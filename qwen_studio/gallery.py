from __future__ import annotations

import json
import re
import uuid
import shutil
import stat
import struct
import threading
import zlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import quote

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_TEXT_BYTES = 4 * 1024 * 1024


@dataclass
class ImageRecord:
    filename: str
    url: str
    width: int | None
    height: int | None
    mtime: float
    created_at: str
    prompt: str | None
    negative_prompt: str | None
    seed: int | None
    steps: int | None
    project: str = ""
    operation: str | None = None
    source: str | None = None
    model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "project": self.project,
            "operation": self.operation,
            "model": self.model,
            "source": self.source,
            "url": self.url,
            "width": self.width,
            "height": self.height,
            "mtime": self.mtime,
            "created_at": self.created_at,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            # JSON numbers cannot represent all ComfyUI seeds in JavaScript.
            "seed": str(self.seed) if self.seed is not None else None,
            "steps": self.steps,
        }


def png_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                return None
            length = struct.unpack(">I", handle.read(4))[0]
            if handle.read(4) != b"IHDR" or length < 8:
                return None
            width, height = struct.unpack(">II", handle.read(8))
            return int(width), int(height)
    except (OSError, struct.error):
        return None


def _inflate(data: bytes) -> bytes:
    decoder = zlib.decompressobj()
    result = decoder.decompress(data, MAX_TEXT_BYTES + 1)
    if len(result) > MAX_TEXT_BYTES or not decoder.eof:
        raise ValueError("invalid or oversized PNG text")
    return result


def _decode_text_chunk(ctype: bytes, data: bytes) -> tuple[str, str] | None:
    try:
        key, separator, rest = data.partition(b"\x00")
        if not separator:
            return None
        if ctype == b"tEXt":
            return key.decode("latin1"), rest.decode("latin1")
        if ctype == b"zTXt" and rest[:1] == b"\x00":
            return key.decode("latin1"), _inflate(rest[1:]).decode("latin1")
        if ctype == b"iTXt" and len(rest) >= 2:
            compressed, method = rest[:2]
            language, sep, rest = rest[2:].partition(b"\x00")
            translated, sep2, value = rest.partition(b"\x00")
            if not sep or not sep2 or compressed not in (0, 1) or method != 0:
                return None
            if compressed:
                value = _inflate(value)
            return key.decode("latin1"), value.decode("utf-8")
    except (ValueError, zlib.error):
        pass
    return None


def png_text_chunks(path: Path) -> dict[str, str]:
    chunks: dict[str, str] = {}
    try:
        with path.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                return chunks
            while header := handle.read(8):
                if len(header) != 8:
                    break
                length, ctype = struct.unpack(">I4s", header)
                if ctype in {b"tEXt", b"zTXt", b"iTXt"} and length <= MAX_TEXT_BYTES:
                    data = handle.read(length)
                    if len(data) != length:
                        break
                    decoded = _decode_text_chunk(ctype, data)
                    if decoded:
                        chunks[decoded[0]] = decoded[1]
                else:
                    # Image data is large and irrelevant to gallery metadata.
                    handle.seek(length, 1)
                if len(handle.read(4)) != 4 or ctype == b"IEND":
                    break
    except (OSError, struct.error):
        pass
    return chunks


def params_from_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    nodes = {str(key): node for key, node in workflow.items() if isinstance(node, dict)}
    ordered = sorted(nodes, key=lambda key: (int(key) if key.isdigit() else -1, key))

    def inputs(node: dict) -> dict:
        value = node.get("inputs")
        return value if isinstance(value, dict) else {}

    def linked(link: Any) -> dict:
        if isinstance(link, list) and link and isinstance(link[0], (str, int)):
            return nodes.get(str(link[0]), {})
        return {}

    encoders = [inputs(nodes[key]).get("text") for key in ordered
                if nodes[key].get("class_type") == "CLIPTextEncode"]
    sampler = next((inputs(nodes[key]) for key in ordered
                    if nodes[key].get("class_type") == "KSampler"), {})
    latent = inputs(linked(sampler.get("latent_image")))
    if not latent:
        latent = next((inputs(nodes[key]) for key in ordered
                       if nodes[key].get("class_type") == "EmptySD3LatentImage"), {})
    positive = inputs(linked(sampler.get("positive"))).get("text")
    negative = inputs(linked(sampler.get("negative"))).get("text")
    return {
        "prompt": positive if isinstance(positive, str) else (encoders[0] if encoders else None),
        "negative_prompt": negative if isinstance(negative, str) else (encoders[1] if len(encoders) > 1 else None),
        "seed": sampler.get("seed"),
        "steps": sampler.get("steps"),
        "width": latent.get("width"),
        "height": latent.get("height"),
    }


def meta_from_png(path: Path) -> dict[str, Any]:
    raw = png_text_chunks(path).get("prompt")
    if not raw:
        return {}
    try:
        workflow = json.loads(raw)
    except (ValueError, RecursionError):
        return {}
    return params_from_workflow(workflow) if isinstance(workflow, dict) else {}


class Gallery:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._children = {}
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[tuple, ImageRecord]] = {}

    @staticmethod
    def _signature(path: Path) -> tuple | None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        return (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)

    def _list_flat(self) -> list[ImageRecord]:
        with self._lock:
            if not self.root.is_dir():
                self._cache.clear()
                return []
            records = []
            seen = set()
            for path in self.root.iterdir():
                if path.suffix.lower() not in IMAGE_EXTS:
                    continue
                try:
                    signature = self._signature(path)
                    if signature is None or not stat.S_ISREG(signature[-1]):
                        continue
                    combined = (signature, self._signature(path.with_suffix(".json")))
                    cached = self._cache.get(path.name)
                    record = cached[1] if cached and cached[0] == combined else self._record(path)
                    self._cache[path.name] = (combined, record)
                    seen.add(path.name)
                    records.append(record)
                except (OSError, ValueError, OverflowError):
                    # A concurrent save/removal must not hide the whole archive.
                    continue
            self._cache = {name: item for name, item in self._cache.items() if name in seen}
            records.sort(key=lambda item: item.mtime, reverse=True)
            return records

    def project_dir(self, project: str) -> Path:
        if not project:
            return self.root
        if not re.fullmatch(r"[a-f0-9]{32}", project):
            raise ValueError("invalid project")
        base = self.root / "projects"
        folder = base / project
        if base.is_symlink() or folder.is_symlink() or not folder.is_dir():
            raise FileNotFoundError("project not found")
        return folder

    def projects(self) -> list[dict]:
        with self._lock:
            base = self.root / "projects"
            if not base.is_dir() or base.is_symlink():
                return []
            result = []
            for folder in base.iterdir():
                if not re.fullmatch(r"[a-f0-9]{32}", folder.name) or folder.is_symlink() or not folder.is_dir():
                    continue
                try:
                    meta = folder / ".project.json"
                    if meta.is_symlink():
                        continue
                    data = json.loads(meta.read_text())
                    if isinstance(data.get("name"), str):
                        result.append({"id": folder.name, "name": data["name"]})
                except (OSError, ValueError, AttributeError):
                    continue
            return sorted(result, key=lambda item: item["name"].casefold())

    def save_project(self, name: str, project: str = "") -> dict:
        name = name.strip()
        if not name or len(name) > 80 or any(ord(char) < 32 for char in name):
            raise ValueError("project name must be 1–80 printable characters")
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            base = self.root / "projects"
            if base.is_symlink():
                raise ValueError("project directory cannot be a symlink")
            if project:
                folder = self.project_dir(project)
            else:
                project = uuid.uuid4().hex
                folder = base / project
                folder.mkdir(parents=True)
            target = folder / ".project.json"
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, delete=False) as f:
                f.write(json.dumps({"name": name}, ensure_ascii=False))
            Path(f.name).replace(target)
            return {"id": project, "name": name}

    def list_images(self) -> list[ImageRecord]:
        with self._lock:
            result = self._list_flat()
            for project in self.projects():
                folder = self.project_dir(project["id"])
                child = self._children.setdefault(project["id"], Gallery(folder))
                for record in child._list_flat():
                    filename = f"projects/{project['id']}/{record.filename}"
                    result.append(replace(record, filename=filename, project=project["id"], url=f"/media/{quote(filename, safe='/')}"))
            return sorted(result, key=lambda item: item.mtime, reverse=True)

    def resolve(self, filename: str) -> Path:
        if not filename or "\x00" in filename or "\\" in filename:
            raise ValueError("invalid image filename")
        parts = filename.split("/")
        if len(parts) == 1:
            folder = self.root
        elif len(parts) == 3 and parts[0] == "projects":
            folder = self.project_dir(parts[1])
        else:
            raise ValueError("invalid image filename")
        name = parts[-1]
        if name in {"", ".", ".."} or Path(name).suffix.lower() not in IMAGE_EXTS:
            raise ValueError("invalid image filename")
        path = folder / name
        if path.is_symlink():
            raise ValueError("image symlinks are not allowed")
        if not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def asset_id(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def move(self, filename: str, project: str) -> str:
        with self._lock:
            source = self.resolve(filename)
            folder = self.project_dir(project)
            if source.parent == folder:
                return filename
            destination = folder / source.name
            if destination.exists() or destination.with_suffix(".json").exists():
                destination = folder / f"{source.stem}-{uuid.uuid4().hex[:8]}{source.suffix}"
            sidecar = source.with_suffix(".json")
            if sidecar.is_symlink() or (sidecar.exists() and not sidecar.is_file()):
                raise ValueError("invalid image metadata file")
            if sidecar.exists():
                shutil.copy2(sidecar, destination.with_suffix(".json"))
            try:
                source.rename(destination)
            except OSError:
                destination.with_suffix(".json").unlink(missing_ok=True)
                raise
            shared = any(other.stem == source.stem and other.suffix.lower() in IMAGE_EXTS and other.is_file() for other in source.parent.iterdir())
            if not shared:
                sidecar.unlink(missing_ok=True)
            return self.asset_id(destination)

    def delete(self, filename: str) -> None:
        with self._lock:
            path = self.resolve(filename)
            sidecar = path.with_suffix(".json")
            if sidecar.is_symlink() or (sidecar.exists() and not sidecar.is_file()):
                raise ValueError("invalid image metadata file")
            # Legacy metadata is shared by files with the same stem.
            shared = any(other != path and other.stem == path.stem
                         and other.suffix.lower() in IMAGE_EXTS and other.is_file()
                         for other in path.parent.iterdir())
            path.unlink()
            self._cache.pop(filename, None)
            if not shared:
                sidecar.unlink(missing_ok=True)

    def write_sidecar(self, image_path: Path, meta: dict[str, Any]) -> None:
        with self._lock:
            path = self.resolve(self.asset_id(image_path))
            if path.resolve() != image_path.resolve():
                raise ValueError("image is outside the gallery")
            sidecar = path.with_suffix(".json")
            if sidecar.is_symlink():
                raise ValueError("metadata symlinks are not allowed")
            payload = {**meta, "filename": path.name}
            temporary = None
            try:
                with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                        prefix=".metadata-", delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
                temporary.replace(sidecar)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            self._cache.pop(path.name, None)

    def _record(self, path: Path) -> ImageRecord:
        info = path.stat()
        created = datetime.fromtimestamp(info.st_mtime, tz=timezone.utc).isoformat()
        sidecar = self._read_sidecar(path)
        png_meta = meta_from_png(path) if path.suffix.lower() == ".png" else {}
        size = png_size(path) if path.suffix.lower() == ".png" else None

        def pick(key, expected):
            for source in (sidecar, png_meta):
                value = source.get(key)
                if expected is int and isinstance(value, str) and value.isascii() and value.isdigit():
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                if isinstance(value, expected) and not isinstance(value, bool):
                    return value
            return None

        width = pick("width", int)
        height = pick("height", int)
        return ImageRecord(
            filename=path.name,
            url=f"/media/{quote(path.name, safe='')}",
            width=width if width is not None else (size[0] if size else None),
            height=height if height is not None else (size[1] if size else None),
            mtime=info.st_mtime,
            created_at=created,
            prompt=pick("prompt", str),
            negative_prompt=pick("negative_prompt", str),
            seed=pick("seed", int),
            steps=pick("steps", int),
            operation=pick("operation", str),
            source=pick("source", str),
            model=pick("model", str),
        )

    def _read_sidecar(self, path: Path) -> dict[str, Any]:
        sidecar = path.with_suffix(".json")
        if sidecar.is_symlink() or not sidecar.is_file():
            return {}
        try:
            if sidecar.stat().st_size > MAX_TEXT_BYTES:
                return {}
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            return {}
        return data if isinstance(data, dict) else {}
