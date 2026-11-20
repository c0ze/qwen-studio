"""User LoRAs: what ComfyUI offers, which base model each was trained on, its trigger words.

ComfyUI's LoraLoaderModelOnly lists the files in models/loras. The base model
and trigger words come from the safetensors header metadata (ai-toolkit and
kohya write `ss_base_model_version` and `ss_tag_frequency`; ai-toolkit can also
write `trigger_word`), falling back to the file name (`..._zimage_turbo_...`).
Only the JSON header is read, never the weights.
"""
from __future__ import annotations

import json
import re
import struct
import threading
from pathlib import Path
from typing import Any

from qwen_studio import nitro

INTERNAL = {nitro.LORA}  # LoRAs a workflow applies by itself
MAX_HEADER = 32 * 1024 * 1024
# Which Studio model can load a LoRA of each base. NitroFusion 2 is a Z-Image
# distillation, but its 1-2 step sampling hasn't been tested with LoRAs.
BASE_MODELS = {"zimage": {"zimage"}}
_NAME_BASES = [(re.compile(r"z[-_]?image"), "zimage"), (re.compile(r"qwen[-_]?image"), "qwen_image"),
               (re.compile(r"flux"), "flux"), (re.compile(r"sdxl"), "sdxl")]

_cache: dict[Path, tuple[float, int, dict[str, Any]]] = {}
_lock = threading.Lock()


def header_metadata(path: Path) -> dict[str, str]:
    try:
        with path.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            if not 0 < length <= MAX_HEADER:
                return {}
            header = json.loads(handle.read(length))
    except (OSError, struct.error, ValueError, RecursionError):
        return {}
    meta = header.get("__metadata__") if isinstance(header, dict) else None
    return {str(k): str(v) for k, v in meta.items()} if isinstance(meta, dict) else {}


def _base(name: str, meta: dict[str, str]) -> str | None:
    declared = meta.get("ss_base_model_version", "").lower()
    for text in (declared, name.lower()):
        for pattern, base in _NAME_BASES:
            if pattern.search(text):
                return base
    return declared or None


def _triggers(meta: dict[str, str]) -> list[str]:
    words: list[str] = []
    for key in ("trigger_word", "trigger_words", "modelspec.trigger_phrase"):
        if meta.get(key):
            words += [w.strip() for w in re.split(r"[,\n]", meta[key]) if w.strip()]
    try:
        freq = json.loads(meta.get("ss_tag_frequency") or "{}")
    except ValueError:
        freq = {}
    if isinstance(freq, dict):
        for tags in freq.values():
            if isinstance(tags, dict):  # kohya counts captions' tags; the trigger is in every caption
                top = max(tags.values(), default=0)
                words += [tag for tag, count in tags.items() if count == top and isinstance(tag, str)][:3]
    return list(dict.fromkeys(words))[:5]


def describe(lora_dir: Path, name: str) -> dict[str, Any]:
    path = (lora_dir / name).resolve()
    info: dict[str, Any] = {"name": name, "label": Path(name).stem, "base": None, "triggers": [], "models": []}
    try:
        if lora_dir.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(name)
        stat = path.stat()
    except OSError:
        info["base"] = _base(name, {})
    else:
        with _lock:
            cached = _cache.get(path)
            if cached and cached[:2] == (stat.st_mtime, stat.st_size):
                return cached[2]
        meta = header_metadata(path) if path.suffix == ".safetensors" else {}
        info["base"] = _base(name, meta)
        info["triggers"] = _triggers(meta)
        if meta.get("ss_output_name"):
            info["label"] = meta["ss_output_name"]
            step = re.search(r"_0*(\d+)$", Path(name).stem)
            if step and not info["label"].endswith(step.group(1)):
                info["label"] += f" · step {step.group(1)}"
    info["models"] = sorted(BASE_MODELS.get(info["base"] or "", set()))
    if path.is_file():
        with _lock:
            _cache[path] = (path.stat().st_mtime, path.stat().st_size, info)
    return info


def available(names: list[str], lora_dir: Path) -> list[dict[str, Any]]:
    return [describe(lora_dir, n) for n in names if isinstance(n, str) and n not in INTERNAL]


def check(choices: list[tuple[str, float]], model: str, offered: list[dict[str, Any]] | None) -> None:
    """Refuse unknown LoRAs and LoRAs of another base model. `offered` is None when ComfyUI is offline."""
    if choices and model not in {m for models in BASE_MODELS.values() for m in models}:
        raise ValueError(f"LoRAs work with Z-Image Turbo only, not {model}")
    if offered is None:
        return
    by_name = {item["name"]: item for item in offered}
    for name, _strength in choices:
        item = by_name.get(name)
        if item is None:
            raise ValueError(f"LoRA not found in ComfyUI/models/loras: {name}")
        if model not in item["models"]:
            base = item["base"] or "an unknown base model"
            raise ValueError(f"{item['label']} was trained on {base}; it can't be used with {model}")
