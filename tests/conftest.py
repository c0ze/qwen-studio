from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qwen_studio.gallery import Gallery
from qwen_studio.server import create_app


def chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def write_png(
    path: Path,
    width: int = 8,
    height: int = 8,
    workflow: dict | None = None,
) -> Path:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + bytes(width * 3) for _ in range(height))
    parts = [
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", ihdr),
        chunk(b"IDAT", zlib.compress(rows)),
    ]
    if workflow is not None:
        text = "prompt\x00" + json.dumps(workflow)
        parts.append(chunk(b"tEXt", text.encode("latin1")))
    parts.append(chunk(b"IEND", b""))
    path.write_bytes(b"".join(parts))
    return path


def sample_workflow(prompt: str = "a red cube", negative: str = "blurry") -> dict:
    return {
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative}},
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 1024, "height": 1536},
        },
        "8": {"class_type": "KSampler", "inputs": {"seed": 99, "steps": 30}},
    }


class FakeComfy:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict] = []
        self._ready = True
        self.fail = False
        self.n = 0

    def ready(self) -> bool:
        return self._ready

    def generate(self, prompt, negative_prompt, width, height, steps, seed):
        if self.fail:
            raise RuntimeError("comfy exploded")
        self.n += 1
        path = self.root / f"Qwen-Image-2512_{self.n:05d}_.png"
        write_png(path, width=8, height=8)
        self.calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
            }
        )
        return path


@pytest.fixture
def studio(tmp_path: Path):
    gallery = Gallery(tmp_path)
    fake = FakeComfy(tmp_path)
    app = create_app(client=fake, gallery=gallery)
    with TestClient(app) as client:
        yield client, fake, gallery
