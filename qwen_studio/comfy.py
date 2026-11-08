from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from qwen_studio import moss, nitro

# Text-to-image models other than Qwen 2512, each a module with workflow() and PREFIX.
EXTRA_MODELS = {"moss": moss, "nitro": nitro}

COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = Path(os.environ.get("COMFYUI_DIR", str(Path.home() / "ComfyUI"))).expanduser().resolve()
OUTPUT_DIR = COMFY_DIR / "output"
GALLERY_DIR = OUTPUT_DIR / "LMStudio"
RUN_SCRIPT = COMFY_DIR / "run-qwen-image.sh"
LOG_PATH = Path.home() / ".local/state/qwen-studio/comfyui.log"
MODEL_NAME = os.environ.get("QWEN_MODEL", "qwen-image-2512-Q4_K_S.gguf")
EDIT_MODEL = os.environ.get("QWEN_EDIT_MODEL", "qwen-image-edit-2511-Q4_K_S.gguf")
ENCODER_NAME = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"
FILENAME_PREFIX = "LMStudio/Qwen-Image-2512"

DEFAULT_NEGATIVE = (
    "low resolution, low quality, malformed anatomy, malformed fingers, "
    "oversaturated, waxy skin, missing facial detail, overly smooth, "
    "AI-generated look, chaotic composition, blurry or distorted text"
)

MIN_SIZE = 512
MAX_SIZE = 1664
SIZE_STEP = 64
MIN_STEPS = 4
MAX_STEPS = 50


def validate_params(prompt: str, width: int, height: int, steps: int, seed: int, model: str = "qwen") -> str:
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if not (MIN_SIZE <= width <= MAX_SIZE and width % SIZE_STEP == 0):
        raise ValueError("width must be a multiple of 64 from 512 through 1664")
    if not (MIN_SIZE <= height <= MAX_SIZE and height % SIZE_STEP == 0):
        raise ValueError("height must be a multiple of 64 from 512 through 1664")
    low, high = (nitro.MIN_STEPS, nitro.MAX_STEPS) if model == "nitro" else (MIN_STEPS, MAX_STEPS)
    if not low <= steps <= high:
        raise ValueError(f"steps must be from {low} through {high}")
    if seed < 0 or seed >= 2**63:
        raise ValueError("seed must be 0 or a positive 63-bit integer")
    return prompt.strip()


def resolve_seed(seed: int) -> int:
    if seed == 0:
        return secrets.randbelow(2**63 - 1) + 1
    return seed


class ComfyClient:
    def __init__(self, url: str = COMFY_URL, run_script: Path = RUN_SCRIPT) -> None:
        self.url = url.rstrip("/")
        self.run_script = run_script

    def request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return json.loads(body) if body else {}

    def ready(self) -> bool:
        try:
            self.request_json("/system_stats", timeout=3)
            return True
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return False

    def release_if_idle(self) -> bool:
        """Unload models so VRAM is free while ComfyUI has no work."""
        try:
            queue = self.request_json("/queue", timeout=5)
            if queue.get("queue_running") or queue.get("queue_pending"):
                return False
            self.request_json(
                "/free",
                method="POST",
                payload={"unload_models": True, "free_memory": True},
            )
            return True
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return False

    def ensure(self) -> None:
        if self.ready():
            return
        if not self.run_script.is_file():
            raise RuntimeError(f"ComfyUI launcher is missing: {self.run_script}")
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("ab") as log:
            subprocess.Popen(
                [str(self.run_script)],
                cwd=COMFY_DIR,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.ready():
                return
            time.sleep(2)
        raise RuntimeError(f"ComfyUI did not start; inspect {LOG_PATH}")

    def workflow(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        model: str = "qwen",
    ) -> dict[str, Any]:
        if model in EXTRA_MODELS:
            return EXTRA_MODELS[model].workflow(prompt, negative_prompt, width, height, steps, seed)
        if model != "qwen":
            raise ValueError(f"unknown model: {model}")
        return {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": MODEL_NAME}},
            "2": {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {"model": ["1", 0], "shift": 3.1},
            },
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": ENCODER_NAME,
                    "type": "qwen_image",
                    "device": "default",
                },
            },
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["3", 0]},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["3", 0]},
            },
            "7": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": 4.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["2", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["7", 0],
                },
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["8", 0], "vae": ["4", 0]},
            },
            "10": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": FILENAME_PREFIX,
                    "images": ["9", 0],
                },
            },
        }

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        output_dir: Path = OUTPUT_DIR,
        model: str = "qwen",
    ) -> Path:
        self.ensure()
        workflow = self.workflow(prompt, negative_prompt, width, height, steps, seed, model)
        # Reused seeds may cache upstream pixels, but each request needs a new
        # saved asset, including after an earlier asset was deleted.
        prefix = EXTRA_MODELS[model].PREFIX if model in EXTRA_MODELS else FILENAME_PREFIX
        workflow["10"]["inputs"]["filename_prefix"] = f"{prefix}-{uuid.uuid4().hex}"
        result = self.request_json(
            "/prompt",
            method="POST",
            payload={
                "prompt": workflow,
                "client_id": str(uuid.uuid4()),
            },
        )
        if result.get("node_errors"):
            raise RuntimeError(f"ComfyUI rejected the workflow: {result['node_errors']}")
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt id: {result}")
        return self._wait_for_image(prompt_id, output_dir)

    def _wait_for_image(self, prompt_id: str, output_dir: Path) -> Path:
        gallery_dir = (output_dir / "LMStudio").resolve()
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            history = self.request_json(f"/history/{prompt_id}", timeout=10)
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise RuntimeError(f"Image generation failed: {messages}")
                # This workflow saves the final image in node 10. Other nodes
                # can expose previews or temporary images that are not assets.
                output = entry.get("outputs", {}).get("10", {})
                for image in output.get("images", []):
                    filename = image.get("filename")
                    subfolder = image.get("subfolder", "")
                    if (
                        image.get("type") != "output"
                        or not isinstance(filename, str)
                        or not isinstance(subfolder, str)
                        or Path(filename).name != filename
                    ):
                        continue
                    path = (output_dir / subfolder / filename).resolve()
                    if path.parent == gallery_dir and path.is_file():
                        return path
                if status.get("completed") or status.get("status_str") == "success":
                    raise RuntimeError(
                        "Generation completed without a valid image in LMStudio"
                    )
            time.sleep(2)
        raise TimeoutError("Image generation exceeded the 30-minute timeout")
