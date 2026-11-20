"""Z-Image Turbo (Tongyi-MAI, Apache-2.0): text-to-image and restyle, with user LoRAs.

The 6B turbo model samples in about 8 steps at CFG 1, so like NitroFusion 2 (a
distillation of it) there is no negative prompt: the negative conditioning is
the positive one zeroed out. It shares NitroFusion's Qwen3-4B text encoder and
FLUX.1 autoencoder. LoRAs trained on Z-Image Turbo (ai-toolkit
`ss_base_model_version: zimage`) chain after the model loader.

Restyle encodes one source image as the starting latent and denoises only part
of the way, so the result keeps the source's composition and silhouette while
the prompt and LoRAs set the look.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

MODEL = os.environ.get("ZIMAGE_MODEL", "z_image_turbo_int8_convrot.safetensors")
ENCODER = "Qwen3-4B-Q8_0.gguf"
VAE = "ae.safetensors"
PREFIX = "LMStudio/Z-Image-Turbo"
MIN_STEPS = 4
MAX_STEPS = 20
DEFAULT_STEPS = 8
SHIFT = 3.0
MAX_LORAS = 3
SOURCE_IMAGE = "restyle-source"  # placeholder replaced with the copied input file


def workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    loras: Sequence[tuple[str, float]] = (),
    source: str | None = None,
    denoise: float = 1.0,
) -> dict[str, Any]:
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise ValueError(f"Z-Image Turbo takes {MIN_STEPS} to {MAX_STEPS} steps")
    if len(loras) > MAX_LORAS:
        raise ValueError(f"use at most {MAX_LORAS} LoRAs")
    if source is None and denoise != 1.0:
        raise ValueError("partial denoise needs a source image")
    if not 0.05 <= denoise <= 1.0:
        raise ValueError("denoise must be from 0.05 to 1")
    graph: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": ENCODER, "type": "lumina2"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
    }
    model: list = ["1", 0]
    for index, (name, strength) in enumerate(loras):
        node = str(20 + index)
        graph[node] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)}}
        model = [node, 0]
    graph["2"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": SHIFT}}
    if source is None:
        graph["7"] = {"class_type": "EmptySD3LatentImage",
                      "inputs": {"width": width, "height": height, "batch_size": 1}}
    else:
        graph["11"] = {"class_type": "LoadImage", "inputs": {"image": source}}
        graph["12"] = {"class_type": "ImageScale", "inputs": {
            "image": ["11", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
        graph["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["12", 0], "vae": ["4", 0]}}
    graph["8"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
        "denoise": float(denoise), "model": ["2", 0],
        "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
    }}
    graph["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}}
    graph["10"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": PREFIX, "images": ["9", 0]}}
    return graph
