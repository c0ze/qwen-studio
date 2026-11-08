"""MOSS-Image text-to-image workflow (OpenMOSS, Apache-2.0).

Needs a ComfyUI with native MOSS support: upstream master after v0.40.0
(e087d8f6 and follow-ups).
"""
from __future__ import annotations

import os
from typing import Any

MODEL = os.environ.get("MOSS_MODEL", "moss_image_sft_int8_convrot.safetensors")
ENCODER = "qwen3.5_9b_int8_convrot.safetensors"
VAE = "flux2-vae_small.safetensors"
PREFIX = "LMStudio/MOSS-Image"


def workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    # Mirrors the reference graph from ComfyUI PR #17095. Skip-layer
    # guidance on layer 22 stands in for the pipeline's uncond_skip_layers.
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "SkipLayerGuidanceDiTSimple",
            "inputs": {
                "model": ["1", 0],
                "double_layers": "",
                "single_layers": "22",
                "start_percent": 0.0,
                "end_percent": 0.5,
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": ENCODER, "type": "moss_image", "device": "default"},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["3", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["3", 0]},
        },
        "7": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 2.5,
                "sampler_name": "er_sde",
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
            "inputs": {"filename_prefix": PREFIX, "images": ["9", 0]},
        },
    }
