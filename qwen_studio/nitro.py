"""NitroFusion 2 text-to-image workflow (Apache-2.0): 1-2 step Z-Image-Turbo.

Mirrors the GGUF workflows published with the model (ChenDY/NitroFusion2,
workflows/nitrofusion2_{1,2}step_gguf.json). CFG stays at 1, so there is no
negative prompt: the negative conditioning is the positive one zeroed out.
"""
from __future__ import annotations

import os
from typing import Any

MODEL = os.environ.get("NITRO_MODEL", "nf2-Q8_0.gguf")
ENCODER = "Qwen3-4B-Q8_0.gguf"
VAE = "ae.safetensors"
LORA = "nf2-dedistill-lora.safetensors"
PREFIX = "LMStudio/NitroFusion2"
MIN_STEPS = 1
MAX_STEPS = 2


def _sampler(model: list, seed: int, steps: int) -> dict[str, Any]:
    return {
        "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
        "scheduler": "simple", "denoise": 1.0, "model": model,
        "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
    }


def _two_step(seed: int) -> dict[str, Any]:
    # Step 1 runs with the de-distillation LoRA at 0.5, step 2 on the bare
    # model; both share the two-step sigma table (shift 6).
    shared = {"add_noise": "disable", "noise_seed": seed, "steps": 2, "cfg": 1.0,
              "sampler_name": "euler", "scheduler": "simple",
              "positive": ["5", 0], "negative": ["6", 0]}
    return {
        "11": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": LORA, "strength_model": 0.5},
        },
        "12": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["11", 0], "shift": 6.0}},
        "8": {
            "class_type": "KSamplerAdvanced",
            "inputs": {**shared, "add_noise": "enable", "model": ["12", 0], "latent_image": ["7", 0],
                       "start_at_step": 0, "end_at_step": 1, "return_with_leftover_noise": "enable"},
        },
        "13": {
            "class_type": "KSamplerAdvanced",
            "inputs": {**shared, "model": ["2", 0], "latent_image": ["8", 0],
                       "start_at_step": 1, "end_at_step": 2, "return_with_leftover_noise": "disable"},
        },
    }


def workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise ValueError("NitroFusion 2 takes 1 or 2 steps")
    graph: dict[str, Any] = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": MODEL}},
        "2": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 3.0 if steps == 1 else 6.0},
        },
        "3": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": ENCODER, "type": "lumina2"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "8": {"class_type": "KSampler", "inputs": _sampler(["2", 0], seed, 1)},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "10": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": PREFIX, "images": ["9", 0]},
        },
    }
    if steps == 2:
        graph.update(_two_step(seed))
        graph["9"]["inputs"]["samples"] = ["13", 0]
    return graph
