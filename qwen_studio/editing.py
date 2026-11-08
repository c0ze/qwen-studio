"""Qwen Image Edit workflows using ComfyUI's native reference conditioning."""
from __future__ import annotations

import uuid
from pathlib import Path

from qwen_studio import moss
from qwen_studio.comfy import COMFY_DIR, EDIT_MODEL, OUTPUT_DIR
from qwen_studio.imaging import load_image, outpaint_input


def _options(nodes: dict, node: str, field: str) -> list:
    """Choices of a loader input; classic nodes use [[...]], V3 combos {"options": [...]}."""
    spec = nodes.get(node, {}).get("input", {}).get("required", {}).get(field) or [[]]
    if isinstance(spec[0], list):
        return spec[0]
    if len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options", [])
    return []


def _moss_status(nodes: dict) -> tuple[bool, str]:
    if "moss_image" not in _options(nodes, "CLIPLoader", "type"):
        return False, "Update ComfyUI to a version with native MOSS-Image support."
    for node, field, name, folder in [
        ("UNETLoader", "unet_name", moss.MODEL, "diffusion_models"),
        ("CLIPLoader", "clip_name", moss.ENCODER, "text_encoders"),
        ("VAELoader", "vae_name", moss.VAE, "vae"),
    ]:
        if name not in _options(nodes, node, field):
            return False, f"Install {name} in ComfyUI/models/{folder}."
    return True, "Ready"


def capabilities(client) -> dict:
    try:
        nodes = client.request_json("/object_info", timeout=5)
        models = _options(nodes, "UnetLoaderGGUF", "unet_name")
        available = EDIT_MODEL in models and "TextEncodeQwenImageEditPlus" in nodes
        moss_ready, moss_reason = _moss_status(nodes)
        return {"online": True, "edit": available, "reference_limit": 3, "edit_model": EDIT_MODEL,
                "reason": "Ready" if available else f"Install {EDIT_MODEL} in ComfyUI/models/diffusion_models and ComfyUI-GGUF.",
                "moss": moss_ready, "moss_model": moss.MODEL, "moss_reason": moss_reason,
                "upscale": "Lanczos (resampling, not AI detail generation)"}
    except (OSError, ValueError, AttributeError):
        return {"online": False, "edit": False, "reference_limit": 3, "edit_model": EDIT_MODEL, "reason": "ComfyUI is unavailable",
                "moss": False, "moss_model": moss.MODEL, "moss_reason": "ComfyUI is unavailable", "upscale": "Lanczos"}


def workflow(client, prompt, negative, width, height, steps, seed, inputs, operation):
    graph = client.workflow(prompt, negative, width, height, steps, seed)
    graph["1"]["inputs"]["unet_name"] = EDIT_MODEL
    refs = {}
    for index, filename in enumerate(inputs, start=1):
        node = str(10 + index)
        graph[node] = {"class_type": "LoadImage", "inputs": {"image": filename}}
        refs[f"image{index}"] = [node, 0]
    for node, text in [("5", prompt), ("6", negative)]:
        graph[node] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": text, **refs,
        }}
    if operation == "extend":
        graph["7"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
            "pixels": ["11", 0], "mask": ["11", 1], "vae": ["4", 0], "grow_mask_by": 0,
        }}
    graph["10"]["inputs"]["filename_prefix"] = f"LMStudio/Qwen-Edit-{uuid.uuid4().hex}"
    return graph


def generate_edit(client, sources: list[Path], prompt: str, negative: str, width: int, height: int, steps: int, seed: int, operation: str) -> Path:
    if not 1 <= len(sources) <= 3:
        raise ValueError("select one to three reference images")
    client.ensure()
    cap = capabilities(client)
    if not cap["edit"]:
        raise ValueError(cap["reason"])
    folder = COMFY_DIR / "input" / "qwen-studio"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for index, source in enumerate(sources):
            path = folder / f"{uuid.uuid4().hex}.png"
            paths.append(path)
            if operation == "extend" and index == 0:
                offset = outpaint_input(source, path, width, height)
            else:
                load_image(source).save(path)
        graph = workflow(client, prompt, negative, width, height, steps, seed,
                         [f"qwen-studio/{p.name}" for p in paths], operation)
        result = client.request_json("/prompt", method="POST", payload={"prompt": graph, "client_id": str(uuid.uuid4())})
        if result.get("node_errors") or not result.get("prompt_id"):
            raise RuntimeError(f"ComfyUI rejected edit: {result}")
        output = client._wait_for_image(result["prompt_id"], OUTPUT_DIR)
        if operation == "extend":
            generated = load_image(output)
            if generated.size != (width, height):
                raise ValueError("unexpected outpaint dimensions")
            # Keep the user's original pixels exact. Only the new border is used.
            generated.paste(load_image(sources[0]), offset)
            generated.save(output)
        return output
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
