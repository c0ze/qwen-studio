from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import pytest

from qwen_studio import editing, loras, nitro, zimage
from qwen_studio.editing import capabilities
from tests.conftest import write_png

LOOPWRIGHT = "gand_loopwright-toybox_zimage_turbo_r16_000001500.safetensors"


def write_lora(folder: Path, name: str, meta: dict | None) -> Path:
    header: dict = {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    if meta is not None:
        header["__metadata__"] = meta
    raw = json.dumps(header).encode()
    path = folder / name
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0\0\0")
    return path


def ai_toolkit_meta(trigger="lwbox"):
    return {"ss_base_model_version": "zimage", "ss_output_name": "gand_loopwright-toybox_zimage_turbo_r16",
            "ss_tag_frequency": json.dumps({f"1_{trigger}": {trigger: 1}})}


def zimage_nodes(lora_names=(LOOPWRIGHT,), unet=None):
    return {
        "UNETLoader": {"input": {"required": {"unet_name": [[zimage.MODEL] if unet is None else unet]}}},
        "CLIPLoaderGGUF": {"input": {"required": {"clip_name": [[zimage.ENCODER]], "type": [["lumina2"]]}}},
        "VAELoader": {"input": {"required": {"vae_name": [[zimage.VAE]]}}},
        "LoraLoaderModelOnly": {"input": {"required": {"lora_name": [[nitro.LORA, *lora_names]]}}},
    }


class InfoClient:
    def __init__(self, nodes):
        self.nodes = nodes

    def request_json(self, path, **kwargs):
        return self.nodes


@pytest.fixture
def lora_dir(tmp_path, monkeypatch):
    folder = tmp_path / "loras"
    folder.mkdir()
    monkeypatch.setattr(editing, "LORA_DIR", folder)
    loras._cache.clear()
    return folder


# Workflow graphs

def test_text_to_image_graph_has_no_negative_and_turbo_sampling():
    graph = zimage.workflow("a toy rocket", "ignored", 1024, 768, 8, 42)
    assert graph["1"]["inputs"]["unet_name"] == zimage.MODEL
    assert graph["6"]["class_type"] == "ConditioningZeroOut"
    sampler = graph["8"]["inputs"]
    assert (sampler["cfg"], sampler["steps"], sampler["seed"], sampler["denoise"]) == (1.0, 8, 42, 1.0)
    assert graph["2"]["inputs"]["model"] == ["1", 0]
    assert graph["7"]["class_type"] == "EmptySD3LatentImage"
    assert (graph["7"]["inputs"]["width"], graph["7"]["inputs"]["height"]) == (1024, 768)


def test_loras_chain_in_order_before_the_sampling_shift():
    graph = zimage.workflow("x", "", 1024, 1024, 8, 1, [("a.safetensors", 0.8), ("b.safetensors", -0.5)])
    assert graph["20"]["inputs"] == {"model": ["1", 0], "lora_name": "a.safetensors", "strength_model": 0.8}
    assert graph["21"]["inputs"] == {"model": ["20", 0], "lora_name": "b.safetensors", "strength_model": -0.5}
    assert graph["2"]["inputs"]["model"] == ["21", 0]


def test_restyle_graph_encodes_the_source_and_denoises_partly():
    graph = zimage.workflow("x", "", 1024, 1024, 8, 1, source="qwen-studio/in.png", denoise=0.55)
    assert graph["11"]["inputs"]["image"] == "qwen-studio/in.png"
    assert graph["12"]["inputs"]["width"] == 1024 and graph["12"]["inputs"]["crop"] == "center"
    assert graph["7"] == {"class_type": "VAEEncode", "inputs": {"pixels": ["12", 0], "vae": ["4", 0]}}
    assert graph["8"]["inputs"]["denoise"] == 0.55


@pytest.mark.parametrize("kwargs, message", [
    ({"steps": 3}, "4 to 20 steps"),
    ({"steps": 21}, "4 to 20 steps"),
    ({"denoise": 0.5}, "needs a source"),
    ({"loras": [(f"{i}.safetensors", 1.0) for i in range(4)]}, "at most 3"),
])
def test_workflow_refuses_bad_settings(kwargs, message):
    args = {"prompt": "x", "negative_prompt": "", "width": 1024, "height": 1024, "steps": 8, "seed": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        zimage.workflow(**args)


# LoRA metadata

def test_lora_base_label_and_trigger_come_from_ai_toolkit_metadata(lora_dir):
    write_lora(lora_dir, LOOPWRIGHT, ai_toolkit_meta())
    info = loras.describe(lora_dir, LOOPWRIGHT)
    assert info["base"] == "zimage"
    assert info["triggers"] == ["lwbox"]
    assert info["models"] == ["zimage"]
    assert info["label"] == "gand_loopwright-toybox_zimage_turbo_r16 · step 1500"


def test_lora_base_falls_back_to_the_file_name(lora_dir):
    write_lora(lora_dir, "style_qwen_image_v1.safetensors", None)
    info = loras.describe(lora_dir, "style_qwen_image_v1.safetensors")
    assert (info["base"], info["models"], info["triggers"]) == ("qwen_image", [], [])


def test_explicit_trigger_word_metadata(lora_dir):
    write_lora(lora_dir, "x_zimage.safetensors", {"trigger_word": "toyworld, felt"})
    assert loras.describe(lora_dir, "x_zimage.safetensors")["triggers"] == ["toyworld", "felt"]


def test_broken_or_missing_lora_files_do_not_fail_the_listing(lora_dir):
    (lora_dir / "broken_zimage.safetensors").write_bytes(b"\xff" * 16)
    listed = loras.available(["broken_zimage.safetensors", "gone_zimage.safetensors", nitro.LORA], lora_dir)
    assert [item["name"] for item in listed] == ["broken_zimage.safetensors", "gone_zimage.safetensors"]
    assert all(item["base"] == "zimage" for item in listed)


def test_lora_outside_the_folder_is_not_read(lora_dir, tmp_path):
    write_lora(tmp_path, "secret_zimage.safetensors", {"trigger_word": "leak"})
    assert loras.describe(lora_dir, "../secret_zimage.safetensors")["triggers"] == []


def test_check_refuses_unknown_and_mismatched_loras(lora_dir):
    write_lora(lora_dir, LOOPWRIGHT, ai_toolkit_meta())
    write_lora(lora_dir, "q_qwen_image.safetensors", None)
    offered = loras.available([LOOPWRIGHT, "q_qwen_image.safetensors"], lora_dir)
    loras.check([(LOOPWRIGHT, 0.8)], "zimage", offered)
    with pytest.raises(ValueError, match="not found"):
        loras.check([("nope.safetensors", 1.0)], "zimage", offered)
    with pytest.raises(ValueError, match="trained on qwen_image"):
        loras.check([("q_qwen_image.safetensors", 1.0)], "zimage", offered)
    with pytest.raises(ValueError, match="Z-Image Turbo only"):
        loras.check([(LOOPWRIGHT, 1.0)], "qwen", offered)
    loras.check([("anything.safetensors", 1.0)], "zimage", None)  # ComfyUI offline: the job checks later


# Capabilities

def test_capabilities_report_zimage_and_user_loras(lora_dir):
    write_lora(lora_dir, LOOPWRIGHT, ai_toolkit_meta())
    cap = capabilities(InfoClient(zimage_nodes()))
    assert (cap["zimage"], cap["zimage_reason"], cap["lora_limit"]) == (True, "Ready", 3)
    assert [item["name"] for item in cap["loras"]] == [LOOPWRIGHT]  # the internal NitroFusion LoRA is hidden
    assert cap["loras"][0]["triggers"] == ["lwbox"]


def test_capabilities_name_the_missing_zimage_model(lora_dir):
    cap = capabilities(InfoClient(zimage_nodes(unet=[])))
    assert cap["zimage"] is False and zimage.MODEL in cap["zimage_reason"]


def test_offline_capabilities_have_no_loras():
    class Offline:
        def request_json(self, *args, **kwargs):
            raise OSError("down")

    cap = capabilities(Offline())
    assert (cap["zimage"], cap["loras"]) == (False, [])


# API

def _wait_job(client, job_id: str) -> dict:
    for _ in range(250):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _ready(fake, lora_dir):
    write_lora(lora_dir, LOOPWRIGHT, ai_toolkit_meta())
    write_lora(lora_dir, "q_qwen_image.safetensors", None)
    fake.object_info = zimage_nodes((LOOPWRIGHT, "q_qwen_image.safetensors"))


def test_generate_with_zimage_and_a_lora_records_it(studio, lora_dir):
    client, fake, gallery = studio
    _ready(fake, lora_dir)
    response = client.post("/api/generate", json={"prompt": "lwbox, a toy rocket", "model": "zimage", "steps": 8,
                                                  "loras": [{"name": LOOPWRIGHT, "strength": 0.8}]})
    assert response.status_code == 200, response.text
    assert response.json()["loras"] == [{"name": LOOPWRIGHT, "strength": 0.8}]
    done = _wait_job(client, response.json()["id"])
    assert done["status"] == "done"
    assert fake.calls[0]["model"] == "zimage" and fake.calls[0]["loras"] == [(LOOPWRIGHT, 0.8)]
    sidecar = json.loads((gallery.root / done["filename"]).with_suffix(".json").read_text())
    assert sidecar["loras"] == [{"name": LOOPWRIGHT, "strength": 0.8}] and sidecar["model"] == "zimage"
    listed = client.get("/api/images").json()["images"][0]
    assert listed["loras"] == [{"name": LOOPWRIGHT, "strength": 0.8}]


@pytest.mark.parametrize("body, message", [
    ({"model": "qwen", "loras": [{"name": LOOPWRIGHT}]}, "Z-Image Turbo only"),
    ({"model": "zimage", "loras": [{"name": "q_qwen_image.safetensors"}]}, "trained on qwen_image"),
    ({"model": "zimage", "loras": [{"name": "missing.safetensors"}]}, "not found"),
    ({"model": "zimage", "loras": [{"name": LOOPWRIGHT}, {"name": LOOPWRIGHT}]}, "once"),
    ({"model": "zimage", "steps": 30}, "steps must be from 4 through 20"),
    ({"model": "zimage", "operation": "edit", "sources": ["x.png"]}, "creates from text or restyles"),
    ({"model": "qwen", "operation": "restyle", "sources": ["x.png"]}, "restyle uses Z-Image"),
])
def test_zimage_requests_are_validated(studio, lora_dir, body, message):
    client, fake, _ = studio
    _ready(fake, lora_dir)
    response = client.post("/api/generate", json={"prompt": "toy", "steps": 8, **body})
    assert response.status_code == 400 and message in response.json()["detail"], response.text


def test_lora_strength_is_bounded(studio, lora_dir):
    client, fake, _ = studio
    _ready(fake, lora_dir)
    response = client.post("/api/generate", json={"prompt": "toy", "model": "zimage", "steps": 8,
                                                  "loras": [{"name": LOOPWRIGHT, "strength": 3}]})
    assert response.status_code == 422


def test_restyle_runs_on_one_source_with_its_denoise(studio, lora_dir, monkeypatch):
    client, fake, gallery = studio
    _ready(fake, lora_dir)
    write_png(gallery.root / "render.png", 16, 16)
    seen = {}

    def fake_restyle(client_, source, prompt, width, height, steps, seed, user_loras, denoise):
        seen.update(source=source.read_bytes(), prompt=prompt, loras=user_loras, denoise=denoise, size=(width, height))
        return write_png(gallery.root / "Z-Image-Turbo-Restyle_00001_.png", width=8, height=8)

    monkeypatch.setattr(editing, "generate_restyle", fake_restyle)
    response = client.post("/api/generate", json={
        "prompt": "lwbox, toy sheep", "model": "zimage", "operation": "restyle", "sources": ["render.png"],
        "steps": 8, "denoise": 0.45, "loras": [{"name": LOOPWRIGHT, "strength": 1.0}]})
    assert response.status_code == 200, response.text
    done = _wait_job(client, response.json()["id"])
    assert done["status"] == "done", done
    assert seen["denoise"] == 0.45 and seen["loras"] == [(LOOPWRIGHT, 1.0)] and seen["size"] == (1024, 1024)
    assert seen["source"] == (gallery.root / "render.png").read_bytes()  # a snapshot of the source
    sidecar = json.loads((gallery.root / done["filename"]).with_suffix(".json").read_text())
    assert (sidecar["operation"], sidecar["model"], sidecar["denoise"], sidecar["source"]) == (
        "restyle", "zimage", 0.45, "render.png")


def test_restyle_needs_exactly_one_source(studio, lora_dir):
    client, fake, gallery = studio
    _ready(fake, lora_dir)
    for name in ("a.png", "b.png"):
        write_png(gallery.root / name)
    response = client.post("/api/generate", json={"prompt": "x", "model": "zimage", "operation": "restyle",
                                                  "steps": 8, "sources": ["a.png", "b.png"]})
    assert response.status_code == 400 and "one source" in response.json()["detail"]


def test_zimage_unavailable_is_a_conflict(studio, lora_dir):
    client, fake, _ = studio
    fake.object_info = zimage_nodes(unet=[])
    response = client.post("/api/generate", json={"prompt": "x", "model": "zimage", "steps": 8})
    assert response.status_code == 409 and zimage.MODEL in response.json()["detail"]


def test_restyle_comfy_payload(tmp_path, monkeypatch):
    """generate_restyle copies the source into ComfyUI's input folder and submits the restyle graph."""
    source = write_png(tmp_path / "src.png", 16, 16)
    monkeypatch.setattr(editing, "COMFY_DIR", tmp_path)

    class Client:
        submitted = None

        def ensure(self):
            pass

        def request_json(self, path, method="GET", payload=None, timeout=30):
            assert (path, method) == ("/prompt", "POST")
            Client.submitted = payload["prompt"]
            name = payload["prompt"]["11"]["inputs"]["image"]
            assert (tmp_path / "input" / name).is_file()  # the copy exists while ComfyUI runs
            return {"prompt_id": "p1"}

        def _wait_for_image(self, prompt_id, output_dir):
            return tmp_path / "out.png"

    out = editing.generate_restyle(Client(), source, "toy", 768, 1024, 8, 7, [(LOOPWRIGHT, 0.9)], 0.5)
    graph = Client.submitted
    assert out == tmp_path / "out.png"
    assert graph["8"]["inputs"]["denoise"] == 0.5 and graph["20"]["inputs"]["lora_name"] == LOOPWRIGHT
    assert graph["10"]["inputs"]["filename_prefix"].startswith(zimage.PREFIX + "-Restyle-")
    assert not list((tmp_path / "input" / "qwen-studio").iterdir())  # the copy is removed afterwards
