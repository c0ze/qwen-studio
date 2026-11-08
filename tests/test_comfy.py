from __future__ import annotations

import json
import urllib.error

import pytest

from qwen_studio.comfy import ComfyClient, resolve_seed, validate_params


def test_validate_params_accepts_legal_values() -> None:
    assert validate_params("  a woman  ", 1024, 1536, 30, 0) == "a woman"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": " ", "width": 1024, "height": 1024, "steps": 30, "seed": 0},
        {"prompt": "x", "width": 1000, "height": 1024, "steps": 30, "seed": 0},
        {"prompt": "x", "width": 1024, "height": 256, "steps": 30, "seed": 0},
        {"prompt": "x", "width": 1024, "height": 1024, "steps": 3, "seed": 0},
        {"prompt": "x", "width": 1024, "height": 1024, "steps": 30, "seed": -1},
    ],
)
def test_validate_params_rejects_bad_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        validate_params(**kwargs)


def test_resolve_seed_randomizes_zero() -> None:
    assert resolve_seed(0) > 0
    assert resolve_seed(42) == 42


def history_client(monkeypatch, outputs, status=None):
    client = ComfyClient()
    monkeypatch.setattr(client, "request_json", lambda *a, **kw: {
        "job": {"outputs": outputs, "status": status or {"completed": True}}
    })
    return client


def test_wait_selects_final_saved_image_over_preview(monkeypatch, tmp_path):
    archive = tmp_path / "LMStudio"
    archive.mkdir()
    final = archive / "final.png"
    final.write_bytes(b"saved")
    (tmp_path / "preview.png").write_bytes(b"preview")
    client = history_client(monkeypatch, {
        "9": {"images": [{"filename": "preview.png", "type": "temp"}]},
        "10": {"images": [{"filename": "final.png", "subfolder": "LMStudio", "type": "output"}]},
    })
    assert client._wait_for_image("job", tmp_path) == final


@pytest.mark.parametrize("image", [
    {"filename": "asset.png", "subfolder": "LMStudio", "type": "temp"},
    {"filename": "asset.png", "subfolder": "", "type": "output"},
    {"filename": "../asset.png", "subfolder": "LMStudio", "type": "output"},
    {"filename": "missing.png", "subfolder": "LMStudio", "type": "output"},
])
def test_completed_without_valid_output_fails_promptly(monkeypatch, tmp_path, image):
    (tmp_path / "LMStudio").mkdir()
    (tmp_path / "asset.png").write_bytes(b"outside archive")
    (tmp_path / "LMStudio" / "asset.png").write_bytes(b"saved")
    client = history_client(monkeypatch, {"10": {"images": [image]}})
    with pytest.raises(RuntimeError, match="without a valid image"):
        client._wait_for_image("job", tmp_path)


def test_completed_preview_only_is_not_success(monkeypatch, tmp_path):
    client = history_client(monkeypatch, {"9": {"images": []}})
    with pytest.raises(RuntimeError, match="without a valid image"):
        client._wait_for_image("job", tmp_path)


def test_saved_symlink_cannot_escape_archive(monkeypatch, tmp_path):
    archive = tmp_path / "LMStudio"
    archive.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (archive / "link.png").symlink_to(outside)
    client = history_client(monkeypatch, {"10": {"images": [
        {"filename": "link.png", "subfolder": "LMStudio", "type": "output"}
    ]}})
    with pytest.raises(RuntimeError, match="without a valid image"):
        client._wait_for_image("job", tmp_path)


def test_execution_error_surfaces_details(monkeypatch, tmp_path):
    client = history_client(monkeypatch, {}, {
        "status_str": "error", "messages": ["out of memory"]
    })
    with pytest.raises(RuntimeError, match="out of memory"):
        client._wait_for_image("job", tmp_path)


def test_launcher_closes_parent_log_handle(monkeypatch, tmp_path):
    from qwen_studio import comfy

    launcher = tmp_path / "run.sh"
    launcher.write_text("#!/bin/sh\n")
    monkeypatch.setattr(comfy, "LOG_PATH", tmp_path / "comfy.log")
    client = ComfyClient(run_script=launcher)
    readiness = iter([False, True])
    monkeypatch.setattr(client, "ready", lambda: next(readiness))
    handles = []
    monkeypatch.setattr(comfy.subprocess, "Popen", lambda *a, **kw: handles.append(kw["stdout"]))
    client.ensure()
    assert handles[0].closed


def test_repeated_generation_saves_a_fresh_asset(monkeypatch, tmp_path):
    client = ComfyClient()
    monkeypatch.setattr(client, 'ensure', lambda: None)
    submissions = []
    def submit(*args, **kwargs):
        submissions.append(kwargs['payload']['prompt'])
        return {'prompt_id': 'test'}
    monkeypatch.setattr(client, 'request_json', submit)
    monkeypatch.setattr(client, '_wait_for_image', lambda *args: tmp_path / 'image.png')
    for _ in range(2):
        client.generate('same prompt', '', 512, 512, 4, 42)
    first, second = submissions
    assert first['10']['inputs']['filename_prefix'] != second['10']['inputs']['filename_prefix']
    assert first['10']['inputs']['filename_prefix'].startswith('LMStudio/Qwen-Image-2512-')
    assert {k: v for k, v in first.items() if k != '10'} == {k: v for k, v in second.items() if k != '10'}


def test_release_if_idle_unloads_only_when_comfy_is_idle(monkeypatch):
    client = ComfyClient()
    calls = []
    queue = {"queue_running": [["busy"]], "queue_pending": []}

    def fake(path, **kwargs):
        calls.append(path)
        return queue if path == "/queue" else {}

    monkeypatch.setattr(client, "request_json", fake)
    assert not client.release_if_idle()
    queue["queue_running"] = []
    assert client.release_if_idle()
    assert calls == ["/queue", "/queue", "/free"]


def test_release_if_idle_tolerates_offline_comfy(monkeypatch):
    client = ComfyClient()

    def offline(*args, **kwargs):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(client, "request_json", offline)
    assert not client.release_if_idle()


def test_qwen_remains_the_default_workflow():
    graph = ComfyClient().workflow("tower", "blurry", 1024, 1024, 30, 7)
    assert graph["1"]["class_type"] == "UnetLoaderGGUF"
    assert graph["3"]["inputs"]["type"] == "qwen_image"


def test_moss_workflow_follows_reference_graph():
    from qwen_studio import moss

    graph = ComfyClient().workflow("tower", "", 1024, 1536, 25, 7, model="moss")
    assert graph["1"]["class_type"] == "UNETLoader"
    assert graph["1"]["inputs"]["unet_name"] == moss.MODEL
    # Skip-layer guidance on layer 22 for the first half replaces the
    # reference pipeline's uncond_skip_layers=[22].
    assert graph["2"]["class_type"] == "SkipLayerGuidanceDiTSimple"
    assert graph["2"]["inputs"]["single_layers"] == "22"
    assert graph["2"]["inputs"]["end_percent"] == 0.5
    assert graph["3"]["inputs"] == {"clip_name": moss.ENCODER, "type": "moss_image", "device": "default"}
    assert graph["4"]["inputs"]["vae_name"] == moss.VAE
    assert graph["7"]["class_type"] == "EmptyFlux2LatentImage"
    assert (graph["7"]["inputs"]["width"], graph["7"]["inputs"]["height"]) == (1024, 1536)
    sampler = graph["8"]["inputs"]
    assert (sampler["cfg"], sampler["sampler_name"], sampler["steps"], sampler["seed"]) == (2.5, "er_sde", 25, 7)
    assert sampler["model"] == ["2", 0]
    assert graph["10"]["class_type"] == "SaveImage"


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        ComfyClient().workflow("tower", "", 1024, 1024, 25, 7, model="sdxl")


def test_moss_generation_saves_under_its_own_prefix(monkeypatch, tmp_path):
    client = ComfyClient()
    monkeypatch.setattr(client, "ensure", lambda: None)
    submissions = []
    monkeypatch.setattr(client, "request_json", lambda *a, **kw: submissions.append(kw["payload"]["prompt"]) or {"prompt_id": "x"})
    monkeypatch.setattr(client, "_wait_for_image", lambda *args: tmp_path / "image.png")
    client.generate("tower", "", 1024, 1024, 25, 7, model="moss")
    assert submissions[0]["1"]["class_type"] == "UNETLoader"
    assert submissions[0]["10"]["inputs"]["filename_prefix"].startswith("LMStudio/MOSS-Image-")


def test_nitro_one_step_follows_published_graph():
    from qwen_studio import nitro

    graph = ComfyClient().workflow("tower", "ignored", 1024, 1536, 1, 7, model="nitro")
    assert graph["1"] == {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": nitro.MODEL}}
    assert graph["2"]["inputs"]["shift"] == 3.0
    assert graph["3"]["inputs"] == {"clip_name": nitro.ENCODER, "type": "lumina2"}
    # CFG 1: the negative is the zeroed positive, never the user's text.
    assert graph["6"] == {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}}
    assert "ignored" not in json.dumps(graph)
    assert (graph["7"]["inputs"]["width"], graph["7"]["inputs"]["height"]) == (1024, 1536)
    sampler = graph["8"]["inputs"]
    assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["seed"]) == (1, 1.0, "euler", 7)
    assert graph["9"]["inputs"]["samples"] == ["8", 0]
    assert graph["10"]["class_type"] == "SaveImage"
    assert not any(node["class_type"] == "LoraLoaderModelOnly" for node in graph.values())


def test_nitro_two_steps_use_the_dedistill_lora_on_the_first_step():
    from qwen_studio import nitro

    graph = ComfyClient().workflow("tower", "", 1024, 1024, 2, 7, model="nitro")
    assert graph["11"]["inputs"] == {"model": ["1", 0], "lora_name": nitro.LORA, "strength_model": 0.5}
    assert graph["12"]["inputs"]["shift"] == graph["2"]["inputs"]["shift"] == 6.0
    first, second = graph["8"]["inputs"], graph["13"]["inputs"]
    assert graph["8"]["class_type"] == graph["13"]["class_type"] == "KSamplerAdvanced"
    assert (first["model"], first["start_at_step"], first["end_at_step"]) == (["12", 0], 0, 1)
    assert (first["add_noise"], first["return_with_leftover_noise"]) == ("enable", "enable")
    assert (second["model"], second["latent_image"], second["start_at_step"], second["end_at_step"]) == (["2", 0], ["8", 0], 1, 2)
    assert second["add_noise"] == "disable"
    assert first["noise_seed"] == second["noise_seed"] == 7
    for sampler in (first, second):
        assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]) == (2, 1.0, "euler", "simple")
    assert second["return_with_leftover_noise"] == "disable"
    assert graph["9"]["inputs"]["samples"] == ["13", 0]


@pytest.mark.parametrize("steps", [0, 3, 30])
def test_nitro_refuses_other_step_counts(steps):
    with pytest.raises(ValueError, match="1 or 2 steps"):
        ComfyClient().workflow("tower", "", 1024, 1024, steps, 7, model="nitro")


def test_step_limits_depend_on_the_model():
    assert validate_params("tower", 1024, 1024, 1, 7, "nitro") == "tower"
    with pytest.raises(ValueError, match="from 1 through 2"):
        validate_params("tower", 1024, 1024, 4, 7, "nitro")
    with pytest.raises(ValueError, match="from 4 through 50"):
        validate_params("tower", 1024, 1024, 1, 7)


def test_nitro_generation_saves_under_its_own_prefix(monkeypatch, tmp_path):
    client = ComfyClient()
    monkeypatch.setattr(client, "ensure", lambda: None)
    submissions = []
    monkeypatch.setattr(client, "request_json", lambda *a, **kw: submissions.append(kw["payload"]["prompt"]) or {"prompt_id": "x"})
    monkeypatch.setattr(client, "_wait_for_image", lambda *args: tmp_path / "image.png")
    client.generate("tower", "", 1024, 1024, 1, 7, model="nitro")
    assert submissions[0]["10"]["inputs"]["filename_prefix"].startswith("LMStudio/NitroFusion2-")
