from __future__ import annotations

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
