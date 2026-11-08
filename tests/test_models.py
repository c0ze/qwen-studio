from __future__ import annotations

import hashlib

import pytest

from qwen_studio import models, moss


def test_existing_verified_files_are_not_downloaded(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(models, "COMFY_DIR", tmp_path)
    data = b"weights"
    (tmp_path / "models/vae").mkdir(parents=True)
    (tmp_path / "models/vae/file.safetensors").write_bytes(data)
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("downloaded"))
    models.install("vae", "file.safetensors", "https://example.invalid", hashlib.sha256(data).hexdigest())
    assert "Already installed" in capsys.readouterr().out


def test_moss_bundle_installs_all_three_files(monkeypatch):
    installed = []
    monkeypatch.setattr(models, "install", lambda *entry: installed.append(entry))
    models.main(["moss"])
    assert [(folder, name) for folder, name, _url, _sha in installed] == [
        ("diffusion_models", "moss_image_sft_int8_convrot.safetensors"),
        ("text_encoders", moss.ENCODER),
        ("vae", moss.VAE),
    ]
    assert all(url.startswith("https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/") for _f, _n, url, _s in installed)


def test_default_bundle_is_qwen_edit(monkeypatch):
    installed = []
    monkeypatch.setattr(models, "install", lambda *entry: installed.append(entry))
    models.main([])
    assert installed == [("diffusion_models", models.NAME, models.URL, models.SHA256)]


def test_unknown_bundle_is_refused():
    with pytest.raises(SystemExit, match="usage"):
        models.main(["sdxl"])


def test_nitro_bundle_installs_model_encoder_lora_and_vae(monkeypatch):
    from qwen_studio import nitro
    installed = []
    monkeypatch.setattr(models, "install", lambda *entry: installed.append(entry))
    models.main(["nitro"])
    assert [(folder, name) for folder, name, _url, _sha in installed] == [
        ("diffusion_models", nitro.MODEL),
        ("text_encoders", nitro.ENCODER),
        ("loras", nitro.LORA),
        ("vae", nitro.VAE),
    ]
    assert all(url.endswith(name) and len(sha) == 64 for _f, name, url, sha in installed)
