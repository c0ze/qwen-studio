from __future__ import annotations

import json
import time

import pytest

from tests.conftest import write_png


def _wait_job(client, job_id: str) -> dict:
    for _ in range(50):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"done", "error"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {payload}")


def test_status_and_empty_gallery(studio) -> None:
    client, fake, _gallery = studio
    status = client.get("/api/status").json()
    assert status["comfy"] is True
    assert status["busy"] is False
    assert client.get("/api/images").json() == {"images": []}
    fake._ready = False
    assert client.get("/api/status").json()["comfy"] is False


def test_generate_rejects_empty_prompt(studio) -> None:
    client, _fake, _gallery = studio
    response = client.post("/api/generate", json={"prompt": "  "})
    assert response.status_code == 400


def test_generate_writes_sidecar_and_lists_image(studio) -> None:
    client, fake, gallery = studio
    response = client.post(
        "/api/generate",
        json={
            "prompt": "teal cemetery",
            "negative_prompt": "daylight",
            "width": 1024,
            "height": 1536,
            "steps": 36,
            "seed": 11,
        },
    )
    assert response.status_code == 200
    job_id = response.json()["id"]
    assert response.json()["seed"] == "11"
    done = _wait_job(client, job_id)
    assert done["status"] == "done"
    assert done["filename"] == "Qwen-Image-2512_00001_.png"
    assert fake.calls[0]["prompt"] == "teal cemetery"
    assert fake.calls[0]["seed"] == 11
    listed = client.get("/api/images").json()["images"]
    assert listed[0]["prompt"] == "teal cemetery"
    sidecar = json.loads(
        (gallery.root / "Qwen-Image-2512_00001_.json").read_text()
    )
    assert sidecar["steps"] == 36
    media = client.get("/media/Qwen-Image-2512_00001_.png")
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("image/")


def test_generate_randomizes_seed_zero(studio) -> None:
    client, fake, _gallery = studio
    job_id = client.post("/api/generate", json={"prompt": "x", "seed": 0}).json()["id"]
    done = _wait_job(client, job_id)
    assert done["status"] == "done"
    assert fake.calls[0]["seed"] > 0


def test_generate_error_surfaces(studio) -> None:
    client, fake, _gallery = studio
    fake.fail = True
    job_id = client.post("/api/generate", json={"prompt": "x"}).json()["id"]
    done = _wait_job(client, job_id)
    assert done["status"] == "error"
    assert "exploded" in done["error"]


def test_media_rejects_escape_and_missing(studio) -> None:
    client, _fake, gallery = studio
    write_png(gallery.root / "ok.png")
    assert client.get("/media/../pyproject.toml").status_code in {400, 404}
    assert client.get("/media/missing.png").status_code == 404


def test_job_not_found(studio) -> None:
    client, _fake, _gallery = studio
    assert client.get("/api/jobs/nope").status_code == 404


def test_index_served(studio) -> None:
    client, _fake, _gallery = studio
    response = client.get("/")
    assert response.status_code == 200
    assert b"Qwen Studio" in response.content


def test_delete_removes_image_sidecar_and_media(studio):
    client, _, gallery = studio
    image = write_png(gallery.root / 'delete me #1.png')
    sidecar = image.with_suffix('.json')
    sidecar.write_text('{"prompt":"delete test"}')
    record = client.get('/api/images').json()['images'][0]
    assert client.get(record['url']).status_code == 200
    response = client.delete('/api/images/delete%20me%20%231.png')
    assert response.status_code == 200
    assert not image.exists()
    assert not sidecar.exists()
    assert client.get(record['url']).status_code == 404
    assert client.get('/api/images').json()['images'] == []
    assert client.delete('/api/images/delete%20me%20%231.png').status_code == 404


def test_delete_rejects_non_images_and_symlinks(studio, tmp_path):
    client, _, gallery = studio
    secret = gallery.root / 'keep.json'
    secret.write_text('{}')
    link = gallery.root / 'link.png'
    link.symlink_to(secret)
    assert client.delete('/api/images/keep.json').status_code == 400
    assert client.delete('/api/images/link.png').status_code in {400, 404}
    assert secret.exists()
    assert link.is_symlink()


def test_delete_permission_error_is_visible(studio, monkeypatch):
    client, _, gallery = studio
    def fail(_filename):
        raise PermissionError('read only')
    monkeypatch.setattr(gallery, 'delete', fail)
    response = client.delete('/api/images/test.png')
    assert response.status_code == 500
    assert 'permissions' in response.json()['detail']


def test_empty_negative_and_exact_seed_roundtrip(studio):
    client, fake, _ = studio
    response = client.post('/api/generate', json={
        'prompt': 'repeatable', 'negative_prompt': '', 'seed': '9223372036854775807',
    })
    assert response.status_code == 200
    done = _wait_job(client, response.json()['id'])
    assert done['seed'] == '9223372036854775807'
    assert fake.calls[0]['seed'] == 9223372036854775807
    assert fake.calls[0]['negative_prompt'] == ''
    image = client.get('/api/images').json()['images'][0]
    assert image['seed'] == '9223372036854775807'
    assert image['negative_prompt'] == ''


def test_jobs_visible_after_reconnect(studio):
    client, _, _ = studio
    job = client.post('/api/generate', json={'prompt': 'queue test'}).json()
    _wait_job(client, job['id'])
    assert client.get('/api/jobs').json()['jobs'][0]['id'] == job['id']
    assert client.delete('/api/jobs/missing').status_code == 404
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 409


def test_rejects_cross_origin_writes(studio):
    client, fake, gallery = studio
    image = write_png(gallery.root / 'keep.png')
    assert client.delete('/api/images/keep.png', headers={'Origin': 'https://evil.example'}).status_code == 403
    assert client.post('/api/generate', json={'prompt': 'x'}, headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403
    assert image.exists()
    assert not fake.calls
    assert client.delete('/api/images/keep.png', headers={'Origin': 'http://testserver'}).status_code == 200


def _moss_ready(fake) -> None:
    from qwen_studio import moss
    fake.object_info = {
        "UNETLoader": {"input": {"required": {"unet_name": [[moss.MODEL]]}}},
        "CLIPLoader": {"input": {"required": {"clip_name": [[moss.ENCODER]], "type": [["moss_image"]]}}},
        "VAELoader": {"input": {"required": {"vae_name": [[moss.VAE]]}}},
    }


def test_generate_with_moss_records_the_model(studio) -> None:
    client, fake, gallery = studio
    _moss_ready(fake)
    response = client.post("/api/generate", json={"prompt": "tower", "model": "moss", "negative_prompt": "", "steps": 25, "seed": 5})
    assert response.status_code == 200
    assert response.json()["model"] == "moss"
    done = _wait_job(client, response.json()["id"])
    assert done["status"] == "done"
    assert fake.calls[0]["model"] == "moss"
    sidecar = json.loads((gallery.root / done["filename"]).with_suffix(".json").read_text())
    assert sidecar["model"] == "moss"
    assert client.get("/api/images").json()["images"][0]["model"] == "moss"


def test_qwen_is_the_default_model(studio) -> None:
    client, fake, _gallery = studio
    done = _wait_job(client, client.post("/api/generate", json={"prompt": "tower"}).json()["id"])
    assert done["model"] == "qwen"
    assert fake.calls[0]["model"] == "qwen"


def test_moss_unavailable_is_a_conflict(studio) -> None:
    client, fake, _gallery = studio
    response = client.post("/api/generate", json={"prompt": "tower", "model": "moss"})
    assert response.status_code == 409
    assert not fake.calls


def test_moss_only_generates_from_text(studio) -> None:
    client, fake, gallery = studio
    _moss_ready(fake)
    write_png(gallery.root / "source.png")
    response = client.post("/api/generate", json={"prompt": "night", "model": "moss", "operation": "edit", "sources": ["source.png"]})
    assert response.status_code == 400
    assert "MOSS" in response.json()["detail"]


def test_unknown_model_is_rejected(studio) -> None:
    client, _fake, _gallery = studio
    assert client.post("/api/generate", json={"prompt": "tower", "model": "sdxl"}).status_code == 422


def test_capabilities_report_moss(studio) -> None:
    client, fake, _gallery = studio
    assert client.get("/api/capabilities").json()["moss"] is False
    _moss_ready(fake)
    assert client.get("/api/capabilities").json()["moss"] is True


def test_moss_queues_while_comfy_is_offline(studio) -> None:
    # Studio starts ComfyUI on demand, so an offline ComfyUI must not block
    # MOSS any more than it blocks Qwen text generation.
    client, fake, _gallery = studio
    fake.object_info = None
    response = client.post("/api/generate", json={"prompt": "tower", "model": "moss"})
    assert response.status_code == 200
    assert _wait_job(client, response.json()["id"])["status"] == "done"
    assert fake.calls[0]["model"] == "moss"


def _nitro_ready(fake) -> None:
    from qwen_studio import nitro
    fake.object_info = {
        "UnetLoaderGGUF": {"input": {"required": {"unet_name": [[nitro.MODEL]]}}},
        "CLIPLoaderGGUF": {"input": {"required": {"clip_name": [[nitro.ENCODER]], "type": [["lumina2"]]}}},
        "VAELoader": {"input": {"required": {"vae_name": [[nitro.VAE]]}}},
        "LoraLoaderModelOnly": {"input": {"required": {"lora_name": [[nitro.LORA]]}}},
    }


def test_generate_with_nitro_in_one_step(studio) -> None:
    client, fake, gallery = studio
    _nitro_ready(fake)
    response = client.post("/api/generate", json={"prompt": "tower", "model": "nitro", "negative_prompt": "", "steps": 1, "seed": 5})
    assert response.status_code == 200
    done = _wait_job(client, response.json()["id"])
    assert done["status"] == "done"
    assert (fake.calls[0]["model"], fake.calls[0]["steps"]) == ("nitro", 1)
    sidecar = json.loads((gallery.root / done["filename"]).with_suffix(".json").read_text())
    assert sidecar["model"] == "nitro"


def test_nitro_rejects_qwen_step_counts(studio) -> None:
    client, fake, _gallery = studio
    _nitro_ready(fake)
    response = client.post("/api/generate", json={"prompt": "tower", "model": "nitro", "steps": 30})
    assert response.status_code == 400
    assert not fake.calls


def test_qwen_still_rejects_one_step(studio) -> None:
    client, fake, _gallery = studio
    assert client.post("/api/generate", json={"prompt": "tower", "steps": 1}).status_code == 400
    assert not fake.calls


def test_nitro_unavailable_is_a_conflict(studio) -> None:
    client, fake, _gallery = studio
    response = client.post("/api/generate", json={"prompt": "tower", "model": "nitro", "steps": 1})
    assert response.status_code == 409
    assert "Install" in response.json()["detail"] or "ComfyUI-GGUF" in response.json()["detail"]
    assert not fake.calls


def test_nitro_only_generates_from_text(studio) -> None:
    client, fake, gallery = studio
    _nitro_ready(fake)
    write_png(gallery.root / "source.png")
    response = client.post("/api/generate", json={"prompt": "night", "model": "nitro", "operation": "edit", "sources": ["source.png"]})
    assert response.status_code == 400
    assert "NitroFusion 2" in response.json()["detail"]


@pytest.mark.parametrize("payload", [
    {"model": "moss", "steps": 2},
    {"operation": "edit", "steps": 2, "sources": ["source.png"]},
    {"operation": "extend", "steps": 3, "sources": ["source.png"]},
])
def test_only_nitro_takes_fewer_than_four_steps(studio, payload) -> None:
    client, fake, gallery = studio
    _moss_ready(fake)
    write_png(gallery.root / "source.png")
    response = client.post("/api/generate", json={"prompt": "tower", **payload})
    assert response.status_code == 400
    assert "4 through 50" in response.json()["detail"]
    assert not fake.calls


@pytest.mark.parametrize("operation", ["crop", "fit", "upscale"])
def test_local_transforms_accept_a_nitro_assets_step_count(studio, operation) -> None:
    # Transforms resend the source's stored steps; a NitroFusion image has 1.
    client, _fake, gallery = studio
    write_png(gallery.root / "source.png")
    response = client.post("/api/generate", json={"operation": operation, "sources": ["source.png"],
                                                  "steps": 1, "width": 768, "height": 512})
    assert response.status_code == 200
