from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qwen_studio.gallery import Gallery, meta_from_png, params_from_workflow, png_size
from tests.conftest import sample_workflow, write_png


def test_png_size_and_workflow_meta(tmp_path: Path) -> None:
    path = write_png(
        tmp_path / "shot.png",
        width=64,
        height=128,
        workflow=sample_workflow("cathedral", "daylight"),
    )
    assert png_size(path) == (64, 128)
    meta = meta_from_png(path)
    assert meta["prompt"] == "cathedral"
    assert meta["negative_prompt"] == "daylight"
    assert meta["seed"] == 99
    assert meta["steps"] == 30
    assert meta["width"] == 1024
    assert meta["height"] == 1536


def test_params_from_workflow_order() -> None:
    params = params_from_workflow(sample_workflow())
    assert params["prompt"] == "a red cube"


def test_gallery_lists_newest_first_and_reads_sidecar(tmp_path: Path) -> None:
    older = write_png(tmp_path / "Qwen-Image-2512_00001_.png")
    newer = write_png(tmp_path / "Qwen-Image-2512_00002_.png")
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_800_000_000, 1_800_000_000))
    (tmp_path / "Qwen-Image-2512_00002_.json").write_text(
        json.dumps({"prompt": "from sidecar", "seed": 7, "steps": 36})
    )
    records = Gallery(tmp_path).list_images()
    assert [item.filename for item in records] == [
        newer.name,
        older.name,
    ]
    assert records[0].prompt == "from sidecar"
    assert records[0].seed == 7
    assert records[0].steps == 36


def test_gallery_falls_back_to_png_prompt(tmp_path: Path) -> None:
    write_png(
        tmp_path / "embedded.png",
        workflow=sample_workflow("embedded prompt", "neg"),
    )
    record = Gallery(tmp_path).list_images()[0]
    assert record.prompt == "embedded prompt"
    assert record.negative_prompt == "neg"


def test_resolve_rejects_path_escape(tmp_path: Path) -> None:
    write_png(tmp_path / "ok.png")
    gallery = Gallery(tmp_path)
    assert gallery.resolve("ok.png").name == "ok.png"
    with pytest.raises(ValueError):
        gallery.resolve("../secret.png")
    with pytest.raises(ValueError):
        gallery.resolve("sub/ok.png")
    with pytest.raises(FileNotFoundError):
        gallery.resolve("missing.png")


def test_delete_removes_asset_and_metadata(tmp_path: Path) -> None:
    image = write_png(tmp_path / "delete me.png")
    sidecar = image.with_suffix(".json")
    sidecar.write_text('{"prompt":"remove"}')
    gallery = Gallery(tmp_path)
    assert len(gallery.list_images()) == 1
    gallery.delete(image.name)
    assert not image.exists()
    assert not sidecar.exists()
    assert gallery.list_images() == []
    with pytest.raises(FileNotFoundError):
        gallery.delete(image.name)


def test_delete_preserves_shared_sidecar(tmp_path: Path) -> None:
    image = write_png(tmp_path / "same.png")
    (tmp_path / "same.webp").write_bytes(b"placeholder")
    image.with_suffix(".json").write_text('{"seed": 9}')
    Gallery(tmp_path).delete(image.name)
    assert (tmp_path / "same.json").is_file()


def test_delete_rejects_escapes_nonimages_and_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "gallery"
    root.mkdir()
    outside = write_png(tmp_path / "outside.png")
    (root / "linked.png").symlink_to(outside)
    (root / "private.json").write_text("secret")
    gallery = Gallery(root)
    for name in ("../outside.png", "private.json", "linked.png", "", "bad\x00.png"):
        with pytest.raises(ValueError):
            gallery.delete(name)
    assert outside.is_file()
    assert gallery.list_images() == []
    image = write_png(root / "valid.png")
    image.with_suffix(".json").symlink_to(outside)
    with pytest.raises(ValueError):
        gallery.delete(image.name)
    assert image.is_file()
    assert outside.is_file()


def test_gallery_survives_truncated_png_and_corrupt_metadata(tmp_path: Path) -> None:
    from tests.conftest import chunk
    (tmp_path / "partial.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "compressed.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"zTXt", b"prompt\x00\x00invalid")
    )
    write_png(tmp_path / "malformed.png", workflow={"1": None, "2": {"inputs": 42}})
    good = write_png(tmp_path / "good.png", workflow=sample_workflow())
    good.with_suffix(".json").write_bytes(b"\xff\xfe")
    records = Gallery(tmp_path).list_images()
    assert len(records) == 4
    assert next(record for record in records if record.filename == good.name).prompt == "a red cube"


def test_sampler_links_determine_prompt_roles() -> None:
    workflow = sample_workflow("negative by ID", "positive by link")
    workflow["8"]["inputs"].update({"positive": ["6", 0], "negative": ["5", 0]})
    params = params_from_workflow(workflow)
    assert params["prompt"] == "positive by link"
    assert params["negative_prompt"] == "negative by ID"


def test_metadata_cache_invalidates_for_sidecar_change_and_removal(tmp_path: Path, monkeypatch) -> None:
    path = write_png(tmp_path / "cached.png", workflow=sample_workflow())
    gallery = Gallery(tmp_path)
    original = gallery._record
    reads = []

    def record(image):
        reads.append(image)
        return original(image)

    monkeypatch.setattr(gallery, "_record", record)
    assert gallery.list_images()[0].prompt == "a red cube"
    assert gallery.list_images()[0].prompt == "a red cube"
    assert len(reads) == 1
    sidecar = path.with_suffix(".json")
    sidecar.write_text('{"prompt": "new prompt"}')
    assert gallery.list_images()[0].prompt == "new prompt"
    sidecar.unlink()
    assert gallery.list_images()[0].prompt == "a red cube"
    assert len(reads) == 3
    path.unlink()
    assert gallery.list_images() == []
    assert gallery._cache == {}


def test_filename_url_encoding_and_seed_precision(tmp_path: Path) -> None:
    image = write_png(tmp_path / "a #? ü.png")
    image.with_suffix(".json").write_text('{"seed": 9223372036854775807}')
    record = Gallery(tmp_path).list_images()[0]
    assert record.url == "/media/a%20%23%3F%20%C3%BC.png"
    assert record.seed == 9223372036854775807
    assert record.as_dict()["seed"] == "9223372036854775807"


def test_gallery_skips_file_removed_during_scan(tmp_path: Path, monkeypatch) -> None:
    path = write_png(tmp_path / "vanish.png")
    gallery = Gallery(tmp_path)

    def disappearing(image):
        image.unlink()
        raise FileNotFoundError(image)

    monkeypatch.setattr(gallery, "_record", disappearing)
    assert gallery.list_images() == []


def test_write_sidecar_rejects_symlink_and_outside_image(tmp_path: Path) -> None:
    root = tmp_path / "gallery"
    root.mkdir()
    image = write_png(root / "same.png")
    outside = write_png(tmp_path / "same.png")
    gallery = Gallery(root)
    with pytest.raises(ValueError):
        gallery.write_sidecar(outside, {"prompt": "no"})
    victim = tmp_path / "victim.json"
    victim.write_text("untouched")
    image.with_suffix(".json").symlink_to(victim)
    with pytest.raises(ValueError):
        gallery.write_sidecar(image, {"prompt": "no"})
    assert victim.read_text() == "untouched"
