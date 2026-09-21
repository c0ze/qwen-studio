from __future__ import annotations

import asyncio
import json
import threading

import pytest

from qwen_studio.gallery import Gallery
from qwen_studio.jobs import JobRunner
from tests.conftest import FakeComfy, write_png


def enqueue(runner, seed=11):
    return runner.enqueue("a cube", "", 1024, 1024, 30, seed)


def test_seed_serialization_preserves_63_bits(tmp_path):
    runner = JobRunner(FakeComfy(tmp_path), Gallery(tmp_path))
    job = enqueue(runner, 2**63 - 1)
    assert job.as_dict()["seed"] == "9223372036854775807"


def test_cancelled_and_missing_jobs_drain_without_generation(tmp_path):
    async def run():
        fake = FakeComfy(tmp_path)
        runner = JobRunner(fake, Gallery(tmp_path))
        cancelled = enqueue(runner)
        assert runner.cancel(cancelled.id)
        with pytest.raises(ValueError, match="Only queued"):
            runner.cancel(cancelled.id)
        assert not runner.cancel("missing")
        runner._queue.put_nowait("missing")
        kept = enqueue(runner)
        runner.start()
        await asyncio.wait_for(runner._queue.join(), 2)
        await runner.stop()
        assert cancelled.status == "cancelled"
        assert kept.status == "done"
        assert len(fake.calls) == 1
        assert not runner.busy()
    asyncio.run(run())


def test_stop_finishes_running_job_and_cancels_queue(tmp_path):
    async def run():
        started = threading.Event()
        release = threading.Event()

        class BlockingComfy(FakeComfy):
            def generate(self, *args):
                started.set()
                assert release.wait(timeout=3)
                return super().generate(*args)

        fake = BlockingComfy(tmp_path)
        runner = JobRunner(fake, Gallery(tmp_path))
        active = enqueue(runner)
        queued = enqueue(runner)
        runner.start()
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.raises(ValueError, match="Only queued"):
            runner.cancel(active.id)
        stopping = asyncio.create_task(runner.stop())
        await asyncio.sleep(0)
        assert queued.status == "cancelled"
        assert not stopping.done()
        with pytest.raises(RuntimeError, match="shutting down"):
            enqueue(runner)
        release.set()
        await asyncio.wait_for(stopping, 2)
        await asyncio.wait_for(runner._queue.join(), 2)
        assert active.status == "done"
        assert json.loads((tmp_path / active.filename).with_suffix(".json").read_text())["seed"] == 11
        assert len(fake.calls) == 1
        assert runner._worker is None
        await runner.stop()
    asyncio.run(run())


def test_generated_external_path_never_writes_sidecar(tmp_path):
    async def run():
        archive = tmp_path / "archive"
        archive.mkdir()
        outside = write_png(tmp_path / "image.png")
        write_png(archive / "image.png")

        class ExternalComfy:
            def generate(self, *args):
                return outside

        runner = JobRunner(ExternalComfy(), Gallery(archive))
        job = enqueue(runner)
        runner.start()
        await asyncio.wait_for(runner._queue.join(), 2)
        await runner.stop()
        assert job.status == "error"
        assert "outside the gallery" in job.error
        assert not outside.with_suffix(".json").exists()
        assert not (archive / "image.json").exists()
    asyncio.run(run())


def test_reserved_filename_gets_escaped_media_url(tmp_path):
    async def run():
        path = write_png(tmp_path / "image #1?.png")

        class NamedComfy:
            def generate(self, *args):
                return path

        runner = JobRunner(NamedComfy(), Gallery(tmp_path))
        job = enqueue(runner)
        runner.start()
        await asyncio.wait_for(runner._queue.join(), 2)
        await runner.stop()
        assert job.status == "done"
        assert job.url == "/media/image%20%231%3F.png"
    asyncio.run(run())


def test_relative_gallery_root_accepts_generated_image(tmp_path, monkeypatch):
    async def run():
        monkeypatch.chdir(tmp_path)
        from pathlib import Path
        archive = Path('archive')
        archive.mkdir()
        runner = JobRunner(FakeComfy(archive), Gallery(archive))
        job = enqueue(runner)
        runner.start()
        await asyncio.wait_for(runner._queue.join(), 2)
        await runner.stop()
        assert job.status == 'done'
    asyncio.run(run())
