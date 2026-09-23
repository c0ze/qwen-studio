from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from qwen_studio.comfy import ComfyClient, resolve_seed
from qwen_studio.gallery import Gallery

Status = Literal["queued", "running", "done", "error", "cancelled"]


@dataclass
class Job:
    id: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    seed: int
    project: str = ""
    operation: str = "generate"
    sources: list[str] = field(default_factory=list)
    snapshots: list[Path] = field(default_factory=list, repr=False)
    workspace: str | None = field(default=None, repr=False)
    status: Status = "queued"
    error: str | None = None
    filename: str | None = None
    url: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project": self.project,
            "operation": self.operation,
            "sources": self.sources,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "seed": str(self.seed),
            "status": self.status,
            "error": self.error,
            "filename": self.filename,
            "url": self.url,
            "created_at": self.created_at,
        }


class JobRunner:
    def __init__(self, client: ComfyClient, gallery: Gallery) -> None:
        self.client = client
        self.gallery = gallery
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._stopping = False

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._stopping = False
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Drain queued work and let the active generation finish safely."""
        if self._stopping:
            if self._worker is not None:
                await asyncio.shield(self._worker)
            return
        self._stopping = True
        for job in self.jobs.values():
            if job.status == "queued":
                job.status = "cancelled"
                self._cleanup(job)
        if self._worker is not None and not self._worker.done():
            self._queue.put_nowait(None)
            await asyncio.shield(self._worker)
        else:
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
        self._worker = None

    def enqueue(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        *, project: str = "", operation: str = "generate", sources: list[str] | None = None,
    ) -> Job:
        if self._stopping:
            raise RuntimeError("Studio is shutting down")
        self.gallery.project_dir(project)
        job = Job(
            id=str(uuid.uuid4()),
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=resolve_seed(seed),
            project=project, operation=operation, sources=list(sources or []),
        )
        if job.sources:
            job.workspace = tempfile.mkdtemp(prefix="qwen-studio-job-")
            try:
                with self.gallery._lock:
                    for index, filename in enumerate(job.sources):
                        original = self.gallery.resolve(filename)
                        target = Path(job.workspace) / f"{index}{original.suffix}"
                        shutil.copyfile(original, target)
                        job.snapshots.append(target)
            except Exception:
                shutil.rmtree(job.workspace)
                raise
        self.jobs[job.id] = job
        self._queue.put_nowait(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def busy(self) -> bool:
        return any(job.status in {"queued", "running"} for job in self.jobs.values())

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        if job.status != "queued":
            raise ValueError("Only queued jobs can be cancelled")
        job.status = "cancelled"
        self._cleanup(job)
        return True

    @staticmethod
    def _cleanup(job: Job) -> None:
        if job.workspace:
            shutil.rmtree(job.workspace, ignore_errors=True)
            job.workspace = None

    async def _run(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = None
            try:
                if job_id is None:
                    return
                job = self.jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue
                job.status = "running"
                path = await asyncio.to_thread(self._generate, job)
                resolved = self.gallery.resolve(path.name).resolve()
                if path.resolve() != resolved:
                    raise ValueError("Generated image is outside the gallery")
                path = resolved
                from qwen_studio.imaging import load_image
                actual = await asyncio.to_thread(load_image, path)
                # Store actual dimensions, including imports and CPU transforms.
                if job.operation != "generate":
                    job.width, job.height = actual.size
                self.gallery.write_sidecar(
                    path,
                    {
                        "prompt": job.prompt,
                        "negative_prompt": job.negative_prompt,
                        "width": job.width,
                        "height": job.height,
                        "steps": job.steps,
                        "seed": job.seed,
                        "created_at": job.created_at,
                        "operation": job.operation,
                        "source": job.sources[0] if job.sources else None,
                    },
                )
                job.filename = self.gallery.move(path.name, job.project)
                job.url = f"/media/{quote(job.filename, safe='/')}"
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 — surface any gen failure to the UI
                if job is not None:
                    job.status = "error"
                    job.error = str(exc)
            finally:
                if job is not None:
                    self._cleanup(job)
                if job_id is not None and self._queue.empty():
                    # Other clients (the MCP server) share ComfyUI, so this
                    # only unloads when ComfyUI's own queue is empty too.
                    try:
                        await asyncio.to_thread(self.client.release_if_idle)
                    except Exception:  # noqa: BLE001 — freeing VRAM is best-effort
                        pass
                self._queue.task_done()

    def _generate(self, job: Job) -> Path:
        self.gallery.root.mkdir(parents=True, exist_ok=True)
        if job.operation in {"crop", "fit", "upscale"}:
            from qwen_studio.imaging import transform
            return transform(job.snapshots[0], self.gallery.root, job.operation, job.width, job.height)
        if job.operation in {"edit", "extend"}:
            from qwen_studio.editing import generate_edit
            return generate_edit(self.client, job.snapshots, job.prompt, job.negative_prompt,
                                 job.width, job.height, job.steps, job.seed, job.operation)
        return self.client.generate(
            job.prompt,
            job.negative_prompt,
            job.width,
            job.height,
            job.steps,
            job.seed,
        )
