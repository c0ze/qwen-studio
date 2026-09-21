from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Literal

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qwen_studio.comfy import (
    DEFAULT_NEGATIVE,
    GALLERY_DIR,
    ComfyClient,
    validate_params,
)
from qwen_studio.gallery import Gallery
from qwen_studio.jobs import JobRunner

STATIC_DIR = Path(__file__).parent / "static"


class GenerateRequest(BaseModel):
    prompt: str = ""
    project: str = ""
    operation: Literal["generate", "edit", "extend", "crop", "fit", "upscale"] = "generate"
    sources: list[str] = Field(default_factory=list, max_length=3)
    negative_prompt: str = DEFAULT_NEGATIVE
    width: int = 1024
    height: int = 1024
    steps: int = Field(default=30, ge=4, le=50)
    seed: int = 0


class ProjectRequest(BaseModel):
    name: str


class MoveRequest(BaseModel):
    filename: str
    project: str = ""


def create_app(
    client: ComfyClient | None = None,
    gallery: Gallery | None = None,
) -> FastAPI:
    client = client or ComfyClient()
    gallery = gallery or Gallery(GALLERY_DIR)
    runner = JobRunner(client, gallery)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runner.start()
        try:
            yield
        finally:
            await runner.stop()

    app = FastAPI(title="Qwen Studio", lifespan=lifespan)
    app.state.client = client
    app.state.gallery = gallery
    app.state.runner = runner

    @app.middleware("http")
    async def same_origin_writes(request: Request, call_next):
        # A local API must not accept writes initiated by an unrelated website.
        if request.method in {"POST", "DELETE", "PUT", "PATCH"}:
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if (origin is not None and origin != expected) or request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "cross-origin writes are not allowed"}, status_code=403)
        response = await call_next(request)
        if request.url.path.startswith("/api/") or request.url.path in {"/", "/static/app.js", "/static/style.css"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def invalid(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing(_request, exc):
        return JSONResponse({"detail": "asset or project not found"}, status_code=404)

    @app.get("/api/capabilities")
    def available_tools():
        from qwen_studio.editing import capabilities
        return capabilities(client)

    @app.get("/api/projects")
    def projects():
        return {"projects": gallery.projects()}

    @app.post("/api/projects")
    def create_project(body: ProjectRequest):
        return gallery.save_project(body.name)

    @app.patch("/api/projects/{project}")
    def rename_project(project: str, body: ProjectRequest):
        return gallery.save_project(body.name, project)

    @app.post("/api/images/move")
    def move_image(body: MoveRequest):
        filename = gallery.move(body.filename, body.project)
        return {"filename": filename}

    @app.post("/api/import")
    async def import_image(request: Request, project: str = ""):
        from qwen_studio.imaging import MAX_UPLOAD, load_image, new_path
        folder = gallery.project_dir(project)
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "image exceeds the 20 MB upload limit")
        def save():
            image = load_image(bytes(data))
            path = new_path(folder, "import")
            image.save(path, format="PNG")
            gallery.write_sidecar(path, {"operation": "import", "width": image.width, "height": image.height})
            return {"filename": gallery.asset_id(path)}
        return await asyncio.to_thread(save)

    @app.get("/api/status")
    async def status() -> dict:
        return {
            "comfy": await asyncio.to_thread(client.ready),
            "busy": runner.busy(),
            "gallery": str(gallery.root),
        }

    @app.get("/api/images")
    def images() -> dict:
        return {"images": [item.as_dict() for item in gallery.list_images()]}

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str) -> dict:
        found = runner.get(job_id)
        if found is None:
            raise HTTPException(404, "job not found")
        return found.as_dict()

    @app.get("/api/jobs")
    async def jobs() -> dict:
        return {"jobs": [job.as_dict() for job in reversed(list(runner.jobs.values()))]}

    @app.delete("/api/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict:
        if runner.get(job_id) is None:
            raise HTTPException(404, "job not found")
        try:
            runner.cancel(job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return runner.get(job_id).as_dict()

    @app.delete("/api/images/{filename:path}")
    def delete_image(filename: str) -> dict:
        try:
            gallery.delete(filename)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "image not found") from exc
        except OSError as exc:
            raise HTTPException(500, "Could not delete asset; check folder permissions and retry") from exc
        return {"deleted": filename}

    @app.post("/api/generate")
    async def generate(body: GenerateRequest) -> dict:
        gallery.project_dir(body.project)
        if body.operation in {"generate", "edit", "extend"}:
            prompt = validate_params(body.prompt, body.width, body.height, body.steps, body.seed)
        else:
            prompt = body.prompt.strip()
            if not (64 <= body.width <= 4096 and 64 <= body.height <= 4096):
                raise ValueError("dimensions must be from 64 to 4096")
            if not 0 <= body.seed < 2**63:
                raise ValueError("invalid seed")
        if body.operation == "generate" and body.sources:
            raise ValueError("choose reference/edit mode to use source images")
        if body.operation != "generate" and not body.sources:
            raise ValueError("select a source image")
        if body.operation in {"extend", "crop", "fit", "upscale"} and len(body.sources) != 1:
            raise ValueError("this operation takes one source image")
        for source in body.sources:
            gallery.resolve(source)
        if body.operation in {"edit", "extend"}:
            from qwen_studio.editing import capabilities
            cap = await asyncio.to_thread(capabilities, client)
            if not cap["edit"]:
                raise HTTPException(409, cap["reason"])
        if body.operation == "extend":
            from qwen_studio.imaging import load_image
            source = await asyncio.to_thread(load_image, gallery.resolve(body.sources[0]))
            if body.width < source.width or body.height < source.height or (body.width, body.height) == source.size:
                raise ValueError("extend needs a larger canvas, without shrinking either dimension")
        job = runner.enqueue(
            prompt,
            body.negative_prompt.strip(),
            body.width,
            body.height,
            body.steps,
            body.seed,
            project=body.project, operation=body.operation, sources=body.sources,
        )
        return job.as_dict()

    @app.get("/media/{filename:path}")
    def media(filename: str) -> FileResponse:
        try:
            path = gallery.resolve(filename)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "not found") from exc
        return FileResponse(path)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("QWEN_STUDIO_HOST", "127.0.0.1")
    port = int(os.environ.get("QWEN_STUDIO_PORT", "8790"))
    uvicorn.run("qwen_studio.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
