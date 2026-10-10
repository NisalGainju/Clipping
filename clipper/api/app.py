"""FastAPI backend: submit a URL/file, poll progress, list and download clips."""
from __future__ import annotations

import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Config, load_config
from ..errors import ClipperError
from ..pipeline import Pipeline

STATIC = Path(__file__).parent / "static"
BASE_CFG: Config = load_config(os.environ.get("CLIPPER_CONFIG"),
                               {"work_dir": os.environ.get("CLIPPER_WORK_DIR")})

app = FastAPI(title="Viral Clipper", version="0.1.0")
app.mount("/files", StaticFiles(directory=str(BASE_CFG.output_dir)), name="files")

JOBS: dict[str, dict] = {}
_lock = threading.Lock()
# One GPU-heavy job at a time; extend with a real queue (RQ/Celery) for multi-user setups.
_executor = ThreadPoolExecutor(max_workers=1)


class JobRequest(BaseModel):
    source: str
    num_clips: Optional[int] = None
    min_clip_s: Optional[float] = None
    max_clip_s: Optional[float] = None
    caption_style: Optional[str] = None
    reframe: Optional[bool] = None


def _public(result: dict) -> dict:
    out = dict(result)
    root = BASE_CFG.output_dir
    clips = []
    for c in result["clips"]:
        c = dict(c)
        for key in ("video_path", "thumbnail"):
            if c.get(key):
                c[key.replace("_path", "_url")] = "/files/" + Path(c[key]).relative_to(root).as_posix()
        clips.append(c)
    out["clips"] = clips
    return out


def _run_job(job_id: str, req: JobRequest) -> None:
    cfg = load_config(os.environ.get("CLIPPER_CONFIG"), {
        "work_dir": BASE_CFG.work_dir, "num_clips": req.num_clips, "min_clip_s": req.min_clip_s,
        "max_clip_s": req.max_clip_s, "caption_style": req.caption_style, "reframe": req.reframe})

    def progress(stage: str, frac: float, msg: str) -> None:
        with _lock:
            JOBS[job_id].update(status="running", stage=stage, progress=round(frac, 3), message=msg)

    try:
        result = Pipeline(cfg, progress).run(req.source)
        with _lock:
            JOBS[job_id].update(status="done", progress=1.0, stage="done", result=_public(result))
    except ClipperError as e:
        with _lock:
            JOBS[job_id].update(status="error", error=str(e))
    except Exception as e:  # unexpected: surface a short message, keep details in server logs
        import logging
        logging.getLogger(__name__).exception("Job failed")
        with _lock:
            JOBS[job_id].update(status="error", error=f"Unexpected error: {str(e)[:300]}")


@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    job_id = uuid.uuid4().hex[:10]
    with _lock:
        JOBS[job_id] = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0.0,
                        "message": "Waiting in queue", "result": None, "error": None}
    _executor.submit(_run_job, job_id, req)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _lock:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "upload.mp4").name
    dest = BASE_CFG.upload_dir / f"{uuid.uuid4().hex[:8]}_{name}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"source": str(dest)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
