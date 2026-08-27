"""FastAPI backend: submit jobs, poll status, fetch ranked results.

Runs the BindForge pipeline in a background thread (one job at a time is fine
for a single-process MVP) and serves a minimal single-page frontend.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from binderforge.config import Config
from binderforge.pipeline import run_pipeline

_JOBS: Dict[str, dict] = {}


def _parse_length(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    n = int(s)
    return n, n


class JobRequest(BaseModel):
    target: str
    n_designs: int = 8
    length: str = "50-80"
    hotspot: Optional[str] = None
    md_top: int = 2
    md_ns: float = 5.0
    dry_run: bool = True
    seed: int = 0


def _run_job(job_id: str, req: JobRequest) -> None:
    job = _JOBS[job_id]
    try:
        job["status"] = "running"
        length_min, length_max = _parse_length(req.length)
        config = Config.from_env()
        summary = run_pipeline(
            target=req.target,
            n_designs=req.n_designs,
            length_min=length_min,
            length_max=length_max,
            hotspot=req.hotspot,
            md_top=req.md_top,
            md_ns=req.md_ns,
            dry_run=req.dry_run,
            results_dir=os.path.join("results", job_id),
            config=config,
            seed=req.seed,
        )
        job["status"] = "done"
        job["result"] = summary.get("results", [])
        job["meta"] = {k: v for k, v in summary.items() if k != "results"}
    except Exception as exc:  # noqa: BLE001 — surface the error to the client
        job["status"] = "failed"
        job["error"] = str(exc)


def create_app() -> FastAPI:
    app = FastAPI(title="BindForge", version="0.1.0")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "jobs": len(_JOBS)}

    @app.post("/api/jobs")
    def submit(req: JobRequest):
        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "created": time.time(),
            "request": req.dict(),
            "result": None,
            "error": None,
        }
        threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
        return {"id": job_id, "status": "queued"}

    @app.get("/api/jobs")
    def list_jobs():
        rows = []
        for j in _JOBS.values():
            rows.append({
                "id": j["id"], "status": j["status"],
                "created": j["created"], "error": j.get("error"),
                "n_results": len(j.get("result") or []),
            })
        return {"jobs": sorted(rows, key=lambda r: r["created"], reverse=True)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "id": job["id"], "status": job["status"],
            "created": job["created"], "error": job.get("error"),
            "result": job.get("result"), "meta": job.get("meta"),
        }

    # Minimal single-page frontend (no build step).
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()
