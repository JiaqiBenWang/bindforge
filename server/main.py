"""FastAPI backend: accounts, submit jobs, poll status, fetch ranked results.

Runs the BindForge pipeline in a background thread (one job at a time is fine
for a single-process MVP) and serves a single-page frontend. All job endpoints
require a signed session token (email/password accounts); jobs are scoped to
the user who created them.
"""

from __future__ import annotations

import io
import os
import threading
import time
import uuid
from contextlib import redirect_stderr
from typing import Dict, Optional

from fastapi import Cookie, Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from binderforge import auth, remote
from binderforge.config import Config
from binderforge.pipeline import load_target_sequence, run_pipeline

_JOBS: Dict[str, dict] = {}

_UPLOAD_DIR = "uploads"
_ALLOWED_SUFFIXES = (".pdb", ".cif", ".mmcif", ".fasta", ".fa", ".faa", ".txt")
_MAX_TARGET_RESIDUES = 500  # business rule: free tier targets must be ≤ 500 aa

# "local" runs the pipeline in-process; "slurm" submits to a remote cluster
# (Beijing supercomputer) over SSH + Slurm. See binderforge/remote.py.
_COMPUTE_BACKEND = os.environ.get("BINDFORGE_COMPUTE", "local").strip().lower()

_DATA_DIR = os.environ.get("BINDFORGE_DATA_DIR", "data")
_STORE = auth.AuthStore(os.path.join(_DATA_DIR, "users.db"))
_SECRET = auth.load_secret(_DATA_DIR)
_COOKIE_NAME = "bindforge_token"


def _parse_length(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    n = int(s)
    return n, n


def _target_residue_count(target: str) -> int:
    """Resolve a target (file path or raw sequence) to its residue count."""
    seq = load_target_sequence(target)
    return len(seq)


def _check_size(target: str) -> None:
    """Reject targets over the 500-aa business limit."""
    try:
        n = _target_residue_count(target)
    except Exception as exc:  # noqa: BLE001 — unparseable target surfaces its own error
        raise HTTPException(status_code=400, detail=f"无法解析靶点 / Cannot parse target: {exc}")
    if n > _MAX_TARGET_RESIDUES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"靶点 {n} aa 超过 {_MAX_TARGET_RESIDUES} aa 限制，请缩短或拆分靶点。"
                f" Target ({n} aa) exceeds the {_MAX_TARGET_RESIDUES}-aa limit."
            ),
        )


class JobRequest(BaseModel):
    target: str
    n_designs: int = 8
    length: str = "50-80"
    hotspot: Optional[str] = None
    target_chain: Optional[str] = None
    design_provider: str = "mock"
    structure_provider: str = "mock"
    md_top: int = 2
    md_ns: float = 5.0
    dry_run: bool = True
    seed: int = 0


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


def _user_id_from_headers(
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Cookie(default=None, alias=_COOKIE_NAME),
) -> str:
    """Extract the session token (Bearer header or cookie) and validate it."""
    raw = None
    if authorization and authorization.lower().startswith("bearer "):
        raw = authorization[7:].strip()
    elif token:
        raw = token
    if not raw:
        raise HTTPException(status_code=401, detail="未登录 / Not logged in")
    user_id = auth.verify_token(_SECRET, raw)
    if user_id is None:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录 / Session expired")
    return user_id


def _public_user(user_id: str) -> dict:
    u = _STORE.get_user(user_id)
    return u or {"id": user_id, "email": None}


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(_COOKIE_NAME, token, httponly=True, samesite="lax",
                        max_age=auth._TOKEN_TTL_SECONDS, path="/")


def _materialize_target(target: str, job_id: str) -> tuple:
    """Return (local_path, remote_name) for a target (file or raw sequence)."""
    if os.path.isfile(target):
        return target, os.path.basename(target)
    seq = load_target_sequence(target)  # validates the raw sequence
    d = os.path.join(_UPLOAD_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "target.fasta")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f">target\n{seq}\n")
    return path, "target.fasta"


def _run_local_job(job_id: str, req: JobRequest) -> None:
    job = _JOBS[job_id]
    try:
        job["status"] = "running"
        length_min, length_max = _parse_length(req.length)
        config = Config.from_env()
        # Capture the MD stage's stderr progress lines into the job log so the
        # frontend can show live progress, not just a spinner.
        buf = io.StringIO()
        with redirect_stderr(buf):
            summary = run_pipeline(
                target=req.target,
                n_designs=req.n_designs,
                length_min=length_min,
                length_max=length_max,
                hotspot=req.hotspot,
                target_chain=req.target_chain,
                design_provider=req.design_provider,
                structure_provider=req.structure_provider,
                md_top=req.md_top,
                md_ns=req.md_ns,
                dry_run=req.dry_run,
                results_dir=os.path.join("results", job_id),
                config=config,
                seed=req.seed,
            )
        job["logs"] = buf.getvalue()
        job["status"] = "done"
        job["result"] = summary.get("results", [])
        job["meta"] = {k: v for k, v in summary.items() if k != "results"}
    except Exception as exc:  # noqa: BLE001 — surface the error to the client
        job["status"] = "failed"
        job["error"] = str(exc)


def _run_remote_job(job_id: str, req: JobRequest) -> None:
    job = _JOBS[job_id]
    try:
        job["status"] = "running"
        cfg = remote.SlurmConfig.from_env()
        target_local, target_remote = _materialize_target(req.target, job_id)

        def _on_status(state: str, log: str) -> None:
            if log:
                job["logs"] = log

        summary = remote.run_remote_job(
            job_id, target_local, target_remote, req.dict(), cfg=cfg, on_status=_on_status
        )
        job["status"] = "done"
        job["result"] = summary.get("results", [])
        job["meta"] = {"remote": True, "slurm_id": summary.get("slurm_id")}
    except Exception as exc:  # noqa: BLE001 — surface the error to the client
        job["status"] = "failed"
        job["error"] = str(exc)


def _run_job(job_id: str, req: JobRequest) -> None:
    if _COMPUTE_BACKEND == "slurm":
        _run_remote_job(job_id, req)
    else:
        _run_local_job(job_id, req)


def _save_upload(upload: UploadFile, job_id: str) -> str:
    """Persist an uploaded target file under uploads/<job_id>/ and return its path."""
    name = os.path.basename(upload.filename or "target.pdb")
    if not name.lower().endswith(_ALLOWED_SUFFIXES):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 / Unsupported file type. Allowed: {', '.join(_ALLOWED_SUFFIXES)}",
        )
    dest_dir = os.path.join(_UPLOAD_DIR, job_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, name)
    data = upload.file.read()
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def create_app() -> FastAPI:
    app = FastAPI(title="BindForge", version="0.2.0")

    # ── Accounts (public) ──────────────────────────────────────────────
    @app.get("/api/health")
    def health():
        return {"status": "ok", "jobs": len(_JOBS)}

    @app.post("/api/register")
    def register(req: RegisterRequest, response: Response):
        try:
            user_id = _STORE.create_user(req.email, req.password)
        except auth.AuthError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        token = auth.issue_token(_SECRET, user_id)
        _set_cookie(response, token)
        return {"token": token, "user": _public_user(user_id)}

    @app.post("/api/login")
    def login(req: LoginRequest, response: Response):
        try:
            user_id = _STORE.verify_login(req.email, req.password)
        except auth.AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc))
        token = auth.issue_token(_SECRET, user_id)
        _set_cookie(response, token)
        return {"token": token, "user": _public_user(user_id)}

    @app.post("/api/logout")
    def logout(response: Response):
        response.delete_cookie(_COOKIE_NAME, path="/")
        return {"status": "ok"}

    @app.get("/api/me")
    def me(user_id: str = Depends(_user_id_from_headers)):
        user = _public_user(user_id)
        user["quota"] = {
            "used_today": _STORE.usage_today(user_id),
            "daily_limit": auth.FREE_DAILY_LIMIT,
        }
        return user

    # ── Jobs (authenticated) ───────────────────────────────────────────
    @app.post("/api/jobs")
    def submit(req: JobRequest, user_id: str = Depends(_user_id_from_headers)):
        _check_size(req.target)
        job_id = uuid.uuid4().hex[:12]
        try:
            _STORE.check_and_charge(user_id, job_id)
        except auth.QuotaExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        _JOBS[job_id] = {
            "id": job_id,
            "user_id": user_id,
            "status": "queued",
            "created": time.time(),
            "request": req.dict(),
            "result": None,
            "error": None,
            "logs": "",
        }
        threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
        return {"id": job_id, "status": "queued"}

    @app.post("/api/upload")
    async def upload_and_run(
        file: UploadFile = File(...),
        n_designs: int = Form(8),
        length: str = Form("50-80"),
        hotspot: Optional[str] = Form(None),
        target_chain: Optional[str] = Form(None),
        md_top: int = Form(2),
        md_ns: float = Form(5.0),
        dry_run: bool = Form(True),
        seed: int = Form(0),
        user_id: str = Depends(_user_id_from_headers),
    ):
        """Upload a PDB/CIF/FASTA target file and run the pipeline on it."""
        job_id = uuid.uuid4().hex[:12]
        path = _save_upload(file, job_id)
        _check_size(path)
        try:
            _STORE.check_and_charge(user_id, job_id)
        except auth.QuotaExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        req = JobRequest(
            target=path, n_designs=n_designs, length=length, hotspot=hotspot,
            target_chain=target_chain,
            md_top=md_top, md_ns=md_ns, dry_run=dry_run, seed=seed,
        )
        _JOBS[job_id] = {
            "id": job_id,
            "user_id": user_id,
            "status": "queued",
            "created": time.time(),
            "request": req.dict(),
            "result": None,
            "error": None,
            "logs": "",
        }
        threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
        return {"id": job_id, "status": "queued", "target": os.path.basename(path)}

    @app.get("/api/jobs")
    def list_jobs(user_id: str = Depends(_user_id_from_headers)):
        rows = []
        for j in _JOBS.values():
            if j.get("user_id") != user_id:
                continue
            rows.append({
                "id": j["id"], "status": j["status"],
                "created": j["created"], "error": j.get("error"),
                "n_results": len(j.get("result") or []),
            })
        return {"jobs": sorted(rows, key=lambda r: r["created"], reverse=True)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, user_id: str = Depends(_user_id_from_headers)):
        job = _JOBS.get(job_id)
        if job is None or job.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "id": job["id"], "status": job["status"],
            "created": job["created"], "error": job.get("error"),
            "result": job.get("result"), "meta": job.get("meta"),
            "logs": job.get("logs", ""),
        }

    @app.get("/api/jobs/{job_id}/structure/{filename}")
    def get_structure(job_id: str, filename: str, user_id: str = Depends(_user_id_from_headers)):
        """Serve a generated complex PDB (for the 3D viewer), scoped to the owner."""
        job = _JOBS.get(job_id)
        if job is None or job.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="job not found")
        if not filename.lower().endswith((".pdb", ".cif")):
            raise HTTPException(status_code=400, detail="only .pdb/.cif files are served")
        # Only ever join a basename into the results dir — no traversal.
        safe = os.path.basename(filename)
        path = os.path.join("results", job_id, safe)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="structure not found")
        from fastapi.responses import FileResponse
        return FileResponse(path, media_type="chemical/x-pdb", filename=safe)

    # Minimal single-page frontend (no build step).
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()
