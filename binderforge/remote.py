"""SSH + Slurm compute backend — offload jobs to a remote supercomputer.

The web server runs jobs locally by default (mock / dry-run). Setting
``BINDFORGE_COMPUTE=slurm`` routes every job to a cluster instead: this module
owns the transport — rendering the sbatch script, uploading the target, running
``sbatch``, polling ``squeue``/``sacct``, and downloading ``ranking.json`` +
complex PDBs back to the server.

``paramiko`` is imported lazily so the package remains importable without it;
install it with ``pip install -e '.[remote]'`` (or the ``web`` extra plus
``paramiko``).
"""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Slurm job states as the web server exposes them (matches local jobs).
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"

_POLL_INTERVAL_SECONDS = 15.0
_SLURM_STATE_PENDING = {"PENDING", "CONFIGURING", "COMPLETING", "RESV_DEL_HOLD"}


class RemoteError(Exception):
    """Raised for SSH/transport failures, misconfiguration, or bad Slurm state."""


@dataclass
class SlurmConfig:
    """Connection + queue settings, read from ``BINDFORGE_SLURM_*`` env vars."""

    host: str = ""
    user: str = ""
    key_path: Optional[str] = None
    password: Optional[str] = None
    port: int = 22
    remote_dir: str = "bindforge_jobs"   # under $HOME unless absolute
    partition: Optional[str] = None
    time: str = "01:00:00"
    cpus: int = 4
    gres: Optional[str] = None           # e.g. "gpu:1"
    conda_env: Optional[str] = None
    modules: str = ""                    # comma-separated, e.g. "cuda/12,openmm"
    timeout: float = 20.0                # SSH connect/auth/IO timeout (s)

    @classmethod
    def from_env(cls) -> "SlurmConfig":
        return cls(
            host=os.environ.get("BINDFORGE_SLURM_HOST", ""),
            user=os.environ.get("BINDFORGE_SLURM_USER", ""),
            key_path=os.environ.get("BINDFORGE_SLURM_KEY") or None,
            password=os.environ.get("BINDFORGE_SLURM_PASSWORD") or None,
            port=int(os.environ.get("BINDFORGE_SLURM_PORT", "22")),
            remote_dir=os.environ.get("BINDFORGE_SLURM_REMOTE_DIR", "bindforge_jobs"),
            partition=os.environ.get("BINDFORGE_SLURM_PARTITION") or None,
            time=os.environ.get("BINDFORGE_SLURM_TIME", "01:00:00"),
            cpus=int(os.environ.get("BINDFORGE_SLURM_CPUS", "4")),
            gres=os.environ.get("BINDFORGE_SLURM_GRES") or None,
            conda_env=os.environ.get("BINDFORGE_SLURM_CONDA") or None,
            modules=os.environ.get("BINDFORGE_SLURM_MODULES", ""),
            timeout=float(os.environ.get("BINDFORGE_SLURM_TIMEOUT", "20")),
        )

    def configured(self) -> bool:
        return bool(self.host and self.user and (self.key_path or self.password))


def render_slurm_script(
    cfg: SlurmConfig,
    job_id: str,
    target_name: str,
    run_args: Dict,
) -> str:
    """Render an sbatch script that runs ``bindforge run`` against ``target_name``.

    Pure and side-effect free, so it can be unit-tested without SSH.
    ``run_args`` carries the same fields as the web ``JobRequest``.
    """
    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name=bf_{job_id}")
    lines.append(f"#SBATCH --output={job_id}.out")
    lines.append(f"#SBATCH --error={job_id}.err")
    lines.append(f"#SBATCH --time={cfg.time}")
    lines.append(f"#SBATCH --cpus-per-task={cfg.cpus}")
    if cfg.gres:
        lines.append(f"#SBATCH --gres={cfg.gres}")
    if cfg.partition:
        lines.append(f"#SBATCH --partition={cfg.partition}")
    lines.append("")
    lines.append("set -e")
    if cfg.conda_env:
        lines.append(f"source activate {shlex.quote(cfg.conda_env)}")
    for mod in (m for m in cfg.modules.split(",") if m.strip()):
        lines.append(f"module load {shlex.quote(mod.strip())}")
    lines.append("")

    cmd = ["bindforge", "run", "--target", target_name]
    cmd += ["--n-designs", str(run_args.get("n_designs", 8))]
    cmd += ["--length", str(run_args.get("length", "50-80"))]
    if run_args.get("hotspot"):
        cmd += ["--hotspot", str(run_args["hotspot"])]
    if run_args.get("design_provider"):
        cmd += ["--design-provider", str(run_args["design_provider"])]
    if run_args.get("structure_provider"):
        cmd += ["--structure-provider", str(run_args["structure_provider"])]
    cmd += ["--md-top", str(run_args.get("md_top", 2))]
    cmd += ["--md-ns", str(run_args.get("md_ns", 5.0))]
    if run_args.get("dry_run"):
        cmd += ["--dry-run"]
    cmd += ["--results-dir", "."]
    cmd += ["--seed", str(run_args.get("seed", 0))]
    lines.append(" ".join(shlex.quote(c) for c in cmd))
    lines.append("")
    return "\n".join(lines)


class SlurmBackend:
    """Minimal paramiko-based SSH+Slurm transport.

    Each public method opens a fresh SSH session (connect → work → close) so
    the long polling loop never holds a single stale connection. Calls that
    need SFTP reuse the session's SFTP channel.
    """

    def __init__(self, cfg: SlurmConfig):
        self.cfg = cfg
        self._client = None
        self._sftp = None
        self._home = None

    # ── session plumbing ───────────────────────────────────────────────
    def _connect(self) -> None:
        if not self.cfg.configured():
            raise RemoteError(
                "Slurm 未配置 / Slurm backend not configured. Set BINDFORGE_SLURM_HOST, "
                "BINDFORGE_SLURM_USER and BINDFORGE_SLURM_KEY (or BINDFORGE_SLURM_PASSWORD)."
            )
        import paramiko  # lazy — optional dependency
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs: dict = dict(
            hostname=self.cfg.host,
            port=self.cfg.port,
            username=self.cfg.user,
            timeout=self.cfg.timeout,
            banner_timeout=self.cfg.timeout,
            auth_timeout=self.cfg.timeout,
        )
        if self.cfg.key_path:
            kwargs["key_filename"] = self.cfg.key_path
        else:
            kwargs["password"] = self.cfg.password
        try:
            client.connect(**kwargs)
        except Exception as exc:  # noqa: BLE001 — wrap transport errors
            raise RemoteError(f"SSH 连接失败 / SSH connect failed: {exc}")
        client.get_transport().set_keepalive(30)
        self._client = client
        self._sftp = client.open_sftp()
        _, out, _ = client.exec_command("echo $HOME")
        self._home = out.read().decode(errors="replace").strip() or "~"

    def _close(self) -> None:
        for obj in (self._sftp, self._client):
            try:
                if obj is not None:
                    obj.close()
            except Exception:  # noqa: BLE001
                pass
        self._sftp = None
        self._client = None
        self._home = None

    @contextmanager
    def _session(self):
        self._connect()
        try:
            yield
        finally:
            self._close()

    def _run(self, cmd: str) -> Tuple[int, str, str]:
        _, stdout, stderr = self._client.exec_command(cmd)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err

    def job_dir(self, job_id: str) -> str:
        base = self.cfg.remote_dir
        if base.startswith("/"):
            return base.rstrip("/") + "/" + job_id
        return self._home.rstrip("/") + "/" + base.strip("/") + "/" + job_id

    def _exists(self, remote_path: str) -> bool:
        try:
            self._sftp.stat(remote_path)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _cat(self, remote_path: str, tail: int = 4000) -> str:
        try:
            with self._sftp.open(remote_path, "r") as f:
                data = f.read().decode(errors="replace")
            return data[-tail:]
        except Exception:  # noqa: BLE001
            return ""

    # ── public operations ──────────────────────────────────────────────
    def submit(self, job_id: str, target_local_path: str,
               target_remote_name: str, script_text: str) -> str:
        """Upload target + sbatch script, submit, return the Slurm job id."""
        with self._session():
            jd = self.job_dir(job_id)
            self._run(f"mkdir -p {shlex.quote(jd)}")
            self._sftp.put(target_local_path, f"{jd}/{target_remote_name}")
            self._sftp.putfo(io.BytesIO(script_text.encode("utf-8")), f"{jd}/submit.slurm")
            rc, out, err = self._run(f"cd {shlex.quote(jd)} && sbatch submit.slurm")
            if rc != 0:
                raise RemoteError(f"sbatch 提交失败 / sbatch failed: {err or out}")
            m = re.search(r"Submitted batch job\s+(\d+)", out + err)
            if not m:
                raise RemoteError(f"无法解析 Slurm 作业号 / could not parse job id: {out + err}")
            return m.group(1)

    def status(self, job_id: str, slurm_id: str) -> Dict[str, str]:
        """Return ``{"state": queued|running|done|failed, "log": ...}``."""
        with self._session():
            jd = self.job_dir(job_id)
            rc, out, _ = self._run(f"squeue -h -j {slurm_id} -o %T")
            state = out.strip()
            if state:
                if state.split()[0].upper() in _SLURM_STATE_PENDING:
                    return {"state": STATE_QUEUED, "log": self._cat(f"{jd}/{job_id}.out")}
                return {"state": STATE_RUNNING, "log": self._cat(f"{jd}/{job_id}.out")}
            # Left the queue — determine outcome.
            if self._exists(f"{jd}/ranking.json"):
                return {"state": STATE_DONE, "log": self._cat(f"{jd}/{job_id}.out")}
            err = self._cat(f"{jd}/{job_id}.err")
            return {"state": STATE_FAILED, "log": (err or self._cat(f"{jd}/{job_id}.out"))}

    def download_results(self, job_id: str, local_dir: str) -> None:
        """Pull ranking.json and all complex PDBs into ``local_dir``."""
        with self._session():
            jd = self.job_dir(job_id)
            os.makedirs(local_dir, exist_ok=True)
            self._sftp.get(f"{jd}/ranking.json", os.path.join(local_dir, "ranking.json"))
            try:
                names = self._sftp.listdir(jd)
            except Exception:  # noqa: BLE001
                names = []
            for name in names:
                if name.endswith(".pdb"):
                    try:
                        self._sftp.get(f"{jd}/{name}", os.path.join(local_dir, name))
                    except Exception:  # noqa: BLE001
                        pass


def run_remote_job(job_id: str, target_local_path: str, target_remote_name: str,
                   run_args: Dict, cfg: Optional[SlurmConfig] = None,
                   poll_interval: float = _POLL_INTERVAL_SECONDS,
                   on_status=None) -> Dict:
    """Submit, poll to completion, download, and return the ranking rows.

    ``on_status(state, log)`` is an optional callback invoked after each poll
    so the caller (the web server) can surface live progress.
    """
    cfg = cfg or SlurmConfig.from_env()
    backend = SlurmBackend(cfg)
    script = render_slurm_script(cfg, job_id, target_remote_name, run_args)
    slurm_id = backend.submit(job_id, target_local_path, target_remote_name, script)

    state = STATE_RUNNING
    while state in (STATE_QUEUED, STATE_RUNNING):
        time.sleep(poll_interval)
        st = backend.status(job_id, slurm_id)
        state = st["state"]
        if on_status:
            on_status(state, st["log"])
        if state == STATE_FAILED:
            raise RemoteError(f"Slurm 作业失败 / Slurm job failed (id {slurm_id}): {st['log'][-2000:]}")

    local_dir = os.path.join("results", job_id)
    backend.download_results(job_id, local_dir)
    with open(os.path.join(local_dir, "ranking.json"), encoding="utf-8") as f:
        rows = json.load(f)
    return {"results": rows, "slurm_id": slurm_id, "remote": True}
