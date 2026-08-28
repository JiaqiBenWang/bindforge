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

import base64
import json
import os
import re
import shlex
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    bindforge_bin: str = "bindforge"     # command or full path to the bindforge CLI
    timeout: float = 20.0                # SSH connect/auth/IO timeout (s)

    @classmethod
    def from_env(cls) -> "SlurmConfig":
        key = os.environ.get("BINDFORGE_SLURM_KEY")
        return cls(
            host=os.environ.get("BINDFORGE_SLURM_HOST", ""),
            user=os.environ.get("BINDFORGE_SLURM_USER", ""),
            key_path=os.path.expanduser(key) if key else None,
            password=os.environ.get("BINDFORGE_SLURM_PASSWORD") or None,
            port=int(os.environ.get("BINDFORGE_SLURM_PORT", "22")),
            remote_dir=os.environ.get("BINDFORGE_SLURM_REMOTE_DIR", "bindforge_jobs"),
            partition=os.environ.get("BINDFORGE_SLURM_PARTITION") or None,
            time=os.environ.get("BINDFORGE_SLURM_TIME", "01:00:00"),
            cpus=int(os.environ.get("BINDFORGE_SLURM_CPUS", "4")),
            gres=os.environ.get("BINDFORGE_SLURM_GRES") or None,
            conda_env=os.environ.get("BINDFORGE_SLURM_CONDA") or None,
            modules=os.environ.get("BINDFORGE_SLURM_MODULES", ""),
            bindforge_bin=os.environ.get("BINDFORGE_SLURM_BINDFORGE", "bindforge"),
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

    cmd = [cfg.bindforge_bin, "run", "--target", target_name]
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
    the long polling loop never holds a single stale connection. Files are
    transferred over the exec channel (base64 via stdin/stdout) rather than
    SFTP, because many HPC login nodes disable the SFTP/SCP subsystem.
    """

    def __init__(self, cfg: SlurmConfig):
        self.cfg = cfg
        self._client = None
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
        _, out, _ = client.exec_command("echo $HOME")
        self._home = out.read().decode(errors="replace").strip() or "~"

    def _close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None
        self._home = None

    @contextmanager
    def _session(self):
        self._connect()
        try:
            yield
        finally:
            self._close()

    def _run(self, cmd: str, stdin_data: Optional[str] = None) -> Tuple[int, str, str]:
        stdin, stdout, stderr = self._client.exec_command(cmd)
        if stdin_data is not None:
            stdin.write(stdin_data)
        stdin.channel.shutdown_write()
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
        rc, _, _ = self._run(f"test -f {shlex.quote(remote_path)}")
        return rc == 0

    def _cat(self, remote_path: str, tail: int = 4000) -> str:
        rc, out, _ = self._run(f"tail -c {tail} {shlex.quote(remote_path)} 2>/dev/null")
        return out if rc == 0 else ""

    # ── file transfer over the exec channel (base64) ───────────────────
    def _put(self, local_path: str, remote_path: str) -> None:
        with open(local_path, "rb") as f:
            self._put_bytes(remote_path, base64.b64encode(f.read()).decode("ascii"))

    def _put_text(self, remote_path: str, text: str) -> None:
        self._put_bytes(remote_path, base64.b64encode(text.encode("utf-8")).decode("ascii"))

    def _put_bytes(self, remote_path: str, b64: str) -> None:
        d = os.path.dirname(remote_path)
        rc, out, err = self._run(
            f"mkdir -p {shlex.quote(d)} && base64 -d > {shlex.quote(remote_path)}",
            stdin_data=b64,
        )
        if rc != 0:
            raise RemoteError(f"上传失败 / upload failed: {err or out}")

    def _get(self, remote_path: str, local_path: str) -> None:
        rc, out, err = self._run(f"base64 {shlex.quote(remote_path)} 2>/dev/null")
        if rc != 0:
            raise RemoteError(f"下载失败 / download failed: {err or out}")
        data = base64.b64decode("".join(out.split()))
        os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)

    def _listdir(self, remote_dir: str) -> List[str]:
        rc, out, _ = self._run(f"ls -1 {shlex.quote(remote_dir)} 2>/dev/null")
        return [x for x in out.splitlines() if x] if rc == 0 else []

    # ── public operations ──────────────────────────────────────────────
    def submit(self, job_id: str, target_local_path: str,
               target_remote_name: str, script_text: str) -> str:
        """Upload target + sbatch script, submit, return the Slurm job id."""
        with self._session():
            jd = self.job_dir(job_id)
            self._put(target_local_path, f"{jd}/{target_remote_name}")
            self._put_text(f"{jd}/submit.slurm", script_text)
            # Submitting with cwd=jd makes Slurm run the job in the same dir,
            # so the script's relative --target/--results-dir paths resolve.
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
            self._get(f"{jd}/ranking.json", os.path.join(local_dir, "ranking.json"))
            for name in self._listdir(jd):
                if name.endswith(".pdb"):
                    try:
                        self._get(f"{jd}/{name}", os.path.join(local_dir, name))
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
