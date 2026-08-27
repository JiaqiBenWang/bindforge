"""Runtime configuration — API keys and MD defaults, read from env / a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader — sets vars only if not already present in the environment."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


@dataclass
class Config:
    # provider API keys
    nvidia_api_key: Optional[str] = None
    boltz_bio_api_key: Optional[str] = None
    chai_api_key: Optional[str] = None

    # MD defaults
    md_solvent: str = "implicit"     # implicit | explicit
    md_ns: float = 5.0
    md_temperature: float = 300.0
    md_dt_fs: float = 2.0
    md_platform: str = "auto"        # auto | CPU | CUDA

    results_dir: str = "results"

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        load_dotenv(env_file or ".env")
        return cls(
            nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
            boltz_bio_api_key=os.environ.get("BOLTZ_BIO_API_KEY"),
            chai_api_key=os.environ.get("CHAI_API_KEY"),
            md_solvent=os.environ.get("BINDFORGE_MD_SOLVENT", "implicit"),
            md_ns=float(os.environ.get("BINDFORGE_MD_NS", "5.0")),
            md_temperature=float(os.environ.get("BINDFORGE_MD_TEMP", "300.0")),
            md_dt_fs=float(os.environ.get("BINDFORGE_MD_DT", "2.0")),
            md_platform=os.environ.get("BINDFORGE_MD_PLATFORM", "auto"),
            results_dir=os.environ.get("BINDFORGE_RESULTS_DIR", "results"),
        )


def has_api_key(config: Config, provider: str) -> bool:
    """True if a provider's key is present."""
    return bool(getattr(config, provider + "_api_key", None))
