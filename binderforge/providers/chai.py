"""Chai-1 structure prediction (backup provider).

Chai-1 (Apache 2.0) currently has no official programmatic REST API — only the
free web UI at https://lab.chaidiscovery.com and third-party gateways (e.g.
SciRouter). This adapter is a placeholder that documents that limitation.
"""

from __future__ import annotations

import os
from typing import Optional

from ..schemas import Binder, ComplexPrediction
from .base import StructureProvider


class ChaiProvider(StructureProvider):
    name = "chai"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("CHAI_API_KEY")

    def predict(self, target_seq, target_struct_path, binder, out_dir=None, **kwargs) -> ComplexPrediction:
        raise NotImplementedError(
            "Chai-1 has no official programmatic API yet. Use the Boltz-2 provider "
            "(NVIDIA NIM) as the primary structure predictor, or the Chai web UI "
            "manually. A SciRouter-based adapter can be added later."
        )
