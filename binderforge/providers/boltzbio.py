"""boltz.bio providers (BoltzGen de novo binder design + Boltz-2 prediction).

boltz.bio (https://boltz.bio/api) offers a commercial API (``boltz-compute`` /
``boltz_api`` SDKs) including BoltzGen for de novo binder design. These adapters
are placeholders: the exact request schema must be confirmed against the current
boltz.bio API docs before first use. Set ``BOLTZ_BIO_API_KEY``.
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..schemas import Binder, ComplexPrediction
from .base import DesignProvider, StructureProvider


class BoltzGenProvider(DesignProvider):
    """De novo binder design via boltz.bio BoltzGen (placeholder)."""

    name = "boltzgen"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BOLTZ_BIO_API_KEY")

    def design(self, target_seq, target_struct_path, n_designs, length_min, length_max,
               hotspot=None, **kwargs) -> List[Binder]:
        raise NotImplementedError(
            "BoltzGenProvider is not wired yet. The boltz.bio BoltzGen API schema "
            "must be confirmed against https://boltz.bio/api before implementing. "
            "For now use --dry-run (mock design) or integrate ProteinMPNN."
        )


class BoltzBioStructureProvider(StructureProvider):
    """Structure prediction via boltz.bio Boltz-2 API (placeholder)."""

    name = "boltzbio"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BOLTZ_BIO_API_KEY")

    def predict(self, target_seq, target_struct_path, binder, out_dir=None, **kwargs) -> ComplexPrediction:
        raise NotImplementedError(
            "BoltzBioStructureProvider is not wired yet. Prefer NvidiaBoltz2Provider "
            "(official Boltz-2 NIM API). See https://boltz.bio/api for the boltz.bio API."
        )
