"""Mock providers for ``--dry-run`` (no API key needed).

They exercise the full pipeline wiring with reproducible, deterministic data and
produce a loadable all-atom complex (target from PDB + built binder) so the real
OpenMM MD stage can run end-to-end.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .. import io
from ..schemas import Binder, ComplexPrediction
from .base import DesignProvider, StructureProvider

AA20 = "ACDEFGHIKLMNPQRSTVWY"
_HYDROPHOBIC = set("AILMFWYV")


def _mock_confidence(seq: str) -> float:
    """Deterministic pseudo-confidence from hydrophobicity (0.35 .. 0.80)."""
    frac = sum(a in _HYDROPHOBIC for a in seq) / max(1, len(seq))
    return 0.35 + 0.45 * frac


class MockDesignProvider(DesignProvider):
    name = "mock"

    def design(self, target_seq, target_struct_path, n_designs, length_min, length_max,
               hotspot=None, seed=0, **kwargs) -> List[Binder]:
        rng = np.random.default_rng(seed)
        binders: List[Binder] = []
        for k in range(n_designs):
            length = int(rng.integers(length_min, length_max + 1))
            seq = "".join(rng.choice(list(AA20), size=length))
            binders.append(Binder(id=f"binder_{k:03d}", sequence=seq, provider=self.name))
        return binders


class MockStructureProvider(StructureProvider):
    name = "mock"

    def predict(self, target_seq, target_struct_path, binder, out_dir=None, **kwargs) -> ComplexPrediction:
        conf = _mock_confidence(binder.sequence)
        structure = None
        structure_path = None

        # Build a real all-atom complex so downstream MD can run (only with a target PDB).
        if target_struct_path:
            if not out_dir:
                out_dir = "."
            os.makedirs(out_dir, exist_ok=True)
            structure_path = os.path.join(out_dir, f"{binder.id}.pdb")
            io.build_complex_pdb(target_struct_path, binder.sequence, structure_path)

        return ComplexPrediction(
            binder_id=binder.id,
            structure=structure,
            structure_path=structure_path,
            ipTM=round(conf, 3),
            pTM=round(max(0.0, conf - 0.05), 3),
            pLDDT=round(conf * 100.0, 1),
            provider=self.name,
        )
