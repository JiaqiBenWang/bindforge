"""Data models for the binder-design pipeline.

Plain dataclasses on purpose — no heavy dependencies, keeps the core package
portable (pydantic is only used by the optional web layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Binder:
    """A designed binder candidate."""

    id: str
    sequence: str
    provider: str = "mock"
    meta: Dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class ComplexPrediction:
    """Structure prediction of the target+binder complex."""

    binder_id: str
    structure: Optional[str] = None        # PDB text (in-memory)
    structure_path: Optional[str] = None   # path to a saved PDB/CIF
    ipTM: Optional[float] = None
    pTM: Optional[float] = None
    pLDDT: Optional[float] = None
    provider: str = "mock"
    meta: Dict = field(default_factory=dict)


@dataclass
class MDResult:
    """MD stability metrics for a complex."""

    binder_id: Optional[str] = None
    solvent: str = "implicit"
    ns: float = 0.0
    rmsd_final: float = 0.0          # binder RMSD vs reference pose (nm)
    rmsd_mean: float = 0.0
    contact_retention: float = 0.0   # fraction of initial interface contacts retained
    dG: float = 0.0                  # MM-GBSA binding energy (kJ/mol; more negative = tighter)
    rmsf: float = 0.0                # mean binder backbone RMSF (nm)
    rmsf_residues: Optional[List[float]] = None  # per-residue CA RMSF profile (nm)
    trajectory_path: Optional[str] = None        # path to final snapshot PDB
    converged: bool = True
    error: Optional[str] = None


@dataclass
class RankedCandidate:
    """A candidate with a combined score, ready for reporting."""

    rank: int
    binder: Binder
    prediction: ComplexPrediction
    md: Optional[MDResult] = None
    score: float = 0.0
    confidence: float = 0.0   # from ipTM/pTM (0..1)
    stability: float = 0.0    # from MD contact retention (0..1)
    binding: float = 0.0      # from -dG (0..1)
    pose: float = 0.0         # from RMSD stability (0..1)
