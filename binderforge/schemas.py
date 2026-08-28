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
    """MD stability metrics for a complex.

    Every metric is Optional and ``None`` means *not measured* — the run failed,
    or the quantity was undefined for this system (e.g. contact retention when
    the starting pose had no interface). None is deliberately distinct from 0.0,
    which is a real measurement meaning "fully lost"; scoring skips None
    components instead of treating them as a zero.
    """

    binder_id: Optional[str] = None
    solvent: str = "implicit"
    ns: float = 0.0
    rmsd_final: Optional[float] = None   # binder RMSD vs reference pose (nm), target-superposed
    rmsd_mean: Optional[float] = None
    contact_retention: Optional[float] = None  # fraction of initial interface contacts retained
    dG: Optional[float] = None           # MM-GBSA binding energy (kJ/mol; more negative = tighter)
    rmsf: Optional[float] = None         # mean binder backbone RMSF (nm)
    rmsf_residues: Optional[List[float]] = None  # per-residue CA RMSF profile (nm)
    trajectory_path: Optional[str] = None        # path to final snapshot PDB
    converged: bool = True
    error: Optional[str] = None


@dataclass
class RankedCandidate:
    """A candidate with a combined score, ready for reporting.

    Component scores are ``None`` when the underlying quantity was not measured;
    `score` is renormalised over whichever components are present, and
    `validated` records whether MD actually contributed to it.
    """

    rank: int
    binder: Binder
    prediction: ComplexPrediction
    md: Optional[MDResult] = None
    score: float = 0.0
    confidence: Optional[float] = None   # from ipTM/pTM (0..1)
    stability: Optional[float] = None    # from MD contact retention (0..1)
    binding: Optional[float] = None      # from -dG (0..1)
    pose: Optional[float] = None         # from RMSD stability (0..1)
    validated: bool = False   # did MD contribute any component to `score`?
