"""Composite scoring and ranking of binder candidates."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .schemas import Binder, ComplexPrediction, MDResult, RankedCandidate


def confidence_score(pred: ComplexPrediction) -> float:
    """Average of ipTM and pTM (0..1); falls back to pLDDT if those are missing."""
    vals = [v for v in (pred.ipTM, pred.pTM) if v is not None]
    if not vals and pred.pLDDT is not None:
        return max(0.0, min(1.0, pred.pLDDT / 100.0))
    if not vals:
        return 0.0
    return max(0.0, min(1.0, float(np.mean(vals))))


def stability_score(md: Optional[MDResult]) -> float:
    """MD interface-contact retention (0..1)."""
    if md is None:
        return 0.0
    return max(0.0, min(1.0, md.contact_retention))


def binding_score(md: Optional[MDResult]) -> float:
    """Map -dG (kJ/mol) to 0..1; -40 kJ/mol maps to ~1."""
    if md is None or md.dG is None:
        return 0.0
    return max(0.0, min(1.0, -md.dG / 40.0))


def pose_score(md: Optional[MDResult]) -> float:
    """Exponential penalty on mean binder RMSD (nm); 0.5 nm characteristic scale."""
    if md is None:
        return 0.0
    return float(np.exp(-md.rmsd_mean / 0.5))


def score_candidate(pred: ComplexPrediction, md: Optional[MDResult]) -> Dict[str, float]:
    """Combined score + per-component breakdown. Weights are tunable."""
    conf = confidence_score(pred)
    stab = stability_score(md)
    bind = binding_score(md)
    pose = pose_score(md)
    total = 0.35 * conf + 0.25 * stab + 0.25 * bind + 0.15 * pose
    return dict(confidence=conf, stability=stab, binding=bind, pose=pose, score=total)


def rank_candidates(
    binders: List[Binder],
    predictions: Dict[str, ComplexPrediction],
    md_results: Dict[str, MDResult],
) -> List[RankedCandidate]:
    """Rank binders by combined score (descending)."""
    rows: List[RankedCandidate] = []
    for b in binders:
        pred = predictions.get(b.id)
        if pred is None:
            continue
        comp = score_candidate(pred, md_results.get(b.id))
        rows.append(RankedCandidate(rank=0, binder=b, prediction=pred,
                                    md=md_results.get(b.id), **comp))
    rows.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(rows, 1):
        r.rank = i
    return rows
