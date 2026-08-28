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


def stability_score(md: Optional[MDResult]) -> Optional[float]:
    """MD interface-contact retention (0..1), or None if it was not measured."""
    if md is None or md.contact_retention is None:
        return None
    return max(0.0, min(1.0, md.contact_retention))


def binding_score(md: Optional[MDResult]) -> Optional[float]:
    """Map -dG (kJ/mol) to 0..1 (-40 kJ/mol -> ~1), or None if not measured."""
    if md is None or md.dG is None:
        return None
    return max(0.0, min(1.0, -md.dG / 40.0))


def pose_score(md: Optional[MDResult]) -> Optional[float]:
    """Exponential penalty on mean binder RMSD (nm); 0.5 nm characteristic scale."""
    if md is None or md.rmsd_mean is None:
        return None
    return float(np.exp(-md.rmsd_mean / 0.5))


# Component weights; a component that was not measured is dropped and the
# remaining weights are renormalised (see `score_candidate`).
WEIGHTS = {"confidence": 0.35, "stability": 0.25, "binding": 0.25, "pose": 0.15}


def score_candidate(pred: ComplexPrediction, md: Optional[MDResult]) -> Dict[str, object]:
    """Combined score + per-component breakdown.

    Unmeasured components (None) are excluded and the weights of the surviving
    components are renormalised, rather than being folded in as zeros. Scoring a
    missing measurement as 0 would punish a candidate for what we simply did not
    run — and would let a crashed MD masquerade as a perfectly rigid pose.
    """
    parts = {
        "confidence": confidence_score(pred),
        "stability": stability_score(md),
        "binding": binding_score(md),
        "pose": pose_score(md),
    }
    available = {k: v for k, v in parts.items() if v is not None}
    total_w = sum(WEIGHTS[k] for k in available)
    total = (sum(WEIGHTS[k] * v for k, v in available.items()) / total_w) if total_w else 0.0
    validated = any(parts[k] is not None for k in ("stability", "binding", "pose"))
    return dict(parts, score=total, validated=validated)


def rank_candidates(
    binders: List[Binder],
    predictions: Dict[str, ComplexPrediction],
    md_results: Dict[str, MDResult],
) -> List[RankedCandidate]:
    """Rank binders by combined score (descending), MD-validated candidates first.

    Because unmeasured components are renormalised away rather than zeroed, a
    candidate that never went through MD is scored on confidence alone and could
    otherwise outrank a validated one on an incomparable basis. Sorting on
    `validated` first keeps the candidates we actually simulated at the top.
    """
    rows: List[RankedCandidate] = []
    for b in binders:
        pred = predictions.get(b.id)
        if pred is None:
            continue
        comp = score_candidate(pred, md_results.get(b.id))
        rows.append(RankedCandidate(rank=0, binder=b, prediction=pred,
                                    md=md_results.get(b.id), **comp))
    rows.sort(key=lambda r: (r.validated, r.score), reverse=True)
    for i, r in enumerate(rows, 1):
        r.rank = i
    return rows
