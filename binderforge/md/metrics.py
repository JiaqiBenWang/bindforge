"""Pure-numpy MD metrics (RMSD / contacts / RMSF). No OpenMM dependency."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _dmat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise distance matrix (n,m) from (n,3) and (m,3)."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


def kabsch_rotation(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Optimal rotation matrix R with ``mobile @ R ~= reference``.

    Both inputs must already be centred on their own centroids. The reflection
    guard (``det``) keeps R a proper rotation rather than a mirror.

    ``np.linalg.svd`` can fail to converge when the anchor is near-collinear
    (e.g. a short peptide after drifting toward a line); in that degenerate case
    the rotation is not even well-defined, so we fall back to the identity
    rather than abort the whole simulation.
    """
    H = mobile.T @ reference
    if not (np.isfinite(H).all() and np.isfinite(reference).all() and np.isfinite(mobile).all()):
        return np.eye(3)
    try:
        V, _, Wt = np.linalg.svd(H)
    except Exception:  # noqa: BLE001 — np raises LinAlgError, sometimes a bare RuntimeError
        return np.eye(3)
    d = np.sign(np.linalg.det(V @ Wt))
    return V @ np.diag([1.0, 1.0, d]) @ Wt


def superpose(mobile_anchor: np.ndarray, reference_anchor: np.ndarray,
              coords: np.ndarray) -> np.ndarray:
    """Map `coords` into the reference frame.

    Computes the rigid transform that best superposes `mobile_anchor` onto
    `reference_anchor` (Kabsch) and applies that same transform to `coords`.

    Used to strip global tumbling: anchoring on the *target* CA atoms means the
    binder RMSD that follows measures motion relative to the target, not the
    free rotation and translation of the whole complex in implicit solvent.
    """
    mc = mobile_anchor.mean(axis=0)
    rc = reference_anchor.mean(axis=0)
    R = kabsch_rotation(mobile_anchor - mc, reference_anchor - rc)
    return (coords - mc) @ R + rc


def ca_rmsd(reference: np.ndarray, coords: np.ndarray) -> float:
    """RMSD (same unit as input, e.g. nm) of `coords` vs `reference` (n,3).

    Positional, not superposed — callers that want pose drift relative to a
    partner chain must pass coordinates already mapped through `superpose`.
    """
    d = coords - reference
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


# Interface contacts are defined on heavy atoms: any target/binder heavy-atom
# pair within 4.5 A. The coarser CA-CA proxy assumes realistic sidechain packing
# to bring backbones within ~8 A, which idealised or extended sidechains never
# satisfy — it reports an empty interface for poses that are genuinely touching.
HEAVY_ATOM_CUTOFF = 0.45  # nm


def interface_contact_mask(target_xyz: np.ndarray, binder_xyz: np.ndarray,
                           cutoff: float = HEAVY_ATOM_CUTOFF) -> np.ndarray:
    """Boolean (nt, nb) mask of target/binder atom pairs within `cutoff` (nm)."""
    return _dmat(target_xyz, binder_xyz) < cutoff


def contact_retention(contact_mask: np.ndarray, target_xyz: np.ndarray,
                      binder_xyz: np.ndarray,
                      cutoff: float = HEAVY_ATOM_CUTOFF) -> Optional[float]:
    """Fraction of reference contacts still present at the current frame.

    Returns ``None`` when the reference has no contacts at all: an interface
    that never existed is *not measurable*, which is a different statement from
    an interface that was fully lost (0.0). Collapsing the two would silently
    score an unbound pair as a maximally unstable binder.
    """
    denom = int(contact_mask.sum())
    if denom == 0:
        return None
    current = _dmat(target_xyz, binder_xyz) < cutoff
    return float((contact_mask & current).sum() / denom)


def rmsf(frames: np.ndarray) -> np.ndarray:
    """Per-atom RMSF over `frames` (n_frames, n_atoms, 3)."""
    mean = frames.mean(axis=0)
    d = frames - mean
    return np.sqrt(np.mean(np.sum(d * d, axis=2), axis=0))
