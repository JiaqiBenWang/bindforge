"""Pure-numpy MD metrics (RMSD / contacts / RMSF). No OpenMM dependency."""

from __future__ import annotations

import numpy as np


def _dmat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise distance matrix (n,m) from (n,3) and (m,3)."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


def ca_rmsd(reference: np.ndarray, coords: np.ndarray) -> float:
    """RMSD (same unit as input, e.g. nm) of `coords` vs `reference` (n,3)."""
    d = coords - reference
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def interface_contact_mask(target_ca: np.ndarray, binder_ca: np.ndarray,
                           cutoff: float = 0.8) -> np.ndarray:
    """Boolean (nt, nb) mask of CA-CA pairs within `cutoff` (nm)."""
    return _dmat(target_ca, binder_ca) < cutoff


def contact_retention(contact_mask: np.ndarray, target_ca: np.ndarray,
                      binder_ca: np.ndarray, cutoff: float = 0.8) -> float:
    """Fraction of reference contacts still present at the current frame."""
    denom = int(contact_mask.sum())
    if denom == 0:
        return 0.0
    current = _dmat(target_ca, binder_ca) < cutoff
    return float((contact_mask & current).sum() / denom)


def rmsf(frames: np.ndarray) -> np.ndarray:
    """Per-atom RMSF over `frames` (n_frames, n_atoms, 3)."""
    mean = frames.mean(axis=0)
    d = frames - mean
    return np.sqrt(np.mean(np.sum(d * d, axis=2), axis=0))
