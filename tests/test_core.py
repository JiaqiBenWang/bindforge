"""Unit tests for io, scoring, and md.metrics (no OpenMM required)."""

import numpy as np
import pytest

from binderforge import io
from binderforge.md import metrics
from binderforge.schemas import Binder, ComplexPrediction, MDResult
from binderforge.scoring import confidence_score, rank_candidates


def test_read_fasta(tmp_path):
    p = tmp_path / "t.fasta"
    p.write_text(">a\nACDE\n>b\nFG\n", encoding="utf-8")
    assert io.read_fasta(str(p)) == ["ACDE", "FG"]


def test_build_peptide_roundtrip(tmp_path):
    atoms = io.build_peptide_pdb("ACDE")
    names = {a["name"] for a in atoms}
    assert {"N", "CA", "C", "O", "OXT"}.issubset(names)  # OXT on C-term
    # no clashing heavy atoms
    xyz = np.array([[a["x"], a["y"], a["z"]] for a in atoms])
    d = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 0.9  # Å


def test_sequence_from_pdb(tmp_path):
    p = tmp_path / "t.pdb"
    io.build_peptide_pdb("MKTA", str(p), chain="A")
    assert io.sequence_from_pdb(str(p)) == "MKTA"


def test_ca_rmsd():
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert metrics.ca_rmsd(ref, ref) == 0.0
    moved = ref + np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert metrics.ca_rmsd(ref, moved) == pytest.approx(1.0)


def test_contact_retention():
    t = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[0.5, 0.0, 0.0]])
    mask = metrics.interface_contact_mask(t, b, cutoff=0.8)
    assert mask[0, 0]
    # moved far away -> retention drops to 0
    assert metrics.contact_retention(mask, t, np.array([[5.0, 0.0, 0.0]])) == 0.0


def test_confidence_fallback():
    assert confidence_score(ComplexPrediction(binder_id="b", ipTM=0.9, pTM=0.7)) == pytest.approx(0.8)
    assert confidence_score(ComplexPrediction(binder_id="b", pLDDT=80.0)) == pytest.approx(0.8)


def test_rank_candidates_sorts():
    b1 = Binder(id="b1", sequence="AAAA")
    b2 = Binder(id="b2", sequence="AAAA")
    preds = {
        "b1": ComplexPrediction(binder_id="b1", ipTM=0.9, pTM=0.9),
        "b2": ComplexPrediction(binder_id="b2", ipTM=0.5, pTM=0.5),
    }
    ranked = rank_candidates([b1, b2], preds, {})
    assert ranked[0].binder.id == "b1"
    assert ranked[0].rank == 1
