"""Unit tests for io, scoring, and md.metrics (no OpenMM required)."""

import numpy as np
import pytest

from binderforge import io
from binderforge.md import metrics
from binderforge.schemas import Binder, ComplexPrediction, MDResult
from binderforge.scoring import confidence_score, rank_candidates, score_candidate


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


def test_build_complex_pdb_makes_an_interface(tmp_path):
    target = tmp_path / "target.pdb"
    io.build_peptide_pdb("MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP", str(target), chain="A")
    out = tmp_path / "complex.pdb"
    io.build_complex_pdb(str(target), "ACDEFGHIKLMNPQRSTVWY" * 3, str(out))

    heavy_a = []
    heavy_b = []
    for a in io.parse_pdb(str(out)):
        xyz = np.array([a["x"], a["y"], a["z"]])
        (heavy_a if a["chain"] == "A" else heavy_b).append(xyz)
    d = np.sqrt(((np.array(heavy_a)[:, None, :] - np.array(heavy_b)[None, :, :]) ** 2).sum(2))
    assert d.min() < 4.5  # heavy atoms are actually in contact
    assert d.min() > 2.0  # but not clashing through each other


def test_build_complex_canonicalises_chain_labels(tmp_path):
    # Real PDBs use arbitrary chain IDs (here "B"); the complex must still map
    # target -> "A" and binder -> "B" so the MD stage's hardcoded A/B lookup works.
    target = tmp_path / "target_chainB.pdb"
    io.build_peptide_pdb("MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP", str(target), chain="B")
    out = tmp_path / "complex.pdb"
    io.build_complex_pdb(str(target), "ACDEFGHIKLMNPQRSTVWY", str(out))

    ca = {}
    for a in io.parse_pdb(str(out)):
        if a["name"] == "CA":
            ca[a["chain"]] = ca.get(a["chain"], 0) + 1
    assert set(ca) == {"A", "B"}
    assert ca["A"] == 37   # the target sequence length
    assert ca["B"] == 20   # the binder sequence length


def test_ca_rmsd():
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert metrics.ca_rmsd(ref, ref) == 0.0
    moved = ref + np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert metrics.ca_rmsd(ref, moved) == pytest.approx(1.0)


def test_kabsch_rotation_degenerate_anchor_is_identity():
    # A collinear (rank-1) anchor or non-finite input must not crash the run;
    # the rotation is ill-defined there, so the safe answer is identity.
    col = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.allclose(metrics.kabsch_rotation(col, col), np.eye(3))
    nan = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.allclose(metrics.kabsch_rotation(nan, col), np.eye(3))


def test_superpose_removes_rigid_body_motion():
    # A reference L-shape; the mobile copy is rotated + translated, with an
    # extra atom that is NOT part of the anchor (a "binder") riding along.
    ref_anchor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    binder_local = np.array([[0.5, 0.0, 3.0]])
    R = metrics.kabsch_rotation(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                                np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    translation = np.array([10.0, -5.0, 2.0])
    mobile_anchor = ref_anchor @ R + translation
    mobile_binder = binder_local @ R + translation
    mapped = metrics.superpose(mobile_anchor, ref_anchor, mobile_binder)
    assert np.allclose(mapped, binder_local, atol=1e-9)


def test_contact_retention():
    # Heavy-atom contact definition: pairs within 0.45 nm.
    t = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[0.3, 0.0, 0.0]])
    mask = metrics.interface_contact_mask(t, b)
    assert mask[0, 0]
    # moved far away -> retention drops to 0
    assert metrics.contact_retention(mask, t, np.array([[5.0, 0.0, 0.0]])) == 0.0


def test_contact_retention_empty_reference_is_none():
    # No contacts in the reference -> the quantity is unmeasurable, not "lost".
    t = np.array([[0.0, 0.0, 0.0]])
    far = np.array([[9.0, 9.0, 9.0]])
    mask = metrics.interface_contact_mask(t, far)
    assert mask.sum() == 0
    assert metrics.contact_retention(mask, t, far) is None


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


def test_score_renormalizes_over_measured_components():
    # Confidence only -> score equals confidence (all MD components unmeasured).
    pred = ComplexPrediction(binder_id="b", ipTM=0.9, pTM=0.9)  # conf = 0.9
    comp = score_candidate(pred, None)
    assert comp["confidence"] == pytest.approx(0.9)
    assert comp["score"] == pytest.approx(0.9)
    assert comp["stability"] is None and comp["validated"] is False

    # A fully-measured MD run -> the normal weighted sum.
    md = MDResult(rmsd_mean=0.3, contact_retention=0.8, dG=-40.0)
    comp = score_candidate(pred, md)
    assert comp["stability"] == pytest.approx(0.8)
    assert comp["binding"] == pytest.approx(1.0)
    assert comp["pose"] == pytest.approx(np.exp(-0.3 / 0.5))
    assert comp["validated"] is True
    # ~0.35*0.9 + 0.25*0.8 + 0.25*1.0 + 0.15*exp(-0.6)
    assert comp["score"] == pytest.approx(
        0.35 * 0.9 + 0.25 * 0.8 + 0.25 * 1.0 + 0.15 * np.exp(-0.6))


def test_failed_md_does_not_get_perfect_pose():
    # A crashed MD (all metrics None) must not yield exp(-0/0.5) == 1.0.
    md = MDResult(rmsd_mean=None, contact_retention=None, dG=None, converged=False)
    comp = score_candidate(ComplexPrediction(binder_id="b", ipTM=0.5, pTM=0.5), md)
    assert comp["pose"] is None
    assert comp["stability"] is None
    assert comp["validated"] is False


def test_validated_candidates_rank_first():
    b_hi = Binder(id="hi", sequence="AAAA")   # unvalidated but high confidence
    b_lo = Binder(id="lo", sequence="AAAA")   # validated but low confidence
    preds = {
        "hi": ComplexPrediction(binder_id="hi", ipTM=0.99, pTM=0.99),
        "lo": ComplexPrediction(binder_id="lo", ipTM=0.4, pTM=0.4),
    }
    md = {"lo": MDResult(rmsd_mean=0.3, contact_retention=0.8, dG=-40.0)}
    ranked = rank_candidates([b_hi, b_lo], preds, md)
    # lo is validated and must come first despite the lower confidence.
    assert ranked[0].binder.id == "lo"
