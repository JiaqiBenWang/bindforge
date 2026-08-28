"""Pipeline orchestration: design -> predict -> MD validate -> rank -> report."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

from . import io
from .config import Config
from .md.engine import run_md
from .providers import get_design_provider, get_structure_provider
from .schemas import Binder, ComplexPrediction, MDResult, RankedCandidate
from .scoring import rank_candidates

_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_target_sequence(target: str) -> str:
    """Resolve a target into a raw sequence (from FASTA, PDB/CIF, or raw text)."""
    if os.path.isfile(target):
        low = target.lower()
        if low.endswith((".pdb", ".cif", ".mmcif")):
            seq = io.sequence_from_pdb(target)
            if seq:
                return seq
            raise ValueError(f"No CA atoms found in structure {target!r}")
        seqs = io.read_fasta(target)
        if seqs:
            return seqs[0]
        raise ValueError(f"Could not parse FASTA from {target!r}")
    raw = "".join(c for c in target.upper() if c in _AA)
    if not raw:
        raise ValueError(f"Target {target!r} is neither a file nor a protein sequence")
    return raw


def _resolve_target(target: str, work_dir: str) -> Tuple[str, Optional[str]]:
    """Return (sequence, structure_path). Builds a target PDB if none is given."""
    seq = load_target_sequence(target)

    struct_path: Optional[str] = None
    if os.path.isfile(target):
        low = target.lower()
        if low.endswith((".pdb", ".cif", ".mmcif")):
            struct_path = os.path.abspath(target)
    if struct_path is None:
        struct_path = os.path.join(work_dir, "target.pdb")
        io.build_peptide_pdb(seq, struct_path, chain="A")
    return seq, struct_path


def _round(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    """Round, passing None (an unmeasured component) straight through."""
    return None if value is None else round(value, ndigits)


def _candidate_to_dict(r: RankedCandidate) -> dict:
    pred, md = r.prediction, r.md
    return {
        "rank": r.rank,
        "binder_id": r.binder.id,
        "sequence": r.binder.sequence,
        "length": r.binder.length,
        "score": _round(r.score),
        "validated": r.validated,
        "confidence": _round(r.confidence),
        "stability": _round(r.stability),
        "binding": _round(r.binding),
        "pose": _round(r.pose),
        "ipTM": pred.ipTM,
        "pTM": pred.pTM,
        "pLDDT": pred.pLDDT,
        "structure_path": pred.structure_path,
        "md_rmsd_final": (md.rmsd_final if md else None),
        "md_rmsd_mean": (md.rmsd_mean if md else None),
        "md_contact_retention": (md.contact_retention if md else None),
        "md_dG": (md.dG if md else None),
        "md_rmsf": (md.rmsf if md else None),
        "md_converged": (md.converged if md else None),
    }


def run_pipeline(
    target: str,
    n_designs: int = 8,
    length_min: int = 50,
    length_max: int = 80,
    hotspot: Optional[str] = None,
    design_provider: str = "mock",
    structure_provider: str = "mock",
    md_top: int = 2,
    md_ns: float = 5.0,
    md_solvent: str = "implicit",
    dry_run: bool = False,
    results_dir: str = "results",
    config: Optional[Config] = None,
    seed: int = 0,
) -> Dict:
    """Run the full pipeline. Returns a summary dict (also written to disk)."""
    os.makedirs(results_dir, exist_ok=True)
    seq, struct_path = _resolve_target(target, results_dir)

    if dry_run:
        design_provider = "mock"
        structure_provider = "mock"

    designer = get_design_provider(design_provider)
    binder_sequences = designer.design(
        seq, struct_path, n_designs, length_min, length_max, hotspot=hotspot, seed=seed
    )

    predictor = get_structure_provider(structure_provider)
    predictions: Dict[str, ComplexPrediction] = {}
    for b in binder_sequences:
        predictions[b.id] = predictor.predict(
            seq, struct_path, b, out_dir=results_dir
        )

    # MD on the top-K predictions (ranked by confidence) to save time.
    md_results: Dict[str, MDResult] = {}
    ranked_top = sorted(
        predictions.values(), key=lambda p: (p.ipTM or 0.0) + (p.pTM or 0.0), reverse=True
    )[:md_top]
    for pred in ranked_top:
        if not pred.structure_path or not os.path.isfile(pred.structure_path):
            continue
        md = run_md(
            pred.structure_path,
            binder_chain="B",
            target_chain="A",
            ns=md_ns,
            solvent=md_solvent,
            platform=(config.md_platform if config else "auto"),
            out_dir=results_dir,
            seed=seed,
        )
        md.binder_id = pred.binder_id
        md_results[pred.binder_id] = md

    ranking = rank_candidates(binder_sequences, predictions, md_results)

    # --- report -----------------------------------------------------------
    summary = write_report(ranking, results_dir)
    summary.update(dict(
        target_sequence=seq,
        target_structure=struct_path,
        n_designs=len(binder_sequences),
        n_md_run=len(md_results),
    ))
    return summary


def write_report(ranking: List[RankedCandidate], results_dir: str) -> Dict:
    """Write ranking.json + ranking.csv and return the rows as a dict."""
    rows = [_candidate_to_dict(r) for r in ranking]

    json_path = os.path.join(results_dir, "ranking.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = os.path.join(results_dir, "ranking.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    return {"results": rows, "ranking_json": json_path, "ranking_csv": csv_path}
