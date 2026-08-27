"""Command-line interface: run / design / predict / md / serve."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _parse_length(s: str):
    """Parse '50' or '50-80' into (min, max)."""
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    n = int(s)
    return n, n


def _cmd_run(args):
    from .config import Config
    from .pipeline import run_pipeline

    length_min, length_max = _parse_length(args.length)
    config = Config.from_env()
    summary = run_pipeline(
        target=args.target,
        n_designs=args.n_designs,
        length_min=length_min,
        length_max=length_max,
        hotspot=args.hotspot,
        design_provider=args.design_provider,
        structure_provider=args.structure_provider,
        md_top=args.md_top,
        md_ns=args.md_ns,
        md_solvent=args.md_solvent,
        dry_run=args.dry_run,
        results_dir=args.results_dir,
        config=config,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


def _cmd_design(args):
    from .config import Config
    from .pipeline import load_target_sequence, _resolve_target
    from .providers import get_design_provider

    seq, struct_path = _resolve_target(args.target, ".")
    length_min, length_max = _parse_length(args.length)
    designer = get_design_provider(args.design_provider)
    binders = designer.design(seq, struct_path, args.n_designs,
                              length_min, length_max, hotspot=args.hotspot, seed=args.seed)
    with open(args.output, "w", encoding="utf-8") as f:
        for b in binders:
            f.write(f">{b.id}\n{b.sequence}\n")
    print(f"Wrote {len(binders)} binders to {args.output}")


def _cmd_predict(args):
    from .config import Config
    from .pipeline import _resolve_target
    from .providers import get_structure_provider
    from .schemas import Binder

    seq, struct_path = _resolve_target(args.target, ".")
    binder_seq = args.binder
    if binder_seq is None:
        raise SystemExit("--binder is required for `predict`")
    binder = Binder(id="binder_000", sequence=binder_seq, provider="cli")
    predictor = get_structure_provider(args.structure_provider)
    pred = predictor.predict(seq, struct_path, binder, out_dir=args.out_dir)
    print(json.dumps({
        "binder_id": pred.binder_id, "ipTM": pred.ipTM, "pTM": pred.pTM,
        "pLDDT": pred.pLDDT, "structure_path": pred.structure_path,
    }, indent=2))


def _cmd_md(args):
    from .md.engine import run_md

    md = run_md(
        args.complex, binder_chain=args.binder_chain, target_chain=args.target_chain,
        ns=args.ns, solvent=args.solvent, platform=args.platform,
        out_dir=args.out_dir, seed=args.seed,
    )
    print(json.dumps({
        "rmsd_final": md.rmsd_final, "rmsd_mean": md.rmsd_mean,
        "contact_retention": md.contact_retention, "dG": md.dG,
        "rmsf": md.rmsf, "converged": md.converged,
        "trajectory_path": md.trajectory_path,
    }, indent=2))


def _cmd_serve(args):
    try:
        from server.main import create_app
    except ImportError:
        raise SystemExit(
            "Web server not available. Install it with `pip install -e '.[web]'` "
            "and ensure the server/ package is on your path."
        )
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bindforge",
                                description="Design binders -> predict complex -> MD validate.")
    p.add_argument("--version", action="version", version=f"bindforge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # run
    r = sub.add_parser("run", help="Full pipeline (design -> predict -> MD -> rank)")
    r.add_argument("--target", required=True, help="Target protein: FASTA/PDB/CIF path or raw sequence")
    r.add_argument("--n-designs", type=int, default=8)
    r.add_argument("--length", default="50-80", help="Binder length range, e.g. 50-80")
    r.add_argument("--hotspot", default=None, help="Optional hotspot residue indices")
    r.add_argument("--design-provider", default="mock")
    r.add_argument("--structure-provider", default="mock")
    r.add_argument("--md-top", type=int, default=2, help="Number of top candidates to MD-validate")
    r.add_argument("--md-ns", type=float, default=5.0)
    r.add_argument("--md-solvent", default="implicit", choices=["implicit"])
    r.add_argument("--dry-run", action="store_true", help="Force mock providers (no API keys)")
    r.add_argument("--results-dir", default="results")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=_cmd_run)

    # design
    d = sub.add_parser("design", help="Design binder sequences only")
    d.add_argument("--target", required=True)
    d.add_argument("--n-designs", type=int, default=8)
    d.add_argument("--length", default="50-80")
    d.add_argument("--hotspot", default=None)
    d.add_argument("--design-provider", default="mock")
    d.add_argument("--output", default="binders.fasta")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=_cmd_design)

    # predict
    pr = sub.add_parser("predict", help="Predict a single target+binder complex")
    pr.add_argument("--target", required=True)
    pr.add_argument("--binder", required=True, help="Binder amino-acid sequence")
    pr.add_argument("--structure-provider", default="mock")
    pr.add_argument("--out-dir", default=".")
    pr.set_defaults(func=_cmd_predict)

    # md
    m = sub.add_parser("md", help="Run MD stability validation on a complex PDB/CIF")
    m.add_argument("--complex", required=True, help="Complex structure (PDB/CIF)")
    m.add_argument("--binder-chain", default="B")
    m.add_argument("--target-chain", default="A")
    m.add_argument("--ns", type=float, default=5.0)
    m.add_argument("--solvent", default="implicit", choices=["implicit"])
    m.add_argument("--platform", default="auto")
    m.add_argument("--out-dir", default="results")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=_cmd_md)

    # serve
    s = sub.add_parser("serve", help="Start the FastAPI web server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 — friendlier top-level error
        print(f"bindforge: error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
