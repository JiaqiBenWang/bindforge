"""OpenMM MD driver: minimize -> NVT equilibrate -> production -> metrics."""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

from ..schemas import MDResult

from . import implicit, metrics

# Standard protein residue names (plus common modified variants) to keep; the
# rest (water, ions, ligands) are stripped before simulation.
_STANDARD_RES = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS HID HIE HIP ILE LEU LYS MET "
    "PHE PRO SER THR TRP TYR VAL MSE SEP TPO".split()
)

_IMPLICIT_NONBONDED = "NoCutoff"
_EQ_PS = 20.0  # short NVT equilibration length (ps)


def _pick_platform(name: str):
    from openmm import Platform

    want = name.lower()
    if want in ("cuda", "opencl"):
        for i in range(Platform.getNumPlatforms()):
            p = Platform.getPlatform(i)
            if p.getName().lower() == want:
                return p
        raise RuntimeError(f"Platform {name!r} requested but not available.")
    if want == "cpu":
        return Platform.getPlatformByName("CPU")
    # auto: prefer CUDA -> OpenCL -> CPU
    for i in range(Platform.getNumPlatforms()):
        p = Platform.getPlatform(i)
        if p.getName() in ("CUDA", "OpenCL"):
            return p
    return Platform.getPlatformByName("CPU")


def _load_structure(path: str):
    from openmm.app import PDBFile, PDBxFile

    low = path.lower()
    if low.endswith((".cif", ".mmcif")):
        return PDBxFile(path)
    return PDBFile(path)


def _clean(modeller):
    """Delete water, ions, ligands and non-protein residues."""
    to_delete = [r for r in modeller.topology.residues() if r.name not in _STANDARD_RES]
    if to_delete:
        modeller.delete(to_delete)
    return modeller


def _ca_indices(topology, chain_id: str) -> List[int]:
    return [a.index for a in topology.atoms()
            if a.name == "CA" and a.residue.chain.id == chain_id]


def _heavy_indices(topology, chain_id: str) -> List[int]:
    """Heavy-atom (non-hydrogen) indices for a chain — used for interface contacts."""
    return [a.index for a in topology.atoms()
            if a.element is not None and a.element.symbol != "H"
            and a.residue.chain.id == chain_id]


def _coords_nm(context, indices: List[int]) -> np.ndarray:
    """Extract selected atom coordinates (nm) from the current context state."""
    pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions()
    return np.array([[pos[i].x, pos[i].y, pos[i].z] for i in indices], dtype=np.float64)


def _mmgbsa_dg(final_pdb_path: str, target_chain: str, binder_chain: str) -> float:
    """Single-trajectory MM-GBSA estimate: dG = E_complex - E_target - E_binder.

    Uses the final (already hydrogenated) snapshot so the three energies are
    consistent. Returns kJ/mol.
    """
    from openmm import Context, LangevinMiddleIntegrator
    from openmm.app import ForceField, Modeller, PDBFile, NoCutoff, HBonds
    from openmm.unit import kelvin, picoseconds, femtoseconds, kilojoule_per_mole

    base = PDBFile(final_pdb_path)
    ff = ForceField(*implicit.FORCEFIELD_XMLS)

    def _energy(modeller):
        system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        integ = LangevinMiddleIntegrator(300 * kelvin, 1 / picoseconds, 2 * femtoseconds)
        ctx = Context(system, integ)
        ctx.setPositions(modeller.positions)
        state = ctx.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    complex_mod = Modeller(base.topology, base.positions)
    e_complex = _energy(complex_mod)

    target_mod = Modeller(base.topology, base.positions)
    target_mod.delete([c for c in target_mod.topology.chains() if c.id != target_chain])
    e_target = _energy(target_mod)

    binder_mod = Modeller(base.topology, base.positions)
    binder_mod.delete([c for c in binder_mod.topology.chains() if c.id != binder_chain])
    e_binder = _energy(binder_mod)

    return float(e_complex - e_target - e_binder)


def run_md(complex_path: str, binder_chain: str = "B", target_chain: str = "A",
           ns: float = 5.0, solvent: str = "implicit", temperature: float = 300.0,
           dt_fs: float = 2.0, platform: str = "auto", out_dir: Optional[str] = None,
           seed: int = 0, quiet: bool = False) -> MDResult:
    """Run an implicit-GBSA MD simulation and return stability metrics.

    Parameters are in the schemas.MDResult units: lengths in nm (RMSD), energy
    in kJ/mol, time in ns.
    """
    from openmm import Context, LangevinMiddleIntegrator, LocalEnergyMinimizer
    from openmm.app import ForceField, Modeller
    from openmm.unit import nanometer, picoseconds, femtoseconds, kelvin

    if solvent != "implicit":
        raise NotImplementedError(
            f"Solvent {solvent!r} not implemented yet (phase 2 explicit solvent). "
            "Use solvent='implicit'."
        )

    def _log(msg):
        if not quiet:
            print(f"[md] {msg}", file=sys.stderr)

    out_dir = out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(complex_path))[0]

    # --- 1. Load & prepare ------------------------------------------------
    struct = _load_structure(complex_path)
    modeller = Modeller(struct.topology, struct.positions)
    _clean(modeller)
    ff = ForceField(*implicit.FORCEFIELD_XMLS)
    modeller.addHydrogens(ff)
    system = implicit.build_system(ff, modeller)

    integ = LangevinMiddleIntegrator(
        temperature * kelvin, 1 / picoseconds, dt_fs * femtoseconds
    )
    integ.setRandomNumberSeed(seed)
    ctx = Context(system, integ, _pick_platform(platform))
    ctx.setPositions(modeller.positions)

    binder_ca = _ca_indices(modeller.topology, binder_chain)
    target_ca = _ca_indices(modeller.topology, target_chain)
    if not binder_ca or not target_ca:
        raise ValueError(
            f"Could not find CA atoms for binder_chain={binder_chain!r} / "
            f"target_chain={target_chain!r} in {complex_path}."
        )
    binder_heavy = _heavy_indices(modeller.topology, binder_chain)
    target_heavy = _heavy_indices(modeller.topology, target_chain)
    _log(f"system ready: {system.getNumParticles()} atoms, "
         f"binder {len(binder_ca)} CA, target {len(target_ca)} CA")

    try:
        # --- 2. Minimize -------------------------------------------------------
        _log("minimizing ...")
        LocalEnergyMinimizer.minimize(ctx, maxIterations=2000)
        ref_pos = _coords_nm(ctx, binder_ca)
        if not np.isfinite(ref_pos).all():
            raise RuntimeError("Minimization produced non-finite coordinates.")

        # Reference interface contacts (heavy-atom pairs within 4.5 A) after
        # minimization. RMSD uses CA; contacts use all heavy atoms because the
        # interface is defined by atom packing, not backbone proximity.
        ref_target = _coords_nm(ctx, target_ca)
        ref_target_heavy = _coords_nm(ctx, target_heavy)
        ref_binder_heavy = _coords_nm(ctx, binder_heavy)
        contact_mask = metrics.interface_contact_mask(ref_target_heavy, ref_binder_heavy)
        n_ref_contacts = int(contact_mask.sum())
        if n_ref_contacts == 0:
            _log("WARNING: no heavy-atom interface contacts in the starting pose "
                 "(chains are not in contact) — contact retention is not "
                 "measurable and will be reported as null, not 0.")
        else:
            _log(f"reference interface: {n_ref_contacts} heavy-atom contacts")

        # --- 3. NVT equilibration --------------------------------------------
        eq_steps = int(_EQ_PS * 1000.0 / dt_fs)
        _log(f"equilibrating {_EQ_PS} ps ({eq_steps} steps) ...")
        integ.step(eq_steps)

        # --- 4. Production ----------------------------------------------------
        total_steps = int(ns * 1e6 / dt_fs)
        save_ps = 10.0 if ns < 1.0 else 100.0
        save_steps = max(1, int(save_ps * 1000.0 / dt_fs))
        n_frames = total_steps // save_steps

        binder_frames = []   # (n_frames, n_binder_ca, 3), target-superposed
        rmsd_series = []     # per-frame binder RMSD vs reference (nm)
        retention_series = []
        for f in range(n_frames):
            integ.step(save_steps)
            b = _coords_nm(ctx, binder_ca)
            t = _coords_nm(ctx, target_ca)
            t_heavy = _coords_nm(ctx, target_heavy)
            b_heavy = _coords_nm(ctx, binder_heavy)
            # Superpose on the target before measuring the binder, so RMSD and
            # RMSF report motion relative to the target rather than the free
            # tumbling of the whole complex (nothing restrains it here).
            b_aligned = metrics.superpose(t, ref_target, b)
            binder_frames.append(b_aligned)
            rmsd_series.append(metrics.ca_rmsd(ref_pos, b_aligned))
            # Contacts are internal distances, so they need no alignment.
            retention_series.append(metrics.contact_retention(contact_mask, t_heavy, b_heavy))

        binder_frames = np.asarray(binder_frames)
        rmsd_series = np.asarray(rmsd_series)

        rmsd_final = float(rmsd_series[-1])
        rmsd_mean = float(rmsd_series.mean())
        # None whenever the reference had no interface at all -> not measurable.
        measured = [r for r in retention_series if r is not None]
        contact_retention_mean = float(np.mean(measured)) if measured else None
        rmsf_arr = metrics.rmsf(binder_frames)  # per-residue nm

        # --- 5. Final snapshot + MM-GBSA --------------------------------------
        from openmm.app import PDBFile
        final_pdb = os.path.join(out_dir, f"{base}_md_final.pdb")
        state = ctx.getState(getPositions=True, enforcePeriodicBox=False)
        PDBFile.writeFile(modeller.topology, state.getPositions(), open(final_pdb, "w"))

        dg = None
        try:
            dg = _mmgbsa_dg(final_pdb, target_chain, binder_chain)
            _log(f"MM-GBSA dG = {dg:.1f} kJ/mol")
        except Exception as exc:  # noqa: BLE001 — dG is best-effort, never fatal
            _log(f"MM-GBSA dG failed ({exc}); leaving unset")

        converged = bool(np.isfinite(rmsd_series).all() and rmsd_final < 2.0)

        retention_str = ("n/a (no starting interface)" if contact_retention_mean is None
                         else f"{contact_retention_mean:.2%}")
        _log(f"done: rmsd_final={rmsd_final:.3f} nm, rmsd_mean={rmsd_mean:.3f} nm, "
             f"contact_retention={retention_str}")

        return MDResult(
            binder_id=None,
            solvent=solvent,
            ns=float(ns),
            rmsd_final=rmsd_final,
            rmsd_mean=rmsd_mean,
            contact_retention=contact_retention_mean,
            dG=dg,
            rmsf=float(rmsf_arr.mean()),
            rmsf_residues=rmsf_arr.tolist(),
            trajectory_path=final_pdb,
            converged=converged,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 — a NaN in one candidate must not kill the run
        _log(f"MD failed: {exc}")
        # Every metric stays None: a failed run measured nothing. Reporting 0.0
        # here would hand the candidate a perfect pose score (exp(-0/0.5) == 1).
        return MDResult(
            binder_id=None,
            solvent=solvent,
            ns=float(ns),
            rmsd_final=None,
            rmsd_mean=None,
            contact_retention=None,
            dG=None,
            rmsf=None,
            rmsf_residues=None,
            trajectory_path=None,
            converged=False,
            error=str(exc),
        )
