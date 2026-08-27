"""Explicit-solvent (TIP3P) system construction — optional high-accuracy mode.

Planned for phase 2: Amber ff14SB + TIP3P water box + 0.15 M NaCl under PME.
Not wired into the run path yet (explicit MD is prohibitively slow on CPU for
typical complexes; it targets GPU/cloud runners).
"""

from __future__ import annotations

FORCEFIELD_XMLS = ["amber14-all.xml", "amber14/tip3p.xml"]


def build_system(forcefield, modeller, padding_nm=1.0, ionic_strength_molar=0.15):
    from openmm.app import PME, HBonds
    from openmm.unit import nanometer, molar

    modeller.addSolvent(
        forcefield, model="tip3p", padding=padding_nm * nanometer,
        ionicStrength=ionic_strength_molar * molar,
    )
    return forcefield.createSystem(
        modeller.topology, nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer, constraints=HBonds,
    )
