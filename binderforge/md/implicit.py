"""Implicit-solvent (GBSA) system construction — the fast default."""

from __future__ import annotations

# Amber ff14SB protein + OBC2 generalized-Born implicit solvent.
FORCEFIELD_XMLS = ["amber14-all.xml", "implicit/obc2.xml"]


def build_system(forcefield, modeller):
    """Create the OpenMM System (no explicit solvent, NoCutoff)."""
    from openmm.app import NoCutoff, HBonds
    return forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds
    )
