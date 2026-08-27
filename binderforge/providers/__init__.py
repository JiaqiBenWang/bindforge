"""Provider registry and factory functions."""

from __future__ import annotations

from typing import Optional, Type

from .base import DesignProvider, StructureProvider
from .mock import MockDesignProvider, MockStructureProvider
from .nvidia_nim import NvidiaBoltz2Provider
from .boltzbio import BoltzGenProvider, BoltzBioStructureProvider
from .chai import ChaiProvider

DESIGN_PROVIDERS = {
    MockDesignProvider.name: MockDesignProvider,
    BoltzGenProvider.name: BoltzGenProvider,
}

STRUCTURE_PROVIDERS = {
    MockStructureProvider.name: MockStructureProvider,
    NvidiaBoltz2Provider.name: NvidiaBoltz2Provider,
    BoltzBioStructureProvider.name: BoltzBioStructureProvider,
    ChaiProvider.name: ChaiProvider,
}


def get_design_provider(name: str, **kwargs) -> DesignProvider:
    cls = DESIGN_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown design provider {name!r}; available: {sorted(DESIGN_PROVIDERS)}")
    return cls(**kwargs)


def get_structure_provider(name: str, **kwargs) -> StructureProvider:
    cls = STRUCTURE_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown structure provider {name!r}; available: {sorted(STRUCTURE_PROVIDERS)}")
    return cls(**kwargs)
