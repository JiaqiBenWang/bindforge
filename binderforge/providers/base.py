"""Provider abstractions for the design and structure-prediction stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..schemas import Binder, ComplexPrediction


class DesignProvider(ABC):
    """Generates candidate binder sequences for a target."""

    name = "base"

    @abstractmethod
    def design(self, target_seq: str, target_struct_path: Optional[str],
               n_designs: int, length_min: int, length_max: int,
               hotspot: Optional[str] = None, **kwargs) -> List[Binder]:
        ...


class StructureProvider(ABC):
    """Predicts the target+binder complex structure and confidence scores."""

    name = "base"

    @abstractmethod
    def predict(self, target_seq: str, target_struct_path: Optional[str],
                binder: Binder, out_dir: Optional[str] = None,
                **kwargs) -> ComplexPrediction:
        ...
