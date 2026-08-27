"""Boltz-2 structure prediction via NVIDIA NIM (hosted at health.api.nvidia.com).

Uses the official ``boltz2-client`` package (MIT, Python 3.8+). Install it with
``pip install boltz2-python-client`` (or ``pip install bindforge[nvidia]``), and
set ``NVIDIA_API_KEY`` (obtain one at https://build.nvidia.com).

The multimer API (``Polymer`` + ``PredictionRequest``) is used to predict the
target (chain A) + binder (chain B) complex. The result CIF and ipTM/pTM/pLDDT
are extracted into a :class:`~binderforge.schemas.ComplexPrediction`.
"""

from __future__ import annotations

import os
from typing import Optional

from ..schemas import Binder, ComplexPrediction
from .base import StructureProvider


class NvidiaBoltz2Provider(StructureProvider):
    name = "nvidia_boltz2"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from boltz2_client import Boltz2SyncClient  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "The NVIDIA Boltz-2 provider requires the 'boltz2-client' package. "
                    "Install it with: pip install boltz2-python-client"
                ) from e
            if not self.api_key:
                raise RuntimeError("NVIDIA_API_KEY is not set (see .env.example).")
            from boltz2_client import Boltz2SyncClient
            self._client = Boltz2SyncClient(
                base_url="https://health.api.nvidia.com",
                api_key=self.api_key,
                endpoint_type="nvidia_hosted",
            )
        return self._client

    def predict(self, target_seq, target_struct_path, binder, out_dir=None,
                recycling_steps=3, sampling_steps=100, **kwargs) -> ComplexPrediction:
        from boltz2_client import Polymer, PredictionRequest

        client = self._get_client()
        chain_a = Polymer(id="A", molecule_type="protein", sequence=target_seq)
        chain_b = Polymer(id="B", molecule_type="protein", sequence=binder.sequence)
        request = PredictionRequest(
            polymers=[chain_a, chain_b],
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
        )
        result = client.predict(request)

        structure = result.structures[0].structure  # CIF text
        structure_path = None
        if structure and out_dir:
            os.makedirs(out_dir, exist_ok=True)
            structure_path = os.path.join(out_dir, f"{binder.id}.cif")
            with open(structure_path, "w", encoding="utf-8") as f:
                f.write(structure)
            structure = None  # keep memory light; MD reads from structure_path

        ipTM = result.iptm_scores[0] if getattr(result, "iptm_scores", None) else None
        pTM = result.ptm_scores[0] if getattr(result, "ptm_scores", None) else None
        plddt = (result.complex_plddt_scores[0]
                 if getattr(result, "complex_plddt_scores", None) else None)

        return ComplexPrediction(
            binder_id=binder.id,
            structure=structure,
            structure_path=structure_path,
            ipTM=float(ipTM) if ipTM is not None else None,
            pTM=float(pTM) if pTM is not None else None,
            pLDDT=float(plddt) if plddt is not None else None,
            provider=self.name,
        )
