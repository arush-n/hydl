"""Authored structure markers, from contract to JAX metadata.

``structure_contract``        shared authored-marker contract for identity
``structure_jax``             fixed-shape JAX metadata for marker structures
``structure_pack``            marker-backed placed structure sidecars
``native_structure_markers``  checked host adapter for authored marker rows
"""

from __future__ import annotations

from . import native_structure_markers as native_structure_markers
from . import structure_contract as structure_contract
from . import structure_jax as structure_jax
from . import structure_pack as structure_pack


__all__ = [
    "native_structure_markers",
    "structure_contract",
    "structure_jax",
    "structure_pack",
]
