"""Compact persistent state representations for multi-actor training."""

from .locomotion import (
    PackedEntityLocomotionState,
    pack_entity_locomotion_state,
    unpack_entity_locomotion_state,
)

__all__ = [
    "PackedEntityLocomotionState",
    "pack_entity_locomotion_state",
    "unpack_entity_locomotion_state",
]
