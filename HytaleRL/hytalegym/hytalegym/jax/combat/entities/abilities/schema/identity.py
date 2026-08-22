"""Canonical identity for a compiled fixed-shape ability bank."""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np


def entity_ability_program_sha256(programs=None) -> str:
    """Hash field names, shapes, dtypes, and canonical little-endian bytes."""

    if programs is None:
        from hytalegym.jax.combat.entities.abilities.factory import (
            hytale_0_5_7_entity_ability_programs,
        )

        programs = hytale_0_5_7_entity_ability_programs()
    digest = hashlib.sha256(b"hytalerl_entity_ability_program_bank_v1\0")
    for name in programs._fields:
        value = np.asarray(getattr(programs, name))
        dtype = value.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(value, dtype=dtype)
        metadata = json.dumps(
            {
                "dtype": dtype.str,
                "field": name,
                "shape": list(canonical.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(struct.pack("<Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack("<Q", canonical.nbytes))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest().upper()


__all__ = ["entity_ability_program_sha256"]
