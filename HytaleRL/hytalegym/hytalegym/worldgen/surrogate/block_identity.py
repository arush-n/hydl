"""Exact per-cell block identity, kept beside the geometry palette.

The geometry palette deduplicates cells by collision and LOS geometry, so
distinct assets can share one cell code (see tests/worldgen/surrogate
test_palette.py). Identity therefore lives in its own table: row 0 is air, and
every stored cell carries the index of the row the generator placed in it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from hytalegym.worldgen.block_actions import block_asset_key, block_semantic_key
from hytalegym.worldgen.surrogate.semantics import LocalBlockSemantics

BLOCK_IDENTITY_PROVENANCE = "surrogate"
AIR_IDENTITY_ROW = 0
IDENTITY_INDEX_DTYPE = np.uint16
MAXIMUM_IDENTITY_ROWS = int(np.iinfo(IDENTITY_INDEX_DTYPE).max) + 1
_AIR_ASSET = "Empty"
_KEY_WORDS = len(block_asset_key(_AIR_ASSET))

ReferenceKey = tuple[str, int, int, int]


@dataclass(frozen=True)
class BlockIdentityRow:
    asset_id: str
    rotation: int
    asset_key: tuple[int, ...]
    semantic_key: tuple[int, ...]


def _air_row() -> BlockIdentityRow:
    zeros = (0,) * _KEY_WORDS
    return BlockIdentityRow("", 0, zeros, zeros)


def _reference_key(value: LocalBlockSemantics) -> ReferenceKey:
    return (value.reference, value.rotation, value.support, value.filler)


@dataclass(frozen=True)
class CompiledBlockIdentity:
    """Rows are distinct resolved references; row 0 is air.

    ``keys[i]`` is the palette reference key of row ``i`` (``None`` for air),
    so a consumer can match a row against a compiled palette.
    """

    rows: tuple[BlockIdentityRow, ...]
    keys: tuple[ReferenceKey | None, ...]
    _row_by_key: dict[ReferenceKey, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.keys) or not self.rows:
            raise ValueError("block identity rows and keys must align")
        if len(self.rows) > MAXIMUM_IDENTITY_ROWS:
            raise ValueError("block identity table exceeds the index dtype")
        if self.rows[AIR_IDENTITY_ROW] != _air_row() or self.keys[0] is not None:
            raise ValueError("block identity row 0 must be air")
        row_by_key: dict[ReferenceKey, int] = {}
        for index, key in enumerate(self.keys[1:], start=1):
            if key is None or key in row_by_key:
                raise ValueError("block identity keys must be unique and non-air")
            row_by_key[key] = index
        object.__setattr__(self, "_row_by_key", row_by_key)

    def row_for(self, semantic: LocalBlockSemantics) -> int:
        """Return the row of an exact resolved reference; air maps to row 0."""

        if not isinstance(semantic, LocalBlockSemantics):
            raise TypeError("semantic must be a LocalBlockSemantics")
        if semantic.asset_id == _AIR_ASSET:
            return AIR_IDENTITY_ROW
        return self.row_for_key(_reference_key(semantic))

    def row_for_key(self, key: ReferenceKey) -> int:
        try:
            return self._row_by_key[key]
        except KeyError as error:
            raise KeyError(f"block identity has no row for {key!r}") from error


def compile_block_identity(
    semantics: Sequence[LocalBlockSemantics],
) -> CompiledBlockIdentity:
    """Give every distinct non-air resolved reference one exact identity row."""

    values = list(semantics)
    if any(not isinstance(value, LocalBlockSemantics) for value in values):
        raise TypeError("semantics must contain LocalBlockSemantics values")
    records: dict[ReferenceKey, BlockIdentityRow] = {}
    for value in values:
        if value.asset_id == _AIR_ASSET:
            continue
        row = BlockIdentityRow(
            asset_id=value.asset_id,
            rotation=value.rotation,
            asset_key=block_asset_key(value.asset_id),
            semantic_key=block_semantic_key(value.asset_id, value.rotation),
        )
        previous = records.setdefault(_reference_key(value), row)
        if previous != row:
            raise ValueError(
                f"block identity {_reference_key(value)!r} resolves inconsistently"
            )
    keys = tuple(sorted(records))
    return CompiledBlockIdentity(
        rows=(_air_row(),) + tuple(records[key] for key in keys),
        keys=(None,) + keys,
    )


def merge_block_identities(
    tables: Sequence[CompiledBlockIdentity],
) -> tuple[CompiledBlockIdentity, tuple[np.ndarray, ...]]:
    """Union identity tables by reference key; return per-source row remaps."""

    source = list(tables)
    if not source:
        raise ValueError("at least one block identity table is required")
    if any(not isinstance(table, CompiledBlockIdentity) for table in source):
        raise TypeError("tables must contain CompiledBlockIdentity values")
    records: dict[ReferenceKey, BlockIdentityRow] = {}
    for table in source:
        for key, row in zip(table.keys[1:], table.rows[1:], strict=True):
            previous = records.setdefault(key, row)
            if previous != row:
                raise ValueError(f"block identity {key!r} conflicts across tables")
    keys = tuple(sorted(records))
    merged = CompiledBlockIdentity(
        rows=(_air_row(),) + tuple(records[key] for key in keys),
        keys=(None,) + keys,
    )
    remaps: list[np.ndarray] = []
    for table in source:
        remap = np.zeros(len(table.rows), dtype=IDENTITY_INDEX_DTYPE)
        for index, key in enumerate(table.keys[1:], start=1):
            remap[index] = merged.row_for_key(key)
        remap.flags.writeable = False
        remaps.append(remap)
    return merged, tuple(remaps)


__all__ = [
    "AIR_IDENTITY_ROW",
    "BLOCK_IDENTITY_PROVENANCE",
    "BlockIdentityRow",
    "CompiledBlockIdentity",
    "IDENTITY_INDEX_DTYPE",
    "compile_block_identity",
    "merge_block_identities",
]
