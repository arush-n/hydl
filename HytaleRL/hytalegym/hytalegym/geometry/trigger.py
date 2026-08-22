"""Bounded decoder for the optional native collision-trigger sidecar."""

from __future__ import annotations

import hashlib
from numbers import Integral
from typing import Any, Iterable

import numpy as np

from hytalegym.geometry.contract import (
    CELL_COUNT,
    MAX_DETAIL_BOXES,
    GeometryCapacityError,
    cell_index,
)


TRIGGER_SCHEMA = "hytale_trigger_transport_v1"
TRIGGER_VERSION = 1
MAX_TRIGGER_PROGRAMS = 64
NO_PROGRAM = -1


def trigger_program_catalog_sha256(program_ids: Iterable[str]) -> str:
    """Return the Java-compatible identity for an ordered program catalog."""

    digest = hashlib.sha256()
    for program_id in program_ids:
        if not isinstance(program_id, str):
            raise ValueError("trigger program IDs must be strings")
        encoded = program_id.encode("utf-8")
        try:
            digest.update(len(encoded).to_bytes(4, "big"))
        except OverflowError as exception:
            raise GeometryCapacityError("trigger program ID is too long") from exception
        digest.update(encoded)
    return digest.hexdigest().upper()


def empty_trigger_transport(
    *, data_unavailable: bool = True, capacity_exceeded: bool = False
) -> dict[str, Any]:
    """Return an explicit fail-closed trigger frame."""

    if not (data_unavailable or capacity_exceeded):
        raise ValueError("unavailable trigger frames require a failure flag")
    return {
        "available": 0,
        "data_unavailable": int(data_unavailable),
        "capacity_exceeded": int(capacity_exceeded),
        "program_catalog_sha256": "",
        "program_ids": (),
        "cell_mask": np.zeros(CELL_COUNT, dtype=np.int8),
        "base_offset": np.zeros((CELL_COUNT, 3), dtype=np.int32),
        "program_index": np.full((CELL_COUNT, 3), NO_PROGRAM, dtype=np.int32),
        "boxes": np.zeros(
            (CELL_COUNT, MAX_DETAIL_BOXES, 6), dtype=np.float32
        ),
        "box_mask": np.zeros(
            (CELL_COUNT, MAX_DETAIL_BOXES), dtype=np.int8
        ),
    }


def parse_trigger_transport(raw: Any) -> dict[str, Any]:
    """Validate and materialize one optional trigger sidecar.

    Missing sidecars and declared failures remain distinguishable from an
    available frame with no active trigger cells. Capacity mismatches raise;
    no engine state is truncated.
    """

    if raw is None:
        return empty_trigger_transport()
    if not isinstance(raw, dict):
        raise ValueError("trigger payload must be an object or null")

    required = {
        "schema",
        "version",
        "available",
        "data_unavailable",
        "capacity_exceeded",
        "program_capacity",
        "cell_capacity",
        "box_capacity_per_cell",
        "program_catalog_sha256",
        "program_ids",
        "cells",
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError(
            f"trigger contract is missing {', '.join(sorted(missing))}"
        )

    version = _integer(raw["version"], "trigger version")
    if raw["schema"] != TRIGGER_SCHEMA or version != TRIGGER_VERSION:
        raise ValueError(
            f"unsupported trigger contract {raw['schema']!r} version {version}"
        )
    _capacity(raw, "program_capacity", MAX_TRIGGER_PROGRAMS)
    _capacity(raw, "cell_capacity", CELL_COUNT)
    _capacity(raw, "box_capacity_per_cell", MAX_DETAIL_BOXES)

    available = _boolean(raw["available"], "available")
    data_unavailable = _boolean(raw["data_unavailable"], "data_unavailable")
    capacity_exceeded = _boolean(raw["capacity_exceeded"], "capacity_exceeded")
    if available == (data_unavailable or capacity_exceeded):
        raise ValueError("trigger availability disagrees with its failure flags")

    program_ids = raw["program_ids"]
    cells = raw["cells"]
    catalog_sha256 = raw["program_catalog_sha256"]
    if not isinstance(program_ids, (list, tuple)):
        raise ValueError("trigger program_ids must be an array")
    if not isinstance(cells, (list, tuple)):
        raise ValueError("trigger cells must be an array")
    if not isinstance(catalog_sha256, str):
        raise ValueError("trigger catalog identity must be a string")

    if not available:
        if catalog_sha256 or program_ids or cells:
            raise ValueError("unavailable trigger frames must fail closed")
        return empty_trigger_transport(
            data_unavailable=data_unavailable,
            capacity_exceeded=capacity_exceeded,
        )

    if len(program_ids) > MAX_TRIGGER_PROGRAMS:
        raise GeometryCapacityError(
            f"trigger has {len(program_ids)} programs; "
            f"capacity is {MAX_TRIGGER_PROGRAMS}"
        )
    if any(not isinstance(item, str) or not item.strip() for item in program_ids):
        raise ValueError("trigger program IDs must be non-empty strings")
    if len(set(program_ids)) != len(program_ids):
        raise ValueError("trigger program IDs must be unique")
    expected_sha256 = trigger_program_catalog_sha256(program_ids)
    if catalog_sha256 != expected_sha256:
        raise ValueError("trigger program catalog identity does not match")
    if len(cells) > CELL_COUNT:
        raise GeometryCapacityError(
            f"trigger has {len(cells)} cells; capacity is {CELL_COUNT}"
        )

    result = empty_trigger_transport()
    result["available"] = 1
    result["data_unavailable"] = 0
    result["program_catalog_sha256"] = catalog_sha256
    result["program_ids"] = tuple(program_ids)
    seen: set[int] = set()
    for cell in cells:
        _materialize_cell(result, cell, len(program_ids), seen)
    return result


def _materialize_cell(
    result: dict[str, Any], cell: Any, program_count: int, seen: set[int]
) -> None:
    if not isinstance(cell, (list, tuple)) or len(cell) != 10:
        raise ValueError("trigger cell must use the 10-field layout")

    integers = [
        _integer(value, "trigger cell integer") for value in cell[:9]
    ]
    if any(
        value < np.iinfo(np.int32).min or value > np.iinfo(np.int32).max
        for value in integers
    ):
        raise GeometryCapacityError("trigger cell integer exceeds int32")
    dx, dy, dz, *base_and_programs = integers
    index = cell_index(dx, dy, dz)
    if index in seen:
        raise ValueError(f"duplicate trigger cell {(dx, dy, dz)}")
    seen.add(index)

    base_offset = base_and_programs[:3]
    program_index = base_and_programs[3:]
    if any(value < NO_PROGRAM or value >= program_count for value in program_index):
        raise ValueError("invalid trigger program index")
    if all(value == NO_PROGRAM for value in program_index):
        raise ValueError("trigger cells require at least one phase program")

    boxes = np.asarray(cell[9], dtype=np.float64).reshape(-1)
    if not boxes.size or boxes.size % 6:
        raise ValueError(
            "trigger boxes must contain complete non-empty six-value rows"
        )
    box_count = boxes.size // 6
    if box_count > MAX_DETAIL_BOXES:
        raise GeometryCapacityError(
            f"trigger cell {(dx, dy, dz)} has {box_count} boxes; "
            f"capacity is {MAX_DETAIL_BOXES}"
        )
    rows = boxes.reshape(box_count, 6)
    if not np.all(np.isfinite(rows)) or np.any(rows[:, :3] > rows[:, 3:]):
        raise ValueError("trigger box bounds must be finite and ordered")

    result["cell_mask"][index] = 1
    result["base_offset"][index] = base_offset
    result["program_index"][index] = program_index
    result["boxes"][index, :box_count] = rows.astype(np.float32)
    result["box_mask"][index, :box_count] = 1


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"trigger {name} must be a boolean")
    return bool(value)


def _capacity(raw: dict[str, Any], name: str, expected: int) -> None:
    actual = _integer(raw[name], f"trigger {name}")
    if actual != expected:
        raise GeometryCapacityError(
            f"trigger {name} is {actual}; expected {expected}"
        )
