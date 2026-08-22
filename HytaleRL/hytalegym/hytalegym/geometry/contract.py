"""Fixed Hytale geometry contract shared by native and accelerated backends."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


GEOMETRY_SCHEMA = "hytale_geometry_v5"
GEOMETRY_VERSION = 5
CELL_RADIUS = 4
CELL_SIDE = CELL_RADIUS * 2 + 1
CELL_COUNT = CELL_SIDE**3
MAX_DETAIL_BOXES = 9
MAX_CONTACTS = 64
MOVEMENT_FEATURES = 9
FLUID_MOVEMENT_FEATURES = 6
CONTACT_FEATURES = 13
GEOMETRY_BINARY_TRANSPORT = "hytale_geometry_binary_v1"
_BINARY_CELL_INTS = 12
_BINARY_CELL_FLOATS = 16

FLAG_SOLID = 1
# Role-independent non-transparent LOS predicate; effective role sets are separate.
FLAG_OPAQUE = 1 << 1
FLAG_FLUID = 1 << 2
FLAG_DAMAGING = 1 << 3
FLAG_CLIMBABLE = 1 << 4
FLAG_BOUNCY = 1 << 5
FLAG_TRIGGER = 1 << 6
FLAG_PROTRUDES_CELL = 1 << 7
FLAG_HAS_MOVEMENT_SETTINGS = 1 << 8
FLAG_HAS_FLUID_MOVEMENT_SETTINGS = 1 << 9


class GeometryCapacityError(ValueError):
    """The engine produced state that cannot fit the certified fixed schema."""


def geometry_space() -> spaces.Dict:
    """Return the model-agnostic fixed-shape Gymnasium geometry space."""

    i32 = np.iinfo(np.int32)
    return spaces.Dict(
        {
            "available": spaces.Discrete(2),
            "exact_collision_shapes": spaces.Discrete(2),
            "origin": spaces.Box(i32.min, i32.max, shape=(3,), dtype=np.int32),
            "cell_mask": spaces.MultiBinary(CELL_COUNT),
            "runtime_block_id": spaces.Box(
                0, i32.max, shape=(CELL_COUNT,), dtype=np.int32
            ),
            "runtime_fluid_id": spaces.Box(
                0, i32.max, shape=(CELL_COUNT,), dtype=np.int32
            ),
            "shape_id": spaces.Box(
                0, i32.max, shape=(CELL_COUNT,), dtype=np.int32
            ),
            "rotation": spaces.Box(
                0, 255, shape=(CELL_COUNT,), dtype=np.int32
            ),
            "flags": spaces.Box(
                0, i32.max, shape=(CELL_COUNT,), dtype=np.int32
            ),
            "fluid_level": spaces.Box(
                0, 255, shape=(CELL_COUNT,), dtype=np.int32
            ),
            "fluid_fill_height": spaces.Box(
                -1.0,
                1.0,
                shape=(CELL_COUNT,),
                dtype=np.float32,
            ),
            "support": spaces.Box(
                i32.min, i32.max, shape=(CELL_COUNT,), dtype=np.int32
            ),
            "block_damage": spaces.Box(
                i32.min, i32.max, shape=(CELL_COUNT,), dtype=np.int32
            ),
            "fluid_damage": spaces.Box(
                i32.min, i32.max, shape=(CELL_COUNT,), dtype=np.int32
            ),
            "movement": spaces.Box(
                -1.0e6,
                1.0e6,
                shape=(CELL_COUNT, MOVEMENT_FEATURES),
                dtype=np.float32,
            ),
            "fluid_movement": spaces.Box(
                -1.0e6,
                1.0e6,
                shape=(CELL_COUNT, FLUID_MOVEMENT_FEATURES),
                dtype=np.float32,
            ),
            "collision_boxes": spaces.Box(
                -1.0e6,
                1.0e6,
                shape=(CELL_COUNT, MAX_DETAIL_BOXES, 6),
                dtype=np.float32,
            ),
            "collision_box_mask": spaces.MultiBinary(
                (CELL_COUNT, MAX_DETAIL_BOXES)
            ),
            "agent_bounds": spaces.Box(
                -1.0e6, 1.0e6, shape=(6,), dtype=np.float32
            ),
            "target_bounds": spaces.Box(
                -1.0e6, 1.0e6, shape=(6,), dtype=np.float32
            ),
            "agent_los_offset": spaces.Box(
                -1.0e6, 1.0e6, shape=(3,), dtype=np.float32
            ),
            "target_los_offset": spaces.Box(
                -1.0e6, 1.0e6, shape=(3,), dtype=np.float32
            ),
            "contacts": spaces.Box(
                -1.0e6,
                1.0e6,
                shape=(MAX_CONTACTS, CONTACT_FEATURES),
                dtype=np.float32,
            ),
            "contact_mask": spaces.MultiBinary(MAX_CONTACTS),
            "grounded": spaces.Discrete(2),
            "ceiling_contact": spaces.Discrete(2),
            "target_los": spaces.Discrete(2),
            "target_los_valid": spaces.Discrete(2),
        }
    )


def empty_geometry() -> dict[str, Any]:
    """Return an explicit unavailable frame, not an implied all-air world."""

    return {
        "available": 0,
        "exact_collision_shapes": 0,
        "origin": np.zeros(3, dtype=np.int32),
        "cell_mask": np.zeros(CELL_COUNT, dtype=np.int8),
        "runtime_block_id": np.zeros(CELL_COUNT, dtype=np.int32),
        "runtime_fluid_id": np.zeros(CELL_COUNT, dtype=np.int32),
        "shape_id": np.zeros(CELL_COUNT, dtype=np.int32),
        "rotation": np.zeros(CELL_COUNT, dtype=np.int32),
        "flags": np.zeros(CELL_COUNT, dtype=np.int32),
        "fluid_level": np.zeros(CELL_COUNT, dtype=np.int32),
        "fluid_fill_height": np.zeros(CELL_COUNT, dtype=np.float32),
        "support": np.zeros(CELL_COUNT, dtype=np.int32),
        "block_damage": np.zeros(CELL_COUNT, dtype=np.int32),
        "fluid_damage": np.zeros(CELL_COUNT, dtype=np.int32),
        "movement": np.zeros(
            (CELL_COUNT, MOVEMENT_FEATURES), dtype=np.float32
        ),
        "fluid_movement": np.zeros(
            (CELL_COUNT, FLUID_MOVEMENT_FEATURES), dtype=np.float32
        ),
        "collision_boxes": np.zeros(
            (CELL_COUNT, MAX_DETAIL_BOXES, 6), dtype=np.float32
        ),
        "collision_box_mask": np.zeros(
            (CELL_COUNT, MAX_DETAIL_BOXES), dtype=np.int8
        ),
        "agent_bounds": np.zeros(6, dtype=np.float32),
        "target_bounds": np.zeros(6, dtype=np.float32),
        "agent_los_offset": np.zeros(3, dtype=np.float32),
        "target_los_offset": np.zeros(3, dtype=np.float32),
        "contacts": np.zeros(
            (MAX_CONTACTS, CONTACT_FEATURES), dtype=np.float32
        ),
        "contact_mask": np.zeros(MAX_CONTACTS, dtype=np.int8),
        "grounded": 0,
        "ceiling_contact": 0,
        "target_los": 0,
        "target_los_valid": 0,
    }


def cell_index(dx: int, dy: int, dz: int) -> int:
    """Map a signed local cell coordinate to the canonical flat index."""

    if any(abs(value) > CELL_RADIUS for value in (dx, dy, dz)):
        raise GeometryCapacityError(
            f"cell offset {(dx, dy, dz)} exceeds radius {CELL_RADIUS}"
        )
    return (
        (dx + CELL_RADIUS) * CELL_SIDE * CELL_SIDE
        + (dy + CELL_RADIUS) * CELL_SIDE
        + (dz + CELL_RADIUS)
    )


def parse_geometry(raw: Any) -> dict[str, Any]:
    """Validate and materialize one wire frame into fixed NumPy arrays.

    Capacity mismatches raise instead of truncating engine state. This is a
    transfer-safety boundary: a backend cannot claim geometry fidelity after
    silently dropping a box, cell, or collision contact.
    """

    if isinstance(raw, dict) and "transport" in raw:
        if raw["transport"] != GEOMETRY_BINARY_TRANSPORT:
            raise ValueError(f"unsupported geometry transport {raw['transport']!r}")
        return _parse_binary_geometry(raw)

    result = empty_geometry()
    if raw is None:
        return result
    if not isinstance(raw, dict):
        raise ValueError("geometry payload must be an object or null")

    if "schema" not in raw or "version" not in raw:
        raise ValueError("geometry contract requires explicit schema and version")
    schema = raw["schema"]
    version = raw["version"]
    if isinstance(version, bool) or not isinstance(version, (int, np.integer)):
        raise ValueError("geometry contract version must be an integer")
    if schema != GEOMETRY_SCHEMA or version != GEOMETRY_VERSION:
        raise ValueError(
            f"unsupported geometry contract {schema!r} version {version}"
        )

    result["available"] = int(bool(raw.get("available", False)))
    result["exact_collision_shapes"] = int(
        bool(raw.get("exact_collision_shapes", False))
    )
    origin = np.asarray(raw.get("origin", [0, 0, 0]), dtype=np.int64).reshape(-1)
    if origin.size != 3 or np.any(origin < np.iinfo(np.int32).min) or np.any(
        origin > np.iinfo(np.int32).max
    ):
        raise GeometryCapacityError("geometry origin must contain three int32 values")
    result["origin"][:] = origin.astype(np.int32)

    seen: set[int] = set()
    cells = raw.get("cells", [])
    if not isinstance(cells, (list, tuple)):
        raise ValueError("geometry cells must be an array")
    if len(cells) > CELL_COUNT:
        raise GeometryCapacityError(
            f"geometry has {len(cells)} cells; capacity is {CELL_COUNT}"
        )
    for cell in cells:
        if not isinstance(cell, (list, tuple)) or len(cell) != 29:
            raise ValueError("geometry cell must use the 29-field layout")
        dx, dy, dz = (int(cell[index]) for index in range(3))
        index = cell_index(dx, dy, dz)
        if index in seen:
            raise ValueError(f"duplicate geometry cell {(dx, dy, dz)}")
        seen.add(index)
        fluid_fill_height = float(cell[9])
        if (
            not np.isfinite(fluid_fill_height)
            or not 0.0 <= fluid_fill_height <= 1.0
        ):
            raise ValueError("fluid fill height must be finite and in [0, 1]")
        boxes = np.asarray(cell[28], dtype=np.float32).reshape(-1)
        if boxes.size % 6 != 0:
            raise ValueError("collision box payload must be divisible by six")
        box_count = boxes.size // 6
        if box_count > MAX_DETAIL_BOXES:
            raise GeometryCapacityError(
                f"cell {(dx, dy, dz)} has {box_count} boxes; "
                f"capacity is {MAX_DETAIL_BOXES}"
            )

        integer_values = [
            int(value) for value in (*cell[3:9], *cell[10:13])
        ]
        if any(
            value < np.iinfo(np.int32).min
            or value > np.iinfo(np.int32).max
            for value in integer_values
        ):
            raise GeometryCapacityError("geometry cell integer exceeds int32")
        (
            runtime_block_id,
            runtime_fluid_id,
            shape_id,
            rotation,
            flags,
            fluid_level,
            support,
            block_damage,
            fluid_damage,
        ) = integer_values
        if runtime_block_id < 0 or runtime_fluid_id < 0 or shape_id < 0:
            raise ValueError(
                "runtime block, fluid, and shape identifiers must be non-negative"
            )
        if not 0 <= rotation <= 255 or not 0 <= fluid_level <= 255:
            raise ValueError("rotation and fluid level must fit uint8")
        is_fluid = bool(flags & FLAG_FLUID)
        if is_fluid and runtime_fluid_id == 0:
            raise ValueError("fluid flags require a runtime fluid identifier")
        if is_fluid and fluid_level > 0 and fluid_fill_height <= 0.0:
            raise ValueError("positive fluid level requires positive fill height")
        if not is_fluid and (
            runtime_fluid_id != 0
            or fluid_level != 0
            or fluid_fill_height != 0.0
        ):
            raise ValueError("non-fluid cells cannot publish fluid state")

        result["cell_mask"][index] = 1
        result["runtime_block_id"][index] = runtime_block_id
        result["runtime_fluid_id"][index] = runtime_fluid_id
        result["shape_id"][index] = shape_id
        result["rotation"][index] = rotation
        result["flags"][index] = flags
        result["fluid_level"][index] = fluid_level
        result["fluid_fill_height"][index] = fluid_fill_height
        result["support"][index] = support
        result["block_damage"][index] = block_damage
        result["fluid_damage"][index] = fluid_damage
        movement = np.asarray(cell[13:22], dtype=np.float32)
        fluid_movement = np.asarray(cell[22:28], dtype=np.float32)
        if not np.all(np.isfinite(movement)) or not np.all(
            np.isfinite(fluid_movement)
        ):
            raise ValueError("geometry movement values must be finite")
        if flags & FLAG_FLUID and fluid_movement[3] <= 0.0:
            raise ValueError(
                "fluid cells require a positive horizontal speed multiplier"
            )
        result["movement"][index] = movement
        result["fluid_movement"][index] = fluid_movement
        if box_count:
            result["collision_boxes"][index, :box_count] = boxes.reshape(
                box_count, 6
            )
            result["collision_box_mask"][index, :box_count] = 1

    for name in ("agent_bounds", "target_bounds"):
        bounds = np.asarray(raw.get(name, np.zeros(6)), dtype=np.float32)
        if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
            raise ValueError(f"{name} must contain six finite values")
        if np.any(bounds[:3] > bounds[3:]):
            raise ValueError(f"{name} must not contain inverted bounds")
        result[name][:] = bounds

    for name in ("agent_los_offset", "target_los_offset"):
        offset = np.asarray(raw.get(name, np.zeros(3)), dtype=np.float32)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError(f"{name} must contain three finite values")
        result[name][:] = offset

    contacts = raw.get("contacts", [])
    if not isinstance(contacts, (list, tuple)):
        raise ValueError("geometry contacts must be an array")
    if len(contacts) > MAX_CONTACTS:
        raise GeometryCapacityError(
            f"geometry has {len(contacts)} contacts; capacity is {MAX_CONTACTS}"
        )
    for index, contact in enumerate(contacts):
        values = np.asarray(contact, dtype=np.float32)
        if values.shape != (CONTACT_FEATURES,) or not np.all(np.isfinite(values)):
            raise ValueError("geometry contact must contain 13 finite values")
        result["contacts"][index] = values
        result["contact_mask"][index] = 1

    result["grounded"] = int(bool(raw.get("grounded", False)))
    result["ceiling_contact"] = int(bool(raw.get("ceiling_contact", False)))
    result["target_los"] = int(bool(raw.get("target_los", False)))
    result["target_los_valid"] = int(
        bool(raw.get("target_los_valid", False))
    )
    return result


def _parse_binary_geometry(raw: dict[str, Any]) -> dict[str, Any]:
    """Materialize the trace-only columnar geometry transport."""

    if raw.get("schema") != GEOMETRY_SCHEMA or raw.get("version") != GEOMETRY_VERSION:
        raise ValueError("binary geometry contract does not match geometry v5")
    result = empty_geometry()
    result["available"] = int(bool(raw.get("available", False)))
    result["exact_collision_shapes"] = int(
        bool(raw.get("exact_collision_shapes", False))
    )

    def column(name: str, dtype: str, size: int) -> np.ndarray:
        payload = raw.get(name)
        itemsize = np.dtype(dtype).itemsize
        if not isinstance(payload, bytes) or len(payload) != size * itemsize:
            raise ValueError(f"{name} must contain {size} {dtype} values")
        return np.frombuffer(payload, dtype=dtype, count=size)

    origin = column("origin_i32_le", "<i4", 3)
    result["origin"][:] = origin
    cell_count = raw.get("cell_count")
    if (
        isinstance(cell_count, bool)
        or not isinstance(cell_count, (int, np.integer))
        or not 0 <= int(cell_count) <= CELL_COUNT
    ):
        raise GeometryCapacityError("binary geometry cell_count exceeds capacity")
    cell_count = int(cell_count)
    integers = column(
        "cell_i32_le", "<i4", cell_count * _BINARY_CELL_INTS
    ).reshape(cell_count, _BINARY_CELL_INTS)
    values = column(
        "cell_f64_le", "<f8", cell_count * _BINARY_CELL_FLOATS
    ).reshape(cell_count, _BINARY_CELL_FLOATS)
    box_counts = column("collision_box_counts_u8", "u1", cell_count)
    if np.any(box_counts > MAX_DETAIL_BOXES):
        raise GeometryCapacityError("binary geometry cell exceeds box capacity")
    boxes = column(
        "collision_boxes_f64_le", "<f8", int(np.sum(box_counts)) * 6
    )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(boxes)):
        raise ValueError("binary geometry contains nonfinite values")

    if cell_count:
        offsets = integers[:, :3].astype(np.int64)
        if np.any(np.abs(offsets) > CELL_RADIUS):
            raise GeometryCapacityError("binary geometry cell exceeds radius")
        indexes = (
            (offsets[:, 0] + CELL_RADIUS) * CELL_SIDE * CELL_SIDE
            + (offsets[:, 1] + CELL_RADIUS) * CELL_SIDE
            + offsets[:, 2]
            + CELL_RADIUS
        )
        if np.unique(indexes).size != cell_count:
            raise ValueError("binary geometry contains duplicate cells")
        if np.any(integers[:, 3:6] < 0):
            raise ValueError("binary geometry identifiers must be nonnegative")
        if np.any((integers[:, 6] < 0) | (integers[:, 6] > 255)) or np.any(
            (integers[:, 8] < 0) | (integers[:, 8] > 255)
        ):
            raise ValueError("binary geometry rotation/fluid level must fit uint8")
        fill = values[:, 0]
        if np.any((fill < 0.0) | (fill > 1.0)):
            raise ValueError("binary geometry fluid fill must be in [0, 1]")
        fluid = (integers[:, 7] & FLAG_FLUID) != 0
        if np.any(fluid & (integers[:, 4] == 0)):
            raise ValueError("binary fluid cells require a fluid identifier")
        if np.any(fluid & (integers[:, 8] > 0) & (fill <= 0.0)):
            raise ValueError("binary positive fluid level requires positive fill")
        if np.any(
            ~fluid
            & ((integers[:, 4] != 0) | (integers[:, 8] != 0) | (fill != 0.0))
        ):
            raise ValueError("binary non-fluid cells publish fluid state")
        if np.any(fluid & (values[:, 13] <= 0.0)):
            raise ValueError("binary fluid cells require horizontal movement")

        result["cell_mask"][indexes] = 1
        for name, column_index in (
            ("runtime_block_id", 3),
            ("runtime_fluid_id", 4),
            ("shape_id", 5),
            ("rotation", 6),
            ("flags", 7),
            ("fluid_level", 8),
            ("support", 9),
            ("block_damage", 10),
            ("fluid_damage", 11),
        ):
            result[name][indexes] = integers[:, column_index]
        result["fluid_fill_height"][indexes] = values[:, 0]
        result["movement"][indexes] = values[:, 1:10]
        result["fluid_movement"][indexes] = values[:, 10:16]
        cursor = 0
        for index, count in zip(indexes, box_counts, strict=True):
            count = int(count)
            width = count * 6
            if count:
                result["collision_boxes"][index, :count] = boxes[
                    cursor : cursor + width
                ].reshape(count, 6)
                result["collision_box_mask"][index, :count] = 1
            cursor += width

    bounds = column("bounds_offsets_f64_le", "<f8", 18)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("binary geometry bounds contain nonfinite values")
    for name, start, stop in (
        ("agent_bounds", 0, 6),
        ("target_bounds", 6, 12),
        ("agent_los_offset", 12, 15),
        ("target_los_offset", 15, 18),
    ):
        result[name][:] = bounds[start:stop]
    for name in ("agent_bounds", "target_bounds"):
        if np.any(result[name][:3] > result[name][3:]):
            raise ValueError(f"{name} must not contain inverted bounds")

    contact_count = raw.get("contact_count")
    if (
        isinstance(contact_count, bool)
        or not isinstance(contact_count, (int, np.integer))
        or not 0 <= int(contact_count) <= MAX_CONTACTS
    ):
        raise GeometryCapacityError("binary geometry contact_count exceeds capacity")
    contact_count = int(contact_count)
    contacts = column(
        "contacts_f64_le", "<f8", contact_count * CONTACT_FEATURES
    ).reshape(contact_count, CONTACT_FEATURES)
    if not np.all(np.isfinite(contacts)):
        raise ValueError("binary geometry contacts contain nonfinite values")
    result["contacts"][:contact_count] = contacts
    result["contact_mask"][:contact_count] = 1

    state_flags = raw.get("state_flags")
    if (
        isinstance(state_flags, bool)
        or not isinstance(state_flags, (int, np.integer))
        or not 0 <= int(state_flags) <= 0xF
    ):
        raise ValueError("binary geometry state_flags contains unknown bits")
    state_flags = int(state_flags)
    result["grounded"] = state_flags & 1
    result["ceiling_contact"] = (state_flags >> 1) & 1
    result["target_los"] = (state_flags >> 2) & 1
    result["target_los_valid"] = (state_flags >> 3) & 1
    return result
