"""Content-addressed terrain references shared by training and replay.

Seeds are selectors, not identities.  Every reference therefore carries the
exact semantic SHA-256 recorded by the environment, and hydration refuses a
different local artifact even when its seed matches.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import math
import re
from threading import Lock
from typing import Any, Mapping

from console.core.worlds import terrain, worldgen


REFERENCE_SCHEMA = "hytalerl_replay_world_reference_v1"
CACHE_SIZE = 32
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_CACHE: OrderedDict[str, dict[str, Any] | None] = OrderedDict()
_LOCK = Lock()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 string")
    return value.lower()


def native_reference(seed: int, semantic_sha256: str) -> dict[str, Any]:
    """Reference one native Region capture without embedding its cell arrays."""

    return {
        "schema": REFERENCE_SCHEMA,
        "version": 1,
        "kind": "native_region",
        "seed": int(seed),
        "semantic_sha256": _sha256(semantic_sha256, "Region semantic"),
        "reference_only": True,
    }


def captured_reference(record: Mapping[str, Any]) -> dict[str, Any]:
    """Reference one discovered composite WorldGen publication."""

    capture_id = _sha256(record.get("capture_id"), "capture id")
    return {
        "schema": REFERENCE_SCHEMA,
        "version": 1,
        "kind": "worldgen_v2_composite",
        "seed": int(record["seed"]),
        "semantic_sha256": capture_id,
        "capture_id": capture_id,
        "structure": str(record["structure"]),
        "reference_only": True,
    }


def worldgen_reference(
    world_identity: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Select one exact Region from an Arena WorldGen pool identity."""

    if not isinstance(world_identity, Mapping):
        raise TypeError("world identity must be a mapping")
    capture_id = world_identity.get("capture_id")
    if capture_id is not None:
        return captured_reference(world_identity)
    selected_seed = world_identity.get("seed") if seed is None else seed
    if isinstance(selected_seed, bool) or not isinstance(selected_seed, int):
        raise ValueError("replay world seed must be an integer")
    rows = world_identity.get("artifacts")
    if not isinstance(rows, (list, tuple)):
        raise ValueError("WorldGen identity has no artifact census")
    row = next(
        (
            item
            for item in rows
            if isinstance(item, Mapping) and item.get("seed") == selected_seed
        ),
        None,
    )
    if row is None:
        raise ValueError(f"WorldGen seed {selected_seed} is absent from the pool")
    structure = world_identity.get("structure") or world_identity.get(
        "world_structure_asset_id"
    )
    if not structure:
        raise ValueError("WorldGen identity has no structure")
    return {
        "schema": REFERENCE_SCHEMA,
        "version": 1,
        "kind": "worldgen_v2_region",
        "seed": selected_seed,
        "semantic_sha256": _sha256(
            row.get("region_semantic_sha256"), "Region semantic"
        ),
        "structure": str(structure),
        "bundle_semantic_sha256": _sha256(
            world_identity.get("bundle_semantic_sha256"), "bundle semantic"
        ),
        "publication_semantic_sha256": world_identity.get(
            "publication_semantic_sha256"
        ),
        "reference_only": True,
    }


def _legacy_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    seed = value.get("seed")
    semantic = value.get("semantic_sha256") or value.get(
        "artifact_semantic_sha256"
    )
    if seed is None:
        raise ValueError("legacy terrain reference has no seed")
    if semantic is None:
        return {
            "schema": REFERENCE_SCHEMA,
            "version": 1,
            "kind": "legacy_native_region_unpinned",
            "seed": int(seed),
            "semantic_sha256": None,
            "reference_only": True,
        }
    return native_reference(int(seed), str(semantic))


def _trajectory_axes(
    trajectory: Mapping[str, Any],
) -> tuple[list[float], list[float], list[float]]:
    axes = []
    for axis in "xyz":
        values = list(trajectory.get(f"agent_{axis}") or ())
        values += list(trajectory.get(f"target_{axis}") or ())
        axes.append([float(value) for value in values])
    if not all(axes):
        raise ValueError("terrain hydration needs actor and target coordinates")
    return axes[0], axes[1], axes[2]


def flat_patch(
    xs: list[float],
    ys: list[float],
    zs: list[float],
    *,
    margin: float = 8.0,
    voxel: float = 1.0,
    maximum_cells: int = 20000,
) -> dict[str, Any] | None:
    """Synthesise the ground slab for a design world, which has no voxels.

    A `world_design` run trains on a mathematical plane -- `source_node -1`, no
    Region atlas, nothing to hydrate -- so every other `resolve` branch has
    nothing to look up and the archive drew an empty scene. The floor is
    recoverable from the trajectory itself: the actors stand on it, so the
    minimum y they ever reach IS the plane.

    Marked `synthesised_from_trajectory` rather than
    `rehydrated_from_capture`, because this is drawn from the run's own
    coordinates, not loaded from captured geometry. A viewer must be able to
    tell a reconstruction from a recording.
    """

    if not xs or not zs or not ys:
        return None
    floor = math.floor(min(ys))
    low_x, high_x = math.floor(min(xs) - margin), math.ceil(max(xs) + margin)
    low_z, high_z = math.floor(min(zs) - margin), math.ceil(max(zs) + margin)
    step = max(voxel, 1.0)
    columns = ((high_x - low_x) / step + 1) * ((high_z - low_z) / step + 1)
    # Coarsen rather than truncate: a clipped slab would draw a floor that
    # stops under the actors, which reads as a hole that is not there.
    while columns > maximum_cells:
        step *= 2.0
        columns = ((high_x - low_x) / step + 1) * ((high_z - low_z) / step + 1)
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    position = low_x
    while position <= high_x:
        depth = low_z
        while depth <= high_z:
            x_values.append(round(float(position), 2))
            y_values.append(float(floor))
            z_values.append(round(float(depth), 2))
            depth += step
        position += step
    return {
        "seed": 0,
        "x": x_values,
        "y": y_values,
        "z": z_values,
        "material": [0] * len(x_values),
        "colors": ["#5A6472"],
        "nodes": len(x_values),
        "voxel_size": step,
        "synthesised_from_trajectory": True,
        "floor_y": float(floor),
        "note": (
            "Design worlds carry no captured geometry, so this ground plane is "
            "reconstructed from the trajectory's own minimum height rather than "
            "loaded from an artifact."
        ),
    }


def flat_from_trajectory(
    trajectory: Mapping[str, Any],
) -> dict[str, Any] | None:
    """The design-world entry point: ground from the run's own coordinates."""

    try:
        xs, ys, zs = _trajectory_axes(trajectory)
    except (ValueError, TypeError):
        return None
    return flat_patch(xs, ys, zs)


def resolve(
    reference: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    cache_key: str,
) -> dict[str, Any] | None:
    """Hydrate one replay crop from its exact local artifact."""

    if not isinstance(reference, Mapping):
        raise TypeError("terrain reference must be a mapping")
    normalized = (
        dict(reference)
        if reference.get("schema") == REFERENCE_SCHEMA
        else _legacy_reference(reference)
    )
    if normalized.get("version") != 1:
        raise ValueError("unsupported replay-world reference version")
    key = f"{cache_key}:{normalized['kind']}:{normalized['semantic_sha256']}"
    with _LOCK:
        if key in _CACHE:
            value = _CACHE.pop(key)
            _CACHE[key] = value
            return deepcopy(value)
    xs, ys, zs = _trajectory_axes(trajectory)
    kind = normalized.get("kind")
    if kind == "native_region":
        value = _native_patch(normalized, xs, ys, zs, key)
    elif kind == "legacy_native_region_unpinned":
        value = terrain.cached_patch(
            key, int(normalized["seed"]), xs, ys, zs
        )
    elif kind == "worldgen_v2_region":
        value = _worldgen_patch(normalized, xs, ys, zs)
    elif kind == "worldgen_v2_composite":
        value = _composite_patch(normalized, xs, ys, zs)
    else:
        raise ValueError(f"unsupported replay-world kind {kind!r}")
    if value is not None:
        value["world_reference"] = normalized
        value["rehydrated_from_capture"] = True
    with _LOCK:
        _CACHE[key] = value
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return deepcopy(value)


def _native_patch(reference, xs, ys, zs, key):
    seed = int(reference["seed"])
    expected = _sha256(reference["semantic_sha256"], "Region semantic")
    actual = terrain.artifact_semantic_sha256(seed)
    # `artifact_semantic_sha256` returns None *only* when the identity cannot be
    # computed -- it swallows the missing-sidecar error -- so None means
    # "unverifiable", never "different". Conflating the two turned every replay
    # on a library without block-semantic sidecars into a bare 500, including
    # runs whose geometry and trajectory are perfectly intact.
    #
    # A real mismatch between two computed identities is still fatal: that is
    # terrain which is not the terrain the run happened on, and drawing it would
    # be a lie. Only the unverifiable case degrades, and it says so in the
    # payload rather than passing itself off as checked.
    if actual is not None and actual != expected:
        raise ValueError(
            f"native Region seed {seed} semantic mismatch: {actual} != {expected}"
        )
    value = terrain.cached_patch(key, seed, xs, ys, zs)
    if value is not None:
        value["semantic_identity_verified"] = actual is not None
    return value


def _worldgen_patch(reference, xs, ys, zs):
    # Arena owns the authenticated bundle loader used by the JAX environment;
    # Console deliberately reuses it instead of maintaining a second catalog.
    from arena import worlds as arena_worlds

    bundle = arena_worlds._bundle()
    expected_bundle = _sha256(
        reference["bundle_semantic_sha256"], "bundle semantic"
    )
    if bundle.semantic_sha256.lower() != expected_bundle:
        raise ValueError("local WorldGen bundle differs from replay reference")
    selected = bundle.select(
        str(reference["structure"]), seeds=(int(reference["seed"]),)
    )[0]
    expected_region = _sha256(reference["semantic_sha256"], "Region semantic")
    if selected.region_semantic_sha256.lower() != expected_region:
        raise ValueError("WorldGen seed resolves to a different Region semantic")
    sidecar = (bundle.artifact_root / selected.block_semantic_path).resolve()
    sidecar.relative_to(bundle.artifact_root.resolve())
    return terrain.semantic_patch(
        seed=selected.seed,
        sidecar=sidecar,
        sidecar_file_sha256=selected.block_semantic_file_sha256,
        region_semantic_sha256=expected_region,
        xs=xs,
        ys=ys,
        zs=zs,
    )


def _composite_patch(reference, xs, ys, zs):
    capture_id = _sha256(reference.get("capture_id"), "capture id")
    preview = worldgen.captured_volume(capture_id, target_columns=96)
    if _sha256(preview.get("digest"), "captured world semantic") != capture_id:
        raise ValueError("captured world semantic differs from replay reference")
    if int(preview["config"]["seed"]) != int(reference["seed"]):
        raise ValueError("captured world seed differs from replay reference")
    return _volume_patch(preview["volume"], int(reference["seed"]), xs, ys, zs)


def _index_bounds(values, origin, unit, size, margin):
    lower = max(0, math.floor((min(values) - margin - origin) / unit))
    upper = min(size, math.ceil((max(values) + margin - origin) / unit))
    return lower, upper


def _volume_patch(volume, seed, xs, ys, zs):
    """Decode only the displayed voxel columns surrounding the trajectory."""

    nx, ny, nz = (int(value) for value in volume["dimensions"])
    ox, oy, oz = (float(value) for value in volume["origin"])
    unit = int(volume["voxel_size"])
    x0, x1 = _index_bounds(xs, ox, unit, nx, terrain.MARGIN)
    y0, y1 = _index_bounds(ys, oy, unit, ny, terrain.BELOW)
    z0, z1 = _index_bounds(zs, oz, unit, nz, terrain.MARGIN)
    cells: dict[tuple[int, int, int], int] = {}
    columns = volume["columns"]
    for z in range(z0, z1):
        for x in range(x0, x1):
            runs = columns[z * nx + x]
            for offset in range(0, len(runs), 3):
                start, length, material = map(int, runs[offset : offset + 3])
                for y in range(max(y0, start), min(y1, start + length)):
                    cells[(x, y, z)] = material
    visible = []
    for cell, material in cells.items():
        x, y, z = cell
        faces = 0
        if (x, y + 1, z) not in cells:
            faces |= terrain.FACE_TOP
        if (x - 1, y, z) not in cells:
            faces |= terrain.FACE_X_NEG
        if (x + 1, y, z) not in cells:
            faces |= terrain.FACE_X_POS
        if (x, y, z - 1) not in cells:
            faces |= terrain.FACE_Z_NEG
        if (x, y, z + 1) not in cells:
            faces |= terrain.FACE_Z_POS
        if faces:
            visible.append((cell, material, faces))
    available = len(visible)
    if available > terrain.MAX_BLOCKS:
        sampled = list(zip(xs[:: max(1, len(xs) // 128)], zs[:: max(1, len(zs) // 128)]))
        visible.sort(key=lambda row: min(
            (ox + (row[0][0] + 0.5) * unit - px) ** 2
            + (oz + (row[0][2] + 0.5) * unit - pz) ** 2
            for px, pz in sampled
        ))
        visible = visible[: terrain.MAX_BLOCKS]
    palette = {int(row["id"]): row for row in volume["palette"]}
    used = sorted({row[1] for row in visible})
    material_index = {material: index for index, material in enumerate(used)}
    blocks = {
        "x": [ox + (row[0][0] + 0.5) * unit for row in visible],
        "y": [oy + (row[0][1] + 1) * unit for row in visible],
        "z": [oz + (row[0][2] + 0.5) * unit for row in visible],
        "material": [material_index[row[1]] for row in visible],
        "faces": [row[2] for row in visible],
        "visible": len(visible),
        "available": available,
        "voxel_size": unit,
    }
    return {
        "seed": seed,
        "semantic_sha256": volume["source_composite_semantic_sha256"],
        "x": blocks["x"],
        "y": blocks["y"],
        "z": blocks["z"],
        "material": blocks["material"],
        "colors": [str(palette[material]["color"]) for material in used],
        "nodes": len(visible),
        "thinned_by": 1,
        "voxel_size": unit,
        "blocks": blocks,
        "note": (
            "Trajectory crop decoded from the immutable captured WorldGen "
            "voxel volume; voxel_size records any explicit display LOD."
        ),
    }


def cache_state() -> dict[str, Any]:
    with _LOCK:
        return {
            "schema": "console-replay-world-cache-v1",
            "entries": len(_CACHE),
            "capacity": CACHE_SIZE,
        }


def clear_cache() -> int:
    with _LOCK:
        count = len(_CACHE)
        _CACHE.clear()
    return count


__all__ = [
    "REFERENCE_SCHEMA",
    "cache_state",
    "captured_reference",
    "clear_cache",
    "native_reference",
    "resolve",
    "worldgen_reference",
]
