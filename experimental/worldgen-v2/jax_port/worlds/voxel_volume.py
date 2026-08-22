"""Browser-sized voxel views derived from exact composite Region artifacts.

This is an adapter, not another generator.  It reads the same immutable
physical, stable-block, and stable-fluid artifacts consumed by the JAX Region
atlas.  A one-Region view can retain one native block per displayed voxel;
larger worlds use an explicit power-of-two display LOD while the source/JAX
artifacts remain exact.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from scipy.ndimage import binary_propagation as _binary_propagation
except ImportError:  # pragma: no cover - the JAX environment ships SciPy
    _binary_propagation = None

from hytalegym.geometry.contract import FLAG_FLUID, FLAG_SOLID
from hytalegym.worldgen.region import (
    CAPTURE_CHUNKS_PER_AXIS,
    CAPTURE_HALO_CHUNKS,
    CHUNK_SIZE,
    CORE_BLOCKS_PER_AXIS,
    HEIGHT_SECTIONS,
    SECTION_VOLUME,
    WORLD_HEIGHT,
)

from .composite_world import (
    COMPOSITE_JAX_WORLD_ID,
    COMPOSITE_WORLD_AUTHORITY,
    COMPOSITE_WORLD_RUNTIME_SOURCE,
    COMPOSITE_WORLD_SCHEMA,
    V2CompositeWorld,
    load_composite_tile_artifacts,
)


VOXEL_VOLUME_SCHEMA = "hytalerl_worldgen_v2_semantic_voxel_volume_v1"
CAPTURED_PREVIEW_SCHEMA = "hytalerl_worldgen_v2_captured_voxel_preview_v1"
DEFAULT_TARGET_COLUMNS = 96
MIN_TARGET_COLUMNS = 24
MAX_TARGET_COLUMNS = 192
MAX_VISUAL_MATERIALS = (1 << 16) - 1

_NATURAL_SOLID_COLORS = (
    "#596166", "#6f746d", "#7e8178", "#67523e",
    "#765d43", "#876d4e", "#9a825c", "#a09169",
    "#536747", "#65784b", "#748950", "#82985b",
    "#3f6548", "#4f7550", "#66745f", "#827762",
)
_NATURAL_FEATURE_COLORS = (
    "#315f3f", "#3f7047", "#4f7d4d", "#628b54",
    "#775c3c", "#896a43", "#718a4c", "#527348",
)


def _dense_capture(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    expected = (
        CAPTURE_CHUNKS_PER_AXIS**2,
        HEIGHT_SECTIONS,
        SECTION_VOLUME,
    )
    if source.shape != expected:
        raise ValueError(
            f"Region section payload has shape {source.shape}, expected {expected}"
        )
    return source.reshape(
        CAPTURE_CHUNKS_PER_AXIS,
        CAPTURE_CHUNKS_PER_AXIS,
        HEIGHT_SECTIONS,
        CHUNK_SIZE,
        CHUNK_SIZE,
        CHUNK_SIZE,
    ).transpose(0, 5, 2, 3, 1, 4).reshape(
        CAPTURE_CHUNKS_PER_AXIS * CHUNK_SIZE,
        WORLD_HEIGHT,
        CAPTURE_CHUNKS_PER_AXIS * CHUNK_SIZE,
    )


def _dense_core(values: np.ndarray) -> np.ndarray:
    start = CAPTURE_HALO_CHUNKS * CHUNK_SIZE
    stop = start + CORE_BLOCKS_PER_AXIS
    return _dense_capture(values)[start:stop, :, start:stop]


def _target_columns(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("target_columns must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target_columns must be an integer") from exc
    if parsed != value or not MIN_TARGET_COLUMNS <= parsed <= MAX_TARGET_COLUMNS:
        raise ValueError(
            f"target_columns must be between {MIN_TARGET_COLUMNS} and "
            f"{MAX_TARGET_COLUMNS}"
        )
    return parsed


def _voxel_unit(width: int, depth: int, target_columns: int) -> int:
    requested = max(1, math.ceil(max(width, depth) / target_columns))
    return 1 << math.ceil(math.log2(requested))


def _hex_words(words: Sequence[int]) -> str:
    return "".join(f"{int(word):08x}" for word in words)


def _stable_color(identity: Any, *, fluid: bool, solid: bool) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    if fluid:
        hue = 0.52 + (hue - 0.5) * 0.10
        saturation, lightness = 0.62, 0.48
        red, green, blue = colorsys.hls_to_rgb(
            hue % 1.0, lightness, saturation
        )
        return (
            f"#{round(red * 255):02x}{round(green * 255):02x}"
            f"{round(blue * 255):02x}"
        )
    palette = _NATURAL_SOLID_COLORS if solid else _NATURAL_FEATURE_COLORS
    return palette[int.from_bytes(digest[:2], "big") % len(palette)]


def _enclosed_display_void(occupied: np.ndarray) -> tuple[np.ndarray, str]:
    air = ~occupied
    boundary = np.zeros_like(air)
    boundary[0, :, :] = air[0, :, :]
    boundary[-1, :, :] = air[-1, :, :]
    boundary[:, 0, :] = air[:, 0, :]
    boundary[:, -1, :] = air[:, -1, :]
    boundary[:, :, 0] = air[:, :, 0]
    boundary[:, :, -1] = air[:, :, -1]
    if _binary_propagation is None:
        # Fail visibly to the older proxy instead of silently claiming exact
        # connectivity when SciPy is absent.
        active = np.any(occupied, axis=2)
        tops = np.zeros(active.shape, dtype=np.int16)
        if np.any(active):
            tops[active] = (
                occupied.shape[2]
                - np.argmax(occupied[:, :, ::-1], axis=2)[active]
            )
        y = np.arange(occupied.shape[2], dtype=np.int16)[None, None, :]
        return (
            active[:, :, None] & (y < tops[:, :, None]) & air,
            "highest_solid_subsurface_proxy_scipy_unavailable",
        )
    outside = _binary_propagation(boundary, mask=air)
    return air & ~outside, "six_connected_enclosed_air"


class _MaterialRegistry:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = [{
            "id": 0,
            "name": "air",
            "color": "#000000",
            "opacity": 0.0,
            "layer": "air",
            "source_identity": {"kind": "canonical_air"},
        }]
        self._ids: dict[tuple[Any, ...], int] = {}
        self.solid: list[bool] = [False]
        self.fluid: list[bool] = [False]

    def add(
        self,
        key: tuple[Any, ...],
        *,
        name: str,
        layer: str,
        opacity: float,
        source_identity: dict[str, Any],
        solid: bool,
        fluid: bool = False,
    ) -> int:
        found = self._ids.get(key)
        if found is not None:
            return found
        identifier = len(self.entries)
        if identifier > MAX_VISUAL_MATERIALS:
            raise ValueError("captured volume exceeds the uint16 visual palette")
        self._ids[key] = identifier
        self.entries.append({
            "id": identifier,
            "name": name,
            "color": _stable_color(key, fluid=fluid, solid=solid),
            "opacity": opacity,
            "layer": layer,
            "source_identity": source_identity,
        })
        self.solid.append(solid)
        self.fluid.append(fluid)
        return identifier

    def block(self, semantic: Any, physical: Any, physical_code: int) -> int:
        flags = int(physical.flags[physical_code])
        solid = bool(flags & FLAG_SOLID)
        if semantic is not None and semantic.valid:
            semantic_hex = _hex_words(semantic.semantic_key)
            asset_hex = _hex_words(semantic.asset_key)
            key = (
                "stable_block",
                semantic_hex,
                asset_hex,
                int(semantic.affordance_tags),
                int(semantic.gather_type_index),
                int(semantic.required_tool_quality),
                int(semantic.rotation_index),
            )
            return self.add(
                key,
                name=f"captured_block_{asset_hex[:12]}",
                layer="captured_blocks",
                opacity=1.0,
                source_identity={
                    "kind": "stable_block_semantic",
                    "semantic_key": list(semantic.semantic_key),
                    "asset_key": list(semantic.asset_key),
                    "rotation_index": int(semantic.rotation_index),
                    "affordance_tags": int(semantic.affordance_tags),
                },
                solid=solid,
            )
        shape_index = int(physical.shape_index[physical_code])
        shape = physical_code
        key = (
            "physical_fallback",
            flags,
            shape_index,
            int(physical.fluid_level[physical_code]),
            round(float(physical.fluid_fill_height[physical_code]), 6),
            int(physical.block_damage[physical_code]),
            int(physical.fluid_damage[physical_code]),
        )
        return self.add(
            key,
            name=f"captured_physical_{shape:04x}",
            layer="captured_blocks",
            opacity=1.0,
            source_identity={
                "kind": "physical_traits_fallback_no_stable_block_id",
                "flags": flags,
                "shape_index": shape_index,
            },
            solid=solid,
        )

    def fluid_material(self, asset_id: str) -> int:
        key = ("stable_fluid", asset_id)
        return self.add(
            key,
            name=f"fluid_{asset_id}",
            layer="water",
            opacity=0.60,
            source_identity={
                "kind": "stable_hytale_fluid_getId",
                "asset_id": asset_id,
            },
            solid=False,
            fluid=True,
        )


def _tile_materials(
    physical_snapshot: Any,
    block_snapshot: Any,
    fluid_snapshot: Any,
    registry: _MaterialRegistry,
) -> tuple[np.ndarray, np.ndarray]:
    physical = _dense_core(physical_snapshot.cell_code)
    semantic = _dense_core(block_snapshot.cell_code)
    fluid = _dense_core(fluid_snapshot.cell_code)
    pair = (
        semantic.astype(np.uint32, copy=False) << np.uint32(16)
    ) | physical.astype(np.uint32, copy=False)
    unique, inverse = np.unique(pair, return_inverse=True)
    identifiers = np.zeros(unique.shape, dtype=np.uint16)
    for index, value in enumerate(unique.tolist()):
        semantic_code = int(value >> 16)
        physical_code = int(value & 0xFFFF)
        if semantic_code == 0 and physical_code == 0:
            continue
        entry = (
            block_snapshot.palette[semantic_code]
            if semantic_code != 0
            else None
        )
        identifiers[index] = registry.block(
            entry, physical_snapshot.cell_palette, physical_code
        )
    material = identifiers[inverse].reshape(pair.shape)
    if np.any(fluid):
        fluid_lookup = np.zeros(len(fluid_snapshot.palette), dtype=np.uint16)
        for code, asset_id in enumerate(fluid_snapshot.palette[1:], start=1):
            fluid_lookup[code] = registry.fluid_material(asset_id)
        material = material.copy()
        fluid_mask = fluid != 0
        material[fluid_mask] = fluid_lookup[fluid[fluid_mask]]
    flags = physical_snapshot.cell_palette.flags[physical]
    solid = (flags & np.uint16(FLAG_SOLID)) != 0
    return material, solid


def _enclosed_air(solid: np.ndarray, occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.any(solid, axis=1)
    top = np.zeros(active.shape, dtype=np.int16)
    if np.any(active):
        top[active] = (
            solid.shape[1]
            - np.argmax(solid[:, ::-1, :], axis=1)[active]
        )
    y = np.arange(solid.shape[1], dtype=np.int16)[None, :, None]
    enclosed = active[:, None, :] & (y < top[:, None, :]) & ~occupied
    return enclosed, top


def _representative(
    material: np.ndarray,
    enclosed: np.ndarray,
    registry: _MaterialRegistry,
    *,
    exact: bool,
) -> int:
    flat = material.reshape(-1)
    if exact:
        return int(flat[0])
    values = flat[flat != 0]
    if not values.size:
        return 0
    # Prefer a real void when at least one fifth of the source cube is a
    # below-surface cavity.  This keeps native tunnels visible at coarse LOD
    # without treating ordinary above-surface air as a cave.
    if np.count_nonzero(enclosed) * 5 >= enclosed.size:
        return 0
    identifiers, counts = np.unique(values, return_counts=True)
    features = np.asarray([
        registry.fluid[int(identifier)] or not registry.solid[int(identifier)]
        for identifier in identifiers
    ], dtype=np.bool_)
    if np.any(features):
        feature_ids = identifiers[features]
        feature_counts = counts[features]
        return int(feature_ids[np.argmax(feature_counts)])
    return int(identifiers[np.argmax(counts)])


def _compact_palette(
    cells: np.ndarray,
    registry: _MaterialRegistry,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    used = np.unique(cells)
    used = used[used != 0]
    lookup = np.zeros(len(registry.entries), dtype=np.uint16)
    palette = [dict(registry.entries[0])]
    for replacement, old in enumerate(used.tolist(), start=1):
        lookup[old] = replacement
        entry = dict(registry.entries[old])
        entry["id"] = replacement
        palette.append(entry)
    return lookup[cells], palette, len(registry.entries) - 1


def _columns(cells: np.ndarray) -> tuple[list[list[int]], dict[int, int], int]:
    nz, nx, ny = cells.shape
    encoded: list[list[int]] = []
    counts: dict[int, int] = {}
    run_count = 0
    for z in range(nz):
        for x in range(nx):
            column = cells[z, x]
            runs: list[int] = []
            y = 0
            while y < ny:
                material = int(column[y])
                if material == 0:
                    y += 1
                    continue
                start = y
                while y < ny and int(column[y]) == material:
                    y += 1
                length = y - start
                runs.extend((start, length, material))
                counts[material] = counts.get(material, 0) + length
                run_count += 1
            encoded.append(runs)
    return encoded, counts, run_count


def composite_voxel_volume(
    world: V2CompositeWorld,
    *,
    target_columns: int = DEFAULT_TARGET_COLUMNS,
) -> dict[str, Any]:
    """Stream one exact composite world into the shared voxel-view schema."""

    if not isinstance(world, V2CompositeWorld):
        raise TypeError("world must be a V2CompositeWorld")
    target = _target_columns(target_columns)
    requested_min = tuple(math.floor(value) for value in world.bounds_min)
    requested_max = tuple(math.ceil(value) for value in world.bounds_max)
    requested_min = (
        requested_min[0],
        max(0, requested_min[1]),
        requested_min[2],
    )
    requested_max = (
        requested_max[0],
        min(WORLD_HEIGHT, requested_max[1]),
        requested_max[2],
    )
    width = requested_max[0] - requested_min[0]
    height = requested_max[1] - requested_min[1]
    depth = requested_max[2] - requested_min[2]
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("composite world has empty voxel bounds")
    unit = _voxel_unit(width, depth, target)
    origin = (
        math.floor(requested_min[0] / unit) * unit,
        math.floor(requested_min[1] / unit) * unit,
        math.floor(requested_min[2] / unit) * unit,
    )
    maximum = (
        math.ceil(requested_max[0] / unit) * unit,
        math.ceil(requested_max[1] / unit) * unit,
        math.ceil(requested_max[2] / unit) * unit,
    )
    nx = (maximum[0] - origin[0]) // unit
    ny = (maximum[1] - origin[1]) // unit
    nz = (maximum[2] - origin[2]) // unit
    cells = np.zeros((nz, nx, ny), dtype=np.uint16)
    registry = _MaterialRegistry()
    source_nonair = 0
    source_solid = 0
    source_fluid = 0
    source_subsurface_void_proxy = 0
    source_counts: dict[int, int] = {}
    surface_heights: list[int] = []

    for tile in world.tiles:
        physical, block, fluid = load_composite_tile_artifacts(world, tile)
        material, solid = _tile_materials(physical, block, fluid, registry)
        occupied = material != 0
        enclosed, top = _enclosed_air(solid, occupied)
        core_x = int(physical.core_min_chunk_xz[0]) * CHUNK_SIZE
        core_z = int(physical.core_min_chunk_xz[1]) * CHUNK_SIZE
        tile_max_x = core_x + CORE_BLOCKS_PER_AXIS
        tile_max_z = core_z + CORE_BLOCKS_PER_AXIS
        x0 = max(requested_min[0], core_x)
        x1 = min(requested_max[0], tile_max_x)
        z0 = max(requested_min[2], core_z)
        z1 = min(requested_max[2], tile_max_z)
        y0 = requested_min[1]
        y1 = requested_max[1]
        if x0 >= x1 or z0 >= z1:
            continue
        local = (
            slice(x0 - core_x, x1 - core_x),
            slice(y0, y1),
            slice(z0 - core_z, z1 - core_z),
        )
        local_material = material[local]
        local_solid = solid[local]
        local_enclosed = enclosed[local]
        source_nonair += int(np.count_nonzero(local_material))
        source_solid += int(np.count_nonzero(local_solid))
        source_subsurface_void_proxy += int(np.count_nonzero(local_enclosed))
        fluid_ids = np.asarray(registry.fluid, dtype=np.bool_)
        source_fluid += int(np.count_nonzero(fluid_ids[local_material]))
        identifiers, counts = np.unique(
            local_material[local_material != 0], return_counts=True
        )
        for identifier, count in zip(identifiers.tolist(), counts.tolist(), strict=True):
            source_counts[identifier] = source_counts.get(identifier, 0) + count
        local_top = top[
            x0 - core_x:x1 - core_x,
            z0 - core_z:z1 - core_z,
        ]
        surface_heights.extend(
            int(value) for value in local_top.reshape(-1) if int(value) > 0
        )

        ox0 = (x0 - origin[0]) // unit
        ox1 = math.ceil((x1 - origin[0]) / unit)
        oy0 = (y0 - origin[1]) // unit
        oy1 = math.ceil((y1 - origin[1]) / unit)
        oz0 = (z0 - origin[2]) // unit
        oz1 = math.ceil((z1 - origin[2]) / unit)
        if unit == 1:
            cells[oz0:oz1, ox0:ox1, oy0:oy1] = np.transpose(
                local_material, (2, 0, 1)
            )
            continue
        for oz in range(oz0, oz1):
            cell_z0 = origin[2] + oz * unit
            world_z0 = max(z0, cell_z0)
            world_z1 = min(z1, cell_z0 + unit)
            lz = slice(world_z0 - core_z, world_z1 - core_z)
            for ox in range(ox0, ox1):
                cell_x0 = origin[0] + ox * unit
                world_x0 = max(x0, cell_x0)
                world_x1 = min(x1, cell_x0 + unit)
                lx = slice(world_x0 - core_x, world_x1 - core_x)
                for oy in range(oy0, oy1):
                    cell_y0 = origin[1] + oy * unit
                    world_y0 = max(y0, cell_y0)
                    world_y1 = min(y1, cell_y0 + unit)
                    ly = slice(world_y0, world_y1)
                    cells[oz, ox, oy] = _representative(
                        material[lx, ly, lz],
                        enclosed[lx, ly, lz],
                        registry,
                        exact=False,
                    )

    cells, palette, registered_materials = _compact_palette(cells, registry)
    columns, material_counts, run_count = _columns(cells)
    palette_names = {int(entry["id"]): str(entry["name"]) for entry in palette}
    surface_floor = max(
        origin[1],
        (int(np.percentile(surface_heights, 5)) - unit * 6)
        if surface_heights else origin[1],
    )
    occupied = cells != 0
    display_void, void_method = _enclosed_display_void(occupied)
    enclosed_display_voxels = int(np.count_nonzero(display_void))
    enclosed_source_cell_equivalent = enclosed_display_voxels * unit**3
    void_y = np.count_nonzero(display_void, axis=(0, 1))
    if np.any(void_y):
        cave_y = int(np.argmax(void_y))
        cut_axis = "y"
        cut_percent = max(5, min(90, round(100 * (1 - (cave_y + 1) / ny))))
    else:
        cut_axis, cut_percent = "z", 48
    fidelity = "exact_native_cell" if unit == 1 else "lossy_display_lod"
    material_voxels = {
        palette_names[identifier]: count
        for identifier, count in sorted(material_counts.items())
    }
    value: dict[str, Any] = {
        "schema": VOXEL_VOLUME_SCHEMA,
        "version": 1,
        "authority": (
            "exact_captured_region_cell_occupancy"
            if unit == 1
            else "exact_captured_source_with_explicit_visual_lod"
        ),
        "source_composite_semantic_sha256": world.semantic_sha256,
        "semantic_sha256": "",
        "axis_order": "x_y_z",
        "origin": [float(value) for value in origin],
        "bounds": {
            "minimum": [float(value) for value in origin],
            "maximum": [float(value) for value in maximum],
            "requested_minimum": [float(value) for value in world.bounds_min],
            "requested_maximum": [float(value) for value in world.bounds_max],
        },
        "voxel_size": unit,
        "source_voxel_size": 1,
        "dimensions": [nx, ny, nz],
        "column_encoding": "flat_y_start_length_material_runs",
        "palette": palette,
        "columns": columns,
        "visual_fidelity": {
            "cell_occupancy": fidelity,
            "stable_material_identity": "exact_for_retained_display_voxels",
            "sub_block_collision_shapes": "not_drawn_cubic_cell_view",
            "lod_reduction": (
                "none"
                if unit == 1
                else "power_of_two_void_preserving_native_cell_aggregation"
            ),
            "source_artifacts_remain_exact": True,
        },
        "recommended_view": {
            "surface_floor_y": int(surface_floor),
            "cutaway_floor_y": int(origin[1]),
            "cut_axis": cut_axis,
            "cut_percent": cut_percent,
        },
        "jax_world": {
            "world_count": 1,
            "world_id": COMPOSITE_JAX_WORLD_ID,
            "region_tiles": len(world.tiles),
            "shared_world_id": True,
            "composite_manifest_schema": COMPOSITE_WORLD_SCHEMA,
        },
        "metrics": {
            "occupied_voxels": int(np.count_nonzero(cells)),
            # Compatibility name consumed by the shared renderer. For native
            # captures this is conservative enclosed air, not a claim that
            # WorldGen authored every void as a cave feature.
            "carved_cave_voxels": enclosed_source_cell_equivalent,
            "cave_metric_kind": void_method,
            "enclosed_void_display_voxels": enclosed_display_voxels,
            "enclosed_void_source_cell_equivalent": (
                enclosed_source_cell_equivalent
            ),
            "subsurface_void_proxy_source_cells": (
                source_subsurface_void_proxy
            ),
            "source_nonair_cells": source_nonair,
            "source_solid_cells": source_solid,
            "source_fluid_cells": source_fluid,
            "source_unique_materials": len(source_counts),
            "registered_materials_before_display_lod": registered_materials,
            "display_materials": len(palette) - 1,
            "run_count": run_count,
            "material_voxels": material_voxels,
        },
    }
    stable = dict(value)
    stable["semantic_sha256"] = ""
    value["semantic_sha256"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def captured_preview(
    manifest_path: str | Path,
    artifact_root: str | Path,
    *,
    target_columns: int = DEFAULT_TARGET_COLUMNS,
) -> dict[str, Any]:
    """Load one published composite and return a renderer-ready receipt."""

    world = V2CompositeWorld.load(
        manifest_path,
        artifact_root,
        verify_artifacts=False,
    )
    volume = composite_voxel_volume(world, target_columns=target_columns)
    widths = (
        float(world.bounds_max[0] - world.bounds_min[0]),
        float(world.bounds_max[2] - world.bounds_min[2]),
    )
    palette = volume["palette"]
    native_origin = volume["origin"]
    spawn = {
        "x": float(world.spawn_position[0]),
        "y": float(world.spawn_position[1]),
        "z": float(world.spawn_position[2]),
        "safe": True,
    }
    surfaces = []
    for z, column in enumerate(volume["columns"]):
        del z
        top = 0
        for index in range(0, len(column), 3):
            top = max(top, column[index] + column[index + 1])
        if top:
            surfaces.append(native_origin[1] + top * volume["voxel_size"])
    mean_height = float(np.mean(surfaces)) if surfaces else native_origin[1]
    relief = float(max(surfaces) - min(surfaces)) if surfaces else 0.0
    return {
        "schema": CAPTURED_PREVIEW_SCHEMA,
        "version": 1,
        "authority": COMPOSITE_WORLD_AUTHORITY,
        "runtime_source": COMPOSITE_WORLD_RUNTIME_SOURCE,
        "digest": world.semantic_sha256,
        "quality": "exact_captured_world",
        "config": {
            "name": world.environment_id,
            "seed": world.seed,
            "world_width": widths[0],
            "world_depth": widths[1],
        },
        "metrics": {
            **volume["metrics"],
            "mean_height": mean_height,
            "relief": relief,
        },
        "spawn": spawn,
        "volume": volume,
        "capture": {
            "environment_id": world.environment_id,
            "structure": world.structure,
            "seed": world.seed,
            "split": world.split,
            "tile_grid": list(world.tile_grid),
            "region_tiles": len(world.tiles),
            "bounds_min": list(world.bounds_min),
            "bounds_max": list(world.bounds_max),
            "palette_entries": len(palette) - 1,
            "manifest_path": str(world.manifest_path),
            "artifact_root": str(world.artifact_root),
            "source_composite_semantic_sha256": world.semantic_sha256,
            "one_jax_world": True,
        },
    }
