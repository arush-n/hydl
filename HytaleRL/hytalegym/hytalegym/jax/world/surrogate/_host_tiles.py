"""Leaf helpers extracted verbatim from traversal.py."""

import math
import numpy as np
from hytalegym.geometry.contract import FLAG_SOLID
from hytalegym.jax.world.surrogate.atlas import GENERATION_READY
from hytalegym.jax.world.surrogate.types import SurrogateAtlas
from hytalegym.jax.world.traversal import GRAPH_DIAGNOSTIC_OFF_GRID_SURFACE
from hytalegym.worldgen.region import (
    CAPTURE_BLOCKS_PER_AXIS,
    CHUNK_SIZE,
    CORE_BLOCKS_PER_AXIS,
    MIN_Y,
    WORLD_HEIGHT,
)
from hytalegym.worldgen.surrogate import TRAVERSAL_STATE_MASK_BITS
from ._geometry import (  # noqa: F401  (re-exported: callers unchanged)
    _EPSILON,
    _SUPPORT_EPSILON,
    _block_key,
    _decode_key,
    _face_intersects_core,
    _gate_allows,
    _host_index,
    _intervals_cover_unit,
    _leaf,
    _movement_bounds,
    _segment_hits_box,
    _segment_interval,
    _strict_overlap,
    _strict_overlap_2d,
    _support_interval,
    _swept_aabb_hits_box,
)


GRAPH_STATE_MASK_BITS = TRAVERSAL_STATE_MASK_BITS


class _HostTileGeometry:
    def __init__(self, atlas: SurrogateAtlas, tile: int):
        self.tile = tile
        self.tile_mask = _leaf(atlas.tile_mask, np.bool_)[tile]
        self.status = _leaf(atlas.generation_status, np.uint8)[tile]
        if not self.tile_mask or self.status != GENERATION_READY:
            raise ValueError("traversal graph requires a ready atlas tile")
        self.core_min = _leaf(atlas.core_min_chunk_xz, np.int32)[tile]
        self.origin = (self.core_min.astype(np.int64) - 1) * CHUNK_SIZE
        self.known = _leaf(atlas.column_known, np.bool_)[tile]
        if not np.all(self.known):
            raise ValueError("traversal graph rejects unknown columns")
        self.height = _leaf(atlas.terrain_height, np.int16)[tile]
        self.surface = _leaf(atlas.surface_cell_code, np.uint16)[tile]
        self.subsurface = _leaf(
            atlas.subsurface_cell_code,
            np.uint16,
        )[tile]
        self.cave_mask = _leaf(atlas.cave_span_mask, np.bool_)[tile]
        self.cave_minimum = _leaf(atlas.cave_min_y, np.int16)[tile]
        self.cave_maximum = _leaf(atlas.cave_max_y, np.int16)[tile]
        structure_mask = _leaf(atlas.structure_mask, np.bool_)[tile]
        structure_key = _leaf(atlas.structure_block_key, np.int32)[tile]
        structure_code = _leaf(
            atlas.structure_cell_code,
            np.uint16,
        )[tile]
        self.structure = {
            int(structure_key[index]): int(structure_code[index])
            for index in np.flatnonzero(structure_mask)
        }
        stateful_mask = _leaf(atlas.stateful_mask, np.bool_)[tile]
        stateful_key = _leaf(atlas.stateful_block_key, np.int32)[tile]
        stateful_runtime = _leaf(
            atlas.stateful_runtime_slot,
            np.int32,
        )[tile]
        stateful_count = _leaf(
            atlas.stateful_state_count,
            np.uint8,
        )[tile]
        stateful_initial = _leaf(
            atlas.stateful_initial_state,
            np.uint8,
        )[tile]
        stateful_variant = _leaf(
            atlas.stateful_variant_cell_code,
            np.uint16,
        )[tile]
        self.stateful_by_key: dict[int, tuple[int, int, int, int]] = {}
        self.state_count_by_runtime: dict[int, int] = {}
        for definition in np.flatnonzero(stateful_mask):
            key = int(stateful_key[definition])
            runtime_slot = int(stateful_runtime[definition])
            state_count = int(stateful_count[definition])
            initial_state = int(stateful_initial[definition])
            if (
                not 0 <= runtime_slot <= np.iinfo(np.int16).max
                or not 1 <= state_count <= GRAPH_STATE_MASK_BITS
                or not 0 <= initial_state < state_count
            ):
                raise ValueError("stateful graph gate exceeds its fixed schema")
            if key in self.stateful_by_key:
                raise ValueError("stateful graph definitions contain duplicate keys")
            previous_count = self.state_count_by_runtime.setdefault(
                runtime_slot,
                state_count,
            )
            if previous_count != state_count:
                raise ValueError("stateful graph identity state counts disagree")
            self.stateful_by_key[key] = (
                int(definition),
                runtime_slot,
                state_count,
                initial_state,
            )
        self.stateful_variant = stateful_variant
        self.cell_count = int(_leaf(atlas.cell_palette_size, np.int32))
        self.flags = _leaf(atlas.cell_flags, np.uint16)
        self.shape_index = _leaf(atlas.cell_shape_index, np.uint16)
        self.shape_count = int(_leaf(atlas.shape_palette_size, np.int32))
        self.boxes = _leaf(atlas.collision_boxes, np.float32)
        self.box_mask = _leaf(atlas.collision_box_mask, np.bool_)
        if np.any(self.shape_index[: self.cell_count] >= self.shape_count):
            raise ValueError("atlas cell palette references an invalid shape")
        active_boxes = self.boxes[: self.shape_count][self.box_mask[: self.shape_count]]
        if active_boxes.size:
            minimum = np.min(active_boxes[:, :3], axis=0)
            maximum = np.max(active_boxes[:, 3:], axis=0)
            self.lower_extension = np.ceil(np.maximum(0.0, -minimum)).astype(np.int32)
            self.upper_extension = np.ceil(np.maximum(0.0, maximum - 1.0)).astype(
                np.int32
            )
        else:
            self.lower_extension = np.zeros(3, dtype=np.int32)
            self.upper_extension = np.zeros(3, dtype=np.int32)

    def candidate_surfaces(
        self,
    ) -> tuple[dict[tuple[int, int], set[np.float32]], int]:
        core_min = CHUNK_SIZE
        core_max = CHUNK_SIZE + CORE_BLOCKS_PER_AXIS
        candidates = {
            (int(self.origin[0]) + x, int(self.origin[1]) + z): {
                np.float32(int(self.height[x, z]) + 1)
            }
            for x in range(core_min, core_max)
            for z in range(core_min, core_max)
        }
        for x in range(core_min, core_max):
            for z in range(core_min, core_max):
                for slot in np.flatnonzero(self.cave_mask[x, z]):
                    candidates[(int(self.origin[0]) + x, int(self.origin[1]) + z)].add(
                        np.float32(self.cave_minimum[x, z, slot])
                    )
        diagnostics = 0
        relevant = set(self.structure)
        for key in self.structure:
            x, y, z = _decode_key(key)
            if y > 0:
                relevant.add(_block_key(x, y - 1, z))
        for key in sorted(relevant):
            x, y, z = _decode_key(key)
            if key in self.stateful_by_key:
                continue
            code = self.cell_code(x, y, z)
            for box, _source_key in self.cell_boxes(x, y, z, code):
                minimum_x = float(box[0])
                maximum_x = float(box[3])
                minimum_z = float(box[2])
                maximum_z = float(box[5])
                foot = np.float32(box[4])
                inserted = False
                for world_x in range(
                    math.ceil(minimum_x - 0.5),
                    math.floor(maximum_x - 0.5) + 1,
                ):
                    for world_z in range(
                        math.ceil(minimum_z - 0.5),
                        math.floor(maximum_z - 0.5) + 1,
                    ):
                        target = candidates.get((world_x, world_z))
                        if target is not None:
                            target.add(foot)
                            inserted = True
                if not inserted and _face_intersects_core(
                    minimum_x,
                    maximum_x,
                    minimum_z,
                    maximum_z,
                    self.origin,
                ):
                    diagnostics |= GRAPH_DIAGNOSTIC_OFF_GRID_SURFACE
        return candidates, diagnostics

    def standable(
        self,
        x: float,
        z: float,
        foot: float,
        bounds: np.ndarray,
        states: dict[int, int] | None = None,
    ) -> tuple[np.float32, int] | None:
        position = np.asarray(
            (x, foot - float(bounds[1]), z),
            dtype=np.float64,
        )
        minimum = position + bounds[:3]
        maximum = position + bounds[3:]
        query_minimum = minimum.copy()
        query_minimum[1] -= _SUPPORT_EPSILON
        support_keys: list[int] = []
        clearance = float(MIN_Y + WORLD_HEIGHT) - foot
        for box, source_key in self.boxes_for_bounds(
            query_minimum,
            maximum,
            states,
        ):
            if _strict_overlap(minimum, maximum, box[:3], box[3:]):
                return None
            horizontal = _strict_overlap_2d(
                minimum,
                maximum,
                box[:3],
                box[3:],
            )
            if horizontal and abs(float(box[4]) - foot) <= _SUPPORT_EPSILON:
                if source_key not in self.stateful_by_key:
                    support_keys.append(source_key)
            if horizontal and float(box[1]) >= foot - _EPSILON:
                clearance = min(clearance, float(box[1]) - foot)
        if not support_keys:
            return None
        return np.float32(max(0.0, clearance)), min(support_keys)

    def gated_standable(
        self,
        x: float,
        z: float,
        foot: float,
        bounds: np.ndarray,
    ) -> tuple[tuple[np.float32, int, int, int] | None, bool]:
        position = np.asarray(
            (x, foot - float(bounds[1]), z),
            dtype=np.float64,
        )
        minimum = position + bounds[:3]
        maximum = position + bounds[3:]
        minimum[1] -= _SUPPORT_EPSILON
        slots = self.stateful_slots_for_bounds(minimum, maximum)
        if len(slots) > 1:
            return None, True
        if not slots:
            value = self.standable(x, z, foot, bounds)
            if value is None:
                return None, False
            return (value[0], value[1], -1, 0), False

        slot = next(iter(slots))
        state_count = self.state_count_by_runtime[slot]
        allowed = 0
        clearances: list[np.float32] = []
        support_keys: list[int] = []
        for state in range(state_count):
            value = self.standable(
                x,
                z,
                foot,
                bounds,
                {slot: state},
            )
            if value is None:
                continue
            allowed |= 1 << state
            clearances.append(value[0])
            support_keys.append(value[1])
        if not allowed:
            return None, False
        full_mask = (1 << state_count) - 1
        gate_slot = -1 if allowed == full_mask else slot
        gate_mask = 0 if gate_slot < 0 else allowed
        return (
            min(clearances),
            min(support_keys),
            gate_slot,
            gate_mask,
        ), False

    def gated_movement(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        bounds: np.ndarray,
        flat_walk: bool,
        source_gate: tuple[int, int],
        target_gate: tuple[int, int],
    ) -> tuple[tuple[int, int] | None, bool]:
        minimum, maximum = _movement_bounds(start, end, bounds)
        slots = self.stateful_slots_for_bounds(minimum, maximum)
        slots.update(slot for slot, _mask in (source_gate, target_gate) if slot >= 0)
        if len(slots) > 1:
            return None, True
        clear = self.flat_walk_clear if flat_walk else self.transition_clear
        if not slots:
            if clear(start, end, bounds):
                return (-1, 0), False
            return None, False

        slot = next(iter(slots))
        state_count = self.state_count_by_runtime.get(slot)
        if state_count is None:
            raise ValueError("graph gate references an unknown runtime slot")
        allowed = 0
        for state in range(state_count):
            if not _gate_allows(source_gate, slot, state) or not _gate_allows(
                target_gate,
                slot,
                state,
            ):
                continue
            if clear(
                start,
                end,
                bounds,
                {slot: state},
            ):
                allowed |= 1 << state
        if not allowed:
            return None, False
        full_mask = (1 << state_count) - 1
        return ((-1, 0) if allowed == full_mask else (slot, allowed)), False

    def transition_clear(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        bounds: np.ndarray,
        states: dict[int, int] | None = None,
    ) -> bool:
        start_position = np.asarray(
            (start[0], start[1] - float(bounds[1]), start[2]),
            dtype=np.float64,
        )
        end_position = np.asarray(
            (end[0], end[1] - float(bounds[1]), end[2]),
            dtype=np.float64,
        )
        if end[1] > start[1]:
            corner = np.asarray(
                (start_position[0], end_position[1], start_position[2]),
            )
        else:
            corner = np.asarray(
                (end_position[0], start_position[1], end_position[2]),
            )
        minimum, maximum = _movement_bounds(start, end, bounds)
        for box, _source_key in self.boxes_for_bounds(
            minimum,
            maximum,
            states,
        ):
            if _swept_aabb_hits_box(start_position, corner, bounds, box):
                return False
            if _swept_aabb_hits_box(corner, end_position, bounds, box):
                return False
        return True

    def flat_walk_clear(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        bounds: np.ndarray,
        states: dict[int, int] | None = None,
    ) -> bool:
        start_position = np.asarray(
            (start[0], start[1] - float(bounds[1]), start[2]),
            dtype=np.float64,
        )
        end_position = np.asarray(
            (end[0], end[1] - float(bounds[1]), end[2]),
            dtype=np.float64,
        )
        minimum = np.minimum(
            start_position + bounds[:3],
            end_position + bounds[:3],
        )
        maximum = np.maximum(
            start_position + bounds[3:],
            end_position + bounds[3:],
        )
        minimum[1] -= _SUPPORT_EPSILON
        boxes = self.boxes_for_bounds(minimum, maximum, states)
        intervals: list[tuple[float, float]] = []
        for box, _source_key in boxes:
            if _segment_hits_box(start_position, end_position, bounds, box):
                return False
            if abs(float(box[4]) - start[1]) <= _SUPPORT_EPSILON:
                interval = _support_interval(
                    start_position,
                    end_position,
                    bounds,
                    box,
                )
                if interval is not None:
                    intervals.append(interval)
        return _intervals_cover_unit(intervals)

    def boxes_for_bounds(
        self,
        minimum: np.ndarray,
        maximum: np.ndarray,
        states: dict[int, int] | None = None,
    ) -> list[tuple[np.ndarray, int]]:
        local_minimum, local_maximum = self._local_cell_bounds(
            minimum,
            maximum,
        )
        result: list[tuple[np.ndarray, int]] = []
        for x in range(int(local_minimum[0]), int(local_maximum[0]) + 1):
            for y in range(int(local_minimum[1]), int(local_maximum[1]) + 1):
                for z in range(
                    int(local_minimum[2]),
                    int(local_maximum[2]) + 1,
                ):
                    if (
                        not 0 <= x < CAPTURE_BLOCKS_PER_AXIS
                        or not 0 <= y < WORLD_HEIGHT
                        or not 0 <= z < CAPTURE_BLOCKS_PER_AXIS
                    ):
                        raise ValueError("actor geometry exceeds the tile capture halo")
                    result.extend(
                        self.cell_boxes(
                            x,
                            y,
                            z,
                            self.cell_code(x, y, z, states),
                        )
                    )
        return result

    def stateful_slots_for_bounds(
        self,
        minimum: np.ndarray,
        maximum: np.ndarray,
    ) -> set[int]:
        local_minimum, local_maximum = self._local_cell_bounds(
            minimum,
            maximum,
        )
        slots: set[int] = set()
        for x in range(int(local_minimum[0]), int(local_maximum[0]) + 1):
            for y in range(int(local_minimum[1]), int(local_maximum[1]) + 1):
                for z in range(
                    int(local_minimum[2]),
                    int(local_maximum[2]) + 1,
                ):
                    if (
                        not 0 <= x < CAPTURE_BLOCKS_PER_AXIS
                        or not 0 <= y < WORLD_HEIGHT
                        or not 0 <= z < CAPTURE_BLOCKS_PER_AXIS
                    ):
                        raise ValueError("actor geometry exceeds the tile capture halo")
                    definition = self.stateful_by_key.get(_block_key(x, y, z))
                    if definition is not None:
                        slots.add(definition[1])
        return slots

    def _local_cell_bounds(
        self,
        minimum: np.ndarray,
        maximum: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        origin = np.asarray((self.origin[0], MIN_Y, self.origin[1]))
        local_minimum = (
            np.floor(minimum - origin).astype(np.int32) - self.lower_extension
        )
        local_maximum = (
            np.floor(maximum - _EPSILON - origin).astype(np.int32)
            + self.upper_extension
        )
        return local_minimum, local_maximum

    def cell_code(
        self,
        x: int,
        y: int,
        z: int,
        states: dict[int, int] | None = None,
    ) -> int:
        key = _block_key(x, y, z)
        definition = self.stateful_by_key.get(key)
        if definition is not None:
            local_slot, runtime_slot, state_count, initial_state = definition
            state = (
                initial_state
                if states is None
                else states.get(runtime_slot, initial_state)
            )
            if not 0 <= state < state_count:
                raise ValueError("stateful graph override exceeds state count")
            return int(self.stateful_variant[local_slot, state])
        override = self.structure.get(key)
        if override is not None:
            return override
        height = int(self.height[x, z])
        if np.any(
            self.cave_mask[x, z]
            & (MIN_Y + y >= self.cave_minimum[x, z])
            & (MIN_Y + y < self.cave_maximum[x, z])
        ):
            return 0
        if y == height:
            return int(self.surface[x, z])
        if y < height:
            return int(self.subsurface[x, z])
        return 0

    def cell_boxes(
        self,
        x: int,
        y: int,
        z: int,
        code: int,
    ) -> list[tuple[np.ndarray, int]]:
        if not 0 <= code < self.cell_count:
            raise ValueError("effective cell code exceeds palette")
        if not int(self.flags[code]) & FLAG_SOLID:
            return []
        shape = int(self.shape_index[code])
        base = np.asarray(
            (
                int(self.origin[0]) + x,
                MIN_Y + y,
                int(self.origin[1]) + z,
                int(self.origin[0]) + x,
                MIN_Y + y,
                int(self.origin[1]) + z,
            ),
            dtype=np.float64,
        )
        key = _block_key(x, y, z)
        return [
            (box.astype(np.float64) + base, key)
            for box in self.boxes[shape, self.box_mask[shape]]
        ]
