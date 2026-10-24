"""Gen-7 feet-block producer (S12; ``sim/gen7/mechanics/feet_block.py``).

The cell rule is the cell the entity is in (floor of the feet point), chosen because the 0.7.2 movement audio reads the
inside block (AudioSystems.txt l.96-135). Each arena row is one dense grid with its own origin; the producer takes [B, E, 3]
positions and returns [B, E] ids. The dense lookup is tested on a stand-in carrying the GeometryState fields it reads
(origin, block_type_id). The Region lookup is not tested here: no Region atlas fixture is built in this increment.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.geometry.contract import CELL_COUNT, CELL_RADIUS
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.world.audio.hearing import load_footstep_content
from hytalegym.sim.gen7.mechanics import feet_block, heard_sounds

ASSETS = os.environ.get("HYTALE_ASSETS_ZIP", "")
BUSH = "Plant_Bush"
SOIL = "Soil_Grass"
SIDE = CELL_RADIUS * 2 + 1
CENTRE = (CELL_RADIUS, CELL_RADIUS, CELL_RADIUS)
BELOW = (CELL_RADIUS, CELL_RADIUS - 1, CELL_RADIUS)
BUSH_RECORD = next(r for r in load_footstep_content(ASSETS).sounds if r.name == "SFX_Bush_MoveIn")


@pytest.fixture(autouse=True)
def _assets(monkeypatch):
    monkeypatch.setenv("HYTALE_ASSETS_ZIP", ASSETS)


def _geometry(origin, cells: dict, batch: int = 1):
    """A stand-in with the GeometryState fields the lookup reads: origin [B, 3] and block_type_id [B, CELL_COUNT].

    ``cells`` keys are local (x, y, z) indices in the grid, which spans origin - CELL_RADIUS .. origin + CELL_RADIUS.
    """

    grid = np.zeros((batch, CELL_COUNT), dtype=np.int32)
    for (x, y, z), value in cells.items():
        grid[0, (x * SIDE + y) * SIDE + z] = value
    return SimpleNamespace(origin=jnp.asarray([origin] * batch, dtype=jnp.int32), block_type_id=jnp.asarray(grid))


# ---------------------------------------------------------------------------------------------------------------------
# The cell rule.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_feet_cell_is_the_floor_of_the_feet_point() -> None:
    cells = feet_block.feet_cells(jnp.asarray([[[10.5, 64.0, 3.25], [10.0, 64.4, 3.0], [-0.5, 64.9, 0.0]]]))
    assert cells.tolist() == [[[10, 64, 3], [10, 64, 3], [-1, 64, 0]]]


def test_a_feet_point_on_a_block_face_reads_the_cell_on_its_positive_side() -> None:
    # Tie rule: a face at x = 10.0 is the minimum corner of cell 10, so the entity reads cell 10 on that side.
    assert feet_block.feet_cells(jnp.asarray([[[10.0, 64.0, 3.0]]])).tolist() == [[[10, 64, 3]]]


# ---------------------------------------------------------------------------------------------------------------------
# The dense lookup: one row per arena, its own grid; the centre cell is the row's origin cell.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_dense_lookup_reads_the_feet_cell_not_the_support_cell_below() -> None:
    geometry = _geometry([4, 64, 4], {CENTRE: semantic_id(BUSH), BELOW: semantic_id(SOIL)})
    bush = feet_block.dense_block_under_feet(geometry, jnp.asarray([[[4.5, 64.3, 4.5]]]))
    assert bush.tolist() == [[semantic_id(BUSH)]]
    soil = feet_block.dense_block_under_feet(geometry, jnp.asarray([[[4.5, 63.5, 4.5]]]))
    assert soil.tolist() == [[semantic_id(SOIL)]]


def test_airborne_above_the_ground_reads_the_air_cell_and_outside_the_window_reads_zero() -> None:
    geometry = _geometry([4, 64, 4], {CENTRE: semantic_id(BUSH)})
    # Mid-jump at 65.7: the cell is y = 65, one above the centre, air (0) in this grid.
    assert feet_block.dense_block_under_feet(geometry, jnp.asarray([[[4.5, 65.7, 4.5]]])).tolist() == [[0]]
    # Far outside the 9x9x9 window.
    assert feet_block.dense_block_under_feet(geometry, jnp.asarray([[[100.5, 64.0, 4.5]]])).tolist() == [[0]]


def test_each_row_reads_only_its_own_grid() -> None:
    row0 = _geometry([4, 64, 4], {CENTRE: semantic_id(BUSH)})
    row1 = _geometry([4, 64, 4], {CENTRE: semantic_id(SOIL)})
    geometry = SimpleNamespace(
        origin=jnp.concatenate([row0.origin, row1.origin]),
        block_type_id=jnp.concatenate([row0.block_type_id, row1.block_type_id]),
    )
    ids = feet_block.dense_block_under_feet(geometry, jnp.asarray([[[4.5, 64.3, 4.5]], [[4.5, 64.3, 4.5]]]))
    assert ids.tolist() == [[semantic_id(BUSH)], [semantic_id(SOIL)]]


def test_no_identity_grid_reads_zero() -> None:
    geometry = SimpleNamespace(origin=jnp.zeros((1, 3), dtype=jnp.int32), block_type_id=None)
    assert feet_block.dense_block_under_feet(geometry, jnp.asarray([[[0.5, 0.5, 0.5]]])).tolist() == [[0]]


# ---------------------------------------------------------------------------------------------------------------------
# The heard list on the producer's output: one row is one arena; entity 0 listens to entity 1.
# ---------------------------------------------------------------------------------------------------------------------


class _Caps(NamedTuple):
    target_candidate_perceptible: Any
    target_candidate_perception_valid: Any
    block_under_feet: Any


def _heard_from_points(rows, blocks_per_row):
    """``rows``: one [entity 0, entity 1] feet pair per arena row, both moving along +x. Each entity's feet block comes from
    the dense producer, on a grid whose centre cell (the entity's own feet cell) holds the given block id."""

    batch = len(rows)
    position = np.zeros((batch, 2, 3), dtype=np.float32)
    velocity = np.zeros((batch, 2, 3), dtype=np.float32)
    feet = np.zeros((batch, 2), dtype=np.int32)
    for row, (pair, blocks) in enumerate(zip(rows, blocks_per_row)):
        for entity in range(2):
            position[row, entity] = pair[entity]
            velocity[row, entity] = (1.0, 0.0, 0.0)
            cell = [int(c) for c in np.floor(pair[entity])]
            geometry = _geometry(cell, {CENTRE: blocks[entity]})
            feet[row, entity] = int(feet_block.dense_block_under_feet(
                geometry, jnp.asarray([[pair[entity]]], dtype=jnp.float32)
            )[0, 0])
    state = SimpleNamespace(
        combat=SimpleNamespace(
            position=jnp.asarray(position),
            velocity=jnp.asarray(velocity),
            tick_count=jnp.zeros((batch,), dtype=jnp.int32),
        )
    )
    flag = jnp.ones((batch, 2, 2), dtype=jnp.bool_)
    caps = _Caps(flag, flag, jnp.asarray(feet, dtype=jnp.int32))
    return heard_sounds.heard_sounds_fields(state, caps, heard_sounds.footstep_tables_from_env())


def test_entity_one_standing_in_a_bush_cell_is_heard_as_a_bush_footstep_by_entity_zero() -> None:
    # Entity 1 is inside a bush cell (feet y = 64.3), 5 blocks from entity 0 on soil (feet y = 63.5, silent).
    rows = [((4.5, 63.5, 4.5), (9.5, 64.3, 4.5))]
    heard = _heard_from_points(rows, [[semantic_id(SOIL), semantic_id(BUSH)]])
    slots = [k for k in range(16) if bool(heard.valid[0, k]) and float(heard.distance[0, k]) > 0.0]
    assert len(slots) == 1
    slot = slots[0]
    assert int(heard.sound_id[0, slot]) == semantic_id("SFX_Bush_MoveIn")
    distance = float(heard.distance[0, slot])
    assert distance == pytest.approx(np.hypot(5.0, 0.8), rel=1e-5)
    expected = BUSH_RECORD.volume_linear * (BUSH_RECORD.max_distance - distance) / (
        BUSH_RECORD.max_distance - BUSH_RECORD.start_distance
    )
    assert float(heard.gain[0, slot]) == pytest.approx(expected, rel=1e-5)


def test_a_soil_floor_under_both_entities_stays_silent() -> None:
    rows = [((4.5, 63.5, 4.5), (9.5, 63.5, 4.5))]
    heard = _heard_from_points(rows, [[semantic_id(SOIL), semantic_id(SOIL)]])
    assert not bool(heard.valid.any())
