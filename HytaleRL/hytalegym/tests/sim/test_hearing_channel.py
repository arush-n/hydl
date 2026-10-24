"""Gen-7 heard sounds (S12; construct ``sim/gen7/mechanics/heard_sounds.py``, switch ``Constructs.hearing``, default False).

One batch row is one arena; its entities are the slots of ``state.combat`` and entity 0 is the listener. The heard list is a
pure function of the entities' positions and velocities, the tick count, the block under each entity's feet and entity 0's
perception flags. The arithmetic is tested on arrays with the real footstep content (Assets.zip). The render is tested with
the switch off and on; the two-entity render case has entity 0 hear entity 1's MoveIn with the curve's gain. The block
under the feet is supplied as ``block_under_feet`` [B, E] on the capabilities, the field no builder publishes yet, so the
render test attaches it by hand.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.world.audio.hearing import load_footstep_content
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7 import overlay as gen7_overlay
from hytalegym.sim.gen7.mechanics import heard_sounds
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import ARCHIVES, mount_fresh, unmount

ASSETS = os.environ.get("HYTALE_ASSETS_ZIP", str(ARCHIVES["gen7"]))
PLANT = "Plant_Bush"
SOIL = "Soil_Grass"
MOVE_IN_RECORD = next(
    record for record in load_footstep_content(ASSETS).sounds if record.name == "SFX_Bush_MoveIn"
)
ENTITY_COUNT = 2


@pytest.fixture(autouse=True)
def _assets(monkeypatch):
    monkeypatch.setenv("HYTALE_ASSETS_ZIP", ASSETS)


def _expected_gain(distance: float) -> float:
    """The curve of hear(): full inside StartAttenuationDistance, linear to zero at MaxDistance, times the linear volume."""

    start, maximum = MOVE_IN_RECORD.start_distance, MOVE_IN_RECORD.max_distance
    if distance <= start:
        curve = 1.0
    else:
        curve = max(0.0, min(1.0, (maximum - distance) / (maximum - start)))
    return MOVE_IN_RECORD.volume_linear * curve


class _Caps(NamedTuple):
    target_candidate_perceptible: Any
    target_candidate_perception_valid: Any


def _caps(batch: int, *, perceptible: bool = True, block_under_feet=None):
    flag = jnp.full((batch, ENTITY_COUNT, ENTITY_COUNT), perceptible, dtype=jnp.bool_)
    caps = _Caps(flag, jnp.ones_like(flag))
    if block_under_feet is None:
        return caps
    fields = list(caps._fields) + ["block_under_feet"]
    cls = NamedTuple("CapsWithBlock", [(name, Any) for name in fields])
    return cls(*caps, jnp.asarray(block_under_feet, dtype=jnp.int32))


def _state(positions, velocities, tick: int):
    batch = len(positions)
    position = np.zeros((batch, ENTITY_COUNT, 3), dtype=np.float32)
    velocity = np.zeros((batch, ENTITY_COUNT, 3), dtype=np.float32)
    for row, (p, v) in enumerate(zip(positions, velocities)):
        position[row, 0] = p
        velocity[row, 0] = v
    combat = SimpleNamespace(
        position=jnp.asarray(position),
        velocity=jnp.asarray(velocity),
        tick_count=jnp.full((batch,), tick, dtype=jnp.int32),
    )
    return SimpleNamespace(combat=combat)


def _expected_gain(distance: float) -> float:
    """The curve of hear(): full inside StartAttenuationDistance, linear to zero at MaxDistance, times the linear volume."""

    start, maximum = MOVE_IN_RECORD.start_distance, MOVE_IN_RECORD.max_distance
    if distance <= start:
        curve = 1.0
    else:
        curve = max(0.0, min(1.0, (maximum - distance) / (maximum - start)))
    return MOVE_IN_RECORD.volume_linear * curve


class _Caps(NamedTuple):
    target_candidate_perceptible: Any
    target_candidate_perception_valid: Any


def _caps(batch: int, *, perceptible: bool = True, block_under_feet=None):
    flag = jnp.full((batch, ENTITY_COUNT, ENTITY_COUNT), perceptible, dtype=jnp.bool_)
    caps = _Caps(flag, jnp.ones_like(flag))
    if block_under_feet is None:
        return caps
    fields = list(caps._fields) + ["block_under_feet"]
    cls = NamedTuple("CapsWithBlock", [(name, Any) for name in fields])
    return cls(*caps, jnp.asarray(block_under_feet, dtype=jnp.int32))


def _state(rows, velocities, tick: int):
    """``rows``: one [entity 0, entity 1] position pair per arena row; ``velocities`` the matching pairs."""

    batch = len(rows)
    position = np.zeros((batch, ENTITY_COUNT, 3), dtype=np.float32)
    velocity = np.zeros((batch, ENTITY_COUNT, 3), dtype=np.float32)
    for row, (pair, vel) in enumerate(zip(rows, velocities)):
        for entity in range(ENTITY_COUNT):
            position[row, entity] = pair[entity]
            velocity[row, entity] = vel[entity]
    combat = SimpleNamespace(
        position=jnp.asarray(position),
        velocity=jnp.asarray(velocity),
        tick_count=jnp.full((batch,), tick, dtype=jnp.int32),
    )
    return SimpleNamespace(combat=combat)


def _other_slot(heard, row: int) -> int:
    """The slot of entity 1's footstep in ``row`` (the only valid slot with a positive distance)."""

    distances = np.asarray(heard.distance[row])
    slots = [k for k in range(distances.shape[0]) if bool(heard.valid[row, k]) and distances[k] > 0.0]
    assert len(slots) == 1, slots
    return slots[0]


def _heard(rows, velocities=None, *, tick=0, block=PLANT, perceptible=True, per_entity=None):
    """The heard list for ``rows`` (one entity pair per row). Both entities stand on ``block`` unless ``per_entity``
    gives the [B, E] semantic ids; ``block=None`` attaches no field (the Soil_Grass default)."""

    batch = len(rows)
    if velocities is None:
        velocities = [[MOVING, MOVING]] * batch
    if per_entity is not None:
        ids = per_entity
    elif block is None:
        ids = None
    else:
        ids = [[semantic_id(block)] * ENTITY_COUNT] * batch
    caps = _caps(batch, perceptible=perceptible, block_under_feet=ids)
    return heard_sounds.heard_sounds_fields(_state(rows, velocities, tick), caps, heard_sounds.footstep_tables_from_env())


MOVING = (1.0, 0.0, 0.0)
STILL = (0.0, 0.0, 0.0)
PAIR_5 = [((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))]


# ---------------------------------------------------------------------------------------------------------------------
# The switch and the construct.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_switch_is_false_in_the_defaults_and_in_both_profiles() -> None:
    assert Constructs().hearing is False
    assert GEN5.constructs.hearing is False
    assert GEN7.constructs.hearing is False


def test_the_construct_returns_the_original_when_the_switch_is_off() -> None:
    def original(*args, **kwargs):
        return args, kwargs

    assert heard_sounds.construct(original, GEN7) is original


def test_a_non_bool_switch_fails_closed() -> None:
    with pytest.raises(ValueError, match="hearing"):
        heard_sounds.construct(lambda *a, **k: None, SimpleNamespace(constructs=SimpleNamespace(hearing="yes")))


# ---------------------------------------------------------------------------------------------------------------------
# The heard list (pure arithmetic on the real content). One row is one arena; entity 0 listens.
# ---------------------------------------------------------------------------------------------------------------------


def test_two_entities_five_blocks_apart_on_a_bush_hear_each_other_at_the_curve_gain() -> None:
    heard = _heard(PAIR_5)
    assert heard.valid.tolist() == [[True, True] + [False] * 14]
    slot = _other_slot(heard, 0)
    assert float(heard.gain[0, slot]) == pytest.approx(_expected_gain(5.0), rel=1e-5)
    assert float(heard.gain[0, slot]) == pytest.approx(MOVE_IN_RECORD.volume_linear * 10.0 / 11.0, rel=1e-5)
    assert int(heard.sound_id[0, slot]) == semantic_id("SFX_Bush_MoveIn")
    assert float(heard.distance[0, slot]) == pytest.approx(5.0)
    assert heard.relative_position[0, slot].tolist() == pytest.approx([5.0, 0.0, 0.0])
    assert int(heard.source_entity_id[0, slot]) == 1


def test_entity_zero_hears_its_own_footstep_at_distance_zero_at_full_volume() -> None:
    heard = _heard(PAIR_5)
    own = [k for k in range(16) if bool(heard.valid[0, k]) and float(heard.distance[0, k]) == 0.0]
    assert own == [0]
    assert float(heard.gain[0, 0]) == pytest.approx(MOVE_IN_RECORD.volume_linear, rel=1e-6)
    assert int(heard.source_entity_id[0, 0]) == 0


def test_rows_do_not_hear_each_other_across_arenas() -> None:
    # Row 0's entity 1 is 5 blocks away; row 1's entities sit 1 block from row 0's entity 0 and are not row 0's sources.
    rows = [((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)), ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
    heard = _heard(rows)
    row0 = sorted(round(float(heard.distance[0, k]), 6) for k in range(16) if bool(heard.valid[0, k]))
    assert row0 == [0.0, 5.0]
    row1 = sorted(round(float(heard.distance[1, k]), 6) for k in range(16) if bool(heard.valid[1, k]))
    assert row1 == [0.0, 0.0]
    assert not any(round(float(heard.distance[k // 16, k % 16]), 6) == 1.0 for k in range(32) if bool(heard.valid[k // 16, k % 16]))


def test_each_entity_footstep_uses_its_own_block() -> None:
    # Entity 0 on a bush cell, entity 1 on soil: only entity 0's (own) bush footstep is audible to entity 0.
    per_entity = [[semantic_id(PLANT), semantic_id(SOIL)]]
    heard = _heard(PAIR_5, per_entity=per_entity)
    assert heard.valid.tolist() == [[True] + [False] * 15]
    assert int(heard.sound_id[0, 0]) == semantic_id("SFX_Bush_MoveIn")


def test_the_floor_is_silent_on_soil_grass() -> None:
    heard = _heard(PAIR_5, per_entity=[[semantic_id(SOIL), semantic_id(SOIL)]])
    assert not bool(heard.valid.any())
    default = _heard(PAIR_5, block=None)
    assert not bool(default.valid.any())


def test_a_source_at_or_beyond_max_distance_is_not_heard() -> None:
    maximum = MOVE_IN_RECORD.max_distance
    for distance in (maximum, maximum + 5.0):
        heard = _heard([((0.0, 0.0, 0.0), (distance, 0.0, 0.0))])
        assert not bool(heard.valid[0, 1])
        assert float(heard.gain[0, 1]) == 0.0


def test_emission_needs_the_repeat_tick_and_a_moving_entity() -> None:
    off_beat = _heard(PAIR_5, tick=1)
    assert not bool(off_beat.valid.any())
    still = _heard(PAIR_5, [[STILL, STILL]])
    assert not bool(still.valid.any())
    on_beat = _heard(PAIR_5, tick=30)
    assert bool(on_beat.valid[0, _other_slot(on_beat, 0)])


def test_the_source_entity_id_follows_entity_zeros_own_perception_flag() -> None:
    seen = _heard(PAIR_5, perceptible=True)
    assert int(seen.source_entity_id[0, 0]) == 0  # its own footstep
    assert int(seen.source_entity_id[0, 1]) == 1  # entity 1, perceived by entity 0
    unseen = _heard(PAIR_5, perceptible=False)
    assert int(unseen.source_entity_id[0, 1]) == -1
    assert int(unseen.source_entity_id[0, 0]) == 0


def test_the_heard_list_has_k_slots_per_row_and_pads_a_two_entity_row() -> None:
    heard = _heard(PAIR_5)
    assert heard.valid.shape == (1, heard_sounds.HEARD_CAPACITY)
    assert heard.relative_position.shape == (1, heard_sounds.HEARD_CAPACITY, 3)
    assert heard.age_seconds.shape == (1, heard_sounds.HEARD_CAPACITY)
    assert not bool(heard.valid[:, 2:].any())


# ---------------------------------------------------------------------------------------------------------------------
# The render: switch off and on.
# ---------------------------------------------------------------------------------------------------------------------


class _Engine:
    """One mounted gen-7 render; each batch row is one arena with its own entities."""

    def __init__(self, package: str) -> None:
        self.package = package
        self.combat = importlib.import_module(package + ".combat")
        self.arsenal = importlib.import_module(package + ".combat.arsenal")
        self.encoder = importlib.import_module(package + ".combat.observation.v3.encoding.encoder")
        self.layout = importlib.import_module(package + ".combat.observation.v3.policy.layout")
        self.params = self.combat.default_combat_params(microticks=1, target_active=False)
        loadout = self.arsenal.hytale_0_5_7_loadouts("iron_sword", target_profiles="iron_sword")
        self.config = self.arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=self.params.sensor_range)

    def config_for(self, batch: int):
        if batch == 1:
            return self.config
        loadout = self.arsenal.hytale_0_5_7_loadouts(["iron_sword"] * batch, target_profiles=["iron_sword"] * batch)
        return self.arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=self.params.sensor_range)

    def reset(self, positions):
        """One row per entry: entity 0 at the point, entity 1 at the default target spot."""

        return self.reset_pairs([(p, (0.0, 0.0, -2.5)) for p in positions])

    def reset_pairs(self, pairs):
        """One row per [entity 0, entity 1] pair."""

        batch = len(pairs)
        position = jnp.asarray([[p0, p1] for p0, p1 in pairs], dtype=jnp.float32)
        health = jnp.full((batch, ENTITY_COUNT), 0.0, dtype=jnp.float32)
        health = health.at[:, 0].set(self.params.agent_max_health).at[:, 1].set(self.params.target_max_health)
        state, _ = self.arsenal.reset_arsenal_batch(
            jax.random.split(jax.random.key(570057), batch),
            self.params,
            self.config_for(batch),
            entity_position=position,
            entity_health=health,
            entity_yaw=jnp.zeros((batch, ENTITY_COUNT), dtype=jnp.float32),
        )
        return state

    def observe(self, state, *, block_under_feet=None):
        batch = state.combat.health.shape[0]
        features = self.combat.empty_injected_world_features(batch)
        caps = self.arsenal.environment.open_flat_arsenal_world_capabilities(state, self.params, config=self.config)
        if block_under_feet is not None:
            if "block_under_feet" in caps._fields:
                caps = caps._replace(block_under_feet=jnp.asarray(block_under_feet, dtype=jnp.int32))
            else:
                fields = list(caps._fields) + ["block_under_feet"]
                cls = NamedTuple("OpenFlatCapsWithBlock", [(name, Any) for name in fields])
                caps = cls(*caps, jnp.asarray(block_under_feet, dtype=jnp.int32))
        return self.encoder.encode_learner_observation_v3(state, self.params, features, caps, self.config_for(batch))


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    name = "hearing_on_profile"
    shim = types.ModuleType(name)
    shim.PROFILE = replace(GEN7, constructs=replace(GEN7.constructs, hearing=True))
    sys.modules[name] = shim
    try:
        off_dir = tmp_path_factory.mktemp("render_off") / "gen7"
        on_dir = tmp_path_factory.mktemp("render_on") / "gen7"
        off_manifest = R.render(gen7_overlay.OVERLAY, off_dir, source=HeadSource())
        on_manifest = R.render(replace(gen7_overlay.OVERLAY, profile=f"{name}:PROFILE"), on_dir, source=HeadSource())
        off = _Engine(mount_fresh(off_dir).__name__)
        on = _Engine(mount_fresh(on_dir).__name__)
        yield SimpleNamespace(off=off, on=on, off_manifest=off_manifest, on_manifest=on_manifest)
        unmount(off.package)
        unmount(on.package)
    finally:
        sys.modules.pop(name, None)


def test_the_render_installs_the_hearing_hook_after_game_clock_in_both_renders(engines) -> None:
    for manifest in (engines.off_manifest, engines.on_manifest):
        applied = [entry for entry in manifest["applied"] if entry["id"] == "hearing"]
        assert len(applied) == 1
        assert applied[0]["construct"] == "hytalegym.sim.gen7.mechanics.heard_sounds:construct"
        assert applied[0]["function"] == "encode_learner_observation_v3"


def test_the_switch_off_observation_has_no_heard_sounds_field(engines) -> None:
    observation = engines.off.observe(engines.off.reset([(0.0, 0.0, 0.0)]))
    assert type(observation).__name__ == "LearnerCombatObservationV3"
    assert not hasattr(observation, "heard_sounds")


def test_the_switch_on_adds_one_field_last_and_keeps_every_v3_field(engines) -> None:
    state = engines.off.reset([(0.0, 0.0, 0.0)])
    off = engines.off.observe(state)
    on = engines.on.observe(engines.on.reset([(0.0, 0.0, 0.0)]))
    assert on._fields[-1] == "heard_sounds"
    assert on._fields[:-1] == off._fields
    for name in off._fields:
        left = jax.tree_util.tree_leaves(getattr(on, name))
        right = jax.tree_util.tree_leaves(getattr(off, name))
        assert len(left) == len(right), name
        for a, b in zip(left, right):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


def test_the_flat_policy_vector_is_unchanged_by_the_switch(engines) -> None:
    off_flat = engines.off.layout.arsenal_policy_observation(engines.off.observe(engines.off.reset([(0.0, 0.0, 0.0)])))
    on_flat = engines.on.layout.arsenal_policy_observation(engines.on.observe(engines.on.reset([(0.0, 0.0, 0.0)])))
    np.testing.assert_array_equal(np.asarray(on_flat), np.asarray(off_flat))


def test_the_render_entity_one_five_blocks_away_is_heard_by_entity_zero(engines) -> None:
    state = engines.on.reset_pairs([((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))])
    moving = state._replace(combat=state.combat._replace(velocity=jnp.asarray(
        [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=jnp.float32
    )))
    observation = engines.on.observe(moving, block_under_feet=[[semantic_id(PLANT), semantic_id(PLANT)]])
    heard = observation.heard_sounds
    assert float(heard.gain[0, _other_slot(heard, 0)]) == pytest.approx(_expected_gain(5.0), rel=1e-5)
    assert heard.valid.shape == (1, heard_sounds.HEARD_CAPACITY)


def test_the_dense_identity_grid_reaches_the_capabilities_and_the_hearing_field(engines) -> None:
    """S12 patch: a GeometryState whose identity grid holds a bush in entity 0's feet cell fills block_under_feet, and the
    render's hearing field carries that bush's MoveIn for entity 0's own footstep. Open-flat capabilities leave it None."""

    from hytalegym.geometry.contract import CELL_COUNT, CELL_RADIUS

    engine = engines.on
    state = engine.reset_pairs([((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))])
    state = state._replace(combat=state.combat._replace(velocity=jnp.asarray(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=jnp.float32
    )))
    open_flat = engines.off.arsenal.environment.open_flat_arsenal_world_capabilities(
        engines.off.reset_pairs([((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))]), engines.off.params, config=engines.off.config
    )
    assert open_flat.block_under_feet is None

    # The empty dense geometry comes from the real package and is re-wrapped in the mounted engine's own GeometryState
    # class: the engine checks isinstance against its copy of the class.
    from hytalegym.jax.evaluation.arsenal_benchmark import exact_empty_local_geometry

    mounted_geometry_state = engine.arsenal.environment.GeometryState
    geometry = mounted_geometry_state(*exact_empty_local_geometry(engine.params))
    side = CELL_RADIUS * 2 + 1
    centre = (CELL_RADIUS * side + CELL_RADIUS) * side + CELL_RADIUS  # the cell at the origin (feet at 0.0, 0.0, 0.0)
    grid = np.zeros((1, CELL_COUNT), dtype=np.int32)
    grid[0, centre] = semantic_id(PLANT)
    geometry = geometry._replace(origin=jnp.zeros((1, 3), dtype=jnp.int32), block_type_id=jnp.asarray(grid))
    caps = engine.arsenal.environment.geometry_arsenal_world_capabilities(
        state, engine.params, geometry=geometry, config=engine.config
    )
    # Entity 0 stands in the bush cell; entity 1 at x = 5 is outside the 9x9x9 window and reads 0.
    assert caps.block_under_feet.tolist() == [[semantic_id(PLANT), 0]]

    features = engine.combat.empty_injected_world_features(1)
    observation = engine.encoder.encode_learner_observation_v3(state, engine.params, features, caps, engine.config)
    heard = observation.heard_sounds
    own = [k for k in range(heard_sounds.HEARD_CAPACITY) if bool(heard.valid[0, k]) and float(heard.distance[0, k]) == 0.0]
    assert len(own) == 1
    assert int(heard.sound_id[0, own[0]]) == semantic_id("SFX_Bush_MoveIn")
    assert float(heard.gain[0, own[0]]) == pytest.approx(MOVE_IN_RECORD.volume_linear, rel=1e-6)
