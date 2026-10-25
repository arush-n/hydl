"""Gen-7 perception channel ``perception_v2`` (``sim/gen7/PERCEPTION_V2.md``; construct ``sim/gen7/mechanics/perception_v2.py``).

The gen-7 render (committed HEAD) is mounted; its encoder is wrapped by the construct the render's hook installs, with the
switch on. The learner observation of a two-entity pair is taken with the open flat world capabilities, and the agent's
line of sight to the target is then set to the outcome of each legal gate: visible, outside the view sector (the
sector test of ``environment.py:2994-2997`` reproduced in numpy from the head yaw), and behind an occluder (the detail
hitbox rule of ``sim/gen7/mechanics/line_of_sight.py`` on a synthetic window). The entity row is blanked in those two
cases, and the channel must blank its kind, held item and role with it.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import line_of_sight as los
from hytalegym.sim.gen7.mechanics import perception_v2
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, render_for_test, unmount

SEED = 570057
LOADOUT = "iron_sword"
WEAPON_ASSET = "Weapon_Sword_Iron"
ON = SimpleNamespace(constructs=SimpleNamespace(perception_v2=True))
HALF_SECTOR = math.radians(60.0)  # a 120-degree full sector, centred on the head yaw


class _Engine:
    """One gen-7 render's arsenal pair: the agent at the origin, the target ``target`` blocks away."""

    def __init__(self, package: str) -> None:
        self.package = package
        self.combat = import_module(package + ".combat")
        self.arsenal = import_module(package + ".combat.arsenal")
        self.environment = import_module(package + ".combat.arsenal.environment")
        self.encoder = import_module(package + ".combat.observation.v3.encoding.encoder")
        self.params = self.combat.default_combat_params(microticks=1, target_active=False)
        self.config = self._config()

    def _config(self, **overrides):
        loadout = self.arsenal.hytale_0_5_7_loadouts(LOADOUT, target_profiles=LOADOUT)
        return self.arsenal.arsenal_runtime_config(
            loadout, specialize=False, sensor_range=self.params.sensor_range, **overrides
        )

    def reset(self, yaw=(0.0, 0.0), target=(0.0, 0.0, -2.5), config=None, agent=(0.0, 0.0, 0.0)):
        position = jnp.asarray([[list(agent), list(target)]], dtype=jnp.float32)
        health = jnp.asarray([[self.params.agent_max_health, self.params.target_max_health]], dtype=jnp.float32)
        state, _ = self.arsenal.reset_arsenal_batch(
            jax.random.split(jax.random.key(SEED), 1),
            self.params,
            config or self.config,
            entity_position=position,
            entity_health=health,
            entity_yaw=jnp.asarray([yaw], dtype=jnp.float32),
        )
        return state

    def capabilities(self, state, *, line_of_sight=None, config=None):
        caps = self.environment.open_flat_arsenal_world_capabilities(state, self.params, config=config or self.config)
        if line_of_sight is not None:
            caps = caps._replace(line_of_sight=caps.line_of_sight.at[:, 0].set(line_of_sight))
        return caps

    def observe(self, state, *, encode, line_of_sight=None, config=None):
        config = config or self.config
        features = self.combat.empty_injected_world_features(1)
        caps = self.capabilities(state, line_of_sight=line_of_sight, config=config)
        return encode(state, self.params, features, caps, config)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    target = tmp_path_factory.mktemp("render") / "gen7"
    target = render_for_test(GEN7_OVERLAY, target, source=HeadSource())
    manifest = R.read_manifest(target)
    built = _Engine(mount_fresh(target).__name__)
    built.manifest = manifest
    built.on = perception_v2.construct(built.encoder.encode_learner_observation_v3, ON)
    yield built
    unmount(built.package)


@pytest.fixture(scope="module")
def on_engine(tmp_path_factory):
    """The gen-7 render with ``perception_v2`` on (its profile names the switch, as ``test_status_clock.py`` does for its
    off render): the encoder carries the channel and the config carries the role ids (``construct_roles``)."""

    name = "perception_v2_on_profile"
    module = types.ModuleType(name)
    module.PROFILE = replace(GEN7, constructs=replace(GEN7.constructs, perception_v2=True))
    sys.modules[name] = module  # the rendered package imports its profile when it mounts, so it stays until teardown
    try:
        target = tmp_path_factory.mktemp("render_on") / "gen7"
        target = render_for_test(replace(GEN7_OVERLAY, profile=f"{name}:PROFILE"), target, source=HeadSource())
        built = _Engine(mount_fresh(target).__name__)
        yield built
        unmount(built.package)
    finally:
        sys.modules.pop(name, None)


def _channel(engine, state, observation, config=None, capacity=perception_v2.ENTITY_CAPACITY):
    return perception_v2.channel(engine.package, state, config or engine.config, observation, entity_capacity=capacity)


def _sector_los(yaw_degrees: float, target) -> bool:
    """The agent's view sector test (head yaw forward, 120 degrees full angle) for the target's horizontal offset."""

    yaw = math.radians(yaw_degrees)
    forward = np.array([-math.sin(yaw), -math.cos(yaw)])
    offset = np.array([target[0], target[2]], dtype=np.float64)
    cosine = float(forward @ offset) / float(np.linalg.norm(offset))
    return math.acos(max(-1.0, min(1.0, cosine))) <= HALF_SECTOR


def _occluded_los() -> bool:
    """The detail-hitbox line of sight of ``sim/gen7/mechanics/line_of_sight.py`` on a 729-cell window with a full-cube
    occluder at cell (0, 0, -1), the cell between the agent (0.5, 0.5, 0.5) and a target at (0.5, 0.5, -2.5)."""

    radius = 4
    offsets = np.asarray([(x, y, z) for x in range(-radius, radius + 1) for y in range(-radius, radius + 1)
                          for z in range(-radius, radius + 1)], dtype=np.float32)
    index = (0 + radius) * 81 + (0 + radius) * 9 + (-1 + radius)
    flags = np.zeros((1, 729), np.int32)
    flags[0, index] = 1
    boxes = np.zeros((1, 729, 9, 6), np.float32)
    boxes[0, index, 0] = (0, 0, 0, 1, 1, 1)
    valid = np.zeros((1, 729, 9), bool)
    valid[0, index, 0] = True
    geometry = SimpleNamespace(
        origin=jnp.zeros((1, 3), jnp.int32),
        cell_mask=jnp.ones((1, 729), bool),
        flags=jnp.asarray(flags),
        collision_boxes=jnp.asarray(boxes),
        collision_box_mask=jnp.asarray(valid),
    )
    namespace = {
        "jnp": jnp,
        "VisibilityResult": SimpleNamespace,
        "_broadcast_geometry": lambda g, batch: g,
        "_CELL_OFFSETS": jnp.asarray(offsets),
        "CELL_RADIUS": radius,
        "FLAG_OPAQUE": 1,
        "_FLOAT_EPSILON": 1.0e-9,
    }
    exec("def line_of_sight_result(geometry, start, end):\n    raise AssertionError('gen-5 not called')\n", namespace)
    detail = los.construct(namespace["line_of_sight_result"], SimpleNamespace(constructs=SimpleNamespace(line_of_sight="detail_hitbox")))
    start = jnp.asarray([[0.5, 0.5, 0.5]], jnp.float32)
    end = jnp.asarray([[0.5, 0.5, -2.5]], jnp.float32)
    return not bool(detail(geometry, start, end).visible[0])


# ---------------------------------------------------------------------------------------------------------------------
# The switch and the render.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_switch_is_false_in_the_defaults_and_in_both_profiles() -> None:
    assert Constructs().perception_v2 is False
    assert GEN5.constructs.perception_v2 is False
    assert GEN7.constructs.perception_v2 is False


def test_the_construct_returns_the_original_when_the_switch_is_off() -> None:
    def original(*args, **kwargs):
        return args, kwargs

    assert perception_v2.construct(original, GEN7) is original


def test_a_non_bool_switch_fails_closed() -> None:
    with pytest.raises(ValueError, match="perception_v2"):
        perception_v2.construct(lambda *args, **kwargs: None, SimpleNamespace(constructs=SimpleNamespace(perception_v2="yes")))


def test_the_gen7_render_installs_the_hook(engine) -> None:
    applied = [entry for entry in engine.manifest["applied"] if entry["id"] == "perception_v2"]
    assert len(applied) == 1
    assert applied[0]["construct"] == "hytalegym.sim.gen7.mechanics.perception_v2:construct"
    assert applied[0]["function"] == "encode_learner_observation_v3"


def test_the_switch_off_keeps_the_v3_observation(engine) -> None:
    state = engine.reset()
    observation = engine.observe(state, encode=engine.encoder.encode_learner_observation_v3)
    assert type(observation).__name__ == "LearnerCombatObservationV3"
    assert not hasattr(observation, "perception_v2")


# ---------------------------------------------------------------------------------------------------------------------
# The switch on: the v3 fields pass through, the channel has its layout, and each entity field follows the row gate.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_switch_on_keeps_every_v3_field_unchanged(engine) -> None:
    state = engine.reset()
    off = engine.observe(state, encode=engine.encoder.encode_learner_observation_v3)
    on = engine.observe(state, encode=engine.on)
    assert on._fields == off._fields + ("perception_v2",)
    for name in off._fields:
        left = jax.tree_util.tree_leaves(getattr(on, name))
        right = jax.tree_util.tree_leaves(getattr(off, name))
        assert len(left) == len(right), name
        for a, b in zip(left, right):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


@pytest.mark.parametrize("capacity", [16, 32])
def test_shapes_and_dtypes(engine, capacity) -> None:
    state = engine.reset()
    observation = engine.observe(state, encode=engine.encoder.encode_learner_observation_v3)
    channel = _channel(engine, state, observation, capacity=capacity)
    expected = {
        "entity_kind": ((1, capacity), jnp.int32),
        "entity_role_id": ((1, capacity), jnp.int32),
        "entity_held_item_id": ((1, capacity), jnp.int32),
        "inventory_i32": ((1, 76, 3), jnp.int32),
        "inventory_f32": ((1, 76, 7), jnp.float32),
        "inventory_mask": ((1, 76), jnp.bool_),
        "body_yaw_sin": ((1,), jnp.float32),
        "body_yaw_cos": ((1,), jnp.float32),
        "self_position": ((1, 3), jnp.float32),
    }
    assert channel._fields == tuple(expected)
    for name, (shape, dtype) in expected.items():
        array = getattr(channel, name)
        assert tuple(array.shape) == shape, name
        assert array.dtype == dtype, name


def test_a_visible_target_fills_kind_and_held_item_and_pads_the_rest(engine) -> None:
    state = engine.reset()
    observation = engine.observe(state, encode=engine.encoder.encode_learner_observation_v3, line_of_sight=True)
    assert bool(observation.base.entity_mask[0, 0])
    channel = _channel(engine, state, observation)
    assert int(channel.entity_kind[0, 0]) == 1  # ENTITY_KIND_NPC
    assert int(channel.entity_held_item_id[0, 0]) == semantic_id(WEAPON_ASSET)
    assert not np.any(np.asarray(channel.entity_kind)[0, 1:])
    assert not np.any(np.asarray(channel.entity_held_item_id)[0, 1:])


def test_a_visible_target_with_the_no_weapon_sentinel_reports_nothing_held(engine) -> None:
    # The loadout marks an empty hand with weapon_id -1; the channel must publish 0, never a non-semantic id
    # (PERCEPTION_V2.md section 4; reported by the skill layer on the open-flat arena, 2026-10-19).
    state = engine.reset()
    observation = engine.observe(state, encode=engine.encoder.encode_learner_observation_v3, line_of_sight=True)
    assert bool(observation.base.entity_mask[0, 0])
    target = getattr(perception_v2, "TARGET_ENTITY", 1)
    config = engine.config
    loadout = config.loadout._replace(weapon_id=config.loadout.weapon_id.at[:, target].set(-1))
    config = config._replace(loadout=loadout) if hasattr(config, "_replace") else __import__("dataclasses").replace(config, loadout=loadout)
    channel = _channel(engine, state, observation, config=config)
    assert int(channel.entity_kind[0, 0]) == 1
    assert int(channel.entity_held_item_id[0, 0]) == 0
    assert not np.any(np.asarray(channel.entity_held_item_id) < 0)


def test_a_target_outside_the_view_sector_blanks_the_row_and_its_fields(engine) -> None:
    # The target stands behind the agent (+z), 180 degrees off the head yaw: outside a 120-degree sector.
    target = (0.0, 0.0, 2.5)
    assert not _sector_los(0.0, target)
    state = engine.reset(target=target)
    observation = engine.observe(state, encode=engine.on, line_of_sight=_sector_los(0.0, target))
    assert not bool(observation.base.entity_mask[0, 0])
    channel = observation.perception_v2
    assert int(channel.entity_kind[0, 0]) == 0
    assert int(channel.entity_held_item_id[0, 0]) == 0
    assert int(channel.entity_role_id[0, 0]) == 0


def test_a_target_behind_an_occluder_blanks_the_row_and_its_fields(engine) -> None:
    assert _occluded_los() is True
    state = engine.reset()
    observation = engine.observe(state, encode=engine.on, line_of_sight=False)
    assert not bool(observation.base.entity_mask[0, 0])
    channel = observation.perception_v2
    assert int(channel.entity_kind[0, 0]) == 0
    assert int(channel.entity_held_item_id[0, 0]) == 0
    assert int(channel.entity_role_id[0, 0]) == 0


def test_a_player_backed_target_has_the_player_kind(engine) -> None:
    config = engine._config(player_backed_actor_mask=jnp.asarray([[False, True]]))
    state = engine.reset(config=config)
    observation = engine.observe(state, encode=engine.on, line_of_sight=True, config=config)
    assert int(observation.perception_v2.entity_kind[0, 0]) == 2  # ENTITY_KIND_PLAYER


def test_body_yaw_is_the_state_yaw_in_radians(engine) -> None:
    state = engine.reset(yaw=(30.0, 0.0))
    observation = engine.observe(state, encode=engine.on)
    channel = observation.perception_v2
    assert float(channel.body_yaw_sin[0]) == pytest.approx(math.sin(math.radians(30.0)), abs=1e-6)
    assert float(channel.body_yaw_cos[0]) == pytest.approx(math.cos(math.radians(30.0)), abs=1e-6)


def test_inventory_tokens_republish_the_v3_floats(engine) -> None:
    inventory = import_module(engine.package + ".combat.observation.v3.tokens.inventory")
    state = engine.reset()
    observation = engine.observe(state, encode=engine.on)
    tokens = inventory.inventory_policy_tokens_from_state(
        state.inventory, engine.config.inventory_layout, actor_valid=observation.valid
    )
    np.testing.assert_array_equal(np.asarray(observation.perception_v2.inventory_f32), np.asarray(tokens.token_f32))
    np.testing.assert_array_equal(np.asarray(observation.perception_v2.inventory_mask), np.asarray(tokens.token_mask))


def test_inventory_item_ids_are_semantic_ids(engine) -> None:
    state = engine.reset()
    item = semantic_id(WEAPON_ASSET)
    inventory = state.inventory._replace(
        item_id=state.inventory.item_id.at[0, 0, 0].set(item),
        quantity=state.inventory.quantity.at[0, 0, 0].set(1),
    )
    state = state._replace(inventory=inventory)
    observation = engine.observe(state, encode=engine.on)
    channel = observation.perception_v2
    present = np.asarray(channel.inventory_mask)[0]
    ids = np.asarray(channel.inventory_i32)[0, :, 2]
    assert item in ids[present].tolist()
    assert not np.any(ids[~present])


# ---------------------------------------------------------------------------------------------------------------------
# Task #96 (2026-10-19): the agent's own position, the last field of the channel. Ruling: a player reads their own
# coordinates, so it is legal; it is gen-7 only, like the rest of the channel.
# ---------------------------------------------------------------------------------------------------------------------


def test_self_position_is_the_combat_state_agent_position(engine) -> None:
    state = engine.reset(agent=(3.5, 1.0, -2.25))
    observation = engine.observe(state, encode=engine.on)
    position = observation.perception_v2.self_position
    assert position.dtype == jnp.float32 and tuple(position.shape) == (1, 3)
    np.testing.assert_array_equal(np.asarray(position), np.asarray(state.combat.position[:, AGENT_ENTITY]))
    assert not np.allclose(np.asarray(position)[0], np.asarray(state.combat.position[0, TARGET_ENTITY]))


def test_self_position_is_appended_and_the_older_fields_keep_their_order() -> None:
    assert perception_v2.PerceptionV2._fields == (
        "entity_kind", "entity_role_id", "entity_held_item_id", "inventory_i32", "inventory_f32", "inventory_mask",
        "body_yaw_sin", "body_yaw_cos", "self_position",
    )


def test_gen5_keeps_its_construct_absent() -> None:
    def original(*args, **kwargs):
        return args, kwargs

    assert GEN5.constructs.perception_v2 is False
    assert perception_v2.construct(original, GEN5) is original


# ---------------------------------------------------------------------------------------------------------------------
# Task #130 item 4 (2026-10-22): the role id. The channel published 0 for every entity: the runtime config did not keep the
# role names (arsenal_runtime_config(entity_role_ids=...) fed the status builder only). construct_roles keeps them as
# semantic ids on the config; the channel publishes the target's (row 0) for a visible NPC target.
# ---------------------------------------------------------------------------------------------------------------------

KWEEBEC = "Kweebec_Razorleaf"  # ruleset matchup agent role (rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json)
TRORK = "Trork_Brawler"  # ruleset matchup target role


def _role_ids(config) -> list[int]:
    return np.asarray(config.entity_role_semantic_ids)[0].tolist()


def _observe_roles(engine, config, **kwargs):
    state = engine.reset(config=config)
    observation = engine.observe(state, encode=engine.encoder.encode_learner_observation_v3, config=config, **kwargs)
    return observation.perception_v2


def test_the_switch_off_config_carries_no_role_ids(engine) -> None:
    assert engine.config.entity_role_semantic_ids is None


def test_the_matchup_default_gives_the_agent_and_the_target_their_role_ids(on_engine) -> None:
    config = on_engine.config
    assert _role_ids(config) == [semantic_id(KWEEBEC), semantic_id(TRORK)]
    channel = _observe_roles(on_engine, config, line_of_sight=True)
    assert int(channel.entity_role_id[0, 0]) == semantic_id(TRORK)
    assert not np.any(np.asarray(channel.entity_role_id)[0, 1:])


def test_the_duel_roles_reach_the_config_and_the_target_channel(on_engine) -> None:
    config = on_engine._config(entity_role_ids=(KWEEBEC, TRORK))
    assert _role_ids(config) == [semantic_id(KWEEBEC), semantic_id(TRORK)]
    assert int(_observe_roles(on_engine, config, line_of_sight=True).entity_role_id[0, 0]) == semantic_id(TRORK)


def test_swapped_duel_roles_put_the_kweebec_on_the_target_row(on_engine) -> None:
    config = on_engine._config(entity_role_ids=(TRORK, KWEEBEC))
    assert _role_ids(config) == [semantic_id(TRORK), semantic_id(KWEEBEC)]
    assert int(_observe_roles(on_engine, config, line_of_sight=True).entity_role_id[0, 0]) == semantic_id(KWEEBEC)


def test_an_empty_role_name_is_no_role(on_engine) -> None:
    # The source genuinely has no role for entity 1: the role name is "" (the padding of entity_role_ids).
    config = on_engine._config(entity_role_ids=(KWEEBEC, ""))
    assert _role_ids(config) == [semantic_id(KWEEBEC), 0]
    channel = _observe_roles(on_engine, config, line_of_sight=True)
    assert bool(np.asarray(channel.entity_kind)[0, 0] == 1)
    assert int(channel.entity_role_id[0, 0]) == 0


def test_a_player_backed_target_has_no_npc_role(on_engine) -> None:
    # A player has no NPC role asset: the kind is PLAYER and the role field reads 0, whatever role name the config names.
    config = on_engine._config(player_backed_actor_mask=jnp.asarray([[False, True]]))
    channel = _observe_roles(on_engine, config, line_of_sight=True)
    assert int(channel.entity_kind[0, 0]) == 2  # ENTITY_KIND_PLAYER
    assert int(channel.entity_role_id[0, 0]) == 0
