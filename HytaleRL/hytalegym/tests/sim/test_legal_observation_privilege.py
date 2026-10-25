"""Legal-observation privilege differential on the rendered simulator (certification row 14).

Each pair differs only in something the learner must not know about an enemy standing behind an opaque wall: its presence,
position, health, held item, mana or role. Two worlds of a pair are built through the arsenal environment's reset seam with
``legal_observation=True`` and compared bit for bit on the structured learner observation (the v3 fields, plus the
``perception_v2`` field when the render carries it) and on the flat v3 policy vector with its action mask. The same
difference with the wall removed is the positive control: it must change the observation, or the pair proves nothing.

The construction follows the three-world leakage gate (tests/jax/combat/arsenal/tests/agents/ppo/test_leakage_three_world.py)
on the rendered package: the same detail-hitbox slab, the same entity-form reset. The flat vector is
``arsenal_policy_observation`` over the carry's action surface, which is what ``make_arsenal_ppo_environment`` computes
(hytalegym/jax/training/arsenal.py). That PPO wrapper is not rendered, so its reset is mirrored here.

Renders: gen 7 (perception_v2 off, its shipped default), gen 7 with perception_v2 on (tests/sim/row14_profiles.py), gen 5.
The role pair needs perception_v2 (the role id is published only there), so it is skipped on the other two renders.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from functools import cache, partial

import numpy as np
import pytest

from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5_OVERLAY
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY

from .support import mount_fresh, render_for_test, unmount

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# One seed: the same hidden world at a second seed shows nothing new for these pairs, and each pair costs a full reset.
SEEDS = (570057,)
AGENT_PROFILE = "iron_sword"
MANA = 1  # RESOURCE_MANA (jax/combat/mechanics/schema/contract.py)
AGENT_ROLE = "Kweebec_Razorleaf"  # the matchup's agent role (the catalogue's default pair)


def _cell(dx: int, dy: int, dz: int) -> int:
    return (dx + 4) * 81 + (dy + 4) * 9 + (dz + 4)


WALL_CELLS = (_cell(0, 0, 0), _cell(0, -1, 0))  # the slab at z in [-1, 0), two blocks high, in front of the agent


@dataclass(frozen=True)
class World:
    """One reset. The defaults are the three-world gate's hidden enemy: behind the wall, alive, iron sword, mana full."""

    wall: bool = True
    enemy_offset: tuple | None = None  # None: the params' target offset (2.5 blocks ahead, behind the wall)
    enemy_health: float | None = None  # None: the params' target max health; 0.0 is the absent slot
    enemy_profile: str = AGENT_PROFILE  # "" gives the enemy no loadout
    enemy_mana: float | None = None  # None: the loadout's mana (full)
    enemy_role: str = "Trork_Brawler"  # the catalogue's target role; "" is inactive padding

    def visible(self) -> "World":
        return replace(self, wall=False)


def _modules(package: str) -> dict:
    return {
        "combat": importlib.import_module(package + ".combat"),
        "environment": importlib.import_module(package + ".combat.arsenal.environment"),
        "arsenal": importlib.import_module(package + ".combat.arsenal"),
        "types": importlib.import_module(package + ".world.types"),
        "differential": importlib.import_module(package + ".combat.observation.v3.encoding.differential"),
    }


def _geometry(m: dict, params, *, wall: bool):
    """The three-world gate's slab geometry (origin (0, 66, -1), agent at (0.5, 65, 0.5)), on the rendered GeometryState."""

    from hytalegym.geometry import contract as C

    types = m["types"]
    flags = np.zeros((1, C.CELL_COUNT), np.int32)
    shape = np.full((1, C.CELL_COUNT), types.COLLISION_SHAPE_NONE, np.int16)
    cubes = list(WALL_CELLS) if wall else []
    for cell in cubes:
        flags[0, cell] = C.FLAG_OPAQUE | C.FLAG_SOLID
        shape[0, cell] = types.COLLISION_SHAPE_FULL_CUBE
    full_cube = np.full((1, 2), -1, np.int16)
    full_cube[0, : len(cubes)] = cubes
    return types.GeometryState(
        origin=jnp.asarray(((0, 66, -1),), dtype=jnp.int32),
        cell_mask=jnp.ones((1, C.CELL_COUNT), dtype=jnp.bool_),
        flags=jnp.asarray(flags),
        fluid_level=jnp.zeros((1, C.CELL_COUNT), dtype=jnp.int32),
        support=jnp.zeros((1, C.CELL_COUNT), dtype=jnp.int32),
        block_damage=jnp.zeros((1, C.CELL_COUNT), dtype=jnp.int32),
        fluid_damage=jnp.zeros((1, C.CELL_COUNT), dtype=jnp.int32),
        movement=jnp.zeros((1, C.CELL_COUNT, C.MOVEMENT_FEATURES), dtype=jnp.float32),
        fluid_movement=jnp.zeros((1, C.CELL_COUNT, C.FLUID_MOVEMENT_FEATURES), dtype=jnp.float32),
        collision_shape_index=jnp.asarray(shape),
        collision_full_cube_cell=jnp.asarray(full_cube),
        collision_exception_cell=jnp.full((1, 1), -1, dtype=jnp.int16),
        collision_exception_box_index=jnp.full((1, 1), -1, dtype=jnp.int16),
        collision_exception_boxes=jnp.zeros((1, 1, 6), dtype=jnp.float32),
        collision_exception_box_cell=jnp.full((1, 1), -1, dtype=jnp.int16),
        collision_world_order=jnp.asarray(((0, 1, 2),), dtype=jnp.int16),
        agent_bounds=params.agent_bounds[None, :],
        target_bounds=params.target_bounds[None, :],
        agent_los_offset=params.agent_eye_offset[None, :],
        target_los_offset=params.target_eye_offset[None, :],
    )


@cache
def _observe(package: str, world: World, seed: int):
    """Reset one world; return (structured learner observation, flat v3 vector, action mask, carry)."""

    m = _modules(package)
    params = m["combat"].default_combat_params(microticks=1)
    loadout = m["arsenal"].hytale_0_5_7_loadouts(
        [AGENT_PROFILE], target_profiles=[world.enemy_profile]
    )
    if world.enemy_mana is not None:
        loadout = loadout._replace(
            resource_initial=loadout.resource_initial.at[:, 1, MANA].set(world.enemy_mana)
        )
    config = m["arsenal"].arsenal_runtime_config(
        loadout, legal_observation=True, entity_role_ids=(AGENT_ROLE, world.enemy_role)
    )
    geometry = _geometry(m, params, wall=world.wall)

    agent_health = jnp.reshape(jnp.asarray(params.agent_max_health, dtype=jnp.float32), (-1,))[0]
    target_health = jnp.reshape(jnp.asarray(params.target_max_health, dtype=jnp.float32), (-1,))[0]
    enemy_health = target_health if world.enemy_health is None else jnp.float32(world.enemy_health)
    offset = params.target_offset if world.enemy_offset is None else jnp.asarray(world.enemy_offset, jnp.float32)
    positions = jnp.stack((params.agent_spawn, params.agent_spawn + offset)).astype(jnp.float32)
    healths = jnp.stack((agent_health, enemy_health))

    def reset_provider(keys):
        batch = keys.shape[0]
        return m["environment"].ArsenalResetBatch(
            entity_position=jnp.broadcast_to(positions, (batch, 2, 3)),
            entity_health=jnp.broadcast_to(healths, (batch, 2)),
        )

    environment = m["environment"].make_arsenal_environment(
        params,
        config,
        maximum_turn_degrees=m["environment"].REFERENCE_SKILL_MAXIMUM_TURN_DEGREES,
        geometry_provider=geometry,
        reset_provider=reset_provider,
        world_capability_provider=partial(
            m["environment"].geometry_arsenal_world_capabilities,
            geometry=geometry,
            config=config,
        ),
    )
    carry, observation = environment.reset(jax.random.split(jax.random.key(seed), 1))
    assert not bool(jnp.any(carry.runtime.combat.geometry_exhausted)), world
    flat = m["combat"].arsenal_policy_observation(
        observation,
        carry.action_surface.block_candidates,
        carry.action_surface.recipe_encoding,
        inventory_tokens=carry.inventory_tokens,
        light_tokens=carry.light_tokens,
    )
    return observation, flat, carry.action_mask


def _unequal(left, right) -> np.ndarray:
    left, right = np.asarray(left), np.asarray(right)
    assert (left.shape, left.dtype) == (right.shape, right.dtype)
    if left.dtype.kind == "f":
        unsigned = np.dtype(f"u{left.dtype.itemsize}")
        return left.view(unsigned) != right.view(unsigned)
    return left != right


def _differences(package: str, left, right) -> list[str]:
    """Name every structured learner field, flat policy group and mask entry that differs (bit inequality)."""

    tree = jax.tree_util
    (left_obs, left_flat, left_mask), (right_obs, right_flat, right_mask) = left, right
    found = [
        f"learner_observation{tree.keystr(path)}"
        for (path, expected), actual in zip(
            tree.tree_leaves_with_path(left_obs),
            tree.tree_leaves(right_obs),
            strict=True,
        )
        if _unequal(expected, actual).any()
    ]
    columns = _unequal(left_flat, right_flat).any(axis=0)
    if columns.any():
        groups = _modules(package)["differential"].arsenal_policy_observation_group_slices()
        names = [name for name, part in groups if columns[part].any()]
        found.append(f"flat policy {int(columns.sum())} column(s) in group(s) {names}")
    mask_entries = int(_unequal(left_mask, right_mask).sum())
    if mask_entries:
        found.append(f"{mask_entries} action mask entr(ies)")
    return found


# Each pair: (hidden world A, hidden world B). The visible control is the same pair with the wall removed.
PAIRS = {
    "enemy_position": (World(), World(enemy_offset=(0.0, 0.0, -4.0))),
    "enemy_presence_armed": (World(), World(enemy_health=0.0)),
    "enemy_presence_unarmed": (World(enemy_profile=""), World(enemy_profile="", enemy_health=0.0)),
    "enemy_health": (World(), World(enemy_health=1.0)),
    "enemy_held_item": (World(), World(enemy_profile="iron_battleaxe")),
    "enemy_mana": (World(), World(enemy_mana=5.0)),
    "enemy_role": (World(), World(enemy_role="")),
}
# Gen 5 has no legal-evidence patches (row 14 is gen-7 only; bit identity keeps gen 5 as it was): these three pairs leak there.
GEN5_LEAK_PAIRS = frozenset({"enemy_held_item", "enemy_presence_armed", "enemy_mana"})
GEN5_LEAK_REASON = (
    "gen 5 publishes the hidden enemy's held item, attack presence and mana; the row-14 patches are gen-7 only, and gen 5 "
    "is out of scope by the bit-identity rule"
)
GEN7_VACUOUS_PAIRS = frozenset({"enemy_mana"})
ROLE_CHANNEL_PAIRS = frozenset({"enemy_role"})  # the role id is published by perception_v2 only


def _pair(name: str, visible: bool):
    left, right = PAIRS[name]
    if visible:
        return left.visible(), right.visible()
    return left, right


def _overlays() -> dict:
    return {
        "gen7": GEN7_OVERLAY,
        "gen7_pv2": replace(
            GEN7_OVERLAY,
            profile="tests.sim.row14_profiles:PERCEPTION_V2",
            package="hytalegym.sim.gen7.engine_pv2",
        ),
        "gen5": GEN5_OVERLAY,
    }


@pytest.fixture(scope="module", params=("gen7", "gen7_pv2", "gen5"))
def render(request, tmp_path_factory):
    variant = request.param
    target = tmp_path_factory.mktemp(f"row14_{variant}") / variant
    target = render_for_test(_overlays()[variant], target, source=HeadSource())
    package = mount_fresh(target).__name__
    yield variant, package
    unmount(package)


def _skip_without_channel(variant: str, name: str) -> None:
    if name in ROLE_CHANNEL_PAIRS and variant != "gen7_pv2":
        pytest.skip(f"{name}: this render publishes no role id (perception_v2 is needed)")
    if name in GEN7_VACUOUS_PAIRS and variant != "gen5":
        pytest.skip(f"{name}: vacuous on {variant}: the visible control is unchanged (enemy mana 5.0 and 20.0 alike), so the "
                    "pair proves nothing here; gen 5 shows it (xfail above)")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", sorted(PAIRS))
def test_hidden_difference_is_invisible_to_the_learner(request, render, name, seed):
    """The privilege differential: the hidden pair must produce a bit-identical legal observation."""

    variant, package = render
    _skip_without_channel(variant, name)
    if variant == "gen5" and name in GEN5_LEAK_PAIRS:
        request.node.add_marker(pytest.mark.xfail(reason=GEN5_LEAK_REASON, strict=False))
    left, right = _pair(name, visible=False)
    found = _differences(package, _observe(package, left, seed), _observe(package, right, seed))
    assert found == [], f"[{variant}] {name} leaks through the legal observation: {found}"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", sorted(PAIRS))
def test_visible_difference_is_seen_by_the_learner(render, name, seed):
    """Positive control: the same difference with no wall must change the observation."""

    variant, package = render
    _skip_without_channel(variant, name)
    left, right = _pair(name, visible=True)
    found = _differences(package, _observe(package, left, seed), _observe(package, right, seed))
    assert found, f"[{variant}] {name} is invisible even when visible: the differential proves nothing"


@pytest.mark.parametrize("seed", SEEDS)
def test_hidden_enemy_is_outside_the_agent_line_of_sight(render, seed):
    """Precondition: the enemy is outside the agent's line of sight behind the wall, and inside it without the wall."""

    variant, package = render
    hidden = _observe(package, World(), seed)
    open_ = _observe(package, World().visible(), seed)
    assert not bool(np.asarray(hidden[0].base.target_mask).any()), variant
    assert bool(np.asarray(open_[0].base.target_mask).any()), variant
