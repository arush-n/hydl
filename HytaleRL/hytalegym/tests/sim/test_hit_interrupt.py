"""Hit interrupt, pure logic (gen 7, row 5a; ``sim/gen7/mechanics/hit_interrupt.py``).

Two kinds of test. The pure-logic tests run the construct's functions on ``StatusState`` arrays of one batch and two entities
(the target is entity 1). The render tests mount the gen-7 render (switch on, switch off, and the committed overlay without the
hit-interrupt rows) and drive the engine's own step: the switch-off render must equal the committed render leaf by leaf, and the
switch-on render must show the measured timeline. Removal follows the stock decrement of ``mechanics/runtime/statuses.py``
tick_statuses (``remaining = max(0, remaining - dt)``, active while > 0) at dt = 1/30 s in float32.

Live check (hit_interrupt_part1 receipt, Skeleton_Fighter, seed 570057): an iron sword swing requested on tick r lands on tick
r + 8 (the swing requested on tick 1 lands on tick 9, the one on tick 21 on tick 29). The render's sword timing (the Replace
default hop, ``sim/gen7/mechanics/replace_default_hop.py``) now lands the swings of this file on steps r + 8 as well: requests
on ticks 2 and 40 hit on steps 10 and 48. The step numbers below are the render's, measured with the shipped gen-7 overlay.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.mechanics.schema.contract import STATUS_FLAG_DISABLE_ABILITIES, STATUS_FLAG_DISABLE_MOVEMENT
from hytalegym.jax.combat.mechanics.schema.types import StatusState
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import hit_interrupt as hi
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, unmount

TARGET = 1
TABLE = hi.load_table(GEN7.data_dir)
DT = np.float32(1.0 / 30.0)
ALL_ON = jnp.asarray([True])
ALL_OFF = jnp.asarray([False])
SWORD = (
    "Weapon_Sword_Primary_Swing_Left",
    "Weapon_Sword_Primary_Swing_Right",
    "Weapon_Sword_Primary_Swing_Down",
    "Weapon_Sword_Primary_Thrust",
)


def _empty(batch: int = 1, entities: int = 2, slots: int = 8) -> StatusState:
    dims = (batch, entities, slots)
    f32 = jnp.zeros(dims, jnp.float32)
    i32 = jnp.zeros(dims, jnp.int32)
    return StatusState(
        effect_id=i32,
        source_entity_id=jnp.full(dims, -1, jnp.int32),
        remaining_seconds=f32,
        cycle_elapsed_seconds=f32,
        cycle_cooldown_seconds=f32,
        damage_per_cycle=f32,
        damage_cause=i32,
        healing_per_cycle=f32,
        resource_id=jnp.full(dims, -1, jnp.int32),
        resource_delta_per_cycle=f32,
        speed_multiplier=jnp.ones(dims, jnp.float32),
        damage_resistance_present=jnp.zeros(dims + (15,), jnp.bool_),
        damage_resistance_flat=jnp.zeros(dims + (15,), jnp.float32),
        damage_resistance_multiplier=jnp.zeros(dims + (15,), jnp.float32),
        flags=jnp.zeros(dims, jnp.uint32),
        overlap_mode=i32,
        active=jnp.zeros(dims, jnp.bool_),
        has_cycled=jnp.zeros(dims, jnp.bool_),
    )


def _slot(status: StatusState, name: str):
    """(active, remaining seconds, speed, flags) of the target's slot for ``name``; active False when absent."""

    sid = semantic_id(name)
    hits = np.flatnonzero(np.asarray(status.active[0, TARGET]) & (np.asarray(status.effect_id[0, TARGET]) == sid))
    if hits.size == 0:
        return False, None, None, None
    index = int(hits[0])
    return (True, float(status.remaining_seconds[0, TARGET, index]), float(status.speed_multiplier[0, TARGET, index]),
            int(status.flags[0, TARGET, index]))


def _tick(status: StatusState) -> StatusState:
    remaining = jnp.maximum(jnp.float32(0.0), status.remaining_seconds - DT)
    return status._replace(remaining_seconds=remaining, active=status.active & (remaining > 0.0))


def _step(status: StatusState, tick: int, *, eligible=ALL_ON) -> StatusState:
    """One microtick: the stock decrement, then the construct's eligibility clock (as the hook would run it)."""

    status = _tick(status)
    return hi.eligible_update(status, TARGET, jnp.asarray([tick], jnp.int32), eligible, TABLE)


def _run_to(status: StatusState, start: int, stop: int) -> tuple[StatusState, int]:
    for tick in range(start, stop + 1):
        status = _step(status, tick)
    return status, stop


def test_the_table_carries_the_eight_effects_and_the_31_attacks() -> None:
    assert len(TABLE.attacks) == len(set(TABLE.attacks)) == 31
    assert set(TABLE.effects) == set(hi.EFFECT_NAMES)
    durations = {name: effect.duration_seconds for name, effect in TABLE.effects.items()}
    assert durations == {
        "Hit_Interrupt_Eligible": 1.0, "Hit_Interrupt_Flinch": 0.5, "Hit_Interrupt_Stunned": 0.7,
        "Hit_Interrupt_Recent_Reaction_A": 4.0, "Hit_Interrupt_Recent_Reaction_B": 4.0,
        "Hit_Interrupt_Recovery_Pending": 2.0, "Hit_Interrupt_Enraged": 8.0, "Hit_Interrupt_Protected_Attack": 8.0,
    }
    assert TABLE.effects["Hit_Interrupt_Flinch"].speed_multiplier == 0.0
    assert TABLE.effects["Hit_Interrupt_Flinch"].flags == STATUS_FLAG_DISABLE_MOVEMENT
    assert TABLE.effects["Hit_Interrupt_Stunned"].flags == STATUS_FLAG_DISABLE_ABILITIES
    families = {prefix: sum(name.startswith(prefix) for name in TABLE.attacks)
                for prefix in ("Weapon_Daggers_", "Weapon_Mace_", "Weapon_Sword_", "Weapon_Battleaxe_", "Ability_")}
    assert families == {"Weapon_Daggers_": 9, "Weapon_Mace_": 7, "Weapon_Sword_": 6, "Weapon_Battleaxe_": 5, "Ability_": 4}


def test_a_foreign_or_partial_table_is_refused(tmp_path) -> None:
    source = json.loads((GEN7.data_dir / hi.DATA_FILE).read_text(encoding="utf-8"))
    wrong = dict(source, family="crafting_augment_tags")
    path = tmp_path / hi.DATA_FILE
    path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError):
        hi._read.__wrapped__(str(path))
    partial = dict(source, rows={k: v for k, v in source["rows"].items() if not k.startswith("effect:Hit_Interrupt_Enraged")})
    path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks effect rows"):
        hi._read.__wrapped__(str(path))


def test_the_semantic_ids_fit_the_int32_status_field() -> None:
    for name in hi.EFFECT_NAMES:
        assert -(2**31) <= semantic_id(name) < 2**31, name


def test_sword_rows_map_to_their_arsenal_roots_and_others_do_not() -> None:
    flagged = set(hi.flagged_ability_ids(TABLE))
    assert {semantic_id(name) for name in SWORD} <= flagged
    assert semantic_id("Weapon_Shortbow_Primary_Shoot") not in flagged


def test_flagged_events_follow_the_ability_event_ranges() -> None:
    ability_id = [semantic_id(SWORD[0]), semantic_id("Weapon_Shortbow_Primary_Shoot"), semantic_id(SWORD[1])]
    start = [0, 4, 8]
    count = [4, 4, 4]
    events = jnp.asarray([[0, 5, 9, 12]], jnp.int32)
    flagged = hi.flagged_events(ability_id, start, count, events, hi.flagged_ability_ids(TABLE))
    assert np.asarray(flagged).tolist() == [[True, False, True, False]]


def test_eligibility_is_added_at_the_measured_tick_and_refreshed_on_the_quarter_second_grid() -> None:
    status = _empty()
    refreshed = []
    for tick in range(1, 61):
        before = status
        status = _step(status, tick)
        present, remaining, _, _ = _slot(status, "Hit_Interrupt_Eligible")
        if tick < hi.ELIGIBLE_DELAY_TICKS:
            assert not present, tick
        elif tick == hi.ELIGIBLE_DELAY_TICKS:
            assert present and remaining == pytest.approx(1.0), tick
        else:
            assert present, tick
            if _slot(before, "Hit_Interrupt_Eligible")[1] is not None and remaining > _slot(before, "Hit_Interrupt_Eligible")[1]:
                refreshed.append(tick)
    assert refreshed == [45, 53, 60]


def test_an_ineligible_target_never_gets_eligibility() -> None:
    status = _empty()
    for tick in range(1, 60):
        status = _step(status, tick, eligible=ALL_OFF)
    assert not _slot(status, "Hit_Interrupt_Eligible")[0]


def test_a_hit_before_eligibility_gives_nothing() -> None:
    status = _empty()
    for tick in range(1, 30):
        status = _step(status, tick, eligible=ALL_OFF)
    status = hi.apply_hit(status, TARGET, ALL_ON, TABLE)
    for name in ("Hit_Interrupt_Stunned", "Hit_Interrupt_Flinch", "Hit_Interrupt_Recent_Reaction_A"):
        assert not _slot(status, name)[0], name


def test_a_hit_while_eligible_gives_the_three_statuses_with_their_durations() -> None:
    status, _ = _run_to(_empty(), 1, hi.ELIGIBLE_DELAY_TICKS)
    status = hi.apply_hit(status, TARGET, ALL_ON, TABLE)
    stunned = _slot(status, "Hit_Interrupt_Stunned")
    flinch = _slot(status, "Hit_Interrupt_Flinch")
    reaction = _slot(status, "Hit_Interrupt_Recent_Reaction_A")
    assert stunned[0] and stunned[1] == pytest.approx(0.7 - hi.HALF_TICK_SECONDS) and stunned[3] == STATUS_FLAG_DISABLE_ABILITIES
    assert flinch[0] and flinch[1] == pytest.approx(0.5 - hi.HALF_TICK_SECONDS) and flinch[2] == 0.0
    assert flinch[3] == STATUS_FLAG_DISABLE_MOVEMENT
    assert reaction[0] and reaction[1] == pytest.approx(4.0 - hi.HALF_TICK_SECONDS)
    assert not _slot(status, "Hit_Interrupt_Recent_Reaction_B")[0]
    assert not _slot(status, "Hit_Interrupt_Recovery_Pending")[0]


def test_precompensated_durations_are_removed_after_15_21_and_120_ticks() -> None:
    """The stock clock (statuses.py: remaining = max(0, remaining - dt), removed at <= 0) in float32 at 1/30 s. The construct
    writes Flinch, Stunned and Reaction A at duration minus half a tick, so the counts are the native frame counts 15, 21, 120."""

    dt = np.float32(1.0 / 30.0)
    counts = {}
    for name in ("Hit_Interrupt_Flinch", "Hit_Interrupt_Stunned", "Hit_Interrupt_Recent_Reaction_A"):
        remaining = np.float32(hi._effect(TABLE, name).duration_seconds)
        steps = 0
        while True:
            steps += 1
            remaining = np.float32(max(np.float32(0.0), np.float32(remaining - dt)))
            if remaining <= 0.0:
                break
        counts[name] = steps
    assert counts == {"Hit_Interrupt_Flinch": 15, "Hit_Interrupt_Stunned": 21, "Hit_Interrupt_Recent_Reaction_A": 120}


def test_the_second_and_third_hits_use_the_reaction_slots_and_enrage_follows_the_flinch() -> None:
    status, tick = _run_to(_empty(), 1, hi.ELIGIBLE_DELAY_TICKS)
    status = hi.apply_hit(status, TARGET, ALL_ON, TABLE)
    status, tick = _run_to(status, tick + 1, tick + 16)
    assert not _slot(status, "Hit_Interrupt_Flinch")[0]
    status = hi.apply_hit(status, TARGET, ALL_ON, TABLE)
    assert _slot(status, "Hit_Interrupt_Recent_Reaction_B")[0]
    assert not _slot(status, "Hit_Interrupt_Recovery_Pending")[0]
    status, tick = _run_to(status, tick + 1, tick + 16)
    status = hi.apply_hit(status, TARGET, ALL_ON, TABLE)
    assert _slot(status, "Hit_Interrupt_Recovery_Pending")[1] == pytest.approx(2.0)
    assert _slot(status, "Hit_Interrupt_Flinch")[0]
    status = hi.enrage_after_flinch(status, TARGET, TABLE)
    assert not _slot(status, "Hit_Interrupt_Enraged")[0]
    status, tick = _run_to(status, tick + 1, tick + 16)
    assert not _slot(status, "Hit_Interrupt_Flinch")[0]
    status = hi.enrage_after_flinch(status, TARGET, TABLE)
    assert _slot(status, "Hit_Interrupt_Enraged")[1] == pytest.approx(8.0)


def test_the_gate_blocks_a_hit_during_flinch() -> None:
    status, tick = _run_to(_empty(), 1, hi.ELIGIBLE_DELAY_TICKS)
    status = hi.apply_hit(status, TARGET, ALL_ON, TABLE)
    status, tick = _run_to(status, tick + 1, tick + 5)
    before = _slot(status, "Hit_Interrupt_Stunned")[1]
    status = hi.apply_hit(status, TARGET, ALL_ON, TABLE)
    assert _slot(status, "Hit_Interrupt_Stunned")[1] == pytest.approx(before)
    assert not _slot(status, "Hit_Interrupt_Recent_Reaction_B")[0]


def test_an_unlanded_hit_changes_nothing() -> None:
    status, _ = _run_to(_empty(), 1, hi.ELIGIBLE_DELAY_TICKS)
    status = hi.apply_hit(status, TARGET, ALL_OFF, TABLE)
    assert not _slot(status, "Hit_Interrupt_Stunned")[0] and not _slot(status, "Hit_Interrupt_Flinch")[0]


def test_the_switch_is_off_by_default_and_the_hooks_pass_through_when_off() -> None:
    sentinel = object()
    assert Constructs().hit_interrupt is False
    assert GEN5.constructs.hit_interrupt is False
    assert hi.IMPLEMENTED is True
    assert hi.construct_roles(sentinel, GEN5) is sentinel
    assert hi.construct_reset(sentinel, GEN5) is sentinel
    assert hi.construct_clock(sentinel, GEN5) is sentinel
    passthrough = hi.construct_hits(lambda *args, **kwargs: (args, kwargs), GEN5)
    assert passthrough(1, 2, engagement_target_id=3, loadout="strip me") == ((1, 2), {"engagement_target_id": 3})


# ---- render tests: the gen-7 render with the switch on, off, and the committed overlay (no hit-interrupt rows) -------------

SEED = 570057
AGENT, TARGET = 0, 1
SWING = 0  # the iron sword's first attack root (Swing_Left; hytale_0_5_7 _iron_sword ability index 0)
DISTANCE = 1.5  # the target, in front of the agent (the agent faces -z at yaw 0)
EARLY, LATE = 2, 40  # swing requests: the hit at step 10 (before tick 42) and the hit at step 48 (the first at or after 42)


class _Scenario:
    """One render's engine: the iron sword against a target 1.5 blocks in front, the target's own motion on."""

    def __init__(self, package: str, roles: Any = None) -> None:
        self.combat = import_module(package + ".combat")
        self.arsenal = import_module(package + ".combat.arsenal")
        environment = import_module(package + ".combat.arsenal.environment")
        self.params = self.combat.default_combat_params(microticks=1, target_active=True)
        loadout = self.arsenal.hytale_0_5_7_loadouts("iron_sword", target_profiles="iron_sword")
        extra = {} if roles is None else {"entity_role_ids": roles}
        self.config = self.arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=self.params.sensor_range, **extra)
        position = jnp.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, -DISTANCE]]], jnp.float32)
        health = jnp.asarray([[self.params.agent_max_health, self.params.target_max_health]], jnp.float32)
        yaw = jnp.zeros((1, 2), jnp.float32)
        self.initial, _ = self.arsenal.reset_arsenal_batch(jax.random.split(jax.random.key(SEED), 1), self.params, self.config,
                                                           entity_position=position, entity_health=health, entity_yaw=yaw)
        self.world = environment.open_flat_arsenal_world_capabilities(self.initial, self.params, config=self.config)

    def run(self, requests: dict[int, int], steps: int) -> tuple[Any, dict[str, np.ndarray]]:
        """The engine's own step for ``steps`` ticks; ``requests`` maps a 1-based tick to an ability slot."""

        slots = jnp.asarray([requests.get(tick, -1) for tick in range(1, steps + 1)], jnp.int32)
        keys = jax.random.split(jax.random.key(161803), steps).reshape(steps, 1)
        action = jnp.zeros((1, self.combat.ACTION_SIZE), jnp.float32)

        def body(state, xs):
            slot, key = xs
            command = self.arsenal.empty_arsenal_commands(1)
            command = command._replace(ability_slot=command.ability_slot.at[:, AGENT].set(slot), world=self.world)
            nxt = self.arsenal.step_arsenal_batch(state, action, key, self.params, command, self.config).state
            statuses = nxt.mechanics.statuses
            return nxt, {"hp": nxt.combat.health[0, TARGET], "active": statuses.active[0, TARGET],
                         "effect_id": statuses.effect_id[0, TARGET], "remaining": statuses.remaining_seconds[0, TARGET],
                         "pos": nxt.combat.position[0, TARGET]}

        final, trace = jax.jit(lambda state, xs: jax.lax.scan(body, state, xs))(self.initial, (slots, keys))
        return final, {name: np.asarray(value) for name, value in trace.items()}


def _steps(trace: dict[str, np.ndarray], name: str) -> list[int]:
    """The 1-based steps on which the target carries the effect ``name``."""

    present = np.any(trace["active"] & (trace["effect_id"] == semantic_id(name)), axis=-1)
    return [int(index) + 1 for index in np.flatnonzero(present)]


def _remaining(trace: dict[str, np.ndarray], name: str) -> np.ndarray:
    match = trace["active"] & (trace["effect_id"] == semantic_id(name))
    return np.where(match, trace["remaining"], 0.0).max(axis=-1)


def _hit_steps(trace: dict[str, np.ndarray]) -> list[int]:
    hp = trace["hp"]
    return [index + 1 for index in range(1, len(hp)) if hp[index] < hp[index - 1]]


def _leaves(tree: Any) -> dict[str, np.ndarray]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


@contextmanager
def _mounted(path: Path) -> Iterator[str]:
    """One render mounted at a time: every render mounts under the same package name, so a test mounts the one it runs."""

    package = mount_fresh(path).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.fixture(scope="module")
def render_dirs(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("hit_interrupt_render")
    committed = replace(
        GEN7_OVERLAY,
        hooks=tuple(hook for hook in GEN7_OVERLAY.hooks if not hook.id.startswith("hit_interrupt_")),
        patches=tuple(patch for patch in GEN7_OVERLAY.patches if not patch.id.startswith("hit_interrupt_")),
    )
    module_name = "hit_interrupt_test_off_profile"
    off_module = types.ModuleType(module_name)
    off_module.PROFILE = replace(GEN7, constructs=replace(GEN7.constructs, hit_interrupt=False))
    sys.modules[module_name] = off_module
    try:
        dirs = {}
        for label, overlay in (("on", GEN7_OVERLAY), ("committed", committed),
                               ("off", replace(GEN7_OVERLAY, profile=f"{module_name}:PROFILE"))):
            R.render(overlay, base / label, source=HeadSource())
            dirs[label] = base / label
        yield dirs
    finally:
        sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def eligible_run(render_dirs):
    with _mounted(render_dirs["on"]) as package:
        return _Scenario(package, roles=("Trork_Brawler", "Skeleton_Fighter")).run({EARLY: SWING, LATE: SWING}, 170)


def test_switch_off_leaves_the_gen7_render_identical_to_the_committed_render(render_dirs) -> None:
    requests = {EARLY: SWING, LATE: SWING}
    with _mounted(render_dirs["off"]) as package:
        off_final, off = _Scenario(package).run(requests, 64)
    with _mounted(render_dirs["committed"]) as package:
        committed_final, committed = _Scenario(package).run(requests, 64)
    # The two renders share every other construct (the shipped profile, replace_default_hop included) and differ only in the
    # hit-interrupt hooks and switch, so the hit steps must agree; their values are pinned by the timing tests below.
    assert len(_hit_steps(committed)) == 2 and _hit_steps(off) == _hit_steps(committed)
    for name in committed:
        assert np.array_equal(off[name], committed[name]), name
    off_leaves, committed_leaves = _leaves(off_final), _leaves(committed_final)
    extra = sorted(set(off_leaves) - set(committed_leaves))
    assert extra and all("hit_interrupt_eligible" in key for key in extra), extra
    assert not set(committed_leaves) - set(off_leaves)
    for key in extra:
        assert not off_leaves[key].any(), key  # the new state field is all False with the switch off
    for key, value in committed_leaves.items():
        assert off_leaves[key].dtype == value.dtype and off_leaves[key].shape == value.shape, key
        assert np.array_equal(off_leaves[key], value), key


def test_the_eligible_target_gets_eligibility_from_tick_42_and_refreshes_on_the_grid(eligible_run) -> None:
    final, trace = eligible_run
    assert np.asarray(final.mechanics.hit_interrupt_eligible)[0].tolist() == [False, True]
    assert _steps(trace, "Hit_Interrupt_Eligible")[0] == 42
    values = _remaining(trace, "Hit_Interrupt_Eligible")
    refreshes = [index + 1 for index in range(1, 70) if values[index - 1] > 0 and values[index] > values[index - 1]]
    assert refreshes == [45, 53, 60, 68]


def test_a_hit_before_tick_42_adds_nothing(eligible_run) -> None:
    _, trace = eligible_run
    assert _hit_steps(trace)[0] == 10  # the swing requested on tick 2 lands on tick 10 (live rule: request + 8)
    for name in ("Hit_Interrupt_Stunned", "Hit_Interrupt_Flinch", "Hit_Interrupt_Recent_Reaction_A"):
        assert all(step >= 42 for step in _steps(trace, name)), name


def test_the_first_hit_at_or_after_tick_42_adds_the_three_statuses_in_one_tick(eligible_run) -> None:
    _, trace = eligible_run
    late = _hit_steps(trace)[1]
    assert late == 48
    for name in ("Hit_Interrupt_Stunned", "Hit_Interrupt_Flinch", "Hit_Interrupt_Recent_Reaction_A"):
        assert late in _steps(trace, name) and late - 1 not in _steps(trace, name), name
    for name in ("Hit_Interrupt_Recent_Reaction_B", "Hit_Interrupt_Recovery_Pending", "Hit_Interrupt_Enraged"):
        assert _steps(trace, name) == [], name


def test_the_statuses_are_removed_after_15_21_and_120_ticks(eligible_run) -> None:
    _, trace = eligible_run
    assert _steps(trace, "Hit_Interrupt_Flinch") == list(range(48, 63))  # removed on step 63: 15 ticks after the hit
    assert _steps(trace, "Hit_Interrupt_Stunned") == list(range(48, 69))  # removed on step 69: 21 ticks
    assert _steps(trace, "Hit_Interrupt_Recent_Reaction_A") == list(range(48, 168))  # removed on step 168: 120 ticks


def test_the_flinch_window_freezes_the_target_motion_on_the_render(eligible_run) -> None:
    """Recorded, not corrected: on the render the flinch's speed multiplier 0 scales the target's whole position change and
    velocity (arsenal/runtime.py _apply_target_controls). The hit's knockback is still moving the target on steps 49 to 53;
    from step 54 to the last flinch step 62 it is still, and it moves again on step 63, when the flinch is gone. The asset
    comment keeps gravity and knockback active, so this freeze is a deviation (sim/gen7/mechanics/hit_interrupt.py, module
    docstring). DISABLE_MOVEMENT alone does not freeze the target (a flag-only run moved on step 1)."""

    _, trace = eligible_run
    pos = trace["pos"]
    assert np.all(pos[53:62] == pos[53])  # the states after steps 54 to 62
    assert not np.array_equal(pos[62], pos[61])  # moving again after the flinch is removed on step 63


def test_a_non_eligible_role_is_never_affected(render_dirs) -> None:
    with _mounted(render_dirs["on"]) as package:
        final, trace = _Scenario(package, roles=("Trork_Brawler", "Trork_Brawler")).run({EARLY: SWING, LATE: SWING}, 60)
    assert np.asarray(final.mechanics.hit_interrupt_eligible)[0].tolist() == [False, False]
    assert _hit_steps(trace)[:2] == [10, 48]  # the swings still land
    for name in ("Hit_Interrupt_Eligible", "Hit_Interrupt_Stunned", "Hit_Interrupt_Flinch", "Hit_Interrupt_Recent_Reaction_A"):
        assert _steps(trace, name) == [], name
