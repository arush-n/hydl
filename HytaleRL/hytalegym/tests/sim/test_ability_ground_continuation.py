"""Ability ground continuation (gen 7, row 5h; ``sim/gen7/mechanics/ability_ground_continuation.py``).

Two kinds of test. The logic test runs the construct's wrapper on a small state tuple with a stub original, so the gate
(the child starts only when the parent finished this tick) and the fields it sets are checked directly. The render tests mount
the gen-7 render (switch on, switch off, and the committed overlay without the row) and drive the engine's own step with the
charged battleaxe, the charged sword and the charged mace, each held on slot 0 against a still iron-sword target.

Measured on the render (scratch lane_119f1fix, 2026-10-21): the Downstrike (slot 4) starts on step 23 and parks in the air;
its ground continuation resolves on step 37 (active -1, continuation slot 5), the tick the leap lands; the Landing (slot 5)
starts on step 37 with scheduler tick 0 and hits for 29 on step 40. Steps are 1-based, one tick is 1/30 s.
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import ability_ground_continuation as agc
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, render_for_test, unmount

AGENT, TARGET = 0, 1
SEED = 570057
HOLD = 30  # slot 0 requested on steps 1 to 30
STEPS = 60
DISTANCES = (1.0, 1.6, 1.7, 2.0, 3.0)  # the target's distance in front of the agent (the agent faces -z)
LANDING = 5
DOWNSTRIKE = 4
RESUME_STEP = 37  # the Downstrike's ground resume (active -1, continuation slot 5) on the render
LANDING_HIT_STEP = 40  # the Landing's first damage tick on the render, three steps after the resume
LANDING_DAMAGE = 29.0


class _State(NamedTuple):
    """The arsenal fields the wrapper reads and writes (a stand-in for ``ArsenalState``)."""

    active_ability_slot: Any
    active_ability_root_slot: Any
    ability_elapsed_seconds: Any
    ability_scheduler_tick: Any
    ability_scheduler_clock_seconds: Any
    ability_selector_hit_bits: Any
    ability_continuation_slot: Any


def _state(active, root, cont, tick):
    """Batch 1, two entities: the agent (entity 0) carries the given values; the target (entity 1) carries none."""

    def row(agent, target):
        return jnp.asarray([[agent, target]], jnp.int32)

    return _State(
        active_ability_slot=row(active, -1),
        active_ability_root_slot=row(root, -1),
        ability_elapsed_seconds=jnp.asarray([[0.4, 0.7]], jnp.float32),
        ability_scheduler_tick=row(tick, 0),
        ability_scheduler_clock_seconds=jnp.full((1, 2, 3), 0.25, jnp.float32),
        ability_selector_hit_bits=jnp.asarray([[5, 9]], jnp.uint32),
        ability_continuation_slot=row(cont, -1),
    )


#: The loadout reads only the scheduler prelude: slot 5 carries 2 prelude ticks, the other slots none.
_LOADOUT = types.SimpleNamespace(ability_scheduler_prelude_ticks=jnp.asarray([[[0, 0, 0, 0, 0, 2, 0, 0]] * 2], jnp.int32))


def _stub(after: _State):
    """A stand-in for ``tick_ability_programs``: returns the given post-tick state and no fired events."""

    def original(state, loadout, *args, **kwargs):
        return after, "fired"

    return original


def test_the_switch_is_off_in_gen5_and_on_in_gen7_and_the_construct_passes_through_when_off() -> None:
    assert Constructs().ability_ground_continuation is False
    assert GEN5.constructs.ability_ground_continuation is False
    assert GEN7.constructs.ability_ground_continuation is True
    assert agc.IMPLEMENTED is True
    sentinel = object()
    assert agc.construct(sentinel, GEN5) is sentinel


def test_the_resume_on_the_finishing_tick_starts_the_child_with_its_prelude_and_keeps_the_requesting_root() -> None:
    before_state = _state(active=DOWNSTRIKE, root=0, cont=-1, tick=12)
    after = _state(active=-1, root=-1, cont=LANDING, tick=0)
    wrapper = agc.construct(_stub(after), GEN7)
    candidate, fired = wrapper(before_state, _LOADOUT, 0.0)
    assert fired == "fired"
    assert int(candidate.active_ability_slot[0, 0]) == LANDING
    assert int(candidate.active_ability_root_slot[0, 0]) == 0  # the requesting root, from before the tick
    assert int(candidate.ability_scheduler_tick[0, 0]) == -2  # minus the child's prelude (slot 5)
    assert float(candidate.ability_elapsed_seconds[0, 0]) == 0.0 and float(candidate.ability_elapsed_seconds[0, 1]) == pytest.approx(0.7)
    clock = np.asarray(candidate.ability_scheduler_clock_seconds)
    assert np.all(clock[0, 0] == 0.0) and np.all(clock[0, 1] == 0.25)  # the target's clock is not touched
    assert int(candidate.ability_selector_hit_bits[0, 0]) == 0 and int(candidate.ability_selector_hit_bits[0, 1]) == 9
    assert int(candidate.ability_continuation_slot[0, 0]) == -1  # the resume is consumed


def test_a_premature_resume_while_the_parent_is_still_active_starts_nothing() -> None:
    """The engine's ground exit reads the previous tick's grounded flag and has no finished term (continuation.py 150):
    slot 5 shows while the Downstrike is still starting. The construct waits for the parent to finish."""

    before_state = _state(active=DOWNSTRIKE, root=0, cont=-1, tick=3)
    after = _state(active=DOWNSTRIKE, root=0, cont=LANDING, tick=4)
    candidate, _ = agc.construct(_stub(after), GEN7)(before_state, _LOADOUT, 0.0)
    assert int(candidate.active_ability_slot[0, 0]) == DOWNSTRIKE
    assert int(candidate.ability_continuation_slot[0, 0]) == LANDING
    assert int(candidate.ability_scheduler_tick[0, 0]) == 4


def test_no_continuation_leaves_the_post_tick_state_alone() -> None:
    before_state = _state(active=0, root=0, cont=-1, tick=7)
    after = _state(active=0, root=0, cont=-1, tick=8)
    candidate, _ = agc.construct(_stub(after), GEN7)(before_state, _LOADOUT, 0.0)
    assert int(candidate.active_ability_slot[0, 0]) == 0 and int(candidate.ability_scheduler_tick[0, 0]) == 8
    assert int(candidate.ability_continuation_slot[0, 0]) == -1


# ---- render tests: the gen-7 render with the switch on, off, and the committed overlay (no row 5h) -------------------------


class _Scenario:
    """The arsenal step on the mounted render: the agent holds slot 0 on steps 1 to 30; the target stands still."""

    def __init__(self, package: str, weapon: str, distances) -> None:
        self.combat = import_module(package + ".combat")
        self.arsenal = import_module(package + ".combat.arsenal")
        environment = import_module(package + ".combat.arsenal.environment")
        self.params = self.combat.default_combat_params(microticks=1, target_active=True)
        self.batch = len(distances)
        loadout = self.arsenal.hytale_0_5_7_loadouts([weapon] * self.batch, target_profiles=["iron_sword"] * self.batch)
        self.config = self.arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=self.params.sensor_range)
        position = np.zeros((self.batch, 2, 3), np.float32)
        position[:, TARGET, 2] = -np.asarray(distances, np.float32)
        health = jnp.tile(jnp.asarray([[self.params.agent_max_health, self.params.target_max_health]], jnp.float32),
                          (self.batch, 1))
        self.initial, _ = self.arsenal.reset_arsenal_batch(
            jax.random.split(jax.random.key(SEED), self.batch), self.params, self.config,
            entity_position=jnp.asarray(position), entity_health=health,
            entity_yaw=jnp.zeros((self.batch, 2), jnp.float32))
        self.world = environment.open_flat_arsenal_world_capabilities(self.initial, self.params, config=self.config)

    def run(self) -> dict[str, np.ndarray]:
        slots = np.full((STEPS, self.batch), -1, np.int32)
        slots[:HOLD] = 0
        keys = jax.random.split(jax.random.key(161803), STEPS * self.batch).reshape(STEPS, self.batch)
        action = jnp.zeros((self.batch, self.combat.ACTION_SIZE), jnp.float32)

        def body(state, xs):
            slot, key = xs
            command = self.arsenal.empty_arsenal_commands(self.batch)
            command = command._replace(ability_slot=command.ability_slot.at[:, AGENT].set(slot), world=self.world)
            nxt = self.arsenal.step_arsenal_batch(state, action, key, self.params, command, self.config).state
            arsenal, combat = nxt.arsenal, nxt.combat
            return nxt, {
                "act": arsenal.active_ability_slot[:, AGENT], "root": arsenal.active_ability_root_slot[:, AGENT],
                "cont": arsenal.ability_continuation_slot[:, AGENT], "tick": arsenal.ability_scheduler_tick[:, AGENT],
                "grounded": combat.agent_grounded, "agent": combat.position[:, AGENT],
                "target": combat.position[:, TARGET], "hp": combat.health[:, TARGET],
            }

        final, trace = jax.jit(lambda state, xs: jax.lax.scan(body, state, xs))(self.initial, (jnp.asarray(slots), keys))
        self.final = final
        return {name: np.asarray(value) for name, value in trace.items()}


def _hits(trace: dict[str, np.ndarray], lane: int) -> list[tuple[int, float]]:
    """(step, damage) of every health drop of the target on one lane."""

    hp = trace["hp"][:, lane]
    return [(step + 1, float(hp[step - 1] - hp[step])) for step in range(1, STEPS) if hp[step] < hp[step - 1]]


def _first_step(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) + 1 if hits.size else None


def _leaves(tree: Any) -> dict[str, np.ndarray]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


@contextmanager
def _mounted(path: Path) -> Iterator[str]:
    package = mount_fresh(path).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.fixture(scope="module")
def render_dirs(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("ability_ground_continuation_render")
    committed = replace(
        GEN7_OVERLAY,
        hooks=tuple(hook for hook in GEN7_OVERLAY.hooks if hook.id != "ability_ground_continuation"),
    )
    module_name = "ability_ground_continuation_test_off_profile"
    off_module = types.ModuleType(module_name)
    off_module.PROFILE = replace(GEN7, constructs=replace(GEN7.constructs, ability_ground_continuation=False))
    sys.modules[module_name] = off_module
    try:
        dirs = {}
        for label, overlay in (("on", GEN7_OVERLAY), ("committed", committed),
                               ("off", replace(GEN7_OVERLAY, profile=f"{module_name}:PROFILE"))):
            dirs[label] = render_for_test(overlay, base / label, source=HeadSource())
        yield dirs
    finally:
        sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def battleaxe(render_dirs) -> dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]]:
    """Per variant: the battleaxe trace (three distances) and the final state's leaves."""

    out = {}
    for label in ("on", "committed", "off"):
        with _mounted(render_dirs[label]) as package:
            scenario = _Scenario(package, "iron_battleaxe", DISTANCES)
            trace = scenario.run()
            out[label] = (trace, _leaves(scenario.final))
    return out


def test_switch_off_leaves_the_gen7_render_identical_to_the_committed_render(battleaxe) -> None:
    off_trace, off_leaves = battleaxe["off"]
    committed_trace, committed_leaves = battleaxe["committed"]
    for name in committed_trace:
        assert np.array_equal(off_trace[name], committed_trace[name]), name
    assert set(off_leaves) == set(committed_leaves)
    for key in committed_leaves:
        assert np.array_equal(off_leaves[key], committed_leaves[key]), key


def test_the_downstrike_parks_and_the_resume_is_seen_before_the_landing_starts(battleaxe) -> None:
    """Committed (the HEAD engine): the Downstrike starts on step 23, a premature slot 5 shows while it is still active,
    the resume with the Downstrike finished is on step 37, and the Landing never starts."""

    committed_trace, _ = battleaxe["committed"]
    lane = DISTANCES.index(3.0)
    act = committed_trace["act"][:, lane]
    cont = committed_trace["cont"][:, lane]
    assert _first_step(act == DOWNSTRIKE) == 23
    assert _first_step((act >= 0) & (cont >= 0)) == 23  # premature: the parent is still starting
    assert _first_step((act == -1) & (cont == LANDING)) == RESUME_STEP
    assert not np.any(act == LANDING)
    assert _hits(committed_trace, lane) == []


def test_the_landing_starts_on_the_resume_tick_and_hits_for_29_three_ticks_later(battleaxe) -> None:
    on_trace, _ = battleaxe["on"]
    lane = DISTANCES.index(3.0)
    act = on_trace["act"][:, lane]
    assert _first_step(act == LANDING) == RESUME_STEP
    assert int(on_trace["tick"][RESUME_STEP - 1, lane]) == 0  # the child's first tick is the next step
    assert _first_step(act == DOWNSTRIKE) == 23
    assert _hits(on_trace, lane) == [(LANDING_HIT_STEP, LANDING_DAMAGE)]
    assert LANDING_HIT_STEP - RESUME_STEP == 3


def test_the_leap_is_unchanged_and_the_construct_only_differs_from_the_resume_on(battleaxe) -> None:
    """Grounded flags, the attacker's position and the Downstrike's slot agree on every step with the construct on or off,
    so the airborne ticks (steps 27 to 35) are the same. The only fields that differ are the Landing's start, its root,
    the scheduler and the consumed continuation slot."""

    on_trace, _ = battleaxe["on"]
    committed_trace, _ = battleaxe["committed"]
    for name in ("grounded", "agent", "act", "target", "hp"):
        diff = np.any(np.asarray(on_trace[name] != committed_trace[name]).reshape(STEPS, -1), axis=1)
        first = int(np.flatnonzero(diff)[0]) + 1 if diff.any() else None
        assert first is None or first >= RESUME_STEP, (name, first)
    lane = DISTANCES.index(3.0)
    airborne = (committed_trace["grounded"][:, lane] == False) & (committed_trace["act"][:, lane] == DOWNSTRIKE)  # noqa: E712
    assert _first_step(airborne) == 27 and int(airborne.sum()) == 9
    assert np.array_equal(on_trace["agent"][:, lane], committed_trace["agent"][:, lane])


def test_targets_at_one_and_one_point_six_blocks_are_missed_and_the_landing_still_runs(battleaxe) -> None:
    """The Downstrike carries the attacker past a close target: the landing point is at -1.78 along the facing, and the
    Landing's forward selector (start 0.1, end 4) misses a target behind that point (measured: 1.0 and 1.6 blocks miss)."""

    on_trace, _ = battleaxe["on"]
    for distance in (1.0, 1.6):
        lane = DISTANCES.index(distance)
        assert _first_step(on_trace["act"][:, lane] == LANDING) == RESUME_STEP, distance
        assert _hits(on_trace, lane) == [], distance


def test_targets_from_one_point_seven_blocks_are_hit_for_29(battleaxe) -> None:
    """Measured: 1.7 blocks hits, 0.08 block behind the landing point. The hit boundary lies between 1.6 and 1.7 on this
    render; the reason (the target's hitbox against the selector) is inferred, not checked."""

    on_trace, _ = battleaxe["on"]
    for distance in (1.7, 2.0, 3.0):
        lane = DISTANCES.index(distance)
        assert _hits(on_trace, lane) == [(LANDING_HIT_STEP, LANDING_DAMAGE)], distance


@pytest.mark.parametrize("weapon", ["iron_sword", "iron_mace"])
def test_other_charged_holds_are_unchanged(render_dirs, weapon: str) -> None:
    """The charged sword thrust and the charged mace hit carry no ground continuation: their full traces equal the committed
    render's, so the construct changes nothing on them."""

    traces = {}
    for label in ("on", "committed"):
        with _mounted(render_dirs[label]) as package:
            traces[label] = _Scenario(package, weapon, DISTANCES).run()
    for name in traces["committed"]:
        assert np.array_equal(traces["on"][name], traces["committed"][name]), (weapon, name)
