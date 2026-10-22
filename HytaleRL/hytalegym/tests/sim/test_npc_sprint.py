"""NPC sprint is run (gen 7, row 1a; ``sim/gen7/mechanics/npc_sprint.py``).

Logic tests run the gait rewrite and both wrappers on stand-ins. Render tests mount the gen-7 render with the switch on, off,
and the committed overlay (the two hooks removed), and drive the engine's own ``step_batch`` on npc_walk and player params:
one sprint lane and one run lane per direction, the pair sharing its reset and step keys.

Expected values come from the ruleset and the gait table: npc_walk MaxSpeed 5.0 (``rulesets/hytale_0_5_7`` agent
``max_speed``) times the run multipliers (forward 1.0, back 0.65, strafe 0.8); the player model 5.5 times the sprint 1.273.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat import ACTION_SIZE, default_combat_params
from hytalegym.jax.combat.types import (
    ACTION_BACK,
    ACTION_FORWARD,
    ACTION_GAIT,
    ACTION_LEFT,
    ACTION_RIGHT,
    AGENT_ENTITY,
    AGENT_MOTION_NPC_WALK,
    AGENT_MOTION_PLAYER,
    GAIT_RUN,
    GAIT_SPRINT,
    PLAYER_STAMINA_MAXIMUM,
)
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import npc_sprint
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, unmount

SWITCH = "npc_walk_sprint_as_run"
HOOK_IDS = ("npc_walk_sprint_learner",)
SEED = 570057
TICKS = 40
#: (forward, back, left, right) per direction; the diagonal is forward and right.
DIRECTIONS = {"forward": (1, 0, 0, 0), "back": (0, 1, 0, 0), "left": (0, 0, 1, 0), "right": (0, 0, 0, 1),
              "diagonal": (1, 0, 0, 1)}
#: Steady horizontal speed of a run on npc_walk (blocks per second): 5.0 x the axis multiplier; the diagonal is hypot(5 x 0.707, 4 x 0.707).
NPC_RUN_SPEED = {"forward": 5.0, "back": 3.25, "left": 4.0, "right": 4.0, "diagonal": 4.528}
SPRINT_FACTOR = 1.273
PLAYER_BASE_SPEED = 5.5


# ---- logic tests: the rewrite and the two wrappers on stand-ins ----------------------------------------------------------


def _params(npc: bool) -> Any:
    return types.SimpleNamespace(agent_motion_model=jnp.int32(AGENT_MOTION_NPC_WALK if npc else AGENT_MOTION_PLAYER))


def _profile(on: bool) -> Any:
    return types.SimpleNamespace(constructs=types.SimpleNamespace(npc_walk_sprint_as_run=on))


def _actions(gaits) -> jnp.ndarray:
    values = np.zeros((len(gaits), ACTION_SIZE), np.float32)
    values[:, ACTION_FORWARD] = 0.25
    values[:, ACTION_GAIT] = gaits
    return jnp.asarray(values)


def test_a_sprint_request_becomes_a_run_only_under_npc_walk_and_nothing_else_changes() -> None:
    gaits = [3.0, 2.6, 2.4, 2.0, 4.0, 0.0, 1.0]
    actions = _actions(gaits)
    npc = np.asarray(npc_sprint.sprint_as_run(actions, _params(npc=True)))
    assert list(npc[:, ACTION_GAIT]) == [2.0, 2.0, 2.4, 2.0, 4.0, 0.0, 1.0]  # 3.0 and 2.6 round to sprint (3); 2.4 rounds to run (2) and is left as it is
    other = [column for column in range(ACTION_SIZE) if column != ACTION_GAIT]
    assert np.array_equal(npc[:, other], np.asarray(actions)[:, other])
    player = npc_sprint.sprint_as_run(actions, _params(npc=False))
    assert np.array_equal(np.asarray(player), np.asarray(actions))


def test_the_learner_wrapper_hands_the_engine_a_run_request() -> None:
    seen = {}

    def original(state, actions, pause_draw, params, *rest, **kwargs):
        seen["gait"] = np.asarray(actions[:, ACTION_GAIT])
        seen["rest"] = (pause_draw, rest, kwargs)
        return "updated", "attack", "accepted"

    wrapper = npc_sprint.construct(original, _profile(True))
    result = wrapper("state", _actions([3.0, 3.0]), "pause", _params(npc=True), "geometry", agent_loadout=None)
    assert result == ("updated", "attack", "accepted")
    assert list(seen["gait"]) == [2.0, 2.0]
    assert seen["rest"] == ("pause", ("geometry",), {"agent_loadout": None})
    npc_sprint.construct(original, _profile(True))("state", _actions([3.0]), "pause", _params(npc=False))
    assert list(seen["gait"]) == [3.0]


def test_gen5_and_the_switch_off_return_the_original_function() -> None:
    sentinel = object()
    assert npc_sprint.construct(sentinel, GEN5) is sentinel
    assert npc_sprint.construct(sentinel, _profile(False)) is sentinel


def test_the_switch_is_off_in_gen5_and_on_in_gen7_and_the_module_is_implemented() -> None:
    assert getattr(Constructs(), SWITCH) is False
    assert getattr(GEN5.constructs, SWITCH) is False
    assert getattr(GEN7.constructs, SWITCH) is True
    assert npc_sprint.SWITCH == SWITCH and npc_sprint.IMPLEMENTED is True


def test_the_overlay_declares_the_learner_hook_on_the_engine_function() -> None:
    hooks = [hook for hook in GEN7_OVERLAY.hooks if hook.id in HOOK_IDS]
    assert [hook.id for hook in hooks] == list(HOOK_IDS)
    learner = hooks[0]
    assert (learner.path, learner.function, learner.construct, learner.switch) == (
        "combat/runtime/step.py", "_apply_action", "hytalegym.sim.gen7.mechanics.npc_sprint:construct", SWITCH)
    assert learner.status == "active" and learner.fidelity_row == "1a"
    ids = [hook.id for hook in GEN7_OVERLAY.hooks]
    assert ids.index("npc_walk_sprint_learner") < ids.index("landing_walk_resume")  # landing stays innermost


def test_the_default_params_carry_the_npc_speed_and_the_motion_model() -> None:
    npc = default_combat_params(microticks=1, motion_model="npc_walk")
    player = default_combat_params(microticks=1, motion_model="player")
    assert float(npc.agent_max_speed) == pytest.approx(5.0)
    assert int(npc.agent_motion_model) == AGENT_MOTION_NPC_WALK
    assert float(player.agent_max_speed) == pytest.approx(PLAYER_BASE_SPEED)
    assert int(player.agent_motion_model) == AGENT_MOTION_PLAYER


# ---- render tests: the gen-7 render with the switch on, off, and the committed overlay (the two hooks removed) --------


@contextmanager
def _mounted(path: Path) -> Iterator[str]:
    package = mount_fresh(path).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.fixture(scope="module")
def render_dirs(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("npc_sprint_render")
    committed = replace(GEN7_OVERLAY, hooks=tuple(hook for hook in GEN7_OVERLAY.hooks if hook.id not in HOOK_IDS))
    module_name = "npc_sprint_test_off_profile"
    off_module = types.ModuleType(module_name)
    off_module.PROFILE = replace(GEN7, constructs=replace(GEN7.constructs, npc_walk_sprint_as_run=False))
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


def _leaves(tree: Any) -> dict[str, np.ndarray]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def _drive(package: str, motion_model: str) -> dict[str, Any]:
    """Lanes 2i (sprint) and 2i+1 (run) for direction i; the pair shares its reset and step keys."""

    combat = import_module(package + ".combat")
    params = combat.default_combat_params(microticks=1, target_active=True, motion_model=motion_model)
    count = len(DIRECTIONS)
    lanes = 2 * count
    pair = np.arange(lanes) // 2
    state, _ = combat.reset_batch(jax.random.split(jax.random.key(SEED), count)[pair], params)
    action = np.zeros((lanes, ACTION_SIZE), np.float32)
    for index, (forward, back, left, right) in enumerate(DIRECTIONS.values()):
        action[2 * index: 2 * index + 2, [ACTION_FORWARD, ACTION_BACK, ACTION_LEFT, ACTION_RIGHT]] = (forward, back, left, right)
        action[2 * index, ACTION_GAIT] = GAIT_SPRINT
        action[2 * index + 1, ACTION_GAIT] = GAIT_RUN
    action = jnp.asarray(action)
    step_keys = jax.random.split(jax.random.key(SEED + 1), TICKS * count).reshape(TICKS, count)[:, pair]
    step = jax.jit(combat.step_batch)
    rows: dict[str, list] = {"velocity": [], "position": [], "stamina": []}
    for tick in range(TICKS):
        state, _, _, _, _ = step(state, action, step_keys[tick], params)
        rows["velocity"].append(np.asarray(state.velocity[:, AGENT_ENTITY]))
        rows["position"].append(np.asarray(state.position[:, AGENT_ENTITY]))
        rows["stamina"].append(np.asarray(state.agent_locomotion_stamina))
    trace = {name: np.stack(values) for name, values in rows.items()}
    trace["speed"] = np.hypot(trace["velocity"][..., 0], trace["velocity"][..., 2])
    trace["final"] = _leaves(state)
    return trace


@pytest.fixture(scope="module")
def traces(render_dirs) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for label in ("on", "committed", "off"):
        with _mounted(render_dirs[label]) as package:
            for model in ("npc_walk", "player"):
                out[(label, model)] = _drive(package, model)
    return out


def _direction(index: int) -> tuple[str, int, int]:
    name = list(DIRECTIONS)[index]
    return name, 2 * index, 2 * index + 1  # (name, sprint lane, run lane)


def test_on_npc_walk_every_sprint_lane_moves_and_stamina_exactly_like_its_run_lane(traces) -> None:
    trace = traces[("on", "npc_walk")]
    for index in range(len(DIRECTIONS)):
        name, sprint, run = _direction(index)
        for key in ("velocity", "position", "stamina", "speed"):
            assert np.array_equal(trace[key][:, sprint], trace[key][:, run]), (name, key)


def test_on_npc_walk_the_sprint_and_run_lanes_match_the_committed_run_lanes_bit_for_bit(traces) -> None:
    on, committed = traces[("on", "npc_walk")], traces[("committed", "npc_walk")]
    for index in range(len(DIRECTIONS)):
        name, sprint, run = _direction(index)
        for key in ("velocity", "position", "stamina"):
            assert np.array_equal(on[key][:, sprint], committed[key][:, run]), (name, "sprint", key)
            assert np.array_equal(on[key][:, run], committed[key][:, run]), (name, "run", key)


def test_on_npc_walk_the_steady_speed_is_the_run_speed_on_every_axis(traces) -> None:
    speed = traces[("on", "npc_walk")]["speed"]
    for index in range(len(DIRECTIONS)):
        name, sprint, _ = _direction(index)
        assert speed[-1, sprint] == pytest.approx(NPC_RUN_SPEED[name], abs=2e-3), name


def test_on_npc_walk_every_sprint_lane_ends_in_the_committed_run_state_field_for_field(traces) -> None:
    """Stronger than the trajectories: every batched state leaf of a sprint lane equals the committed run lane's."""

    on, committed = traces[("on", "npc_walk")]["final"], traces[("committed", "npc_walk")]["final"]
    assert set(on) == set(committed)
    lanes = 2 * len(DIRECTIONS)
    compared = 0
    for key, leaf in on.items():
        if leaf.ndim == 0 or leaf.shape[0] != lanes:
            continue
        for index in range(len(DIRECTIONS)):
            name, sprint, run = _direction(index)
            assert np.array_equal(leaf[sprint], committed[key][run]), (name, key)
            compared += 1
    assert compared > 20, compared


def test_on_npc_walk_sprint_does_not_drain_stamina(traces) -> None:
    stamina = traces[("on", "npc_walk")]["stamina"]
    for index in range(len(DIRECTIONS)):
        _, sprint, _ = _direction(index)
        assert np.all(stamina[:, sprint] == PLAYER_STAMINA_MAXIMUM)


def test_switch_off_npc_walk_sprint_is_the_stock_sprint_and_drains_stamina(traces) -> None:
    off = traces[("off", "npc_walk")]
    committed = traces[("committed", "npc_walk")]
    for key in ("velocity", "position", "stamina", "speed"):
        assert np.array_equal(off[key], committed[key]), key
    forward_sprint = off["speed"][-1, 0]
    assert forward_sprint == pytest.approx(5.0 * SPRINT_FACTOR, abs=2e-3)
    assert off["stamina"][:, 0].min() < PLAYER_STAMINA_MAXIMUM


def test_player_model_sprint_is_unchanged_by_the_construct(traces) -> None:
    on, committed = traces[("on", "player")], traces[("committed", "player")]
    for key in ("velocity", "position", "stamina", "speed"):
        assert np.array_equal(on[key], committed[key]), key
    _, sprint, _ = _direction(0)
    assert on["speed"][-1, sprint] == pytest.approx(PLAYER_BASE_SPEED * SPRINT_FACTOR, abs=2e-3)

