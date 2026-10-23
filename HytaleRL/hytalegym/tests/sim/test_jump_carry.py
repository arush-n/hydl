"""Jump carry (gen 7, row 1c; owner ruling 2026-10-22): a running jump keeps its take-off velocity, falls by the engine's air
drag, slides on the ground and walks at full speed on L+5. Construct: sim/gen7/mechanics/npc_jump_carry.py. A standing jump
(zero take-off speed) keeps the L+3 hold of landing_walk_resume.

The live list is the velocity field of the receipt knockback_landing_jump_Kweebec_Razorleaf_carry_DBB3736B.json (bridge
DBB3736B with the momentum change; learner Kweebec_Razorleaf, forward run held, a jump request every 30 ticks; two seeds, five
jumps each, all eight jumps identical tick by tick). Jump request on tick 40, touchdown on L = 50.
"""

from __future__ import annotations

import math
from dataclasses import replace
from importlib import import_module
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import npc_jump_carry
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, unmount

SEED = 0
#: The gait run factor the live probes hold (run, forward).
RUN_FORWARD = 9
#: Blocks per second; the receipt's velocity field is read to three decimals, so the bar is 0.02 per tick.
TOLERANCE = 0.02
#: Horizontal speed by tick for the jump requested on tick 40 (touchdown L = 50, walk on L+5).
LIVE = {40: 5.000, 41: 4.608, 42: 4.247, 43: 3.914, 44: 3.607, 45: 3.324, 46: 3.064, 47: 2.823, 48: 2.602, 49: 2.398,
        50: 2.210, 51: 1.486, 52: 0.999, 53: 0.671, 54: 0.203, 55: 5.000}
MOVING = 1.0e-3


class _Learner:
    """The single-agent arsenal environment on a mounted render: the iron-sword learner on the npc_walk body, no target."""

    def __init__(self, package: str) -> None:
        combat = import_module(package + ".combat")
        arsenal = import_module(package + ".combat.arsenal")
        environment = import_module(package + ".combat.arsenal.environment")
        layout = import_module(package + ".combat.observation.v3.policy.layout")
        params = combat.default_combat_params(microticks=1, target_active=False, motion_model="npc_walk")
        config = arsenal.arsenal_runtime_config(arsenal.hytale_0_5_7_loadouts(["iron_sword"]))
        self.env = environment.make_arsenal_environment(params, config, maximum_turn_degrees=45.0)
        names = list(layout.ARSENAL_POLICY_ACTION_HEAD_NAMES)
        self.neutral = layout.neutral_arsenal_policy_action_factors(1)
        self.forward = self.neutral.at[:, names.index("locomotion_gait_compass")].set(RUN_FORWARD)
        self.jump = self.forward.at[:, names.index("jump_off_on")].set(1)
        self.step = jax.jit(self.env.step_factors)

    def run(self, jump_ticks: set[int], steps: int, knock_ticks: frozenset = frozenset()) -> list[dict]:
        """Rows 0..steps: the state after each tick; forward is held throughout and a jump is requested on ``jump_ticks``.

        After each tick in ``knock_ticks`` the state is hit (:func:`_knock`) before its row is recorded.
        """

        carry, _ = self.env.reset(jax.random.split(jax.random.key(SEED), 1))
        rows = [self._row(carry)]
        for tick in range(1, steps + 1):
            factors = self.jump if tick in jump_ticks else self.forward
            carry, _, _, _, _ = self.step(carry, factors, jax.random.split(jax.random.key(1000 * SEED + tick), 1))
            if tick in knock_ticks:
                carry = _knock(carry)
            rows.append(self._row(carry))
        return rows

    @staticmethod
    def _row(carry) -> dict:
        combat = carry.runtime.combat
        velocity = np.asarray(combat.velocity)[0, 0]
        return {
            "pos": np.asarray(combat.position)[0, 0].tolist(),
            "hs": math.hypot(float(velocity[0]), float(velocity[2])),
            "grounded": bool(np.asarray(combat.agent_grounded)[0]),
            "count": int(np.asarray(combat.agent_jump_carry_ticks)[0]),
        }


def _knock(carry):
    """The state a hit leaves behind: the knockback control lock set and a horizontal knockback force of 6 blocks/s along -z."""

    combat = carry.runtime.combat
    force = np.asarray(combat.agent_force_velocity).copy()
    force[0] = (0.0, 0.0, -6.0)
    combat = combat._replace(knockback_control_lock=jnp.asarray([True]), agent_force_velocity=jnp.asarray(force))
    return carry._replace(runtime=carry.runtime._replace(combat=combat))


def _render_rows(overlay, base, label: str, jump_ticks: set[int], steps: int, knock_ticks: frozenset = frozenset()) -> list[dict]:
    R.render(overlay, base / label, source=HeadSource())
    package = mount_fresh(base / label).__name__
    try:
        return _Learner(package).run(jump_ticks, steps, knock_ticks)
    finally:
        unmount(package)


def _planar(rows: list[dict], tick: int) -> float:
    a, b = rows[tick - 1]["pos"], rows[tick]["pos"]
    return math.hypot(b[0] - a[0], b[2] - a[2])


def _touchdowns(rows: list[dict]) -> list[int]:
    return [t for t in range(1, len(rows)) if rows[t]["grounded"] and not rows[t - 1]["grounded"]]


@pytest.fixture(scope="module")
def carried(tmp_path_factory) -> list[dict]:
    """Gen 7 (switch on): forward run from tick 1, jump requests on 10, 40 and 70."""

    return _render_rows(GEN7_OVERLAY, tmp_path_factory.mktemp("jump_carry_on"), "gen7", {10, 40, 70}, 100)


@pytest.fixture(scope="module")
def standing(tmp_path_factory) -> list[dict]:
    """Gen 7 (switch on): a jump requested on tick 1, before the body has any speed, then forward held."""

    return _render_rows(GEN7_OVERLAY, tmp_path_factory.mktemp("jump_carry_standing"), "gen7", {1}, 30)


@pytest.fixture(scope="module")
def knocked(tmp_path_factory) -> list[dict]:
    """Gen 7 (switch on): the running jump requested on tick 40 is hit in the air after tick 45 (see ``_knock``)."""

    return _render_rows(GEN7_OVERLAY, tmp_path_factory.mktemp("jump_carry_knock"), "gen7_knock", {40}, 60, frozenset({45}))


@pytest.fixture(scope="module")
def without_carry(tmp_path_factory) -> list[dict]:
    """Gen 7 with ``npc_jump_carry`` False: the arc before this construct (the running jump keeps 5.0 in the air)."""

    module_name = "jump_carry_off_profile"
    off_module = types.ModuleType(module_name)
    off_module.PROFILE = replace(GEN7, constructs=replace(GEN7.constructs, npc_jump_carry=False))
    sys.modules[module_name] = off_module
    try:
        overlay = replace(GEN7_OVERLAY, profile=f"{module_name}:PROFILE")
        return _render_rows(overlay, tmp_path_factory.mktemp("jump_carry_off"), "gen7_off", {10, 40, 70}, 100)
    finally:
        sys.modules.pop(module_name, None)


def test_a_running_jump_follows_the_live_speed_on_every_tick(carried) -> None:
    rows = carried
    for tick, live in LIVE.items():
        assert abs(rows[tick]["hs"] - live) <= TOLERANCE, (tick, rows[tick]["hs"], live)
    assert _touchdowns(rows)[:2] == [20, 50]  # the first jump lands on 20 (its own take-off), the second on L = 50
    assert [rows[t]["grounded"] for t in range(40, 50)] == [False] * 10
    assert all(rows[t]["grounded"] for t in range(50, 56))


def test_the_next_running_jump_repeats_the_same_list(carried) -> None:
    rows = carried
    for k in range(16):
        assert abs(rows[70 + k]["hs"] - LIVE[40 + k]) <= TOLERANCE, (k, rows[70 + k]["hs"], LIVE[40 + k])


def test_the_walk_resumes_at_full_speed_on_the_fifth_tick_after_touchdown(carried) -> None:
    rows = carried
    assert rows[55]["hs"] == pytest.approx(5.0, abs=TOLERANCE)
    assert rows[54]["hs"] < 0.25  # the last slide tick is 0.203, not a walk
    assert _planar(rows, 55) == pytest.approx(5.0 / 30.0, abs=1.0e-3)


def test_a_standing_jump_still_walks_on_the_third_tick_after_touchdown(standing) -> None:
    rows = standing
    landing = _touchdowns(rows)[0]
    assert _planar(rows, landing + 1) == 0.0 and _planar(rows, landing + 2) == 0.0  # the two-tick hold
    assert next(k for k in range(1, 8) if _planar(rows, landing + k) > MOVING) == 3


def test_the_switch_off_is_the_arc_before_the_construct(without_carry) -> None:
    rows = without_carry
    # Before this construct the body keeps its run speed in the air, and the landing walk resumes on L+3 from rest.
    assert all(abs(rows[t]["hs"] - 5.0) <= 1.0e-6 for t in range(41, 50))
    landing = _touchdowns(rows)[1]
    assert rows[landing + 1]["hs"] == pytest.approx(0.0, abs=1.0e-6)
    assert rows[landing + 3]["hs"] > 0.0


def test_a_knockback_during_the_carry_ends_the_carry(knocked) -> None:
    rows = knocked
    assert rows[45]["count"] == 1  # still airborne in the carry when the hit lands
    assert abs(rows[46]["hs"] - 6.0) <= 1.0e-6  # the knockback's own force is the velocity of the next tick
    assert all(rows[t]["count"] == 0 for t in range(46, 61))  # the carry is over: no tail, no release
    assert rows[55]["hs"] < 1.0  # the knockback decays on its own; the walk does not take over on L+5


def test_gen5_and_the_default_keep_the_construct_off() -> None:
    assert Constructs().npc_jump_carry is False and GEN5.constructs.npc_jump_carry is False
    sentinel = object()
    assert npc_jump_carry.construct(sentinel, GEN5) is sentinel
