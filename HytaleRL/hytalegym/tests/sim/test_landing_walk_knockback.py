"""Walk after a landing, by cause (gen 7, row 1c; rendered and run on the npc_walk body).

Live 0.7.2 (knockback_landing receipts, 2026-10-22, learner Kweebec_Razorleaf, Trork melee): a hit knocks the idle learner
up three airborne ticks; the touchdown tick L and L+1 do not move, and with forward held the body walks from L+2. A jump
landing walks from L+3 (construct landing_walk_resume), and a walk-off with no jump walks from L+1 (the construct's
docstring). The render reproduces the knockback timing without the construct: the one-tick hold after a hit is the engine's
knockback control lock (CombatState.knockback_control_lock: set by the hit, still set on L, cleared on L+1). The jump and
walk-off tests pin the construct's other two cases.

The opponent is the engine's legacy Trork melee: a target with no arsenal loadout. An arsenal-loaded target is controlled by
its own programs and never swings at this range (the coordinator's quick script, 2026-10-22).
"""

from __future__ import annotations

import math
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY

from .support import mount_fresh, unmount

SEED = 0
#: The gait run factor the live probes hold (run, forward).
RUN_FORWARD = 9
#: A planar displacement above this counts as walking (the live probe's MOVING_EPSILON).
MOVING = 1.0e-3


class _Learner:
    """The single-agent arsenal environment on a mounted render: the iron-sword learner on the npc_walk body."""

    def __init__(self, package: str, *, target_active: bool) -> None:
        combat = import_module(package + ".combat")
        arsenal = import_module(package + ".combat.arsenal")
        environment = import_module(package + ".combat.arsenal.environment")
        layout = import_module(package + ".combat.observation.v3.policy.layout")
        params = combat.default_combat_params(microticks=1, target_active=target_active, motion_model="npc_walk")
        config = arsenal.arsenal_runtime_config(arsenal.hytale_0_5_7_loadouts(["iron_sword"]))
        self.env = environment.make_arsenal_environment(params, config, maximum_turn_degrees=45.0)
        names = list(layout.ARSENAL_POLICY_ACTION_HEAD_NAMES)
        self.neutral = layout.neutral_arsenal_policy_action_factors(1)
        self.forward = self.neutral.at[:, names.index("locomotion_gait_compass")].set(RUN_FORWARD)
        self.jump = self.forward.at[:, names.index("jump_off_on")].set(1)
        self.step = jax.jit(self.env.step_factors)

    def reset(self):
        carry, _ = self.env.reset(jax.random.split(jax.random.key(SEED), 1))
        return carry

    def row(self, carry) -> dict:
        combat = carry.runtime.combat
        return {
            "pos": np.asarray(combat.position)[0, 0].tolist(),
            "hp": float(np.asarray(combat.health)[0, 0]),
            "grounded": bool(np.asarray(combat.agent_grounded)[0]),
            "lock": bool(np.asarray(combat.knockback_control_lock)[0]),
        }

    def tick(self, carry, factors, tick: int):
        carry, _, _, done, _ = self.step(carry, factors, jax.random.split(jax.random.key(1000 * SEED + tick), 1))
        return carry, bool(np.asarray(done).reshape(-1)[0])


def _planar(rows: list[dict], tick: int) -> float:
    """The planar displacement that arrives on ``tick`` (the live probe's definition)."""

    a, b = rows[tick - 1]["pos"], rows[tick]["pos"]
    return math.hypot(b[0] - a[0], b[2] - a[2])


def _touchdowns(rows: list[dict]) -> list[int]:
    return [t for t in range(1, len(rows)) if rows[t]["grounded"] and not rows[t - 1]["grounded"]]


def _walk_after(rows: list[dict], landing: int) -> int:
    """k >= 1: the walk starts k ticks after touchdown (the first tick after L that moves)."""

    return next(k for k in range(1, len(rows) - landing) if _planar(rows, landing + k) > MOVING)


def _lift(carry, height: float):
    combat = carry.runtime.combat
    position = np.asarray(combat.position).copy()
    position[0, 0, 1] += height
    return carry._replace(runtime=carry.runtime._replace(combat=combat._replace(position=jnp.asarray(position))))


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    path = tmp_path_factory.mktemp("landing_walk_knockback") / "gen7"
    R.render(GEN7_OVERLAY, path, source=HeadSource())
    name = mount_fresh(path).__name__
    yield name
    unmount(name)


@pytest.fixture(scope="module")
def hit_rows(package):
    """The live probe's closed loop: idle until hit; hit n sets forward (n odd) or idle (n even) until 8 ticks after touchdown."""

    learner = _Learner(package, target_active=True)
    carry = learner.reset()
    rows, arms = [learner.row(carry)], [None]
    hits, arm, release, airborne = 0, None, None, False
    for tick in range(1, 171):  # hits at 58, 103, 161: forward landings at 61 and 164
        factors = learner.forward if arm == "forward" else learner.neutral
        carry, done = learner.tick(carry, factors, tick)
        rows.append(learner.row(carry))
        arms.append(arm)
        if rows[-1]["hp"] < rows[-2]["hp"]:
            hits, arm, release = hits + 1, ("forward" if (hits + 1) % 2 else "idle"), None
        if arm is not None and not rows[-1]["grounded"]:
            airborne = True
        if arm is not None and airborne and rows[-1]["grounded"] and release is None:
            release, airborne = tick + 8, False
        if release is not None and tick >= release:
            arm, release = None, None
        if done:
            break
    return rows, arms


def test_a_knockback_landing_walks_on_the_second_tick_after_touchdown(hit_rows) -> None:
    rows, arms = hit_rows
    hit = next(t for t in range(1, len(rows)) if rows[t]["hp"] < rows[t - 1]["hp"])
    landings = _touchdowns(rows)
    forward = [t for t in landings if arms[t] == "forward"]
    idle = [t for t in landings if arms[t] == "idle"]
    assert len(forward) >= 2 and idle, (landings, arms)
    assert landings[0] - hit == 3, (hit, landings)  # three airborne ticks after the hit
    for landing in forward[:2]:
        assert _planar(rows, landing) == 0.0  # no planar move in the air
        assert rows[landing]["lock"] and not rows[landing + 1]["lock"]  # the knockback control lock: held on L, cleared on L+1
        assert _planar(rows, landing + 1) == 0.0  # the one-tick hold
        assert _walk_after(rows, landing) == 2
    for landing in idle[:1]:
        assert all(_planar(rows, t) == 0.0 for t in range(landing, landing + 8))  # idle: no sliding after touchdown


def test_a_jump_landing_still_walks_on_the_third_tick_after_touchdown(package) -> None:
    learner = _Learner(package, target_active=False)
    carry = learner.reset()
    rows = [learner.row(carry)]
    for tick in range(1, 31):
        carry, _ = learner.tick(carry, learner.jump if tick % 30 == 10 else learner.forward, tick)
        rows.append(learner.row(carry))
    landing = _touchdowns(rows)[0]
    assert _planar(rows, landing + 1) == 0.0 and _planar(rows, landing + 2) == 0.0  # the two-tick hold (the construct)
    assert _walk_after(rows, landing) == 3


def test_a_walk_off_landing_without_a_jump_walks_on_the_first_tick_after_touchdown(package) -> None:
    learner = _Learner(package, target_active=False)
    carry = learner.reset()
    rows = [learner.row(carry)]
    for tick in range(1, 31):
        if tick == 4:
            # Walk-off stand-in: still grounded at the start of this step, lifted 1.5 blocks; no jump and no hit.
            carry = _lift(carry, 1.5)
        carry, _ = learner.tick(carry, learner.forward, tick)
        rows.append(learner.row(carry))
    landing = _touchdowns(rows)[0]
    assert not any(rows[t]["lock"] for t in range(1, len(rows)))
    assert _walk_after(rows, landing) == 1
