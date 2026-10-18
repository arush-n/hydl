"""Landing walk resume (gen 7, row 1c, F1c): an accepted jump arms a countdown that holds the planar walk on the two
grounded ticks after its touchdown.

Measured in FIDELITY-0.7.2.md section (o): after a jump the native NPC walks again on the third tick after landing (saved
capture single-jump-30: grounded from step 11, first displacement on step 14). After a walk-off landing with no jump (saved
capture forward-30) it walks on the first tick, so a landing alone must not arm the hold.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import landing_walk
from hytalegym.sim.gen7.overlay import HOOKS, PATCHES
from hytalegym.sim.gen7.profile import PROFILE as GEN7


def _run(xp, ticks):
    """Replay (grounded at tick start, accepted jump this tick) pairs; return the hold of each tick."""

    stored = xp.zeros((1,), dtype=xp.int32)
    holds = []
    for grounded, jumped in ticks:
        g = xp.asarray([grounded])
        hold = landing_walk.holding(xp, g, stored)
        stored = landing_walk.countdown(xp, g, xp.asarray([jumped]) & g, stored)
        holds.append(bool(hold[0]))
    return holds


@pytest.mark.parametrize(
    ("ticks", "expected"),
    (
        # standing, jump accepted on tick 1, airborne 2-3, grounded from 4: held on 4 and 5, walks on 6
        (
            [(True, False), (True, True), (False, False), (False, False), (True, False), (True, False), (True, False), (True, False)],
            [False, False, False, False, True, True, False, False],
        ),
        # walk off the ledge (no jump), airborne 1-2, landing on 3: no hold, the walk resumes at once
        (
            [(True, False), (False, False), (False, False), (True, False), (True, False)],
            [False, False, False, False, False],
        ),
        # a re-jump on the first held tick (tick 4, still held) restarts the countdown at the next landing
        (
            [(True, False), (True, True), (False, False), (True, False), (True, True), (False, False), (True, False), (True, False)],
            [False, False, False, True, True, False, True, True],
        ),
        # standing still with no jump: never held
        ([(True, False)] * 4, [False] * 4),
    ),
)
def test_a_jump_arms_two_held_grounded_ticks_and_a_plain_landing_does_not(ticks, expected) -> None:
    assert _run(np, ticks) == expected
    assert _run(jnp, ticks) == expected


def test_countdown_spends_one_per_grounded_tick_and_never_goes_below_zero() -> None:
    stored = np.asarray([2], dtype=np.int32)
    grounded = np.asarray([True])
    no_jump = np.asarray([False])
    stored = landing_walk.countdown(np, grounded, no_jump, stored)
    assert int(stored[0]) == 1
    stored = landing_walk.countdown(np, grounded, no_jump, stored)
    assert int(stored[0]) == 0
    stored = landing_walk.countdown(np, grounded, no_jump, stored)
    assert int(stored[0]) == 0


def test_an_airborne_tick_keeps_the_count_and_a_jump_resets_it() -> None:
    airborne = np.asarray([False])
    assert int(landing_walk.countdown(np, airborne, np.asarray([False]), np.asarray([2], np.int32))[0]) == 2
    assert int(landing_walk.countdown(np, np.asarray([True]), np.asarray([True]), np.asarray([0], np.int32))[0]) == 2


def test_gen5_and_at_once_return_the_original_function() -> None:
    def original(state, actions, pause_draw, params, *rest, **kwargs):  # pragma: no cover - identity only
        return None

    assert landing_walk.construct(original, GEN5) is original
    at_once = replace(GEN7, constructs=replace(GEN7.constructs, landing_walk_resume="at_once"))
    assert landing_walk.construct(original, at_once) is original


def test_unknown_switch_value_refuses() -> None:
    bad = replace(GEN7, constructs=replace(GEN7.constructs, landing_walk_resume="maybe"))
    with pytest.raises(ValueError):
        landing_walk.construct(lambda *a, **k: None, bad)


def test_profiles_set_the_switch() -> None:
    assert Constructs().landing_walk_resume == "at_once"
    assert GEN5.constructs.landing_walk_resume == "at_once"
    assert GEN7.constructs.landing_walk_resume == "third_tick"


def test_overlay_declares_the_hook_and_the_countdown_patches() -> None:
    hook = next(h for h in HOOKS if h.id == "landing_walk_resume")
    assert (hook.path, hook.function, hook.switch, hook.fidelity_row) == (
        "combat/runtime/step.py", "_apply_action", "landing_walk_resume", "1c")
    assert hook.construct == "hytalegym.sim.gen7.mechanics.landing_walk:construct"
    fields = {p.id: p for p in PATCHES if p.id.startswith("landing_walk_state")}
    assert set(fields) == {"landing_walk_state_field", "landing_walk_state_init"}
    assert all(p.fidelity_row == "1c" and p.differential for p in fields.values())
