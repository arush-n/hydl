"""Status clock (gen 7, task #106; ``sim/gen7/mechanics/status_clock.py``).

Pure logic: the stock ``tick_statuses`` (``jax/combat/mechanics/runtime/statuses.py``) and the construct's wrapper on one status
slot, dt = 1/30 s in float32. Render: the gen-7 render with the switch on, the switch off, and the committed render (the overlay
without the ``status_clock`` row); the switch-off render must equal the committed one leaf by leaf. Durations that are whole ticks
are measured (the live 0.5 s and 0.7 s hit-interrupt statuses, 15 and 21 frames); the non-whole rows are the construct's stated
rule, not a measurement.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pytest

from hytalegym.jax.combat.mechanics.factory import default_mechanics_rules, empty_mechanics_state
from hytalegym.jax.combat.mechanics.runtime import statuses as stock
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import hit_interrupt as hi
from hytalegym.sim.gen7.mechanics import status_clock as sc
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, render_for_test, unmount

DT = np.float32(1.0 / 30.0)
TARGET = 1
SLOT = (0, TARGET, 0)
STEPS = 400
TRACE_STEPS = 25


def _profile(on: bool) -> SimpleNamespace:
    return SimpleNamespace(constructs=SimpleNamespace(status_clock=on))


def _slot_state(duration: float) -> tuple[Any, Any]:
    rules = default_mechanics_rules(1)
    state = empty_mechanics_state(1, rules)
    statuses = state.statuses
    statuses = statuses._replace(
        effect_id=statuses.effect_id.at[SLOT].set(1),
        active=statuses.active.at[SLOT].set(True),
        remaining_seconds=statuses.remaining_seconds.at[SLOT].set(duration),
    )
    return rules, state._replace(statuses=statuses)


def _ticks_until_inactive(tick: Any, duration: float) -> int:
    """The 1-based tick on which the target's slot first reads inactive, with ``tick`` applied to a status of ``duration``."""

    rules, state = _slot_state(duration)
    for step in range(1, STEPS + 1):
        state, _ = tick(state, DT, rules)
        if not bool(state.statuses.active[SLOT]):
            return step
    raise AssertionError(f"a {duration} s status is still active after {STEPS} ticks")


def test_a_half_second_status_ends_after_15_ticks_on_the_clock_and_16_on_the_stock_clock() -> None:
    # Stock (the gen-5 engine): 0.5 - k * float32(1/30) stays positive after 15 sequential subtractions (2.2e-8).
    assert _ticks_until_inactive(stock.tick_statuses, 0.5) == 16
    assert _ticks_until_inactive(sc.construct(stock.tick_statuses, _profile(True)), 0.5) == 15


#: (authored duration in seconds, ticks on the construct). The whole-tick rows are the 0.7.2 hit-interrupt durations and the
#: 1 tick case; the rest are the rule's rounding (half up, a tie at a half tick), assumed and not measured.
DURATIONS = [
    (0.5, 15),
    (0.7, 21),
    (2.0, 60),
    (4.0, 120),
    (8.0, 240),
    (1.0 / 30.0, 1),
    (0.01, 1),
    (0.25, 8),
    (0.51, 15),
    (0.52, 16),
    (5.05, 152),
]


@pytest.mark.parametrize(("duration", "ticks"), DURATIONS)
def test_the_construct_lasts_round_of_the_duration_over_the_tick(duration: float, ticks: int) -> None:
    assert _ticks_until_inactive(sc.construct(stock.tick_statuses, _profile(True)), duration) == ticks


def test_the_stock_clock_is_one_tick_late_on_whole_durations_except_where_float32_rounds_below() -> None:
    assert _ticks_until_inactive(stock.tick_statuses, 2.0) == 61
    assert _ticks_until_inactive(stock.tick_statuses, 4.0) == 121
    assert _ticks_until_inactive(stock.tick_statuses, 0.7) == 21  # float32(0.7) rounds below 0.7, so the residue is negative


def test_the_construct_changes_only_the_active_flag() -> None:
    rules, stock_state = _slot_state(0.51)
    _, on_state = _slot_state(0.51)
    tick_on = sc.construct(stock.tick_statuses, _profile(True))
    for _ in range(20):
        stock_state, _ = stock.tick_statuses(stock_state, DT, rules)
        on_state, _ = tick_on(on_state, DT, rules)
        np.testing.assert_array_equal(
            np.asarray(stock_state.statuses.remaining_seconds), np.asarray(on_state.statuses.remaining_seconds)
        )


def test_switch_off_returns_the_stock_function() -> None:
    assert sc.construct(stock.tick_statuses, _profile(False)) is stock.tick_statuses
    with pytest.raises(ValueError):
        sc.construct(stock.tick_statuses, _profile(None))


def test_gen5_keeps_the_stock_clock_and_gen7_turns_it_on() -> None:
    assert GEN5.constructs.status_clock is False
    assert GEN7.constructs.status_clock is True


def test_hit_interrupt_writes_full_durations_when_the_clock_rounds() -> None:
    """Pre-compensation (duration minus half a tick) is for the stock clock only; with the clock on, the stored duration is D."""

    table = hi.load_table(GEN7.data_dir)
    assert hi._effect(table, "Hit_Interrupt_Flinch").duration_seconds == pytest.approx(0.5 - 0.5 / 30.0)
    assert hi._effect(table, "Hit_Interrupt_Flinch", precompensate=False).duration_seconds == pytest.approx(0.5)
    assert hi._effect(table, "Hit_Interrupt_Eligible", precompensate=False).duration_seconds == pytest.approx(1.0)


def _render_trace(package: str) -> dict[str, Any]:
    """Per authored duration: the tick the slot reads inactive, and the first steps' remaining and active leaves."""

    tick = import_module(package + ".combat.mechanics.runtime.statuses").tick_statuses
    factory = import_module(package + ".combat.mechanics.factory")
    out: dict[str, Any] = {}
    for duration, _ in DURATIONS:
        rules = factory.default_mechanics_rules(1)
        state = factory.empty_mechanics_state(1, rules)
        statuses = state.statuses
        statuses = statuses._replace(
            effect_id=statuses.effect_id.at[SLOT].set(1),
            active=statuses.active.at[SLOT].set(True),
            remaining_seconds=statuses.remaining_seconds.at[SLOT].set(duration),
        )
        state = state._replace(statuses=statuses)
        remaining, active, until = [], [], None
        for step in range(1, STEPS + 1):
            state, _ = tick(state, DT, rules)
            if step <= TRACE_STEPS:
                remaining.append(np.asarray(state.statuses.remaining_seconds))
                active.append(np.asarray(state.statuses.active))
            if until is None and not bool(state.statuses.active[SLOT]):
                until = step
        out[str(duration)] = {"until": until, "remaining": np.stack(remaining), "active": np.stack(active)}
    return out


@contextmanager
def _mounted(path: Path) -> Iterator[str]:
    package = mount_fresh(path).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.fixture(scope="module")
def render_dirs(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("status_clock_render")
    committed = replace(GEN7_OVERLAY, hooks=tuple(hook for hook in GEN7_OVERLAY.hooks if hook.id != "status_clock"))
    module_name = "status_clock_test_off_profile"
    off_module = types.ModuleType(module_name)
    off_module.PROFILE = replace(GEN7, constructs=replace(GEN7.constructs, status_clock=False))
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
def traces(render_dirs) -> dict[str, dict[str, Any]]:
    out = {}
    for label, path in render_dirs.items():
        with _mounted(path) as package:
            out[label] = _render_trace(package)
    return out


def test_the_rendered_clock_ends_a_half_second_status_after_15_ticks(traces) -> None:
    assert traces["on"]["0.5"]["until"] == 15
    assert traces["off"]["0.5"]["until"] == 16
    assert traces["committed"]["0.5"]["until"] == 16


def test_the_rendered_clock_matches_the_construct_on_every_row(traces) -> None:
    for duration, ticks in DURATIONS:
        assert traces["on"][str(duration)]["until"] == ticks, duration


def test_switch_off_leaves_the_gen7_render_identical_to_the_committed_render(traces) -> None:
    for duration, _ in DURATIONS:
        off, committed = traces["off"][str(duration)], traces["committed"][str(duration)]
        assert off["until"] == committed["until"], duration
        np.testing.assert_array_equal(off["remaining"], committed["remaining"], err_msg=str(duration))
        np.testing.assert_array_equal(off["active"], committed["active"], err_msg=str(duration))
