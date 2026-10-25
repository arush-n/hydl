"""Gen-7 guard on the policy mask (row 5d; overlay patch ``guard_reentry_closes_new_claim`` in sim/gen7/overlay.py).

The re-entry countdown (construct ``guard_cooldown``) closes a NEW guard claim; a guard already held stays on the mask. Before
the fix the term closed the held guard too: the policy dropped it on its second tick and it never engaged (task: the skill
layer's encoding audit, 2026-10-21). Live 0.7.2 (FIDELITY-0.7.2.md section (af), 24 runs): claim on tick 1, activation on
tick 3. The replays drive the policy-level environment (``make_arsenal_environment``, ``step_factors``) on the gen-7 render,
so the mask under test is the one the policy sees.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator

import jax
import numpy as np
import pytest

from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY

from .support import mount_fresh, render_for_test, unmount

SEED = 0


class _Env:
    """The single-agent arsenal environment on a mounted render: the agent's iron sword against an iron-sword target."""

    def __init__(self, package: str) -> None:
        combat = import_module(package + ".combat")
        arsenal = import_module(package + ".combat.arsenal")
        environment = import_module(package + ".combat.arsenal.environment")
        layout = import_module(package + ".combat.observation.v3.policy.layout")
        params = combat.default_combat_params(microticks=1)
        config = arsenal.arsenal_runtime_config(
            arsenal.hytale_0_5_7_loadouts(["iron_sword"], target_profiles=["iron_sword"]),
        )
        self.env = environment.make_arsenal_environment(params, config, maximum_turn_degrees=45.0)
        names = list(layout.ARSENAL_POLICY_ACTION_HEAD_NAMES)
        sizes = list(layout.ARSENAL_POLICY_ACTION_HEAD_SIZES)
        self.head = names.index("guard_off_on")
        self.on_bit = sum(sizes[: self.head]) + 1
        self.neutral = layout.neutral_arsenal_policy_action_factors(1)

    def run(self, on_ticks: set[int], ticks: int) -> list[dict[str, Any]]:
        carry, _ = self.env.reset(jax.random.split(jax.random.key(SEED), 1))
        rows = []
        for tick in range(1, ticks + 1):
            legal = bool(np.asarray(carry.action_mask)[0, self.on_bit])
            factors = self.neutral.at[:, self.head].set(1 if tick in on_ticks else 0)
            carry, _, _, _, info = self.env.step_factors(carry, factors, jax.random.split(jax.random.key(tick), 1))
            mechanics = carry.runtime.mechanics
            rows.append({
                "tick": tick,
                "on_legal_before": legal,
                "valid": bool(np.asarray(info.action_valid).reshape(-1)[0]),
                "held": bool(np.asarray(mechanics.guard_held).reshape(-1)[0]),
                "active": bool(np.asarray(mechanics.guard_active).reshape(-1)[0]),
            })
        return rows


@contextmanager
def _mounted(path: Path) -> Iterator[str]:
    package = mount_fresh(path).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.fixture(scope="module")
def traces(tmp_path_factory) -> dict[str, list[dict[str, Any]]]:
    path = tmp_path_factory.mktemp("guard_mask_held") / "gen7"
    path = render_for_test(GEN7_OVERLAY, path, source=HeadSource())
    with _mounted(path) as package:
        env = _Env(package)
        return {
            "hold12": env.run(set(range(1, 13)), 14),
            "reclaim": env.run({1, 2} | set(range(4, 17)), 16),
        }


def _ticks(rows: list[dict[str, Any]], key: str) -> list[int]:
    return [row["tick"] for row in rows if row[key]]


def test_a_held_guard_stays_on_the_mask_and_engages_on_tick_3(traces) -> None:
    rows = traces["hold12"]
    assert _ticks(rows, "on_legal_before")[:12] == list(range(1, 13)), rows
    assert all(row["valid"] for row in rows[:12]), rows
    assert _ticks(rows, "held")[:12] == list(range(1, 13)), rows
    assert _ticks(rows, "active")[0] == 3, rows
    assert set(range(3, 13)) <= set(_ticks(rows, "active")), rows


def test_a_new_claim_after_a_release_waits_for_the_countdown(traces) -> None:
    rows = traces["reclaim"]
    closed = [row["tick"] for row in rows if 4 <= row["tick"] and not row["on_legal_before"]]
    reopened = next(row["tick"] for row in rows if row["tick"] >= 4 and row["on_legal_before"])
    assert closed and closed[0] == 4, rows
    # Live (section (af)): the entry choice reopens at START + 10 or START + 11 after a claim on tick 1.
    assert 11 <= reopened <= 13, rows
    assert not any(row["active"] for row in rows if 4 <= row["tick"] < reopened), rows
