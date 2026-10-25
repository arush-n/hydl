"""Shield bash offered only under a held and active guard (gen 7, FIDELITY section (x), row 5d; construct
``sim/gen7/mechanics/shield_bash_fork.py``, switch ``Constructs.shield_bash_root``).

The replays drive the engine's own step (``arsenal.step_arsenal_batch``) for the agent on the iron shield, on the gen-7 and
gen-5 renders, under one compiled ``lax.scan`` per render. Tick k is the state after action k and tick 0 the reset (the learner
row's convention). The legal bit is the observable mask (``ability_legal_mask`` on the lifecycle view, as the v3 encoder builds
it) for slot 0, the bash; ``accepted`` is the admission event of the request on that tick.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5_OVERLAY
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import shield_bash_fork
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, render_for_test, unmount

SEED = 570057
TICKS = 16
HOLD = range(1, 13)  # a 12-tick hold: requested on ticks 1 to 12, released from tick 13
SECTION_X_ADVERTISED = list(range(3, 13))  # the live 0.7.2 rows: advertised on ticks 3 to 12 only
AGENT = 0


class _Replay:
    """One render's engine: the iron shield loadout, the open flat world, and the compiled per-tick scan."""

    def __init__(self, package: str) -> None:
        self.package = package
        combat = import_module(package + ".combat")
        arsenal = import_module(package + ".combat.arsenal")
        environment = import_module(package + ".combat.arsenal.environment")
        self.ability = import_module(package + ".combat.arsenal.programs.ability")
        self.arsenal = arsenal
        self.combat = combat
        self.params = combat.default_combat_params(microticks=1, target_active=False)
        self.config = arsenal.arsenal_runtime_config(
            arsenal.hytale_0_5_7_loadouts(("iron_shield",)),
            specialize=False,
            sensor_range=self.params.sensor_range,
        )
        state, _ = arsenal.reset_arsenal_batch(jax.random.split(jax.random.key(SEED), 1), self.params, self.config)
        self.world = environment.open_flat_arsenal_world_capabilities(state, self.params, config=self.config)
        self.reset_legal = bool(self._legal(state))
        self._scan = jax.jit(self._run)

    def _legal(self, state):
        view = self.ability.ability_lifecycle_legality_view(state.arsenal, self.config.loadout)
        mask = self.ability.ability_legal_mask(
            view, state.mechanics, self.config.loadout, self.world, state.combat.health > 0.0
        )
        return mask[0, AGENT, 0]

    def _run(self, state, guard, request, keys):
        def body(current, xs):
            guard_on, request_slot, key = xs
            command = self.arsenal.empty_arsenal_commands(1)
            command = command._replace(
                ability_slot=command.ability_slot.at[:, AGENT].set(request_slot),
                defense=command.defense._replace(guard_held=command.defense.guard_held.at[:, AGENT].set(guard_on)),
                world=self.world,
            )
            action = jnp.zeros((1, self.combat.ACTION_SIZE), dtype=jnp.float32)
            transition = self.arsenal.step_arsenal_batch(current, action, key, self.params, command, self.config)
            nxt = transition.state
            out = {
                "legal": self._legal(nxt),
                "held": nxt.mechanics.guard_held[0, AGENT],
                "active": nxt.mechanics.guard_active[0, AGENT],
                "accepted": transition.arsenal_info.ability_accepted[0, AGENT],
                "active_slot": nxt.arsenal.active_ability_slot[0, AGENT],
            }
            return nxt, out

        return jax.lax.scan(body, state, (guard, request, keys))

    def run(self, guard_ticks=(), request_ticks=()) -> list[dict]:
        """Rows for ticks 1 to ``TICKS``: a guard request on ``guard_ticks``, a bash request on ``request_ticks``."""

        held, requested = set(guard_ticks), set(request_ticks)
        guard = jnp.asarray([t in held for t in range(1, TICKS + 1)], dtype=jnp.bool_)
        request = jnp.asarray([0 if t in requested else -1 for t in range(1, TICKS + 1)], dtype=jnp.int32)
        keys = jax.random.split(jax.random.key(161803), TICKS).reshape(TICKS, 1)
        initial, _ = self.arsenal.reset_arsenal_batch(jax.random.split(jax.random.key(SEED), 1), self.params, self.config)
        _, trace = self._scan(initial, guard, request, keys)
        rows = []
        for index in range(TICKS):
            row = {name: trace[name][index].item() for name in trace}
            rows.append({"tick": index + 1, **row})
        return rows


@pytest.fixture(scope="module")
def replays(tmp_path_factory):
    engines = {}
    for name, overlay in (("gen7", GEN7_OVERLAY), ("gen5", GEN5_OVERLAY)):
        target = tmp_path_factory.mktemp("render") / name
        target = render_for_test(overlay, target, source=HeadSource())
        engines[name] = _Replay(mount_fresh(target).__name__)
    yield engines
    for engine in engines.values():
        unmount(engine.package)


def _ticks(rows, key):
    return [row["tick"] for row in rows if row[key]]


# ---------------------------------------------------------------------------------------------------------------------
# Switch plumbing: gen 5 keeps the stock mask, an unknown value fails closed.
# ---------------------------------------------------------------------------------------------------------------------


def test_switch_defaults_name_the_gen5_plain_root_and_the_gen7_fork() -> None:
    assert Constructs().shield_bash_root == "plain_root"
    assert GEN5.constructs.shield_bash_root == "plain_root"
    assert GEN7.constructs.shield_bash_root == "held_guard_fork"


def test_gen5_returns_the_original_mask() -> None:
    def original(*args):
        return args

    assert shield_bash_fork.root_admission(original, GEN5) is original


def test_an_unknown_switch_value_fails_closed() -> None:
    def original(*args):
        return args

    typo = SimpleNamespace(constructs=replace(GEN5.constructs, shield_bash_root="typo"))
    with pytest.raises(ValueError):
        shield_bash_fork.root_admission(original, typo)


# ---------------------------------------------------------------------------------------------------------------------
# Render replays (gen 7): the bash is offered and admitted only while the guard is held and active.
# ---------------------------------------------------------------------------------------------------------------------


def test_without_a_guard_the_bash_is_not_offered_and_a_request_is_refused(replays) -> None:
    engine = replays["gen7"]
    rows = engine.run(request_ticks=range(1, 7))
    assert not engine.reset_legal
    assert _ticks(rows, "legal") == []
    assert _ticks(rows, "accepted") == []
    assert all(row["active_slot"] == -1 for row in rows)


def test_section_x_hold_offers_the_bash_on_ticks_3_to_12_only(replays) -> None:
    rows = replays["gen7"].run(guard_ticks=HOLD)
    assert _ticks(rows, "legal") == SECTION_X_ADVERTISED
    assert [row["active"] for row in rows[:3]] == [False, False, True]


def test_a_request_before_the_guard_is_active_is_refused(replays) -> None:
    rows = replays["gen7"].run(guard_ticks=HOLD, request_ticks={2})
    assert _ticks(rows, "accepted") == []
    assert all(row["active_slot"] == -1 for row in rows)


def test_a_request_while_held_and_active_is_admitted(replays) -> None:
    rows = replays["gen7"].run(guard_ticks=HOLD, request_ticks={4})
    assert _ticks(rows, "accepted") == [4]
    assert rows[3]["active_slot"] == 0


def test_after_release_the_bash_is_neither_offered_nor_admitted(replays) -> None:
    # Tick 13 is the release tick (the guard is still active on it, so its admission is not asserted here); from tick 14
    # the guard is neither held nor active and a request is refused.
    rows = replays["gen7"].run(guard_ticks=HOLD, request_ticks={14, 15})
    assert _ticks(rows, "accepted") == []
    assert not any(row["legal"] for row in rows if row["tick"] >= 13)


# ---------------------------------------------------------------------------------------------------------------------
# Gen-5 control: the stock mask keeps the held-level offer (no active term).
# ---------------------------------------------------------------------------------------------------------------------


def test_gen5_keeps_the_stock_offer_from_the_first_held_tick(replays) -> None:
    engine = replays["gen5"]
    rows = engine.run(guard_ticks=HOLD)
    assert not engine.reset_legal
    assert _ticks(rows, "legal") == list(range(1, 13))
