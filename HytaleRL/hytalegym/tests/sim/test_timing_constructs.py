"""Gen-7 timing constructs (FIDELITY-0.7.2.md section (af)): the guard claim cooldown, the one-drain ability start and the
shortbow's empty reset.

The receipts are reduced to ``timing_receipts_072.json`` (claim, activation, reopen, start, finish, legal-on and reset
ticks of the live 0.7.2 bridge). The replays drive the construct functions themselves on the simulator's exact 1/30 s
tick. The native ticks differ by one tick in 17 of the 24 guard runs (load jitter recorded in section (af)); the tests
assert that difference rather than hide it.
"""

from __future__ import annotations

import json
from collections import namedtuple
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import pytest

import importlib

from hytalegym.sim.common import bench
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import ability_start_split, guard_cooldown, shortbow_reset_ammo
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

FIXTURE = json.loads((Path(__file__).parent / "timing_receipts_072.json").read_text(encoding="utf-8"))
#: Names the construct reads from a rendered module's globals (the shortbow loadout builder's module).
RESOURCE_AMMO = 5
AGENT_ENTITY = 0
RULES = namedtuple("RULES", "server_ticks_per_second")(jnp.float32(30.0))
GUARD_CLAIM_TICK = 1


class _Profile:
    """Only the ``constructs`` attribute is read by the construct functions."""

    def __init__(self, constructs: Constructs) -> None:
        self.constructs = constructs


def _profile(base, **overrides) -> _Profile:
    return _Profile(replace(base.constructs, **overrides))


# ---------------------------------------------------------------------------------------------------------------------
# Switch plumbing: gen 5 keeps the original function, an unknown value fails closed.
# ---------------------------------------------------------------------------------------------------------------------


def test_gen5_profile_keeps_every_timing_switch_at_its_default() -> None:
    assert GEN5.constructs.guard_reentry_cooldown == "none"
    assert GEN5.constructs.ability_start_split == "claim_tick"
    assert GEN5.constructs.shortbow_reset_ammo == "loadout_one"
    assert Constructs().guard_reentry_cooldown == "none"


def test_gen7_profile_sets_the_timing_switches() -> None:
    assert GEN7.constructs.guard_reentry_cooldown == "claim_0_35s"
    assert GEN7.constructs.ability_start_split == "next_drain"
    assert GEN7.constructs.shortbow_reset_ammo == "native_zero"


@pytest.mark.parametrize(
    ("module", "switch"),
    (
        (guard_cooldown, "guard_reentry_cooldown"),
        (ability_start_split, "ability_start_split"),
        (shortbow_reset_ammo, "shortbow_reset_ammo"),
    ),
)
def test_gen5_returns_the_original_function(module, switch) -> None:
    def original(*args, **kwargs):
        return args

    assert module.SWITCH == switch
    assert module.IMPLEMENTED is True
    assert module.construct(original, _profile(GEN5)) is original


def test_an_unknown_switch_value_fails_closed() -> None:
    def original(*args, **kwargs):
        return args

    with pytest.raises(ValueError):
        guard_cooldown.construct(original, _profile(GEN5, guard_reentry_cooldown="typo"))
    with pytest.raises(ValueError):
        ability_start_split.construct(original, _profile(GEN5, ability_start_split="typo"))
    with pytest.raises(ValueError):
        shortbow_reset_ammo.construct(original, _profile(GEN5, shortbow_reset_ammo="typo"))


# ---------------------------------------------------------------------------------------------------------------------
# Guard claim cooldown: the countdown on the simulator's 1/30 s tick against the 24 native guard runs.
# ---------------------------------------------------------------------------------------------------------------------


def _guard_trace(request_ticks, last_tick):
    """Drive the countdown with a held guard request on ``request_ticks`` (released on every other tick).

    Mirrors the construct: a request is refused while the countdown stored after the previous tick is positive; an
    admitted claim sets it to 0.35 s; otherwise it falls by one nominal tick and floors at zero. Returns the admitted
    ticks and the countdown after each tick.
    """

    stored = jnp.float32(0.0)
    held = False
    admitted_ticks, after = [], {}
    for tick in range(1, last_tick + 1):
        requested = tick in request_ticks
        blocked = bool(guard_cooldown.refused(jnp, stored, jnp.bool_(requested), jnp.bool_(held)))
        admitted = requested and not held and not blocked
        if admitted:
            admitted_ticks.append(tick)
        held = (held or admitted) if requested else False
        stored = guard_cooldown.countdown_after(
            jnp, stored, jnp.bool_(admitted), guard_cooldown.tick_seconds(jnp, RULES)
        )
        after[tick] = float(stored)
    return admitted_ticks, after


def test_a_single_claim_reopens_on_claim_plus_eleven_at_the_simulator_clock() -> None:
    # The claim is tick 1; 0.35 s at 1/30 s runs out on tick 12 (claim + 11).
    _, after = _guard_trace({GUARD_CLAIM_TICK}, last_tick=14)
    reopen = next(t for t in range(2, 15) if after[t] <= 0.0 and after[t - 1] > 0.0)
    assert reopen == 12


@pytest.mark.parametrize("run", FIXTURE["guard_reopen"], ids=lambda r: f"seed{r['seed']}-release{r['release_ticks']}")
def test_each_native_guard_run_is_replayed_on_the_same_claim(run) -> None:
    assert run["request_tick"] == 5 + run["release_ticks"]
    assert run["native_reopen_tick"] in (11, 12)
    _, after = _guard_trace({GUARD_CLAIM_TICK}, last_tick=14)
    assert next(t for t in range(2, 15) if after[t] <= 0.0 and after[t - 1] > 0.0) == 12


def test_the_simulator_reopen_matches_seven_of_the_twenty_four_native_runs() -> None:
    native = [run["native_reopen_tick"] for run in FIXTURE["guard_reopen"]]
    assert len(native) == 24
    # Under the loaded server's long ticks the choice reopened on claim + 10 (tick 11) in 17 runs and on claim + 11
    # (tick 12) in 7. The simulator's tick 12 is the native tick in those 7; the other 17 are one tick late (recorded).
    assert sum(1 for tick in native if tick == 11) == 17
    assert sum(1 for tick in native if tick == 12) == 7


@pytest.mark.parametrize("release", range(1, 13))
def test_a_held_request_is_admitted_on_the_tick_after_the_reopen_when_its_window_reaches_it(release) -> None:
    # Held guard on ticks 1 to 4, released for ``release`` ticks, requested again for four ticks from 5 + release.
    request = 5 + release
    window = set(range(request, request + 4))
    admitted, _ = _guard_trace(set(range(1, 5)) | window, last_tick=request + 8)
    assert admitted[0] == 1
    reopen_admission = max(request, 13)
    if reopen_admission <= request + 3:
        assert admitted[1:] == [reopen_admission]
    else:
        assert admitted[1:] == []


def test_an_idle_guard_never_runs_the_countdown() -> None:
    admitted, after = _guard_trace(set(), last_tick=30)
    assert admitted == []
    assert set(after.values()) == {0.0}


def test_the_guard_wrapper_refuses_a_rising_claim_inside_the_cooldown() -> None:
    State = namedtuple("State", "guard_held guard_reentry_seconds")
    Commands = namedtuple("Commands", "guard_held")

    def admit_all(state, commands, yaw, alive, rules, params):
        # Stands in for the original admission: a rising request is admitted and a released one drops.
        return state._replace(guard_held=commands.guard_held), None

    wrapped = guard_cooldown.construct(admit_all, _profile(GEN7))
    state = State(guard_held=jnp.bool_(False), guard_reentry_seconds=jnp.float32(0.0))

    def step(state, held):
        return wrapped(state, Commands(guard_held=jnp.bool_(held)), None, None, RULES, None)[0]

    state = step(state, True)  # claim admitted on tick 1: the countdown is 0.35 s
    assert bool(state.guard_held) is True
    assert float(state.guard_reentry_seconds) == pytest.approx(0.35)
    state = step(state, False)  # release on tick 2
    assert bool(state.guard_held) is False
    state = step(state, True)  # a rising request on tick 3 is inside the cooldown: refused
    assert bool(state.guard_held) is False


# ---------------------------------------------------------------------------------------------------------------------
# Ability chain start: the admitted rows start one drain later; the construct does not touch the charges.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_ability_wrapper_delays_only_the_admitted_rows() -> None:
    Arsenal = namedtuple("Arsenal", "ability_scheduler_tick ability_charge_count")

    def admit(arsenal, mechanics, requested, loadout, world, alive, rules, admission=None):
        accepted = jnp.asarray([[True, False]])
        tick = jnp.where(accepted, jnp.int32(0), arsenal.ability_scheduler_tick)
        return arsenal._replace(ability_scheduler_tick=tick), mechanics, requested, accepted

    wrapped = ability_start_split.construct(admit, _profile(GEN7))
    arsenal = Arsenal(
        ability_scheduler_tick=jnp.asarray([[-7, -7]], dtype=jnp.int32),
        ability_charge_count=jnp.asarray([[3, 3]], dtype=jnp.int32),
    )
    out, _, _, accepted = wrapped(arsenal, None, None, None, None, None, None)
    # The admitted row sits one scheduler tick further back (its start and finish move one drain later).
    assert out.ability_scheduler_tick.tolist() == [[-1, -7]]
    # The construct leaves the charge counts to the original: they are spent on the claim.
    assert out.ability_charge_count.tolist() == [[3, 3]]
    assert accepted.tolist() == [[True, False]]


# ---------------------------------------------------------------------------------------------------------------------
# Shortbow reset ammo: only the shortbow agent starts at zero; other rows and resources are unchanged.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_shortbow_wrapper_zeroes_only_the_shortbow_agent() -> None:
    Loadout = namedtuple("Loadout", "resource_initial")
    initial = jnp.ones((2, 2, 6), dtype=jnp.float32)

    def build(agent_profiles, *, target_profiles=None):
        return Loadout(resource_initial=initial)

    wrapped = shortbow_reset_ammo.construct(build, _profile(GEN7))
    out = wrapped(["iron_shortbow", "iron_sword"])
    assert float(out.resource_initial[0, AGENT_ENTITY, RESOURCE_AMMO]) == 0.0
    assert float(out.resource_initial[1, AGENT_ENTITY, RESOURCE_AMMO]) == 1.0
    assert float(out.resource_initial[0, AGENT_ENTITY, 4]) == 1.0
    assert float(out.resource_initial[0, 1, RESOURCE_AMMO]) == 1.0


def test_the_recorded_shortbow_reset_row_matches_the_native_receipts() -> None:
    for seed in ("570057", "913771"):
        assert FIXTURE["shortbow_reset"][seed]["ammo_ticks_0_to_4"] == [0.0, 0.0, 1.0, 1.0, 1.0]
        assert FIXTURE["shortbow_reset"][seed]["claim_tick"] == 1
        assert FIXTURE["shortbow_reset"][seed]["accepted_tick"] == 2


def test_tick_zero_head_and_body_yaw_are_zero_on_both_receipt_seeds() -> None:
    for seed in ("570057", "913771"):
        assert FIXTURE["tick0_yaw"][seed] == {"yaw_degrees": 0.0, "head_yaw_degrees": 0.0}


def test_the_recorded_chain_timelines_are_one_drain_after_the_claim() -> None:
    for seed in ("570057", "913771"):
        assert FIXTURE["kunai_slot_zero"][seed]["accepted"] == [2]
        assert FIXTURE["kunai_slot_zero"][seed]["finished"] == [11]
        assert FIXTURE["iron_sword_slot_zero"][seed]["accepted"] == [2]
        assert FIXTURE["iron_sword_slot_zero"][seed]["finished"] == [14]


# ---------------------------------------------------------------------------------------------------------------------
# Render build: the hooks run when the gen-7 render is imported. A hook that reads a name the wrapped module does not
# bind raises here, not in the text render (the KeyError of the first draft).
# ---------------------------------------------------------------------------------------------------------------------


def test_the_pinned_constants_are_the_engine_values() -> None:
    from hytalegym.jax.combat.mechanics.schema.contract import RESOURCE_AMMO as engine_ammo
    from hytalegym.jax.combat.types import AGENT_ENTITY as engine_agent

    assert shortbow_reset_ammo.RESOURCE_AMMO == engine_ammo == 5
    assert shortbow_reset_ammo.AGENT_ENTITY == engine_agent == 0


def test_the_gen7_render_imports_with_the_three_timing_hooks(tmp_path) -> None:
    rendered = tmp_path / "gen7"
    R.render(GEN7_OVERLAY, rendered, source=HeadSource())
    package = bench.package_for("gen7", rendered)
    hooked = {
        "combat.mechanics.runtime.motion": "apply_defense_commands",
        "combat.arsenal.programs.ability": "start_abilities",
        "combat.arsenal.profiles.hytale_0_5_7": "hytale_0_5_7_loadouts",
    }
    for module_name, function in hooked.items():
        module = importlib.import_module(f"{package}.{module_name}")
        assert getattr(getattr(module, function), "__wrapped__", None) is not None, (module_name, function)
