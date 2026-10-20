"""Hit interrupt, pure logic (gen 7, row 5a; ``sim/gen7/mechanics/hit_interrupt.py``).

The engine hooks are not wired (``IMPLEMENTED = False``), so these tests run the construct's functions on ``StatusState``
arrays of one batch and two entities (the target is entity 1). Removal follows the stock decrement of
``mechanics/runtime/statuses.py`` tick_statuses (``remaining = max(0, remaining - dt)``, active while > 0) at dt = 1/30 s in
float32. Native receipts (FIDELITY (an) Part 1) remove the flinch after 15 frames; the stock clock removes it after 16 (see
``test_removal_ticks_follow_the_stock_clock``). Not tested: the engine render (no hook is wired).
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.mechanics.schema.contract import STATUS_FLAG_DISABLE_ABILITIES, STATUS_FLAG_DISABLE_MOVEMENT
from hytalegym.jax.combat.mechanics.schema.types import StatusState
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import hit_interrupt as hi
from hytalegym.sim.gen7.profile import PROFILE as GEN7

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
    assert stunned[0] and stunned[1] == pytest.approx(0.7) and stunned[3] == STATUS_FLAG_DISABLE_ABILITIES
    assert flinch[0] and flinch[1] == pytest.approx(0.5) and flinch[2] == 0.0 and flinch[3] == STATUS_FLAG_DISABLE_MOVEMENT
    assert reaction[0] and reaction[1] == pytest.approx(4.0)
    assert not _slot(status, "Hit_Interrupt_Recent_Reaction_B")[0]
    assert not _slot(status, "Hit_Interrupt_Recovery_Pending")[0]


def test_removal_ticks_follow_the_stock_clock() -> None:
    """Stock: the flinch (0.5 s) is removed on tick 16 and the stun (0.7 s) on tick 21. Native: 15 and 21 frames (FIDELITY
    (an) Part 1). The difference is the float32 decrement, not this module (see the module docstring)."""

    status, tick = _run_to(_empty(), 1, hi.ELIGIBLE_DELAY_TICKS)
    status = hi.apply_hit(status, TARGET, ALL_ON, TABLE)
    removed = {}
    for step in range(1, 130):
        status = _tick(status)
        for name in ("Hit_Interrupt_Flinch", "Hit_Interrupt_Stunned"):
            if name not in removed and not _slot(status, name)[0]:
                removed[name] = step
    assert removed == {"Hit_Interrupt_Flinch": 16, "Hit_Interrupt_Stunned": 21}


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


def test_the_switch_is_off_by_default_and_the_unwired_construct_fails_closed() -> None:
    sentinel = object()
    assert Constructs().hit_interrupt is False
    assert GEN5.constructs.hit_interrupt is False
    assert hi.construct(sentinel, GEN5) is sentinel
    assert hi.IMPLEMENTED is False
    with pytest.raises(NotImplementedError):
        hi.construct(sentinel, GEN7)
