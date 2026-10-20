"""Hit interrupt, stun and flinch (gen 7, row 5a; 0.7.2). Switch ``Constructs.hit_interrupt``.

STATUS: pure logic only. The engine hooks are not wired (``IMPLEMENTED = False``; the overlay rows stay ``stub``), so a
gen-7 render is unchanged by this module. Two seams are open and need a decision before wiring: the eligible target's role
(``ArsenalRuntimeConfig`` carries no role field; role ids reach the engine only as initial statuses at reset), and the
row identity of a landed hit (``DamageEvents`` carries none; the call-site seam is ``process_direct_events`` in
``combat/arsenal/effects/events.py``, fed the loadout from ``arsenal/runtime.py``). Design: ``sim/gen7/mechanics_072.md``.

Measured live (FIDELITY-0.7.2.md section (an), Part 1, 2026-10-20; receipts ``~/.claude/jobs/31947bd7/tmp/hit_interrupt_part1``):
``Hit_Interrupt_Eligible`` is added to an eligible Skeleton_Fighter at env tick 42 (43 in some runs) and refreshed every 0.25 s;
a hit before eligibility gives Red_Flash only; the first hit while eligible adds Stunned 0.7 s, Flinch 0.5 s and Recent_Reaction_A
4.0 s in one tick; the flinch is removed 15 frames later and the stun 21 frames later.

Asset (``sim/gen7/data/hit_interrupt.json``, extracted by tools/content_extract/hit_interrupt.py): the eight effect durations,
overlap rules and horizontal speed multipliers, and the 31 attack interactions that carry ``DamageEntityParent_Hit_Interrupt``.
Asset comments give the mechanism: Stunned blocks attack starts (mapped to ``STATUS_FLAG_DISABLE_ABILITIES``); Flinch stops
walking with gravity and knockback active (speed multiplier 0, mapped to ``STATUS_FLAG_DISABLE_MOVEMENT``).
The engine's movement gate for DISABLE_MOVEMENT is read for the agent's desired velocity (arsenal/runtime.py:1506-1512);
the target's motion path is not checked, so the flinch's effect on the target is unverified.

NOT MEASURED (asset values kept, marked unmeasured): Recent_Reaction_B 4 s on the second flinch; Recovery_Pending 2 s on the
third; Enraged 8 s when that flinch ends; Protected_Attack 8 s on attacks started while Enraged (its reader is not located);
the cancel of the target's Attack-tagged root (not modelled).

The stock status clock (``mechanics/runtime/statuses.py`` tick_statuses) subtracts the step dt from ``remaining_seconds``
in float32 and removes a status when it is <= 0. A 0.5 s status is therefore removed on the 16th tick at 1/30 s (the native
receipt shows 15 frames); a 0.7 s status on the 21st. This module does not correct the stock clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.mechanics.schema.contract import (
    DAMAGE_PHYSICAL,
    STATUS_FLAG_DISABLE_ABILITIES,
    STATUS_FLAG_DISABLE_MOVEMENT,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
)

SWITCH = "hit_interrupt"
IMPLEMENTED = False
DATA_FILE = "hit_interrupt.json"
FAMILY = "hit_interrupt"
TABLE_SCHEMA = "hytalerl_content_table_v1"
#: The measured delay: Hit_Interrupt_Eligible is first added at env tick 42 (43 in the period-14 runs).
ELIGIBLE_DELAY_TICKS = 42
#: The role refreshes eligibility every 0.25 s (Hit_Interrupt_Eligible asset comment). At 30 Hz that is 7.5 ticks.
REFRESH_PERIOD_SECONDS = 0.25
TICKS_PER_SECOND = 30
#: The measured eligible roles (static reading: the 22 roles whose reference chain reaches the instruction are the
#: candidates; Skeleton_Fighter and Goblin_Scrapper are the two measured). Configurable through ``params_from_tables``.
MEASURED_ELIGIBLE_ROLES = frozenset({"Skeleton_Fighter", "Goblin_Scrapper"})
#: Effect name -> the mechanics flag the engine honours for it (asset comments). Other effects carry no flag.
EFFECT_FLAGS = {
    "Hit_Interrupt_Stunned": STATUS_FLAG_DISABLE_ABILITIES,
    "Hit_Interrupt_Flinch": STATUS_FLAG_DISABLE_MOVEMENT,
}
ATTACK_SUFFIX = "_Damage"
EFFECT_NAMES = (
    "Hit_Interrupt_Eligible",
    "Hit_Interrupt_Flinch",
    "Hit_Interrupt_Stunned",
    "Hit_Interrupt_Recent_Reaction_A",
    "Hit_Interrupt_Recent_Reaction_B",
    "Hit_Interrupt_Recovery_Pending",
    "Hit_Interrupt_Enraged",
    "Hit_Interrupt_Protected_Attack",
)


@dataclass(frozen=True)
class Effect:
    name: str
    duration_seconds: float
    overlap: int
    speed_multiplier: float
    flags: int


@dataclass(frozen=True)
class HitInterruptTable:
    effects: dict[str, Effect]
    attacks: tuple[str, ...]


@dataclass(frozen=True)
class HitInterruptParams:
    eligible_roles: frozenset[str]
    table: HitInterruptTable


def _overlap(text: str) -> int:
    if text == "Ignore":
        return STATUS_OVERLAP_IGNORE
    if text == "Overwrite":
        return STATUS_OVERLAP_OVERWRITE
    raise ValueError(f"unknown OverlapBehavior {text!r}")


@lru_cache(maxsize=None)
def _read(path: str) -> HitInterruptTable:
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("schema") != TABLE_SCHEMA or table.get("family") != FAMILY:
        raise ValueError(f"{path} is not a {FAMILY} table ({TABLE_SCHEMA})")
    effects: dict[str, Effect] = {}
    attacks: list[str] = []
    for key in sorted(table["rows"]):
        row = table["rows"][key]
        if key.startswith("effect:"):
            name = key.split(":", 1)[1]
            effects[name] = Effect(name, float(row["duration_seconds"]), _overlap(row["overlap"]),
                                   float(row["speed_multiplier"]), EFFECT_FLAGS.get(name, 0))
        elif key.startswith("attack:"):
            attacks.append(row["interaction"])
        else:
            raise ValueError(f"unknown row {key!r} in {path}")
    missing = set(EFFECT_NAMES) - set(effects)
    if missing:
        raise ValueError(f"{path} lacks effect rows {sorted(missing)}")
    return HitInterruptTable(effects, tuple(attacks))


def load_table(data_dir: str | Path) -> HitInterruptTable:
    """The checked-in table in ``data_dir``. It never reads the archive."""

    return _read(str(Path(data_dir) / DATA_FILE))


def params_from_tables(tables: HitInterruptTable, eligible_roles: frozenset[str] = MEASURED_ELIGIBLE_ROLES) -> HitInterruptParams:
    return HitInterruptParams(frozenset(eligible_roles), tables)


def flagged_ability_ids(table: HitInterruptTable) -> tuple[int, ...]:
    """Semantic ids of the attack roots whose damage row is flagged, as the arsenal names them (the row minus ``_Damage``).

    Ability rows (Ability_Fireball_Hit and the like) carry no ``_Damage`` suffix and no arsenal root of that name is
    checked here, so they are not in the set.
    """

    return tuple(sorted({semantic_id(name[: -len(ATTACK_SUFFIX)]) for name in table.attacks if name.endswith(ATTACK_SUFFIX)}))


def flagged_events(ability_id: Any, ability_event_start: Any, ability_event_count: Any, event_index: Any, flagged: Any) -> Any:
    """bool[B, F]: the fired event sits in an ability whose id is flagged.

    ``ability_id``, ``ability_event_start`` and ``ability_event_count`` are [A] (per loadout ability); ``event_index`` is [B, F]
    (``FiredEvents.ability_event_index``); ``flagged`` is the array of ``flagged_ability_ids``."""

    ability_id = jnp.asarray(ability_id, jnp.int32)
    start = jnp.asarray(ability_event_start, jnp.int32)
    count = jnp.asarray(ability_event_count, jnp.int32)
    event = jnp.asarray(event_index, jnp.int32)
    in_range = (event[..., None] >= start) & (event[..., None] < start + count)
    is_flagged = jnp.any(ability_id[:, None] == jnp.asarray(flagged, jnp.int32)[None, :], axis=-1)
    return jnp.any(in_range & is_flagged, axis=-1)


# ---- status slots (the status rows of the 0.7.2 table, written by the construct) -----------------------------------

def _has(status: Any, entity: int, effect_id: int) -> Any:
    return jnp.any(status.active[:, entity] & (status.effect_id[:, entity] == jnp.int32(effect_id)), axis=-1)


def write_status(status: Any, entity: int, effect: Effect, apply: Any) -> Any:
    """Write ``effect`` into ``entity``'s slots where ``apply`` [B] is true (an overlap-aware insert).

    ``Overwrite`` refreshes an active slot of the same effect; ``Ignore`` leaves it. A new effect takes the first free slot.
    The stock applier (``apply_statuses``) rejects a speed multiplier of 0 (statuses.py:207-208), so Flinch is written here.
    """

    effect_id = semantic_id(effect.name)
    active = status.active[:, entity]
    match = active & (status.effect_id[:, entity] == jnp.int32(effect_id))
    has = jnp.any(match, axis=-1)
    free = ~active
    any_free = jnp.any(free, axis=-1)
    first_free = jax.nn.one_hot(jnp.argmax(free, axis=-1), free.shape[-1], dtype=jnp.bool_) & any_free[:, None]
    refresh = apply & has & (effect.overlap == STATUS_OVERLAP_OVERWRITE)
    insert = apply & ~has & any_free
    slot = jnp.where(has[:, None], match, first_free)
    new_values = {
        "effect_id": effect_id,
        "source_entity_id": 0,
        "cycle_elapsed_seconds": 0.0,
        "cycle_cooldown_seconds": 0.0,
        "damage_per_cycle": 0.0,
        "damage_cause": DAMAGE_PHYSICAL,
        "healing_per_cycle": 0.0,
        "resource_id": -1,
        "resource_delta_per_cycle": 0.0,
        "speed_multiplier": effect.speed_multiplier,
        "damage_resistance_present": False,
        "damage_resistance_flat": 0,
        "damage_resistance_multiplier": 0.0,
        "flags": effect.flags,
        "overlap_mode": effect.overlap,
        "active": True,
        "has_cycled": False,
    }
    updated = {}
    for field in status._fields:
        column = getattr(status, field)[:, entity]
        if field == "remaining_seconds":
            mask = slot & (refresh | insert)[:, None]
            value = jnp.float32(effect.duration_seconds)
        else:
            mask = slot & insert[:, None]
            value = new_values.get(field)
            if value is None:
                updated[field] = column
                continue
        expanded = mask.reshape(mask.shape + (1,) * (column.ndim - 2))
        updated[field] = jnp.where(expanded, jnp.asarray(value, column.dtype), column)
    return type(status)(*(getattr(status, f).at[:, entity].set(updated[f]) for f in status._fields))


def eligible_update(status: Any, entity: int, tick: Any, eligible: Any, table: HitInterruptTable) -> Any:
    """Keep ``Hit_Interrupt_Eligible`` on ``entity`` from the measured tick on (``tick`` [B]: the microtick count after the
    step; ``eligible`` [B]: the entity's role is in the eligible set). It is added at ``ELIGIBLE_DELAY_TICKS`` and refreshed
    when the 0.25 s grid (7.5 ticks at 30 Hz, index ``floor(2 * tick / 15)``) advances."""

    effect = table.effects["Hit_Interrupt_Eligible"]
    grid_now = (2 * tick) // 15
    grid_before = (2 * (tick - 1)) // 15
    crossed = grid_now != grid_before
    present = _has(status, entity, semantic_id("Hit_Interrupt_Eligible"))
    due = eligible & (tick >= ELIGIBLE_DELAY_TICKS) & (~present | crossed)
    return write_status(status, entity, effect, due)


def hit_gate(status: Any, entity: int) -> Any:
    """The gate chain (weapons/interaction:DamageEntityParent_Hit_Interrupt_On_Hit): Eligible on, and none of Flinch,
    Enraged and Recovery_Pending."""

    eligible = _has(status, entity, semantic_id("Hit_Interrupt_Eligible"))
    blocked = (_has(status, entity, semantic_id("Hit_Interrupt_Flinch"))
               | _has(status, entity, semantic_id("Hit_Interrupt_Enraged"))
               | _has(status, entity, semantic_id("Hit_Interrupt_Recovery_Pending")))
    return eligible & ~blocked


def apply_hit(status: Any, entity: int, landed: Any, table: HitInterruptTable) -> Any:
    """A landed hit on ``entity`` (``landed`` [B]). Where the gate holds: Stunned, Flinch and a reaction. The reaction is
    Recent_Reaction_A on the first hit, B on the second, and the third queues Recovery_Pending (asset values; unmeasured)."""

    gate = landed & hit_gate(status, entity)
    a_present = _has(status, entity, semantic_id("Hit_Interrupt_Recent_Reaction_A"))
    b_present = _has(status, entity, semantic_id("Hit_Interrupt_Recent_Reaction_B"))
    status = write_status(status, entity, table.effects["Hit_Interrupt_Stunned"], gate)
    status = write_status(status, entity, table.effects["Hit_Interrupt_Flinch"], gate)
    status = write_status(status, entity, table.effects["Hit_Interrupt_Recent_Reaction_A"], gate & ~a_present)
    status = write_status(status, entity, table.effects["Hit_Interrupt_Recent_Reaction_B"], gate & a_present & ~b_present)
    return write_status(status, entity, table.effects["Hit_Interrupt_Recovery_Pending"], gate & a_present & b_present)


def enrage_after_flinch(status: Any, entity: int, table: HitInterruptTable) -> Any:
    """Enraged (asset 8 s) starts when the flinch of a third hit ends: Recovery_Pending on, Flinch off, not yet Enraged."""

    pending = _has(status, entity, semantic_id("Hit_Interrupt_Recovery_Pending"))
    flinch = _has(status, entity, semantic_id("Hit_Interrupt_Flinch"))
    enraged = _has(status, entity, semantic_id("Hit_Interrupt_Enraged"))
    return write_status(status, entity, table.effects["Hit_Interrupt_Enraged"], pending & ~flinch & ~enraged)


def construct(original: Any, profile: Any) -> Any:
    """Not wired. The gen-7 hooks stay ``stub`` in the overlay, so the render never reaches this function. Gen 5 and an
    off switch return the original."""

    if profile.constructs.hit_interrupt is False:
        return original
    raise NotImplementedError("hit interrupt: the engine hooks are not wired (see the module docstring: open seams)")
