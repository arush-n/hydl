"""Hit interrupt, stun and flinch (gen 7, row 5a; 0.7.2). Switch ``Constructs.hit_interrupt``; IMPLEMENTED.

Hooks (sim/gen7/overlay.py, rows ``hit_interrupt_*``; the switch-off path returns the original, except the hit wrapper, which
strips the loadout keyword the call-site patch adds):

- ``construct_roles`` on ``arsenal_runtime_config``: the config's ``hit_interrupt_eligible`` [entity] flags from ``entity_role_ids``
  (the measured eligible roles by default; the 22-role static reading of npc_roles is a comment only).
- ``construct_reset`` on ``reset_arsenal_batch``: copies the flags into ``CombatMechanicsState.hit_interrupt_eligible``.
- ``construct_clock`` on ``step_arsenal_batch``: before a step, keeps Hit_Interrupt_Eligible on the eligible entities from
  tick 42 (the step's microtick count, one per 1/30 s step at microticks 1) and refreshes it on the 0.25 s grid; Enraged starts
  when a flinch ends while Recovery_Pending holds.
- ``construct_hits`` on ``process_direct_events``: a requested direct hit on the target from a flagged attack of the agent's
  loadout (``ability_event_index`` -> ability range -> ``ability_id`` in the flagged set, sim/gen7 table) applies the gate chain
  and the effects.

Measured live (FIDELITY-0.7.2.md section (an), Part 1, 2026-10-20; receipts ``~/.claude/jobs/31947bd7/tmp/hit_interrupt_part1``):
Hit_Interrupt_Eligible on an eligible Skeleton_Fighter from env tick 42 (43 in some runs), refreshed every 0.25 s; a hit before
eligibility gives Red_Flash only; the first hit while eligible adds Stunned 0.7 s, Flinch 0.5 s and Recent_Reaction_A 4.0 s in one
tick; the flinch is removed 15 frames later and the stun 21.

Asset (``sim/gen7/data/hit_interrupt.json``, extracted by tools/content_extract/hit_interrupt.py): the eight effect durations,
overlap rules and horizontal speed multipliers, and the 31 attack interactions with ``DamageEntityParent_Hit_Interrupt``. Stunned
blocks attack starts (``STATUS_FLAG_DISABLE_ABILITIES``, asset comment); Flinch stops walking with gravity and knockback active
(speed multiplier 0 with ``STATUS_FLAG_DISABLE_MOVEMENT``, asset comment).

Removal timing: with ``Constructs.status_clock`` on (gen 7, ``mechanics/status_clock.py``, task #106) a status lasts round(D / tick)
ticks, so the construct writes Flinch, Stunned and Reaction A at their full durations and they are removed after 15, 21 and 120
ticks, the native frame counts. With it off (the stock clock, ``mechanics/runtime/statuses.py``: removed at <= 0 in float32) the
construct writes them at duration minus half a tick, which gives the same counts; the stock clock alone removes a 0.5 s status on
the 16th tick at 1/30 s.

Motion on the render (tests/sim/test_hit_interrupt.py): DISABLE_MOVEMENT alone does not move the target (a flag-only run moved on
step 1). The flinch's speed multiplier 0 does: arsenal/runtime.py _apply_target_controls scales the target's whole position change
and velocity by the status speed, so the target stands still while the flinch lasts (steps 53 to 61 after a hit on 47) and
moves again when it is removed. Gravity and knockback stop with it, unlike the asset comment. Recorded, not corrected: a
horizontal-only stop needs the walking part separated from knockback in the engine.

NOT MEASURED (asset values, unmeasured): Recent_Reaction_B 4 s on the second flinch; Recovery_Pending 2 s on the third; Enraged 8 s
when that flinch ends; Protected_Attack 8 s on attacks started while Enraged (no reader located; not written); the cancel of the
target's Attack-tagged root (not modelled); Red_Flash (not in the engine's status table; not modelled).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.mechanics.schema.contract import (
    DAMAGE_PHYSICAL,
    STATUS_FLAG_DISABLE_ABILITIES,
    STATUS_FLAG_DISABLE_MOVEMENT,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
)
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY

SWITCH = "hit_interrupt"
IMPLEMENTED = True
DATA_FILE = "hit_interrupt.json"
FAMILY = "hit_interrupt"
TABLE_SCHEMA = "hytalerl_content_table_v1"
#: The measured delay: Hit_Interrupt_Eligible is first added at env tick 42 (43 in the period-14 runs).
ELIGIBLE_DELAY_TICKS = 42
#: The role refreshes eligibility every 0.25 s (Hit_Interrupt_Eligible asset comment). At 30 Hz that is 7.5 ticks.
REFRESH_PERIOD_SECONDS = 0.25
TICKS_PER_SECOND = 30
HALF_TICK_SECONDS = 0.5 / TICKS_PER_SECOND
#: The arsenal's entity rows (agent, target); the role flags are per row.
ENTITY_COUNT = 2
#: The measured eligible roles. The 22 roles whose reference chain reaches Component_Instruction_Hit_Interrupt (a static reading
#: of npc_roles) are candidates only; Skeleton_Fighter and Goblin_Scrapper are the two measured.
MEASURED_ELIGIBLE_ROLES = frozenset({"Skeleton_Fighter", "Goblin_Scrapper"})
#: Effect name -> the mechanics flag the engine honours for it (asset comments). Other effects carry no flag.
EFFECT_FLAGS = {
    "Hit_Interrupt_Stunned": STATUS_FLAG_DISABLE_ABILITIES,
    "Hit_Interrupt_Flinch": STATUS_FLAG_DISABLE_MOVEMENT,
}
#: Written at duration minus half a tick when the stock clock runs (``status_clock`` off; see the module docstring).
PRECOMPENSATED = ("Hit_Interrupt_Flinch", "Hit_Interrupt_Stunned", "Hit_Interrupt_Recent_Reaction_A")
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


def flagged_ability_ids(table: HitInterruptTable) -> tuple[int, ...]:
    """Semantic ids of the attack roots whose damage row is flagged, as the arsenal names them (the row minus ``_Damage``).

    Ability rows (Ability_Fireball_Hit and the like) carry no ``_Damage`` suffix and no arsenal root of that name is
    checked here, so they are not in the set.
    """

    return tuple(sorted({semantic_id(name[: -len(ATTACK_SUFFIX)]) for name in table.attacks if name.endswith(ATTACK_SUFFIX)}))


def flagged_events(ability_id: Any, ability_event_start: Any, ability_event_count: Any, event_index: Any, flagged: Any) -> Any:
    """bool[B, F]: the fired event sits in an ability whose id is flagged.

    ``ability_id``, ``ability_event_start`` and ``ability_event_count`` are [A] (one loadout) or [B, A]; ``event_index`` is
    [B, F] (``FiredEvents.ability_event_index``); ``flagged`` is the array of ``flagged_ability_ids``."""

    ability_id = jnp.asarray(ability_id, jnp.int32)
    start = jnp.asarray(ability_event_start, jnp.int32)
    count = jnp.asarray(ability_event_count, jnp.int32)
    if ability_id.ndim == 1:
        ability_id, start, count = ability_id[None], start[None], count[None]
    event = jnp.asarray(event_index, jnp.int32)
    in_range = (event[..., None] >= start[:, None, :]) & (event[..., None] < (start + count)[:, None, :])
    is_flagged = jnp.any(ability_id[..., None] == jnp.asarray(flagged, jnp.int32), axis=-1)
    return jnp.any(in_range & is_flagged[:, None, :], axis=-1)


def landed_flagged(loadout: Any, fired: Any, damage: Any, flagged: Any) -> Any:
    """bool[B]: a requested direct hit on the target from a flagged attack of the agent (the landed approximation: the cone
    request of ``process_direct_events``; blocks and invulnerability are not resolved here)."""

    batch = fired.requested.shape[0]
    flat = jax.tree_util.tree_map(lambda value: value.reshape((batch, -1) + value.shape[3:]), fired)
    # The loadout arrays are per entity ([batch, entity, ability] in the config; the agent's row is AGENT_ENTITY).
    attack = flagged_events(loadout.ability_id[..., AGENT_ENTITY, :], loadout.ability_event_start[..., AGENT_ENTITY, :],
                            loadout.ability_event_count[..., AGENT_ENTITY, :], flat.ability_event_index, flagged)
    from_agent = flat.source_entity_id == AGENT_ENTITY
    on_target = damage.target_entity_id == TARGET_ENTITY
    return jnp.any(damage.requested & attack & from_agent & on_target, axis=-1)


def role_mask(role_ids: Any, eligible_roles: frozenset[str]) -> np.ndarray:
    """bool[ENTITY_COUNT]: the entity rows whose role id is in ``eligible_roles`` (no role ids: none eligible)."""

    roles = list(role_ids or ())[:ENTITY_COUNT]
    roles += [None] * (ENTITY_COUNT - len(roles))
    return np.asarray([role in eligible_roles for role in roles], dtype=np.bool_)


# ---- status slots (the rows of the 0.7.2 status table, written by the construct) ------------------------------------

def _has(status: Any, entity: int, effect_id: int) -> Any:
    return jnp.any(status.active[:, entity] & (status.effect_id[:, entity] == jnp.int32(effect_id)), axis=-1)


def write_status(status: Any, entity: int, effect: Effect, apply: Any) -> Any:
    """Write ``effect`` into ``entity``'s slots where ``apply`` [B] is true (an overlap-aware insert).

    ``Overwrite`` refreshes an active slot of the same effect; ``Ignore`` leaves it. A new effect takes the first free slot.
    The stock applier (``apply_statuses``) rejects a speed multiplier of 0 (statuses.py:208), so Flinch is written here.
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
        "source_entity_id": AGENT_ENTITY,
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
    """Keep ``Hit_Interrupt_Eligible`` on ``entity`` from the measured tick on (``tick`` [B]: the microtick the step runs; ``eligible``
    [B]: the entity's role is in the eligible set). It is added at ``ELIGIBLE_DELAY_TICKS`` and refreshed when the 0.25 s grid
    (7.5 ticks at 30 Hz, index ``floor(2 * tick / 15)``) advances."""

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


def _status_clock_on(profile: Any) -> bool:
    """The half-tick pre-compensation is for the stock clock; the status_clock construct rounds on its own, so it must not get it."""

    return profile.constructs.status_clock is True


def _effect(table: HitInterruptTable, name: str, precompensate: bool = True) -> Effect:
    effect = table.effects[name]
    if precompensate and name in PRECOMPENSATED:
        return replace(effect, duration_seconds=effect.duration_seconds - HALF_TICK_SECONDS)
    return effect


def apply_hit(status: Any, entity: int, landed: Any, table: HitInterruptTable, precompensate: bool = True) -> Any:
    """A landed hit on ``entity`` (``landed`` [B]). Where the gate holds: Stunned, Flinch and a reaction. The reaction is
    Recent_Reaction_A on the first hit, B on the second, and the third queues Recovery_Pending (asset values; unmeasured)."""

    gate = landed & hit_gate(status, entity)
    a_present = _has(status, entity, semantic_id("Hit_Interrupt_Recent_Reaction_A"))
    b_present = _has(status, entity, semantic_id("Hit_Interrupt_Recent_Reaction_B"))
    status = write_status(status, entity, _effect(table, "Hit_Interrupt_Stunned", precompensate), gate)
    status = write_status(status, entity, _effect(table, "Hit_Interrupt_Flinch", precompensate), gate)
    status = write_status(status, entity, _effect(table, "Hit_Interrupt_Recent_Reaction_A", precompensate), gate & ~a_present)
    status = write_status(status, entity, _effect(table, "Hit_Interrupt_Recent_Reaction_B", precompensate), gate & a_present & ~b_present)
    return write_status(status, entity, _effect(table, "Hit_Interrupt_Recovery_Pending", precompensate), gate & a_present & b_present)


def enrage_after_flinch(status: Any, entity: int, table: HitInterruptTable) -> Any:
    """Enraged (asset 8 s) starts when the flinch of a third hit ends: Recovery_Pending on, Flinch off, not yet Enraged."""

    pending = _has(status, entity, semantic_id("Hit_Interrupt_Recovery_Pending"))
    flinch = _has(status, entity, semantic_id("Hit_Interrupt_Flinch"))
    enraged = _has(status, entity, semantic_id("Hit_Interrupt_Enraged"))
    return write_status(status, entity, table.effects["Hit_Interrupt_Enraged"], pending & ~flinch & ~enraged)


def clock_update(status: Any, tick: Any, eligible: Any, table: HitInterruptTable) -> Any:
    """One step's clock for every entity row: Eligible (``eligible`` [B, ENTITY_COUNT]) and the flinch-end Enraged."""

    for entity in range(ENTITY_COUNT):
        status = eligible_update(status, entity, tick, eligible[:, entity], table)
        status = enrage_after_flinch(status, entity, table)
    return status


# ---- hooks ------------------------------------------------------------------------------------------------------------

def _enabled(profile: Any) -> bool:
    value = profile.constructs.hit_interrupt
    if value is False:
        return False
    if value is not True:
        raise ValueError(f"hit_interrupt must be a bool, got {value!r}")
    return True


def construct_roles(original: Any, profile: Any) -> Any:
    """Hook for ``arsenal_runtime_config``: the config's per-entity eligible-role flags from ``entity_role_ids``."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def arsenal_runtime_config(*args: Any, **kwargs: Any) -> Any:
        # The builder's role catalog rejects a name it has no initial statuses for (role_statuses.py:125); the measured
        # eligible roles have none, so they are passed as "" (no role) and the flags come from the names given here.
        roles = kwargs.get("entity_role_ids")
        mask = role_mask(roles, MEASURED_ELIGIBLE_ROLES)
        if roles is not None:
            kwargs = dict(kwargs, entity_role_ids=tuple("" if role in MEASURED_ELIGIBLE_ROLES else role for role in roles))
        config = original(*args, **kwargs)
        return config._replace(hit_interrupt_eligible=jnp.asarray(mask))

    return arsenal_runtime_config


def construct_reset(original: Any, profile: Any) -> Any:
    """Hook for ``reset_arsenal_batch``: copy the config's eligible flags into the mechanics state."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def reset_arsenal_batch(keys: Any, params: Any, config: Any, *args: Any, **kwargs: Any) -> Any:
        state, observation = original(keys, params, config, *args, **kwargs)
        mask = config.hit_interrupt_eligible
        if mask is None:
            return state, observation
        batch = state.combat.tick_count.shape[0]
        eligible = jnp.broadcast_to(jnp.asarray(mask, jnp.bool_)[None, :], (batch, ENTITY_COUNT))
        state = state._replace(mechanics=state.mechanics._replace(hit_interrupt_eligible=eligible))
        return state, observation

    return reset_arsenal_batch


def construct_clock(original: Any, profile: Any) -> Any:
    """Hook for ``step_arsenal_batch``: before the step, the clock for the microtick the step runs (tick = count + 1)."""

    if not _enabled(profile):
        return original
    table = load_table(profile.data_dir)

    @wraps(original)
    def step_arsenal_batch(state: Any, actions: Any, keys: Any, params: Any, commands: Any, config: Any, *args: Any,
                           **kwargs: Any) -> Any:
        if config.hit_interrupt_eligible is not None:
            tick = state.combat.tick_count + 1
            batch = tick.shape[0]
            eligible = jnp.broadcast_to(jnp.asarray(config.hit_interrupt_eligible, jnp.bool_)[None, :], (batch, ENTITY_COUNT))
            statuses = clock_update(state.mechanics.statuses, tick, eligible, table)
            state = state._replace(mechanics=state.mechanics._replace(statuses=statuses))
        return original(state, actions, keys, params, commands, config, *args, **kwargs)

    return step_arsenal_batch


def construct_hits(original: Any, profile: Any) -> Any:
    """Hook for ``process_direct_events``: a flagged direct hit on the target applies the gate chain and the effects. With the
    switch off the loadout keyword the call-site patch passes is stripped and the original runs."""

    if not _enabled(profile):
        @wraps(original)
        def process_direct_events(*args: Any, loadout: Any = None, **kwargs: Any) -> Any:
            return original(*args, **kwargs)

        return process_direct_events
    table = load_table(profile.data_dir)
    flagged = flagged_ability_ids(table)

    @wraps(original)
    def process_direct_events(combat: Any, mechanics: Any, arsenal: Any, events: Any, world: Any, params: Any, rules: Any,
                              *, engagement_target_id: Any = None, loadout: Any = None) -> Any:
        out = original(combat, mechanics, arsenal, events, world, params, rules, engagement_target_id=engagement_target_id)
        if loadout is None:
            return out
        mechanics_out, arsenal_out, health, damage, event_count = out
        landed = landed_flagged(loadout, events, damage, flagged)
        statuses = apply_hit(mechanics_out.statuses, TARGET_ENTITY, landed, table,
                             precompensate=not _status_clock_on(profile))
        return mechanics_out._replace(statuses=statuses), arsenal_out, health, damage, event_count

    return process_direct_events
