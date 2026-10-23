"""The event log and the activity counters (``docs/VIRTUAL_HUMAN.md`` section 21).

The **event log** is a short raw record of the discrete changes the agent registers: a health drop, an inventory count
that changed, a request that ended. Each event is marked **observed** (a count or a request's status, read directly) or
**inferred** (a health drop credited to an entity by ``state/affect.py``'s attribution, or a consumable's count falling,
taken as eating). The mark is provenance: an inferred hit is never taken as a fact until the simulator says who hit whom
(S11). The log is a ring of ``EVENT_SLOTS`` events per agent, the oldest giving way. It is not memory (21.2): an event
that rolls off the ring is gone from the log, and the memories that need it keep what they need on their own terms
(the owner's ruling, 2026-10-22).

The **counters** are lifetime totals per activity: the distance walked, sprinted, swum and climbed (blocks), damage
dealt and taken, items gained, and food eaten. They add up over the episode and survive a death (21.2). They will feed
the levels of section 16.6, whose proficiency curve ``p = 1 - exp(-N / tau)`` needs a time constant per activity that
is not set yet, so only the counts are kept here.

Every detector but the requests' compares ``state.last`` (last tick's observation) with ``observed`` (this tick's), and
none of those fires on the first tick of an episode (``seen`` False), since there is no change to register. A request's
outcome is reported by the runtime, not a change between ticks, so it is logged on any tick. With ``h`` my health
fraction, ``e`` an entity's health fraction, ``n`` an item count and ``d`` my position's change in blocks, the rules
are:

- **damage taken** (inferred when attributed): ``drop = h_last - h`` where ``h_last >= 0`` and ``drop > 0``. The amount
  is ``drop`` and the other party is the entity the caller attributes (-1 if none).
- **damage dealt** (inferred): for each entity perceived now and last tick, with ``drop = e_last - e > 0``, when my
  attack was executing within the last ``DEALT_WINDOW_TICKS`` [E] (``now - last_attack <= 4``, 0.18 s) and the entity
  is within ``THREAT_REACH_BLOCKS`` (the reach ``affect.py`` assumes for an attacker, an estimate there). The attack is
  stamped before this rule reads it, so a hit on the same tick counts.
- **items** (observed; per item id, the counts summed over the slots holding it): a gain ``total_now - total_last > 0``
  gives ``item_gained``. A loss ``total_last - total_now > 0`` gives ``ate`` when that slot was a consumable (inferred:
  a consumable's count dropping is taken as eating, 16.5), else ``item_lost``. A move between slots with the same total
  gives nothing.
- **requests** (observed): outcome 1, 2 or 3 gives ``request_done``, ``request_failed`` or ``request_aborted``, with the
  request's verb as its detail.
- **distance** (observed): a position change ``|d|`` above ``TELEPORT_BLOCKS`` [E] is a respawn or a teleport and
  counts nowhere. Otherwise, with ``h = |(dx, dz)|``, ``v = |dy|`` and ``full = |d|``: climbing adds ``v`` to the
  climbed distance; else swimming or in a fluid adds ``full`` to the swum; else running or sprinting adds ``h`` to the
  sprinted (under ``npc_walk`` a requested sprint is a run, 21.1, so there is no speed threshold); else walking adds
  ``h`` to the walked. The movement flags are this tick's.

The ring: each tick's candidates are written in a fixed order (the damage taken, each entity row's damage dealt, each
inventory slot's gain, each last-tick slot's loss, each request slot), and ``count`` of them are valid. A valid
candidate with rank ``r`` (its place among the valid ones) goes to slot ``(next + r) % EVENT_SLOTS``, and only the last
``EVENT_SLOTS`` of them are kept when more arrive in one tick; then ``next`` advances by ``count``.

Estimated constants are marked [E]: ``DEALT_WINDOW_TICKS`` and ``TELEPORT_BLOCKS``. ``EVENT_SLOTS`` is a capacity, a
token budget sized like the other capacities and changeable without retraining (21.5), not an estimate.

**Grounding** (21.5): the log is a short store that the memories read from, as in the multi-store model (Atkinson and
Shiffrin 1968). Events are the points where the registered features change, as in event segmentation (Zacks et al.
2007). The observed or inferred mark is source monitoring (Johnson, Hashtroudi and Lindsay 1993). The counters follow
Minecraft's lifetime statistics (distances, damage, items) and use-based progression, where a skill gains a little each
time it is used (Skyrim's "Leveling", UESP).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.state.affect import THREAT_REACH_BLOCKS

EVENT_SLOTS: int = 32
"""The ring's capacity per agent (21.2)."""
EVENT_KINDS: tuple[str, ...] = ("damage_taken", "damage_dealt", "item_gained", "item_lost", "ate", "request_done",
                                "request_failed", "request_aborted")
COUNTERS: tuple[str, ...] = ("distance_walked", "distance_sprinted", "distance_swum", "distance_climbed",
                             "damage_dealt", "damage_taken", "items_gained", "food_eaten")
MOVES: tuple[str, ...] = ("walking", "running", "sprinting", "swimming", "in_fluid", "climbing")
"""The columns of ``Observed.moving``: the self token's movement flags."""
OUTCOME_SLOTS: int = 4
"""The request outcomes per agent per tick: the combiner's active-request table holds 4 (``experimental/neural_agent/
docs/MIDDLE_LAYER.md`` section 5)."""
DEALT_WINDOW_TICKS: int = 4
"""An opponent's health drop counts as my hit when my attack was executing within this many ticks [E]: 0.18 s at
0.045 s a tick."""
TELEPORT_BLOCKS: float = 2.0
"""A position change larger than this in one tick is not movement (a respawn or a teleport) [E]."""
_NEVER: int = -1_000_000
"""``last_attack`` before any attack: far enough back that no swing is within the window."""
_KIND = {name: index for index, name in enumerate(EVENT_KINDS)}
_MOVE = {name: index for index, name in enumerate(MOVES)}
_OUTCOME_KIND = jnp.asarray([-1, _KIND["request_done"], _KIND["request_failed"], _KIND["request_aborted"]], jnp.int32)
"""The event kind of each request outcome code (0 none, 1 done, 2 failed, 3 aborted)."""


class Observed(NamedTuple):
    """What the caller reads from the agent's own observation each tick (all of it legal, 21.1), for B agents."""

    health: jax.Array  # float32[B]: my health fraction
    position: jax.Array  # float32[B, 3]: my world position, in blocks
    has_position: jax.Array  # bool[B]
    moving: jax.Array  # float32[B, len(MOVES)]: 0 or 1 per column of MOVES
    attack_executing: jax.Array  # bool[B]: my attack is executing this tick
    attributed: jax.Array  # int32[B]: the entity the affect rules credit with this tick's damage taken; -1 none
    entity_ids: jax.Array  # int32[B, N]: the perceived entities' ids; -1 padding
    entity_health: jax.Array  # float32[B, N]: each one's health fraction
    entity_distance: jax.Array  # float32[B, N]: each one's distance from me, in blocks
    item_ids: jax.Array  # int32[B, S]: the inventory slots' item semantic ids; 0 for an empty slot
    item_counts: jax.Array  # float32[B, S]
    item_consumable: jax.Array  # bool[B, S]
    outcomes: jax.Array  # int32[B, R]: this tick's request outcomes; 0 none, 1 done, 2 failed, 3 aborted
    outcome_verbs: jax.Array  # int32[B, R]: each request's verb id in ``neural_skills/spec.py`` ``VERBS`` (the
    # modules' append-only vocabulary, the ids the body's requests use); -1 for a verb outside it


class EventLog(NamedTuple):
    """The ring of B agents, E = ``EVENT_SLOTS`` slots each. An empty slot has kind -1 and tick -1."""

    kind: jax.Array  # int32[B, E]: an index into EVENT_KINDS; -1 for an empty slot
    tick: jax.Array  # int32[B, E]: the tick it happened
    other: jax.Array  # int32[B, E]: the other party's entity id; -1 none
    amount: jax.Array  # float32[B, E]: health as a fraction of maximum health, or an item count; 0 for requests
    detail: jax.Array  # int32[B, E]: the item's semantic id (item events and ``ate``), the verb (requests); else 0
    world: jax.Array  # float32[B, E, 3]: my position when it happened; zeros without a position
    inferred: jax.Array  # bool[B, E]: True for the inferred events (attributed damage, damage dealt, eating)
    next: jax.Array  # int32[B]: the slot the next event is written to


class EventState(NamedTuple):
    """The events of B agents: the log, the counters, and what the detectors compare against."""

    now: jax.Array  # int32[B]: ticks since each agent's episode began
    log: EventLog
    counters: jax.Array  # float32[B, len(COUNTERS)]: lifetime totals, in COUNTERS' order
    last: Observed  # last tick's observation (health -1 before the first tick)
    seen: jax.Array  # bool[B]: False until the first tick has been taken
    last_attack: jax.Array  # int32[B]: the tick my attack last executed; ``_NEVER`` before any


class Recent(NamedTuple):
    """The last ``count`` events of each agent, oldest first (B agents, ``count`` columns)."""

    kind: jax.Array  # int32[B, count]
    tick: jax.Array  # int32[B, count]
    other: jax.Array  # int32[B, count]
    amount: jax.Array  # float32[B, count]
    detail: jax.Array  # int32[B, count]
    world: jax.Array  # float32[B, count, 3]
    inferred: jax.Array  # bool[B, count]
    valid: jax.Array  # bool[B, count]: False where the ring holds no event there yet


def initial_events(batch: int, *, entities: int, slots: int, outcomes: int) -> EventState:
    """An empty log, zero counters and no tick yet. ``entities`` (N), ``slots`` (S) and ``outcomes`` (R) set the shapes
    of ``last``, which is all padding with health -1; ``seen`` is False and ``last_attack`` is ``_NEVER``."""

    empty = EventLog(
        kind=jnp.full((batch, EVENT_SLOTS), -1, jnp.int32),
        tick=jnp.full((batch, EVENT_SLOTS), -1, jnp.int32),
        other=jnp.full((batch, EVENT_SLOTS), -1, jnp.int32),
        amount=jnp.zeros((batch, EVENT_SLOTS), jnp.float32),
        detail=jnp.zeros((batch, EVENT_SLOTS), jnp.int32),
        world=jnp.zeros((batch, EVENT_SLOTS, 3), jnp.float32),
        inferred=jnp.zeros((batch, EVENT_SLOTS), jnp.bool_),
        next=jnp.zeros((batch,), jnp.int32),
    )
    last = Observed(
        health=jnp.full((batch,), -1.0, jnp.float32),
        position=jnp.zeros((batch, 3), jnp.float32),
        has_position=jnp.zeros((batch,), jnp.bool_),
        moving=jnp.zeros((batch, len(MOVES)), jnp.float32),
        attack_executing=jnp.zeros((batch,), jnp.bool_),
        attributed=jnp.full((batch,), -1, jnp.int32),
        entity_ids=jnp.full((batch, entities), -1, jnp.int32),
        entity_health=jnp.zeros((batch, entities), jnp.float32),
        entity_distance=jnp.zeros((batch, entities), jnp.float32),
        item_ids=jnp.zeros((batch, slots), jnp.int32),
        item_counts=jnp.zeros((batch, slots), jnp.float32),
        item_consumable=jnp.zeros((batch, slots), jnp.bool_),
        outcomes=jnp.zeros((batch, outcomes), jnp.int32),
        outcome_verbs=jnp.zeros((batch, outcomes), jnp.int32),
    )
    return EventState(jnp.zeros((batch,), jnp.int32), empty, jnp.zeros((batch, len(COUNTERS)), jnp.float32), last,
                      jnp.zeros((batch,), jnp.bool_), jnp.full((batch,), _NEVER, jnp.int32))


def _totals(query: jax.Array, ids: jax.Array, counts: jax.Array) -> jax.Array:
    """float32[B, S]: for each query slot, the counts summed over the slots of ``ids`` holding the same nonzero id."""

    match = (query[:, :, None] == ids[:, None, :]) & (ids[:, None, :] != 0)
    return jnp.sum(jnp.where(match, counts[:, None, :], 0.0), axis=2)


def _first(ids: jax.Array) -> jax.Array:
    """bool[B, S]: the slot is the first one holding its id."""

    slots = jnp.arange(ids.shape[1])
    same = ids[:, :, None] == ids[:, None, :]
    earlier = slots[None, :] < slots[:, None]  # [s, t]: slot t comes before slot s
    return ~jnp.any(same & earlier, axis=2)


def _block(valid, kind, other, amount, detail, world, inferred):
    """One detector's candidates as [B, K] columns (world [B, K, 3]); scalars and [B, 1] columns broadcast."""

    shape = valid.shape
    return (valid,
            jnp.broadcast_to(jnp.asarray(kind, jnp.int32), shape),
            jnp.broadcast_to(jnp.asarray(other, jnp.int32), shape),
            jnp.broadcast_to(jnp.asarray(amount, jnp.float32), shape),
            jnp.broadcast_to(jnp.asarray(detail, jnp.int32), shape),
            jnp.broadcast_to(world, shape + (3,)),
            jnp.broadcast_to(jnp.asarray(inferred, jnp.bool_), shape))


def update(state: EventState, observed: Observed, reset: jax.Array | None = None) -> EventState:
    """The state after this tick (module docstring). Lanes where ``reset`` bool[B] is True start from
    :func:`initial_events` first; then the detectors run on ``state.last`` against ``observed``, their events enter the
    ring, the counters add, and ``observed`` becomes ``last``."""

    batch = state.now.shape[0]
    if reset is not None:
        fresh = initial_events(batch, entities=state.last.entity_ids.shape[1], slots=state.last.item_ids.shape[1],
                               outcomes=state.last.outcomes.shape[1])
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(reset.reshape((batch,) + (1,) * (old.ndim - 1)), new, old), fresh, state)
    last, seen, log = state.last, state.seen, state.log
    now = state.now + 1
    last_attack = jnp.where(observed.attack_executing, now, state.last_attack)  # before the dealt rule reads it
    here = jnp.where(observed.has_position[:, None], observed.position, 0.0)

    drop = last.health - observed.health
    taken = seen & (last.health >= 0.0) & (drop > 0.0)
    taken_amount = jnp.where(taken, drop, 0.0)

    ids = observed.entity_ids
    same = (ids[:, :, None] == last.entity_ids[:, None, :]) & (ids[:, :, None] >= 0)
    dealt_drop = jnp.take_along_axis(last.entity_health, jnp.argmax(same, axis=2), axis=1) - observed.entity_health
    dealt = (seen[:, None] & jnp.any(same, axis=2) & (dealt_drop > 0.0)
             & ((now - last_attack)[:, None] <= DEALT_WINDOW_TICKS)
             & (observed.entity_distance <= THREAT_REACH_BLOCKS))
    dealt_amount = jnp.where(dealt, dealt_drop, 0.0)

    now_ids, now_counts = observed.item_ids, observed.item_counts
    old_ids, old_counts, old_consumable = last.item_ids, last.item_counts, last.item_consumable
    gain = _totals(now_ids, now_ids, now_counts) - _totals(now_ids, old_ids, old_counts)
    gained = seen[:, None] & (now_ids != 0) & _first(now_ids) & (gain > 0.0)
    gain_amount = jnp.where(gained, gain, 0.0)
    loss = _totals(old_ids, old_ids, old_counts) - _totals(old_ids, now_ids, now_counts)
    lost = seen[:, None] & (old_ids != 0) & _first(old_ids) & (loss > 0.0)
    eaten_amount = jnp.where(lost & old_consumable, loss, 0.0)

    outcomes = observed.outcomes
    requested = (outcomes >= 1) & (outcomes <= 3)  # reported by the runtime: logged on any tick, the first included

    delta = observed.position - last.position
    moved = jnp.linalg.norm(delta, axis=-1)
    horizontal = jnp.hypot(delta[:, 0], delta[:, 2])
    vertical = jnp.abs(delta[:, 1])
    moves = observed.moving > 0.5
    walk = moves[:, _MOVE["walking"]]
    run = moves[:, _MOVE["running"]] | moves[:, _MOVE["sprinting"]]
    swim = moves[:, _MOVE["swimming"]] | moves[:, _MOVE["in_fluid"]]
    climb = moves[:, _MOVE["climbing"]]
    real = seen & observed.has_position & last.has_position & (moved <= TELEPORT_BLOCKS)
    climbed = jnp.where(real & climb, vertical, 0.0)
    swum = jnp.where(real & ~climb & swim, moved, 0.0)
    sprinted = jnp.where(real & ~climb & ~swim & run, horizontal, 0.0)
    walked = jnp.where(real & ~climb & ~swim & ~run & walk, horizontal, 0.0)

    blocks = [
        _block(taken[:, None], _KIND["damage_taken"], observed.attributed[:, None], drop[:, None], 0, here[:, None],
               observed.attributed[:, None] >= 0),
        _block(dealt, _KIND["damage_dealt"], ids, dealt_drop, 0, here[:, None], True),
        _block(gained, _KIND["item_gained"], -1, gain, now_ids, here[:, None], False),
        _block(lost, jnp.where(old_consumable, _KIND["ate"], _KIND["item_lost"]), -1, loss, old_ids, here[:, None],
               old_consumable),
        _block(requested, _OUTCOME_KIND[jnp.clip(outcomes, 0, 3)], -1, 0.0, observed.outcome_verbs, here[:, None],
               False),
    ]
    valid, kind, other, amount, detail, world, inferred = (
        jnp.concatenate(column, axis=1) for column in zip(*blocks))
    count = jnp.sum(valid, axis=1, dtype=jnp.int32)
    rank = jnp.cumsum(valid, axis=1, dtype=jnp.int32) - 1
    keep = valid & (rank >= (count - EVENT_SLOTS)[:, None])
    slot = jnp.where(keep, (log.next[:, None] + rank) % EVENT_SLOTS, EVENT_SLOTS)  # out of range: dropped
    rows = jnp.arange(batch)[:, None]
    ring = EventLog(
        kind=log.kind.at[rows, slot].set(kind, mode="drop"),
        tick=log.tick.at[rows, slot].set(jnp.broadcast_to(now[:, None], kind.shape), mode="drop"),
        other=log.other.at[rows, slot].set(other, mode="drop"),
        amount=log.amount.at[rows, slot].set(amount, mode="drop"),
        detail=log.detail.at[rows, slot].set(detail, mode="drop"),
        world=log.world.at[rows, slot].set(world, mode="drop"),
        inferred=log.inferred.at[rows, slot].set(inferred, mode="drop"),
        next=(log.next + count) % EVENT_SLOTS,
    )

    increments = {
        "distance_walked": walked,
        "distance_sprinted": sprinted,
        "distance_swum": swum,
        "distance_climbed": climbed,
        "damage_dealt": jnp.sum(dealt_amount, axis=1),
        "damage_taken": taken_amount,
        "items_gained": jnp.sum(gain_amount, axis=1),
        "food_eaten": jnp.sum(eaten_amount, axis=1),
    }
    counters = state.counters + jnp.stack([increments[name] for name in COUNTERS], axis=1)
    return EventState(now, ring, counters, observed, jnp.ones_like(seen), last_attack)


def recent(state: EventState, count: int) -> Recent:
    """The last ``count`` events of each agent's ring, oldest first, newest last. ``valid`` is False where the ring
    holds fewer events than ``count`` (an empty slot has kind -1)."""

    if not 0 < count <= EVENT_SLOTS:
        raise ValueError(f"count must be in 1..{EVENT_SLOTS}, got {count}")
    batch = state.now.shape[0]
    log = state.log
    slots = (log.next[:, None] - count + jnp.arange(count)[None, :]) % EVENT_SLOTS
    rows = jnp.arange(batch)[:, None]
    kind = log.kind[rows, slots]
    return Recent(kind, log.tick[rows, slots], log.other[rows, slots], log.amount[rows, slots],
                  log.detail[rows, slots], log.world[rows, slots], log.inferred[rows, slots], kind >= 0)


__all__ = ["COUNTERS", "DEALT_WINDOW_TICKS", "EVENT_KINDS", "EVENT_SLOTS", "EventLog", "EventState", "MOVES",
           "OUTCOME_SLOTS", "Observed", "Recent", "TELEPORT_BLOCKS", "initial_events", "recent", "update"]
