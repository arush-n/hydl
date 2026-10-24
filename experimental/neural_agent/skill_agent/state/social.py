"""The interaction facts of the people the agent knows (``docs/VIRTUAL_HUMAN.md`` section 26, step 9): one record per
individual it has perceived, with what was seen of it and what passed between them.

The relationship the decision layer writes about someone (closeness, trust, feelings, roles; ``state/written.py``) is a
written entry and lives there. This module holds the **facts**. Rules compute them from the senses and the event log,
and they are never written (15.3: "interaction facts and reputation are computed from events, not written"):

- **met:** the first and the last tick it was perceived, and where it was last perceived;
- **meetings:** how many separate meetings (a new one begins when it is perceived again after ``MEETING_GAP_SECONDS``
  unseen), and how many ticks it has been in view;
- **fights:** the hits taken from it and dealt to it, counts and amounts, from the event log's ``damage_taken`` and
  ``damage_dealt``, each inferred until the simulator says who hit whom (S11); and who struck first in the current
  meeting;
- **spared:** the chances to strike me that it did not take (a chance is seen in reach and facing me, and pauses while
  it is out of sight; see :func:`update`);
- **the last interaction:** its tick and the event's kind.

The table holds ``SOCIAL_SLOTS`` records per agent. When it is full, a newcomer replaces the record seen longest ago
among those with no written entry: the time since the last contact is the strongest single sign of a weak tie, and the
outer layers turn over fastest (26.6; Sutcliffe, Dunbar, Binder and Arrow 2012; MacCarron, Kaski and Dunbar, as cited in
search results). A record with a written entry is never replaced (19.7). When every record has one, the newcomer is not
recorded, and ``unrecorded`` counts it. Like the places (22.5), the table survives a respawn and a save and load, and an
episode boundary clears it, since the perceived ids restart then.

:func:`space_target` is the personal-space distance a body keeps from a person (26.2): Hall's zones by the closeness
band, pushed out toward the public distance by hostility or fear. It is a rule over the written relationship and the
feelings, not a record.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.state import affect as feeling
from experimental.neural_agent.skill_agent.state import clock, events

SOCIAL_SLOTS: int = 150
"""The records per agent: Dunbar's outer layer (section 26.2, 15.3); a capacity, not an estimate."""
MEETING_GAP_SECONDS: float = 60.0
"""Unseen this long, and the next sighting is a new meeting [E] (section 26.2); turned into ticks by the world's tick
(section 27)."""
STARTED: tuple[str, ...] = ("me", "them", "both")
"""Who struck first in the current meeting: ``started`` holds -1 (no blow yet this meeting) or an index here."""
CLOSENESS_EDGES: tuple[float, ...] = (1 / 6, 1 / 3, 1 / 2)
"""Where the closeness bands start: below 1/6 distant, 1/6 to 1/3 rather close, 1/3 to 1/2 close, above 1/2 very close
(section 26.2; Humanoid Agents' 5, 10 and 15 on a scale of 30, scaled to [0, 1])."""
SPACE_ZONES: tuple[float, ...] = (2.85, 1.65, 0.98, 0.6)
"""The personal-space distance of each band, in blocks [E], from distant to very close: the midpoints of Hall's social
far zone (2.1 to 3.6 m), social near, personal far and personal near (Hall 1966, through secondary sources)."""
SPACE_PUBLIC_BLOCKS: float = 5.6
"""The distance that hostility or fear pushes toward, in blocks [E]: past Hall's social zone, into the public one."""
SPACE_SCALE_BLOCKS: float = 7.6
"""The distance is read over this many blocks [E]: the public zone's far edge (section 26.2)."""
SPARE_REACH_BLOCKS: float = feeling.THREAT_REACH_BLOCKS
"""The reach of a chance to strike me: the reach the emotion rules assume for an attacker."""
SPARE_FACING_DEGREES: float = 45.0
"""A chance faces me within this angle [E]: between its facing and the direction from it to me."""
MIN_CHANCE_SECONDS: float = 0.25
"""A chance counts as spared only if it lasted this long [E] (section 26.8): about a person's simple reaction time to a
visual signal (recalled: 0.2 to 0.25 s), so not striking within less is not a choice. Pauses count toward it."""
_TAKEN: int = events.EVENT_KINDS.index("damage_taken")
_DEALT: int = events.EVENT_KINDS.index("damage_dealt")
_ME, _THEM, _BOTH = (STARTED.index(name) for name in ("me", "them", "both"))
_NOT_A_CANDIDATE: int = 2**30
"""The replacement order's key for a record that cannot be replaced this tick: after every candidate."""


class Records(NamedTuple):
    """The records of B agents, S = ``SOCIAL_SLOTS`` each. An empty slot is not valid."""

    key_id: jax.Array  # int32[B, S]: the perceived entity id; -1 for an empty slot
    first: jax.Array  # int32[B, S]: the tick it was first perceived; -1 for an empty slot
    last: jax.Array  # int32[B, S]: the tick it was last perceived; -1 for an empty slot
    world: jax.Array  # float32[B, S, 3]: where it was last perceived, in world blocks
    world_known: jax.Array  # bool[B, S]
    meetings: jax.Array  # int32[B, S]: the separate meetings with it
    meeting_start: jax.Array  # int32[B, S]: the tick the current meeting began
    ticks_in_view: jax.Array  # int32[B, S]: the ticks it has been perceived
    hits_taken: jax.Array  # int32[B, S]: the hits it gave me (inferred until S11)
    hits_dealt: jax.Array  # int32[B, S]: the hits I gave it (inferred until S11)
    damage_taken: jax.Array  # float32[B, S]: their amounts, as the event log's (fractions of health)
    damage_dealt: jax.Array  # float32[B, S]
    started: jax.Array  # int32[B, S]: who struck first in the current meeting: -1, or an index in STARTED
    last_interaction: jax.Array  # int32[B, S]: the tick of the last event with it as the other party; -1 none
    last_kind: jax.Array  # int32[B, S]: that event's index in events.EVENT_KINDS; -1 none
    spared: jax.Array  # int32[B, S]: the completed chances it had to strike me and did not take
    chance_open: jax.Array  # bool[B, S]: a chance is open now (it is in reach and facing me)
    chance_hit: jax.Array  # bool[B, S]: a hit from it landed during the open chance
    chance_start: jax.Array  # int32[B, S]: the tick the open chance began (its length counts pauses); -1 if none
    valid: jax.Array  # bool[B, S]


class SocialState(NamedTuple):
    """The records of B agents, and the newcomers not recorded because every record had a written entry."""

    records: Records
    unrecorded: jax.Array  # int32[B]


def _empty(batch: int, slots: int) -> Records:
    zeros = jnp.zeros((batch, slots), jnp.int32)
    minus = jnp.full((batch, slots), -1, jnp.int32)
    floats = jnp.zeros((batch, slots), jnp.float32)
    return Records(key_id=minus, first=minus, last=minus, world=jnp.zeros((batch, slots, 3), jnp.float32),
                   world_known=jnp.zeros((batch, slots), jnp.bool_), meetings=zeros, meeting_start=zeros,
                   ticks_in_view=zeros, hits_taken=zeros, hits_dealt=zeros, damage_taken=floats, damage_dealt=floats,
                   started=minus, last_interaction=minus, last_kind=minus, spared=zeros,
                   chance_open=jnp.zeros((batch, slots), jnp.bool_), chance_hit=jnp.zeros((batch, slots), jnp.bool_),
                   chance_start=jnp.full((batch, slots), -1, jnp.int32),
                   valid=jnp.zeros((batch, slots), jnp.bool_))


def initial_social(batch: int, slots: int = SOCIAL_SLOTS) -> SocialState:
    """The table at an episode's start: no records and no newcomer refused."""

    return SocialState(_empty(batch, slots), jnp.zeros((batch,), jnp.int32))


def _match(records: Records, ids: jax.Array) -> jax.Array:
    """bool[B, N, S]: the valid record that each id (int32[B, N]) is; -1 matches nothing."""

    return records.valid[:, None, :] & (records.key_id[:, None, :] == ids[:, :, None]) & (ids[:, :, None] >= 0)


def lookup(state: SocialState, ids: jax.Array) -> jax.Array:
    """int32[B, N]: the record slot of each id (int32[B, N]), -1 where there is none."""

    match = _match(state.records, ids)
    return jnp.where(jnp.any(match, axis=2), jnp.argmax(match, axis=2), -1).astype(jnp.int32)


def update(state: SocialState, seen_ids: jax.Array, seen_world: jax.Array | None, seen_chance: jax.Array,
           logged: events.EventState, written_ids: jax.Array, reset: jax.Array | None = None,
           tick_seconds: float = clock.DEFAULT_TICK_SECONDS, seen_alive: jax.Array | None = None) -> SocialState:
    """The table after this tick (module docstring).

    A blow from it during a chance, on the tick the chance opens included, takes the chance: only an unhit one counts.

    Args:
        state: Last tick's table.
        seen_ids: int32[B, N], this tick's perceived entity ids (not the remembered ones), -1 padding.
        seen_world: float32[B, N, 3], their world positions; None without a position channel: the positions stay, and a
            new record's ``world_known`` is False.
        seen_chance: bool[B, N], aligned with ``seen_ids``: this tick the person is alive, in reach and facing me.
        logged: The event state after this tick's ``events.update``. Its ``now`` is this tick, and this tick's events
            are the ring entries stamped with it.
        written_ids: int32[B, K], the keys of the live written entity entries (-1 padding): their records are kept.
        reset: bool[B], the lanes that start empty first (an episode boundary). A respawn changes nothing.
        tick_seconds: The world's tick (section 27), which turns ``MEETING_GAP_SECONDS`` and ``MIN_CHANCE_SECONDS``
            into ticks.
        seen_alive: bool[B, N], aligned with ``seen_ids``: the person is seen alive (None: every one is). A chance that
            closes because the person is seen dead is not counted: it did not choose.
    """

    batch, slots = state.records.key_id.shape
    if reset is not None:
        fresh = initial_social(batch, slots)
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(reset.reshape((batch,) + (1,) * (old.ndim - 1)), new, old), fresh, state)
    rec, now = state.records, logged.now[:, None]

    # Sightings, in parallel: each record perceived this tick is matched by its id.
    match = _match(rec, seen_ids)  # bool[B, N, S]
    seen = jnp.any(match, axis=1)  # bool[B, S]
    new_meeting = seen & (now - rec.last > clock.about(MEETING_GAP_SECONDS, tick_seconds))
    if seen_world is None:
        world, world_known = rec.world, rec.world_known
    else:
        row = jnp.argmax(match, axis=1)  # int32[B, S]: the row that matched
        at = jnp.take_along_axis(seen_world, jnp.broadcast_to(row[..., None], row.shape + (3,)), axis=1)
        world, world_known = jnp.where(seen[..., None], at, rec.world), rec.world_known | seen
    rec = rec._replace(
        last=jnp.where(seen, now, rec.last), ticks_in_view=rec.ticks_in_view + seen.astype(jnp.int32),
        meetings=rec.meetings + new_meeting.astype(jnp.int32),
        meeting_start=jnp.where(new_meeting, now, rec.meeting_start),
        started=jnp.where(new_meeting, -1, rec.started), world=world, world_known=world_known)

    # Newcomers: free slots first, then the valid records not seen this tick and not written, oldest sighting first.
    newcomer = (seen_ids >= 0) & ~jnp.any(match, axis=2)  # bool[B, N]
    held = rec.valid & jnp.any((rec.key_id[:, :, None] == written_ids[:, None, :]) & (written_ids[:, None, :] >= 0),
                               axis=2)
    replaceable = rec.valid & ~seen & ~held
    rank_key = jnp.where(~rec.valid, -1, jnp.where(replaceable, rec.last, _NOT_A_CANDIDATE))
    order = jnp.argsort(rank_key, axis=1, stable=True)  # int32[B, S]: the slots in the order they are taken
    candidates = jnp.sum(~rec.valid | replaceable, axis=1, dtype=jnp.int32)
    rank = jnp.cumsum(newcomer, axis=1, dtype=jnp.int32) - 1  # the newcomer's place among the newcomers
    placed = newcomer & (rank < candidates[:, None])
    target = jnp.where(placed, jnp.take_along_axis(order, jnp.clip(rank, 0, slots - 1), axis=1), -1)
    unrecorded = state.unrecorded + jnp.sum(newcomer & ~placed, axis=1, dtype=jnp.int32)
    takes = target[:, :, None] == jnp.arange(slots)[None, None, :]  # bool[B, N, S]: newcomer -> slot, one at most each
    hit = jnp.any(takes, axis=1)  # bool[B, S]
    source = jnp.argmax(takes, axis=1)  # int32[B, S]: the newcomer row that takes each slot
    minus = jnp.full((batch, slots), -1, jnp.int32)
    zeros = jnp.zeros((batch, slots), jnp.int32)
    floats = jnp.zeros((batch, slots), jnp.float32)
    falses = jnp.zeros((batch, slots), jnp.bool_)
    taken_world = jnp.zeros((batch, slots, 3), jnp.float32) if seen_world is None else jnp.take_along_axis(
        seen_world, jnp.broadcast_to(source[..., None], (batch, slots, 3)), axis=1)
    fresh = Records(
        key_id=jnp.take_along_axis(seen_ids, source, axis=1), first=jnp.broadcast_to(now, (batch, slots)),
        last=jnp.broadcast_to(now, (batch, slots)), world=taken_world,
        world_known=jnp.full((batch, slots), seen_world is not None, jnp.bool_),
        meetings=jnp.ones_like(zeros), meeting_start=jnp.broadcast_to(now, (batch, slots)),
        ticks_in_view=jnp.ones_like(zeros), hits_taken=zeros, hits_dealt=zeros, damage_taken=floats,
        damage_dealt=floats, started=minus, last_interaction=minus, last_kind=minus, spared=zeros,
        chance_open=falses, chance_hit=falses, chance_start=zeros - 1, valid=jnp.ones((batch, slots), jnp.bool_))
    rec = jax.tree_util.tree_map(
        lambda new, old: jnp.where(hit.reshape(hit.shape + (1,) * (old.ndim - hit.ndim)), new, old), fresh, rec)

    # This tick's events with a known other party, matched to the records (a newcomer's included).
    log = logged.log
    this_tick = (log.tick == now) & (log.other >= 0)  # bool[B, E]
    pair = rec.valid[:, None, :] & (rec.key_id[:, None, :] == log.other[:, :, None]) & this_tick[:, :, None]
    taken = pair & (log.kind == _TAKEN)[:, :, None]
    dealt = pair & (log.kind == _DEALT)[:, :, None]
    rec = rec._replace(
        hits_taken=rec.hits_taken + jnp.sum(taken, axis=1, dtype=jnp.int32),
        damage_taken=rec.damage_taken + jnp.sum(jnp.where(taken, log.amount[:, :, None], 0.0), axis=1),
        hits_dealt=rec.hits_dealt + jnp.sum(dealt, axis=1, dtype=jnp.int32),
        damage_dealt=rec.damage_dealt + jnp.sum(jnp.where(dealt, log.amount[:, :, None], 0.0), axis=1))
    got, gave = jnp.any(taken, axis=1), jnp.any(dealt, axis=1)
    blow = jnp.where(got & gave, _BOTH, jnp.where(got, _THEM, jnp.where(gave, _ME, -1)))
    rec = rec._replace(started=jnp.where(rec.started == -1, blow, rec.started))
    # The last event with it, in this tick's order: the ring writes a tick's events oldest first, from ``next`` back.
    interacted = jnp.any(pair, axis=1)
    count = jnp.sum(this_tick, axis=1, dtype=jnp.int32)
    order_in_tick = (jnp.arange(events.EVENT_SLOTS)[None, :] - (log.next - count)[:, None]) % events.EVENT_SLOTS
    latest = jnp.argmax(jnp.where(pair, order_in_tick[:, :, None], -1), axis=1)  # int32[B, S]
    rec = rec._replace(
        last_interaction=jnp.where(interacted, now, rec.last_interaction),
        last_kind=jnp.where(interacted, jnp.take_along_axis(log.kind, latest, axis=1), rec.last_kind))
    # The chances to strike me (``spared``), after the sightings and the events. A chance opens on a tick the person is
    # seen in reach and facing me (a newcomer's included) and closes on a tick it is seen and is not. While the person
    # is out of sight the chance pauses: losing sight of it is my doing, not its choice. A hit from it during the chance
    # (its first tick and a pause included) means the chance was taken: it closes unspared. A chance left open from an
    # earlier meeting is dropped uncounted. It counts only if it lasted ``MIN_CHANCE_SECONDS`` (pauses included) and the
    # person is seen alive as it closes (section 26.8).
    seen_now = seen | hit  # perceived this tick, a newcomer included
    chance_seen = (jnp.any(match & seen_chance[:, :, None], axis=1)
                   | (hit & jnp.take_along_axis(seen_chance, source, axis=1)))
    was_open, was_hit = rec.chance_open & ~new_meeting & ~hit, rec.chance_hit & ~new_meeting & ~hit
    chance = jnp.where(seen_now, chance_seen, was_open)
    struck = was_hit | ((was_open | chance) & got)
    alive = seen_now if seen_alive is None else (jnp.any(match & seen_alive[:, :, None], axis=1)
                                                 | (hit & jnp.take_along_axis(seen_alive, source, axis=1)))
    lasted = now - rec.chance_start >= clock.at_least(MIN_CHANCE_SECONDS, tick_seconds)  # read where it was open
    rec = rec._replace(spared=rec.spared + (was_open & ~chance & ~struck & alive & lasted).astype(jnp.int32),
                       chance_open=chance, chance_hit=struck & chance,
                       chance_start=jnp.where(chance, jnp.where(was_open, rec.chance_start, now), -1))
    return SocialState(rec, unrecorded)


def space_target(closeness: jax.Array, hostility: jax.Array, fear: jax.Array) -> jax.Array:
    """float32, in blocks, elementwise: the personal-space distance a body keeps from a person (section 26.2). The
    closeness band gives the zone (``SPACE_ZONES``, by ``CLOSENESS_EDGES``); the larger of hostility and fear, clipped
    to [0, 1], moves it toward ``SPACE_PUBLIC_BLOCKS``: ``d = zone + (SPACE_PUBLIC_BLOCKS - zone) h``."""

    closeness = jnp.asarray(closeness, jnp.float32)
    zone = jnp.asarray(SPACE_ZONES, jnp.float32)[
        jnp.sum(closeness[..., None] >= jnp.asarray(CLOSENESS_EDGES, jnp.float32), axis=-1)]
    h = jnp.clip(jnp.maximum(jnp.asarray(hostility, jnp.float32), jnp.asarray(fear, jnp.float32)), 0.0, 1.0)
    return zone + (SPACE_PUBLIC_BLOCKS - zone) * h


__all__ = ["CLOSENESS_EDGES", "MEETING_GAP_SECONDS", "MIN_CHANCE_SECONDS", "Records", "SOCIAL_SLOTS",
           "SPACE_PUBLIC_BLOCKS", "SPACE_SCALE_BLOCKS", "SPACE_ZONES", "SPARE_FACING_DEGREES", "SPARE_REACH_BLOCKS",
           "STARTED", "SocialState", "initial_social", "lookup", "space_target", "update"]
