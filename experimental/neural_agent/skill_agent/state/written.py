"""The top two layers' structured writes into the agent's state (``docs/VIRTUAL_HUMAN.md`` section 19, rulings 19.7).

A **write** is one record from the decision layer or the LLM (:class:`Write`): a writer, a declared field, a value in
the field's range, and a key when the field is about someone or somewhere. :func:`lower` checks each write on the host
against the registry (``contract/variables.py``) before it enters the tick. It refuses a write, with a reason that goes
back to the writer, when the field is unknown, the writer is not the field's writer, the value is out of range, the key
is missing or of the wrong kind, or the tick has no slot left for it (``WRITE_SLOTS`` per agent).

Written entries live in two fixed-size tables per agent: the entries about an entity (``ENTITY_ENTRY_SLOTS``, keyed by
the entity id) and the entries about a place (``PLACE_ENTRY_SLOTS``, keyed by world position; two positions within
``memory.SAME_CELL_BLOCKS`` on every axis are the same place). Each entry's fields are declared by name in
``variables.ENTITY_ENTRIES`` and ``variables.PLACE_ENTRIES``.

:func:`apply` runs between ticks, at the start of a tick, before the stores update. The tick's writes apply in slot
order, so a later write to the same field wins. A write about a new key takes the first free entry; when none is free
the write is refused as full and nothing is evicted. An entry stays until it is ejected: a write that sets every field
back to its default frees it. Each entry keeps the writer and the tick of its last write. Nothing fades, and the state
layer never drops an entry on its own (19.7, 1). No written scalar is declared yet (19.7, 2), so a keyless write is
refused as an unknown field.

:func:`tags` derives the live ``remember`` entries as the stores' tags (``state/memory.py``): the stores pin a
remembered thing and read it as ``tagged``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract import variables
from experimental.neural_agent.skill_agent.state import memory

ENTITY_ENTRY_SLOTS: int = 16
PLACE_ENTRY_SLOTS: int = 16
WRITE_SLOTS: int = 8
"""Writes per agent per tick. A passing write beyond these is refused as "too many writes this tick"."""
_ENTITY_DEFAULTS = np.asarray([v.default for v in variables.ENTITY_ENTRIES], np.float32)
_PLACE_DEFAULTS = np.asarray([v.default for v in variables.PLACE_ENTRIES], np.float32)
_ENTITY_NAMES = tuple(v.name for v in variables.ENTITY_ENTRIES)
_PLACE_NAMES = tuple(v.name for v in variables.PLACE_ENTRIES)


@dataclass(frozen=True)
class Write:
    """One write as a layer emits it, before the host check (:func:`lower`). ``entity`` or ``place`` gives its key;
    neither makes it keyless."""

    writer: str
    name: str
    value: float
    entity: int | None = None
    place: tuple[float, float, float] | None = None


class Writes(NamedTuple):
    """One tick's writes of B agents, W = ``WRITE_SLOTS`` slots each, the unused slots padding (table 0)."""

    table: jax.Array  # int32[B, W]: 0 padding, 1 an entity entry, 2 a place entry
    field: jax.Array  # int32[B, W]: the field's index in that table's declared list
    entity: jax.Array  # int32[B, W]: the entity id; -1 unless an entity write
    place: jax.Array  # float32[B, W, 3]: the place's world position; zeros unless a place write
    value: jax.Array  # float32[B, W]
    writer: jax.Array  # int32[B, W]: the writer's index in ``variables.WRITERS``


class Entries(NamedTuple):
    """One table of entries of B agents, K slots each, C fields each."""

    key_id: jax.Array  # int32[B, K]: the entity id; -1 for an empty entry, and always for a place
    world: jax.Array  # float32[B, K, 3]: a place's world position; zeros otherwise
    values: jax.Array  # float32[B, K, C]: the fields, in the table's declared order
    written_at: jax.Array  # int32[B, K]: the tick of the last write; -1 for an empty entry
    writer: jax.Array  # int32[B, K]: the index in ``variables.WRITERS`` of the last writer; -1 for an empty entry
    valid: jax.Array  # bool[B, K]


class Written(NamedTuple):
    """The written state of B agents."""

    now: jax.Array  # int32[B]: ticks since each agent's episode began
    entity: Entries  # the entries about entities
    place: Entries  # the entries about places
    applied: jax.Array  # bool[B, W]: which of this tick's writes took effect (False for padding and a refusal as full)


def _empty(batch: int, slots: int, defaults: np.ndarray) -> Entries:
    return Entries(jnp.full((batch, slots), -1, jnp.int32), jnp.zeros((batch, slots, 3), jnp.float32),
                   jnp.broadcast_to(jnp.asarray(defaults), (batch, slots, len(defaults))),
                   jnp.full((batch, slots), -1, jnp.int32), jnp.full((batch, slots), -1, jnp.int32),
                   jnp.zeros((batch, slots), jnp.bool_))


def initial_written(batch: int) -> Written:
    """The written state at an episode's start: no entries, every field at its default, the clock at 0."""

    return Written(jnp.zeros((batch,), jnp.int32), _empty(batch, ENTITY_ENTRY_SLOTS, _ENTITY_DEFAULTS),
                   _empty(batch, PLACE_ENTRY_SLOTS, _PLACE_DEFAULTS), jnp.zeros((batch, WRITE_SLOTS), jnp.bool_))


def no_writes(batch: int) -> Writes:
    """Every slot padding: no writes this tick."""

    return Writes(jnp.zeros((batch, WRITE_SLOTS), jnp.int32), jnp.zeros((batch, WRITE_SLOTS), jnp.int32),
                  jnp.full((batch, WRITE_SLOTS), -1, jnp.int32), jnp.zeros((batch, WRITE_SLOTS, 3), jnp.float32),
                  jnp.zeros((batch, WRITE_SLOTS), jnp.float32), jnp.zeros((batch, WRITE_SLOTS), jnp.int32))


def _declared(write: Write) -> tuple[int, tuple[variables.Variable, ...]]:
    """The table a write goes to (1 entity, 2 place) and its declared fields. A keyless write has none: the registry's
    written scalars, of which there are none yet (19.7, 2)."""

    if write.entity is not None:
        return 1, variables.ENTITY_ENTRIES
    if write.place is not None:
        return 2, variables.PLACE_ENTRIES
    return 0, ()


def _check(write: Write) -> str | None:
    """The refusal reason of ``write``, or None when it passes the host check."""

    if write.entity is not None and write.place is not None:
        return "two keys"
    table, declared = _declared(write)
    names = [field.name for field in declared]
    if write.name not in names:
        return "unknown field"
    field = declared[names.index(write.name)]
    if write.writer != field.writer:
        return "wrong writer"
    value = float(write.value)
    if not (math.isfinite(value) and field.low <= value <= field.high):
        return "out of range"
    if table == 1:
        key_ok = isinstance(write.entity, numbers.Integral) and not isinstance(write.entity, bool) and write.entity >= 0
    else:
        place = np.asarray(write.place, np.float64)
        key_ok = place.shape == (3,) and bool(np.isfinite(place).all())
    return None if key_ok else "bad key"


def lower(writes: Sequence[Sequence[Write]]) -> tuple[Writes, list[list[str | None]]]:
    """The device writes of B agents' writes this tick (agent i's writes in its slots, in order, padding after them),
    and each input write's refusal reason (None when it passes). A refused write takes no slot."""

    batch = len(writes)
    table = np.zeros((batch, WRITE_SLOTS), np.int32)
    field = np.zeros((batch, WRITE_SLOTS), np.int32)
    entity = np.full((batch, WRITE_SLOTS), -1, np.int32)
    place = np.zeros((batch, WRITE_SLOTS, 3), np.float32)
    value = np.zeros((batch, WRITE_SLOTS), np.float32)
    writer = np.zeros((batch, WRITE_SLOTS), np.int32)
    reasons: list[list[str | None]] = []
    for agent, agent_writes in enumerate(writes):
        slot = 0
        agent_reasons: list[str | None] = []
        for write in agent_writes:
            reason = _check(write)
            if reason is None and slot == WRITE_SLOTS:
                reason = "too many writes this tick"
            agent_reasons.append(reason)
            if reason is not None:
                continue
            which, declared = _declared(write)
            table[agent, slot] = which
            field[agent, slot] = [item.name for item in declared].index(write.name)
            if which == 1:
                entity[agent, slot] = write.entity
            else:
                place[agent, slot] = write.place
            value[agent, slot] = write.value
            writer[agent, slot] = variables.WRITERS.index(write.writer)
            slot += 1
        reasons.append(agent_reasons)
    device = Writes(jnp.asarray(table), jnp.asarray(field), jnp.asarray(entity), jnp.asarray(place),
                    jnp.asarray(value), jnp.asarray(writer))
    return device, reasons


def _write_table(entries: Entries, match: jax.Array, defaults: np.ndarray, field: jax.Array, value: jax.Array,
                 writer: jax.Array, now: jax.Array, active: jax.Array, key: jax.Array | int, world: jax.Array
                 ) -> tuple[Entries, jax.Array]:
    """One write to one table of one agent (``match`` bool[K]: the valid entries it is about). Returns the table and
    whether the write applied: to a matching entry; or, with none matching, ejecting an absent entry (nothing changes)
    or taking the first free entry; refused only when the table is full. An entry whose fields all equal their
    defaults is freed."""

    slots = entries.valid.shape[0]
    known = jnp.any(match)
    free = ~entries.valid
    ejects = value == jnp.asarray(defaults)[field]
    target = jnp.where(known, jnp.argmax(match), jnp.argmax(free))
    write = active & (known | (~ejects & jnp.any(free)))
    applied = active & (known | ejects | jnp.any(free))
    chosen = (jnp.arange(slots) == target) & write
    base = jnp.where(known, entries.values[target], jnp.asarray(defaults))
    values = base.at[field].set(value)
    gone = jnp.all(values == jnp.asarray(defaults))
    key_now = jnp.where(known, entries.key_id[target], key)
    world_now = jnp.where(known, entries.world[target], world)

    def pick(new, old):  # the chosen slot takes ``new`` (a scalar or a row), the others keep ``old``
        return jnp.where(chosen.reshape(chosen.shape + (1,) * (old.ndim - 1)), new, old)

    return Entries(pick(jnp.where(gone, -1, key_now), entries.key_id),
                   pick(jnp.where(gone, 0.0, world_now), entries.world),
                   pick(jnp.where(gone, jnp.asarray(defaults), values), entries.values),
                   pick(jnp.where(gone, -1, now), entries.written_at),
                   pick(jnp.where(gone, -1, writer), entries.writer),
                   pick(~gone, entries.valid)), applied


def _apply_agent(entity: Entries, place: Entries, now: jax.Array, writes: Writes
                 ) -> tuple[Entries, Entries, jax.Array]:
    """One agent's writes in slot order, and which took effect (vmapped over agents by :func:`apply`)."""

    def slot(tables, write):
        entity_table, place_table = tables
        table, field, ident, position, value, writer = write
        near = jnp.all(jnp.abs(place_table.world - position) < memory.SAME_CELL_BLOCKS, axis=-1)
        entity_table, entity_ok = _write_table(entity_table, entity_table.valid & (entity_table.key_id == ident),
                                               _ENTITY_DEFAULTS, field, value, writer, now, table == 1, ident,
                                               jnp.zeros((3,), jnp.float32))
        place_table, place_ok = _write_table(place_table, place_table.valid & near, _PLACE_DEFAULTS, field, value,
                                             writer, now, table == 2, -1, position)
        return (entity_table, place_table), entity_ok | place_ok

    (entity, place), applied = jax.lax.scan(
        slot, (entity, place), (writes.table, writes.field, writes.entity, writes.place, writes.value, writes.writer))
    return entity, place, applied


def apply(written: Written, writes: Writes | None, reset: jax.Array | None = None) -> Written:
    """The written state after this tick: lanes where ``reset`` bool[B] is True start from :func:`initial_written`; the
    clock advances by one; then each agent's writes apply in slot order (module docstring). ``writes=None``: none."""

    batch = written.now.shape[0]
    if reset is not None:
        fresh = initial_written(batch)
        written = jax.tree_util.tree_map(
            lambda new, old: jnp.where(reset.reshape((batch,) + (1,) * (old.ndim - 1)), new, old), fresh, written)
    writes = no_writes(batch) if writes is None else writes
    now = written.now + 1
    entity, place, applied = jax.vmap(_apply_agent)(written.entity, written.place, now, writes)
    return Written(now, entity, place, applied)


def tags(written: Written) -> memory.Tags:
    """The live ``remember`` entries as the stores' tags (``state/memory.py``), K = ``ENTITY_ENTRY_SLOTS`` +
    ``PLACE_ENTRY_SLOTS`` slots: the entity entries first, then the place entries, in slot order. An entry is live when
    it is valid and its ``remember`` is above 0; a dead slot has id -1, no position, priority 0."""

    entity, place = written.entity, written.place
    live_entity = entity.valid & (entity.values[..., _ENTITY_NAMES.index("remember")] > 0)
    live_place = place.valid & (place.values[..., _PLACE_NAMES.index("remember")] > 0)
    return memory.Tags(
        jnp.concatenate([jnp.where(live_entity, entity.key_id, -1), jnp.full(live_place.shape, -1, jnp.int32)], axis=1),
        jnp.concatenate([jnp.zeros_like(entity.world), jnp.where(live_place[..., None], place.world, 0.0)], axis=1),
        jnp.concatenate([jnp.zeros_like(live_entity), live_place], axis=1),
        jnp.concatenate([jnp.where(live_entity, entity.values[..., _ENTITY_NAMES.index("remember")], 0.0),
                         jnp.where(live_place, place.values[..., _PLACE_NAMES.index("remember")], 0.0)], axis=1))


def _read(entries: Entries, match: jax.Array, defaults: np.ndarray) -> tuple[jax.Array, jax.Array]:
    """float32[B, C] and bool[B]: the matching entry's fields (the defaults when none matches) and whether one matches.
    At most one entry matches, since an entity or a cell holds one entry."""

    found = jnp.any(match, axis=1)
    values = jnp.sum(jnp.where(match[..., None], entries.values, 0.0), axis=1)
    return jnp.where(found[:, None], values, jnp.asarray(defaults)), found


def entity_entry(written: Written, ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The entry about each agent's entity ``ids`` int32[B]: its fields float32[B, C] and whether one is held
    (bool[B])."""

    entries = written.entity
    return _read(entries, entries.valid & (entries.key_id == ids[:, None]), _ENTITY_DEFAULTS)


def place_entry(written: Written, positions: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The entry about each agent's place ``positions`` float32[B, 3]: its fields float32[B, C] and whether it is held
    (bool[B]), a cell being the same as another within ``memory.SAME_CELL_BLOCKS`` on every axis."""

    entries = written.place
    near = jnp.all(jnp.abs(entries.world - positions[:, None, :]) < memory.SAME_CELL_BLOCKS, axis=-1)
    return _read(entries, entries.valid & near, _PLACE_DEFAULTS)


__all__ = ["ENTITY_ENTRY_SLOTS", "Entries", "PLACE_ENTRY_SLOTS", "WRITE_SLOTS", "Write", "Writes", "Written", "apply",
           "entity_entry", "initial_written", "lower", "no_writes", "place_entry", "tags"]
