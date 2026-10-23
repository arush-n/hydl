"""The neural layer's stored memory: the spatial map and the salient store (``docs/PERCEPTION.md`` sections 10.1 and
10.3; owner, 2026-10-20 night: "store important information and spatial information separately ... these important
tags should be able to be passed down by the middle layer").

Two stores, both fixed-capacity arrays in the neural layer's carry, anchored to the world by the agent's own position
(the perception channel's ``self_position``) so that what was seen stays where it was while the agent moves:

- **the spatial map** (``geometry``): the geometry cells the agent has seen. Kept by recency; when full, the oldest
  cell gives way. Lossy is acceptable: space is bulk.
- **the salient store** (``entity`` and ``block``): the things that matter, with identity. Entities are keyed by their
  id, blocks by their cell. When full, the least important gives way: a tagged thing is pinned; then salience (a
  hostile entity; an ore, door or stateful block) and recency decide.

Each tick :func:`update` writes this tick's perceived rows into the stores (an upsert: a known key is refreshed, a new
one takes a free slot or the least important slot), and :func:`read` gives back every stored row that was not
perceived this tick, re-expressed in the current body frame with its ``where`` block, ``remembered`` 1, its ``age``
and its ``tagged`` priority (an entity also reads ``visible`` 0). Columns that are directions in the body frame (an
entity's velocity and facing, a cell's traversal edges) are stored in world axes and turned into the current body
frame on reading. Everything else reads as last perceived, including an entity's facing toward me and its velocity
relative to mine at that moment (the gym scales entity and self velocity by different speeds, so the agent's own
motion since cannot be taken out here: ``docs/PERCEPTION.md`` section 11).

**Tags.** The middle layer marks a thing as important with a ``remember`` request (a scope-less verb). The runtime
lowers the live tags to :class:`Tags` each tick: an entity id, or a world position, with a priority. A tagged entity or
block is pinned while its tag lives and reads ``tagged`` with its priority; a tagged position the agent has not seen
becomes a placeholder block row (content unknown), so the layer remembers the place it was told about.

Without the agent's position (no perception channel: stock gen 5, or gen 7 before the coordinator's S1) the stores
cannot anchor: :func:`update` leaves them unchanged and :func:`read` gives nothing.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.state.senses import (
    ENTITY_CENTRE_HEIGHT,
    Perceived,
    Pose,
    Remembered,
    where_of,
)
from experimental.neural_agent.skill_agent.state.frames import (
    POSITION_SCALE_BLOCKS,
    body_to_world,
    world_to_body,
)

GEOMETRY_SLOTS: int = v3.CAPACITIES_V3["geometry"] - 44
ENTITY_SLOTS: int = 16
BLOCK_SLOTS: int = v3.CAPACITIES_V3["block_candidate"] - 16
HORIZON_TICKS: int = 1800
"""``age`` saturates here (60 s at 30 Hz)."""
SAME_CELL_BLOCKS: float = 0.25
"""Two positions closer than this on every axis are the same cell."""
_PINNED, _FREE = 1e9, -1e9
_REFERENCE_HEIGHT = {"entity": ENTITY_CENTRE_HEIGHT, "geometry": 0.0, "block_candidate": 0.5}
_SALIENT_BLOCK = tuple(f"affordance_{name}" for name in ("ore", "door", "stateful", "harvestable"))
_COLUMNS = {name: {feature: index for index, feature in enumerate(cls.FEATURES)}
            for name, cls in v3.TOKEN_TYPES_V3.items()}
_VECTORS = {"entity": (("v_forward", "v_right", "v_up"),),
            "geometry": tuple((f"edge_{edge}_to_forward", f"edge_{edge}_to_right", f"edge_{edge}_to_up")
                              for edge in range(v3.GEOMETRY_EDGES))}
"""Columns beyond ``where`` that are vectors in the body frame: kept in world axes while stored."""
_ANGLES = {"entity": (("yaw_sin", "yaw_cos"),)}
"""Columns that are a heading relative to the body: kept as a world heading while stored."""


class Store(NamedTuple):
    """One store, per agent and slot: the key (a world position; an entity id, or -1), the row as last perceived, the
    tick it was last perceived, whether the slot holds anything, its middle-layer tag, its salience, and the thing's
    own latent in the trunk (its object slot, ``docs/VIRTUAL_HUMAN.md`` section 18: kept while the slot holds the same
    thing, zero for a new one; width 0 without object slots)."""

    world: jax.Array  # float32[B, C, 3]
    key_id: jax.Array  # int32[B, C]
    rows: jax.Array  # float32[B, C, F]
    tick: jax.Array  # int32[B, C]
    valid: jax.Array  # bool[B, C]
    tag: jax.Array  # float32[B, C]
    salience: jax.Array  # float32[B, C]
    latent: jax.Array  # float32[B, C, L]


class SpatialMemory(NamedTuple):
    """The stores of B agents and their clocks (ticks since each agent's episode began)."""

    now: jax.Array  # int32[B]
    geometry: Store
    entity: Store
    block: Store


class Tags(NamedTuple):
    """The middle layer's live tags of B agents, K each: an entity id (-1 for none) or a world position (with its
    presence), and a priority (0 for an empty slot)."""

    entity_id: jax.Array  # int32[B, K]
    world: jax.Array  # float32[B, K, 3]
    world_present: jax.Array  # bool[B, K]
    priority: jax.Array  # float32[B, K]


def _empty_store(batch: int, slots: int, type_name: str, latent_width: int = 0) -> Store:
    width = len(v3.TOKEN_TYPES_V3[type_name].FEATURES)
    return Store(jnp.zeros((batch, slots, 3), jnp.float32), jnp.full((batch, slots), -1, jnp.int32),
                 jnp.zeros((batch, slots, width), jnp.float32), jnp.zeros((batch, slots), jnp.int32),
                 jnp.zeros((batch, slots), jnp.bool_), jnp.zeros((batch, slots), jnp.float32),
                 jnp.zeros((batch, slots), jnp.float32), jnp.zeros((batch, slots, latent_width), jnp.float32))


def initial_memory(batch: int, *, geometry_slots: int = GEOMETRY_SLOTS, entity_slots: int = ENTITY_SLOTS,
                   block_slots: int = BLOCK_SLOTS, latent_width: int = 0) -> SpatialMemory:
    """Empty stores at an episode's start. ``latent_width``: the trunk's width when it keeps object slots
    (``EncoderConfig.object_types``), else 0; the important stores (entities and blocks) hold one latent a slot, the
    spatial map none."""

    return SpatialMemory(jnp.zeros((batch,), jnp.int32), _empty_store(batch, geometry_slots, "geometry"),
                         _empty_store(batch, entity_slots, "entity", latent_width),
                         _empty_store(batch, block_slots, "block_candidate", latent_width))


def no_tags(batch: int, slots: int = 1) -> Tags:
    """No live tag."""

    return Tags(jnp.full((batch, slots), -1, jnp.int32), jnp.zeros((batch, slots, 3), jnp.float32),
                jnp.zeros((batch, slots), jnp.bool_), jnp.zeros((batch, slots), jnp.float32))


def _upsert(store: Store, rows, valid, world, key_id, salience, now, by_id: bool) -> Store:
    """Write N candidate rows into one agent's store (vmapped over agents) in one parallel pass (2026-10-22,
    ``docs/VIRTUAL_HUMAN.md`` section 18.4: the serial write's cost grew with the rows a tick can carry and the store's
    size, not with how many are seen; on real Region rows this one is 6 to 8 times faster).

    A candidate whose key is stored refreshes that slot. New candidates take the free slots first, then the least
    important unpinned ones (salience, then age), the most salient candidate first. When candidates compete:
    - two with the same key this tick: the later one is written (as the serial write left it);
    - a slot refreshed this tick, or pinned by a tag, is never taken;
    - more new candidates than slots: the least salient are not written.
    With room for every new candidate, the result is the serial write's, slot for slot."""

    capacity = store.valid.shape[0]
    if by_id:
        same = store.valid[None, :] & (store.key_id[None, :] == key_id[:, None])
        twin = key_id[:, None] == key_id[None, :]
    else:
        same = store.valid[None, :] & jnp.all(jnp.abs(store.world[None, :, :] - world[:, None, :]) < SAME_CELL_BLOCKS,
                                              axis=-1)
        twin = jnp.all(jnp.abs(world[:, None, :] - world[None, :, :]) < SAME_CELL_BLOCKS, axis=-1)
    superseded = jnp.any(jnp.triu(twin & valid[None, :], k=1), axis=1)  # a later candidate has the same key
    live = valid & ~superseded
    known = live & same.any(axis=1)
    fresh = live & ~known
    refreshed = jnp.any(same & known[:, None], axis=0)
    score = jnp.where(~store.valid, _FREE, jnp.where(store.tag > 0, _PINNED,
                                                     store.salience * HORIZON_TICKS - (now - store.tick)))
    score = jnp.where(refreshed, _PINNED, score)
    slots = jnp.argsort(score, stable=True)  # the slots new candidates take, in order
    room = jnp.sum(score < _PINNED)
    by_importance = jnp.argsort(jnp.where(fresh, -salience, jnp.inf), stable=True)
    importance = jnp.zeros(valid.shape, jnp.int32).at[by_importance].set(jnp.arange(valid.shape[0], dtype=jnp.int32))
    placed = fresh & (importance < room)  # without room for all, the most salient
    rank = jnp.cumsum(placed) - 1  # placed candidates take the slots in their own order
    take = slots[jnp.clip(rank, 0, capacity - 1)]
    target = jnp.where(known, jnp.argmax(same, axis=1), jnp.where(placed, take, -1))
    write = target[:, None] == jnp.arange(capacity)[None, :]  # [N, C]: candidate -> slot, one at most each
    hit = write.any(axis=0)
    source = jnp.argmax(write, axis=0)  # the candidate each written slot takes

    def put(new, old):
        return jnp.where(hit.reshape((-1,) + (1,) * (old.ndim - 1)), new[source], old)

    keep_tag = jnp.where(refreshed, store.tag, 0.0)
    new_thing = hit & ~refreshed  # its object latent starts at zero; a refreshed thing keeps its own
    return Store(put(world, store.world), put(key_id, store.key_id), put(rows, store.rows),
                 jnp.where(hit, now, store.tick), store.valid | hit, jnp.where(hit, keep_tag, store.tag),
                 put(salience, store.salience), jnp.where(new_thing[:, None], 0.0, store.latent))


def _world_of(rows: jax.Array, pose: Pose) -> jax.Array:
    """The world positions float32[B, N, 3] of rows whose ``where`` block leads (blocks from the feet)."""

    body = rows[..., :3] * jnp.float32(POSITION_SCALE_BLOCKS)
    return pose.position[:, None, :] + body_to_world(body, pose.body_yaw[:, None])


def _turn(rows: jax.Array, key: str, body_yaw: jax.Array, to_world: bool) -> jax.Array:
    """``rows`` float32[B, N, F] of type ``key`` with their body-frame vectors and headings (``_VECTORS``,
    ``_ANGLES``) turned into world axes (``to_world``) or back into the body frame of ``body_yaw`` float32[B], so a
    stored row reads in the current body frame after the agent turns."""

    columns = _COLUMNS[key]
    yaw = body_yaw[:, None]
    for names in _VECTORS.get(key, ()):
        index = jnp.asarray([columns[name] for name in names])
        turn = body_to_world if to_world else world_to_body
        rows = rows.at[..., index].set(turn(rows[..., index], yaw))
    for sine, cosine in _ANGLES.get(key, ()):
        heading = jnp.arctan2(rows[..., columns[sine]], rows[..., columns[cosine]]) + (yaw if to_world else -yaw)
        rows = rows.at[..., columns[sine]].set(jnp.sin(heading)).at[..., columns[cosine]].set(jnp.cos(heading))
    return rows


def _tagged(store: Store, tags: Tags, by_id: bool) -> jax.Array:
    """Each slot's tag priority float32[B, C]: the largest live tag that names it (0 when none does)."""

    if by_id:
        match = (store.key_id[:, :, None] == tags.entity_id[:, None, :]) & (tags.entity_id[:, None, :] >= 0)
    else:
        match = tags.world_present[:, None, :] & jnp.all(
            jnp.abs(store.world[:, :, None, :] - tags.world[:, None, :, :]) < SAME_CELL_BLOCKS, axis=-1)
    return jnp.max(jnp.where(match & store.valid[:, :, None], tags.priority[:, None, :], 0.0), axis=-1)


def update(memory: SpatialMemory, perceived: Perceived, tags: Tags | None = None,
           reset: jax.Array | None = None) -> SpatialMemory:
    """The stores after this tick: lanes where ``reset`` bool[B] is True start empty; the clock advances; this tick's
    perceived geometry, entities and block candidates are written; the live tags are applied (an unseen tagged place
    becomes a placeholder block row). Without the agent's position the stores are returned unchanged but the clock."""

    batch = memory.now.shape[0]
    if reset is not None:
        fresh = initial_memory(batch, geometry_slots=memory.geometry.valid.shape[1],
                               entity_slots=memory.entity.valid.shape[1], block_slots=memory.block.valid.shape[1],
                               latent_width=memory.entity.latent.shape[-1])
        memory = jax.tree_util.tree_map(
            lambda new, old: jnp.where(reset.reshape((batch,) + (1,) * (old.ndim - 1)), new, old), fresh, memory)
    now = memory.now + 1
    pose = perceived.pose
    if pose.position is None:
        return memory._replace(now=now)
    upsert = jax.vmap(_upsert, in_axes=(0, 0, 0, 0, 0, 0, 0, None))
    stores = {}
    for name, key in (("geometry", "geometry"), ("entity", "entity"), ("block", "block_candidate")):
        found = perceived.rows.get(key)
        store = getattr(memory, name)
        if found is not None:
            rows, valid = found
            count = rows.shape[1]
            ids = perceived.entity_ids[:, :count] if name == "entity" else jnp.full(valid.shape, -1, jnp.int32)
            salience = _salience(name, rows)
            world = _world_of(rows, pose)
            rows = _turn(rows, key, pose.body_yaw, to_world=True)
            store = upsert(store, rows, valid, world, ids, salience, now, name == "entity")
        stores[name] = store
    if tags is not None:
        stores["entity"] = stores["entity"]._replace(tag=_tagged(stores["entity"], tags, True))
        block = stores["block"]
        unseen = tags.world_present & (tags.priority > 0) & ~jnp.any(
            block.valid[:, :, None] & jnp.all(jnp.abs(block.world[:, :, None, :] - tags.world[:, None, :, :])
                                              < SAME_CELL_BLOCKS, axis=-1), axis=1)
        placeholder = jnp.zeros(tags.priority.shape + (block.rows.shape[-1],), jnp.float32)
        # Only when some tagged place is unseen: with none, the write leaves the store as it was, and skipping it saves
        # about a quarter of the update at batch 64 (docs/VIRTUAL_HUMAN.md 19.8).
        block = jax.lax.cond(jnp.any(unseen), lambda b: upsert(b, placeholder, unseen, tags.world,
                                                                jnp.full(unseen.shape, -1, jnp.int32),
                                                                jnp.ones(unseen.shape, jnp.float32), now, False),
                             lambda b: b, block)
        stores["block"] = block._replace(tag=_tagged(block, tags, False))
    return SpatialMemory(now, stores["geometry"], stores["entity"], stores["block"])


def _salience(store: str, rows: jax.Array) -> jax.Array:
    """float32[B, N]: an entity's hostility (the registry's attitude, or the engine's flag while the role is unknown),
    a block's ore, door, stateful or harvestable affordance; geometry 0."""

    if store == "entity":
        columns = _COLUMNS["entity"]
        return jnp.maximum(rows[..., columns["attitude_hostile"]], rows[..., columns["hostile"]])
    if store == "block":
        return sum(rows[..., _COLUMNS["block_candidate"][name]] for name in _SALIENT_BLOCK)
    return jnp.zeros(rows.shape[:2], jnp.float32)


def object_slots(memory: SpatialMemory, tokens: dict, entity_ids: jax.Array, pose: Pose) -> dict[str, jax.Array]:
    """Each important token row's slot in its store, where its object latent lives (``docs/VIRTUAL_HUMAN.md`` section
    18): entity rows by id (``entity_ids`` int32[B, E], aligned with them), block-candidate rows by cell, perceived or
    remembered; -1 for a row the stores do not hold (and every row without the agent's position)."""

    store = memory.entity
    match = (store.valid[:, None, :] & (store.key_id[:, None, :] == entity_ids[:, :, None])
             & (entity_ids[:, :, None] >= 0))
    out = {"entity": jnp.where(match.any(axis=-1), jnp.argmax(match, axis=-1), -1).astype(jnp.int32)}
    rows, valid = tokens["block_candidate"]
    if pose.position is None:
        out["block_candidate"] = jnp.full(valid.shape, -1, jnp.int32)
        return out
    world, block = _world_of(rows, pose), memory.block
    match = valid[:, :, None] & block.valid[:, None, :] & jnp.all(
        jnp.abs(block.world[:, None, :, :] - world[:, :, None, :]) < SAME_CELL_BLOCKS, axis=-1)
    out["block_candidate"] = jnp.where(match.any(axis=-1), jnp.argmax(match, axis=-1), -1).astype(jnp.int32)
    return out


def read(memory: SpatialMemory, pose: Pose) -> Remembered | None:
    """Every stored row not perceived this tick, in the current body frame (module docstring); None without the
    agent's position."""

    if pose.position is None:
        return None
    out = {}
    for name, key in (("entity", "entity"), ("geometry", "geometry"), ("block", "block_candidate")):
        store = getattr(memory, name)
        body = world_to_body(store.world - pose.position[:, None, :], pose.body_yaw[:, None])
        rows = _turn(store.rows, key, pose.body_yaw, to_world=False)
        rows = rows.at[..., :len(v3.WHERE_FEATURES)].set(where_of(body, pose, _REFERENCE_HEIGHT[key]))
        columns = _COLUMNS[key]
        age = jnp.minimum(jnp.log1p((memory.now[:, None] - store.tick).astype(jnp.float32)) / np.log1p(HORIZON_TICKS),
                          1.0)
        rows = rows.at[..., columns["remembered"]].set(1.0).at[..., columns["age"]].set(age)
        rows = rows.at[..., columns["tagged"]].set(store.tag)
        if key == "entity":
            rows = rows.at[..., columns["visible"]].set(0.0)
        mask = store.valid & (store.tick < memory.now[:, None])
        out[key] = (jnp.where(mask[..., None], rows, 0.0), mask)
    return Remembered(out["entity"], out["geometry"], out["block_candidate"])


__all__ = ["BLOCK_SLOTS", "ENTITY_SLOTS", "GEOMETRY_SLOTS", "HORIZON_TICKS", "SAME_CELL_BLOCKS", "SpatialMemory",
           "object_slots",
           "Store", "Tags", "initial_memory", "no_tags", "read", "update"]
