"""Hearing, the first slice (``docs/VIRTUAL_HUMAN.md`` sections 17.2 and 17.8, on the inputs of section 28; S12-a, 2026-10-24).

The simulator publishes this tick's positional footsteps as ``heard_sounds`` on the learner observation, when the gen-7
construct ``hearing`` is on (``sim/gen7/mechanics/heard_sounds.py``). Without the field nothing is heard. This module
applies the hearing rule of 17.2 to them and keeps a ring of recent sounds per agent (:class:`HearingState`):

1. threshold: a sound is heard only at a gain of ``HEARING_THRESHOLD`` or more;
2. masking: within a tick, a sound is not heard when a louder one within ``MASK_DEGREES`` of its bearing is more than
   ``MASK_RATIO`` times louder;
3. kind: the sound's registry row (``perception/sounds.py``: its description and category); the unknown row when the
   registry does not hold it, and the sound is still heard;
4. direction: the bearing in the body frame, blurred the way a person's is (the spread grows from
   ``BLUR_FRONT_DEGREES`` straight ahead to ``BLUR_SIDE_DEGREES`` at the side), the elevation blurred coarsely;
5. judged distance: a power function of the true distance (Zahorik 2002); only the judged distance reaches a token;
6. who: known when the simulator says the source is an entity the agent perceives (``source_entity_id`` at least 0);
7. repeats: a sound of the same kind from the same source, heard again within ``MERGE_SECONDS``, merges into its entry
   (the same entity when both are known, else a bearing from the listener within ``MERGE_DEGREES``), which takes the new
   judged position, loudness, last tick and count; the rate is ``count - 1`` over the seconds from the first hearing to
   the last, and 0 until a repeat;
8. the ring: ``HEARD_SLOTS`` entries per agent; a new sound takes a free slot, else the one heard longest ago; an entry
   not heard for ``FORGET_SECONDS`` is dropped; the ring is cleared on a reset and on a respawn.

Windows are stated in seconds and turned into ticks by ``state/clock.py``. Every constant is a named module constant;
those marked [E] are estimates for the owner to rule on (17.2, 17.3).

Positions. With a pose (the perception channel publishes the agent's world position, and the yaw comes with the pose), a
heard sound is kept in world axes at its judged position, and every entry is turned into this tick's body frame before
the token and the views read it: an entry stays where the source was heard as the agent turns or moves. A merge compares
bearings from the listener in the current body frame. With no pose (no channel), the first slice's behaviour holds: an
entry keeps the body frame of its last hearing, and drifts as the agent turns or moves until its next merge. The true
source offset is kept beside it for later memory.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.perception import adapter
from experimental.neural_agent.skill_agent.perception import sounds as sound_registry
from experimental.neural_agent.skill_agent.state import clock
from experimental.neural_agent.skill_agent.state.frames import body_to_world, world_to_body
from experimental.neural_agent.skill_agent.state.senses import Pose, _put, _zeros, where_of

HEARING_THRESHOLD: float = 0.01
"""A linear gain: -40 dB of full volume [E]. Alarm would lower it and fatigue raise it, through a regulator (17.3), which
is future, so it is fixed here."""
MASK_DEGREES: float = 30.0
"""A masker counts within this many degrees of the sound's bearing [E] (17.2 item 1)."""
MASK_RATIO: float = 10.0
"""A masker must be more than this many times louder, 20 dB [E] (17.2 item 1)."""
BLUR_FRONT_DEGREES: float = 1.0
"""The spread of the bearing straight ahead, in degrees: the smallest change a listener detects there (Mills 1958);
used as the blur's spread [E]."""
BLUR_SIDE_DEGREES: float = 3.0
"""The spread at the side, in degrees (a 2021 replication's 3.1); used as the blur's spread [E]."""
BLUR_ELEVATION_DEGREES: float = 10.0
"""The spread of the elevation, in degrees: elevation is heard coarsely [E] (17.2 item 3)."""
JUDGED_K: float = 1.3
"""Judged distance = ``JUDGED_K * distance ** JUDGED_EXPONENT`` in blocks (Zahorik 2002, JASA 111(4); quoted from
arXiv 1704.06008, read 2026-10-24)."""
JUDGED_EXPONENT: float = 0.39
"""See ``JUDGED_K``: the exponent of the power function, averaging 0.39 in Zahorik 2002."""
MERGE_DEGREES: float = 20.0
"""A sound from an unknown source merges into an entry whose judged bearing is within this many degrees [E]."""
MERGE_SECONDS: float = 2.0
"""A repeat merges into an entry heard within this many seconds [E]."""
FORGET_SECONDS: float = 10.0
"""An entry not heard for this many seconds is dropped [E]."""
PRIVATE_REGISTRY: Path = Path(__file__).resolve().parents[4] / "agent_skills/training_runs/perception/sounds_072"
"""The private sound registry (``perception/sounds.py``, built from the 0.7.2 archive): the default kinds of heard
sounds (:func:`default_tables`)."""
HEARD_SLOTS: int = v3.CAPACITIES_V3["sound"]
"""Entries per agent: the token's slots (16)."""
BLUR_SEED: int = 0x5E4D
"""The fixed key the blur draws fold into, so a replay repeats the blur."""
CATEGORIES: tuple[str, ...] = v3.SOUND_CATEGORIES
CATEGORY_START: int = len(sound_registry.SOUND_FEATURES)
"""The first category column of the registry's feature rows (after the fixed columns)."""


class HearingState(NamedTuple):
    """The ring of one agent batch: per slot (``HEARD_SLOTS`` each) and the tick counter."""

    valid: jax.Array  # bool[B, S]
    sound_id: jax.Array  # int32[B, S], the semantic id of the sound's asset
    row: jax.Array  # int32[B, S], the sound's registry table row (0: the unknown row)
    body: jax.Array  # float32[B, S, 3], the judged position in the body frame (forward, right, up), at the last hearing
    true_offset: jax.Array  # float32[B, S, 3], the true source minus the listener at the last hearing, world axes
    world: jax.Array  # float32[B, S, 3], the judged source in world axes (the pose's position plus the judged offset)
    anchored: jax.Array  # bool[B, S], the entry has a world position (the agent had a pose when it was heard)
    distance: jax.Array  # float32[B, S], the judged distance, blocks
    gain: jax.Array  # float32[B, S], the loudness: the linear gain at the last hearing
    first: jax.Array  # int32[B, S], the tick the sound was first heard
    last: jax.Array  # int32[B, S], the tick it was last heard
    count: jax.Array  # int32[B, S], the hearings the entry holds
    known: jax.Array  # bool[B, S], the source is an entity the agent perceives
    entity: jax.Array  # int32[B, S], that entity's slot when known, else -1
    now: jax.Array  # int32[B], the tick counter (one per :func:`update`)


def initial(batch: int) -> HearingState:
    """An empty ring for ``batch`` agents."""

    zeros_int = jnp.zeros((batch, HEARD_SLOTS), jnp.int32)
    return HearingState(
        valid=jnp.zeros((batch, HEARD_SLOTS), jnp.bool_), sound_id=zeros_int, row=zeros_int,
        body=jnp.zeros((batch, HEARD_SLOTS, 3), jnp.float32), true_offset=jnp.zeros((batch, HEARD_SLOTS, 3), jnp.float32),
        world=jnp.zeros((batch, HEARD_SLOTS, 3), jnp.float32), anchored=jnp.zeros((batch, HEARD_SLOTS), jnp.bool_),
        distance=jnp.zeros((batch, HEARD_SLOTS), jnp.float32), gain=jnp.zeros((batch, HEARD_SLOTS), jnp.float32),
        first=zeros_int, last=zeros_int, count=zeros_int, known=jnp.zeros((batch, HEARD_SLOTS), jnp.bool_),
        entity=jnp.full((batch, HEARD_SLOTS), -1, jnp.int32), now=jnp.zeros((batch,), jnp.int32))


def _wrap(angle: jax.Array) -> jax.Array:
    return jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi


def _normals(lanes: Any, ticks: Any, slots: Any, sounds: Any) -> jax.Array:
    """Two standard normals per draw, float32[..., 2]: one fixed key folded with the lane, the tick, the slot of the
    simulator's heard list and the sound id, so the same four give the same draw."""

    arguments = (lanes, ticks, slots, sounds)
    shape = jnp.broadcast_shapes(*(jnp.shape(argument) for argument in arguments))
    flat = [jnp.broadcast_to(jnp.asarray(argument, jnp.int32), shape).reshape(-1) for argument in arguments]
    base = jax.random.key(BLUR_SEED)

    def draw(lane, tick, slot, sound):
        key = base
        for datum in (lane, tick, slot, sound):
            key = jax.random.fold_in(key, datum)
        return jax.random.normal(key, (2,))

    return jax.vmap(draw)(*flat).reshape(shape + (2,))


def judge(body: jax.Array, distance: jax.Array, lanes: Any, ticks: Any, slots: Any, sounds: Any
          ) -> tuple[jax.Array, jax.Array]:
    """The judged position and distance of sources at true body-frame positions ``body`` float32[..., 3] (forward,
    right, up) and true ``distance`` [...]: the position float32[..., 3] in the body frame, and the judged distance
    [...]. The bearing is blurred by ``BLUR_FRONT_DEGREES`` straight ahead to ``BLUR_SIDE_DEGREES`` at the side, the
    elevation by ``BLUR_ELEVATION_DEGREES``; the distance is compressed (17.2 items 4 and 5)."""

    forward, right, up = body[..., 0], body[..., 1], body[..., 2]
    azimuth = jnp.arctan2(right, forward)
    elevation = jnp.arctan2(up, jnp.hypot(forward, right))
    noise = _normals(lanes, ticks, slots, sounds)
    spread = jnp.radians(BLUR_FRONT_DEGREES + (BLUR_SIDE_DEGREES - BLUR_FRONT_DEGREES) * jnp.abs(jnp.sin(azimuth)))
    judged_azimuth = azimuth + spread * noise[..., 0]
    judged_elevation = elevation + jnp.radians(BLUR_ELEVATION_DEGREES) * noise[..., 1]
    judged_distance = JUDGED_K * jnp.power(distance, JUDGED_EXPONENT)
    ground = judged_distance * jnp.cos(judged_elevation)
    position = jnp.stack((ground * jnp.cos(judged_azimuth), ground * jnp.sin(judged_azimuth),
                          judged_distance * jnp.sin(judged_elevation)), axis=-1)
    return position.astype(jnp.float32), judged_distance.astype(jnp.float32)


def _set(old: jax.Array, mask: jax.Array, new: jax.Array) -> jax.Array:
    """``old`` with ``new`` (one value per lane, or one vector per lane) where ``mask`` float [B, S]."""

    new = jnp.asarray(new)
    new = new[:, None] if new.ndim == 1 else new[:, None, :]
    return jnp.where(mask if old.ndim == 2 else mask[..., None], new, old)


def _same_source(state: HearingState, known: jax.Array, entity: jax.Array, body: jax.Array) -> jax.Array:
    """bool[B, S]: each entry is the same source as a sound: the same entity when both are known, else a judged bearing
    within ``MERGE_DEGREES``."""

    entry = jnp.arctan2(state.body[..., 1], state.body[..., 0])
    heard = jnp.arctan2(body[:, 1], body[:, 0])
    near = jnp.abs(_wrap(entry - heard[:, None])) <= jnp.radians(MERGE_DEGREES)
    return jnp.where(state.known & known[:, None], state.entity == entity[:, None], near)


def _hear_one(state: HearingState, sound: tuple[jax.Array, ...], now: jax.Array, merge_ticks: int) -> HearingState:
    """One of this tick's heard sounds, per lane: merged into its entry when one matches, else into a free slot or the
    one heard longest ago. A lane with nothing heard is unchanged."""

    heard, sound_id, row, body, offset, distance, gain, known, entity, world, anchored = sound
    match = (state.valid & (state.sound_id == sound_id[:, None]) & (now[:, None] - state.last <= merge_ticks)
             & _same_source(state, known, entity, body))
    merging = jnp.any(match, axis=1)
    oldest = jnp.argmin(jnp.where(state.valid, state.last, -1), axis=1)
    at = jnp.where(merging, jnp.argmax(match, axis=1), oldest)
    write = (jnp.arange(HEARD_SLOTS)[None, :] == at[:, None]) & heard[:, None]
    fresh = write & ~merging[:, None]
    joined = write & merging[:, None]
    return state._replace(
        valid=state.valid | write,
        sound_id=_set(state.sound_id, fresh, sound_id),
        row=_set(state.row, fresh, row),
        body=_set(state.body, write, body),
        true_offset=_set(state.true_offset, write, offset),
        world=_set(state.world, write, world),
        anchored=_set(state.anchored, write, anchored),
        distance=_set(state.distance, write, distance),
        gain=_set(state.gain, write, gain),
        first=_set(state.first, fresh, now),
        last=_set(state.last, write, now),
        count=jnp.where(fresh, 1, jnp.where(joined, state.count + 1, state.count)),
        known=_set(state.known, fresh, known),
        entity=_set(state.entity, fresh, entity),
        now=state.now)


def update(state: HearingState, heard: Any, *, body_yaw: jax.Array, position: jax.Array | None,
           tables: sound_registry.SoundTables | None, tick_seconds: float, reset: jax.Array | None = None,
           respawn: jax.Array | None = None) -> HearingState:
    """The ring after this tick (the module docstring, items 1 to 8).

    Args:
        state: Last tick's ring.
        heard: The observation's ``heard_sounds`` (a ``HeardSounds``: ``valid``, ``sound_id``, ``relative_position``,
            ``distance``, ``gain``, ``source_entity_id``), or None when the observation has none: nothing is heard.
        body_yaw: float32[B], the body yaw in radians (the frame the sounds are turned into).
        position: float32[B, 3], the agent's world position (the pose's, from the perception channel), or None. With a
            position, a heard sound is kept in world axes and every anchored entry is turned into this tick's body frame
            before anything else; with none (no channel), an entry keeps the body frame of its last hearing.
        tables: The sound registry's tables (``perception/sounds.sound_tables``), or None: every sound reads the
            unknown row.
        tick_seconds: The world's tick length (``state/clock.py``).
        reset: bool[B], lanes that start an episode: their ring is cleared.
        respawn: bool[B], lanes that respawned: their ring is cleared.
    """

    batch = state.valid.shape[0]
    forget = clock.about(FORGET_SECONDS, tick_seconds)
    merge = clock.about(MERGE_SECONDS, tick_seconds)
    now = state.now + 1
    restart = jnp.zeros((batch,), jnp.bool_)
    for flag in (reset, respawn):
        if flag is not None:
            restart = restart | jnp.asarray(flag, jnp.bool_)
    state = state._replace(valid=state.valid & ~restart[:, None] & (now[:, None] - state.last <= forget), now=now)
    yaw = jnp.asarray(body_yaw, jnp.float32)
    here = None if position is None else jnp.asarray(position, jnp.float32)
    if here is not None:  # each anchored entry in this tick's body frame: the belief stays where it was heard
        turned = world_to_body(state.world - here[:, None, :], yaw[:, None])
        state = state._replace(body=jnp.where((state.valid & state.anchored)[..., None], turned, state.body))
    if heard is None:
        return state

    relative = jnp.asarray(heard.relative_position, jnp.float32)
    body = world_to_body(relative, jnp.asarray(body_yaw, jnp.float32)[:, None])
    gain = jnp.asarray(heard.gain, jnp.float32)
    distance = jnp.asarray(heard.distance, jnp.float32)
    sound_id = jnp.asarray(heard.sound_id, jnp.int32)
    entity = jnp.asarray(heard.source_entity_id, jnp.int32)
    known = entity >= 0

    audible = jnp.asarray(heard.valid, jnp.bool_) & (gain >= HEARING_THRESHOLD)
    azimuth = jnp.arctan2(body[..., 1], body[..., 0])
    near_bearing = jnp.abs(_wrap(azimuth[:, :, None] - azimuth[:, None, :])) <= jnp.radians(MASK_DEGREES)
    louder = gain[:, None, :] > MASK_RATIO * gain[:, :, None]
    masked = jnp.any(louder & near_bearing & audible[:, None, :], axis=2)
    hear = audible & ~masked

    lanes = jnp.arange(batch, dtype=jnp.int32)[:, None]
    slots = jnp.arange(sound_id.shape[1], dtype=jnp.int32)[None, :]
    judged, judged_distance = judge(body, distance, lanes, now[:, None], slots, sound_id)
    if tables is None:
        row = jnp.zeros_like(sound_id)
    else:
        row, _ = adapter.lookup(tables.keys, tables.rows, sound_id)

    if here is None:
        world = jnp.zeros_like(judged)
    else:
        world = here[:, None, :] + body_to_world(judged, yaw[:, None])
    anchored = jnp.full(sound_id.shape, here is not None, jnp.bool_)
    sounds = tuple(jnp.moveaxis(x, 1, 0) for x in (hear, sound_id, row, judged, relative, judged_distance, gain, known,
                                                    entity, world, anchored))
    state, _ = jax.lax.scan(lambda carry, sound: (_hear_one(carry, sound, now, merge), None), state, sounds)
    return state


def rows(state: HearingState, pose: Pose, tables: sound_registry.SoundTables | None, *, tick_seconds: float
         ) -> tuple[jax.Array, jax.Array]:
    """The ``sound`` token rows (float32[B, S, F], mask bool[B, S]): each entry's ``where`` block from its judged position
    in the body frame, its registry description and category, its loudness, age, rate and whether its source is known,
    and its confidence (17.8)."""

    batch, slots = state.valid.shape
    now = state.now[:, None]
    if tables is None:
        description = jnp.zeros((batch, slots, v3.DESCRIPTION_WIDTH), jnp.float32)
        category = jnp.zeros((batch, slots, len(CATEGORIES)), jnp.float32)
    else:
        description = tables.descriptions[state.row]
        category = tables.features[state.row][..., CATEGORY_START:]
    age = jnp.clip((now - state.last).astype(jnp.float32) * tick_seconds / FORGET_SECONDS, 0.0, 1.0)
    rate = rate_of(state.count, state.first, state.last, tick_seconds)
    confidence = jnp.clip(jnp.log10(jnp.maximum(state.gain, HEARING_THRESHOLD) / HEARING_THRESHOLD) / 2.0, 0.0, 1.0)
    features = _put(_zeros(batch, slots, "sound"), {
        "forward": where_of(state.body, pose, 0.0),
        "description_0": description,
        f"category_{CATEGORIES[0]}": category,
        "loudness": state.gain, "age": age, "rate": rate, "known": state.known, "confidence": confidence,
    }, "sound")
    return jnp.where(state.valid[..., None], features, 0.0), state.valid


def rate_of(count: Any, first: Any, last: Any, tick_seconds: float) -> jax.Array:
    """The rate of repeats, per second: ``count - 1`` over the seconds from the first hearing to the last. 0 until a
    repeat, and where the repeats fall in one tick (no span). Footsteps every 30 ticks over 3 s read exactly 1 a second
    (the rate the coordinator set on 2026-10-24)."""

    count = jnp.asarray(count, jnp.int32)
    span = (jnp.asarray(last, jnp.int32) - jnp.asarray(first, jnp.int32)).astype(jnp.float32) * tick_seconds
    usable = (count >= 2) & (span > 0.0)
    rate = (count - 1).astype(jnp.float32) / jnp.where(usable, span, 1.0)
    return jnp.where(usable, rate, 0.0).astype(jnp.float32)


def unknown_tables() -> sound_registry.SoundTables:
    """Tables that hold no sound: every sound reads the unknown row (zeros), as with no registry."""

    return sound_registry.SoundTables(
        keys=jnp.zeros((0,), jnp.int32), rows=jnp.zeros((0,), jnp.int32),
        features=jnp.zeros((1, CATEGORY_START + len(CATEGORIES)), jnp.float32),
        descriptions=jnp.zeros((1, v3.DESCRIPTION_WIDTH), jnp.float32))


@functools.lru_cache(maxsize=None)
def default_tables() -> sound_registry.SoundTables:
    """The private registry's tables, read once from ``PRIVATE_REGISTRY``; the unknown-row tables when it is absent. The
    layer uses them when a step has a heard field and no ``sound_tables``."""

    if not (PRIVATE_REGISTRY.with_suffix(".npz").exists() and PRIVATE_REGISTRY.with_suffix(".json").exists()):
        return unknown_tables()
    return tables_for(sound_registry.load_sounds(PRIVATE_REGISTRY))


def tables_for(registry: sound_registry.SoundRegistry) -> sound_registry.SoundTables:
    """The sound tables of a registry, refused unless its categories and description width are the schema's (a later
    archive's new category would otherwise shift the token's columns silently)."""

    if tuple(registry.manifest["categories"]) != CATEGORIES:
        raise ValueError("the sound registry's categories differ from tokens_v3.SOUND_CATEGORIES")
    if registry.manifest["description_width"] != v3.DESCRIPTION_WIDTH:
        raise ValueError("the sound registry's description width differs from the token's")
    return sound_registry.sound_tables(registry)


__all__ = ["BLUR_FRONT_DEGREES", "BLUR_SIDE_DEGREES", "FORGET_SECONDS", "HEARD_SLOTS", "HEARING_THRESHOLD",
           "HearingState", "JUDGED_EXPONENT", "JUDGED_K", "MASK_DEGREES", "MASK_RATIO", "MERGE_DEGREES",
           "MERGE_SECONDS", "PRIVATE_REGISTRY", "default_tables", "initial", "judge", "rate_of", "rows", "tables_for",
           "unknown_tables", "update"]
