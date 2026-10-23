"""Opt-in server-faithful footsteps and the heard-sounds channel (S12, increment 1).

Nothing imports this module. It is off unless a caller runs it, so the existing
audio ring, the contract hash and the gen-5 render stay bit-identical.

Footsteps follow 0.7.2 ``AudioSystems$TickMovementAudio``: on a change of the
block under the feet it plays the block's ``MoveOut`` (of the block left) and
``MoveIn`` (of the block entered) when the sets author them; while the body is
not idle, ``MoveIn`` repeats on the set's repeat range. Only block sets that
author MoveIn or MoveOut make sound. ``Walk`` is a client-side sound that a
server-perceiving agent does not get, so Grass and Stone stay silent here.

[E] The engine draws each repeat from ``BlockSoundSet``'s range (default
0.5 to 1.5 s, ``BlockSoundSet.javap`` static init). This module uses the
midpoint, 1.0 s, so the cadence is deterministic.

[E] Sound ids are ``semantic_id`` of the event's asset name, used as
``sound_event_index`` because the simulator has no runtime index table.

Heard sounds are the ring's events within each sound's ``MaxDistance`` (the
server's own gate), with gain = linear volume times a linear curve from
``StartAttenuationDistance`` (gain 1) to ``MaxDistance`` (gain 0). [E] The
engine's real curve and wall occlusion are unknown; the server sends the
parameters to the client, which mixes them.

Gaps, not built here: the source entity id is always 0 because
``audio_observation_step`` requires network id 0 on 3-D events, so the
perceptible-source id needs a ring change. Block hit and break sounds are
increment 2.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.world.audio import (
    AUDIO_CATEGORY_SFX,
    AUDIO_EVENT_FAMILY_BLOCK,
    AUDIO_PACKET_ID_3D,
    AUDIO_ROUTING_SPATIAL_BROADCAST,
    BLOCK_SOUND_EVENT_MOVE_IN,
    BLOCK_SOUND_EVENT_MOVE_OUT,
    AudioEventBatch,
    AudioObservation,
    AudioStepResult,
    audio_observation_step,
)

Array = jax.Array

FOOTSTEP_BLOCK_ASSETS = (
    "Plant_Bush",
    "Deco_SpiderWeb_Flat",
    "Fluid_Water",
    "Soil_Grass",
    "Rock_Stone",
)
"""Block assets the content table covers. Soil_Grass (set Grass) and Rock_Stone (set Stone) author no MoveIn."""

ENGINE_MAX_DISTANCE = 16.0
ENGINE_START_ATTENUATION = 2.0
REPEAT_SECONDS = 1.0
"""[E] Midpoint of the default MoveIn repeat range (0.5 to 1.5 s)."""
NO_REPEAT = -1.0
EVENT_SLOTS = 2
"""Slot 0 carries MoveOut on a block change; slot 1 carries MoveIn (a change or a repeat)."""


class SoundRecord(NamedTuple):
    name: str
    sound_id: int
    max_distance: float
    start_distance: float
    volume_linear: float


@dataclass(frozen=True)
class FootstepContent:
    """Per block: its asset id and its MoveIn/MoveOut sound ids (0 when not authored), plus the sound records."""

    blocks: dict[str, tuple[int, int, int]]
    sounds: tuple[SoundRecord, ...]


class FootstepTables(NamedTuple):
    block_keys: Array  # int32 [Nb], sorted semantic block ids
    move_in: Array  # int32 [Nb], sound id or 0
    move_out: Array  # int32 [Nb], sound id or 0
    sound_keys: Array  # int32 [Ns], sorted sound ids
    sound_max: Array  # float32 [Ns]
    sound_start: Array  # float32 [Ns]
    sound_volume: Array  # float32 [Ns], linear


class HeardSounds(NamedTuple):
    heard: Array  # bool [B, K]
    gain: Array  # float32 [B, K]
    relative_position: Array  # float32 [B, K, 3]
    distance: Array  # float32 [B, K]
    sound_event_index: Array  # int32 [B, K]
    event_family: Array  # int32 [B, K]
    event_ordinal: Array  # int32 [B, K]
    category: Array  # int32 [B, K]
    age_seconds: Array  # float32 [B, K]
    source_network_id: Array  # int32 [B, K], 0 in this increment


def _stem_lookup(names: list[str], prefix: str, stem: str) -> str:
    hits = [n for n in names if n.startswith(prefix) and (n.endswith("/" + stem + ".json") or n == prefix + stem + ".json")]
    if len(hits) != 1:
        raise ValueError(f"expected one asset named {stem} under {prefix}, found {len(hits)}")
    return hits[0]


def _chain_value(zipped: zipfile.ZipFile, path: str, key: str, default):
    current = path
    while True:
        document = json.loads(zipped.read(current))
        if key in document:
            return document[key]
        parent = document.get("Parent")
        if not isinstance(parent, str):
            return default
        current = _stem_lookup(zipped.namelist(), "Server/Audio/SoundEvents/", parent)


def load_footstep_content(assets_zip: str | Path) -> FootstepContent:
    """Read the footstep sets from the game's Assets.zip. Ambiguous or missing assets raise."""

    return _load(str(assets_zip))


@lru_cache(maxsize=4)
def _load(assets_zip: str) -> FootstepContent:
    blocks: dict[str, tuple[int, int, int]] = {}
    sounds: dict[int, SoundRecord] = {}
    with zipfile.ZipFile(assets_zip) as zipped:
        names = zipped.namelist()
        for block in FOOTSTEP_BLOCK_ASSETS:
            path = _stem_lookup(names, "Server/Item/Items/", block)
            set_name = json.loads(zipped.read(path))["BlockType"]["BlockSoundSetId"]
            events = json.loads(zipped.read(f"Server/Item/Block/Sounds/{set_name}.json"))["SoundEvents"]
            ids = []
            for family in ("MoveIn", "MoveOut"):
                event = events.get(family)
                if event is None:
                    ids.append(0)
                    continue
                sound_path = _stem_lookup(names, "Server/Audio/SoundEvents/", event)
                sound_id = semantic_id(event)
                max_distance = float(_chain_value(zipped, sound_path, "MaxDistance", ENGINE_MAX_DISTANCE))
                start = float(_chain_value(zipped, sound_path, "StartAttenuationDistance", ENGINE_START_ATTENUATION))
                volume_db = float(_chain_value(zipped, sound_path, "Volume", 0.0))
                sounds[sound_id] = SoundRecord(
                    name=event,
                    sound_id=sound_id,
                    max_distance=max_distance,
                    start_distance=start,
                    volume_linear=float(10.0 ** (volume_db / 20.0)),
                )
                ids.append(sound_id)
            blocks[block] = (semantic_id(block), ids[0], ids[1])
    return FootstepContent(blocks=blocks, sounds=tuple(sorted(sounds.values(), key=lambda s: s.sound_id)))


def footstep_tables(content: FootstepContent) -> FootstepTables:
    rows = sorted(content.blocks.values(), key=lambda row: row[0])
    sounds = content.sounds
    return FootstepTables(
        block_keys=jnp.asarray([row[0] for row in rows], dtype=jnp.int32),
        move_in=jnp.asarray([row[1] for row in rows], dtype=jnp.int32),
        move_out=jnp.asarray([row[2] for row in rows], dtype=jnp.int32),
        sound_keys=jnp.asarray([s.sound_id for s in sounds], dtype=jnp.int32),
        sound_max=jnp.asarray([s.max_distance for s in sounds], dtype=jnp.float32),
        sound_start=jnp.asarray([s.start_distance for s in sounds], dtype=jnp.float32),
        sound_volume=jnp.asarray([s.volume_linear for s in sounds], dtype=jnp.float32),
    )


def _find(keys: Array, query: Array) -> tuple[Array, Array]:
    index = jnp.clip(jnp.searchsorted(keys, query), 0, keys.shape[0] - 1)
    return index, (keys[index] == query) & (query != 0)


def _block_sound(tables: FootstepTables, block: Array, table: Array) -> Array:
    index, found = _find(tables.block_keys, block)
    return jnp.where(found, table[index], 0)


def footstep_step(
    tables: FootstepTables,
    previous_block: Array,
    current_block: Array,
    idle: Array,
    repeat_remaining: Array,
    position: Array,
    delta_seconds: float,
) -> tuple[AudioEventBatch, Array]:
    """One tick of footsteps for B bodies. Returns the staged events (2 slots) and the next repeat timers.

    ``repeat_remaining`` is NO_REPEAT (-1) when no repeat is armed. ``current_block`` is the block under the feet
    (semantic id, 0 for air or unknown). Only the events of sets that author them are staged.
    """

    changed = current_block != previous_block
    out_key = _block_sound(tables, previous_block, tables.move_out)
    in_key = _block_sound(tables, current_block, tables.move_in)
    armed = repeat_remaining >= 0.0
    moving = ~idle

    ticked = repeat_remaining - delta_seconds
    repeat_fire = ~changed & moving & armed & (ticked <= 0.0) & (in_key != 0)
    out_valid = changed & (out_key != 0)
    in_valid = (changed & (in_key != 0)) | repeat_fire

    next_on_change = jnp.where(in_key != 0, REPEAT_SECONDS, NO_REPEAT)
    next_on_repeat = jnp.where(repeat_fire, REPEAT_SECONDS, jnp.where(in_key != 0, ticked, NO_REPEAT))
    next_remaining = jnp.where(
        changed,
        next_on_change,
        jnp.where(moving & armed, next_on_repeat, repeat_remaining),
    )

    def slot(valid: Array, key: Array) -> dict[str, Array]:
        index, _ = _find(tables.sound_keys, key)
        batch = valid.shape[0]
        return {
            "valid": valid,
            "packet_id": jnp.full((batch,), AUDIO_PACKET_ID_3D, dtype=jnp.int32),
            "routing": jnp.full((batch,), AUDIO_ROUTING_SPATIAL_BROADCAST, dtype=jnp.int32),
            "sound_event_index": key.astype(jnp.int32),
            "category": jnp.full((batch,), AUDIO_CATEGORY_SFX, dtype=jnp.int32),
            "event_family": jnp.full((batch,), AUDIO_EVENT_FAMILY_BLOCK, dtype=jnp.int32),
            "max_distance": tables.sound_max[index],
        }

    out_slot = slot(out_valid, out_key)
    in_slot = slot(in_valid, in_key)
    fields = {}
    names = ("valid", "packet_id", "routing", "sound_event_index", "category", "event_family", "max_distance")
    for name in names:
        fields[name] = jnp.stack([out_slot[name], in_slot[name]], axis=1)
    fields["event_ordinal"] = jnp.stack(
        [
            jnp.full_like(out_slot["valid"], BLOCK_SOUND_EVENT_MOVE_OUT, dtype=jnp.int32),
            jnp.full_like(in_slot["valid"], BLOCK_SOUND_EVENT_MOVE_IN, dtype=jnp.int32),
        ],
        axis=1,
    )
    batch = position.shape[0]
    fields["source_position"] = jnp.broadcast_to(position[:, None, :], (batch, EVENT_SLOTS, 3)).astype(jnp.float32)
    fields["source_network_id"] = jnp.zeros((batch, EVENT_SLOTS), dtype=jnp.int32)
    fields["volume_modifier"] = jnp.ones((batch, EVENT_SLOTS), dtype=jnp.float32)
    fields["pitch_modifier"] = jnp.zeros((batch, EVENT_SLOTS), dtype=jnp.float32)
    fields["age_seconds"] = jnp.zeros((batch, EVENT_SLOTS), dtype=jnp.float32)
    return AudioEventBatch(**fields), next_remaining


def hear(
    tables: FootstepTables,
    observation: AudioObservation,
) -> HeardSounds:
    """The ring's events each actor hears: inside the sound's MaxDistance, with the linear attenuation gain.

    The sound's start distance and linear volume come from the content table by sound id. An unknown sound is not
    heard. A 2-D event (distance 0, max distance 0) is never heard here.
    """

    index, found = _find(tables.sound_keys, observation.sound_event_index)
    start = jnp.where(found, tables.sound_start[index], 0.0)
    volume = jnp.where(found, tables.sound_volume[index], 0.0)
    distance = observation.distance
    max_distance = observation.max_distance
    span = max_distance - start
    safe_span = jnp.where(span > 0.0, span, 1.0)
    fade = jnp.clip((max_distance - distance) / safe_span, 0.0, 1.0)
    curve = jnp.where(distance <= start, 1.0, jnp.where(span > 0.0, fade, 0.0))
    heard = observation.valid & found & (distance < max_distance)
    gain = jnp.where(heard, volume * curve, 0.0).astype(jnp.float32)
    return HeardSounds(
        heard=heard,
        gain=gain,
        relative_position=observation.relative_position,
        distance=distance,
        sound_event_index=observation.sound_event_index,
        event_family=observation.event_family,
        event_ordinal=observation.event_ordinal,
        category=observation.category,
        age_seconds=observation.age_seconds,
        source_network_id=jnp.zeros_like(observation.source_network_id),
    )


def footstep_observation_step(
    tables: FootstepTables,
    previous: AudioObservation,
    listener_position: Array,
    source_position: Array,
    previous_block: Array,
    current_block: Array,
    idle: Array,
    repeat_remaining: Array,
    delta_seconds: float,
) -> tuple[AudioStepResult, Array]:
    """Stage this tick's footsteps for one body per environment row and fold them into that row's ring.

    Limit: the event axis holds this body's two slots only, so a ring hears only its own body's footsteps. Hearing
    other bodies needs 2 slots per emitting body in one batch (not built here).
    """

    events, next_remaining = footstep_step(
        tables,
        previous_block,
        current_block,
        idle,
        repeat_remaining,
        source_position,
        delta_seconds,
    )
    result = audio_observation_step(previous, listener_position, events, delta_seconds)
    return result, next_remaining


__all__ = [
    "FOOTSTEP_BLOCK_ASSETS",
    "FootstepContent",
    "FootstepTables",
    "HeardSounds",
    "NO_REPEAT",
    "REPEAT_SECONDS",
    "footstep_observation_step",
    "footstep_step",
    "footstep_tables",
    "hear",
    "load_footstep_content",
]
