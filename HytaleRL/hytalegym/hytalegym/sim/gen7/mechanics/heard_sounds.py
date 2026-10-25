"""Gen-7 heard sounds (S12): the footsteps of the entities in each arena row, heard by the row's entity 0.

Switch ``Constructs.hearing``, default False. Hook: ``encode_learner_observation_v3`` (sim/gen7/overlay.py row ``hearing``,
chained after ``game_clock``). When the switch is on, the learner observation gains one field, ``heard_sounds``, last. The
flat policy vector and the contract hashes do not read it. Content and gain come from ``jax/world/audio/hearing.py``
(``load_footstep_content``, the linear curve of ``hear``); the grounding is DESIGN.md sections 1, 3 and 5 in the lane folder.

One batch row is one arena. Its entities are the slots 0..E-1 of ``state.combat``; entity 0 is the learner, and the other
slots are the sources it can hear. A row never hears another row: each row's list holds only its own entities.

Approximations (each is a cost, stated here so the skill layer can judge them):

- Switch ``hearing_entry`` (on): the footsteps are movein_timing.py's, the native entry and 0.5-1.5 s repeat, read from
  ``state.combat.footstep.emit``; the stateless cadence below is then unused. The text below describes the switch-off path.
- Emission is stateless (switch off). Each entity emits MoveIn of the block in its feet cell on ticks where ``tick_count % 30 == 0``
  while it moves horizontally (30 ticks = 1.0 s, the midpoint of the engine's repeat range; [E] in hearing.py). The phase is
  shared by every entity. Cost: no MoveOut and no entry burst on a block change, because the policy state carries the
  previous block nowhere; the first footstep can come up to 1 s late.
- The heard list holds this tick's emissions only, so ``age_seconds`` is 0. The K = 16 ring is the skill layer's to keep.
  Sources are filled in entity order, heard ones first, and padded to K.
- The block under each entity is read from ``world_capabilities.block_under_feet`` (int32 [B, E], semantic id, 0 unknown). The
  producer is mechanics/feet_block.py (the feet cell, the dense grid or the Region atlas), but no capability builder attaches
  its result yet, so the field is absent and every entity stands on Soil_Grass, which authors no MoveIn and is silent.
- ``source_entity_id``: 0 for entity 0's own footsteps; j for another entity j when entity 0's perception of j holds
  (``target_candidate_perceptible[:, 0, j]`` and its validity flag); else -1.
"""

from __future__ import annotations

import os
from functools import lru_cache, wraps
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.world.audio.hearing import FootstepTables, footstep_tables, load_footstep_content

SWITCH = "hearing"
IMPLEMENTED = True

#: Heard slots per listener (the wire AUDIO_MAX_EVENTS and the ring capacity).
HEARD_CAPACITY = 16
#: 1.0 s at the gen-7 step (1/30 s): the midpoint of the engine's MoveIn repeat range.
REPEAT_TICKS = 30
#: Horizontal speed above which an entity counts as moving (blocks per second).
MOVING_SPEED = 1e-3
#: The default floor: Soil_Grass (Server/Item/Items/Soil_Grass, set Grass), the live fixture floor; it authors no MoveIn.
FLOOR_BLOCK = "Soil_Grass"
LISTENER = 0
ASSETS_ENV = "HYTALE_ASSETS_ZIP"


class HeardSounds(NamedTuple):
    """Per arena row, K slots for its entity-0 listener. Each array is [B, K] except ``relative_position`` [B, K, 3]."""

    valid: Any  # bool
    sound_id: Any  # int32, the semantic id of the MoveIn event's name
    relative_position: Any  # float32, source minus listener, world axes
    distance: Any  # float32, blocks
    gain: Any  # float32, linear volume times the attenuation curve
    age_seconds: Any  # float32, 0 (this tick's emissions only)
    source_entity_id: Any  # int32, the source's entity slot, -1 when not perceptible


@lru_cache(maxsize=4)
def _tables(assets_zip: str) -> FootstepTables:
    # Cached, so built concretely even when the first call is inside a jitted reset or step: a traced build would cache
    # that trace's tracers (test_hearing_channel.py test_tables_first_built_inside_a_trace_stay_concrete).
    with jax.ensure_compile_time_eval():
        return footstep_tables(load_footstep_content(assets_zip))


def footstep_tables_from_env() -> FootstepTables:
    """The footstep content tables, read once from the Assets.zip named by HYTALE_ASSETS_ZIP."""

    path = os.environ.get(ASSETS_ENV)
    if not path:
        raise ValueError(f"the hearing construct reads footstep sounds from {ASSETS_ENV}, which is not set")
    return _tables(path)


def _lookup(keys, query, table):
    index = jnp.clip(jnp.searchsorted(keys, query), 0, keys.shape[0] - 1)
    found = (keys[index] == query) & (query != 0)
    return jnp.where(found, table[index], 0), found, index


def heard_sounds_fields(state, world_capabilities, tables: FootstepTables, *, entry_timed: bool = False) -> HeardSounds:
    """The heard list of each arena row's entity 0, from this tick's footsteps of the row's entities (E sources).

    ``entry_timed`` (switch hearing_entry): the footsteps are the ones movein_timing.py stored this step in
    ``state.combat.footstep.emit``; off, the stateless 30-tick cadence below.
    """

    position = state.combat.position.astype(jnp.float32)  # [B, E, 3]
    batch, entities = position.shape[0], position.shape[1]
    if entry_timed:
        footstep = state.combat.footstep  # movein_timing.FootstepState; None only for a state the reset hook did not make
        if footstep is None:
            sound_id = jnp.zeros((batch, entities), dtype=jnp.int32)
        else:
            sound_id = footstep.emit.astype(jnp.int32)  # [B, E], 0 for no footstep this step
        emits = sound_id != 0
    else:
        velocity = state.combat.velocity.astype(jnp.float32)
        tick = state.combat.tick_count.astype(jnp.int32)  # [B]

        block = getattr(world_capabilities, "block_under_feet", None)
        if block is None:
            block = jnp.full((batch, entities), semantic_id(FLOOR_BLOCK), dtype=jnp.int32)
        block = block.astype(jnp.int32)  # [B, E]
        horizontal = jnp.sqrt(velocity[..., 0] ** 2 + velocity[..., 2] ** 2)
        moving = horizontal > MOVING_SPEED
        repeat_due = ((tick % REPEAT_TICKS) == 0)[:, None]
        block_in, block_known, _ = _lookup(tables.block_keys, block, tables.move_in)
        emits = moving & repeat_due & block_known & (block_in != 0)
        sound_id = jnp.where(emits, block_in, 0).astype(jnp.int32)  # [B, E]

    sound_index = jnp.clip(jnp.searchsorted(tables.sound_keys, sound_id), 0, tables.sound_keys.shape[0] - 1)
    sound_known = (tables.sound_keys[sound_index] == sound_id) & emits
    max_distance = jnp.where(sound_known, tables.sound_max[sound_index], 0.0)  # [B, E] per source
    start = jnp.where(sound_known, tables.sound_start[sound_index], 0.0)
    volume = jnp.where(sound_known, tables.sound_volume[sound_index], 0.0)

    # Listener = entity 0 of each row; source j = entity j of the same row, as hear() in hearing.py.
    relative = position - position[:, LISTENER : LISTENER + 1, :]  # [B, E, 3] = source j minus the listener
    distance = jnp.sqrt(jnp.sum(relative * relative, axis=-1))  # [B, E]
    span = max_distance - start
    safe_span = jnp.where(span > 0.0, span, 1.0)
    fade = jnp.clip((max_distance - distance) / safe_span, 0.0, 1.0)
    curve = jnp.where(distance <= start, 1.0, jnp.where(span > 0.0, fade, 0.0))
    heard = sound_known & (distance < max_distance)  # [B, E]
    gain = jnp.where(heard, volume * curve, 0.0).astype(jnp.float32)

    source = jnp.arange(entities, dtype=jnp.int32)[None, :]  # [1, E]
    perceived = world_capabilities.target_candidate_perceptible[:, LISTENER, :entities] & (
        world_capabilities.target_candidate_perception_valid[:, LISTENER, :entities]
    )  # [B, E]
    source_entity = jnp.where(source == LISTENER, LISTENER, jnp.where(perceived, source, -1)).astype(jnp.int32)

    # Slots in entity order, the heard ones first. The source axis pads to K so a small row still fills K slots.
    width = max(entities, HEARD_CAPACITY)
    pad = ((0, 0), (0, width - entities))
    heard_p = jnp.pad(heard, pad)
    order = jnp.argsort(~heard_p, axis=1, stable=True)[:, :HEARD_CAPACITY]  # [B, K] source indices

    valid = jnp.take_along_axis(heard_p, order, axis=1)

    def slot(values2d):
        """[B, E] -> [B, K] for the chosen sources."""
        return jnp.take_along_axis(jnp.pad(values2d, pad), order, axis=1)

    relative_padded = jnp.pad(relative, pad + ((0, 0),))
    relative_slots = jnp.take_along_axis(relative_padded, order[:, :, None], axis=1)
    return HeardSounds(
        valid=valid,
        sound_id=jnp.where(valid, slot(sound_id), 0).astype(jnp.int32),
        relative_position=jnp.where(valid[..., None], relative_slots, 0.0).astype(jnp.float32),
        distance=jnp.where(valid, slot(distance), 0.0).astype(jnp.float32),
        gain=jnp.where(valid, slot(gain), 0.0).astype(jnp.float32),
        age_seconds=jnp.zeros((batch, HEARD_CAPACITY), dtype=jnp.float32),
        source_entity_id=jnp.where(valid, slot(source_entity), -1).astype(jnp.int32),
    )


@lru_cache(maxsize=None)
def observation_type(base: type) -> type:
    """``base`` with one more field, ``heard_sounds``, last. One class per observation class."""

    fields = [(name, Any) for name in base._fields] + [("heard_sounds", Any)]
    return NamedTuple(f"{base.__name__}WithHeardSounds", fields)


def construct(original, profile):
    """Return ``original`` when the switch is off; the encoder with the heard list when it is on (the render's hook)."""

    value = profile.constructs.hearing
    if value is False:
        return original
    if value is not True:
        raise ValueError(f"hearing must be a bool, got {value!r}")

    entry_timed = profile.constructs.hearing_entry
    if entry_timed is not True and entry_timed is not False:
        raise ValueError(f"hearing_entry must be a bool, got {entry_timed!r}")

    @wraps(original)  # perception_v2 unwraps the encoder it receives to find the rendered package
    def encode_with_heard_sounds(state, params, world_features, world_capabilities, config, *, world_geometry=None):
        observation = original(
            state, params, world_features, world_capabilities, config, world_geometry=world_geometry
        )
        heard = heard_sounds_fields(
            state, world_capabilities, footstep_tables_from_env(), entry_timed=entry_timed
        )
        return observation_type(type(observation))(*observation, heard)

    return encode_with_heard_sounds
