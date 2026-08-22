"""Validity predicates and masking for audio observations."""
from __future__ import annotations


import jax
import jax.numpy as jnp

from hytalegym.jax.world.audio._types import (
    AudioObservation,
)


Array = jax.Array

AUDIO_OBSERVATION_SCHEMA = "hytalerl_audio_observation_v1"
AUDIO_OBSERVATION_VERSION = 1

# protocol/packets/world/PlaySoundEvent2D.java:12-20
# protocol/packets/world/PlaySoundEvent3D.java:14-22
# protocol/packets/world/PlaySoundEventEntity.java:11-19
AUDIO_PACKET_ID_2D = 154
AUDIO_PACKET_ID_3D = 155
AUDIO_PACKET_ID_ENTITY = 156

# protocol/SoundCategory.java:5-9
AUDIO_CATEGORY_MUSIC = 0
AUDIO_CATEGORY_AMBIENT = 1
AUDIO_CATEGORY_SFX = 2
AUDIO_CATEGORY_UI = 3

# protocol/BlockSoundEvent.java:5-14
BLOCK_SOUND_EVENT_WALK = 0
BLOCK_SOUND_EVENT_LAND = 1
BLOCK_SOUND_EVENT_MOVE_IN = 2
BLOCK_SOUND_EVENT_MOVE_OUT = 3
BLOCK_SOUND_EVENT_HIT = 4
BLOCK_SOUND_EVENT_BREAK = 5
BLOCK_SOUND_EVENT_BUILD = 6
BLOCK_SOUND_EVENT_CLONE = 7
BLOCK_SOUND_EVENT_HARVEST = 8

# protocol/ItemSoundEvent.java:5-7
ITEM_SOUND_EVENT_DRAG = 0
ITEM_SOUND_EVENT_DROP = 1

# JAX-only namespace tags. Native block and item ordinals overlap.
AUDIO_EVENT_FAMILY_GENERIC = 0
AUDIO_EVENT_FAMILY_BLOCK = 1
AUDIO_EVENT_FAMILY_ITEM = 2
AUDIO_EVENT_NONE = -1
AUDIO_CATEGORY_NONE = -1

# JAX-only routing provenance. An outbound packet watcher observes mode 0
# after native recipient selection; modes 1 and 2 model the two SoundUtil
# routing paths before the packet write.
AUDIO_ROUTING_SERVER_DELIVERED = 0
AUDIO_ROUTING_SPATIAL_BROADCAST = 1
AUDIO_ROUTING_TARGETED_3D = 2

# server/core/asset/type/soundevent/config/SoundEvent.java:131-134
SOUND_EVENT_DEFAULT_START_ATTENUATION_DISTANCE = 2.0
SOUND_EVENT_DEFAULT_MAX_DISTANCE = 16.0
SOUND_EVENT_DEFAULT_MAX_INSTANCE = 50


def _semantic_event_valid(family: Array, ordinal: Array) -> Array:
    generic_event = (family == AUDIO_EVENT_FAMILY_GENERIC) & (
        ordinal == AUDIO_EVENT_NONE
    )
    block_event = (
        (family == AUDIO_EVENT_FAMILY_BLOCK)
        & (ordinal >= BLOCK_SOUND_EVENT_WALK)
        & (ordinal <= BLOCK_SOUND_EVENT_HARVEST)
    )
    item_event = (
        (family == AUDIO_EVENT_FAMILY_ITEM)
        & (ordinal >= ITEM_SOUND_EVENT_DRAG)
        & (ordinal <= ITEM_SOUND_EVENT_DROP)
    )
    return generic_event | block_event | item_event


def _observation_values_valid(observation: AudioObservation) -> Array:
    valid = observation.valid
    packet_2d = observation.packet_id == AUDIO_PACKET_ID_2D
    packet_3d = observation.packet_id == AUDIO_PACKET_ID_3D
    packet_entity = observation.packet_id == AUDIO_PACKET_ID_ENTITY
    packet_valid = packet_2d | packet_3d | packet_entity
    delivered = observation.routing == AUDIO_ROUTING_SERVER_DELIVERED
    broadcast = observation.routing == AUDIO_ROUTING_SPATIAL_BROADCAST
    targeted = observation.routing == AUDIO_ROUTING_TARGETED_3D
    routing_valid = ((packet_2d | packet_entity) & delivered) | (
        packet_3d & (delivered | broadcast | targeted)
    )
    category_valid = (packet_entity & (observation.category == AUDIO_CATEGORY_NONE)) | (
        (packet_2d | packet_3d)
        & (observation.category >= AUDIO_CATEGORY_MUSIC)
        & (observation.category <= AUDIO_CATEGORY_UI)
    )
    source_finite = jnp.all(jnp.isfinite(observation.source_position), axis=2)
    relative_finite = jnp.all(
        jnp.isfinite(observation.relative_position),
        axis=2,
    )
    distance_finite = jnp.isfinite(observation.distance) & (observation.distance >= 0.0)
    max_distance_finite = jnp.isfinite(observation.max_distance) & (
        observation.max_distance >= 0.0
    )
    source_zero = jnp.all(observation.source_position == 0.0, axis=2)
    relative_zero = jnp.all(observation.relative_position == 0.0, axis=2)
    nonspatial_canonical = (
        source_zero
        & relative_zero
        & (observation.distance == 0.0)
        & (observation.max_distance == 0.0)
        & (packet_entity | (observation.source_network_id == 0))
    )
    relative_norm = jnp.sqrt(
        jnp.maximum(
            jnp.sum(
                observation.relative_position * observation.relative_position,
                axis=2,
            ),
            0.0,
        )
    )
    spatial_canonical = (
        source_finite
        & relative_finite
        & distance_finite
        & max_distance_finite
        & (observation.source_network_id == 0)
        & jnp.isclose(
            observation.distance,
            relative_norm,
            rtol=1e-5,
            atol=1e-5,
        )
    )
    slot_valid = (
        packet_valid
        & routing_valid
        & (observation.sound_event_index > 0)
        & category_valid
        & _semantic_event_valid(
            observation.event_family,
            observation.event_ordinal,
        )
        & jnp.isfinite(observation.volume_modifier)
        & jnp.isfinite(observation.pitch_modifier)
        & jnp.isfinite(observation.age_seconds)
        & (observation.age_seconds >= 0.0)
        & (
            (packet_3d & spatial_canonical)
            | ((packet_2d | packet_entity) & nonspatial_canonical)
        )
    )
    slots_valid = jnp.all(~valid | slot_valid, axis=1)
    padding_then_valid = ~jnp.any(
        valid[:, :-1] & ~valid[:, 1:],
        axis=1,
    )
    age_ordered = jnp.all(
        ~(valid[:, :-1] & valid[:, 1:])
        | (observation.age_seconds[:, :-1] >= observation.age_seconds[:, 1:]),
        axis=1,
    )
    return slots_valid & padding_then_valid & age_ordered


def _mask_observation(
    observation: AudioObservation,
    available: Array,
) -> AudioObservation:
    def mask(value: Array) -> Array:
        selector = available.reshape((available.shape[0],) + (1,) * (value.ndim - 1))
        return jnp.where(selector, value, jnp.zeros_like(value))

    return jax.tree.map(mask, observation)
