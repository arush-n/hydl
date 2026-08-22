"""Normalization of incoming audio observations and event batches."""
from __future__ import annotations


import jax
import jax.numpy as jnp

from hytalegym.jax.world.audio._arrays import (
    _float_array,
    _integer_array,
)
from hytalegym.jax.world.audio._types import (
    AudioEventBatch,
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


def _normalize_observation(value: AudioObservation) -> AudioObservation:
    if not isinstance(value, AudioObservation):
        raise TypeError("previous must be AudioObservation")
    valid = jnp.asarray(value.valid)
    if valid.dtype != jnp.bool_:
        raise TypeError("previous.valid must be boolean")
    if valid.ndim != 2 or valid.shape[0] == 0 or valid.shape[1] == 0:
        raise ValueError("previous.valid must have non-empty shape [batch, capacity]")
    scalar_shape = valid.shape
    vector_shape = (*scalar_shape, 3)
    normalized = AudioObservation(
        valid=valid,
        packet_id=_integer_array(
            value.packet_id,
            scalar_shape,
            "previous.packet_id",
        ),
        routing=_integer_array(
            value.routing,
            scalar_shape,
            "previous.routing",
        ),
        sound_event_index=_integer_array(
            value.sound_event_index,
            scalar_shape,
            "previous.sound_event_index",
        ),
        category=_integer_array(
            value.category,
            scalar_shape,
            "previous.category",
        ),
        event_family=_integer_array(
            value.event_family,
            scalar_shape,
            "previous.event_family",
        ),
        event_ordinal=_integer_array(
            value.event_ordinal,
            scalar_shape,
            "previous.event_ordinal",
        ),
        source_position=_float_array(
            value.source_position,
            vector_shape,
            "previous.source_position",
        ),
        relative_position=_float_array(
            value.relative_position,
            vector_shape,
            "previous.relative_position",
        ),
        distance=_float_array(
            value.distance,
            scalar_shape,
            "previous.distance",
        ),
        max_distance=_float_array(
            value.max_distance,
            scalar_shape,
            "previous.max_distance",
        ),
        source_network_id=_integer_array(
            value.source_network_id,
            scalar_shape,
            "previous.source_network_id",
        ),
        volume_modifier=_float_array(
            value.volume_modifier,
            scalar_shape,
            "previous.volume_modifier",
        ),
        pitch_modifier=_float_array(
            value.pitch_modifier,
            scalar_shape,
            "previous.pitch_modifier",
        ),
        age_seconds=_float_array(
            value.age_seconds,
            scalar_shape,
            "previous.age_seconds",
        ),
    )
    return normalized


def _normalize_events(value: AudioEventBatch) -> AudioEventBatch:
    if not isinstance(value, AudioEventBatch):
        raise TypeError("events must be AudioEventBatch")
    valid = jnp.asarray(value.valid)
    if valid.dtype != jnp.bool_:
        raise TypeError("events.valid must be boolean")
    if valid.ndim != 2 or valid.shape[0] == 0 or valid.shape[1] == 0:
        raise ValueError("events.valid must have non-empty shape [batch, staging]")
    scalar_shape = valid.shape
    return AudioEventBatch(
        valid=valid,
        packet_id=_integer_array(
            value.packet_id,
            scalar_shape,
            "events.packet_id",
        ),
        routing=_integer_array(
            value.routing,
            scalar_shape,
            "events.routing",
        ),
        sound_event_index=_integer_array(
            value.sound_event_index,
            scalar_shape,
            "events.sound_event_index",
        ),
        category=_integer_array(
            value.category,
            scalar_shape,
            "events.category",
        ),
        event_family=_integer_array(
            value.event_family,
            scalar_shape,
            "events.event_family",
        ),
        event_ordinal=_integer_array(
            value.event_ordinal,
            scalar_shape,
            "events.event_ordinal",
        ),
        source_position=_float_array(
            value.source_position,
            (*scalar_shape, 3),
            "events.source_position",
        ),
        max_distance=_float_array(
            value.max_distance,
            scalar_shape,
            "events.max_distance",
        ),
        source_network_id=_integer_array(
            value.source_network_id,
            scalar_shape,
            "events.source_network_id",
        ),
        volume_modifier=_float_array(
            value.volume_modifier,
            scalar_shape,
            "events.volume_modifier",
        ),
        pitch_modifier=_float_array(
            value.pitch_modifier,
            scalar_shape,
            "events.pitch_modifier",
        ),
        age_seconds=_float_array(
            value.age_seconds,
            scalar_shape,
            "events.age_seconds",
        ),
    )
