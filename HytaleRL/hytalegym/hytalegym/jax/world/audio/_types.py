"""Fixed-shape tuple records internal to the audio package."""
from __future__ import annotations

from typing import NamedTuple

import jax


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


class AudioEventBatch(NamedTuple):
    """Per-environment staged transient emissions in source order."""

    valid: Array
    packet_id: Array
    routing: Array
    sound_event_index: Array
    category: Array
    event_family: Array
    event_ordinal: Array
    source_position: Array
    max_distance: Array
    source_network_id: Array
    volume_modifier: Array
    pitch_modifier: Array
    age_seconds: Array


class AudioObservation(NamedTuple):
    """Chronological fixed-capacity ring; padding precedes valid events."""

    valid: Array
    packet_id: Array
    routing: Array
    sound_event_index: Array
    category: Array
    event_family: Array
    event_ordinal: Array
    source_position: Array
    relative_position: Array
    distance: Array
    max_distance: Array
    source_network_id: Array
    volume_modifier: Array
    pitch_modifier: Array
    age_seconds: Array


class AudioStepResult(NamedTuple):
    """One source-ordered auditory observation update."""

    observation: AudioObservation
    available: Array
    audible_event: Array
    invalid_event: Array
    invalid_previous: Array
    overwritten_count: Array
