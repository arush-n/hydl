"""Array coercion and shape helpers internal to audio observations."""
from __future__ import annotations


import jax
import jax.numpy as jnp


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


def _positive_size(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")


def _integer_array(
    value: Array,
    shape: tuple[int, ...],
    label: str,
) -> Array:
    result = jnp.asarray(value)
    if not jnp.issubdtype(result.dtype, jnp.integer):
        raise TypeError(f"{label} must be integer")
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {list(shape)}")
    return result.astype(jnp.int32)


def _float_array(
    value: Array,
    shape: tuple[int, ...],
    label: str,
) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {list(shape)}")
    return result


def _batch_float(
    value: Array | float,
    batch: int,
    label: str,
) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{label} must be scalar or shape [batch]")
    return result


def _append_field(current: Array, value: Array, append: Array) -> Array:
    shifted = jnp.concatenate(
        (current[:, 1:], value[:, None, ...]),
        axis=1,
    )
    mask = append.reshape((append.shape[0],) + (1,) * (current.ndim - 1))
    return jnp.where(mask, shifted, current)
