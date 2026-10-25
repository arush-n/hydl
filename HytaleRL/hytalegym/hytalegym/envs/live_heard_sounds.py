"""The live heard list (S12): the bridge's 3-D footstep events, built as the gen-7 sim builds ``heard_sounds``.

Opt-in ``HytaleEnv(learner_heard_sounds=True)``. One parsed audio frame (``hytalegym/audio.py``) becomes one
``HeardSounds`` (``sim/gen7/mechanics/heard_sounds.py``) with batch 1, K = ``HEARD_CAPACITY`` slots, so a consumer reads
live as it reads the sim. Mirrors ``heard_sounds_fields`` for one listener:

- only 3-D (kind 1), spatial, valid events whose asset id is a footstep sound (``FootstepTables.sound_keys``, MoveIn or
  MoveOut) are kept;
- an event is heard when its distance is below the sound's MaxDistance; the gain is the sound's linear volume times the
  curve of ``jax/world/audio/hearing.py`` ``hear()``, times the event's volume modifier (1.0 in the sim);
- ``source_entity_id``: 0 for the learner's own footsteps (source slot 0); 1 for the target when the bridge reports it
  perceptible; else -1.

Slots are filled in the frame's order (oldest first), then padded to K, as the sim pads.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hytalegym.audio import (
    AUDIO_KIND_3D,
    AUDIO_SOURCE_SLOT_LEARNER,
    AUDIO_SOURCE_SLOT_TARGET,
)
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.world.audio.hearing import FootstepTables
from hytalegym.sim.gen7.mechanics.heard_sounds import HEARD_CAPACITY, HeardSounds


def live_heard_sounds(
    audio: Mapping[str, Any],
    asset_id_of_index: Mapping[int, str],
    tables: FootstepTables,
    target_perceptible: bool,
) -> HeardSounds:
    """The heard list (batch 1) of one parsed audio frame; all slots invalid when nothing is heard."""

    sound_keys = np.asarray(tables.sound_keys)
    sound_start = np.asarray(tables.sound_start, dtype=np.float32)
    sound_max = np.asarray(tables.sound_max, dtype=np.float32)
    sound_volume = np.asarray(tables.sound_volume, dtype=np.float32)
    key_to_row = {int(key): row for row, key in enumerate(sound_keys)}

    valid = np.zeros((1, HEARD_CAPACITY), dtype=np.bool_)
    sound_id = np.zeros((1, HEARD_CAPACITY), dtype=np.int32)
    relative = np.zeros((1, HEARD_CAPACITY, 3), dtype=np.float32)
    distance_out = np.zeros((1, HEARD_CAPACITY), dtype=np.float32)
    gain = np.zeros((1, HEARD_CAPACITY), dtype=np.float32)
    age = np.zeros((1, HEARD_CAPACITY), dtype=np.float32)
    source = np.full((1, HEARD_CAPACITY), -1, dtype=np.int32)

    slot = 0
    for event in range(len(audio["valid"])):
        if not audio["valid"][event] or audio["kind"][event] != AUDIO_KIND_3D or not audio["spatial_valid"][event]:
            continue
        asset_id = asset_id_of_index.get(int(audio["sound_event_index"][event]))
        if asset_id is None:
            continue
        sid = semantic_id(asset_id)
        row = key_to_row.get(sid)
        if row is None:
            continue  # not a footstep sound (MoveIn or MoveOut) in the tables
        offset = np.asarray(audio["relative_position"][event], dtype=np.float32)
        distance = np.float32(np.sqrt(np.sum(offset * offset)))
        maximum, start = sound_max[row], sound_start[row]
        if not distance < maximum:
            continue  # the sim keeps only heard sounds
        span = maximum - start
        if distance <= start:
            curve = np.float32(1.0)
        elif span > 0.0:
            curve = np.float32(np.clip((maximum - distance) / span, 0.0, 1.0))
        else:
            curve = np.float32(0.0)
        if slot >= HEARD_CAPACITY:
            break
        slot_source = int(audio["source_slot"][event])
        if slot_source == AUDIO_SOURCE_SLOT_LEARNER:
            entity = AUDIO_SOURCE_SLOT_LEARNER
        elif slot_source == AUDIO_SOURCE_SLOT_TARGET and target_perceptible:
            entity = AUDIO_SOURCE_SLOT_TARGET
        else:
            entity = -1

        valid[0, slot] = True
        sound_id[0, slot] = sid
        relative[0, slot] = offset
        distance_out[0, slot] = distance
        gain[0, slot] = np.float32(sound_volume[row] * curve * np.float32(audio["volume_modifier"][event]))
        source[0, slot] = entity
        slot += 1

    return HeardSounds(
        valid=valid,
        sound_id=sound_id,
        relative_position=relative,
        distance=distance_out,
        gain=gain,
        age_seconds=age,
        source_entity_id=source,
    )


__all__ = ["live_heard_sounds"]
