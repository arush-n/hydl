"""Fixed-shape, source-audited auditory event observations."""

from __future__ import annotations

import hashlib
import json

import jax
import jax.numpy as jnp

# Re-exported to preserve this module's attribute surface.
from hytalegym.jax.world.audio._arrays import (  # noqa: F401
    _append_field,
    _batch_float,
    _float_array,
    _integer_array,
    _positive_size,
)
from hytalegym.jax.world.audio._normalize import (  # noqa: F401
    _normalize_events,
    _normalize_observation,
)
from hytalegym.jax.world.audio._types import (  # noqa: F401
    AudioEventBatch,
    AudioObservation,
    AudioStepResult,
)
from hytalegym.jax.world.audio._validate import (  # noqa: F401
    _mask_observation,
    _observation_values_valid,
    _semantic_event_valid,
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


def empty_audio_event_batch(
    batch_size: int,
    staging_capacity: int,
) -> AudioEventBatch:
    """Return explicit padding for a fixed incoming-event shape."""

    _positive_size(batch_size, "batch_size")
    _positive_size(staging_capacity, "staging_capacity")
    scalar_shape = (batch_size, staging_capacity)
    return AudioEventBatch(
        valid=jnp.zeros(scalar_shape, dtype=jnp.bool_),
        packet_id=jnp.zeros(scalar_shape, dtype=jnp.int32),
        routing=jnp.full(
            scalar_shape,
            AUDIO_ROUTING_SERVER_DELIVERED,
            dtype=jnp.int32,
        ),
        sound_event_index=jnp.zeros(scalar_shape, dtype=jnp.int32),
        category=jnp.full(
            scalar_shape,
            AUDIO_CATEGORY_NONE,
            dtype=jnp.int32,
        ),
        event_family=jnp.full(
            scalar_shape,
            AUDIO_EVENT_FAMILY_GENERIC,
            dtype=jnp.int32,
        ),
        event_ordinal=jnp.full(
            scalar_shape,
            AUDIO_EVENT_NONE,
            dtype=jnp.int32,
        ),
        source_position=jnp.zeros(
            (*scalar_shape, 3),
            dtype=jnp.float32,
        ),
        max_distance=jnp.zeros(scalar_shape, dtype=jnp.float32),
        source_network_id=jnp.zeros(scalar_shape, dtype=jnp.int32),
        volume_modifier=jnp.zeros(scalar_shape, dtype=jnp.float32),
        pitch_modifier=jnp.zeros(scalar_shape, dtype=jnp.float32),
        age_seconds=jnp.zeros(scalar_shape, dtype=jnp.float32),
    )


def empty_audio_observation(
    batch_size: int,
    event_capacity: int,
) -> AudioObservation:
    """Return an empty caller-sized auditory ring."""

    _positive_size(batch_size, "batch_size")
    _positive_size(event_capacity, "event_capacity")
    scalar_shape = (batch_size, event_capacity)
    vector_shape = (*scalar_shape, 3)
    return AudioObservation(
        valid=jnp.zeros(scalar_shape, dtype=jnp.bool_),
        packet_id=jnp.zeros(scalar_shape, dtype=jnp.int32),
        routing=jnp.full(
            scalar_shape,
            AUDIO_ROUTING_SERVER_DELIVERED,
            dtype=jnp.int32,
        ),
        sound_event_index=jnp.zeros(scalar_shape, dtype=jnp.int32),
        category=jnp.full(
            scalar_shape,
            AUDIO_CATEGORY_NONE,
            dtype=jnp.int32,
        ),
        event_family=jnp.full(
            scalar_shape,
            AUDIO_EVENT_FAMILY_GENERIC,
            dtype=jnp.int32,
        ),
        event_ordinal=jnp.full(
            scalar_shape,
            AUDIO_EVENT_NONE,
            dtype=jnp.int32,
        ),
        source_position=jnp.zeros(vector_shape, dtype=jnp.float32),
        relative_position=jnp.zeros(vector_shape, dtype=jnp.float32),
        distance=jnp.zeros(scalar_shape, dtype=jnp.float32),
        max_distance=jnp.zeros(scalar_shape, dtype=jnp.float32),
        source_network_id=jnp.zeros(scalar_shape, dtype=jnp.int32),
        volume_modifier=jnp.zeros(scalar_shape, dtype=jnp.float32),
        pitch_modifier=jnp.zeros(scalar_shape, dtype=jnp.float32),
        age_seconds=jnp.zeros(scalar_shape, dtype=jnp.float32),
    )


def audio_observation_step(
    previous: AudioObservation,
    listener_position: Array,
    events: AudioEventBatch,
    delta_seconds: Array | float,
) -> AudioStepResult:
    """Append valid audible emissions and retain the last fixed-capacity set.

    The input event axis is chronological. Native spatial broadcasts use the
    KD-tree's strict radius test, while the targeted-player overload uses an
    inclusive radius test. Events captured from the public outbound packet
    adapter are already recipient-routed and must not be range-filtered again.
    Two-dimensional and entity-attached packets are always server-delivered.
    Incoming ages are measured at the observation boundary, cannot exceed the
    current step duration, and must run oldest-to-newest with the event axis.

    Structural inputs are mandatory and raise when absent or malformed.
    Invalid event values are ignored and diagnosed. Invalid previous state,
    listener, or time values fail the whole environment closed through
    ``available``; no Python branch depends on an array value.
    """

    state = _normalize_observation(previous)
    incoming = _normalize_events(events)
    batch, capacity = state.valid.shape
    if incoming.valid.shape[0] != batch:
        raise ValueError("events batch axis must match previous")
    listeners = jnp.asarray(listener_position, dtype=jnp.float32)
    if listeners.shape != (batch, 3):
        raise ValueError("listener_position must have shape [batch, 3]")
    delta = _batch_float(delta_seconds, batch, "delta_seconds")

    input_valid = (
        jnp.all(jnp.isfinite(listeners), axis=1) & jnp.isfinite(delta) & (delta >= 0.0)
    )
    previous_valid = _observation_values_valid(state)
    candidate_age = (
        state.age_seconds
        + jnp.where(
            jnp.isfinite(delta) & (delta >= 0.0),
            delta,
            0.0,
        )[:, None]
    )
    age_update_valid = jnp.all(
        ~state.valid | jnp.isfinite(candidate_age),
        axis=1,
    )
    available = input_valid & previous_valid & age_update_valid

    packet_2d = incoming.packet_id == AUDIO_PACKET_ID_2D
    packet_3d = incoming.packet_id == AUDIO_PACKET_ID_3D
    packet_entity = incoming.packet_id == AUDIO_PACKET_ID_ENTITY
    packet_valid = packet_2d | packet_3d | packet_entity
    delivered = incoming.routing == AUDIO_ROUTING_SERVER_DELIVERED
    broadcast = incoming.routing == AUDIO_ROUTING_SPATIAL_BROADCAST
    targeted = incoming.routing == AUDIO_ROUTING_TARGETED_3D
    routing_valid = ((packet_2d | packet_entity) & delivered) | (
        packet_3d & (delivered | broadcast | targeted)
    )
    category_valid = (packet_entity & (incoming.category == AUDIO_CATEGORY_NONE)) | (
        (packet_2d | packet_3d)
        & (incoming.category >= AUDIO_CATEGORY_MUSIC)
        & (incoming.category <= AUDIO_CATEGORY_UI)
    )
    semantic_event_valid = _semantic_event_valid(
        incoming.event_family,
        incoming.event_ordinal,
    )
    event_age_valid = (
        jnp.isfinite(incoming.age_seconds)
        & (incoming.age_seconds >= 0.0)
        & (incoming.age_seconds <= delta[:, None])
    )
    earlier = jnp.arange(incoming.valid.shape[1])[:, None]
    later = jnp.arange(incoming.valid.shape[1])[None, :]
    age_order_violation = jnp.any(
        (earlier < later)[None, :, :]
        & incoming.valid[:, :, None]
        & incoming.valid[:, None, :]
        & (incoming.age_seconds[:, :, None] < incoming.age_seconds[:, None, :]),
        axis=(1, 2),
    )
    event_age_valid = event_age_valid & ~age_order_violation[:, None]
    position_finite = jnp.all(
        jnp.isfinite(incoming.source_position),
        axis=2,
    )
    position_zero = jnp.all(incoming.source_position == 0.0, axis=2)
    max_distance_valid = jnp.isfinite(incoming.max_distance) & (
        incoming.max_distance >= 0.0
    )
    safe_listeners = jnp.where(input_valid[:, None], listeners, 0.0)
    relative = incoming.source_position - safe_listeners[:, None, :]
    squared_distance = jnp.sum(relative * relative, axis=2)
    distance = jnp.sqrt(jnp.maximum(squared_distance, 0.0))
    radius_squared = incoming.max_distance * incoming.max_distance
    derived_geometry_valid = (
        jnp.all(jnp.isfinite(relative), axis=2)
        & jnp.isfinite(squared_distance)
        & jnp.isfinite(distance)
        & (delivered | jnp.isfinite(radius_squared))
    )
    nonspatial_canonical = (
        position_zero
        & (incoming.max_distance == 0.0)
        & (packet_entity | (incoming.source_network_id == 0))
    )
    spatial_canonical = (
        position_finite
        & max_distance_valid
        & derived_geometry_valid
        & (incoming.source_network_id == 0)
    )
    value_valid = (
        packet_valid
        & routing_valid
        & (incoming.sound_event_index > 0)
        & category_valid
        & semantic_event_valid
        & event_age_valid
        & jnp.isfinite(incoming.volume_modifier)
        & jnp.isfinite(incoming.pitch_modifier)
        & (
            (packet_3d & spatial_canonical)
            | ((packet_2d | packet_entity) & nonspatial_canonical)
        )
    )
    invalid_event = incoming.valid & ~value_valid

    route_reachable = (
        ~packet_3d
        | delivered
        | (broadcast & (squared_distance < radius_squared))
        | (targeted & (squared_distance <= radius_squared))
    )
    audible = incoming.valid & value_valid & available[:, None] & route_reachable

    safe_state = _mask_observation(state, available)
    safe_delta = jnp.where(available, delta, 0.0)
    aged = safe_state._replace(
        age_seconds=jnp.where(
            safe_state.valid,
            safe_state.age_seconds + safe_delta[:, None],
            0.0,
        )
    )
    overwritten = jnp.zeros((batch,), dtype=jnp.int32)

    def append_one(
        index: int,
        carry: tuple[AudioObservation, Array],
    ) -> tuple[AudioObservation, Array]:
        observation, overwritten_count = carry
        append = audible[:, index]
        spatial = packet_3d[:, index]
        event_relative = jnp.where(
            spatial[:, None],
            relative[:, index, :],
            0.0,
        )
        event_position = jnp.where(
            spatial[:, None],
            incoming.source_position[:, index, :],
            0.0,
        )
        event_distance = jnp.where(
            spatial,
            distance[:, index],
            0.0,
        )
        next_observation = AudioObservation(
            valid=_append_field(
                observation.valid,
                jnp.ones((batch,), dtype=jnp.bool_),
                append,
            ),
            packet_id=_append_field(
                observation.packet_id,
                incoming.packet_id[:, index],
                append,
            ),
            routing=_append_field(
                observation.routing,
                incoming.routing[:, index],
                append,
            ),
            sound_event_index=_append_field(
                observation.sound_event_index,
                incoming.sound_event_index[:, index],
                append,
            ),
            category=_append_field(
                observation.category,
                incoming.category[:, index],
                append,
            ),
            event_family=_append_field(
                observation.event_family,
                incoming.event_family[:, index],
                append,
            ),
            event_ordinal=_append_field(
                observation.event_ordinal,
                incoming.event_ordinal[:, index],
                append,
            ),
            source_position=_append_field(
                observation.source_position,
                event_position,
                append,
            ),
            relative_position=_append_field(
                observation.relative_position,
                event_relative,
                append,
            ),
            distance=_append_field(
                observation.distance,
                event_distance,
                append,
            ),
            max_distance=_append_field(
                observation.max_distance,
                jnp.where(
                    spatial,
                    incoming.max_distance[:, index],
                    0.0,
                ),
                append,
            ),
            source_network_id=_append_field(
                observation.source_network_id,
                incoming.source_network_id[:, index],
                append,
            ),
            volume_modifier=_append_field(
                observation.volume_modifier,
                incoming.volume_modifier[:, index],
                append,
            ),
            pitch_modifier=_append_field(
                observation.pitch_modifier,
                incoming.pitch_modifier[:, index],
                append,
            ),
            age_seconds=_append_field(
                observation.age_seconds,
                incoming.age_seconds[:, index],
                append,
            ),
        )
        overwritten_count = overwritten_count + (
            append & observation.valid[:, 0]
        ).astype(jnp.int32)
        return next_observation, overwritten_count

    observation, overwritten = jax.lax.fori_loop(
        0,
        incoming.valid.shape[1],
        append_one,
        (aged, overwritten),
    )
    observation = _mask_observation(observation, available)
    if observation.valid.shape != (batch, capacity):
        raise AssertionError("audio ring shape changed during update")
    return AudioStepResult(
        observation=observation,
        available=available,
        audible_event=audible,
        invalid_event=invalid_event,
        invalid_previous=~previous_valid,
        overwritten_count=overwritten,
    )


def audio_contract() -> dict[str, object]:
    """Return the pinnable model-only auditory observation contract."""

    return {
        "schema": AUDIO_OBSERVATION_SCHEMA,
        "version": AUDIO_OBSERVATION_VERSION,
        "function": "audio_observation_step",
        "transport": {
            "packet_ids": {
                "PlaySoundEvent2D": AUDIO_PACKET_ID_2D,
                "PlaySoundEvent3D": AUDIO_PACKET_ID_3D,
                "PlaySoundEventEntity": AUDIO_PACKET_ID_ENTITY,
            },
            "common_fields": [
                "sound_event_index",
                "volume_modifier",
                "pitch_modifier",
            ],
            "category_values": {
                "Music": AUDIO_CATEGORY_MUSIC,
                "Ambient": AUDIO_CATEGORY_AMBIENT,
                "SFX": AUDIO_CATEGORY_SFX,
                "UI": AUDIO_CATEGORY_UI,
            },
            "packet_spatial_fields": ["position"],
            "source_asset_enrichment": ["max_distance"],
            "entity_field": "network_id",
        },
        "semantic_roles": {
            "block": {
                "Walk": BLOCK_SOUND_EVENT_WALK,
                "Land": BLOCK_SOUND_EVENT_LAND,
                "MoveIn": BLOCK_SOUND_EVENT_MOVE_IN,
                "MoveOut": BLOCK_SOUND_EVENT_MOVE_OUT,
                "Hit": BLOCK_SOUND_EVENT_HIT,
                "Break": BLOCK_SOUND_EVENT_BREAK,
                "Build": BLOCK_SOUND_EVENT_BUILD,
                "Clone": BLOCK_SOUND_EVENT_CLONE,
                "Harvest": BLOCK_SOUND_EVENT_HARVEST,
            },
            "item": {
                "Drag": ITEM_SOUND_EVENT_DRAG,
                "Drop": ITEM_SOUND_EVENT_DROP,
            },
            "verified_server_fired": {
                "block": [
                    "Walk",
                    "MoveIn",
                    "MoveOut",
                    "Hit",
                    "Break",
                    "Harvest",
                ],
                "item": ["Drop"],
            },
            "asset_or_protocol_only_not_verified_server_fired": {
                "block": ["Land", "Build", "Clone"],
                "item": ["Drag"],
            },
            "generic_family": (
                "required_when_the_transport_emission_has_no_preserved"
                "_block_or_item_role"
            ),
        },
        "sound_event_defaults": {
            "start_attenuation_distance": (
                SOUND_EVENT_DEFAULT_START_ATTENUATION_DISTANCE
            ),
            "max_distance": SOUND_EVENT_DEFAULT_MAX_DISTANCE,
            "max_instance": SOUND_EVENT_DEFAULT_MAX_INSTANCE,
            "runtime_max_distance_input_required": True,
            "outbound_packet_omits_max_distance": True,
        },
        "routing": {
            "2d": "server_recipient_routed_no_distance_test",
            "3d": {
                "server_delivered": {
                    "tag": AUDIO_ROUTING_SERVER_DELIVERED,
                    "predicate": "already_routed_no_second_distance_test",
                },
                "spatial_broadcast": {
                    "tag": AUDIO_ROUTING_SPATIAL_BROADCAST,
                    "predicate": "distanceSquared_strict_less_than_maxDistance_squared",
                },
                "targeted_player": {
                    "tag": AUDIO_ROUTING_TARGETED_3D,
                    "predicate": "distanceSquared_less_equal_maxDistance_squared",
                },
            },
            "entity": ("server_recipient_routed_entity_position_not_in_packet"),
            "routing_tag": "jax_only_provenance_not_a_native_packet_field",
            "hidden_source_predicate": (
                "required_for_upstream_broadcast_staging_but_already_reflected"
                "_by_server_delivered_capture"
            ),
        },
        "observable_server_api": {
            "outbound_interceptor": (
                "PacketAdapters.registerOutbound(PlayerPacketWatcher)"
            ),
            "recipient": "PlayerRef",
            "packet": "Packet",
            "cleanup": "PacketAdapters.deregisterOutbound(returned_PacketFilter)",
            "call_order": "inline_before_packet_cache_and_Netty_write",
            "watcher_suppresses_packet": False,
            "ordering_caveat": (
                "an_earlier_registered_filter_returning_true_prevents_later"
                "_watchers_from_observing_the_write"
            ),
            "stability": "public_in_installed_0_5_7_not_api_stability_certified",
            "sound_specific_event_bus": False,
            "scope": "connected_PlayerRef_recipients_only",
        },
        "agent_integration": {
            "connected_player": {
                "producer": "outbound_PlayerPacketWatcher",
                "status": "source_supported_bridge_unimplemented",
            },
            "native_npc": {
                "producer": "HeadlessPacketHandler_step_buffer",
                "status": "partial_available_full_spatial_unavailable",
                "reasons": [
                    "controlled_actor_is_NPCEntity_not_Player",
                    "PlayerSpatialSystem_indexes_Player_plus_Transform",
                    "headless_tick_anchor_is_not_an_entity_listener",
                    (
                        "HeadlessPacketHandler_observes_world_broadcasts_but"
                        "_is_not_selected_by_PlayerSpatialSystem"
                    ),
                ],
                "partial_headless_capture": {
                    "status": "implemented_agent_visible_and_incomplete",
                    "producer": "HeadlessPacketHandler_step_buffer",
                    "covers": [
                        "world_broadcast_PlaySoundEvent2D",
                        "world_broadcast_PlaySoundEventEntity",
                    ],
                    "why": (
                        "PlayerUtil_broadcasts_iterate_world_PlayerRefs_and_the"
                        "_tick_anchor_is_tracked_as_one"
                    ),
                    "does_not_cover": [
                        "PlaySoundEvent3D_PlayerSpatialSystem_delivery",
                        "targeted_PlayerRef_writes_to_other_recipients",
                        "entity_direction_without_network_replication_join",
                    ],
                },
                "wire_contract": {
                    "schema": "hytalerl_audio_frame_v1",
                    "capture_mask_order": ["2d", "3d", "entity"],
                    "current_capture_mask": [True, False, True],
                    "capacity": 16,
                    "parser": "HytaleEnv_padding_first_strict_fixed_shape",
                    "availability_semantics": (
                        "structurally_valid_not_complete_capture"
                    ),
                },
                "required_fix": (
                    "world_scoped_emission_hook_routed_at_emission_time"
                    "_against_the_controlled_NPC_position"
                ),
            },
            "simulator": {
                "producer": "task_authored_synthetic_emissions",
                "status": "unimplemented_and_must_remain_labelled_modelled",
            },
            "entity_packet_join": (
                "network_id_requires_entity_replication_lookup_for_direction"
            ),
        },
        "step_frame": {
            "collection": "events_emitted_during_the_current_agent_step",
            "routing_time": "listener_position_at_each_emission_not_step_end",
            "event_age": "seconds_before_immutable_step_observation_boundary",
            "event_age_bounds": "zero_through_delta_seconds",
            "event_order": "oldest_to_newest_nonincreasing_age",
            "reset": "clear_after_warmup_and_fixture_setup",
            "observe": "idempotent_return_last_frame_without_draining",
            "serialization": "serialize_attached_immutable_frame_never_live_queue",
            "overflow": "retain_last_N_and_report_overwritten_count",
        },
        "policy_consumption": {
            "surface": "masked_structured_ring",
            "categorical_fields": "embed_or_one_hot_do_not_scale_as_continuous",
            "spatial_fields": "relative_position_distance_and_valid_mask",
            "recurrent_agents": "age_seconds_preserves_cross_step_recency",
            "reward": "observation_only_no_implicit_audio_reward_shaping",
        },
        "ring": {
            "capacity": "caller_static_positive_integer",
            "incoming_capacity": "caller_static_positive_integer",
            "ordering": "padding_then_oldest_to_newest",
            "overflow": "retain_last_N_audible_events",
            "incoming_age": "step_boundary_age_zero_through_delta_seconds",
            "batching": "all_shapes_fixed_and_jit_vmap_safe",
        },
        "fail_closed": {
            "missing_or_malformed_structure": "raise",
            "invalid_listener": "unavailable_and_zero_observation",
            "invalid_event": "diagnostic_true_and_not_appended",
            "invalid_event_age": (
                "nonfinite_negative_after_step_or_out_of_order_not_appended"
            ),
            "invalid_previous": "diagnostic_true_unavailable_and_zero_observation",
            "sound_event_index_zero": "invalid_native_empty_id",
            "missing_max_distance": "malformed_or_invalid_not_defaulted",
        },
        "source": {
            "enums": [
                "protocol/BlockSoundEvent.java:5-14",
                "protocol/ItemSoundEvent.java:5-7",
                "protocol/SoundCategory.java:5-9",
            ],
            "packets": [
                "protocol/packets/world/PlaySoundEvent2D.java:12-24",
                "protocol/packets/world/PlaySoundEvent3D.java:14-28",
                "protocol/packets/world/PlaySoundEventEntity.java:11-22",
            ],
            "emission_and_routing": [
                "server/core/universe/world/SoundUtil.java:44-64",
                "server/core/universe/world/SoundUtil.java:68-101",
                "server/core/universe/world/SoundUtil.java:106-140",
                "server/core/universe/world/SoundUtil.java:273-288",
                "component/spatial/KDTree.java:99-102",
                "component/spatial/KDTree.java:275-309",
            ],
            "outbound_observation": [
                "server/core/io/PacketHandler.java:221-246",
                "server/core/io/adapter/PacketAdapters.java:31-42",
                "server/core/io/adapter/PacketAdapters.java:74-95",
                "server/core/io/adapter/PacketAdapters.java:116-118",
                "server/core/io/adapter/PlayerPacketWatcher.java:7-9",
            ],
            "bridge_agent_backend": [
                "nativebackend/NativeEnvironmentSession.java:1578-1605",
                "nativebackend/NativeEnvironmentSession.java:1761-1775",
                "nativebackend/HeadlessPacketHandler.java:26-47",
                "server/core/universe/world/PlayerUtil.java:91-112",
                "server/core/modules/entity/system/PlayerSpatialSystem.java:19-27",
            ],
        },
        "fidelity": {
            "status": "model_only",
            "certified": False,
            "evidence": "surrogate_audio_source_audit.py",
            "native_compare": None,
        },
        "unsupported": [
            "client_attenuation_curve",
            "client_layer_probability_and_randomization",
            "client_audio_file_realization",
            "persistent_AudioComponent_AudioUpdate_loops",
            "complete_native_NPC_3D_emission_capture",
        ],
    }


def audio_contract_sha256() -> str:
    payload = json.dumps(
        audio_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AUDIO_CATEGORY_AMBIENT",
    "AUDIO_CATEGORY_MUSIC",
    "AUDIO_CATEGORY_NONE",
    "AUDIO_CATEGORY_SFX",
    "AUDIO_CATEGORY_UI",
    "AUDIO_EVENT_FAMILY_BLOCK",
    "AUDIO_EVENT_FAMILY_GENERIC",
    "AUDIO_EVENT_FAMILY_ITEM",
    "AUDIO_EVENT_NONE",
    "AUDIO_OBSERVATION_SCHEMA",
    "AUDIO_OBSERVATION_VERSION",
    "AUDIO_PACKET_ID_2D",
    "AUDIO_PACKET_ID_3D",
    "AUDIO_PACKET_ID_ENTITY",
    "AUDIO_ROUTING_SERVER_DELIVERED",
    "AUDIO_ROUTING_SPATIAL_BROADCAST",
    "AUDIO_ROUTING_TARGETED_3D",
    "BLOCK_SOUND_EVENT_BREAK",
    "BLOCK_SOUND_EVENT_BUILD",
    "BLOCK_SOUND_EVENT_CLONE",
    "BLOCK_SOUND_EVENT_HARVEST",
    "BLOCK_SOUND_EVENT_HIT",
    "BLOCK_SOUND_EVENT_LAND",
    "BLOCK_SOUND_EVENT_MOVE_IN",
    "BLOCK_SOUND_EVENT_MOVE_OUT",
    "BLOCK_SOUND_EVENT_WALK",
    "ITEM_SOUND_EVENT_DRAG",
    "ITEM_SOUND_EVENT_DROP",
    "SOUND_EVENT_DEFAULT_MAX_DISTANCE",
    "SOUND_EVENT_DEFAULT_MAX_INSTANCE",
    "SOUND_EVENT_DEFAULT_START_ATTENUATION_DISTANCE",
    "AudioEventBatch",
    "AudioObservation",
    "AudioStepResult",
    "audio_contract",
    "audio_contract_sha256",
    "audio_observation_step",
    "empty_audio_event_batch",
    "empty_audio_observation",
]
