"""Strict host transport for UUID-pinned native NPC transition traces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from hytalegym.geometry.contract import CELL_COUNT, CELL_RADIUS, parse_geometry
from hytalegym.worldgen.native_channels import (
    NATIVE_PERCEPTION_CHANNEL_SCHEMA,
    NATIVE_PERCEPTION_CHANNEL_VERSION,
    NativePerceptionChannelCapture,
    native_perception_channel_capture_from_wire,
)


NATIVE_NPC_TRACE_SCHEMA = "hytalerl_native_npc_transition_trace_v13"
NATIVE_NPC_TRACE_VERSION = 13
NATIVE_NPC_TRACE_STATE_LAYOUT = (
    "x,y,z,vx,vy,vz,body_yaw,body_pitch,body_roll,head_yaw,head_pitch,"
    "health,max_health,on_ground"
)
NATIVE_NPC_TRACE_CONTROL_LAYOUT = (
    "body_x,body_y,body_z,body_yaw,body_pitch,head_yaw,head_pitch,"
    "body_turn_speed,head_turn_speed"
)
NATIVE_NPC_TRACE_CONTROL_MASK_LAYOUT = (
    "bit0=body_translation,bit1=body_yaw,bit2=body_pitch,bit3=head_yaw,"
    "bit4=head_pitch,bit5=body_steering_present,bit6=head_steering_present"
)
NATIVE_NPC_TRACE_ATTACK_ACTION_LAYOUT = (
    "label:utf8,active:bool,triggered:bool,ready:bool,"
    "aiming_seconds_remaining:f32,charge_seconds:f32,"
    "interaction_type:utf8,interaction_id:utf8,path:utf8"
)
NATIVE_NPC_ATTACK_EXECUTION_CAUSE_LAYOUT = (
    "available:bool,executed:bool,candidate_index:i32,interaction_type:utf8,"
    "interaction_id:utf8,path:utf8"
)
NATIVE_NPC_TRACE_INTERACTION_LAYOUT = (
    "source:utf8,type:utf8,base_type:utf8,chain_id:i32,initial_root_id:utf8,root_id:utf8,"
    "server_state:utf8,client_state:utf8,final_state:utf8,time_seconds:f32,"
    "time_shift:f32,operation_counter:i32,simulated_operation_counter:i32,"
    "operation_index:i32,client_operation_index:i32,call_depth:i32,"
    "simulated_call_depth:i32,predicted:bool,requires_client:bool,"
    "first_run:bool,pre_ticked:bool,desynced:bool,target_uuid:uuid?"
)
NATIVE_NPC_TRACE_INTERNAL_LAYOUT = (
    "state_name:utf8,state_index:i32,substate_index:i32,busy:bool,"
    "transitioning:bool,role_change_requested:bool,terminal_action:bool,"
    "backing_away:bool,steering_motion:utf8,motion_controller_present:bool,"
    "motion_in_progress:bool,obstructed:bool,current_speed:f64,"
    "maximum_speed:f64,avoidance_steering:f64[3],"
    "separation_steering:f64[3],marked_targets:marked_target[]"
)
NATIVE_NPC_DAMAGE_EVENT_LAYOUT = (
    "source_type:utf8,environment_type:utf8,damage_cause_index:i32,"
    "damage_cause_id:utf8,initial_amount:f32,final_amount:f32,"
    "cancelled:bool,blocked:bool,actor_source:bool,actor_target:bool,"
    "source_entity_index:i32,source_uuid:uuid?,target_entity_index:i32,"
    "target_uuid:uuid?,projectile_entity_index:i32,projectile_uuid:uuid?,"
    "hit_location_available:bool,hit_location:f64[4],health_available:bool,"
    "target_health_after:f32,target_max_health:f32,lethal:bool"
)
NATIVE_NPC_DAMAGE_EVENT_CAPACITY = 32
NATIVE_NPC_EVENT_CONTRACT_SCHEMA = "hytalerl_native_npc_event_contract_v1"
NATIVE_NPC_EVENT_CONTRACT_VERSION = 1
_NATIVE_NPC_EVENT_CONTRACT = {
    "schema": NATIVE_NPC_EVENT_CONTRACT_SCHEMA,
    "version": NATIVE_NPC_EVENT_CONTRACT_VERSION,
    "boundary": "inspect_damage_group_after_apply_damage",
    "layout": NATIVE_NPC_DAMAGE_EVENT_LAYOUT,
    "capacity": NATIVE_NPC_DAMAGE_EVENT_CAPACITY,
    "overflow": "count_gt_emitted_events",
    "perspective": "actor_source_and_actor_target_are_trace_relative",
    "amounts": {
        "initial": "pre_filter_server_damage_amount",
        "final": "post_filter_server_amount_rounded_when_applied",
    },
}
NATIVE_NPC_EVENT_CONTRACT_SHA256 = (
    hashlib.sha256(
        json.dumps(
            _NATIVE_NPC_EVENT_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    .hexdigest()
    .upper()
)
NATIVE_NPC_LIFECYCLE_EVENT_LAYOUT = (
    "kind:utf8,origin:utf8,subject:utf8,key:utf8,index:i32,value_before:f64,"
    "value_after:f64,auxiliary_0:f64,auxiliary_1:f64,text_before:utf8,"
    "text_after:utf8,successful:bool,complete:bool,entity_index:i32,"
    "entity_uuid:uuid?,owner_uuid:uuid?,position_available:bool,"
    "position:f64[3],flags:i32"
)
NATIVE_NPC_LIFECYCLE_EVENT_CAPACITY = 64
NATIVE_NPC_LIFECYCLE_SOURCE_LAYOUT = (
    "bit0=stats,bit1=status,bit2=inventory,bit3=projectiles,bit4=actor_lifecycle"
)
NATIVE_NPC_LIFECYCLE_SOURCE_STATS = 1
NATIVE_NPC_LIFECYCLE_SOURCE_STATUS = 1 << 1
NATIVE_NPC_LIFECYCLE_SOURCE_INVENTORY = 1 << 2
NATIVE_NPC_LIFECYCLE_SOURCE_PROJECTILES = 1 << 3
NATIVE_NPC_LIFECYCLE_SOURCE_ACTOR_LIFECYCLE = 1 << 4
NATIVE_NPC_LIFECYCLE_ALL_SOURCE_BITS = (1 << 5) - 1
NATIVE_NPC_LIFECYCLE_CONTRACT_SCHEMA = "hytalerl_native_npc_lifecycle_contract_v1"
NATIVE_NPC_LIFECYCLE_CONTRACT_VERSION = 1
_NATIVE_NPC_LIFECYCLE_CONTRACT = {
    "schema": NATIVE_NPC_LIFECYCLE_CONTRACT_SCHEMA,
    "version": NATIVE_NPC_LIFECYCLE_CONTRACT_VERSION,
    "boundary": "inter_row_context_and_pre_behavior_to_after_native_tick",
    "capacity": NATIVE_NPC_LIFECYCLE_EVENT_CAPACITY,
    "event_order": (
        "inter_row_deltas_then_server_inventory_arrival_then_transition_deltas"
    ),
    "layout": NATIVE_NPC_LIFECYCLE_EVENT_LAYOUT,
    "missingness": (
        "available_union_partial_if_any_observed_subinterval_unavailable_or_overflow"
    ),
    "origins": (
        "server_inventory_event_or_server_boundary_state_delta_or_"
        "server_inter_row_state_delta_or_server_death_component"
    ),
    "source_layout": NATIVE_NPC_LIFECYCLE_SOURCE_LAYOUT,
}
NATIVE_NPC_LIFECYCLE_CONTRACT_SHA256 = (
    hashlib.sha256(
        json.dumps(
            _NATIVE_NPC_LIFECYCLE_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    .hexdigest()
    .upper()
)
NATIVE_NPC_ACTION_CONTRACT_SCHEMA = "hytalerl_native_npc_action_contract_v2"
NATIVE_NPC_ACTION_CONTRACT_VERSION = 2
_NATIVE_NPC_ACTION_CONTRACT = {
    "schema": NATIVE_NPC_ACTION_CONTRACT_SCHEMA,
    "version": NATIVE_NPC_ACTION_CONTRACT_VERSION,
    "boundary": {
        "observation": "after_pre_behavior_support_before_behavior",
        "decision": "after_behavior_and_avoidance_before_bridge_and_steering",
        "next_observation": "after_native_tick",
    },
    "control": {
        "layout": NATIVE_NPC_TRACE_CONTROL_LAYOUT,
        "mask_layout": NATIVE_NPC_TRACE_CONTROL_MASK_LAYOUT,
        "native_control": "true_if_no_bridge_override_was_marked_for_transition",
    },
    "attack": {
        "layout": NATIVE_NPC_TRACE_ATTACK_ACTION_LAYOUT,
        "activation": (
            "selected_attack_action_active_and_pre_behavior_attack_action_inactive"
        ),
        "execution": "combat_support_execution_at_decision_and_next_boundaries",
        "execution_cause": {
            "availability": "false_on_candidate_overflow_or_non_unique_join",
            "candidate_join": (
                "same_path_next_candidate_interaction_id_equals_chain_initial_root_id"
            ),
            "interaction_start": "decision_boundary_combat_support_first_run",
            "layout": NATIVE_NPC_ATTACK_EXECUTION_CAUSE_LAYOUT,
        },
    },
    "interaction": {"layout": NATIVE_NPC_TRACE_INTERACTION_LAYOUT},
    "internal": {"layout": NATIVE_NPC_TRACE_INTERNAL_LAYOUT},
    "identity": {
        "labels": "raw_java_strings",
        "paths": "behavior_tree_paths",
        "symbols": "capture_specific_table_sha256_stamped_separately",
    },
}
NATIVE_NPC_ACTION_CONTRACT_SHA256 = (
    hashlib.sha256(
        json.dumps(
            _NATIVE_NPC_ACTION_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    .hexdigest()
    .upper()
)
NATIVE_NPC_WORLDVIEW_SCHEMA = "hytalerl_npc_world_snapshot_v4"
NATIVE_NPC_WORLDVIEW_VERSION = 4
NATIVE_NPC_WORLDVIEW_ACTOR_CAPACITY = 256
NATIVE_NPC_WORLDVIEW_ENTITY_CAPACITY = 512
NATIVE_NPC_WORLDVIEW_ACTOR_LAYOUT = (
    "uuid:uuid,role:utf8,state:f64[14],state_name:utf8,target_uuid:uuid?,"
    "combat_attack:bool,attack_pause_seconds:f32"
)
NATIVE_NPC_WORLDVIEW_ENTITY_LAYOUT = (
    "entity_index:i32,uuid:uuid?,flags:i32,asset_id:utf8,model_asset_id:utf8,"
    "position:f64[3],rotation:f64[3],velocity_available:bool,velocity:f64[3],"
    "bounds_available:bool,bounds:f64[6],owner_uuid:uuid?,physics_state:utf8,"
    "lifecycle_timing_available:bool,lifecycle_start_epoch_second:i64,"
    "lifecycle_start_nano:i32,lifecycle_end_epoch_second:i64,"
    "lifecycle_end_nano:i32"
)
NATIVE_NPC_WORLDVIEW_ENTITY_NPC = 1
NATIVE_NPC_WORLDVIEW_ENTITY_LEGACY_PROJECTILE = 1 << 1
NATIVE_NPC_WORLDVIEW_ENTITY_STANDARD_PROJECTILE = 1 << 2
NATIVE_NPC_WORLDVIEW_ENTITY_PREDICTED_PROJECTILE = 1 << 3
NATIVE_NPC_WORLDVIEW_ENTITY_ON_GROUND = 1 << 4
NATIVE_NPC_WORLDVIEW_ENTITY_IN_FLUID = 1 << 5
NATIVE_NPC_WORLDVIEW_ENTITY_IMPACTED_OR_BOUNCED = 1 << 6
NATIVE_NPC_WORLDVIEW_ENTITY_RESTING_OR_SLIDING = 1 << 7
NATIVE_NPC_WORLDVIEW_ENTITY_DEPLOYABLE = 1 << 8
NATIVE_NPC_WORLDVIEW_ENTITY_ALL_FLAGS = (1 << 9) - 1
NATIVE_NPC_OBSERVATION_SCHEMA = "hytalerl_native_npc_observation_v5"
NATIVE_NPC_OBSERVATION_VERSION = 5
_LEGACY_NPC_OBSERVATION_SCHEMA = "hytalerl_native_npc_observation_v4"
_LEGACY_NPC_OBSERVATION_VERSION = 4
_LEGACY_NPC_OBSERVATION_CONTRACT_SHA256 = (
    "6D1E2BFA924138D075FFEF949DE9DE2C4D2060A142BF82F7DA383DB22DC142C7"
)
NATIVE_NPC_OBSERVATION_ATTACK_CANDIDATE_CAPACITY = 64
NATIVE_NPC_OBSERVATION_PERCEPTIBLE_NPC_CAPACITY = 256
NATIVE_NPC_OBSERVATION_PERCEPTIBLE_ENTITY_CAPACITY = 512
NATIVE_NPC_OBSERVATION_GROUP_LAYOUT = (
    "bit0=actor_core,bit1=target_core,bit2=combat,bit3=resources,"
    "bit4=weapon,bit5=defense,bit6=status,bit7=ability,"
    "bit8=actor_world,bit9=movement,bit10=geometry,bit11=collision,"
    "bit12=perception,bit13=nearby_npcs,bit14=inventory,"
    "bit15=world_clock,bit16=decision_state,bit17=active_interactions,"
    "bit18=interaction_candidates,bit19=traversal,bit20=projectiles,"
    "bit21=dynamic_hazards,bit22=audio,bit23=non_npc_entities,"
    "bit24=role_opacity,bit25=light,bit26=environment_channels"
)
NATIVE_NPC_OBSERVATION_ALL_GROUP_BITS = (1 << 27) - 1
NATIVE_NPC_ACTOR_EVIDENCE_SCHEMA = "hytalerl_native_learner_v3_actor_evidence_inputs_v4"
NATIVE_NPC_ACTOR_EVIDENCE_VERSION = 4
NATIVE_NPC_ACTOR_EVIDENCE_CONTRACT_SHA256 = (
    "1A7CFF4EE96218C4AA48F10E5F65BA0F92B8372C508C66F1EE28D2A811723CD5"
)
_NATIVE_NPC_OBSERVATION_CONTRACT = {
    "schema": NATIVE_NPC_OBSERVATION_SCHEMA,
    "version": NATIVE_NPC_OBSERVATION_VERSION,
    "group_layout": NATIVE_NPC_OBSERVATION_GROUP_LAYOUT,
    "group_status": "complete_partial_unavailable_partition",
    "boundaries": "authoritative_pre_and_post_tick",
    "actor_evidence": {
        "schema": NATIVE_NPC_ACTOR_EVIDENCE_SCHEMA,
        "version": NATIVE_NPC_ACTOR_EVIDENCE_VERSION,
        "sha256": NATIVE_NPC_ACTOR_EVIDENCE_CONTRACT_SHA256,
    },
    "action_capability_evidence": {
        "schema": NATIVE_NPC_ACTOR_EVIDENCE_SCHEMA,
        "version": NATIVE_NPC_ACTOR_EVIDENCE_VERSION,
        "sha256": NATIVE_NPC_ACTOR_EVIDENCE_CONTRACT_SHA256,
        "boundary": "same_as_containing_observation",
        "availability": "controlled_actor_and_target_aligned_and_reset_negotiated",
    },
    "geometry": {
        "schema": "hytale_geometry_v5",
        "version": 5,
        "cell_count": CELL_COUNT,
        "cell_order": "geometry_cell_index_dx_dy_dz",
    },
    "inventory": {
        "schema": "hytalerl_native_inventory_v2",
        "version": 2,
        "sha256": "CC393B3EF1AAD2D54D5DA3FE7CB736C8439D9675C1A1D5D0E85E6BAAA82D0D8F",
    },
    "attack_candidates": {
        "layout": NATIVE_NPC_TRACE_ATTACK_ACTION_LAYOUT,
        "capacity": NATIVE_NPC_OBSERVATION_ATTACK_CANDIDATE_CAPACITY,
        "subjects": ["actor", "target"],
        "boundary": "pre_native_policy_decision",
    },
    "combat_lifecycle": {
        "boundary": "pre_native_policy_decision",
        "fields": [
            "available",
            "attack_executing",
            "attack_pause_seconds",
        ],
        "missing": "availability_false_and_zero_values",
    },
    "perceptible_npcs": {
        "identity": "uuid",
        "capacity": NATIVE_NPC_OBSERVATION_PERCEPTIBLE_NPC_CAPACITY,
    },
    "perceptible_entities": {
        "identity": "worldview_entity_index",
        "capacity": NATIVE_NPC_OBSERVATION_PERCEPTIBLE_ENTITY_CAPACITY,
        "scope": "live_npcs_and_projectiles_with_native_los_within_nearby_radius",
        "attributes": "join_same_boundary_worldview_entities_by_index",
    },
    "role_opacity": {
        "availability": "independent",
        "mask": f"bool[{CELL_COUNT}]",
        "cell_order": "geometry_cell_index_dx_dy_dz",
    },
    "local_perception": {
        "schema": NATIVE_PERCEPTION_CHANNEL_SCHEMA,
        "version": NATIVE_PERCEPTION_CHANNEL_VERSION,
        "positions": "implicit_geometry_cube",
        "validity": "per_sample_per_channel",
    },
}
NATIVE_NPC_OBSERVATION_CONTRACT_SHA256 = (
    hashlib.sha256(
        json.dumps(
            _NATIVE_NPC_OBSERVATION_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    .hexdigest()
    .upper()
)
NATIVE_NPC_OBSERVATION_COMPATIBLE_IDENTITIES = frozenset(
    {
        (
            NATIVE_NPC_OBSERVATION_SCHEMA,
            NATIVE_NPC_OBSERVATION_VERSION,
            NATIVE_NPC_OBSERVATION_CONTRACT_SHA256,
        ),
        (
            _LEGACY_NPC_OBSERVATION_SCHEMA,
            _LEGACY_NPC_OBSERVATION_VERSION,
            _LEGACY_NPC_OBSERVATION_CONTRACT_SHA256,
        ),
    }
)
NATIVE_NPC_TRACE_STATE_WIDTH = 14
NATIVE_NPC_TRACE_CONTROL_WIDTH = 9
NATIVE_NPC_TRACE_DEFAULT_CAPACITY = 4096
NATIVE_NPC_TRACE_MAX_CAPACITY = 65_536
NATIVE_NPC_TRACE_MAX_DRAIN = 32
NATIVE_NPC_TRACE_MAX_ACTIVE = 8
_STATUSES = frozenset(
    {"recording", "stopped", "target_missing", "role_changed", "overflow"}
)
_LOCAL_CELL_OFFSETS = np.asarray(
    [
        (dx, dy, dz)
        for dx in range(-CELL_RADIUS, CELL_RADIUS + 1)
        for dy in range(-CELL_RADIUS, CELL_RADIUS + 1)
        for dz in range(-CELL_RADIUS, CELL_RADIUS + 1)
    ],
    dtype="<i4",
)
_LOCAL_CELL_OFFSETS.flags.writeable = False


@dataclass(frozen=True, slots=True)
class NativeNpcAttackAction:
    """One selected Java ActionAttack with raw lifecycle and charge state."""

    label: str
    active: bool
    triggered: bool
    ready: bool
    aiming_seconds_remaining: float
    charge_seconds: float
    interaction_type: str
    interaction_id: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class NativeNpcAttackExecutionCause:
    """Fail-closed Java association from a chain start to its authored action."""

    available: bool
    executed: bool
    candidate_index: int
    interaction_type: str
    interaction_id: str
    path: str


@dataclass(frozen=True, slots=True)
class NativeNpcInteraction:
    """One exact public Java interaction-chain snapshot."""

    source: str
    type: str
    base_type: str
    chain_id: int
    initial_root_id: str
    root_id: str
    server_state: str
    client_state: str
    final_state: str
    time_seconds: float
    time_shift: float
    operation_counter: int
    simulated_operation_counter: int
    operation_index: int
    client_operation_index: int
    call_depth: int
    simulated_call_depth: int
    predicted: bool
    requires_client: bool
    first_run: bool
    pre_ticked: bool
    desynced: bool
    target_uuid: uuid.UUID | None


@dataclass(frozen=True, slots=True)
class NativeNpcDamageEvent:
    """One Java damage event after server filtering and application."""

    source_type: str
    environment_type: str
    damage_cause_index: int
    damage_cause_id: str
    initial_amount: float
    final_amount: float
    cancelled: bool
    blocked: bool
    actor_source: bool
    actor_target: bool
    source_entity_index: int
    source_uuid: uuid.UUID | None
    target_entity_index: int
    target_uuid: uuid.UUID | None
    projectile_entity_index: int
    projectile_uuid: uuid.UUID | None
    hit_location_available: bool
    hit_location: tuple[float, float, float, float]
    health_available: bool
    target_health_after: float
    target_max_health: float
    lethal: bool


@dataclass(frozen=True, slots=True)
class NativeNpcLifecycleEvent:
    """One generic Java lifecycle fact with explicit origin and completeness."""

    kind: str
    origin: str
    subject: str
    key: str
    index: int
    value_before: float
    value_after: float
    auxiliary_0: float
    auxiliary_1: float
    text_before: str
    text_after: str
    successful: bool
    complete: bool
    entity_index: int
    entity_uuid: uuid.UUID | None
    owner_uuid: uuid.UUID | None
    position_available: bool
    position: tuple[float, float, float]
    flags: int


@dataclass(frozen=True, slots=True)
class NativeNpcMarkedTarget:
    """One named Java role-memory target slot."""

    slot: int
    name: str
    uuid: uuid.UUID


@dataclass(frozen=True, slots=True)
class NativeNpcInternalState:
    """Decision-relevant Java role and motion-controller state."""

    state_name: str
    state_index: int
    substate_index: int
    busy: bool
    transitioning: bool
    role_change_requested: bool
    terminal_action: bool
    backing_away: bool
    steering_motion: str
    motion_controller_present: bool
    motion_in_progress: bool
    obstructed: bool
    current_speed: float
    maximum_speed: float
    avoidance_steering: tuple[float, float, float]
    separation_steering: tuple[float, float, float]
    marked_targets: tuple[NativeNpcMarkedTarget, ...]


@dataclass(frozen=True, slots=True)
class NativeNpcWorldActor:
    """One UUID-stable actor in the bounded omniscient NPC worldview."""

    uuid: uuid.UUID
    role: str
    state: tuple[float, ...]
    state_name: str
    target_uuid: uuid.UUID | None
    combat_attack: bool
    attack_pause_seconds: float


@dataclass(frozen=True, slots=True)
class NativeNpcWorldEntity:
    """One raw transform-bearing ECS entity in the omniscient worldview."""

    entity_index: int
    uuid: uuid.UUID | None
    flags: int
    asset_id: str
    model_asset_id: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    velocity_available: bool
    velocity: tuple[float, float, float]
    bounds_available: bool
    bounds: tuple[float, ...]
    owner_uuid: uuid.UUID | None
    physics_state: str
    lifecycle_timing_available: bool
    lifecycle_start_epoch_second: int
    lifecycle_start_nano: int
    lifecycle_end_epoch_second: int
    lifecycle_end_nano: int

    @property
    def lifecycle_duration_seconds(self) -> float | None:
        """Return the engine-owned scheduled lifetime without epoch subtraction."""

        if not self.lifecycle_timing_available:
            return None
        return (
            float(self.lifecycle_end_epoch_second - self.lifecycle_start_epoch_second)
            + float(self.lifecycle_end_nano - self.lifecycle_start_nano) / 1e9
        )


@dataclass(frozen=True, slots=True)
class NativeNpcWorldSnapshot:
    """All live NPC actors plus native world-clock state at one boundary."""

    world_tick: int
    game_time_epoch_second: int
    game_time_nano: int
    day_progress: float
    sunlight_factor: float
    moon_phase: int
    npc_count: int
    overflow: bool
    actors: tuple[NativeNpcWorldActor, ...]
    entity_count: int
    entity_overflow: bool
    entities: tuple[NativeNpcWorldEntity, ...]
    simulation_time_epoch_second: int
    simulation_time_nano: int


@dataclass(frozen=True, slots=True)
class NativeNpcObservation:
    """Model-neutral Java evidence aligned to one trace state boundary."""

    world_tick: int
    complete_group_bits: int
    partial_group_bits: int
    unavailable_group_bits: int
    action_capability_evidence: Mapping[str, Any]
    geometry: Mapping[str, Any]
    inventory: Mapping[str, Any]
    combat_lifecycle_available: bool
    combat_attack_executing: bool
    attack_pause_seconds: float
    attack_candidates: tuple[NativeNpcAttackAction, ...]
    attack_candidate_count: int
    attack_candidate_overflow: bool
    target_attack_candidates: tuple[NativeNpcAttackAction, ...]
    target_attack_candidate_count: int
    target_attack_candidate_overflow: bool
    perception_available: bool
    target_present: bool
    target_perceptible: bool
    target_distance: float
    perceptible_npc_uuids: tuple[uuid.UUID, ...]
    perceptible_npc_count: int
    perceptible_npc_overflow: bool
    perceptible_entity_indices: tuple[int, ...]
    perceptible_entity_count: int
    perceptible_entity_overflow: bool
    status_failure_bits: int
    role_opaque_cell_mask_available: bool
    role_opaque_cell_mask: npt.NDArray[np.bool_]
    local_perception: NativePerceptionChannelCapture


@dataclass(frozen=True, slots=True)
class NativeNpcTraceCapture:
    """One validated, columnar drain from the Java server."""

    bridge_sha256: str
    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    trace_uuid: uuid.UUID
    npc_uuid: uuid.UUID
    role: str
    environment_parameters: Mapping[str, str]
    actor_evidence_schema: str
    actor_evidence_version: int
    actor_evidence_contract_sha256: str
    action_contract_schema: str
    action_contract_version: int
    action_contract_sha256: str
    event_contract_schema: str
    event_contract_version: int
    event_contract_sha256: str
    lifecycle_contract_schema: str
    lifecycle_contract_version: int
    lifecycle_contract_sha256: str
    observation_schema: str
    observation_version: int
    observation_contract_sha256: str
    status: str
    capacity: int
    total_captured: int
    tick: npt.NDArray[np.int64]
    delta_seconds: npt.NDArray[np.float32]
    state: npt.NDArray[np.float64]
    control: npt.NDArray[np.float64]
    control_mask: npt.NDArray[np.int32]
    next_state: npt.NDArray[np.float64]
    native_control: npt.NDArray[np.bool_]
    attack_action_active: npt.NDArray[np.bool_]
    attack_activation: npt.NDArray[np.bool_]
    attack_execution_cause: tuple[NativeNpcAttackExecutionCause, ...]
    combat_attack: npt.NDArray[np.bool_]
    next_combat_attack: npt.NDArray[np.bool_]
    attack_pause_seconds: npt.NDArray[np.float32]
    next_attack_pause_seconds: npt.NDArray[np.float32]
    target_uuid: tuple[uuid.UUID | None, ...]
    next_target_uuid: tuple[uuid.UUID | None, ...]
    decision_target_uuid: tuple[uuid.UUID | None, ...]
    state_name: tuple[str, ...]
    next_state_name: tuple[str, ...]
    internal_state: tuple[NativeNpcInternalState, ...]
    decision_internal_state: tuple[NativeNpcInternalState, ...]
    next_internal_state: tuple[NativeNpcInternalState, ...]
    body_instruction: tuple[str, ...]
    body_decision_path: tuple[str, ...]
    head_instruction: tuple[str, ...]
    head_decision_path: tuple[str, ...]
    active_actions: tuple[tuple[str, ...], ...]
    active_action_paths: tuple[tuple[str, ...], ...]
    attack_actions: tuple[tuple[NativeNpcAttackAction, ...], ...]
    next_attack_actions: tuple[tuple[NativeNpcAttackAction, ...], ...]
    interactions: tuple[tuple[NativeNpcInteraction, ...], ...]
    next_interactions: tuple[tuple[NativeNpcInteraction, ...], ...]
    damage_events: tuple[tuple[NativeNpcDamageEvent, ...], ...]
    damage_event_count: npt.NDArray[np.int32]
    damage_event_overflow: npt.NDArray[np.bool_]
    lifecycle_events: tuple[tuple[NativeNpcLifecycleEvent, ...], ...]
    lifecycle_event_count: npt.NDArray[np.int32]
    lifecycle_event_overflow: npt.NDArray[np.bool_]
    lifecycle_source_available: npt.NDArray[np.int32]
    lifecycle_source_partial: npt.NDArray[np.int32]
    actor_evidence: tuple[Mapping[str, Any], ...]
    next_actor_evidence: tuple[Mapping[str, Any], ...]
    target_actor_evidence: tuple[Mapping[str, Any], ...]
    next_target_actor_evidence: tuple[Mapping[str, Any], ...]
    observation: tuple[NativeNpcObservation, ...]
    next_observation: tuple[NativeNpcObservation, ...]
    worldview: tuple[NativeNpcWorldSnapshot, ...]
    next_worldview: tuple[NativeNpcWorldSnapshot, ...]

    @property
    def emitted_count(self) -> int:
        return int(self.tick.size)

    @classmethod
    def from_response(cls, response: Mapping[str, Any]) -> NativeNpcTraceCapture:
        source = dict(response)
        if source.get("type") != "npc_transition_trace":
            raise ValueError("native response is not an NPC transition trace")
        if (
            source.get("schema"),
            source.get("version"),
        ) != (NATIVE_NPC_TRACE_SCHEMA, NATIVE_NPC_TRACE_VERSION):
            raise ValueError("unsupported NPC imitation trace schema")
        if source.get("state_layout") != NATIVE_NPC_TRACE_STATE_LAYOUT:
            raise ValueError("unsupported NPC trace state layout")
        if source.get("control_layout") != NATIVE_NPC_TRACE_CONTROL_LAYOUT:
            raise ValueError("unsupported NPC trace control layout")
        if source.get("control_mask_layout") != NATIVE_NPC_TRACE_CONTROL_MASK_LAYOUT:
            raise ValueError("unsupported NPC trace control-mask layout")
        if source.get("attack_action_layout") != NATIVE_NPC_TRACE_ATTACK_ACTION_LAYOUT:
            raise ValueError("unsupported NPC trace attack-action layout")
        if (
            source.get("attack_execution_cause_layout")
            != NATIVE_NPC_ATTACK_EXECUTION_CAUSE_LAYOUT
        ):
            raise ValueError("unsupported NPC attack-execution cause layout")
        if source.get("interaction_layout") != NATIVE_NPC_TRACE_INTERACTION_LAYOUT:
            raise ValueError("unsupported NPC trace interaction layout")
        if source.get("internal_layout") != NATIVE_NPC_TRACE_INTERNAL_LAYOUT:
            raise ValueError("unsupported NPC trace internal-state layout")
        if (
            source.get("event_contract_schema") != NATIVE_NPC_EVENT_CONTRACT_SCHEMA
            or source.get("event_contract_version") != NATIVE_NPC_EVENT_CONTRACT_VERSION
            or source.get("event_contract_sha256") != NATIVE_NPC_EVENT_CONTRACT_SHA256
            or source.get("damage_event_layout") != NATIVE_NPC_DAMAGE_EVENT_LAYOUT
            or source.get("damage_event_capacity") != NATIVE_NPC_DAMAGE_EVENT_CAPACITY
        ):
            raise ValueError("unsupported NPC event contract")
        if (
            source.get("lifecycle_contract_schema")
            != NATIVE_NPC_LIFECYCLE_CONTRACT_SCHEMA
            or source.get("lifecycle_contract_version")
            != NATIVE_NPC_LIFECYCLE_CONTRACT_VERSION
            or source.get("lifecycle_contract_sha256")
            != NATIVE_NPC_LIFECYCLE_CONTRACT_SHA256
            or source.get("lifecycle_event_layout") != NATIVE_NPC_LIFECYCLE_EVENT_LAYOUT
            or source.get("lifecycle_event_capacity")
            != NATIVE_NPC_LIFECYCLE_EVENT_CAPACITY
            or source.get("lifecycle_source_layout")
            != NATIVE_NPC_LIFECYCLE_SOURCE_LAYOUT
        ):
            raise ValueError("unsupported NPC lifecycle contract")
        if (
            source.get("action_contract_schema") != NATIVE_NPC_ACTION_CONTRACT_SCHEMA
            or source.get("action_contract_version")
            != NATIVE_NPC_ACTION_CONTRACT_VERSION
            or source.get("action_contract_sha256") != NATIVE_NPC_ACTION_CONTRACT_SHA256
        ):
            raise ValueError("unsupported NPC action contract")
        if (
            source.get("worldview_schema") != NATIVE_NPC_WORLDVIEW_SCHEMA
            or source.get("worldview_version") != NATIVE_NPC_WORLDVIEW_VERSION
            or source.get("worldview_actor_capacity")
            != NATIVE_NPC_WORLDVIEW_ACTOR_CAPACITY
            or source.get("worldview_actor_layout") != NATIVE_NPC_WORLDVIEW_ACTOR_LAYOUT
            or source.get("worldview_entity_capacity")
            != NATIVE_NPC_WORLDVIEW_ENTITY_CAPACITY
            or source.get("worldview_entity_layout")
            != NATIVE_NPC_WORLDVIEW_ENTITY_LAYOUT
        ):
            raise ValueError("unsupported NPC trace worldview contract")
        observation_identity = (
            source.get("observation_schema"),
            source.get("observation_version"),
            source.get("observation_contract_sha256"),
        )
        current_observation = (
            NATIVE_NPC_OBSERVATION_SCHEMA,
            NATIVE_NPC_OBSERVATION_VERSION,
            NATIVE_NPC_OBSERVATION_CONTRACT_SHA256,
        )
        if (
            observation_identity not in NATIVE_NPC_OBSERVATION_COMPATIBLE_IDENTITIES
            or source.get("observation_group_layout")
            != NATIVE_NPC_OBSERVATION_GROUP_LAYOUT
            or source.get("observation_attack_candidate_capacity")
            != NATIVE_NPC_OBSERVATION_ATTACK_CANDIDATE_CAPACITY
            or source.get("observation_perceptible_npc_capacity")
            != NATIVE_NPC_OBSERVATION_PERCEPTIBLE_NPC_CAPACITY
            or source.get("observation_perceptible_entity_capacity")
            != NATIVE_NPC_OBSERVATION_PERCEPTIBLE_ENTITY_CAPACITY
        ):
            raise ValueError("unsupported NPC semantic-observation contract")
        observation_current = observation_identity == current_observation
        if source.get("angle_units") != "radians":
            raise ValueError("unsupported NPC trace angle units")

        count = _integer(
            source, "emitted_count", minimum=0, maximum=NATIVE_NPC_TRACE_MAX_DRAIN
        )
        capacity = _integer(
            source,
            "capacity",
            minimum=1,
            maximum=NATIVE_NPC_TRACE_MAX_CAPACITY,
        )
        total = _integer(source, "total_captured", minimum=count)
        status = _text(source, "status")
        if status not in _STATUSES:
            raise ValueError(f"unknown NPC trace status: {status}")
        bridge_sha256 = _text(source, "bridge_sha256")
        if re.fullmatch(r"[0-9A-Fa-f]{64}", bridge_sha256) is None:
            raise ValueError("bridge_sha256 must be a SHA-256")
        actor_schema = _text(source, "actor_evidence_schema")
        actor_version = _integer(source, "actor_evidence_version")
        actor_contract = _text(source, "actor_evidence_contract_sha256").upper()
        if (
            actor_schema != NATIVE_NPC_ACTOR_EVIDENCE_SCHEMA
            or actor_version != NATIVE_NPC_ACTOR_EVIDENCE_VERSION
            or actor_contract != NATIVE_NPC_ACTOR_EVIDENCE_CONTRACT_SHA256
        ):
            raise ValueError("unsupported NPC trace actor-evidence contract")

        tick = _array(source, "ticks_i64_le", "<i8", (count,))
        delta = _array(source, "delta_seconds_f32_le", "<f4", (count,))
        state = _array(
            source,
            "state_f64_le_rows",
            "<f8",
            (count, NATIVE_NPC_TRACE_STATE_WIDTH),
        )
        control = _array(
            source,
            "control_f64_le_rows",
            "<f8",
            (count, NATIVE_NPC_TRACE_CONTROL_WIDTH),
        )
        control_mask = _array(
            source,
            "control_mask_i32_le",
            "<i4",
            (count,),
        )
        next_state = _array(
            source,
            "next_state_f64_le_rows",
            "<f8",
            (count, NATIVE_NPC_TRACE_STATE_WIDTH),
        )
        native_control = _bool_array(source, "native_control_u8", count)
        attack_action_active = _bool_array(source, "attack_action_active_u8", count)
        attack_activation = _bool_array(source, "attack_activation_u8", count)
        attack_cause_available = _bool_array(
            source, "attack_execution_cause_available_u8", count
        )
        attack_cause_executed = _bool_array(
            source, "attack_execution_cause_executed_u8", count
        )
        attack_cause_candidate = _array(
            source,
            "attack_execution_cause_candidate_i32_le",
            "<i4",
            (count,),
        )
        attack_cause_types = _strings(
            source, "attack_execution_cause_interaction_types", count
        )
        attack_cause_ids = _strings(
            source, "attack_execution_cause_interaction_ids", count
        )
        attack_cause_paths = _strings(source, "attack_execution_cause_paths", count)
        attack_execution_cause = tuple(
            NativeNpcAttackExecutionCause(
                bool(available),
                bool(executed),
                int(candidate),
                interaction_type,
                interaction_id,
                path,
            )
            for available, executed, candidate, interaction_type, interaction_id, path in zip(
                attack_cause_available,
                attack_cause_executed,
                attack_cause_candidate,
                attack_cause_types,
                attack_cause_ids,
                attack_cause_paths,
                strict=True,
            )
        )
        combat_attack = _bool_array(source, "combat_attack_u8", count)
        next_combat_attack = _bool_array(source, "next_combat_attack_u8", count)
        attack_pause = _array(source, "attack_pause_seconds_f32_le", "<f4", (count,))
        next_attack_pause = _array(
            source, "next_attack_pause_seconds_f32_le", "<f4", (count,)
        )
        damage_event_count = _array(
            source, "damage_event_counts_i32_le", "<i4", (count,)
        )
        damage_event_overflow = _bool_array(source, "damage_event_overflow_u8", count)
        damage_events = _damage_event_rows(source, "damage_events", count)
        lifecycle_event_count = _array(
            source, "lifecycle_event_counts_i32_le", "<i4", (count,)
        )
        lifecycle_event_overflow = _bool_array(
            source, "lifecycle_event_overflow_u8", count
        )
        lifecycle_source_available = _array(
            source, "lifecycle_source_available_i32_le", "<i4", (count,)
        )
        lifecycle_source_partial = _array(
            source, "lifecycle_source_partial_i32_le", "<i4", (count,)
        )
        lifecycle_events = _lifecycle_event_rows(source, "lifecycle_events", count)

        if count > 1 and np.any(tick[1:] <= tick[:-1]):
            raise ValueError("NPC trace ticks must be strictly increasing")
        if not np.all(np.isfinite(delta)) or np.any(delta < 0.0):
            raise ValueError("NPC trace deltas must be finite and nonnegative")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(control)):
            raise ValueError("NPC trace rows must be finite")
        if not np.all(np.isfinite(next_state)):
            raise ValueError("NPC trace next-state rows must be finite")
        if np.any((control_mask < 0) | (control_mask > 0x7F)):
            raise ValueError("NPC trace control mask has unknown bits")
        if (
            not np.all(np.isfinite(attack_pause))
            or not np.all(np.isfinite(next_attack_pause))
            or np.any(attack_pause < 0.0)
            or np.any(next_attack_pause < 0.0)
        ):
            raise ValueError("NPC trace attack pauses must be finite and nonnegative")
        if np.any(attack_activation & ~attack_action_active):
            raise ValueError("NPC attack activation must be an active-action edge")
        for cause in attack_execution_cause:
            identity = (
                cause.candidate_index >= 0
                and bool(cause.interaction_type)
                and bool(cause.interaction_id)
                and bool(cause.path)
            )
            if (cause.executed and (not cause.available or not identity)) or (
                not cause.executed
                and (
                    cause.candidate_index != -1
                    or cause.interaction_type
                    or cause.interaction_id
                    or cause.path
                )
            ):
                raise ValueError("NPC attack execution cause is not fail-closed")
        if np.any(damage_event_count < 0):
            raise ValueError("NPC damage event counts must be nonnegative")
        for emitted, total_count, overflow in zip(
            damage_events,
            damage_event_count,
            damage_event_overflow,
            strict=True,
        ):
            if (
                len(emitted) > NATIVE_NPC_DAMAGE_EVENT_CAPACITY
                or int(total_count) < len(emitted)
                or bool(overflow) != (int(total_count) > len(emitted))
            ):
                raise ValueError("NPC damage event count/overflow is inconsistent")
        if (
            np.any(lifecycle_event_count < 0)
            or np.any(
                (lifecycle_source_available < 0)
                | (lifecycle_source_available > NATIVE_NPC_LIFECYCLE_ALL_SOURCE_BITS)
            )
            or np.any(lifecycle_source_partial & ~lifecycle_source_available)
        ):
            raise ValueError("NPC lifecycle event source masks are invalid")
        for emitted, total_count, overflow in zip(
            lifecycle_events,
            lifecycle_event_count,
            lifecycle_event_overflow,
            strict=True,
        ):
            if (
                len(emitted) > NATIVE_NPC_LIFECYCLE_EVENT_CAPACITY
                or int(total_count) < len(emitted)
                or bool(overflow) != (int(total_count) > len(emitted))
            ):
                raise ValueError("NPC lifecycle event count/overflow is inconsistent")

        targets = _uuid_rows(source, "target", count)
        next_targets = _uuid_rows(source, "next_target", count)
        decision_targets = _uuid_rows(source, "decision_target", count)

        strings = {
            name: _strings(source, key, count)
            for name, key in (
                ("state_name", "state_names"),
                ("next_state_name", "next_state_names"),
                ("body_instruction", "body_instructions"),
                ("body_decision_path", "body_decision_paths"),
                ("head_instruction", "head_instructions"),
                ("head_decision_path", "head_decision_paths"),
            )
        }
        actions = _string_rows(source, "active_actions", count)
        action_paths = _string_rows(source, "active_action_paths", count)

        capture = cls(
            bridge_sha256=bridge_sha256.upper(),
            server_version=_text(source, "server_version"),
            world=_text(source, "world"),
            worldgen_provider=_text(source, "worldgen_provider"),
            worldgen_version=_text(source, "worldgen_version"),
            seed=_integer(source, "seed"),
            trace_uuid=_uuid(source, "trace_uuid_bytes"),
            npc_uuid=_uuid(source, "npc_uuid_bytes"),
            role=_text(source, "role", nonempty=True),
            environment_parameters=_string_map(source, "environment_parameters"),
            actor_evidence_schema=actor_schema,
            actor_evidence_version=actor_version,
            actor_evidence_contract_sha256=actor_contract,
            action_contract_schema=NATIVE_NPC_ACTION_CONTRACT_SCHEMA,
            action_contract_version=NATIVE_NPC_ACTION_CONTRACT_VERSION,
            action_contract_sha256=NATIVE_NPC_ACTION_CONTRACT_SHA256,
            event_contract_schema=NATIVE_NPC_EVENT_CONTRACT_SCHEMA,
            event_contract_version=NATIVE_NPC_EVENT_CONTRACT_VERSION,
            event_contract_sha256=NATIVE_NPC_EVENT_CONTRACT_SHA256,
            lifecycle_contract_schema=NATIVE_NPC_LIFECYCLE_CONTRACT_SCHEMA,
            lifecycle_contract_version=NATIVE_NPC_LIFECYCLE_CONTRACT_VERSION,
            lifecycle_contract_sha256=NATIVE_NPC_LIFECYCLE_CONTRACT_SHA256,
            observation_schema=observation_identity[0],
            observation_version=observation_identity[1],
            observation_contract_sha256=observation_identity[2],
            status=status,
            capacity=capacity,
            total_captured=total,
            tick=tick,
            delta_seconds=delta,
            state=state,
            control=control,
            control_mask=control_mask,
            next_state=next_state,
            native_control=native_control,
            attack_action_active=attack_action_active,
            attack_activation=attack_activation,
            attack_execution_cause=attack_execution_cause,
            combat_attack=combat_attack,
            next_combat_attack=next_combat_attack,
            attack_pause_seconds=attack_pause,
            next_attack_pause_seconds=next_attack_pause,
            target_uuid=targets,
            next_target_uuid=next_targets,
            decision_target_uuid=decision_targets,
            state_name=strings["state_name"],
            next_state_name=strings["next_state_name"],
            internal_state=_internal_rows(source, "internal_states", count),
            decision_internal_state=_internal_rows(
                source, "decision_internal_states", count
            ),
            next_internal_state=_internal_rows(source, "next_internal_states", count),
            body_instruction=strings["body_instruction"],
            body_decision_path=strings["body_decision_path"],
            head_instruction=strings["head_instruction"],
            head_decision_path=strings["head_decision_path"],
            active_actions=actions,
            active_action_paths=action_paths,
            attack_actions=_attack_action_rows(source, "attack_actions", count),
            next_attack_actions=_attack_action_rows(
                source, "next_attack_actions", count
            ),
            interactions=_interaction_rows(source, "interactions", count),
            next_interactions=_interaction_rows(source, "next_interactions", count),
            damage_events=damage_events,
            damage_event_count=damage_event_count,
            damage_event_overflow=damage_event_overflow,
            lifecycle_events=lifecycle_events,
            lifecycle_event_count=lifecycle_event_count,
            lifecycle_event_overflow=lifecycle_event_overflow,
            lifecycle_source_available=lifecycle_source_available,
            lifecycle_source_partial=lifecycle_source_partial,
            actor_evidence=_actor_rows(source, "actor_evidence", count, 0, True),
            next_actor_evidence=_actor_rows(
                source, "next_actor_evidence", count, 0, True
            ),
            target_actor_evidence=_actor_rows(
                source, "target_actor_evidence", count, 1, False
            ),
            next_target_actor_evidence=_actor_rows(
                source, "next_target_actor_evidence", count, 1, False
            ),
            observation=_observation_rows(
                source,
                "observations",
                count,
                provenance=source,
                combat_lifecycle_required=observation_current,
            ),
            next_observation=_observation_rows(
                source,
                "next_observations",
                count,
                provenance=source,
                combat_lifecycle_required=observation_current,
            ),
            worldview=_worldview_rows(source, "worldviews", count),
            next_worldview=_worldview_rows(source, "next_worldviews", count),
        )
        for value in (
            tick,
            delta,
            state,
            control,
            control_mask,
            next_state,
            native_control,
            attack_action_active,
            attack_activation,
            combat_attack,
            next_combat_attack,
            attack_pause,
            next_attack_pause,
            damage_event_count,
            damage_event_overflow,
            lifecycle_event_count,
            lifecycle_event_overflow,
            lifecycle_source_available,
            lifecycle_source_partial,
        ):
            value.flags.writeable = False
        if any(row["role_id"] != capture.role for row in capture.actor_evidence):
            raise ValueError("NPC trace actor evidence disagrees with the pinned role")
        if any(row["role_id"] != capture.role for row in capture.next_actor_evidence):
            raise ValueError("NPC next actor evidence disagrees with the pinned role")
        _validate_observation_alignment(capture)
        _validate_attack_execution_causes(capture)
        if any(
            active != any(action.active for action in row)
            for active, row in zip(
                capture.attack_action_active, capture.attack_actions, strict=True
            )
        ):
            raise ValueError("NPC attack-action state disagrees with its snapshots")
        if np.any((capture.attack_pause_seconds > 0.0) & ~capture.combat_attack):
            raise ValueError("NPC attack pause requires combat-attack state")
        if np.any(
            (capture.next_attack_pause_seconds > 0.0) & ~capture.next_combat_attack
        ):
            raise ValueError("NPC next attack pause requires combat-attack state")
        if any(
            name != internal.state_name
            for name, internal in zip(
                capture.state_name, capture.internal_state, strict=True
            )
        ):
            raise ValueError(
                "NPC pre-decision state name disagrees with internal state"
            )
        if any(
            name != internal.state_name
            for name, internal in zip(
                capture.next_state_name, capture.next_internal_state, strict=True
            )
        ):
            raise ValueError("NPC post-tick state name disagrees with internal state")
        for worldview in capture.worldview + capture.next_worldview:
            if not worldview.overflow and capture.npc_uuid not in {
                actor.uuid for actor in worldview.actors
            }:
                raise ValueError("complete NPC worldview omits the traced actor")
        for row in capture.damage_events:
            for event in row:
                if event.actor_source and event.source_uuid != capture.npc_uuid:
                    raise ValueError("NPC damage source perspective is inconsistent")
                if event.actor_target and event.target_uuid != capture.npc_uuid:
                    raise ValueError("NPC damage target perspective is inconsistent")
        actor_kinds = {
            "stat_changed",
            "status_added",
            "status_removed",
            "status_refreshed",
            "inventory_transaction",
            "inventory_slot",
            "actor_death",
            "actor_revived",
        }
        for row, available in zip(
            capture.lifecycle_events,
            capture.lifecycle_source_available,
            strict=True,
        ):
            for event in row:
                if not (int(available) & _LIFECYCLE_KIND_SOURCE[event.kind]):
                    raise ValueError("NPC lifecycle event source is unavailable")
                if event.kind in actor_kinds and event.entity_uuid != capture.npc_uuid:
                    raise ValueError("NPC lifecycle actor perspective is inconsistent")
        return capture


def native_npc_trace_start_request(
    npc_uuid: uuid.UUID | str | bytes,
    *,
    expected_role: str | None = None,
    capacity: int = NATIVE_NPC_TRACE_DEFAULT_CAPACITY,
) -> dict[str, object]:
    """Start one exact-UUID trace; role pinning is optional but recommended."""

    capacity = _bounded(capacity, "capacity", 1, NATIVE_NPC_TRACE_MAX_CAPACITY)
    request: dict[str, object] = {
        "type": "npc_transition_trace",
        "command": "start",
        "npc_uuid_bytes": _as_uuid(npc_uuid).bytes,
        "capacity": capacity,
    }
    if expected_role is not None:
        if not isinstance(expected_role, str) or not expected_role.strip():
            raise ValueError("expected_role must be a non-empty string")
        request["expected_role"] = expected_role.strip()
    return request


def native_npc_trace_poll_request(
    trace_uuid: uuid.UUID | str | bytes,
    *,
    max_frames: int = NATIVE_NPC_TRACE_MAX_DRAIN,
) -> dict[str, object]:
    return _drain_request(trace_uuid, "poll", max_frames)


def native_npc_trace_stop_request(
    trace_uuid: uuid.UUID | str | bytes,
    *,
    max_frames: int = NATIVE_NPC_TRACE_MAX_DRAIN,
) -> dict[str, object]:
    return _drain_request(trace_uuid, "stop", max_frames)


def concatenate_native_npc_traces(
    captures: Sequence[NativeNpcTraceCapture],
) -> NativeNpcTraceCapture:
    """Concatenate ordered drains while preserving one trace identity."""

    rows = tuple(captures)
    if not rows:
        raise ValueError("at least one NPC trace capture is required")
    first = rows[0]
    identity = _identity(first)
    if any(_identity(row) != identity for row in rows[1:]):
        raise ValueError("NPC trace drains have different identities")
    ticks = np.concatenate([row.tick for row in rows])
    if ticks.size > 1 and np.any(ticks[1:] <= ticks[:-1]):
        raise ValueError("NPC trace drains overlap or are out of order")
    emitted = sum(row.emitted_count for row in rows)
    total = rows[-1].total_captured
    if emitted > total:
        raise ValueError("NPC trace drains exceed the server capture count")
    arrays = {
        "tick": ticks,
        "delta_seconds": np.concatenate([row.delta_seconds for row in rows]),
        "state": np.concatenate([row.state for row in rows]),
        "control": np.concatenate([row.control for row in rows]),
        "control_mask": np.concatenate([row.control_mask for row in rows]),
        "next_state": np.concatenate([row.next_state for row in rows]),
        "native_control": np.concatenate([row.native_control for row in rows]),
        "attack_action_active": np.concatenate(
            [row.attack_action_active for row in rows]
        ),
        "attack_activation": np.concatenate([row.attack_activation for row in rows]),
        "combat_attack": np.concatenate([row.combat_attack for row in rows]),
        "next_combat_attack": np.concatenate([row.next_combat_attack for row in rows]),
        "attack_pause_seconds": np.concatenate(
            [row.attack_pause_seconds for row in rows]
        ),
        "next_attack_pause_seconds": np.concatenate(
            [row.next_attack_pause_seconds for row in rows]
        ),
        "damage_event_count": np.concatenate([row.damage_event_count for row in rows]),
        "damage_event_overflow": np.concatenate(
            [row.damage_event_overflow for row in rows]
        ),
        "lifecycle_event_count": np.concatenate(
            [row.lifecycle_event_count for row in rows]
        ),
        "lifecycle_event_overflow": np.concatenate(
            [row.lifecycle_event_overflow for row in rows]
        ),
        "lifecycle_source_available": np.concatenate(
            [row.lifecycle_source_available for row in rows]
        ),
        "lifecycle_source_partial": np.concatenate(
            [row.lifecycle_source_partial for row in rows]
        ),
    }
    for value in arrays.values():
        value.flags.writeable = False
    return NativeNpcTraceCapture(
        bridge_sha256=first.bridge_sha256,
        server_version=first.server_version,
        world=first.world,
        worldgen_provider=first.worldgen_provider,
        worldgen_version=first.worldgen_version,
        seed=first.seed,
        trace_uuid=first.trace_uuid,
        npc_uuid=first.npc_uuid,
        role=first.role,
        environment_parameters=first.environment_parameters,
        actor_evidence_schema=first.actor_evidence_schema,
        actor_evidence_version=first.actor_evidence_version,
        actor_evidence_contract_sha256=first.actor_evidence_contract_sha256,
        action_contract_schema=first.action_contract_schema,
        action_contract_version=first.action_contract_version,
        action_contract_sha256=first.action_contract_sha256,
        event_contract_schema=first.event_contract_schema,
        event_contract_version=first.event_contract_version,
        event_contract_sha256=first.event_contract_sha256,
        lifecycle_contract_schema=first.lifecycle_contract_schema,
        lifecycle_contract_version=first.lifecycle_contract_version,
        lifecycle_contract_sha256=first.lifecycle_contract_sha256,
        observation_schema=first.observation_schema,
        observation_version=first.observation_version,
        observation_contract_sha256=first.observation_contract_sha256,
        status=rows[-1].status,
        capacity=first.capacity,
        total_captured=total,
        tick=arrays["tick"],
        delta_seconds=arrays["delta_seconds"],
        state=arrays["state"],
        control=arrays["control"],
        control_mask=arrays["control_mask"],
        next_state=arrays["next_state"],
        native_control=arrays["native_control"],
        attack_action_active=arrays["attack_action_active"],
        attack_activation=arrays["attack_activation"],
        attack_execution_cause=sum((row.attack_execution_cause for row in rows), ()),
        combat_attack=arrays["combat_attack"],
        next_combat_attack=arrays["next_combat_attack"],
        attack_pause_seconds=arrays["attack_pause_seconds"],
        next_attack_pause_seconds=arrays["next_attack_pause_seconds"],
        target_uuid=sum((row.target_uuid for row in rows), ()),
        next_target_uuid=sum((row.next_target_uuid for row in rows), ()),
        decision_target_uuid=sum((row.decision_target_uuid for row in rows), ()),
        state_name=sum((row.state_name for row in rows), ()),
        next_state_name=sum((row.next_state_name for row in rows), ()),
        internal_state=sum((row.internal_state for row in rows), ()),
        decision_internal_state=sum((row.decision_internal_state for row in rows), ()),
        next_internal_state=sum((row.next_internal_state for row in rows), ()),
        body_instruction=sum((row.body_instruction for row in rows), ()),
        body_decision_path=sum((row.body_decision_path for row in rows), ()),
        head_instruction=sum((row.head_instruction for row in rows), ()),
        head_decision_path=sum((row.head_decision_path for row in rows), ()),
        active_actions=sum((row.active_actions for row in rows), ()),
        active_action_paths=sum((row.active_action_paths for row in rows), ()),
        attack_actions=sum((row.attack_actions for row in rows), ()),
        next_attack_actions=sum((row.next_attack_actions for row in rows), ()),
        interactions=sum((row.interactions for row in rows), ()),
        next_interactions=sum((row.next_interactions for row in rows), ()),
        damage_events=sum((row.damage_events for row in rows), ()),
        damage_event_count=arrays["damage_event_count"],
        damage_event_overflow=arrays["damage_event_overflow"],
        lifecycle_events=sum((row.lifecycle_events for row in rows), ()),
        lifecycle_event_count=arrays["lifecycle_event_count"],
        lifecycle_event_overflow=arrays["lifecycle_event_overflow"],
        lifecycle_source_available=arrays["lifecycle_source_available"],
        lifecycle_source_partial=arrays["lifecycle_source_partial"],
        actor_evidence=sum((row.actor_evidence for row in rows), ()),
        next_actor_evidence=sum((row.next_actor_evidence for row in rows), ()),
        target_actor_evidence=sum((row.target_actor_evidence for row in rows), ()),
        next_target_actor_evidence=sum(
            (row.next_target_actor_evidence for row in rows), ()
        ),
        observation=sum((row.observation for row in rows), ()),
        next_observation=sum((row.next_observation for row in rows), ()),
        worldview=sum((row.worldview for row in rows), ()),
        next_worldview=sum((row.next_worldview for row in rows), ()),
    )


def _drain_request(value: uuid.UUID | str | bytes, command: str, maximum: int):
    return {
        "type": "npc_transition_trace",
        "command": command,
        "trace_uuid_bytes": _as_uuid(value).bytes,
        "max_frames": _bounded(maximum, "max_frames", 1, NATIVE_NPC_TRACE_MAX_DRAIN),
    }


def _as_uuid(value: uuid.UUID | str | bytes) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        return uuid.UUID(value)
    if isinstance(value, bytes) and len(value) == 16:
        return uuid.UUID(bytes=value)
    raise TypeError("UUID must be uuid.UUID, canonical string, or 16 bytes")


def _identity(capture: NativeNpcTraceCapture) -> tuple[object, ...]:
    return (
        capture.bridge_sha256,
        capture.server_version,
        capture.world,
        capture.worldgen_provider,
        capture.worldgen_version,
        capture.seed,
        capture.trace_uuid,
        capture.npc_uuid,
        capture.role,
        tuple(sorted(capture.environment_parameters.items())),
        capture.actor_evidence_schema,
        capture.actor_evidence_version,
        capture.actor_evidence_contract_sha256,
        capture.action_contract_schema,
        capture.action_contract_version,
        capture.action_contract_sha256,
        capture.event_contract_schema,
        capture.event_contract_version,
        capture.event_contract_sha256,
        capture.lifecycle_contract_schema,
        capture.lifecycle_contract_version,
        capture.lifecycle_contract_sha256,
        capture.observation_schema,
        capture.observation_version,
        capture.observation_contract_sha256,
        capture.capacity,
    )


def _array(source, key: str, dtype: str, shape: tuple[int, ...]):
    raw = _bytes(source, key, int(np.prod(shape)) * np.dtype(dtype).itemsize)
    return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def _bool_array(source, key: str, count: int):
    value = _array(source, key, "u1", (count,))
    if np.any(value > 1):
        raise ValueError(f"{key} must contain only 0 or 1")
    return value.astype(np.bool_)


def _bytes(source, key: str, size: int) -> bytes:
    value = source.get(key)
    if not isinstance(value, bytes) or len(value) != size:
        raise TypeError(f"{key} must contain exactly {size} bytes")
    return value


def _uuid(source, key: str) -> uuid.UUID:
    return uuid.UUID(bytes=_bytes(source, key, 16))


def _uuid_rows(
    source: Mapping[str, Any], prefix: str, count: int
) -> tuple[uuid.UUID | None, ...]:
    present = _bool_array(source, f"{prefix}_present_u8", count)
    payload = _bytes(source, f"{prefix}_uuid_bytes", count * 16)
    result: list[uuid.UUID | None] = []
    for index, exists in enumerate(present):
        raw = payload[index * 16 : (index + 1) * 16]
        if exists:
            result.append(uuid.UUID(bytes=raw))
        elif any(raw):
            raise ValueError(f"absent NPC trace {prefix} must use zero UUID bytes")
        else:
            result.append(None)
    return tuple(result)


_OBSERVATION_FIELDS = frozenset(
    {
        "world_tick",
        "complete_group_bits",
        "partial_group_bits",
        "unavailable_group_bits",
        "action_capability_evidence",
        "geometry",
        "inventory",
        "role_opaque_cell_mask_available",
        "role_opaque_cell_mask_u8",
        "local_perception",
        "combat_lifecycle_available",
        "combat_attack_executing",
        "attack_pause_seconds",
        "attack_candidate_count",
        "attack_candidate_overflow",
        "attack_candidates",
        "target_attack_candidate_count",
        "target_attack_candidate_overflow",
        "target_attack_candidates",
        "perception_available",
        "target_present",
        "target_perceptible",
        "target_distance",
        "perceptible_npc_count",
        "perceptible_npc_overflow",
        "perceptible_npc_uuids",
        "perceptible_entity_count",
        "perceptible_entity_overflow",
        "perceptible_entity_indices",
        "status_failure_bits",
    }
)


def _observation_rows(
    source: Mapping[str, Any],
    key: str,
    count: int,
    *,
    provenance: Mapping[str, Any],
    combat_lifecycle_required: bool,
) -> tuple[NativeNpcObservation, ...]:
    value = source.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise TypeError(f"{key} must match emitted_count")
    return tuple(
        _observation(
            row,
            f"{key}[{index}]",
            provenance,
            combat_lifecycle_required=combat_lifecycle_required,
        )
        for index, row in enumerate(value)
    )


def _observation(
    value: Any,
    key: str,
    provenance: Mapping[str, Any],
    *,
    combat_lifecycle_required: bool,
) -> NativeNpcObservation:
    from hytalegym.jax.combat.inventory.adapters.native import (
        parse_native_inventory_frame,
    )

    fields = _OBSERVATION_FIELDS if combat_lifecycle_required else (
        _OBSERVATION_FIELDS
        - {
            "combat_lifecycle_available",
            "combat_attack_executing",
            "attack_pause_seconds",
        }
    )
    row = _mapping(value, key, fields)
    world_tick = _wire_int(row["world_tick"], f"{key}.world_tick")
    if world_tick < 0:
        raise ValueError(f"{key}.world_tick must be nonnegative")
    group_bits = tuple(
        _wire_int(row[name], f"{key}.{name}")
        for name in (
            "complete_group_bits",
            "partial_group_bits",
            "unavailable_group_bits",
        )
    )
    if (
        any(
            bits < 0 or bits > NATIVE_NPC_OBSERVATION_ALL_GROUP_BITS
            for bits in group_bits
        )
        or (group_bits[0] & group_bits[1])
        or (group_bits[0] & group_bits[2])
        or (group_bits[1] & group_bits[2])
        or (group_bits[0] | group_bits[1] | group_bits[2])
        != NATIVE_NPC_OBSERVATION_ALL_GROUP_BITS
    ):
        raise ValueError(f"{key} feature-family masks must partition the contract")

    from hytalegym.jax.combat.observation.v3.native.codec.decode import (
        parse_native_actor_evidence_frame,
    )

    capability_wire = row["action_capability_evidence"]
    if not isinstance(capability_wire, Mapping):
        raise TypeError(f"{key}.action_capability_evidence must be an object")
    # Unavailable capability rows carry no learner values.  Older v4 captures
    # remain safe to decode after the live actor-evidence contract advances;
    # available rows still require the exact current parser and hash.
    legacy_unavailable = (
        capability_wire.get("evidence_contract_sha256")
        == NATIVE_NPC_ACTOR_EVIDENCE_CONTRACT_SHA256
        and capability_wire.get("available") is False
    )
    capability = (
        {"available": False, "world_tick": world_tick}
        if legacy_unavailable
        else parse_native_actor_evidence_frame(capability_wire)
    )
    if capability["available"] and capability["world_tick"] != world_tick:
        raise ValueError(f"{key} action capability is not boundary-aligned")

    geometry = row["geometry"]
    inventory = row["inventory"]
    if not isinstance(geometry, Mapping) or not isinstance(inventory, Mapping):
        raise TypeError(f"{key} geometry and inventory must be objects")
    materialized_geometry = parse_geometry(dict(geometry))
    parsed_inventory = parse_native_inventory_frame(inventory)
    role_opacity_available = _wire_bool(
        row["role_opaque_cell_mask_available"],
        f"{key}.role_opaque_cell_mask_available",
    )
    role_opacity_u8 = np.frombuffer(
        _bytes(row, "role_opaque_cell_mask_u8", CELL_COUNT),
        dtype=np.uint8,
    ).copy()
    if np.any(role_opacity_u8 > 1):
        raise ValueError(f"{key}.role_opaque_cell_mask_u8 must contain 0 or 1")
    role_opacity = role_opacity_u8.astype(np.bool_)
    if not role_opacity_available and np.any(role_opacity):
        raise ValueError(f"{key} unavailable role opacity must be zero-masked")
    role_opacity.flags.writeable = False
    local_perception = _local_perception(
        row["local_perception"],
        materialized_geometry,
        provenance,
        f"{key}.local_perception",
    )

    combat_lifecycle_available = (
        _wire_bool(
            row["combat_lifecycle_available"],
            f"{key}.combat_lifecycle_available",
        )
        if combat_lifecycle_required
        else False
    )
    combat_attack_executing = (
        _wire_bool(
            row["combat_attack_executing"],
            f"{key}.combat_attack_executing",
        )
        if combat_lifecycle_required
        else False
    )
    attack_pause_seconds = (
        _wire_number(row["attack_pause_seconds"], f"{key}.attack_pause_seconds")
        if combat_lifecycle_required
        else 0.0
    )
    if attack_pause_seconds < 0.0 or (
        not combat_lifecycle_available
        and (combat_attack_executing or attack_pause_seconds != 0.0)
    ):
        raise ValueError(f"{key} combat lifecycle is invalid")

    candidates = _attack_action_row(row["attack_candidates"], key)
    candidate_count = _wire_int(
        row["attack_candidate_count"], f"{key}.attack_candidate_count"
    )
    candidate_overflow = _wire_bool(
        row["attack_candidate_overflow"], f"{key}.attack_candidate_overflow"
    )
    _validate_bounded_rows(
        candidates,
        candidate_count,
        candidate_overflow,
        NATIVE_NPC_OBSERVATION_ATTACK_CANDIDATE_CAPACITY,
        f"{key}.attack_candidates",
    )
    target_candidates = _attack_action_row(row["target_attack_candidates"], key)
    target_candidate_count = _wire_int(
        row["target_attack_candidate_count"],
        f"{key}.target_attack_candidate_count",
    )
    target_candidate_overflow = _wire_bool(
        row["target_attack_candidate_overflow"],
        f"{key}.target_attack_candidate_overflow",
    )
    _validate_bounded_rows(
        target_candidates,
        target_candidate_count,
        target_candidate_overflow,
        NATIVE_NPC_OBSERVATION_ATTACK_CANDIDATE_CAPACITY,
        f"{key}.target_attack_candidates",
    )

    perception_available = _wire_bool(
        row["perception_available"], f"{key}.perception_available"
    )
    target_present = _wire_bool(row["target_present"], f"{key}.target_present")
    target_perceptible = _wire_bool(
        row["target_perceptible"], f"{key}.target_perceptible"
    )
    target_distance = _wire_number(row["target_distance"], f"{key}.target_distance")
    if target_distance < 0.0 or (
        not target_present
        and (target_perceptible or target_distance != 0.0 or target_candidate_count)
    ):
        raise ValueError(f"{key} target observation is inconsistent")
    if target_perceptible and not perception_available:
        raise ValueError(f"{key} unavailable perception sees the target")

    visible_raw = row["perceptible_npc_uuids"]
    if not isinstance(visible_raw, list) or any(
        not isinstance(item, bytes) or len(item) != 16 for item in visible_raw
    ):
        raise TypeError(f"{key}.perceptible_npc_uuids must contain UUID bytes")
    visible = tuple(uuid.UUID(bytes=item) for item in visible_raw)
    if tuple(item.bytes for item in visible) != tuple(
        sorted(item.bytes for item in visible)
    ) or len(set(visible)) != len(visible):
        raise ValueError(f"{key}.perceptible_npc_uuids must be uniquely sorted")
    visible_count = _wire_int(
        row["perceptible_npc_count"], f"{key}.perceptible_npc_count"
    )
    visible_overflow = _wire_bool(
        row["perceptible_npc_overflow"], f"{key}.perceptible_npc_overflow"
    )
    _validate_bounded_rows(
        visible,
        visible_count,
        visible_overflow,
        NATIVE_NPC_OBSERVATION_PERCEPTIBLE_NPC_CAPACITY,
        f"{key}.perceptible_npc_uuids",
    )
    if not perception_available and (visible or visible_count or visible_overflow):
        raise ValueError(f"{key} unavailable perception carries visible NPCs")
    entity_indices_raw = row["perceptible_entity_indices"]
    if not isinstance(entity_indices_raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in entity_indices_raw
    ):
        raise TypeError(f"{key}.perceptible_entity_indices must contain indexes")
    entity_indices = tuple(entity_indices_raw)
    if entity_indices != tuple(sorted(set(entity_indices))):
        raise ValueError(f"{key}.perceptible_entity_indices must be uniquely sorted")
    entity_count = _wire_int(
        row["perceptible_entity_count"], f"{key}.perceptible_entity_count"
    )
    entity_overflow = _wire_bool(
        row["perceptible_entity_overflow"], f"{key}.perceptible_entity_overflow"
    )
    _validate_bounded_rows(
        entity_indices,
        entity_count,
        entity_overflow,
        NATIVE_NPC_OBSERVATION_PERCEPTIBLE_ENTITY_CAPACITY,
        f"{key}.perceptible_entity_indices",
    )
    if not perception_available and (entity_indices or entity_count or entity_overflow):
        raise ValueError(f"{key} unavailable perception carries visible entities")
    status_failure_bits = _wire_int(
        row["status_failure_bits"], f"{key}.status_failure_bits"
    )
    if not 0 <= status_failure_bits <= 0xF:
        raise ValueError(f"{key}.status_failure_bits has unknown bits")

    inventory_copy = MappingProxyType(dict(parsed_inventory))
    return NativeNpcObservation(
        world_tick=world_tick,
        complete_group_bits=group_bits[0],
        partial_group_bits=group_bits[1],
        unavailable_group_bits=group_bits[2],
        action_capability_evidence=MappingProxyType(dict(capability_wire)),
        geometry=MappingProxyType(dict(geometry)),
        inventory=inventory_copy,
        combat_lifecycle_available=combat_lifecycle_available,
        combat_attack_executing=combat_attack_executing,
        attack_pause_seconds=attack_pause_seconds,
        attack_candidates=candidates,
        attack_candidate_count=candidate_count,
        attack_candidate_overflow=candidate_overflow,
        target_attack_candidates=target_candidates,
        target_attack_candidate_count=target_candidate_count,
        target_attack_candidate_overflow=target_candidate_overflow,
        perception_available=perception_available,
        target_present=target_present,
        target_perceptible=target_perceptible,
        target_distance=target_distance,
        perceptible_npc_uuids=visible,
        perceptible_npc_count=visible_count,
        perceptible_npc_overflow=visible_overflow,
        perceptible_entity_indices=entity_indices,
        perceptible_entity_count=entity_count,
        perceptible_entity_overflow=entity_overflow,
        status_failure_bits=status_failure_bits,
        role_opaque_cell_mask_available=role_opacity_available,
        role_opaque_cell_mask=role_opacity,
        local_perception=local_perception,
    )


_LOCAL_PERCEPTION_FIELDS = frozenset(
    {
        "schema",
        "version",
        "cell_order",
        "sample_count",
        "available_u8",
        "channel_validity_u8_bits",
        "heightmap_i16_le",
        "sky_light_u8",
        "block_light_rgb_u8",
        "environment_i32_le",
        "tint_argb_i32_le",
    }
)


def _local_perception(
    value: Any,
    geometry: Mapping[str, Any],
    provenance: Mapping[str, Any],
    key: str,
) -> NativePerceptionChannelCapture:
    row = _mapping(value, key, _LOCAL_PERCEPTION_FIELDS)
    schema = _text(row, "schema")
    version = _wire_int(row["version"], f"{key}.version")
    cell_order = _text(row, "cell_order")
    sample_count = _wire_int(row["sample_count"], f"{key}.sample_count")
    if (
        schema != NATIVE_PERCEPTION_CHANNEL_SCHEMA
        or version != NATIVE_PERCEPTION_CHANNEL_VERSION
        or cell_order != "geometry_cell_index_dx_dy_dz"
        or sample_count != CELL_COUNT
    ):
        raise ValueError(f"{key} contract is unsupported")
    origin = np.asarray(geometry["origin"], dtype=np.int32)
    positions = np.asarray(_LOCAL_CELL_OFFSETS + origin, dtype="<i4")
    return native_perception_channel_capture_from_wire(
        {
            "type": "perception_channels",
            "schema": schema,
            "version": version,
            "server_version": _text(provenance, "server_version"),
            "world": _text(provenance, "world"),
            "worldgen_provider": _text(provenance, "worldgen_provider"),
            "worldgen_version": _text(provenance, "worldgen_version"),
            "seed": _integer(provenance, "seed"),
            "sample_count": CELL_COUNT,
            "positions_i32_le_xyz": positions.tobytes(),
            "available_u8": row["available_u8"],
            "channel_validity_u8_bits": row["channel_validity_u8_bits"],
            "heightmap_i16_le": row["heightmap_i16_le"],
            "sky_light_u8": row["sky_light_u8"],
            "block_light_rgb_u8": row["block_light_rgb_u8"],
            "environment_i32_le": row["environment_i32_le"],
            "tint_argb_i32_le": row["tint_argb_i32_le"],
        }
    )


def _validate_bounded_rows(
    rows: Sequence[Any], total: int, overflow: bool, capacity: int, key: str
) -> None:
    if total < len(rows) or len(rows) > capacity or overflow != (total > len(rows)):
        raise ValueError(f"{key} metadata is inconsistent")


def _validate_observation_alignment(capture: NativeNpcTraceCapture) -> None:
    boundaries = (
        (
            capture.tick,
            capture.target_uuid,
            capture.actor_evidence,
            capture.target_actor_evidence,
            capture.observation,
            capture.worldview,
        ),
        (
            tuple(row.world_tick for row in capture.next_worldview),
            capture.next_target_uuid,
            capture.next_actor_evidence,
            capture.next_target_actor_evidence,
            capture.next_observation,
            capture.next_worldview,
        ),
    )
    for ticks, targets, actors, target_actors, observations, worldviews in boundaries:
        for tick, target_uuid, actor, target_actor, observation, worldview in zip(
            ticks,
            targets,
            actors,
            target_actors,
            observations,
            worldviews,
            strict=True,
        ):
            if observation.world_tick != int(tick) or worldview.world_tick != int(tick):
                raise ValueError("NPC semantic observation tick is misaligned")
            if observation.target_present != target_actor["present"] or (
                observation.target_perceptible != target_actor["perceptible"]
            ):
                raise ValueError("NPC target evidence disagrees with perception")
            if observation.target_present != (target_uuid is not None):
                raise ValueError("NPC target presence disagrees with its UUID")
            if target_uuid is not None and (
                (target_uuid in observation.perceptible_npc_uuids)
                != observation.target_perceptible
            ):
                raise ValueError(
                    "NPC target visibility disagrees with nearby perception"
                )
            if not worldview.overflow and not set(
                observation.perceptible_npc_uuids
            ).issubset({row.uuid for row in worldview.actors}):
                raise ValueError(
                    "NPC perception contains an actor outside the worldview"
                )
            if not worldview.entity_overflow and not set(
                observation.perceptible_entity_indices
            ).issubset({row.entity_index for row in worldview.entities}):
                raise ValueError(
                    "NPC perception contains an entity outside the worldview"
                )
            _validate_active_slot(
                actor["active_ability_slot"],
                observation.attack_candidates,
                observation.attack_candidate_overflow,
                "actor",
            )
            _validate_active_slot(
                target_actor["active_ability_slot"],
                observation.target_attack_candidates,
                observation.target_attack_candidate_overflow,
                "target",
            )


def _validate_attack_execution_causes(capture: NativeNpcTraceCapture) -> None:
    expected = tuple(
        _attack_execution_cause(current, next_value, interactions)
        for current, next_value, interactions in zip(
            capture.observation,
            capture.next_observation,
            capture.interactions,
            strict=True,
        )
    )
    if capture.attack_execution_cause != expected:
        raise ValueError("NPC attack execution cause disagrees with source evidence")


def _attack_execution_cause(current, next_value, interactions):
    if (
        current.attack_candidate_overflow
        or next_value.attack_candidate_overflow
        or current.attack_candidate_count != next_value.attack_candidate_count
    ):
        return NativeNpcAttackExecutionCause(False, False, -1, "", "", "")
    before = current.attack_candidates
    after = next_value.attack_candidates
    if any(
        left.path != right.path or left.label != right.label
        for left, right in zip(before, after, strict=True)
    ):
        return NativeNpcAttackExecutionCause(False, False, -1, "", "", "")
    starts = {
        (row.initial_root_id, row.type)
        for row in interactions
        if row.source == "combat_support" and row.first_run and row.initial_root_id
    }
    if not starts:
        return NativeNpcAttackExecutionCause(True, False, -1, "", "", "")
    if len(starts) != 1:
        return NativeNpcAttackExecutionCause(False, False, -1, "", "", "")
    interaction_id, interaction_type = next(iter(starts))
    matches = tuple(
        (index, candidate)
        for index, candidate in enumerate(after)
        if candidate.interaction_id == interaction_id
    )
    if len(matches) != 1 or matches[0][1].interaction_type != interaction_type:
        return NativeNpcAttackExecutionCause(False, False, -1, "", "", "")
    index, candidate = matches[0]
    return NativeNpcAttackExecutionCause(
        True,
        True,
        index,
        interaction_type,
        interaction_id,
        candidate.path,
    )


def _validate_active_slot(
    slot: int,
    candidates: Sequence[NativeNpcAttackAction],
    overflow: bool,
    label: str,
) -> None:
    active = tuple(index for index, value in enumerate(candidates) if value.active)
    if slot >= 0 and (slot >= len(candidates) or slot not in active):
        raise ValueError(f"NPC {label} active ability slot is inconsistent")
    if slot < -1 or (slot == -1 and active and not overflow):
        raise ValueError(f"NPC {label} active ability sentinel is inconsistent")


def _attack_action_rows(
    source: Mapping[str, Any], key: str, count: int
) -> tuple[tuple[NativeNpcAttackAction, ...], ...]:
    value = source.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise TypeError(f"{key} must match emitted_count")
    return tuple(_attack_action_row(row, key) for row in value)


def _attack_action_row(value: Any, key: str) -> tuple[NativeNpcAttackAction, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{key} rows must be arrays")
    result: list[NativeNpcAttackAction] = []
    for action in value:
        if not isinstance(action, list) or len(action) != 9:
            raise TypeError(f"{key} actions must match attack_action_layout")
        if any(not isinstance(action[index], str) for index in (0, 6, 7, 8)):
            raise TypeError(f"{key} action text fields must be strings")
        if any(not isinstance(action[index], bool) for index in (1, 2, 3)):
            raise TypeError(f"{key} action flags must be booleans")
        times = action[4:6]
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not np.isfinite(item)
            for item in times
        ):
            raise TypeError(f"{key} action times must be finite numbers")
        if action[5] < 0.0:
            raise ValueError(f"{key} charge time must be nonnegative")
        result.append(
            NativeNpcAttackAction(
                label=action[0],
                active=action[1],
                triggered=action[2],
                ready=action[3],
                aiming_seconds_remaining=float(action[4]),
                charge_seconds=float(action[5]),
                interaction_type=action[6],
                interaction_id=action[7],
                path=action[8],
            )
        )
    return tuple(result)


def _interaction_rows(
    source: Mapping[str, Any], key: str, count: int
) -> tuple[tuple[NativeNpcInteraction, ...], ...]:
    value = source.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise TypeError(f"{key} must match emitted_count")
    return tuple(_interaction_row(row, key) for row in value)


def _damage_event_rows(
    source: Mapping[str, Any], key: str, count: int
) -> tuple[tuple[NativeNpcDamageEvent, ...], ...]:
    value = source.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise TypeError(f"{key} must match emitted_count")
    rows: list[tuple[NativeNpcDamageEvent, ...]] = []
    for row in value:
        if not isinstance(row, list):
            raise TypeError(f"{key} rows must be arrays")
        rows.append(tuple(_damage_event(item, key) for item in row))
    return tuple(rows)


def _damage_event(value: Any, key: str) -> NativeNpcDamageEvent:
    if not isinstance(value, list) or len(value) != 22:
        raise TypeError(f"{key} events must match damage_event_layout")
    if any(not isinstance(value[index], str) for index in (0, 1, 3)):
        raise TypeError(f"{key} damage event text fields must be strings")
    if value[0] not in {"other", "entity", "projectile", "environment"}:
        raise ValueError(f"{key} damage source type is unknown")
    if (value[0] == "environment") != bool(value[1]):
        raise ValueError(f"{key} environment source identity is inconsistent")
    for index in (2, 10, 12, 14):
        if isinstance(value[index], bool) or not isinstance(value[index], int):
            raise TypeError(f"{key} damage event indexes must be integers")
    if any(value[index] < -1 for index in (10, 12, 14)):
        raise ValueError(f"{key} damage entity indexes must use -1 as absent")
    if any(not isinstance(value[index], bool) for index in (6, 7, 8, 9, 16, 18, 21)):
        raise TypeError(f"{key} damage event flags must be booleans")
    if not value[8] and not value[9]:
        raise ValueError(f"{key} damage event does not involve the traced actor")
    identifiers = tuple(
        _nullable_wire_uuid(value[index], f"{key}.uuid") for index in (11, 13, 15)
    )
    initial = _wire_number(value[4], f"{key}.initial_amount")
    final = _wire_number(value[5], f"{key}.final_amount")
    health = _wire_number(value[19], f"{key}.target_health_after")
    maximum = _wire_number(value[20], f"{key}.target_max_health")
    hit = _numbers(value[17], 4, f"{key}.hit_location")
    if initial < 0.0 or final < 0.0:
        raise ValueError(f"{key} damage amounts must be nonnegative")
    if not value[16] and any(hit):
        raise ValueError(f"{key} unavailable hit location must be zero")
    if (
        health < 0.0
        or maximum < 0.0
        or (not value[18] and (health != 0.0 or maximum != 0.0))
        or (value[18] and health > maximum)
        or (value[21] and (not value[18] or health > 0.0))
    ):
        raise ValueError(f"{key} damage health outcome is inconsistent")
    return NativeNpcDamageEvent(
        source_type=value[0],
        environment_type=value[1],
        damage_cause_index=value[2],
        damage_cause_id=value[3],
        initial_amount=initial,
        final_amount=final,
        cancelled=value[6],
        blocked=value[7],
        actor_source=value[8],
        actor_target=value[9],
        source_entity_index=value[10],
        source_uuid=identifiers[0],
        target_entity_index=value[12],
        target_uuid=identifiers[1],
        projectile_entity_index=value[14],
        projectile_uuid=identifiers[2],
        hit_location_available=value[16],
        hit_location=hit,
        health_available=value[18],
        target_health_after=health,
        target_max_health=maximum,
        lethal=value[21],
    )


_LIFECYCLE_KINDS = frozenset(
    {
        "stat_changed",
        "status_added",
        "status_removed",
        "status_refreshed",
        "inventory_transaction",
        "inventory_slot",
        "projectile_spawned",
        "projectile_despawned",
        "projectile_impacted",
        "actor_death",
        "actor_revived",
    }
)
_LIFECYCLE_ORIGINS = frozenset(
    {
        "server_boundary_state_delta",
        "server_inter_row_state_delta",
        "server_inventory_event",
        "server_death_component",
    }
)
_LIFECYCLE_KIND_SOURCE = {
    "stat_changed": NATIVE_NPC_LIFECYCLE_SOURCE_STATS,
    "status_added": NATIVE_NPC_LIFECYCLE_SOURCE_STATUS,
    "status_removed": NATIVE_NPC_LIFECYCLE_SOURCE_STATUS,
    "status_refreshed": NATIVE_NPC_LIFECYCLE_SOURCE_STATUS,
    "inventory_transaction": NATIVE_NPC_LIFECYCLE_SOURCE_INVENTORY,
    "inventory_slot": NATIVE_NPC_LIFECYCLE_SOURCE_INVENTORY,
    "projectile_spawned": NATIVE_NPC_LIFECYCLE_SOURCE_PROJECTILES,
    "projectile_despawned": NATIVE_NPC_LIFECYCLE_SOURCE_PROJECTILES,
    "projectile_impacted": NATIVE_NPC_LIFECYCLE_SOURCE_PROJECTILES,
    "actor_death": NATIVE_NPC_LIFECYCLE_SOURCE_ACTOR_LIFECYCLE,
    "actor_revived": NATIVE_NPC_LIFECYCLE_SOURCE_ACTOR_LIFECYCLE,
}


def _lifecycle_event_rows(
    source: Mapping[str, Any], key: str, count: int
) -> tuple[tuple[NativeNpcLifecycleEvent, ...], ...]:
    value = source.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise TypeError(f"{key} must match emitted_count")
    rows: list[tuple[NativeNpcLifecycleEvent, ...]] = []
    for row in value:
        if not isinstance(row, list):
            raise TypeError(f"{key} rows must be arrays")
        rows.append(tuple(_lifecycle_event(item, key) for item in row))
    return tuple(rows)


def _lifecycle_event(value: Any, key: str) -> NativeNpcLifecycleEvent:
    if not isinstance(value, list) or len(value) != 19:
        raise TypeError(f"{key} events must match lifecycle_event_layout")
    if any(not isinstance(value[index], str) for index in (0, 1, 2, 3, 9, 10)):
        raise TypeError(f"{key} lifecycle text fields must be strings")
    if value[0] not in _LIFECYCLE_KINDS or value[1] not in _LIFECYCLE_ORIGINS:
        raise ValueError(f"{key} lifecycle identity is unknown")
    expected_origin = (
        "server_inventory_event"
        if value[0].startswith("inventory_")
        else "server_death_component"
        if value[0].startswith("actor_")
        else "server_boundary_state_delta"
    )
    allowed_origins = {expected_origin}
    if not value[0].startswith("inventory_"):
        allowed_origins.add("server_inter_row_state_delta")
    if value[1] not in allowed_origins:
        raise ValueError(f"{key} lifecycle origin disagrees with its kind")
    for index in (4, 13, 18):
        if isinstance(value[index], bool) or not isinstance(value[index], int):
            raise TypeError(f"{key} lifecycle indexes must be integers")
    if value[4] < -1 or value[13] < -1 or value[18] < 0:
        raise ValueError(f"{key} lifecycle integers are out of range")
    if any(not isinstance(value[index], bool) for index in (11, 12, 16)):
        raise TypeError(f"{key} lifecycle flags must be booleans")
    numbers = tuple(_wire_number(value[index], f"{key}.value") for index in range(5, 9))
    entity_uuid = _nullable_wire_uuid(value[14], f"{key}.entity_uuid")
    owner_uuid = _nullable_wire_uuid(value[15], f"{key}.owner_uuid")
    position = _numbers(value[17], 3, f"{key}.position")
    if not value[16] and any(position):
        raise ValueError(f"{key} unavailable lifecycle position must be zero")
    return NativeNpcLifecycleEvent(
        kind=value[0],
        origin=value[1],
        subject=value[2],
        key=value[3],
        index=value[4],
        value_before=numbers[0],
        value_after=numbers[1],
        auxiliary_0=numbers[2],
        auxiliary_1=numbers[3],
        text_before=value[9],
        text_after=value[10],
        successful=value[11],
        complete=value[12],
        entity_index=value[13],
        entity_uuid=entity_uuid,
        owner_uuid=owner_uuid,
        position_available=value[16],
        position=position,
        flags=value[18],
    )


def _interaction_row(value: Any, key: str) -> tuple[NativeNpcInteraction, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{key} rows must be arrays")
    return tuple(_interaction(item, key) for item in value)


def _interaction(value: Any, key: str) -> NativeNpcInteraction:
    if not isinstance(value, list) or len(value) != 23:
        raise TypeError(f"{key} interactions must match interaction_layout")
    text_indexes = (0, 1, 2, 4, 5, 6, 7, 8)
    if any(not isinstance(value[index], str) for index in text_indexes):
        raise TypeError(f"{key} interaction text fields must be strings")
    if value[0] not in {
        "interaction_manager",
        "combat_support",
        "interaction_manager+combat_support",
    }:
        raise ValueError(f"{key} interaction source is unknown")
    integer_indexes = (3, 11, 12, 13, 14, 15, 16)
    if any(
        isinstance(value[index], bool) or not isinstance(value[index], int)
        for index in integer_indexes
    ):
        raise TypeError(f"{key} interaction counters must be integers")
    times = value[9:11]
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not np.isfinite(item)
        for item in times
    ):
        raise TypeError(f"{key} interaction times must be finite numbers")
    if any(not isinstance(value[index], bool) for index in range(17, 22)):
        raise TypeError(f"{key} interaction flags must be booleans")
    target = value[22]
    if target is not None and (not isinstance(target, bytes) or len(target) != 16):
        raise TypeError(f"{key} interaction target must be a UUID or nil")
    return NativeNpcInteraction(
        source=value[0],
        type=value[1],
        base_type=value[2],
        chain_id=value[3],
        initial_root_id=value[4],
        root_id=value[5],
        server_state=value[6],
        client_state=value[7],
        final_state=value[8],
        time_seconds=float(value[9]),
        time_shift=float(value[10]),
        operation_counter=value[11],
        simulated_operation_counter=value[12],
        operation_index=value[13],
        client_operation_index=value[14],
        call_depth=value[15],
        simulated_call_depth=value[16],
        predicted=value[17],
        requires_client=value[18],
        first_run=value[19],
        pre_ticked=value[20],
        desynced=value[21],
        target_uuid=None if target is None else uuid.UUID(bytes=target),
    )


def _internal_rows(
    source: Mapping[str, Any], key: str, count: int
) -> tuple[NativeNpcInternalState, ...]:
    value = source.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise TypeError(f"{key} must match emitted_count")
    return tuple(_internal(row, key) for row in value)


def _internal(value: Any, key: str) -> NativeNpcInternalState:
    if not isinstance(value, list) or len(value) != 17:
        raise TypeError(f"{key} rows must match internal_layout")
    if any(not isinstance(value[index], str) for index in (0, 8)):
        raise TypeError(f"{key} text fields must be strings")
    if any(
        isinstance(value[index], bool) or not isinstance(value[index], int)
        for index in (1, 2)
    ):
        raise TypeError(f"{key} state indexes must be integers")
    if any(not isinstance(value[index], bool) for index in range(3, 8)) or any(
        not isinstance(value[index], bool) for index in range(9, 12)
    ):
        raise TypeError(f"{key} flags must be booleans")
    current_speed = _wire_number(value[12], f"{key}.current_speed")
    maximum_speed = _wire_number(value[13], f"{key}.maximum_speed")
    if current_speed < 0.0 or maximum_speed < 0.0:
        raise ValueError(f"{key} speeds must be nonnegative")
    marked = value[16]
    if not isinstance(marked, list):
        raise TypeError(f"{key}.marked_targets must be an array")
    targets: list[NativeNpcMarkedTarget] = []
    for item in marked:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or isinstance(item[0], bool)
            or not isinstance(item[0], int)
            or item[0] < 0
            or not isinstance(item[1], str)
            or not isinstance(item[2], bytes)
            or len(item[2]) != 16
        ):
            raise TypeError(f"{key}.marked_targets rows must be [slot,name,uuid]")
        targets.append(
            NativeNpcMarkedTarget(item[0], item[1], uuid.UUID(bytes=item[2]))
        )
    if len({target.slot for target in targets}) != len(targets):
        raise ValueError(f"{key}.marked_targets contains duplicate slots")
    if not value[9] and (
        value[8]
        or value[10]
        or value[11]
        or current_speed != 0.0
        or maximum_speed != 0.0
    ):
        raise ValueError(f"{key} missing motion controller carries motion state")
    return NativeNpcInternalState(
        state_name=value[0],
        state_index=value[1],
        substate_index=value[2],
        busy=value[3],
        transitioning=value[4],
        role_change_requested=value[5],
        terminal_action=value[6],
        backing_away=value[7],
        steering_motion=value[8],
        motion_controller_present=value[9],
        motion_in_progress=value[10],
        obstructed=value[11],
        current_speed=current_speed,
        maximum_speed=maximum_speed,
        avoidance_steering=_numbers(value[14], 3, f"{key}.avoidance_steering"),
        separation_steering=_numbers(value[15], 3, f"{key}.separation_steering"),
        marked_targets=tuple(targets),
    )


def _worldview_rows(
    source: Mapping[str, Any], key: str, count: int
) -> tuple[NativeNpcWorldSnapshot, ...]:
    value = source.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise TypeError(f"{key} must match emitted_count")
    return tuple(_worldview(row, key) for row in value)


def _worldview(value: Any, key: str) -> NativeNpcWorldSnapshot:
    if not isinstance(value, list) or len(value) != 14:
        raise TypeError(f"{key} rows must match the worldview schema")
    integer_indexes = (0, 1, 2, 5, 6, 9, 12, 13)
    if any(
        isinstance(value[index], bool) or not isinstance(value[index], int)
        for index in integer_indexes
    ):
        raise TypeError(f"{key} clock and count fields must be integers")
    day = _wire_number(value[3], f"{key}.day_progress")
    sunlight = _wire_number(value[4], f"{key}.sunlight_factor")
    if (
        not isinstance(value[7], bool)
        or not isinstance(value[8], list)
        or not isinstance(value[10], bool)
        or not isinstance(value[11], list)
    ):
        raise TypeError(f"{key} overflow/actors fields have wrong types")
    actors = tuple(_world_actor(item, key) for item in value[8])
    uuids = tuple(actor.uuid.bytes for actor in actors)
    if uuids != tuple(sorted(uuids)) or len(set(uuids)) != len(uuids):
        raise ValueError(f"{key} actors must be uniquely UUID-sorted")
    if (
        value[0] < 0
        or not 0 <= value[2] < 1_000_000_000
        or not 0 <= value[13] < 1_000_000_000
        or not 0.0 <= day <= 1.0
        or not 0.0 <= sunlight <= 1.0
        or value[5] < 0
        or value[6] < len(actors)
        or len(actors) > NATIVE_NPC_WORLDVIEW_ACTOR_CAPACITY
        or value[7] != (value[6] > len(actors))
    ):
        raise ValueError(f"{key} worldview metadata is inconsistent")
    entities = tuple(_world_entity(item, key) for item in value[11])
    indexes = tuple(entity.entity_index for entity in entities)
    if indexes != tuple(sorted(indexes)) or len(set(indexes)) != len(indexes):
        raise ValueError(f"{key} entities must be uniquely index-sorted")
    if (
        value[9] < len(entities)
        or len(entities) > NATIVE_NPC_WORLDVIEW_ENTITY_CAPACITY
        or value[10] != (value[9] > len(entities))
    ):
        raise ValueError(f"{key} entity-worldview metadata is inconsistent")
    return NativeNpcWorldSnapshot(
        world_tick=value[0],
        game_time_epoch_second=value[1],
        game_time_nano=value[2],
        day_progress=day,
        sunlight_factor=sunlight,
        moon_phase=value[5],
        npc_count=value[6],
        overflow=value[7],
        actors=actors,
        entity_count=value[9],
        entity_overflow=value[10],
        entities=entities,
        simulation_time_epoch_second=value[12],
        simulation_time_nano=value[13],
    )


def _world_actor(value: Any, key: str) -> NativeNpcWorldActor:
    if (
        not isinstance(value, list)
        or len(value) != 7
        or not isinstance(value[0], bytes)
        or len(value[0]) != 16
        or not isinstance(value[1], str)
        or not value[1]
        or not isinstance(value[3], str)
        or not isinstance(value[5], bool)
    ):
        raise TypeError(f"{key} worldview actor has wrong types")
    target = value[4]
    if target is not None and (not isinstance(target, bytes) or len(target) != 16):
        raise TypeError(f"{key} worldview target must be a UUID or nil")
    pause = _wire_number(value[6], f"{key}.attack_pause_seconds")
    if pause < 0.0 or (pause > 0.0 and not value[5]):
        raise ValueError(f"{key} worldview combat lifecycle is inconsistent")
    return NativeNpcWorldActor(
        uuid=uuid.UUID(bytes=value[0]),
        role=value[1],
        state=_numbers(value[2], NATIVE_NPC_TRACE_STATE_WIDTH, f"{key}.state"),
        state_name=value[3],
        target_uuid=None if target is None else uuid.UUID(bytes=target),
        combat_attack=value[5],
        attack_pause_seconds=pause,
    )


def _world_entity(value: Any, key: str) -> NativeNpcWorldEntity:
    if (
        not isinstance(value, list)
        or len(value) != 18
        or isinstance(value[0], bool)
        or not isinstance(value[0], int)
        or value[0] < 0
        or isinstance(value[2], bool)
        or not isinstance(value[2], int)
        or not 0 <= value[2] <= NATIVE_NPC_WORLDVIEW_ENTITY_ALL_FLAGS
        or any(not isinstance(value[index], str) for index in (3, 4, 12))
        or not isinstance(value[7], bool)
        or not isinstance(value[9], bool)
        or not isinstance(value[13], bool)
        or any(
            isinstance(value[index], bool) or not isinstance(value[index], int)
            for index in (14, 15, 16, 17)
        )
    ):
        raise TypeError(f"{key} worldview entity has wrong types")
    identity = _nullable_wire_uuid(value[1], f"{key}.uuid")
    owner = _nullable_wire_uuid(value[11], f"{key}.owner_uuid")
    velocity = _numbers(value[8], 3, f"{key}.velocity")
    bounds = _numbers(value[10], 6, f"{key}.bounds")
    if not value[7] and any(velocity):
        raise ValueError(f"{key} unavailable entity velocity is nonzero")
    if not value[9] and any(bounds):
        raise ValueError(f"{key} unavailable entity bounds are nonzero")
    timing_values = (value[14], value[15], value[16], value[17])
    if not 0 <= value[15] < 1_000_000_000 or not 0 <= value[17] < 1_000_000_000:
        raise ValueError(f"{key} entity lifecycle nanoseconds are invalid")
    if not value[13] and any(timing_values):
        raise ValueError(f"{key} unavailable entity lifecycle timing is nonzero")
    if value[13] and (value[16], value[17]) < (value[14], value[15]):
        raise ValueError(f"{key} entity lifecycle end precedes its start")
    return NativeNpcWorldEntity(
        entity_index=value[0],
        uuid=identity,
        flags=value[2],
        asset_id=value[3],
        model_asset_id=value[4],
        position=_numbers(value[5], 3, f"{key}.position"),
        rotation=_numbers(value[6], 3, f"{key}.rotation"),
        velocity_available=value[7],
        velocity=velocity,
        bounds_available=value[9],
        bounds=bounds,
        owner_uuid=owner,
        physics_state=value[12],
        lifecycle_timing_available=value[13],
        lifecycle_start_epoch_second=value[14],
        lifecycle_start_nano=value[15],
        lifecycle_end_epoch_second=value[16],
        lifecycle_end_nano=value[17],
    )


def _nullable_wire_uuid(value: Any, key: str) -> uuid.UUID | None:
    if value is None:
        return None
    if not isinstance(value, bytes) or len(value) != 16:
        raise TypeError(f"{key} must be UUID bytes or nil")
    return uuid.UUID(bytes=value)


_ACTOR_FIELDS = frozenset(
    {
        "entity_id",
        "present",
        "perceptible",
        "role_id",
        "item_id",
        "item_runtime_index",
        "active_ability_slot",
        "position",
        "velocity",
        "motion_force",
        "yaw_degrees",
        "pitch_degrees",
        "health",
        "max_health",
        "resource_values",
        "resource_maximums",
        "resource_available",
        "defense_values",
        "defense_available",
        "statuses",
        "actor_world_values",
        "actor_world_available",
        "movement_states",
    }
)
_MOTION_FIELDS = frozenset(
    {
        "legacy_external_available",
        "legacy_external_velocity",
        "configured_applied_available",
        "configured_applied_velocity",
        "configured_applied_count",
        "pending_knockback_available",
        "pending_knockback_velocity",
        "projected_available",
        "projected_velocity",
        "projection_source",
    }
)
_STATUS_FIELDS = frozenset(
    {
        "runtime_effect_index",
        "effect_id",
        "initial_duration_seconds",
        "remaining_duration_seconds",
        "infinite",
        "debuff",
        "invulnerable",
    }
)


def _actor_rows(
    source: Mapping[str, Any],
    key: str,
    count: int,
    entity_id: int,
    require_present: bool,
) -> tuple[Mapping[str, Any], ...]:
    value = source.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise TypeError(f"{key} must match emitted_count")
    return tuple(_actor(row, key, entity_id, require_present) for row in value)


def _actor(
    value: Any, key: str, entity_id: int, require_present: bool
) -> Mapping[str, Any]:
    row = _mapping(value, key, _ACTOR_FIELDS)
    if _wire_int(row["entity_id"], f"{key}.entity_id") != entity_id:
        raise ValueError(f"{key} must use trace-local entity_id {entity_id}")
    for name in ("present", "perceptible"):
        if not isinstance(row[name], bool):
            raise TypeError(f"{key}.{name} must be boolean")
    if require_present and (not row["present"] or not row["perceptible"]):
        raise ValueError(f"{key} must contain the traced actor")
    if not row["present"] and row["perceptible"]:
        raise ValueError(f"{key} absent actor cannot be perceptible")
    for name in ("role_id", "item_id"):
        if not isinstance(row[name], str) or (
            name == "role_id" and row["present"] and not row[name]
        ):
            raise TypeError(f"{key}.{name} must be a string")
    result = {
        "entity_id": entity_id,
        "present": row["present"],
        "perceptible": row["perceptible"],
        "role_id": row["role_id"],
        "item_id": row["item_id"],
        "item_runtime_index": _wire_int(
            row["item_runtime_index"], f"{key}.item_runtime_index"
        ),
        "active_ability_slot": _wire_int(
            row["active_ability_slot"], f"{key}.active_ability_slot"
        ),
        "position": _numbers(row["position"], 3, f"{key}.position"),
        "velocity": _numbers(row["velocity"], 3, f"{key}.velocity"),
        "motion_force": _motion(row["motion_force"], key),
        "yaw_degrees": _wire_number(row["yaw_degrees"], f"{key}.yaw_degrees"),
        "pitch_degrees": _wire_number(row["pitch_degrees"], f"{key}.pitch_degrees"),
        "health": _wire_number(row["health"], f"{key}.health"),
        "max_health": _wire_number(row["max_health"], f"{key}.max_health"),
        "resource_values": _numbers(
            row["resource_values"], 7, f"{key}.resource_values"
        ),
        "resource_maximums": _numbers(
            row["resource_maximums"], 7, f"{key}.resource_maximums"
        ),
        "resource_available": _booleans(
            row["resource_available"], 7, f"{key}.resource_available"
        ),
        "defense_values": _numbers(row["defense_values"], 7, f"{key}.defense_values"),
        "defense_available": _booleans(
            row["defense_available"], 7, f"{key}.defense_available"
        ),
        "statuses": _statuses(row["statuses"], key),
        "actor_world_values": _numbers(
            row["actor_world_values"], 5, f"{key}.actor_world_values"
        ),
        "actor_world_available": _booleans(
            row["actor_world_available"], 3, f"{key}.actor_world_available"
        ),
        "movement_states": _movement(row["movement_states"], key),
    }
    if result["present"] and (
        result["max_health"] <= 0.0
        or not 0.0 <= result["health"] <= result["max_health"] + 1e-6
    ):
        raise ValueError(f"{key} health is outside its declared range")
    if not result["present"] and (
        result["role_id"]
        or result["item_id"]
        or result["health"] != 0.0
        or result["max_health"] != 0.0
        or any(result["resource_available"])
        or any(result["defense_available"])
        or result["statuses"]
        or any(result["actor_world_available"])
        or result["movement_states"]["available"]
    ):
        raise ValueError(f"{key} absent actor carries usable evidence")
    return MappingProxyType(result)


def _motion(value: Any, key: str) -> Mapping[str, Any]:
    row = _mapping(value, f"{key}.motion_force", _MOTION_FIELDS)
    result: dict[str, Any] = {}
    for name in (
        "legacy_external_available",
        "configured_applied_available",
        "pending_knockback_available",
        "projected_available",
    ):
        if not isinstance(row[name], bool):
            raise TypeError(f"{key}.motion_force.{name} must be boolean")
        result[name] = row[name]
    for name in (
        "legacy_external_velocity",
        "configured_applied_velocity",
        "pending_knockback_velocity",
        "projected_velocity",
    ):
        result[name] = _numbers(row[name], 3, f"{key}.motion_force.{name}")
    result["configured_applied_count"] = _wire_int(
        row["configured_applied_count"], f"{key}.motion_force.configured_applied_count"
    )
    result["projection_source"] = _wire_int(
        row["projection_source"], f"{key}.motion_force.projection_source"
    )
    return MappingProxyType(result)


def _statuses(value: Any, key: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or len(value) > 8:
        raise TypeError(f"{key}.statuses must fit the native status capacity")
    result = []
    for index, item in enumerate(value):
        prefix = f"{key}.statuses[{index}]"
        row = _mapping(item, prefix, _STATUS_FIELDS)
        if not isinstance(row["effect_id"], str):
            raise TypeError(f"{prefix}.effect_id must be a string")
        for name in ("infinite", "debuff", "invulnerable"):
            if not isinstance(row[name], bool):
                raise TypeError(f"{prefix}.{name} must be boolean")
        result.append(
            MappingProxyType(
                {
                    "runtime_effect_index": _wire_int(
                        row["runtime_effect_index"], f"{prefix}.runtime_effect_index"
                    ),
                    "effect_id": row["effect_id"],
                    "initial_duration_seconds": _wire_number(
                        row["initial_duration_seconds"],
                        f"{prefix}.initial_duration_seconds",
                    ),
                    "remaining_duration_seconds": _wire_number(
                        row["remaining_duration_seconds"],
                        f"{prefix}.remaining_duration_seconds",
                    ),
                    "infinite": row["infinite"],
                    "debuff": row["debuff"],
                    "invulnerable": row["invulnerable"],
                }
            )
        )
    return tuple(result)


def _movement(value: Any, key: str) -> Mapping[str, Any]:
    row = _mapping(value, f"{key}.movement_states", frozenset({"available", "bits"}))
    if not isinstance(row["available"], bool):
        raise TypeError(f"{key}.movement_states.available must be boolean")
    bits = _wire_int(row["bits"], f"{key}.movement_states.bits")
    if not 0 <= bits < 1 << 23 or (not row["available"] and bits):
        raise ValueError(f"{key}.movement_states bits are invalid")
    return MappingProxyType({"available": row["available"], "bits": bits})


def _mapping(value: Any, key: str, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TypeError(f"{key} does not match its native evidence schema")
    return dict(value)


def _numbers(value: Any, width: int, key: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != width:
        raise TypeError(f"{key} must contain {width} numbers")
    return tuple(_wire_number(item, key) for item in value)


def _booleans(value: Any, width: int, key: str) -> tuple[bool, ...]:
    if (
        not isinstance(value, list)
        or len(value) != width
        or any(not isinstance(item, bool) for item in value)
    ):
        raise TypeError(f"{key} must contain {width} booleans")
    return tuple(value)


def _wire_number(value: Any, key: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise TypeError(f"{key} must be a finite number")
    return float(value)


def _wire_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _wire_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be boolean")
    return value


def _strings(source, key: str, count: int) -> tuple[str, ...]:
    value = source.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise TypeError(f"{key} must match emitted_count")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a string array")
    return tuple(value)


def _string_rows(
    source: Mapping[str, Any], key: str, count: int
) -> tuple[tuple[str, ...], ...]:
    value = source.get(key)
    if not isinstance(value, list) or len(value) != count:
        raise TypeError(f"{key} must match emitted_count")
    rows: list[tuple[str, ...]] = []
    for row in value:
        if not isinstance(row, list) or any(not isinstance(item, str) for item in row):
            raise TypeError(f"{key} rows must be string arrays")
        if len(set(row)) != len(row):
            raise ValueError(f"{key} rows must not contain duplicates")
        rows.append(tuple(row))
    return tuple(rows)


def _string_map(source: Mapping[str, Any], key: str) -> Mapping[str, str]:
    value = source.get(key)
    if not isinstance(value, Mapping) or any(
        not isinstance(name, str) or not name or not isinstance(item, str)
        for name, item in value.items()
    ):
        raise TypeError(f"{key} must be a string map")
    return MappingProxyType(dict(value))


def _text(source, key: str, *, nonempty: bool = False) -> str:
    value = source.get(key)
    if not isinstance(value, str) or (nonempty and not value):
        raise TypeError(f"{key} must be a{' non-empty' if nonempty else ''} string")
    return value


def _integer(
    source, key: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be at most {maximum}")
    return value


def _bounded(value: int, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


__all__ = [
    "NATIVE_NPC_ATTACK_EXECUTION_CAUSE_LAYOUT",
    "NATIVE_NPC_ACTION_CONTRACT_SCHEMA",
    "NATIVE_NPC_ACTION_CONTRACT_SHA256",
    "NATIVE_NPC_ACTION_CONTRACT_VERSION",
    "NATIVE_NPC_DAMAGE_EVENT_CAPACITY",
    "NATIVE_NPC_DAMAGE_EVENT_LAYOUT",
    "NATIVE_NPC_EVENT_CONTRACT_SCHEMA",
    "NATIVE_NPC_EVENT_CONTRACT_SHA256",
    "NATIVE_NPC_EVENT_CONTRACT_VERSION",
    "NATIVE_NPC_LIFECYCLE_ALL_SOURCE_BITS",
    "NATIVE_NPC_LIFECYCLE_CONTRACT_SCHEMA",
    "NATIVE_NPC_LIFECYCLE_CONTRACT_SHA256",
    "NATIVE_NPC_LIFECYCLE_CONTRACT_VERSION",
    "NATIVE_NPC_LIFECYCLE_EVENT_CAPACITY",
    "NATIVE_NPC_LIFECYCLE_EVENT_LAYOUT",
    "NATIVE_NPC_LIFECYCLE_SOURCE_ACTOR_LIFECYCLE",
    "NATIVE_NPC_LIFECYCLE_SOURCE_INVENTORY",
    "NATIVE_NPC_LIFECYCLE_SOURCE_LAYOUT",
    "NATIVE_NPC_LIFECYCLE_SOURCE_PROJECTILES",
    "NATIVE_NPC_LIFECYCLE_SOURCE_STATS",
    "NATIVE_NPC_LIFECYCLE_SOURCE_STATUS",
    "NATIVE_NPC_TRACE_ATTACK_ACTION_LAYOUT",
    "NATIVE_NPC_TRACE_CONTROL_LAYOUT",
    "NATIVE_NPC_TRACE_CONTROL_MASK_LAYOUT",
    "NATIVE_NPC_TRACE_CONTROL_WIDTH",
    "NATIVE_NPC_TRACE_DEFAULT_CAPACITY",
    "NATIVE_NPC_TRACE_MAX_CAPACITY",
    "NATIVE_NPC_TRACE_MAX_ACTIVE",
    "NATIVE_NPC_TRACE_MAX_DRAIN",
    "NATIVE_NPC_TRACE_INTERACTION_LAYOUT",
    "NATIVE_NPC_TRACE_INTERNAL_LAYOUT",
    "NATIVE_NPC_TRACE_SCHEMA",
    "NATIVE_NPC_TRACE_STATE_LAYOUT",
    "NATIVE_NPC_TRACE_STATE_WIDTH",
    "NATIVE_NPC_TRACE_VERSION",
    "NATIVE_NPC_WORLDVIEW_ACTOR_CAPACITY",
    "NATIVE_NPC_WORLDVIEW_ACTOR_LAYOUT",
    "NATIVE_NPC_WORLDVIEW_ENTITY_ALL_FLAGS",
    "NATIVE_NPC_WORLDVIEW_ENTITY_CAPACITY",
    "NATIVE_NPC_WORLDVIEW_ENTITY_DEPLOYABLE",
    "NATIVE_NPC_WORLDVIEW_ENTITY_IMPACTED_OR_BOUNCED",
    "NATIVE_NPC_WORLDVIEW_ENTITY_IN_FLUID",
    "NATIVE_NPC_WORLDVIEW_ENTITY_LAYOUT",
    "NATIVE_NPC_WORLDVIEW_ENTITY_LEGACY_PROJECTILE",
    "NATIVE_NPC_WORLDVIEW_ENTITY_NPC",
    "NATIVE_NPC_WORLDVIEW_ENTITY_ON_GROUND",
    "NATIVE_NPC_WORLDVIEW_ENTITY_PREDICTED_PROJECTILE",
    "NATIVE_NPC_WORLDVIEW_ENTITY_RESTING_OR_SLIDING",
    "NATIVE_NPC_WORLDVIEW_ENTITY_STANDARD_PROJECTILE",
    "NATIVE_NPC_WORLDVIEW_SCHEMA",
    "NATIVE_NPC_WORLDVIEW_VERSION",
    "NATIVE_NPC_ACTOR_EVIDENCE_CONTRACT_SHA256",
    "NATIVE_NPC_ACTOR_EVIDENCE_SCHEMA",
    "NATIVE_NPC_ACTOR_EVIDENCE_VERSION",
    "NATIVE_NPC_OBSERVATION_ALL_GROUP_BITS",
    "NATIVE_NPC_OBSERVATION_ATTACK_CANDIDATE_CAPACITY",
    "NATIVE_NPC_OBSERVATION_COMPATIBLE_IDENTITIES",
    "NATIVE_NPC_OBSERVATION_CONTRACT_SHA256",
    "NATIVE_NPC_OBSERVATION_GROUP_LAYOUT",
    "NATIVE_NPC_OBSERVATION_PERCEPTIBLE_ENTITY_CAPACITY",
    "NATIVE_NPC_OBSERVATION_PERCEPTIBLE_NPC_CAPACITY",
    "NATIVE_NPC_OBSERVATION_SCHEMA",
    "NATIVE_NPC_OBSERVATION_VERSION",
    "NativeNpcAttackAction",
    "NativeNpcAttackExecutionCause",
    "NativeNpcDamageEvent",
    "NativeNpcInternalState",
    "NativeNpcInteraction",
    "NativeNpcLifecycleEvent",
    "NativeNpcMarkedTarget",
    "NativeNpcObservation",
    "NativeNpcTraceCapture",
    "NativeNpcWorldActor",
    "NativeNpcWorldEntity",
    "NativeNpcWorldSnapshot",
    "concatenate_native_npc_traces",
    "native_npc_trace_poll_request",
    "native_npc_trace_start_request",
    "native_npc_trace_stop_request",
]
