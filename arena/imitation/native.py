"""Adapters from authoritative Java NPC captures to composable Arena traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import numpy as np

from arena.imitation.contract import (
    TraceComponent,
    TraceContract,
    TraceSequence,
    TraceView,
    numeric_component,
    structured_component,
)
from arena.imitation.corpus import TraceCorpus
from arena.imitation.projectors import install_episode_evidence
from hytalegym.worldgen import (
    NATIVE_NPC_TRACE_ATTACK_ACTION_LAYOUT,
    NATIVE_NPC_TRACE_CONTROL_LAYOUT,
    NATIVE_NPC_TRACE_CONTROL_MASK_LAYOUT,
    NATIVE_NPC_DAMAGE_EVENT_LAYOUT,
    NATIVE_NPC_EVENT_CONTRACT_SHA256,
    NATIVE_NPC_LIFECYCLE_CONTRACT_SHA256,
    NATIVE_NPC_LIFECYCLE_EVENT_LAYOUT,
    NATIVE_NPC_LIFECYCLE_SOURCE_ACTOR_LIFECYCLE,
    NATIVE_NPC_TRACE_INTERACTION_LAYOUT,
    NATIVE_NPC_TRACE_INTERNAL_LAYOUT,
    NATIVE_NPC_OBSERVATION_COMPATIBLE_IDENTITIES,
    NATIVE_NPC_OBSERVATION_GROUP_LAYOUT,
    NATIVE_NPC_TRACE_SCHEMA,
    NATIVE_NPC_TRACE_STATE_LAYOUT,
    NATIVE_NPC_TRACE_VERSION,
    NATIVE_NPC_WORLDVIEW_ACTOR_LAYOUT,
    NATIVE_NPC_WORLDVIEW_ENTITY_CAPACITY,
    NATIVE_NPC_WORLDVIEW_ENTITY_LAYOUT,
    NATIVE_NPC_WORLDVIEW_SCHEMA,
    NATIVE_NPC_WORLDVIEW_VERSION,
    NativeNpcTraceCapture,
)
from hytalegym.jax.combat.observation.v3.native.channels.events import (
    NativeDamageEventBatch,
    pack_native_damage_events,
)
from hytalegym.jax.combat.observation.v3.native.channels.group import (
    NativeGroupStepBatch,
    NativeGroupStepEvidence,
    pack_native_group_step_evidence,
)
from hytalegym.jax.combat.observation.v3.native.channels.lifecycle import (
    NativeLifecycleEventBatch,
    pack_native_lifecycle_events,
)
from hytalegym.jax.combat.observation.v3.native.capabilities import (
    NATIVE_NPC_ACTION_CAPABILITY_SCHEMA,
    NATIVE_NPC_ACTION_CAPABILITY_SHA256,
    NATIVE_NPC_ACTION_CAPABILITY_VERSION,
    project_native_npc_action_capabilities,
)
from hytalegym.jax.combat.observation.v3.policy import (
    arsenal_policy_contract_sha256,
)
from hytalegym.geometry import parse_geometry
from hytalegym.jax.combat.observation.v3.schema.contract import (
    ACTOR_WORLD_FLOAT_FEATURES,
    ACTOR_WORLD_MASK_FEATURES,
    DEFENSE_FLOAT_FEATURES,
    DEFENSE_FLOAT_SIZE,
    LEARNER_OBSERVATION_V3_SCHEMA,
    LEARNER_OBSERVATION_V3_VERSION,
    MOVEMENT_STATE_FEATURES,
)
from hytalegym.jax.combat.observation.v3.schema.spec import (
    learner_observation_v3_contract_sha256,
)


_LEARNER_OBSERVATION_SHA256 = learner_observation_v3_contract_sha256()
NATIVE_HYTALE_ACTION_MASK = "hytale.arsenal_action_mask"
NATIVE_EPISODE_PROJECTOR_NAME = "arena.native_actor_lifecycle"
NATIVE_EPISODE_PROJECTOR_VERSION = 1
_RELATIVE_LAYOUT = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_vx",
    "delta_vy",
    "delta_vz",
    "distance",
    "planar_distance",
    "bearing_sin",
    "bearing_cos",
    "target_health_fraction",
)
_DECISION_CHANGE_LAYOUT = (
    "control",
    "control_mask",
    "control_owner",
    "body_instruction",
    "head_instruction",
    "active_actions",
    "internal_state",
    "target",
    "combat_phase",
    "attack_actions",
    "interactions",
)
_DECISION_SUPPORT_CHANGE_LAYOUT = (
    "state",
    "busy",
    "transitioning",
    "role_change_requested",
    "terminal_action",
    "backing_away",
    "steering_motion",
    "motion_in_progress",
    "obstructed",
    "speed",
    "marked_targets",
    "locked_target",
)
NATIVE_CORPUS_COMPATIBILITY_KEYS = (
    "native.action_contract_sha256",
    "native.event_contract_sha256",
    "native.lifecycle_contract_sha256",
    "native.actor_evidence_contract_sha256",
    "native.action_capability_contract_sha256",
    "native.bridge_sha256",
    "native.learner_observation_contract_sha256",
    "native.schema",
    "native.semantic_observation_schema",
    "native.semantic_observation_contract_sha256",
    "native.semantic_observation_version",
    "native.server_version",
    "native.version",
)


NATIVE_BEHAVIOR_VIEW = TraceView(
    inputs=(
        "native.actor_state",
        "native.semantic_observation",
        "native.internal_state",
        "native.behavior_state",
        "native.combat.phase",
        "native.interactions",
        "native.attack_candidates",
    ),
    targets=(
        "native.control",
        "native.attack_activation",
        "native.attack_execution_cause",
        "native.combat_attack",
    ),
    context=(
        "native.control_mask",
        "native.target_uuid",
        "native.active_actions",
        "native.combat.attack_actions",
    ),
)
NATIVE_CONTEXTUAL_BEHAVIOR_VIEW = TraceView(
    inputs=(
        "native.actor_state",
        "spatial.opponent_relative",
        "native.semantic_observation",
        "native.actor_view_pose",
        "native.internal_state",
        "native.decision_internal_state",
        "native.behavior_state",
        "native.geometry",
        "native.role_opaque_cell_mask",
        "native.local_perception_channels",
        "native.worldview",
    ),
    targets=(
        "native.control",
        "native.body_instruction",
        "native.head_instruction",
        "native.attack_activation",
        "native.attack_execution_cause",
        "native.combat_attack",
    ),
    context=(
        "native.control_mask",
        "native.next_decision_valid",
        "native.target_uuid",
        "native.decision_change",
        "native.combat.phase",
        "native.attack_candidates",
    ),
)
NATIVE_CONTEXTUAL_STEERING_VIEW = TraceView(
    inputs=(
        "native.actor_state",
        "spatial.opponent_relative",
        "native.semantic_observation",
        "native.actor_view_pose",
        "native.internal_state",
        "native.behavior_state",
        "native.geometry",
        "native.role_opaque_cell_mask",
        "native.local_perception_channels",
        "native.worldview",
    ),
    targets=("native.control",),
    context=("native.control_mask", "native.target_uuid"),
)
NATIVE_WORLD_MODEL_VIEW = TraceView(
    inputs=(
        "native.worldview",
        "native.world_entities",
        "native.world_clock",
        "native.actor_state",
        "native.actor_observation",
        "native.semantic_observation",
        "native.geometry",
        "native.role_opaque_cell_mask",
        "native.local_perception_channels",
        "native.inventory",
        "native.perceptible_npc_uuids",
        "native.perceived_entities",
        "native.actor_view_pose",
        "native.control",
        "native.attack_activation",
        "native.attack_execution_cause",
        "native.combat_attack",
    ),
    targets=(
        "native.worldview.next",
        "native.world_entities.next",
        "native.world_clock.next",
        "native.actor_state.next",
        "native.combat_attack.next",
        "native.interactions.next",
        "native.actor_observation.next",
        "native.semantic_observation.next",
        "native.geometry.next",
        "native.role_opaque_cell_mask.next",
        "native.local_perception_channels.next",
        "native.inventory.next",
        "native.perceptible_npc_uuids.next",
        "native.perceived_entities.next",
        "native.actor_view_pose.next",
        "native.damage_events",
        "native.lifecycle_events",
    ),
    context=("native.interactions", "native.target_uuid"),
)


def native_trace_sequence(
    capture: NativeNpcTraceCapture,
    *extensions: TraceComponent,
    parameters: Mapping[str, Any] | None = None,
) -> TraceSequence:
    """Preserve every native field as a trainer-neutral state transition."""

    if not isinstance(capture, NativeNpcTraceCapture):
        raise TypeError("capture must be NativeNpcTraceCapture")
    observation_identity = (
        capture.observation_schema,
        capture.observation_version,
        capture.observation_contract_sha256,
    )
    if observation_identity not in NATIVE_NPC_OBSERVATION_COMPATIBLE_IDENTITIES:
        raise ValueError("unsupported native semantic-observation contract")
    time_to_attack, time_to_attack_valid = _time_to_event(
        capture.attack_activation, capture.delta_seconds
    )
    view_pose, view_pose_valid = _actor_view_pose(capture.state, capture.observation)
    next_view_pose, next_view_pose_valid = _actor_view_pose(
        capture.next_state, capture.next_observation
    )
    decision_next_valid = _decision_continuity(capture.tick)
    control_next = _shift(capture.control)
    control_mask_next = _shift(capture.control_mask)
    native_control_next = _shift(capture.native_control)
    attack_active_next = _shift(capture.attack_action_active)
    body_next = _shift(capture.body_instruction)
    body_path_next = _shift(capture.body_decision_path)
    head_next = _shift(capture.head_instruction)
    head_path_next = _shift(capture.head_decision_path)
    active_actions_next = _shift(capture.active_actions)
    active_action_paths_next = _shift(capture.active_action_paths)
    decision_internal_next = _shift(capture.decision_internal_state)
    decision_target_next = _shift(capture.decision_target_uuid)
    phase = _combat_phase(
        capture.attack_actions,
        capture.interactions,
        capture.attack_pause_seconds,
        capture.combat_attack,
    )
    post_phase = _combat_phase(
        capture.next_attack_actions,
        capture.next_interactions,
        capture.next_attack_pause_seconds,
        capture.next_combat_attack,
    )
    decision_changes = _decision_changes(
        capture,
        phase,
        control_next,
        control_mask_next,
        body_next,
        head_next,
        active_actions_next,
        decision_internal_next,
    )
    components = (
        numeric_component(
            "native.actor_state",
            capture.state,
            next=capture.next_state,
            kind="state",
            layout=NATIVE_NPC_TRACE_STATE_LAYOUT.split(","),
            units="blocks,radians,health",
            layer="transition",
        ),
        numeric_component(
            "native.actor_view_pose",
            view_pose,
            next=next_view_pose,
            kind="observation",
            layout=("origin_x", "origin_y", "origin_z", "head_yaw", "head_pitch"),
            valid=view_pose_valid,
            next_valid=next_view_pose_valid,
            units="blocks,radians",
            layer="internal",
            description=(
                "Native LOS origin plus raw head rotation; camera convention/FOV "
                "remain calibration-owned."
            ),
        ),
        numeric_component(
            "native.control",
            capture.control,
            next=control_next,
            kind="action",
            layout=NATIVE_NPC_TRACE_CONTROL_LAYOUT.split(","),
            next_valid=decision_next_valid,
            layer="decision",
            description="Selected control; .next is the next contiguous NPC decision.",
        ),
        numeric_component(
            "native.next_decision_valid",
            decision_next_valid,
            kind="mask",
            layer="decision",
            description="A next contiguous native decision exists for this row.",
        ),
        numeric_component(
            "native.control_mask",
            capture.control_mask,
            next=control_mask_next,
            kind="mask",
            next_valid=decision_next_valid,
            layer="decision",
        ),
        numeric_component(
            "native.control_owned",
            capture.native_control,
            next=native_control_next,
            kind="context",
            next_valid=decision_next_valid,
            layer="decision",
        ),
        numeric_component(
            "native.attack_action_active",
            capture.attack_action_active,
            next=attack_active_next,
            kind="action",
            next_valid=decision_next_valid,
            layer="decision",
        ),
        numeric_component(
            "native.attack_activation",
            capture.attack_activation,
            kind="event",
            layer="decision",
        ),
        structured_component(
            "native.attack_execution_cause",
            capture.attack_execution_cause,
            dtype="hytale.attack_execution_cause",
            kind="action",
            valid=np.asarray(
                [cause.available for cause in capture.attack_execution_cause]
            ),
            layer="decision",
            description="Exact authored combat-chain start; unavailable rows abstain.",
        ),
        numeric_component(
            "native.time_to_attack_activation",
            time_to_attack,
            kind="context",
            valid=time_to_attack_valid,
            units="seconds",
            layer="decision",
        ),
        numeric_component(
            "native.combat_attack",
            capture.combat_attack,
            next=capture.next_combat_attack,
            kind="state",
            layer="internal",
        ),
        numeric_component(
            "native.combat.attack_pause_seconds",
            capture.attack_pause_seconds,
            next=capture.next_attack_pause_seconds,
            kind="state",
            units="seconds",
            layer="internal",
        ),
        structured_component(
            "native.combat.attack_actions",
            capture.attack_actions,
            next=capture.next_attack_actions,
            dtype="hytale.action_attack[]",
            kind="state",
            layer="decision",
        ),
        structured_component(
            "native.combat.phase",
            phase,
            next=post_phase,
            dtype="idle|windup|aiming|interaction|completion|recovery",
            kind="state",
            layer="internal",
        ),
        structured_component(
            "native.target_uuid",
            capture.target_uuid,
            next=capture.next_target_uuid,
            dtype="uuid?",
            kind="state",
            layer="external",
        ),
        structured_component(
            "native.decision_target_uuid",
            capture.decision_target_uuid,
            next=decision_target_next,
            dtype="uuid?",
            kind="action",
            next_valid=decision_next_valid,
            layer="decision",
            description="Locked target after selection; .next is the next decision.",
        ),
        structured_component(
            "native.behavior_state",
            capture.state_name,
            next=capture.next_state_name,
            dtype="utf8",
            kind="state",
            layer="internal",
        ),
        structured_component(
            "native.body_instruction",
            capture.body_instruction,
            next=body_next,
            dtype="utf8",
            kind="action",
            next_valid=decision_next_valid,
            layer="decision",
        ),
        structured_component(
            "native.body_decision_path",
            capture.body_decision_path,
            next=body_path_next,
            dtype="utf8",
            kind="action",
            next_valid=decision_next_valid,
            layer="decision",
        ),
        structured_component(
            "native.head_instruction",
            capture.head_instruction,
            next=head_next,
            dtype="utf8",
            kind="action",
            next_valid=decision_next_valid,
            layer="decision",
        ),
        structured_component(
            "native.head_decision_path",
            capture.head_decision_path,
            next=head_path_next,
            dtype="utf8",
            kind="action",
            next_valid=decision_next_valid,
            layer="decision",
        ),
        structured_component(
            "native.active_actions",
            capture.active_actions,
            next=active_actions_next,
            dtype="utf8[]",
            kind="action",
            next_valid=decision_next_valid,
            layer="decision",
        ),
        structured_component(
            "native.active_action_paths",
            capture.active_action_paths,
            next=active_action_paths_next,
            dtype="utf8[]",
            kind="action",
            next_valid=decision_next_valid,
            layer="decision",
        ),
        structured_component(
            "native.interactions",
            capture.interactions,
            next=capture.next_interactions,
            dtype="hytale.interaction_chain[]",
            kind="state",
            layer="internal",
        ),
        structured_component(
            "native.damage_events",
            capture.damage_events,
            dtype="hytale.native_damage_event_v1[]",
            kind="event",
            valid=~capture.damage_event_overflow,
            layer="omniscient",
            description=(
                "Post-filter Java damage admission and outcome; overflow rows "
                "remain explicit partial evidence."
            ),
        ),
        structured_component(
            "native.lifecycle_events",
            capture.lifecycle_events,
            dtype="hytale.native_lifecycle_event_v1[]",
            kind="event",
            valid=~capture.lifecycle_event_overflow,
            layer="omniscient",
            description=(
                "Server inventory events plus state-boundary stat, status, "
                "projectile, and actor lifecycle facts."
            ),
        ),
        numeric_component(
            "native.lifecycle_event_count",
            capture.lifecycle_event_count,
            kind="mask",
            layer="omniscient",
        ),
        numeric_component(
            "native.lifecycle_event_overflow",
            capture.lifecycle_event_overflow,
            kind="mask",
            layer="omniscient",
        ),
        numeric_component(
            "native.lifecycle_source_available",
            capture.lifecycle_source_available,
            kind="mask",
            layer="omniscient",
        ),
        numeric_component(
            "native.lifecycle_source_partial",
            capture.lifecycle_source_partial,
            kind="mask",
            layer="omniscient",
        ),
        numeric_component(
            "native.damage_event_count",
            capture.damage_event_count,
            kind="mask",
            layer="omniscient",
        ),
        numeric_component(
            "native.damage_event_overflow",
            capture.damage_event_overflow,
            kind="mask",
            layer="omniscient",
        ),
        structured_component(
            "native.actor_observation",
            capture.actor_evidence,
            next=capture.next_actor_evidence,
            dtype="hytale.native_actor_evidence_v4",
            kind="observation",
            layer="internal",
        ),
        structured_component(
            "native.action_capability",
            tuple(row.action_capability_evidence for row in capture.observation),
            next=tuple(
                row.action_capability_evidence for row in capture.next_observation
            ),
            dtype="hytale.native_actor_action_capability_v4",
            kind="mask",
            valid=_action_capability_available(capture.observation),
            next_valid=_action_capability_available(capture.next_observation),
            layer="internal",
            description=(
                "Reset-bound native action inputs captured at the exact "
                "observation boundary."
            ),
        ),
        *_observation_components(capture),
        structured_component(
            "native.internal_state",
            capture.internal_state,
            next=capture.next_internal_state,
            dtype="hytale.npc_internal_state_v1",
            kind="state",
            layer="internal",
            description="Role support before behavior and after the native tick.",
        ),
        structured_component(
            "native.decision_internal_state",
            capture.decision_internal_state,
            next=decision_internal_next,
            dtype="hytale.npc_internal_state_v1",
            kind="action",
            next_valid=decision_next_valid,
            layer="decision",
            description="Role support after selection; .next is the next decision.",
        ),
        structured_component(
            "native.target_actor_observation",
            capture.target_actor_evidence,
            next=capture.next_target_actor_evidence,
            dtype="hytale.native_actor_evidence_v4",
            kind="observation",
            valid=_actor_present(capture.target_actor_evidence),
            next_valid=_actor_present(capture.next_target_actor_evidence),
            layer="external",
        ),
        numeric_component(
            "spatial.target_relative",
            _evidence_relative(capture.actor_evidence, capture.target_actor_evidence),
            next=_evidence_relative(
                capture.next_actor_evidence, capture.next_target_actor_evidence
            ),
            kind="observation",
            layout=_RELATIVE_LAYOUT,
            valid=_actor_present(capture.target_actor_evidence),
            next_valid=_actor_present(capture.next_target_actor_evidence),
            layer="external",
            description="Exact locked-target relation from authoritative actor evidence.",
        ),
        structured_component(
            "native.worldview",
            capture.worldview,
            next=capture.next_worldview,
            dtype=NATIVE_NPC_WORLDVIEW_SCHEMA,
            kind="observation",
            valid=np.asarray(
                [
                    not row.overflow and not row.entity_overflow
                    for row in capture.worldview
                ]
            ),
            next_valid=np.asarray(
                [
                    not row.overflow and not row.entity_overflow
                    for row in capture.next_worldview
                ]
            ),
            layer="omniscient",
            description="Bounded live-NPC and transform-entity worldview.",
        ),
        structured_component(
            "native.world_entities",
            tuple(row.entities for row in capture.worldview),
            next=tuple(row.entities for row in capture.next_worldview),
            dtype="hytale.world_entity_v1[]",
            kind="observation",
            valid=np.asarray([not row.entity_overflow for row in capture.worldview]),
            next_valid=np.asarray(
                [not row.entity_overflow for row in capture.next_worldview]
            ),
            layer="omniscient",
            description="Every bounded transform-bearing ECS entity, index-sorted.",
        ),
        numeric_component(
            "native.world_clock",
            _world_clock(capture.worldview),
            next=_world_clock(capture.next_worldview),
            kind="state",
            layout=(
                "world_tick",
                "game_time_epoch_second",
                "game_time_nano",
                "day_progress",
                "sunlight_factor",
                "moon_phase",
                "npc_count",
            ),
            layer="omniscient",
        ),
        numeric_component(
            "native.decision_change",
            decision_changes,
            kind="event",
            layout=_DECISION_CHANGE_LAYOUT,
            valid=decision_next_valid,
            layer="decision",
            description="Exact changes from this intent to the next contiguous intent.",
        ),
        numeric_component(
            "native.decision_support_change",
            _support_changes(capture),
            kind="event",
            layout=_DECISION_SUPPORT_CHANGE_LAYOUT,
            layer="decision",
            description="Role-support mutations between observation and selection.",
        ),
        *_actor_components(capture),
        *extensions,
    )
    symbols = _symbols(capture)
    provenance = {
        "native.angle_units": "radians",
        "native.action_contract_schema": capture.action_contract_schema,
        "native.action_contract_version": capture.action_contract_version,
        "native.action_contract_sha256": capture.action_contract_sha256,
        "native.event_contract_schema": capture.event_contract_schema,
        "native.event_contract_version": capture.event_contract_version,
        "native.event_contract_sha256": capture.event_contract_sha256,
        "native.lifecycle_contract_schema": capture.lifecycle_contract_schema,
        "native.lifecycle_contract_version": capture.lifecycle_contract_version,
        "native.lifecycle_contract_sha256": capture.lifecycle_contract_sha256,
        "native.lifecycle_event_layout": NATIVE_NPC_LIFECYCLE_EVENT_LAYOUT,
        "native.damage_event_layout": NATIVE_NPC_DAMAGE_EVENT_LAYOUT,
        "native.event_scope": (
            "server_damage_after_filter_and_apply;server_inventory_events;"
            "boundary_stat_status_projectile_actor_lifecycle"
        ),
        "native.boundary_semantics": {
            "state.current": "after_pre_behavior_support_before_behavior",
            "decision.current": "after_behavior_and_avoidance_before_bridge_and_steering",
            "state.next": "after_native_tick",
            "decision.next": "next_contiguous_decision;see_component_next_valid",
        },
        "native.episode_structure": {
            "episode_id": "unavailable;trace_uuid_identifies_only_a_capture_segment",
            "episode_start": "unavailable;capture_may_begin_after_warmup",
            "terminated": "unavailable;task_terminal_semantics_are_not_captured",
            "truncated": "unavailable;capture_stop_is_not_a_task_truncation",
        },
        "native.bridge_sha256": capture.bridge_sha256,
        "native.capacity": capture.capacity,
        "native.control_layout": NATIVE_NPC_TRACE_CONTROL_LAYOUT,
        "native.control_mask_layout": NATIVE_NPC_TRACE_CONTROL_MASK_LAYOUT,
        "native.emitted_count": capture.emitted_count,
        "native.environment_parameters": dict(capture.environment_parameters),
        "native.actor_evidence_schema": capture.actor_evidence_schema,
        "native.actor_evidence_version": capture.actor_evidence_version,
        "native.actor_evidence_contract_sha256": (
            capture.actor_evidence_contract_sha256
        ),
        "native.action_capability_schema": capture.actor_evidence_schema,
        "native.action_capability_version": capture.actor_evidence_version,
        "native.action_capability_contract_sha256": (
            capture.actor_evidence_contract_sha256
        ),
        "native.action_capability_boundary": ("same_as_native_semantic_observation"),
        "native.attack_action_layout": NATIVE_NPC_TRACE_ATTACK_ACTION_LAYOUT,
        "native.interaction_layout": NATIVE_NPC_TRACE_INTERACTION_LAYOUT,
        "native.internal_layout": NATIVE_NPC_TRACE_INTERNAL_LAYOUT,
        "native.worldview_schema": NATIVE_NPC_WORLDVIEW_SCHEMA,
        "native.worldview_version": NATIVE_NPC_WORLDVIEW_VERSION,
        "native.worldview_actor_layout": NATIVE_NPC_WORLDVIEW_ACTOR_LAYOUT,
        "native.worldview_entity_capacity": NATIVE_NPC_WORLDVIEW_ENTITY_CAPACITY,
        "native.worldview_entity_layout": NATIVE_NPC_WORLDVIEW_ENTITY_LAYOUT,
        "native.worldview_scope": (
            "all_live_npcs_world_clock_and_transform_bearing_entities"
        ),
        "native.semantic_observation_schema": capture.observation_schema,
        "native.semantic_observation_version": capture.observation_version,
        "native.semantic_observation_group_layout": (
            NATIVE_NPC_OBSERVATION_GROUP_LAYOUT
        ),
        "native.semantic_observation_contract_sha256": (
            capture.observation_contract_sha256
        ),
        "native.semantic_observation_present": True,
        "native.npc_uuid": str(capture.npc_uuid),
        "native.role": capture.role,
        "native.schema": NATIVE_NPC_TRACE_SCHEMA,
        "native.seed": capture.seed,
        "native.server_version": capture.server_version,
        "native.state_layout": NATIVE_NPC_TRACE_STATE_LAYOUT,
        "native.status": capture.status,
        "native.total_captured": capture.total_captured,
        "native.trace_uuid": str(capture.trace_uuid),
        "native.version": NATIVE_NPC_TRACE_VERSION,
        "native.world": capture.world,
        "native.worldgen_provider": capture.worldgen_provider,
        "native.worldgen_version": capture.worldgen_version,
        "native.learner_observation_schema": LEARNER_OBSERVATION_V3_SCHEMA,
        "native.learner_observation_version": LEARNER_OBSERVATION_V3_VERSION,
        "native.learner_observation_contract_sha256": _LEARNER_OBSERVATION_SHA256,
        "native.learner_observation_present": False,
        "native.learner_observation_absence_reason": (
            "dense_policy_row_not_captured;structured_semantic_evidence_available"
        ),
        "native.decision_candidate_scope": (
            "complete_bounded_role_actionattack_tree_plus_selected_instructions"
        ),
        "native.decision_candidate_complete": False,
        "native.attack_candidate_scope_complete": bool(
            all(not row.attack_candidate_overflow for row in capture.observation)
            and all(
                not row.attack_candidate_overflow for row in capture.next_observation
            )
        ),
        "native.missing_decision_inputs": (
            "behavior_tree_alternative_scores",
            "sensor_predicate_outcomes",
            "non_attack_behavior_tree_candidates",
            "learner_ability_affordable",
            "learner_ability_requirement_bits",
        ),
        "native.missing_observation_groups": (
            "normalized_status_features",
            "audio",
            "actor_visible_non_npc_non_projectile_entity_perception",
        ),
        "native.observation_coverage": {
            "movement_state": "23/23 with availability",
            "defense_inputs": "7/7 with availability; raw native units",
            "actor_world_inputs": "5 values/3 masks; unavailable fields masked",
            "resources": "7 values/7 maxima/7 availability",
            "weapon": "item id and diagnostic runtime index",
            "status": (
                "raw effect IDs/durations/flags plus overflow/invalid bits; "
                "learner normalization remains host-owned"
            ),
            "ability": (
                "full bounded role ActionAttack candidates, readiness, charge, "
                "lifecycle and active slot; affordability predicates unavailable"
            ),
            "target": (
                "privileged state plus separate policy perceptibility, exact "
                "relative pose, LOS and visible-NPC/entity membership"
            ),
            "perceived_entities": (
                "complete bounded native-LOS NPC/projectile membership joined "
                "to same-boundary worldview transforms; overflow validity-masked"
            ),
            "actor_view_pose": (
                "native LOS origin and raw head angles; projection convention "
                "and FOV require paired calibration"
            ),
            "worldview": (
                "all live NPCs, world clock, and bounded transform-bearing ECS "
                "entities with projectile/physics flags; overflow validity-masked"
            ),
            "geometry": (
                "exact local cells, fluids, movement semantics, collision boxes, "
                "contacts, bounds and LOS"
            ),
            "role_opacity": (
                "authored NPC-role opacity for every geometry cell with an "
                "independent availability mask"
            ),
            "light_environment": (
                "tick-aligned heightmap, sky light, RGB block light, environment "
                "code, and tint with per-sample/per-channel validity"
            ),
            "inventory": (
                "six native containers, capacities, active slots, quantities, "
                "durability and per-container validity"
            ),
            "decision": "instruction paths, actions, control, internal state, and changes",
            "decision_candidates": (
                "selected instruction rows plus the role's bounded complete "
                "ActionAttack tree for self and target"
            ),
        },
        "native.trace_layers": _layer_manifest(components, capture),
        "native.symbol_table_version": 1,
        "native.symbol_table": symbols,
        "native.symbol_table_sha256": _symbol_sha256(symbols),
    }
    overlap = set(provenance) & set(parameters or {})
    if overlap:
        raise ValueError(
            f"extension parameters overwrite native provenance: {sorted(overlap)}"
        )
    provenance.update(parameters or {})
    count = capture.emitted_count
    unavailable = np.zeros(count, dtype=np.bool_)
    return TraceSequence(
        contract=TraceContract(
            tuple(component.spec for component in components), tuple(provenance)
        ),
        tick=capture.tick,
        delta_seconds=capture.delta_seconds,
        episode_id=np.zeros(count, dtype=np.int64),
        episode_start=unavailable,
        terminated=np.zeros(count, dtype=np.bool_),
        truncated=np.zeros(count, dtype=np.bool_),
        episode_id_available=unavailable,
        episode_start_available=unavailable,
        terminated_available=unavailable,
        truncated_available=unavailable,
        components={component.spec.name: component for component in components},
        parameters=provenance,
    )


def native_duel_trace_sequences(
    captures: Sequence[NativeNpcTraceCapture], *, spatial: bool = True
) -> tuple[TraceSequence, TraceSequence]:
    """Build synchronized actor traces, optionally adding opponent-relative state."""

    pair = tuple(captures)
    if len(pair) != 2 or not all(
        isinstance(capture, NativeNpcTraceCapture) for capture in pair
    ):
        raise TypeError("captures must contain exactly two native NPC traces")
    if pair[0].npc_uuid == pair[1].npc_uuid or _capture_world(pair[0]) != (
        _capture_world(pair[1])
    ):
        raise ValueError("duel traces must identify two NPCs from one world reset")
    if not np.array_equal(pair[0].tick, pair[1].tick):
        raise ValueError("duel traces must contain the same native-control ticks")
    if not spatial:
        return tuple(native_trace_sequence(capture) for capture in pair)
    return tuple(
        native_trace_sequence(
            pair[index],
            _opponent_relative(pair[index], pair[1 - index]),
            parameters={"spatial.opponent_uuid": str(pair[1 - index].npc_uuid)},
        )
        for index in range(2)
    )


def native_concurrent_trace_sequences(
    captures: Sequence[NativeNpcTraceCapture], *, spatial: bool = True
) -> tuple[TraceSequence, TraceSequence]:
    """Build actor-local traces from UUID-aligned Java world snapshots.

    Native roles can receive control callbacks on different ticks. Each actor
    keeps its own decision rows, while missing opponent context stays masked.
    """

    pair = tuple(captures)
    if len(pair) != 2 or not all(
        isinstance(capture, NativeNpcTraceCapture) for capture in pair
    ):
        raise TypeError("captures must contain exactly two native NPC traces")
    if pair[0].npc_uuid == pair[1].npc_uuid or _capture_world(pair[0]) != (
        _capture_world(pair[1])
    ):
        raise ValueError(
            "concurrent traces must identify two NPCs from one world reset"
        )
    if not spatial:
        return tuple(native_trace_sequence(capture) for capture in pair)
    return tuple(
        native_trace_sequence(
            pair[index],
            _worldview_opponent_relative(pair[index], pair[1 - index].npc_uuid),
            parameters={
                "spatial.opponent_uuid": str(pair[1 - index].npc_uuid),
                "spatial.source": "native.worldview.uuid",
            },
        )
        for index in range(2)
    )


def native_trace_corpus(
    captures: Sequence[NativeNpcTraceCapture],
) -> TraceCorpus:
    """Group captures without concatenating rows or mixing engine contracts."""

    return TraceCorpus(
        tuple(native_trace_sequence(capture) for capture in captures),
        NATIVE_CORPUS_COMPATIBILITY_KEYS,
    )


def attach_native_hytale_action_capabilities(
    trace: TraceSequence,
    capture: NativeNpcTraceCapture,
    *,
    agent_profile: str,
    target_profile: str = "",
) -> TraceSequence:
    """Attach same-boundary learner-v3 factor masks from native trace facts."""

    if NATIVE_HYTALE_ACTION_MASK in trace.components:
        raise ValueError("trace already contains a Hytale action mask")
    if not isinstance(capture, NativeNpcTraceCapture):
        raise TypeError("capture must be NativeNpcTraceCapture")
    if (
        trace.rows != capture.emitted_count
        or not np.array_equal(trace.tick, capture.tick)
        or trace.parameters.get("native.trace_uuid") != str(capture.trace_uuid)
        or trace.parameters.get("native.bridge_sha256") != capture.bridge_sha256
        or trace.parameters.get("native.semantic_observation_contract_sha256")
        != capture.observation_contract_sha256
    ):
        raise ValueError("native action capabilities do not belong to this trace")
    projected = project_native_npc_action_capabilities(
        capture,
        agent_profile=agent_profile,
        target_profile=target_profile,
    )
    return trace.compose(
        numeric_component(
            NATIVE_HYTALE_ACTION_MASK,
            projected.action_mask,
            next=projected.next_action_mask,
            kind="mask",
            valid=projected.valid,
            next_valid=projected.next_valid,
            layer="external",
            description=(
                "Same-boundary learner-v3 factor legality reconstructed from "
                "native rule, cooldown, charge, and actor evidence."
            ),
        ),
        parameters={
            "hytale.action_mask_schema": "hytale-arsenal-factor-mask-v1",
            "hytale.action_mask_policy_sha256": arsenal_policy_contract_sha256(),
            "hytale.action_capability_schema": NATIVE_NPC_ACTION_CAPABILITY_SCHEMA,
            "hytale.action_capability_version": NATIVE_NPC_ACTION_CAPABILITY_VERSION,
            "hytale.action_capability_sha256": NATIVE_NPC_ACTION_CAPABILITY_SHA256,
            "hytale.action_capability_agent_profile": agent_profile,
            "hytale.action_capability_target_profile": target_profile,
            "hytale.action_capability_unavailable_reason": (
                projected.unavailable_reason
            ),
            "hytale.action_capability_next_unavailable_reason": (
                projected.next_unavailable_reason
            ),
        },
    )


def install_native_lifecycle_episodes(
    trace: TraceSequence,
    *,
    reset_boundary: bool,
    task_terminated: Any,
    task_truncated: Any,
    task_terminated_available: Any,
    task_truncated_available: Any,
    initial_episode_id: int | None = None,
    name: str = NATIVE_EPISODE_PROJECTOR_NAME,
    version: int = NATIVE_EPISODE_PROJECTOR_VERSION,
    evidence: Sequence[str] | None = None,
    derivation: str = (
        "task end, in-transition actor death, or in-transition revival reset "
        "ends; revival or capture reset starts; inter-row death cuts identity"
    ),
) -> TraceSequence:
    """Fuse task outcomes with complete Java actor-lifecycle evidence.

    Task outcomes are mandatory because actor survival does not prove that a
    task continued. Unknown or partial evidence remains unavailable.
    """

    if not isinstance(reset_boundary, bool):
        raise TypeError("reset_boundary must be bool")
    if trace.parameters.get("native.lifecycle_contract_sha256") != (
        NATIVE_NPC_LIFECYCLE_CONTRACT_SHA256
    ):
        raise ValueError("trace has no supported native lifecycle contract")
    events = trace.components.get("native.lifecycle_events")
    source = trace.components.get("native.lifecycle_source_available")
    partial = trace.components.get("native.lifecycle_source_partial")
    if (
        events is None
        or events.spec.kind != "event"
        or events.spec.dtype != "hytale.native_lifecycle_event_v1[]"
        or source is None
        or source.spec.kind != "mask"
        or source.spec.dtype != "int32"
        or source.spec.shape != ()
        or partial is None
        or partial.spec.kind != "mask"
        or partial.spec.dtype != "int32"
        or partial.spec.shape != ()
    ):
        raise ValueError("trace has malformed native lifecycle components")

    def boolean_rows(value: Any, field: str) -> np.ndarray:
        rows = np.asarray(value)
        if rows.dtype != np.bool_ or rows.shape != (trace.rows,):
            raise ValueError(f"{field} must be bool with shape [{trace.rows}]")
        return rows

    task_end = boolean_rows(task_terminated, "task_terminated")
    task_cut = boolean_rows(task_truncated, "task_truncated")
    task_end_known = boolean_rows(
        task_terminated_available, "task_terminated_available"
    )
    task_cut_known = boolean_rows(task_truncated_available, "task_truncated_available")
    if np.any(task_end & ~task_end_known) or np.any(task_cut & ~task_cut_known):
        raise ValueError("unknown task episode outcomes must use false placeholders")

    source_values = np.asarray(source.current, dtype=np.int32)
    partial_values = np.asarray(partial.current, dtype=np.int32)
    actor_source = NATIVE_NPC_LIFECYCLE_SOURCE_ACTOR_LIFECYCLE
    lifecycle_known = (
        events.validity()
        & source.validity()
        & partial.validity()
        & ((source_values & actor_source) != 0)
        & ((partial_values & actor_source) == 0)
    )
    starts = np.zeros(trace.rows, dtype=np.bool_)
    starts_known = lifecycle_known.copy()
    actor_death = np.zeros(trace.rows, dtype=np.bool_)
    actor_death_known = lifecycle_known.copy()
    actor_reset = np.zeros(trace.rows, dtype=np.bool_)
    actor_reset_known = lifecycle_known.copy()
    if trace.rows and reset_boundary:
        starts[0] = starts_known[0] = True
    elif trace.rows and initial_episode_id is not None:
        starts_known[0] = True

    for index, row in enumerate(events.current):
        if not lifecycle_known[index]:
            continue
        deaths = tuple(event for event in row if event.kind == "actor_death")
        revivals = tuple(event for event in row if event.kind == "actor_revived")
        actor_events = deaths + revivals
        contradictory = (
            len(deaths) > 1
            or len(revivals) > 1
            or (deaths and revivals and deaths[0].origin == revivals[0].origin)
            or any(
                event.origin
                not in {"server_death_component", "server_inter_row_state_delta"}
                for event in actor_events
            )
        )
        if contradictory:
            if any(
                event.origin == "server_inter_row_state_delta" for event in actor_events
            ):
                starts[index] = False
                starts_known[index] = False
            actor_death_known[index] = False
            actor_reset_known[index] = False
            if index + 1 < trace.rows and any(
                event.kind == "actor_revived"
                and event.origin == "server_death_component"
                for event in actor_events
            ):
                starts[index + 1] = False
                starts_known[index + 1] = False
            continue
        if deaths:
            origin = deaths[0].origin
            if origin == "server_death_component":
                actor_death[index] = True
            elif origin == "server_inter_row_state_delta":
                starts_known[index] = False
        if revivals:
            origin = revivals[0].origin
            if origin == "server_inter_row_state_delta":
                starts[index] = True
            elif origin == "server_death_component":
                actor_reset[index] = True
                if index + 1 < trace.rows:
                    starts[index + 1] = bool(lifecycle_known[index + 1])
                    starts_known[index + 1] = bool(lifecycle_known[index + 1])

    terminated = task_end | actor_death | actor_reset
    terminated_known = terminated | (
        task_end_known & actor_death_known & actor_reset_known
    )
    truncated = task_cut & ~terminated
    truncated_known = task_cut_known | terminated
    return install_episode_evidence(
        trace,
        name=name,
        version=version,
        episode_start=starts,
        terminated=terminated,
        truncated=truncated,
        episode_start_available=starts_known,
        terminated_available=terminated_known,
        truncated_available=truncated_known,
        evidence=(
            (
                "task.terminated",
                "task.truncated",
                "native.lifecycle_events.actor_death",
                "native.lifecycle_events.actor_revived",
                "native.lifecycle_source_available.actor_lifecycle",
                "native.lifecycle_source_partial.actor_lifecycle",
                "capture.reset_boundary",
            )
            if evidence is None
            else evidence
        ),
        derivation=derivation,
        initial_episode_id=initial_episode_id,
    )


def terrain_component(
    current: Any,
    next: Any,
    *,
    name: str = "terrain.local",
    layout: Sequence[str] = (),
    valid: Any = None,
) -> TraceComponent:
    """Attach any aligned terrain encoding without coupling Arena to its sampler."""

    if not name.startswith("terrain."):
        raise ValueError("terrain feature names must use the terrain namespace")
    return numeric_component(
        name,
        current,
        next=next,
        kind="observation",
        layout=layout,
        valid=valid,
        layer="extension",
        description="Caller-owned terrain encoding aligned to native engine ticks.",
    )


def _opponent_relative(
    actor: NativeNpcTraceCapture, opponent: NativeNpcTraceCapture
) -> TraceComponent:
    return numeric_component(
        "spatial.opponent_relative",
        _relative(actor.state, opponent.state),
        next=_relative(actor.next_state, opponent.next_state),
        kind="observation",
        layout=(
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_vx",
            "delta_vy",
            "delta_vz",
            "distance",
            "planar_distance",
            "bearing_sin",
            "bearing_cos",
            "opponent_health_fraction",
        ),
        description="Exact opponent-relative kinematics derived from paired Java states.",
        layer="external",
    )


def _worldview_opponent_relative(
    actor: NativeNpcTraceCapture, opponent_uuid: UUID
) -> TraceComponent:
    current, current_valid = _world_actor_states(actor.worldview, opponent_uuid)
    following, following_valid = _world_actor_states(
        actor.next_worldview, opponent_uuid
    )
    return numeric_component(
        "spatial.opponent_relative",
        _relative(actor.state, current),
        next=_relative(actor.next_state, following),
        kind="observation",
        layout=(
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_vx",
            "delta_vy",
            "delta_vz",
            "distance",
            "planar_distance",
            "bearing_sin",
            "bearing_cos",
            "opponent_health_fraction",
        ),
        valid=current_valid,
        next_valid=following_valid,
        description=(
            "Opponent-relative kinematics from the actor-boundary Java worldview."
        ),
        layer="external",
    )


def _world_actor_states(worldviews, opponent_uuid):
    states = np.zeros((len(worldviews), 14), dtype=np.float64)
    valid = np.zeros(len(worldviews), dtype=np.bool_)
    for index, worldview in enumerate(worldviews):
        matches = [row for row in worldview.actors if row.uuid == opponent_uuid]
        if len(matches) != 1 or len(matches[0].state) != states.shape[1]:
            continue
        states[index] = matches[0].state
        valid[index] = True
    return states, valid


def _observation_components(
    capture: NativeNpcTraceCapture,
) -> tuple[TraceComponent, ...]:
    current, following = capture.observation, capture.next_observation
    return (
        structured_component(
            "native.semantic_observation",
            current,
            next=following,
            dtype=capture.observation_schema,
            kind="observation",
            layer="external",
            description="Java semantic evidence with per-family fidelity state.",
        ),
        structured_component(
            "native.geometry",
            tuple(row.geometry for row in current),
            next=tuple(row.geometry for row in following),
            dtype="hytale_geometry_v5",
            kind="observation",
            valid=np.asarray([bool(row.geometry["available"]) for row in current]),
            next_valid=np.asarray(
                [bool(row.geometry["available"]) for row in following]
            ),
            layer="external",
        ),
        numeric_component(
            "native.role_opaque_cell_mask",
            np.stack([row.role_opaque_cell_mask for row in current]),
            next=np.stack([row.role_opaque_cell_mask for row in following]),
            kind="observation",
            valid=np.asarray([row.role_opaque_cell_mask_available for row in current]),
            next_valid=np.asarray(
                [row.role_opaque_cell_mask_available for row in following]
            ),
            layer="external",
            description="Role-authored opacity aligned to geometry cell order.",
        ),
        structured_component(
            "native.local_perception_channels",
            tuple(row.local_perception for row in current),
            next=tuple(row.local_perception for row in following),
            dtype="hytalerl_native_perception_channels_v1",
            kind="observation",
            layer="external",
            description=(
                "Geometry-aligned light, environment, tint, and heightmap with "
                "per-channel validity."
            ),
        ),
        structured_component(
            "native.inventory",
            tuple(row.inventory for row in current),
            next=tuple(row.inventory for row in following),
            dtype="hytalerl_native_inventory_v2",
            kind="observation",
            valid=np.asarray([bool(row.inventory["available"]) for row in current]),
            next_valid=np.asarray(
                [bool(row.inventory["available"]) for row in following]
            ),
            layer="internal",
        ),
        structured_component(
            "native.attack_candidates",
            tuple(row.attack_candidates for row in current),
            next=tuple(row.attack_candidates for row in following),
            dtype="hytale.action_attack[]",
            kind="observation",
            valid=np.asarray([not row.attack_candidate_overflow for row in current]),
            next_valid=np.asarray(
                [not row.attack_candidate_overflow for row in following]
            ),
            layer="decision",
        ),
        structured_component(
            "native.target_attack_candidates",
            tuple(row.target_attack_candidates for row in current),
            next=tuple(row.target_attack_candidates for row in following),
            dtype="hytale.action_attack[]",
            kind="observation",
            valid=np.asarray(
                [
                    row.target_present and not row.target_attack_candidate_overflow
                    for row in current
                ]
            ),
            next_valid=np.asarray(
                [
                    row.target_present and not row.target_attack_candidate_overflow
                    for row in following
                ]
            ),
            layer="external",
        ),
        structured_component(
            "native.perceptible_npc_uuids",
            tuple(row.perceptible_npc_uuids for row in current),
            next=tuple(row.perceptible_npc_uuids for row in following),
            dtype="uuid[]",
            kind="observation",
            valid=np.asarray(
                [
                    row.perception_available and not row.perceptible_npc_overflow
                    for row in current
                ]
            ),
            next_valid=np.asarray(
                [
                    row.perception_available and not row.perceptible_npc_overflow
                    for row in following
                ]
            ),
            layer="external",
        ),
        _perceived_entity_component(capture),
        numeric_component(
            "native.observation_group_bits",
            _observation_group_bits(current),
            next=_observation_group_bits(following),
            kind="mask",
            layout=("complete", "partial", "unavailable"),
            layer="external",
        ),
        numeric_component(
            "native.target_perception",
            _target_perception(current),
            next=_target_perception(following),
            kind="observation",
            layout=("available", "present", "perceptible", "distance"),
            layer="external",
        ),
        numeric_component(
            "native.status_failure_bits",
            np.asarray([row.status_failure_bits for row in current], dtype=np.int32),
            next=np.asarray(
                [row.status_failure_bits for row in following], dtype=np.int32
            ),
            kind="mask",
            layer="internal",
        ),
    )


def _observation_group_bits(rows: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [
            (
                row.complete_group_bits,
                row.partial_group_bits,
                row.unavailable_group_bits,
            )
            for row in rows
        ],
        dtype=np.int64,
    ).reshape((len(rows), 3))


def _perceived_entity_component(capture: NativeNpcTraceCapture) -> TraceComponent:
    current, current_valid = _perceived_entities(capture.observation, capture.worldview)
    following, following_valid = _perceived_entities(
        capture.next_observation, capture.next_worldview
    )
    return structured_component(
        "native.perceived_entities",
        current,
        next=following,
        dtype="hytale.world_entity_v1[]",
        kind="observation",
        valid=current_valid,
        next_valid=following_valid,
        layer="external",
        description=(
            "Native-LOS NPC/projectile membership joined to exact same-boundary "
            "worldview entities."
        ),
    )


def _perceived_entities(observations, worldviews):
    rows = []
    valid = []
    for observation, worldview in zip(observations, worldviews, strict=True):
        by_index = {entity.entity_index: entity for entity in worldview.entities}
        selected = tuple(
            by_index[index]
            for index in observation.perceptible_entity_indices
            if index in by_index
        )
        rows.append(selected)
        valid.append(
            observation.perception_available
            and not observation.perceptible_entity_overflow
            and not worldview.entity_overflow
            and len(selected) == len(observation.perceptible_entity_indices)
        )
    return tuple(rows), np.asarray(valid, dtype=np.bool_)


def _actor_view_pose(states: np.ndarray, observations):
    values = np.zeros((len(states), 5), dtype=np.float64)
    valid = np.zeros(len(states), dtype=np.bool_)
    for index, (state, observation) in enumerate(
        zip(states, observations, strict=True)
    ):
        geometry = observation.geometry
        if "agent_los_offset" not in geometry:
            try:
                geometry = parse_geometry(dict(geometry))
            except (TypeError, ValueError):
                continue
        offset = geometry.get("agent_los_offset") if geometry.get("available") else None
        if offset is None:
            continue
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            continue
        values[index, :3] = state[:3] + offset
        values[index, 3:] = state[9:11]
        valid[index] = True
    return values, valid


def _target_perception(rows: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [
            (
                row.perception_available,
                row.target_present,
                row.target_perceptible,
                row.target_distance,
            )
            for row in rows
        ],
        dtype=np.float64,
    ).reshape((len(rows), 4))


def _actor_components(capture: NativeNpcTraceCapture) -> tuple[TraceComponent, ...]:
    current, following = capture.actor_evidence, capture.next_actor_evidence
    movement, movement_next = _movement(current), _movement(following)
    movement_mask, movement_mask_next = (
        _movement_mask(current),
        _movement_mask(following),
    )
    return (
        numeric_component(
            "native.movement_state",
            movement,
            next=movement_next,
            kind="state",
            layout=MOVEMENT_STATE_FEATURES,
            layer="internal",
        ),
        numeric_component(
            "native.movement_state_mask",
            movement_mask,
            next=movement_mask_next,
            kind="mask",
            layout=MOVEMENT_STATE_FEATURES,
            layer="internal",
        ),
        numeric_component(
            "native.resources",
            _join(current, "resource_values", "resource_maximums"),
            next=_join(following, "resource_values", "resource_maximums"),
            kind="state",
            layout=tuple(f"value_{index}" for index in range(7))
            + tuple(f"maximum_{index}" for index in range(7)),
            layer="internal",
        ),
        numeric_component(
            "native.resource_mask",
            _field(current, "resource_available", np.bool_, 7),
            next=_field(following, "resource_available", np.bool_, 7),
            kind="mask",
            layer="internal",
        ),
        numeric_component(
            "native.defense",
            # DERIVED, not restated. `DEFENSE_FLOAT_FEATURES` grew to 8 when
            # `locomotion_stamina_fraction` was added to the observation, and
            # these widths stayed at 7 -- so every TraceSpec here was built
            # with an 8-name layout over a 7-wide shape and `contract.py`
            # refused it, taking 19 of the 25 arena failures with it.
            _field(current, "defense_values", np.float64, DEFENSE_FLOAT_SIZE),
            next=_field(
                following, "defense_values", np.float64, DEFENSE_FLOAT_SIZE
            ),
            kind="state",
            layout=DEFENSE_FLOAT_FEATURES,
            layer="internal",
        ),
        numeric_component(
            "native.defense_mask",
            _field(current, "defense_available", np.bool_, DEFENSE_FLOAT_SIZE),
            next=_field(
                following, "defense_available", np.bool_, DEFENSE_FLOAT_SIZE
            ),
            kind="mask",
            layout=DEFENSE_FLOAT_FEATURES,
            layer="internal",
        ),
        numeric_component(
            "native.actor_world",
            _field(current, "actor_world_values", np.float64, 5),
            next=_field(following, "actor_world_values", np.float64, 5),
            kind="state",
            layout=ACTOR_WORLD_FLOAT_FEATURES,
            layer="internal",
        ),
        numeric_component(
            "native.actor_world_mask",
            _field(current, "actor_world_available", np.bool_, 3),
            next=_field(following, "actor_world_available", np.bool_, 3),
            kind="mask",
            layout=ACTOR_WORLD_MASK_FEATURES,
            layer="internal",
        ),
        numeric_component(
            "native.item_runtime_index",
            _integer(current, "item_runtime_index"),
            next=_integer(following, "item_runtime_index"),
            kind="state",
            description="Diagnostic server runtime index; item_id is stable evidence.",
            layer="internal",
        ),
        numeric_component(
            "native.active_ability_slot",
            _integer(current, "active_ability_slot"),
            next=_integer(following, "active_ability_slot"),
            kind="state",
            valid=np.asarray(
                [not row.attack_candidate_overflow for row in capture.observation]
            ),
            next_valid=np.asarray(
                [not row.attack_candidate_overflow for row in capture.next_observation]
            ),
            description="Role candidate index; -1 means no active attack action.",
            layer="internal",
        ),
        structured_component(
            "native.item_id",
            tuple(row["item_id"] for row in current),
            next=tuple(row["item_id"] for row in following),
            dtype="utf8",
            kind="state",
            layer="internal",
        ),
        structured_component(
            "native.statuses",
            tuple(row["statuses"] for row in current),
            next=tuple(row["statuses"] for row in following),
            dtype="hytale.status[]",
            kind="state",
            layer="internal",
        ),
        structured_component(
            "native.motion_force",
            tuple(row["motion_force"] for row in current),
            next=tuple(row["motion_force"] for row in following),
            dtype="hytale.motion_force",
            kind="state",
            layer="internal",
        ),
    )


def _field(
    rows: Sequence[Mapping[str, Any]], name: str, dtype: Any, width: int
) -> np.ndarray:
    if not rows:
        return np.empty((0, width), dtype=dtype)
    return np.asarray([row[name] for row in rows], dtype=dtype)


def _join(rows: Sequence[Mapping[str, Any]], *names: str) -> np.ndarray:
    return np.concatenate([_field(rows, name, np.float64, 7) for name in names], axis=1)


def _integer(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    return np.asarray([row[name] for row in rows], dtype=np.int32)


def _movement(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    bits = np.asarray([row["movement_states"]["bits"] for row in rows], dtype=np.int32)
    return (bits[:, None] & (1 << np.arange(23, dtype=np.int32))) != 0


def _movement_mask(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    available = np.asarray(
        [row["movement_states"]["available"] for row in rows], dtype=np.bool_
    )
    return np.broadcast_to(available[:, None], (len(rows), 23)).copy()


def _time_to_event(
    events: np.ndarray, delta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(events), dtype=np.float32)
    valid = np.zeros(len(events), dtype=np.bool_)
    elapsed = 0.0
    seen = False
    for index in range(len(events) - 1, -1, -1):
        if events[index]:
            elapsed, seen = 0.0, True
        elif seen:
            elapsed += float(delta[index])
        values[index], valid[index] = elapsed, seen
    return values, valid


def _decision_continuity(ticks: np.ndarray) -> np.ndarray:
    valid = np.zeros(len(ticks), dtype=np.bool_)
    if len(ticks) > 1:
        valid[:-1] = ticks[1:] == ticks[:-1] + 1
    return valid


def _shift(values: Any) -> Any:
    if isinstance(values, np.ndarray):
        following = values.copy()
        if len(values) > 1:
            following[:-1] = values[1:]
        return following
    rows = tuple(values)
    return rows[1:] + rows[-1:] if rows else ()


def _actor_present(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([row["present"] for row in rows], dtype=np.bool_)


def _action_capability_available(rows: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [row.action_capability_evidence["available"] for row in rows],
        dtype=np.bool_,
    )


def _evidence_relative(
    actor: Sequence[Mapping[str, Any]], target: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    if not actor:
        return np.empty((0, len(_RELATIVE_LAYOUT)), dtype=np.float64)
    actor_position = np.asarray([row["position"] for row in actor], dtype=np.float64)
    target_position = np.asarray([row["position"] for row in target], dtype=np.float64)
    actor_velocity = np.asarray([row["velocity"] for row in actor], dtype=np.float64)
    target_velocity = np.asarray([row["velocity"] for row in target], dtype=np.float64)
    delta = target_position - actor_position
    velocity = target_velocity - actor_velocity
    distance = np.linalg.norm(delta, axis=1)
    planar = np.hypot(delta[:, 0], delta[:, 2])
    yaw = np.deg2rad([row["yaw_degrees"] for row in actor])
    bearing = np.arctan2(delta[:, 0], -delta[:, 2]) - yaw
    health = np.divide(
        [row["health"] for row in target],
        [row["max_health"] for row in target],
        out=np.zeros(len(target), dtype=np.float64),
        where=np.asarray([row["max_health"] > 0.0 for row in target]),
    )
    return np.column_stack(
        (delta, velocity, distance, planar, np.sin(bearing), np.cos(bearing), health)
    )


def _world_clock(rows: Sequence[Any]) -> np.ndarray:
    if not rows:
        return np.empty((0, 7), dtype=np.float64)
    return np.asarray(
        [
            (
                row.world_tick,
                row.game_time_epoch_second,
                row.game_time_nano,
                row.day_progress,
                row.sunlight_factor,
                row.moon_phase,
                row.npc_count,
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def _decision_changes(
    capture: NativeNpcTraceCapture,
    phase: tuple[str, ...],
    control_next: np.ndarray,
    control_mask_next: np.ndarray,
    body_next: tuple[str, ...],
    head_next: tuple[str, ...],
    active_actions_next: tuple[tuple[str, ...], ...],
    internal_next: tuple[Any, ...],
) -> np.ndarray:
    rows = capture.emitted_count
    if not rows:
        return np.empty((0, len(_DECISION_CHANGE_LAYOUT)), dtype=np.bool_)
    body = tuple(zip(capture.body_instruction, capture.body_decision_path, strict=True))
    body_following = tuple(
        zip(body_next, _shift(capture.body_decision_path), strict=True)
    )
    head = tuple(zip(capture.head_instruction, capture.head_decision_path, strict=True))
    head_following = tuple(
        zip(head_next, _shift(capture.head_decision_path), strict=True)
    )
    actions = tuple(
        zip(capture.active_actions, capture.active_action_paths, strict=True)
    )
    actions_following = tuple(
        zip(active_actions_next, _shift(capture.active_action_paths), strict=True)
    )
    columns = (
        np.any(capture.control != control_next, axis=1),
        capture.control_mask != control_mask_next,
        capture.native_control != _shift(capture.native_control),
        _changed(body, body_following),
        _changed(head, head_following),
        _changed(actions, actions_following),
        _changed(capture.decision_internal_state, internal_next),
        _changed(
            capture.decision_target_uuid,
            _shift(capture.decision_target_uuid),
        ),
        _changed(phase, _shift(phase)),
        _changed(capture.attack_actions, _shift(capture.attack_actions)),
        _changed(capture.interactions, _shift(capture.interactions)),
    )
    return np.column_stack(columns)


def _changed(current: Sequence[Any], following: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [left != right for left, right in zip(current, following, strict=True)],
        dtype=np.bool_,
    )


def _support_changes(capture: NativeNpcTraceCapture) -> np.ndarray:
    if not capture.emitted_count:
        return np.empty((0, len(_DECISION_SUPPORT_CHANGE_LAYOUT)), dtype=np.bool_)
    result = []
    for before, after, target, decision_target in zip(
        capture.internal_state,
        capture.decision_internal_state,
        capture.target_uuid,
        capture.decision_target_uuid,
        strict=True,
    ):
        result.append(
            (
                (before.state_name, before.state_index, before.substate_index)
                != (after.state_name, after.state_index, after.substate_index),
                before.busy != after.busy,
                before.transitioning != after.transitioning,
                before.role_change_requested != after.role_change_requested,
                before.terminal_action != after.terminal_action,
                before.backing_away != after.backing_away,
                before.steering_motion != after.steering_motion,
                before.motion_in_progress != after.motion_in_progress,
                before.obstructed != after.obstructed,
                (before.current_speed, before.maximum_speed)
                != (after.current_speed, after.maximum_speed),
                before.marked_targets != after.marked_targets,
                target != decision_target,
            )
        )
    return np.asarray(result, dtype=np.bool_)


def _layer_manifest(
    components: Sequence[TraceComponent], capture: NativeNpcTraceCapture
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for layer in (
        "omniscient",
        "internal",
        "external",
        "decision",
        "transition",
        "extension",
    ):
        selected = tuple(
            component for component in components if component.spec.layer == layer
        )
        if not selected:
            continue
        manifest[layer] = {
            "components": tuple(component.spec.name for component in selected),
            "current_valid_rows": {
                component.spec.name: int(np.count_nonzero(component.valid))
                for component in selected
            },
            "next_valid_rows": {
                component.spec.name: int(np.count_nonzero(component.next_valid))
                for component in selected
                if component.next_valid is not None
            },
        }
    manifest["scope"] = {
        "rows": capture.emitted_count,
        "omniscient": "bounded_transform_bearing_entities_plus_world_clock",
        "explicit_extensions": (
            "terrain.local",
            "spatial.line_of_sight",
            "spatial.camera_calibration",
            "world.dynamic_entities",
            "reward.*",
        ),
    }
    return manifest


def _capture_world(capture: NativeNpcTraceCapture) -> tuple[object, ...]:
    return (
        capture.bridge_sha256,
        capture.action_contract_sha256,
        capture.event_contract_sha256,
        capture.server_version,
        capture.world,
        capture.worldgen_provider,
        capture.worldgen_version,
        capture.seed,
        tuple(sorted(capture.environment_parameters.items())),
    )


def _symbols(capture: NativeNpcTraceCapture) -> tuple[str, ...]:
    values = set(capture.state_name + capture.next_state_name)
    values.update(capture.body_instruction)
    values.update(capture.body_decision_path)
    values.update(capture.head_instruction)
    values.update(capture.head_decision_path)
    values.update(action for row in capture.active_actions for action in row)
    values.update(action for row in capture.active_action_paths for action in row)
    for rows in (
        capture.internal_state,
        capture.decision_internal_state,
        capture.next_internal_state,
    ):
        for row in rows:
            values.update((row.state_name, row.steering_motion))
            values.update(target.name for target in row.marked_targets)
    for rows in (capture.worldview, capture.next_worldview):
        for row in rows:
            for actor in row.actors:
                values.update((actor.role, actor.state_name))
    for rows in (capture.attack_actions, capture.next_attack_actions):
        for row in rows:
            for action in row:
                values.update(
                    (action.label, action.interaction_type, action.interaction_id)
                )
                values.add(action.path)
    for cause in capture.attack_execution_cause:
        values.update((cause.interaction_type, cause.interaction_id, cause.path))
    for rows in (capture.interactions, capture.next_interactions):
        for row in rows:
            for chain in row:
                values.update(
                    (
                        chain.source,
                        chain.type,
                        chain.base_type,
                        chain.initial_root_id,
                        chain.root_id,
                    )
                )
    for row in capture.damage_events:
        for event in row:
            values.update(
                (event.source_type, event.environment_type, event.damage_cause_id)
            )
    for row in capture.lifecycle_events:
        for event in row:
            values.update(
                (
                    event.kind,
                    event.origin,
                    event.subject,
                    event.key,
                    event.text_before,
                    event.text_after,
                )
            )
    values.update(row["item_id"] for row in capture.actor_evidence)
    values.discard("")
    return tuple(sorted(values))


def native_damage_event_batch(trace: TraceSequence) -> NativeDamageEventBatch:
    """Transfer one Arena trace's authoritative event ledger to fixed JAX arrays."""

    if trace.parameters.get("native.event_contract_sha256") != (
        NATIVE_NPC_EVENT_CONTRACT_SHA256
    ):
        raise ValueError("trace has no supported native event contract")
    return pack_native_damage_events(
        trace.resolve("native.damage_events"),
        count=trace.resolve("native.damage_event_count"),
        overflow=trace.resolve("native.damage_event_overflow"),
    )


def native_lifecycle_event_batch(
    trace: TraceSequence,
) -> NativeLifecycleEventBatch:
    """Transfer generic Java lifecycle facts to fixed-shape JAX arrays."""

    if trace.parameters.get("native.lifecycle_contract_sha256") != (
        NATIVE_NPC_LIFECYCLE_CONTRACT_SHA256
    ):
        raise ValueError("trace has no supported native lifecycle contract")
    return pack_native_lifecycle_events(
        trace.resolve("native.lifecycle_events"),
        count=trace.resolve("native.lifecycle_event_count"),
        overflow=trace.resolve("native.lifecycle_event_overflow"),
        source_available=trace.resolve("native.lifecycle_source_available"),
        source_partial=trace.resolve("native.lifecycle_source_partial"),
    )


def native_group_step_batch(
    info: Mapping[str, object] | NativeGroupStepEvidence,
) -> NativeGroupStepBatch:
    """Preserve Java action validation receipts as fixed-shape JAX arrays."""

    evidence = (
        info
        if isinstance(info, NativeGroupStepEvidence)
        else NativeGroupStepEvidence.from_info(info)
    )
    return pack_native_group_step_evidence(evidence)


def _combat_phase(actions, interactions, pauses, executing) -> tuple[str, ...]:
    phases: list[str] = []
    for row_actions, chains, pause, active in zip(
        actions, interactions, pauses, executing, strict=True
    ):
        if any(
            action.active and action.aiming_seconds_remaining > 0.0
            for action in row_actions
        ):
            phase = "aiming"
        elif any(chain.server_state == "NotFinished" for chain in chains):
            phase = "interaction"
        elif chains:
            phase = "completion"
        elif float(pause) > 0.0 or active:
            phase = "recovery"
        elif any(action.active for action in row_actions):
            phase = "windup"
        else:
            phase = "idle"
        phases.append(phase)
    return tuple(phases)


def _symbol_sha256(symbols: tuple[str, ...]) -> str:
    import hashlib
    import json

    return (
        hashlib.sha256(json.dumps(symbols, separators=(",", ":")).encode())
        .hexdigest()
        .upper()
    )


def _relative(actor: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    delta = opponent[:, :3] - actor[:, :3]
    velocity = opponent[:, 3:6] - actor[:, 3:6]
    distance = np.linalg.norm(delta, axis=1)
    planar = np.hypot(delta[:, 0], delta[:, 2])
    bearing = np.arctan2(delta[:, 0], -delta[:, 2]) - actor[:, 6]
    health = np.divide(
        opponent[:, 11],
        opponent[:, 12],
        out=np.zeros_like(opponent[:, 11]),
        where=opponent[:, 12] > 0.0,
    )
    return np.column_stack(
        (delta, velocity, distance, planar, np.sin(bearing), np.cos(bearing), health)
    )


__all__ = [
    "NATIVE_EPISODE_PROJECTOR_NAME",
    "NATIVE_EPISODE_PROJECTOR_VERSION",
    "NATIVE_HYTALE_ACTION_MASK",
    "NATIVE_BEHAVIOR_VIEW",
    "NATIVE_CORPUS_COMPATIBILITY_KEYS",
    "NATIVE_CONTEXTUAL_BEHAVIOR_VIEW",
    "NATIVE_CONTEXTUAL_STEERING_VIEW",
    "NATIVE_WORLD_MODEL_VIEW",
    "native_concurrent_trace_sequences",
    "native_duel_trace_sequences",
    "native_damage_event_batch",
    "attach_native_hytale_action_capabilities",
    "native_group_step_batch",
    "install_native_lifecycle_episodes",
    "native_lifecycle_event_batch",
    "native_trace_sequence",
    "native_trace_corpus",
    "terrain_component",
]
