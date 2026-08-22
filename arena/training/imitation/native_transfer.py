"""Fail-closed native-behavior warm start for the JAX Arsenal policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any

import jax.numpy as jnp
import numpy as np

from arena.jax_contract import HEAD_SPANS
from hytalegym.jax.combat.observation.v1.schema.contract import (
    SELF_FLOAT_FEATURES,
    TARGET_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.encoding.differential import (
    arsenal_policy_observation_group_slices,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    arsenal_policy_observation_size,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    world_geometry_shared_visual_dense_offsets,
    world_geometry_shared_visual_manifest,
    world_geometry_shared_visual_sha256,
    world_geometry_shared_visual_size,
)
from hytalegym.jax.combat.observation.v3.schema.contract import (
    ACTOR_WORLD_FLOAT_FEATURES,
    ACTOR_WORLD_MASK_FEATURES,
    MOVEMENT_STATE_FEATURES,
)
from hytalegym.jax.combat.arsenal.profiles import (
    NATIVE_NPC_ROLE_PROFILE_NAMES,
    NATIVE_NPC_ROLE_PROFILE_SCHEMA,
    hytale_0_5_7_native_npc_role_bindings,
    hytale_0_5_7_native_npc_role_program_content_sha256,
    hytale_0_5_7_native_profile_bindings,
    hytale_0_5_7_program_content_sha256,
)
from hytalegym.jax.combat.types import default_combat_params
from hytalegym.jax.training.types import PPOConfig, RecurrentPolicyParams

from .native_replay import (
    NATIVE_REPLAY_ATTACK_BLACKBOARD_FEATURES,
    NATIVE_REPLAY_ACTION_SCHEMA,
    NATIVE_REPLAY_OBSERVATION_SCHEMA,
    NATIVE_REPLAY_VISUAL_SHA256,
    NATIVE_REPLAY_VISUAL_SIZE,
    NativeBehaviorPolicy,
    NativeReplayCorpus,
    native_replay_action_head_coverage,
)


NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA = (
    "arena_native_arsenal_shared_behavior_context_v6"
)
NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA = (
    "arena_native_arsenal_conditioned_behavior_context_v5"
)
NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA = (
    "arena_native_arsenal_visual_shared_behavior_context_v5"
)
NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA = (
    "arena_native_arsenal_visual_conditioned_behavior_context_v5"
)
NATIVE_ARSENAL_TRANSFER_SCHEMA = "arena_native_arsenal_policy_warm_start_v11"
NATIVE_ARSENAL_ABILITY_BINDING_SCHEMA = "arena_native_arsenal_role_ability_binding_v2"
NATIVE_ARSENAL_ABILITY_SLOT_CONTRACT_SCHEMA = (
    "arena_native_arsenal_runtime_ability_slots_v1"
)
NATIVE_ARSENAL_ACTION_HEAD_READINESS_SCHEMA = (
    "arena_native_arsenal_action_head_readiness_v2"
)
NATIVE_ARSENAL_TRANSFERABLE_HEADS = (
    "base_action",
    "locomotion_gait_compass",
    "yaw_delta_bins",
    "pitch_delta_bins",
)
_NEUTRAL_LOGIT_MARGIN = 8.0
_ATTACK_BLACKBOARD_INDEX = {
    name: 932 + index
    for index, name in enumerate(NATIVE_REPLAY_ATTACK_BLACKBOARD_FEATURES)
}


@dataclass(frozen=True, slots=True)
class NativeArsenalAbilityBinding:
    """Pinned projection from one Java role into a concrete JAX layout."""

    projection: str
    actor_role: str
    source_ability_slot_contract_sha256: str
    target_ability_slot_contract_sha256: str
    target_program_content_sha256: str
    source_slot_indices: tuple[int, ...]
    source_slots: tuple[tuple[str, str], ...]
    target_slots: tuple[tuple[str, str], ...]
    source_to_target_slots: tuple[int, ...]


_SELF_INDICES = tuple(
    SELF_FLOAT_FEATURES.index(name)
    for name in (
        "forward_velocity",
        "right_velocity",
        "vertical_velocity",
        "health_fraction",
        "yaw_sin",
        "yaw_cos",
        "pitch",
        "grounded",
        "attack_executing",
        "attack_cooldown",
        "alive",
    )
)
_TARGET_INDICES = tuple(
    TARGET_FLOAT_FEATURES.index(name)
    for name in (
        "relative_forward",
        "relative_right",
        "relative_up",
        "relative_velocity_forward",
        "relative_velocity_right",
        "relative_velocity_up",
        "planar_distance",
        "distance",
        "health_fraction",
        "bearing_sin",
        "bearing_cos",
        "visible",
    )
)
_MOVEMENT_NAMES = (
    "idle",
    "horizontal_idle",
    "jumping",
    "flying",
    "walking",
    "running",
    "sprinting",
    "falling",
    "climbing",
    "in_fluid",
    "swimming",
    "swim_jumping",
    "on_ground",
)
_MOVEMENT_INDICES = tuple(
    MOVEMENT_STATE_FEATURES.index(name) for name in _MOVEMENT_NAMES
)
_GROUP_SELECTIONS = (
    ("self_f32", _SELF_INDICES),
    ("target_f32", _TARGET_INDICES),
    ("target_mask", (0,)),
    ("defense_f32", (0, 1, 6)),
    ("actor_world_f32", (ACTOR_WORLD_FLOAT_FEATURES.index("controller_in_fluid"),)),
    (
        "actor_world_mask",
        (ACTOR_WORLD_MASK_FEATURES.index("controller_medium_available"),),
    ),
    ("movement_state_f32", _MOVEMENT_INDICES),
    ("movement_state_mask", _MOVEMENT_INDICES),
    # The native fundamental head collapses every authored Java attack root
    # onto the target profile's default slot.  Keep these at the end so v3
    # retains the complete v2 physical prefix byte-for-byte.
    ("ability_mask", (0,)),
    ("ability_legal", (0,)),
)


def _dense_indices() -> tuple[int, ...]:
    groups = dict(arsenal_policy_observation_group_slices())
    return tuple(
        groups[name].start + index
        for name, indexes in _GROUP_SELECTIONS
        for index in indexes
    )


NATIVE_ARSENAL_SHARED_DENSE_INDICES = _dense_indices()
NATIVE_ARSENAL_SHARED_FEATURE_LAYOUT = tuple(
    f"{name}.{index}" for name, indexes in _GROUP_SELECTIONS for index in indexes
)
NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE = len(NATIVE_ARSENAL_SHARED_DENSE_INDICES)
_WORLD_GROUP = dict(arsenal_policy_observation_group_slices())["world_geometry"]
NATIVE_ARSENAL_VISUAL_DENSE_INDICES = tuple(
    _WORLD_GROUP.start + index for index in world_geometry_shared_visual_dense_offsets()
)
NATIVE_ARSENAL_VISUAL_SIZE = world_geometry_shared_visual_size()
NATIVE_ARSENAL_VISUAL_SHARED_DENSE_INDICES = (
    *NATIVE_ARSENAL_SHARED_DENSE_INDICES,
    *NATIVE_ARSENAL_VISUAL_DENSE_INDICES,
)
NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE = len(
    NATIVE_ARSENAL_VISUAL_SHARED_DENSE_INDICES
)
NATIVE_ARSENAL_VISUAL_SHARED_FEATURE_LAYOUT = (
    *NATIVE_ARSENAL_SHARED_FEATURE_LAYOUT,
    *(
        f"world_geometry.shared_visual.{index}"
        for index in range(NATIVE_ARSENAL_VISUAL_SIZE)
    ),
)

_params = default_combat_params()
_NORMALIZATION = {
    "agent_max_speed": float(np.asarray(_params.agent_max_speed)),
    "vertical_speed_scale": float(np.asarray(_params.vertical_speed_scale)),
    "target_chase_speed": float(np.asarray(_params.target_chase_speed)),
    "target_radius_blocks": 24.0,
    "wire_fixed_point_scale": float(np.asarray(_params.wire_fixed_point_scale)),
    "agent_attack_pause_max_seconds": float(
        np.asarray(_params.agent_attack_pause_max_seconds)
    ),
}
_CONTRACT = {
    "schema": NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA,
    # V6 restores executing and cooldown from Java's pre-decision blackboard.
    # BASIC mask/legality stay last.
    "source_schema": NATIVE_REPLAY_OBSERVATION_SCHEMA,
    "dense_observation_size": arsenal_policy_observation_size(),
    "dense_indices": list(NATIVE_ARSENAL_SHARED_DENSE_INDICES),
    "feature_layout": list(NATIVE_ARSENAL_SHARED_FEATURE_LAYOUT),
    "normalization": _NORMALIZATION,
    "target_evidence": "zero_unless_perceptible",
    "excluded": {
        "resources": "native rows lack the profile-pinned JAX minimum check",
        "ability_f32": (
            "native attack duration/progress is not numerically identical to the "
            "profile-pinned JAX ability bank"
        ),
        "ability_slots_1_to_15": (
            "fundamental Java BASIC is intentionally bound only to target slot zero"
        ),
        "defense_2_to_5": "native rows carry raw engine units",
        "actor_world_submersion_and_drop": (
            "the Java capture marks these channels unavailable"
        ),
        "externally_owned_movement_states": (
            "the WorldGen combat runtime does not author these fields"
        ),
        "observation_valid": "portable replay omits actor-evidence failure bits",
    },
    "geometry": "not_in_v1_shared_view",
}
NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256 = (
    hashlib.sha256(
        json.dumps(_CONTRACT, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    .hexdigest()
    .upper()
)
_VISUAL_CONTRACT = {
    "schema": NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
    "physical_shared_sha256": NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256,
    "visual_sha256": world_geometry_shared_visual_sha256(),
    "visual": world_geometry_shared_visual_manifest(),
    "dense_indices": list(NATIVE_ARSENAL_VISUAL_SHARED_DENSE_INDICES),
    "jax_only": "zero_initialized_trainable_supplemental_after_warm_start",
}
NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256 = (
    hashlib.sha256(
        json.dumps(_VISUAL_CONTRACT, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    )
    .hexdigest()
    .upper()
)


def native_arsenal_conditioned_observation_sha256(roles: tuple[str, ...]) -> str:
    """Pin the ordered actor/opponent style vocabulary layered on shared state."""

    _require_roles(roles)
    contract = {
        "schema": NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA,
        "shared_observation_sha256": NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256,
        "roles": list(roles),
        "conditioning": ["actor_role_one_hot", "opponent_role_one_hot"],
        "deployment": "fixed style is folded into the first-layer bias",
    }
    return (
        hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        .hexdigest()
        .upper()
    )


def native_arsenal_visual_conditioned_observation_sha256(
    roles: tuple[str, ...],
) -> str:
    """Pin style conditioning layered on the shared physical+visual view."""

    _require_roles(roles)
    contract = {
        "schema": NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
        "shared_observation_sha256": (NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256),
        "roles": list(roles),
        "conditioning": ["actor_role_one_hot", "opponent_role_one_hot"],
        "deployment": "fixed style is folded into the first-layer bias",
    }
    return (
        hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        .hexdigest()
        .upper()
    )


def native_arsenal_ability_binding(
    corpus: NativeReplayCorpus,
    actor_role: str,
    target_slots: tuple[tuple[str, str], ...],
    target_program_content_sha256: str,
) -> NativeArsenalAbilityBinding:
    """Bind every known Java root to the same root in a pinned JAX program."""

    source_slot_indices, source_slots = _source_role_slots(corpus, actor_role)
    target_slots = _require_root_slots(target_slots, "target")
    if not set(source_slots).issubset(target_slots):
        raise ValueError("known native ability roots are not present in the JAX layout")
    target_program_content_sha256 = _require_sha256(
        target_program_content_sha256,
        "target program content",
    )
    binding = NativeArsenalAbilityBinding(
        projection="exact_root_injection",
        actor_role=actor_role,
        source_ability_slot_contract_sha256=_require_sha256(
            corpus.ability_slot_contract_sha256,
            "source ability-slot contract",
        ),
        target_ability_slot_contract_sha256=(
            native_arsenal_ability_slot_contract_sha256(
                target_slots,
                target_program_content_sha256,
            )
        ),
        target_program_content_sha256=target_program_content_sha256,
        source_slot_indices=source_slot_indices,
        source_slots=source_slots,
        target_slots=target_slots,
        source_to_target_slots=tuple(target_slots.index(slot) for slot in source_slots),
    )
    _validate_ability_binding(binding)
    return binding


def _source_role_slots(corpus, actor_role):
    if corpus.action_schema != NATIVE_REPLAY_ACTION_SCHEMA:
        raise ValueError("native ability binding requires the causal action schema")
    if not corpus.ability_slot_contract_sha256 or len(
        corpus.ability_slot_layout
    ) != len(corpus.roles):
        raise ValueError("native corpus lacks a role ability-slot contract")
    try:
        role_index = corpus.roles.index(actor_role)
    except ValueError as exc:
        raise ValueError("native ability binding names an unknown actor role") from exc
    source_rows = tuple(
        (index, (slot[2], slot[1]))
        for index, slot in enumerate(corpus.ability_slot_layout[role_index])
        if slot[2]
    )
    source_slot_indices = tuple(index for index, _ in source_rows)
    source_slots = tuple(slot for _, slot in source_rows)
    _require_root_slots(source_slots, "source")
    if not source_slots:
        raise ValueError("native role has no causally observed attack root")
    return source_slot_indices, source_slots


def native_arsenal_profile_basic_attack_binding(
    corpus: NativeReplayCorpus,
    actor_role: str,
    target_profile: str,
) -> NativeArsenalAbilityBinding:
    """Bind Java attack onset to the certified default attack of a JAX profile."""

    source_indices, source_slots = _source_role_slots(corpus, actor_role)
    profile = hytale_0_5_7_native_profile_bindings(target_profile)
    target_slots = tuple(
        (row.interaction_id, row.interaction_type) for row in profile.abilities
    )
    authored = tuple(profile.authored_ability_slots)
    if not target_slots or authored != tuple(range(len(target_slots))):
        raise ValueError(
            "basic-attack transfer requires a contiguous JAX ability layout"
        )
    if authored[0] != 0 or target_slots[0][1] != "Primary":
        raise ValueError("JAX profile slot zero is not a certified primary attack")
    program = hytale_0_5_7_program_content_sha256()
    binding = NativeArsenalAbilityBinding(
        projection="basic_attack_to_default_slot",
        actor_role=actor_role,
        source_ability_slot_contract_sha256=_require_sha256(
            corpus.ability_slot_contract_sha256,
            "source ability-slot contract",
        ),
        target_ability_slot_contract_sha256=(
            native_arsenal_ability_slot_contract_sha256(target_slots, program)
        ),
        target_program_content_sha256=program,
        source_slot_indices=source_indices,
        source_slots=source_slots,
        target_slots=target_slots,
        source_to_target_slots=(0,),
    )
    _validate_ability_binding(binding)
    return binding


def native_arsenal_npc_role_ability_binding(
    corpus: NativeReplayCorpus,
    actor_role: str,
) -> NativeArsenalAbilityBinding:
    """Bind a native role to its source-pinned executable JAX NPC program."""

    profile = hytale_0_5_7_native_npc_role_bindings(actor_role)
    return native_arsenal_ability_binding(
        corpus,
        actor_role,
        tuple((row.interaction_id, row.interaction_type) for row in profile.abilities),
        hytale_0_5_7_native_npc_role_program_content_sha256(actor_role),
    )


def native_arsenal_action_head_readiness(corpus: NativeReplayCorpus) -> dict[str, Any]:
    """Publish exact label/runtime coverage without treating abstention as no-op."""

    if corpus.action_schema != NATIVE_REPLAY_ACTION_SCHEMA:
        raise ValueError("native head readiness requires the causal action schema")
    if len(corpus.ability_slot_layout) != len(corpus.roles):
        raise ValueError("native head readiness requires role ability-slot layouts")

    executable = set(NATIVE_NPC_ROLE_PROFILE_NAMES)
    role_rows = []
    for role, layout in zip(corpus.roles, corpus.ability_slot_layout, strict=True):
        roots = tuple(slot[2] for slot in layout if slot[2])
        if not roots:
            status = "no_observed_ability_root"
            program_sha256 = None
        elif role not in executable:
            status = "labelled_runtime_program_unavailable"
            program_sha256 = None
        else:
            binding = native_arsenal_npc_role_ability_binding(corpus, role)
            status = "exact_label_and_executable_program"
            program_sha256 = binding.target_program_content_sha256
        role_rows.append(
            {
                "role": role,
                "known_roots": list(roots),
                "basic_attack_status": (
                    "portable_to_certified_profile_slot_zero"
                    if roots
                    else "unavailable_no_causal_attack"
                ),
                "status": status,
                "program_content_sha256": program_sha256,
            }
        )

    coverage = native_replay_action_head_coverage(corpus)
    exact_generic = {
        "locomotion_gait_compass",
        "yaw_delta_bins",
        "pitch_delta_bins",
    }
    runtime = {
        name: (
            "abstain_without_penalty"
            if row["training_behavior"] != "supervise_exact_rows"
            else "portable_basic_attack_profile_slot_zero"
            if name == "ability_none_plus_slots"
            else "exact_generic"
            if name in exact_generic
            else "labelled_runtime_unavailable"
        )
        for name, row in coverage.items()
    }
    return {
        "schema": NATIVE_ARSENAL_ACTION_HEAD_READINESS_SCHEMA,
        "npc_program_schema": NATIVE_NPC_ROLE_PROFILE_SCHEMA,
        "heads": {
            name: {
                "label_status": row["status"],
                "runtime_status": runtime[name],
                "labelled_rows": row["labelled_rows"],
                "training_behavior": row["training_behavior"],
            }
            for name, row in coverage.items()
        },
        "ability_roles": role_rows,
        "exact_runtime_roles": [
            row["role"]
            for row in role_rows
            if row["status"] == "exact_label_and_executable_program"
        ],
        "portable_basic_attack_roles": [
            row["role"]
            for row in role_rows
            if row["basic_attack_status"] == "portable_to_certified_profile_slot_zero"
        ],
        "blocked_runtime_roles": [
            row["role"]
            for row in role_rows
            if row["status"] == "labelled_runtime_program_unavailable"
        ],
    }


def native_arsenal_ability_slot_contract_sha256(
    target_slots: tuple[tuple[str, str], ...],
    target_program_content_sha256: str,
) -> str:
    """Pin ordered JAX ability roots together with their executable program."""

    slots = _require_root_slots(target_slots, "target")
    program = _require_sha256(target_program_content_sha256, "target program content")
    payload = {
        "schema": NATIVE_ARSENAL_ABILITY_SLOT_CONTRACT_SCHEMA,
        "slots": [
            {
                "slot": index,
                "interaction_id": interaction_id,
                "interaction_type": interaction_type,
            }
            for index, (interaction_id, interaction_type) in enumerate(slots)
        ],
        "program_content_sha256": program,
        "factor_encoding": "choice_zero_none_else_slot_plus_one",
    }
    return (
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        .hexdigest()
        .upper()
    )


def native_arsenal_ability_binding_sha256(
    binding: NativeArsenalAbilityBinding,
) -> str:
    """Return a stable identity for one verified role-local slot permutation."""

    _validate_ability_binding(binding)
    payload = {
        "schema": NATIVE_ARSENAL_ABILITY_BINDING_SCHEMA,
        "projection": binding.projection,
        "actor_role": binding.actor_role,
        "source_ability_slot_contract_sha256": (
            binding.source_ability_slot_contract_sha256
        ),
        "target_ability_slot_contract_sha256": (
            binding.target_ability_slot_contract_sha256
        ),
        "target_program_content_sha256": binding.target_program_content_sha256,
        "source_slot_indices": list(binding.source_slot_indices),
        "source_slots": [list(slot) for slot in binding.source_slots],
        "target_slots": [list(slot) for slot in binding.target_slots],
        "source_to_target_slots": list(binding.source_to_target_slots),
        "factor_encoding": "choice_zero_none_else_slot_plus_one",
    }
    return (
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        .hexdigest()
        .upper()
    )


def native_arsenal_shared_replay_corpus(
    corpus: NativeReplayCorpus,
) -> NativeReplayCorpus:
    """Project native rows onto the exact native/Arsenal feature intersection."""

    if corpus.observation_schema == NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA:
        return corpus
    if corpus.observation_schema != NATIVE_REPLAY_OBSERVATION_SCHEMA:
        raise ValueError("native corpus has no declared Arsenal shared projection")
    source = np.asarray(corpus.tensors.observation, dtype=np.float32)
    if source.shape[-1] < 942:
        raise ValueError("native omniscient observation is missing fixed context")

    valid = np.asarray(corpus.tensors.valid, dtype=np.bool_)
    lifecycle_available = (
        source[..., _ATTACK_BLACKBOARD_INDEX["combat_lifecycle_available"]] > 0.5
    )
    if np.any(valid & ~lifecycle_available):
        raise ValueError(
            "native shared projection requires the pre-decision combat blackboard"
        )
    actor = source[..., 751:834]
    defense_available = actor[..., (26, 27, 32)]
    if np.any(valid[..., None] & (defense_available < 0.5)):
        raise ValueError("shared defense features require native availability")

    yaw_sin, yaw_cos = actor[..., 1], actor[..., 2]
    velocity = source[..., :3] * np.float32(16.0)
    self_f32 = np.stack(
        (
            (-yaw_sin * velocity[..., 0] - yaw_cos * velocity[..., 2])
            / _NORMALIZATION["agent_max_speed"],
            (yaw_cos * velocity[..., 0] - yaw_sin * velocity[..., 2])
            / _NORMALIZATION["agent_max_speed"],
            velocity[..., 1] / _NORMALIZATION["vertical_speed_scale"],
            actor[..., 0],
            yaw_sin,
            yaw_cos,
            actor[..., 3],
            actor[..., 58] * actor[..., 66],
            source[..., _ATTACK_BLACKBOARD_INDEX["attack_executing"]],
            source[..., _ATTACK_BLACKBOARD_INDEX["attack_pause_seconds/5"]]
            * np.float32(5.0 / _NORMALIZATION["agent_attack_pause_max_seconds"]),
            (actor[..., 0] > 0.0).astype(np.float32),
        ),
        axis=-1,
    )
    relative = source[..., 10:21]
    scale = _NORMALIZATION["wire_fixed_point_scale"]
    offset = _java_round(relative[..., :3] * np.float32(32.0), scale)
    relative_velocity = relative[..., 3:6] * np.float32(16.0)
    forward = -yaw_sin * offset[..., 0] - yaw_cos * offset[..., 2]
    right = yaw_cos * offset[..., 0] - yaw_sin * offset[..., 2]
    velocity_forward = (
        -yaw_sin * relative_velocity[..., 0] - yaw_cos * relative_velocity[..., 2]
    )
    velocity_right = (
        yaw_cos * relative_velocity[..., 0] - yaw_sin * relative_velocity[..., 2]
    )
    planar = np.hypot(forward, right)
    distance = np.linalg.norm(offset, axis=-1)
    radius = _NORMALIZATION["target_radius_blocks"]
    safe_planar = np.maximum(planar, np.float32(1.0e-6))
    visible = source[..., 21] > 0.5
    target_f32 = np.stack(
        (
            forward / radius,
            right / radius,
            offset[..., 1] / radius,
            velocity_forward / _NORMALIZATION["target_chase_speed"],
            velocity_right / _NORMALIZATION["target_chase_speed"],
            relative_velocity[..., 1] / _NORMALIZATION["vertical_speed_scale"],
            planar / radius,
            distance / radius,
            relative[..., 10],
            right / safe_planar,
            forward / safe_planar,
            visible.astype(np.float32),
        ),
        axis=-1,
    )
    target_f32 = np.where(visible[..., None], target_f32, 0.0)
    movement_available = actor[..., 66:67]
    if corpus.ability_slot_layout:
        if len(corpus.ability_slot_layout) != len(corpus.roles):
            raise ValueError("native role ability-slot layout is incomplete")
        authored_by_role = np.asarray(
            [bool(slots) for slots in corpus.ability_slot_layout],
            dtype=np.bool_,
        )
        role = np.asarray(corpus.tensors.role, dtype=np.int32)
        if np.any((role < 0) | (role >= authored_by_role.size)):
            raise ValueError("native replay role index is outside its vocabulary")
        basic_authored = np.broadcast_to(
            authored_by_role[role][:, None, None],
            (*valid.shape, 1),
        )
    else:
        # Legacy synthetic fixtures carry no pinned role layout.  Absence is
        # unavailable evidence, never an authored-false guess.
        basic_authored = np.zeros((*valid.shape, 1), dtype=np.bool_)
    basic_legal = (
        basic_authored[..., 0]
        & np.asarray(corpus.tensors.basic_attack_legal, dtype=np.bool_)
    )[..., None]
    shared = np.concatenate(
        (
            self_f32,
            target_f32,
            visible[..., None],
            actor[..., (19, 20, 25)],
            actor[..., 33:34],
            actor[..., 38:39],
            actor[..., np.asarray(_MOVEMENT_INDICES) + 43] * movement_available,
            np.repeat(movement_available, len(_MOVEMENT_INDICES), axis=-1),
            basic_authored.astype(np.float32),
            basic_legal.astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    shared = np.where(valid[..., None], np.clip(shared, -1.0, 1.0), 0.0)
    if shared.shape[-1] != NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE:
        raise RuntimeError("native Arsenal shared observation width drift")
    tensors = corpus.tensors._replace(observation=shared)
    return replace(
        corpus,
        tensors=tensors,
        feature_layout=NATIVE_ARSENAL_SHARED_FEATURE_LAYOUT,
        tensor_sha256=_tensor_sha256(tensors),
        observation_schema=NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA,
    )


def native_arsenal_visual_shared_replay_corpus(
    corpus: NativeReplayCorpus,
) -> NativeReplayCorpus:
    """Project native rows onto exact shared physics plus JAX visual tokens."""

    if corpus.observation_schema == NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA:
        return corpus
    if corpus.observation_schema != NATIVE_REPLAY_OBSERVATION_SCHEMA:
        raise ValueError("native visual projection requires replay observation v6")
    if (
        not corpus.feature_layout
        or corpus.feature_layout[-1]
        != f"jax_shared_world_visual[{NATIVE_REPLAY_VISUAL_SIZE}]"
        or NATIVE_REPLAY_VISUAL_SHA256 != world_geometry_shared_visual_sha256()
    ):
        raise ValueError("native replay visual contract is absent or stale")
    shared = native_arsenal_shared_replay_corpus(corpus)
    source = np.asarray(corpus.tensors.observation, dtype=np.float32)
    visual = source[..., -NATIVE_REPLAY_VISUAL_SIZE:]
    valid = np.asarray(corpus.tensors.valid, dtype=np.bool_)
    if (
        not np.all(np.isfinite(visual))
        or np.any(np.abs(visual[valid]) > np.float32(1.0))
        or np.any(visual[~valid] != 0.0)
    ):
        raise ValueError("native replay visual rows violate the shared contract")
    observation = np.concatenate(
        (np.asarray(shared.tensors.observation), visual), axis=-1
    ).astype(np.float32)
    tensors = shared.tensors._replace(observation=observation)
    return replace(
        shared,
        tensors=tensors,
        feature_layout=NATIVE_ARSENAL_VISUAL_SHARED_FEATURE_LAYOUT,
        tensor_sha256=_tensor_sha256(tensors),
        observation_schema=NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
    )


def native_arsenal_visual_conditioned_replay_corpus(
    corpus: NativeReplayCorpus,
) -> NativeReplayCorpus:
    """Append actor/opponent style to the exact shared visual projection."""

    if (
        corpus.observation_schema
        == NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA
    ):
        return corpus
    if corpus.observation_schema == NATIVE_REPLAY_OBSERVATION_SCHEMA:
        shared = native_arsenal_shared_replay_corpus(corpus)
        source = np.asarray(corpus.tensors.observation, dtype=np.float32)
        visual = source[..., -NATIVE_REPLAY_VISUAL_SIZE:]
        valid_rows = np.asarray(corpus.tensors.valid, dtype=np.bool_)
        if (
            not np.all(np.isfinite(visual))
            or np.any(np.abs(visual[valid_rows]) > np.float32(1.0))
            or np.any(visual[~valid_rows] != 0.0)
        ):
            raise ValueError("native replay visual rows violate the shared contract")
        base = (np.asarray(shared.tensors.observation), visual)
    else:
        shared = native_arsenal_visual_shared_replay_corpus(corpus)
        base = (np.asarray(shared.tensors.observation),)
    roles = _require_roles(shared.roles)
    count = len(roles)
    eye = np.eye(count, dtype=np.float32)
    valid = np.asarray(shared.tensors.valid, dtype=np.float32)[..., None]
    actor = np.broadcast_to(
        eye[np.asarray(shared.tensors.role)][:, None, :],
        (*valid.shape[:2], count),
    )
    opponent = np.broadcast_to(
        eye[np.asarray(shared.tensors.opponent_role)][:, None, :],
        (*valid.shape[:2], count),
    )
    observation = np.concatenate(
        (*base, actor * valid, opponent * valid),
        axis=-1,
    )
    tensors = shared.tensors._replace(observation=observation)
    return replace(
        shared,
        tensors=tensors,
        feature_layout=(
            *NATIVE_ARSENAL_VISUAL_SHARED_FEATURE_LAYOUT,
            *(f"actor_style.{role}" for role in roles),
            *(f"opponent_style.{role}" for role in roles),
        ),
        tensor_sha256=_tensor_sha256(tensors),
        observation_schema=NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
    )


def native_arsenal_conditioned_replay_corpus(
    corpus: NativeReplayCorpus,
) -> NativeReplayCorpus:
    """Append explicit actor/opponent style to the portable physical context."""

    if corpus.observation_schema == NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA:
        return corpus
    shared = native_arsenal_shared_replay_corpus(corpus)
    roles = _require_roles(shared.roles)
    count = len(roles)
    eye = np.eye(count, dtype=np.float32)
    valid = np.asarray(shared.tensors.valid, dtype=np.float32)[..., None]
    actor = np.broadcast_to(
        eye[np.asarray(shared.tensors.role)][:, None, :],
        (*valid.shape[:2], count),
    )
    opponent = np.broadcast_to(
        eye[np.asarray(shared.tensors.opponent_role)][:, None, :],
        (*valid.shape[:2], count),
    )
    observation = np.concatenate(
        (np.asarray(shared.tensors.observation), actor * valid, opponent * valid),
        axis=-1,
    )
    tensors = shared.tensors._replace(observation=observation)
    layout = (
        *NATIVE_ARSENAL_SHARED_FEATURE_LAYOUT,
        *(f"actor_style.{role}" for role in roles),
        *(f"opponent_style.{role}" for role in roles),
    )
    return replace(
        shared,
        tensors=tensors,
        feature_layout=layout,
        tensor_sha256=_tensor_sha256(tensors),
        observation_schema=NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA,
    )


def arsenal_shared_observation(dense: Any) -> jnp.ndarray:
    """Select the same shared columns from a live 8,271-value policy row."""

    value = jnp.asarray(dense, dtype=jnp.float32)
    expected = arsenal_policy_observation_size()
    if value.shape[-1] != expected:
        raise ValueError(f"Arsenal policy observation width must be {expected}")
    return jnp.take(
        value,
        jnp.asarray(NATIVE_ARSENAL_SHARED_DENSE_INDICES, dtype=jnp.int32),
        axis=-1,
    )


def arsenal_visual_shared_observation(dense: Any) -> jnp.ndarray:
    """Select exact native/JAX physical and visual fields from a live row."""

    value = jnp.asarray(dense, dtype=jnp.float32)
    expected = arsenal_policy_observation_size()
    if value.shape[-1] != expected:
        raise ValueError(f"Arsenal policy observation width must be {expected}")
    return jnp.take(
        value,
        jnp.asarray(NATIVE_ARSENAL_VISUAL_SHARED_DENSE_INDICES, dtype=jnp.int32),
        axis=-1,
    )


def arsenal_conditioned_observation(
    dense: Any,
    roles: tuple[str, ...],
    style: tuple[str, str],
) -> jnp.ndarray:
    """Project live state and append one explicitly selected behavior style."""

    roles = _require_roles(roles)
    actor, opponent = _require_style(roles, style)
    shared = arsenal_shared_observation(dense)
    condition = jnp.concatenate(
        (
            jax_one_hot(roles.index(actor), len(roles)),
            jax_one_hot(roles.index(opponent), len(roles)),
        )
    )
    return jnp.concatenate(
        (shared, jnp.broadcast_to(condition, (*shared.shape[:-1], condition.size))),
        axis=-1,
    )


def arsenal_visual_conditioned_observation(
    dense: Any,
    roles: tuple[str, ...],
    style: tuple[str, str],
) -> jnp.ndarray:
    """Project shared world vision and append one selected behavior style."""

    roles = _require_roles(roles)
    actor, opponent = _require_style(roles, style)
    shared = arsenal_visual_shared_observation(dense)
    condition = jnp.concatenate(
        (
            jax_one_hot(roles.index(actor), len(roles)),
            jax_one_hot(roles.index(opponent), len(roles)),
        )
    )
    return jnp.concatenate(
        (shared, jnp.broadcast_to(condition, (*shared.shape[:-1], condition.size))),
        axis=-1,
    )


def native_behavior_observation_projector(
    policy: NativeBehaviorPolicy,
    style: tuple[str, str] | None = None,
):
    """Return the exact live projector required by a native behavior policy."""

    schema = policy.metadata.get("native_replay_observation_schema")
    if schema == NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA:
        if style is not None:
            raise ValueError("unconditioned native policy does not accept a style")
        return arsenal_shared_observation
    if schema == NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA:
        if style is not None:
            raise ValueError("unconditioned native policy does not accept a style")
        return arsenal_visual_shared_observation
    conditioned = {
        NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA: (
            arsenal_conditioned_observation
        ),
        NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA: (
            arsenal_visual_conditioned_observation
        ),
    }
    if schema not in conditioned:
        raise ValueError("native policy has no live Arsenal projector")
    roles = tuple(policy.metadata.get("native_behavior_style_roles", ()))
    selected = _require_style(_require_roles(roles), style)
    return lambda dense: conditioned[schema](dense, roles, selected)


def transplant_native_behavior_policy(
    policy: NativeBehaviorPolicy,
    target_params: RecurrentPolicyParams,
    target_config: PPOConfig,
    target_combat_params: Any,
    style: tuple[str, str] | None = None,
    *,
    ability_binding: NativeArsenalAbilityBinding | None = None,
    target_ability_slot_contract_sha256: str | None = None,
) -> tuple[RecurrentPolicyParams, dict[str, Any]]:
    """Warm-start shared encoder/GRU and only transferable action heads."""

    schema = policy.metadata.get("native_replay_observation_schema")
    if policy.metadata.get("native_arsenal_shared_observation_sha256") != (
        NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256
    ):
        raise ValueError("native behavior policy was not trained on the shared view")
    conditioning = None
    source_kernel = policy.params.encoder_input.kernel
    source_bias = policy.params.encoder_input.bias
    dense_indices = NATIVE_ARSENAL_SHARED_DENSE_INDICES
    if schema == NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA:
        if style is not None:
            raise ValueError("unconditioned native policy does not accept a style")
        expected_size = NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE
    elif schema == NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA:
        roles = _require_roles(
            tuple(policy.metadata.get("native_behavior_style_roles", ()))
        )
        selected = _require_style(roles, style)
        expected_hash = native_arsenal_conditioned_observation_sha256(roles)
        if (
            policy.metadata.get("native_arsenal_conditioned_observation_sha256")
            != expected_hash
        ):
            raise ValueError("native behavior style contract mismatch")
        count = len(roles)
        actor, opponent = (roles.index(value) for value in selected)
        source_bias = (
            source_bias
            + source_kernel[NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE + actor]
            + source_kernel[NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE + count + opponent]
        )
        source_kernel = source_kernel[:NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE]
        expected_size = NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE + 2 * count
        conditioning = {
            "actor_role": selected[0],
            "opponent_role": selected[1],
            "roles": list(roles),
            "method": "one_hot_first_layer_contribution_folded_into_bias",
        }
    elif schema == NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA:
        if style is not None:
            raise ValueError("unconditioned native policy does not accept a style")
        if (
            policy.metadata.get("native_arsenal_visual_shared_observation_sha256")
            != NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256
        ):
            raise ValueError("native behavior visual contract mismatch")
        dense_indices = NATIVE_ARSENAL_VISUAL_SHARED_DENSE_INDICES
        expected_size = NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE
    elif schema == NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA:
        roles = _require_roles(
            tuple(policy.metadata.get("native_behavior_style_roles", ()))
        )
        selected = _require_style(roles, style)
        if policy.metadata.get(
            "native_arsenal_visual_shared_observation_sha256"
        ) != NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256 or policy.metadata.get(
            "native_arsenal_visual_conditioned_observation_sha256"
        ) != native_arsenal_visual_conditioned_observation_sha256(roles):
            raise ValueError("native behavior visual style contract mismatch")
        count = len(roles)
        actor, opponent = (roles.index(value) for value in selected)
        source_bias = (
            source_bias
            + source_kernel[NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE + actor]
            + source_kernel[
                NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE + count + opponent
            ]
        )
        source_kernel = source_kernel[:NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE]
        dense_indices = NATIVE_ARSENAL_VISUAL_SHARED_DENSE_INDICES
        expected_size = NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE + 2 * count
        conditioning = {
            "actor_role": selected[0],
            "opponent_role": selected[1],
            "roles": list(roles),
            "method": "one_hot_first_layer_contribution_folded_into_bias",
        }
    else:
        raise ValueError("native behavior policy was not trained on a portable view")
    if policy.config.observation_size != expected_size:
        raise ValueError("native behavior policy observation width drift")
    if (
        target_config.observation_size != arsenal_policy_observation_size()
        or tuple(target_config.action_head_sizes)
        != tuple(ARSENAL_POLICY_ACTION_HEAD_SIZES)
        or tuple(HEAD_SPANS) != tuple(ARSENAL_POLICY_ACTION_HEAD_NAMES)
    ):
        raise ValueError("target policy is not the current Arsenal surface")
    if (
        policy.config.encoder_size != target_config.encoder_size
        or policy.config.recurrent_size != target_config.recurrent_size
    ):
        raise ValueError("native and Arsenal policy hidden widths must match")
    target_normalization = {
        name: float(np.asarray(getattr(target_combat_params, name)))
        for name in (
            "agent_max_speed",
            "vertical_speed_scale",
            "target_chase_speed",
            "wire_fixed_point_scale",
            "agent_attack_pause_max_seconds",
        )
    }
    if target_normalization != {
        name: _NORMALIZATION[name] for name in target_normalization
    }:
        raise ValueError("target combat normalization differs from native shared view")

    input_kernel = (
        jnp.zeros_like(target_params.encoder_input.kernel)
        .at[jnp.asarray(dense_indices), :]
        .set(source_kernel)
    )
    # Unknown heads must not inherit a random target policy.  A finite neutral
    # prior starts them at choice zero while preserving gradients for later RL.
    actor_kernel = jnp.zeros_like(target_params.actor.kernel)
    actor_bias = jnp.zeros_like(target_params.actor.bias)
    for start, _ in HEAD_SPANS.values():
        actor_bias = actor_bias.at[start].set(_NEUTRAL_LOGIT_MARGIN)
    transferred = [
        name
        for name in NATIVE_ARSENAL_TRANSFERABLE_HEADS
        if name in policy.supervised_heads
    ]
    for name in transferred:
        start, size = HEAD_SPANS[name]
        span = slice(start, start + size)
        actor_kernel = actor_kernel.at[:, span].set(policy.params.actor.kernel[:, span])
        actor_bias = actor_bias.at[span].set(policy.params.actor.bias[span])
    ability_transfer = None
    ability_head = "ability_none_plus_slots"
    source_action_schema = policy.metadata.get("native_replay_action_schema")
    if ability_binding is not None and ability_head not in policy.supervised_heads:
        raise ValueError("ability binding supplied for an unsupervised native head")
    if ability_head in policy.supervised_heads and ability_binding is not None:
        if schema not in {
            NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA,
            NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
        }:
            raise ValueError(
                "role-local ability transfer requires a conditioned policy"
            )
        if style is None or style[0] != ability_binding.actor_role:
            raise ValueError("ability binding does not match the selected actor role")
        _validate_ability_binding(ability_binding)
        if (
            policy.metadata.get("native_replay_ability_slot_contract_sha256")
            != ability_binding.source_ability_slot_contract_sha256
        ):
            raise ValueError("native policy ability-slot contract mismatch")
        if (
            target_ability_slot_contract_sha256
            != ability_binding.target_ability_slot_contract_sha256
        ):
            raise ValueError("target JAX ability-slot contract mismatch")
        start, _ = HEAD_SPANS[ability_head]
        _, ability_size = HEAD_SPANS[ability_head]
        ability_span = slice(start, start + ability_size)
        actor_bias = actor_bias.at[ability_span].set(-_NEUTRAL_LOGIT_MARGIN)
        if source_action_schema == NATIVE_REPLAY_ACTION_SCHEMA:
            if not ability_binding.source_to_target_slots:
                raise ValueError("basic attack binding has no executable JAX root")
            choices = ((0, 0), (1, ability_binding.source_to_target_slots[0] + 1))
            projection = "basic_attack_onset_to_first_exact_role_root"
        else:
            choices = ((0, 0),) + tuple(
                (source + 1, target + 1)
                for source, target in zip(
                    ability_binding.source_slot_indices,
                    ability_binding.source_to_target_slots,
                    strict=True,
                )
            )
            projection = "exact_role_local_root_permutation"
        for source, target in choices:
            actor_kernel = actor_kernel.at[:, start + target].set(
                policy.params.actor.kernel[:, start + source]
            )
            actor_bias = actor_bias.at[start + target].set(
                policy.params.actor.bias[start + source]
            )
        transferred.append(ability_head)
        ability_transfer = {
            "schema": NATIVE_ARSENAL_ABILITY_BINDING_SCHEMA,
            "contract_sha256": native_arsenal_ability_binding_sha256(ability_binding),
            "actor_role": ability_binding.actor_role,
            "source_slot_indices": list(ability_binding.source_slot_indices),
            "source_to_target_slots": list(ability_binding.source_to_target_slots),
            "projection": projection,
            "copied_choices": [list(choice) for choice in choices],
            "target_ability_slot_contract_sha256": (
                ability_binding.target_ability_slot_contract_sha256
            ),
        }
    transferred = tuple(
        name for name in policy.supervised_heads if name in set(transferred)
    )
    params = target_params._replace(
        encoder_input=target_params.encoder_input._replace(
            kernel=input_kernel,
            bias=source_bias,
        ),
        encoder_hidden=policy.params.encoder_hidden,
        gru=policy.params.gru,
        actor=target_params.actor._replace(kernel=actor_kernel, bias=actor_bias),
    )
    blocked = tuple(name for name in policy.supervised_heads if name not in transferred)
    neutral = tuple(name for name in HEAD_SPANS if name not in transferred)
    return params, {
        "schema": NATIVE_ARSENAL_TRANSFER_SCHEMA,
        "shared_observation_schema": NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA,
        "shared_observation_sha256": NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256,
        "visual_shared_observation_sha256": (
            NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256
            if schema
            in {
                NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA,
                NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA,
            }
            else None
        ),
        "transferred_input_features": len(dense_indices),
        "normalization": target_normalization,
        "conditioning": conditioning,
        "source_checkpoint_sha256": policy.checkpoint_sha256,
        "transferred_heads": list(transferred),
        "blocked_heads": list(blocked),
        "neutral_initialized_heads": list(neutral),
        "neutral_initialization": {
            "choice": 0,
            "logit_margin": _NEUTRAL_LOGIT_MARGIN,
            "purpose": "trainable_abstention_until_rl_supplies_evidence",
        },
        "ability_binding": ability_transfer,
        "head_mapping": {
            "base_action": "Java Primary chain start -> Arsenal SKILL_ATTACK",
            "ability_none_plus_slots": (
                "causal Java basic-attack onset -> bound default JAX role root"
                if ability_transfer is not None
                and source_action_schema == NATIVE_REPLAY_ACTION_SCHEMA
                else "exact Java root identity -> bound JAX runtime root identity"
                if ability_transfer is not None
                else "blocked until the selected Java role-local slot layout is "
                "bound to an exact JAX runtime ability layout"
            ),
            "locomotion_gait_compass": "native body steering -> compass bin",
            "yaw_delta_bins": "native requested head yaw -> look-delta bin",
            "pitch_delta_bins": "native requested head pitch -> look-delta bin",
        },
        "blocked_reason": (
            "untransferred heads lack an exact native-to-runtime semantic map; "
            "Java ability labels require a bijective role-local root binding"
        ),
        "nonshared_input_initialization": "zero; PPO may learn after warm start",
    }


def _require_root_slots(value, name):
    try:
        slots = tuple(tuple(slot) for slot in value)
    except TypeError as exc:
        raise TypeError(f"{name} ability slots must be pairs") from exc
    if any(
        len(slot) != 2 or not all(isinstance(item, str) and item for item in slot)
        for slot in slots
    ):
        raise ValueError(f"{name} ability slots require root id and interaction type")
    if len(slots) > HEAD_SPANS["ability_none_plus_slots"][1] - 1:
        raise ValueError(f"{name} ability slots exceed the Arena head capacity")
    if len(set(slots)) != len(slots):
        raise ValueError(f"{name} ability roots must be unique")
    return slots


def _require_sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.upper()
        or any(character not in "0123456789ABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be an uppercase SHA-256")
    return value


def _validate_ability_binding(binding):
    if not isinstance(binding, NativeArsenalAbilityBinding):
        raise TypeError("ability binding has the wrong type")
    if not binding.actor_role:
        raise ValueError("ability binding actor role must be nonempty")
    source = _require_root_slots(binding.source_slots, "source")
    target = _require_root_slots(binding.target_slots, "target")
    source_indices = binding.source_slot_indices
    maximum = HEAD_SPANS["ability_none_plus_slots"][1] - 1
    if (
        len(source_indices) != len(source)
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in source_indices
        )
        or tuple(sorted(set(source_indices))) != source_indices
        or any(not 0 <= index < maximum for index in source_indices)
    ):
        raise ValueError("ability binding source slot indices are invalid")
    _require_sha256(
        binding.source_ability_slot_contract_sha256,
        "source ability-slot contract",
    )
    _require_sha256(
        binding.target_ability_slot_contract_sha256,
        "target ability-slot contract",
    )
    if binding.projection == "exact_root_injection":
        if not source or not set(source).issubset(target):
            raise ValueError("ability binding is not an exact root injection")
        expected = tuple(target.index(slot) for slot in source)
    elif binding.projection == "basic_attack_to_default_slot":
        expected = (0,)
        if not source or not target or target[0][1] != "Primary":
            raise ValueError("ability binding lacks a default primary attack")
    else:
        raise ValueError("ability binding projection is unknown")
    if binding.source_to_target_slots != expected:
        raise ValueError("ability binding slot projection is invalid")
    program = _require_sha256(
        binding.target_program_content_sha256,
        "target program content",
    )
    if binding.target_ability_slot_contract_sha256 != (
        native_arsenal_ability_slot_contract_sha256(target, program)
    ):
        raise ValueError("ability binding target slot contract is stale")


def _java_round(value: np.ndarray, scale: float) -> np.ndarray:
    return np.floor(value * np.float32(scale) + np.float32(0.5)) / np.float32(scale)


def jax_one_hot(index: int, count: int) -> jnp.ndarray:
    return jnp.arange(count, dtype=jnp.int32) == index


def _require_roles(roles: tuple[str, ...]) -> tuple[str, ...]:
    if not roles or len(set(roles)) != len(roles) or not all(roles):
        raise ValueError("native behavior style roles must be unique and nonempty")
    return roles


def _require_style(
    roles: tuple[str, ...], style: tuple[str, str] | None
) -> tuple[str, str]:
    if (
        not isinstance(style, tuple)
        or len(style) != 2
        or any(value not in roles for value in style)
    ):
        raise ValueError("native behavior style must name an actor and opponent role")
    return style


def _tensor_sha256(tensors: Any) -> str:
    digest = hashlib.sha256()
    for name, value in zip(tensors._fields, tensors, strict=True):
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest().upper()


__all__ = [
    "NATIVE_ARSENAL_ABILITY_BINDING_SCHEMA",
    "NATIVE_ARSENAL_ACTION_HEAD_READINESS_SCHEMA",
    "NATIVE_ARSENAL_ABILITY_SLOT_CONTRACT_SCHEMA",
    "NATIVE_ARSENAL_CONDITIONED_OBSERVATION_SCHEMA",
    "NATIVE_ARSENAL_SHARED_DENSE_INDICES",
    "NATIVE_ARSENAL_SHARED_FEATURE_LAYOUT",
    "NATIVE_ARSENAL_SHARED_OBSERVATION_SCHEMA",
    "NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256",
    "NATIVE_ARSENAL_SHARED_OBSERVATION_SIZE",
    "NATIVE_ARSENAL_VISUAL_CONDITIONED_OBSERVATION_SCHEMA",
    "NATIVE_ARSENAL_VISUAL_DENSE_INDICES",
    "NATIVE_ARSENAL_VISUAL_SHARED_DENSE_INDICES",
    "NATIVE_ARSENAL_VISUAL_SHARED_FEATURE_LAYOUT",
    "NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SCHEMA",
    "NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256",
    "NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SIZE",
    "NATIVE_ARSENAL_TRANSFERABLE_HEADS",
    "NATIVE_ARSENAL_TRANSFER_SCHEMA",
    "NativeArsenalAbilityBinding",
    "arsenal_conditioned_observation",
    "arsenal_shared_observation",
    "arsenal_visual_conditioned_observation",
    "arsenal_visual_shared_observation",
    "native_arsenal_conditioned_observation_sha256",
    "native_arsenal_ability_binding",
    "native_arsenal_ability_binding_sha256",
    "native_arsenal_action_head_readiness",
    "native_arsenal_ability_slot_contract_sha256",
    "native_arsenal_npc_role_ability_binding",
    "native_arsenal_profile_basic_attack_binding",
    "native_arsenal_conditioned_replay_corpus",
    "native_arsenal_visual_conditioned_observation_sha256",
    "native_arsenal_visual_conditioned_replay_corpus",
    "native_arsenal_visual_shared_replay_corpus",
    "native_behavior_observation_projector",
    "native_arsenal_shared_replay_corpus",
    "transplant_native_behavior_policy",
]
