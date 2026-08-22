"""Frozen learner-facing arsenal observation and action surface."""

# ruff: noqa: F401,F403

from hytalegym.jax.combat.observation.v3.policy.surface import (
    ACTION_SURFACE_STAGING_SCHEMA,
    ACTION_SURFACE_STAGING_VERSION,
    ActionSurfaceLayout,
    StagedActionSurfaceDecode,
    action_surface_layout,
    action_surface_staging_contract_manifest,
    action_surface_staging_contract_sha256,
    decode_staged_action_surface_factors,
    staged_action_surface_mask,
)
from hytalegym.jax.combat.observation.v3.policy.actions import (
    decode_learner_arsenal_action,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.blocks import (
    BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE,
    BLOCK_AFFORDANCE_POLICY_FEATURE_SIZE,
    BlockActionCandidatePolicyView,
    BlockActionCandidateSelection,
    BlockAffordancePolicyTokens,
    block_affordance_policy_staging_contract_manifest,
    block_affordance_policy_staging_contract_sha256,
    empty_block_action_candidate_policy_view,
    empty_block_affordance_policy_tokens,
    encode_block_action_candidate_policy_view,
    encode_block_affordance_policy_tokens,
    mask_block_action_candidate_policy_view,
    mask_block_affordance_policy_tokens,
    resolve_block_action_candidate,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.contract import (
    RECIPE_CANDIDATE_EMBEDDING_SIZE,
    RECIPE_CANDIDATE_ENCODING_SCHEMA,
    RECIPE_CANDIDATE_ENCODING_VERSION,
    recipe_candidate_encoding_contract_manifest,
    recipe_candidate_encoding_contract_sha256,
    recipe_candidate_encoding_parameter_seed,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
    RECIPE_CANDIDATE_ENCODER_STAGING_SCHEMA,
    RECIPE_CANDIDATE_ENCODER_STAGING_VERSION,
    RecipeCandidateEncoderConfig,
    RecipeCandidateEncoderParams,
    RecipeCandidateEncoding,
    encode_recipe_candidates,
    initialize_recipe_candidate_encoder,
    recipe_candidate_encoder_staging_contract_manifest,
    recipe_candidate_encoder_staging_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.schema.contract import *
from hytalegym.jax.combat.observation.v3.encoding.encoder import (
    encode_learner_observation_v3,
    encode_learner_observation_v3_from_actor_evidence,
    project_learner_observation_v3_actor_evidence,
)
from hytalegym.jax.combat.observation.v3.encoding.differential import (
    arsenal_policy_action_group_slices,
    arsenal_policy_observation_group_slices,
    dense_row_differential,
)
from hytalegym.jax.combat.observation.v3.tokens.inventory import *
from hytalegym.jax.combat.observation.v3.tokens.light import *
from hytalegym.jax.combat.observation.v3.native.channels.light import (
    NativePerceptionCapture,
    capture_native_actor_light_policy_tokens,
)
from hytalegym.jax.combat.observation.v3.native.channels.events import (
    NativeDamageEventBatch,
    NativeDamageTotals,
    native_damage_totals,
    pack_native_damage_events,
)
from hytalegym.jax.combat.observation.v3.native.channels.lifecycle import (
    NATIVE_LIFECYCLE_KIND_ID,
    NATIVE_LIFECYCLE_ORIGIN_ID,
    NativeLifecycleEventBatch,
    native_lifecycle_kind_counts,
    native_lifecycle_symbol_id,
    native_lifecycle_symbol_table,
    pack_native_lifecycle_events,
)
from hytalegym.jax.combat.observation.v3.native.channels.group import (
    NativeGroupStepBatch,
    NativeGroupStepEvidence,
    pack_native_group_step_evidence,
)
from hytalegym.jax.combat.observation.v3.native.channels.actor_major import (
    NativeActorMajorCapabilities,
    NativeActorMajorDecision,
    NativeActorMajorHostComposer,
    NativeActorMajorObservation,
    NativeActorMajorRow,
    NativeActorMajorSession,
    NativeActorMajorStepResult,
    actor_first_order,
    native_actor_major_reset_options,
)
from hytalegym.jax.combat.observation.v3.native.codec.transport import (
    NATIVE_MOVEMENT_STATE_SCHEMA,
    NATIVE_MOVEMENT_STATE_VERSION,
    NATIVE_POLICY_ACTION_PROTOCOL_VERSION,
    NATIVE_WORLD_VERB_LIFECYCLE_CONTRACT_SHA256,
    NATIVE_WORLD_VERB_LIFECYCLE_SCHEMA,
    NATIVE_WORLD_VERB_LIFECYCLE_VERSION,
    NativeActionTranslation,
    NativeMovementStateEvidence,
    NativePolicyCapabilities,
    decode_native_movement_state_evidence,
    summarize_native_action_accounting,
    translate_learner_arsenal_action_to_native,
)
from hytalegym.jax.combat.observation.v3.policy import *
from hytalegym.jax.combat.observation.v3.native.evidence import (
    NATIVE_ACTOR_EVIDENCE_SCHEMA,
    NATIVE_ACTOR_EVIDENCE_UNKNOWN_STATUS_FAILURE,
    NATIVE_ACTOR_EVIDENCE_VERSION,
    NativeActorEvidenceAssembler,
    NativeActorEvidenceAssembly,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.recipes import (
    RECIPE_CANDIDATE_SOURCE_POLICY_FIELDS,
    RecipeCandidatePolicyView,
    RecipeCandidateSelection,
    empty_recipe_candidate_policy_view,
    encode_recipe_candidate_policy_view,
    mask_recipe_candidate_policy_view,
    recipe_candidate_policy_staging_contract_manifest,
    recipe_candidate_policy_staging_contract_sha256,
    resolve_recipe_candidate,
)
from hytalegym.jax.combat.observation.v3.encoding.runtime import (
    reset_learner_arsenal_batch,
    step_learner_arsenal_batch,
)
from hytalegym.jax.combat.observation.v3.schema.spec import (
    learner_observation_v3_actor_evidence_contract_json,
    learner_observation_v3_actor_evidence_contract_manifest,
    learner_observation_v3_actor_evidence_contract_sha256,
    learner_observation_v3_contract_json,
    learner_observation_v3_contract_manifest,
    learner_observation_v3_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.schema.types import *
from hytalegym.jax.combat.observation.v3.tokens.world import *

__all__ = [name for name in globals() if not name.startswith("_")]

# Keep historical module imports working while implementations live in focused branches.
# New code should use the organized paths documented in DEV.md.
import sys as _sys

_legacy_module_targets = {
    "_decode": "native.codec.decode",
    "action_surface": "policy.surface",
    "actions": "policy.actions",
    "block_affordances": "policy.candidates.blocks",
    "candidate_encoder": "policy.candidates.encoder",
    "candidate_encoding": "policy.candidates.contract",
    "contract": "schema.contract",
    "differential": "encoding.differential",
    "encoder": "encoding.encoder",
    "inventory_tokens": "tokens.inventory",
    "light_policy_tokens": "tokens.light",
    "native_evidence": "native.evidence",
    "native_light_policy": "native.channels.light",
    "native_transport": "native.codec.transport",
    "recipe_candidates": "policy.candidates.recipes",
    "runtime": "encoding.runtime",
    "spec": "schema.spec",
    "types": "schema.types",
    "world_tokens": "tokens.world",
}
for _legacy_name, _target_name in _legacy_module_targets.items():
    _module = _sys.modules[f"{__name__}.{_target_name}"]
    _sys.modules.setdefault(f"{__name__}.{_legacy_name}", _module)
    setattr(_sys.modules[__name__], _legacy_name, _module)
del _legacy_name, _legacy_module_targets, _module, _sys, _target_name
