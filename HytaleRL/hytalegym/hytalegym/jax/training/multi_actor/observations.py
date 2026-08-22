"""Actor-major consumption of legally projected learner-v3 evidence."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.encoding.encoder import (
    encode_learner_observation_v3_from_actor_evidence,
    project_learner_observation_v3_actor_evidence,
)
from hytalegym.jax.combat.observation import empty_injected_world_features
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalEnvironmentState,
    ArsenalRuntimeConfig,
    ArsenalWorldCapabilities,
)
from hytalegym.jax.combat.arsenal.environment import (
    ArsenalActionSurfaceEvidence,
)
from hytalegym.jax.combat.block_interactions import (
    InteractionMovementConstraints,
)
from hytalegym.jax.combat.types import CombatParams
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerCombatObservationV3,
    LearnerObservationV3ActorEvidence,
)
from hytalegym.jax.combat.observation.v3.policy import (
    arsenal_policy_action_mask,
    arsenal_policy_observation,
)
from hytalegym.jax.combat.observation.v3.tokens.inventory import (
    InventoryPolicyTokens,
    inventory_policy_tokens_from_state,
)
from hytalegym.jax.combat.observation.v3.tokens.light import (
    ActorLightPolicyTokens,
)

from .assignment import PolicyActorAssignment
from .locomotion import EntityLocomotionState
from .views import actor_first_arsenal_view


class PolicyActorObservation(NamedTuple):
    """Actor-major structured and dense inputs for flat shared arenas."""

    evidence: LearnerObservationV3ActorEvidence
    structured: LearnerCombatObservationV3
    inventory: InventoryPolicyTokens
    dense: jax.Array
    action_mask: jax.Array


def project_policy_actor_evidence_flat(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    capabilities: ArsenalWorldCapabilities,
    config: ArsenalRuntimeConfig,
    locomotion: EntityLocomotionState,
    assignment: PolicyActorAssignment,
) -> LearnerObservationV3ActorEvidence:
    """Project legal actor-major evidence for bounded flat shared arenas.

    Each policy slot receives an actor-first, entity-ID-remapped view and then
    uses the unchanged learner-v3 projector. Actor-world scalar fields remain
    unavailable for nonzero actors because the current World producer is
    actor-zero-only. Region geometry/tokens must gain an actor-major producer
    before this boundary is widened beyond flat shared arenas.
    """

    batch, entity_count = state.combat.health.shape
    actor_shape = assignment.actor_index.shape
    if actor_shape[0] != batch:
        raise ValueError("assignment batch does not match arena state")
    if any(
        leaf.shape[:2] != (batch, entity_count)
        for leaf in jax.tree_util.tree_leaves(locomotion)
    ):
        raise ValueError("locomotion leaves must begin with [B,N]")
    rows = []
    for slot in range(actor_shape[1]):
        actor = jnp.where(
            assignment.active[:, slot],
            assignment.actor_index[:, slot],
            jnp.int32(0),
        )
        actor_state, actor_config, actor_capabilities = actor_first_arsenal_view(
            state,
            config,
            capabilities,
            locomotion,
            actor,
        )
        row = project_learner_observation_v3_actor_evidence(
            actor_state,
            params,
            empty_injected_world_features(batch),
            actor_capabilities,
            actor_config,
        )
        if actor_shape[1] > 1:
            row = _mask_non_actor_entity_rows(row)
        rows.append(row)
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values, axis=1), *rows)


def _mask_non_actor_entity_rows(
    evidence: LearnerObservationV3ActorEvidence,
) -> LearnerObservationV3ActorEvidence:
    """Remove private nonactor resource/status/loadout rows after projection."""

    entity_count = evidence.weapon_i32.shape[1]
    keep = jnp.arange(entity_count, dtype=jnp.int32)[None, :] == 0

    def mask(value):
        selector = keep.reshape(keep.shape + (1,) * (value.ndim - 2))
        return jnp.where(selector, value, jnp.zeros_like(value))

    return evidence._replace(
        weapon_i32=mask(evidence.weapon_i32),
        resource_f32=mask(evidence.resource_f32),
        resource_mask=mask(evidence.resource_mask),
        defense_f32=mask(evidence.defense_f32),
        status_f32=mask(evidence.status_f32),
        status_i32=mask(evidence.status_i32),
        status_flags=mask(evidence.status_flags),
        status_mask=mask(evidence.status_mask),
        ability_f32=mask(evidence.ability_f32),
        ability_i32=mask(evidence.ability_i32),
        ability_mask=mask(evidence.ability_mask),
        ability_legal=mask(evidence.ability_legal),
    )


def encode_policy_actor_observations(
    evidence: LearnerObservationV3ActorEvidence,
    assignment: PolicyActorAssignment,
) -> LearnerCombatObservationV3:
    """Encode legal evidence rows shaped ``[B,P,...]`` without row mirroring.

    This function is deliberately downstream of evidence projection. It gives
    every actor slot the exact shipped v3 encoder while keeping privileged
    JAX-state projection out of this package. Inactive actor slots are masked
    to all-zero, invalid observations with no legal action.
    """

    if not isinstance(evidence, LearnerObservationV3ActorEvidence):
        raise TypeError("evidence must be LearnerObservationV3ActorEvidence")
    actor_shape = assignment.actor_index.shape
    leaves = jax.tree_util.tree_leaves(evidence)
    if not leaves or any(
        leaf.ndim < 2 or leaf.shape[:2] != actor_shape for leaf in leaves
    ):
        raise ValueError("every actor-evidence leaf must begin with [B,P]")
    flat_count = actor_shape[0] * actor_shape[1]
    flat_evidence = jax.tree_util.tree_map(
        lambda leaf: leaf.reshape((flat_count,) + leaf.shape[2:]),
        evidence,
    )
    flat_observation = encode_learner_observation_v3_from_actor_evidence(
        flat_evidence
    )
    observation = jax.tree_util.tree_map(
        lambda leaf: leaf.reshape(actor_shape + leaf.shape[1:]),
        flat_observation,
    )
    active = assignment.active
    masked = jax.tree_util.tree_map(
        lambda leaf: jnp.where(
            active.reshape(actor_shape + (1,) * (leaf.ndim - 2)),
            leaf,
            jnp.zeros_like(leaf),
        ),
        observation,
    )
    return masked._replace(valid=observation.valid & active)


def observe_policy_actors_flat(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    capabilities: ArsenalWorldCapabilities,
    config: ArsenalRuntimeConfig,
    locomotion: EntityLocomotionState,
    assignment: PolicyActorAssignment,
) -> PolicyActorObservation:
    """Build complete legal policy inputs for a bounded flat shared arena.

    World-action candidate surfaces stay closed here. Their mutable Region
    state is not actor-major yet, so exposing those heads would claim a
    producer this adapter does not own. Combat, look, movement, defense, and
    semantic inventory all use the shipped learner-v3/policy encoders.
    """

    evidence = project_policy_actor_evidence_flat(
        state,
        params,
        capabilities,
        config,
        locomotion,
        assignment,
    )
    structured = encode_policy_actor_observations(evidence, assignment)
    inventory_rows = []
    for slot in range(assignment.actor_index.shape[1]):
        actor = jnp.where(
            assignment.active[:, slot],
            assignment.actor_index[:, slot],
            jnp.int32(0),
        )
        actor_state, _, _ = actor_first_arsenal_view(
            state,
            config,
            capabilities,
            locomotion,
            actor,
        )
        inventory_rows.append(
            inventory_policy_tokens_from_state(
                actor_state.inventory,
                config.inventory_layout,
                actor_valid=structured.valid[:, slot],
            )
        )
    inventory = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values, axis=1),
        *inventory_rows,
    )
    return _assemble_policy_actor_observation(
        evidence,
        structured,
        inventory,
        assignment,
    )


def _assemble_policy_actor_observation(
    evidence: LearnerObservationV3ActorEvidence,
    structured: LearnerCombatObservationV3,
    inventory: InventoryPolicyTokens,
    assignment: PolicyActorAssignment,
    *,
    action_surface: ArsenalActionSurfaceEvidence | None = None,
    light_tokens: ActorLightPolicyTokens | None = None,
    interaction_movement: InteractionMovementConstraints | None = None,
) -> PolicyActorObservation:
    """Flatten actor-major legal evidence through the shipped policy surface."""

    batch, policy_slots = assignment.actor_index.shape
    flat_count = batch * policy_slots

    def flatten(value):
        return value.reshape((flat_count,) + value.shape[2:])

    flat_structured = jax.tree_util.tree_map(flatten, structured)
    flat_inventory = jax.tree_util.tree_map(flatten, inventory)
    flat_surface = (
        None
        if action_surface is None
        else jax.tree_util.tree_map(flatten, action_surface)
    )
    flat_light = (
        None if light_tokens is None else jax.tree_util.tree_map(flatten, light_tokens)
    )
    flat_movement = (
        None
        if interaction_movement is None
        else jax.tree_util.tree_map(flatten, interaction_movement)
    )
    dense = arsenal_policy_observation(
        flat_structured,
        None if flat_surface is None else flat_surface.block_candidates,
        None if flat_surface is None else flat_surface.recipe_encoding,
        inventory_tokens=flat_inventory,
        light_tokens=flat_light,
    ).reshape((batch, policy_slots, -1))
    action_mask = arsenal_policy_action_mask(
        flat_structured,
        None if flat_surface is None else flat_surface.block_candidates,
        None if flat_surface is None else flat_surface.recipe_candidates,
        use_available=(
            None if flat_surface is None else flat_surface.use_available
        ),
        block_trigger_available=(
            None
            if flat_surface is None
            else flat_surface.block_trigger_available
        ),
        interaction_movement=flat_movement,
    ).reshape((batch, policy_slots, -1))
    active = assignment.active & structured.valid
    dense = jnp.where(active[..., None], dense, jnp.float32(0.0))
    action_mask = jnp.where(active[..., None], action_mask, False)
    return PolicyActorObservation(
        evidence=evidence,
        structured=structured,
        inventory=inventory,
        dense=dense,
        action_mask=action_mask,
    )


__all__ = [
    "encode_policy_actor_observations",
    "observe_policy_actors_flat",
    "PolicyActorObservation",
    "project_policy_actor_evidence_flat",
]
