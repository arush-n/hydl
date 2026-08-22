"""Combat-owned action half of native block interaction chains."""

from hytalegym.jax.combat.block_interactions.schema.contract import *  # noqa: F403
from hytalegym.jax.combat.block_interactions.adapters.native import (
    CompiledNativeBlockInteraction,
    block_interaction_program_from_native_node,
)
from hytalegym.jax.combat.block_interactions.execution.runtime import (
    apply_interaction_movement_constraints,
    empty_block_interaction_command,
    empty_block_interaction_evidence,
    empty_block_interaction_program,
    empty_block_interaction_state,
    empty_resolved_block_drops,
    interaction_movement_constraints,
    step_block_interactions,
)
from hytalegym.jax.combat.block_interactions.schema.types import (
    BlockInteractionCommand,
    BlockInteractionEvidence,
    BlockInteractionInfo,
    BlockInteractionProgram,
    BlockInteractionState,
    InteractionMovementConstraints,
    ResolvedBlockDrops,
)
from hytalegym.jax.combat.block_interactions.adapters.world_binding import (
    BLOCK_INTERACTION_WORLD_BINDING_SCHEMA as BLOCK_INTERACTION_WORLD_BINDING_SCHEMA,
    BLOCK_INTERACTION_WORLD_BINDING_VERSION as BLOCK_INTERACTION_WORLD_BINDING_VERSION,
    BLOCK_INTERACTION_WORLD_BINDING_DIAGNOSTIC_SELECTION as BLOCK_INTERACTION_WORLD_BINDING_DIAGNOSTIC_SELECTION,
    BlockInteractionWorldInputs,
    bind_world_placement_support,
    bind_world_removal_cascade_safety,
    bind_world_block_interaction_inputs,
    block_interaction_world_binding_contract,
    block_interaction_world_binding_contract_sha256,
    select_actor_block_action_target,
    step_block_interactions_from_world,
)


__all__ = [
    "BlockInteractionCommand",
    "CompiledNativeBlockInteraction",
    "BlockInteractionEvidence",
    "BlockInteractionInfo",
    "BlockInteractionProgram",
    "BlockInteractionState",
    "BlockInteractionWorldInputs",
    "InteractionMovementConstraints",
    "ResolvedBlockDrops",
    "apply_interaction_movement_constraints",
    "bind_world_placement_support",
    "bind_world_removal_cascade_safety",
    "bind_world_block_interaction_inputs",
    "block_interaction_program_from_native_node",
    "block_interaction_world_binding_contract",
    "block_interaction_world_binding_contract_sha256",
    "empty_block_interaction_command",
    "empty_block_interaction_evidence",
    "empty_block_interaction_program",
    "empty_block_interaction_state",
    "empty_resolved_block_drops",
    "interaction_movement_constraints",
    "select_actor_block_action_target",
    "step_block_interactions",
    "step_block_interactions_from_world",
]
__all__ += [
    name
    for name in globals()
    if name.isupper() or name.startswith("block_interaction_contract_")
]
