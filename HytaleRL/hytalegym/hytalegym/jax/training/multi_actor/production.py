"""Production composition for flat multi-policy Arsenal training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalRuntimeCapacity,
    ArsenalRuntimeConfig,
)
from hytalegym.jax.combat.arsenal.environment import (
    ArsenalOpponentAbilityProvider,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_ACTION_SIZE,
    ARSENAL_STANDARD_ROOT_DISTRIBUTION,
    arsenal_policy_observation_size,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    WorldGeometryPolicyConfig,
)
from hytalegym.jax.combat.types import CombatParams
from hytalegym.jax.training.types import PPOConfig, RecurrentPolicyParams

from .assignment import (
    PolicyActorAssignment,
    single_policy_actor_assignment,
    validate_policy_actor_assignment,
)
from .checkpoint.provenance import (
    RuntimeTrainingEntitySpec,
    arsenal_runtime_config_content_sha256,
    attest_policy_training_contexts,
)
from .population_training import (
    MultiActorPopulationTrainer,
    MultiActorPopulationTrainingState,
    make_multi_actor_population_trainer,
    plan_multi_actor_population,
)
from .rollout import (
    ArenaCapabilityProvider,
    ArenaResetProvider,
    make_multi_actor_rollout_collector,
)
from .runtime import MultiActorArenaState, initialize_multi_actor_arena_state


@dataclass(frozen=True)
class FlatCombatPopulationEntryPoint:
    """Complete flat-arena reset, rollout, optimizer, and checkpoint seam."""

    params: CombatParams
    runtime_config: ArsenalRuntimeConfig
    assignment: PolicyActorAssignment
    ppo_config: PPOConfig
    reset_provider: ArenaResetProvider
    capability_provider: ArenaCapabilityProvider
    opponent_ability_provider: ArsenalOpponentAbilityProvider
    collector: Any
    trainer: MultiActorPopulationTrainer
    entity_count: int
    training_entity_specs: tuple[RuntimeTrainingEntitySpec, ...]
    policy_training_contexts: tuple[dict[str, Any], ...]
    policy_training_context_attestations: tuple[dict[str, Any], ...]
    runtime_config_content_sha256: str

    def initialize(
        self,
        policy_bank: RecurrentPolicyParams,
        key: jax.Array,
    ) -> MultiActorPopulationTrainingState:
        """Reset every arena and initialize isolated optimizer ownership."""

        bank_size = validate_population_policy_bank(policy_bank, self.ppo_config)
        plan_multi_actor_population(
            self.assignment,
            self.ppo_config,
            bank_size=bank_size,
        )
        arena = self._reset_arena(key)
        return self.trainer.initialize(policy_bank, arena)

    def train_step(
        self,
        state: MultiActorPopulationTrainingState,
        key: jax.Array,
    ):
        """Run one actor-major rollout and one update per trainable policy."""

        return self.trainer.train_step(state, key)

    def save_checkpoint(
        self,
        path: str | Path,
        state: MultiActorPopulationTrainingState,
        *,
        selected_policy_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        arsenal_runtime_capacity: (
            ArsenalRuntimeCapacity | Mapping[str, int] | None
        ) = None,
        world_geometry_config: (
            WorldGeometryPolicyConfig | Mapping[str, object] | None
        ) = None,
    ) -> Path:
        """Save the portable v1 population state at an update boundary."""

        from .checkpoint import save_population_checkpoint_v1

        return save_population_checkpoint_v1(
            path,
            state,
            self.assignment,
            self.ppo_config,
            self.trainer,
            entity_count=self.entity_count,
            selected_policy_id=selected_policy_id,
            metadata=metadata,
            arsenal_runtime_capacity=arsenal_runtime_capacity,
            world_geometry_config=world_geometry_config,
            runtime_config=self.runtime_config,
            training_entity_specs=self.training_entity_specs,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        key: jax.Array,
        *,
        arsenal_runtime_capacity: (
            ArsenalRuntimeCapacity | Mapping[str, int] | None
        ) = None,
        world_geometry_config: (
            WorldGeometryPolicyConfig | Mapping[str, object] | None
        ) = None,
    ):
        """Restore bank/optimizers/counters onto one freshly reset arena."""

        from .checkpoint import load_population_checkpoint_v1

        return load_population_checkpoint_v1(
            path,
            self.trainer,
            self._reset_arena(key),
            expected_assignment=self.assignment,
            expected_config=self.ppo_config,
            entity_count=self.entity_count,
            arsenal_runtime_capacity=arsenal_runtime_capacity,
            world_geometry_config=world_geometry_config,
            expected_runtime_config=self.runtime_config,
            expected_training_entity_specs=self.training_entity_specs,
        )

    def _reset_arena(self, key: jax.Array) -> MultiActorArenaState:
        batch = self.assignment.actor_index.shape[0]
        reset_state = self.reset_provider(jax.random.split(key, batch))
        expected = (batch, self.entity_count)
        if reset_state.combat.health.shape != expected:
            raise ValueError(
                "reset_provider combat entity shape differs from assignment: "
                f"{reset_state.combat.health.shape} != {expected}"
            )
        return initialize_multi_actor_arena_state(reset_state, self.params)


def make_flat_combat_population_entrypoint(
    params: CombatParams,
    runtime_config: ArsenalRuntimeConfig,
    assignment: PolicyActorAssignment,
    ppo_config: PPOConfig,
    *,
    reset_provider: ArenaResetProvider,
    capability_provider: ArenaCapabilityProvider,
    opponent_ability_provider: ArsenalOpponentAbilityProvider | None = None,
    training_entity_specs: tuple[RuntimeTrainingEntitySpec, ...] | None = None,
    compile: bool = True,
) -> FlatCombatPopulationEntryPoint:
    """Compose the shipped flat collector with independent policy optimizers."""

    if int(params.microticks) != 1:
        raise ValueError("flat multi-actor population v1 requires microticks=1")
    if runtime_config.loadout.weapon_id.ndim != 2:
        raise ValueError("runtime loadout must have shape [B,N,...]")
    batch, entity_count = runtime_config.loadout.weapon_id.shape
    validate_policy_actor_assignment(assignment, entity_count=entity_count)
    if assignment.actor_index.shape[0] != batch:
        raise ValueError("assignment batch does not match runtime config")
    if training_entity_specs is None:
        normalized_training_specs: tuple[RuntimeTrainingEntitySpec, ...] = ()
        policy_training_contexts: tuple[dict[str, Any], ...] = ()
        policy_training_context_attestations: tuple[dict[str, Any], ...] = ()
        runtime_config_sha256 = arsenal_runtime_config_content_sha256(
            runtime_config
        )
    else:
        normalized_training_specs = tuple(training_entity_specs)
        (
            policy_training_contexts,
            policy_training_context_attestations,
            runtime_config_sha256,
        ) = attest_policy_training_contexts(
            runtime_config,
            assignment,
            normalized_training_specs,
        )
    _validate_arsenal_ppo_surface(ppo_config)
    if opponent_ability_provider is None:
        from hytalegym.jax.combat.opponents.runtime.policy import (
            first_legal_opponent_ability_slots,
        )

        opponent_ability_provider = first_legal_opponent_ability_slots
    collector = make_multi_actor_rollout_collector(
        params,
        runtime_config,
        assignment,
        reset_provider=reset_provider,
        capability_provider=capability_provider,
        rollout_steps=ppo_config.rollout_steps,
        opponent_ability_provider=opponent_ability_provider,
    )
    trainer = make_multi_actor_population_trainer(
        collector,
        assignment,
        ppo_config,
        compile=compile,
    )
    return FlatCombatPopulationEntryPoint(
        params=params,
        runtime_config=runtime_config,
        assignment=assignment,
        ppo_config=ppo_config,
        reset_provider=reset_provider,
        capability_provider=capability_provider,
        opponent_ability_provider=opponent_ability_provider,
        collector=collector,
        trainer=trainer,
        entity_count=entity_count,
        training_entity_specs=normalized_training_specs,
        policy_training_contexts=policy_training_contexts,
        policy_training_context_attestations=(
            policy_training_context_attestations
        ),
        runtime_config_content_sha256=runtime_config_sha256,
    )


def make_single_actor_flat_combat_population_entrypoint(
    params: CombatParams,
    runtime_config: ArsenalRuntimeConfig,
    ppo_config: PPOConfig,
    *,
    reset_provider: ArenaResetProvider,
    capability_provider: ArenaCapabilityProvider,
    opponent_ability_provider: ArsenalOpponentAbilityProvider | None = None,
    actor_index: int = 0,
    policy_id: int = 0,
    trainable: bool = True,
    training_entity_specs: tuple[RuntimeTrainingEntitySpec, ...] | None = None,
    compile: bool = True,
) -> FlatCombatPopulationEntryPoint:
    """Build the legacy actor-zero path as the exact ``P=1`` population case."""

    batch, entity_count = runtime_config.loadout.weapon_id.shape
    if batch != ppo_config.num_envs:
        raise ValueError(
            "single-policy runtime batch must equal PPO num_envs: "
            f"{batch} != {ppo_config.num_envs}"
        )
    assignment = single_policy_actor_assignment(
        batch,
        actor_index=actor_index,
        policy_id=policy_id,
        trainable=trainable,
        entity_count=entity_count,
    )
    return make_flat_combat_population_entrypoint(
        params,
        runtime_config,
        assignment,
        ppo_config,
        reset_provider=reset_provider,
        capability_provider=capability_provider,
        opponent_ability_provider=opponent_ability_provider,
        training_entity_specs=training_entity_specs,
        compile=compile,
    )


def validate_population_policy_bank(
    policy_bank: RecurrentPolicyParams,
    config: PPOConfig,
) -> int:
    """Validate one static ``K``-row recurrent policy bank and return ``K``."""

    named = {
        "encoder_input.kernel": policy_bank.encoder_input.kernel,
        "encoder_input.bias": policy_bank.encoder_input.bias,
        "encoder_hidden.kernel": policy_bank.encoder_hidden.kernel,
        "encoder_hidden.bias": policy_bank.encoder_hidden.bias,
        "gru.input_kernel": policy_bank.gru.input_kernel,
        "gru.recurrent_kernel": policy_bank.gru.recurrent_kernel,
        "gru.bias": policy_bank.gru.bias,
        "actor.kernel": policy_bank.actor.kernel,
        "actor.bias": policy_bank.actor.bias,
        "critic.kernel": policy_bank.critic.kernel,
        "critic.bias": policy_bank.critic.bias,
    }
    first = next(iter(named.values()))
    if first.ndim < 1 or first.shape[0] < 1:
        raise ValueError("policy bank must have a nonempty leading K axis")
    bank_size = int(first.shape[0])
    expected = {
        "encoder_input.kernel": (config.observation_size, config.encoder_size),
        "encoder_input.bias": (config.encoder_size,),
        "encoder_hidden.kernel": (config.encoder_size, config.encoder_size),
        "encoder_hidden.bias": (config.encoder_size,),
        "gru.input_kernel": (config.encoder_size, 3 * config.recurrent_size),
        "gru.recurrent_kernel": (
            config.recurrent_size,
            3 * config.recurrent_size,
        ),
        "gru.bias": (3 * config.recurrent_size,),
        "actor.kernel": (config.recurrent_size, config.action_size),
        "actor.bias": (config.action_size,),
        "critic.kernel": (config.recurrent_size, 1),
        "critic.bias": (1,),
    }
    for name, value in named.items():
        if value.shape != (bank_size,) + expected[name]:
            raise ValueError(
                f"policy bank {name} has shape {value.shape}, expected "
                f"{(bank_size,) + expected[name]}"
            )
        if value.dtype != jnp.dtype(jnp.float32):
            raise ValueError(f"policy bank {name} must be float32")
    return bank_size


def _validate_arsenal_ppo_surface(config: PPOConfig) -> None:
    expected = {
        "observation_size": arsenal_policy_observation_size(),
        "action_size": ARSENAL_POLICY_ACTION_SIZE,
        "action_head_sizes": tuple(ARSENAL_POLICY_ACTION_HEAD_SIZES),
        "action_transport": "factors",
        "action_distribution": ARSENAL_STANDARD_ROOT_DISTRIBUTION,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(
                f"flat population PPO {name} must match current Arsenal: "
                f"{getattr(config, name)!r} != {value!r}"
            )


__all__ = [
    "FlatCombatPopulationEntryPoint",
    "make_flat_combat_population_entrypoint",
    "make_single_actor_flat_combat_population_entrypoint",
    "validate_population_policy_bank",
]
