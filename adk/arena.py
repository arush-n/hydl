"""One-way ADK access to Arena's direct JAX tasks and training tools.

Arena remains framework-independent: this module imports Arena only after an
agent explicitly asks for ``handle.arena`` functionality.  ADK policies and
live action contracts are adapted here; Arena never imports ADK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ArenaPolicy:
    """An ADK policy plus the recurrent carry Arena starts each rollout with."""

    policy: Callable
    initial_carry: Any = None

    def __post_init__(self) -> None:
        if not callable(self.policy):
            raise TypeError("policy must satisfy the ADK Policy protocol")

    def __call__(self, carry: Any, actor_input: Any, key: Any):
        return self.policy(carry, actor_input, key)


@dataclass(frozen=True, slots=True)
class ArenaTools:
    """Arena tasks and reusable techniques bound to one :class:`AgentHandle`."""

    handle: Any

    @property
    def tasks(self):
        """Arena's task-authoring and conformance framework."""

        import arena.tasks as tasks

        return tasks

    @property
    def games(self):
        """Authored Arena games, including the WorldGen V2 task ports."""

        import arena.tasks.games as games

        return games

    @property
    def training(self):
        """JAX PPO, learned rewards, imitation, self-play, and evolution."""

        import arena.training as training

        return training

    @property
    def imitation(self):
        """Algorithm-neutral trace contracts for IL and world-model training."""

        import arena.imitation as imitation

        return imitation

    @property
    def selfplay(self):
        """ADK's immutable population and multi-actor self-play transport."""

        import adk.selfplay as selfplay

        return selfplay

    @property
    def techniques(self) -> tuple[str, ...]:
        return tuple(self.training.TECHNIQUES)

    def describe(self) -> dict[str, Any]:
        """Return the exact agent-facing Arena contract."""

        self._require_task_contract()
        return {
            "engine": "hytalegym.jax",
            "dependency_direction": "adk -> arena",
            "action_head_names": list(self.handle.action_head_names),
            "action_head_sizes": list(self.handle.action_head_sizes),
            "techniques": list(self.techniques),
            "trace_contract": "arena_composable_transition_trace_v3",
            "trace_layers": [
                "omniscient",
                "internal",
                "external",
                "decision",
                "transition",
                "extension",
            ],
            "world_generation": "worldgen-v2",
        }

    def manager(
        self,
        *games: Any,
        build_scenes: bool = True,
        cache_limit: int | None = None,
        component_steps: int = 3,
    ):
        """Create a manager and register only the requested games/components."""

        from arena.manager import GameManager

        self._require_task_contract()
        manager = GameManager(cache_limit=cache_limit)
        for game in games:
            manager.register(
                game,
                build_scenes=build_scenes,
                steps=component_steps,
            )
        return manager

    def training_scene(self, task: Any, difficulty: str, *, batch: int | None = None):
        """Build the exact rewarded JAX scene an ADK agent will train in."""

        self._require_task_contract()
        lanes = getattr(self.handle, "batch", None) if batch is None else batch
        if lanes is None:
            raise ValueError("batch is required when the ADK handle has no batch")
        return task.build_scene(difficulty, batch=lanes)

    def ppo_config(self, scene: Any, **overrides: Any):
        """Derive recurrent PPO shapes from an Arena scene, never constants."""

        self._require_task_contract()
        return self.training.ppo_config(scene, **overrides)

    def bind_policy(
        self, policy: Callable, *, initial_carry: Any = None
    ) -> ArenaPolicy:
        """Attach Arena's initial-carry convention to any ADK policy."""

        self._require_task_contract()
        return ArenaPolicy(policy, initial_carry)

    def recurrent_policy(
        self,
        parameters: Any,
        *,
        batch: int | None = None,
        decode_mode: str = "factored_argmax",
    ):
        """Adapt recurrent weights without recompiling for every checkpoint."""

        from arena.jax_env import ParameterizedPolicy

        self._require_task_contract()
        parameters = getattr(parameters, "policy_params", parameters)
        _, carry = self.handle.recurrent_policy(
            parameters,
            batch=batch,
            decode_mode=decode_mode,
        )

        def apply(candidate, state, actor_input, key):
            policy, _ = self.handle.recurrent_policy(
                candidate,
                batch=batch,
                decode_mode=decode_mode,
            )
            return policy(state, actor_input, key)

        lanes = self.handle.batch if batch is None else batch
        return ParameterizedPolicy(
            apply,
            parameters,
            carry,
            compilation_key=("adk.recurrent", id(self.handle), lanes, decode_mode),
        )

    def behavior_cloning_config(self, **overrides: Any):
        """Build a BC config pinned to this handle's live action-head ABI."""

        from arena.training import BehaviorCloningConfig

        expected = tuple(self.handle.action_head_sizes)
        requested = tuple(overrides.pop("head_sizes", expected))
        if requested != expected:
            raise ValueError(
                "behavior-cloning head_sizes must match the built agent: "
                f"expected {expected}, got {requested}"
            )
        return BehaviorCloningConfig(head_sizes=expected, **overrides)

    def demonstration_batch(
        self,
        observation: Any,
        action: Any,
        *,
        action_mask: Any = None,
        supervision_mask: Any = None,
        weight: Any = None,
        config: Any = None,
        **config_overrides: Any,
    ):
        """Validate demonstrations against the built agent's factored ABI."""

        from arena.training import demonstration_batch

        config = self._behavior_cloning_config(config, config_overrides)
        return demonstration_batch(
            observation,
            action,
            action_mask=action_mask,
            supervision_mask=supervision_mask,
            weight=weight,
            config=config,
        )

    def behavior_cloning_trainer(
        self,
        apply: Callable,
        config: Any = None,
        *,
        optimizer: Any = None,
        compile: bool = True,
        **config_overrides: Any,
    ):
        """Create Arena's JIT BC update with the live ADK action contract."""

        from arena.training import make_behavior_cloning_trainer

        config = self._behavior_cloning_config(config, config_overrides)
        return make_behavior_cloning_trainer(
            apply,
            config,
            optimizer=optimizer,
            compile=compile,
        )

    def native_npc_demonstrations(
        self,
        capture: Any,
        *,
        observation: Any = None,
        action_mask: Any = None,
        weight: Any = None,
        config: Any = None,
        **config_overrides: Any,
    ):
        """Project a Java NPC trace against this agent's action ABI."""

        from arena.training import native_npc_demonstrations

        config = self._behavior_cloning_config(config, config_overrides)
        return native_npc_demonstrations(
            capture,
            observation=observation,
            action_mask=action_mask,
            weight=weight,
            config=config,
        )

    def projected_demonstrations(
        self,
        trace: Any,
        *,
        observation: str,
        action_labels: Any,
        action_mask: str | None = None,
        weight: str | None = None,
        config: Any = None,
        **config_overrides: Any,
    ):
        """Bind agent-owned trace projections to this ADK action ABI."""

        from arena.training import projected_demonstrations

        config = self._behavior_cloning_config(config, config_overrides)
        return projected_demonstrations(
            trace,
            observation=observation,
            action_labels=action_labels,
            action_mask=action_mask,
            weight=weight,
            head_names=self.handle.action_head_names,
            config=config,
        )

    def native_trace_sequence(self, capture: Any, *extensions: Any, **parameters: Any):
        """Expose every Java transition field without choosing an IL algorithm."""

        return self.imitation.native_trace_sequence(
            capture, *extensions, parameters=parameters
        )

    def native_duel_trace_sequences(self, captures: Any, *, spatial: bool = True):
        """Expose synchronized actor traces with optional spatial composition."""

        return self.imitation.native_duel_trace_sequences(captures, spatial=spatial)

    def native_damage_event_batch(self, trace: Any):
        """Pack Java-validated damage events into fixed-shape JAX arrays."""

        return self.imitation.native_damage_event_batch(trace)

    def native_lifecycle_event_batch(self, trace: Any):
        """Pack generic Java lifecycle facts into fixed-shape JAX arrays."""

        return self.imitation.native_lifecycle_event_batch(trace)

    def native_group_step_batch(self, info: Any):
        """Pack Java action-validation receipts into fixed-shape JAX arrays."""

        return self.imitation.native_group_step_batch(info)

    def trace_projectors(self, *projectors: Any):
        """Compose agent-defined, versioned trace transformations."""

        return self.imitation.TraceProjectorRegistry(projectors)

    def apply_episode_structure(self, trace: Any, **structure: Any):
        """Install evidence-derived episode boundaries with projector identity."""

        return self.imitation.apply_episode_structure(trace, **structure)

    def install_episode_evidence(self, trace: Any, **evidence: Any):
        """Derive episode IDs from capture-provided starts and terminal flags."""

        return self.imitation.install_episode_evidence(trace, **evidence)

    def install_native_lifecycle_episodes(self, trace: Any, **outcomes: Any):
        """Fuse Java actor lifecycle with caller-supplied task outcomes."""

        return self.imitation.install_native_lifecycle_episodes(trace, **outcomes)

    def reward_component(self, reward: Any, **options: Any):
        """Declare one versionable task-owned transition reward."""

        return self.imitation.reward_component(reward, **options)

    def categorical_action_component(
        self, name: str, action: Any, supported: Any, **options: Any
    ):
        """Declare exact categorical labels with explicit abstention."""

        return self.imitation.categorical_action_component(
            name, action, supported, **options
        )

    def component_requirement(self, reference: str, **options: Any):
        """Require a current or next trace component and its validity."""

        return self.imitation.component_requirement(reference, **options)

    def trace_layer(self, trace: Any, name: str):
        """Select one zero-copy trace layer for an ADK-built learner."""

        return trace.layer(name)

    def collect_native_npc_traces(self, jobs: Any, **options: Any):
        """Run bounded multi-world collection through the JAX environment bridge."""

        from hytalegym.worldgen import collect_native_npc_traces

        return collect_native_npc_traces(jobs, **options)

    def load_native_behavior_policy(self, checkpoint: Any, corpus: Any):
        """Load a contract-pinned native-context behavior prior."""

        self._require_task_contract()
        from arena.training import load_native_behavior_policy

        return load_native_behavior_policy(checkpoint, corpus)

    def native_behavior_policy_logits(self, policy: Any, corpus: Any):
        """Evaluate a prior; consume only ``policy.supervised_heads``."""

        self._require_task_contract()
        from arena.training import native_behavior_policy_logits

        return native_behavior_policy_logits(policy, corpus)

    def native_npc_duel_demonstrations(
        self,
        captures: Any,
        *,
        observations: Any = None,
        action_masks: Any = None,
        weights: Any = None,
        config: Any = None,
        **config_overrides: Any,
    ):
        """Project both synchronized Java NPC traces against the agent ABI."""

        from arena.training import native_npc_duel_demonstrations

        config = self._behavior_cloning_config(config, config_overrides)
        return native_npc_duel_demonstrations(
            captures,
            observations=observations,
            action_masks=action_masks,
            weights=weights,
            config=config,
        )

    def _behavior_cloning_config(self, config: Any, overrides: dict[str, Any]):
        if config is not None and overrides:
            raise ValueError("pass a behavior-cloning config or overrides, not both")
        config = config or self.behavior_cloning_config(**overrides)
        expected = tuple(self.handle.action_head_sizes)
        if tuple(config.head_sizes) != expected:
            raise ValueError(
                "behavior-cloning config does not match the built agent's "
                f"action heads: expected {expected}, got {tuple(config.head_sizes)}"
            )
        return config

    def _require_task_contract(self) -> None:
        from arena.jax_contract import HEAD_SPANS

        expected_names = tuple(HEAD_SPANS)
        expected_sizes = tuple(size for _, size in HEAD_SPANS.values())
        actual_names = tuple(self.handle.action_head_names)
        actual_sizes = tuple(self.handle.action_head_sizes)
        if (actual_names, actual_sizes) != (expected_names, expected_sizes):
            raise ValueError(
                "the built agent's action contract cannot run Arena tasks: "
                f"expected {expected_names}/{expected_sizes}, got "
                f"{actual_names}/{actual_sizes}"
            )


__all__ = ["ArenaPolicy", "ArenaTools"]
