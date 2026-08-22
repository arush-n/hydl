"""High-level Agent Development Kit.

Developers work with :class:`AgentKit` and :class:`AgentHandle`.  JAX factory
arguments, backend resolution, bridge translation, leases, and artifact
identity remain behind this surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.jax.combat.arsenal.environment import ArsenalEnvironment

if TYPE_CHECKING:
    from hytalegym.jax.training.ppo import PPOEnvironment
    from hytalegym.jax.training.types import (
        PPOConfig,
        PPOTrainState,
        RecurrentPolicyParams,
    )

from adk.architecture.inputs import (
    ActorPolicyInput,
    JaxTrainingInput,
    actor_policy_input,
    jax_training_input,
)
from adk.architecture.decision_context import (
    ActionLifecycle,
    ActionVerb,
    DecisionContext,
    DecisionContextFrame,
    DecisionEvent,
    DecisionEvents,
    DecisionTiming,
    ExecutionState,
    jax_decision_context,
    manager_decision_due,
    native_decision_context,
)
from adk.architecture.inference import (
    InferenceMetadata,
    InferenceValidation,
    InferenceValidationContext,
    StaleInferenceReason,
    action_mask_hash,
    capture_inference_metadata,
    validate_inference_result,
)
from adk.architecture.hierarchy import (
    HierarchicalAgentCore,
    HierarchicalCoreState,
)
from adk.core.registry import (
    get as get_registered_agent,
    list_agents as list_registered_agents,
    register as register_agent_spec,
)
from adk.core.spec import AgentSpec
from adk.development import (
    ActionHead,
    ActionIntent,
    ActionSpace,
    AgentFrame,
    BaseAction,
    BlockTrigger,
    DodgeDirection,
    EnvironmentDiagnostics,
    StepResult,
    WorldDirection,
)

if TYPE_CHECKING:
    from adk.core.lifecycle import LoadedCheckpoint
    from adk.evaluation import EvaluationResult, EvaluationSuite
from adk.policy import Policy
from adk.runtime.bridge_client import (
    BridgePolicyInput,
    BridgeTransition,
    HYTALE_ENV_PARAMETER_BINDINGS,
    NativeBridgeSession,
)
from adk.runtime.actions import (
    ActionHead as CompositeActionHead,
    CompositeActionSpec,
    EnvironmentActionBoundary,
    EnvironmentActionReceipt,
    JointActionCapability,
    JointActionRejected,
    JointExecutionFailed,
    JointExecutionUnavailable,
    JointValidationUnavailable,
    MaskedCompositeSample,
    StructuredActionCodec,
)
from adk.runtime.env_adapter import (
    BuiltAgent,
    EnvironmentStep,
    JaxBuildInputs,
    PolicyInput,
    build as build_jax_agent,
    decode as decode_action_factors,
    encode as encode_structured_actions,
    neutral_actions as make_neutral_actions,
    prepare_jax_build_inputs,
    step_with_boundary,
)
from adk.runtime.episode import EpisodeBoundary
from adk.runtime.policy_surface import DensePolicySurface
from adk.runtime.jax_runtime import (
    JaxRuntimeSettings,
    configure_jax_compilation_cache,
)
from adk.runtime.loop import (
    ContextualLoopState,
    ContextualLoopTransition,
    ContextualPolicy,
    ContextualPolicyInput,
    ContextualRecordFn,
    LoopState,
    LoopTransition,
    RecordFn,
    collect as collect_transitions,
    collect_contextual as collect_contextual_transitions,
    unknown_decision_context,
)
from adk.runtime.hierarchical import (
    HierarchicalActorInputFn,
    HierarchicalContextFn,
    HierarchicalLoopState,
    HierarchicalLoopTransition,
    HierarchicalRecordFn,
    HierarchicalRuntimeDiagnostics,
    collect_hierarchical as collect_hierarchical_transitions,
    compile_hierarchical_collector as make_hierarchical_collector,
    jax_hierarchical_context,
)
from adk.runtime.rollout import Trajectory, rollout as run_rollout
from adk.runtime.scene_builder import (
    COMBAT_PARAMETER_NAMES,
    RUNTIME_CONFIG_ARGUMENTS,
    SceneDescription,
    describe_scene,
    list_loadouts,
    make_scene,
)
from adk.runtime.scenes import (
    FAIL_CLOSED_SCENE,
    JAX_SCENE_PROVIDER_ARGUMENTS,
    JaxScene,
    OPEN_FLAT_CONTROL_SCENE,
    scene_from_region_fixture,
)
from adk.runtime.tree import (
    ENVIRONMENT_STATE_SELECTOR_SCHEMA,
    select_environment_rows,
)
from adk.runtime.spawns import (
    describe_spawns,
    fixed_spawns,
    lit_surface_spawns,
    region_pool_spawns,
    shifted,
    swapped,
)


class ResetResult(NamedTuple):
    """Named form of ``reset``'s pair.

    Still a plain tuple, so ``state, policy_input = handle.reset(key)`` keeps
    working; the names exist so callers do not have to learn the order by
    introspection.
    """

    state: Any
    policy_input: Any


def _unwrap_policy_params(value: Any) -> Any:
    """Accept either policy params or a train state that carries them.

    Passing a ``PPOTrainState`` where params are expected otherwise fails deep
    inside the policy constructor with ``AttributeError: 'PPOTrainState' object
    has no attribute 'gru'``, which leaks an internal and names nothing the
    caller can act on.
    """

    carried = getattr(value, "policy_params", None)
    return value if carried is None else carried


@dataclass(frozen=True, slots=True)
class AgentHandle:
    """One built agent with the common development workflows attached."""

    built: BuiltAgent

    @property
    def spec(self) -> AgentSpec:
        return self.built.spec

    def with_shaping(self, shaping: Any) -> "AgentHandle":
        """Return a handle whose reward carries an additive shaping term.

        `shaping` is anything :func:`adk.scenarios.shaping.resolve` accepts --
        a callable ``(next_state, info) -> (batch,)``, a ``Shaping``, a
        minigame, or a list of them. It is applied inside
        :meth:`BuiltAgent.step_factors`, so PPO, the collector and a bare
        rollout all see the same reward; attaching it at any one of those call
        sites instead would shape that one and silently leave the others on the
        native reward.

        Frozen, so this returns a new handle rather than mutating. The
        observation and action contracts are untouched, so a checkpoint trained
        through a shaped handle still loads into an unshaped one -- which is
        correct, because shaping is a training-time reward change and not a
        change to what the policy is.
        """

        import dataclasses

        return dataclasses.replace(self, built=self.built.with_shaping(shaping))

    def with_episode_limit(self, ticks: int | None) -> "AgentHandle":
        """Return a handle whose episodes end after `ticks` engine ticks.

        30 ticks is one second of game time. The environment's own rule ends an
        episode only on death or a world error, so without this an episode has
        no upper bound -- measured, they ran past 512 ticks while a 256-step
        update completed 0-3 of them, which starves a value function of the
        boundaries it has to fit.

        The cap is published as `truncated`, distinct from `terminated`, so a
        learner can bootstrap through a clock expiry and not through a death.
        Like shaping, it applies inside :meth:`BuiltAgent.step_factors`, so the
        collector, PPO and a bare rollout all see the same episodes.
        """

        import dataclasses

        return dataclasses.replace(self, built=self.built.with_episode_limit(ticks))

    @property
    def stamp(self):
        return self.built.stamp

    @property
    def batch(self) -> int:
        return self.built.batch

    @property
    def observation_size(self) -> int:
        return self.built.observation_size

    @property
    def action_head_sizes(self) -> tuple[int, ...]:
        return self.built.stamp.action_head_sizes

    @property
    def action_head_names(self) -> tuple[str, ...]:
        return self.built.stamp.action_head_names

    @property
    def action_space(self) -> ActionSpace:
        """Describe and construct actions from the exact live factor ABI."""

        return ActionSpace.from_contract(
            self.action_head_names,
            self.action_head_sizes,
        )

    @property
    def composite_action_spec(self) -> CompositeActionSpec:
        """Expose the lossless, algorithm-neutral live factor contract."""

        return CompositeActionSpec.from_live_contract(self.stamp)

    @property
    def structured_action_codec(self) -> StructuredActionCodec[Any]:
        """Borrow Gym's exact structured codec behind the live factor ABI."""

        return StructuredActionCodec(
            spec=self.composite_action_spec,
            encoder=encode_structured_actions,
            decoder=decode_action_factors,
            raw_codec=(encode_structured_actions, decode_action_factors),
        )

    @property
    def jax_environment(self) -> ArsenalEnvironment:
        """Borrow the canonical framework-neutral structured environment."""

        return self.built.environment

    @property
    def jax_environment_spec(self) -> Any:
        """Borrow the upstream structured observation/action specification."""

        return self.built.environment.spec

    @property
    def policy_surface_spec(self) -> Any:
        """Describe the optional dense bridge-compatible policy projection."""

        return self.built.policy_surface.spec

    def ppo_environment(self):
        """Create an optional PPO adapter over this same structured runtime."""

        from adk.training.ppo_adapter import as_ppo_environment

        return as_ppo_environment(self.built)

    @property
    def ppo(self):
        """Opt into the reference PPO learner built on the SDK runtime."""

        from adk.algorithms.ppo import PPOTools

        return PPOTools(self.built)

    @property
    def arena(self):
        """Opt into Arena tasks and generic JAX training techniques.

        The adapter lives on the ADK side so Arena remains a reusable sibling
        with no dependency on agent-framework code.
        """

        from adk.arena import ArenaTools

        return ArenaTools(self)

    @property
    def jax_combat_params(self) -> Any:
        """Borrow the exact upstream combat parameters used by this build."""

        return self.built.combat_params

    @property
    def jax_runtime_config(self) -> Any:
        """Borrow the exact upstream Arsenal runtime configuration."""

        return self.built.runtime_config

    @property
    def jax_scene_providers(self) -> Mapping[str, Any]:
        """Borrow the exact provider arguments bound to this scene."""

        return self.built.scene.environment_kwargs

    @property
    def jax_runtime_capacity(self) -> Any:
        """Borrow the upstream Arsenal runtime-capacity descriptor."""

        return self.built.runtime_capacity

    def reset(self, key: jax.Array):
        """Reset every JAX lane from one root key."""

        state, policy_input = self.built.reset(jax.random.split(key, self.batch))
        return ResetResult(state, policy_input)

    def start(self, key: jax.Array) -> AgentFrame:
        """Reset into one lossless developer-facing frame."""

        state, policy_input = self.reset(key)
        return AgentFrame(state, policy_input)

    @staticmethod
    def actor_input(frame: AgentFrame) -> ActorPolicyInput:
        """Expose Gym's exact structured legal observation to a policy.

        The returned value also retains the unchanged dense observation and
        action mask used by the bridge-compatible reference surface.
        """

        if not isinstance(frame, AgentFrame):
            raise TypeError("frame must be returned by AgentHandle.start/step")
        return actor_policy_input(frame.state, frame.policy_input)

    @staticmethod
    def training_input(frame: AgentFrame) -> JaxTrainingInput:
        """Expose simulator truth on an explicit teacher/critic-only path."""

        if not isinstance(frame, AgentFrame):
            raise TypeError("frame must be returned by AgentHandle.start/step")
        return jax_training_input(frame.state, frame.policy_input)

    def make_action(
        self,
        intent: ActionIntent,
        *,
        frame: AgentFrame | None = None,
        policy_input: Any | None = None,
    ) -> jax.Array:
        """Encode a semantic intent and reject choices masked in this frame."""

        if frame is not None:
            if not isinstance(frame, AgentFrame):
                raise TypeError("frame must be returned by AgentHandle.start/step")
            if policy_input is not None:
                raise ValueError("pass frame or policy_input, not both")
            policy_input = frame.policy_input
        return self.action_space.encode_intent(
            intent,
            batch=self.batch,
            policy_input=policy_input,
        )

    def make_action_factors(
        self,
        *,
        frame: AgentFrame | None = None,
        policy_input: Any | None = None,
        choices: Mapping[str, Any] | None = None,
        **named_choices: Any,
    ) -> jax.Array:
        """Build mask-checked factors from live action-head names."""

        if frame is not None:
            if not isinstance(frame, AgentFrame):
                raise TypeError("frame must be returned by AgentHandle.start/step")
            if policy_input is not None:
                raise ValueError("pass frame or policy_input, not both")
            policy_input = frame.policy_input
        return self.action_space.compose(
            batch=self.batch,
            policy_input=policy_input,
            choices=choices,
            **named_choices,
        )

    def step(
        self,
        frame: AgentFrame,
        action: ActionIntent | Any,
        key: jax.Array,
    ) -> StepResult:
        """Apply one semantic intent or exact factor array without data loss."""

        if not isinstance(frame, AgentFrame):
            raise TypeError("frame must be returned by AgentHandle.start/step")
        if isinstance(action, ActionIntent):
            factors = self.make_action(action, frame=frame)
        else:
            factors = self.action_space.require_legal(
                action,
                frame.policy_input,
            )
        environment_step = step_with_boundary(
            self.built,
            frame.state,
            factors,
            jax.random.split(key, self.batch),
        )
        return StepResult(
            frame=AgentFrame(
                environment_step.state,
                environment_step.policy_input,
            ),
            action_factors=factors,
            reward=environment_step.reward,
            done=environment_step.done,
            info=environment_step.info,
            terminated=environment_step.terminated,
            truncated=environment_step.truncated,
        )

    @staticmethod
    def diagnostics(info: Any) -> EnvironmentDiagnostics:
        """Name common diagnostics while preserving the complete raw value."""

        return EnvironmentDiagnostics.from_info(info)

    def decision_context(
        self,
        result: StepResult,
        *,
        previous_frame: AgentFrame | None = None,
    ) -> DecisionContextFrame:
        """Normalize one JAX step into PLAN timing/lifecycle/event terms.

        Pass the frame used to produce ``result`` to enable actor-observable
        edge events. Without it, timing and lifecycle are still mapped while
        perceptual events remain explicitly unknown. The exact step ``info``
        remains available through ``returned.diagnostics.raw``.
        """

        if not isinstance(result, StepResult):
            raise TypeError("result must be returned by AgentHandle.step")
        if previous_frame is None:
            return jax_decision_context(result.info)
        if not isinstance(previous_frame, AgentFrame):
            raise TypeError("previous_frame must be returned by AgentHandle.start/step")
        return jax_decision_context(
            result.info,
            actor_input=self.actor_input(result.frame),
            previous_actor_input=self.actor_input(previous_frame),
        )

    def rollout(
        self,
        policy,
        key: jax.Array,
        steps: int,
        *,
        initial_carry: Any = None,
    ) -> tuple[Any, Trajectory]:
        """Collect a compiled-friendly time-major trajectory."""

        loop_carry, trajectory = run_rollout(
            self.built,
            policy,
            key,
            steps,
            initial_carry=initial_carry,
        )
        return loop_carry[2], trajectory

    def collect(
        self,
        policy: Policy,
        record: RecordFn,
        key: jax.Array,
        steps: int,
        *,
        initial_carry: Any = None,
    ) -> tuple[LoopState, Any]:
        """Collect any fixed-shape transition PyTree through one shared loop.

        The record function receives a lossless :class:`LoopTransition` and
        chooses what an algorithm stores.  This is the general primitive on
        which compact rollouts and learner-specific batches are built.
        """

        return collect_transitions(
            self.built,
            policy,
            record,
            key,
            steps,
            initial_carry=initial_carry,
        )

    def collect_contextual(
        self,
        policy: ContextualPolicy,
        record: ContextualRecordFn,
        key: jax.Array,
        steps: int,
        *,
        initial_carry: Any = None,
    ) -> tuple[ContextualLoopState, Any]:
        """Collect with actor-safe decision context computed inside JAX.

        Existing :meth:`collect` policies remain unchanged. This variant gives
        a policy the exact :class:`ActorPolicyInput` together with normalized
        timing, lifecycle, and event evidence and owns terminal-row context
        reset in the same compiled scan.
        """

        return collect_contextual_transitions(
            self.built,
            policy,
            record,
            key,
            steps,
            initial_carry=initial_carry,
        )

    def collect_hierarchical(
        self,
        core: HierarchicalAgentCore,
        record: HierarchicalRecordFn,
        key: jax.Array,
        steps: int,
        *,
        initial_state: HierarchicalCoreState,
        context_fn: HierarchicalContextFn = jax_hierarchical_context,
        actor_input_fn: HierarchicalActorInputFn = actor_policy_input,
    ) -> tuple[HierarchicalLoopState, Any]:
        """Run a general belief/manager/actor hierarchy over this runtime."""

        return collect_hierarchical_transitions(
            self.built,
            core,
            record,
            key,
            steps,
            initial_state=initial_state,
            context_fn=context_fn,
            actor_input_fn=actor_input_fn,
        )

    def compile_update(
        self,
        *,
        policy_of: Callable[[Any], Policy],
        record: RecordFn,
        loss: Callable[..., Any],
        optimizer: Any,
        steps: int,
        initial_carry: Any = None,
        has_aux: bool = False,
        compile: bool = True,
    ):
        """Compile one algorithm-neutral collect-then-descend update.

        The caller owns the parameters, the policy they induce, what is
        recorded, the loss, and the optimiser; the ADK owns collection and the
        compiled update.  Behaviour cloning, DAgger, and auxiliary-loss agents
        use this instead of importing the PPO module.
        """

        from adk.algorithms.neutral import make_update

        return make_update(
            self,
            policy_of=policy_of,
            record=record,
            loss=loss,
            optimizer=optimizer,
            steps=steps,
            initial_carry=initial_carry,
            has_aux=has_aux,
            compile=compile,
        )

    def compile_collector(
        self,
        policy: Policy,
        record: RecordFn,
        steps: int,
        *,
        initial_carry: Any = None,
    ):
        """JIT a reusable algorithm-defined collection loop."""

        if not callable(policy):
            raise TypeError("policy must satisfy the ADK Policy protocol")
        if not callable(record):
            raise TypeError("record must be callable")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps < 1:
            raise ValueError("steps must be positive")

        def compiled(root_key, carry=initial_carry):
            return collect_transitions(
                self.built,
                policy,
                record,
                root_key,
                steps,
                initial_carry=carry,
            )

        return jax.jit(compiled)

    def compile_contextual_collector(
        self,
        policy: ContextualPolicy,
        record: ContextualRecordFn,
        steps: int,
        *,
        initial_carry: Any = None,
    ):
        """JIT a reusable contextual collection loop."""

        if not callable(policy):
            raise TypeError("policy must be callable")
        if not callable(record):
            raise TypeError("record must be callable")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps < 1:
            raise ValueError("steps must be positive")

        def compiled(root_key, carry=initial_carry):
            return collect_contextual_transitions(
                self.built,
                policy,
                record,
                root_key,
                steps,
                initial_carry=carry,
            )

        return jax.jit(compiled)

    def compile_hierarchical_collector(
        self,
        core: HierarchicalAgentCore,
        record: HierarchicalRecordFn,
        steps: int,
        *,
        initial_state: HierarchicalCoreState,
        context_fn: HierarchicalContextFn = jax_hierarchical_context,
        actor_input_fn: HierarchicalActorInputFn = actor_policy_input,
    ):
        """JIT a reusable non-PPO hierarchical environment collector."""

        return make_hierarchical_collector(
            self.built,
            core,
            record,
            steps,
            initial_state=initial_state,
            context_fn=context_fn,
            actor_input_fn=actor_input_fn,
        )

    def compile_rollout(
        self,
        policy: Policy,
        steps: int,
        *,
        initial_carry: Any = None,
    ):
        """JIT one reusable host-sync-free rollout callable.

        The returned function accepts ``(key, carry=initial_carry)``.  Policy,
        provider arrays, and the static step count are captured once, while a
        caller may still pass a new recurrent carry without rebuilding the
        environment.
        """

        if not callable(policy):
            raise TypeError("policy must satisfy the ADK Policy protocol")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps < 1:
            raise ValueError("steps must be positive")

        def compiled(root_key, carry=initial_carry):
            loop_carry, trajectory = run_rollout(
                self.built,
                policy,
                root_key,
                steps,
                initial_carry=carry,
            )
            return loop_carry[2], trajectory

        return jax.jit(compiled)

    def neutral_actions(self) -> jax.Array:
        """Return one all-neutral factor row for every built JAX lane."""

        return make_neutral_actions(self.batch)

    def neutral_policy(self) -> Policy:
        """A do-nothing policy satisfying the public ``Policy`` protocol.

        Every rollout and collection entry point needs a ``Policy``, and the
        first thing a new caller wants is to exercise the loop before they have
        a trained one.  Named explicitly rather than defaulted: a rollout that
        silently used a neutral policy would produce meaningless data that looks
        real.
        """

        actions = self.neutral_actions()

        def policy(carry, _actor_input, _key):
            return carry, actions

        return policy

    def sample_actions(
        self,
        logits: jax.Array,
        policy_input: PolicyInput | ActorPolicyInput,
        key: jax.Array,
    ) -> jax.Array:
        """Sample every categorical head through the exact current mask."""

        if not isinstance(policy_input, (PolicyInput, ActorPolicyInput)):
            raise TypeError(
                "policy_input must be returned by the ADK policy/frame surface"
            )
        return self.sample_composite_action(
            logits,
            policy_input,
            key,
        ).factors

    def sample_composite_action(
        self,
        logits: jax.Array,
        policy_input: PolicyInput | ActorPolicyInput,
        key: jax.Array,
    ) -> MaskedCompositeSample:
        """Sample factors and retain availability, legality, and log-prob."""

        if not isinstance(policy_input, (PolicyInput, ActorPolicyInput)):
            raise TypeError(
                "policy_input must be returned by the ADK policy/frame surface"
            )
        return self.composite_action_spec.sample_masked(
            key,
            logits,
            policy_input.action_mask,
        )

    def action_log_probability(
        self,
        logits: jax.Array,
        factors: jax.Array,
        policy_input: PolicyInput | ActorPolicyInput,
    ) -> jax.Array:
        """Score explicit factors without depending on a training algorithm."""

        if not isinstance(policy_input, (PolicyInput, ActorPolicyInput)):
            raise TypeError(
                "policy_input must be returned by the ADK policy/frame surface"
            )
        return self.composite_action_spec.masked_log_probability(
            logits,
            factors,
            policy_input.action_mask,
        )

    def action_receipt(self, result: StepResult) -> EnvironmentActionReceipt:
        """Expose joint-action evidence without upgrading marginal masks.

        The current JAX step publishes detailed verb lifecycle diagnostics but
        no whole-composite validation receipt.  Preserve the raw environment
        value and report that capability as unavailable until upstream owns
        that contract explicitly.
        """

        if not isinstance(result, StepResult):
            raise TypeError("result must be returned by AgentHandle.step")
        return EnvironmentActionReceipt.unavailable(
            result.action_factors,
            reason=(
                "the current environment does not publish whole-composite "
                "joint validation"
            ),
            raw_receipt=result.info,
        )

    @staticmethod
    def encode_actions(structured_actions: Any) -> jax.Array:
        """Encode Gym structured actions as explicit factor rows."""

        return encode_structured_actions(structured_actions)

    @staticmethod
    def decode_actions(factors: jax.Array) -> Any:
        """Decode explicit factor rows to the Gym structured action."""

        return decode_action_factors(factors)

    def training_config(self, **overrides: Any) -> PPOConfig:
        """Compatibility alias for ``handle.ppo.config``."""

        return self.ppo.config(**overrides)

    def initialize_training(
        self,
        key: jax.Array,
        config: PPOConfig | None = None,
        **config_overrides: Any,
    ) -> tuple[PPOConfig, PPOTrainState]:
        """Compatibility alias for ``handle.ppo.initialize``."""

        return self.ppo.initialize(key, config, **config_overrides)

    def training_step(
        self,
        config: PPOConfig,
        *,
        compile: bool = True,
    ):
        """Compatibility alias for ``handle.ppo.train_step``."""

        return self.ppo.train_step(config, compile=compile)

    def evaluate(
        self,
        policy,
        key: jax.Array,
        *,
        max_steps: int = 512,  # EVALUATION_HORIZON_DEFAULT; literal keeps
        # adk.evaluation (and the Gym) out of module import time.
        initial_carry: Any = None,
        compile: bool = True,
        allow_short_horizon: bool = False,
    ) -> EvaluationResult:
        from adk.evaluation import evaluate as evaluate_agent

        return evaluate_agent(
            self.built,
            policy,
            key,
            max_steps=max_steps,
            initial_carry=initial_carry,
            compile=compile,
            allow_short_horizon=allow_short_horizon,
        )

    def recurrent_policy(
        self,
        params: RecurrentPolicyParams,
        *,
        decode_mode: str = "factored_argmax",
        batch: int | None = None,
    ) -> tuple[Policy, jax.Array]:
        """Expose trained Gym parameters through the public Policy protocol.

        Accepts either ``RecurrentPolicyParams`` or a train state carrying them.
        """

        lanes = self.batch if batch is None else batch
        if isinstance(lanes, bool) or not isinstance(lanes, int):
            raise TypeError("policy carry batch must be an integer")
        if lanes < 1:
            raise ValueError("policy carry batch must be positive")
        policy, one_row_carry = self.ppo.policy(
            _unwrap_policy_params(params),
            decode_mode=decode_mode,
        )
        return policy, jnp.broadcast_to(
            one_row_carry,
            (lanes, one_row_carry.shape[1]),
        )

    def evaluate_controls(
        self,
        trained_params: RecurrentPolicyParams,
        config: PPOConfig,
        key: jax.Array,
        *,
        max_steps: int = 512,  # EVALUATION_HORIZON_DEFAULT; literal keeps
        # adk.evaluation (and the Gym) out of module import time.
        trained_decode_mode: str = "factored_argmax",
        compile: bool = True,
        allow_short_horizon: bool = False,
    ) -> EvaluationSuite:
        return self.ppo.evaluate_controls(
            trained_params,
            config,
            key,
            max_steps=max_steps,
            trained_decode_mode=trained_decode_mode,
            compile=compile,
            allow_short_horizon=allow_short_horizon,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        policy_params: RecurrentPolicyParams,
        config: PPOConfig,
        *,
        seed: int,
        updates: int | None = None,
        environment_steps: int | None = None,
        combat_target_active: bool = True,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save a stamped checkpoint.

        ``policy_params`` accepts either ``RecurrentPolicyParams`` or a train
        state carrying them.  When a train state is supplied, ``updates`` and
        ``environment_steps`` default to the counters it already tracks, so the
        common call is ``save_checkpoint(path, train_state, config, seed=...)``.
        """

        resolved_updates, resolved_steps = self._resolve_progress_counters(
            policy_params,
            updates,
            environment_steps,
        )
        return self.ppo.save(
            path,
            _unwrap_policy_params(policy_params),
            config,
            seed=seed,
            updates=resolved_updates,
            environment_steps=resolved_steps,
            combat_target_active=combat_target_active,
            run_metadata=run_metadata,
        )

    @staticmethod
    def _resolve_progress_counters(
        policy_params: Any,
        updates: int | None,
        environment_steps: int | None,
    ) -> tuple[int, int]:
        """Fill checkpoint progress counters from a train state when present."""

        resolved = {"updates": updates, "environment_steps": environment_steps}
        sources = {
            "updates": "update_count",
            "environment_steps": "total_environment_steps",
        }
        for name, attribute in sources.items():
            if resolved[name] is not None:
                continue
            counter = getattr(policy_params, attribute, None)
            if counter is None:
                raise TypeError(
                    f"{name} is required because the supplied policy params "
                    f"carry no {attribute!r}; pass {name}= explicitly, or pass "
                    "the train state instead of bare policy params"
                )
            resolved[name] = int(counter)
        return resolved["updates"], resolved["environment_steps"]

    def load_checkpoint(self, path: str | Path) -> LoadedCheckpoint:
        return self.ppo.load(path)

    def native_session(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5556,
        purpose: str = "agent_adk_native_transfer",
        **options: Any,
    ) -> NativeBridgeSession:
        """Create a lease-owning native transfer session; no socket opens yet."""

        options.setdefault(
            "learner_v3_world_geometry_config",
            self.built.world_geometry_config,
        )
        return NativeBridgeSession(
            self.spec,
            backend={
                "backend": "native",
                "host": host,
                "port": port,
                "entry_point": "native_evidence",
            },
            stamp=self.stamp,
            purpose=purpose,
            **options,
        )


_AGENT_FAMILY_ROOT = Path(__file__).resolve().parent / "agents"


def _import_agent_family(family: str) -> None:
    """Import one optional agent family so its ``register`` calls run.

    Registration is an import side effect, so a family that nobody has
    imported is invisible to the registry even though ``build`` can resolve
    it.  A missing optional family is a normal registry miss; a dependency
    missing *inside* a real family module is not.
    """

    module_name = f"adk.agents.{family}.spec"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing = str(error.name or "")
        if not (missing == module_name or module_name.startswith(f"{missing}.")):
            raise


def _import_agent_families() -> None:
    """Import every on-disk agent family that declares specs.

    ``list_agents`` must see exactly what ``build`` can resolve.  Families are
    directories under ``adk/agents`` containing ``spec.py``; README-only
    placeholders are skipped.
    """

    if not _AGENT_FAMILY_ROOT.is_dir():
        return
    for entry in sorted(_AGENT_FAMILY_ROOT.iterdir()):
        if entry.name.startswith((".", "_")) or not entry.is_dir():
            continue
        if (entry / "spec.py").is_file():
            _import_agent_family(entry.name)


class AgentKit:
    """Registry and factory for ADK agents and their JAX scene bindings."""

    def __init__(
        self,
        *,
        compilation_cache: str | Path | None = None,
    ) -> None:
        self.jax_runtime_settings: JaxRuntimeSettings | None = None
        if compilation_cache is not None:
            self.jax_runtime_settings = configure_jax_compilation_cache(
                compilation_cache
            )
        self._scenes: dict[str, JaxScene] = {
            FAIL_CLOSED_SCENE.name: FAIL_CLOSED_SCENE,
            OPEN_FLAT_CONTROL_SCENE.name: OPEN_FLAT_CONTROL_SCENE,
        }

    def register(self, spec: AgentSpec) -> AgentSpec:
        """Register one declarative agent definition."""

        return register_agent_spec(spec)

    def register_scene(self, scene: JaxScene) -> JaxScene:
        """Register a provider bundle by stable name and contract identity."""

        if not isinstance(scene, JaxScene):
            raise TypeError("scene must be a JaxScene")
        existing = self._scenes.get(scene.name)
        if existing is not None and existing.contract_sha256 != scene.contract_sha256:
            raise ValueError(
                f"scene name already registered differently: {scene.name!r}"
            )
        if existing is not None:
            if existing.metadata() != scene.metadata():
                raise ValueError(
                    "scene digest was reused with different declared "
                    f"metadata: {scene.name!r}"
                )
            # The digest is the provider-bundle identity.  Keep the first
            # bound runtime objects instead of comparing arbitrary arrays or
            # callables (whose equality may be ambiguous) or silently
            # replacing them.
            return existing
        self._scenes[scene.name] = scene
        return scene

    def make_scene(self, name: str, **options: Any) -> JaxScene:
        """Build a parameterized scene and register it in one call.

        Thin wrapper over :func:`adk.runtime.scene_builder.make_scene`; the
        contract digest is derived from the declared configuration, so calling
        this twice with identical options is idempotent.
        """

        return self.register_scene(make_scene(name, **options))

    def load_region_scene(
        self,
        spec: AgentSpec | str,
        *,
        name: str,
        contract_sha256: str,
        batch: int,
        loader: Callable[..., Any],
        loader_kwargs: Mapping[str, Any] | None = None,
        world_geometry_config: Any | None = None,
        explosion_candidate_provider: Any | None = None,
        recipe_candidate_encoder_params: Any | None = None,
        opponent_ability_provider: Any | None = None,
    ) -> JaxScene:
        """Load and register an exact Region fixture once on the host.

        ``loader`` is the Gym's artifact loader (or a compatible future
        library entry point).  The SDK supplies its exact params, runtime, and
        batch, then retains those same objects for the eventual JAX build.
        No loader or filesystem work is captured by the compiled environment.
        """

        if not callable(loader):
            raise TypeError("loader must be callable")
        options = dict(loader_kwargs or {})
        protected = {"params", "runtime", "batch_size"}.intersection(options)
        if protected:
            raise ValueError(
                "Region loader build inputs are SDK-owned: "
                + ", ".join(sorted(protected))
            )
        resolved = self._resolve_spec(spec)
        identity = JaxScene(
            name=name,
            contract_sha256=contract_sha256,
            environment_kwargs={},
            expected_batch=batch,
            expected_loadout=resolved.loadout,
        )
        existing = self._scenes.get(identity.name)
        if existing is not None:
            if existing.contract_sha256 != identity.contract_sha256:
                raise ValueError(
                    f"scene name already registered differently: {identity.name!r}"
                )
            if existing.expected_batch != batch:
                raise ValueError(
                    "registered Region scene has a different batch identity"
                )
            if existing.expected_loadout != resolved.loadout:
                raise ValueError(
                    "registered Region scene has a different loadout identity"
                )
            return existing
        prepared = prepare_jax_build_inputs(resolved, batch)
        fixture = loader(
            params=prepared.combat_params,
            runtime=prepared.runtime_config,
            batch_size=batch,
            **options,
        )
        scene = scene_from_region_fixture(
            name=name,
            contract_sha256=contract_sha256,
            fixture=fixture,
            runtime_config=prepared.runtime_config,
            runtime_capacity=prepared.runtime_capacity,
            expected_batch=batch,
            expected_loadout=resolved.loadout,
            world_geometry_config=world_geometry_config,
            explosion_candidate_provider=explosion_candidate_provider,
            recipe_candidate_encoder_params=(recipe_candidate_encoder_params),
            opponent_ability_provider=opponent_ability_provider,
        )
        return self.register_scene(scene)

    def list_agents(self) -> list[str]:
        _import_agent_families()
        return list_registered_agents()

    def list_scenes(self) -> list[str]:
        return sorted(self._scenes)

    def build(
        self,
        spec: AgentSpec | str,
        *,
        batch: int | None = None,
        num_envs: int | None = None,
        backend: str | Mapping[str, Any] = "jax",
    ) -> AgentHandle:
        """Build a registered name or an explicit spec into an SDK handle.

        ``num_envs`` is accepted as an alias for ``batch`` because that is the
        prevailing name in the surrounding RL ecosystem.
        """

        if batch is not None and num_envs is not None and batch != num_envs:
            raise TypeError(
                "batch and num_envs are aliases; pass only one "
                f"(got batch={batch}, num_envs={num_envs})"
            )
        if batch is None:
            batch = 1 if num_envs is None else num_envs

        resolved = self._resolve_spec(spec)
        try:
            scene = self._scenes[resolved.scene]
        except KeyError as error:
            raise KeyError(
                f"scene {resolved.scene!r} is not registered in this AgentKit"
            ) from error
        return AgentHandle(
            build_jax_agent(
                resolved,
                batch=batch,
                backend=backend,
                scene=scene,
            )
        )

    def make(
        self,
        spec: AgentSpec | str,
        *,
        batch: int | None = None,
        num_envs: int | None = None,
        backend: str | Mapping[str, Any] = "jax",
    ) -> AgentHandle:
        """Alias for :meth:`build`; ``make`` is the more common spelling."""

        return self.build(
            spec,
            batch=batch,
            num_envs=num_envs,
            backend=backend,
        )

    @staticmethod
    def _resolve_spec(spec: AgentSpec | str) -> AgentSpec:
        if isinstance(spec, AgentSpec):
            return spec
        if not isinstance(spec, str):
            raise TypeError("agent must be an AgentSpec or registered name")
        try:
            return get_registered_agent(spec)
        except KeyError:
            _import_agent_family(spec.split("/", 1)[0])
            return get_registered_agent(spec)


default_kit = AgentKit()

# Low-level form for runtimes that already own a ``BuiltAgent``. Most agent
# authors should use ``AgentHandle.collect_contextual`` or its compiled form.
collect_contextual = collect_contextual_transitions
collect_hierarchical = collect_hierarchical_transitions
compile_hierarchical_collector = make_hierarchical_collector


def build(
    spec: AgentSpec | str,
    *,
    batch: int = 1,
    backend: str | Mapping[str, Any] = "jax",
) -> AgentHandle:
    """Convenience form of ``default_kit.build(...)``."""

    return default_kit.build(spec, batch=batch, backend=backend)


_LAZY_EXPORTS = {
    "EvaluationResult": ("adk.evaluation", "EvaluationResult"),
    "EvaluationSuite": ("adk.evaluation", "EvaluationSuite"),
    "LoadedCheckpoint": ("adk.core.lifecycle", "LoadedCheckpoint"),
    "PPOConfig": ("hytalegym.jax.training.types", "PPOConfig"),
    "PPOEnvironment": ("hytalegym.jax.training.ppo", "PPOEnvironment"),
    "PPOTrainState": ("hytalegym.jax.training.types", "PPOTrainState"),
    "RecurrentPolicyParams": (
        "hytalegym.jax.training.types",
        "RecurrentPolicyParams",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve optional PPO/checkpoint exports only when explicitly used."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "ActionHead",
    "CompositeActionHead",
    "CompositeActionSpec",
    "ActionIntent",
    "ActionLifecycle",
    "ActionSpace",
    "ActionVerb",
    "AgentFrame",
    "AgentHandle",
    "AgentKit",
    "AgentSpec",
    "ActorPolicyInput",
    "BridgePolicyInput",
    "BridgeTransition",
    "BaseAction",
    "BlockTrigger",
    "BuiltAgent",
    "ContextualLoopState",
    "ContextualLoopTransition",
    "ContextualPolicy",
    "ContextualPolicyInput",
    "ContextualRecordFn",
    "DensePolicySurface",
    "DecisionContext",
    "DecisionContextFrame",
    "DecisionEvent",
    "DecisionEvents",
    "DecisionTiming",
    "EvaluationResult",
    "EvaluationSuite",
    "DodgeDirection",
    "EnvironmentDiagnostics",
    "EnvironmentStep",
    "EnvironmentActionBoundary",
    "EnvironmentActionReceipt",
    "EpisodeBoundary",
    "ExecutionState",
    "ENVIRONMENT_STATE_SELECTOR_SCHEMA",
    "COMBAT_PARAMETER_NAMES",
    "FAIL_CLOSED_SCENE",
    "HYTALE_ENV_PARAMETER_BINDINGS",
    "HytaleEnv",
    "HierarchicalActorInputFn",
    "HierarchicalContextFn",
    "HierarchicalLoopState",
    "HierarchicalLoopTransition",
    "HierarchicalRecordFn",
    "HierarchicalRuntimeDiagnostics",
    "JAX_SCENE_PROVIDER_ARGUMENTS",
    "JaxBuildInputs",
    "LoopState",
    "LoopTransition",
    "JaxTrainingInput",
    "InferenceMetadata",
    "InferenceValidation",
    "InferenceValidationContext",
    "JaxRuntimeSettings",
    "JaxScene",
    "RUNTIME_CONFIG_ARGUMENTS",
    "SceneDescription",
    "LoadedCheckpoint",
    "JointActionCapability",
    "JointActionRejected",
    "JointExecutionFailed",
    "JointExecutionUnavailable",
    "JointValidationUnavailable",
    "NativeBridgeSession",
    "OPEN_FLAT_CONTROL_SCENE",
    "PPOConfig",
    "PPOEnvironment",
    "PPOTrainState",
    "Policy",
    "PolicyInput",
    "MaskedCompositeSample",
    "RecordFn",
    "StepResult",
    "StructuredActionCodec",
    "StaleInferenceReason",
    "Trajectory",
    "WorldDirection",
    "action_mask_hash",
    "build",
    "configure_jax_compilation_cache",
    "capture_inference_metadata",
    "collect_contextual",
    "collect_hierarchical",
    "compile_hierarchical_collector",
    "default_kit",
    "prepare_jax_build_inputs",
    "jax_decision_context",
    "jax_hierarchical_context",
    "manager_decision_due",
    "native_decision_context",
    "describe_scene",
    "list_loadouts",
    "make_scene",
    "scene_from_region_fixture",
    "describe_spawns",
    "fixed_spawns",
    "lit_surface_spawns",
    "region_pool_spawns",
    "select_environment_rows",
    "shifted",
    "swapped",
    "unknown_decision_context",
    "validate_inference_result",
]
