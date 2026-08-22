"""Current learner-v3 bridge session for native transfer evidence.

The Gym package already owns MessagePack, native actor-evidence assembly, the
12-head decoder, and policy-to-``AgentAction`` translation.  This ADK layer
adds ownership: explicit backend selection, the shared lease, stamp pinning,
one-shot session semantics, and result validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from types import MappingProxyType
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.envs.hytale_env import HytaleEnv

from adk.architecture.inputs import ActorPolicyInput
from adk.contracts.stamp import ContractStamp, current_stamp, require_current
from adk.core.spec import AgentSpec
from adk.development import ActionIntent, ActionSpace, EnvironmentDiagnostics
from adk.policy import Policy
from adk.runtime.backends import BackendSelection, resolve_backend
from adk.runtime.actions import (
    CompositeActionSpec,
    EnvironmentActionReceipt,
)
from adk.runtime.episode import EpisodeBoundary
from adk.runtime.lease import NativeEvidenceLease, acquire_native_lease


HYTALE_ENV_PARAMETER_BINDINGS = MappingProxyType(
    {
        "task": "task",
        "host": "backend.host",
        "port": "backend.port",
        "max_episode_steps": "max_episode_steps",
        "ticks_per_step": "ticks_per_step",
        "backend": "bridge_backend",
        "world": "world",
        "npc_role": "npc_role",
        # Optional native combat target role, added by the Gym on 2026-08-12.
        # It is distinct from learner_v3_target_profile: the former selects the
        # spawned NPC role while the latter selects actor-policy equipment.
        "combat_target_role": "combat_target_role",
        "spawn": "spawn",
        "combat_target_active": "combat_target_active",
        "fidelity_fixture": "fidelity_fixture",
        "learner_v3_profile": "spec.loadout",
        "learner_v3_target_profile": "target_profile",
        "learner_v3_world_geometry_config": ("learner_v3_world_geometry_config"),
        # Added by the Gym on 2026-08-03 for policy-selected world actions
        # (block/place/break/craft).  The ADK forwards it but does not yet
        # build one -- the JAX-side seams are still unbound, see
        # adk/HANDOFF-COMBAT.md §11.  Defaults to None, which is the Gym's own
        # default, so forwarding it changes no behaviour.
        "learner_v3_world_action_adapter": ("learner_v3_world_action_adapter"),
        "render_mode": "render_mode",
        # Added by the Gym on 2026-08-09 to select the worldgen V2 structure
        # preset.  The ADK forwards it unchanged; the Gym's own default is
        # "Default", so forwarding it changes no behaviour.
        "worldgen_structure": "worldgen_structure",
        # Server-owned timing controls. Pair 30 * speed TPS with speed dilation
        # to preserve the native 1/30-second logical delta while fast-forwarding.
        "native_tick_rate": "native_tick_rate",
        "native_time_dilation": "native_time_dilation",
    }
)


def _verify_hytale_environment_surface() -> None:
    actual = tuple(inspect.signature(HytaleEnv).parameters)
    expected = tuple(HYTALE_ENV_PARAMETER_BINDINGS)
    if actual != expected:
        raise RuntimeError(
            "Gym HytaleEnv constructor surface changed; update the ADK bridge "
            f"port (expected {expected!r}, found {actual!r})"
        )
    if not isinstance(
        getattr(HytaleEnv, "native_actor_assembly", None),
        property,
    ):
        raise RuntimeError(
            "Gym HytaleEnv must publish native_actor_assembly; the ADK does "
            "not read private Gym state or rebuild structured actor input"
        )


_verify_hytale_environment_surface()


@dataclass(frozen=True, slots=True)
class BridgePolicyInput:
    """One validated native learner-v3 observation and its exact action mask."""

    observation: np.ndarray
    action_mask: np.ndarray
    observation_valid: bool
    failure_bits: int
    mechanics_failure_bits: int
    arsenal_failure_bits: int
    loadout_failure: bool
    host_policy_action_legal: bool
    host_policy_action_reject_reasons: tuple[str, ...]
    native_server_process_uptime_seconds: float | None
    info: Mapping[str, Any]
    legal_observation: Any = None
    actor_evidence: Any = None
    world_capabilities: Any = None
    actor_yaw_degrees: float | None = None
    raw_actor_assembly: Any = None

    def as_policy_input(self) -> ActorPolicyInput:
        """Convert one native row to the same actor input used by JAX."""

        return ActorPolicyInput(
            observation=jnp.asarray(self.observation)[None, :],
            action_mask=jnp.asarray(self.action_mask)[None, :],
            legal_observation=self.legal_observation,
            action_surface=None,
        )

    def as_actor_input(self) -> ActorPolicyInput:
        """Explicit alias for structured-policy code."""

        return self.as_policy_input()

    @property
    def diagnostics(self) -> EnvironmentDiagnostics:
        """Return stable diagnostic names plus the complete bridge ``info``."""

        return EnvironmentDiagnostics.from_info(self.info)

    def decision_context(
        self,
        *,
        previous: "BridgePolicyInput | None" = None,
    ):
        """Normalize this native frame without discarding bridge evidence.

        Supplying the preceding frame enables perceptual edge events from the
        two exact legal observations. With no preceding frame those events
        remain unknown, while timing and native action lifecycle still map.
        """

        from adk.architecture.decision_context import native_decision_context

        if previous is None:
            return native_decision_context(self.info)
        if not isinstance(previous, BridgePolicyInput):
            raise TypeError("previous must be a BridgePolicyInput")
        return native_decision_context(
            self.info,
            actor_input=self.as_actor_input(),
            previous_actor_input=previous.as_actor_input(),
        )


@dataclass(frozen=True, slots=True)
class BridgeTransition:
    """One native evidence transition on the current learner-v3 surface."""

    policy_input: BridgePolicyInput
    action_factors: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    bridge_sha256: str
    native_server_process_uptime_seconds: float | None

    @property
    def frame(self) -> BridgePolicyInput:
        """Use the same next-frame term as the JAX ``StepResult``."""

        return self.policy_input

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def boundary(self) -> EpisodeBoundary:
        """Native Gymnasium causes are both authoritative and available."""

        return EpisodeBoundary(
            done=self.done,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    @property
    def info(self) -> Mapping[str, Any]:
        return self.policy_input.info

    @property
    def diagnostics(self) -> EnvironmentDiagnostics:
        return self.policy_input.diagnostics

    @property
    def action_receipt(self) -> EnvironmentActionReceipt:
        """Retain native translator evidence without claiming atomic execution."""

        detail = (
            "; ".join(self.host_policy_action_reject_reasons)
            if self.host_policy_action_reject_reasons
            else (
                "native bridge does not yet publish whole-composite server "
                "validation and execution evidence"
            )
        )
        return EnvironmentActionReceipt.unavailable(
            self.action_factors,
            reason=detail,
            raw_receipt=self.policy_input.info,
        )

    @property
    def host_policy_action_legal(self) -> bool:
        return self.policy_input.host_policy_action_legal

    @property
    def host_policy_action_reject_reasons(self) -> tuple[str, ...]:
        return self.policy_input.host_policy_action_reject_reasons


def _native_selection(
    backend: str | Mapping[str, Any] | BackendSelection,
) -> BackendSelection:
    if isinstance(backend, BackendSelection):
        selection = backend
    elif isinstance(backend, str):
        selection = resolve_backend(
            {
                "backend": backend,
                "entry_point": "native_evidence",
            }
        )
    elif isinstance(backend, Mapping):
        options = dict(backend)
        options.setdefault("entry_point", "native_evidence")
        selection = resolve_backend(options)
    else:
        raise TypeError("backend must be a name, mapping, or BackendSelection")
    if selection.backend != "native":
        raise ValueError(
            "the published learner-v3 bridge surface currently requires "
            "backend='native'"
        )
    if selection.is_training:
        raise RuntimeError("native bridge sessions are evidence-only")
    return selection


class NativeBridgeSession:
    """Lease-owning, one-reset native learner-v3 evidence session."""

    def __init__(
        self,
        spec: AgentSpec,
        *,
        backend: str | Mapping[str, Any] | BackendSelection = "native",
        stamp: ContractStamp | None = None,
        purpose: str = "agent_adk_native_transfer",
        bridge_backend: str = "native",
        task: str = "kill_trork",
        world: str = "flat",
        worldgen_structure: str = "Default",
        npc_role: str | None = None,
        combat_target_role: str | None = None,
        spawn: tuple[float, float, float] | None = None,
        target_profile: str = "",
        max_episode_steps: int = 6000,
        ticks_per_step: int = 4,
        native_tick_rate: int = 30,
        native_time_dilation: float = 1.0,
        combat_target_active: bool = True,
        fidelity_fixture: str = "default",
        learner_v3_world_geometry_config: Any | None = None,
        learner_v3_world_action_adapter: Any | None = None,
        render_mode: str | None = None,
    ) -> None:
        if not isinstance(spec, AgentSpec):
            raise TypeError("spec must be an AgentSpec")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("purpose must be a non-empty string")
        if bridge_backend not in {"native", "headless"}:
            raise ValueError("bridge_backend must be 'native' or 'headless'")
        for name, value in (
            ("task", task),
            ("world", world),
            ("worldgen_structure", worldgen_structure),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if npc_role is not None and (
            not isinstance(npc_role, str)
            or not npc_role
            or npc_role != npc_role.strip()
        ):
            raise ValueError("npc_role must be None or a non-empty string")
        if combat_target_role is not None:
            if (
                not isinstance(combat_target_role, str)
                or not combat_target_role
                or combat_target_role != combat_target_role.strip()
            ):
                raise ValueError(
                    "combat_target_role must be None or a non-empty string"
                )
            if task != "kill_trork" or bridge_backend != "native":
                raise ValueError(
                    "combat_target_role requires kill_trork and a native bridge"
                )
        normalized_spawn = None
        if spawn is not None:
            if not isinstance(spawn, (list, tuple)) or len(spawn) != 3:
                raise ValueError("spawn must contain exactly three coordinates")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                for value in spawn
            ):
                raise ValueError("spawn coordinates must be finite numbers")
            normalized_spawn = tuple(float(value) for value in spawn)
        for name, value in (
            ("max_episode_steps", max_episode_steps),
            ("ticks_per_step", ticks_per_step),
            ("native_tick_rate", native_tick_rate),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if native_tick_rate > 2048:
            raise ValueError("native_tick_rate must be at most 2048")
        if (
            isinstance(native_time_dilation, bool)
            or not isinstance(native_time_dilation, (int, float))
            or not np.isfinite(float(native_time_dilation))
            or not 0.01 < float(native_time_dilation) <= 4.0
        ):
            raise ValueError("native_time_dilation must be finite and in (0.01, 4]")
        if not isinstance(combat_target_active, bool):
            raise TypeError("combat_target_active must be bool")
        if not isinstance(target_profile, str):
            raise TypeError("target_profile must be a string")
        if (
            not isinstance(fidelity_fixture, str)
            or not fidelity_fixture
            or fidelity_fixture != fidelity_fixture.strip()
        ):
            raise ValueError("fidelity_fixture must be a non-empty string")
        if render_mode is not None and (
            not isinstance(render_mode, str)
            or not render_mode
            or render_mode != render_mode.strip()
        ):
            raise ValueError("render_mode must be None or a non-empty string")
        self.spec = spec
        self.backend = _native_selection(backend)
        self.bridge_backend = bridge_backend
        self.learner_v3_world_geometry_config = learner_v3_world_geometry_config
        self.learner_v3_world_action_adapter = learner_v3_world_action_adapter
        self.render_mode = render_mode
        self.stamp = (
            current_stamp(learner_v3_world_geometry_config) if stamp is None else stamp
        )
        require_current(
            self.stamp,
            self.learner_v3_world_geometry_config,
        )
        self.purpose = purpose.strip()
        self.task = task
        self.world = world
        self.worldgen_structure = worldgen_structure
        self.npc_role = npc_role
        self.combat_target_role = combat_target_role
        self.spawn = normalized_spawn
        self.target_profile = target_profile
        self.max_episode_steps = max_episode_steps
        self.ticks_per_step = ticks_per_step
        self.native_tick_rate = native_tick_rate
        self.native_time_dilation = float(native_time_dilation)
        self.combat_target_active = combat_target_active
        self.fidelity_fixture = fidelity_fixture
        self._lease: NativeEvidenceLease | None = None
        self._environment: HytaleEnv | None = None
        self._policy_input: BridgePolicyInput | None = None
        self._reset_used = False
        self._can_step = False
        self._last_uptime: float | None = None
        self._closed = False

    def __enter__(self) -> "NativeBridgeSession":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def opened(self) -> bool:
        return self._lease is not None and self._environment is not None

    @property
    def policy_input(self) -> BridgePolicyInput | None:
        return self._policy_input

    @property
    def action_space(self) -> ActionSpace:
        """Expose the same named factor contract used by the JAX handle."""

        return ActionSpace.from_contract(
            self.stamp.action_head_names,
            self.stamp.action_head_sizes,
        )

    @property
    def composite_action_spec(self) -> CompositeActionSpec:
        """Expose the same lossless factor contract as the JAX handle."""

        return CompositeActionSpec.from_live_contract(self.stamp)

    def make_action(self, intent: ActionIntent) -> jax.Array:
        """Encode a semantic intent against the current native action mask."""

        if self._policy_input is None:
            raise RuntimeError("reset the native bridge session before making actions")
        return self.action_space.encode_intent(
            intent,
            batch=1,
            policy_input=self._policy_input,
        )

    def make_action_factors(
        self,
        *,
        choices: Mapping[str, Any] | None = None,
        **named_choices: Any,
    ) -> jax.Array:
        """Build named factors against the current native action mask."""

        if self._policy_input is None:
            raise RuntimeError("reset the native bridge session before making actions")
        return self.action_space.compose(
            batch=1,
            policy_input=self._policy_input,
            choices=choices,
            **named_choices,
        )

    @property
    def raw_environment(self) -> HytaleEnv | None:
        """Borrow the actual upstream ``HytaleEnv`` while this session is open."""

        return self._environment

    @property
    def raw_lease(self) -> NativeEvidenceLease | None:
        """Borrow the ADK's process-wide native-evidence lease while open."""

        return self._lease

    def open(self) -> "NativeBridgeSession":
        if self.opened:
            return self
        if self._closed:
            raise RuntimeError(
                "native bridge sessions are single-use; create a new session"
            )
        require_current(
            self.stamp,
            self.learner_v3_world_geometry_config,
        )
        lease = acquire_native_lease(
            self.backend,
            purpose=self.purpose,
            evidence_kinds=("learner_v3_native_transfer",),
        )
        try:
            environment = HytaleEnv(
                task=self.task,
                host=self.backend.host,
                port=self.backend.port,
                max_episode_steps=self.max_episode_steps,
                ticks_per_step=self.ticks_per_step,
                native_tick_rate=self.native_tick_rate,
                native_time_dilation=self.native_time_dilation,
                backend=self.bridge_backend,
                world=self.world,
                worldgen_structure=self.worldgen_structure,
                npc_role=self.npc_role,
                combat_target_role=self.combat_target_role,
                spawn=self.spawn,
                combat_target_active=self.combat_target_active,
                fidelity_fixture=self.fidelity_fixture,
                learner_v3_profile=self.spec.loadout,
                learner_v3_target_profile=self.target_profile,
                learner_v3_world_geometry_config=(
                    self.learner_v3_world_geometry_config
                ),
                learner_v3_world_action_adapter=(self.learner_v3_world_action_adapter),
                render_mode=self.render_mode,
            )
        except Exception:
            lease.close()
            raise
        self._lease = lease
        self._environment = environment
        return self

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> BridgePolicyInput:
        if not self.opened or self._environment is None:
            raise RuntimeError("open the native bridge session before reset")
        if self._reset_used:
            raise RuntimeError(
                "native learner-v3 reset is one-shot; create a new session "
                "for another episode"
            )
        reset_options = dict(options or {})
        configured_backend = reset_options.get(
            "backend",
            self.bridge_backend,
        )
        if configured_backend != self.bridge_backend:
            raise ValueError("reset options cannot override the bridge backend")
        reset_options["backend"] = self.bridge_backend
        require_current(
            self.stamp,
            self.learner_v3_world_geometry_config,
        )
        # A native reset attempt can mutate the listener even when parsing or
        # validation later fails, so retries on this object are unsafe.
        self._reset_used = True
        observation, info = self._environment.reset(
            seed=seed,
            options=reset_options,
        )
        self._policy_input = self._validate_policy_input(observation, info)
        self._last_uptime = self._policy_input.native_server_process_uptime_seconds
        self._can_step = True
        require_current(
            self.stamp,
            self.learner_v3_world_geometry_config,
        )
        return self._policy_input

    def step(self, factors: ActionIntent | Any) -> BridgeTransition:
        if not self.opened or self._environment is None or self._policy_input is None:
            raise RuntimeError("reset the native bridge session before step")
        if not self._can_step:
            raise RuntimeError(
                "native bridge episode is terminal or its prior step failed"
            )
        require_current(
            self.stamp,
            self.learner_v3_world_geometry_config,
        )
        if isinstance(factors, ActionIntent):
            factors = self.make_action(factors)
        values = np.asarray(factors)
        if values.shape == (1, len(self.stamp.action_head_sizes)):
            values = values[0]
        expected_shape = (len(self.stamp.action_head_sizes),)
        if values.dtype != np.dtype(np.int32):
            raise TypeError("native action factors must have dtype int32")
        if values.shape != expected_shape:
            raise ValueError(f"native action factors must have shape {expected_shape}")
        offset = 0
        for index, size in enumerate(self.stamp.action_head_sizes):
            value = int(values[index])
            if value < 0 or value >= size:
                raise ValueError(f"native action factor {index} is out of range")
            if not bool(self._policy_input.action_mask[offset + value]):
                raise ValueError(f"native action factor {index} is masked out")
            offset += size

        # A transport attempt can mutate native state even if parsing fails.
        # Leave the session unsteppable unless a valid nonterminal response
        # completes the transaction.
        self._can_step = False
        observation, reward, terminated, truncated, info = self._environment.step(
            values
        )
        policy_input = self._validate_policy_input(observation, info)
        uptime = policy_input.native_server_process_uptime_seconds
        if (
            self._last_uptime is not None
            and uptime is not None
            and uptime < self._last_uptime
        ):
            raise ValueError(
                "native server process uptime decreased during the session"
            )
        require_current(
            self.stamp,
            self.learner_v3_world_geometry_config,
        )
        self._policy_input = policy_input
        self._last_uptime = uptime
        self._can_step = not (bool(terminated) or bool(truncated))
        return BridgeTransition(
            policy_input=policy_input,
            action_factors=values.copy(),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            bridge_sha256=self.stamp.contracts["bridge_jar"],
            native_server_process_uptime_seconds=uptime,
        )

    def step_policy(
        self,
        policy: Policy,
        carry: Any,
        key: jax.Array,
    ) -> tuple[Any, BridgeTransition]:
        """Run one ADK policy decision on the current native observation."""

        if self._policy_input is None:
            raise RuntimeError("reset the native bridge session before policy step")
        if not self._can_step:
            raise RuntimeError(
                "native bridge episode is terminal or its prior step failed"
            )
        if not callable(policy):
            raise TypeError("policy must satisfy the ADK Policy protocol")
        next_carry, factors = policy(
            carry,
            self._policy_input.as_policy_input(),
            key,
        )
        return next_carry, self.step(jax.device_get(factors))

    def close(self) -> None:
        environment, self._environment = self._environment, None
        lease, self._lease = self._lease, None
        self._policy_input = None
        self._can_step = False
        self._last_uptime = None
        self._closed = True
        try:
            if environment is not None:
                environment.close()
        finally:
            if lease is not None:
                lease.close()

    def _validate_policy_input(
        self,
        observation: Any,
        info: Mapping[str, Any],
    ) -> BridgePolicyInput:
        if not isinstance(info, Mapping):
            raise TypeError("native learner-v3 info must be a mapping")
        bridge_sha256 = info.get("bridge_sha256")
        if (
            not isinstance(bridge_sha256, str)
            or bridge_sha256.strip().upper() != self.stamp.contracts["bridge_jar"]
        ):
            raise ValueError("native listener bridge identity does not match the stamp")
        evidence_sha256 = info.get("learner_v3_actor_evidence_contract_sha256")
        if (
            not isinstance(evidence_sha256, str)
            or evidence_sha256.strip().upper()
            != self.stamp.contracts["learner_v3_actor_evidence"]
        ):
            raise ValueError(
                "native learner-v3 actor-evidence identity does not match the stamp"
            )
        values = np.asarray(observation)
        if values.dtype != np.dtype(np.float32):
            raise TypeError("native learner-v3 observation must have dtype float32")
        if values.shape != (self.stamp.observation_size,):
            raise ValueError(
                "native learner-v3 observation does not match the JAX surface"
            )
        if not bool(np.all(np.isfinite(values))):
            raise ValueError("native learner-v3 observation contains non-finite values")
        mask = np.asarray(info.get("learner_v3_action_mask"))
        if mask.dtype != np.dtype(np.bool_):
            raise TypeError("native learner-v3 action mask must have dtype bool")
        if mask.shape != (sum(self.stamp.action_head_sizes),):
            raise ValueError("native learner-v3 action mask has the wrong width")
        offset = 0
        for index, size in enumerate(self.stamp.action_head_sizes):
            if not bool(np.any(mask[offset : offset + size])):
                raise ValueError(
                    f"native learner-v3 action head {index} has no legal choice"
                )
            offset += size
        observation_valid = _required_bool(
            info,
            "learner_v3_observation_valid",
        )
        failure_bits = _required_failure_bits(
            info,
            "learner_v3_failure_bits",
        )
        mechanics_failure_bits = _required_failure_bits(
            info,
            "learner_v3_mechanics_failure_bits",
        )
        arsenal_failure_bits = _required_failure_bits(
            info,
            "learner_v3_arsenal_failure_bits",
        )
        loadout_failure = _required_bool(
            info,
            "learner_v3_loadout_failure",
        )
        host_action_legal = _required_bool(
            info,
            "host_policy_action_legal",
        )
        reject_reasons = info.get("host_policy_action_reject_reasons")
        if not isinstance(reject_reasons, (list, tuple)) or any(
            not isinstance(reason, str) for reason in reject_reasons
        ):
            raise TypeError(
                "host_policy_action_reject_reasons must be a string sequence"
            )
        uptime = _optional_uptime(info)
        assembly = None
        if self._environment is not None:
            # Gym owns this object and creates it from actor evidence before
            # returning the dense row. Retain the exact object through Gym's
            # public seam; never rebuild a structured observation from the
            # flattened projection.
            assembly = self._environment.native_actor_assembly
        if assembly is not None:
            assembly_observation = np.asarray(assembly.policy_observation)
            assembly_mask = np.asarray(assembly.policy_action_mask)
            if not np.array_equal(assembly_observation, values):
                raise ValueError(
                    "native structured assembly and returned policy row differ"
                )
            if not np.array_equal(assembly_mask, mask):
                raise ValueError(
                    "native structured assembly and returned action mask differ"
                )
        return BridgePolicyInput(
            observation=values.copy(),
            action_mask=mask.copy(),
            observation_valid=observation_valid,
            failure_bits=failure_bits,
            mechanics_failure_bits=mechanics_failure_bits,
            arsenal_failure_bits=arsenal_failure_bits,
            loadout_failure=loadout_failure,
            host_policy_action_legal=host_action_legal,
            host_policy_action_reject_reasons=tuple(reject_reasons),
            native_server_process_uptime_seconds=uptime,
            info=dict(info),
            legal_observation=(None if assembly is None else assembly.observation),
            actor_evidence=(None if assembly is None else assembly.evidence),
            world_capabilities=(
                None if assembly is None else assembly.world_capabilities
            ),
            actor_yaw_degrees=(
                None if assembly is None else assembly.actor_yaw_degrees
            ),
            raw_actor_assembly=assembly,
        )


def _required_bool(info: Mapping[str, Any], name: str) -> bool:
    value = info.get(name)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be bool")
    return bool(value)


def _required_failure_bits(info: Mapping[str, Any], name: str) -> int:
    value = info.get(name)
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise TypeError(f"{name} must be a nonnegative integer")
    return int(value)


def _optional_uptime(info: Mapping[str, Any]) -> float | None:
    uptime = info.get("native_server_process_uptime_seconds")
    if uptime is None:
        return None
    if (
        isinstance(uptime, bool)
        or not isinstance(uptime, (int, float))
        or not np.isfinite(float(uptime))
        or float(uptime) < 0.0
    ):
        raise ValueError("native server uptime is invalid")
    return float(uptime)


# The old ADK class never spoke the learner-v3 surface.  Keep the familiar
# import name while routing all new callers through the current session.
NativeBridgeClient = NativeBridgeSession


__all__ = [
    "BridgePolicyInput",
    "BridgeTransition",
    "HYTALE_ENV_PARAMETER_BINDINGS",
    "NativeBridgeClient",
    "NativeBridgeSession",
]
