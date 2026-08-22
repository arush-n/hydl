"""Lossless developer-facing terms for JAX and native agent interaction.

This module intentionally contains no environment mechanics.  It names the
objects an agent developer handles while preserving the upstream Gym arrays,
state, structured actions, and diagnostic PyTrees without projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.mechanics import (
    DODGE_BACK,
    DODGE_FORWARD,
    DODGE_LEFT,
    DODGE_NONE,
    DODGE_RIGHT,
)
from hytalegym.jax.combat.observation import (
    ACTION_DOOR_CLOSE,
    ACTION_DOOR_OPEN,
    ACTION_DOOR_USE,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ACTOR_RECIPE_CANDIDATE_CAPACITY,
    ARSENAL_BLOCK_TRIGGER_PRIMARY,
    ARSENAL_BLOCK_TRIGGER_SECONDARY,
    ARSENAL_POLICY_COMPASS_DIRECTIONS,
    DODGE_ACTION_COUNT,
)
from hytalegym.jax.combat.skills import (
    SKILL_APPROACH,
    SKILL_APPROACH_ATTACK,
    SKILL_ATTACK,
    SKILL_FACE_TARGET,
    SKILL_IDLE,
    SKILL_RETREAT,
    SKILL_RETREAT_ATTACK,
    SKILL_STRAFE_LEFT,
    SKILL_STRAFE_RIGHT,
)

from adk.runtime.env_adapter import (
    JaxEnvironmentState,
    PolicyInput,
    decode,
    encode,
    neutral_actions,
)
from adk.runtime.episode import EpisodeBoundary
from adk.runtime.actions import CompositeActionSpec


class BaseAction(IntEnum):
    """Semantic choices in the live ``base_action`` head."""

    IDLE = SKILL_IDLE
    FACE_TARGET = SKILL_FACE_TARGET
    APPROACH = SKILL_APPROACH
    RETREAT = SKILL_RETREAT
    STRAFE_LEFT = SKILL_STRAFE_LEFT
    STRAFE_RIGHT = SKILL_STRAFE_RIGHT
    ATTACK = SKILL_ATTACK
    APPROACH_ATTACK = SKILL_APPROACH_ATTACK
    RETREAT_ATTACK = SKILL_RETREAT_ATTACK
    DOOR_OPEN = ACTION_DOOR_OPEN
    DOOR_CLOSE = ACTION_DOOR_CLOSE
    DOOR_USE = ACTION_DOOR_USE


class DodgeDirection(IntEnum):
    """Semantic dodge choices, encoded in the tail of ``locomotion_gait_compass``."""

    NONE = DODGE_NONE
    FORWARD = DODGE_FORWARD
    BACK = DODGE_BACK
    LEFT = DODGE_LEFT
    RIGHT = DODGE_RIGHT


class WorldDirection(IntEnum):
    """World-XZ compass choices, clockwise from North."""

    NONE = 0
    NORTH = 1
    NORTH_EAST = 2
    EAST = 3
    SOUTH_EAST = 4
    SOUTH = 5
    SOUTH_WEST = 6
    WEST = 7
    NORTH_WEST = 8


class BlockTrigger(IntEnum):
    """Authored equipped-item trigger used with a block candidate."""

    PRIMARY = ARSENAL_BLOCK_TRIGGER_PRIMARY
    SECONDARY = ARSENAL_BLOCK_TRIGGER_SECONDARY


@dataclass(frozen=True, slots=True)
class ActionIntent:
    """One semantic action intent, scalar or batched.

    Candidate slots are zero-based; ``-1`` means no ability, recipe, or block
    target.  Fields may be scalars or arrays of shape ``(B,)``.  Encoding is
    delegated to the upstream Gym action encoder, then checked against the
    exact mask from the frame on which the action will operate.
    """

    base_action: Any = BaseAction.IDLE
    ability_slot: Any = -1
    guard: Any = False
    dodge: Any = DodgeDirection.NONE
    jump: Any = False
    world_direction: Any = WorldDirection.NONE
    yaw_delta_degrees: Any = 0.0
    pitch_delta_degrees: Any = 0.0
    use: Any = False
    block_trigger: Any = BlockTrigger.PRIMARY
    recipe_candidate: Any = -1
    block_candidate: Any = -1


@dataclass(frozen=True, slots=True)
class ActionHead:
    """One named categorical head in the exact live action ABI."""

    index: int
    name: str
    size: int
    offset: int
    neutral_choice: int


@dataclass(frozen=True, slots=True)
class ActionSpace:
    """Named view over the exact factored action contract.

    The space derives names, sizes, offsets, and physical neutral choices from
    the live stamp and upstream neutral-action implementation.  It never
    flattens the factors into a scalar action.
    """

    heads: tuple[ActionHead, ...]
    composite: CompositeActionSpec

    @classmethod
    def from_contract(
        cls,
        head_names: tuple[str, ...],
        head_sizes: tuple[int, ...],
    ) -> "ActionSpace":
        names = tuple(head_names)
        sizes = tuple(head_sizes)
        if not names or len(names) != len(sizes):
            raise ValueError(
                "action head names and sizes must be non-empty and aligned"
            )
        if len(set(names)) != len(names):
            raise ValueError("action head names must be unique")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in sizes
        ):
            raise ValueError("action head sizes must be positive integers")
        neutral = np.asarray(jax.device_get(neutral_actions(1)))
        if neutral.shape != (1, len(names)):
            raise RuntimeError(
                "upstream neutral action disagrees with the live head count"
            )
        offset = 0
        heads: list[ActionHead] = []
        for index, (name, size) in enumerate(zip(names, sizes, strict=True)):
            choice = int(neutral[0, index])
            if choice < 0 or choice >= size:
                raise RuntimeError(f"upstream neutral choice is invalid for {name!r}")
            heads.append(ActionHead(index, name, size, offset, choice))
            offset += size
        return cls(
            tuple(heads),
            CompositeActionSpec.from_contract(names, sizes),
        )

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(head.name for head in self.heads)

    @property
    def head_sizes(self) -> tuple[int, ...]:
        return tuple(head.size for head in self.heads)

    @property
    def logit_size(self) -> int:
        return sum(self.head_sizes)

    def head(self, name: str) -> ActionHead:
        for head in self.heads:
            if head.name == name:
                return head
        raise KeyError(f"unknown action head: {name!r}")

    def metadata(self) -> dict[str, object]:
        """Return a JSON-safe description of the exact live action layout."""

        return {
            "transport": "factors",
            "factor_dtype": "int32",
            "logit_size": self.logit_size,
            "heads": [
                {
                    "index": head.index,
                    "name": head.name,
                    "size": head.size,
                    "offset": head.offset,
                    "neutral_choice": head.neutral_choice,
                }
                for head in self.heads
            ],
        }

    def neutral(self, batch: int) -> jax.Array:
        values = neutral_actions(batch)
        self.validate(values, expected_batch=batch)
        return values

    def split_factors(self, factors: Any) -> dict[str, jax.Array]:
        return self.composite.split_factors(factors)

    def split_mask(self, policy_input_or_mask: Any) -> dict[str, jax.Array]:
        mask = self._mask(policy_input_or_mask)
        return self.composite.split_mask(mask)

    def compose(
        self,
        *,
        batch: int,
        policy_input: Any | None = None,
        choices: Mapping[str, Any] | None = None,
        **named_choices: Any,
    ) -> jax.Array:
        """Build factors from named head choices, defaulting to physical no-op.

        This is a host-side developer helper.  Compiled policies should emit
        their factor array directly or use the handle's masked sampler.
        """

        selected = dict(choices or {})
        overlap = selected.keys() & named_choices.keys()
        if overlap:
            raise ValueError(
                "action heads were provided twice: " + ", ".join(sorted(overlap))
            )
        selected.update(named_choices)
        unknown = selected.keys() - set(self.head_names)
        if unknown:
            raise KeyError("unknown action heads: " + ", ".join(sorted(unknown)))
        factors = self.neutral(batch)
        for name, raw_choice in selected.items():
            head = self.head(name)
            choice = _integer_batch_value(raw_choice, batch, name)
            host_choice = np.asarray(jax.device_get(choice))
            if bool(np.any((host_choice < 0) | (host_choice >= head.size))):
                raise ValueError(f"{name} choices must be in [0, {head.size - 1}]")
            factors = factors.at[:, head.index].set(choice)
        return self.require_legal(factors, policy_input)

    def encode_intent(
        self,
        intent: ActionIntent,
        *,
        batch: int,
        policy_input: Any | None = None,
    ) -> jax.Array:
        """Encode semantic terms through Gym's exact structured-action codec."""

        if not isinstance(intent, ActionIntent):
            raise TypeError("intent must be an ActionIntent")
        _validate_intent_host(intent, batch, self)
        structured = decode(self.neutral(batch))._replace(
            skill_id=_integer_batch_value(intent.base_action, batch, "base_action"),
            ability_slot=_integer_batch_value(
                intent.ability_slot,
                batch,
                "ability_slot",
            ),
            guard_held=_bool_batch_value(intent.guard, batch, "guard"),
            dodge_direction=_integer_batch_value(intent.dodge, batch, "dodge"),
            jump_held=_bool_batch_value(intent.jump, batch, "jump"),
            world_move_direction=_integer_batch_value(
                intent.world_direction,
                batch,
                "world_direction",
            ),
            yaw_delta_degrees=_float_batch_value(
                intent.yaw_delta_degrees,
                batch,
                "yaw_delta_degrees",
            ),
            pitch_delta_degrees=_float_batch_value(
                intent.pitch_delta_degrees,
                batch,
                "pitch_delta_degrees",
            ),
            use_requested=_bool_batch_value(intent.use, batch, "use"),
            block_interaction_trigger=_integer_batch_value(
                intent.block_trigger,
                batch,
                "block_trigger",
            ),
            recipe_candidate_index=_integer_batch_value(
                intent.recipe_candidate,
                batch,
                "recipe_candidate",
            ),
            block_candidate_index=_integer_batch_value(
                intent.block_candidate,
                batch,
                "block_candidate",
            ),
        )
        factors = encode(structured)
        self.validate(factors, expected_batch=batch)
        return self.require_legal(factors, policy_input)

    def decode(self, factors: Any) -> Any:
        """Return Gym's exact structured learner action without projection."""

        return decode(self.validate(factors))

    def validate(
        self,
        factors: Any,
        *,
        expected_batch: int | None = None,
    ) -> jax.Array:
        return self.composite.validate_factors(
            factors,
            expected_batch=expected_batch,
        )

    def require_legal(
        self,
        factors: Any,
        policy_input: Any | None,
    ) -> jax.Array:
        values = self.validate(factors)
        if policy_input is None:
            return values
        return self.composite.require_per_head_legal(
            values,
            self._mask(policy_input),
        )

    def _mask(self, policy_input_or_mask: Any) -> jax.Array:
        raw = getattr(policy_input_or_mask, "action_mask", policy_input_or_mask)
        mask = jnp.asarray(raw)
        if mask.dtype != jnp.dtype(jnp.bool_):
            raise TypeError("action mask must have dtype bool")
        if mask.ndim == 1:
            mask = mask[None, :]
        if mask.ndim != 2 or mask.shape[1] != self.logit_size:
            raise ValueError(f"action mask must have shape (B, {self.logit_size})")
        return mask


class AgentFrame(NamedTuple):
    """Exact JAX environment state and the policy input derived from it."""

    state: JaxEnvironmentState
    policy_input: PolicyInput

    @property
    def observation(self) -> jax.Array:
        return self.policy_input.observation

    @property
    def action_mask(self) -> jax.Array:
        return self.policy_input.action_mask


class StepResult(NamedTuple):
    """One lossless JAX transition in developer-facing terms.

    ``done`` remains the compatibility boundary signal.  The optional cause
    fields are ``None`` when the backend published only collapsed ``done``;
    callers that require time-limit-aware bootstrapping should use
    ``boundary.require_split()`` and fail closed until the backend supplies
    both values.
    """

    frame: AgentFrame
    action_factors: jax.Array
    reward: jax.Array
    done: jax.Array
    info: Any
    terminated: jax.Array | None = None
    truncated: jax.Array | None = None

    @property
    def boundary(self) -> EpisodeBoundary:
        """Episode-boundary values and explicit cause availability."""

        return EpisodeBoundary(
            done=self.done,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    @property
    def diagnostics(self) -> "EnvironmentDiagnostics":
        return EnvironmentDiagnostics.from_info(self.info)


@dataclass(frozen=True, slots=True)
class EnvironmentDiagnostics:
    """Stable names for common diagnostics plus the complete original value.

    Fields absent on one backend are ``None``.  ``raw`` is never filtered, so
    new upstream fields remain reachable before the ADK gives them a stable
    convenience name.
    """

    raw: Any
    observation_valid: Any = None
    action_valid: Any = None
    action_surface_legal: Any = None
    world_verb_legal: Any = None
    action_surface_lifecycle: Any = None
    combat: Any = None
    arsenal: Any = None
    failure_bits: Any = None
    mechanics_failure_bits: Any = None
    arsenal_failure_bits: Any = None
    loadout_failure: Any = None
    host_action_legal: Any = None
    host_action_reject_reasons: Any = None
    bridge_sha256: Any = None
    native_server_process_uptime_seconds: Any = None

    @classmethod
    def from_info(cls, info: Any) -> "EnvironmentDiagnostics":
        return cls(
            raw=info,
            observation_valid=_read(
                info,
                "observation_valid",
                "learner_v3_observation_valid",
            ),
            action_valid=_read(info, "action_valid"),
            action_surface_legal=_read(info, "action_surface_legal"),
            world_verb_legal=_read(info, "world_verb_legal"),
            action_surface_lifecycle=_read(info, "action_surface_lifecycle"),
            combat=_read(info, "combat_info", "combat"),
            arsenal=_read(info, "arsenal_info", "arsenal"),
            failure_bits=_read(info, "failure_bits", "learner_v3_failure_bits"),
            mechanics_failure_bits=_read(
                info,
                "mechanics_failure_bits",
                "learner_v3_mechanics_failure_bits",
            ),
            arsenal_failure_bits=_read(
                info,
                "arsenal_failure_bits",
                "learner_v3_arsenal_failure_bits",
            ),
            loadout_failure=_read(
                info,
                "loadout_failure",
                "learner_v3_loadout_failure",
            ),
            host_action_legal=_read(
                info,
                "host_action_legal",
                "host_policy_action_legal",
            ),
            host_action_reject_reasons=_read(
                info,
                "host_action_reject_reasons",
                "host_policy_action_reject_reasons",
            ),
            bridge_sha256=_read(info, "bridge_sha256"),
            native_server_process_uptime_seconds=_read(
                info,
                "native_server_process_uptime_seconds",
            ),
        )

    def to_host(self) -> "EnvironmentDiagnostics":
        """Copy every array to host while preserving the complete structure."""

        return EnvironmentDiagnostics(
            **{
                name: _to_host(getattr(self, name))
                for name in self.__dataclass_fields__
            }
        )

    def standard(self, *, include_raw: bool = False) -> dict[str, Any]:
        """Return named fields for logging without discarding ``raw``."""

        names = tuple(self.__dataclass_fields__)
        if not include_raw:
            names = tuple(name for name in names if name != "raw")
        return {name: getattr(self, name) for name in names}


def _validate_intent_host(
    intent: ActionIntent,
    batch: int,
    space: ActionSpace,
) -> None:
    ranges = (
        ("base_action", intent.base_action, 0, space.head("base_action").size - 1),
        (
            "ability_slot",
            intent.ability_slot,
            -1,
            space.head("ability_none_plus_slots").size - 2,
        ),
        # Dodge, world direction and recipe candidate no longer have heads of
        # their own -- the first two are spans inside `locomotion_gait_compass`
        # and the third left the policy surface entirely -- so these bounds come
        # from the contract constants rather than from a head width.
        ("dodge", intent.dodge, 0, DODGE_ACTION_COUNT),
        (
            "world_direction",
            intent.world_direction,
            0,
            ARSENAL_POLICY_COMPASS_DIRECTIONS,
        ),
        (
            "recipe_candidate",
            intent.recipe_candidate,
            -1,
            ACTOR_RECIPE_CANDIDATE_CAPACITY - 1,
        ),
        (
            "block_candidate",
            intent.block_candidate,
            -1,
            space.head("block_none_plus_candidates").size - 2,
        ),
    )
    for name, value, minimum, maximum in ranges:
        host = np.asarray(jax.device_get(_integer_batch_value(value, batch, name)))
        if bool(np.any((host < minimum) | (host > maximum))):
            raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    trigger = np.asarray(
        jax.device_get(
            _integer_batch_value(intent.block_trigger, batch, "block_trigger")
        )
    )
    if not bool(
        np.all(
            (trigger == int(BlockTrigger.PRIMARY))
            | (trigger == int(BlockTrigger.SECONDARY))
        )
    ):
        raise ValueError("block_trigger must be PRIMARY or SECONDARY")
    for name, value in (
        ("yaw_delta_degrees", intent.yaw_delta_degrees),
        ("pitch_delta_degrees", intent.pitch_delta_degrees),
    ):
        host = np.asarray(jax.device_get(_float_batch_value(value, batch, name)))
        if bool(np.any(~np.isfinite(host))) or bool(np.any(np.abs(host) > 45.0)):
            raise ValueError(f"{name} must be finite and in [-45, 45]")


def _integer_batch_value(value: Any, batch: int, name: str) -> jax.Array:
    raw = np.asarray(jax.device_get(value))
    if raw.dtype == np.dtype(np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if raw.ndim == 0:
        raw = np.full((batch,), raw.item(), dtype=np.int32)
    if raw.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have shape ({batch},)")
    return jnp.asarray(raw, dtype=jnp.int32)


def _bool_batch_value(value: Any, batch: int, name: str) -> jax.Array:
    raw = np.asarray(jax.device_get(value))
    if raw.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must contain booleans")
    if raw.ndim == 0:
        raw = np.full((batch,), bool(raw), dtype=np.bool_)
    if raw.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have shape ({batch},)")
    return jnp.asarray(raw, dtype=jnp.bool_)


def _float_batch_value(value: Any, batch: int, name: str) -> jax.Array:
    raw = np.asarray(jax.device_get(value))
    if raw.dtype == np.dtype(np.bool_) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must contain numbers")
    if raw.ndim == 0:
        raw = np.full((batch,), raw.item(), dtype=np.float32)
    if raw.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have shape ({batch},)")
    return jnp.asarray(raw, dtype=jnp.float32)


def _read(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _to_host(value: Any) -> Any:
    if value is None:
        return None
    try:
        return jax.device_get(value)
    except (TypeError, ValueError):
        return value


__all__ = [
    "ActionHead",
    "ActionIntent",
    "ActionSpace",
    "AgentFrame",
    "BaseAction",
    "BlockTrigger",
    "DodgeDirection",
    "EpisodeBoundary",
    "EnvironmentDiagnostics",
    "StepResult",
    "WorldDirection",
]
