"""Conservative Arena labels from native NPC steering traces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import uuid
from typing import Any

import numpy as np
import numpy.typing as npt

from arena.imitation import (
    TraceSequence,
    native_duel_trace_sequences,
    native_trace_sequence,
)
from arena.jax_contract import HEAD_SPANS
from arena.training.imitation import (
    NATIVE_HEAD_SIZES,
    BehaviorCloningConfig,
    DemonstrationBatch,
    demonstration_batch,
)
from hytalegym.jax.combat.observation.v3.policy.look_deltas import (
    look_delta_values,
)
from hytalegym.jax.combat.skills import SKILL_ATTACK, SKILL_IDLE
from hytalegym.worldgen import NativeNpcTraceCapture


_HEAD_NAMES = tuple(HEAD_SPANS)
_BASE = _HEAD_NAMES.index("base_action")
_ABILITY = _HEAD_NAMES.index("ability_none_plus_slots")
_MOVE = _HEAD_NAMES.index("locomotion_gait_compass")
_YAW = _HEAD_NAMES.index("yaw_delta_bins")
_PITCH = _HEAD_NAMES.index("pitch_delta_bins")
_BODY_TRANSLATION = 1 << 0
_HEAD_YAW = 1 << 3
_HEAD_PITCH = 1 << 4
_BODY_PRESENT = 1 << 5
_HEAD_PRESENT = 1 << 6


@dataclass(frozen=True, slots=True)
class NativeNpcBehaviorLabels:
    """Exact native labels aligned to one actor's BC rows."""

    delta_seconds: npt.NDArray[np.float32]
    control: npt.NDArray[np.float64]
    control_mask: npt.NDArray[np.int32]
    attack_action_active: npt.NDArray[np.bool_]
    attack_activation: npt.NDArray[np.bool_]
    attack_execution_cause: tuple[Any, ...]
    combat_attack: npt.NDArray[np.bool_]
    next_combat_attack: npt.NDArray[np.bool_]
    attack_pause_seconds: npt.NDArray[np.float32]
    next_attack_pause_seconds: npt.NDArray[np.float32]
    target_uuid: tuple[uuid.UUID | None, ...]
    next_target_uuid: tuple[uuid.UUID | None, ...]
    state_name: tuple[str, ...]
    next_state_name: tuple[str, ...]
    body_instruction: tuple[str, ...]
    head_instruction: tuple[str, ...]
    active_actions: tuple[tuple[str, ...], ...]
    attack_actions: tuple[tuple[Any, ...], ...]
    next_attack_actions: tuple[tuple[Any, ...], ...]
    interactions: tuple[tuple[Any, ...], ...]
    next_interactions: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True, slots=True)
class NativeNpcDemonstrations:
    """A BC batch plus its exact source rows in the native trace."""

    batch: DemonstrationBatch
    source_index: npt.NDArray[np.int64]
    tick: npt.NDArray[np.int64]
    trace_uuid: uuid.UUID
    trace: TraceSequence
    native_labels: NativeNpcBehaviorLabels
    attack_rows: int
    attack_activation_rows: int
    attack_execution_rows: int
    ability_start_rows: int
    target_lock_rows: int
    health_delta: float


@dataclass(frozen=True, slots=True)
class NativeNpcDuelDemonstrations:
    """Two synchronized, independently labelled native NPC behaviors."""

    actors: tuple[NativeNpcDemonstrations, NativeNpcDemonstrations]
    npc_uuid: tuple[uuid.UUID, uuid.UUID]
    role: tuple[str, str]

    @property
    def batches(self) -> tuple[DemonstrationBatch, DemonstrationBatch]:
        return tuple(actor.batch for actor in self.actors)

    @property
    def traces(self) -> tuple[TraceSequence, TraceSequence]:
        return tuple(actor.trace for actor in self.actors)

    @property
    def attack_rows(self) -> tuple[int, int]:
        return tuple(actor.attack_rows for actor in self.actors)

    @property
    def health_delta(self) -> tuple[float, float]:
        return tuple(actor.health_delta for actor in self.actors)


def native_npc_demonstrations(
    capture: NativeNpcTraceCapture,
    *,
    observation: Any = None,
    action_mask: Any = None,
    weight: Any = None,
    movement_epsilon: float = 1.0e-6,
    project_attack_execution: bool = False,
    project_basic_attacks: bool = False,
    project_combat_abilities: bool = False,
    config: BehaviorCloningConfig = BehaviorCloningConfig(),
    _trace: TraceSequence | None = None,
) -> NativeNpcDemonstrations:
    """Project uncontaminated native steering into representable Arena heads.

    The default observation is the trace's 14-value privileged state. Callers
    training the 8271-value learner policy must supply aligned policy
    observations and legal masks. Movement, yaw, and pitch are always
    labelled. ``project_attack_execution`` retains the legacy coarse mapping
    to ``base_action``. ``project_basic_attacks`` maps every causally proven
    attack start to choice one of the ability head, regardless of the authored
    swing root; this is the portable locomotion-and-basic-attack foundation.
    ``project_combat_abilities`` instead preserves the exact role-local root.
    Later execution frames remain NONE. Combat projections are mutually
    exclusive and never infer an input from damage or animation effects.
    """

    if not isinstance(capture, NativeNpcTraceCapture):
        raise TypeError("capture must be NativeNpcTraceCapture")
    if tuple(config.head_sizes) != NATIVE_HEAD_SIZES:
        raise ValueError("native NPC projection requires Arena's action-head ABI")
    if not np.isfinite(movement_epsilon) or movement_epsilon < 0.0:
        raise ValueError("movement_epsilon must be finite and nonnegative")
    if sum(
        (project_attack_execution, project_basic_attacks, project_combat_abilities)
    ) > 1:
        raise ValueError("native combat projections select different semantics")

    source_index = np.flatnonzero(capture.native_control).astype(np.int64)
    if source_index.size == 0:
        raise ValueError("trace has no uncontaminated native-control frames")
    observations = np.asarray(capture.state if observation is None else observation)
    if observations.ndim != 2 or observations.shape[0] != capture.emitted_count:
        raise ValueError("observation must have shape [emitted_count, features]")

    count = source_index.size
    actions = np.zeros((count, len(NATIVE_HEAD_SIZES)), dtype=np.int32)
    actions[:, _YAW] = NATIVE_HEAD_SIZES[_YAW] // 2
    actions[:, _PITCH] = NATIVE_HEAD_SIZES[_PITCH] // 2
    supervised = np.zeros_like(actions, dtype=np.bool_)
    control = capture.control[source_index]
    mask = capture.control_mask[source_index]

    body_present = (mask & _BODY_PRESENT) != 0
    translation = (mask & _BODY_TRANSLATION) != 0
    planar = np.hypot(control[:, 0], control[:, 2])
    planar_only = np.abs(control[:, 1]) <= movement_epsilon
    move_rows = body_present & (~translation | planar_only)
    actions[move_rows, _MOVE] = _compass_choice(
        control[move_rows, 0],
        control[move_rows, 2],
        translation[move_rows] & (planar[move_rows] > movement_epsilon),
    )
    supervised[move_rows, _MOVE] = True

    head_present = (mask & _HEAD_PRESENT) != 0
    yaw_requested = (mask & _HEAD_YAW) != 0
    pitch_requested = (mask & _HEAD_PITCH) != 0
    current = capture.state[source_index]
    desired_yaw = np.where(yaw_requested, control[:, 5], current[:, 9])
    desired_pitch = np.where(pitch_requested, control[:, 6], current[:, 10])
    actions[head_present, _YAW] = _signed_choice(
        desired_yaw[head_present],
        current[head_present, 9],
        NATIVE_HEAD_SIZES[_YAW],
        wrap=True,
    )
    actions[head_present, _PITCH] = _signed_choice(
        desired_pitch[head_present],
        current[head_present, 10],
        NATIVE_HEAD_SIZES[_PITCH],
        wrap=False,
    )
    supervised[head_present, _YAW] = True
    supervised[head_present, _PITCH] = True
    if project_attack_execution:
        causes = tuple(capture.attack_execution_cause[index] for index in source_index)
        cause_available = np.asarray(
            [cause.available for cause in causes], dtype=np.bool_
        )
        cause_executed = np.asarray(
            [cause.executed for cause in causes], dtype=np.bool_
        )
        # Only a Primary interaction proves the Arena base-attack head. Other
        # authored roots may belong to abilities or future typed heads; an
        # ActionAttack effect alone is not enough to choose among them.
        base_attack = cause_executed & np.asarray(
            [cause.interaction_type == "Primary" for cause in causes],
            dtype=np.bool_,
        )
        base_known = cause_available & (~cause_executed | base_attack)
        actions[base_known, _BASE] = SKILL_IDLE
        actions[base_attack, _BASE] = SKILL_ATTACK
        supervised[base_known, _BASE] = True
    ability_start_rows = 0
    if project_basic_attacks:
        ability_start_rows = _project_basic_attacks(
            capture,
            source_index,
            actions,
            supervised,
        )
    elif project_combat_abilities:
        ability_start_rows = _project_combat_abilities(
            capture,
            source_index,
            actions,
            supervised,
        )
    if not np.any(supervised):
        raise ValueError("trace has no Arena-representable steering labels")

    batch = demonstration_batch(
        observations[source_index],
        actions,
        action_mask=_selected(action_mask, source_index, capture.emitted_count),
        supervision_mask=supervised,
        weight=_selected(weight, source_index, capture.emitted_count),
        config=config,
    )
    ticks = capture.tick[source_index].copy()
    source_index.flags.writeable = False
    ticks.flags.writeable = False
    labels = NativeNpcBehaviorLabels(
        delta_seconds=_frozen_rows(capture.delta_seconds, source_index),
        control=_frozen_rows(capture.control, source_index),
        control_mask=_frozen_rows(capture.control_mask, source_index),
        attack_action_active=_frozen_rows(capture.attack_action_active, source_index),
        attack_activation=_frozen_rows(capture.attack_activation, source_index),
        attack_execution_cause=_selected_tuple(
            capture.attack_execution_cause, source_index
        ),
        combat_attack=_frozen_rows(capture.combat_attack, source_index),
        next_combat_attack=_frozen_rows(capture.next_combat_attack, source_index),
        attack_pause_seconds=_frozen_rows(capture.attack_pause_seconds, source_index),
        next_attack_pause_seconds=_frozen_rows(
            capture.next_attack_pause_seconds, source_index
        ),
        target_uuid=_selected_tuple(capture.target_uuid, source_index),
        next_target_uuid=_selected_tuple(capture.next_target_uuid, source_index),
        state_name=_selected_tuple(capture.state_name, source_index),
        next_state_name=_selected_tuple(capture.next_state_name, source_index),
        body_instruction=_selected_tuple(capture.body_instruction, source_index),
        head_instruction=_selected_tuple(capture.head_instruction, source_index),
        active_actions=_selected_tuple(capture.active_actions, source_index),
        attack_actions=_selected_tuple(capture.attack_actions, source_index),
        next_attack_actions=_selected_tuple(capture.next_attack_actions, source_index),
        interactions=_selected_tuple(capture.interactions, source_index),
        next_interactions=_selected_tuple(capture.next_interactions, source_index),
    )
    return NativeNpcDemonstrations(
        batch=batch,
        source_index=source_index,
        tick=ticks,
        trace_uuid=capture.trace_uuid,
        trace=_trace or native_trace_sequence(capture),
        native_labels=labels,
        attack_rows=int(np.count_nonzero(labels.combat_attack)),
        attack_activation_rows=int(np.count_nonzero(labels.attack_activation)),
        attack_execution_rows=sum(
            cause.executed for cause in labels.attack_execution_cause
        ),
        ability_start_rows=ability_start_rows,
        target_lock_rows=sum(target is not None for target in labels.target_uuid),
        health_delta=float(
            capture.next_state[source_index[-1], 11]
            - capture.state[source_index[0], 11]
        ),
    )


def _project_basic_attacks(capture, source_index, actions, supervised):
    """Label attack onset separately from the authored swing selection."""

    causes = tuple(capture.attack_execution_cause[index] for index in source_index)
    current_slot = np.asarray(
        [capture.actor_evidence[index]["active_ability_slot"] for index in source_index],
        dtype=np.int32,
    )
    next_slot = np.asarray(
        [
            capture.next_actor_evidence[index]["active_ability_slot"]
            for index in source_index
        ],
        dtype=np.int32,
    )
    activation = np.asarray(capture.attack_activation[source_index], dtype=np.bool_)
    available = np.asarray([row.available for row in causes], dtype=np.bool_)
    executed = np.asarray([row.executed for row in causes], dtype=np.bool_)

    if np.any(activation & (next_slot < 0)):
        raise ValueError("native attack activation lacks an active authored slot")
    instant = executed & (current_slot < 0) & (next_slot < 0)
    start = activation | instant
    known = available | activation
    actions[known, _ABILITY] = 0
    actions[start, _ABILITY] = 1
    supervised[known, _ABILITY] = True
    return int(np.count_nonzero(start))


def _project_combat_abilities(capture, source_index, actions, supervised):
    """Encode one initiating Java attack decision, never its later effects."""

    causes = tuple(capture.attack_execution_cause[index] for index in source_index)
    current_slot = np.asarray(
        [
            capture.actor_evidence[index]["active_ability_slot"]
            for index in source_index
        ],
        dtype=np.int32,
    )
    next_slot = np.asarray(
        [
            capture.next_actor_evidence[index]["active_ability_slot"]
            for index in source_index
        ],
        dtype=np.int32,
    )
    activation = np.asarray(capture.attack_activation[source_index], dtype=np.bool_)
    cause_available = np.asarray([row.available for row in causes], dtype=np.bool_)
    executed = np.asarray([row.executed for row in causes], dtype=np.bool_)
    candidate = np.asarray([row.candidate_index for row in causes], dtype=np.int32)

    if np.any(activation & (next_slot < 0)):
        raise ValueError("native attack activation lacks an active authored slot")
    instant = executed & (current_slot < 0) & (next_slot < 0)
    both = activation & executed
    if np.any(both & (next_slot != candidate)):
        raise ValueError("native attack causes disagree on the initiating slot")
    start = activation | instant
    slot = np.where(activation, next_slot, candidate)
    capacity = NATIVE_HEAD_SIZES[_ABILITY] - 1
    if np.any(start & ((slot < 0) | (slot >= capacity))):
        raise ValueError("native authored attack slot exceeds the Arena ability head")

    # A complete cause row proves that no other authored root started. An
    # activation edge remains usable even when unrelated cause joining is
    # unavailable. Executions of an already-active slot are deliberately NONE.
    known = cause_available | activation
    actions[known, _ABILITY] = 0
    actions[start, _ABILITY] = slot[start] + 1
    supervised[known, _ABILITY] = True
    return int(np.count_nonzero(start))


def native_npc_duel_demonstrations(
    captures: Sequence[NativeNpcTraceCapture],
    *,
    observations: Sequence[Any] | None = None,
    action_masks: Sequence[Any] | None = None,
    weights: Sequence[Any] | None = None,
    movement_epsilon: float = 1.0e-6,
    config: BehaviorCloningConfig = BehaviorCloningConfig(),
) -> NativeNpcDuelDemonstrations:
    """Project both sides of one simultaneous Java NPC duel."""

    pair = _pair(captures, "captures")
    if not all(isinstance(item, NativeNpcTraceCapture) for item in pair):
        raise TypeError("captures must contain NativeNpcTraceCapture values")
    if pair[0].npc_uuid == pair[1].npc_uuid:
        raise ValueError("duel traces must identify different NPCs")

    def identity(item):
        return (
            item.bridge_sha256,
            item.server_version,
            item.world,
            item.worldgen_provider,
            item.worldgen_version,
            item.seed,
        )

    if identity(pair[0]) != identity(pair[1]):
        raise ValueError("duel traces must come from the same native world reset")

    inputs = (
        _pair(observations, "observations", default=None),
        _pair(action_masks, "action_masks", default=None),
        _pair(weights, "weights", default=None),
    )
    traces = native_duel_trace_sequences(pair)
    actors = tuple(
        native_npc_demonstrations(
            capture,
            observation=inputs[0][index],
            action_mask=inputs[1][index],
            weight=inputs[2][index],
            movement_epsilon=movement_epsilon,
            config=config,
            _trace=traces[index],
        )
        for index, capture in enumerate(pair)
    )
    if not np.array_equal(actors[0].tick, actors[1].tick):
        raise ValueError("duel traces must contain the same native-control ticks")
    return NativeNpcDuelDemonstrations(
        actors=actors,
        npc_uuid=(pair[0].npc_uuid, pair[1].npc_uuid),
        role=(pair[0].role, pair[1].role),
    )


def _pair(value: Sequence[Any] | None, name: str, *, default=...) -> tuple[Any, Any]:
    if value is None and default is not ...:
        return (default, default)
    result = tuple(value) if value is not None else ()
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two actor-aligned values")
    return result


def _selected(value: Any, index: npt.NDArray[np.int64], rows: int):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != rows:
        raise ValueError("aligned trace data must start with emitted_count")
    return array[index]


def _frozen_rows(value, index: npt.NDArray[np.int64]):
    selected = np.asarray(value)[index].copy()
    selected.flags.writeable = False
    return selected


def _selected_tuple(value, index: npt.NDArray[np.int64]) -> tuple:
    return tuple(value[int(row)] for row in index)


def _compass_choice(x, z, moving):
    angle = np.mod(np.arctan2(x, -z), 2.0 * np.pi)
    choice = (np.floor(angle / (np.pi / 4.0) + 0.5).astype(np.int32) % 8) + 1
    return np.where(moving, choice, 0)


def _signed_choice(desired, current, size: int, *, wrap: bool):
    delta = desired - current
    if wrap:
        delta = np.arctan2(np.sin(delta), np.cos(delta))
    degrees = np.asarray(np.rad2deg(delta), dtype=np.float32)
    table = np.asarray(look_delta_values(size), dtype=np.float32)
    return np.argmin(
        np.abs(degrees[..., None] - table),
        axis=-1,
    ).astype(np.int32)


__all__ = [
    "NativeNpcBehaviorLabels",
    "NativeNpcDemonstrations",
    "NativeNpcDuelDemonstrations",
    "native_npc_demonstrations",
    "native_npc_duel_demonstrations",
]
