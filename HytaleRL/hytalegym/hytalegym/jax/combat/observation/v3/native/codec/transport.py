"""Host-side translation from the public Arsenal action to the bridge wire.

The bridge advertises each verb independently.  Translation is atomic: if a
requested factor is unavailable or still needs a selected-target resolver, the
whole decision becomes a safe no-op instead of partially executing a different
action on native Hytale.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from hytalegym.jax.combat.arsenal.schema.contract import ABILITY_CAPACITY
from hytalegym.jax.combat.observation.v3.schema.contract import (
    MOVEMENT_STATE_FEATURES,
    MOVEMENT_STATE_SIZE,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_BLOCK_TRIGGER_PRIMARY,
    ARSENAL_BLOCK_TRIGGER_SECONDARY,
)
from hytalegym.jax.combat.observation.v3.schema.types import LearnerArsenalAction
from hytalegym.jax.combat.types import (
    ACTION_ATTACK,
    ACTION_BACK,
    ACTION_FORWARD,
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_PITCH_DELTA,
    ACTION_RIGHT,
    ACTION_BODY_YAW_DELTA,
    ACTION_HOTBAR_SLOT,
    ACTION_YAW_DELTA,
    ACTION_SIZE,
)
from hytalegym.worldgen import (
    NativeWorldVerbRequest,
    native_world_verb_lifecycle_contract,
    native_world_verb_lifecycle_contract_sha256,
)

NATIVE_POLICY_ACTION_PROTOCOL_VERSION = 1
NATIVE_MOVEMENT_STATE_SCHEMA = "hytalerl_native_movement_states_v1"
NATIVE_MOVEMENT_STATE_VERSION = 1
_WORLD_VERB_LIFECYCLE_CONTRACT = native_world_verb_lifecycle_contract()
NATIVE_WORLD_VERB_LIFECYCLE_SCHEMA = str(_WORLD_VERB_LIFECYCLE_CONTRACT["schema"])
NATIVE_WORLD_VERB_LIFECYCLE_VERSION = int(_WORLD_VERB_LIFECYCLE_CONTRACT["version"])
NATIVE_WORLD_VERB_LIFECYCLE_CONTRACT_SHA256 = (
    native_world_verb_lifecycle_contract_sha256().upper()
)
_NATIVE_ACTION_LIFECYCLE_FIELDS = {
    "guard": (
        "native_guard_requested",
        "native_guard_accepted",
        "native_guard_started",
        "native_guard_active",
        "native_guard_finished",
    ),
    "ability": (
        "native_ability_requested",
        "native_ability_accepted",
        "native_ability_started",
        "native_ability_finished",
        "native_ability_failed",
    ),
    "attack": (
        "native_attack_requested",
        "native_attack_accepted",
        "native_attack_executing",
    ),
    "dodge": (
        "native_dodge_requested",
        "native_dodge_accepted",
        "native_dodge_started",
        "native_dodge_finished",
        "native_dodge_failed",
    ),
    "use": (
        "native_use_requested",
        "native_use_accepted",
        "native_use_started",
        "native_use_active",
        "native_use_finished",
        "native_use_failed",
    ),
    "place_block": (
        "native_place_block_requested",
        "native_place_block_accepted",
    ),
    "break_block": (
        "native_break_block_requested",
        "native_break_block_accepted",
    ),
    "craft_recipe": (
        "native_craft_recipe_requested",
        "native_craft_recipe_accepted",
    ),
}
_NATIVE_ACTION_REJECT_FIELDS = {
    "guard": "native_guard_reject_reason",
    "ability": "native_ability_reject_reason",
    "dodge": "native_dodge_reject_reason",
    "use": "native_use_reject_reason",
    "place_block": "native_place_block_reject_reason",
    "break_block": "native_break_block_reject_reason",
    "craft_recipe": "native_craft_recipe_reject_reason",
}


@dataclass(frozen=True)
class NativePolicyCapabilities:
    protocol_version: int
    supported_actions: frozenset[str]

    @classmethod
    def from_info(cls, info: Mapping[str, Any]) -> NativePolicyCapabilities:
        version = info.get("native_policy_combat_protocol_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("native policy-combat protocol version is unavailable")
        actions = info.get("supported_actions")
        if not isinstance(actions, str):
            raise ValueError("native supported-actions evidence is unavailable")
        return cls(
            protocol_version=version,
            supported_actions=frozenset(
                value.strip() for value in actions.split(",") if value.strip()
            ),
        )

    def supports(self, action: str) -> bool:
        return action in self.supported_actions


@dataclass(frozen=True)
class NativeActionTranslation:
    action: dict[str, object]
    legal: bool
    reject_reasons: tuple[str, ...]


@dataclass(frozen=True)
class NativeMovementStateEvidence:
    """One actor-visible movement-state row ready for the v3 evidence record."""

    movement_state_f32: np.ndarray
    movement_state_mask: np.ndarray


def summarize_native_action_accounting(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce all three native action-accounting layers over a rollout.

    Every lifecycle key is required on every row. Missing telemetry is not
    equivalent to a false event: rejecting it here prevents a bridge version
    mismatch from producing a reassuring but incomplete zero histogram.
    """

    materialized = tuple(rows)
    host_rejects: Counter[str] = Counter()
    bridge_rejects: Counter[str] = Counter()
    lifecycle = {
        verb: {field: 0 for field in fields}
        for verb, fields in _NATIVE_ACTION_LIFECYCLE_FIELDS.items()
    }
    native_rejects = {verb: Counter() for verb in _NATIVE_ACTION_REJECT_FIELDS}

    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise TypeError(f"native action-accounting row {index} is not a mapping")
        _validate_native_world_verb_lifecycle(row, index=index)
        host_rejects.update(
            _native_reason_values(
                row.get("host_policy_action_reject_reasons", ()),
                "host_policy_action_reject_reasons",
            )
        )
        bridge_rejects.update(
            _native_reason_values(
                row.get("requested_unsupported_actions", ""),
                "requested_unsupported_actions",
            )
        )
        for verb, fields in _NATIVE_ACTION_LIFECYCLE_FIELDS.items():
            for field in fields:
                value = row.get(field)
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError(
                        f"native action-accounting row {index} has no boolean {field!r}"
                    )
                lifecycle[verb][field] += int(value)
        for verb, field in _NATIVE_ACTION_REJECT_FIELDS.items():
            native_rejects[verb].update(_native_reason_values(row.get(field), field))

    return {
        "schema": "hytalerl_native_action_accounting_v2",
        "steps": len(materialized),
        "host_reject_reasons": dict(sorted(host_rejects.items())),
        "bridge_unsupported_actions": dict(sorted(bridge_rejects.items())),
        "native_lifecycle": {
            verb: {
                "events": dict(events),
                "reject_reasons": dict(sorted(native_rejects.get(verb, {}).items())),
            }
            for verb, events in lifecycle.items()
        },
    }


def _validate_native_world_verb_lifecycle(
    row: Mapping[str, Any],
    *,
    index: int,
) -> None:
    expected = (
        (
            "native_world_verb_lifecycle_schema",
            NATIVE_WORLD_VERB_LIFECYCLE_SCHEMA,
        ),
        (
            "native_world_verb_lifecycle_version",
            NATIVE_WORLD_VERB_LIFECYCLE_VERSION,
        ),
        (
            "native_world_verb_lifecycle_contract_sha256",
            NATIVE_WORLD_VERB_LIFECYCLE_CONTRACT_SHA256,
        ),
    )
    for field, value in expected:
        if row.get(field) != value:
            raise ValueError(
                f"native action-accounting row {index} has incompatible "
                f"{field!r}: {row.get(field)!r} != {value!r}"
            )


def decode_native_movement_state_evidence(
    info: Mapping[str, Any],
    *,
    subject: str = "agent",
) -> NativeMovementStateEvidence:
    """Decode the bridge bitset without inventing unavailable native state."""

    expected_order = ",".join(MOVEMENT_STATE_FEATURES)
    metadata = (
        ("movement_states_schema", NATIVE_MOVEMENT_STATE_SCHEMA),
        ("movement_states_version", NATIVE_MOVEMENT_STATE_VERSION),
        ("movement_states_count", MOVEMENT_STATE_SIZE),
        ("movement_states_order", expected_order),
    )
    for field, expected in metadata:
        actual = info.get(field)
        if isinstance(expected, int) and isinstance(actual, bool):
            actual = None
        if actual != expected:
            raise ValueError(
                f"native movement-state {field} mismatch: {actual!r} != {expected!r}"
            )

    available = info.get(f"{subject}_movement_states_available")
    bits = info.get(f"{subject}_movement_states_bits")
    if not isinstance(available, bool):
        raise ValueError(f"native {subject} movement-state availability is unavailable")
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise ValueError(f"native {subject} movement-state bits are unavailable")
    if bits < 0 or bits >= (1 << MOVEMENT_STATE_SIZE):
        raise ValueError(f"native {subject} movement-state bits are out of range")
    if not available and bits != 0:
        raise ValueError(
            f"native {subject} unavailable movement-state evidence has nonzero bits"
        )

    indices = np.arange(MOVEMENT_STATE_SIZE, dtype=np.uint32)
    values = ((np.uint32(bits) >> indices) & np.uint32(1)).astype(np.float32)
    mask = np.full((MOVEMENT_STATE_SIZE,), available, dtype=np.bool_)
    values = np.where(mask, values, np.float32(0.0)).astype(np.float32)
    return NativeMovementStateEvidence(
        movement_state_f32=values[None, :],
        movement_state_mask=mask[None, :],
    )


def translate_learner_arsenal_action_to_native(
    action: LearnerArsenalAction,
    low_level_action,
    backend_info: Mapping[str, Any],
    *,
    lane: int = 0,
    requested_charge_time_seconds: float = 0.0,
    native_ability_slot: int | None = None,
    native_world_verb_request: NativeWorldVerbRequest | None = None,
    policy_legal: bool = True,
    policy_reject_reasons: tuple[str, ...] = (),
) -> NativeActionTranslation:
    """Translate one already mask-checked policy decision to ``AgentAction``.

    Block and recipe factors remain privileged: their actor slot must first be
    rechecked and converted to a concrete ``NativeWorldVerbRequest``.  This
    function accepts only a request whose verb exactly matches the selected
    learner factors, and merges it into the same atomic bridge action.  An
    unresolved or mismatched selection fails the whole decision closed.
    """

    capabilities = NativePolicyCapabilities.from_info(backend_info)
    if capabilities.protocol_version != NATIVE_POLICY_ACTION_PROTOCOL_VERSION:
        raise ValueError(
            "native policy-combat protocol mismatch: "
            f"{capabilities.protocol_version} != "
            f"{NATIVE_POLICY_ACTION_PROTOCOL_VERSION}"
        )
    low_level = np.asarray(low_level_action, dtype=np.float32)
    if low_level.ndim != 2 or low_level.shape[1] != ACTION_SIZE:
        raise ValueError(f"low_level_action must have shape (B, {ACTION_SIZE})")
    if lane < 0 or lane >= low_level.shape[0]:
        raise ValueError("lane is out of range")
    charge = float(requested_charge_time_seconds)
    if not np.isfinite(charge) or charge < 0.0:
        raise ValueError("requested_charge_time_seconds must be finite and nonnegative")
    if not isinstance(policy_legal, (bool, np.bool_)):
        raise TypeError("policy_legal must be boolean")
    if not isinstance(policy_reject_reasons, tuple) or not all(
        isinstance(reason, str) and reason for reason in policy_reject_reasons
    ):
        raise TypeError("policy_reject_reasons must be a tuple of nonempty strings")

    semantic = {
        name: _lane_scalar(getattr(action, name), lane, name)
        for name in LearnerArsenalAction._fields
    }
    row = low_level[lane]
    authored_ability_slot = int(semantic["ability_slot"])
    if native_ability_slot is None:
        transport_ability_slot = authored_ability_slot
    else:
        if isinstance(native_ability_slot, (bool, np.bool_)) or not isinstance(
            native_ability_slot,
            (int, np.integer),
        ):
            raise TypeError("native_ability_slot must be an integer or None")
        transport_ability_slot = int(native_ability_slot)
        if not -1 <= transport_ability_slot < ABILITY_CAPACITY:
            raise ValueError("native_ability_slot is out of range")
    requested = {
        "world_move_direction": int(semantic["world_move_direction"]) != 0,
        "guard_held": bool(semantic["guard_held"]),
        "dodge_direction": int(semantic["dodge_direction"]) != 0,
        "ability_slot": authored_ability_slot >= 0,
        "use": bool(semantic["use_requested"]),
    }
    reject_reasons = list(policy_reject_reasons)
    if not policy_legal and not reject_reasons:
        reject_reasons.append("policy_action_illegal")
    if requested["ability_slot"] and transport_ability_slot < 0:
        reject_reasons.append("native_ability_slot_unbound")
    reject_reasons.extend(
        f"native_action_unavailable:{name}"
        for name, active in requested.items()
        if active and not capabilities.supports(name)
    )
    block_trigger = int(semantic["block_interaction_trigger"])
    block_candidate = int(semantic["block_candidate_index"])
    recipe_candidate = int(semantic["recipe_candidate_index"])
    block_requested = block_candidate >= 0
    recipe_requested = recipe_candidate >= 0
    use_requested = requested["use"]
    block_verb_requested = block_requested and not use_requested
    typed_selections = (
        int(use_requested) + int(block_verb_requested) + int(recipe_requested)
    )
    expected_world_verb: str | None = None
    if typed_selections > 1:
        reject_reasons.append("native_world_verb_selection_conflict")
    elif use_requested and native_world_verb_request is not None:
        # Use owns the shared block target. A selected target is not a second
        # verb and the inactive Primary/Secondary factor must not double-run.
        expected_world_verb = "use"
    elif block_verb_requested:
        if block_trigger == ARSENAL_BLOCK_TRIGGER_PRIMARY:
            expected_world_verb = "break_block"
        elif block_trigger == ARSENAL_BLOCK_TRIGGER_SECONDARY:
            expected_world_verb = "place_block"
        else:
            reject_reasons.append("native_block_trigger_invalid")
    elif recipe_requested:
        expected_world_verb = "craft_recipe"

    main_interaction_count = (
        int(row[ACTION_ATTACK] > 0.0)
        + int(requested["ability_slot"])
        + int(requested["guard_held"])
        + int(use_requested or block_verb_requested or recipe_requested)
    )
    if main_interaction_count > 1:
        reject_reasons.append("native_standard_input_selection_conflict")

    if native_world_verb_request is not None and not isinstance(
        native_world_verb_request,
        NativeWorldVerbRequest,
    ):
        raise TypeError(
            "native_world_verb_request must be NativeWorldVerbRequest or None"
        )
    if expected_world_verb is not None:
        if native_world_verb_request is None:
            if block_verb_requested:
                reject_reasons.extend(
                    (
                        "native_block_trigger_unresolved",
                        "native_block_candidate_unresolved",
                    )
                )
            elif recipe_requested:
                reject_reasons.append("native_recipe_candidate_unresolved")
        elif not native_world_verb_request.policy_candidate_evidence_bound:
            reject_reasons.append(
                "native_policy_world_action_evidence_required"
            )
        elif native_world_verb_request.verb != expected_world_verb:
            reject_reasons.append(
                "native_world_verb_request_mismatch:"
                f"{expected_world_verb}:{native_world_verb_request.verb}"
            )
        elif not capabilities.supports(expected_world_verb):
            reject_reasons.append(
                f"native_action_unavailable:{expected_world_verb}"
            )
    elif native_world_verb_request is not None:
        reject_reasons.append("native_world_verb_request_unselected")
    if reject_reasons:
        return NativeActionTranslation(
            action=_native_noop(),
            legal=False,
            reject_reasons=tuple(dict.fromkeys(reject_reasons)),
        )

    # BODY-RELATIVE despite the name. Choice one is the body's own forward,
    # three its right, in 45-degree steps -- the same eight directions the
    # forward/back/left/right booleans below express, which is why the two are
    # alternatives rather than a fallback to a weaker channel.
    #
    # The key is still spelled `world_move_direction` across the semantic dict,
    # this transport and Java's `AgentAction`; renaming that chain is a separate
    # change. `NativeEnvironmentSession` rotates the bin by `desiredBodyYaw`,
    # matching `policy/actions.py`. It used to be an absolute compass on BOTH
    # sides, so the integer's meaning changed consistently -- but a deployed
    # jar older than that change will steer this bin as a WORLD bearing.
    world_direction = int(semantic["world_move_direction"])
    native = _native_noop()
    if world_direction == 0:
        native.update(
            {
                "forward": int(row[ACTION_FORWARD] > 0.0),
                "back": int(row[ACTION_BACK] > 0.0),
                "left": int(row[ACTION_LEFT] > 0.0),
                "right": int(row[ACTION_RIGHT] > 0.0),
            }
        )
    native.update(
        {
            "jump": int(row[ACTION_JUMP] > 0.0),
            "attack": int(row[ACTION_ATTACK] > 0.0),
            "guard_held": int(bool(semantic["guard_held"])),
            "dodge_direction": int(semantic["dodge_direction"]),
            "ability_slot": (
                transport_ability_slot
                if requested["ability_slot"]
                else -1
            ),
            "world_move_direction": world_direction,
            "use": int(
                bool(semantic["use_requested"])
                and native_world_verb_request is None
            ),
            "camera_delta_yaw": float(row[ACTION_YAW_DELTA]),
            # The body is steered independently of the camera. Without this the
            # server sets body := head (`NativeEnvironmentSession` 3957/3964) and
            # a policy that strafes while tracking would have its chest snapped
            # round to its aim, which is not what it was trained on.
            "body_delta_yaw": float(row[ACTION_BODY_YAW_DELTA]),
            "camera_delta_pitch": float(row[ACTION_PITCH_DELTA]),
            # Carried as slot + 1 on the wire's source channel; -1 here is the
            # bridge's own "no switch" sentinel, matching `AgentAction.hotbarSlot`
            # which is applied only when it lands in [0, 9).
            "hotbar_slot": int(row[ACTION_HOTBAR_SLOT]) - 1,
            "requested_charge_time": charge,
        }
    )
    if native_world_verb_request is not None:
        native.update(native_world_verb_request.to_action_fields())
    return NativeActionTranslation(
        action=native,
        legal=True,
        reject_reasons=(),
    )


def _lane_scalar(value, lane: int, name: str):
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.ndim != 1 or lane >= array.shape[0]:
        raise ValueError(f"{name} must be scalar or have shape (B,)")
    return array[lane].item()


def _native_reason_values(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        raise ValueError(f"native action-accounting field {field!r} is unavailable")
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (tuple, list)):
        if not all(isinstance(part, str) and part for part in value):
            raise ValueError(
                f"native action-accounting field {field!r} has invalid reasons"
            )
        return tuple(value)
    raise ValueError(
        f"native action-accounting field {field!r} has invalid reason encoding"
    )


def _native_noop() -> dict[str, object]:
    return {
        "forward": 0,
        "back": 0,
        "left": 0,
        "right": 0,
        "jump": 0,
        "attack": 0,
        "use": 0,
        "guard_held": 0,
        "dodge_direction": 0,
        "ability_slot": -1,
        "world_move_direction": 0,
        "camera_delta_yaw": 0.0,
        "body_delta_yaw": 0.0,
        "camera_delta_pitch": 0.0,
        "hotbar_slot": -1,
        "requested_charge_time": 0.0,
    }


__all__ = [
    "NATIVE_MOVEMENT_STATE_SCHEMA",
    "NATIVE_MOVEMENT_STATE_VERSION",
    "NATIVE_POLICY_ACTION_PROTOCOL_VERSION",
    "NATIVE_WORLD_VERB_LIFECYCLE_CONTRACT_SHA256",
    "NATIVE_WORLD_VERB_LIFECYCLE_SCHEMA",
    "NATIVE_WORLD_VERB_LIFECYCLE_VERSION",
    "NativeActionTranslation",
    "NativeMovementStateEvidence",
    "NativePolicyCapabilities",
    "decode_native_movement_state_evidence",
    "summarize_native_action_accounting",
    "translate_learner_arsenal_action_to_native",
]
