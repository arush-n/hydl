"""Production same-connection composer for actor-major native policy rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import uuid

import numpy as np

from hytalegym.jax.combat.observation.v3.native.codec.transport import (
    NativeActionTranslation,
    translate_learner_arsenal_action_to_native,
)
from hytalegym.jax.combat.observation.v3.native.evidence import (
    NativeActorEvidenceAssembler,
    NativeActorEvidenceAssembly,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
    RecipeCandidateEncoderParams,
)
from hytalegym.jax.combat.types import ENTITY_COUNT
from hytalegym.jax.crafting import PackedRecipeTable

from ..group import (
    NativeGroupActionMessage,
    NativeGroupStepEvidence,
    native_group_policy_combat_reset_options,
    native_group_step_message,
)
from ..world_action_capture import NativePolicyWorldActionCaptureAdapter
from .projection import (
    NativeActorMajorCapabilities,
    actor_first_order,
    project_actor_first_wire,
    project_actor_info,
)


@dataclass(frozen=True, slots=True)
class NativeActorMajorRow:
    """One composer-owned actor-first row and its exact host context."""

    actor_slot: int
    target_slot: int
    assembly: NativeActorEvidenceAssembly
    info: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NativeActorMajorObservation:
    """Canonical actor-major rows derived from one bridge observation."""

    rows: tuple[NativeActorMajorRow, ...]
    capabilities: NativeActorMajorCapabilities
    group_evidence: NativeGroupStepEvidence | None
    environment_step: int

    @property
    def policy_observation(self) -> np.ndarray:
        """Dense actor-major policy input in canonical slot order."""

        return np.stack(
            [row.assembly.policy_observation for row in self.rows],
            axis=0,
        )

    @property
    def policy_action_mask(self) -> np.ndarray:
        """Actor-major flattened factor masks matching ``policy_observation``."""

        return np.stack(
            [row.assembly.policy_action_mask for row in self.rows],
            axis=0,
        )

    def row(self, actor_slot: int) -> NativeActorMajorRow:
        for row in self.rows:
            if row.actor_slot == actor_slot:
                return row
        raise KeyError(actor_slot)


@dataclass(frozen=True, slots=True)
class NativeActorMajorDecision:
    """Host legality and translated wire row for one policy actor."""

    actor_slot: int
    translation: NativeActionTranslation
    requested_factors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NativeActorMajorStepResult:
    """One atomic group transition with mandatory validated Java evidence."""

    observation: NativeActorMajorObservation
    decisions: tuple[NativeActorMajorDecision, ...]
    group_message: NativeGroupActionMessage
    group_evidence: NativeGroupStepEvidence
    reward: float
    terminated: bool
    truncated: bool


class NativeActorMajorHostComposer:
    """Compose all negotiated actor rows over one borrowed BridgeConnection.

    The composer never opens or closes a socket.  Each actor owns a distinct
    assembler, capture adapter, privileged snapshot, and monotonic request
    sequence; every adapter borrows the exact same ``send_and_recv`` bound
    method.  Policy rows are resolved independently, then sent in one sorted
    group message.  A step is not returned until the bridge's complete group
    receipt and the next actor surfaces have both validated.
    """

    def __init__(
        self,
        connection: object,
        *,
        actor_profiles: Mapping[int, str],
        actor_identities: Mapping[int, str],
        target_slots: Mapping[int, int] | None = None,
        packed_recipes: PackedRecipeTable | None = None,
        recipe_encoder_params: RecipeCandidateEncoderParams | None = None,
        initial_request_ids: Mapping[int, int] | None = None,
        ticks_per_step: int = 4,
    ) -> None:
        request = getattr(connection, "send_and_recv", None)
        if not callable(request):
            raise TypeError("connection must expose callable send_and_recv")
        self._connection = connection
        self._request = request
        self._actor_profiles = _canonical_slot_mapping(
            actor_profiles,
            "actor_profiles",
            value_type=str,
        )
        if tuple(self._actor_profiles) != tuple(range(ENTITY_COUNT)):
            raise ValueError(
                "actor_profiles must negotiate every actor-capacity slot"
            )
        if any(not value for value in self._actor_profiles.values()):
            raise ValueError("actor profiles must be nonempty")
        identities = _canonical_slot_mapping(
            actor_identities,
            "actor_identities",
            value_type=str,
        )
        if tuple(identities) != tuple(self._actor_profiles):
            raise ValueError("actor identities and profiles must own equal slots")
        self._actor_identities = {
            slot: _canonical_uuid(identity)
            for slot, identity in identities.items()
        }
        self._target_slots = self._normalize_target_slots(target_slots)
        request_ids = (
            {slot: 0 for slot in self._actor_profiles}
            if initial_request_ids is None
            else _canonical_slot_mapping(
                initial_request_ids,
                "initial_request_ids",
                value_type=int,
            )
        )
        if tuple(request_ids) != tuple(self._actor_profiles):
            raise ValueError("initial request IDs must cover every actor slot")
        if any(
            isinstance(value, bool) or value < 0 or value > 2**63 - 1
            for value in request_ids.values()
        ):
            raise ValueError("initial request IDs must be nonnegative int64")

        self._assemblers: dict[int, NativeActorEvidenceAssembler] = {}
        self._adapters: dict[int, NativePolicyWorldActionCaptureAdapter] = {}
        for actor_slot, profile in self._actor_profiles.items():
            target_slot = self._target_slots[actor_slot]
            assembler = NativeActorEvidenceAssembler(
                profile,
                target_profile=self._actor_profiles[target_slot],
                ticks_per_step=ticks_per_step,
            )
            adapter = NativePolicyWorldActionCaptureAdapter(
                packed_recipes=packed_recipes,
                recipe_encoder_params=recipe_encoder_params,
                actor_slot=actor_slot,
                expected_actor_identity=self._actor_identities[actor_slot],
                initial_request_id=request_ids[actor_slot],
            )
            adapter.bind_bridge_request(self._request)
            self._assemblers[actor_slot] = assembler
            self._adapters[actor_slot] = adapter

        negotiations = [
            assembler.negotiation_options
            for assembler in self._assemblers.values()
        ]
        if any(value != negotiations[0] for value in negotiations[1:]):
            raise ValueError(
                "actor profiles require incompatible native evidence negotiation"
            )
        self._negotiation_options = dict(negotiations[0])
        self._environment_step = 0
        self._observation: NativeActorMajorObservation | None = None
        self._faulted = False

    @property
    def reset_options(self) -> dict[str, object]:
        """Return one reset payload fragment for the composed actor group."""

        result = dict(self._negotiation_options)
        compatibility = self._assemblers[0].reset_options
        for name, value in compatibility.items():
            existing = result.get(name, value)
            if existing != value:
                raise ValueError(f"reset option {name!r} differs across actors")
            result[name] = value
        nonzero_profiles = {
            slot: profile
            for slot, profile in self._actor_profiles.items()
            if slot != 0
        }
        if nonzero_profiles:
            result.update(
                native_group_policy_combat_reset_options(nonzero_profiles)
            )
        return result

    @property
    def observation(self) -> NativeActorMajorObservation:
        if self._observation is None:
            raise RuntimeError("accept_reset_response must precede actor-major step")
        return self._observation

    def accept_reset_response(
        self,
        response: Mapping[str, Any],
    ) -> NativeActorMajorObservation:
        """Start an episode from the reset response on the borrowed connection."""

        self._environment_step = 0
        self._faulted = False
        for assembler in self._assemblers.values():
            assembler.reset()
        for adapter in self._adapters.values():
            adapter.reset()
        try:
            observation = self._assemble_response(response, receipt=None)
        except Exception:
            self._faulted = True
            raise
        self._observation = observation
        return observation

    def step(
        self,
        factors_by_slot: Mapping[int, Any],
    ) -> NativeActorMajorStepResult:
        """Resolve every actor independently and execute one atomic group step."""

        if self._faulted:
            raise RuntimeError(
                "actor-major composer is faulted; accept a fresh reset response"
            )
        current = self.observation
        factors = _canonical_slot_mapping(
            factors_by_slot,
            "factors_by_slot",
            value_type=None,
        )
        expected_slots = tuple(self._actor_profiles)
        if tuple(factors) != expected_slots:
            raise ValueError("one policy factor row is required per actor slot")

        decisions = []
        translations = []
        for actor_slot in expected_slots:
            row = current.row(actor_slot)
            decision = self._resolve_row(
                actor_slot,
                factors[actor_slot],
                row,
            )
            decisions.append(decision)
            translations.append(decision.translation)
        group_message = native_group_step_message(
            expected_slots,
            translations,
        )

        self._environment_step += 1
        try:
            response = self._request(group_message.message)
            _require_observation(response)
            info = _response_info(response, self._environment_step)
            receipt = NativeGroupStepEvidence.from_info(info)
            if receipt.entity_id != group_message.entity_id:
                raise ValueError(
                    "native group receipt actor IDs differ from the sent group"
                )
            observation = self._assemble_response(response, receipt=receipt)
        except Exception:
            # A failed synchronous request has uncertain engine state.  Never
            # retry stale snapshots or counters inside the same episode.
            self._faulted = True
            raise
        self._observation = observation
        return NativeActorMajorStepResult(
            observation=observation,
            decisions=tuple(decisions),
            group_message=group_message,
            group_evidence=receipt,
            reward=float(response.get("reward", 0.0)),
            terminated=bool(response.get("terminated", False)),
            truncated=bool(response.get("truncated", False)),
        )

    def unbind(self) -> None:
        """Release the borrowed callback without closing the connection."""

        for adapter in self._adapters.values():
            adapter.unbind_bridge_request(self._request)

    def _assemble_response(
        self,
        response: Mapping[str, Any],
        *,
        receipt: NativeGroupStepEvidence | None,
    ) -> NativeActorMajorObservation:
        _require_observation(response)
        info = _response_info(response, self._environment_step)
        capabilities = NativeActorMajorCapabilities.from_info(info)
        if capabilities.actor_identities != tuple(
            self._actor_identities.values()
        ):
            raise ValueError("native policy actor ownership changed")
        capabilities.validate_profiles(self._actor_profiles)
        raw_observation = response.get("obs")
        if not isinstance(raw_observation, Mapping):
            raise TypeError("native group observation payload must be a mapping")
        wire = raw_observation.get("native_actor_evidence")
        if not isinstance(wire, Mapping):
            raise ValueError("native group observation lacks actor evidence")

        rows = []
        for actor_slot in self._actor_profiles:
            adapter = self._adapters[actor_slot]
            # Capture first: unlike the compatibility observation, this exact
            # inventory belongs to the requested actor slot.
            surface = adapter.surface(None, info)
            inventory = adapter.last_surface_inventory
            if inventory is None:
                raise RuntimeError("native actor surface lost its inventory")
            row_info = project_actor_info(
                info,
                actor_slot=actor_slot,
                inventory=inventory,
                capabilities=capabilities,
                receipt=receipt,
            )
            actor_wire = project_actor_first_wire(
                wire,
                actor_slot=actor_slot,
                target_slot=self._target_slots[actor_slot],
                actor_profile=self._actor_profiles[actor_slot],
                capabilities=capabilities,
                receipt=receipt,
            )
            assembler = self._assemblers[actor_slot]
            assembly = assembler.assemble({}, actor_wire, row_info)
            assembly = assembler.bind_policy_action_surface(
                assembly,
                surface,
                row_info,
            )
            rows.append(
                NativeActorMajorRow(
                    actor_slot=actor_slot,
                    target_slot=self._target_slots[actor_slot],
                    assembly=assembly,
                    info=row_info,
                )
            )
        return NativeActorMajorObservation(
            rows=tuple(rows),
            capabilities=capabilities,
            group_evidence=receipt,
            environment_step=self._environment_step,
        )

    def _resolve_row(
        self,
        actor_slot: int,
        factors: Any,
        row: NativeActorMajorRow,
    ) -> NativeActorMajorDecision:
        values = np.asarray(factors)
        expected_shape = (len(ARSENAL_POLICY_ACTION_HEAD_SIZES),)
        if values.shape != expected_shape:
            raise ValueError(
                f"actor {actor_slot} policy factors must have shape "
                f"{expected_shape}"
            )
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("actor-major policy factors must be integers")
        values_i32 = values.astype(np.int32, copy=False)
        assembler = self._assemblers[actor_slot]
        action, decoded = assembler.decode_policy_action(
            values_i32,
            row.assembly,
        )
        selected_legal = []
        offset = 0
        for value, size in zip(
            values_i32,
            ARSENAL_POLICY_ACTION_HEAD_SIZES,
            strict=True,
        ):
            if not 0 <= int(value) < int(size):
                raise ValueError("actor-major policy factor is out of range")
            selected_legal.append(
                bool(row.assembly.policy_action_mask[offset + int(value)])
            )
            offset += int(size)
        reject_reasons = tuple(
            f"policy_head_masked:{index}"
            for index, legal in enumerate(selected_legal)
            if not legal
        )
        policy_legal = bool(np.asarray(decoded.valid)[0]) and not reject_reasons
        if not policy_legal and not reject_reasons:
            reject_reasons = ("policy_decode_rejected",)

        native_world_verb_request = None
        if policy_legal:
            snapshot = row.assembly.action_surface_snapshot
            if snapshot is None:
                policy_legal = False
                reject_reasons = ("native_world_action_snapshot_missing",)
            else:
                resolution = self._adapters[actor_slot].resolve(
                    action,
                    snapshot,
                    row.info,
                )
                native_world_verb_request = resolution.request
                if not resolution.legal:
                    policy_legal = False
                    reject_reasons = tuple(
                        dict.fromkeys(
                            reject_reasons + resolution.reject_reasons
                        )
                    )

        native_ability_slot = assembler.native_ability_slot(action)
        translation = translate_learner_arsenal_action_to_native(
            action,
            decoded.low_level_action,
            row.info,
            requested_charge_time_seconds=(
                assembler.requested_charge_time_seconds(action)
            ),
            native_ability_slot=native_ability_slot,
            native_world_verb_request=native_world_verb_request,
            policy_legal=policy_legal,
            policy_reject_reasons=reject_reasons,
        )
        return NativeActorMajorDecision(
            actor_slot=actor_slot,
            translation=translation,
            requested_factors=tuple(int(value) for value in values_i32),
        )

    def _normalize_target_slots(
        self,
        value: Mapping[int, int] | None,
    ) -> dict[int, int]:
        if ENTITY_COUNT < 2:
            raise ValueError("actor-major composition requires target capacity")
        if value is None:
            result = {
                actor_slot: next(
                    slot
                    for slot in range(ENTITY_COUNT)
                    if slot != actor_slot
                )
                for actor_slot in self._actor_profiles
            }
        else:
            result = _canonical_slot_mapping(
                value,
                "target_slots",
                value_type=int,
            )
        if tuple(result) != tuple(self._actor_profiles):
            raise ValueError("target slots must cover every policy actor")
        for actor_slot, target_slot in result.items():
            actor_first_order(actor_slot, target_slot)
        return result


def _canonical_slot_mapping(
    value: Mapping[int, Any],
    name: str,
    *,
    value_type: type | None,
) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = {}
    for slot, item in value.items():
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError(f"{name} actor slots must be integers")
        if slot < 0 or slot >= ENTITY_COUNT:
            raise ValueError(f"{name} actor slot is outside capacity")
        if value_type is not None and (
            isinstance(item, bool) and value_type is int
            or not isinstance(item, value_type)
        ):
            raise TypeError(f"{name} has an invalid value")
        result[slot] = item
    return dict(sorted(result.items()))


def native_actor_major_reset_options(
    actor_profiles: Mapping[int, str],
    *,
    ticks_per_step: int = 4,
) -> dict[str, object]:
    """Build the reset fragment before actor UUIDs have been negotiated."""

    profiles = _canonical_slot_mapping(
        actor_profiles,
        "actor_profiles",
        value_type=str,
    )
    if tuple(profiles) != tuple(range(ENTITY_COUNT)) or any(
        not profile for profile in profiles.values()
    ):
        raise ValueError(
            "actor profiles must name every actor-capacity slot"
        )
    assemblers = {
        slot: NativeActorEvidenceAssembler(
            profile,
            target_profile=profiles[
                next(index for index in range(ENTITY_COUNT) if index != slot)
            ],
            ticks_per_step=ticks_per_step,
        )
        for slot, profile in profiles.items()
    }
    negotiations = [
        assembler.negotiation_options for assembler in assemblers.values()
    ]
    if any(value != negotiations[0] for value in negotiations[1:]):
        raise ValueError(
            "actor profiles require incompatible native evidence negotiation"
        )
    result = dict(negotiations[0])
    for name, value in assemblers[0].reset_options.items():
        if name in result and result[name] != value:
            raise ValueError(f"reset option {name!r} differs across actors")
        result[name] = value
    nonzero = {slot: profile for slot, profile in profiles.items() if slot != 0}
    if nonzero:
        result.update(native_group_policy_combat_reset_options(nonzero))
    return result


def _canonical_uuid(value: str) -> str:
    try:
        canonical = str(uuid.UUID(value))
    except (AttributeError, ValueError) as error:
        raise ValueError("actor identity must be a canonical UUID") from error
    if value != canonical:
        raise ValueError("actor identity must be canonical UUID text")
    return value


def _response_info(
    response: Mapping[str, Any],
    environment_step: int,
) -> dict[str, Any]:
    raw = response.get("info", {})
    if not isinstance(raw, Mapping):
        raise TypeError("native observation info must be a mapping")
    result = dict(raw)
    result["step_count"] = environment_step
    return result


def _require_observation(response: object) -> None:
    if not isinstance(response, Mapping):
        raise TypeError("bridge response must be a mapping")
    if response.get("type") != "observation":
        raise ConnectionError(
            f"expected observation response, got {response.get('type')!r}"
        )


__all__ = [
    "NativeActorMajorDecision",
    "NativeActorMajorHostComposer",
    "NativeActorMajorObservation",
    "NativeActorMajorRow",
    "NativeActorMajorStepResult",
    "native_actor_major_reset_options",
]
