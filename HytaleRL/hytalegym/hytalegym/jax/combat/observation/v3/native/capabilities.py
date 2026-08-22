"""Project same-boundary native trace evidence into learner-v3 action masks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np
import numpy.typing as npt

from hytalegym.jax.combat.observation.v3.native.evidence import (
    NativeActorEvidenceAssembler,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_SIZE,
    arsenal_policy_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.schema.spec import (
    learner_observation_v3_contract_sha256,
)
from hytalegym.geometry import parse_geometry
from hytalegym.worldgen.native_npc_traces import (
    NATIVE_NPC_ACTOR_EVIDENCE_CONTRACT_SHA256,
    NATIVE_NPC_OBSERVATION_CONTRACT_SHA256,
    NativeNpcObservation,
    NativeNpcTraceCapture,
)


NATIVE_NPC_ACTION_CAPABILITY_SCHEMA = (
    "hytalerl_native_npc_learner_v3_action_capability_v1"
)
NATIVE_NPC_ACTION_CAPABILITY_VERSION = 1
_CONTRACT = {
    "schema": NATIVE_NPC_ACTION_CAPABILITY_SCHEMA,
    "version": NATIVE_NPC_ACTION_CAPABILITY_VERSION,
    "source_observation_sha256": NATIVE_NPC_OBSERVATION_CONTRACT_SHA256,
    "source_actor_evidence_sha256": NATIVE_NPC_ACTOR_EVIDENCE_CONTRACT_SHA256,
    "learner_observation_sha256": learner_observation_v3_contract_sha256(),
    "policy_sha256": arsenal_policy_contract_sha256(),
    "output": f"bool[{ARSENAL_POLICY_ACTION_SIZE}]",
    "boundary": "same_as_native_semantic_observation",
    "temporal_state": "reset_for_each_boundary",
    "native_authority": "host_legal_rule_cooldown_and_charge_admission",
    "shared_projection": "current_health_resource_status_loadout_geometry_and_factors",
    "unavailable": "all_false_with_explicit_validity_false",
}
NATIVE_NPC_ACTION_CAPABILITY_SHA256 = hashlib.sha256(
    json.dumps(_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class NativeNpcActionCapabilityBoundary:
    """One side of a transition's factor-mask projection."""

    action_mask: npt.NDArray[np.bool_]
    valid: npt.NDArray[np.bool_]
    unavailable_reason: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NativeNpcActionCapabilityProjection:
    """Current/next factor masks with honest per-boundary availability."""

    action_mask: npt.NDArray[np.bool_]
    next_action_mask: npt.NDArray[np.bool_]
    valid: npt.NDArray[np.bool_]
    next_valid: npt.NDArray[np.bool_]
    unavailable_reason: tuple[str, ...]
    next_unavailable_reason: tuple[str, ...]
    agent_profile: str
    target_profile: str

    def __post_init__(self) -> None:
        rows = len(self.unavailable_reason)
        masks = {
            "action_mask": (self.action_mask, (rows, ARSENAL_POLICY_ACTION_SIZE)),
            "next_action_mask": (
                self.next_action_mask,
                (rows, ARSENAL_POLICY_ACTION_SIZE),
            ),
            "valid": (self.valid, (rows,)),
            "next_valid": (self.next_valid, (rows,)),
        }
        for name, (value, shape) in masks.items():
            frozen = np.asarray(value, dtype=np.bool_).copy()
            if frozen.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)
        if len(self.next_unavailable_reason) != rows:
            raise ValueError("next unavailable reasons must match current rows")


def project_native_npc_action_capabilities(
    capture: NativeNpcTraceCapture,
    *,
    agent_profile: str,
    target_profile: str = "",
) -> NativeNpcActionCapabilityProjection:
    """Build exact learner-v3 factor masks without assuming pre-trace history."""

    if not isinstance(capture, NativeNpcTraceCapture):
        raise TypeError("capture must be NativeNpcTraceCapture")
    current = project_native_npc_action_capability_boundary(
        capture.observation,
        capture.combat_attack,
        agent_profile=agent_profile,
        target_profile=target_profile,
    )
    following = project_native_npc_action_capability_boundary(
        capture.next_observation,
        capture.next_combat_attack,
        agent_profile=agent_profile,
        target_profile=target_profile,
    )
    return NativeNpcActionCapabilityProjection(
        action_mask=current.action_mask,
        next_action_mask=following.action_mask,
        valid=current.valid,
        next_valid=following.valid,
        unavailable_reason=current.unavailable_reason,
        next_unavailable_reason=following.unavailable_reason,
        agent_profile=agent_profile,
        target_profile=target_profile,
    )


def project_native_npc_action_capability_boundary(
    observations: Sequence[NativeNpcObservation],
    combat_attack: npt.NDArray[np.bool_],
    *,
    agent_profile: str,
    target_profile: str = "",
) -> NativeNpcActionCapabilityBoundary:
    """Project one current-or-next observation sequence independently."""

    assembler = NativeActorEvidenceAssembler(
        agent_profile,
        target_profile=target_profile,
        ticks_per_step=1,
    )
    rows = len(observations)
    if np.asarray(combat_attack).shape != (rows,):
        raise ValueError("combat-attack rows must align with observations")
    masks = np.zeros((rows, ARSENAL_POLICY_ACTION_SIZE), dtype=np.bool_)
    valid = np.zeros(rows, dtype=np.bool_)
    reasons: list[str] = []
    for index, observation in enumerate(observations):
        wire = observation.action_capability_evidence
        if not bool(wire.get("available", False)):
            reasons.append(str(wire.get("unavailable_reason") or "unavailable"))
            continue
        assembler.reset()
        try:
            geometry = observation.geometry
            if "cell_mask" not in geometry:
                geometry = parse_geometry(dict(geometry))
            assembly = assembler.assemble(
                {"geometry": geometry},
                wire,
                {
                    "native_inventory": observation.inventory,
                    "combat_agent_attack_executing": bool(combat_attack[index]),
                },
            )
        except ValueError as error:
            reasons.append(f"projection_failed:{error}")
            continue
        row_valid = bool(np.asarray(assembly.observation.valid)[0])
        if not row_valid:
            reasons.append("learner_observation_invalid")
            continue
        masks[index] = assembly.policy_action_mask
        valid[index] = True
        reasons.append("")
    masks.flags.writeable = False
    valid.flags.writeable = False
    return NativeNpcActionCapabilityBoundary(masks, valid, tuple(reasons))


__all__ = [
    "NATIVE_NPC_ACTION_CAPABILITY_SCHEMA",
    "NATIVE_NPC_ACTION_CAPABILITY_SHA256",
    "NATIVE_NPC_ACTION_CAPABILITY_VERSION",
    "NativeNpcActionCapabilityBoundary",
    "NativeNpcActionCapabilityProjection",
    "project_native_npc_action_capability_boundary",
    "project_native_npc_action_capabilities",
]
