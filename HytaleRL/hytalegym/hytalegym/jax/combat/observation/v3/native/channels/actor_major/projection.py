"""Capacity-derived actor-first projection for the native group host.

The bridge's learner-v3 evidence frame is presently encoded in entity order
and its capability/geometry tail belongs to the compatibility actor.  This
module keeps the entity rows, which are genuinely actor-major, while replacing
every singleton capability with the reset-negotiated group row for the chosen
ego.  Geometry is closed for every composed row until the bridge publishes an
actor-major geometry producer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import uuid
from typing import Any

from hytalegym.jax.combat.arsenal.profiles import (
    hytale_0_5_7_native_profile_bindings,
)
from hytalegym.jax.combat.arsenal.schema.contract import ABILITY_CAPACITY
from hytalegym.jax.combat.observation.v3.native.codec.decode import (
    NATIVE_ACTOR_EVIDENCE_GEOMETRY_CELL_COUNT,
    parse_native_actor_evidence_frame,
)
from hytalegym.jax.combat.types import ENTITY_COUNT

from ..group import (
    NATIVE_GROUP_ACTION_SCHEMA,
    NATIVE_GROUP_ACTION_VERSION,
    NATIVE_POLICY_COMBAT_BINDING_SCHEMA,
    NATIVE_POLICY_COMBAT_BINDING_VERSION,
    NativeGroupStepEvidence,
    native_group_action_contract_sha256,
)


@dataclass(frozen=True, slots=True)
class NativeActorMajorCapabilities:
    """Reset-pinned actor ownership and policy-combat capabilities."""

    actor_identities: tuple[str, ...]
    policy_combat_bound: tuple[bool, ...]
    ability_slot_masks: tuple[int, ...]
    guard_available: tuple[bool, ...]
    dodge_available: tuple[bool, ...]
    item_ids: tuple[str, ...]

    @classmethod
    def from_info(
        cls,
        info: Mapping[str, object],
    ) -> NativeActorMajorCapabilities:
        """Decode group metadata that is present on reset and every step."""

        if not isinstance(info, Mapping):
            raise TypeError("native actor-major info must be a mapping")
        expected = {
            "native_group_action_schema": NATIVE_GROUP_ACTION_SCHEMA,
            "native_group_action_version": NATIVE_GROUP_ACTION_VERSION,
            "native_group_action_contract_sha256": (
                native_group_action_contract_sha256()
            ),
            "native_group_action_actor_capacity": ENTITY_COUNT,
            "native_group_action_binding_schema": (
                NATIVE_POLICY_COMBAT_BINDING_SCHEMA
            ),
            "native_group_action_binding_version": (
                NATIVE_POLICY_COMBAT_BINDING_VERSION
            ),
        }
        for name, value in expected.items():
            actual = info.get(name)
            if name.endswith("sha256"):
                actual = str(actual).upper()
                value = str(value).upper()
            if isinstance(value, int) and isinstance(actual, bool):
                actual = None
            if actual != value:
                raise ValueError(
                    f"native actor-major {name} mismatch: "
                    f"{actual!r} != {value!r}"
                )
        if not bool(info.get("native_group_action_available", False)):
            raise ValueError("native actor-major group actions are unavailable")

        identities = _actor_identities(
            info.get("native_policy_actor_identities")
        )
        result = cls(
            actor_identities=identities,
            policy_combat_bound=_csv_bits(
                info.get("native_group_action_policy_combat_bound")
            ),
            ability_slot_masks=_csv_ints(
                info.get("native_group_action_ability_slot_masks")
            ),
            guard_available=_csv_bits(
                info.get("native_group_action_guard_available")
            ),
            dodge_available=_csv_bits(
                info.get("native_group_action_dodge_available")
            ),
            item_ids=_csv_strings(
                info.get("native_group_action_item_ids")
            ),
        )
        for name, values in (
            ("actor identities", result.actor_identities),
            ("policy combat", result.policy_combat_bound),
            ("ability masks", result.ability_slot_masks),
            ("guard capability", result.guard_available),
            ("dodge capability", result.dodge_available),
            ("item bindings", result.item_ids),
        ):
            if len(values) != ENTITY_COUNT:
                raise ValueError(
                    f"native actor-major {name} differs from actor capacity"
                )
        if any(value < 0 for value in result.ability_slot_masks):
            raise ValueError("native actor-major ability mask is negative")
        return result

    def validate_profiles(self, actor_profiles: Mapping[int, str]) -> None:
        """Require exact reset bindings for every composer-owned slot."""

        for actor_slot, profile in actor_profiles.items():
            binding = hytale_0_5_7_native_profile_bindings(profile)
            if not self.policy_combat_bound[actor_slot]:
                raise ValueError(
                    f"native policy actor slot {actor_slot} is not bound"
                )
            expected_mask = (1 << len(binding.abilities)) - 1
            if self.ability_slot_masks[actor_slot] != expected_mask:
                raise ValueError(
                    f"native policy actor slot {actor_slot} ability binding "
                    "differs from its profile"
                )
            if self.guard_available[actor_slot] != (binding.guard is not None):
                raise ValueError(
                    f"native policy actor slot {actor_slot} guard binding "
                    "differs from its profile"
                )
            if self.item_ids[actor_slot] != binding.item_id:
                raise ValueError(
                    f"native policy actor slot {actor_slot} item binding "
                    "differs from its profile"
                )


def actor_first_order(
    actor_slot: int,
    target_slot: int,
    *,
    capacity: int = ENTITY_COUNT,
) -> tuple[int, ...]:
    """Return ego, selected target, then remaining entities in stable order."""

    for name, value in (("actor_slot", actor_slot), ("target_slot", target_slot)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0 or value >= capacity:
            raise ValueError(f"{name} is outside actor capacity")
    if actor_slot == target_slot:
        raise ValueError("actor and target slots must differ")
    return (actor_slot, target_slot) + tuple(
        slot
        for slot in range(capacity)
        if slot not in {actor_slot, target_slot}
    )


def project_actor_first_wire(
    wire_value: Mapping[str, Any],
    *,
    actor_slot: int,
    target_slot: int,
    actor_profile: str,
    capabilities: NativeActorMajorCapabilities,
    receipt: NativeGroupStepEvidence | None,
) -> dict[str, Any]:
    """Build one actor-first wire without borrowing singleton actor-0 facts."""

    parse_native_actor_evidence_frame(wire_value)
    order = actor_first_order(actor_slot, target_slot)
    wire = deepcopy(dict(wire_value))
    raw_actors = wire.get("actors")
    if not isinstance(raw_actors, list) or len(raw_actors) != ENTITY_COUNT:
        raise ValueError("native actor evidence actor rows differ from capacity")
    actors: list[dict[str, Any]] = []
    for projected_slot, source_slot in enumerate(order):
        row = deepcopy(raw_actors[source_slot])
        if not isinstance(row, dict) or row.get("entity_id") != source_slot:
            raise ValueError("native actor evidence ownership order changed")
        row["entity_id"] = projected_slot
        actors.append(row)
    wire["actors"] = actors

    binding = hytale_0_5_7_native_profile_bindings(actor_profile)
    ability_mask = capabilities.ability_slot_masks[actor_slot]
    active_slot = -1
    active = False
    if receipt is not None:
        active = receipt.ability_active[actor_slot]
        active_slot = receipt.ability_slots[actor_slot] if active else -1
    if active and not 0 <= active_slot < len(binding.abilities):
        raise ValueError("active group ability is outside the bound profile")
    world_tick = int(wire.get("world_tick", -1))
    abilities = []
    for slot in range(ABILITY_CAPACITY):
        authored = slot < len(binding.abilities)
        profile_row = binding.abilities[slot] if authored else None
        row_active = authored and active and slot == active_slot
        abilities.append(
            {
                "slot": slot,
                "interaction_id": (
                    "" if profile_row is None else profile_row.interaction_id
                ),
                "interaction_type": (
                    "" if profile_row is None else profile_row.interaction_type
                ),
                "authored": authored,
                "host_legal": authored and bool(ability_mask & (1 << slot)),
                "active": row_active,
                "start_world_tick": world_tick if row_active else -1,
                "finish_world_tick": -1,
            }
        )
    wire["abilities"] = abilities
    actors[0]["active_ability_slot"] = active_slot

    # These are singleton fields in the current bridge frame.  Replacing them
    # for every row (including slot zero) gives the new composer one uniform
    # law and leaves the existing single HytaleEnv path unchanged.
    wire["world_geometry_available"] = False
    wire["role_opaque_cell_mask_available"] = False
    wire["role_opaque_cell_mask"] = [
        False
    ] * NATIVE_ACTOR_EVIDENCE_GEOMETRY_CELL_COUNT
    wire["skill_action_mask"] = [True] * 6 + [
        capabilities.policy_combat_bound[actor_slot]
    ] * 3
    wire["jump_action_available"] = bool(actors[0].get("present", False))
    wire["guard_action_available"] = capabilities.guard_available[actor_slot]
    lateral = capabilities.dodge_available[actor_slot]
    wire["dodge_action_mask"] = [False, False, lateral, lateral]
    wire["door_action_mask"] = [[False] * 3 for _ in range(8)]
    return wire


def project_actor_info(
    info: Mapping[str, Any],
    *,
    actor_slot: int,
    inventory: Mapping[str, object],
    capabilities: NativeActorMajorCapabilities,
    receipt: NativeGroupStepEvidence | None,
) -> dict[str, Any]:
    """Replace actor-0 scalar capability/lifecycle fields for one ego row."""

    result = dict(info)
    result["native_inventory"] = inventory
    supported = {
        value.strip()
        for value in str(info.get("supported_actions", "")).split(",")
        if value.strip()
    }
    supported.difference_update(
        {"attack", "ability_slot", "guard_held", "dodge_direction"}
    )
    if capabilities.policy_combat_bound[actor_slot]:
        supported.add("attack")
    if capabilities.ability_slot_masks[actor_slot]:
        supported.add("ability_slot")
    if capabilities.guard_available[actor_slot]:
        supported.add("guard_held")
    if capabilities.dodge_available[actor_slot]:
        supported.add("dodge_direction")
    result["supported_actions"] = ",".join(sorted(supported))

    # The compatibility combat scalars have no actor-major producer.  Do not
    # reinterpret actor zero's attack/aim state as another actor's evidence.
    result.update(
        {
            "combat_agent_attack_executing": 0.0,
            "combat_target_attack_phase": 0.0,
            "combat_target_attack_index": -1.0,
            "combat_target_attack_elapsed_ticks": 0.0,
            "combat_target_head_yaw_degrees": 0.0,
            "combat_target_head_pitch_degrees": 0.0,
            "native_use_available": False,
        }
    )
    accepted = False if receipt is None else receipt.ability_accepted[actor_slot]
    accepted_slot = (
        -1 if receipt is None else receipt.ability_slots[actor_slot]
    )
    result["native_ability_accepted"] = accepted
    result["native_ability_accepted_slot"] = accepted_slot
    result["native_ability_slot"] = accepted_slot
    return result


def _actor_identities(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise TypeError("native policy actor identities must be an array")
    result = []
    for identity in value:
        if not isinstance(identity, str):
            raise TypeError("native policy actor identity must be a string")
        try:
            canonical = str(uuid.UUID(identity))
        except ValueError as error:
            raise ValueError("native policy actor identity is not a UUID") from error
        if identity != canonical:
            raise ValueError("native policy actor identity is not canonical")
        result.append(identity)
    return tuple(result)


def _csv_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError("native actor-major capability must be a CSV string")
    return tuple(value.split(","))


def _csv_bits(value: object) -> tuple[bool, ...]:
    values = _csv_strings(value)
    if any(item not in {"0", "1"} for item in values):
        raise ValueError("native actor-major boolean capability is malformed")
    return tuple(item == "1" for item in values)


def _csv_ints(value: object) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _csv_strings(value))
    except ValueError as error:
        raise ValueError("native actor-major integer capability is malformed") from error


__all__ = [
    "NativeActorMajorCapabilities",
    "actor_first_order",
    "project_actor_first_wire",
    "project_actor_info",
]
