"""Bounded actor-major messages for one shared native engine step."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.profiles import (
    hytale_0_5_7_native_profile_bindings,
)
from hytalegym.jax.combat.types import ENTITY_COUNT

from ..codec.transport import NativeActionTranslation


NATIVE_GROUP_ACTION_SCHEMA = "hytalerl_native_group_action_v5"
NATIVE_GROUP_ACTION_VERSION = 5
NATIVE_POLICY_COMBAT_BINDING_SCHEMA = (
    "hytalerl_native_policy_combat_binding_v1"
)
NATIVE_POLICY_COMBAT_BINDING_VERSION = 1


def native_group_action_contract_manifest() -> dict[str, object]:
    """Return the same canonical, capacity-derived contract as Java."""

    return {
        "actor_capacity": ENTITY_COUNT,
        "actor_order": "entity_id_ascending",
        "atomicity": "one_shared_engine_tick_actor_major",
        "binding_schema": NATIVE_POLICY_COMBAT_BINDING_SCHEMA,
        "binding_version": NATIVE_POLICY_COMBAT_BINDING_VERSION,
        "empty_actor_step": "native_ai_unoverridden",
        "entity_zero_only_surface": "legacy_use_place_break_craft",
        "nonzero_actor_precondition": (
            "kill_trork_and_combat_target_active_false"
        ),
        "nonzero_actor_surface": (
            "movement_look_jump_hotbar_role_attack_"
            "policy_item_ability_guard_dodge_charge"
            "_typed_world_verbs"
        ),
        "schema": NATIVE_GROUP_ACTION_SCHEMA,
        "typed_world_verb_evidence": (
            "candidate_generation_and_selected_semantic_sha256_required"
        ),
        "version": NATIVE_GROUP_ACTION_VERSION,
        "world_verb_arbitration": (
            "entity_id_ascending_first_admitted_per_exact_target"
        ),
    }


def native_group_action_contract_sha256() -> str:
    canonical = json.dumps(
        native_group_action_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper()


@dataclass(frozen=True)
class NativeGroupActionMessage:
    """Wire message plus host legality retained for actor-level accounting."""

    message: dict[str, object]
    entity_id: tuple[int, ...]
    legal: tuple[bool, ...]
    reject_reasons: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class NativeGroupStepEvidence:
    """Validated bridge lifecycle for one bounded group step."""

    entity_id: tuple[int, ...]
    control_ticks: tuple[int, ...]
    attack_requested: tuple[bool, ...]
    attack_accepted: tuple[bool, ...]
    attack_reject_reasons: tuple[str, ...]
    policy_combat_bound: tuple[bool, ...]
    ability_slot_masks: tuple[int, ...]
    guard_available: tuple[bool, ...]
    dodge_available: tuple[bool, ...]
    item_ids: tuple[str, ...]
    ability_requested: tuple[bool, ...]
    ability_accepted: tuple[bool, ...]
    ability_started: tuple[bool, ...]
    ability_finished: tuple[bool, ...]
    ability_failed: tuple[bool, ...]
    ability_active: tuple[bool, ...]
    ability_slots: tuple[int, ...]
    ability_reject_reasons: tuple[str, ...]
    ability_interaction_ids: tuple[str, ...]
    guard_requested: tuple[bool, ...]
    guard_accepted: tuple[bool, ...]
    guard_started: tuple[bool, ...]
    guard_finished: tuple[bool, ...]
    guard_active: tuple[bool, ...]
    guard_reject_reasons: tuple[str, ...]
    dodge_requested: tuple[bool, ...]
    dodge_accepted: tuple[bool, ...]
    dodge_started: tuple[bool, ...]
    dodge_finished: tuple[bool, ...]
    dodge_failed: tuple[bool, ...]
    dodge_active: tuple[bool, ...]
    dodge_directions: tuple[int, ...]
    dodge_reject_reasons: tuple[str, ...]
    dodge_interaction_ids: tuple[str, ...]
    world_verb_requested: tuple[bool, ...]
    world_verb_accepted: tuple[bool, ...]
    world_verb_started: tuple[bool, ...]
    world_verb_finished: tuple[bool, ...]
    world_verb_failed: tuple[bool, ...]
    world_verb_active: tuple[bool, ...]
    world_verb_reject_reasons: tuple[str, ...]
    world_verb_verbs: tuple[str, ...]
    world_verb_candidate_generations: tuple[str, ...]
    world_verb_selected_semantics: tuple[str, ...]

    @classmethod
    def from_info(cls, info: Mapping[str, object]) -> NativeGroupStepEvidence:
        if not isinstance(info, Mapping):
            raise TypeError("native group step info must be a mapping")
        expected = native_group_action_contract_manifest()
        if info.get("native_group_action_schema") != expected["schema"]:
            raise ValueError("native group action schema mismatch")
        if int(info.get("native_group_action_version", -1)) != int(
            expected["version"]
        ):
            raise ValueError("native group action version mismatch")
        if (
            str(info.get("native_group_action_contract_sha256", "")).upper()
            != native_group_action_contract_sha256()
        ):
            raise ValueError("native group action contract mismatch")
        if int(info.get("native_group_action_actor_capacity", -1)) != ENTITY_COUNT:
            raise ValueError("native group action actor capacity mismatch")
        if (
            info.get("native_group_action_binding_schema")
            != NATIVE_POLICY_COMBAT_BINDING_SCHEMA
            or int(info.get("native_group_action_binding_version", -1))
            != NATIVE_POLICY_COMBAT_BINDING_VERSION
        ):
            raise ValueError("native group policy-combat binding mismatch")
        if not bool(info.get("native_group_action_available", False)):
            raise ValueError("native group actions are unavailable in this reset")
        if (
            info.get("native_group_action_empty_actor_step")
            != expected["empty_actor_step"]
        ):
            raise ValueError("native group empty-actor semantics mismatch")

        entity_ids = _csv_ints(
            info.get("native_group_action_requested_entity_ids", ""),
            allow_empty=True,
        )
        if tuple(sorted(set(entity_ids))) != entity_ids:
            raise ValueError("native group step actor IDs are not canonical")
        if any(value < 0 or value >= ENTITY_COUNT for value in entity_ids):
            raise ValueError("native group step actor ID is outside capacity")
        control_ticks = _csv_ints(
            info.get("native_group_action_control_ticks", "")
        )
        attack_requested = _csv_bits(
            info.get("native_group_action_attack_requested", "")
        )
        attack_accepted = _csv_bits(
            info.get("native_group_action_attack_accepted", "")
        )
        reject_reasons = _csv_strings(
            info.get("native_group_action_attack_reject_reasons", "")
        )
        bit_fields = {
            name: _csv_bits(info.get(f"native_group_action_{name}", ""))
            for name in (
                "policy_combat_bound",
                "guard_available",
                "dodge_available",
                "ability_requested",
                "ability_accepted",
                "ability_started",
                "ability_finished",
                "ability_failed",
                "ability_active",
                "guard_requested",
                "guard_accepted",
                "guard_started",
                "guard_finished",
                "guard_active",
                "dodge_requested",
                "dodge_accepted",
                "dodge_started",
                "dodge_finished",
                "dodge_failed",
                "dodge_active",
                "world_verb_requested",
                "world_verb_accepted",
                "world_verb_started",
                "world_verb_finished",
                "world_verb_failed",
                "world_verb_active",
            )
        }
        int_fields = {
            name: _csv_ints(info.get(f"native_group_action_{name}", ""))
            for name in (
                "ability_slot_masks",
                "ability_slots",
                "dodge_directions",
            )
        }
        string_fields = {
            name: _csv_strings(info.get(f"native_group_action_{name}", ""))
            for name in (
                "item_ids",
                "ability_reject_reasons",
                "ability_interaction_ids",
                "guard_reject_reasons",
                "dodge_reject_reasons",
                "dodge_interaction_ids",
                "world_verb_reject_reasons",
                "world_verb_verbs",
                "world_verb_candidate_generations",
                "world_verb_selected_semantics",
            )
        }
        for name, values in (
            ("control_ticks", control_ticks),
            ("attack_requested", attack_requested),
            ("attack_accepted", attack_accepted),
            ("attack_reject_reasons", reject_reasons),
            *bit_fields.items(),
            *int_fields.items(),
            *string_fields.items(),
        ):
            if len(values) != ENTITY_COUNT:
                raise ValueError(
                    f"native group {name} length differs from actor capacity"
                )
        return cls(
            entity_id=entity_ids,
            control_ticks=control_ticks,
            attack_requested=attack_requested,
            attack_accepted=attack_accepted,
            attack_reject_reasons=reject_reasons,
            policy_combat_bound=bit_fields["policy_combat_bound"],
            ability_slot_masks=int_fields["ability_slot_masks"],
            guard_available=bit_fields["guard_available"],
            dodge_available=bit_fields["dodge_available"],
            item_ids=string_fields["item_ids"],
            ability_requested=bit_fields["ability_requested"],
            ability_accepted=bit_fields["ability_accepted"],
            ability_started=bit_fields["ability_started"],
            ability_finished=bit_fields["ability_finished"],
            ability_failed=bit_fields["ability_failed"],
            ability_active=bit_fields["ability_active"],
            ability_slots=int_fields["ability_slots"],
            ability_reject_reasons=string_fields[
                "ability_reject_reasons"
            ],
            ability_interaction_ids=string_fields[
                "ability_interaction_ids"
            ],
            guard_requested=bit_fields["guard_requested"],
            guard_accepted=bit_fields["guard_accepted"],
            guard_started=bit_fields["guard_started"],
            guard_finished=bit_fields["guard_finished"],
            guard_active=bit_fields["guard_active"],
            guard_reject_reasons=string_fields["guard_reject_reasons"],
            dodge_requested=bit_fields["dodge_requested"],
            dodge_accepted=bit_fields["dodge_accepted"],
            dodge_started=bit_fields["dodge_started"],
            dodge_finished=bit_fields["dodge_finished"],
            dodge_failed=bit_fields["dodge_failed"],
            dodge_active=bit_fields["dodge_active"],
            dodge_directions=int_fields["dodge_directions"],
            dodge_reject_reasons=string_fields["dodge_reject_reasons"],
            dodge_interaction_ids=string_fields[
                "dodge_interaction_ids"
            ],
            world_verb_requested=bit_fields["world_verb_requested"],
            world_verb_accepted=bit_fields["world_verb_accepted"],
            world_verb_started=bit_fields["world_verb_started"],
            world_verb_finished=bit_fields["world_verb_finished"],
            world_verb_failed=bit_fields["world_verb_failed"],
            world_verb_active=bit_fields["world_verb_active"],
            world_verb_reject_reasons=string_fields[
                "world_verb_reject_reasons"
            ],
            world_verb_verbs=string_fields["world_verb_verbs"],
            world_verb_candidate_generations=string_fields[
                "world_verb_candidate_generations"
            ],
            world_verb_selected_semantics=string_fields[
                "world_verb_selected_semantics"
            ],
        )


class NativeGroupStepBatch(NamedTuple):
    """Fixed-shape JAX view of exact Java action lifecycle receipts.

    Action axes are attack, ability, guard, dodge. Lifecycle axes omit attack
    because its exact server receipt is an acceptance edge, not a held state.
    Symbolic IDs and reject reasons remain available on the host evidence.
    """

    actor_requested: jax.Array
    control_tick: jax.Array
    requested: jax.Array
    accepted: jax.Array
    rejected: jax.Array
    started: jax.Array
    finished: jax.Array
    active: jax.Array
    failed: jax.Array
    capability: jax.Array
    ability_slot_mask: jax.Array
    ability_slot: jax.Array
    dodge_direction: jax.Array
    world_verb_requested: jax.Array
    world_verb_accepted: jax.Array
    world_verb_started: jax.Array
    world_verb_finished: jax.Array
    world_verb_failed: jax.Array
    world_verb_active: jax.Array
    world_verb_rejected: jax.Array


def pack_native_group_step_evidence(
    evidence: NativeGroupStepEvidence,
) -> NativeGroupStepBatch:
    """Transfer one validated actor-major Java receipt into JAX arrays."""

    if not isinstance(evidence, NativeGroupStepEvidence):
        raise TypeError("evidence must be NativeGroupStepEvidence")
    entity_ids = np.asarray(evidence.entity_id, dtype=np.int32)
    if (
        np.any(entity_ids < 0)
        or np.any(entity_ids >= ENTITY_COUNT)
        or len(np.unique(entity_ids)) != len(entity_ids)
    ):
        raise ValueError("native group evidence actor IDs are invalid")
    actor_requested = np.zeros(ENTITY_COUNT, dtype=np.bool_)
    actor_requested[entity_ids] = True

    def columns(names: tuple[str, ...], dtype: object) -> jax.Array:
        value = np.asarray(
            [getattr(evidence, name) for name in names], dtype=dtype
        ).T
        if value.shape != (ENTITY_COUNT, len(names)):
            raise ValueError("native group evidence is not actor-major")
        return jnp.asarray(value)

    rejected = np.asarray(
        [
            evidence.attack_reject_reasons,
            evidence.ability_reject_reasons,
            evidence.guard_reject_reasons,
            evidence.dodge_reject_reasons,
        ],
        dtype=object,
    ).T
    return NativeGroupStepBatch(
        actor_requested=jnp.asarray(actor_requested),
        control_tick=columns(("control_ticks",), np.int32)[:, 0],
        requested=columns(
            (
                "attack_requested",
                "ability_requested",
                "guard_requested",
                "dodge_requested",
            ),
            np.bool_,
        ),
        accepted=columns(
            (
                "attack_accepted",
                "ability_accepted",
                "guard_accepted",
                "dodge_accepted",
            ),
            np.bool_,
        ),
        rejected=jnp.asarray(rejected != ""),
        started=columns(
            ("ability_started", "guard_started", "dodge_started"), np.bool_
        ),
        finished=columns(
            ("ability_finished", "guard_finished", "dodge_finished"),
            np.bool_,
        ),
        active=columns(
            ("ability_active", "guard_active", "dodge_active"), np.bool_
        ),
        failed=columns(("ability_failed", "dodge_failed"), np.bool_),
        capability=columns(
            ("policy_combat_bound", "guard_available", "dodge_available"),
            np.bool_,
        ),
        ability_slot_mask=columns(("ability_slot_masks",), np.int32)[:, 0],
        ability_slot=columns(("ability_slots",), np.int32)[:, 0],
        dodge_direction=columns(("dodge_directions",), np.int32)[:, 0],
        world_verb_requested=columns(
            ("world_verb_requested",), np.bool_
        )[:, 0],
        world_verb_accepted=columns(
            ("world_verb_accepted",), np.bool_
        )[:, 0],
        world_verb_started=columns(
            ("world_verb_started",), np.bool_
        )[:, 0],
        world_verb_finished=columns(
            ("world_verb_finished",), np.bool_
        )[:, 0],
        world_verb_failed=columns(
            ("world_verb_failed",), np.bool_
        )[:, 0],
        world_verb_active=columns(
            ("world_verb_active",), np.bool_
        )[:, 0],
        world_verb_rejected=jnp.asarray(
            np.asarray(evidence.world_verb_reject_reasons, dtype=object) != ""
        ),
    )


def native_group_policy_combat_reset_options(
    actor_profiles: Mapping[int, str],
) -> dict[str, object]:
    """Build reset-pinned native item roots for selected policy actors.

    Profile-local authored slots remain a host concern.  The bridge receives
    the same compact slot order used by ``NativeActorEvidenceAssembler`` and
    never switches on weapon/profile names at execution time.
    """

    if not isinstance(actor_profiles, Mapping) or not actor_profiles:
        raise ValueError("actor_profiles must be a nonempty mapping")
    rows: list[dict[str, object]] = []
    for entity_id, profile in sorted(actor_profiles.items()):
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("native policy-combat entity IDs must be integers")
        if entity_id <= 0 or entity_id >= ENTITY_COUNT:
            raise ValueError("native policy-combat entity ID is outside capacity")
        if not isinstance(profile, str) or not profile:
            raise TypeError("native policy-combat profiles must be nonempty strings")
        binding = hytale_0_5_7_native_profile_bindings(profile)
        if not binding.item_id or (
            not binding.abilities and binding.guard is None
        ):
            raise ValueError(
                f"profile {profile!r} has no native policy-item binding"
            )
        row: dict[str, object] = {
            "entity_id": entity_id,
            "item_id": binding.item_id,
            "ability_slots": list(range(len(binding.abilities))),
            "ability_interaction_ids": [
                ability.interaction_id for ability in binding.abilities
            ],
            "ability_interaction_types": [
                ability.interaction_type for ability in binding.abilities
            ],
        }
        if binding.guard is not None:
            row.update(
                {
                    "guard_interaction_id": binding.guard.interaction_id,
                    "guard_interaction_type": binding.guard.interaction_type,
                }
            )
        rows.append(row)
    return {"native_group_policy_combat_bindings": rows}


def native_group_step_message(
    entity_ids: Sequence[int],
    translations: Sequence[NativeActionTranslation],
) -> NativeGroupActionMessage:
    """Pack unique actor actions without collapsing their legality outcomes."""

    ids = tuple(entity_ids)
    rows = tuple(translations)
    if len(ids) != len(rows):
        raise ValueError(
            "entity_ids and translations must be equally sized"
        )
    if len(ids) > ENTITY_COUNT:
        raise ValueError("native group action exceeds actor capacity")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise TypeError("native group entity IDs must be integers")
    if any(value < 0 or value >= ENTITY_COUNT for value in ids):
        raise ValueError("native group entity ID is outside actor capacity")
    if len(set(ids)) != len(ids):
        raise ValueError("native group entity IDs must be unique")
    if any(not isinstance(row, NativeActionTranslation) for row in rows):
        raise TypeError(
            "translations must contain NativeActionTranslation rows"
        )

    ordered = tuple(
        sorted(zip(ids, rows, strict=True), key=lambda pair: pair[0])
    )
    return NativeGroupActionMessage(
        message={
            "type": "step",
            "actions": [
                {"entity_id": entity_id, "action": dict(row.action)}
                for entity_id, row in ordered
            ],
        },
        entity_id=tuple(entity_id for entity_id, _ in ordered),
        legal=tuple(row.legal for _, row in ordered),
        reject_reasons=tuple(row.reject_reasons for _, row in ordered),
    )


def native_autonomous_step_message() -> NativeGroupActionMessage:
    """Advance one shared step without overriding any Java NPC."""

    return native_group_step_message((), ())


def _csv_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError("native group lifecycle field must be a string")
    return tuple(value.split(","))


def _csv_ints(
    value: object,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    values = _csv_strings(value)
    if allow_empty and values == ("",):
        return ()
    try:
        return tuple(int(item) for item in values)
    except ValueError as error:
        raise ValueError("native group integer lifecycle field is malformed") from error


def _csv_bits(value: object) -> tuple[bool, ...]:
    values = _csv_strings(value)
    if any(item not in {"0", "1"} for item in values):
        raise ValueError("native group bit lifecycle field is malformed")
    return tuple(item == "1" for item in values)


__all__ = [
    "NATIVE_GROUP_ACTION_SCHEMA",
    "NATIVE_GROUP_ACTION_VERSION",
    "NATIVE_POLICY_COMBAT_BINDING_SCHEMA",
    "NATIVE_POLICY_COMBAT_BINDING_VERSION",
    "NativeGroupActionMessage",
    "NativeGroupStepBatch",
    "NativeGroupStepEvidence",
    "native_group_action_contract_manifest",
    "native_group_action_contract_sha256",
    "native_autonomous_step_message",
    "native_group_policy_combat_reset_options",
    "pack_native_group_step_evidence",
    "native_group_step_message",
]
