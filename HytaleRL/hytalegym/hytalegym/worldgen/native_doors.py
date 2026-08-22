"""Strict host contract for native vertical-door transition evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

NATIVE_DOOR_EVIDENCE_SCHEMA = "hytalerl_native_door_transition_evidence_v1"
NATIVE_DOOR_EVIDENCE_VERSION = 1
NATIVE_DOOR_ASSET_ID = "Furniture_Kweebec_Door"
NATIVE_DOOR_ROOT_INTERACTION_ID = "Door"
NATIVE_DOOR_TRANSITION_EXECUTION = (
    "native_vertical_door_interaction_direct_on_world_thread"
)
NATIVE_DOOR_YAWS = (0, 90, 180, 270)

_EXPECTED_KEYS = {
    "type",
    "schema",
    "version",
    "server_version",
    "world",
    "worldgen_provider",
    "worldgen_version",
    "seed",
    "door_asset_id",
    "root_interaction_id",
    "transition_execution",
    "timing_certified",
    "rows",
}
_ROW_KEYS = {
    "yaw_degrees",
    "partner_offset",
    "single_front_open_state",
    "single_front_open_hitbox_type",
    "single_closed_after_front",
    "single_back_open_state",
    "single_back_open_hitbox_type",
    "single_closed_after_back",
    "double_root_open_state",
    "double_root_open_hitbox_type",
    "double_partner_open_state",
    "double_partner_open_hitbox_type",
    "double_root_closed",
    "double_partner_closed",
}


@dataclass(frozen=True)
class NativeDoorTransitionRow:
    """One yaw's exact single- and paired-door physical transitions."""

    yaw_degrees: int
    partner_offset: tuple[int, int, int]
    single_front_open_state: str
    single_front_open_hitbox_type: str
    single_closed_after_front: bool
    single_back_open_state: str
    single_back_open_hitbox_type: str
    single_closed_after_back: bool
    double_root_open_state: str
    double_root_open_hitbox_type: str
    double_partner_open_state: str
    double_partner_open_hitbox_type: str
    double_root_closed: bool
    double_partner_closed: bool

    @classmethod
    def from_wire(cls, value: Any) -> NativeDoorTransitionRow:
        if not isinstance(value, Mapping) or set(value) != _ROW_KEYS:
            raise ValueError("native door row fields do not match v1")
        offset = tuple(
            _integer(component, "partner_offset")
            for component in _sequence(value["partner_offset"], "partner_offset")
        )
        if (
            len(offset) != 3
            or offset[1] != 0
            or sum(component != 0 for component in offset) != 1
        ):
            raise ValueError("door partner offset must be axis-aligned horizontal")
        return cls(
            yaw_degrees=_integer(value["yaw_degrees"], "yaw_degrees"),
            partner_offset=offset,  # type: ignore[arg-type]
            single_front_open_state=_text(
                value["single_front_open_state"],
                "single_front_open_state",
            ),
            single_front_open_hitbox_type=_text(
                value["single_front_open_hitbox_type"],
                "single_front_open_hitbox_type",
            ),
            single_closed_after_front=_boolean(
                value["single_closed_after_front"],
                "single_closed_after_front",
            ),
            single_back_open_state=_text(
                value["single_back_open_state"],
                "single_back_open_state",
            ),
            single_back_open_hitbox_type=_text(
                value["single_back_open_hitbox_type"],
                "single_back_open_hitbox_type",
            ),
            single_closed_after_back=_boolean(
                value["single_closed_after_back"],
                "single_closed_after_back",
            ),
            double_root_open_state=_text(
                value["double_root_open_state"],
                "double_root_open_state",
            ),
            double_root_open_hitbox_type=_text(
                value["double_root_open_hitbox_type"],
                "double_root_open_hitbox_type",
            ),
            double_partner_open_state=_text(
                value["double_partner_open_state"],
                "double_partner_open_state",
            ),
            double_partner_open_hitbox_type=_text(
                value["double_partner_open_hitbox_type"],
                "double_partner_open_hitbox_type",
            ),
            double_root_closed=_boolean(
                value["double_root_closed"],
                "double_root_closed",
            ),
            double_partner_closed=_boolean(
                value["double_partner_closed"],
                "double_partner_closed",
            ),
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class NativeDoorTransitionEvidence:
    """Validated native evidence; interaction timing is explicitly excluded."""

    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    rows: tuple[NativeDoorTransitionRow, ...]

    @classmethod
    def from_response(
        cls,
        response: Mapping[str, Any],
    ) -> NativeDoorTransitionEvidence:
        if set(response) != _EXPECTED_KEYS:
            raise ValueError("native door response fields do not match v1")
        if response["type"] != "door_evidence":
            raise ValueError("bridge returned the wrong native door message type")
        if (
            response["schema"] != NATIVE_DOOR_EVIDENCE_SCHEMA
            or _integer(response["version"], "version")
            != NATIVE_DOOR_EVIDENCE_VERSION
        ):
            raise ValueError("unsupported native door evidence contract")
        if response["door_asset_id"] != NATIVE_DOOR_ASSET_ID:
            raise ValueError("native door fixture asset changed")
        if response["root_interaction_id"] != NATIVE_DOOR_ROOT_INTERACTION_ID:
            raise ValueError("native door root interaction changed")
        if (
            response["transition_execution"]
            != NATIVE_DOOR_TRANSITION_EXECUTION
        ):
            raise ValueError("native door execution path changed")
        if _boolean(response["timing_certified"], "timing_certified"):
            raise ValueError("native door v1 must not claim transition timing")
        rows = tuple(
            NativeDoorTransitionRow.from_wire(row)
            for row in _sequence(response["rows"], "rows")
        )
        if tuple(row.yaw_degrees for row in rows) != NATIVE_DOOR_YAWS:
            raise ValueError("native door rows must cover ordered cardinal yaws")
        return cls(
            server_version=_text(response["server_version"], "server_version"),
            world=_text(response["world"], "world"),
            worldgen_provider=_text(
                response["worldgen_provider"],
                "worldgen_provider",
            ),
            worldgen_version=_text(
                response["worldgen_version"],
                "worldgen_version",
            ),
            seed=_integer(response["seed"], "seed"),
            rows=rows,
        )

    def transition_digest(self) -> str:
        return _digest([row.semantic_payload() for row in self.rows])

    def semantic_digest(self) -> str:
        return _digest(
            {
                "schema": NATIVE_DOOR_EVIDENCE_SCHEMA,
                "version": NATIVE_DOOR_EVIDENCE_VERSION,
                "server_version": self.server_version,
                "worldgen_provider": self.worldgen_provider,
                "worldgen_version": self.worldgen_version,
                "seed": self.seed,
                "rows": [row.semantic_payload() for row in self.rows],
            }
        )


def native_door_transition_contract() -> dict[str, object]:
    return {
        "schema": NATIVE_DOOR_EVIDENCE_SCHEMA,
        "version": NATIVE_DOOR_EVIDENCE_VERSION,
        "message_type": "door_evidence",
        "published_provenance": "native",
        "source_evidence": "installed_0.5.7_source_pin",
        "live_evidence": NATIVE_DOOR_TRANSITION_EXECUTION,
        "fixture": {
            "world": "isolated_flat",
            "asset_id": NATIVE_DOOR_ASSET_ID,
            "root_interaction_id": NATIVE_DOOR_ROOT_INTERACTION_ID,
            "yaw_degrees": list(NATIVE_DOOR_YAWS),
            "side_definition": {
                "normal": "(sin(yaw),0,cos(yaw))",
                "front": "dot(actor-door_center,normal)<0",
                "back": "dot(actor-door_center,normal)>=0",
            },
        },
        "certifies": [
            "single_door_front_and_back_physical_state",
            "single_door_close",
            "double_door_reciprocal_state",
            "double_door_close",
            "open_state_hitbox_type",
        ],
        "does_not_certify": [
            "client_or_network_transition_timing",
            "animation",
            "sound",
            "lock_or_permission",
            "soft_block_breaking_or_drops",
            "entity_occupied_clearance",
            "horizontal_trapdoors",
            "root_interaction_cooldown_or_chain_scheduling",
        ],
        "surrogate_fallback": {
            "source": "asset_derived_vertical_door_state_table",
            "closed_actor_front": "OpenDoorIn",
            "closed_actor_back": "OpenDoorOut",
            "open": "closed",
        },
        "unsupported_mechanics": "fail_closed",
    }


def native_door_transition_contract_sha256() -> str:
    return _digest(native_door_transition_contract())


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be boolean")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be non-empty text")
    return value


__all__ = [
    "NATIVE_DOOR_ASSET_ID",
    "NATIVE_DOOR_EVIDENCE_SCHEMA",
    "NATIVE_DOOR_EVIDENCE_VERSION",
    "NATIVE_DOOR_ROOT_INTERACTION_ID",
    "NATIVE_DOOR_TRANSITION_EXECUTION",
    "NATIVE_DOOR_YAWS",
    "NativeDoorTransitionEvidence",
    "NativeDoorTransitionRow",
    "native_door_transition_contract",
    "native_door_transition_contract_sha256",
]
