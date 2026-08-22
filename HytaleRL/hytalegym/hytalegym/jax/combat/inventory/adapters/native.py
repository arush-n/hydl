"""Host validation for the sparse native six-container inventory frame."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


NATIVE_INVENTORY_SCHEMA = "hytalerl_native_inventory_v2"
NATIVE_INVENTORY_VERSION = 2
NATIVE_INVENTORY_V1_SCHEMA = "hytalerl_native_inventory_v1"
NATIVE_INVENTORY_V1_VERSION = 1
NATIVE_INVENTORY_CONTAINER_ORDER = (
    "storage",
    "armor",
    "hotbar",
    "utility",
    "tools",
    "backpack",
)
NATIVE_INVENTORY_SECTION_IDS = (-2, -3, -1, -5, -8, -9)


def _native_inventory_v1_contract_manifest() -> dict[str, Any]:
    return {
        "schema": NATIVE_INVENTORY_V1_SCHEMA,
        "version": NATIVE_INVENTORY_V1_VERSION,
        "containers": [
            {"name": name, "native_section_id": section_id}
            for name, section_id in zip(
                NATIVE_INVENTORY_CONTAINER_ORDER,
                NATIVE_INVENTORY_SECTION_IDS,
                strict=True,
            )
        ],
        "capacity": "native_ItemContainer_getCapacity_per_frame",
        "encoding": "sparse_occupied_slots_with_explicit_local_slot",
        "active_slots": ["hotbar", "utility", "tools"],
        "slot_fields": {
            "item_id": "exact_native_string",
            "item_runtime_index": (
                "sorted_Hytale_0_5_7_item_asset_index_zero_when_unmapped"
            ),
            "quantity": "positive_int32",
            "durability": "finite_nonnegative_float64",
            "max_durability": "finite_nonnegative_float64",
            "metadata_present": "boolean_no_metadata_payload",
        },
        "legacy_inventory": (
            "preserved_int36_hotbar_then_storage_ids_for_compatibility"
        ),
        "policy_boundary": ("telemetry_only_until_an_announced_actor_observation_move"),
    }


def native_inventory_contract_manifest() -> dict[str, Any]:
    """Describe the partial-validity, variable-capacity bridge telemetry."""

    manifest = _native_inventory_v1_contract_manifest()
    manifest.update(
        {
            "schema": NATIVE_INVENTORY_SCHEMA,
            "version": NATIVE_INVENTORY_VERSION,
            "frame_availability": (
                "inventory_component_present; false carries no container rows"
            ),
            "container_availability": (
                "independent_per_container; unavailable is distinct from empty"
            ),
            "unavailable_container": {
                "capacity": 0,
                "occupied_slots": [],
                "active_slot": -1,
                "reason": "required_nonempty_string",
            },
        }
    )
    return manifest


def _contract_sha256(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def native_inventory_contract_sha256() -> str:
    return _contract_sha256(native_inventory_contract_manifest())


def native_inventory_v1_contract_sha256() -> str:
    """Return the deployed v1 identity accepted during the v2 transition."""

    return _contract_sha256(_native_inventory_v1_contract_manifest())


def parse_native_inventory_frame(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one v1/v2 bridge frame without inventing missing containers."""

    if not isinstance(value, Mapping):
        raise TypeError("native inventory frame must be an object")
    schema = value.get("schema")
    version = value.get("version")
    if (
        schema == NATIVE_INVENTORY_SCHEMA
        and version == NATIVE_INVENTORY_VERSION
    ):
        contract_sha256 = native_inventory_contract_sha256()
        per_container_validity = True
    elif (
        schema == NATIVE_INVENTORY_V1_SCHEMA
        and version == NATIVE_INVENTORY_V1_VERSION
    ):
        contract_sha256 = native_inventory_v1_contract_sha256()
        per_container_validity = False
    else:
        raise ValueError("schema/version does not match a native inventory contract")
    _require_equal(value, "contract_sha256", contract_sha256)
    available = _boolean(value.get("available"), "available")
    reason = value.get("unavailable_reason")
    if not isinstance(reason, str):
        raise TypeError("unavailable_reason must be a string")

    active_raw = value.get("active_slots")
    if not isinstance(active_raw, Mapping):
        raise TypeError("active_slots must be an object")
    active_slots = {
        name: _integer(active_raw.get(name), f"active_slots.{name}")
        for name in ("hotbar", "utility", "tools")
    }
    containers_raw = value.get("containers")
    if not _is_sequence(containers_raw):
        raise TypeError("containers must be an array")

    if not available:
        if not reason:
            raise ValueError("unavailable frame must carry a reason")
        if containers_raw:
            raise ValueError("unavailable frame cannot carry containers")
        return {
            "schema": schema,
            "version": version,
            "contract_sha256": contract_sha256,
            "available": False,
            "unavailable_reason": reason,
            "active_slots": active_slots,
            "containers": [],
        }
    if reason:
        raise ValueError("available frame cannot carry an unavailable reason")
    if len(containers_raw) != len(NATIVE_INVENTORY_CONTAINER_ORDER):
        raise ValueError("available frame must carry all six containers")

    containers = [
        _parse_container(
            raw,
            name,
            section_id,
            per_container_validity=per_container_validity,
        )
        for raw, name, section_id in zip(
            containers_raw,
            NATIVE_INVENTORY_CONTAINER_ORDER,
            NATIVE_INVENTORY_SECTION_IDS,
            strict=True,
        )
    ]
    capacity_by_name = {
        container["name"]: container["capacity"] for container in containers
    }
    for name, slot in active_slots.items():
        container = next(
            row for row in containers if row["name"] == name
        )
        if not container["available"] and slot != -1:
            raise ValueError(
                f"active_slots.{name} must be -1 when its container is unavailable"
            )
        capacity = capacity_by_name[name]
        if container["available"] and (slot < -1 or slot >= capacity):
            raise ValueError(f"active_slots.{name} is outside its native capacity")

    return {
        "schema": schema,
        "version": version,
        "contract_sha256": contract_sha256,
        "available": True,
        "unavailable_reason": "",
        "active_slots": active_slots,
        "containers": containers,
    }


def _parse_container(
    value: Any,
    expected_name: str,
    expected_section_id: int,
    *,
    per_container_validity: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"container {expected_name} must be an object")
    _require_equal(value, "name", expected_name)
    _require_equal(value, "section_id", expected_section_id)
    capacity = _integer(value.get("capacity"), f"{expected_name}.capacity")
    if capacity < 0:
        raise ValueError(f"{expected_name}.capacity must be nonnegative")
    occupied_raw = value.get("occupied_slots")
    if not _is_sequence(occupied_raw):
        raise TypeError(f"{expected_name}.occupied_slots must be an array")
    if per_container_validity:
        available = _boolean(
            value.get("available"),
            f"{expected_name}.available",
        )
        reason = value.get("unavailable_reason")
        if not isinstance(reason, str):
            raise TypeError(f"{expected_name}.unavailable_reason must be a string")
    else:
        available = True
        reason = ""
    if not available:
        if not reason:
            raise ValueError(
                f"unavailable {expected_name} container must carry a reason"
            )
        if capacity != 0 or occupied_raw:
            raise ValueError(
                f"unavailable {expected_name} container cannot carry contents"
            )
        return {
            "name": expected_name,
            "section_id": expected_section_id,
            "available": False,
            "unavailable_reason": reason,
            "capacity": 0,
            "occupied_slots": [],
        }
    if reason:
        raise ValueError(
            f"available {expected_name} container cannot carry an unavailable reason"
        )

    occupied = []
    seen_slots: set[int] = set()
    for raw in occupied_raw:
        slot = _parse_slot(raw, expected_name, capacity)
        index = slot["slot"]
        if index in seen_slots:
            raise ValueError(f"{expected_name} repeats occupied slot {index}")
        seen_slots.add(index)
        occupied.append(slot)
    if [slot["slot"] for slot in occupied] != sorted(seen_slots):
        raise ValueError(
            f"{expected_name}.occupied_slots must be in ascending slot order"
        )
    return {
        "name": expected_name,
        "section_id": expected_section_id,
        "available": True,
        "unavailable_reason": "",
        "capacity": capacity,
        "occupied_slots": occupied,
    }


def _parse_slot(
    value: Any,
    container_name: str,
    capacity: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{container_name} slot must be an object")
    slot = _integer(value.get("slot"), f"{container_name}.slot")
    if not 0 <= slot < capacity:
        raise ValueError(f"{container_name}.slot is outside native capacity")
    item_id = value.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(f"{container_name}.item_id must be nonempty")
    runtime_index = _integer(
        value.get("item_runtime_index"),
        f"{container_name}.item_runtime_index",
    )
    if runtime_index < 0:
        raise ValueError("item_runtime_index must be nonnegative")
    quantity = _integer(value.get("quantity"), f"{container_name}.quantity")
    if not 0 < quantity <= (2**31 - 1):
        raise ValueError("quantity must be a positive int32")
    durability = _finite_nonnegative(
        value.get("durability"),
        f"{container_name}.durability",
    )
    maximum = _finite_nonnegative(
        value.get("max_durability"),
        f"{container_name}.max_durability",
    )
    if durability > maximum:
        raise ValueError("durability cannot exceed max_durability")
    metadata_present = _boolean(
        value.get("metadata_present"),
        f"{container_name}.metadata_present",
    )
    return {
        "slot": slot,
        "item_id": item_id,
        "item_runtime_index": runtime_index,
        "quantity": quantity,
        "durability": durability,
        "max_durability": maximum,
        "metadata_present": metadata_present,
    }


def _require_equal(value: Mapping[str, Any], key: str, expected: Any) -> None:
    if value.get(key) != expected:
        raise ValueError(f"{key} does not match the native inventory contract")


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


__all__ = [
    "NATIVE_INVENTORY_CONTAINER_ORDER",
    "NATIVE_INVENTORY_SCHEMA",
    "NATIVE_INVENTORY_SECTION_IDS",
    "NATIVE_INVENTORY_VERSION",
    "NATIVE_INVENTORY_V1_SCHEMA",
    "NATIVE_INVENTORY_V1_VERSION",
    "native_inventory_contract_manifest",
    "native_inventory_contract_sha256",
    "native_inventory_v1_contract_sha256",
    "parse_native_inventory_frame",
]
