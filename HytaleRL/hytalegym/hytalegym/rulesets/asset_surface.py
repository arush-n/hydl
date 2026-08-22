"""Versioned activation audit for unmodelled Hytale combat asset surfaces.

The server codecs expose more mechanics than the current compiled ruleset uses.
This module scans only the shipped asset families that can contain those combat
codecs, records every activation, and compares the result with a manifest tied
to the exact ``Assets.zip`` identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

from hytalegym.combat.assets import (
    HYTALE_0_5_7_ASSETS_SHA256,
    file_sha256,
)


ASSET_SURFACE_AUDIT_SCHEMA = "hytalerl_combat_asset_surface_v1"
ASSET_SURFACE_AUDIT_RESOURCE = "hytale_0_5_7/asset_surface_v1.json"
HYTALE_ASSETS_ZIP_ENV = "HYTALE_ASSETS_ZIP"

# These are the asset families decoded through the server codecs under audit.
# Prefabs and client media cannot instantiate DamageCalculator,
# WieldingInteraction, DamageEffects, ItemArmor, ItemDurability, or NPC role
# fields.
ASSET_SURFACE_SCAN_PREFIXES = (
    "Server/Entity/Effects/",
    "Server/Entity/HitboxCollision/",
    "Server/Entity/Repulsion/",
    "Server/GameplayConfigs/",
    "Server/Item/Interactions/",
    "Server/Item/Items/",
    "Server/NPC/Balancing/",
    "Server/NPC/Roles/",
    "Server/Projectiles/",
)

_ARMOR_ITEM_PREFIX = "Server/Item/Items/Armor/"
_ENTITY_EFFECT_PREFIX = "Server/Entity/Effects/"
_GAMEPLAY_CONFIG_PREFIX = "Server/GameplayConfigs/"
_HITBOX_COLLISION_PREFIX = "Server/Entity/HitboxCollision/"
_BALANCE_PREFIX = "Server/NPC/Balancing/"
_ROLE_PREFIX = "Server/NPC/Roles/"
_MATCHUP_ROLE_IDS = ("Kweebec_Razorleaf", "Trork_Brawler")
_MATCHUP_GUARD_PATH = (
    "Server/Item/Interactions/Weapons/Spear/Attacks/Block/Spear_Block_Damage.json"
)
_MATCHUP_DAMAGE_PATH = "Server/Item/Interactions/NPCs/NPC_Attack_Melee_Damage.json"

_SURFACE_NAMES = (
    "damage_calculator_multi_cause",
    "damage_calculator_dps",
    "sequential_modifier_step",
    "sequential_modifier_minimum",
    "stamina_drain_multiplier",
    "item_armor_damage_resistance",
    "item_armor_base_damage_resistance",
    "item_armor_knockback_resistances",
    "broken_item_resistance_penalty",
    "entity_effect_damage_resistance",
    "wielding_root_damage_modifiers",
    "wielding_root_knockback_modifiers",
    "wielding_stamina_cost",
    "angled_wielding_angle",
    "angled_wielding_nonzero_angle",
    "angled_wielding_damage_modifiers",
    "angled_wielding_knockback_modifiers",
    "hitbox_collision_soft_offset_ratio",
    "npc_role_use_projected_distance",
    "matchup_role_chain",
    "matchup_role_apply_avoidance",
    "matchup_role_apply_separation",
    "matchup_role_min_hit_slowdown",
    "matchup_role_armor_assignments",
    "matchup_role_repulsion_config_index",
    "matchup_role_hitbox_collision_config_index",
    "matchup_role_combat_action_evaluator",
    "matchup_role_predictability_range",
    "matchup_guard_stamina_cost",
    "matchup_guard_root_damage_modifiers",
    "matchup_guard_root_knockback_modifiers",
    "matchup_guard_angled_angle",
    "matchup_guard_angled_damage_modifiers",
    "matchup_guard_angled_knockback_modifiers",
    "matchup_damage_stamina_drain_multiplier",
)


@dataclass(frozen=True)
class AssetSurfaceOccurrence:
    """One authored value at a stable archive path and JSON pointer."""

    path: str
    pointer: str
    value: Any


@dataclass(frozen=True)
class AssetSurfaceReport:
    """Exact activations found in one pinned asset archive."""

    assets_sha256: str
    scanned_document_count: int
    scanned_paths_sha256: str
    surfaces: dict[str, tuple[AssetSurfaceOccurrence, ...]]

    def summary(self) -> dict[str, Any]:
        """Return the compact, deterministic form stored in the ruleset."""

        return {
            "schema": ASSET_SURFACE_AUDIT_SCHEMA,
            "hytale_version": "0.5.7",
            "assets_sha256": self.assets_sha256,
            "scan_prefixes": list(ASSET_SURFACE_SCAN_PREFIXES),
            "scanned_document_count": self.scanned_document_count,
            "scanned_paths_sha256": self.scanned_paths_sha256,
            "surfaces": {
                name: {
                    "count": len(self.surfaces[name]),
                    "occurrences_sha256": _canonical_sha256(
                        [asdict(item) for item in self.surfaces[name]]
                    ),
                }
                for name in _SURFACE_NAMES
            },
        }


def _configured_hytale_assets_archive() -> Path:
    """Resolve the explicit or standard installed ``Assets.zip`` path."""

    explicit = os.environ.get(HYTALE_ASSETS_ZIP_ENV)
    if explicit:
        return Path(explicit)
    app_data = os.environ.get("APPDATA")
    if not app_data:
        raise FileNotFoundError(f"{HYTALE_ASSETS_ZIP_ENV} and APPDATA are unavailable")
    return (
        Path(app_data)
        / "Hytale"
        / "install"
        / "release"
        / "package"
        / "game"
        / "latest"
        / "Assets.zip"
    )


def default_hytale_assets_archive() -> Path:
    """Return the pinned Hytale 0.5.7 ``Assets.zip`` archive.

    The public default is identity-safe because nearly every caller reads
    native mechanics immediately after resolving it.  Tests that intentionally
    exercise another archive pass that path explicitly to their compiler.
    """

    return require_hytale_0_5_7_assets_archive()


@lru_cache(maxsize=8)
def _cached_asset_archive_sha256(
    resolved_path: str,
    size: int,
    modified_ns: int,
    changed_ns: int,
) -> str:
    """Hash one immutable-on-disk archive identity.

    The file metadata values are deliberately part of the cache key.  A
    replaced archive at the same path must never inherit the identity of the
    previous file.
    """

    del size, modified_ns, changed_ns
    return file_sha256(Path(resolved_path)).upper()


def require_hytale_0_5_7_assets_archive(
    archive: str | Path | None = None,
) -> Path:
    """Return an archive only when it is the pinned Hytale 0.5.7 asset set.

    Asset-backed fidelity checks use this function before reading semantics so
    that an older, structurally valid archive fails as an identity mismatch
    rather than as a misleading mechanics regression.
    """

    path = _configured_hytale_assets_archive() if archive is None else Path(archive)
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Hytale Assets.zip does not exist: {path}") from exc
    resolved = path.resolve()
    actual_sha256 = _cached_asset_archive_sha256(
        str(resolved),
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    expected_sha256 = HYTALE_0_5_7_ASSETS_SHA256.upper()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Hytale 0.5.7 Assets.zip SHA-256 mismatch: "
            f"path={resolved}, expected={expected_sha256}, "
            f"actual={actual_sha256}"
        )
    return path


def load_asset_surface_manifest() -> dict[str, Any]:
    """Load the packaged 0.5.7 surface expectation."""

    resource = files("hytalegym.rulesets").joinpath(ASSET_SURFACE_AUDIT_RESOURCE)
    value = json.loads(resource.read_bytes())
    if value.get("schema") != ASSET_SURFACE_AUDIT_SCHEMA:
        raise ValueError("unsupported combat asset-surface manifest")
    if value.get("hytale_version") != "0.5.7":
        raise ValueError("asset-surface manifest is not for Hytale 0.5.7")
    if value.get("assets_sha256") != HYTALE_0_5_7_ASSETS_SHA256:
        raise ValueError("asset-surface manifest has the wrong asset identity")
    return value


def build_asset_surface_report(
    archive: str | Path,
    *,
    expected_sha256: str = HYTALE_0_5_7_ASSETS_SHA256,
    verify_sha256: bool = True,
) -> AssetSurfaceReport:
    """Enumerate the combat codec surfaces in one asset archive."""

    path = Path(archive)
    actual_sha256 = (
        file_sha256(path).upper() if verify_sha256 else expected_sha256.upper()
    )
    if actual_sha256 != expected_sha256.upper():
        raise ValueError(
            "Assets.zip SHA-256 mismatch: "
            f"expected {expected_sha256.upper()}, got {actual_sha256}"
        )

    with ZipFile(path) as source:
        selected_paths = tuple(
            sorted(
                info.filename
                for info in source.infolist()
                if info.filename.endswith(".json")
                and info.filename.startswith(ASSET_SURFACE_SCAN_PREFIXES)
            )
        )
        documents = {
            asset_path: _json_object(source.read(asset_path), asset_path)
            for asset_path in selected_paths
        }

    buckets: dict[str, list[AssetSurfaceOccurrence]] = {
        name: [] for name in _SURFACE_NAMES
    }
    wielding_ids = _wielding_asset_ids(documents)
    for asset_path in selected_paths:
        document = documents[asset_path]
        for pointer, node in _walk_objects(document):
            _scan_codec_node(
                buckets,
                asset_path,
                pointer,
                node,
                wielding_ids,
            )

    _scan_matchup_roles(buckets, documents)
    _copy_path_surface(
        buckets,
        "wielding_stamina_cost",
        "matchup_guard_stamina_cost",
        _MATCHUP_GUARD_PATH,
    )
    _copy_path_surface(
        buckets,
        "wielding_root_damage_modifiers",
        "matchup_guard_root_damage_modifiers",
        _MATCHUP_GUARD_PATH,
    )
    _copy_path_surface(
        buckets,
        "wielding_root_knockback_modifiers",
        "matchup_guard_root_knockback_modifiers",
        _MATCHUP_GUARD_PATH,
    )
    _copy_path_surface(
        buckets,
        "angled_wielding_angle",
        "matchup_guard_angled_angle",
        _MATCHUP_GUARD_PATH,
    )
    _copy_path_surface(
        buckets,
        "angled_wielding_damage_modifiers",
        "matchup_guard_angled_damage_modifiers",
        _MATCHUP_GUARD_PATH,
    )
    _copy_path_surface(
        buckets,
        "angled_wielding_knockback_modifiers",
        "matchup_guard_angled_knockback_modifiers",
        _MATCHUP_GUARD_PATH,
    )
    _copy_path_surface(
        buckets,
        "stamina_drain_multiplier",
        "matchup_damage_stamina_drain_multiplier",
        _MATCHUP_DAMAGE_PATH,
    )

    surfaces = {
        name: tuple(
            sorted(
                buckets[name],
                key=lambda item: (
                    item.path,
                    item.pointer,
                    _canonical_json(item.value),
                ),
            )
        )
        for name in _SURFACE_NAMES
    }
    return AssetSurfaceReport(
        assets_sha256=actual_sha256,
        scanned_document_count=len(selected_paths),
        scanned_paths_sha256=_canonical_sha256(selected_paths),
        surfaces=surfaces,
    )


def audit_hytale_0_5_7_asset_surface(
    archive: str | Path | None = None,
) -> AssetSurfaceReport:
    """Fail closed unless the shipped surface matches the packaged manifest."""

    manifest = load_asset_surface_manifest()
    report = build_asset_surface_report(
        _configured_hytale_assets_archive() if archive is None else archive
    )
    actual = report.summary()
    expected = {
        key: manifest[key]
        for key in (
            "schema",
            "hytale_version",
            "assets_sha256",
            "scan_prefixes",
            "scanned_document_count",
            "scanned_paths_sha256",
            "surfaces",
        )
    }
    if actual != expected:
        differences = _surface_differences(expected, actual)
        raise ValueError(
            "Hytale combat asset surface changed: " + "; ".join(differences)
        )
    return report


def _scan_codec_node(
    buckets: dict[str, list[AssetSurfaceOccurrence]],
    asset_path: str,
    pointer: str,
    node: dict[str, Any],
    wielding_ids: frozenset[str],
) -> None:
    base_damage = node.get("BaseDamage")
    if isinstance(base_damage, dict) and len(base_damage) > 1:
        _add(
            buckets,
            "damage_calculator_multi_cause",
            asset_path,
            pointer,
            "BaseDamage",
            base_damage,
        )

    if node.get("Type") == "DPS":
        _add(
            buckets,
            "damage_calculator_dps",
            asset_path,
            pointer,
            "Type",
            "DPS",
        )

    for key, surface in (
        ("SequentialModifierStep", "sequential_modifier_step"),
        ("SequentialModifierMinimum", "sequential_modifier_minimum"),
        ("StaminaDrainMultiplier", "stamina_drain_multiplier"),
    ):
        if key in node:
            _add(buckets, surface, asset_path, pointer, key, node[key])

    if asset_path.startswith(_ARMOR_ITEM_PREFIX):
        for key, surface in (
            ("DamageResistance", "item_armor_damage_resistance"),
            ("BaseDamageResistance", "item_armor_base_damage_resistance"),
            ("KnockbackResistances", "item_armor_knockback_resistances"),
        ):
            if key in node:
                _add(buckets, surface, asset_path, pointer, key, node[key])

    if asset_path.startswith(_ENTITY_EFFECT_PREFIX) and "DamageResistance" in node:
        _add(
            buckets,
            "entity_effect_damage_resistance",
            asset_path,
            pointer,
            "DamageResistance",
            node["DamageResistance"],
        )

    if (
        asset_path.startswith(_HITBOX_COLLISION_PREFIX)
        and "SoftCollisionOffsetRatio" in node
    ):
        _add(
            buckets,
            "hitbox_collision_soft_offset_ratio",
            asset_path,
            pointer,
            "SoftCollisionOffsetRatio",
            node["SoftCollisionOffsetRatio"],
        )

    if asset_path.startswith(_ROLE_PREFIX) and "UseProjectedDistance" in node:
        _add(
            buckets,
            "npc_role_use_projected_distance",
            asset_path,
            pointer,
            "UseProjectedDistance",
            node["UseProjectedDistance"],
        )

    broken_penalties = node.get("BrokenPenalties")
    if (
        asset_path.startswith(_GAMEPLAY_CONFIG_PREFIX)
        and isinstance(broken_penalties, dict)
        and "Weapon" in broken_penalties
    ):
        _add(
            buckets,
            "broken_item_resistance_penalty",
            asset_path,
            _pointer(pointer, "BrokenPenalties"),
            "Weapon",
            broken_penalties["Weapon"],
        )

    parent = node.get("Parent")
    is_wielding = node.get("Type") == "Wielding" or (
        isinstance(parent, str) and parent in wielding_ids
    )
    if not is_wielding:
        return
    for key, surface in (
        ("DamageModifiers", "wielding_root_damage_modifiers"),
        ("KnockbackModifiers", "wielding_root_knockback_modifiers"),
        ("StaminaCost", "wielding_stamina_cost"),
    ):
        if key in node:
            _add(buckets, surface, asset_path, pointer, key, node[key])

    angled = node.get("AngledWielding")
    if not isinstance(angled, dict):
        return
    angled_pointer = _pointer(pointer, "AngledWielding")
    if "Angle" in angled:
        _add(
            buckets,
            "angled_wielding_angle",
            asset_path,
            angled_pointer,
            "Angle",
            angled["Angle"],
        )
        angle = angled["Angle"]
        if not isinstance(angle, (int, float)) or float(angle) != 0.0:
            _add(
                buckets,
                "angled_wielding_nonzero_angle",
                asset_path,
                angled_pointer,
                "Angle",
                angle,
            )
    for key, surface in (
        ("DamageModifiers", "angled_wielding_damage_modifiers"),
        ("KnockbackModifiers", "angled_wielding_knockback_modifiers"),
    ):
        if key in angled:
            _add(
                buckets,
                surface,
                asset_path,
                angled_pointer,
                key,
                angled[key],
            )


def _scan_matchup_roles(
    buckets: dict[str, list[AssetSurfaceOccurrence]],
    documents: dict[str, dict[str, Any]],
) -> None:
    role_paths: dict[str, list[str]] = {}
    balance_paths: dict[str, list[str]] = {}
    for asset_path in documents:
        if asset_path.startswith(_ROLE_PREFIX):
            role_paths.setdefault(Path(asset_path).stem, []).append(asset_path)
        elif asset_path.startswith(_BALANCE_PREFIX):
            balance_paths.setdefault(Path(asset_path).stem, []).append(asset_path)

    for matchup_role_id in _MATCHUP_ROLE_IDS:
        pending = [matchup_role_id]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            paths = role_paths.get(current, [])
            if len(paths) != 1:
                raise ValueError(
                    f"expected one NPC role asset named {current}, got {paths}"
                )
            seen.add(current)
            asset_path = paths[0]
            document = documents[asset_path]
            references: set[str] = set()
            for pointer, node in _walk_objects(document):
                reference = node.get("Reference")
                if isinstance(reference, str):
                    references.add(reference)
                for key, surface in (
                    ("ApplyAvoidance", "matchup_role_apply_avoidance"),
                    ("ApplySeparation", "matchup_role_apply_separation"),
                ):
                    if key in node:
                        _add(
                            buckets,
                            surface,
                            asset_path,
                            pointer,
                            key,
                            node[key],
                        )
                for key, surface in (
                    (
                        "RepulsionConfigIndex",
                        "matchup_role_repulsion_config_index",
                    ),
                    (
                        "HitboxCollisionConfigIndex",
                        "matchup_role_hitbox_collision_config_index",
                    ),
                    (
                        "MinHitSlowdown",
                        "matchup_role_min_hit_slowdown",
                    ),
                ):
                    if key in node:
                        _add_matchup(
                            buckets,
                            surface,
                            asset_path,
                            pointer,
                            key,
                            matchup_role_id,
                            node[key],
                        )
                if node.get("Type") == "CombatActionEvaluator":
                    _add_matchup(
                        buckets,
                        "matchup_role_combat_action_evaluator",
                        asset_path,
                        pointer,
                        "Type",
                        matchup_role_id,
                        "CombatActionEvaluator",
                    )
                    _scan_matchup_predictability(
                        buckets,
                        matchup_role_id,
                        asset_path,
                        pointer,
                        node,
                    )
                for field_name in ("CombatConfig", "_CombatConfig"):
                    if field_name in node:
                        _scan_matchup_combat_config(
                            buckets,
                            documents,
                            balance_paths,
                            matchup_role_id,
                            asset_path,
                            pointer,
                            field_name,
                            node[field_name],
                        )
            for pointer, value in _role_armor_values(document):
                if _authored_nonempty(value):
                    buckets["matchup_role_armor_assignments"].append(
                        AssetSurfaceOccurrence(asset_path, pointer, value)
                    )
            buckets["matchup_role_chain"].append(
                AssetSurfaceOccurrence(
                    asset_path,
                    "",
                    {
                        "matchup_role_id": matchup_role_id,
                        "role_asset_id": current,
                        "references": sorted(references),
                    },
                )
            )
            pending.extend(
                reference
                for reference in sorted(references, reverse=True)
                if reference not in seen
            )


def _scan_matchup_combat_config(
    buckets: dict[str, list[AssetSurfaceOccurrence]],
    documents: dict[str, dict[str, Any]],
    balance_paths: dict[str, list[str]],
    matchup_role_id: str,
    role_path: str,
    role_pointer: str,
    field_name: str,
    value: Any,
) -> None:
    """Record a role's utility-combat attachment and authored predictability."""

    if isinstance(value, dict):
        # Directly typed inline evaluators are found by the ordinary object
        # walk. This branch covers the nested codec form without duplicating
        # that occurrence.
        evaluator = value.get("CombatActionEvaluator")
        if (
            value.get("Type") == "CombatActionEvaluator"
            or (
                isinstance(evaluator, dict)
                and evaluator.get("Type") == "CombatActionEvaluator"
            )
            or not isinstance(evaluator, dict)
        ):
            return
        buckets["matchup_role_combat_action_evaluator"].append(
            AssetSurfaceOccurrence(
                role_path,
                _pointer(role_pointer, field_name),
                {
                    "matchup_role_id": matchup_role_id,
                    "balance_asset_id": None,
                },
            )
        )
        _scan_matchup_predictability(
            buckets,
            matchup_role_id,
            role_path,
            _pointer(role_pointer, field_name),
            value,
        )
        return

    if not isinstance(value, str):
        return

    balance_chain: list[tuple[str, dict[str, Any]]] = []
    balance_asset_id: str | None = value
    seen: set[str] = set()
    while balance_asset_id is not None:
        if balance_asset_id in seen:
            raise ValueError(
                f"NPC balance asset parent cycle at {balance_asset_id}"
            )
        seen.add(balance_asset_id)
        paths = balance_paths.get(balance_asset_id, [])
        if len(paths) != 1:
            raise ValueError(
                "expected one NPC balance asset named "
                f"{balance_asset_id}, got {paths}"
            )
        config_path = paths[0]
        config = documents[config_path]
        balance_chain.append((config_path, config))
        parent = config.get("Parent")
        balance_asset_id = parent if isinstance(parent, str) else None

    if not any(
        config.get("Type") == "CombatActionEvaluator"
        or isinstance(config.get("CombatActionEvaluator"), dict)
        for _, config in balance_chain
    ):
        return
    buckets["matchup_role_combat_action_evaluator"].append(
        AssetSurfaceOccurrence(
            role_path,
            _pointer(role_pointer, field_name),
            {
                "matchup_role_id": matchup_role_id,
                "balance_asset_id": value,
            },
        )
    )
    for config_path, config in balance_chain:
        _scan_matchup_predictability(
            buckets,
            matchup_role_id,
            config_path,
            "",
            config,
        )


def _scan_matchup_predictability(
    buckets: dict[str, list[AssetSurfaceOccurrence]],
    matchup_role_id: str,
    asset_path: str,
    pointer: str,
    value: dict[str, Any],
) -> None:
    for nested_pointer, node in _walk_objects(value, pointer):
        if "PredictabilityRange" in node:
            _add_matchup(
                buckets,
                "matchup_role_predictability_range",
                asset_path,
                nested_pointer,
                "PredictabilityRange",
                matchup_role_id,
                node["PredictabilityRange"],
            )


def _role_armor_values(
    document: dict[str, Any],
) -> Iterable[tuple[str, Any]]:
    if "Armor" in document:
        yield "/Armor", document["Armor"]
    modify = document.get("Modify")
    if isinstance(modify, dict) and "Armor" in modify:
        yield "/Modify/Armor", modify["Armor"]
    parameters = document.get("Parameters")
    if isinstance(parameters, dict) and "Armor" in parameters:
        value = parameters["Armor"]
        if isinstance(value, dict) and "Value" in value:
            yield "/Parameters/Armor/Value", value["Value"]
        else:
            yield "/Parameters/Armor", value


def _wielding_asset_ids(
    documents: dict[str, dict[str, Any]],
) -> frozenset[str]:
    ids = {
        Path(asset_path).stem
        for asset_path, document in documents.items()
        if document.get("Type") == "Wielding"
    }
    changed = True
    while changed:
        changed = False
        for asset_path, document in documents.items():
            parent = document.get("Parent")
            asset_id = Path(asset_path).stem
            if isinstance(parent, str) and parent in ids and asset_id not in ids:
                ids.add(asset_id)
                changed = True
    return frozenset(ids)


def _walk_objects(
    value: Any,
    pointer: str = "",
) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        yield pointer, value
        for key, child in value.items():
            yield from _walk_objects(child, _pointer(pointer, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_objects(child, _pointer(pointer, str(index)))


def _add(
    buckets: dict[str, list[AssetSurfaceOccurrence]],
    surface: str,
    asset_path: str,
    pointer: str,
    key: str,
    value: Any,
) -> None:
    buckets[surface].append(
        AssetSurfaceOccurrence(
            asset_path,
            _pointer(pointer, key),
            value,
        )
    )


def _add_matchup(
    buckets: dict[str, list[AssetSurfaceOccurrence]],
    surface: str,
    asset_path: str,
    pointer: str,
    key: str,
    matchup_role_id: str,
    value: Any,
) -> None:
    buckets[surface].append(
        AssetSurfaceOccurrence(
            asset_path,
            _pointer(pointer, key),
            {
                "matchup_role_id": matchup_role_id,
                "authored_value": value,
            },
        )
    )


def _copy_path_surface(
    buckets: dict[str, list[AssetSurfaceOccurrence]],
    source: str,
    target: str,
    asset_path: str,
) -> None:
    buckets[target].extend(item for item in buckets[source] if item.path == asset_path)


def _pointer(parent: str, component: str) -> str:
    escaped = str(component).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}"


def _authored_nonempty(value: Any) -> bool:
    return value not in (None, False, "", (), [], {})


def _json_object(content: bytes, asset_path: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{asset_path} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{asset_path} must contain a JSON object")
    return value


def _surface_differences(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    differences: list[str] = []
    for key in (
        "assets_sha256",
        "scan_prefixes",
        "scanned_document_count",
        "scanned_paths_sha256",
    ):
        if expected.get(key) != actual.get(key):
            differences.append(
                f"{key} expected {expected.get(key)!r}, got {actual.get(key)!r}"
            )
    expected_surfaces = expected.get("surfaces", {})
    actual_surfaces = actual.get("surfaces", {})
    for name in sorted(set(expected_surfaces) | set(actual_surfaces)):
        if expected_surfaces.get(name) != actual_surfaces.get(name):
            differences.append(
                f"{name} expected {expected_surfaces.get(name)!r}, "
                f"got {actual_surfaces.get(name)!r}"
            )
    return differences or ["manifest metadata differs"]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest().upper()


__all__ = [
    "ASSET_SURFACE_AUDIT_RESOURCE",
    "ASSET_SURFACE_AUDIT_SCHEMA",
    "ASSET_SURFACE_SCAN_PREFIXES",
    "AssetSurfaceOccurrence",
    "AssetSurfaceReport",
    "HYTALE_ASSETS_ZIP_ENV",
    "audit_hytale_0_5_7_asset_surface",
    "build_asset_surface_report",
    "default_hytale_assets_archive",
    "load_asset_surface_manifest",
    "require_hytale_0_5_7_assets_archive",
]
