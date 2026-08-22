"""Deterministic, host-only inventory of Hytale 0.5.7 combat assets."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from hytalegym.combat.assets.weapon_variants import SCALAR_VARIANTS
from hytalegym.combat.assets.weapon_archetypes import (
    DOUBLE_INCANDESCENT_SPEAR_ASSET_ID,
    DOUBLE_INCANDESCENT_SPEAR_PROFILE_NAME,
    DEPLOYABLE_SOURCE_PATHS,
    DEPLOYABLE_SPECS,
    GUN_SPECS,
    KNIFE_WEAPON_VARIANTS,
    LEGACY_CASTER_VARIANTS,
    LEGACY_MELEE_VARIANTS,
    PROTOTYPE_BOW_SPECS,
    TRIBAL_BLOWGUN_ASSET_ID,
    TRIBAL_BLOWGUN_PROFILE_NAME,
    TRIBAL_CLAWS_ASSET_ID,
    TRIBAL_CLAWS_PROFILE_NAME,
    VOID_SCYTHE_ASSET_ID,
    VOID_SCYTHE_PROFILE_NAME,
)


COMBAT_ASSET_CATALOG_SCHEMA = "hytalerl_combat_asset_catalog_v1"
HYTALE_0_5_7_ASSETS_SHA256 = (
    "1B8802C284C228AE4549DAC037716C175BC6B94B0FC9AAC2B7039AC6BD2FFD5D"
)
HYTALE_0_5_7_COMBAT_CATALOG_SHA256 = (
    "E1F71C9164ABB156FC0519F2E960ADF1CD2506CB3024462E9BB2380178B9CB57"
)

_STANDARD_SPEAR_IDS = (
    "Adamantite",
    "Adamantite_Saurian",
    "Bone",
    "Bronze",
    "Cobalt",
    "Copper",
    "Crude",
    "Fishbone",
    "Iron",
    "Leaf",
    "Mithril",
    "Onyxium",
    "Scrap",
    "Stone_Trork",
    "Thorium",
    "Tribal",
)
COMPILED_PROFILE_SOURCE_ASSETS = {
    "iron_sword": ("Weapon_Sword_Iron",),
    "iron_mace": ("Weapon_Mace_Iron",),
    "iron_battleaxe": ("Weapon_Battleaxe_Iron",),
    "iron_daggers": ("Weapon_Daggers_Iron",),
    "iron_shortbow": ("Weapon_Shortbow_Iron",),
    "iron_crossbow": ("Weapon_Crossbow_Iron",),
    "iron_shield": ("Weapon_Shield_Iron",),
    "flame_staff": ("Weapon_Staff_Crystal_Flame",),
    "ice_staff": ("Weapon_Staff_Crystal_Ice",),
    "bombs": (
        "Weapon_Bomb",
        "Weapon_Bomb_Stun",
        "Weapon_Bomb_Popberry",
    ),
    "potions": ("Potion_Health", "Potion_Stamina", "Potion_Antidote"),
    "stoneskin_wand": ("Weapon_Wand_Stoneskin",),
    "root_wand": ("Weapon_Wand_Root",),
    "skeleton_mage_spellbook": ("Weapon_Spellbook_Grimoire_Purple",),
    "kunai": ("Weapon_Kunai",),
    **{
        f"{spear_id.lower()}_spear": (f"Weapon_Spear_{spear_id}",)
        for spear_id in _STANDARD_SPEAR_IDS
        if spear_id != "Adamantite_Saurian"
    },
    "adamantite_saurian_spear": ("Weapon_Spear_Adamantite_Saurian",),
    **{row.profile: (row.asset_id,) for row in SCALAR_VARIANTS},
    VOID_SCYTHE_PROFILE_NAME: (VOID_SCYTHE_ASSET_ID,),
    **{row.profile: (row.asset_id,) for row in LEGACY_MELEE_VARIANTS},
    **{row.profile: (row.asset_id,) for row in LEGACY_CASTER_VARIANTS},
    **{row.profile: (row.asset_id,) for row in KNIFE_WEAPON_VARIANTS},
    DOUBLE_INCANDESCENT_SPEAR_PROFILE_NAME: (DOUBLE_INCANDESCENT_SPEAR_ASSET_ID,),
    TRIBAL_CLAWS_PROFILE_NAME: (TRIBAL_CLAWS_ASSET_ID,),
    TRIBAL_BLOWGUN_PROFILE_NAME: (TRIBAL_BLOWGUN_ASSET_ID,),
    **{row.profile: (row.asset_id,) for row in PROTOTYPE_BOW_SPECS},
    **{row.profile: (row.asset_id,) for row in GUN_SPECS},
    **{row.profile: (row.asset_id,) for row in DEPLOYABLE_SPECS},
}
COMPILED_AUXILIARY_SOURCE_PATHS = {
    "kunai": (
        "Server/Item/Interactions/Weapons/Kunai/Kunai_Throw.json",
        "Server/Models/Projectiles/Weapons/Kunai/Kunai.json",
        "Server/ProjectileConfigs/Weapons/Throwables/Projectile_Config_Kunai.json",
    ),
    "root_wand": (
        "Server/Entity/Effects/Status/Root.json",
        "Server/Entity/Stats/Immunity.json",
        "Server/Item/Interactions/Weapons/Wand/Root_Cast.json",
    ),
    "skeleton_mage_spellbook": (
        "Server/Item/Interactions/NPCs/Undead/Skeleton_Sand_Mage/"
        "Skeleton_Sand_Mage_Spellbook_Corruption_Orb.json",
        "Server/Projectiles/NPCs/Undead/Skeleton_Mage/"
        "Skeleton_Mage_Corruption_Orb.json",
        "Server/Projectiles/Player/Staff/Staff_Wood_Rotten_Corruption_Orb.json",
    ),
    "standard_spears": tuple(
        path
        for spear_id in _STANDARD_SPEAR_IDS
        for path in (
            f"Server/Projectiles/Player/Spear/Spear_{spear_id}.json",
            f"Server/Models/Projectiles/Weapons/Spear/Spear_{spear_id}.json",
        )
    ),
    "prototype_bows": (
        "Server/ProjectileConfigs/Weapons/Arrows/Projectile_Config_Arrow_Base.json",
        "Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat.json",
        "Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Combat_Trishot.json",
        "Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Pull_Pullshot.json",
        "Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Arrow_Ricochet_Base.json",
        "Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Arrow_Ricochet.json",
        "Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Bomb_Boomshot.json",
        "Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp.json",
        "Server/ProjectileConfigs/Weapons/Bows/Prototype/Projectile_Config_Bow_Vamp_Supershot.json",
        "Server/Item/Interactions/Weapons/Bow/Prototype/CombatBow/Bow_Combat_Projectile_Damage.json",
        "Server/Item/Interactions/Weapons/Bow/Prototype/CombatBow/Bow_Combat_Projectile_Trishot_Damage.json",
        "Server/Item/Interactions/Weapons/Bow/Prototype/PullBow/Bow_Pull_Projectile_Pullshot_Damage.json",
        "Server/Item/Interactions/Weapons/Bow/Prototype/BombBow/Bow_Bomb_Boomshot_Explosion.json",
        "Server/Item/Interactions/Weapons/Bow/Prototype/VampBow/Bow_Vamp_Projectile_Damage.json",
        "Server/Item/Interactions/Weapons/Bow/Prototype/VampBow/Bow_Vamp_Projectile_Supershot_Damage.json",
        *(
            "Server/ProjectileConfigs/Weapons/Bows/Prototype/"
            f"Projectile_Config_Bow_Combat_Charge_{index:02d}.json"
            for index in range(1, 11)
        ),
        *(
            "Server/ProjectileConfigs/Weapons/Bows/Prototype/"
            f"Projectile_Config_Bow_Ricochet_Charge_{index:02d}.json"
            for index in range(1, 11)
        ),
        *(
            "Server/ProjectileConfigs/Weapons/Bows/Prototype/"
            f"Projectile_Config_Bow_Vamp_Charge_{index:02d}.json"
            for index in range(1, 11)
        ),
    ),
    "legacy_guns": (
        "Server/Projectiles/Hypixel/Minigames/GunPvP/GunPvP_Assault_Rifle_Bullet.json",
        "Server/Projectiles/Hypixel/Minigames/GunPvP/GunPvP_Handgun_Bullet.json",
        "Server/Projectiles/Player/Gun/Gun_Blunderbuss_Bullet.json",
    ),
    "legacy_fireball_staffs": (
        "Server/Projectiles/Spells/Fireball.json",
        "Server/Models/Projectiles/Spells/Fireball.json",
    ),
    "projectile_terminal_deployables": tuple(
        path
        for profile in sorted(DEPLOYABLE_SOURCE_PATHS)
        for path in DEPLOYABLE_SOURCE_PATHS[profile][1:]
    ),
}
COMPILED_SOURCE_IDS = frozenset(
    asset for assets in COMPILED_PROFILE_SOURCE_ASSETS.values() for asset in assets
)

WORLD_INTERACTION_TYPES = frozenset(
    {
        "BreakBlock",
        "ChangeBlock",
        "DestroyBlock",
        "HarvestCrop",
        "PlaceBlock",
        "PlaceFluid",
        "SpawnDeployableAtLocation",
        "SpawnDeployableFromRaycast",
        "SpawnEntity",
        "SpawnNPC",
        "UseBlock",
    }
)

_ITEM_PREFIX = "Server/Item/Items/"
_WEAPON_PREFIX = f"{_ITEM_PREFIX}Weapon/"
_ROOT_PREFIX = "Server/Item/RootInteractions/"
_INTERACTION_PREFIX = "Server/Item/Interactions/"
_GRAPH_REFERENCE_KEYS = frozenset(
    {"DefaultValue", "Failed", "Interactions", "Next", "Parent"}
)
_EFFECT_KEYS = frozenset({"ApplyEffects", "EffectId", "EntityEffectId"})


@dataclass(frozen=True)
class CombatAssetRecord:
    """One item plus the interaction graph reachable from its equipped roots."""

    asset_id: str
    path: str
    parent_chain: tuple[str, ...]
    root_interactions: tuple[str, ...]
    interaction_vars: tuple[str, ...]
    graph_ids: tuple[str, ...]
    interaction_types: tuple[str, ...]
    effect_ids: tuple[str, ...]
    unresolved_graph_ids: tuple[str, ...]
    world_interaction_types: tuple[str, ...]
    coverage: str
    source_semantic_sha256: str


@dataclass(frozen=True)
class CombatAssetCatalog:
    """Canonical evidence manifest; this object is never placed on device."""

    assets_sha256: str
    records: tuple[CombatAssetRecord, ...]
    auxiliary_source_semantic_sha256: tuple[tuple[str, str], ...] = ()

    def manifest(self) -> dict[str, Any]:
        records = [asdict(record) for record in self.records]
        coverage: dict[str, int] = {}
        for record in self.records:
            coverage[record.coverage] = coverage.get(record.coverage, 0) + 1
        return {
            "schema": COMBAT_ASSET_CATALOG_SCHEMA,
            "hytale_version": "0.5.7",
            "assets_sha256": self.assets_sha256,
            "record_count": len(records),
            "coverage_counts": dict(sorted(coverage.items())),
            "compiled_profile_sources": COMPILED_PROFILE_SOURCE_ASSETS,
            "compiled_auxiliary_sources": COMPILED_AUXILIARY_SOURCE_PATHS,
            "auxiliary_source_semantic_sha256": dict(
                self.auxiliary_source_semantic_sha256
            ),
            "records": records,
        }

    def semantic_sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def file_sha256(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    """Hash a local artifact without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_combat_asset_catalog(
    archive: str | Path,
    *,
    expected_sha256: str = HYTALE_0_5_7_ASSETS_SHA256,
    verify_sha256: bool = True,
) -> CombatAssetCatalog:
    """Audit all weapon items and the exact source items used by the gym."""

    path = Path(archive)
    actual_sha256 = file_sha256(path) if verify_sha256 else expected_sha256
    if actual_sha256.upper() != expected_sha256.upper():
        raise ValueError(
            "Assets.zip SHA-256 mismatch: "
            f"expected {expected_sha256.upper()}, got {actual_sha256.upper()}"
        )

    with ZipFile(path) as source:
        names = source.namelist()
        item_paths = _index_paths(names, _ITEM_PREFIX)
        roots = _index_paths(names, _ROOT_PREFIX)
        interactions = _index_paths(names, _INTERACTION_PREFIX)
        documents: dict[str, dict[str, Any]] = {}

        def read(asset_path: str) -> dict[str, Any]:
            if asset_path not in documents:
                value = json.loads(source.read(asset_path))
                if not isinstance(value, dict):
                    raise ValueError(f"{asset_path} must contain a JSON object")
                documents[asset_path] = value
            return documents[asset_path]

        missing = sorted(COMPILED_SOURCE_IDS - item_paths.keys())
        if missing:
            raise ValueError(f"compiled source assets are missing: {missing}")
        auxiliary_paths = tuple(
            sorted(
                path
                for paths in COMPILED_AUXILIARY_SOURCE_PATHS.values()
                for path in paths
            )
        )
        missing_auxiliary = sorted(set(auxiliary_paths) - set(names))
        if missing_auxiliary:
            raise ValueError(
                f"compiled auxiliary sources are missing: {missing_auxiliary}"
            )

        semantic_weapon_ids = {
            asset_id
            for asset_id in item_paths
            if not asset_id.startswith("Template_")
            and _is_semantic_weapon(_resolve_item(asset_id, item_paths, read)[0])
        }
        selected = (
            {
                asset_id
                for asset_id, asset_path in item_paths.items()
                if asset_path.startswith(_WEAPON_PREFIX)
                and not asset_id.startswith("Template_")
            }
            | semantic_weapon_ids
            | COMPILED_SOURCE_IDS
        )
        records = tuple(
            _record(
                asset_id,
                item_paths,
                roots,
                interactions,
                read,
            )
            for asset_id in sorted(selected)
        )
        auxiliary_sources = tuple(
            (source_path, _canonical_sha256(read(source_path)))
            for source_path in auxiliary_paths
        )
    catalog = CombatAssetCatalog(
        actual_sha256.upper(),
        records,
        auxiliary_sources,
    )
    catalog_sha256 = catalog.semantic_sha256()
    if (
        actual_sha256.upper() == HYTALE_0_5_7_ASSETS_SHA256
        and catalog_sha256 != HYTALE_0_5_7_COMBAT_CATALOG_SHA256
    ):
        raise ValueError(
            "pinned combat asset catalog identity drift: "
            f"expected {HYTALE_0_5_7_COMBAT_CATALOG_SHA256}, "
            f"got {catalog_sha256}"
        )
    return catalog


def _record(asset_id, items, roots, interactions, read) -> CombatAssetRecord:
    resolved, parents = _resolve_item(asset_id, items, read)
    variables = resolved.get("InteractionVars", {})
    if not isinstance(variables, dict):
        raise ValueError(f"{asset_id}.InteractionVars must be an object")
    visitor = _GraphVisitor(roots, interactions, read, variables)
    equipped = resolved.get("Interactions", {})
    visitor.visit(
        equipped,
        "Interactions",
        prefer_root=True,
        allow_graph=True,
    )
    root_ids = tuple(sorted(_known_ids(equipped, roots)))
    graph_bundle = {
        "item": resolved,
        "graphs": {
            graph_id: visitor.graph_documents[graph_id]
            for graph_id in sorted(visitor.graph_documents)
        },
    }
    world_types = tuple(sorted(visitor.types & WORLD_INTERACTION_TYPES))
    coverage = (
        "compiled_profile_source"
        if asset_id in COMPILED_SOURCE_IDS
        else (
            "world_or_entity_graph_not_compiled"
            if world_types
            else "combat_graph_not_compiled"
        )
    )
    return CombatAssetRecord(
        asset_id=asset_id,
        path=items[asset_id],
        parent_chain=parents,
        root_interactions=root_ids,
        interaction_vars=tuple(sorted(variables)),
        graph_ids=tuple(sorted(visitor.graph_documents)),
        interaction_types=tuple(sorted(visitor.types)),
        effect_ids=tuple(sorted(visitor.effect_ids)),
        unresolved_graph_ids=tuple(sorted(visitor.unresolved)),
        world_interaction_types=world_types,
        coverage=coverage,
        source_semantic_sha256=_canonical_sha256(graph_bundle),
    )


class _GraphVisitor:
    def __init__(self, roots, interactions, read, variables):
        self.roots = roots
        self.interactions = interactions
        self.read = read
        self.variables = variables
        self.graph_documents: dict[str, dict[str, Any]] = {}
        self.types: set[str] = set()
        self.effect_ids: set[str] = set()
        self.unresolved: set[str] = set()

    def visit(
        self,
        value: Any,
        key: str = "",
        *,
        prefer_root: bool = False,
        allow_graph: bool = False,
    ) -> None:
        if isinstance(value, str):
            if not allow_graph:
                return
            namespace, path = (
                ("root", self.roots.get(value))
                if prefer_root
                else ("interaction", self.interactions.get(value))
            )
            if path is None:
                namespace, path = (
                    ("interaction", self.interactions.get(value))
                    if prefer_root
                    else ("root", self.roots.get(value))
                )
            if path:
                self._visit_named(namespace, value, path)
            elif key in _GRAPH_REFERENCE_KEYS:
                self.unresolved.add(value)
            return
        if isinstance(value, list):
            for item in value:
                self.visit(
                    item,
                    key,
                    prefer_root=prefer_root,
                    allow_graph=allow_graph,
                )
            return
        if not isinstance(value, dict):
            return

        interaction_type = value.get("Type")
        if isinstance(interaction_type, str):
            self.types.add(interaction_type)
        for effect_key in _EFFECT_KEYS:
            _collect_strings(value.get(effect_key), self.effect_ids)

        parent = value.get("Parent")
        if isinstance(parent, str):
            self.visit(parent, "Parent", allow_graph=True)
        replace = interaction_type == "Replace"
        replacement_key = value.get("Var")
        if replace and isinstance(replacement_key, str):
            replacement = self.variables.get(replacement_key, value.get("DefaultValue"))
            self.visit(
                replacement,
                "DefaultValue",
                allow_graph=True,
            )

        reference_container = allow_graph and interaction_type is None
        for child_key, child in value.items():
            if child_key == "Parent":
                continue
            if replace and child_key in {"DefaultValue", "Var"}:
                continue
            self.visit(
                child,
                child_key,
                prefer_root=prefer_root,
                allow_graph=(child_key in _GRAPH_REFERENCE_KEYS or reference_container),
            )

    def _visit_named(self, namespace: str, graph_id: str, path: str) -> None:
        identity = f"{namespace}:{graph_id}"
        if identity in self.graph_documents:
            return
        document = self.read(path)
        self.graph_documents[identity] = document
        self.visit(document)


def _index_paths(names: list[str], prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in names:
        if not path.startswith(prefix) or not path.endswith(".json"):
            continue
        asset_id = path.rsplit("/", 1)[-1][:-5]
        previous = result.setdefault(asset_id, path)
        if previous != path:
            raise ValueError(f"duplicate asset id {asset_id}: {previous}, {path}")
    return result


def _resolve_item(asset_id, items, read, active=()):
    if asset_id in active:
        raise ValueError(f"item parent cycle: {' -> '.join((*active, asset_id))}")
    child = read(items[asset_id])
    parent_id = child.get("Parent")
    if not isinstance(parent_id, str):
        return copy.deepcopy(child), ()
    if parent_id not in items:
        raise ValueError(f"{asset_id} has unknown item parent {parent_id}")
    parent, chain = _resolve_item(parent_id, items, read, (*active, asset_id))
    return _merge(parent, child), (*chain, parent_id)


def _merge(parent: Any, child: Any) -> Any:
    if not isinstance(parent, dict) or not isinstance(child, dict):
        return copy.deepcopy(child)
    result = copy.deepcopy(parent)
    for key, value in child.items():
        result[key] = (
            _merge(result[key], value) if key in result else copy.deepcopy(value)
        )
    return result


def _known_ids(value, index) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        if value in index:
            found.add(value)
    elif isinstance(value, list):
        for item in value:
            found.update(_known_ids(item, index))
    elif isinstance(value, dict):
        for item in value.values():
            found.update(_known_ids(item, index))
    return found


def _is_semantic_weapon(item: dict[str, Any]) -> bool:
    """Recognize weapon items even when legacy content lives off-path."""

    categories = item.get("Categories", ())
    tags = item.get("Tags", {})
    tag_types = tags.get("Type", ()) if isinstance(tags, dict) else ()
    return (isinstance(categories, list) and "Items.Weapons" in categories) or (
        isinstance(tag_types, list) and "Weapon" in tag_types
    )


def _collect_strings(value: Any, target: set[str]) -> None:
    if isinstance(value, str):
        target.add(value)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, target)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()
