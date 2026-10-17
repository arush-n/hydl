"""``load(profile) -> ContentTables``: validated, typed, overlay-aware tables.

Raw rows are exactly what the extractors wrote. A generation may carry a
calibration overlay (gen 5: ``calibration_057.json``): hand-maintained values
the engine keeps although the tables say otherwise, each with its
classification and evidence. Typed accessors return overlay-applied values;
``row()`` returns the raw extraction. ``content_sha256`` identifies the
effective content (tables plus overlay) for the contract stamp.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from hytalegym.sim.common.content.records import (
    BlockType,
    Consumable,
    DurabilityEntry,
    Effect,
    InteractionNode,
    LegacyProjectile,
    Model,
    MovementConfig,
    NpcRole,
    ProjectileConfig,
    Recipe,
    StatDefinition,
    WeaponItem,
    _frozen,
)
from hytalegym.sim.common.content.registry import Selector, sim_profile
from hytalegym.sim.common.content.schema import (
    FAMILIES,
    canonical_sha256,
    validate_manifest,
    validate_overlay,
    validate_table,
)
from hytalegym.sim.common.profile import SimProfile

MANIFEST_NAME = "MANIFEST.json"


class ContentError(ValueError):
    """A data directory or overlay failed validation."""


@dataclass(frozen=True)
class Table:
    family: str
    game_version: str
    assets_sha256: str
    extractor_version: int
    rows_sha256: str
    source_count: int
    sources_sha256: str
    rows: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Override:
    family: str
    row: str
    path: tuple[str, ...]
    value: Any
    extracted: Any
    classification: str
    evidence: str
    consumer: str
    constants: tuple[str, ...] = ()


def _set_path(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node: Any = target
    for part in path[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    last = path[-1]
    if isinstance(node, list):
        node[int(last)] = copy.deepcopy(value)
    else:
        node[last] = copy.deepcopy(value)


class ContentTables:
    """All content tables of one generation."""

    def __init__(
        self,
        profile: SimProfile,
        manifest: Mapping[str, Any],
        tables: Mapping[str, Table],
        overrides: tuple[Override, ...],
    ):
        self.profile = profile
        self.manifest = manifest
        self.tables = tables
        self.overrides = overrides
        self._overridden: dict[tuple[str, str], tuple[Override, ...]] = {}
        for override in overrides:
            self._overridden.setdefault((override.family, override.row), ())
            self._overridden[(override.family, override.row)] += (override,)
        self.content_sha256 = (
            manifest["content_sha256"]
            if not overrides
            else canonical_sha256(
                {
                    "tables": manifest["content_sha256"],
                    "overrides": [
                        [o.family, o.row, list(o.path), o.value] for o in overrides
                    ],
                }
            )
        )

    @property
    def generation(self) -> str:
        return self.profile.generation

    @property
    def game_version(self) -> str:
        return self.manifest["game_version"]

    @property
    def extracted_content_sha256(self) -> str:
        return self.manifest["content_sha256"]

    def table(self, family: str) -> Table:
        return self.tables[family]

    def row(self, family: str, key: str) -> Mapping[str, Any]:
        """The raw extracted row (no overlay)."""

        return self.tables[family].rows[key]

    def effective_row(self, family: str, key: str) -> dict[str, Any]:
        """The row with this generation's calibration overlay applied."""

        row = copy.deepcopy(dict(self.tables[family].rows[key]))
        for override in self._overridden.get((family, key), ()):
            _set_path(row, override.path, override.value)
        return row

    def keys(self, family: str, kind: str) -> tuple[str, ...]:
        prefix = f"{kind}:"
        return tuple(key for key in self.tables[family].rows if key.startswith(prefix))

    # -- typed accessors (overlay applied) ------------------------------
    def movement_config(self, name: str | None = None) -> MovementConfig:
        if name is None:
            name = self.effective_row("movement", "binding:player")["movement_config"]
        key = f"config:{name}"
        return MovementConfig.from_row(key, self.effective_row("movement", key))

    def stat(self, stat_id: str) -> StatDefinition:
        key = f"stat:{stat_id}"
        return StatDefinition.from_row(key, self.effective_row("stats", key))

    def stamina_plugin(self) -> Mapping[str, Any]:
        return _frozen(self.effective_row("stats", "plugin:stamina"))

    def durability(self, item_id: str) -> DurabilityEntry | None:
        key = f"item:{item_id}"
        if key not in self.tables["durability"].rows:
            return None
        return DurabilityEntry.from_row(key, self.effective_row("durability", key))

    def weapon_item(self, item_id: str) -> WeaponItem:
        key = f"item:{item_id}"
        return WeaponItem.from_row(key, self.effective_row("weapons", key))

    def interaction(self, node_id: str, kind: str = "interaction") -> InteractionNode:
        key = f"{kind}:{node_id}"
        return InteractionNode.from_row(key, self.effective_row("weapons", key))

    def npc_interaction(self, node_id: str) -> InteractionNode:
        key = f"npc_interaction:{node_id}"
        return InteractionNode.from_row(key, self.effective_row("npc_roles", key))

    def effect(self, effect_id: str) -> Effect:
        key = f"effect:{effect_id}"
        family = "weapons" if key in self.tables["weapons"].rows else "consumables"
        return Effect.from_row(key, self.effective_row(family, key))

    def projectile_config(self, config_id: str) -> ProjectileConfig:
        key = f"config:{config_id}"
        return ProjectileConfig.from_row(key, self.effective_row("projectiles", key))

    def legacy_projectile(self, projectile_id: str) -> LegacyProjectile:
        key = f"projectile:{projectile_id}"
        return LegacyProjectile.from_row(key, self.effective_row("projectiles", key))

    def model(self, model_id: str) -> Model:
        key = f"model:{model_id}"
        family = "projectiles" if key in self.tables["projectiles"].rows else "npc_roles"
        return Model.from_row(key, self.effective_row(family, key))

    def consumable(self, item_id: str) -> Consumable:
        key = f"item:{item_id}"
        return Consumable.from_row(key, self.effective_row("consumables", key))

    def block(self, block_id: str) -> BlockType:
        key = f"block:{block_id}"
        return BlockType.from_row(key, self.effective_row("blocks", key))

    def recipe(self, recipe_id: str) -> Recipe:
        key = f"recipe:{recipe_id}"
        return Recipe.from_row(key, self.effective_row("crafting", key))

    def recipes(self) -> tuple[Recipe, ...]:
        return tuple(self.recipe(key.split(":", 1)[1]) for key in self.keys("crafting", "recipe"))

    def role_parameters(self, role_id: str) -> dict[str, Any]:
        """Full parameters: the reference chain's, then this role's delta."""

        row = self.effective_row("npc_roles", f"role:{role_id}")
        if row["parameters_complete"]:
            return copy.deepcopy(row["parameters"])
        parameters = self.role_parameters(row["reference_chain"][-1])
        parameters.update(copy.deepcopy(row["parameters"]))
        return parameters

    def role(self, role_id: str) -> NpcRole:
        row = self.effective_row("npc_roles", f"role:{role_id}")
        return NpcRole(
            role_id,
            row["source"],
            row.get("type"),
            tuple(row.get("reference_chain") or ()),
            None if row.get("max_health") is None else float(row["max_health"]),
            None if row.get("knockback_scale") is None else float(row["knockback_scale"]),
            row.get("appearance"),
            _frozen(list(row.get("motion_controllers") or [])),
            _frozen(self.role_parameters(role_id)),
            _frozen(dict(row.get("fields") or {})),
        )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _load(data_dir: str, overlay: str | None, generation_key: str, validate: bool) -> tuple[Mapping[str, Any], Mapping[str, Table], tuple[Override, ...]]:
    directory = Path(data_dir)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ContentError(f"content manifest missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    raw_tables = {}
    for family in FAMILIES:
        path = directory / f"{family}.json"
        if not path.is_file():
            raise ContentError(f"content table missing: {path}")
        raw_tables[family] = _read_json(path)
    if validate:
        errors = [f"{family}: {error}" for family, table in raw_tables.items() for error in validate_table(table, family=family)]
        errors += validate_manifest(manifest, raw_tables)
        if errors:
            raise ContentError(f"{generation_key} content invalid: " + "; ".join(errors[:20]))
    tables = {
        family: Table(
            family,
            table["game_version"],
            table["assets_sha256"],
            table["extractor_version"],
            table["rows_sha256"],
            table["source_count"],
            table["sources_sha256"],
            table["rows"],
        )
        for family, table in raw_tables.items()
    }
    overrides: tuple[Override, ...] = ()
    if overlay is not None:
        document = _read_json(Path(overlay))
        errors = validate_overlay(document)
        if errors:
            raise ContentError(f"{generation_key} overlay invalid: " + "; ".join(errors))
        overrides = tuple(
            Override(
                entry["family"],
                entry["row"],
                tuple(str(part) for part in entry["path"]),
                entry["value"],
                entry["extracted"],
                entry["classification"],
                entry["evidence"],
                entry["consumer"],
                tuple(entry.get("constants", ())),
            )
            for entry in document.get("overrides", [])
        )
        for override in overrides:
            if override.row not in tables[override.family].rows:
                raise ContentError(f"overlay names an absent row {override.family}/{override.row}")
    return manifest, tables, overrides


def load(selector: Selector = None, *, validate: bool = True, apply_overlay: bool = True) -> ContentTables:
    """Load (and by default validate) one generation's content tables.

    ``apply_overlay=False`` returns the raw extraction (used by the A1 gate to
    list exactly what the overlay changes).
    """

    profile = sim_profile(selector)
    overlay = profile.calibration_overlay
    manifest, tables, overrides = _load(
        str(profile.data_dir),
        str(overlay) if overlay is not None and Path(overlay).is_file() else None,
        profile.generation,
        validate,
    )
    return ContentTables(profile, manifest, tables, overrides if apply_overlay else ())


def content_sha256(selector: Selector = None) -> str:
    """The effective content identity of a generation (for the contract stamp)."""

    return load(selector).content_sha256
