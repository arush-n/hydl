"""Typed, immutable views of content-table rows.

Each record keeps the asset ``source`` path. Fields the engine consumes are
typed; the rest of a row stays reachable through ``row`` (read-only mapping).
Engine defaults that the assets omit (block Material/Opacity/HitboxType/IsDoor,
the codec defaults the extractors already applied) are named here, once.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

# Block codec defaults (worldgen/surrogate/semantics.py::_BLOCK_DEFAULTS).
BLOCK_DEFAULTS = MappingProxyType(
    {"material": "Empty", "opacity": "Solid", "hitbox_type": "Full", "is_door": False}
)


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(child) for child in value)
    return value


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class MovementConfig:
    name: str
    source: str
    fields: Mapping[str, Any]

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "MovementConfig":
        return cls(key.split(":", 1)[1], row["source"], _frozen(dict(row["fields"])))

    def number(self, field: str) -> float:
        return float(self.fields[field])


@dataclass(frozen=True)
class RegenRule:
    interval: float
    amount: float
    regen_type: str | None
    clamp_at_zero: bool
    conditions: tuple[Mapping[str, Any], ...]

    def condition_ids(self) -> tuple[tuple[str, bool], ...]:
        return tuple((c.get("Id"), bool(c.get("Inverse", False))) for c in self.conditions)


@dataclass(frozen=True)
class StatDefinition:
    stat_id: str
    source: str
    initial_value: float | None
    minimum: float | None
    maximum: float | None
    regenerating: tuple[RegenRule, ...]
    row: Mapping[str, Any]

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "StatDefinition":
        rules = tuple(
            RegenRule(
                float(rule["interval"]),
                float(rule["amount"]),
                rule.get("regen_type"),
                bool(rule.get("clamp_at_zero")),
                _frozen(list(rule.get("conditions") or [])),
            )
            for rule in row.get("regenerating") or []
        )
        return cls(
            key.split(":", 1)[1],
            row["source"],
            _float(row.get("initial_value")),
            _float(row.get("min")),
            _float(row.get("max")),
            rules,
            _frozen(dict(row)),
        )


@dataclass(frozen=True)
class DurabilityEntry:
    item_id: str
    max_durability: float
    durability_loss_on_hit: float
    source: str

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "DurabilityEntry":
        return cls(
            row["item_asset_id"],
            float(row["max_durability"]),
            float(row["durability_loss_on_hit"]),
            row["source"],
        )


@dataclass(frozen=True)
class DamageEntry:
    variable: str
    origin: str
    base: Mapping[str, float] | None
    base_first: float | None
    angled: Mapping[str, float] | None
    angled_first: float | None


@dataclass(frozen=True)
class WeaponItem:
    item_id: str
    source: str
    parent_chain: tuple[str, ...]
    folder: str | None
    max_durability: float | None
    durability_loss_on_hit: float | None
    guard_stamina_value: float | None
    interactions: Mapping[str, str]
    interaction_vars: tuple[str, ...]
    damage: Mapping[str, DamageEntry]

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "WeaponItem":
        damage = {
            name: DamageEntry(
                name,
                entry["origin"],
                _frozen(entry.get("base")),
                _float(entry.get("base_first")),
                _frozen(entry.get("angled")),
                _float(entry.get("angled_first")),
            )
            for name, entry in row["damage"].items()
        }
        return cls(
            key.split(":", 1)[1],
            row["source"],
            tuple(row["parent_chain"]),
            row.get("folder"),
            _float(row.get("max_durability")),
            _float(row.get("durability_loss_on_hit")),
            _float(row.get("guard_stamina_value")),
            _frozen(dict(row.get("interactions") or {})),
            tuple(row["interaction_vars"]),
            MappingProxyType(damage),
        )


@dataclass(frozen=True)
class Knockback:
    type: str | None
    force: float | None
    direction: tuple[float, float, float] | None
    velocity_type: str | None
    row: Mapping[str, Any]


@dataclass(frozen=True)
class InteractionNode:
    """A named interaction/root/projectile-config document and its attack facts."""

    node_id: str
    kind: str
    source: str
    type: str | None
    parent_chain: tuple[str, ...]
    hit_interrupt: bool
    damage_class: str | None
    base_damage: Mapping[str, float] | None
    knockback: Knockback | None
    selector: Mapping[str, Any] | None
    charge_thresholds: tuple[float, ...]
    chain: Mapping[str, Any] | None
    document: Mapping[str, Any]

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "InteractionNode":
        knockback = row.get("knockback")
        return cls(
            key.split(":", 1)[1],
            row["kind"],
            row["source"],
            row.get("type"),
            tuple(row.get("parent_chain") or ()),
            bool(row.get("hit_interrupt", False)),
            row.get("damage_class"),
            _frozen(row.get("base_damage")),
            None
            if not knockback
            else Knockback(
                knockback.get("type"),
                _float(knockback.get("force")),
                None if knockback.get("direction") is None else tuple(knockback["direction"]),
                knockback.get("velocity_type"),
                _frozen(dict(knockback)),
            ),
            _frozen(row.get("selector")),
            tuple(row.get("charge_thresholds") or ()),
            _frozen(row.get("chain")),
            _frozen(row.get("document") or {}),
        )


@dataclass(frozen=True)
class Effect:
    effect_id: str
    source: str | None
    duration: float | None
    document: Mapping[str, Any]

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "Effect":
        return cls(key.split(":", 1)[1], row.get("source"), _float(row.get("duration")), _frozen(row.get("document") or {}))


@dataclass(frozen=True)
class ProjectileConfig:
    config_id: str
    source: str
    model: str | None
    launch_force: float | None
    gravity: float | None
    terminal_velocity_air: float | None
    spawn_offset: tuple[float, float, float] | None
    spawn_rotation_offset: Mapping[str, float] | None
    interaction_vars: tuple[str, ...]

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "ProjectileConfig":
        offset = row.get("spawn_offset")
        return cls(
            key.split(":", 1)[1],
            row["source"],
            row.get("model"),
            _float(row.get("launch_force")),
            _float(row.get("gravity")),
            _float(row.get("terminal_velocity_air")),
            None if offset is None else tuple(offset),
            _frozen(row.get("spawn_rotation_offset")),
            tuple(row.get("interaction_vars") or ()),
        )


@dataclass(frozen=True)
class LegacyProjectile:
    projectile_id: str
    source: str
    appearance: str | None
    muzzle_velocity: float | None
    terminal_velocity: float | None
    gravity: float | None
    damage: float | None
    dead_time: float | None
    time_to_live: float | None
    horizontal_center_shot: float | None
    vertical_center_shot: float | None
    depth_shot: float | None

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "LegacyProjectile":
        return cls(
            key.split(":", 1)[1],
            row["source"],
            row.get("appearance"),
            *(
                _float(row.get(name))
                for name in (
                    "muzzle_velocity",
                    "terminal_velocity",
                    "gravity",
                    "damage",
                    "dead_time",
                    "time_to_live",
                    "horizontal_center_shot",
                    "vertical_center_shot",
                    "depth_shot",
                )
            ),
        )


@dataclass(frozen=True)
class Model:
    model_id: str
    source: str
    eye_height: float | None
    min_scale: float | None
    max_scale: float | None
    hitbox_min: tuple[float, float, float] | None
    hitbox_max: tuple[float, float, float] | None

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "Model":
        box = row.get("hitbox") or {}
        return cls(
            key.split(":", 1)[1],
            row["source"],
            _float(row.get("eye_height")),
            _float(row.get("min_scale")),
            _float(row.get("max_scale")),
            tuple(box["min"]) if box.get("min") else None,
            tuple(box["max"]) if box.get("max") else None,
        )


@dataclass(frozen=True)
class Consumable:
    item_id: str
    source: str
    max_stack: int | None
    effects_applied: tuple[str, ...]
    effects_cleared: tuple[str, ...]
    stat_changes: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "Consumable":
        return cls(
            key.split(":", 1)[1],
            row["source"],
            row.get("max_stack"),
            tuple(row["effects_applied"]),
            tuple(row["effects_cleared"]),
            _frozen(list(row["stat_changes"])),
        )


@dataclass(frozen=True)
class BlockType:
    block_id: str
    source: str
    material: str
    opacity: str
    hitbox_type: str
    is_door: bool
    draw_type: str | None
    climbable: bool
    movement_settings: Mapping[str, Any]
    row: Mapping[str, Any]

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "BlockType":
        return cls(
            key.split(":", 1)[1],
            row["source"],
            row.get("material", BLOCK_DEFAULTS["material"]),
            row.get("opacity", BLOCK_DEFAULTS["opacity"]),
            row.get("hitbox_type", BLOCK_DEFAULTS["hitbox_type"]),
            bool(row.get("is_door", BLOCK_DEFAULTS["is_door"])),
            row.get("draw_type"),
            bool(row.get("climbable", False)),
            _frozen(dict(row.get("movement_settings") or {})),
            _frozen(dict(row)),
        )

    @property
    def blocks_sight_v0_5(self) -> bool:
        """0.5.x LOS rule: any non-``Transparent`` opacity stops sight."""

        return self.opacity != "Transparent"


@dataclass(frozen=True)
class Material:
    quantity: int
    item_id: str | None = None
    resource_type_id: str | None = None
    item_tag: str | None = None
    excluded_item_ids: tuple[str, ...] = ()

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> "Material":
        return cls(
            int(entry["quantity"]),
            entry.get("item_id"),
            entry.get("resource_type_id"),
            entry.get("item_tag"),
            tuple(entry.get("excluded_item_ids") or ()),
        )


@dataclass(frozen=True)
class BenchRequirement:
    type: str | None
    bench_id: str | None
    categories: tuple[str, ...]
    required_tier_level: int


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    source: str
    inputs: tuple[Material, ...]
    outputs: tuple[Material, ...]
    primary_output: Material | None
    bench_requirements: tuple[BenchRequirement, ...]
    time_seconds: float
    knowledge_required: bool
    required_memories_level: int
    generated_from_item: str | None

    @classmethod
    def from_row(cls, key: str, row: Mapping[str, Any]) -> "Recipe":
        primary = row.get("primary_output")
        return cls(
            key.split(":", 1)[1],
            row["source"],
            tuple(Material.from_entry(entry) for entry in row["inputs"]),
            tuple(Material.from_entry(entry) for entry in row["outputs"]),
            None if primary is None else Material.from_entry(primary),
            tuple(
                BenchRequirement(
                    entry.get("type"),
                    entry.get("id"),
                    tuple(entry.get("categories") or ()),
                    int(entry.get("required_tier_level", 0)),
                )
                for entry in row["bench_requirements"]
            ),
            float(row["time_seconds"]),
            bool(row["knowledge_required"]),
            int(row["required_memories_level"]),
            row.get("generated_from_item"),
        )


@dataclass(frozen=True)
class NpcRole:
    role_id: str
    source: str
    type: str | None
    reference_chain: tuple[str, ...]
    max_health: float | None
    knockback_scale: float | None
    appearance: str | None
    motion_controllers: tuple[Mapping[str, Any], ...]
    parameters: Mapping[str, Any]
    fields: Mapping[str, Any]

    def walk_controller(self) -> Mapping[str, Any] | None:
        for controller in self.motion_controllers:
            if controller.get("Type") == "Walk":
                return controller
        return None
