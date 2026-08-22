"""Installed falling-block impact dispatch for Hytale 0.5.7."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

from hytalegym.worldgen.block_actions import block_semantic_key


FALLING_BLOCK_IMPACT_SCHEMA = "hytalerl_falling_block_impact_v1"
FALLING_BLOCK_IMPACT_VERSION = 1

FALLING_BLOCK_IMPACT_NONE = 0
FALLING_BLOCK_IMPACT_PLACE = 1
FALLING_BLOCK_IMPACT_BREAK = 2
FALLING_BLOCK_IMPACT_EXPLODE = 3
FALLING_BLOCK_IMPACT_TYPES = ("none", "place", "break", "explode")
NATIVE_FALLING_BLOCK_TRACE_SCHEMA = (
    "hytalerl_native_falling_block_trace_v2"
)
NATIVE_FALLING_BLOCK_TRACE_VERSION = 2
NATIVE_FALLING_BLOCK_TRACE_STAGES = (
    "spawn_before_first_tick",
    "after_block_entity_physics_before_falling_gravity",
    "ground_impact_after_falling_ticker",
)

_IMPACT_BY_NAME = {
    "Place": FALLING_BLOCK_IMPACT_PLACE,
    "Break": FALLING_BLOCK_IMPACT_BREAK,
    "Explode": FALLING_BLOCK_IMPACT_EXPLODE,
}


@dataclass(frozen=True)
class LocalFallingBlockImpact:
    """One inherited block asset's native ground-impact dispatch."""

    semantic_key: tuple[int, ...]
    impact_type: int
    hitbox_collision_config_id: str | None
    explosion_configured: bool


@dataclass(frozen=True, slots=True)
class NativeFallingBlockTrace:
    """One bridge-attributed dry falling-block motion sample."""

    bridge_sha256: str
    fixture_available: bool
    sample_available: bool
    stage: str
    block_asset_id: str
    runtime_block_id: int
    impact_type: str
    sample_index: int
    world_tick: int
    delta_seconds: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    on_ground: bool
    entity_present: bool
    impact_observed: bool
    impact_cell: tuple[int, int, int]
    mass: float
    drag_coefficient: float
    inverted_gravity: bool
    box_width: float
    box_depth: float
    dry_static_fixture: bool
    public_cascade_certified: bool

    @classmethod
    def from_info(
        cls,
        info: Mapping[str, object],
        *,
        expected_bridge_sha256: str | None = None,
    ) -> NativeFallingBlockTrace:
        version = info.get("native_falling_block_trace_version")
        if (
            info.get("native_falling_block_trace_schema")
            != NATIVE_FALLING_BLOCK_TRACE_SCHEMA
            or isinstance(version, bool)
            or version != NATIVE_FALLING_BLOCK_TRACE_VERSION
        ):
            raise ValueError("native falling-block trace schema changed")
        contract_sha256 = _sha256(
            info.get("native_falling_block_trace_contract_sha256"),
            "native_falling_block_trace_contract_sha256",
        )
        if contract_sha256.lower() != (
            native_falling_block_trace_contract_sha256()
        ):
            raise ValueError("native falling-block trace contract changed")
        bridge = _sha256(info.get("bridge_sha256"), "bridge_sha256")
        if (
            expected_bridge_sha256 is not None
            and bridge.lower()
            != _sha256(
                expected_bridge_sha256,
                "expected_bridge_sha256",
            ).lower()
        ):
            raise ValueError("falling-block trace came from another bridge")
        result = cls(
            bridge_sha256=bridge,
            fixture_available=_boolean(
                info,
                "native_falling_block_fixture_available",
            ),
            sample_available=_boolean(
                info,
                "native_falling_block_sample_available",
            ),
            stage=_string(info, "native_falling_block_stage"),
            block_asset_id=_string(
                info,
                "native_falling_block_asset_id",
            ),
            runtime_block_id=_integer(
                info,
                "native_falling_block_runtime_id",
                minimum=0,
            ),
            impact_type=_string(
                info,
                "native_falling_block_impact_type",
            ),
            sample_index=_integer(
                info,
                "native_falling_block_sample_index",
                minimum=-1,
            ),
            world_tick=_integer(
                info,
                "native_falling_block_world_tick",
                minimum=-1,
            ),
            delta_seconds=_real(
                info,
                "native_falling_block_delta_seconds",
                minimum=0.0,
            ),
            position=tuple(
                _real(info, f"native_falling_block_position_{axis}")
                for axis in "xyz"
            ),
            velocity=tuple(
                _real(info, f"native_falling_block_velocity_{axis}")
                for axis in "xyz"
            ),
            on_ground=_boolean(
                info,
                "native_falling_block_on_ground",
            ),
            entity_present=_boolean(
                info,
                "native_falling_block_entity_present",
            ),
            impact_observed=_boolean(
                info,
                "native_falling_block_impact_observed",
            ),
            impact_cell=tuple(
                _integer(info, f"native_falling_block_impact_{axis}")
                for axis in "xyz"
            ),
            mass=_real(
                info,
                "native_falling_block_mass",
                minimum=0.0,
            ),
            drag_coefficient=_real(
                info,
                "native_falling_block_drag_coefficient",
                minimum=0.0,
            ),
            inverted_gravity=_boolean(
                info,
                "native_falling_block_inverted_gravity",
            ),
            box_width=_real(
                info,
                "native_falling_block_box_width",
                minimum=0.0,
            ),
            box_depth=_real(
                info,
                "native_falling_block_box_depth",
                minimum=0.0,
            ),
            dry_static_fixture=_boolean(
                info,
                "native_falling_block_dry_static_fixture",
            ),
            public_cascade_certified=_boolean(
                info,
                "native_falling_block_public_cascade_certified",
            ),
        )
        _validate_native_falling_block_trace(result)
        return result


def resolve_local_falling_block_impact(
    asset_id: str,
    block: Mapping[str, object],
    *,
    rotation_index: int = 0,
) -> LocalFallingBlockImpact | None:
    """Resolve native default-Place and explicit impact dispatch."""

    if not isinstance(block, Mapping):
        raise TypeError("block must be a mapping")
    settings_value = block.get("FallingBlockSettings")
    if settings_value is None:
        return None
    settings = _mapping(settings_value, "FallingBlockSettings")
    impact_value = settings.get("Impact")
    if impact_value is None:
        impact_type = FALLING_BLOCK_IMPACT_PLACE
    else:
        impact = _mapping(impact_value, "FallingBlockSettings.Impact")
        name = impact.get("Type")
        if not isinstance(name, str) or name not in _IMPACT_BY_NAME:
            raise ValueError("falling-block impact Type is unknown")
        impact_type = _IMPACT_BY_NAME[name]

    collision = settings.get("HitboxCollisionConfig")
    if collision is not None and (
        not isinstance(collision, str) or not collision
    ):
        raise ValueError("HitboxCollisionConfig must be a non-empty string")
    explosion = block.get("ExplosionConfig")
    if explosion is not None and not isinstance(explosion, Mapping):
        raise ValueError("ExplosionConfig must be a mapping or null")
    return LocalFallingBlockImpact(
        semantic_key=block_semantic_key(asset_id, rotation_index),
        impact_type=impact_type,
        hitbox_collision_config_id=collision,
        explosion_configured=explosion is not None,
    )


def falling_block_impact_contract() -> dict[str, object]:
    """Return the source-derived dispatch boundary."""

    return {
        "schema": FALLING_BLOCK_IMPACT_SCHEMA,
        "version": FALLING_BLOCK_IMPACT_VERSION,
        "server_version": "0.5.7",
        "source": {
            "spawn": "FallingBlock.fallBlock_and_generateFallingBlock",
            "ground_dispatch": "FallingBlockTickingSystem.tick",
            "default": "missing_impact_dispatches_Place",
            "place": "PlaceFallingBlockImpact_floor_position_then_place",
            "place_failure": "BreakFallingBlockImpact",
            "break": "BreakFallingBlockImpact",
            "explode": (
                "ExplodeFallingBlockImpact_only_when_ExplosionConfig_exists"
            ),
        },
        "identity": "asset_and_rotation_semantic_sha256_words",
        "outputs": [
            "impact_type",
            "hitbox_collision_config_id",
            "explosion_configured",
        ],
        "scope": "impact_dispatch_not_falling_entity_physics_or_effect_kernel",
        "unsupported": [
            "gravity_and_collision_trajectory",
            "place_mutation_acknowledgement",
            "break_drop_spawning",
            "explosion_terrain_entity_and_knockback_effects",
        ],
    }


def falling_block_impact_contract_sha256() -> str:
    payload = json.dumps(
        falling_block_impact_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def native_falling_block_trace_contract() -> dict[str, object]:
    """Return the native dry-motion/ground-impact evidence contract."""

    return {
        "schema": NATIVE_FALLING_BLOCK_TRACE_SCHEMA,
        "version": NATIVE_FALLING_BLOCK_TRACE_VERSION,
        "server_version": "0.5.7",
        "fixture": {
            "world": "flat_static_dry",
            "block_asset": "runtime_bound_installed_falling_block",
            "spawn": "FallingBlock.generateFallingBlock",
        },
        "sample_stage": (
            "after_BlockEntitySystems_Ticking_before_"
            "FallingBlockTickingSystem_applyGravity"
        ),
        "sequence": {
            "sample_index": "consecutive_fixture_system_invocations",
            "world_tick": (
                "strictly_monotonic_scheduler_diagnostic_not_sample_ordinal"
            ),
            "admission": (
                "sample_index_delta_one_and_world_tick_strictly_increases"
            ),
        },
        "fields": [
            "asset_and_runtime_block_identity",
            "impact_type",
            "sample_index_and_world_tick",
            "delta_seconds",
            "position_xyz",
            "velocity_xyz_before_falling_gravity",
            "on_ground_entity_present_impact_observed",
            "impact_cell",
            "mass_drag_inverted_gravity",
            "bounding_box_width_depth",
        ],
        "dry_air_recurrence": {
            "terminal_velocity": (
                "PhysicsMath.getTerminalVelocity_mass_density_area_drag"
            ),
            "vertical_acceleration": (
                "32*(1-abs(vy/terminal_velocity)^3)"
            ),
            "order": (
                "source_ItemPrePhysicsSystem_gravity_uses_source_delta_"
                "then_destination_BlockEntitySystems_motion_uses_"
                "destination_delta"
            ),
            "jax": "falling_block_dry_air_step",
        },
        "certifies": [
            "native_component_initialization",
            "block_entity_physics_then_falling_ticker_order",
            "dry_air_position_and_velocity_recurrence",
            "first_ground_contact_impact_cell_and_entity_removal",
        ],
        "does_not_certify": [
            "fluid_entry_or_exit",
            "dynamic_support_cascade_enumeration",
            "explosion_effect_kernel",
            "authenticated_player_action",
        ],
        "public_cascade_certified": False,
    }


def native_falling_block_trace_contract_sha256() -> str:
    payload = json.dumps(
        native_falling_block_trace_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_native_falling_block_trace(
    value: NativeFallingBlockTrace,
) -> None:
    if value.public_cascade_certified:
        raise ValueError(
            "fixture trace cannot certify general falling cascades"
        )
    if value.dry_static_fixture != value.fixture_available:
        raise ValueError("falling-block fixture scope is inconsistent")
    if not value.sample_available:
        if (
            value.stage != "unavailable"
            or value.block_asset_id
            or value.runtime_block_id != 0
            or value.impact_type
            or value.sample_index != -1
            or value.world_tick != -1
            or value.delta_seconds != 0.0
            or value.position != (0.0, 0.0, 0.0)
            or value.velocity != (0.0, 0.0, 0.0)
            or value.on_ground
            or value.entity_present
            or value.impact_observed
            or value.impact_cell != (0, 0, 0)
            or value.mass != 0.0
            or value.drag_coefficient != 0.0
            or value.inverted_gravity
            or value.box_width != 0.0
            or value.box_depth != 0.0
        ):
            raise ValueError("unavailable falling-block trace is inconsistent")
        return
    if not value.fixture_available:
        raise ValueError("falling-block sample exists outside its fixture")
    if (
        value.stage not in NATIVE_FALLING_BLOCK_TRACE_STAGES
        or not value.block_asset_id
        or value.runtime_block_id <= 0
        or value.impact_type not in {"Place", "Break", "Explode"}
        or value.sample_index < 0
        or value.world_tick < 0
        or value.mass <= 0.0
        or value.drag_coefficient <= 0.0
        or value.box_width <= 0.0
        or value.box_depth <= 0.0
    ):
        raise ValueError("available falling-block sample is incomplete")
    if value.impact_observed:
        if (
            value.stage != "ground_impact_after_falling_ticker"
            or not value.on_ground
            or value.entity_present
            or value.impact_cell
            != tuple(math.floor(axis) for axis in value.position)
        ):
            raise ValueError("falling-block impact sample is inconsistent")
    elif (
        value.stage == "ground_impact_after_falling_ticker"
        or value.on_ground
        or not value.entity_present
        or value.impact_cell != (0, 0, 0)
    ):
        raise ValueError("falling-block in-flight sample is inconsistent")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256")
    result = value.upper()
    if len(result) != 64 or any(
        character not in "0123456789ABCDEF" for character in result
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return result


def _boolean(info: Mapping[str, object], key: str) -> bool:
    value = info.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _string(info: Mapping[str, object], key: str) -> str:
    value = info.get(key)
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{key} must be a canonical string")
    return value


def _integer(
    info: Mapping[str, object],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} is below its minimum")
    return value


def _real(
    info: Mapping[str, object],
    key: str,
    *,
    minimum: float | None = None,
) -> float:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{key} is below its minimum")
    return result


__all__ = [
    "FALLING_BLOCK_IMPACT_BREAK",
    "FALLING_BLOCK_IMPACT_EXPLODE",
    "FALLING_BLOCK_IMPACT_NONE",
    "FALLING_BLOCK_IMPACT_PLACE",
    "FALLING_BLOCK_IMPACT_SCHEMA",
    "FALLING_BLOCK_IMPACT_TYPES",
    "FALLING_BLOCK_IMPACT_VERSION",
    "LocalFallingBlockImpact",
    "NATIVE_FALLING_BLOCK_TRACE_SCHEMA",
    "NATIVE_FALLING_BLOCK_TRACE_STAGES",
    "NATIVE_FALLING_BLOCK_TRACE_VERSION",
    "NativeFallingBlockTrace",
    "falling_block_impact_contract",
    "falling_block_impact_contract_sha256",
    "native_falling_block_trace_contract",
    "native_falling_block_trace_contract_sha256",
    "resolve_local_falling_block_impact",
]
