"""Host-only compiler for projectile-terminal deployable-area graphs.

The compiler consumes resolved JSON objects, never item names.  It admits the
small mechanical family whose equipped Primary is a timed projectile launch
and whose world-contact terminal creates one typed AOE carrying one
entity effect.  Presentation-only fields are ignored through explicit
allowlists; unknown interaction or mechanical payloads fail closed.
"""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


DEPLOYABLE_ATTACK_OWNER = 1 << 0
DEPLOYABLE_ATTACK_TEAM = 1 << 1
DEPLOYABLE_ATTACK_ENEMIES = 1 << 2
DEPLOYABLE_ATTACK_FLAG_MASK = (
    DEPLOYABLE_ATTACK_OWNER | DEPLOYABLE_ATTACK_TEAM | DEPLOYABLE_ATTACK_ENEMIES
)
DEPLOYABLE_AREA_SHAPES = frozenset({"Sphere", "Cylinder"})
PROJECTILE_ROTATION_MODES = frozenset(
    {"None", "Velocity", "VelocityDamped", "VelocityRoll"}
)
PROJECTILE_COMPONENT_LIFETIME_SECONDS = 300.0
# DeployablesUtils creates EntityStatMap then update() installs the native
# global Health default even when DeployableConfig.Stats is omitted. This is
# pinned exclusion metadata: the bounded JAX AreaState does not execute
# damageable deployable ECS entities.
DEPLOYABLE_DEFAULT_HEALTH = 100.0
DEPLOYABLE_DEFAULT_HEALTH_CANONICAL_SHA256 = (
    "7020360FC1F27EA1C5AC8A2199464078AE3BBA3821F10F1173488B70F1E469F0"
)

_DEFAULT_HEALTH_STAT_KEYS = frozenset(
    {
        "InitialValue",
        "Min",
        "Max",
        "Shared",
        "ResetType",
        "Regenerating",
    }
)

_ITEM_KEYS = frozenset(
    {
        "Categories",
        "DroppedItemAnimation",
        "Icon",
        "IconProperties",
        "Interactions",
        "ItemLevel",
        "MaxStack",
        "Model",
        "PlayerAnimationsId",
        "Quality",
        "Recipe",
        "Scale",
        "Tags",
        "Texture",
        "TranslationProperties",
    }
)
_PRIMARY_KEYS = frozenset({"Cooldown", "Interactions"})
_COOLDOWN_KEYS = frozenset({"Cooldown", "Id"})
_SERIAL_KEYS = frozenset({"Interactions", "Type"})
_SIMPLE_KEYS = frozenset({"Effects", "RunTime", "Type"})
_SIMPLE_EFFECT_KEYS = frozenset({"ItemAnimationId", "ItemPlayerAnimationsId"})
_PROJECTILE_INTERACTION_KEYS = frozenset({"Config", "Type"})
_PROJECTILE_KEYS = frozenset(
    {
        "Interactions",
        "LaunchForce",
        "Model",
        "Physics",
        "SpawnOffset",
        "SpawnRotationOffset",
    }
)
_STANDARD_PHYSICS_KEYS = frozenset(
    {
        "AllowRolling",
        "BounceCount",
        "BounceLimit",
        "Bounciness",
        "Gravity",
        "RollingFrictionFactor",
        "RotationMode",
        "SticksVertically",
        "TerminalVelocityAir",
        "TerminalVelocityWater",
        "Type",
    }
)
_SPAWN_ROTATION_KEYS = frozenset({"Pitch", "Roll", "Yaw"})
_PROJECTILE_MISS_KEYS = frozenset({"Interactions"})
_SPAWN_DEPLOYABLE_KEYS = frozenset({"Config", "Type"})
_REMOVE_ENTITY_KEYS = frozenset({"Entity", "Type"})
_PROJECTILE_MODEL_KEYS = frozenset(
    {
        "AnimationSets",
        "CrouchOffset",
        "DefaultAttachments",
        "EyeHeight",
        "HitBox",
        "MaxScale",
        "MinScale",
        "Model",
        "Particles",
        "Texture",
        "Trails",
    }
)
_HITBOX_KEYS = frozenset({"Min", "Max"})
_EFFECT_KEYS = frozenset(
    {
        "ApplicationEffects",
        "DamageCalculatorCooldown",
        "Debuff",
        "Duration",
        "OverlapBehavior",
        "StatModifiers",
        "StatusEffectIcon",
    }
)

_AREA_PRESENTATION_KEYS = frozenset(
    {
        "AmbientSoundEventId",
        "DebugVisuals",
        "DeploySoundEventId",
        "DespawnParticles",
        "DespawnSoundEventId",
        "DieSoundEventId",
        "ModelPreview",
        "SpawnParticles",
        "WireframeDebugVisuals",
    }
)
_AREA_MECHANICAL_KEYS = frozenset(
    {
        "AllowPlaceOnWalls",
        "ApplyEffects",
        "AttackEnemies",
        "AttackOwner",
        "AttackTeam",
        "CountTowardsGlobalLimit",
        "DamageAmount",
        "DamageCause",
        "DamageInterval",
        "EndRadius",
        "Height",
        "HitboxCollisionConfig",
        "Id",
        "Invulnerable",
        "LiveDuration",
        "MaxLiveCount",
        "Model",
        "ModelScale",
        "RadiusChangeTime",
        "Shape",
        "StartRadius",
        "Stats",
        "Type",
    }
)
_EFFECT_PRESENTATION_KEYS = frozenset(
    {
        "EntityBottomTint",
        "EntityTopTint",
        "LocalSoundEventId",
        "ScreenEffect",
    }
)


@dataclass(frozen=True)
class DeployableStatusSpec:
    """One generically executable entity-effect payload."""

    effect_asset_id: str
    duration_seconds: float
    calculator_cooldown_seconds: float
    healing_per_cycle: float
    horizontal_speed_multiplier: float
    overlap_behavior: str
    debuff: bool


@dataclass(frozen=True)
class ProjectileTerminalDeployableSpec:
    """Resolved host record for one projectile -> typed area -> effect graph."""

    profile: str
    asset_id: str
    family_id: int
    projectile_config_id: str
    deployable_id: str
    cooldown_group_id: str
    launch_event_seconds: float
    ability_duration_seconds: float
    cooldown_seconds: float
    launch_force: float
    gravity: float
    terminal_velocity_air: float
    terminal_velocity_water: float
    rotation_mode: str
    collision_min: tuple[float, float, float]
    collision_max: tuple[float, float, float]
    projectile_lifetime_seconds: float
    spawn_offset: tuple[float, float, float]
    spawn_yaw_degrees: float
    spawn_pitch_degrees: float
    bounciness: float
    bounce_limit: float
    bounce_count: int
    allow_rolling: bool
    rolling_friction_factor: float
    sticks_vertically: bool
    area_shape: str
    deployable_collision_min: tuple[float, float, float]
    deployable_collision_max: tuple[float, float, float]
    deployable_hard_collision: bool
    deployable_count_towards_global_limit: bool
    deployable_max_live_count: int
    deployable_default_health: float
    area_duration_seconds: float
    area_interval_seconds: float
    area_start_radius: float
    area_end_radius: float
    area_height: float
    area_radius_change_seconds: float
    area_damage: float
    area_damage_cause: str
    area_attack_flags: int
    status: DeployableStatusSpec

    @property
    def collision_half_extent(self) -> tuple[float, float, float]:
        return tuple(
            (maximum - minimum) * 0.5
            for minimum, maximum in zip(
                self.collision_min,
                self.collision_max,
                strict=True,
            )
        )

    @property
    def collision_center_offset(self) -> tuple[float, float, float]:
        return tuple(
            (minimum + maximum) * 0.5
            for minimum, maximum in zip(
                self.collision_min,
                self.collision_max,
                strict=True,
            )
        )


def compile_projectile_terminal_deployable(
    *,
    profile: str,
    asset_id: str,
    family_id: int,
    items: Mapping[str, Mapping[str, Any]],
    projectile_configs: Mapping[str, Mapping[str, Any]],
    projectile_models: Mapping[str, Mapping[str, Any]],
    deployable_models: Mapping[str, Mapping[str, Any]],
    hitbox_collision_configs: Mapping[str, Mapping[str, Any]],
    default_health_stat: Mapping[str, Any],
    effects: Mapping[str, Mapping[str, Any]],
) -> ProjectileTerminalDeployableSpec:
    """Compile one strict typed graph from already resolved asset documents."""

    _nonempty(profile, "profile")
    _nonempty(asset_id, "asset_id")
    if isinstance(family_id, bool) or not isinstance(family_id, int) or family_id <= 0:
        raise ValueError("family_id must be a positive integer")

    _reject_unknown(
        default_health_stat,
        _DEFAULT_HEALTH_STAT_KEYS,
        "default Health stat",
    )
    default_health_sha256 = (
        hashlib.sha256(
            json.dumps(
                default_health_stat,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        )
        .hexdigest()
        .upper()
    )
    if default_health_sha256 != DEPLOYABLE_DEFAULT_HEALTH_CANONICAL_SHA256:
        raise ValueError(
            "default Health stat must match the complete pinned 0.5.7 "
            "record; regeneration and damageability are excluded mechanics"
        )
    default_health = _finite(
        default_health_stat.get("InitialValue"),
        "default Health.InitialValue",
        minimum=0.0,
        strict=True,
    )
    if default_health != _finite(
        default_health_stat.get("Max"),
        "default Health.Max",
        minimum=0.0,
        strict=True,
    ):
        raise ValueError("default Health initial value must equal its maximum")

    try:
        item = items[asset_id]
    except KeyError as error:
        raise ValueError(f"missing item asset {asset_id!r}") from error
    _reject_unknown(item, _ITEM_KEYS, "Item")
    item_interactions = _mapping(item.get("Interactions"), "Interactions")
    if set(item_interactions) != {"Primary"}:
        raise ValueError("Item.Interactions must contain only Primary")
    primary = _mapping(item_interactions.get("Primary"), "Primary")
    _reject_unknown(primary, _PRIMARY_KEYS, "Primary")
    primary_steps = _list(primary.get("Interactions"), "Primary.Interactions")
    if len(primary_steps) != 1:
        raise ValueError("Primary must contain exactly one Serial interaction")
    serial = _mapping(primary_steps[0], "Primary.Serial")
    _reject_unknown(serial, _SERIAL_KEYS, "Primary.Serial")
    if serial.get("Type") != "Serial":
        raise ValueError("Primary terminal deployable root must be Serial")
    serial_steps = _list(serial.get("Interactions"), "Serial.Interactions")
    if len(serial_steps) != 2:
        raise ValueError("Serial must contain Simple then Projectile")
    delay_node = _mapping(serial_steps[0], "Serial.Simple")
    launch_node = _mapping(serial_steps[1], "Serial.Projectile")
    _reject_unknown(delay_node, _SIMPLE_KEYS, "Serial.Simple")
    _reject_unknown(launch_node, _PROJECTILE_INTERACTION_KEYS, "Serial.Projectile")
    if delay_node.get("Type") != "Simple" or launch_node.get("Type") != "Projectile":
        raise ValueError("Serial must contain Simple then Projectile")
    launch_time = _finite(delay_node.get("RunTime"), "Simple.RunTime", minimum=0.0)
    simple_effects = _mapping(delay_node.get("Effects", {}), "Simple.Effects")
    _reject_unknown(simple_effects, _SIMPLE_EFFECT_KEYS, "Simple.Effects")
    projectile_config_id = _nonempty(launch_node.get("Config"), "Projectile.Config")
    try:
        projectile = projectile_configs[projectile_config_id]
    except KeyError as error:
        raise ValueError(
            f"missing projectile config {projectile_config_id!r}"
        ) from error
    cooldown_node = _mapping(primary.get("Cooldown"), "Primary.Cooldown")
    _reject_unknown(cooldown_node, _COOLDOWN_KEYS, "Primary.Cooldown")
    cooldown_group_id = _nonempty(
        cooldown_node.get("Id"),
        "Primary.Cooldown.Id",
    )
    cooldown = _finite(
        cooldown_node.get("Cooldown"),
        "Primary.Cooldown.Cooldown",
        minimum=0.0,
    )

    _reject_unknown(projectile, _PROJECTILE_KEYS, "Projectile")
    projectile_model_id = _nonempty(projectile.get("Model"), "Projectile.Model")
    try:
        projectile_model = projectile_models[projectile_model_id]
    except KeyError as error:
        raise ValueError(f"missing projectile model {projectile_model_id!r}") from error
    physics = _mapping(projectile.get("Physics"), "Projectile.Physics")
    _reject_unknown(physics, _STANDARD_PHYSICS_KEYS, "Projectile.Physics")
    if physics.get("Type") != "Standard":
        raise ValueError("projectile Physics.Type must be Standard")
    gravity = _finite(physics.get("Gravity", 0.0), "Physics.Gravity", minimum=0.0)
    terminal_velocity = _finite(
        physics.get("TerminalVelocityAir", 1.0),
        "Physics.TerminalVelocityAir",
        minimum=0.0,
        strict=True,
    )
    terminal_velocity_water = _finite(
        physics.get("TerminalVelocityWater", 1.0),
        "Physics.TerminalVelocityWater",
        minimum=0.0,
        strict=True,
    )
    rotation_mode = physics.get("RotationMode", "VelocityDamped")
    if rotation_mode not in PROJECTILE_ROTATION_MODES:
        raise ValueError(f"unsupported Physics.RotationMode {rotation_mode!r}")
    bounciness = _finite(
        physics.get("Bounciness", 0.0),
        "Physics.Bounciness",
        minimum=0.0,
        maximum=1.0,
    )
    bounce_limit = _finite(
        physics.get("BounceLimit", 0.4),
        "Physics.BounceLimit",
        minimum=0.0,
    )
    bounce_count = physics.get("BounceCount", -1)
    if (
        isinstance(bounce_count, bool)
        or not isinstance(bounce_count, int)
        or bounce_count < -1
    ):
        raise ValueError("Physics.BounceCount must be -1 or a non-negative integer")
    allow_rolling = _boolean(physics.get("AllowRolling", False), "Physics.AllowRolling")
    sticks_vertically = _boolean(
        physics.get("SticksVertically", False),
        "Physics.SticksVertically",
    )
    rolling_friction = _finite(
        physics.get("RollingFrictionFactor", 0.99),
        "Physics.RollingFrictionFactor",
        minimum=0.0,
    )
    launch_force = _finite(
        projectile.get("LaunchForce"), "LaunchForce", minimum=0.0, strict=True
    )
    spawn_offset = _vector(projectile.get("SpawnOffset", {}), "SpawnOffset")
    rotation = _mapping(
        projectile.get("SpawnRotationOffset", {}), "SpawnRotationOffset"
    )
    _reject_unknown(rotation, _SPAWN_ROTATION_KEYS, "SpawnRotationOffset")
    spawn_yaw = _finite(rotation.get("Yaw", 0.0), "SpawnRotationOffset.Yaw")
    spawn_pitch = _finite(rotation.get("Pitch", 0.0), "SpawnRotationOffset.Pitch")
    _finite(rotation.get("Roll", 0.0), "SpawnRotationOffset.Roll")

    collision_min, collision_max = _bounds(projectile_model)
    interactions = _mapping(projectile.get("Interactions"), "Projectile.Interactions")
    if set(interactions) != {"ProjectileMiss"}:
        raise ValueError("terminal deployable projectile supports only ProjectileMiss")
    miss = _mapping(interactions["ProjectileMiss"], "ProjectileMiss")
    _reject_unknown(miss, _PROJECTILE_MISS_KEYS, "ProjectileMiss")
    miss_steps = _list(miss.get("Interactions"), "ProjectileMiss.Interactions")
    if len(miss_steps) != 2:
        raise ValueError(
            "ProjectileMiss must spawn a deployable then remove the projectile"
        )
    spawn = _mapping(miss_steps[0], "ProjectileMiss.Spawn")
    remove = _mapping(miss_steps[1], "ProjectileMiss.Remove")
    _reject_unknown(spawn, _SPAWN_DEPLOYABLE_KEYS, "ProjectileMiss.Spawn")
    _reject_unknown(remove, _REMOVE_ENTITY_KEYS, "ProjectileMiss.Remove")
    if spawn.get("Type") != "SpawnDeployableAtHitLocation":
        raise ValueError("ProjectileMiss first interaction must spawn at hit location")
    if remove.get("Type") != "RemoveEntity" or remove.get("Entity") != "User":
        raise ValueError("ProjectileMiss must remove its projectile user")
    area = _mapping(spawn.get("Config"), "SpawnDeployableAtHitLocation.Config")
    unknown_area = set(area) - _AREA_PRESENTATION_KEYS - _AREA_MECHANICAL_KEYS
    if unknown_area:
        raise ValueError(
            f"unknown deployable-area payload keys: {sorted(unknown_area)}"
        )
    if area.get("Type") != "Aoe":
        raise ValueError("terminal deployable must be an Aoe")
    deployable_id = _nonempty(area.get("Id"), "Aoe.Id")
    deployable_model_id = _nonempty(area.get("Model"), "Aoe.Model")
    try:
        deployable_model = deployable_models[deployable_model_id]
    except KeyError as error:
        raise ValueError(f"missing deployable model {deployable_model_id!r}") from error
    deployable_collision_min, deployable_collision_max = _bounds(
        _mapping(deployable_model, f"deployable model {deployable_model_id}")
    )
    model_scale = _finite(
        area.get("ModelScale", 1.0), "Aoe.ModelScale", minimum=0.0, strict=True
    )
    deployable_collision_min = tuple(
        value * model_scale for value in deployable_collision_min
    )
    deployable_collision_max = tuple(
        value * model_scale for value in deployable_collision_max
    )
    collision_config_id = _nonempty(
        area.get("HitboxCollisionConfig"),
        "Aoe.HitboxCollisionConfig",
    )
    try:
        collision_config = _mapping(
            hitbox_collision_configs[collision_config_id],
            f"hitbox collision {collision_config_id}",
        )
    except KeyError as error:
        raise ValueError(
            f"missing hitbox collision config {collision_config_id!r}"
        ) from error
    _reject_unknown(
        collision_config,
        frozenset({"CollisionType"}),
        f"hitbox collision {collision_config_id}",
    )
    if collision_config.get("CollisionType") != "Hard":
        raise ValueError("terminal deployable requires Hard collision")
    if _boolean(area.get("Invulnerable", False), "Aoe.Invulnerable"):
        raise ValueError("invulnerable terminal deployables are not supported")
    stats = _mapping(area.get("Stats", {}), "Aoe.Stats")
    if stats:
        raise ValueError("authored deployable stat overrides are not supported")
    if _boolean(area.get("AllowPlaceOnWalls", False), "Aoe.AllowPlaceOnWalls"):
        raise ValueError("wall-placeable terminal deployables are not supported")
    count_towards_global = _boolean(
        area.get("CountTowardsGlobalLimit", True),
        "Aoe.CountTowardsGlobalLimit",
    )
    if not count_towards_global:
        raise ValueError(
            "terminal deployable subset requires CountTowardsGlobalLimit=true"
        )
    max_live_count = area.get("MaxLiveCount", 2_147_483_647)
    if (
        isinstance(max_live_count, bool)
        or not isinstance(max_live_count, int)
        or max_live_count <= 0
    ):
        raise ValueError("Aoe.MaxLiveCount must be a positive integer")
    if max_live_count != 2_147_483_647:
        raise ValueError(
            "terminal deployable subset requires the default unbounded MaxLiveCount"
        )
    area_shape = area.get("Shape", "Sphere")
    if area_shape not in DEPLOYABLE_AREA_SHAPES:
        raise ValueError(f"unsupported deployable-area shape {area_shape!r}")
    effect_ids = _list(area.get("ApplyEffects"), "Aoe.ApplyEffects")
    if len(effect_ids) != 1:
        raise ValueError("terminal deployable supports exactly one entity effect")
    effect_id = _nonempty(effect_ids[0], "Aoe.ApplyEffects[0]")
    try:
        effect = effects[effect_id]
    except KeyError as error:
        raise ValueError(f"missing entity-effect payload {effect_id!r}") from error
    status = _compile_status(effect_id, _mapping(effect, f"effect {effect_id}"))

    attack_flags = 0
    if _boolean(area.get("AttackOwner", False), "Aoe.AttackOwner"):
        attack_flags |= DEPLOYABLE_ATTACK_OWNER
    if _boolean(area.get("AttackTeam", False), "Aoe.AttackTeam"):
        attack_flags |= DEPLOYABLE_ATTACK_TEAM
    if _boolean(area.get("AttackEnemies", True), "Aoe.AttackEnemies"):
        attack_flags |= DEPLOYABLE_ATTACK_ENEMIES

    area_duration = _finite(
        area.get("LiveDuration", 1.0), "Aoe.LiveDuration", minimum=0.0, strict=True
    )
    area_interval = _finite(
        area.get("DamageInterval", 1.0), "Aoe.DamageInterval", minimum=0.0, strict=True
    )
    start_radius = _finite(area.get("StartRadius", 1.0), "Aoe.StartRadius", minimum=0.0)
    end_radius = _finite(area.get("EndRadius"), "Aoe.EndRadius", minimum=0.0)
    radius_change = _finite(
        area.get("RadiusChangeTime"), "Aoe.RadiusChangeTime", minimum=0.0, strict=True
    )
    height = _finite(area.get("Height", 1.0), "Aoe.Height", minimum=0.0, strict=True)
    damage = _finite(area.get("DamageAmount", 1.0), "Aoe.DamageAmount", minimum=0.0)
    if damage != 0.0:
        raise ValueError(
            "terminal deployable area damage is unsupported until deployable "
            "entities share the complete damage-target axis"
        )
    damage_cause = _nonempty(area.get("DamageCause", "Physical"), "Aoe.DamageCause")

    return ProjectileTerminalDeployableSpec(
        profile=profile,
        asset_id=asset_id,
        family_id=family_id,
        projectile_config_id=projectile_config_id,
        deployable_id=deployable_id,
        cooldown_group_id=cooldown_group_id,
        launch_event_seconds=launch_time,
        ability_duration_seconds=launch_time,
        cooldown_seconds=cooldown,
        launch_force=launch_force,
        gravity=gravity,
        terminal_velocity_air=terminal_velocity,
        terminal_velocity_water=terminal_velocity_water,
        rotation_mode=rotation_mode,
        collision_min=collision_min,
        collision_max=collision_max,
        projectile_lifetime_seconds=PROJECTILE_COMPONENT_LIFETIME_SECONDS,
        spawn_offset=spawn_offset,
        spawn_yaw_degrees=spawn_yaw,
        spawn_pitch_degrees=spawn_pitch,
        bounciness=bounciness,
        bounce_limit=bounce_limit,
        bounce_count=bounce_count,
        allow_rolling=allow_rolling,
        rolling_friction_factor=rolling_friction,
        sticks_vertically=sticks_vertically,
        area_shape=area_shape,
        deployable_collision_min=deployable_collision_min,
        deployable_collision_max=deployable_collision_max,
        deployable_hard_collision=True,
        deployable_count_towards_global_limit=count_towards_global,
        deployable_max_live_count=max_live_count,
        deployable_default_health=default_health,
        area_duration_seconds=area_duration,
        area_interval_seconds=area_interval,
        area_start_radius=start_radius,
        area_end_radius=end_radius,
        area_height=height,
        area_radius_change_seconds=radius_change,
        area_damage=damage,
        area_damage_cause=damage_cause,
        area_attack_flags=attack_flags,
        status=status,
    )


def _compile_status(effect_id: str, effect: Mapping[str, Any]) -> DeployableStatusSpec:
    _reject_unknown(effect, _EFFECT_KEYS, f"EntityEffect {effect_id}")
    stat_modifiers = _mapping(
        effect.get("StatModifiers", {}), "EntityEffect.StatModifiers"
    )
    unknown_stats = set(stat_modifiers) - {"Health"}
    if unknown_stats:
        raise ValueError(
            f"unknown entity-effect stat payloads: {sorted(unknown_stats)}"
        )
    healing = _finite(
        stat_modifiers.get("Health", 0.0), "StatModifiers.Health", minimum=0.0
    )
    application = _mapping(
        effect.get("ApplicationEffects", {}), "EntityEffect.ApplicationEffects"
    )
    unknown_application = (
        set(application) - _EFFECT_PRESENTATION_KEYS - {"HorizontalSpeedMultiplier"}
    )
    if unknown_application:
        raise ValueError(
            f"unknown entity-effect application payloads: {sorted(unknown_application)}"
        )
    speed = _finite(
        application.get("HorizontalSpeedMultiplier", 1.0),
        "ApplicationEffects.HorizontalSpeedMultiplier",
        minimum=0.0,
        strict=True,
    )
    if healing == 0.0 and speed == 1.0:
        raise ValueError("entity effect has no supported healing or status modifier")
    cooldown = _finite(
        effect.get("DamageCalculatorCooldown", 0.0),
        "EntityEffect.DamageCalculatorCooldown",
        minimum=0.0,
    )
    if healing > 0.0 and cooldown <= 0.0:
        raise ValueError("healing stat changes require a positive calculator cooldown")
    overlap = effect.get("OverlapBehavior", "Ignore")
    if overlap not in {"Ignore", "Extend", "Overwrite"}:
        raise ValueError(f"unsupported entity-effect overlap behavior {overlap!r}")
    return DeployableStatusSpec(
        effect_asset_id=effect_id,
        duration_seconds=_finite(
            effect.get("Duration"), "EntityEffect.Duration", minimum=0.0, strict=True
        ),
        calculator_cooldown_seconds=cooldown,
        healing_per_cycle=healing,
        horizontal_speed_multiplier=speed,
        overlap_behavior=overlap,
        debuff=_boolean(effect.get("Debuff", False), "EntityEffect.Debuff"),
    )


def _bounds(
    model: Mapping[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    _reject_unknown(model, _PROJECTILE_MODEL_KEYS, "ProjectileModel")
    hitbox = _mapping(model.get("HitBox"), "ProjectileModel.HitBox")
    _reject_unknown(hitbox, _HITBOX_KEYS, "ProjectileModel.HitBox")
    minimum = _vector(hitbox.get("Min"), "ProjectileModel.HitBox.Min")
    maximum = _vector(hitbox.get("Max"), "ProjectileModel.HitBox.Max")
    if any(high <= low for low, high in zip(minimum, maximum, strict=True)):
        raise ValueError(
            "projectile-model hitbox must have positive extent on every axis"
        )
    return minimum, maximum


def _vector(value: Any, label: str) -> tuple[float, float, float]:
    mapping = _mapping(value, label)
    unknown = set(mapping) - {"X", "Y", "Z"}
    if unknown:
        raise ValueError(f"{label} has unknown axes: {sorted(unknown)}")
    return tuple(_finite(mapping.get(axis, 0.0), f"{label}.{axis}") for axis in "XYZ")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} has unknown keys: {sorted(unknown)}")


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a bool")
    return value


def _finite(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        comparator = ">" if strict else ">="
        raise ValueError(f"{label} must be {comparator} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


__all__ = [
    "DEPLOYABLE_AREA_SHAPES",
    "DEPLOYABLE_DEFAULT_HEALTH",
    "DEPLOYABLE_DEFAULT_HEALTH_CANONICAL_SHA256",
    "DEPLOYABLE_ATTACK_ENEMIES",
    "DEPLOYABLE_ATTACK_FLAG_MASK",
    "DEPLOYABLE_ATTACK_OWNER",
    "DEPLOYABLE_ATTACK_TEAM",
    "DeployableStatusSpec",
    "PROJECTILE_COMPONENT_LIFETIME_SECONDS",
    "PROJECTILE_ROTATION_MODES",
    "ProjectileTerminalDeployableSpec",
    "compile_projectile_terminal_deployable",
]
