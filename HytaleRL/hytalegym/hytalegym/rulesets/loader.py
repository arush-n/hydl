"""Strict loaders for packaged, versioned Hytale mechanics data."""

from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import math
from typing import Any

from hytalegym.combat.assets import HYTALE_0_5_7_ASSETS_SHA256


COMBAT_RULESET_RESOURCE = (
    "hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json"
)
_COMBAT_SCHEMA = "hytalerl_combat_ruleset_v6"
_COMBAT_VERSION = 9


def load_combat_ruleset() -> dict[str, Any]:
    """Return an isolated copy of the validated 0.5.7 combat ruleset."""

    return copy.deepcopy(_load_combat_ruleset())


def combat_ruleset_sha256() -> str:
    """Return the exact packaged ruleset fingerprint used by this build."""

    return hashlib.sha256(_combat_ruleset_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _combat_ruleset_bytes() -> bytes:
    resource = files("hytalegym.rulesets").joinpath(COMBAT_RULESET_RESOURCE)
    return resource.read_bytes()


@lru_cache(maxsize=1)
def _load_combat_ruleset() -> dict[str, Any]:
    """Load and minimally validate the shared 0.5.7 combat ruleset."""

    data = json.loads(_combat_ruleset_bytes())
    if (
        data.get("schema") != _COMBAT_SCHEMA
        or int(data.get("version", -1)) != _COMBAT_VERSION
    ):
        raise ValueError("unsupported packaged combat ruleset")
    if data.get("hytale_server_version") != "0.5.7":
        raise ValueError("combat ruleset is not calibrated for Hytale 0.5.7")
    required = {
        "engine",
        "fixture",
        "agent",
        "target",
        "damage_interaction",
        "regeneration",
        "reward",
        "observation_normalization",
        "provenance",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"combat ruleset is missing {sorted(missing)}")
    surface = data["provenance"].get("asset_surface_audit")
    if (
        not isinstance(surface, dict)
        or surface.get("schema") != "hytalerl_combat_asset_surface_v1"
        or surface.get("resource")
        != "hytale_0_5_7/asset_surface_v1.json"
        or surface.get("assets_sha256") != HYTALE_0_5_7_ASSETS_SHA256
    ):
        raise ValueError("combat ruleset asset-surface provenance is invalid")
    if len(data["agent"]["attacks"]) != 3:
        raise ValueError("Razorleaf ruleset must contain three attacks")
    if len(data["target"]["attacks"]) != 5:
        raise ValueError("Brawler ruleset must contain five attacks")
    agent = data["agent"]
    target = data["target"]
    chase_senses = target.get("chase_senses")
    if (
        not isinstance(chase_senses, dict)
        or float(chase_senses.get("view_range", 0.0)) <= 0.0
        or float(chase_senses.get("hearing_range", -1.0)) < 0.0
        or not isinstance(
            chase_senses.get("hearing_suppressed_by_crouching"),
            bool,
        )
    ):
        raise ValueError("target chase senses contract is invalid")
    for name, actor in (("agent", agent), ("target", target)):
        minimum = float(actor["attack_pause_min_seconds"])
        maximum = float(actor["attack_pause_max_seconds"])
        if minimum < 0 or maximum < minimum:
            raise ValueError(f"{name} attack pause range is invalid")
    if not math.isclose(
        float(agent["max_speed"]),
        float(agent["asset_max_walk_speed"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("agent max speed must match its role asset")
    if float(agent["max_speed"]) <= 0.0:
        raise ValueError("agent effective max speed must be positive")
    if float(agent["jump_velocity_gravity_floor"]) <= 0.0:
        raise ValueError("agent jump velocity gravity floor must be positive")
    if float(agent["steering_relative_turn_speed"]) <= 0.0:
        raise ValueError("agent relative turn speed must be positive")
    # The head is a second steering seam, not a view of the body one:
    # `MotionControllerBase` drives body and head from separate `Steering`
    # objects with their own rotation ceilings, and bounds the head to an
    # authored yaw/pitch window around the body. Same shape as the target
    # contract below.
    if (
        float(agent["max_head_rotation_degrees_per_second"]) <= 0.0
        or float(agent["head_yaw_max_degrees"])
        <= float(agent["head_yaw_min_degrees"])
        or float(agent["head_pitch_max_degrees"])
        <= float(agent["head_pitch_min_degrees"])
    ):
        raise ValueError("agent head-motion contract is invalid")
    if (
        float(agent["knockback_scale"]) <= 0.0
        or float(agent["movement_velocity_resistance"]) <= 0.0
        or float(agent["min_walk_speed"]) < 0.0
        or not 0.0 <= float(agent["min_hit_slowdown"]) <= 1.0
    ):
        raise ValueError("agent knockback controller contract is invalid")
    walk = agent.get("walk_controller")
    if not isinstance(walk, dict):
        raise ValueError("agent walk controller must be an object")
    if (
        float(walk["gravity"]) <= 0.0
        or float(walk["fall_acceleration_multiplier"]) <= 0.0
        or float(walk["gravity_drag_exponent"]) <= 0.0
        or float(walk["max_fall_speed"]) <= 0.0
        or float(walk["max_sink_speed_fluid"]) <= 0.0
        or float(walk["max_climb_height"]) <= 0.0
        or float(walk["max_drop_height"])
        <= float(walk["max_climb_height"])
    ):
        raise ValueError("agent walk controller contract is invalid")
    _validate_bounds(agent["bounding_box"], "agent bounding box")
    if not math.isclose(
        float(agent["effective_eye_height"]),
        float(agent["asset_eye_height"]) * float(agent["model_scale"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("agent effective eye height derivation is invalid")
    if float(data["engine"]["steering_slowdown_falloff"]) <= 0.0:
        raise ValueError("steering slowdown falloff must be positive")
    if float(data["engine"]["horizontal_selector_pi"]) <= 0.0:
        raise ValueError("horizontal selector pi must be positive")
    if (
        float(data["engine"]["legacy_horizontal_knockback_scale"]) <= 0.0
        or float(
            data["engine"]["legacy_motion_controller_horizontal_factor"]
        )
        <= 0.0
    ):
        raise ValueError("legacy knockback scaling must be positive")
    if target["velocity_control"] != "STEERING_TRANSLATION":
        raise ValueError("unsupported target velocity control")
    if target["agent_collision_mode"] != "NON_BLOCKING":
        raise ValueError("unsupported target/agent collision mode")
    if int(target["chase_reaction_ticks"]) != 1:
        raise ValueError("target chase must begin on its first active tick")
    _validate_bounds(target["bounding_box"], "target bounding box")
    if not math.isclose(
        float(target["effective_eye_height"]),
        float(target["asset_eye_height"]) * float(target["model_scale"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("target effective eye height derivation is invalid")
    if (
        float(target["max_head_rotation_degrees_per_second"]) <= 0.0
        or float(target["head_aim_relative_turn_speed"]) <= 0.0
        or float(target["head_default_relative_turn_speed"]) <= 0.0
        or float(target["head_yaw_max_degrees"])
        <= float(target["head_yaw_min_degrees"])
        or float(target["head_pitch_max_degrees"])
        <= float(target["head_pitch_min_degrees"])
    ):
        raise ValueError("target head-motion contract is invalid")
    chase = target["chase"]
    if (
        float(chase["stop_distance"]) < 0.0
        or float(chase["slowdown_distance"])
        <= float(chase["stop_distance"])
    ):
        raise ValueError("target chase distances are invalid")
    maintain = target["maintain_distance"]
    if (
        float(maintain["activation_range"]) <= 0.0
        or float(maintain["desired_distance_min"]) < 0.0
        or float(maintain["desired_distance_max"])
        < float(maintain["desired_distance_min"])
        or float(maintain["move_threshold"]) <= 0.0
        or not 0.0 <= float(maintain["target_distance_factor"]) <= 1.0
        or float(maintain["move_towards_slowdown_distance"]) < 0.0
        or float(maintain["relative_forward_speed"]) <= 0.0
        or float(maintain["relative_backward_speed"]) <= 0.0
    ):
        raise ValueError("target maintain-distance contract is invalid")
    for prefix in ("strafing_duration", "strafing_frequency"):
        minimum = float(maintain[f"{prefix}_min_seconds"])
        maximum = float(maintain[f"{prefix}_max_seconds"])
        if minimum < 0.0 or maximum < minimum:
            raise ValueError(f"target {prefix} range is invalid")
    if (
        float(maintain["strafing_yaw_offset_degrees"]) < 0.0
        or float(maintain["strafing_translation_offset_degrees"]) < 0.0
    ):
        raise ValueError("target strafing offsets are invalid")
    expected_chase_speed = (
        float(target["asset_max_walk_speed"])
        * float(target["relative_chase_speed"])
    )
    if not math.isclose(
        float(target["chase_speed"]),
        expected_chase_speed,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "target chase speed must match max walk speed times relative speed"
        )
    for attack in target["attacks"]:
        if (
            float(attack["selector_runtime_seconds"]) <= 0.0
            or float(attack["start_distance"]) < 0.0
            or float(attack["end_distance"])
            <= float(attack["start_distance"])
            or float(attack["arc_degrees"]) <= 0.0
            or float(attack["extend_top"]) < 0.0
            or float(attack["extend_bottom"]) < 0.0
        ):
            raise ValueError("target horizontal selector is invalid")
    damage_interaction = data["damage_interaction"]
    if damage_interaction.get("interaction_id") != "NPC_Attack_Melee_Damage":
        raise ValueError("unsupported target damage interaction")
    knockback = damage_interaction.get("knockback")
    if not isinstance(knockback, dict):
        raise ValueError("target damage knockback must be an object")
    if (
        knockback.get("type") != "DIRECTIONAL"
        or knockback.get("velocity_type") != "SET"
        or float(knockback["force"]) < 0.0
        or float(knockback["duration_seconds"]) != 0.0
        or "velocity_config" not in knockback
        or knockback["velocity_config"] is not None
    ):
        raise ValueError("unsupported target directional knockback contract")
    for component in ("relative_x", "relative_z", "velocity_y"):
        if not math.isfinite(float(knockback[component])):
            raise ValueError("target knockback vector must be finite")
    if float(data["reward"]["target_damage_scale"]) < 0.0:
        raise ValueError("target damage reward scale must be non-negative")
    if float(data["reward"]["agent_damage_scale"]) > 0.0:
        raise ValueError("agent damage reward scale must be non-positive")
    return data


def _validate_bounds(value: Any, name: str) -> None:
    bounds = [float(component) for component in value]
    if (
        len(bounds) != 6
        or not all(math.isfinite(component) for component in bounds)
        or any(bounds[axis + 3] <= bounds[axis] for axis in range(3))
    ):
        raise ValueError(f"{name} is invalid")
