"""Projectiles: launch force, gravity, terminal velocity, damage, hitboxes.

* ``config:<id>``: ``Server/ProjectileConfigs/**`` resolved through
  ``Parent`` (launch force, ``Physics``, spawn offset/rotation, model).
* ``projectile:<id>``: legacy ``Server/Projectiles/**`` ballistic rows
  (``MuzzleVelocity``, ``Gravity``, ``Damage``, ``DeadTime``, shot offsets),
  resolved through ``Parent``.
* ``model:<id>``: the hitbox and scale of every model a projectile names
  (config ``Model`` or legacy ``Appearance``) plus everything under
  ``Server/Models/Projectiles/``; the simulator's half-extents come from these.

Per-tier projectile damage is a weapon ``InteractionVars`` entry and lives in
the weapons table.
"""

from __future__ import annotations

from typing import Any

from .archive import GameArchive
from .common import gameplay, resolve_parented

FAMILY = "projectiles"
VERSION = 1
CONFIG_PREFIX = "Server/ProjectileConfigs/"
LEGACY_PREFIX = "Server/Projectiles/"
MODEL_PREFIX = "Server/Models/"
PROJECTILE_MODEL_PREFIX = "Server/Models/Projectiles/"


def _xyz(value: Any) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    return [float(value.get(axis, 0.0)) for axis in ("X", "Y", "Z")]


def _replace_vars(value: Any) -> list[str]:
    """``Replace`` variable names under a config's ``Interactions`` (hit/miss)."""

    found: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("Type") == "Replace" and isinstance(node.get("Var"), str):
                found.add(node["Var"])
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return sorted(found)


def _hitbox(model: dict[str, Any]) -> dict[str, Any] | None:
    box = model.get("HitBox")
    if not isinstance(box, dict):
        return None
    return {"min": _xyz(box.get("Min")), "max": _xyz(box.get("Max"))}


def extract(archive: GameArchive) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    models_needed: set[str] = set()
    configs = archive.index(CONFIG_PREFIX)
    for stem in sorted(configs):
        document, chain = resolve_parented(stem, configs, archive.read_json)
        physics = document.get("Physics") if isinstance(document.get("Physics"), dict) else {}
        rotation = document.get("SpawnRotationOffset")
        model = document.get("Model")
        if isinstance(model, str):
            models_needed.add(model)
        rows[f"config:{stem}"] = {
            "kind": "projectile_config",
            "source": configs[stem],
            "parent_chain": list(chain),
            "model": model,
            "launch_force": document.get("LaunchForce"),
            "physics_type": physics.get("Type"),
            "gravity": physics.get("Gravity"),
            "terminal_velocity_air": physics.get("TerminalVelocityAir"),
            "terminal_velocity_water": physics.get("TerminalVelocityWater"),
            "bounciness": physics.get("Bounciness"),
            "spawn_offset": _xyz(document.get("SpawnOffset")),
            "spawn_rotation_offset": gameplay(rotation) if isinstance(rotation, dict) else None,
            "interaction_vars": _replace_vars(document.get("Interactions")),
            "document": gameplay(
                {key: value for key, value in document.items() if key != "Interactions"}
            ),
        }
    legacy = archive.index(LEGACY_PREFIX)
    for stem in sorted(legacy):
        document, chain = resolve_parented(stem, legacy, archive.read_json)
        appearance = document.get("Appearance")
        if isinstance(appearance, str):
            models_needed.add(appearance)
        rows[f"projectile:{stem}"] = {
            "kind": "legacy_projectile",
            "source": legacy[stem],
            "parent_chain": list(chain),
            "appearance": appearance,
            "muzzle_velocity": document.get("MuzzleVelocity"),
            "terminal_velocity": document.get("TerminalVelocity"),
            "gravity": document.get("Gravity"),
            "damage": document.get("Damage"),
            "dead_time": document.get("DeadTime"),
            "time_to_live": document.get("TimeToLive"),
            "horizontal_center_shot": document.get("HorizontalCenterShot"),
            "vertical_center_shot": document.get("VerticalCenterShot"),
            "depth_shot": document.get("DepthShot"),
            "document": gameplay(document),
        }
    models = archive.index(MODEL_PREFIX)
    model_ids = {stem for stem, path in models.items() if path.startswith(PROJECTILE_MODEL_PREFIX)}
    model_ids |= {stem for stem in models_needed if stem in models}
    for stem in sorted(model_ids):
        model, model_chain = resolve_parented(stem, models, archive.read_json)
        rows[f"model:{stem}"] = {
            "kind": "model",
            "source": models[stem],
            "parent_chain": list(model_chain),
            "hitbox": _hitbox(model),
            "min_scale": model.get("MinScale"),
            "max_scale": model.get("MaxScale"),
            "eye_height": model.get("EyeHeight"),
        }
    return rows
