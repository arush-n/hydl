"""Parameterize the JAX environment without hand-building a provider bundle.

``JaxScene`` is the ADK's environment identity, but constructing one by hand
means importing Gym loadout/runtime factories, knowing that ``build()`` prefers
a scene's ``combat_params`` over the spec-derived defaults, and inventing a
64-hex contract digest.  That is the whole reason custom environments felt
out of reach.

:func:`make_scene` closes that gap.  It exposes the knobs the Gym already
accepts but the ADK previously collapsed:

* ``prepare_jax_build_inputs`` forces ``[spec.loadout] * batch``, yet
  ``hytale_0_5_7_loadouts`` takes a *sequence* plus separate ``target_profiles``
  -- so per-environment agents and opponents were upstream-supported and
  ADK-blocked.
* ``arsenal_runtime_config`` accepts ``entity_team_id``, ``entity_role_ids``,
  and ``opponent_controller_mask``; the ADK called it with defaults only.
* ``CombatParams`` has 116 fields (rewards, gravity, health, ranges); only
  ``microticks`` was reachable.

The contract digest is derived from the declared configuration, so an identical
configuration produces an identical name+digest and re-registration is
idempotent.  It identifies the *declaration*, not the Gym code behind it --
upstream dynamics changes are the contract stamp's job, not this digest's.
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import hashlib
import json
import warnings
from typing import Any, Mapping, Sequence

import numpy as np

from hytalegym.jax.combat.arsenal.factory import arsenal_runtime_capacity
from hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7 import (
    PROFILE_NAMES,
    hytale_0_5_7_loadouts,
)
from hytalegym.jax.combat.arsenal.runtime import arsenal_runtime_config
from hytalegym.jax.combat.observation.v3.world_tokens import (
    normalize_world_geometry_policy_config,
)
from hytalegym.jax.combat.types import default_combat_params

from adk.runtime.scenes import (
    JAX_SCENE_PROVIDER_ARGUMENTS,
    SCENE_CONTRACT_SCHEMA,
    JaxScene,
)


#: Every field of the Gym's ``CombatParams``, for validation and discovery.
COMBAT_PARAMETER_NAMES: tuple[str, ...] = tuple(
    default_combat_params(microticks=1)._fields
)

#: Runtime knobs ``arsenal_runtime_config`` accepts beyond the loadout itself.
RUNTIME_CONFIG_ARGUMENTS: tuple[str, ...] = (
    "specialize",
    "entity_team_id",
    "distance_component_selector",
    "sensor_range",
    "backpack_capacity",
    "entity_role_ids",
    "opponent_controller_mask",
)


def list_loadouts() -> tuple[str, ...]:
    """Return the combat profile names a scene may select.

    Published because guessing fails loudly but unhelpfully: the Gym rejects
    ``'iron_bow'`` and only then lists what it would have accepted.
    """

    return tuple(PROFILE_NAMES)


def _digest_component(value: Any) -> Any:
    """Reduce a knob to something canonical JSON can hash by content."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _digest_component(item) for key, item in value.items()}
    if callable(value):
        # A provider is code; its identity is its name, not its bytes.
        module = getattr(value, "__module__", "?")
        qualname = getattr(value, "__qualname__", repr(value))
        return f"callable:{module}.{qualname}"
    fields = getattr(value, "_fields", None)
    if fields is not None:  # NamedTuple, e.g. CombatParams
        return {
            name: _digest_component(getattr(value, name)) for name in fields
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # e.g. WorldGeometryPolicyConfig, whose capacities move the observation.
        return {
            field.name: _digest_component(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return f"opaque:{type(value).__name__}"
    if array.dtype == object:
        if array.ndim == 0:
            return f"opaque:{type(value).__name__}"
        return [_digest_component(item) for item in array.tolist()]
    return {"shape": list(array.shape), "values": array.tolist()}


def _configuration_sha256(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {key: _digest_component(value) for key, value in manifest.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def _normalize_profiles(value: str | Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        profiles: tuple[str, ...] = (value,)
    elif isinstance(value, Sequence):
        profiles = tuple(value)
    else:
        raise TypeError(f"{label} must be a profile name or a sequence of them")
    if not profiles:
        raise ValueError(f"{label} must not be empty")
    unknown = sorted({name for name in profiles if name not in PROFILE_NAMES})
    if unknown:
        raise ValueError(
            f"unknown combat profile(s) in {label}: {unknown}; "
            f"choose from {list_loadouts()}"
        )
    return profiles


def _apply_parameter_overrides(
    params: Any,
    overrides: Mapping[str, Any] | None,
) -> Any:
    if not overrides:
        return params
    unknown = sorted(set(overrides) - set(COMBAT_PARAMETER_NAMES))
    if unknown:
        raise ValueError(
            "unknown combat parameter(s): "
            + ", ".join(unknown)
            + f"; CombatParams declares {len(COMBAT_PARAMETER_NAMES)} fields "
            "(see adk.runtime.scene_builder.COMBAT_PARAMETER_NAMES)"
        )
    if "microticks" in overrides:
        raise ValueError(
            "set microticks through make_scene(microticks=...) so the scene "
            "and the AgentSpec cannot disagree"
        )
    return params._replace(**dict(overrides))


def defensive_providers(extra: dict | None = None) -> dict:
    """Providers for a scene where guard and dodge can actually be measured.

    **Use this for any defensive work.** The default scripted target
    (``opponents=None``) attacks through the legacy damage path, which applies
    damage at ``hytalegym/jax/combat/_step_core.py:419`` as a flat subtraction
    with **no mitigation term** -- ``grep -c guard`` is 0 across
    ``_step_core.py``, ``_target_attack.py`` and ``_agent.py``. Guard
    mitigation exists only in ``mechanics/kernel.py:242`` (``victim_guard``),
    on the arsenal path, which is what an armed opponent uses.

    Measured 2026-08-03, seed 11, 256 ticks, batch 4:

    ======================  ========  =========
    scene                   guard on  guard off
    ======================  ========  =========
    scripted target         414.0     414.0
    armed arsenal opponent   60.0      70.0
    ======================  ========  =========

    Byte-identical on the scripted path, and guard still spends stamina there,
    so it is a strict penalty rather than merely inert.

    Pair with ``opponents=[profile] * batch``::

        scene = kit.make_scene(
            name,
            loadouts=["iron_sword"] * batch,
            opponents=["iron_sword"] * batch,
            providers=defensive_providers(
                dict(OPEN_FLAT_CONTROL_SCENE.environment_kwargs)
            ),
        )
    """

    try:
        # Moved to `opponents.runtime.policy` in the Gym's restructure, and the
        # parent package does not re-export the name, so both spellings are kept
        # to import against the live tree and the frozen snapshot alike.
        from hytalegym.jax.combat.opponents.runtime.policy import (
            first_legal_opponent_ability_slots,
        )
    except ImportError:  # pragma: no cover - depends on the Gym tree on the path
        from hytalegym.jax.combat.opponents.policy import (
            first_legal_opponent_ability_slots,
        )

    providers = dict(extra or {})
    providers["opponent_ability_provider"] = first_legal_opponent_ability_slots
    return providers


def make_scene(
    name: str,
    *,
    loadouts: str | Sequence[str] = "iron_sword",
    opponents: str | Sequence[str] | None = None,
    microticks: int = 1,
    target_active: bool = True,
    parameters: Mapping[str, Any] | None = None,
    world_geometry: Any | None = None,
    spawns: Any | None = None,
    providers: Mapping[str, Any] | None = None,
    **runtime_options: Any,
) -> JaxScene:
    """Build a contract-stamped :class:`JaxScene` from declared parameters.

    ``loadouts`` may be a single profile (applied to every environment) or one
    profile per environment.  When they differ, the scene's runtime owns them
    and ``AgentSpec.loadout`` is *not* consulted -- ``describe_scene`` reports
    what is actually bound.

    ``parameters`` overrides any of the 116 ``CombatParams`` fields, which is
    how reward shaping (``completion_reward``, ``death_reward``,
    ``agent_damage_reward_scale``) and world physics (``world_gravity``,
    ``agent_max_health``) are reached.

    ``runtime_options`` are forwarded to the Gym's ``arsenal_runtime_config``:
    ``entity_team_id``, ``entity_role_ids``, ``opponent_controller_mask``,
    ``sensor_range``, ``backpack_capacity``, ``specialize``, and
    ``distance_component_selector``.

    ``providers`` is the escape hatch to raw environment kwargs for anything
    this signature does not name; keys are checked against
    ``JAX_SCENE_PROVIDER_ARGUMENTS``.

    ``microticks`` must equal the consuming ``AgentSpec.microticks``; ``build()``
    rejects a mismatch rather than silently preferring one.

    Declaring ``opponents`` without a ``world_capability_provider`` warns: the
    scene builds and runs, but the agent cannot perceive the opponent, so
    ``target_visible`` stays 0.0, no attack is ever legal, and reward stays 0.

    ``spawns`` is a composable spawn source (``adk.runtime.spawns``) bound as
    the scene's ``reset_provider``. A Region scene needs one: without it both
    fighters take the flat default height, which on captured terrain is
    underground -- and underground reads exactly like the blind case above.
    """

    unknown_runtime = sorted(set(runtime_options) - set(RUNTIME_CONFIG_ARGUMENTS))
    if unknown_runtime:
        raise ValueError(
            "unknown runtime option(s): "
            + ", ".join(unknown_runtime)
            + f"; arsenal_runtime_config accepts {list(RUNTIME_CONFIG_ARGUMENTS)}"
        )
    agent_profiles = _normalize_profiles(loadouts, "loadouts")
    target_profiles = (
        None if opponents is None else _normalize_profiles(opponents, "opponents")
    )
    if target_profiles is not None and len(target_profiles) != len(agent_profiles):
        if len(target_profiles) == 1:
            target_profiles = target_profiles * len(agent_profiles)
        elif len(agent_profiles) == 1:
            agent_profiles = agent_profiles * len(target_profiles)
        else:
            raise ValueError(
                "loadouts and opponents must describe the same number of "
                f"environments: {len(agent_profiles)} vs {len(target_profiles)}"
            )

    supplied_providers = dict(providers or {})
    unknown_providers = sorted(
        set(supplied_providers) - set(JAX_SCENE_PROVIDER_ARGUMENTS)
    )
    if unknown_providers:
        raise ValueError(
            "unknown provider argument(s): " + ", ".join(unknown_providers)
        )
    if world_geometry is not None:
        if "world_geometry_config" in supplied_providers:
            raise ValueError(
                "pass world geometry once: either world_geometry= or "
                "providers={'world_geometry_config': ...}"
            )
        supplied_providers["world_geometry_config"] = (
            normalize_world_geometry_policy_config(world_geometry)
        )
    if spawns is not None:
        if "reset_provider" in supplied_providers:
            raise ValueError(
                "pass spawn placement once: either spawns= or "
                "providers={'reset_provider': ...}"
            )
        if not callable(spawns):
            raise TypeError(
                "spawns must be a spawn source -- keys -> ArsenalResetBatch; "
                "see adk.runtime.spawns"
            )
        supplied_providers["reset_provider"] = spawns

    # ``world_runtime_provider`` replaces the capability/feature/token trio --
    # make_arsenal_environment rejects binding both -- so either one satisfies
    # this check. Naming only the capability provider here made a Region scene
    # that binds the runtime provider warn falsely.
    perception_providers = ("world_capability_provider", "world_runtime_provider")
    if target_profiles is not None and not any(
        name in supplied_providers for name in perception_providers
    ):
        warnings.warn(
            f"scene {name!r} declares opponents but binds neither "
            "world_capability_provider nor world_runtime_provider, so the "
            "agent cannot perceive them: "
            "combat_f32.target_visible stays 0.0 for the whole episode, no "
            "attack is ever legal, and reward stays 0. Bind one -- e.g. "
            "providers=dict(OPEN_FLAT_CONTROL_SCENE.environment_kwargs) -- or "
            "pass opponents=None if a movement-only scene is intended.",
            RuntimeWarning,
            stacklevel=2,
        )

    # REMOVED 2026-08-03: an inert-opponent warning used to fire here.
    #
    # It warned that declaring `opponents=` without `opponent_ability_provider`
    # left the target unable to swing, on a 2026-08-02 measurement where
    # opponents="iron_mace" pinned agent_health at 105.0 over 192 ticks while
    # the identical scene with opponents=None took it to 82.0.
    #
    # Upstream closed that the same day. Re-measured on the current tree, the
    # same two scenes give agent_health minima of 47.0 (armed, 116.0 damage
    # taken) and 82.0 (scripted, 46.0) -- so the configuration the warning
    # named is now the *harder* one, not the broken one. The warning had become
    # false, and a false warning is worse than none: it would push callers away
    # from the configuration that fights hardest.
    #
    # Do not reinstate without re-measuring both arms.

    params = default_combat_params(
        microticks=microticks,
        target_active=target_active,
    )
    params = _apply_parameter_overrides(params, parameters)

    loadout = hytale_0_5_7_loadouts(
        list(agent_profiles),
        target_profiles=(
            None if target_profiles is None else list(target_profiles)
        ),
    )
    runtime_config = arsenal_runtime_config(loadout, **runtime_options)
    runtime_capacity = arsenal_runtime_capacity(loadout)

    manifest = {
        "schema": SCENE_CONTRACT_SCHEMA,
        "name": name,
        "agent_profiles": list(agent_profiles),
        "target_profiles": (
            None if target_profiles is None else list(target_profiles)
        ),
        "microticks": microticks,
        "target_active": bool(target_active),
        "parameters": dict(parameters or {}),
        "providers": {
            key: supplied_providers[key] for key in sorted(supplied_providers)
        },
        "runtime_options": {
            key: runtime_options[key] for key in sorted(runtime_options)
        },
    }

    homogeneous = len(set(agent_profiles)) == 1
    return JaxScene(
        name=name,
        contract_sha256=_configuration_sha256(manifest),
        environment_kwargs=supplied_providers,
        combat_params=params,
        runtime_config=runtime_config,
        runtime_capacity=runtime_capacity,
        expected_batch=len(agent_profiles),
        expected_loadout=agent_profiles[0] if homogeneous else None,
    )


@dataclass(frozen=True, slots=True)
class SceneDescription:
    """What a scene actually binds, for printing and for run metadata."""

    name: str
    contract_sha256: str
    expected_batch: int | None
    expected_loadout: str | None
    active_providers: tuple[str, ...]
    world_geometry: dict[str, Any] | None
    combat_params_bound: bool
    runtime_config_bound: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contract_sha256": self.contract_sha256,
            "expected_batch": self.expected_batch,
            "expected_loadout": self.expected_loadout,
            "active_providers": list(self.active_providers),
            "world_geometry": self.world_geometry,
            "combat_params_bound": self.combat_params_bound,
            "runtime_config_bound": self.runtime_config_bound,
        }


def describe_scene(scene: JaxScene) -> SceneDescription:
    """Report what a scene binds without reaching into its internals."""

    if not isinstance(scene, JaxScene):
        raise TypeError("scene must be a JaxScene")
    metadata = scene.metadata()
    geometry = scene.environment_kwargs.get("world_geometry_config")
    return SceneDescription(
        name=scene.name,
        contract_sha256=scene.contract_sha256,
        expected_batch=scene.expected_batch,
        expected_loadout=scene.expected_loadout,
        active_providers=tuple(metadata["active_arguments"]),
        world_geometry=(
            None
            if geometry is None
            else {
                field: getattr(geometry, field)
                for field in ("token_capacity", "edge_capacity", "maximum_distance")
            }
        ),
        combat_params_bound=bool(metadata["combat_params_bound"]),
        runtime_config_bound=bool(metadata["runtime_config_bound"]),
    )


__all__ = [
    "COMBAT_PARAMETER_NAMES",
    "RUNTIME_CONFIG_ARGUMENTS",
    "SceneDescription",
    "describe_scene",
    "list_loadouts",
    "make_scene",
]
