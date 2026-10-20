"""Arena selection for the composition runs: one builder, two arenas.

:func:`build_arena` returns a :class:`BuiltArena`, the object that
``eval/run_composition.py`` consumes. Its ``built`` has ``batch``, ``reset`` and
``step_detailed``, so both ``compile_hierarchical_collector`` and
``session.run_session`` can step it.

Arenas (:data:`ARENA_NAMES`):

* ``flat_pursuit`` delegates to :func:`tasks.flat_arena.build_flat_arena`, the
  pursuit candidate's flat two-actor arena. Combat is closed there, so no actor
  takes damage and DAMAGE_RECEIVED never fires. A
  terminated lane is read as a catch, because no actor can die
  (``flat_arena.FlatPursuitEnvironment.step_detailed``).
* ``combat_baseline`` is ``AgentKit().build('combat/baseline', batch=B)
  .with_episode_limit(ticks).built``, the only agent name that AgentKit's
  registry resolves (``adk/agents/combat/spec.py``). DAMAGE_RECEIVED fires there
  with no injection: ``probes/out/events.json`` (written by
  ``probes/probe_events.py``) measured 17 of 1,024 lane-steps for a neutral
  actor at batch 8 over 128 steps. The composition task's evade trigger is
  therefore real on this arena and absent on ``flat_pursuit``. This arena has
  no catch. ``terminated`` is the environment's own end rule, a death or a
  world error, and the episode limit is published as ``truncated``
  (``adk/runtime/env_adapter.py``, ``BuiltAgent.step_detailed``). The same probe
  run showed one step per engine tick: with a limit of 128, every lane ended on
  step index 127. The scene is ``combat/fail_closed`` (``adk/core/spec.py``),
  which, by the comment in ``adk/environments/config.py``, denies line of sight
  and target selection, so the learner has no attack there (not tested here).
* ``combat_cover`` is the open-flat legal scene (same reset, opponent and sector) with
  solid pillars in the local geometry window (:func:`_combat_cover`). Line of sight is
  occluded by the pillars. The window is 9 by 9 blocks about the spawn block, so the
  pillars sit within 4 blocks of the spawn on each axis, not at the 5 to 15 blocks asked for.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Mapping, NamedTuple

import jax

FLAT_PURSUIT: str = "flat_pursuit"
COMBAT_BASELINE: str = "combat_baseline"
COMBAT_OPEN_FLAT: str = "combat_open_flat"
COMBAT_COVER: str = "combat_cover"
#: The legal view sector of the combat arena: 120 degrees, as the certified legal observation uses (2026-10-19).
COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS: float = 2.0943951023931953
#: The arena names that :func:`build_arena` accepts.
ARENA_NAMES: tuple[str, ...] = (FLAT_PURSUIT, COMBAT_BASELINE, COMBAT_OPEN_FLAT, COMBAT_COVER)
#: The AgentKit name this module registers for the open-flat combat scene, the
#: one shipped scene where the target is visible AND damage events occur
#: (adk/DEV.md scene table: fail_closed never shows the target; probes/out/events.json:
#: DAMAGE_RECEIVED on open_flat 17 of 1,024 lane-steps with a neutral actor).
COMBAT_OPEN_FLAT_AGENT: str = "neural_agent/open_flat_control"
COMBAT_OPEN_FLAT_SCENE: str = "combat/open_flat_control"
#: Our registration of that scene in the Gym's legal observation mode (2026-10-17): the same
#: world capabilities, a runtime config built with ``legal_observation=True`` for the batch.
#: The Gym default stays privileged for other consumers; this architecture must not cheat.
COMBAT_OPEN_FLAT_LEGAL_SCENE: str = "neural_agent/open_flat_legal"
#: The AgentKit name that the combat baseline arena builds.
COMBAT_BASELINE_AGENT: str = "combat/baseline"
COMBAT_BASELINE_SCHEMA: str = "experimental-neural-agent-combat-baseline-v1"
COMBAT_OPEN_FLAT_SCHEMA: str = "experimental-neural-agent-combat-open-flat-v2"  # v2: legal observation mode
#: combat_cover (task #98, 2026-10-20): the open-flat legal scene plus solid pillars in the local geometry window.
#: ``build_arena(..., loadout=...)`` selects one of COMBAT_COVER_LOADOUTS; the opponent's loadout never changes.
COMBAT_COVER_DEFAULT_LOADOUT: str = "iron_sword"
COMBAT_COVER_LOADOUTS: tuple[str, ...] = (COMBAT_COVER_DEFAULT_LOADOUT, "iron_shortbow")
#: The default loadout's AgentKit names. Other loadouts use neural_agent/cover_<loadout> and
#: neural_agent/cover_legal_<loadout>, because agent names are process-wide.
COMBAT_COVER_AGENT: str = "neural_agent/cover_iron_sword"
COMBAT_COVER_SCENE: str = "neural_agent/cover_legal_iron_sword"
COMBAT_COVER_SCHEMA: str = "experimental-neural-agent-combat-cover-v1"
#: Pillar columns as block offsets (dx, dz) from the spawn block (the agent stands at its centre). Each column is
#: one block square and COMBAT_COVER_PILLAR_HEIGHT blocks high. Every cell lies inside the local window
#: (hytalegym.geometry.contract CELL_RADIUS 4). No column overlaps the default target's spawn footprint
#: (block (0, -2)), so the default reset starts clear. The radii are in the description record.
COMBAT_COVER_PILLARS: tuple[tuple[int, int], ...] = ((3, 0), (1, 3), (-1, 3), (-3, 0), (-1, -3), (1, -3))
COMBAT_COVER_PILLAR_HEIGHT: int = 3


class BuiltArena(NamedTuple):
    """One built arena, ready for the composition runners.

    Attributes:
        built: The environment, with ``batch``, ``reset(keys)`` and ``step_detailed``.
        reset_keys: PRNG keys with leading axis ``batch``, for the first reset.
        learner_role_index: Physical actor row of the learner.
        opponent_role_index: Physical actor row of the opponent.
        catch_from_terminated: True when a terminated lane is a catch
            (``flat_pursuit``). False when ``terminated`` is a death or a world
            error (``combat_baseline``), so the catch-based task columns cannot
            be measured on that arena.
        description: The construction record. :meth:`describe` returns a copy.
    """

    built: Any
    reset_keys: jax.Array
    learner_role_index: int
    opponent_role_index: int
    catch_from_terminated: bool
    description: Mapping[str, Any]

    def describe(self) -> dict[str, Any]:
        """Return a shallow copy of the construction record."""

        return dict(self.description)


def _whole(value: Any, name: str, *, minimum: int | None) -> int:
    """Return ``value`` as an int, rejecting bools, non-integers and values below ``minimum``."""

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


def build_arena(
    name: str,
    batch: int,
    ticks: int,
    seed: int,
    *,
    loadout: str | None = None,
    arrows: int | None = None,
) -> BuiltArena:
    """Build the named arena with ``batch`` lanes and an episode horizon of ``ticks``.

    Args:
        name: One of :data:`ARENA_NAMES`.
        batch: Number of lanes B, a positive int.
        ticks: Episode horizon in decision ticks, a positive int. ``flat_pursuit``
            accepts 9 to 1000 (its stage config refuses fewer).
        seed: Seed of the first reset keys, an int. ``jax.random.key(seed)`` is
            split into ``batch`` keys.
        loadout: The agent's loadout, for ``combat_cover`` only, one of
            :data:`COMBAT_COVER_LOADOUTS`. ``None`` keeps the default ``iron_sword``.
            Any other arena refuses a loadout.
        arrows: The agent's arrow stack at reset, for ``combat_cover`` only: a non-negative int, the quantity of
            its ``Weapon_Arrow_Crude`` stack. ``None`` keeps the reset's bare inventory, which holds no arrows, so a
            shortbow tap spawns nothing. A positive count on a loadout with no arrow stack (``iron_sword``) raises.
            Any other arena refuses arrows.

    Returns:
        A :class:`BuiltArena`. ``describe()["arena"]`` is ``name``.

    Raises:
        ValueError: If ``name`` is unknown, or a count is out of range.
        TypeError: If ``batch``, ``ticks`` or ``seed`` is not an integer.
    """

    if name not in ARENA_NAMES:
        raise ValueError(f"unknown arena {name!r}; expected one of {ARENA_NAMES}")
    lanes = _whole(batch, "batch", minimum=1)
    horizon = _whole(ticks, "ticks", minimum=1)
    root = _whole(seed, "seed", minimum=None)
    if loadout is not None and name != COMBAT_COVER:
        raise ValueError(f"loadout is selectable for {COMBAT_COVER} only, not {name!r}")
    if arrows is not None and name != COMBAT_COVER:
        raise ValueError(f"arrows is selectable for {COMBAT_COVER} only, not {name!r}")
    count = None if arrows is None else _whole(arrows, "arrows", minimum=0)
    if name == COMBAT_COVER:
        profile = COMBAT_COVER_DEFAULT_LOADOUT if loadout is None else loadout
        if profile not in COMBAT_COVER_LOADOUTS:
            raise ValueError(f"loadout must be one of {COMBAT_COVER_LOADOUTS}, got {loadout!r}")
        return _combat_cover(lanes, horizon, root, profile, count)
    if name == FLAT_PURSUIT:
        return _flat_pursuit(lanes, horizon, root)
    if name == COMBAT_OPEN_FLAT:
        return _combat_open_flat(lanes, horizon, root)
    return _combat_baseline(lanes, horizon, root)


def _flat_pursuit(batch: int, ticks: int, seed: int) -> BuiltArena:
    """Delegate to the flat pursuit builder and keep its record."""

    from experimental.neural_agent.tasks.flat_arena import build_flat_arena

    flat = build_flat_arena(batch, seed, ticks=ticks)
    return BuiltArena(
        built=flat.built,
        reset_keys=flat.reset_keys,
        learner_role_index=flat.learner_role_index,
        opponent_role_index=flat.opponent_role_index,
        catch_from_terminated=True,
        description={"arena": FLAT_PURSUIT, **flat.describe()},
    )


def _combat_baseline(batch: int, ticks: int, seed: int) -> BuiltArena:
    """Build ``combat/baseline`` through AgentKit, with an episode limit of ``ticks``."""

    from adk.api import AgentKit
    from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY

    handle = AgentKit().build(COMBAT_BASELINE_AGENT, batch=batch).with_episode_limit(ticks)
    description = {
        "schema": COMBAT_BASELINE_SCHEMA,
        "arena": COMBAT_BASELINE,
        "built_by": (
            f"adk.api.AgentKit().build({COMBAT_BASELINE_AGENT!r}, batch=B)"
            ".with_episode_limit(ticks).built"
        ),
        "agent": COMBAT_BASELINE_AGENT,
        "scene": handle.spec.scene,
        "loadout": handle.spec.loadout,
        "batch": batch,
        "seed": seed,
        "ticks": ticks,
        "terminated": "death or world error (adk/runtime/env_adapter.py BuiltAgent.step_detailed); not a catch",
        "truncated": "the episode limit of `ticks` steps",
        "damage_received": (
            "fires without injection: probes/out/events.json, 17 of 1,024 lane-steps, "
            "neutral actor, batch 8, 128 steps"
        ),
        "learner_role_index": AGENT_ENTITY,
        "opponent_role_index": TARGET_ENTITY,
    }
    return BuiltArena(
        built=handle.built,
        reset_keys=jax.random.split(jax.random.key(seed), batch),
        learner_role_index=AGENT_ENTITY,
        opponent_role_index=TARGET_ENTITY,
        catch_from_terminated=False,
        description=description,
    )


__all__ = [
    "ARENA_NAMES",
    "BuiltArena",
    "COMBAT_BASELINE",
    "COMBAT_BASELINE_AGENT",
    "COMBAT_BASELINE_SCHEMA",
    "FLAT_PURSUIT",
    "build_arena",
    "COMBAT_OPEN_FLAT",
    "COMBAT_OPEN_FLAT_AGENT",
    "COMBAT_OPEN_FLAT_SCENE",
    "COMBAT_OPEN_FLAT_LEGAL_SCENE",
    "COMBAT_OPEN_FLAT_SCHEMA",
    "COMBAT_COVER",
    "COMBAT_COVER_AGENT",
    "COMBAT_COVER_SCENE",
    "COMBAT_COVER_SCHEMA",
    "COMBAT_COVER_LOADOUTS",
]

def _combat_open_flat(batch: int, ticks: int, seed: int) -> BuiltArena:
    """Build the open-flat combat scene through AgentKit, with an episode limit of ``ticks``.

    Registers ``COMBAT_OPEN_FLAT_AGENT`` on a fresh kit, bound to ``neural_agent/open_flat_legal``:
    the shipped ``combat/open_flat_control`` scene in the Gym's legal observation mode (2026-10-17). This is the arena the composition task
    needs: the target is visible there and DAMAGE_RECEIVED fires, so a damage-triggered
    evade, a resumed pursuit and a re-engagement are all observable. ``combat/baseline``
    binds the fail_closed scene, where the target is never visible (adk/DEV.md).
    """

    import hashlib

    from adk.api import AgentKit
    from adk.core.spec import AgentSpec
    from adk.runtime.scenes import OPEN_FLAT_CONTROL_SCENE, JaxScene
    from hytalegym.jax.combat import arsenal_runtime_config, hytale_0_5_7_loadouts
    from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY
    from hytalegym.jax.combat.arsenal.environment import open_flat_arsenal_world_capabilities

    kit = AgentKit()
    spec = AgentSpec(name=COMBAT_OPEN_FLAT_AGENT, scene=COMBAT_OPEN_FLAT_LEGAL_SCENE)
    # The shipped open-flat scene, in the Gym's legal observation mode (2026-10-17): the same
    # world capabilities, and a runtime config for this batch with legal_observation=True.
    # adk's build() takes a scene's runtime_config over the default one (env_adapter.py).
    # Scenes live on the kit instance while agent names are process-wide, so the scene is
    # registered on every fresh kit (register_scene is idempotent for an identical scene).
    def _open_flat_legal_capabilities(state, params, *, config=None):
        # Both legal-observation terms of the entity-row gate, as the open-flat fixture's own opt-in keywords
        # (environment.py, open_flat_arsenal_world_capabilities, the two-entity path): the opponent beyond
        # params.sensor_range is blanked, and the 120-degree sector centred on the agent's head yaw hides a
        # target behind the agent. The sector has to enter the provider call: the skill layer's probe at
        # 9159cfd showed that a field replaced on the returned capabilities (ff7ef22) had no effect, and the
        # environment factory takes no sector argument (adk/runtime/scenes.py lists its provider arguments).
        return open_flat_arsenal_world_capabilities(
            state,
            params,
            config=config,
            legal_observation_sensor_range=True,
            view_sector_full_angle_radians=COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS,
        )

    legal_scene = JaxScene(
        name=COMBAT_OPEN_FLAT_LEGAL_SCENE,
        contract_sha256=hashlib.sha256(
            f"{OPEN_FLAT_CONTROL_SCENE.contract_sha256}:legal_observation=true"
            f":view_sector_full_angle_radians={COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS!r}"
            ":legal_observation_sensor_range=true".encode("ascii")
        ).hexdigest(),
        environment_kwargs={
            **OPEN_FLAT_CONTROL_SCENE.environment_kwargs,
            "world_capability_provider": _open_flat_legal_capabilities,
        },
        runtime_config=arsenal_runtime_config(
            hytale_0_5_7_loadouts([spec.loadout] * batch), legal_observation=True
        ),
        expected_batch=batch,
        expected_loadout=spec.loadout,
    )
    kit.register_scene(legal_scene)
    if COMBAT_OPEN_FLAT_AGENT not in kit.list_agents():
        kit.register(spec)
    handle = kit.build(COMBAT_OPEN_FLAT_AGENT, batch=batch).with_episode_limit(ticks)
    description = {
        "schema": COMBAT_OPEN_FLAT_SCHEMA,
        "arena": COMBAT_OPEN_FLAT,
        "built_by": (
            f"AgentKit().register_scene(<{COMBAT_OPEN_FLAT_SCENE} with legal_observation=True>)"
            f".register(AgentSpec({COMBAT_OPEN_FLAT_AGENT!r}, scene={COMBAT_OPEN_FLAT_LEGAL_SCENE!r})).build(..., batch=B)"
            ".with_episode_limit(ticks).built"
        ),
        "agent": COMBAT_OPEN_FLAT_AGENT,
        "scene": handle.spec.scene,
        "base_scene": COMBAT_OPEN_FLAT_SCENE,
        "legal_observation": True,
        "view_sector_full_angle_radians": COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS,
        "legal_observation_sensor_range": True,
        "loadout": handle.spec.loadout,
        "batch": batch,
        "seed": seed,
        "ticks": ticks,
        "terminated": "death or world error (adk/runtime/env_adapter.py BuiltAgent.step_detailed); not a catch",
        "truncated": "the episode limit of `ticks` steps",
        "damage_received": (
            "fires without injection: probes/out/events.json, 17 of 1,024 lane-steps, "
            "neutral actor, batch 8, 128 steps"
        ),
        "learner_role_index": AGENT_ENTITY,
        "opponent_role_index": TARGET_ENTITY,
    }
    return BuiltArena(
        built=handle.built,
        reset_keys=jax.random.split(jax.random.key(seed), batch),
        learner_role_index=AGENT_ENTITY,
        opponent_role_index=TARGET_ENTITY,
        catch_from_terminated=False,
        description=description,
    )


def _cover_origin(params: Any) -> tuple[int, int, int]:
    """Return the block that holds the spawn: the window's centre cell, ``floor(agent_spawn)``."""

    return tuple(math.floor(float(value)) for value in params.agent_spawn)


def _cover_geometry(params: Any) -> Any:
    """Return combat_cover's local geometry: a solid floor over the window, and the pillar columns.

    The window spans ``CELL_RADIUS`` blocks each way about :func:`_cover_origin`, 9 by 9 by 9 cells.
    The floor is the layer one block below the spawn, so its top face is the agent's feet at
    ``agent_spawn[1]``. Pillar cells are solid and opaque. The numpy frame follows the engine's exact-floor
    test fixture (``_flat_exact_geometry`` in tests/jax/combat/arsenal/tests/environment).
    """

    import numpy as np

    from hytalegym.geometry.contract import CELL_RADIUS, FLAG_OPAQUE, FLAG_SOLID, cell_index, empty_geometry
    from hytalegym.jax.world import geometry_state_from_numpy

    frame = empty_geometry()
    frame["available"] = 1
    frame["exact_collision_shapes"] = 1
    frame["origin"][:] = _cover_origin(params)
    frame["cell_mask"][:] = 1
    frame["agent_bounds"][:] = np.asarray(params.agent_bounds)
    frame["target_bounds"][:] = np.asarray(params.target_bounds)
    frame["agent_los_offset"][:] = np.asarray(params.agent_eye_offset)
    frame["target_los_offset"][:] = np.asarray(params.target_eye_offset)
    solid = [
        (x, -1, z)
        for x in range(-CELL_RADIUS, CELL_RADIUS + 1)
        for z in range(-CELL_RADIUS, CELL_RADIUS + 1)
    ]
    solid += [
        (dx, y, dz) for dx, dz in COMBAT_COVER_PILLARS for y in range(COMBAT_COVER_PILLAR_HEIGHT)
    ]
    for x, y, z in solid:
        index = cell_index(x, y, z)
        frame["flags"][index] = FLAG_SOLID | FLAG_OPAQUE
        frame["collision_boxes"][index, 0] = (0, 0, 0, 1, 1, 1)
        frame["collision_box_mask"][index, 0] = 1
    return geometry_state_from_numpy(frame)


def _combat_cover(batch: int, ticks: int, seed: int, loadout: str, arrows: int | None = None) -> BuiltArena:
    """Build combat_cover through AgentKit, with an episode limit of ``ticks``.

    The scene binds ``geometry_provider`` (:func:`_cover_geometry`) and a world-capability provider that
    calls ``geometry_arsenal_world_capabilities`` with the 120-degree sector and the sensor-range flag that
    combat_open_flat's legal scene uses. The reset is the default one and the opponent is the default melee
    NPC, so the agent's loadout is the only input that changes. The environment calls the provider with
    the state and the params only, and the geometry path needs the runtime config, so the closure binds this
    batch's config. With ``arrows``, every reset seeds the agent's arrow stack (``inventory_reset_provider``), and the
    names, the manifest and the description carry the count; ``None`` leaves all of them as the default build's.
    """

    import hashlib
    import json

    from adk.api import AgentKit
    from adk.core.spec import AgentSpec
    from adk.runtime.scenes import JaxScene
    from hytalegym.geometry.contract import CELL_RADIUS
    from hytalegym.jax.combat import arsenal_runtime_config, hytale_0_5_7_loadouts
    from hytalegym.jax.combat.arsenal.environment import geometry_arsenal_world_capabilities
    from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY, default_combat_params

    # Agent names are process-wide and the scene's environment kwargs differ with arrows, so an arrows build gets its own.
    suffix = "" if arrows is None else f"_arrows{arrows}"
    agent_name = f"neural_agent/cover_{loadout}{suffix}"
    scene_name = f"neural_agent/cover_legal_{loadout}{suffix}"
    spec = AgentSpec(name=agent_name, scene=scene_name, loadout=loadout)
    params = default_combat_params(microticks=spec.microticks)
    origin = _cover_origin(params)
    geometry = _cover_geometry(params)
    runtime = arsenal_runtime_config(hytale_0_5_7_loadouts([loadout] * batch), legal_observation=True)

    def _cover_capabilities(state, combat_params):
        return geometry_arsenal_world_capabilities(
            state,
            combat_params,
            geometry=geometry,
            config=runtime,
            view_sector_full_angle_radians=COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS,
            legal_observation_sensor_range=True,
        )

    manifest = {
        "schema": COMBAT_COVER_SCHEMA,
        "name": scene_name,
        "base": "combat/open_flat_control reset, opponent and legal observation",
        "geometry": {
            "origin": list(origin),
            "cell_radius": CELL_RADIUS,
            "floor_offset_y": -1,
            "pillars": [list(column) for column in COMBAT_COVER_PILLARS],
            "pillar_height": COMBAT_COVER_PILLAR_HEIGHT,
        },
        "world_capabilities": "geometry_arsenal_world_capabilities",
        "view_sector_full_angle_radians": COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS,
        "legal_observation": True,
        "legal_observation_sensor_range": True,
        "loadout": loadout,
    }
    environment_kwargs = {
        "geometry_provider": geometry,
        "world_capability_provider": _cover_capabilities,
    }
    if arrows is not None:
        from experimental.neural_agent.tasks.duel_arena import _arrow_inventory

        manifest["arrows"] = arrows
        # The agent holds the stack and the opponent none (count 0). The environment calls this provider from every
        # reset, and the autoreset of the collectors resets through the same call.
        seeded = _arrow_inventory(runtime, (arrows, 0))

        def _cover_inventory(keys, config):
            return seeded

        environment_kwargs["inventory_reset_provider"] = _cover_inventory
    contract = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    scene = JaxScene(
        name=scene_name,
        contract_sha256=contract,
        environment_kwargs=environment_kwargs,
        runtime_config=runtime,
        expected_batch=batch,
        expected_loadout=loadout,
    )
    kit = AgentKit()
    kit.register_scene(scene)
    if agent_name not in kit.list_agents():
        kit.register(spec)
    handle = kit.build(agent_name, batch=batch).with_episode_limit(ticks)
    radii = [math.hypot(dx, dz) for dx, dz in COMBAT_COVER_PILLARS]
    description = {
        "schema": COMBAT_COVER_SCHEMA,
        "arena": COMBAT_COVER,
        "built_by": (
            f"AgentKit().register_scene(<{scene_name}: geometry_provider + "
            "geometry_arsenal_world_capabilities, legal_observation=True>)"
            f".register(AgentSpec({agent_name!r}, scene={scene_name!r})).build(..., batch=B)"
            ".with_episode_limit(ticks).built"
        ),
        "agent": agent_name,
        "scene": scene_name,
        "contract_sha256": contract,
        "legal_observation": True,
        "view_sector_full_angle_radians": COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS,
        "legal_observation_sensor_range": True,
        "window_origin": list(origin),
        "window_radius_blocks": CELL_RADIUS,
        "pillars": [
            {
                "offset": [dx, dz],
                "radius_blocks": math.hypot(dx, dz),
                "angle_degrees": math.degrees(math.atan2(dz, dx)),
            }
            for dx, dz in COMBAT_COVER_PILLARS
        ],
        "pillar_count": len(COMBAT_COVER_PILLARS),
        "pillar_size_blocks": [1, 1],
        "pillar_height_blocks": COMBAT_COVER_PILLAR_HEIGHT,
        "pillar_radius_range_blocks": [min(radii), max(radii)],
        "radius_note": (
            "5 to 15 blocks is beyond the local geometry window: a cell outside origin +/- CELL_RADIUS is "
            "unavailable, and a line of sight through it is not clear (hytalegym/jax/world/perception/los.py)"
        ),
        "loadout": loadout,
        "opponent": "combat_open_flat's default melee NPC; no opponent provider, loadout unchanged",
        "batch": batch,
        "seed": seed,
        "ticks": ticks,
        "terminated": "death or world error (adk/runtime/env_adapter.py BuiltAgent.step_detailed); not a catch",
        "truncated": "the episode limit of `ticks` steps",
        "damage_received": "not probed on this arena; the composition run's damage column is the measurement",
        "learner_role_index": AGENT_ENTITY,
        "opponent_role_index": TARGET_ENTITY,
    }
    if arrows is not None:
        description["arrows"] = arrows
    return BuiltArena(
        built=handle.built,
        reset_keys=jax.random.split(jax.random.key(seed), batch),
        learner_role_index=AGENT_ENTITY,
        opponent_role_index=TARGET_ENTITY,
        catch_from_terminated=False,
        description=description,
    )
