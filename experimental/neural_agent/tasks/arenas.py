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
"""

from __future__ import annotations

import numbers
from typing import Any, Mapping, NamedTuple

import jax

FLAT_PURSUIT: str = "flat_pursuit"
COMBAT_BASELINE: str = "combat_baseline"
COMBAT_OPEN_FLAT: str = "combat_open_flat"
#: The legal view sector of the combat arena: 120 degrees, as the certified legal observation uses (2026-10-19).
COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS: float = 2.0943951023931953
#: The arena names that :func:`build_arena` accepts.
ARENA_NAMES: tuple[str, ...] = (FLAT_PURSUIT, COMBAT_BASELINE, COMBAT_OPEN_FLAT)
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


def build_arena(name: str, batch: int, ticks: int, seed: int) -> BuiltArena:
    """Build the named arena with ``batch`` lanes and an episode horizon of ``ticks``.

    Args:
        name: One of :data:`ARENA_NAMES`.
        batch: Number of lanes B, a positive int.
        ticks: Episode horizon in decision ticks, a positive int. ``flat_pursuit``
            accepts 9 to 1000 (its stage config refuses fewer).
        seed: Seed of the first reset keys, an int. ``jax.random.key(seed)`` is
            split into ``batch`` keys.

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
        # The sensor-range term of the entity-row gate (environment.py, the two-entity path): the opponent
        # beyond params.sensor_range is blanked. The view sector is not a capability field; it is the
        # environment's own argument (view_sector_full_angle_radians, below), read by the gate before the
        # capabilities are returned, so it is passed through the scene's environment kwargs (the skill
        # layer's acceptance of perception_v2, 2026-10-19, found the arena line-of-sight only, and its
        # second probe showed a field replaced here after the fact has no effect).
        return open_flat_arsenal_world_capabilities(
            state, params, config=config, legal_observation_sensor_range=True
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
            # 120 degrees: a target behind the agent is not perceived (make_arsenal_environment's own argument).
            "view_sector_full_angle_radians": COMBAT_OPEN_FLAT_VIEW_SECTOR_RADIANS,
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
