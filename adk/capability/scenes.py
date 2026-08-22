"""Scenes built for measurement rather than for play.

A capability test wants the *fewest* moving parts that still exercise the
mechanic.  These builders make that explicit, and they set health high enough
that nothing dies and truncates a run mid-measurement.
"""

from __future__ import annotations

import warnings
from typing import Any

from adk import AgentKit, AgentSpec
from adk.runtime.scenes import OPEN_FLAT_CONTROL_SCENE

#: Health high enough that no measurement ends early because something died.
#: A truncated episode silently shortens the denominator of every rate.
DURABLE = {"agent_max_health": 4000.0, "target_max_health": 4000.0}


def control_scene(
    name: str,
    *,
    batch: int,
    loadout: str = "iron_sword",
    parameters: dict[str, float] | None = None,
) -> Any:
    """An open flat scene with the scripted target and no arsenal opponent.

    Binds ``OPEN_FLAT_CONTROL_SCENE``'s providers, so the agent can perceive
    the world.  Without them ``combat_f32.target_visible`` stays 0.0 for the
    whole episode and every target-conditioned mechanic silently reads dead --
    the failure this project has repeated most often.

    **This scene is not passive: the scripted target attacks.**  A perfectly
    neutral action still yields damage and knockback -- measured at ticks 58
    and 91 of a 120-tick run, each lifting the body off the floor for three
    ticks.  Any test that assumes "do nothing" means "nothing happens" will
    read that as a defect.  Raise ``target_max_health`` rather than expecting
    quiet, and attribute vertical motion to damage before calling it physics.
    """

    kit = AgentKit()
    with warnings.catch_warnings():
        # make_scene warns about opponents=None; that is the intent here.
        warnings.simplefilter("ignore", RuntimeWarning)
        scene = kit.make_scene(
            name,
            loadouts=[loadout] * batch,
            opponents=None,
            providers=dict(OPEN_FLAT_CONTROL_SCENE.environment_kwargs),
            parameters=dict(parameters or DURABLE),
        )
    kit.register(AgentSpec(name=name, scene=scene.name, loadout=loadout))
    return kit.make(name, num_envs=batch)
