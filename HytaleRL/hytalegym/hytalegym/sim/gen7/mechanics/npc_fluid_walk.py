"""Water for the npc_walk body (row 10c, F10c). Switch ``Constructs.swimming`` (value ``npc_walk_072``); lane ``lane_swim``, 2026-10-23.

What 0.7.2 does to a Walk-only NPC in water (the learner roles: ``MotionControllerList`` is only ``Walk``, no ``BreathesInWater``):

- ``MotionControllerWalk.postReadPosition``: ``inWater`` from the feet cell; the horizontal multiplier is the FluidFX
  ``HorizontalSpeedMultiplier`` (0.6 for Water). The sim already does this (``world/geometry`` ``movement_medium_result``).
- ``MotionControllerWalk.computeMove``: the fluid vertical cap (``maxVerticalSpeedFluid``, ``MaxSinkSpeedFluid`` default 4.0) is
  selected by ``inWater()`` and stored in a local that is never read again. ``computeNewFallSpeed`` has no fluid branch. So a
  walk NPC in water falls exactly as on land, with no sink cap.
- No swim controller: the Walk mode is the only one a non-``BreathesInWater`` role gets (``getDefaultSpawnMovementModes``).
  FluidFX ``SwimUpSpeed``/``SinkSpeed`` are not read by the Walk controller. The sim does not read them either.

Gen 5 caps the in-fluid fall at ``max_sink_speed_fluid`` (4.0, 0.5.7 ruleset). This construct replaces that cap with the dry cap
for the npc_walk model only, and leaves the player model alone. The dry cap is the sim's own dry value; the 0.7.2 terminal fall
speed itself is not confirmed here (INFERRED: the same physics as on land).

Evidence: javap of the 0.7.2 jar (``MotionControllerWalk`` computeMove, postReadPosition, computeNewFallSpeed; BMCWalk MaxSinkSpeedFluid
default). Not differential-gated: no 0.7.2 water fixture exists and no live probe was run (see ``mechanics_072.md``, row 10c).

Hook: ``_tick_agent_geometry_motion`` (``combat/runtime/agent.py``). The fall cap is read there only through ``maximum_fall_speed``.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any

SWITCH = "swimming"
#: The 0.7.2 value of ``Constructs.swimming``; any other value (gen 5's ``fluidfx_v0_5``) returns the original function.
VALUE = "npc_walk_072"
IMPLEMENTED = True


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` unless ``swimming`` is ``npc_walk_072``; then the motion stage with the in-fluid cap set to the dry cap."""

    if profile.constructs.swimming != VALUE:
        return original
    # Unwrap to the rendered function (auto_step_up and npc_jump_arc also wrap this stage); its globals carry the jnp and the model.
    env = inspect.unwrap(original).__globals__
    jnp = env["jnp"]
    npc_walk = env["AGENT_MOTION_NPC_WALK"]

    @wraps(original)
    def _tick_agent_geometry_motion(
        geometry: Any,
        position: Any,
        velocity: Any,
        impulse: Any,
        residual: Any,
        control_lock: Any,
        grounded_at_start: Any,
        applied_vertical_velocity: Any,
        fall_speed: Any,
        fall_start_y: Any,
        move_speed: Any,
        in_fluid: Any,
        motion_delta: Any,
        params: Any,
    ) -> Any:
        dry_cap_in_fluid = jnp.where(
            params.agent_motion_model == npc_walk,
            params.agent_walk_max_fall_speed,
            params.agent_walk_max_sink_speed_fluid,
        )
        return original(
            geometry, position, velocity, impulse, residual, control_lock, grounded_at_start,
            applied_vertical_velocity, fall_speed, fall_start_y, move_speed, in_fluid, motion_delta,
            params._replace(agent_walk_max_sink_speed_fluid=dry_cap_in_fluid),
        )

    return _tick_agent_geometry_motion
