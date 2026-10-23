"""Automatic step-up for the npc_walk body (row 1f, F1f). Switch ``Constructs.auto_step_up``; lane ``lane_traversal``, 2026-10-23.

Gen 5 has no step: a grounded npc_walk move blocked by a ledge stops at the face (``resolve_aabb_motion`` is a pure sweep).
Gen 7 (this construct): when a grounded tick's horizontal move is blocked, try a raised move. Lift by the walk's
``agent_walk_max_climb_height``, move the same horizontal distance, then settle down. Accept only when (i) there is headroom
for the lift, (ii) the raised move covers the full horizontal distance, and (iii) the settle lands on a floor that is higher
than the start by more than zero and at most the lift. Otherwise the blocked result stands.

Parameter: the 0.5.7 ruleset ``max_climb_height`` 1.3 (``rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json:51``).
The 0.7.2 learner role carries no ``MaxClimbHeight`` (``sim/gen7/data/npc_roles.json``), so 1.3 is INFERRED for 0.7.2.

Evidence status: NOT differential-gated. No 0.7.2 up-step fixture exists (FIDELITY-0.7.2.md row 1d), and no live probe was run.
The unit tests check the geometry behaviour the construct claims (tests/sim/test_auto_step_up.py), not parity with the server.

Hook: ``_tick_agent_geometry_motion`` (``combat/runtime/agent.py``), outermost. Positions and velocities are the rendered
module's own: the resolver and ``jnp`` come from ``original.__globals__`` so no rendered module imports ``hytalegym.jax``.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any

SWITCH = "auto_step_up"
IMPLEMENTED = True
#: Shortfall tolerance in blocks: a move counts as full or blocked within this distance.
TOLERANCE = 1.0e-3


def _shortfall(xp: Any, applied: Any, requested: Any) -> Any:
    """Horizontal distance by which an applied move falls short of the requested one, per row ``[B]``."""

    gap = requested[:, [0, 2]] - applied[:, [0, 2]]
    return xp.sqrt(xp.sum(gap * gap, axis=1))


def step_target(xp: Any, resolve: Any, geometry: Any, position: Any, horizontal: Any, height: Any) -> tuple[Any, Any, Any]:
    """The settled position of a raised move, the accept mask, and the exhaustion mask, per row.

    ``resolve``: the rendered ``_resolve_aabb_motion_result`` (or ``resolve_aabb_motion`` in tests); ``position`` ``[B,3]``;
    ``horizontal`` ``[B,3]`` with y = 0; ``height`` the lift in blocks (scalar or ``[B]``).
    """

    zeros = xp.zeros_like(position[:, 1])
    lift_height = xp.zeros_like(zeros) + height
    lift = xp.stack((zeros, lift_height, zeros), axis=1)
    flat = resolve(geometry, position, horizontal)
    blocked = _shortfall(xp, flat.applied_displacement, horizontal) > TOLERANCE
    head = resolve(geometry, position, lift)
    head_clear = head.applied_displacement[:, 1] >= lift_height - TOLERANCE
    raised = resolve(geometry, position + lift, horizontal)
    raised_clear = _shortfall(xp, raised.applied_displacement, horizontal) <= TOLERANCE
    settle = resolve(geometry, raised.position, -lift)
    rise = settle.position[:, 1] - position[:, 1]
    lands = settle.grounded & (rise > TOLERANCE) & (rise <= lift_height + TOLERANCE)
    accepted = blocked & head_clear & raised_clear & lands
    exhausted = blocked & (head.geometry_exhausted | raised.geometry_exhausted | settle.geometry_exhausted)
    return settle.position, accepted, exhausted


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` unless ``auto_step_up`` is set; then the motion stage with a step-up for grounded npc_walk rows."""

    if not profile.constructs.auto_step_up:
        return original
    # Unwrap to the rendered function: another construct on this function (npc_jump_arc) is listed after this one and wraps it,
    # and that wrapper's __globals__ belong to the construct module, not to the engine.
    env = inspect.unwrap(original).__globals__
    jnp = env["jnp"]
    npc_walk = env["AGENT_MOTION_NPC_WALK"]
    resolve = env["_resolve_aabb_motion_result"]

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
        out = original(
            geometry, position, velocity, impulse, residual, control_lock, grounded_at_start,
            applied_vertical_velocity, fall_speed, fall_start_y, move_speed, in_fluid, motion_delta, params,
        )
        grounded = grounded_at_start & (velocity[:, 1] <= 0.0)
        walk = (
            (jnp.abs(applied_vertical_velocity) <= jnp.float32(1.0e-9))
            & ~control_lock & ~impulse & ~residual & grounded
            & (params.agent_motion_model == npc_walk)
        )
        zeros = jnp.zeros_like(motion_delta)
        horizontal = jnp.stack((velocity[:, 0] * motion_delta, zeros, velocity[:, 2] * motion_delta), axis=1)
        target, accepted, exhausted = step_target(
            jnp, resolve, geometry, position, horizontal, params.agent_walk_max_climb_height,
        )
        accepted = accepted & walk
        dt = jnp.maximum(motion_delta, jnp.finfo(jnp.float32).tiny)
        stepped_velocity = jnp.stack(
            ((target[:, 0] - position[:, 0]) / dt, zeros, (target[:, 2] - position[:, 2]) / dt), axis=1
        )
        position_out = jnp.where(accepted[:, None], target, out[0])
        velocity_out = jnp.where(accepted[:, None], stepped_velocity, out[1])
        grounded_out = out[5] | accepted
        exhausted_out = out[6] | (walk & exhausted)
        return (position_out, velocity_out, out[2], out[3], out[4], grounded_out, exhausted_out, *out[7:])

    return _tick_agent_geometry_motion
