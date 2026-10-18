"""NPC learner jump arc: the server's fall-speed integration. Switch ``Constructs.npc_jump_arc``; row 1c (F1c).

The 0.7.2 NPC walk controller carries a jump as a fixed applied vertical velocity plus an external vertical speed
that starts at zero. Each tick it moves by the sum times the tick's duration, and then updates the external speed
with ``MotionControllerWalk.computeNewFallSpeed``: ``e += -PhysicsMath.getAcceleration(e, T) * dt``, where
``getAcceleration`` is ``32 * (1 - |e / T|^3)`` and the terminal speed ``T = PhysicsMath.getTerminalVelocity(mass,
0.001225, hitbox width x depth, drag coefficient) = sqrt(64 * 1000 * mass / (0.001225 * 1e6 * area * drag))``.
A model without ``PhysicsValues`` (the learner's ``Kweebec_Sapling``) takes the ``ModelAsset`` default (mass 68, drag
0.5), and its 0.6 x 0.6 hitbox gives T = 140.488 m/s. So the vertical speed on airborne step k is
``v_k = v_(k-1) - 32 * (1 - |(v_(k-1) - v_jump) / T|^3) * dt_(k-1)``: the previous tick's duration and a cubic drag
on the speed gained since takeoff. In double precision this reproduces four live 0.7.2 jump captures
(``native_motion_differential``, Kweebec_Razorleaf, ledge fixture) to 1e-13 in speed and position.

The engine's legacy (non walk-controller) branch of ``runtime/agent._tick_agent_geometry_motion``, which carries a
jump until it lands, integrates ``v_k = v_(k-1) - g * dt_k``. The two differ through both terms. The duration term
needs varying tick durations (the native replays), so it vanishes on the constant gen-7 clock. The drag term
does not, but it is small: 32 * (32 t / T)^3 integrates to about 0.07 mm over a jump in place (0.33 s) and about
2 cm over a one-second fall.

Gen 7 (``physics_math``) hooks two functions:

- ``capture_previous_delta`` wraps ``runtime/step._microtick`` and holds ``state.last_motion_delta_seconds`` (the
  previous microtick's duration; ``_microtick`` overwrites it before the agent moves) for the call;
- ``fall_speed_integration`` wraps ``runtime/agent._tick_agent_geometry_motion``: on ``npc_walk`` rows of the legacy
  branch that are airborne after the jump impulse (a jump's applied vertical velocity, no knockback lock, no
  impulse or residual this step, not grounded at the start), it adds ``g * dt_k - g * (1 - r) * dt_(k-1)`` to the
  vertical speed before the original integrates, so the original's ``v - g * dt_k`` is the server's ``v_k`` and its
  displacement ``v_k * dt_k``. T comes from the engine's agent bounds and the two ``ModelAsset`` defaults.

Outside a microtick (no captured duration) the original runs unchanged. Not changed: the walk controller's fall
(``accelerateDragCapped`` over the current tick; the ledge capture already matches), the terminal clamp (never
reached on a jump: |e| < T for the first 4 s of a fall), the no-geometry integrator, knockback arcs and the player
model. Gen 5 (``constant_gravity``) returns both originals.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

SWITCH = "npc_jump_arc"
IMPLEMENTED = True
CONSTANT_GRAVITY = "constant_gravity"
PHYSICS_MATH = "physics_math"
#: ``ModelAsset``'s default ``PhysicsValues(68.0, 0.5, false)``, used by a model that authors none.
MODEL_DEFAULT_MASS = 68.0
MODEL_DEFAULT_DRAG_COEFFICIENT = 0.5
#: ``computeNewFallSpeed``'s air density (``PhysicsMath.getRelativeDensity``).
AIR_DENSITY = 0.001225

#: ``state.last_motion_delta_seconds`` of each ``_microtick`` being traced (innermost last).
_PREVIOUS_DELTAS: list[Any] = []


def _enabled(profile: Any) -> bool:
    value = profile.constructs.npc_jump_arc
    if value == CONSTANT_GRAVITY:
        return False
    if value != PHYSICS_MATH:
        raise ValueError(f"unknown npc_jump_arc switch {value!r}")
    return True


def terminal_speed(width: Any, depth: Any, mass: float = MODEL_DEFAULT_MASS,
                   drag: float = MODEL_DEFAULT_DRAG_COEFFICIENT) -> Any:
    """``PhysicsMath.getTerminalVelocity(mass, 0.001225, |width * depth|, drag)``."""

    return (64.0 * 1000.0 * mass / (AIR_DENSITY * 1.0e6 * abs(width * depth) * drag)) ** 0.5


def capture_previous_delta(original: Any, profile: Any) -> Any:
    """Hook for ``_microtick``: expose the previous tick's duration to the agent's motion."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _microtick(state, *args, **kwargs):
        _PREVIOUS_DELTAS.append(state.last_motion_delta_seconds)
        try:
            return original(state, *args, **kwargs)
        finally:
            _PREVIOUS_DELTAS.pop()

    return _microtick


def fall_speed_integration(original: Any, profile: Any) -> Any:
    """Hook for ``_tick_agent_geometry_motion``: an NPC jump's vertical speed as the server integrates it."""

    if not _enabled(profile):
        return original
    env = original.__globals__
    jnp = env["jnp"]
    npc_walk = env["AGENT_MOTION_NPC_WALK"]

    @wraps(original)
    def _tick_agent_geometry_motion(
        geometry,
        position,
        velocity,
        impulse,
        residual,
        control_lock,
        grounded_at_start,
        applied_vertical_velocity,
        *rest,
    ):
        if _PREVIOUS_DELTAS:
            # rest = fall_speed, fall_start_y, move_speed, in_fluid, motion_delta, params
            motion_delta, params = rest[-2], rest[-1]
            bounds = params.agent_bounds
            terminal = terminal_speed(bounds[3] - bounds[0], bounds[5] - bounds[2])
            drag = jnp.abs((velocity[:, 1] - applied_vertical_velocity) / terminal) ** 3
            jump_airborne = (
                (params.agent_motion_model == jnp.int32(npc_walk))
                & (jnp.abs(applied_vertical_velocity) > jnp.float32(1.0e-9))
                & ~control_lock
                & ~impulse
                & ~residual
                & ~grounded_at_start
            )
            velocity = velocity.at[:, 1].add(
                jnp.where(
                    jump_airborne,
                    params.world_gravity * (motion_delta - (1.0 - drag) * _PREVIOUS_DELTAS[-1]),
                    jnp.float32(0.0),
                )
            )
        return original(
            geometry,
            position,
            velocity,
            impulse,
            residual,
            control_lock,
            grounded_at_start,
            applied_vertical_velocity,
            *rest,
        )

    return _tick_agent_geometry_motion
