"""Landing walk resume (0.7.2 NPC walk). Switch ``Constructs.landing_walk_resume``; row 1c (F1c).

Measured on the live 0.7.2 bridge (FIDELITY-0.7.2.md section (o), row 1c): after a jump the NPC lands on tick L and, with
forward held, starts walking on tick L + 3; HEAD's engine walks on L + 1. The saved capture ``single-jump-30`` (seed 570057,
ledge fixture) shows it: grounded from step 11, native z unchanged on steps 12 and 13, first displacement on step 14. The
engine's walk curve starts on step 12; shifted two ticks later it matches native's (max position error 1.2e-5 m with the
hold, against 0.89 m without).

The hold is armed by an accepted jump, not by a landing. The saved capture ``forward-30`` walks off the ledge without a jump
and lands with forward velocity: native walks on the first tick after touchdown, and the at_once engine matches it exactly
(max position error 1.1e-5 m). A version that armed on every landing held that walk for two ticks and failed it (0.99 m).
Arming on the jump is FIDELITY (o)'s inference (a force-pushed state after a jump blocks steering); the two captures
above are its only evidence, not a measured mechanism.

State: ``CombatState.agent_landing_hold_ticks``, an int32 countdown (patches ``landing_walk_state_field`` and
``landing_walk_state_init`` in ``sim/gen7/overlay.py``). A tick that begins grounded holds the planar walk (forward, back,
left, right set to zero) while the countdown is positive, and spends one; an accepted jump sets it to ``HOLD_TICKS``; an
airborne tick leaves it unchanged. The jump request itself is not held. Only the ``npc_walk`` motion model is held, the model
the captures measure (the player model is unmeasured). Gen 5 (``at_once``) returns the original function unchanged.

Hook: ``_apply_action`` (``combat/runtime/step.py``), which ``_prepare_step`` calls once per policy step: one call per engine
tick at ``microticks=1``, the setting of every production path and of the differential. A render with several microticks per
step would count per step.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

SWITCH = "landing_walk_resume"
IMPLEMENTED = True
AT_ONCE = "at_once"
THIRD_TICK = "third_tick"
#: Grounded ticks after an accepted jump's touchdown with the planar walk held (native walks on the third tick after landing).
HOLD_TICKS = 2


def _enabled(profile: Any) -> bool:
    value = profile.constructs.landing_walk_resume
    if value == AT_ONCE:
        return False
    if value != THIRD_TICK:
        raise ValueError(f"unknown landing_walk_resume switch {value!r}")
    return True


def holding(xp: Any, grounded: Any, stored: Any) -> Any:
    """The planar walk is held on a tick that begins grounded while the countdown is positive. ``xp``: jax.numpy or numpy."""

    return grounded & (stored > 0)


def countdown(xp: Any, grounded: Any, jumped: Any, stored: Any) -> Any:
    """The countdown stored for the next tick.

    ``grounded``: contact at the start of the tick; ``jumped``: this tick's jump was accepted, so the contact ends. An accepted
    jump sets ``HOLD_TICKS``; a grounded tick spends one; an airborne tick keeps the stored count for the landing.
    """

    spent = xp.maximum(stored - 1, 0)
    return xp.where(grounded, xp.where(jumped, HOLD_TICKS, spent), stored)


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``third_tick`` the ``_apply_action`` that holds the planar walk after a jump's landing."""

    if not _enabled(profile):
        return original
    env = original.__globals__
    jnp = env["jnp"]
    npc_walk = env["AGENT_MOTION_NPC_WALK"]
    planar_columns = jnp.asarray(
        (env["ACTION_FORWARD"], env["ACTION_BACK"], env["ACTION_LEFT"], env["ACTION_RIGHT"]), dtype=jnp.int32
    )

    @wraps(original)
    def _apply_action(state: Any, actions: Any, pause_draw: Any, params: Any, *rest: Any, **kwargs: Any) -> Any:
        walking_model = params.agent_motion_model == jnp.int32(npc_walk)
        hold = holding(jnp, state.agent_grounded, state.agent_landing_hold_ticks) & walking_model
        columns = jnp.zeros((actions.shape[1],), dtype=jnp.bool_).at[planar_columns].set(True)
        held = jnp.where(hold[:, None] & columns[None, :], jnp.zeros_like(actions), actions)
        updated, attack, accepted = original(state, held, pause_draw, params, *rest, **kwargs)
        jumped = state.agent_grounded & ~updated.agent_grounded
        following = countdown(jnp, state.agent_grounded, jumped, state.agent_landing_hold_ticks)
        following = jnp.where(walking_model, following, jnp.int32(0))
        return updated._replace(agent_landing_hold_ticks=following), attack, accepted

    return _apply_action
