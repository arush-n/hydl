"""NPC sprint is run (gen 7, row 1a). Switch ``Constructs.npc_walk_sprint_as_run``: ``False`` (gen 5 and the stock) or
``True`` (gen 7).

Owner ruling 2026-10-21: "Sim: sprint = run". Measured on the live 0.7.2 bridge (bridge 6DFD8682, 2026-10-21; the lane's
receipts, not re-measured by this module): the agent body is an NPC (Kweebec_Razorleaf, MaxSpeed 5.0,
``rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json`` agent ``max_speed``). Every gait's wire velocity equals
the npc_walk simulator except sprint: live 5.0 against the simulator's 5.0 x 1.273 = 6.37, because the NPC body cannot exceed
its top speed. The NPC controller never publishes a sprinting state (``jax/world/entities/movement_states.py``,
``sprint_boundary.npc_walk_sprinting`` = ``source_pinned_false_no_npc_writer_sets_true``).

Engine (HEAD): a sprint request is granted while stamina lasts and the travel is forward-dominant (``combat/runtime/step.py``
209 to 213). A granted sprint scales the forward gait by 1.273, drains stamina and blocks regeneration.

Construct: the hook on ``_apply_action`` (``combat/runtime/step.py``, actor zero) rewrites the gait column of the action
array before the original runs. Under ``npc_walk`` (``params.agent_motion_model``, set in ``default_combat_params``) a
requested SPRINT becomes RUN; every other value is passed through bit for bit. The original then takes its run branch:
``sprinting`` is False, the gait is RUN, no stamina is drained and regeneration runs as it does for run. The player model
keeps its sprint (1.273 x), and gen 5 returns the original function.

Not covered: the entity path (``jax/training/multi_actor/locomotion.py``, ``_apply_entity_action``, lines 352 to 360) has its
own sprint block. It is outside the gen-7 render's module scope (``overlay.modules == ("combat/",)``), and the multi-actor
runtime that steps the duel's slot 1 imports the HEAD ``hytalegym.jax.combat``, not the render. Measured and tests:
``tests/sim/test_npc_sprint.py``; ``sim/gen7/mechanics_072.md`` (row 1a).
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import jax.numpy as jnp

from hytalegym.jax.combat.types import ACTION_GAIT, AGENT_MOTION_NPC_WALK, GAIT_RUN, GAIT_SPRINT

SWITCH = "npc_walk_sprint_as_run"
IMPLEMENTED = True


def _enabled(profile: Any) -> bool:
    return bool(profile.constructs.npc_walk_sprint_as_run)


def sprint_as_run(actions: Any, params: Any) -> Any:
    """Return ``actions`` with the gait of every sprint request under npc_walk replaced by RUN.

    The engine rounds the gait the same way (``jnp.round``, then clipped to the gait range), and the clip cannot move a
    value of 3 out of range, so ``round(gait) == GAIT_SPRINT`` is exactly the engine's sprint request.
    """

    gait = actions[:, ACTION_GAIT]
    sprint = jnp.round(gait).astype(jnp.int32) == jnp.int32(GAIT_SPRINT)
    npc_walk = params.agent_motion_model == jnp.int32(AGENT_MOTION_NPC_WALK)
    run = jnp.asarray(GAIT_RUN, dtype=gait.dtype)
    return actions.at[:, ACTION_GAIT].set(jnp.where(npc_walk & sprint, run, gait))


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for gen 7 the ``_apply_action`` that runs a sprint request as a run under npc_walk."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _apply_action(state: Any, actions: Any, pause_draw: Any, params: Any, *rest: Any, **kwargs: Any) -> Any:
        return original(state, sprint_as_run(actions, params), pause_draw, params, *rest, **kwargs)

    return _apply_action
