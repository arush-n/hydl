"""NPC jump carries its take-off velocity (gen 7, row 1c). Switch ``Constructs.npc_jump_carry``: ``False`` (gen 5, HEAD) or
``True`` (gen 7). Owner ruling 2026-10-22: the bridge's jump keeps the body's horizontal momentum.

Measured on the live 0.7.2 bridge DBB3736B (the scratch bridge with the change; learner Kweebec_Razorleaf, forward run held,
a jump request every 30 ticks). Receipt ``HytaleRL/artifacts/fidelity_evidence/gen7_probes/knockback_landing/
knockback_landing_jump_Kweebec_Razorleaf_carry_DBB3736B.json``: two seeds, five jumps each, and all eight jumps give the
same horizontal speed on every tick (velocity field, blocks per second; the report's displacement column is noisier). The
request is on tick T = 40 and the touchdown on L = 50:

- take-off: 5.000 (the run speed; the velocity of the request tick);
- airborne, T+1 .. L-1: 4.608, 4.247, 3.914, 3.607, 3.324, 3.064, 2.823, 2.602, 2.398; each is the previous value times 0.9216
  (0.96 squared), with no steering whatever the input does;
- touchdown L: 2.210 (the air factor still applies on the touchdown tick);
- grounded, L+1 .. L+3: 1.486, 0.999, 0.671 (each is the previous value times 0.6724, which is 0.82 squared);
- L+4: 0.203, which is 0.45 times the ground factor applied to 0.671 (see below);
- L+5: 5.000, the run speed at once, with no ramp.

HEAD already has these ratios. A knockback's horizontal force is damped by ``_damp_force_velocity`` (``jax/combat/runtime/
agent.py``): the factor is ``air_drag_min`` 0.96 below 6 blocks per second (``ground_drag_base`` 0.82 on a grounded tick)
raised to ``reference_ticks_per_second / ticks_per_second`` = 60 / 30, with a 0.1 per-axis deadzone (rulesets/hytale_0_5_7
``walk_controller.force_velocity_damping``). That is 0.9216 and 0.6724, the live ratios exactly. The force path also holds
the move speed and ignores steering (``_tick_agent_motion``, ``force_active``). So the construct only sets the force at the
take-off and adds the two things HEAD's release does not do: the release on L+5 and the L+4 factor.

HEAD's release is the deadzone: the force is zeroed only when its damped value falls to 0.1, which from the chain above is
after L+7 (the walk on L+8). The live release is on L+5, so the construct times it.

Construct: a hook on ``_apply_action`` (``combat/runtime/step.py``) under ``npc_walk``. Counter ``CombatState.agent_jump_carry_ticks``
(int32; 0 idle, otherwise 1 + the number of grounded ticks since take-off, so L+k starts with k + 1):

- an accepted jump with horizontal speed > 0 sets the force velocity to the take-off horizontal velocity (y zero) and the
  counter to 1;
- a tick that starts grounded advances a live counter by one; on L+4 (counter 5) the force's horizontal part is multiplied by
  ``TAIL_FACTOR`` 0.45; on L+5 (counter 6) the force is cleared and the counter goes to 0, so the walk resumes with the move speed
  the force path held (5.0);
- a knockback during the carry (the knockback control lock is set) ends it: the counter is cleared, so the knockback's own
  force is not scaled or cleared by the tail and the release;
- a standing jump (zero take-off speed) is not carried: ``landing_walk_resume`` holds it, as before (saved capture
  ``single-jump-30``, a standing jump). The landing walk checks the same speed condition when this switch is on.

``TAIL_FACTOR`` is fitted to one value: the live L+4 speed 0.203 against the ground factor's 0.452 (0.203 / 0.4517 = 0.4495).
The mechanism is not identified: no HEAD term that I found gives 0.45 here (the deadzone and the movement resistance 0.242 do
not). Inferred, not measured: a standing jump on the new bridge (assumed unchanged), jumps whose take-off is a walk ramp, a
re-jump during a carry. Gen 5 returns the original function.

Tests: ``tests/sim/test_jump_carry.py`` (the per-tick speed list within 0.02 blocks/s, touchdown on L, the walk at full speed
on L+5, the standing jump on L+3, a knockback during the carry, the switch off).
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import jax.numpy as jnp

from hytalegym.jax.combat.types import AGENT_ENTITY, AGENT_MOTION_NPC_WALK

SWITCH = "npc_jump_carry"
IMPLEMENTED = True
#: The horizontal force of the L+4 tick is scaled by this factor (fitted to the live 0.203; see the docstring).
TAIL_FACTOR = 0.45
#: Counter values (1 on take-off, plus the grounded ticks since): L+4 starts at 5, L+5 at 6.
TAIL_COUNT = 5
RELEASE_COUNT = 6


def _enabled(profile: Any) -> bool:
    return bool(profile.constructs.npc_jump_carry)


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for gen 7 the ``_apply_action`` that carries a running jump's take-off velocity."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def _apply_action(state: Any, actions: Any, pause_draw: Any, params: Any, *rest: Any, **kwargs: Any) -> Any:
        updated, attack, accepted = original(state, actions, pause_draw, params, *rest, **kwargs)
        npc_walk = params.agent_motion_model == jnp.int32(AGENT_MOTION_NPC_WALK)
        jumped = state.agent_grounded & ~updated.agent_grounded
        vx = state.velocity[:, AGENT_ENTITY, 0]
        vz = state.velocity[:, AGENT_ENTITY, 2]
        moving = (vx != 0.0) | (vz != 0.0)
        start = jumped & npc_walk & moving
        # A knockback during the carry sets the control lock and writes its own force: the carry ends, so the knockback's
        # force is neither scaled on L+4 nor cleared on L+5.
        count = jnp.where(state.knockback_control_lock, jnp.int32(0), state.agent_jump_carry_ticks)
        advanced = jnp.where((count > 0) & state.agent_grounded & ~start, count + 1, count)
        tail = (advanced == TAIL_COUNT) & ~start
        release = (advanced == RELEASE_COUNT) & ~start
        force = updated.agent_force_velocity
        takeoff = jnp.stack([vx, jnp.zeros_like(vx), vz], axis=1)
        force = jnp.where(start[:, None], takeoff, force)
        scale = jnp.where(
            tail[:, None],
            jnp.asarray([TAIL_FACTOR, 1.0, TAIL_FACTOR], dtype=force.dtype),
            jnp.asarray([1.0, 1.0, 1.0], dtype=force.dtype),
        )
        force = jnp.where(release[:, None], jnp.zeros_like(force), force * scale)
        following = jnp.where(start, jnp.int32(1), jnp.where(release, jnp.int32(0), advanced))
        return updated._replace(agent_force_velocity=force, agent_jump_carry_ticks=following), attack, accepted

    return _apply_action
