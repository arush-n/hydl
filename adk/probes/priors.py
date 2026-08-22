"""Hand-written reactive agents, a few lines each.

Random sampling explores the action space evenly, which is the wrong shape for
finding combat bugs: uniform-legal spends most of its steps wandering and
engages only by accident.  These priors read the observation and commit to an
obvious behaviour, so they reach engagement, low health, and termination far
sooner than chance does.

None of them learn and none are meant to be good.  They exist because a bug in
the attack path only shows up if something attacks.

All of them build **logits** and sample through the mask rather than emitting
factors directly, so a prior can never produce an illegal action -- if the
behaviour it wants is masked off, the mask wins and a legal action is chosen
instead.  That fallback is deliberate: pair a prior with
``adk.probes.liveness`` to see whether the environment ever actually offered
what the prior asked for.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_LOCOMOTION_DODGE_START,
    DODGE_ACTION_COUNT,
)

from adk.policy.actions import action_index
from adk.policy.fields import field
from adk.probes.policies import (
    ACTION_LOGIT_WIDTH,
    PREFERENCE_STRENGTH,
    head_span,
)
from adk.runtime.env_adapter import legal_mask, sample_actions


#: Planar distance below which the target counts as "in reach".  The
#: observation normalizes distance against the scene's sensor range, so this is
#: a fraction of that range rather than metres.
ENGAGE_DISTANCE = 0.35

#: Health fraction below which the defensive prior starts protecting itself.
HURT_FRACTION = 0.5

#: The observation reports distance as a fraction of the opponent's sensor
#: range, not in blocks.  Measured by fitting observed distance against true
#: world separation: ``observed = world / 16.0`` (intercept ~0), which matches
#: the ruleset's ``target.sensor_range = 16.0``.  Use :func:`blocks` rather
#: than writing normalized literals, which are unreadable and easy to
#: mis-scale if the range ever moves.
SENSOR_RANGE_BLOCKS = 16.0


def blocks(distance: float) -> float:
    """Convert a distance in blocks to the observation's normalized units."""

    return distance / SENSOR_RANGE_BLOCKS


#: How close the aggressive prior wants to be before it swings.
STRIKE_DISTANCE = blocks(2.5)

#: The band the evasive prior tries to hold: close enough to stay engaged,
#: far enough that the opponent's swing does not reach.
SHADOW_NEAR = blocks(3.0)
SHADOW_FAR = blocks(5.0)


def _row_bias(head: str, choice: jax.Array, strength: float) -> jax.Array:
    """One-hot logit bias on ``head``, chosen per row.  Traced."""

    offset, size = head_span(head)
    within = jnp.clip(jnp.asarray(choice), 0, size - 1)
    onehot = jax.nn.one_hot(offset + within, ACTION_LOGIT_WIDTH, dtype=jnp.float32)
    return onehot * strength


def _combat(observation: Any, name: str) -> jax.Array:
    return field(observation.base.combat_f32, "combat_f32", name)


def approach_target(*, engage: float = ENGAGE_DISTANCE,
                    strength: float = PREFERENCE_STRENGTH):
    """Close the distance, then hold and face.

    The simplest prior that reaches engagement: it turns wandering into
    contact, which is what the combat invariants need in order to mean
    anything.
    """

    approach = action_index("action_mask", "approach")
    face = action_index("action_mask", "face_target")
    idle = action_index("action_mask", "idle")

    def policy(carry, actor_input, key):
        observation = actor_input.legal_observation
        mask = legal_mask(actor_input)
        visible = _combat(observation, "target_visible") > 0.5
        distance = _combat(observation, "visible_target_planar_distance")

        choice = jnp.where(
            visible,
            jnp.where(distance > engage, approach, face),
            idle,
        )
        bias = _row_bias("base_action", choice, strength)
        return carry, sample_actions(bias, mask, key)

    return policy


def strike_when_ready(*, engage: float = ENGAGE_DISTANCE,
                      strength: float = PREFERENCE_STRENGTH):
    """Approach, then fire an ability once one is available.

    Attacks live on the ability head, not on ``base_action`` -- driving
    ``base_action`` toward "attack" is a silent no-op because the policy mask
    never offers it.  This prior therefore biases both heads: movement on
    ``base_action`` and the strike itself on ``ability_none_plus_slots``.
    """

    approach = action_index("action_mask", "approach")
    face = action_index("action_mask", "face_target")
    ability_offset, ability_size = head_span("ability_none_plus_slots")

    def policy(carry, actor_input, key):
        observation = actor_input.legal_observation
        mask = legal_mask(actor_input)
        visible = _combat(observation, "target_visible") > 0.5
        distance = _combat(observation, "visible_target_planar_distance")
        in_reach = visible & (distance <= engage)

        move = jnp.where(distance > engage, approach, face)
        bias = _row_bias("base_action", move, strength)

        # Slot 0 is "no ability"; prefer the lowest offered real slot.
        ability_mask = mask[..., ability_offset : ability_offset + ability_size]
        real = ability_mask.at[..., 0].set(False)
        has_ability = jnp.any(real, axis=-1)
        slot = jnp.argmax(real, axis=-1)
        strike = in_reach & has_ability
        bias = bias + jnp.where(
            strike[:, None],
            _row_bias("ability_none_plus_slots", slot, strength),
            0.0,
        )
        return carry, sample_actions(bias, mask, key)

    return policy


def guard_when_hurt(*, hurt: float = HURT_FRACTION,
                    strength: float = PREFERENCE_STRENGTH):
    """Raise guard below a health threshold, otherwise press forward.

    Drives the guard/stamina interaction, which is where
    ``guard_implies_stamina`` can fire and where a random policy rarely spends
    enough consecutive steps to matter.
    """

    approach = action_index("action_mask", "approach")
    retreat = action_index("action_mask", "retreat")

    def policy(carry, actor_input, key):
        observation = actor_input.legal_observation
        mask = legal_mask(actor_input)
        health = field(observation.base.self_f32, "self_f32", "health_fraction")
        hurting = health < hurt

        bias = _row_bias(
            "base_action", jnp.where(hurting, retreat, approach), strength
        )
        bias = bias + jnp.where(
            hurting[:, None],
            _row_bias("guard_off_on", jnp.ones((), jnp.int32), strength),
            0.0,
        )
        return carry, sample_actions(bias, mask, key)

    return policy


def wander(*, strength: float = PREFERENCE_STRENGTH):
    """Move and look around, never engage.

    The control arm.  Anything a combat prior trips that ``wander`` does not is
    attributable to engagement rather than to movement or to time passing.

    "Never engage" has to be enforced, not just intended: every head this
    prior does not bias is sampled uniformly from the legal mask, so an
    unbiased ability head made the control arm attack on roughly half of all
    steps and deal real damage.  A control that engages is not a control, so
    the ability head is pinned to ``none`` here.
    """

    # Idle plus the four gait rings, stopping short of the dodge options: a
    # wandering control that dodged would not be a movement control.
    move_size = ARSENAL_POLICY_LOCOMOTION_DODGE_START
    _yaw_offset, yaw_size = head_span("yaw_delta_bins")

    def policy(carry, actor_input, key):
        mask = legal_mask(actor_input)
        batch = mask.shape[0]
        move_key, yaw_key, sample_key = jax.random.split(key, 3)
        bias = _row_bias(
            "locomotion_gait_compass",
            jax.random.randint(move_key, (batch,), 0, move_size),
            strength,
        ) + _row_bias(
            "yaw_delta_bins",
            jax.random.randint(yaw_key, (batch,), 0, yaw_size),
            strength,
        ) + _row_bias("ability_none_plus_slots", 0, strength * 2.0)
        return carry, sample_actions(bias, mask, sample_key)

    return policy


def _still(strength: float) -> jax.Array:
    """Hold jump and dodge off.

    Every head a prior does not bias is sampled uniformly from the legal
    mask, so an unbiased ``jump_off_on`` makes the agent jump on roughly half
    of all steps.  On flat ground that only looks odd; on terrain it drives
    the body into slopes and dominates the trace.  A prior about movement has
    to hold these still or it is measuring its own noise.

    Dodge no longer has its own head -- it is the tail of
    ``locomotion_gait_compass`` -- so holding it off means pushing those four
    options down rather than pinning the head to a neutral value.  Pinning
    would also pin movement to idle, which is the opposite of what a movement
    prior wants.
    """

    bias = _row_bias("jump_off_on", 0, strength * 4.0)
    for choice in range(
        ARSENAL_POLICY_LOCOMOTION_DODGE_START,
        ARSENAL_POLICY_LOCOMOTION_DODGE_START + DODGE_ACTION_COUNT,
    ):
        bias = bias + _row_bias(
            "locomotion_gait_compass", choice, -strength * 4.0
        )
    return bias


def press_attack(*, reach: float = STRIKE_DISTANCE,
                 strength: float = PREFERENCE_STRENGTH):
    """Close the distance and keep swinging.

    Walks in with ``approach_attack`` while out of reach, holds and faces once
    inside it, and fires the lowest offered ability slot whenever one is
    available.  The strike goes on the ability head because ``base_action``'s
    own ``attack`` is never offered by the policy mask.

    Pairs with :func:`shadow_without_damage` as the aggressive half of a
    two-agent comparison against the same opponent.
    """

    # ``approach_attack`` is never offered by the policy mask -- attacking is an
    # ability in the v3 space -- so biasing toward it does nothing and the mask
    # substitutes a random legal move instead.  Walk in on plain ``approach``.
    approach = action_index("action_mask", "approach")
    face = action_index("action_mask", "face_target")
    idle = action_index("action_mask", "idle")
    ability_offset, ability_size = head_span("ability_none_plus_slots")

    def policy(carry, actor_input, key):
        observation = actor_input.legal_observation
        mask = legal_mask(actor_input)
        visible = _combat(observation, "target_visible") > 0.5
        distance = _combat(observation, "visible_target_planar_distance")
        in_reach = visible & (distance <= reach)

        move = jnp.where(visible, jnp.where(in_reach, face, approach), idle)
        bias = _row_bias("base_action", move, strength) + _still(strength)

        ability_mask = mask[..., ability_offset : ability_offset + ability_size]
        real = ability_mask.at[..., 0].set(False)
        # Fire whenever a real slot is offered, as the docstring promises.
        # Gating this on ``in_reach`` as well made the prior attack less often
        # than uniform sampling: it only ever struck inside 2.5 blocks, which
        # it rarely reached once the illegal approach bias above was ignored.
        strike = visible & jnp.any(real, axis=-1)
        bias = bias + jnp.where(
            strike[:, None],
            _row_bias("ability_none_plus_slots", jnp.argmax(real, axis=-1),
                      strength),
            0.0,
        )
        return carry, sample_actions(bias, mask, key)

    return policy


def shadow_without_damage(*, near: float = SHADOW_NEAR,
                          far: float = SHADOW_FAR,
                          strength: float = PREFERENCE_STRENGTH):
    """Stay with the opponent without trading blows.

    Holds a standoff band: walks in when the opponent drifts beyond ``far``,
    backs off when it closes inside ``near``, and strafes while the band
    holds, so it keeps circling rather than standing still.  Guard goes up
    whenever the opponent is both close and facing this way.  It never fires
    an ability -- staying alive next to something is the whole behaviour.

    Threat is judged from distance and ``target_facing_error`` rather than the
    target's attack phase: ``target_phase_*`` and ``target_attack_progress``
    are published but never populated on the arsenal path, so a prior that
    keyed on them would silently read zeros.
    """

    approach = action_index("action_mask", "approach")
    retreat = action_index("action_mask", "retreat")
    strafe_left = action_index("action_mask", "strafe_left")
    strafe_right = action_index("action_mask", "strafe_right")
    idle = action_index("action_mask", "idle")

    def policy(carry, actor_input, key):
        observation = actor_input.legal_observation
        mask = legal_mask(actor_input)
        visible = _combat(observation, "target_visible") > 0.5
        distance = _combat(observation, "visible_target_planar_distance")
        facing_error = _combat(observation, "target_facing_error")

        # Circle toward the side the opponent is not facing, so the prior
        # stays stateless -- every other prior here keeps ``carry`` None and
        # the test suite pins that.  Which of the two strafes corresponds to
        # the blind side depends on the sign convention of
        # ``target_facing_error``, which is not verified here; either polarity
        # produces orbiting rather than a one-way drift, which is what the
        # band needs.
        orbit = jnp.where(facing_error > 0.0, strafe_left, strafe_right)

        held = jnp.where(distance < near, retreat,
                         jnp.where(distance > far, approach, orbit))
        move = jnp.where(visible, held, idle)
        bias = _row_bias("base_action", move, strength) + _still(strength)

        threatened = visible & (distance < far) & (jnp.abs(facing_error) < 0.25)
        bias = bias + jnp.where(
            threatened[:, None],
            _row_bias("guard_off_on", 1, strength),
            _row_bias("guard_off_on", 0, strength),
        )
        # Never attack: pin the ability head to "none".
        bias = bias + _row_bias("ability_none_plus_slots", 0, strength * 2.0)
        return carry, sample_actions(bias, mask, key)

    return policy


def run_away(*, strength: float = PREFERENCE_STRENGTH):
    """Break away and keep breaking away.

    The runner half of a runner-versus-combat pairing.  Unlike
    :func:`shadow_without_damage`, which holds a band, this one simply
    retreats whenever it can see the opponent and guards while it does, so
    separation grows for as long as the opponent keeps chasing.

    Note the speed asymmetry it runs into: the ruleset gives the agent
    ``asset_max_walk_speed = 5.0`` and the Trork Brawler ``8.0`` with a
    ``6.4`` chase speed, so a straight-line retreat loses ground.  That is the
    behaviour, not a bug in the prior.
    """

    retreat = action_index("action_mask", "retreat")
    idle = action_index("action_mask", "idle")

    def policy(carry, actor_input, key):
        observation = actor_input.legal_observation
        mask = legal_mask(actor_input)
        visible = _combat(observation, "target_visible") > 0.5

        bias = _row_bias(
            "base_action", jnp.where(visible, retreat, idle), strength
        ) + _still(strength)
        bias = bias + jnp.where(
            visible[:, None],
            _row_bias("guard_off_on", 1, strength),
            _row_bias("guard_off_on", 0, strength),
        )
        bias = bias + _row_bias("ability_none_plus_slots", 0, strength * 2.0)
        return carry, sample_actions(bias, mask, key)

    return policy


#: Every prior, for sweeping them all against one scene.
PRIORS = {
    "approach_target": approach_target,
    "strike_when_ready": strike_when_ready,
    "guard_when_hurt": guard_when_hurt,
    "wander": wander,
    "press_attack": press_attack,
    "shadow_without_damage": shadow_without_damage,
    "run_away": run_away,
}


__all__ = [
    "ENGAGE_DISTANCE",
    "HURT_FRACTION",
    "PRIORS",
    "SENSOR_RANGE_BLOCKS",
    "SHADOW_FAR",
    "SHADOW_NEAR",
    "STRIKE_DISTANCE",
    "approach_target",
    "blocks",
    "guard_when_hurt",
    "press_attack",
    "run_away",
    "shadow_without_damage",
    "strike_when_ready",
    "wander",
]
