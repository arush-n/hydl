"""Survive a frozen pursuer while staying close to it, on a stamina handicap.

The mirror of the pursuit lesson over the same two-actor arena, and the second
half of a recursive pair: the learner is the EVADER and the pursuer runs a
frozen checkpoint, so each side is trained against the other's last promoted
policy.

Distinct from :mod:`arena.training.contracts.evasion`, which is a dodge-timing
lesson against an attack. This one is about position over a whole episode.

The objective moves with the rung the curriculum is on, and the contract carries
both shapes:

* WITH a leading band (`band_leads`), the objective is NOT "get away". Both
  actors run the same gait table at the same top speed, so an evader that simply
  holds a heading can never be caught on open ground -- it would win every
  episode, teach the pursuer nothing, and stall the recursive pair on its first
  turn. `contact_reward` is what makes holding station near the pursuer pay.
* WITHOUT one (`separation_reward` on, band off), the objective IS to open the
  gap, and this is the bottom rung of the objective ladder: an evader that
  cannot yet steer away from a pursuer has nothing to hold a band with.

The reward is kept thin, and every term earns its place:

* `survival_reward` at the horizon, `caught_penalty` on a catch -- the
  objective;
* `separation_reward` per block the evader's own motion adds to the gap, which
  is the ONLY term here keyed on distance rather than on motion or posture;
* `contact_reward` over a band, so holding station near the pursuer is what
  pays and running away forfeits income;
* `travel_reward`, `strafe_reward`, `sprint_reward` and `turn_reward`, all
  SMALL, because the band alone buys a policy that parks.

Those four price a MANOEUVRE rather than motion. `travel_reward` alone pays the
same for a straight line as for a circle, so it buys movement without buying
evasion. `strafe_reward` pays only the component of a step that runs ACROSS the
bearing to the pursuer, which is the part a straight-line run does not have;
`sprint_reward` pays for reaching sprint speed on top of the per-block rate;
and `turn_reward` pays for swinging the BODY, which is what travel rides. Their
combined per-tick income at sprint is guarded below `contact_reward`, so
manoeuvring seasons the standoff and never replaces it.

That group is not decoration. With the band as the only positional term the
evader solved it exactly as written: it sat at 5.36 blocks -- just inside the
6.0 peak -- and let itself be caught, surviving 0.002 of episodes. Standing at
the paying distance IS optimal income right up until the catch, so a reward that
only prices POSITION teaches station-keeping, not evasion. Paying a little for
ground covered and for changing heading is what makes moving away from a closing
pursuer worth more than holding the spot that pays.

They are capped as a GROUP, not one at a time: `__post_init__` prices the best
tick any of them can produce -- a purely lateral sprint with a turn -- and
refuses the config if that sum reaches `contact_reward`. Four individually-safe
rates can otherwise add up past the band. So fleeing past the radius still
forfeits the band and collects only the manoeuvre crumbs.

`airborne_idle_cost` and `tick_cost` stay at zero -- the first because the jump
head is closed outright, the second because charging per tick would tax the very
thing the evader is trying to buy.

`airborne_idle_cost` has a longer story worth not repeating. Hopping in place
was a real pathology -- airborne on 92% of ticks, 2.9 blocks covered in a whole
episode -- and it survived the simulation being corrected to forbid mid-air
acceleration, because the dynamics only made hopping INEFFECTIVE and never made
it UNREWARDING. The fix is not to price the behaviour, though: it is to close
the jump head, which `evader_duel_action_scope` now does. A mechanic the agent
cannot invoke needs no reward term arguing against it, and a cost that can never
be charged is dead weight in a contract whose whole point is to stay thin.

`PursuitStage.evader_*` cannot express the BAND. `evader_progress_scale`
multiplies the separation gain, so it pays for opening the gap, and `pursuit.py`
refuses negative reward values, so it cannot be inverted with a minus sign.

It does express separation, though -- and that is exactly the term this contract
was missing. `evader_progress_scale` (4.0) only ever reaches the evader through
`target_reward`, i.e. when the evader is the frozen OPPONENT. Trained as the
learner, the evader was paid by this module, which had no separation term at
all, so nothing in its per-tick reward could tell fleeing from circling. Run
20260822T011816Z-d8c5f7e3 measured the consequence: caught in 821/821 then
893/893 episodes while moving on 96% of ticks. `separation_reward` is the
mirror, held at the same 4.0 so neither role is handicapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from arena.training.contracts.common import (
    LOCOMOTION_HEADS,
    HorizonConfig,
    contract_hash,
    head_scope,
    lane_shapes_match,
    normalized_manifest,
    require_finite_nonnegative,
)
from arena.training.skills.program import FactoredActionScope


EVADER_DUEL_SCHEMA = "arena-contract-evader-duel-v1"

#: Full player stamina, from `hytalegym.jax.combat.types.PLAYER_STAMINA_MAXIMUM`.
#: Mirrored rather than imported so this config stays importable without the
#: Gym. `stamina_fraction` is validated against it; the runtime applies it.
PLAYER_STAMINA_MAXIMUM = 10.0


@dataclass(frozen=True, slots=True)
class EvaderDuelConfig:
    """What the evader is paid for, and the handicap that stops it fleeing."""

    horizon: HorizonConfig = HorizonConfig()

    #: Contact range. Inside it the pursuer has caught the evader. Matches
    #: `PursuitStage.catch_distance` so both halves of a recursive pair agree
    #: on what a catch is -- if these disagree, each policy is training against
    #: a rule the other does not play by.
    catch_distance: float = 1.0
    #: Where the contact reward PEAKS. Credit is zero at `catch_distance`,
    #: rises to 1.0 here, and falls back to zero at `contact_radius`.
    #:
    #: The first version of this ramped straight to 1.0 at `catch_distance`,
    #: which made the reward strictly increase as the evader closed -- it was
    #: paid most at the exact range where it dies. Run
    #: 20260821T030402Z-7e7a0372 solved that objective exactly as written:
    #: 2461 of 2461 episodes caught, mean episode 48.7 ticks, shorter than
    #: before training. A standoff has to have a peak the evader can hold, not
    #: a gradient into the jaws.
    safe_distance: float = 6.0
    #: Credit returns to zero here, so there is no gradient rewarding a longer
    #: sprint once the evader has left the band.
    contact_radius: float = 11.0

    contact_reward: float = 0.10
    #: Paid once at truncation if never caught. This has to dominate: without
    #: it the evader would trade its life for a few more ticks of contact
    #: income, since being caught only stops the income rather than reversing
    #: it.
    survival_reward: float = 1000.0
    #: Subtracted on a catch. Separate from the absence of `survival_reward`
    #: rather than implied by it, so "caught at tick 5" and "caught at tick
    #: 500" are both unambiguously worse than surviving.
    caught_penalty: float = 1000.0
    #: Paid per BLOCK of horizontal ground covered. Vertical motion earns
    #: nothing, which is the point: run 20260821T034339Z-4b6ba468 requested
    #: sprint on 84% of ticks yet travelled 0.078 blocks/tick while airborne
    #: 93% of the time -- 3.12 blocks of horizontal path against 8.67 of
    #: vertical, net displacement 0.77. It was hopping on the spot. Paying
    #: for ground covered makes that worthless without inventing a jump cost
    #: Hytale does not author.
    travel_reward: float = 0.12
    #: Paid per block the evader's OWN motion adds to the gap, normalized by
    #: `separation_clip_blocks`. Zero by default so no existing config changes.
    #:
    #: THE TERM THAT WAS MISSING. Every other per-tick reward here pays for
    #: motion or posture and none of them can tell fleeing from circling:
    #: `travel_reward` pays ground covered whatever the direction,
    #: `body_away_reward` pays where the body POINTS rather than where it ends
    #: up. Run 20260822T011816Z-d8c5f7e3 is what that produces -- the evader
    #: moved on 96% of ticks, turned on 98%, covered 23.4 blocks against its
    #: pursuer's 49.2, and was caught in 821/821 then 893/893 episodes. It was
    #: farming motion income, which is the mirror of the chaser farming the
    #: sight terms before `stationary_cost` was re-keyed onto radial progress.
    #:
    #: The scale matches `PursuitStageConfig.evader_progress_scale` (4.0), which
    #: is what already pays the evader for separation on the OTHER side of the
    #: swap -- via `target_reward`, and therefore only ever when the evader is
    #: the opponent rather than the learner. Same number keeps the two roles
    #: symmetric, which is the whole premise of the recursive loop.
    #:
    #: Attributed to the evader's own movement, not to the raw gap: a pursuer
    #: that wanders off must not pay the evader for standing still. Same
    #: principle as `target_separation_gain = -opponent_radial_contribution`.
    #: Paid per block of the step that runs ALONG the body's facing. The only
    #: term that pays for travelling where you are pointed.
    #:
    #: Movement is compass-driven and body-relative, so an actor can move in any
    #: direction without ever turning -- and strafing keeps the pursuer in view,
    #: which gated perception rewards. The evader took that option on every
    #: round: `aligned_fraction` 0.028, body almost never aligned with travel.
    #:
    #: It cannot work. Authored speeds are sprint-forward 7.00, run-forward 5.50,
    #: run-strafe 4.40, run-backward 3.58 blocks/s against a pursuer sprinting at
    #: 7.00. Only forward sprint keeps up, and `sprinting` requires
    #: `forward_dominant`, so a strafing evader is not slow by choice -- it is
    #: physically incapable of escaping. `travel_reward` cannot express this: it
    #: pays a strafe and a sprint identically per block.
    #: Paid on every valid tick the evader is still alive. Zero by default.
    #:
    #: `survival_reward` pays ONCE, at the horizon, so it is all-or-nothing:
    #: surviving 250 of 256 ticks and surviving 6 of them score the same. This
    #: pays for the length of the episode itself, which is the only shape that
    #: says "the longer it stays alive the better" -- and it gives a policy that
    #: never reaches the horizon a gradient it can actually climb, which under a
    #: 1.000 catch rate is the difference between a signal and nothing.
    alive_reward: float = 0.0
    forward_travel_reward: float = 0.0
    separation_reward: float = 0.0
    #: Sprint blocks per tick. Normalizing by the physical maximum makes the
    #: term a fraction of best-possible flight, so it cannot be out-scaled by a
    #: faster gait table later.
    separation_clip_blocks: float = 0.238
    #: Paid per block of the step that runs ACROSS the bearing to the pursuer.
    #: A straight-line flight is almost entirely radial and collects nothing
    #: here; circling the standoff collects the whole step. That is the
    #: difference between covering ground and evading, and `travel_reward`
    #: alone cannot express it -- it pays a run and a circle identically.
    #:
    #: It is bounded by the step itself, so the worst case for the anti-farm
    #: guard is a purely lateral sprint, where this and `travel_reward` both
    #: pay in full on the same tick.
    strafe_reward: float = 0.12
    #: Paid per tick spent at sprint speed, ON TOP of the per-block rate, which
    #: already pays sprint 3.8x a walk (0.238 vs 0.063 blocks/tick).
    sprint_reward: float = 0.005
    #: Sprint is detected from GROUND COVERED, not from the requested gait.
    #: Run 20260821T034339Z-4b6ba468 requested sprint on 84% of ticks while
    #: travelling 0.078 blocks/tick, so paying the request would have paid a
    #: policy hopping on the spot. Sprint is 0.238 blocks/tick and the
    #: directional multipliers cut that for non-forward motion, so the
    #: threshold sits below the forward figure rather than at it.
    sprint_travel_blocks: float = 0.20
    #: Paid for changing heading -- BODY yaw, which is what travel rides, not
    #: head yaw, which only moves the camera. Small on purpose: it is a
    #: tie-breaker that makes weaving cheaper than committing to one bearing,
    #: not an objective. Large enough to matter and it buys a policy that spins
    #: on the spot.
    turn_reward: float = 0.02
    #: Below this, a heading change is noise rather than a manoeuvre.
    turn_deadband_degrees: float = 5.0
    #: Paid for pointing the BODY away from the pursuer, scaled by how directly
    #: away it points: 1.0 at exactly 180 degrees off the bearing, 0.0 when
    #: staring straight at it.
    #:
    #: This is the posture half of evading, and it is the half the policy did
    #: not have. Travel pays for covering ground and strafe for crossing the
    #: bearing, but neither says which way to FACE, and movement is
    #: body-relative -- so an evader facing its pursuer can only retreat by
    #: walking backwards, at the reduced non-forward multiplier. Turning the
    #: body away is what makes fleeing fast, and head yaw stays free to look
    #: back, which is exactly the behaviour gated perception asks for.
    #:
    #: GATED ON `contact`, deliberately. Ungated it pays a policy that turns
    #: its back and runs to the far side of the arena, which is the farm the
    #: motion guard exists to stop. Gated, it is only earnable while holding
    #: the band, so it prices posture WITHIN the objective rather than
    #: competing with it -- and a fleeing policy earns exactly zero of it.
    body_away_reward: float = 0.04
    #: Whether the standoff band is the OBJECTIVE or merely a nudge.
    #:
    #: True is the original contract: proximity is what the evader is for, so
    #: motion, posture and turning must each stay below `contact_reward` or the
    #: policy farms them instead of holding station.
    #:
    #: False is the locomotion-first phase. The band is kept small and only
    #: pulls the evader back toward the fight so it does not simply run to the
    #: arena wall and park; the per-tick motion terms are meant to out-earn it,
    #: which is exactly what those guards forbid. It also ungates
    #: `body_away_reward` -- gating posture on the band is what made closing the
    #: distance the precondition for being paid to face away.
    #:
    #: NOT derived from the magnitudes. Inferring it from
    #: `contact_reward > motion_income` would make the guard tautological and
    #: delete the safety mechanism by construction, so the intent is declared.
    band_leads: bool = True
    #: Off. A per-tick charge is a cost on *time survived*, which is the one
    #: thing this objective is trying to buy.
    tick_cost: float = 0.0
    #: Charged per tick spent AIRBORNE while covering less than
    #: `hop_travel_blocks` of ground. Jumping is not penalised -- a jump that
    #: carries the evader somewhere is a legitimate manoeuvre -- but hopping
    #: on the spot is. Run 20260821T034339Z-4b6ba468 sat airborne on 93% of
    #: ticks with 0.77 blocks of net displacement over 41.
    #:
    #: This is REWARD SHAPING, not a physics claim: Hytale authors no stamina
    #: cost for jumping and none is added to the simulation.
    airborne_idle_cost: float = 0.0
    #: Ground covered in a tick below which an airborne tick counts as a hop.
    #: Walk is 0.063 blocks/tick, so this sits just under a walking step.
    hop_travel_blocks: float = 0.05

    #: Share of the pursuer's stamina bar the evader may hold. 1.0 is parity:
    #: the recursive pair is a fair fight, so neither side carries a handicap
    #: the other does not. Holding station is enforced by `contact_reward`, not
    #: by crippling the evader.
    stamina_fraction: float = 1.0

    schema: str = EVADER_DUEL_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, HorizonConfig):
            raise TypeError("evader duel horizon must be a HorizonConfig")
        require_finite_nonnegative(
            {
                "catch_distance": self.catch_distance,
                "safe_distance": self.safe_distance,
                "contact_radius": self.contact_radius,
                "contact_reward": self.contact_reward,
                "survival_reward": self.survival_reward,
                "caught_penalty": self.caught_penalty,
                "travel_reward": self.travel_reward,
                "alive_reward": self.alive_reward,
                "forward_travel_reward": self.forward_travel_reward,
                "separation_reward": self.separation_reward,
                "separation_clip_blocks": self.separation_clip_blocks,
                "strafe_reward": self.strafe_reward,
                "sprint_reward": self.sprint_reward,
                "sprint_travel_blocks": self.sprint_travel_blocks,
                "turn_reward": self.turn_reward,
                "turn_deadband_degrees": self.turn_deadband_degrees,
                "body_away_reward": self.body_away_reward,
                "airborne_idle_cost": self.airborne_idle_cost,
                "hop_travel_blocks": self.hop_travel_blocks,
                "tick_cost": self.tick_cost,
                "stamina_fraction": self.stamina_fraction,
            },
            "evader duel",
        )
        if not self.catch_distance < self.safe_distance < self.contact_radius:
            raise ValueError(
                "the band must satisfy catch_distance < safe_distance < "
                f"contact_radius, got {self.catch_distance:g} / "
                f"{self.safe_distance:g} / {self.contact_radius:g}"
            )
        if not 0.0 < self.stamina_fraction <= 1.0:
            raise ValueError("stamina_fraction must be in (0, 1]")
        # The catch has to outweigh a whole episode of loitering, or closing to
        # contact and dying is a profitable trade.
        #
        # Only while a band pays. Without one `loitering` is zero, so the check
        # reads "0 must exceed 0" and refuses a legitimate config: a rung with
        # no proximity income and no catch penalty, where being caught is
        # punished by ENDING the episode and with it the per-tick income. That
        # is a positive-only objective, not a loophole -- there is no contact
        # reward to farm by dying.
        # A caught_penalty of exactly ZERO is the positive-only objective and is
        # allowed with a band, 2026-08-22. The per-tick income is non-negative
        # and the episode ENDS on a catch, so dying at tick T earns the contact
        # accumulated before T and surviving to the horizon earns strictly more:
        # there is no tick at which dying pays better than continuing. What the
        # guard still refuses is a TOKEN penalty -- nonzero but under a full
        # episode of income -- which does make an early death a real trade.
        loitering = self.contact_reward * self.horizon.maximum_ticks
        if (
            self.band_active
            and self.caught_penalty > 0.0
            and self.caught_penalty <= loitering
        ):
            raise ValueError(
                f"caught_penalty {self.caught_penalty:g} must exceed a full "
                f"episode of contact income ({loitering:g}), or be exactly 0 "
                "for a positive-only objective; a token penalty makes being "
                "caught worth paying for"
            )
        # Only meaningful while SURVIVING is the objective.
        #
        # Without a band `loitering` is zero and the comparison says nothing.
        # With a band that LEADS, proximity is the objective by declaration and
        # the survival bonus is explicitly on a schedule down to zero -- that is
        # the intended end state, "hold contact for the whole episode", not a
        # misconfiguration. `caught_penalty` is what keeps dying bad at every
        # rung, and it is guarded unconditionally above.
        if (
            self.band_active
            and not self.band_leads
            and self.survival_reward <= loitering
        ):
            raise ValueError(
                f"survival_reward {self.survival_reward:g} must exceed a full "
                f"episode of contact income ({loitering:g}), or surviving is "
                "not the objective"
            )
        # Every guard below prices something AGAINST the standoff band. With
        # `contact_reward` at zero there is no band: proximity pays nothing,
        # fleeing forfeits nothing, and "moving must not out-earn holding" is a
        # comparison against a term that does not exist. Guarding anyway would
        # refuse every motion-only config, since any positive rate exceeds zero.
        #
        # This is the LEARN-TO-MOVE configuration and it is deliberate. Paying
        # for proximity while the policy cannot yet control its motion teaches
        # it to close the distance, which is the one thing an evader must not
        # do; the band only becomes meaningful once it can already flee.
        if self.band_governs:
            # Manoeuvring must not out-earn holding the band, or the evader
            # sprints away and farms motion. The worst case is one tick of
            # purely lateral sprint: 0.238 blocks per tick paying travel AND
            # strafe together, plus the sprint bonus and a turn past the
            # deadband. Guarding the terms separately would let four
            # individually-safe rates sum past the band.
            sprint_blocks = 0.238
            motion_income = (
                (self.travel_reward + self.strafe_reward) * sprint_blocks
                + self.sprint_reward
                + self.turn_reward
            )
            if motion_income >= self.contact_reward:
                raise ValueError(
                    f"travel_reward {self.travel_reward:g}, strafe_reward "
                    f"{self.strafe_reward:g}, sprint_reward "
                    f"{self.sprint_reward:g} and turn_reward "
                    f"{self.turn_reward:g} pay {motion_income:g} together on "
                    f"one lateral sprint tick, which must stay below "
                    f"contact_reward {self.contact_reward:g} or moving is "
                    "better than holding the standoff"
                )
            # `body_away_reward` is deliberately NOT in `motion_income`: with a
            # band it is gated on `contact`, so a policy that has fled cannot
            # earn it and it cannot fund running away. What it must not do is
            # out-earn the position it scales, or the evader would rather face
            # away at the band edge than hold the peak facing anywhere.
            if self.body_away_reward >= self.contact_reward:
                raise ValueError(
                    f"body_away_reward {self.body_away_reward:g} must stay "
                    f"below contact_reward {self.contact_reward:g}, or posture "
                    "pays better than the position it is supposed to modify"
                )
            # Spinning in place must not out-earn holding the gap, or the
            # evader farms the turn term instead of evading.
            if self.turn_reward >= self.contact_reward:
                raise ValueError(
                    "turn_reward must stay below contact_reward, or spinning "
                    "on the spot pays better than staying close"
                )
        if self.schema != EVADER_DUEL_SCHEMA:
            raise ValueError("evader duel schema is not current")

    @property
    def band_active(self) -> bool:
        """Whether proximity pays at all.

        A zero `contact_reward` is the LEARN-TO-MOVE configuration: no standoff
        band, so nothing rewards closing and every band-relative guard is a
        comparison against a term that does not exist.
        """

        return self.contact_reward > 0.0

    @property
    def band_governs(self) -> bool:
        """Whether proximity is the objective the other terms are priced against.

        Both halves matter. A band that pays nothing governs nothing, and a band
        the caller has declared a nudge (`band_leads=False`) is deliberately
        allowed to be out-earned by motion. This is what the ordering guards and
        the posture gate key on -- `band_active` alone would re-impose the
        band-is-everything premise the moment a small band came back.
        """

        return self.band_active and self.band_leads

    @property
    def stamina_maximum(self) -> float:
        """The evader's stamina ceiling in bar units, for the runtime to apply."""

        return PLAYER_STAMINA_MAXIMUM * self.stamina_fraction

    def manifest(self) -> dict[str, object]:
        return normalized_manifest(
            schema=self.schema,
            objective="survive_the_pursuer_while_holding_contact_range",
            reward={
                "contact_reward": self.contact_reward,
                "survival_reward": self.survival_reward,
                "travel_reward": self.travel_reward,
                "alive_reward": self.alive_reward,
                "forward_travel_reward": self.forward_travel_reward,
                "separation_reward": self.separation_reward,
                "strafe_reward": self.strafe_reward,
                "sprint_reward": self.sprint_reward,
                "turn_reward": self.turn_reward,
                "body_away_reward": self.body_away_reward,
            },
            cost={
                "airborne_idle_cost": self.airborne_idle_cost,
                "caught_penalty": self.caught_penalty,
                "tick_cost": self.tick_cost,
            },
            anti_farm=(
                "proximity_is_what_pays_so_fleeing_forfeits_income_and_the_four_"
                "manoeuvre_terms_travel_strafe_sprint_and_turn_are_guarded_to_"
                "sum_below_contact_reward_on_one_lateral_sprint_tick_so_moving_"
                "can_never_replace_holding_the_standoff"
            ),
            horizon=self.horizon,
            role="learner_is_the_evader",
            opponent="frozen_policy_checkpoint",
            band={
                "catch_distance": self.catch_distance,
                "safe_distance": self.safe_distance,
                "contact_radius": self.contact_radius,
                "shape": "zero_at_catch_distance_peak_at_safe_distance_zero_at_radius",
            },
            handicap={
                "stamina_fraction": self.stamina_fraction,
                "stamina_maximum": self.stamina_maximum,
                "pursuer_stamina_maximum": PLAYER_STAMINA_MAXIMUM,
            },
        )

    @property
    def contract_sha256(self) -> str:
        return contract_hash(self.manifest())


class EvaderDuelSignals(NamedTuple):
    """Evader credit plus the evidence behind it."""

    reward: jax.Array
    contact: jax.Array
    survived: jax.Array
    caught: jax.Array


def evader_duel_signals(
    config: EvaderDuelConfig,
    *,
    distance: jax.Array,
    heading_change_degrees: jax.Array,
    ground_travelled: jax.Array,
    lateral_travelled: jax.Array,
    forward_travelled: jax.Array,
    separation_gain: jax.Array,
    facing_away: jax.Array,
    airborne: jax.Array,
    caught: jax.Array,
    truncated: jax.Array,
    valid: jax.Array,
) -> EvaderDuelSignals:
    """Score one evader tick against the gap it held and whether it survived.

    ``truncated`` marks the horizon tick, which is the only tick
    ``survival_reward`` can pay on -- a per-tick survival bonus would be
    indistinguishable from a negative tick cost and would pay a fleeing policy
    just as well as a manoeuvring one.
    """

    gap = jnp.asarray(distance, dtype=jnp.float32)
    turned = jnp.asarray(heading_change_degrees, dtype=jnp.float32)
    travelled = jnp.asarray(ground_travelled, dtype=jnp.float32)
    across = jnp.asarray(lateral_travelled, dtype=jnp.float32)
    # Clamped to the step and to non-negative: reversing is not forward travel,
    # and a caller cannot pay more along-body ground than was actually covered.
    ahead = jnp.clip(jnp.asarray(forward_travelled, dtype=jnp.float32), 0.0,
                     jnp.asarray(ground_travelled, dtype=jnp.float32))
    # Signed and SYMMETRICALLY clipped: closing has to cost what opening pays,
    # or an evader could bank a sprint away and give the ground back for free.
    # Normalized by sprint speed so the term reads as a fraction of the best
    # flight physically available.
    opened = jnp.clip(
        jnp.asarray(separation_gain, dtype=jnp.float32)
        / jnp.maximum(jnp.float32(config.separation_clip_blocks), jnp.float32(1.0e-6)),
        -1.0,
        1.0,
    )
    # Clipped rather than trusted: the caller derives this from a cosine, and a
    # value outside [0, 1] would silently scale the term past its configured
    # ceiling and break the guard that keeps it under `contact_reward`.
    away = jnp.clip(jnp.asarray(facing_away, dtype=jnp.float32), 0.0, 1.0)
    aloft = jnp.asarray(airborne, dtype=jnp.bool_)
    was_caught = jnp.asarray(caught, dtype=jnp.bool_)
    ended = jnp.asarray(truncated, dtype=jnp.bool_)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    lane_shapes_match(
        gap, turned, travelled, across, ahead, opened, away, aloft, was_caught,
        ended, evidence,
    )

    near = jnp.maximum(
        jnp.float32(config.safe_distance - config.catch_distance),
        jnp.float32(1.0e-6),
    )
    far = jnp.maximum(
        jnp.float32(config.contact_radius - config.safe_distance),
        jnp.float32(1.0e-6),
    )
    # A peak the evader can hold: 0 at contact, 1 at `safe_distance`, 0 again at
    # the radius. Closing past the peak LOSES reward, which is what turns it
    # around before contact. Ramping straight to 1.0 at `catch_distance` paid
    # most at the range where it dies and was solved by walking in.
    contact = jnp.clip(
        jnp.minimum(
            (gap - jnp.float32(config.catch_distance)) / near,
            (jnp.float32(config.contact_radius) - gap) / far,
        ),
        0.0,
        1.0,
    )
    # Static on a config field, not a traced value, so this branch is resolved
    # at trace time and both arms stay jit-safe.
    posture_gate = contact if config.band_governs else jnp.float32(1.0)
    manoeuvred = jnp.abs(turned) > jnp.float32(config.turn_deadband_degrees)
    # Clamped to the step so a caller cannot pay more lateral ground than was
    # actually covered, which would make the anti-farm guard a lie.
    lateral = jnp.clip(across, 0.0, travelled)
    sprinting = travelled >= jnp.float32(config.sprint_travel_blocks)
    # Airborne AND going nowhere. A jump that covers ground is untouched.
    hopping = evidence & aloft & (travelled < jnp.float32(config.hop_travel_blocks))
    survived = evidence & ended & ~was_caught

    reward = (
        jnp.float32(config.contact_reward) * contact
        + jnp.float32(config.travel_reward) * travelled
        + jnp.float32(config.strafe_reward) * lateral
        + jnp.float32(config.sprint_reward) * sprinting
        + jnp.float32(config.turn_reward) * manoeuvred
        # Distance actually GAINED, which is the only term here that can tell
        # fleeing from circling -- every other one pays a run and a circle the
        # same. Ungated on purpose: it is the objective on the rung that carries
        # it, and it is set to zero on the rungs where holding the band is.
        + jnp.float32(config.forward_travel_reward) * ahead
        + jnp.float32(config.separation_reward) * opened
        # With a band, scaled by `contact`, so facing away is only payable
        # while the band is being held -- outside it both factors collapse and
        # the term is zero, which is what stops it funding a run.
        #
        # WITHOUT a band that gate is exactly wrong. `contact` is zero
        # everywhere, so the term would be dead; worse, when it was live the
        # gate made "get close" the precondition for being paid to face away,
        # which pulled the evader TOWARD the pursuer. With nothing rewarding
        # proximity there is no farm to prevent, and facing away is the whole
        # lesson, so it is paid wherever the evader is.
        + jnp.float32(config.body_away_reward) * posture_gate * away
        + jnp.float32(config.alive_reward) * evidence
        + jnp.float32(config.survival_reward) * survived
        - jnp.float32(config.airborne_idle_cost) * hopping
        - jnp.float32(config.caught_penalty) * (evidence & was_caught)
        - jnp.float32(config.tick_cost)
    )
    return EvaderDuelSignals(
        jnp.where(evidence, reward, jnp.float32(0.0)),
        contact,
        survived,
        evidence & was_caught,
    )


def evader_duel_action_scope() -> FactoredActionScope:
    """Movement and aim only. The evader does not attack; it survives.

    JUMP IS CLOSED. Every world in the pair's current rotation is a plane, so
    there is nothing to jump over, and leaving the head open cost real ground:
    the evader stayed airborne on 92% of ticks and covered 2.9 blocks in a whole
    episode, because an airborne actor cannot accelerate and steers only weakly.
    Closing the head pins it to its no-op, which removes the behaviour outright
    instead of pricing it and hoping the policy works that out.

    Reopen this together with the pursuer's when the rotation carries terrain
    worth jumping over -- and restore `airborne_idle_cost` at the same time.
    """

    return head_scope(*LOCOMOTION_HEADS)


__all__ = [
    "EVADER_DUEL_SCHEMA",
    "PLAYER_STAMINA_MAXIMUM",
    "EvaderDuelConfig",
    "EvaderDuelSignals",
    "evader_duel_action_scope",
    "evader_duel_signals",
]
