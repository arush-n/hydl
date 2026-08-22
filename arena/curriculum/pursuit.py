"""The pursuit/evasion curriculum, as data rather than as a loop.

The recursive self-play driver used to carry its own list of terrain rungs and
its own advancement rule. That put the definition of "what gets harder, and
when" in a scratch script, where the console could not read it and no test
covered it. It lives here now, and `arena.curriculum.ladder.Ladder` owns the
promote/demote decision, so there is exactly one such rule in the repository.

Three axes move, and they are deliberately separate:

**Objective** is a ladder, and it is the one that runs first. `objective_ladder`
asks what the evader is being told to DO -- survive, then hold station, then
train recursively -- because a reward the policy cannot yet act on does not
teach it, it misleads it. Terrain difficulty is meaningless until the agent
understands the game it is playing on that terrain.

**Terrain** is the ladder. Rungs go flat plane -> a few gentle captured worlds
-> more of them -> the rugged tail. Ordering matters more than count: picked
blind, `region-library-v3-288` skews toward near-worst-case ground, and one
earlier selection trained entirely inside a cave at 0% open sky. `flattest`
ranks the split by gentleness so the early rungs are terrain a policy can
actually move through.

**Horizon** is NOT a ladder rung. It is a dial the loop moves within a round to
keep a matchup contested, and it moves in OPPOSITE directions for the two sides
-- shorter favours the evader, longer favours the chaser. Promoting on it would
mean promoting one side by handicapping the other. `HORIZON_BAND` records the
range and the reasoning; the loop owns the stepping.
"""

from __future__ import annotations

from arena.curriculum.ladder import Ladder, SettingsStage

#: Ticks. The simulation advances 30 per second, so these are 4.3 s, 8.5 s and
#: 17 s. The floor is not lower because an episode under about four seconds
#: stops being an evasion problem and becomes a reaction-time one, and a win
#: there does not transfer. The ceiling is not the default because an earlier
#: lineage survived 0.6% of episodes at 512 -- no positive examples, so no
#: gradient.
HORIZON_BAND = (128, 256, 512)

#: A rung promotes on sustained success, not one lucky evaluation. `window=3`
#: with `minimum_evaluations=3` means three consecutive rounds averaging above
#: the bar; `demote_below` catches a pair that advanced onto ground it cannot
#: actually handle, which is otherwise invisible until several rungs later.
_GATE = {
    "promote_at": 0.60,
    "demote_below": 0.15,
    "window": 3,
    "minimum_evaluations": 3,
}


def terrain_ladder() -> Ladder:
    """Flat plane first, then captured worlds, gentlest ordering first.

    `world_design` and the Region selectors are mutually exclusive by
    construction: setting `world_design` is what ROUTES a run onto the flat
    path, and leaving it unset is what selects captured worlds. Each rung
    therefore carries exactly one of the two shapes, and a caller splats
    `stage.settings` into its launch spec without deciding anything.
    """

    return Ladder([
        SettingsStage(
            "flat-plane",
            {"world_design": "completely_flat"},
            **_GATE,
        ),
        SettingsStage(
            "4-flattest",
            {"world_count": 4, "world_order": "flattest",
             "environment_diversity": True},
            **_GATE,
        ),
        SettingsStage(
            "8-flattest",
            {"world_count": 8, "world_order": "flattest",
             "environment_diversity": True},
            **_GATE,
        ),
        SettingsStage(
            "16-flattest",
            {"world_count": 16, "world_order": "flattest",
             "environment_diversity": True},
            **_GATE,
        ),
        SettingsStage(
            "16-mixed",
            {"world_count": 16, "world_order": "mixed",
             "environment_diversity": True},
            **_GATE,
        ),
        SettingsStage(
            "32-mixed",
            {"world_count": 32, "world_order": "mixed",
             "environment_diversity": True},
            **_GATE,
        ),
    ])


#: A rung of the objective ladder is a whole lesson rather than a harder map,
#: so it is held longer before advancing: five evaluations averaged, three of
#: them at minimum. Advancing early is what produced an evader that had a
#: proximity reward before it could steer.
_OBJECTIVE_GATE = {
    "promote_at": 0.60,
    "demote_below": 0.10,
    "window": 5,
    "minimum_evaluations": 3,
}

#: The rung at which the loop starts swapping roles. Below it, one side trains
#: against a frozen opponent and the other is never touched -- recursion is the
#: LAST lesson, not the frame the others happen inside.
RECURSIVE_STAGE = "recursive"


def objective_ladder() -> Ladder:
    """What the evader is being asked to do, in the order it can learn it.

    Terrain asks "can you handle this ground". This asks "do you know what the
    game is", and the two are independent -- an evader can be fluent on a plane
    and still not understand the objective.

    The order is not cosmetic. Every rung here was reached by watching the
    previous one fail:

    1. **long_range** -- open the gap and stay alive. `separation` leads and no
       proximity term is paid at all. Round 29 gave the evader a standoff band
       AND a posture
       reward gated on that band, and it ran TOWARDS its pursuer: being close
       was the precondition for collecting the term meant to teach it to face
       away. A policy that cannot yet steer must not be handed a reward whose
       optimum is "stand near the thing that kills you".

    2. **closing** -- the survival bonus starts coming down and the band comes
       on, so being close AND surviving is worth more than surviving alone.
       The motion shaping is still paid, because this rung is a handover and
       not yet the destination.

    3. **contact** -- holding the band IS the job. The motion shaping comes off
       and the survival bonus drops again, leaving proximity as what the evader
       is actually optimising.

    4. **recursive** -- the same objective, and the loop begins swapping roles:
       freeze the evader, train the chaser against it, and repeat. At this rung
       the evader's terminal is holding contact for the WHOLE episode -- which
       the band already pays tick by tick, so the survival bonus is zero -- and
       the chaser's objective is simply catching it.

       Last for two reasons. A swap hands the opponent whatever the previous
       side learned, so freezing a policy that does not understand the
       objective teaches its opponent to beat a mistake. And the sparser the
       reward, the more it needs a policy that can already generate the
       behaviour often enough to be graded on it.

    Rungs 3 and 4 share a reward shape on purpose -- the difference is the
    LOOP, and `RECURSIVE_STAGE` is how the driver reads it. Inventing a
    settings difference to make them look distinct would be a lie about what
    changes.

    SCAFFOLDING GOES ON AT THE BOTTOM AND COMES OFF AS THE RUNG RISES. Every
    motion term is authored on `long_range`, which is where the policy is
    learning to control its body at all, and each higher rung drops some of
    them: 8 terms, then 5, then 1, then 1. A term withheld from the bottom
    teaches nothing, and a term never removed is a second objective wearing the
    first one's name.

    ORDERING IS WHAT MAKES A FULL TERM SET SAFE. Compass movement lets the
    evader travel without turning, so `strafe` and `travel` pay for circling,
    which is what it did instead of fleeing on every round from 29 to 34.
    Circling is also unwinnable -- run-strafe is 4.40 blocks/s against a pursuer
    sprinting at 7.00, and `sprinting` gates on forward-dominant travel, so a
    strafing evader is locked out of the only gait that keeps up. The answer is
    not to remove those terms but to price them under `forward_travel`, which
    pays the component of a step along the body's own facing and is the only
    term that relates travel to posture. At the authored speeds one tick pays
    3.45 for a forward sprint, 0.72 for a lateral strafe and 0.50 for standing
    still, so the behaviour that wins is also the one that pays best.
    """

    return Ladder([
        SettingsStage(
            "long_range",
            {
                # EVERY TERM ON. This is where the policy learns to control its
                # body, and the terms are ordered rather than withheld.
                #
                # `separation` pays distance the evader's own motion adds and is
                # the objective every higher rung keeps. `forward_travel` pays
                # the component of a step along the body's own facing and is set
                # ABOVE `travel` and `strafe` on purpose: those two pay a step
                # taken in any direction, which a circling evader satisfies
                # without ever turning, so the relationship between facing and
                # travel is what has to be priced highest. `sprint` pays only at
                # real sprint speed, which the gym gates on forward-dominant
                # travel, so it prices the turn as well as the gait.
                # `body_away` and `turn` oppose the chaser prior these weights
                # arrive with -- without them nothing says "stop looking at the
                # thing chasing you".
                #
                # OPENING THE GAP IS THE OBJECTIVE, and `separation` is the only
                # term that can express it: every other one pays a run and a
                # circle identically, and `separation` pays per block the
                # evader's OWN motion adds to the distance. `contact_reward` is
                # zero, which is what `EvaderDuelConfig.band_active` reads to
                # mean "no standoff band" -- so nothing pays for closing, and
                # the band-relative ordering guards are comparisons against a
                # term that does not exist. The two objectives cannot both lead;
                # here it is distance, and the band takes over from `closing` up.
                #
                # 4.0 matches `PursuitStageConfig.evader_progress_scale`, which
                # is what already pays the evader for separation on the other
                # side of the swap. Same number keeps the two roles symmetric,
                # which is the premise of the recursive rung.
                "evader_contact_reward": 0.0,
                "evader_separation_reward": 4.0,
                # Motion is scaffolding, and additive per tick. Normalized by
                # `separation_clip_blocks` (0.238, one sprint tick), a tick of
                # pure radial flight pays 4.0 against 0.0248 for the whole motion
                # mixture, so fleeing outearns circling by ~160x rather than
                # relying on a guard to forbid the farm.
                "evader_forward_travel_reward": 0.02,
                "evader_sprint_reward": 0.02,
                "evader_body_away_reward": 0.02,
                "evader_travel_reward": 0.01,
                "evader_strafe_reward": 0.01,
                # ZERO. It paid a flat 1.0 per tick for any turn past the
                # deadband -- unscaled by distance or progress -- which is 33%
                # of a circling tick's income and free to a spinner.
                "evader_turn_reward": 0.0,
                # Survival is the TERMINAL bonus (`evader_terminal_reward`),
                # paid once for reaching the horizon uncaught. A per-tick alive
                # term pays existing, and the cheapest way to exist is to stand
                # still, which is why it is zero here.
                "evader_alive_reward": 0.0,
                # The band does NOT lead here, and with `contact_reward` at zero
                # there is no band to lead. Declaring it anyway is what keeps
                # `band_governs` false if a small band is ever restored, and it
                # ungates `body_away_reward`: with a band that term is scaled by
                # `contact`, so being close is the precondition for being paid to
                # face away, which pulled the evader TOWARDS its pursuer. Facing
                # away is the whole lesson here, so it is paid wherever it is.
                "evader_band_leads": False,
                # ZERO. Positive-only objective: being caught is punished by
                # ENDING the episode, which ends the per-tick separation income
                # and forfeits the survival bonus. A 200 penalty against 5.5 of
                # realised income made every episode worth about -194 with no
                # variance -- 2212 catches in 2212 episodes, so the gradient
                # carried no signal and there were no positive examples at all.
                "evader_caught_penalty": 0.0,
                # CLOSED, and therefore uncharged. Jump opens at `closing`, and
                # the cost opens with it: a charge that can never be levied is
                # dead weight in a contract whose point is to stay thin. This is
                # the same scaffolding order as every other term here -- the
                # bottom rung teaches ground locomotion, and a head the policy
                # cannot yet use productively is one more way to not learn it.
                "evader_allow_jump": False,
                "evader_airborne_idle_cost": 0.0,
            },
            **_OBJECTIVE_GATE,
        ),
        SettingsStage(
            "closing",
            {
                # FEWER TERMS AND SMALLER ONES. `forward_travel`, `sprint` and
                # `alive` come off: a policy that reaches this rung already
                # sprints away, and a term that has done its job is scaffolding.
                # What is left is halved, because it is now a nudge on top of
                # `separation` rather than the thing being taught. Jump opens,
                # so `airborne_idle_cost` comes on to price a hop that covers no
                # ground.
                "evader_contact_reward": 0.10,
                "evader_separation_reward": 0.0,
                "evader_forward_travel_reward": 0.02,
                "evader_sprint_reward": 0.02,
                "evader_body_away_reward": 0.0,
                "evader_travel_reward": 0.0,
                "evader_strafe_reward": 0.0,
                "evader_turn_reward": 0.0,
                "evader_alive_reward": 0.0,
                "evader_band_leads": True,
                # 0.10 * 1024 = 102.4 of band income in a full episode.
                "evader_caught_penalty": 0.0,
                "evader_allow_jump": True,
                "evader_airborne_idle_cost": 0.05,
            },
            **_OBJECTIVE_GATE,
        ),
        SettingsStage(
            "contact",
            {
                # ONE TERM. Open ground, by whatever means it has learned.
                "evader_contact_reward": 0.10,
                "evader_separation_reward": 0.0,
                "evader_sprint_reward": 0.0,
                "evader_body_away_reward": 0.0,
                "evader_alive_reward": 0.0,
                "evader_travel_reward": 0.0,
                "evader_forward_travel_reward": 0.0,
                "evader_strafe_reward": 0.0,
                "evader_turn_reward": 0.0,
                "evader_band_leads": True,
                "evader_caught_penalty": 0.0,
                "evader_allow_jump": True,
                "evader_airborne_idle_cost": 0.05,
            },
            **_OBJECTIVE_GATE,
        ),
        SettingsStage(
            RECURSIVE_STAGE,
            {
                # Same single term. What changes here is the LOOP, not the
                # reward: the sides start swapping. Inventing a reward
                # difference to make this rung look distinct would be a lie
                # about what actually changes.
                "evader_contact_reward": 0.10,
                "evader_separation_reward": 0.0,
                "evader_sprint_reward": 0.0,
                "evader_body_away_reward": 0.0,
                "evader_alive_reward": 0.0,
                "evader_travel_reward": 0.0,
                "evader_forward_travel_reward": 0.0,
                "evader_strafe_reward": 0.0,
                "evader_turn_reward": 0.0,
                "evader_band_leads": True,
                "evader_caught_penalty": 0.0,
                "evader_allow_jump": True,
                "evader_airborne_idle_cost": 0.05,
            },
            **_OBJECTIVE_GATE,
        ),
    ])


#: How much of a full episode of per-tick income the SURVIVAL bonus is worth,
#: per rung. It shrinks as proximity takes over, which is the handover the whole
#: ladder exists to perform.
#:
#: Zero at the top is the point, not an edge case: the evader's terminal there
#: IS holding contact for the whole episode, which the band already pays tick by
#: tick. `caught_penalty` is decoupled and stays large throughout, so "no
#: survival bonus" never means "being caught is free".
#:
#: A SCHEDULE, not a per-cycle decay. The decay was keyed on the terrain cycle,
#: which only advances once both sides have won -- and that cannot happen before
#: the recursive rung, so on every rung below it the terminal never actually
#: moved.
#: Episode length per rung, in ticks. 30 per second, so 512 is 17 s and 1024 is
#: 34 s.
#:
#: It only ever goes UP. A shorter round is a reward for outlasting a clock: at
#: 64-128 ticks the chaser has not crossed the opening gap, so the evader
#: "survives" without ever opening ground and gets promoted for waiting. Rounds
#: 11-13 and 17-19 were all won that way and none transferred. There is no
#: easing and no dial -- the rung picks the length and the driver refuses
#: anything shorter than `long_range` carries.
#:
#: 1024 from `closing` onward and it stays there: once the band is the job, the
#: evader has to hold a matchup rather than reach an end, and a longer episode
#: is the only thing that distinguishes the two.
HORIZON_BY_STAGE = {
    "long_range": 512,
    "closing": 1024,
    "contact": 1024,
    RECURSIVE_STAGE: 1024,
}

#: SURVIVING IS THE OBJECTIVE, so the terminal has to dominate what a whole
#: episode of proximity pays. `terminal_for` computes
#: `multiple * horizon * tick_scale`, so at the old 2.0 the bonus for lasting a
#: full 512 ticks was 40.96 against 51.2 of band income over the same episode --
#: reaching the end was worth LESS than the shaping accumulated getting there,
#: and with `caught_penalty` now zero there was nothing left making a catch bad.
#: At 20.0 the terminal is 409.6, eight times the band, so the return is led by
#: whether the episode was survived and shaped by how close it was held.
#:
#: The decay across rungs is kept: by `contact` the pair should be holding a
#: matchup rather than racing a clock, and `recursive` retires the bonus so the
#: band is the whole return once the sides start swapping.
TERMINAL_MULTIPLE_BY_STAGE = {
    "long_range": 20.0,
    "closing": 10.0,
    "contact": 2.5,
    RECURSIVE_STAGE: 0.0,
}


def describe() -> dict[str, object]:
    """The curriculum as a payload, for the console and for a run receipt."""

    ladder = terrain_ladder()
    objectives = objective_ladder()
    return {
        "schema": "arena-pursuit-curriculum-v1",
        "objectives": {
            "gate": dict(_OBJECTIVE_GATE),
            "recursive_stage": RECURSIVE_STAGE,
            "terminal_multiple": dict(TERMINAL_MULTIPLE_BY_STAGE),
            # Exposed because a rung is not launchable without it. `settings`
            # alone omits the episode length, so a caller building a run from
            # this payload silently used the form's horizon and every rung
            # trained at the same length -- which is the one axis
            # `HORIZON_BY_STAGE` exists to move.
            "horizon_by_stage": dict(HORIZON_BY_STAGE),
            # The learner is the EVADER on every rung here, and `pursuit_run`
            # refuses `objective="evade"` unless the opponent actually pursues.
            # Stated once, next to the rungs, so a caller cannot infer a role
            # from the `evader_` prefix and get the pairing wrong.
            "objective": "evade",
            "target_controller": "frozen_source_policy",
            "stages": [
                {"name": stage.name, "settings": dict(stage.settings)}
                for stage in objectives
            ],
        },
        "horizon_band": {
            "minimum": HORIZON_BAND[0],
            "default": HORIZON_BAND[1],
            "maximum": HORIZON_BAND[2],
            "ticks_per_second": 30,
            "note": "a dial the loop moves within a round, not a ladder rung",
        },
        "gate": dict(_GATE),
        "stages": [
            {"name": stage.name, "settings": dict(stage.settings)}
            for stage in ladder
        ],
    }


__all__ = [
    "HORIZON_BAND",
    "TERMINAL_MULTIPLE_BY_STAGE",
    "RECURSIVE_STAGE",
    "describe",
    "objective_ladder",
    "terrain_ladder",
]
