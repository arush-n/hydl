package com.hytalerlbridge.combat.ruleset;

import static com.hytalerlbridge.combat.ruleset.RulesetChecks.require;

/** Extracted verbatim from {@code CombatRuleset}. */
public record WalkController(
    double gravity,
    double fallAccelerationMultiplier,
    double gravityDragExponent,
    double maxFallSpeed,
    double maxSinkSpeedFluid,
    double maxClimbHeight,
    double maxDropHeight
) {
    public WalkController {
        require(
            gravity > 0.0
                && fallAccelerationMultiplier > 0.0
                && gravityDragExponent > 0.0
                && maxFallSpeed > 0.0
                && maxSinkSpeedFluid > 0.0
                && maxClimbHeight > 0.0
                && maxDropHeight > maxClimbHeight,
            "agent walk controller values"
        );
    }
}
