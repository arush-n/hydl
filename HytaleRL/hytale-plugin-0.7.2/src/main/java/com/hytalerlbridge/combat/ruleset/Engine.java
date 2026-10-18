package com.hytalerlbridge.combat.ruleset;

import static com.hytalerlbridge.combat.ruleset.RulesetChecks.require;

/** Extracted verbatim from {@code CombatRuleset}. */
public record Engine(
    int ticksPerSecond,
    double nominalDeltaSeconds,
    double loadedDeltaSeconds,
    int motionTimingProfileCount,
    double gravity,
    double steeringSlowdownFalloff,
    double horizontalSelectorPi,
    double legacyHorizontalKnockbackScale,
    double legacyMotionControllerHorizontalFactor
) {
    public Engine {
        require(
            horizontalSelectorPi > 0.0
                && legacyHorizontalKnockbackScale > 0.0
                && legacyMotionControllerHorizontalFactor > 0.0,
            "engine selector and knockback constants"
        );
    }
}
