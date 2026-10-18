package com.hytalerlbridge.combat.ruleset;

import static com.hytalerlbridge.combat.ruleset.RulesetChecks.require;

/** Extracted verbatim from {@code CombatRuleset}. */
public record MaintainDistance(
    double activationRange,
    double desiredDistanceMin,
    double desiredDistanceMax,
    double moveThreshold,
    double targetDistanceFactor,
    double moveTowardsSlowdownDistance,
    double relativeForwardSpeed,
    double relativeBackwardSpeed,
    double strafingDurationMinSeconds,
    double strafingDurationMaxSeconds,
    double strafingFrequencyMinSeconds,
    double strafingFrequencyMaxSeconds,
    double strafingYawOffsetDegrees,
    double strafingTranslationOffsetDegrees
) {
    public MaintainDistance {
        require(activationRange > 0.0, "maintain-distance activation range");
        require(
            desiredDistanceMin >= 0.0
                && desiredDistanceMax >= desiredDistanceMin,
            "maintain-distance desired range"
        );
        require(moveThreshold > 0.0, "maintain-distance move threshold");
        require(
            targetDistanceFactor >= 0.0 && targetDistanceFactor <= 1.0,
            "maintain-distance target factor"
        );
        require(
            moveTowardsSlowdownDistance >= 0.0,
            "maintain-distance slowdown"
        );
        require(
            relativeForwardSpeed > 0.0 && relativeBackwardSpeed > 0.0,
            "maintain-distance relative speeds"
        );
        require(
            validRange(
                strafingDurationMinSeconds,
                strafingDurationMaxSeconds
            ),
            "maintain-distance strafe duration"
        );
        require(
            validRange(
                strafingFrequencyMinSeconds,
                strafingFrequencyMaxSeconds
            ),
            "maintain-distance strafe frequency"
        );
        require(
            strafingYawOffsetDegrees >= 0.0
                && strafingTranslationOffsetDegrees >= 0.0,
            "maintain-distance strafe offsets"
        );
    }

    private static boolean validRange(double minimum, double maximum) {
        return minimum >= 0.0 && maximum >= minimum;
    }
}
