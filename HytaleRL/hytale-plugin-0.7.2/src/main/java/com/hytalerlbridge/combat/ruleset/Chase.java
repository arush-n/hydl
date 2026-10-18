package com.hytalerlbridge.combat.ruleset;

import static com.hytalerlbridge.combat.ruleset.RulesetChecks.require;

/** Extracted verbatim from {@code CombatRuleset}. */
public record Chase(double stopDistance, double slowdownDistance) {
    public Chase {
        require(stopDistance >= 0.0, "target chase stop distance");
        require(
            slowdownDistance > stopDistance,
            "target chase slowdown distance"
        );
    }
}
