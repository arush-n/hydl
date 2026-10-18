package com.hytalerlbridge.combat.ruleset;

import static com.hytalerlbridge.combat.ruleset.RulesetChecks.require;

/** Extracted verbatim from {@code CombatRuleset}. */
public record DirectionalKnockback(
    KnockbackType type,
    double force,
    double relativeX,
    double relativeZ,
    double velocityY,
    KnockbackVelocityType velocityType,
    double durationSeconds
) {
    public DirectionalKnockback {
        require(type == KnockbackType.DIRECTIONAL, "knockback type");
        require(
            velocityType == KnockbackVelocityType.SET,
            "knockback velocity type"
        );
        require(force >= 0.0, "knockback force");
        require(durationSeconds == 0.0, "knockback duration");
    }
}
