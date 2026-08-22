package com.hytalerlbridge.combat.ruleset;

import com.hytalerlbridge.combat.MeleeAttackProfile;

/** Extracted verbatim from {@code CombatRuleset}. */
public record TargetAttack(
    String interactionId,
    int windupTicks,
    int sweepTicks,
    int recoveryTicks,
    double selectorRuntimeSeconds,
    double startDistance,
    double endDistance,
    double arcDegrees,
    MeleeAttackProfile.SweepDirection sweepDirection,
    double yawStartOffsetDegrees,
    double pitchOffsetDegrees,
    double rollOffsetDegrees,
    double extendTop,
    double extendBottom,
    boolean requiresLineOfSight,
    double damage
) {
    public MeleeAttackProfile profile() {
        return new MeleeAttackProfile(
            interactionId,
            windupTicks,
            sweepTicks,
            recoveryTicks,
            selectorRuntimeSeconds,
            startDistance,
            endDistance,
            arcDegrees,
            sweepDirection,
            yawStartOffsetDegrees,
            pitchOffsetDegrees,
            rollOffsetDegrees,
            extendTop,
            extendBottom,
            requiresLineOfSight,
            damage
        );
    }
}
