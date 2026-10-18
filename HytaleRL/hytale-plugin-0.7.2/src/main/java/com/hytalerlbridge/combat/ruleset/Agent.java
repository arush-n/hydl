package com.hytalerlbridge.combat.ruleset;

import java.util.List;
import static com.hytalerlbridge.combat.ruleset.RulesetChecks.require;
import static com.hytalerlbridge.combat.ruleset.RulesetChecks.checkedBounds;

/** Extracted verbatim from {@code CombatRuleset}. */
public record Agent(
    double maxHealth,
    double assetMaxWalkSpeed,
    double maxSpeed,
    double acceleration,
    double jumpVelocityGravityFloor,
    double jumpHeightParameter,
    double knockbackScale,
    double movementVelocityResistance,
    double minWalkSpeed,
    double minHitSlowdown,
    double landingVelocityScale,
    double steeringRelativeTurnSpeed,
    double turnDegreesPerSecond,
    WalkController walkController,
    double[] boundingBox,
    double damage,
    double attackPauseMinSeconds,
    double attackPauseMaxSeconds,
    List<AgentAttack> attacks
) {
    public Agent {
        boundingBox = checkedBounds(boundingBox, "agent boundingBox");
        require(
            Math.abs(maxSpeed - assetMaxWalkSpeed) <= 1.0e-12,
            "agent max speed derivation"
        );
        require(
            knockbackScale > 0.0
                && movementVelocityResistance > 0.0
                && minWalkSpeed >= 0.0
                && minHitSlowdown >= 0.0
                && minHitSlowdown <= 1.0,
            "agent knockback controller"
        );
        require(walkController != null, "agent walk controller");
        require(
            attackPauseMinSeconds >= 0.0
                && attackPauseMaxSeconds >= attackPauseMinSeconds,
            "agent attack pause range"
        );
    }

    @Override
    public double[] boundingBox() {
        return boundingBox.clone();
    }
}
