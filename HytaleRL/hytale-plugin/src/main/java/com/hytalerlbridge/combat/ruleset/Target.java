package com.hytalerlbridge.combat.ruleset;

import java.util.List;
import static com.hytalerlbridge.combat.ruleset.RulesetChecks.require;
import static com.hytalerlbridge.combat.ruleset.RulesetChecks.checkedBounds;

/** Extracted verbatim from {@code CombatRuleset}. */
public record Target(
    double maxHealth,
    double assetMaxWalkSpeed,
    double relativeChaseSpeed,
    double chaseSpeed,
    double acceleration,
    TargetVelocityControl velocityControl,
    AgentCollisionMode agentCollisionMode,
    double assetEyeHeight,
    double modelScale,
    double effectiveEyeHeight,
    double[] boundingBox,
    double maxHeadRotationDegreesPerSecond,
    double headAimRelativeTurnSpeed,
    double headDefaultRelativeTurnSpeed,
    double headYawMinDegrees,
    double headYawMaxDegrees,
    double headPitchMinDegrees,
    double headPitchMaxDegrees,
    Chase chase,
    MaintainDistance maintainDistance,
    int chaseReactionTicks,
    double turnDegreesPerSecond,
    int activationMinTick,
    int activationMaxTick,
    int decisionDelayTicks,
    double sensorRange,
    double attackPauseMinSeconds,
    double attackPauseMaxSeconds,
    List<TargetAttack> attacks
) {
    public Target {
        boundingBox = checkedBounds(boundingBox, "target boundingBox");
        require(
            Math.abs(
                chaseSpeed - assetMaxWalkSpeed * relativeChaseSpeed
            ) <= 1.0e-12,
            "target chase speed derivation"
        );
        require(
            velocityControl == TargetVelocityControl.STEERING_TRANSLATION,
            "target velocity control"
        );
        require(
            agentCollisionMode == AgentCollisionMode.NON_BLOCKING,
            "target/agent collision mode"
        );
        require(chaseReactionTicks == 1, "target chase reaction");
        require(
            Math.abs(
                effectiveEyeHeight - assetEyeHeight * modelScale
            ) <= 1.0e-12,
            "target effective eye height derivation"
        );
        require(
            maxHeadRotationDegreesPerSecond > 0.0
                && headAimRelativeTurnSpeed > 0.0
                && headDefaultRelativeTurnSpeed > 0.0,
            "target head speeds"
        );
        require(
            headYawMaxDegrees > headYawMinDegrees
                && headPitchMaxDegrees > headPitchMinDegrees,
            "target head ranges"
        );
        require(chase != null, "target chase");
        require(maintainDistance != null, "target maintain distance");
        require(
            attackPauseMinSeconds >= 0.0
                && attackPauseMaxSeconds >= attackPauseMinSeconds,
            "target attack pause range"
        );
    }

    @Override
    public double[] boundingBox() {
        return boundingBox.clone();
    }
}
