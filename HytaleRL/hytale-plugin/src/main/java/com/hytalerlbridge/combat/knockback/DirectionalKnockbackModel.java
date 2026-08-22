package com.hytalerlbridge.combat.knockback;

import com.hytalerlbridge.combat.CombatRuleset;
import com.hytalerlbridge.combat.ruleset.Agent;
import com.hytalerlbridge.combat.ruleset.DirectionalKnockback;
import com.hytalerlbridge.combat.ruleset.Engine;

/**
 * Exact flat-fixture 0.5.7 directional-knockback and NPC forced-walk math.
 *
 * <p>The authored interaction first builds a direction from victim to
 * attacker and adds its head-relative vector. The legacy damage path scales
 * horizontal components separately from vertical velocity. A
 * {@code VelocityType=Set} NPC instruction clears the controller velocity,
 * applies that vector, and then adds a direction-dependent fraction of the
 * controller's pre-hit walk speed.</p>
 */
public final class DirectionalKnockbackModel {

    private DirectionalKnockbackModel() {}

    public static Velocity pendingExternalVelocity(
        double attackerX,
        double attackerY,
        double attackerZ,
        double victimX,
        double victimY,
        double victimZ,
        double attackerHeadYawDegrees,
        CombatRuleset rules
    ) {
        DirectionalKnockback knockback =
            rules.damageInteraction().knockback();
        double yaw = Math.toRadians(attackerHeadYawDegrees);

        double baseX = attackerX - victimX;
        double baseY = attackerY - victimY;
        double baseZ = attackerZ - victimZ;
        double baseLength = Math.sqrt(
            baseX * baseX + baseY * baseY + baseZ * baseZ
        );
        if (baseLength > 0.0) {
            baseX /= baseLength;
            baseZ /= baseLength;
        } else {
            baseX = -Math.sin(yaw);
            baseZ = -Math.cos(yaw);
        }

        // JOML Vector3d.rotateY: x' = x*cos + z*sin,
        // z' = -x*sin + z*cos.
        double relativeX =
            knockback.relativeX() * Math.cos(yaw)
                + knockback.relativeZ() * Math.sin(yaw);
        double relativeZ =
            -knockback.relativeX() * Math.sin(yaw)
                + knockback.relativeZ() * Math.cos(yaw);
        double authoredX = (baseX + relativeX) * knockback.force();
        double authoredZ = (baseZ + relativeZ) * knockback.force();

        Agent agent = rules.agent();
        Engine engine = rules.engine();
        double horizontalScale =
            engine.legacyHorizontalKnockbackScale()
                * agent.knockbackScale()
                * engine.legacyMotionControllerHorizontalFactor()
                * agent.movementVelocityResistance();
        return new Velocity(
            authoredX * horizontalScale,
            knockback.velocityY() * agent.knockbackScale(),
            authoredZ * horizontalScale
        );
    }

    public static ForcedPush forcePushedMotion(
        Velocity externalVelocity,
        double priorMoveSpeed,
        double victimYawDegrees,
        CombatRuleset rules
    ) {
        double resultX = externalVelocity.x();
        double resultZ = externalVelocity.z();
        double resultingMoveSpeed = priorMoveSpeed;
        double externalLength = Math.hypot(resultX, resultZ);
        if (externalLength > 0.0 && priorMoveSpeed > 0.0) {
            double yaw = Math.toRadians(victimYawDegrees);
            double headingX = -Math.sin(yaw);
            double headingZ = -Math.cos(yaw);
            double alignment = (
                headingX * resultX + headingZ * resultZ
            ) / externalLength;
            double maxWalkSpeedAfterHitMultiplier =
                1.0 - rules.agent().minHitSlowdown();
            double walkMultiplier = Math.min(
                (alignment + 1.0) / 2.0,
                maxWalkSpeedAfterHitMultiplier
            );
            resultingMoveSpeed = priorMoveSpeed * walkMultiplier;
            if (resultingMoveSpeed > rules.agent().minWalkSpeed()) {
                resultX += headingX * resultingMoveSpeed;
                resultZ += headingZ * resultingMoveSpeed;
            } else {
                resultingMoveSpeed = 0.0;
            }
        }
        return new ForcedPush(
            new Velocity(
                resultX,
                externalVelocity.y(),
                resultZ
            ),
            resultingMoveSpeed
        );
    }

    public record ForcedPush(Velocity velocity, double moveSpeed) {
        public ForcedPush {
            if (velocity == null
                || !Double.isFinite(moveSpeed)
                || moveSpeed < 0.0) {
                throw new IllegalArgumentException(
                    "forced-push result must be finite"
                );
            }
        }
    }

    public record Velocity(double x, double y, double z) {
        public Velocity {
            if (!Double.isFinite(x)
                || !Double.isFinite(y)
                || !Double.isFinite(z)) {
                throw new IllegalArgumentException(
                    "knockback velocity must be finite"
                );
            }
        }
    }
}
