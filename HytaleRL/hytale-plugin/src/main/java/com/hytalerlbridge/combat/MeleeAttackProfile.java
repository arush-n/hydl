package com.hytalerlbridge.combat;

/** Immutable 30 Hz projection of one authored Hytale melee selector. */
public record MeleeAttackProfile(
    String interactionId,
    int windupTicks,
    int sweepTicks,
    int recoveryTicks,
    double selectorRuntimeSeconds,
    double startDistance,
    double endDistance,
    double arcDegrees,
    SweepDirection direction,
    double yawStartOffsetDegrees,
    double pitchOffsetDegrees,
    double rollOffsetDegrees,
    double extendTop,
    double extendBottom,
    boolean testLineOfSight,
    double damage
) {
    public enum SweepDirection {
        LEFT,
        RIGHT,
        CENTER
    }

    public MeleeAttackProfile {
        if (interactionId == null || interactionId.isBlank()) {
            throw new IllegalArgumentException("interactionId is required");
        }
        if (windupTicks < 0 || sweepTicks < 1 || recoveryTicks < 0) {
            throw new IllegalArgumentException("attack phase ticks are invalid");
        }
        if (!Double.isFinite(selectorRuntimeSeconds)
            || selectorRuntimeSeconds <= 0.0) {
            throw new IllegalArgumentException("selector runtime is invalid");
        }
        if (startDistance < 0.0 || endDistance <= startDistance) {
            throw new IllegalArgumentException("selector distance is invalid");
        }
        if (!Double.isFinite(arcDegrees)
            || arcDegrees <= 0.0
            || arcDegrees > 360.0
            || !Double.isFinite(yawStartOffsetDegrees)
            || !Double.isFinite(pitchOffsetDegrees)
            || !Double.isFinite(rollOffsetDegrees)
            || !Double.isFinite(damage)
            || damage < 0.0) {
            throw new IllegalArgumentException("selector arc/damage is invalid");
        }
        if (!Double.isFinite(extendTop)
            || !Double.isFinite(extendBottom)
            || extendTop < 0.0
            || extendBottom < 0.0) {
            throw new IllegalArgumentException("selector vertical extension is invalid");
        }
    }

    public int chainTicks() {
        return windupTicks + sweepTicks + recoveryTicks;
    }

    public CombatPhase phaseAt(int elapsedTicks) {
        if (elapsedTicks <= 0) return CombatPhase.IDLE;
        if (elapsedTicks <= windupTicks) return CombatPhase.WINDUP;
        if (elapsedTicks <= windupTicks + sweepTicks) return CombatPhase.SWEEP;
        return CombatPhase.RECOVERY;
    }

    public double phaseProgressAt(int elapsedTicks) {
        CombatPhase phase = phaseAt(elapsedTicks);
        return switch (phase) {
            case WINDUP -> fraction(elapsedTicks, windupTicks);
            case SWEEP -> fraction(elapsedTicks - windupTicks, sweepTicks);
            case RECOVERY -> fraction(
                elapsedTicks - windupTicks - sweepTicks,
                Math.max(1, recoveryTicks)
            );
            case IDLE -> 0.0;
            case COOLDOWN -> 1.0;
        };
    }

    /** Zero-based selector sample; the first native sweep sample has delta zero. */
    public int selectorSampleAt(int elapsedTicks) {
        return Math.max(0, elapsedTicks - windupTicks - 1);
    }

    private static double fraction(int numerator, int denominator) {
        if (denominator <= 0) return 1.0;
        return Math.max(0.0, Math.min(1.0, numerator / (double) denominator));
    }
}
