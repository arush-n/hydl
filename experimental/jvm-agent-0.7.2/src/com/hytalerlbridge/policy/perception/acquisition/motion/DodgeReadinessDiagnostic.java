package com.hytalerlbridge.policy.perception.acquisition.motion;

import java.util.Locale;

/** Pure classifier separating corridor acquisition from final policy gates. */
public final class DodgeReadinessDiagnostic {

    public enum Reason {
        READY,
        ACQUISITION_PRECONDITION,
        MOTION_EVALUATION_FAILED,
        SWEEP_UNAVAILABLE,
        CORRIDOR_BLOCKED,
        MOVEMENT_DISABLED,
        ACTOR_DEAD,
        STAMINA_UNAVAILABLE,
        STAMINA_NONFINITE,
        INSUFFICIENT_STAMINA,
        AUTHORED_DIRECTION_CLOSED,
    }

    public record Snapshot(
        Reason reason,
        ServerDodgeCorridorReader.Capture corridor,
        boolean movementEnabled,
        boolean alive,
        boolean staminaAvailable,
        float stamina,
        float dodgeCost,
        boolean[] finalDirectionMask
    ) {
        public Snapshot {
            if (reason == null || corridor == null
                || finalDirectionMask == null
                || finalDirectionMask.length
                    != DodgeCorridorEvaluator.DIRECTION_COUNT) {
                throw new IllegalArgumentException(
                    "invalid Dodge readiness diagnostic");
            }
            finalDirectionMask = finalDirectionMask.clone();
        }

        @Override public boolean[] finalDirectionMask() {
            return finalDirectionMask.clone();
        }

        /** Stable one-line receipt intended for saved server stdout. */
        public String receipt() {
            StringBuilder value = new StringBuilder(512);
            value.append("schema=hytalerl_dodge_readiness_v1")
                .append(" reason=").append(reason.name().toLowerCase(Locale.ROOT))
                .append(" acquisition=")
                .append(corridor.acquisitionReason().name().toLowerCase(Locale.ROOT))
                .append(" evaluation=")
                .append(corridor.evaluation().reason().name().toLowerCase(Locale.ROOT))
                .append(" raw_mask=").append(mask(corridor.clear()))
                .append(" final_mask=").append(mask(finalDirectionMask))
                .append(" movement_enabled=").append(movementEnabled)
                .append(" alive=").append(alive)
                .append(" stamina_available=").append(staminaAvailable)
                .append(" stamina=").append(number(stamina))
                .append(" dodge_cost=").append(number(dodgeCost))
                .append(" grounded=").append(corridor.grounded())
                .append(" world_tps=").append(corridor.worldTicksPerSecond())
                .append(" expected_tps=").append(corridor.expectedTicksPerSecond())
                .append(" bounds=")
                .append(number(corridor.boundsWidth())).append('x')
                .append(number(corridor.boundsHeight())).append('x')
                .append(number(corridor.boundsDepth()))
                .append(" motion_ticks=").append(corridor.evaluation().path().ticks())
                .append(" motion_complete=")
                .append(corridor.evaluation().path().complete());
            for (ServerDodgeCorridorReader.SweepEvidence sweep
                    : corridor.sweeps()) {
                value.append(' ')
                    .append(sweep.direction().name().toLowerCase(Locale.ROOT))
                    .append('=')
                    .append(sweep.result().name().toLowerCase(Locale.ROOT))
                    .append(":blocks=").append(sweep.blockCollisionCount())
                    .append(":slide=").append(sweep.supportingSlide())
                    .append(":obstructed=").append(sweep.obstructed())
                    .append(":stop_t=").append(number(sweep.stopFraction()))
                    .append(":blocking_index=")
                    .append(sweep.blockingCollisionIndex())
                    .append(":ignored_slide=")
                    .append(sweep.ignoredSlideContacts())
                    .append(":ignored_grazing=")
                    .append(sweep.ignoredGrazingContacts())
                    .append(":late_approach=")
                    .append(sweep.lateApproachContacts())
                    .append(":first_t=").append(number(sweep.firstCollisionStart()))
                    .append(":normal=")
                    .append(number(sweep.firstNormalX())).append(',')
                    .append(number(sweep.firstNormalY())).append(',')
                    .append(number(sweep.firstNormalZ()))
                    .append(":upward_normal=")
                    .append(sweep.firstUpwardNormal())
                    .append(":touching=").append(sweep.firstTouching())
                    .append(":overlap=").append(sweep.firstOverlapping())
                    .append(":block=")
                    .append(sweep.firstBlockX()).append(',')
                    .append(sweep.firstBlockY()).append(',')
                    .append(sweep.firstBlockZ()).append(',')
                    .append(sweep.firstBlockId())
                    .append(":unavailable=")
                    .append(sweep.unavailableType().isEmpty()
                        ? "none" : sweep.unavailableType());
            }
            return value.toString();
        }
    }

    private DodgeReadinessDiagnostic() {}

    public static Snapshot classify(
        ServerDodgeCorridorReader.Capture corridor,
        boolean movementEnabled,
        boolean alive,
        boolean staminaAvailable,
        float stamina,
        float dodgeCost,
        boolean[] finalDirectionMask
    ) {
        Reason reason;
        if (corridor.acquisitionReason()
                != ServerDodgeCorridorReader.AcquisitionReason.READY) {
            reason = Reason.ACQUISITION_PRECONDITION;
        } else if (corridor.evaluation().reason()
                != DodgeCorridorEvaluator.EvaluationReason.READY) {
            reason = Reason.MOTION_EVALUATION_FAILED;
        } else if (hasUnavailable(corridor.sweeps())) {
            reason = Reason.SWEEP_UNAVAILABLE;
        } else if (!any(corridor.clear())) {
            reason = Reason.CORRIDOR_BLOCKED;
        } else if (!movementEnabled) {
            reason = Reason.MOVEMENT_DISABLED;
        } else if (!alive) {
            reason = Reason.ACTOR_DEAD;
        } else if (!staminaAvailable) {
            reason = Reason.STAMINA_UNAVAILABLE;
        } else if (!Float.isFinite(stamina) || !Float.isFinite(dodgeCost)) {
            reason = Reason.STAMINA_NONFINITE;
        } else if (stamina < dodgeCost) {
            reason = Reason.INSUFFICIENT_STAMINA;
        } else if (!any(finalDirectionMask)) {
            reason = Reason.AUTHORED_DIRECTION_CLOSED;
        } else {
            reason = Reason.READY;
        }
        return new Snapshot(
            reason,
            corridor,
            movementEnabled,
            alive,
            staminaAvailable,
            stamina,
            dodgeCost,
            finalDirectionMask
        );
    }

    private static boolean hasUnavailable(
        ServerDodgeCorridorReader.SweepEvidence[] values
    ) {
        for (ServerDodgeCorridorReader.SweepEvidence value : values) {
            if (value.result() == DodgeCorridorEvaluator.SweepResult.UNAVAILABLE) {
                return true;
            }
        }
        return false;
    }

    private static boolean any(boolean[] values) {
        for (boolean value : values) if (value) return true;
        return false;
    }

    private static String mask(boolean[] values) {
        StringBuilder result = new StringBuilder(values.length);
        for (boolean value : values) result.append(value ? '1' : '0');
        return result.toString();
    }

    private static String number(double value) {
        return Double.isFinite(value)
            ? String.format(Locale.ROOT, "%.6f", value)
            : "na";
    }
}
