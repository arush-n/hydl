package com.hytalerlbridge.nativebackend.group;

/** First-control-tick admission result for one policy actor's base attack. */
public record NativeAttackAdmission(boolean accepted, String rejectReason) {

    public NativeAttackAdmission {
        rejectReason = rejectReason == null ? "" : rejectReason;
        if (accepted && !rejectReason.isEmpty()) {
            throw new IllegalArgumentException(
                "accepted native attack cannot carry a rejection reason"
            );
        }
        if (!accepted && rejectReason.isEmpty()) {
            throw new IllegalArgumentException(
                "rejected native attack requires a reason"
            );
        }
    }

    public static NativeAttackAdmission admitted() {
        return new NativeAttackAdmission(true, "");
    }

    public static NativeAttackAdmission rejected(String reason) {
        return new NativeAttackAdmission(false, reason);
    }
}
