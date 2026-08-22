package com.hytalerlbridge.nativebackend.model;


/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record TargetState(
    boolean present,
    String roleName,
    double health,
    double maxHealth,
    double distance,
    double x,
    double y,
    double z,
    double yawDegrees,
    double headYawDegrees,
    double headPitchDegrees,
    double facingErrorDegrees,
    double velocityX,
    double velocityZ
) {
    public static TargetState absent() {
        return new TargetState(
            false,
            "",
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        );
    }
}
