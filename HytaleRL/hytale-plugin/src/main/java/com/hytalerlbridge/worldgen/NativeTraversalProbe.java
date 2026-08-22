package com.hytalerlbridge.worldgen;

/**
 * Bounded native CollisionModule evidence for stationary traversal samples.
 *
 * <p>Positions are actor transform positions. Validation codes use
 * CollisionModule's -1/0/1/2/3 invalid/clear/ground/ceiling/ground+ceiling
 * contract. Upward distance is measured only for non-overlapping samples.</p>
 */
public record NativeTraversalProbe(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    double[] positions,
    double[] upwardLimits,
    double[] actorBounds,
    byte[] validationCodes,
    double[] upwardCollisionDistances
) {
    public static final String SCHEMA = "hytalerl_native_traversal_probe_v1";
    public static final int VERSION = 1;
    public static final int MAX_SAMPLES = 4096;
    public static final int VALIDATE_INVALID = -1;
    public static final int VALIDATE_CLEAR = 0;
    public static final int VALIDATE_ON_GROUND = 1;
    public static final int VALIDATE_TOUCH_CEILING = 2;

    public NativeTraversalProbe {
        requireText(serverVersion, "serverVersion");
        requireText(worldName, "worldName");
        requireText(worldgenProvider, "worldgenProvider");
        requireText(worldgenVersion, "worldgenVersion");
        positions = positions == null ? new double[0] : positions.clone();
        upwardLimits = upwardLimits == null ? new double[0] : upwardLimits.clone();
        actorBounds = actorBounds == null ? new double[0] : actorBounds.clone();
        validationCodes = validationCodes == null
            ? new byte[0]
            : validationCodes.clone();
        upwardCollisionDistances = upwardCollisionDistances == null
            ? new double[0]
            : upwardCollisionDistances.clone();
        int count = validationCodes.length;
        if (
            count < 1
                || count > MAX_SAMPLES
                || positions.length != count * 3
                || upwardLimits.length != count
                || actorBounds.length != 6
                || upwardCollisionDistances.length != count
        ) {
            throw new IllegalArgumentException(
                "Traversal probe arrays disagree or exceed capacity"
            );
        }
        requireFinite(positions, "positions");
        requireFinite(upwardLimits, "upwardLimits");
        requireFinite(actorBounds, "actorBounds");
        requireFinite(upwardCollisionDistances, "upwardCollisionDistances");
        if (
            actorBounds[0] >= actorBounds[3]
                || actorBounds[1] >= actorBounds[4]
                || actorBounds[2] >= actorBounds[5]
        ) {
            throw new IllegalArgumentException("actorBounds must be a positive AABB");
        }
        for (int index = 0; index < count; index++) {
            int code = validationCodes[index];
            double limit = upwardLimits[index];
            double distance = upwardCollisionDistances[index];
            if (code < VALIDATE_INVALID || code > 3) {
                throw new IllegalArgumentException("Unknown traversal validation code");
            }
            if (limit < 0.0 || distance < 0.0 || distance > limit + 1.0e-9) {
                throw new IllegalArgumentException(
                    "Traversal upward distance lies outside its requested limit"
                );
            }
            if (code == VALIDATE_INVALID && distance != 0.0) {
                throw new IllegalArgumentException(
                    "Invalid traversal samples must not publish clearance"
                );
            }
        }
    }

    public int sampleCount() {
        return validationCodes.length;
    }

    @Override
    public double[] positions() {
        return positions.clone();
    }

    @Override
    public double[] upwardLimits() {
        return upwardLimits.clone();
    }

    @Override
    public double[] actorBounds() {
        return actorBounds.clone();
    }

    @Override
    public byte[] validationCodes() {
        return validationCodes.clone();
    }

    @Override
    public double[] upwardCollisionDistances() {
        return upwardCollisionDistances.clone();
    }

    private static void requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
    }

    private static void requireFinite(double[] values, String name) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(name + " must be finite");
            }
        }
    }
}
