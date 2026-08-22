package com.hytalerlbridge.worldgen;

/** Bounded native MotionControllerWalk evidence for traversal graph edges. */
public record NativeTraversalEdgeProbe(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    double[] startPositions,
    double[] targetPositions,
    double[] horizontalArrivalTolerances,
    double[] verticalArrivalTolerances,
    double[] actorBounds,
    double[] directionComponentSelector,
    double maximumClimbHeight,
    double maximumDropHeight,
    byte[] reachable,
    byte[] edgeBlocked,
    double[] finalPositions,
    double[] travelledDistances
) {
    public static final String SCHEMA =
        "hytalerl_native_traversal_edge_probe_v1";
    public static final int VERSION = 1;
    public static final int MAX_SAMPLES = 4096;

    public NativeTraversalEdgeProbe {
        requireText(serverVersion, "serverVersion");
        requireText(worldName, "worldName");
        requireText(worldgenProvider, "worldgenProvider");
        requireText(worldgenVersion, "worldgenVersion");
        startPositions = clone(startPositions);
        targetPositions = clone(targetPositions);
        horizontalArrivalTolerances = clone(horizontalArrivalTolerances);
        verticalArrivalTolerances = clone(verticalArrivalTolerances);
        actorBounds = clone(actorBounds);
        directionComponentSelector = clone(directionComponentSelector);
        reachable = clone(reachable);
        edgeBlocked = clone(edgeBlocked);
        finalPositions = clone(finalPositions);
        travelledDistances = clone(travelledDistances);
        int count = horizontalArrivalTolerances.length;
        if (
            count < 1
                || count > MAX_SAMPLES
                || startPositions.length != count * 3
                || targetPositions.length != count * 3
                || verticalArrivalTolerances.length != count
                || actorBounds.length != 6
                || directionComponentSelector.length != 3
                || reachable.length != count
                || edgeBlocked.length != count
                || finalPositions.length != count * 3
                || travelledDistances.length != count
        ) {
            throw new IllegalArgumentException(
                "Traversal edge probe arrays disagree or exceed capacity"
            );
        }
        requireFinite(startPositions, "startPositions");
        requireFinite(targetPositions, "targetPositions");
        requireFinite(
            horizontalArrivalTolerances,
            "horizontalArrivalTolerances"
        );
        requireFinite(verticalArrivalTolerances, "verticalArrivalTolerances");
        requireFinite(actorBounds, "actorBounds");
        requireFinite(directionComponentSelector, "directionComponentSelector");
        requireFinite(finalPositions, "finalPositions");
        requireFinite(travelledDistances, "travelledDistances");
        if (
            actorBounds[0] >= actorBounds[3]
                || actorBounds[1] >= actorBounds[4]
                || actorBounds[2] >= actorBounds[5]
        ) {
            throw new IllegalArgumentException(
                "actorBounds must be a positive AABB"
            );
        }
        if (
            !Double.isFinite(maximumClimbHeight)
                || maximumClimbHeight < 0.0
                || !Double.isFinite(maximumDropHeight)
                || maximumDropHeight < 0.0
        ) {
            throw new IllegalArgumentException(
                "Traversal climb/drop limits must be finite and non-negative"
            );
        }
        for (int index = 0; index < count; index++) {
            if (
                horizontalArrivalTolerances[index] < 0.0
                    || verticalArrivalTolerances[index] < 0.0
                    || travelledDistances[index] < 0.0
                    || reachable[index] < 0
                    || reachable[index] > 1
                    || edgeBlocked[index] < 0
                    || edgeBlocked[index] > 1
            ) {
                throw new IllegalArgumentException(
                    "Traversal edge values are outside their contract"
                );
            }
        }
    }

    public int sampleCount() {
        return horizontalArrivalTolerances.length;
    }

    @Override
    public double[] startPositions() {
        return startPositions.clone();
    }

    @Override
    public double[] targetPositions() {
        return targetPositions.clone();
    }

    @Override
    public double[] horizontalArrivalTolerances() {
        return horizontalArrivalTolerances.clone();
    }

    @Override
    public double[] verticalArrivalTolerances() {
        return verticalArrivalTolerances.clone();
    }

    @Override
    public double[] actorBounds() {
        return actorBounds.clone();
    }

    @Override
    public double[] directionComponentSelector() {
        return directionComponentSelector.clone();
    }

    @Override
    public byte[] reachable() {
        return reachable.clone();
    }

    @Override
    public byte[] edgeBlocked() {
        return edgeBlocked.clone();
    }

    @Override
    public double[] finalPositions() {
        return finalPositions.clone();
    }

    @Override
    public double[] travelledDistances() {
        return travelledDistances.clone();
    }

    private static double[] clone(double[] values) {
        return values == null ? new double[0] : values.clone();
    }

    private static byte[] clone(byte[] values) {
        return values == null ? new byte[0] : values.clone();
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
