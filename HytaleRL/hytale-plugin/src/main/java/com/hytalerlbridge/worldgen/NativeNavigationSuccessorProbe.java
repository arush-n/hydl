package com.hytalerlbridge.worldgen;

/** Exact per-direction inputs to Hytale 0.5.7 AStarBase successor creation. */
public record NativeNavigationSuccessorProbe(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    double[] startPositions,
    byte[] directionIndices,
    double[] directions,
    double[] actorBounds,
    double[] directionComponentSelector,
    double[] directionDistances,
    double[] travelledDistances,
    byte[] reachedHalfStep,
    byte[] reachedFullStep,
    byte[] validPositions,
    byte[] edgeBlocked,
    double[] halfStepPositions,
    double[] successorPositions
) {
    public static final String SCHEMA =
        "hytalerl_native_navigation_successor_probe_v1";
    public static final int VERSION = 1;
    public static final int MAX_SAMPLES = 4096;

    public NativeNavigationSuccessorProbe {
        requireText(serverVersion, "serverVersion");
        requireText(worldName, "worldName");
        requireText(worldgenProvider, "worldgenProvider");
        requireText(worldgenVersion, "worldgenVersion");
        startPositions = clone(startPositions);
        directionIndices = clone(directionIndices);
        directions = clone(directions);
        actorBounds = clone(actorBounds);
        directionComponentSelector = clone(directionComponentSelector);
        directionDistances = clone(directionDistances);
        travelledDistances = clone(travelledDistances);
        reachedHalfStep = clone(reachedHalfStep);
        reachedFullStep = clone(reachedFullStep);
        validPositions = clone(validPositions);
        edgeBlocked = clone(edgeBlocked);
        halfStepPositions = clone(halfStepPositions);
        successorPositions = clone(successorPositions);
        int count = directionIndices.length;
        if (
            count < 1
                || count > MAX_SAMPLES
                || startPositions.length != count * 3
                || directions.length != count * 3
                || actorBounds.length != 6
                || directionComponentSelector.length != 3
                || directionDistances.length != count
                || travelledDistances.length != count
                || reachedHalfStep.length != count
                || reachedFullStep.length != count
                || validPositions.length != count
                || edgeBlocked.length != count
                || halfStepPositions.length != count * 3
                || successorPositions.length != count * 3
        ) {
            throw new IllegalArgumentException(
                "Navigation successor arrays disagree or exceed capacity"
            );
        }
        requireFinite(startPositions, "startPositions");
        requireFinite(directions, "directions");
        requireFinite(actorBounds, "actorBounds");
        requireFinite(
            directionComponentSelector,
            "directionComponentSelector"
        );
        requireFinite(directionDistances, "directionDistances");
        requireFinite(travelledDistances, "travelledDistances");
        requireFinite(halfStepPositions, "halfStepPositions");
        requireFinite(successorPositions, "successorPositions");
        if (
            actorBounds[0] >= actorBounds[3]
                || actorBounds[1] >= actorBounds[4]
                || actorBounds[2] >= actorBounds[5]
        ) {
            throw new IllegalArgumentException(
                "actorBounds must be a positive AABB"
            );
        }
        for (int index = 0; index < count; index++) {
            if (
                directionIndices[index] < 0
                    || directionIndices[index] >= 26
                    || directionDistances[index] <= 0.0
                    || travelledDistances[index] < 0.0
            ) {
                throw new IllegalArgumentException(
                    "Navigation successor values are outside their contract"
                );
            }
            requireFlag(reachedHalfStep[index]);
            requireFlag(reachedFullStep[index]);
            requireFlag(validPositions[index]);
            requireFlag(edgeBlocked[index]);
            if (
                reachedFullStep[index] > reachedHalfStep[index]
                    || validPositions[index] > reachedHalfStep[index]
            ) {
                throw new IllegalArgumentException(
                    "Navigation successor flags are inconsistent"
                );
            }
        }
    }

    public int sampleCount() {
        return directionIndices.length;
    }

    @Override
    public double[] startPositions() {
        return startPositions.clone();
    }

    @Override
    public byte[] directionIndices() {
        return directionIndices.clone();
    }

    @Override
    public double[] directions() {
        return directions.clone();
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
    public double[] directionDistances() {
        return directionDistances.clone();
    }

    @Override
    public double[] travelledDistances() {
        return travelledDistances.clone();
    }

    @Override
    public byte[] reachedHalfStep() {
        return reachedHalfStep.clone();
    }

    @Override
    public byte[] reachedFullStep() {
        return reachedFullStep.clone();
    }

    @Override
    public byte[] validPositions() {
        return validPositions.clone();
    }

    @Override
    public byte[] edgeBlocked() {
        return edgeBlocked.clone();
    }

    @Override
    public double[] halfStepPositions() {
        return halfStepPositions.clone();
    }

    @Override
    public double[] successorPositions() {
        return successorPositions.clone();
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

    private static void requireFlag(byte value) {
        if (value < 0 || value > 1) {
            throw new IllegalArgumentException(
                "Navigation successor flags must be binary"
            );
        }
    }
}
