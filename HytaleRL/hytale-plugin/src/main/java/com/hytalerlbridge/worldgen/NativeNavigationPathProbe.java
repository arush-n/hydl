package com.hytalerlbridge.worldgen;

/**
 * Bounded native AStarWithTarget evidence for exact Region navigation paths.
 *
 * <p>Paths use the full predecessor chain rather than the runtime's
 * corner-only optimization so a consumer can compare every node. Arrays are
 * fixed to {@code sampleCount * maximumPathLength}; unused rows are zero.</p>
 */
public record NativeNavigationPathProbe(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    double[] startPositions,
    double[] targetPositions,
    double[] actorBounds,
    double[] directionComponentSelector,
    int maximumPathLength,
    int openNodesLimit,
    int totalNodesLimit,
    int nodesPerIteration,
    byte[] progress,
    int[] iterations,
    int[] visitedCounts,
    int[] openCounts,
    int[] pathNodeCounts,
    double[] pathPositions,
    float[] pathTravelCosts
) {
    public static final String SCHEMA =
        "hytalerl_native_navigation_path_probe_v1";
    public static final int VERSION = 1;
    public static final int MAX_SAMPLES = 32;
    public static final int MAXIMUM_PATH_LENGTH = 200;
    public static final int MAX_OPEN_NODES = 200;
    public static final int MAX_TOTAL_NODES = 900;
    public static final int MAX_NODES_PER_ITERATION = 900;

    public NativeNavigationPathProbe {
        requireText(serverVersion, "serverVersion");
        requireText(worldName, "worldName");
        requireText(worldgenProvider, "worldgenProvider");
        requireText(worldgenVersion, "worldgenVersion");
        startPositions = clone(startPositions);
        targetPositions = clone(targetPositions);
        actorBounds = clone(actorBounds);
        directionComponentSelector = clone(directionComponentSelector);
        progress = clone(progress);
        iterations = clone(iterations);
        visitedCounts = clone(visitedCounts);
        openCounts = clone(openCounts);
        pathNodeCounts = clone(pathNodeCounts);
        pathPositions = clone(pathPositions);
        pathTravelCosts = clone(pathTravelCosts);
        int count = progress.length;
        int pathCapacity = Math.multiplyExact(count, maximumPathLength);
        if (
            count < 1
                || count > MAX_SAMPLES
                || startPositions.length != count * 3
                || targetPositions.length != count * 3
                || actorBounds.length != 6
                || directionComponentSelector.length != 3
                || maximumPathLength < 1
                || maximumPathLength > MAXIMUM_PATH_LENGTH
                || openNodesLimit < 1
                || openNodesLimit > MAX_OPEN_NODES
                || totalNodesLimit < 1
                || totalNodesLimit > MAX_TOTAL_NODES
                || nodesPerIteration < 1
                || nodesPerIteration > MAX_NODES_PER_ITERATION
                || iterations.length != count
                || visitedCounts.length != count
                || openCounts.length != count
                || pathNodeCounts.length != count
                || pathPositions.length != pathCapacity * 3
                || pathTravelCosts.length != pathCapacity
        ) {
            throw new IllegalArgumentException(
                "Navigation path probe arrays or capacities disagree"
            );
        }
        requireFinite(startPositions, "startPositions");
        requireFinite(targetPositions, "targetPositions");
        requireFinite(actorBounds, "actorBounds");
        requireFinite(directionComponentSelector, "directionComponentSelector");
        requireFinite(pathPositions, "pathPositions");
        requireFinite(pathTravelCosts, "pathTravelCosts");
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
                progress[index] < 0
                    || progress[index] > 6
                    || iterations[index] < 0
                    || visitedCounts[index] < 0
                    || visitedCounts[index] > MAX_TOTAL_NODES + 8
                    || openCounts[index] < 0
                    || openCounts[index] > MAX_OPEN_NODES + 8
                    || pathNodeCounts[index] < 0
                    || pathNodeCounts[index] > maximumPathLength
            ) {
                throw new IllegalArgumentException(
                    "Navigation path probe values exceed their contract"
                );
            }
        }
    }

    public int sampleCount() {
        return progress.length;
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
    public double[] actorBounds() {
        return actorBounds.clone();
    }

    @Override
    public double[] directionComponentSelector() {
        return directionComponentSelector.clone();
    }

    @Override
    public byte[] progress() {
        return progress.clone();
    }

    @Override
    public int[] iterations() {
        return iterations.clone();
    }

    @Override
    public int[] visitedCounts() {
        return visitedCounts.clone();
    }

    @Override
    public int[] openCounts() {
        return openCounts.clone();
    }

    @Override
    public int[] pathNodeCounts() {
        return pathNodeCounts.clone();
    }

    @Override
    public double[] pathPositions() {
        return pathPositions.clone();
    }

    @Override
    public float[] pathTravelCosts() {
        return pathTravelCosts.clone();
    }

    private static double[] clone(double[] values) {
        return values == null ? new double[0] : values.clone();
    }

    private static float[] clone(float[] values) {
        return values == null ? new float[0] : values.clone();
    }

    private static int[] clone(int[] values) {
        return values == null ? new int[0] : values.clone();
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

    private static void requireFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must be finite");
            }
        }
    }
}
