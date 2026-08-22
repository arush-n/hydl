package com.hytalerlbridge.policy.perception.projection;

/** Pure learner projection for World-owned, actor-legal geometry tokens. */
public final class GeometryProjection {

    public static final int TOKEN_CAPACITY = 44;
    public static final int EDGE_CAPACITY = 12;
    public static final int DETAIL_BOX_CAPACITY = 9;
    public static final int BASE_FEATURE_SIZE = 8;
    public static final int EDGE_FEATURE_SIZE = 5;
    public static final int BOX_FEATURE_SIZE = 7;
    public static final int FEATURE_SIZE = BASE_FEATURE_SIZE
        + EDGE_CAPACITY * EDGE_FEATURE_SIZE
        + DETAIL_BOX_CAPACITY * BOX_FEATURE_SIZE;
    public static final int VALUE_COUNT = TOKEN_CAPACITY * FEATURE_SIZE;
    public static final float MAXIMUM_DISTANCE = 4.0f;
    public static final int MAXIMUM_TOKEN_KIND = 2;
    public static final int MAXIMUM_PROVENANCE = 3;

    /** Raw public World token row before policy normalization. */
    public record Input(
        boolean sourceAvailable,
        boolean[] tokenMask,
        int[] tokenKind,
        int[] tokenProvenance,
        float[] relativePosition,
        float[] clearance,
        int[] semanticFlags,
        boolean[] dynamicBlocked,
        boolean[] edgeMask,
        int[] edgeDestination,
        float[] edgeCost,
        int[] edgeKind,
        int[] edgeFlags,
        boolean[] collisionBoxMask,
        float[] collisionBoxesRelative,
        boolean actorValid
    ) {
        public Input {
            require(tokenMask, TOKEN_CAPACITY, "geometry token mask");
            require(tokenKind, TOKEN_CAPACITY, "geometry token kind");
            require(tokenProvenance, TOKEN_CAPACITY, "geometry provenance");
            require(relativePosition, TOKEN_CAPACITY * 3, "relative position");
            require(clearance, TOKEN_CAPACITY, "clearance");
            require(semanticFlags, TOKEN_CAPACITY, "semantic flags");
            require(dynamicBlocked, TOKEN_CAPACITY, "dynamic blocked");
            int edges = axis(edgeMask.length, EDGE_CAPACITY, "edge");
            require(edgeDestination, TOKEN_CAPACITY * edges, "edge destination");
            require(edgeCost, TOKEN_CAPACITY * edges, "edge cost");
            require(edgeKind, TOKEN_CAPACITY * edges, "edge kind");
            require(edgeFlags, TOKEN_CAPACITY * edges, "edge flags");
            int boxes = axis(
                collisionBoxMask.length, DETAIL_BOX_CAPACITY, "collision box");
            require(
                collisionBoxesRelative,
                TOKEN_CAPACITY * boxes * 6,
                "collision boxes");
            tokenMask = tokenMask.clone();
            tokenKind = tokenKind.clone();
            tokenProvenance = tokenProvenance.clone();
            relativePosition = relativePosition.clone();
            clearance = clearance.clone();
            semanticFlags = semanticFlags.clone();
            dynamicBlocked = dynamicBlocked.clone();
            edgeMask = edgeMask.clone();
            edgeDestination = edgeDestination.clone();
            edgeCost = edgeCost.clone();
            edgeKind = edgeKind.clone();
            edgeFlags = edgeFlags.clone();
            collisionBoxMask = collisionBoxMask.clone();
            collisionBoxesRelative = collisionBoxesRelative.clone();
        }

        public int sourceEdgeCapacity() {
            return edgeMask.length / TOKEN_CAPACITY;
        }

        public int sourceCollisionBoxCapacity() {
            return collisionBoxMask.length / TOKEN_CAPACITY;
        }

        @Override public boolean[] tokenMask() { return tokenMask.clone(); }
        @Override public int[] tokenKind() { return tokenKind.clone(); }
        @Override public int[] tokenProvenance() {
            return tokenProvenance.clone();
        }
        @Override public float[] relativePosition() {
            return relativePosition.clone();
        }
        @Override public float[] clearance() { return clearance.clone(); }
        @Override public int[] semanticFlags() { return semanticFlags.clone(); }
        @Override public boolean[] dynamicBlocked() {
            return dynamicBlocked.clone();
        }
        @Override public boolean[] edgeMask() { return edgeMask.clone(); }
        @Override public int[] edgeDestination() {
            return edgeDestination.clone();
        }
        @Override public float[] edgeCost() { return edgeCost.clone(); }
        @Override public int[] edgeKind() { return edgeKind.clone(); }
        @Override public int[] edgeFlags() { return edgeFlags.clone(); }
        @Override public boolean[] collisionBoxMask() {
            return collisionBoxMask.clone();
        }
        @Override public float[] collisionBoxesRelative() {
            return collisionBoxesRelative.clone();
        }

        public static Input unavailable(boolean actorValid) {
            return new Input(
                false,
                new boolean[TOKEN_CAPACITY],
                new int[TOKEN_CAPACITY],
                new int[TOKEN_CAPACITY],
                new float[TOKEN_CAPACITY * 3],
                new float[TOKEN_CAPACITY],
                new int[TOKEN_CAPACITY],
                new boolean[TOKEN_CAPACITY],
                new boolean[0],
                new int[0],
                new float[0],
                new int[0],
                new int[0],
                new boolean[0],
                new float[0],
                actorValid
            );
        }
    }

    public record Result(boolean available, float[] values, boolean[] tokenMask) {
        public Result {
            require(values, VALUE_COUNT, "geometry values");
            require(tokenMask, TOKEN_CAPACITY, "geometry token mask");
            values = values.clone();
            tokenMask = tokenMask.clone();
        }

        @Override public float[] values() { return values.clone(); }
        @Override public boolean[] tokenMask() { return tokenMask.clone(); }
    }

    private GeometryProjection() {
    }

    /**
     * Match {@code encode_world_geometry_policy_tokens} and its outer actor
     * validity gate. Candidate selection, LOS, and traversal stay World-owned.
     */
    public static Result project(Input input) {
        // Input is immutable-by-copy at its public boundary. Direct nested
        // storage access avoids cloning the full edge/box payload every tick.
        boolean[] rawTokenMask = input.tokenMask;
        int[] tokenKind = input.tokenKind;
        int[] provenance = input.tokenProvenance;
        float[] position = input.relativePosition;
        float[] clearance = input.clearance;
        int[] semantic = input.semanticFlags;
        boolean[] dynamic = input.dynamicBlocked;
        boolean[] rawEdgeMask = input.edgeMask;
        int[] destination = input.edgeDestination;
        float[] edgeCost = input.edgeCost;
        int[] edgeKind = input.edgeKind;
        int[] edgeFlags = input.edgeFlags;
        boolean[] rawBoxMask = input.collisionBoxMask;
        float[] boxes = input.collisionBoxesRelative;
        int sourceEdges = input.sourceEdgeCapacity();
        int sourceBoxes = input.sourceCollisionBoxCapacity();

        boolean[] activeToken = new boolean[TOKEN_CAPACITY];
        for (int token = 0; token < TOKEN_CAPACITY; token++) {
            activeToken[token] = input.sourceAvailable() && rawTokenMask[token];
        }

        boolean rowValid = true;
        for (int token = 0; token < TOKEN_CAPACITY; token++) {
            if (!activeToken[token]) {
                continue;
            }
            int xyz = token * 3;
            rowValid &= Float.isFinite(position[xyz])
                && Float.isFinite(position[xyz + 1])
                && Float.isFinite(position[xyz + 2])
                && Float.isFinite(clearance[token])
                && tokenKind[token] <= MAXIMUM_TOKEN_KIND
                && provenance[token] <= MAXIMUM_PROVENANCE;
            for (int edge = 0; edge < sourceEdges; edge++) {
                int index = token * sourceEdges + edge;
                if (!rawEdgeMask[index]) {
                    continue;
                }
                int target = destination[index];
                rowValid &= target >= 0
                    && target < TOKEN_CAPACITY
                    && activeToken[Math.max(0, Math.min(TOKEN_CAPACITY - 1, target))]
                    && Float.isFinite(edgeCost[index]);
            }
            for (int box = 0; box < sourceBoxes; box++) {
                int maskIndex = token * sourceBoxes + box;
                if (!rawBoxMask[maskIndex]) {
                    continue;
                }
                int base = maskIndex * 6;
                for (int component = 0; component < 6; component++) {
                    rowValid &= Float.isFinite(boxes[base + component]);
                }
            }
        }

        boolean available = input.sourceAvailable() && rowValid && input.actorValid();
        boolean[] tokenMask = new boolean[TOKEN_CAPACITY];
        float[] values = new float[VALUE_COUNT];
        if (!available) {
            return new Result(false, values, tokenMask);
        }

        for (int token = 0; token < TOKEN_CAPACITY; token++) {
            boolean active = rawTokenMask[token];
            tokenMask[token] = active;
            if (!active) {
                continue;
            }
            int inputXyz = token * 3;
            int out = token * FEATURE_SIZE;
            values[out] = clamp(tokenKind[token] / (float) MAXIMUM_TOKEN_KIND, -1, 1);
            values[out + 1] = clamp(
                provenance[token] / (float) MAXIMUM_PROVENANCE, -1, 1);
            values[out + 2] = clamp(position[inputXyz] / MAXIMUM_DISTANCE, -1, 1);
            values[out + 3] = clamp(
                position[inputXyz + 1] / MAXIMUM_DISTANCE, -1, 1);
            values[out + 4] = clamp(
                position[inputXyz + 2] / MAXIMUM_DISTANCE, -1, 1);
            values[out + 5] = clamp(
                clearance[token] / MAXIMUM_DISTANCE, -1, 1);
            values[out + 6] = clamp(semantic[token] / 65535.0f, -1, 1);
            values[out + 7] = dynamic[token] ? 1.0f : 0.0f;

            int edgeOut = out + BASE_FEATURE_SIZE;
            for (int edge = 0; edge < EDGE_CAPACITY; edge++) {
                boolean supplied = edge < sourceEdges;
                int source = supplied ? token * sourceEdges + edge : -1;
                boolean edgeActive = supplied && rawEdgeMask[source];
                int feature = edgeOut + edge * EDGE_FEATURE_SIZE;
                values[feature] = edgeActive ? 1.0f : 0.0f;
                values[feature + 1] = edgeActive
                    ? (destination[source] + 1.0f) / (TOKEN_CAPACITY + 1.0f)
                    : 0.0f;
                values[feature + 2] = supplied
                    ? clamp(edgeCost[source] / MAXIMUM_DISTANCE, 0, 1) : 0.0f;
                values[feature + 3] = supplied
                    ? edgeKind[source] / 255.0f : 0.0f;
                values[feature + 4] = supplied
                    ? edgeFlags[source] / 255.0f : 0.0f;
            }

            int boxOut = edgeOut + EDGE_CAPACITY * EDGE_FEATURE_SIZE;
            for (int box = 0; box < DETAIL_BOX_CAPACITY; box++) {
                boolean supplied = box < sourceBoxes;
                int sourceMask = supplied ? token * sourceBoxes + box : -1;
                int feature = boxOut + box * BOX_FEATURE_SIZE;
                values[feature] = supplied && rawBoxMask[sourceMask]
                    ? 1.0f : 0.0f;
                for (int component = 0; component < 6; component++) {
                    values[feature + 1 + component] = supplied
                        ? clamp(
                            boxes[(sourceMask * 6) + component]
                                / MAXIMUM_DISTANCE,
                            -1, 1)
                        : 0.0f;
                }
            }
        }
        return new Result(true, values, tokenMask);
    }

    private static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static int axis(int length, int maximum, String name) {
        if (length % TOKEN_CAPACITY != 0) {
            throw new IllegalArgumentException(
                name + " axis does not divide the token capacity: " + length);
        }
        int capacity = length / TOKEN_CAPACITY;
        if (capacity > maximum) {
            throw new IllegalArgumentException(
                name + " capacity " + capacity + " exceeds " + maximum);
        }
        return capacity;
    }

    private static void require(boolean[] values, int expected, String name) {
        if (values == null || values.length != expected) {
            throw width(name, values == null ? -1 : values.length, expected);
        }
    }

    private static void require(int[] values, int expected, String name) {
        if (values == null || values.length != expected) {
            throw width(name, values == null ? -1 : values.length, expected);
        }
    }

    private static void require(float[] values, int expected, String name) {
        if (values == null || values.length != expected) {
            throw width(name, values == null ? -1 : values.length, expected);
        }
    }

    private static IllegalArgumentException width(
        String name, int actual, int expected
    ) {
        return new IllegalArgumentException(
            name + " width drift: "
                + (actual < 0 ? "null" : Integer.toString(actual))
                + " != " + expected);
    }
}
