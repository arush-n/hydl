package com.hytalerlbridge.policy.testing.placement;

/** Pure deterministic square-ring search shared by server placement tests. */
public final class FixturePlacementPlanner {

    public enum Reason {
        READY,
        SEARCH_EXHAUSTED,
    }

    /** The server adapter owns the typed detail for an individual rejection. */
    public record Candidate(
        boolean accepted,
        String detail,
        double agentX,
        double agentY,
        double agentZ,
        double targetX,
        double targetY,
        double targetZ
    ) {
        public Candidate {
            if (detail == null) {
                throw new IllegalArgumentException(
                    "placement candidate detail is required");
            }
            if (accepted && (!finite(agentX, agentY, agentZ)
                    || !finite(targetX, targetY, targetZ))) {
                throw new IllegalArgumentException(
                    "accepted placement coordinates must be finite");
            }
        }

        public static Candidate reject(String detail) {
            return new Candidate(
                false, detail,
                Double.NaN, Double.NaN, Double.NaN,
                Double.NaN, Double.NaN, Double.NaN
            );
        }

        private static boolean finite(double x, double y, double z) {
            return Double.isFinite(x)
                && Double.isFinite(y)
                && Double.isFinite(z);
        }
    }

    public record Result(
        Reason reason,
        Candidate candidate,
        int offsetX,
        int offsetZ,
        int attempts,
        String lastRejection
    ) {
        public Result {
            if (reason == null || candidate == null || attempts < 1
                || lastRejection == null) {
                throw new IllegalArgumentException(
                    "invalid placement search result");
            }
            if ((reason == Reason.READY) != candidate.accepted()) {
                throw new IllegalArgumentException(
                    "placement result and candidate disagree");
            }
        }
    }

    @FunctionalInterface
    public interface Oracle {
        Candidate evaluate(int offsetX, int offsetZ);
    }

    private FixturePlacementPlanner() {}

    /**
     * Search center first, then each square ring in stable north/south/west/east
     * order. This mirrors Hytale's respawn-search shape without fixing a world
     * coordinate or block identity in source.
     */
    public static Result search(int radius, Oracle oracle) {
        if (radius < 0 || radius > 64 || oracle == null) {
            throw new IllegalArgumentException(
                "placement radius/oracle is invalid");
        }
        Counter counter = new Counter();
        Result center = evaluate(0, 0, oracle, counter);
        if (center != null) return center;
        for (int distance = 1; distance <= radius; distance++) {
            for (int offset = -distance; offset <= distance; offset++) {
                Result north = evaluate(
                    offset, -distance, oracle, counter);
                if (north != null) return north;
                Result south = evaluate(
                    offset, distance, oracle, counter);
                if (south != null) return south;
            }
            for (int offset = -distance + 1;
                    offset < distance; offset++) {
                Result west = evaluate(
                    -distance, offset, oracle, counter);
                if (west != null) return west;
                Result east = evaluate(
                    distance, offset, oracle, counter);
                if (east != null) return east;
            }
        }
        return new Result(
            Reason.SEARCH_EXHAUSTED,
            FixturePlacementPlanner.Candidate.reject(counter.lastRejection),
            0,
            0,
            counter.attempts,
            counter.lastRejection
        );
    }

    private static Result evaluate(
        int x,
        int z,
        Oracle oracle,
        Counter counter
    ) {
        Candidate candidate = oracle.evaluate(x, z);
        if (candidate == null) {
            candidate = Candidate.reject("oracle_returned_null");
        }
        counter.attempts++;
        counter.lastRejection = candidate.detail();
        return candidate.accepted()
            ? new Result(
                Reason.READY,
                candidate,
                x,
                z,
                counter.attempts,
                ""
            )
            : null;
    }

    private static final class Counter {
        private int attempts;
        private String lastRejection = "not_evaluated";
    }
}
