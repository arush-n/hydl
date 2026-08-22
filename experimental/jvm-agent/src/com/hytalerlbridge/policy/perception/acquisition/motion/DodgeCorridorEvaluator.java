package com.hytalerlbridge.policy.perception.acquisition.motion;

/** Pure prospective Dodge motion plus a caller-supplied swept-AABB oracle. */
public final class DodgeCorridorEvaluator {

    public static final int DIRECTION_COUNT = 4;
    private static final float EPSILON = 1.0e-6f;

    /** Policy direction order: forward, back, left, right. */
    public enum Direction {
        FORWARD,
        BACK,
        LEFT,
        RIGHT,
    }

    public enum SweepResult {
        CLEAR,
        BLOCKED,
        UNAVAILABLE,
    }

    /** Why a prospective corridor evaluation did or did not reach geometry. */
    public enum EvaluationReason {
        READY,
        INVALID_INPUT,
        MOTION_INCOMPLETE,
        DEGENERATE_PATH,
        SWEEP_EXCEPTION,
    }

    public record Displacement(double x, double y, double z) {
        public Displacement {
            if (!Double.isFinite(x) || !Double.isFinite(y)
                || !Double.isFinite(z)) {
                throw new IllegalArgumentException(
                    "Dodge displacement must be finite");
            }
        }
    }

    public record MotionPath(
        Displacement displacement,
        int ticks,
        boolean complete
    ) {
        public MotionPath {
            if (displacement == null || ticks < 0) {
                throw new IllegalArgumentException("invalid Dodge motion path");
            }
        }

        public double distanceSquared() {
            return displacement.x() * displacement.x()
                + displacement.y() * displacement.y()
                + displacement.z() * displacement.z();
        }
    }

    /** One direction's exact displacement and geometry result. */
    public record DirectionEvaluation(
        Direction direction,
        Displacement displacement,
        SweepResult sweep
    ) {
        public DirectionEvaluation {
            if (direction == null || displacement == null || sweep == null) {
                throw new IllegalArgumentException(
                    "invalid Dodge direction evaluation");
            }
        }
    }

    /** Typed result used by live diagnostics without changing policy semantics. */
    public record Evaluation(
        boolean[] clear,
        EvaluationReason reason,
        MotionPath path,
        DirectionEvaluation[] directions
    ) {
        public Evaluation {
            if (clear == null || clear.length != DIRECTION_COUNT
                || reason == null || path == null || directions == null
                || directions.length != DIRECTION_COUNT) {
                throw new IllegalArgumentException(
                    "invalid Dodge corridor evaluation");
            }
            clear = clear.clone();
            directions = directions.clone();
        }

        @Override public boolean[] clear() {
            return clear.clone();
        }

        @Override public DirectionEvaluation[] directions() {
            return directions.clone();
        }
    }

    @FunctionalInterface
    public interface SweptAabb {
        SweepResult query(Direction direction, Displacement displacement);
    }

    private DodgeCorridorEvaluator() {}

    /** Integrate exactly the open-space vector recurrence used by JAX. */
    public static MotionPath integrate(
        double yawDegrees,
        long startingTick,
        DodgeMotionParameters parameters,
        boolean grounded
    ) {
        if (parameters == null || !Double.isFinite(yawDegrees)) {
            return unavailablePath();
        }
        boolean nativeProfile = parameters.executionProfile()
            == DodgeMotionParameters.NATIVE_NPC_NULL_CONFIG;
        double radians = Math.toRadians(yawDegrees);
        float scale = parameters.authoredForce()
            * parameters.knockbackScale();
        if (nativeProfile) {
            scale *= parameters.nativeHorizontalFactor()
                * parameters.nativeMovementVelocityResistance();
        }
        float velocityX = (float) (-Math.sin(radians) * scale);
        float velocityZ = (float) (-Math.cos(radians) * scale);
        if (!Float.isFinite(velocityX) || !Float.isFinite(velocityZ)
            || velocityX == 0.0f && velocityZ == 0.0f) {
            return unavailablePath();
        }

        float displacementX = 0.0f;
        float displacementZ = 0.0f;
        int tick = 0;
        while (tick < parameters.clearanceTickCapacity()
            && moving(velocityX, velocityZ, nativeProfile,
                parameters.velocityRemovalSquared())) {
            float deltaSeconds = motionDelta(
                startingTick + tick + 1L,
                parameters
            );
            displacementX += velocityX * deltaSeconds;
            displacementZ += velocityZ * deltaSeconds;
            Velocity damped = damp(
                velocityX,
                velocityZ,
                grounded,
                nativeProfile,
                parameters
            );
            velocityX = damped.x();
            velocityZ = damped.z();
            tick++;
        }
        boolean complete = !moving(
            velocityX,
            velocityZ,
            nativeProfile,
            parameters.velocityRemovalSquared()
        );
        return new MotionPath(
            new Displacement(displacementX, 0.0, displacementZ),
            tick,
            complete
        );
    }

    /**
     * Evaluate all four geometrical directions. Authored direction filtering
     * remains in ActionMaskProjection; this producer owns geometry only.
     */
    public static boolean[] evaluate(
        double yawDegrees,
        long startingTick,
        boolean grounded,
        DodgeMotionParameters parameters,
        SweptAabb sweep
    ) {
        return evaluateDetailed(
            yawDegrees, startingTick, grounded, parameters, sweep).clear();
    }

    /** Evaluate with typed failure and per-direction evidence for live audits. */
    public static Evaluation evaluateDetailed(
        double yawDegrees,
        long startingTick,
        boolean grounded,
        DodgeMotionParameters parameters,
        SweptAabb sweep
    ) {
        boolean[] result = new boolean[DIRECTION_COUNT];
        DirectionEvaluation[] evidence = emptyDirections();
        if (!Double.isFinite(yawDegrees) || parameters == null || sweep == null) {
            return new Evaluation(
                result,
                EvaluationReason.INVALID_INPUT,
                unavailablePath(),
                evidence
            );
        }
        MotionPath path = integrate(
            yawDegrees, startingTick, parameters, grounded);
        if (!path.complete()) {
            return new Evaluation(
                result, EvaluationReason.MOTION_INCOMPLETE, path, evidence);
        }
        if (path.distanceSquared() <= EPSILON * EPSILON) {
            return new Evaluation(
                result, EvaluationReason.DEGENERATE_PATH, path, evidence);
        }

        double forwardX = path.displacement().x();
        double forwardY = path.displacement().y();
        double forwardZ = path.displacement().z();
        Displacement[] displacement = {
            new Displacement(forwardX, forwardY, forwardZ),
            new Displacement(-forwardX, forwardY, -forwardZ),
            new Displacement(forwardZ, forwardY, -forwardX),
            new Displacement(-forwardZ, forwardY, forwardX),
        };
        Direction[] directions = Direction.values();
        for (int index = 0; index < result.length; index++) {
            SweepResult value;
            try {
                value = sweep.query(directions[index], displacement[index]);
            } catch (RuntimeException | LinkageError unavailable) {
                return new Evaluation(
                    new boolean[DIRECTION_COUNT],
                    EvaluationReason.SWEEP_EXCEPTION,
                    path,
                    evidence
                );
            }
            if (value == null) value = SweepResult.UNAVAILABLE;
            evidence[index] = new DirectionEvaluation(
                directions[index], displacement[index], value);
            result[index] = value == SweepResult.CLEAR;
        }
        return new Evaluation(
            result, EvaluationReason.READY, path, evidence);
    }

    private static Velocity damp(
        float velocityX,
        float velocityZ,
        boolean grounded,
        boolean nativeProfile,
        DodgeMotionParameters parameters
    ) {
        float speedSquared = velocityX * velocityX + velocityZ * velocityZ;
        float speed = (float) Math.sqrt(speedSquared);
        float resistance;
        float referenceTicks;
        if (nativeProfile) {
            if (grounded) {
                resistance = parameters.nativeGroundDragBase();
            } else {
                float span = parameters.nativeAirDragMaxSpeed()
                    - parameters.nativeAirDragMinSpeed();
                float blend = clip((speed
                    - parameters.nativeAirDragMinSpeed()) / span);
                resistance = parameters.nativeAirDragMin()
                    * (1.0f - blend)
                    + parameters.nativeAirDragMax() * blend;
            }
            referenceTicks = parameters.nativeReferenceTicksPerSecond();
        } else {
            float minimum = grounded
                ? parameters.configuredGroundResistance()
                : parameters.configuredAirResistance();
            float maximum = grounded
                ? parameters.configuredGroundResistanceMax()
                : parameters.configuredAirResistanceMax();
            float ratio = speed / parameters.configuredResistanceThreshold();
            float blend = parameters.configuredResistanceStyle()
                    == DodgeMotionParameters.EXPONENTIAL_RESISTANCE
                ? clip(ratio * ratio)
                : clip(ratio);
            resistance = minimum * blend + maximum * (1.0f - blend);
            referenceTicks = 60.0f;
        }
        float scale = (float) Math.pow(
            resistance,
            referenceTicks / parameters.serverTicksPerSecond()
        );
        float dampedX = velocityX * scale;
        float dampedZ = velocityZ * scale;
        if (nativeProfile) {
            if (Math.abs(dampedX) <= parameters.nativePerAxisDeadzone()) {
                dampedX = 0.0f;
            }
            if (Math.abs(dampedZ) <= parameters.nativePerAxisDeadzone()) {
                dampedZ = 0.0f;
            }
        }
        if (!nativeProfile
            && dampedX * dampedX + dampedZ * dampedZ
                < parameters.velocityRemovalSquared()) {
            dampedX = 0.0f;
            dampedZ = 0.0f;
        }
        if (!Float.isFinite(dampedX) || !Float.isFinite(dampedZ)) {
            return new Velocity(0.0f, 0.0f);
        }
        return new Velocity(dampedX, dampedZ);
    }

    private static boolean moving(
        float velocityX,
        float velocityZ,
        boolean nativeProfile,
        float removalSquared
    ) {
        return nativeProfile
            ? velocityX != 0.0f || velocityZ != 0.0f
            : velocityX * velocityX + velocityZ * velocityZ >= removalSquared;
    }

    private static MotionPath unavailablePath() {
        return new MotionPath(new Displacement(0.0, 0.0, 0.0), 0, false);
    }

    private static DirectionEvaluation[] emptyDirections() {
        DirectionEvaluation[] result = new DirectionEvaluation[DIRECTION_COUNT];
        Direction[] values = Direction.values();
        for (int index = 0; index < result.length; index++) {
            result[index] = new DirectionEvaluation(
                values[index],
                new Displacement(0.0, 0.0, 0.0),
                SweepResult.UNAVAILABLE
            );
        }
        return result;
    }

    private record Velocity(float x, float z) {}

    /** Match JAX `_motion_delta(tick_count + index + 1, profile, params)`. */
    private static float motionDelta(
        long prospectiveTick,
        DodgeMotionParameters parameters
    ) {
        int profile = parameters.motionTimingProfile();
        boolean odd = Math.floorMod(prospectiveTick, 2L) == 1L;
        boolean loadedTick = profile == 1 ? odd : !odd;
        return profile != 0 && loadedTick
            ? parameters.loadedDeltaSeconds()
            : parameters.nominalDeltaSeconds();
    }

    private static float clip(float value) {
        return Math.max(0.0f, Math.min(1.0f, value));
    }
}
