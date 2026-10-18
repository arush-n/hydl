package com.hytalerlbridge.policy;

/**
 * Projects the policy's local locomotion command into the vector consumed by
 * Hytale's public NPC steering API.
 *
 * <p>The JAX combat runtime first normalizes the local forward/strafe command,
 * applies the authored player gait multipliers, and then rotates that vector
 * by the commanded body yaw. The policy wire field is still named
 * {@code world_move_direction} for compatibility, but its eight values are a
 * body-relative ring: forward, forward-right, right, back-right, back,
 * back-left, left, and forward-left.
 *
 * <p>The NPC API does not expose the player's client-side gait transition or
 * stamina system. This projection therefore uses the gait as the magnitude of
 * the NPC steering vector. {@code MotionControllerWalk} interprets that
 * magnitude as a relative speed against the role's authored maximum. The
 * direction frame and authored directional multipliers are preserved; the
 * absolute player speed, client sprint transition, and sprint stamina drain
 * remain explicit transfer gaps.
 */
public final class PlayerLocomotionProjection {

    /** Number of entries in the local direction ring. */
    private static final int DIRECTION_COUNT = 8;
    /**
     * Treat mathematically diagonal travel as non-forward-dominant. The JAX
     * source uses float32 trigonometry, while this Java projection receives a
     * float yaw but computes its local ring in double precision; without this
     * epsilon an exact diagonal can change sprint state from rounding noise.
     */
    private static final double SPRINT_FORWARD_DOMINANCE_EPSILON = 1.0e-7;

    /*
     * These are the player values in
     * HytaleRL/hytalegym/hytalegym/jax/combat/types.py. Keep the index order
     * identical to ActionDecoder.GAIT_*: idle, walk, run, sprint, sneak.
     */
    private static final double[] FORWARD_MULTIPLIERS = {
        1.0, 0.3, 1.0, 1.273, 0.55
    };
    private static final double[] BACKWARD_MULTIPLIERS = {
        1.0, 0.3, 0.65, 1.273, 0.4
    };
    private static final double[] STRAFE_MULTIPLIERS = {
        1.0, 0.3, 0.8, 1.273, 0.45
    };

    /** Resulting horizontal steering vector and the gait actually applied. */
    public record Vector(double x, double z, int effectiveGait) {
        /** Whether this command asks the NPC controller to move. */
        public boolean moving() {
            return Math.hypot(x, z) > 1.0e-9;
        }
    }

    /** Project a decoded policy action using the commanded body yaw. */
    public static Vector project(ActionDecoder.Decoded action, double bodyYawRadians) {
        return project(
            action.forward(),
            action.back(),
            action.left(),
            action.right(),
            action.worldMoveDirection(),
            action.gait(),
            bodyYawRadians
        );
    }

    /**
     * Project the action fields without requiring a Hytale classpath.
     *
     * <p>This overload is useful to contract tests because it makes the
     * local-frame and gait rules independently exercisable from the decoded
     * action record.
     */
    public static Vector project(
        boolean forward,
        boolean back,
        boolean left,
        boolean right,
        int worldMoveDirection,
        int requestedGait,
        double bodyYawRadians
    ) {
        double longitudinal;
        double lateral;

        if (worldMoveDirection >= 1 && worldMoveDirection <= DIRECTION_COUNT) {
            double localAngle = (worldMoveDirection - 1) * (Math.PI / 4.0);
            longitudinal = Math.cos(localAngle);
            lateral = Math.sin(localAngle);
        } else {
            longitudinal = (forward ? 1.0 : 0.0) - (back ? 1.0 : 0.0);
            lateral = (right ? 1.0 : 0.0) - (left ? 1.0 : 0.0);
            double length = Math.hypot(longitudinal, lateral);
            if (length <= 1.0e-9) {
                return new Vector(0.0, 0.0, clampGait(requestedGait));
            }
            longitudinal /= length;
            lateral /= length;
        }

        int gait = clampGait(requestedGait);
        boolean sprintRequested = gait == ActionDecoder.GAIT_SPRINT;
        boolean forwardDominant = longitudinal
            > Math.abs(lateral) + SPRINT_FORWARD_DOMINANCE_EPSILON;
        // The player runtime falls back to run when sprint is not forward
        // dominant. Stamina is intentionally not guessed here; the NPC public
        // API gives this sink no authoritative player stamina transition.
        if (sprintRequested && !forwardDominant) {
            gait = ActionDecoder.GAIT_RUN;
        }

        double longitudinalMultiplier = longitudinal >= 0.0
            ? FORWARD_MULTIPLIERS[gait]
            : BACKWARD_MULTIPLIERS[gait];
        double lateralMultiplier = STRAFE_MULTIPLIERS[gait];
        longitudinal *= longitudinalMultiplier;
        lateral *= lateralMultiplier;

        // This is the same x/z rotation as
        // hytalegym/jax/combat/runtime/step.py. Positive local longitudinal
        // means forward, which is -Z at zero body yaw.
        double sin = Math.sin(bodyYawRadians);
        double cos = Math.cos(bodyYawRadians);
        double x = longitudinal * -sin + lateral * cos;
        double z = longitudinal * -cos - lateral * sin;
        return new Vector(x, z, gait);
    }

    private static int clampGait(int gait) {
        return Math.max(ActionDecoder.GAIT_IDLE,
            Math.min(ActionDecoder.GAIT_SNEAK, gait));
    }

    private PlayerLocomotionProjection() {
    }
}
