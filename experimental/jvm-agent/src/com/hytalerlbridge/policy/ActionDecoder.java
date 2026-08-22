package com.hytalerlbridge.policy;

import java.util.Arrays;

/**
 * Turns the 12 policy head indices into the action the server can apply.
 *
 * <p>In Python this is two stages: {@code decode_staged_action_surface_factors}
 * (policy/surface.py) produces a semantic {@code LearnerArsenalAction}, then the
 * native translate step (v3/native/transport.py) flattens it into the wire map
 * that the plugin's {@code AgentAction.fromMap} reads. In-process the wire
 * disappears, so both stages collapse into this one class.
 *
 * <p>Dependency-free on purpose -- it references no plugin and no engine type,
 * so {@code ActionTest} can certify it against JAX-exported reference cases
 * without a Hytale server on the classpath. {@code PolicyAgentAdapter} does the
 * (trivial) mapping from {@link Decoded} to the plugin record.
 *
 * <p>Three behaviours here are easy to get subtly wrong and are all pinned by
 * exported cases rather than reasoned about:
 * <ul>
 *   <li><b>Arbitration.</b> At most one "standard input root" may start per
 *       action -- attack, ability, guard, use, block, recipe. Two at once makes
 *       the <i>whole</i> action illegal, not just the loser.
 *   <li><b>Illegality is total.</b> One out-of-range or mask-forbidden head
 *       zeroes every field, including movement, not only the offending head.
 *   <li><b>world_move suppresses the strafe booleans but not attack.</b> When a
 *       compass direction is requested, forward/back/left/right stay false and
 *       the server derives the vector from the direction itself; attack still
 *       fires.
 * </ul>
 */
public final class ActionDecoder {

    /** Head order, matching {@code Policy.HEAD_SIZES}. */
    public static final int BASE_ACTION = 0;
    public static final int ABILITY = 1;
    public static final int GUARD = 2;
    public static final int JUMP = 3;
    public static final int LOCOMOTION = 4;
    public static final int YAW_BINS = 5;
    public static final int BODY_YAW_BINS = 6;
    public static final int PITCH_BINS = 7;
    public static final int HOTBAR = 8;
    public static final int USE = 9;
    public static final int BLOCK_TRIGGER = 10;
    public static final int BLOCK = 11;

    // ---- locomotion_gait_compass ------------------------------------------
    //
    // One 37-wide categorical carrying idle, the gait ring, and dodge, because
    // all three occupy the same physical slot. Mirrors `split_locomotion_choice`
    // in `observation/v3/policy/layout.py`; the constants are duplicated here
    // rather than derived because this side has no access to the Python module,
    // and `ContractTest` is what proves they still agree.

    /** Compass directions in the gait ring: N, NE, E, SE, S, SW, W, NW. */
    public static final int COMPASS_DIRECTIONS = 8;
    /** Choices at or above this are a dodge rather than a step. */
    public static final int LOCOMOTION_DODGE_START = 33;
    /** Choice 0: standing still. */
    public static final int LOCOMOTION_IDLE = 0;

    public static final int GAIT_IDLE = 0;
    public static final int GAIT_WALK = 1;
    public static final int GAIT_RUN = 2;
    public static final int GAIT_SPRINT = 3;
    public static final int GAIT_SNEAK = 4;

    public static final int SKILL_IDLE = 0;
    public static final int SKILL_FACE_TARGET = 1;
    public static final int SKILL_APPROACH = 2;
    public static final int SKILL_RETREAT = 3;
    public static final int SKILL_STRAFE_LEFT = 4;
    public static final int SKILL_STRAFE_RIGHT = 5;
    public static final int SKILL_ATTACK = 6;
    public static final int SKILL_APPROACH_ATTACK = 7;
    public static final int SKILL_RETREAT_ATTACK = 8;
    public static final int SKILL_COUNT = 9;

    /** base_action values at or above this are door intents, not skills. */
    public static final int ACTION_DOOR_OPEN = 9;

    public static final int BLOCK_TRIGGER_NONE = 0;
    public static final int BLOCK_TRIGGER_PRIMARY = 1;

    public static final double MAXIMUM_TURN_DEGREES = 45.0;
    public static final double FINE_TURN_DEGREES = 1.0;

    /**
     * A decoded action. The first block is the semantic surface; the second is
     * what the server actually applies.
     */
    public record Decoded(
        boolean actionLegal,
        int skillId,
        int abilitySlot,
        boolean guardHeld,
        int dodgeDirection,
        boolean jumpHeld,
        int worldMoveDirection,
        /** Which gait the ring asked for: one of the {@code GAIT_*} values. */
        int gait,
        double yawDeltaDegrees,
        /**
         * Body steer, independent of {@code yawDeltaDegrees}.
         *
         * <p>Head yaw aims and body yaw steers. Collapsing them would make an
         * agent unable to look one way while travelling another, which is the
         * whole reason the contract publishes two heads of identical width.
         */
        double bodyYawDeltaDegrees,
        double pitchDeltaDegrees,
        /** Requested hotbar slot, or -1 for "no switch". */
        int hotbarSlot,
        boolean useRequested,
        int blockInteractionTrigger,
        int recipeCandidateIndex,
        int blockCandidateIndex,
        // derived: what applyControl consumes
        boolean forward,
        boolean back,
        boolean left,
        boolean right,
        boolean attack
    ) {
        /**
         * Whether this action's executed root is a world verb.
         *
         * <p><b>A selected block candidate is not enough.</b> {@code use}
         * suppresses block -- the same rule used by root arbitration
         * ({@code block_requested = block > 0 && !use} in {@code surface.py}) -- so an
         * action with both set executes as one *use* root. It must still enter
         * the World-verb sink: the block slot supplies Use's selected target,
         * while the Use bit selects which interaction root executes.
         */
        public boolean requestsWorldVerb() {
            return useRequested
                || recipeCandidateIndex >= 0
                || blockCandidateIndex >= 0;
        }
    }

    /** The all-neutral action, used whenever any head is illegal. */
    public static final Decoded REJECTED = new Decoded(
        false, -1, -1, false, 0, false, 0, GAIT_IDLE, 0.0, 0.0, 0.0, -1, false,
        BLOCK_TRIGGER_NONE, -1, -1,
        false, false, false, false, false
    );

    /** Gait, compass direction and dodge, fanned out of one locomotion choice. */
    public record Locomotion(int gait, int direction, int dodge) {
    }

    /**
     * Fan one {@code locomotion_gait_compass} choice into its three fields.
     *
     * <p>Mirrors {@code split_locomotion_choice} in
     * {@code observation/v3/policy/layout.py}. The published head is a single
     * categorical because the gaits and a dodge occupy the same physical slot;
     * everything downstream still consumes the separate fields.
     *
     * <ul>
     *   <li>{@code 0} -- idle: no gait, no direction, no dodge;
     *   <li>{@code 1..32} -- a step: {@code ring = choice - 1}, gait is
     *       {@code ring / 8 + 1} and direction is {@code ring % 8 + 1};
     *   <li>{@code 33..36} -- a dodge: {@code choice - 33 + 1}.
     * </ul>
     */
    public static Locomotion splitLocomotion(int choice) {
        if (choice >= LOCOMOTION_DODGE_START) {
            return new Locomotion(
                GAIT_IDLE, 0, choice - LOCOMOTION_DODGE_START + 1);
        }
        if (choice <= LOCOMOTION_IDLE) {
            return new Locomotion(GAIT_IDLE, 0, 0);
        }
        int ring = choice - 1;
        return new Locomotion(
            ring / COMPASS_DIRECTIONS + 1, ring % COMPASS_DIRECTIONS + 1, 0);
    }

    /**
     * The locomotion choice that requests dodge {@code direction} (1-based).
     *
     * <p>The inverse of {@link #splitLocomotion}. Callers that used to address
     * a dedicated dodge head by direction go through this rather than open-code
     * the offset, so the dodge block can move again without a hunt.
     */
    public static int locomotionDodgeChoice(int direction) {
        if (direction < 1
            || direction > Policy.HEAD_SIZES[LOCOMOTION] - LOCOMOTION_DODGE_START) {
            throw new IllegalArgumentException(
                "dodge direction out of range: " + direction);
        }
        return LOCOMOTION_DODGE_START + direction - 1;
    }

    /** How many dodge directions the locomotion head publishes. */
    public static int dodgeDirectionCount() {
        return Policy.HEAD_SIZES[LOCOMOTION] - LOCOMOTION_DODGE_START;
    }

    /** The locomotion choice for a gait (1-based) and compass direction (1-based). */
    public static int locomotionStepChoice(int gait, int direction) {
        if (gait < 1 || direction < 1 || direction > COMPASS_DIRECTIONS) {
            throw new IllegalArgumentException(
                "step out of range: gait " + gait + " direction " + direction);
        }
        int choice = (gait - 1) * COMPASS_DIRECTIONS + (direction - 1) + 1;
        if (choice >= LOCOMOTION_DODGE_START) {
            throw new IllegalArgumentException("gait out of range: " + gait);
        }
        return choice;
    }

    private ActionDecoder() {
    }

    /**
     * Decode greedy head indices against the same legality mask the policy was
     * given.
     *
     * @param factors 12 head indices, one per head, in {@code Policy.HEAD_SIZES} order
     * @param legal   99 per-logit legality bits, concatenated in head order
     */
    public static Decoded decode(int[] factors, boolean[] legal) {
        int[] sizes = Policy.HEAD_SIZES;
        if (factors.length != sizes.length) {
            throw new IllegalArgumentException(
                "expected " + sizes.length + " action factors, got " + factors.length);
        }
        int total = 0;
        for (int size : sizes) {
            total += size;
        }
        if (legal.length != total) {
            throw new IllegalArgumentException(
                "expected " + total + " mask bits, got " + legal.length);
        }

        // A head is usable only if its chosen index is in range *and* that
        // logit was legal. Both are required; range alone would let the policy
        // pick a masked-off ability and have it silently applied.
        boolean usable = true;
        int offset = 0;
        for (int head = 0; head < sizes.length; head++) {
            int value = factors[head];
            if (value < 0 || value >= sizes[head] || !legal[offset + value]) {
                usable = false;
            }
            offset += sizes[head];
        }

        boolean useRequested = factors[USE] == 1;
        boolean legacyAttack = factors[BASE_ACTION] == SKILL_ATTACK
            || factors[BASE_ACTION] == SKILL_APPROACH_ATTACK
            || factors[BASE_ACTION] == SKILL_RETREAT_ATTACK;
        // A block request is suppressed by `use`, so the pair does not count as
        // two roots -- matching surface.py, where block_requested carries
        // `& ~use_requested`.
        boolean blockRequested = factors[BLOCK] > 0 && !useRequested;
        // The recipe head is no longer published for a combat agent, so it can
        // no longer contribute a root. `recipeCandidateIndex` stays on the
        // record as a constant -1 because the world-action executor and the
        // native channels still read the field.
        int roots = (legacyAttack ? 1 : 0)
            + (factors[ABILITY] > 0 ? 1 : 0)
            + (factors[GUARD] == 1 ? 1 : 0)
            + (useRequested ? 1 : 0)
            + (blockRequested ? 1 : 0);

        if (!usable || roots > 1) {
            return REJECTED;
        }

        int skillId = factors[BASE_ACTION];
        Locomotion locomotion = splitLocomotion(factors[LOCOMOTION]);
        int worldMove = locomotion.direction();

        // Door intents occupy base_action 9..11 and carry no locomotion; the
        // Python decoder folds them (and any illegal action) to IDLE before
        // reaching skills_to_actions.
        int movementSkill = skillId < SKILL_COUNT ? skillId : SKILL_IDLE;
        boolean attack = movementSkill == SKILL_ATTACK
            || movementSkill == SKILL_APPROACH_ATTACK
            || movementSkill == SKILL_RETREAT_ATTACK;
        boolean forward = movementSkill == SKILL_APPROACH
            || movementSkill == SKILL_APPROACH_ATTACK;
        boolean back = movementSkill == SKILL_RETREAT
            || movementSkill == SKILL_RETREAT_ATTACK;
        boolean left = movementSkill == SKILL_STRAFE_LEFT;
        boolean right = movementSkill == SKILL_STRAFE_RIGHT;
        if (worldMove != 0) {
            // The compass direction replaces the strafe booleans; the server
            // derives dx/dz from the direction. Attack is unaffected.
            forward = false;
            back = false;
            left = false;
            right = false;
        }

        return new Decoded(
            true,
            skillId,
            factors[ABILITY] - 1,
            factors[GUARD] == 1,
            locomotion.dodge(),
            factors[JUMP] == 1,
            worldMove,
            locomotion.gait(),
            signedDelta(factors[YAW_BINS], sizes[YAW_BINS]),
            signedDelta(factors[BODY_YAW_BINS], sizes[BODY_YAW_BINS]),
            signedDelta(factors[PITCH_BINS], sizes[PITCH_BINS]),
            factors[HOTBAR] - 1,
            useRequested,
            factors[BLOCK_TRIGGER] + BLOCK_TRIGGER_PRIMARY,
            -1,
            factors[BLOCK] - 1,
            forward, back, left, right, attack
        );
    }

    /** Map a bin through the policy contract's reusable look codebook. */
    private static double signedDelta(int choice, int size) {
        return lookDeltaValues(
            size,
            MAXIMUM_TURN_DEGREES,
            FINE_TURN_DEGREES
        )[choice];
    }

    /**
     * Build the symmetric geometric coarse-to-fine look codebook.
     *
     * <p>The outer pair retains the maximum turn, the inner pair retains a
     * fine correction, and any intervening pairs are distributed
     * geometrically. Each magnitude is rounded to float32 exactly as JAX does
     * before the bridge receives it. The rule depends only on head cardinality
     * and limits, so new policy layouts reuse it without fixture or weapon
     * branches.
     */
    public static double[] lookDeltaValues(
        int size,
        double maximumDegrees,
        double fineDegrees
    ) {
        if (size < 3 || size % 2 == 0) {
            throw new IllegalArgumentException(
                "signed delta heads must have odd size >= 3, got " + size);
        }
        if (!Double.isFinite(maximumDegrees) || maximumDegrees <= 0.0) {
            throw new IllegalArgumentException(
                "maximumDegrees must be finite and positive");
        }
        if (!Double.isFinite(fineDegrees)
            || fineDegrees <= 0.0
            || fineDegrees > maximumDegrees) {
            throw new IllegalArgumentException(
                "fineDegrees must be finite, positive, and at most maximumDegrees");
        }
        int half = (size - 1) / 2;
        if (half > 1 && fineDegrees >= maximumDegrees) {
            throw new IllegalArgumentException(
                "fineDegrees must be less than maximumDegrees for heads wider than 3");
        }
        double[] values = new double[size];
        double ratio = half == 1
            ? 1.0
            : Math.pow(maximumDegrees / fineDegrees, 1.0 / (half - 1));
        for (int index = 0; index < half; index++) {
            double magnitude = half == 1
                ? maximumDegrees
                : index == 0
                    ? fineDegrees
                    : index == half - 1
                    ? maximumDegrees
                    : fineDegrees * Math.pow(ratio, index);
            double wireMagnitude = (double) (float) magnitude;
            if (index > 0 && wireMagnitude <= values[half + index]) {
                throw new IllegalArgumentException(
                    "look delta magnitudes must remain distinct after float32 rounding");
            }
            values[half + 1 + index] = wireMagnitude;
            values[half - 1 - index] = -wireMagnitude;
        }
        return values;
    }

    /** All-legal mask, for callers that have no evidence-derived one. */
    public static boolean[] permissive() {
        int total = 0;
        for (int size : Policy.HEAD_SIZES) {
            total += size;
        }
        boolean[] legal = new boolean[total];
        Arrays.fill(legal, true);
        return legal;
    }
}
