package com.hytalerlbridge.policy.perception.projection;

/** Derivations for policy action masks that depend on live actor evidence. */
public final class ActionMaskProjection {

    public static final int DODGE_DIRECTION_COUNT = 4;

    /**
     * Hytale 0.5.7's native NPC dodge surface authors only left and right.
     * Forward and back remain structurally closed rather than becoming
     * available merely because the swept corridor is clear.
     */
    private static final boolean[] DODGE_AUTHORED_ACTION_MASK = {
        false, false, true, true,
    };

    public record DodgeInput(
        boolean[] corridorClear,
        boolean movementEnabled,
        boolean alive,
        float stamina,
        float dodgeCost
    ) {
        public DodgeInput {
            if (corridorClear.length != DODGE_DIRECTION_COUNT) {
                throw new IllegalArgumentException(
                    "dodge corridor width drift: " + corridorClear.length);
            }
            corridorClear = corridorClear.clone();
        }

        @Override
        public boolean[] corridorClear() {
            return corridorClear.clone();
        }
    }

    private ActionMaskProjection() {
    }

    /** Match learner-v3's four directional dodge legality bits. */
    public static boolean[] dodge(DodgeInput input) {
        boolean[] corridor = input.corridorClear();
        boolean actorReady = input.movementEnabled()
            && input.alive()
            && input.stamina() >= input.dodgeCost();
        boolean[] result = new boolean[DODGE_DIRECTION_COUNT];
        for (int direction = 0; direction < DODGE_DIRECTION_COUNT; direction++) {
            result[direction] = corridor[direction]
                && DODGE_AUTHORED_ACTION_MASK[direction]
                && actorReady;
        }
        return result;
    }
}
