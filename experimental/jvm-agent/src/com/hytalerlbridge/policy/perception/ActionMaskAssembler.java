package com.hytalerlbridge.policy.perception;

import com.hytalerlbridge.policy.Policy;

/** Builds the twelve-head policy legality mask from actor-safe evidence. */
public final class ActionMaskAssembler {

    public static final int SKILL_COUNT = 9;
    public static final int DOOR_INTENT_COUNT = 3;
    public static final int ABILITY_COUNT = 16;
    public static final int DODGE_DIRECTION_COUNT = 4;
    public static final int CANDIDATE_COUNT = 16;

    /**
     * Inputs already filtered by the native host at the current tick.
     *
     * <p>The two candidate masks and their availability bits remain separate:
     * an all-false available row is not interchangeable with an unavailable
     * producer. The policy mask happens to close both, but the observation
     * carries that distinction.
     */
    public record Input(
        boolean valid,
        boolean[] skill,
        boolean[] door,
        boolean[] ability,
        boolean guard,
        boolean[] dodge,
        boolean jump,
        boolean movement,
        boolean look,
        boolean use,
        boolean[] blockTrigger,
        boolean blockAvailable,
        boolean[] blockCandidates,
        boolean recipeAvailable,
        boolean[] recipeCandidates
    ) {
        public Input {
            require(skill, SKILL_COUNT, "skill");
            require(door, DOOR_INTENT_COUNT, "door");
            require(ability, ABILITY_COUNT, "ability");
            require(dodge, DODGE_DIRECTION_COUNT, "dodge");
            require(blockTrigger, 2, "block trigger");
            require(blockCandidates, CANDIDATE_COUNT, "block candidates");
            require(recipeCandidates, CANDIDATE_COUNT, "recipe candidates");
            skill = skill.clone();
            door = door.clone();
            ability = ability.clone();
            dodge = dodge.clone();
            blockTrigger = blockTrigger.clone();
            blockCandidates = blockCandidates.clone();
            recipeCandidates = recipeCandidates.clone();
        }

        private static void require(boolean[] values, int width, String name) {
            if (values.length != width) {
                throw new IllegalArgumentException(
                    name + " mask has " + values.length + " bits, expected " + width);
            }
        }
    }

    private ActionMaskAssembler() {
    }

    /**
     * Match {@code arsenal_policy_action_mask} and
     * {@code staged_action_surface_mask} for one actor.
     */
    public static boolean[] assemble(Input input) {
        if (!input.valid()) {
            throw new IllegalStateException(
                "an invalid observation must skip the policy tick, not emit a mask");
        }
        boolean[] result = new boolean[actionSize()];
        int offset = 0;

        offset = copyGated(result, offset, input.skill(), input.valid());
        offset = copyGated(result, offset, input.door(), input.valid());

        // Every factored head retains a legal neutral choice even when the
        // actor or producer is unavailable.
        result[offset++] = true;
        offset = copyGated(result, offset, input.ability(), input.valid());
        result[offset++] = true;
        result[offset++] = input.valid() && input.guard();
        result[offset++] = true;
        offset = copyGated(result, offset, input.dodge(), input.valid());
        result[offset++] = true;
        result[offset++] = input.valid() && input.jump();

        offset = neutralOrAll(result, offset, Policy.HEAD_SIZES[5],
            input.valid() && input.movement(), 0);
        offset = neutralOrAll(result, offset, Policy.HEAD_SIZES[6],
            input.valid() && input.look(), Policy.HEAD_SIZES[6] / 2);
        offset = neutralOrAll(result, offset, Policy.HEAD_SIZES[7],
            input.valid() && input.look(), Policy.HEAD_SIZES[7] / 2);

        result[offset++] = true;
        result[offset++] = input.valid() && input.use();

        boolean anyTrigger = any(input.blockTrigger());
        boolean anyBlock = any(input.blockCandidates());
        boolean blockAction = input.valid()
            && input.blockAvailable()
            && anyBlock
            && anyTrigger;
        if (blockAction) {
            offset = copyGated(result, offset, input.blockTrigger(), true);
        } else {
            result[offset++] = true;
            result[offset++] = false;
        }

        result[offset++] = true;
        offset = copyGated(
            result,
            offset,
            input.recipeCandidates(),
            input.valid() && input.recipeAvailable());

        boolean blockTarget = input.valid()
            && input.blockAvailable()
            && anyBlock
            && (anyTrigger || input.use());
        result[offset++] = true;
        offset = copyGated(result, offset, input.blockCandidates(), blockTarget);

        if (offset != result.length) {
            throw new IllegalStateException(
                "assembled " + offset + " action bits, expected " + result.length);
        }
        assertHeadLiveness(result);
        return result;
    }

    /** Fail loudly if any factored head has no legal choice. */
    public static void assertHeadLiveness(boolean[] mask) {
        int expected = actionSize();
        if (mask.length != expected) {
            throw new IllegalArgumentException(
                "action mask has " + mask.length + " bits, expected "
                    + expected);
        }
        int offset = 0;
        for (int head = 0; head < Policy.HEAD_SIZES.length; head++) {
            boolean live = false;
            for (int index = 0; index < Policy.HEAD_SIZES[head]; index++) {
                live |= mask[offset + index];
            }
            if (!live) {
                throw new IllegalStateException(
                    "action head " + head + " has no legal choice");
            }
            offset += Policy.HEAD_SIZES[head];
        }
    }

    private static int copyGated(
        boolean[] destination,
        int offset,
        boolean[] source,
        boolean gate
    ) {
        for (boolean value : source) {
            destination[offset++] = gate && value;
        }
        return offset;
    }

    private static int neutralOrAll(
        boolean[] destination,
        int offset,
        int width,
        boolean available,
        int neutral
    ) {
        for (int index = 0; index < width; index++) {
            destination[offset++] = available || index == neutral;
        }
        return offset;
    }

    private static boolean any(boolean[] values) {
        for (boolean value : values) {
            if (value) {
                return true;
            }
        }
        return false;
    }

    private static int actionSize() {
        int total = 0;
        for (int width : Policy.HEAD_SIZES) {
            total += width;
        }
        return total;
    }
}
