package com.hytalerlbridge.policy.perception;

import com.hytalerlbridge.policy.Policy;

/** Builds the twelve-head policy legality mask from actor-safe evidence. */
public final class ActionMaskAssembler {

    public static final int SKILL_COUNT = 9;
    public static final int DOOR_INTENT_COUNT = 3;
    public static final int ABILITY_COUNT = 16;
    public static final int DODGE_DIRECTION_COUNT = 4;
    public static final int CANDIDATE_COUNT = 16;
    public static final int LOCOMOTION_DODGE_START = 33;
    public static final int LOCOMOTION_HEAD_SIZE = 37;
    public static final int HOTBAR_HEAD_SIZE = 10;

    /**
     * Inputs already filtered by the native host at the current tick.
     *
     * <p>The recipe candidate evidence is intentionally not an input here:
     * recipe selection is no longer a published combat-policy head. Recipe
     * evidence may still be carried by the observation/world-action seam, but
     * it must not change the twelve-head combat mask.
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
        boolean use,
        boolean[] blockTrigger,
        boolean blockAvailable,
        boolean[] blockCandidates
    ) {
        public Input {
            require(skill, SKILL_COUNT, "skill");
            require(door, DOOR_INTENT_COUNT, "door");
            require(ability, ABILITY_COUNT, "ability");
            require(dodge, DODGE_DIRECTION_COUNT, "dodge");
            require(blockTrigger, 2, "block trigger");
            require(blockCandidates, CANDIDATE_COUNT, "block candidates");
            skill = skill.clone();
            door = door.clone();
            ability = ability.clone();
            dodge = dodge.clone();
            blockTrigger = blockTrigger.clone();
            blockCandidates = blockCandidates.clone();
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

        // Base heads: the first choice of each added binary/categorical head
        // is the neutral action. This is the exact 33-column base prefix in
        // layout.py: 9 skills, 3 doors, ability none+16, guard off/on, and
        // jump off/on.
        result[offset++] = true;
        offset = copyGated(result, offset, input.ability(), input.valid());
        result[offset++] = true;
        result[offset++] = input.valid() && input.guard();
        result[offset++] = true;
        result[offset++] = input.valid() && input.jump();

        // Locomotion is one categorical head. Choice 0 is idle, choices 1..32
        // are the gait/compass ring, and choices 33..36 are the authored dodge
        // directions. JAX gates the ring on translation availability while
        // preserving per-direction dodge legality.
        result[offset++] = true;
        for (int index = 1; index < LOCOMOTION_DODGE_START; index++) {
            result[offset++] = input.valid() && input.movement();
        }
        offset = copyGated(result, offset, input.dodge(), input.valid());

        // Look choices are legal whenever the observation is valid. A missing
        // native look actuator is represented by the producer returning an
        // invalid frame, not by silently shrinking the policy surface.
        offset = neutralOrAll(result, offset, Policy.HEAD_SIZES[5],
            input.valid(), Policy.HEAD_SIZES[5] / 2);
        offset = neutralOrAll(result, offset, Policy.HEAD_SIZES[6],
            input.valid(), Policy.HEAD_SIZES[6] / 2);
        offset = neutralOrAll(result, offset, Policy.HEAD_SIZES[7],
            input.valid(), Policy.HEAD_SIZES[7] / 2);

        // Choice 0 leaves the live slot alone; every number-key slot is a
        // legal request, including an empty slot. Loadout resolution decides
        // whether the selected slot has an applicable item.
        offset = neutralOrAll(result, offset, HOTBAR_HEAD_SIZE,
            input.valid(), 0);

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
