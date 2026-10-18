package com.hytalerlbridge.policy.action;

import com.hytalerlbridge.policy.ActionDecoder;

/**
 * Greedy selection under Hytale's one-standard-input-root constraint.
 *
 * <p>The network publishes twelve categorical heads, but base attack, ability,
 * guard and use/block are alternative standard roots. Selecting every head
 * independently can make the whole decision illegal. This class finds the
 * highest-scoring legal joint factor vector without changing the exported head
 * layout. Locomotion, look, body steer, jump, hotbar and the retained
 * block-trigger factor remain independent, matching the JAX distribution used
 * for PPO.
 *
 * <p>Recipe was a fifth root until the head stopped being published for a
 * combat agent. Dodge was independent until it merged into
 * `locomotion_gait_compass`, where it is mutually exclusive with a step by
 * construction rather than by arbitration.
 */
public final class StandardRootSelector {

    public static final String SCHEMA = "arsenal_standard_root_v1";
    private static final float MASKED_LOGIT_CUTOFF = -5.0e8f;

    private static final int CATEGORY_NONE = 0;
    private static final int CATEGORY_BASE_ATTACK = 1;
    private static final int CATEGORY_ABILITY = 2;
    private static final int CATEGORY_GUARD = 3;
    private static final int CATEGORY_BLOCK_OR_USE = 4;

    private StandardRootSelector() {
    }

    /** Return the same constrained greedy factors as JAX inference. */
    public static int[] select(float[] logits, int[] headSizes) {
        int[] offsets = offsets(headSizes, logits.length);
        requirePublishedLayout(headSizes);

        Choice baseNeutral = best(
            logits,
            offsets[ActionDecoder.BASE_ACTION],
            headSizes[ActionDecoder.BASE_ACTION],
            false,
            true
        );
        Choice baseAttack = best(
            logits,
            offsets[ActionDecoder.BASE_ACTION],
            headSizes[ActionDecoder.BASE_ACTION],
            true,
            false
        );
        Choice abilityNone = singleton(
            logits, offsets[ActionDecoder.ABILITY], 0);
        Choice abilityActive = active(
            logits, offsets[ActionDecoder.ABILITY], headSizes[ActionDecoder.ABILITY]);
        Choice guardOff = singleton(logits, offsets[ActionDecoder.GUARD], 0);
        Choice guardOn = singleton(logits, offsets[ActionDecoder.GUARD], 1);
        Choice useOff = singleton(logits, offsets[ActionDecoder.USE], 0);
        Choice useAny = all(logits, offsets[ActionDecoder.USE], headSizes[ActionDecoder.USE]);
        Choice blockNone = singleton(logits, offsets[ActionDecoder.BLOCK], 0);
        Choice blockActive = active(
            logits, offsets[ActionDecoder.BLOCK], headSizes[ActionDecoder.BLOCK]);

        float fixedNone = sum(
            abilityNone.score(), guardOff.score(), useOff.score(),
            blockNone.score());
        float[] categoryScores = {
            sum(baseNeutral.score(), fixedNone),
            sum(baseAttack.score(), fixedNone),
            sum(baseNeutral.score(), abilityActive.score(), guardOff.score(),
                useOff.score(), blockNone.score()),
            sum(baseNeutral.score(), abilityNone.score(), guardOn.score(),
                useOff.score(), blockNone.score()),
            sum(baseNeutral.score(), abilityNone.score(), guardOff.score(),
                useAny.score(), blockActive.score())
        };
        int category = argmax(categoryScores);

        int[] factors = new int[headSizes.length];
        for (int head = 0; head < headSizes.length; head++) {
            factors[head] = all(logits, offsets[head], headSizes[head]).index();
        }
        factors[ActionDecoder.BASE_ACTION] = category == CATEGORY_BASE_ATTACK
            ? baseAttack.index() : baseNeutral.index();
        factors[ActionDecoder.ABILITY] = category == CATEGORY_ABILITY
            ? abilityActive.index() : abilityNone.index();
        factors[ActionDecoder.GUARD] = category == CATEGORY_GUARD
            ? guardOn.index() : guardOff.index();
        factors[ActionDecoder.USE] = category == CATEGORY_BLOCK_OR_USE
            ? useAny.index() : useOff.index();
        factors[ActionDecoder.BLOCK] = category == CATEGORY_BLOCK_OR_USE
            ? blockActive.index() : blockNone.index();
        return factors;
    }

    private static int[] offsets(int[] headSizes, int logitCount) {
        if (headSizes == null || headSizes.length == 0) {
            throw new IllegalArgumentException("action head sizes must not be empty");
        }
        int[] result = new int[headSizes.length];
        int total = 0;
        for (int head = 0; head < headSizes.length; head++) {
            if (headSizes[head] < 1) {
                throw new IllegalArgumentException("action head sizes must be positive");
            }
            result[head] = total;
            total += headSizes[head];
        }
        if (logitCount != total) {
            throw new IllegalArgumentException(
                "expected " + total + " logits, got " + logitCount);
        }
        return result;
    }

    private static void requirePublishedLayout(int[] sizes) {
        if (sizes.length != 12
            || sizes[ActionDecoder.BASE_ACTION] != 12
            || sizes[ActionDecoder.ABILITY] != 17
            || sizes[ActionDecoder.GUARD] != 2
            || sizes[ActionDecoder.JUMP] != 2
            || sizes[ActionDecoder.LOCOMOTION] != 37
            || sizes[ActionDecoder.YAW_BINS] != 9
            || sizes[ActionDecoder.BODY_YAW_BINS] != 9
            || sizes[ActionDecoder.PITCH_BINS] != 5
            || sizes[ActionDecoder.HOTBAR] != 10
            || sizes[ActionDecoder.USE] != 2
            || sizes[ActionDecoder.BLOCK_TRIGGER] != 2
            || sizes[ActionDecoder.BLOCK] != 17) {
            throw new IllegalArgumentException(
                SCHEMA + " requires the published 12-head Arsenal layout");
        }
    }

    private static Choice singleton(float[] logits, int offset, int index) {
        float score = available(logits[offset + index])
            ? logits[offset + index] : Float.NEGATIVE_INFINITY;
        return new Choice(index, score);
    }

    private static Choice active(float[] logits, int offset, int size) {
        return range(logits, offset, 1, size);
    }

    private static Choice all(float[] logits, int offset, int size) {
        return range(logits, offset, 0, size);
    }

    private static Choice best(
        float[] logits,
        int offset,
        int size,
        boolean attacks,
        boolean nonAttacks
    ) {
        Choice best = new Choice(0, Float.NEGATIVE_INFINITY);
        for (int index = 0; index < size; index++) {
            boolean attack = index == ActionDecoder.SKILL_ATTACK
                || index == ActionDecoder.SKILL_APPROACH_ATTACK
                || index == ActionDecoder.SKILL_RETREAT_ATTACK;
            if ((attack && attacks) || (!attack && nonAttacks)) {
                best = better(best, index, logits[offset + index]);
            }
        }
        return best;
    }

    private static Choice range(float[] logits, int offset, int start, int end) {
        Choice best = new Choice(start, Float.NEGATIVE_INFINITY);
        for (int index = start; index < end; index++) {
            best = better(best, index, logits[offset + index]);
        }
        return best;
    }

    private static Choice better(Choice current, int index, float score) {
        if (available(score) && score > current.score()) {
            return new Choice(index, score);
        }
        return current;
    }

    private static boolean available(float value) {
        return Float.isFinite(value) && value > MASKED_LOGIT_CUTOFF;
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int index = 1; index < values.length; index++) {
            if (values[index] > values[best]) {
                best = index;
            }
        }
        return best;
    }

    private static float sum(float... values) {
        float total = 0.0f;
        for (float value : values) {
            total += value;
        }
        return total;
    }

    private record Choice(int index, float score) {
    }
}
