package com.hytalerlbridge.policy.perception.projection;

/** Learner-v3 ability feature derivation for two entity rows and 16 slots. */
public final class AbilityProjection {

    public static final int ENTITY_COUNT = 2;
    public static final int ABILITY_CAPACITY = 16;
    public static final int RESOURCE_COUNT = 7;
    public static final int FEATURE_COUNT = 4 + RESOURCE_COUNT;
    public static final int SLOT_COUNT = ENTITY_COUNT * ABILITY_CAPACITY;
    public static final int VALUE_COUNT = SLOT_COUNT * FEATURE_COUNT;
    public static final int RESOURCE_VALUE_COUNT = ENTITY_COUNT * RESOURCE_COUNT;
    public static final int ABILITY_RESOURCE_VALUE_COUNT =
        SLOT_COUNT * RESOURCE_COUNT;

    public static final int RESOURCE_COST_NONE = 0;
    public static final int RESOURCE_COST_SINGLE_APPLICATION = 1;
    public static final int RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT = 2;
    public static final int
        RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT_STAMINA_BREAK_IMMUNE = 3;
    public static final int NATIVE_NPC_INSTANT_ADD_STAT_APPLICATIONS = 2;

    /**
     * Raw numeric inputs plus authoritative per-slot legality.
     *
     * <p>{@code legal} is intentionally an input. Native interaction-chain
     * arbitration and world requirements decide admission; duplicating those
     * rules here would create a second combat runtime. This projector owns the
     * learner features and structural mask, while acquisition must supply the
     * server's current legality result.
     */
    public record Input(
        float[] resources,
        float[] resourceSpan,
        float[] abilityDurationSeconds,
        float[] authoredCooldownSeconds,
        float[] remainingCooldownSeconds,
        int[] activeAbilitySlot,
        float[] abilityElapsedSeconds,
        float[] authoredResourceCost,
        int[] resourceCostKind,
        float[] resourceMinimum,
        boolean[] authoredAbilityMask,
        boolean[] equipped,
        boolean[] overflow,
        boolean[] legal
    ) {
        public Input {
            require(resources.length, RESOURCE_VALUE_COUNT, "resources");
            require(resourceSpan.length, RESOURCE_VALUE_COUNT, "resource span");
            require(abilityDurationSeconds.length, SLOT_COUNT, "duration");
            require(authoredCooldownSeconds.length, SLOT_COUNT, "authored cooldown");
            require(remainingCooldownSeconds.length, SLOT_COUNT, "remaining cooldown");
            require(activeAbilitySlot.length, ENTITY_COUNT, "active slot");
            require(abilityElapsedSeconds.length, ENTITY_COUNT, "ability elapsed");
            require(authoredResourceCost.length, ABILITY_RESOURCE_VALUE_COUNT,
                "authored resource cost");
            require(resourceCostKind.length, ABILITY_RESOURCE_VALUE_COUNT,
                "resource cost kind");
            require(resourceMinimum.length, ABILITY_RESOURCE_VALUE_COUNT,
                "resource minimum");
            require(authoredAbilityMask.length, SLOT_COUNT, "ability mask");
            require(equipped.length, ENTITY_COUNT, "equipped");
            require(overflow.length, ENTITY_COUNT, "overflow");
            require(legal.length, SLOT_COUNT, "legal");
            resources = resources.clone();
            resourceSpan = resourceSpan.clone();
            abilityDurationSeconds = abilityDurationSeconds.clone();
            authoredCooldownSeconds = authoredCooldownSeconds.clone();
            remainingCooldownSeconds = remainingCooldownSeconds.clone();
            activeAbilitySlot = activeAbilitySlot.clone();
            abilityElapsedSeconds = abilityElapsedSeconds.clone();
            authoredResourceCost = authoredResourceCost.clone();
            resourceCostKind = resourceCostKind.clone();
            resourceMinimum = resourceMinimum.clone();
            authoredAbilityMask = authoredAbilityMask.clone();
            equipped = equipped.clone();
            overflow = overflow.clone();
            legal = legal.clone();
        }
    }

    public record Result(float[] values, boolean[] mask, boolean[] legal) {
        public Result {
            require(values.length, VALUE_COUNT, "ability values");
            require(mask.length, SLOT_COUNT, "ability mask");
            require(legal.length, SLOT_COUNT, "ability legal");
            values = values.clone();
            mask = mask.clone();
            legal = legal.clone();
        }

        @Override
        public float[] values() {
            return values.clone();
        }

        @Override
        public boolean[] mask() {
            return mask.clone();
        }

        @Override
        public boolean[] legal() {
            return legal.clone();
        }
    }

    private AbilityProjection() {
    }

    /** Match the float32 learner-v3 ability projection and structural mask. */
    public static Result project(Input input) {
        float[] resources = input.resources();
        float[] spans = input.resourceSpan();
        float[] durations = input.abilityDurationSeconds();
        float[] authoredCooldowns = input.authoredCooldownSeconds();
        float[] cooldowns = input.remainingCooldownSeconds();
        int[] activeSlots = input.activeAbilitySlot();
        float[] elapsed = input.abilityElapsedSeconds();
        float[] authoredCosts = input.authoredResourceCost();
        int[] costKinds = input.resourceCostKind();
        float[] minima = input.resourceMinimum();
        boolean[] authoredMask = input.authoredAbilityMask();
        boolean[] equipped = input.equipped();
        boolean[] overflow = input.overflow();

        float[] values = new float[VALUE_COUNT];
        boolean[] mask = new boolean[SLOT_COUNT];
        for (int entity = 0; entity < ENTITY_COUNT; entity++) {
            int entitySlot = entity * ABILITY_CAPACITY;
            int entityResource = entity * RESOURCE_COUNT;
            float durationScale = 0.0f;
            for (int localSlot = 0; localSlot < ABILITY_CAPACITY; localSlot++) {
                int slot = entitySlot + localSlot;
                if (authoredMask[slot]) {
                    durationScale = Math.max(durationScale, durations[slot]);
                }
            }
            durationScale = Math.max(durationScale, 1.0f);

            for (int localSlot = 0; localSlot < ABILITY_CAPACITY; localSlot++) {
                int slot = entitySlot + localSlot;
                mask[slot] = authoredMask[slot]
                    && equipped[entity]
                    && !overflow[entity];
                if (!mask[slot]) {
                    continue;
                }
                int at = slot * FEATURE_COUNT;
                values[at] = durations[slot] / durationScale;
                values[at + 1] = cooldowns[slot]
                    / Math.max(authoredCooldowns[slot], 1.0f);
                values[at + 2] = localSlot == activeSlots[entity]
                    ? elapsed[entity] / Math.max(durations[slot], 1.0e-6f)
                    : 0.0f;
                boolean affordable = true;
                int abilityResource = slot * RESOURCE_COUNT;
                for (int resource = 0; resource < RESOURCE_COUNT; resource++) {
                    affordable &= resources[entityResource + resource] + 1.0e-6f
                        >= minima[abilityResource + resource];
                    float normalizedCost = 0.0f;
                    if (spans[entityResource + resource] > 0.0f) {
                        normalizedCost = authoredCosts[abilityResource + resource]
                            * costApplications(costKinds[abilityResource + resource])
                            / Math.max(spans[entityResource + resource], 1.0e-6f);
                    }
                    values[at + 4 + resource] = clip(normalizedCost);
                }
                values[at + 3] = affordable ? 1.0f : 0.0f;
                for (int feature = 0; feature < 4; feature++) {
                    values[at + feature] = clip(values[at + feature]);
                }
            }
        }
        return new Result(values, mask, input.legal());
    }

    private static float costApplications(int kind) {
        if (kind == RESOURCE_COST_SINGLE_APPLICATION) {
            return 1.0f;
        }
        if (kind == RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT
            || kind
                == RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT_STAMINA_BREAK_IMMUNE) {
            return NATIVE_NPC_INSTANT_ADD_STAT_APPLICATIONS;
        }
        return 0.0f;
    }

    private static float clip(float value) {
        return Math.max(-1.0f, Math.min(1.0f, value));
    }

    private static void require(int actual, int expected, String name) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                name + " width drift: " + actual + " != " + expected);
        }
    }
}
