package com.hytalerlbridge.policy.perception.projection;

/** Learner projections backed by combat-mechanics state and rules. */
public final class MechanicsProjection {

    public static final int ENTITY_COUNT = 2;
    public static final int RESOURCE_COUNT = 7;
    public static final int RESOURCE_VALUES = ENTITY_COUNT * RESOURCE_COUNT;
    public static final int DEFENSE_FEATURES = 7;
    public static final int DEFENSE_VALUES = ENTITY_COUNT * DEFENSE_FEATURES;
    public static final float CONTROL_IMMUNITY_MAXIMUM = 100.0f;
    public static final int STATUS_CAPACITY = 8;
    public static final int STATUS_FEATURES = 6;
    public static final int STATUS_SLOTS = ENTITY_COUNT * STATUS_CAPACITY;
    public static final int STATUS_VALUES = STATUS_SLOTS * STATUS_FEATURES;
    public static final float STATUS_DURATION_SCALE_SECONDS = 30.0f;

    /** Entity-major current/minimum/maximum resource values. */
    public record ResourceInput(
        float[] current,
        float[] minimum,
        float[] maximum,
        boolean[] available
    ) {
        public ResourceInput(
            float[] current,
            float[] minimum,
            float[] maximum
        ) {
            this(current, minimum, maximum, positiveSpans(minimum, maximum));
        }

        public ResourceInput {
            requireWidth(current, "current");
            requireWidth(minimum, "minimum");
            requireWidth(maximum, "maximum");
            if (available.length != RESOURCE_VALUES) {
                throw new IllegalArgumentException(
                    "resource availability width drift");
            }
            current = current.clone();
            minimum = minimum.clone();
            maximum = maximum.clone();
            available = available.clone();
        }

        @Override
        public float[] current() {
            return current.clone();
        }

        @Override
        public float[] minimum() {
            return minimum.clone();
        }

        @Override
        public float[] maximum() {
            return maximum.clone();
        }

        @Override
        public boolean[] available() {
            return available.clone();
        }

        private static void requireWidth(float[] values, String name) {
            if (values.length != RESOURCE_VALUES) {
                throw new IllegalArgumentException(
                    "resource " + name + " has " + values.length
                        + " values, expected " + RESOURCE_VALUES);
            }
        }

        private static boolean[] positiveSpans(
            float[] minimum, float[] maximum
        ) {
            if (minimum.length != maximum.length) {
                throw new IllegalArgumentException(
                    "resource minimum/maximum width mismatch");
            }
            boolean[] result = new boolean[minimum.length];
            for (int index = 0; index < result.length; index++) {
                result[index] = maximum[index] - minimum[index] > 0.0f;
            }
            return result;
        }
    }

    public record ResourceResult(float[] values, boolean[] mask) {
        public ResourceResult {
            if (values.length != RESOURCE_VALUES || mask.length != RESOURCE_VALUES) {
                throw new IllegalArgumentException("resource projection width drift");
            }
            values = values.clone();
            mask = mask.clone();
        }

        @Override
        public float[] values() {
            return values.clone();
        }

        @Override
        public boolean[] mask() {
            return mask.clone();
        }
    }

    /** Raw entity-major fields used by the seven defense columns. */
    public record DefenseInput(
        boolean[] guardActive,
        boolean[] staminaBroken,
        float[] dodgeInvulnerabilityRemaining,
        float[] dodgeInvulnerabilityDuration,
        float[] appliedVelocity,
        float[] dodgeForce,
        float[] staminaRegenDelay,
        float[] controlImmunity,
        float[] health
    ) {
        public DefenseInput {
            requireEntityWidth(guardActive, "guardActive");
            requireEntityWidth(staminaBroken, "staminaBroken");
            requireEntityWidth(dodgeInvulnerabilityRemaining, "dodge remaining");
            requireEntityWidth(dodgeInvulnerabilityDuration, "dodge duration");
            if (appliedVelocity.length != ENTITY_COUNT * 3) {
                throw new IllegalArgumentException("applied velocity width drift");
            }
            requireEntityWidth(dodgeForce, "dodge force");
            requireEntityWidth(staminaRegenDelay, "stamina regen delay");
            requireEntityWidth(controlImmunity, "control immunity");
            requireEntityWidth(health, "health");
            guardActive = guardActive.clone();
            staminaBroken = staminaBroken.clone();
            dodgeInvulnerabilityRemaining = dodgeInvulnerabilityRemaining.clone();
            dodgeInvulnerabilityDuration = dodgeInvulnerabilityDuration.clone();
            appliedVelocity = appliedVelocity.clone();
            dodgeForce = dodgeForce.clone();
            staminaRegenDelay = staminaRegenDelay.clone();
            controlImmunity = controlImmunity.clone();
            health = health.clone();
        }
    }

    public record DefenseResult(float[] values) {
        public DefenseResult {
            if (values.length != DEFENSE_VALUES) {
                throw new IllegalArgumentException("defense projection width drift");
            }
            values = values.clone();
        }

        @Override
        public float[] values() {
            return values.clone();
        }
    }

    /** Raw active-effect fields in entity-major, then slot-major order. */
    public record StatusInput(
        float[] maximumHealth,
        float[] resourceSpan,
        float[] remainingSeconds,
        float[] cycleElapsedSeconds,
        float[] cycleCooldownSeconds,
        float[] damagePerCycle,
        float[] healingPerCycle,
        int[] resourceId,
        float[] resourceDeltaPerCycle,
        float[] speedMultiplier,
        boolean[] active
    ) {
        public StatusInput {
            requireEntityWidth(maximumHealth, "maximum health");
            if (resourceSpan.length != RESOURCE_VALUES) {
                throw new IllegalArgumentException("status resource span width drift");
            }
            requireStatusWidth(remainingSeconds, "remaining seconds");
            requireStatusWidth(cycleElapsedSeconds, "cycle elapsed");
            requireStatusWidth(cycleCooldownSeconds, "cycle cooldown");
            requireStatusWidth(damagePerCycle, "damage per cycle");
            requireStatusWidth(healingPerCycle, "healing per cycle");
            if (resourceId.length != STATUS_SLOTS) {
                throw new IllegalArgumentException("status resource id width drift");
            }
            requireStatusWidth(resourceDeltaPerCycle, "resource delta");
            requireStatusWidth(speedMultiplier, "speed multiplier");
            if (active.length != STATUS_SLOTS) {
                throw new IllegalArgumentException("status active width drift");
            }
            maximumHealth = maximumHealth.clone();
            resourceSpan = resourceSpan.clone();
            remainingSeconds = remainingSeconds.clone();
            cycleElapsedSeconds = cycleElapsedSeconds.clone();
            cycleCooldownSeconds = cycleCooldownSeconds.clone();
            damagePerCycle = damagePerCycle.clone();
            healingPerCycle = healingPerCycle.clone();
            resourceId = resourceId.clone();
            resourceDeltaPerCycle = resourceDeltaPerCycle.clone();
            speedMultiplier = speedMultiplier.clone();
            active = active.clone();
        }
    }

    public record StatusResult(float[] values, boolean[] mask) {
        public StatusResult {
            if (values.length != STATUS_VALUES || mask.length != STATUS_SLOTS) {
                throw new IllegalArgumentException("status projection width drift");
            }
            values = values.clone();
            mask = mask.clone();
        }

        @Override
        public float[] values() {
            return values.clone();
        }

        @Override
        public boolean[] mask() {
            return mask.clone();
        }
    }

    private MechanicsProjection() {
    }

    /** Normalize each resource by its authored [minimum, maximum] interval. */
    public static ResourceResult resources(ResourceInput input) {
        float[] current = input.current();
        float[] minimum = input.minimum();
        float[] maximum = input.maximum();
        boolean[] available = input.available();
        float[] values = new float[RESOURCE_VALUES];
        boolean[] mask = new boolean[RESOURCE_VALUES];
        for (int index = 0; index < RESOURCE_VALUES; index++) {
            float span = maximum[index] - minimum[index];
            mask[index] = span > 0.0f && available[index];
            values[index] = mask[index]
                ? (current[index] - minimum[index])
                    / Math.max(span, Float.MIN_NORMAL)
                : 0.0f;
        }
        return new ResourceResult(values, mask);
    }

    /** Match the seven learner-v3 defensive-mechanics columns per entity. */
    public static DefenseResult defense(DefenseInput input) {
        float[] values = new float[DEFENSE_VALUES];
        for (int entity = 0; entity < ENTITY_COUNT; entity++) {
            int at = entity * DEFENSE_FEATURES;
            int velocity = entity * 3;
            float vx = input.appliedVelocity()[velocity];
            float vy = input.appliedVelocity()[velocity + 1];
            float vz = input.appliedVelocity()[velocity + 2];
            float speed = (float) Math.sqrt(vx * vx + vy * vy + vz * vz);
            values[at] = input.guardActive()[entity] ? 1.0f : 0.0f;
            values[at + 1] = input.staminaBroken()[entity] ? 1.0f : 0.0f;
            values[at + 2] = input.dodgeInvulnerabilityRemaining()[entity]
                / Math.max(input.dodgeInvulnerabilityDuration()[entity], 1.0e-6f);
            values[at + 3] = speed
                / Math.max(input.dodgeForce()[entity], 1.0e-6f);
            values[at + 4] = Math.max(-1.0f, Math.min(
                0.0f, input.staminaRegenDelay()[entity] / 2.0f));
            values[at + 5] = input.controlImmunity()[entity]
                / CONTROL_IMMUNITY_MAXIMUM;
            values[at + 6] = input.health()[entity] > 0.0f ? 1.0f : 0.0f;
        }
        return new DefenseResult(values);
    }

    /** Project active effects into six clipped learner floats per status slot. */
    public static StatusResult statuses(StatusInput input) {
        float[] values = new float[STATUS_VALUES];
        boolean[] mask = input.active().clone();
        float[] maximumHealth = input.maximumHealth();
        float[] resourceSpan = input.resourceSpan();
        float[] remaining = input.remainingSeconds();
        float[] elapsed = input.cycleElapsedSeconds();
        float[] cooldown = input.cycleCooldownSeconds();
        float[] damage = input.damagePerCycle();
        float[] healing = input.healingPerCycle();
        int[] resourceIds = input.resourceId();
        float[] resourceDelta = input.resourceDeltaPerCycle();
        float[] speed = input.speedMultiplier();
        for (int slot = 0; slot < STATUS_SLOTS; slot++) {
            if (!mask[slot]) {
                continue;
            }
            int entity = slot / STATUS_CAPACITY;
            int resource = Math.max(0, Math.min(
                RESOURCE_COUNT - 1, resourceIds[slot]));
            float selectedSpan = resourceSpan[entity * RESOURCE_COUNT + resource];
            int at = slot * STATUS_FEATURES;
            values[at] = remaining[slot] / STATUS_DURATION_SCALE_SECONDS;
            values[at + 1] = cooldown[slot] > 0.0f
                ? elapsed[slot] / Math.max(cooldown[slot], 1.0e-6f)
                : 0.0f;
            values[at + 2] = damage[slot] / maximumHealth[entity];
            values[at + 3] = healing[slot] / maximumHealth[entity];
            values[at + 4] = resourceDelta[slot] / Math.max(selectedSpan, 1.0f);
            values[at + 5] = speed[slot] / 2.0f;
            for (int feature = 0; feature < STATUS_FEATURES; feature++) {
                int index = at + feature;
                values[index] = Math.max(-1.0f, Math.min(1.0f, values[index]));
            }
        }
        return new StatusResult(values, mask);
    }

    private static void requireEntityWidth(boolean[] values, String name) {
        if (values.length != ENTITY_COUNT) {
            throw new IllegalArgumentException(name + " width drift");
        }
    }

    private static void requireEntityWidth(float[] values, String name) {
        if (values.length != ENTITY_COUNT) {
            throw new IllegalArgumentException(name + " width drift");
        }
    }

    private static void requireStatusWidth(float[] values, String name) {
        if (values.length != STATUS_SLOTS) {
            throw new IllegalArgumentException(name + " width drift");
        }
    }
}
