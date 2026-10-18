package com.hytalerlbridge.policy.perception.projection;

/** Dense learner-v3 inventory tokens from actor-safe semantic stack rows. */
public final class InventoryProjection {

    public static final int CONTAINER_COUNT = 6;
    public static final int TOKEN_CAPACITY = 76;
    public static final int TOKEN_FEATURE_COUNT = 7;
    public static final int TOKEN_VALUE_COUNT =
        TOKEN_CAPACITY * TOKEN_FEATURE_COUNT;

    /** Storage, armor, hotbar, utility, tools, and variable backpack. */
    public static final int[] DEFAULT_CONTAINER_CAPACITIES = {
        36, 4, 9, 4, 23, 0,
    };

    public record Input(
        boolean available,
        boolean[] containerAvailable,
        int[] containerCapacity,
        boolean[] tokenMask,
        int[] containerId,
        int[] containerSlot,
        int[] itemId,
        int[] quantity,
        float[] durabilityFraction,
        boolean[] active
    ) {
        public Input {
            require(containerAvailable.length, CONTAINER_COUNT,
                "container available");
            require(containerCapacity.length, CONTAINER_COUNT,
                "container capacity");
            require(tokenMask.length, TOKEN_CAPACITY, "token mask");
            require(containerId.length, TOKEN_CAPACITY, "container id");
            require(containerSlot.length, TOKEN_CAPACITY, "container slot");
            require(itemId.length, TOKEN_CAPACITY, "item id");
            require(quantity.length, TOKEN_CAPACITY, "quantity");
            require(durabilityFraction.length, TOKEN_CAPACITY,
                "durability fraction");
            require(active.length, TOKEN_CAPACITY, "active");
            for (int capacity : containerCapacity) {
                if (capacity < 0) {
                    throw new IllegalArgumentException(
                        "container capacity cannot be negative");
                }
            }
            containerAvailable = containerAvailable.clone();
            containerCapacity = containerCapacity.clone();
            tokenMask = tokenMask.clone();
            containerId = containerId.clone();
            containerSlot = containerSlot.clone();
            itemId = itemId.clone();
            quantity = quantity.clone();
            durabilityFraction = durabilityFraction.clone();
            active = active.clone();
        }
    }

    public record Result(
        boolean available,
        float[] containerValues,
        boolean[] containerMask,
        float[] tokenValues,
        boolean[] tokenMask
    ) {
        public Result {
            require(containerValues.length, CONTAINER_COUNT,
                "inventory container values");
            require(containerMask.length, CONTAINER_COUNT,
                "inventory container mask");
            require(tokenValues.length, TOKEN_VALUE_COUNT,
                "inventory token values");
            require(tokenMask.length, TOKEN_CAPACITY,
                "inventory token mask");
            containerValues = containerValues.clone();
            containerMask = containerMask.clone();
            tokenValues = tokenValues.clone();
            tokenMask = tokenMask.clone();
        }

        @Override
        public float[] containerValues() {
            return containerValues.clone();
        }

        @Override
        public boolean[] containerMask() {
            return containerMask.clone();
        }

        @Override
        public float[] tokenValues() {
            return tokenValues.clone();
        }

        @Override
        public boolean[] tokenMask() {
            return tokenMask.clone();
        }
    }

    private InventoryProjection() {
    }

    /** Match {@code encode_inventory_policy_tokens} in float32 order. */
    public static Result project(Input input) {
        boolean[] containerAvailable = input.containerAvailable();
        int[] capacities = input.containerCapacity();
        boolean[] containerMask = new boolean[CONTAINER_COUNT];
        boolean available = input.available();
        for (int container = 0; container < CONTAINER_COUNT; container++) {
            containerMask[container] = containerAvailable[container] && available;
        }
        boolean capacityValid = true;
        for (int container = 0; container < CONTAINER_COUNT; container++) {
            int configured = DEFAULT_CONTAINER_CAPACITIES[container];
            capacityValid &= !containerMask[container]
                || configured <= 0
                || capacities[container] <= configured;
        }
        available &= capacityValid;

        float[] containerValues = new float[CONTAINER_COUNT];
        for (int container = 0; container < CONTAINER_COUNT; container++) {
            containerMask[container] &= available;
            if (!containerMask[container]) {
                continue;
            }
            int configured = DEFAULT_CONTAINER_CAPACITIES[container];
            containerValues[container] = configured > 0
                ? (float) capacities[container] / Math.max((float) configured, 1.0f)
                : (float) capacities[container] / (float) Integer.MAX_VALUE;
        }

        boolean[] sourceTokenMask = input.tokenMask();
        int[] containerIds = input.containerId();
        int[] slots = input.containerSlot();
        int[] itemIds = input.itemId();
        int[] quantities = input.quantity();
        float[] durability = input.durabilityFraction();
        boolean[] active = input.active();
        boolean[] tokenMask = new boolean[TOKEN_CAPACITY];
        float[] tokenValues = new float[TOKEN_VALUE_COUNT];
        for (int token = 0; token < TOKEN_CAPACITY; token++) {
            tokenMask[token] = sourceTokenMask[token] && available;
            if (!tokenMask[token]) {
                continue;
            }
            int container = Math.max(0, Math.min(
                CONTAINER_COUNT - 1, containerIds[token]));
            int capacity = capacities[container];
            int at = token * TOKEN_FEATURE_COUNT;
            tokenValues[at] = clip01(
                (float) containerIds[token] / (float) (CONTAINER_COUNT - 1));
            tokenValues[at + 1] = clip01(
                ((float) slots[token] + 1.0f) / ((float) capacity + 1.0f));
            tokenValues[at + 2] = clip01(
                (float) (itemIds[token] & 0xFFFF) / (float) 0xFFFF);
            tokenValues[at + 3] = clip01(
                (float) ((itemIds[token] >> 16) & 0x7FFF)
                    / (float) 0x7FFF);
            tokenValues[at + 4] = clip01(
                (float) (Math.log((float) quantities[token] + 1.0f)
                    / Math.log(2.0)) / 31.0f);
            tokenValues[at + 5] = clip01(durability[token]);
            tokenValues[at + 6] = active[token] ? 1.0f : 0.0f;
        }
        return new Result(
            available, containerValues, containerMask, tokenValues, tokenMask);
    }

    private static float clip01(float value) {
        return Math.max(0.0f, Math.min(1.0f, value));
    }

    private static void require(int actual, int expected, String name) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                name + " width drift: " + actual + " != " + expected);
        }
    }
}
