package com.hytalerlbridge.policy.perception.projection;

import java.util.Arrays;

/** Pure projection for geometry-aligned native light evidence. */
public final class LightProjection {

    public static final int TOKEN_CAPACITY = 44;
    public static final int CHANNEL_COUNT = 3;
    public static final int FEATURE_SIZE = 10;
    public static final int VALUE_COUNT = TOKEN_CAPACITY * FEATURE_SIZE;

    /** Raw uint8-equivalent World evidence plus the geometry row it accompanies. */
    public record Input(
        boolean sourceAvailable,
        boolean[] sourceTokenMask,
        boolean[] lightValid,
        int[] skyLight,
        int[] blockLightRgb,
        int[] tintRgb,
        boolean[] geometryTokenMask,
        boolean geometryAvailable,
        boolean actorValid
    ) {
        public Input {
            require(sourceTokenMask, TOKEN_CAPACITY, "light token mask");
            require(lightValid, TOKEN_CAPACITY * CHANNEL_COUNT, "light validity");
            require(skyLight, TOKEN_CAPACITY, "sky light");
            require(blockLightRgb, TOKEN_CAPACITY * 3, "block-light RGB");
            require(tintRgb, TOKEN_CAPACITY * 3, "tint RGB");
            require(geometryTokenMask, TOKEN_CAPACITY, "geometry token mask");
            requireUint8(skyLight, "sky light");
            requireUint8(blockLightRgb, "block-light RGB");
            requireUint8(tintRgb, "tint RGB");
            sourceTokenMask = sourceTokenMask.clone();
            lightValid = lightValid.clone();
            skyLight = skyLight.clone();
            blockLightRgb = blockLightRgb.clone();
            tintRgb = tintRgb.clone();
            geometryTokenMask = geometryTokenMask.clone();
        }

        @Override public boolean[] sourceTokenMask() {
            return sourceTokenMask.clone();
        }
        @Override public boolean[] lightValid() { return lightValid.clone(); }
        @Override public int[] skyLight() { return skyLight.clone(); }
        @Override public int[] blockLightRgb() { return blockLightRgb.clone(); }
        @Override public int[] tintRgb() { return tintRgb.clone(); }
        @Override public boolean[] geometryTokenMask() {
            return geometryTokenMask.clone();
        }

        public static Input unavailable(
            boolean[] geometryTokenMask, boolean geometryAvailable,
            boolean actorValid
        ) {
            return new Input(
                false,
                new boolean[TOKEN_CAPACITY],
                new boolean[TOKEN_CAPACITY * CHANNEL_COUNT],
                new int[TOKEN_CAPACITY],
                new int[TOKEN_CAPACITY * 3],
                new int[TOKEN_CAPACITY * 3],
                geometryTokenMask,
                geometryAvailable,
                actorValid
            );
        }
    }

    /** Policy values and the aligned token mask (the latter reuses geometry). */
    public record Result(boolean available, float[] values, boolean[] tokenMask) {
        public Result {
            require(values, VALUE_COUNT, "light values");
            require(tokenMask, TOKEN_CAPACITY, "light token mask");
            values = values.clone();
            tokenMask = tokenMask.clone();
        }

        @Override public float[] values() { return values.clone(); }
        @Override public boolean[] tokenMask() { return tokenMask.clone(); }
    }

    private LightProjection() {
    }

    /**
     * Match {@code encode_actor_light_policy_tokens} followed by
     * {@code align_actor_light_policy_tokens}.
     *
     * <p>Sky and block-light values above 15 invalidate the complete row only
     * when their channel is valid on an active source token. Tint remains the
     * full uint8 range. A source mask that differs from geometry also closes
     * the complete row; partial positional alignment is never accepted.
     */
    public static Result project(Input input) {
        // Input is immutable-by-copy at its public boundary. As the enclosing
        // projector we can read the private record storage directly and avoid
        // six defensive-array allocations on every NPC tick.
        boolean[] sourceMask = input.sourceTokenMask;
        boolean[] geometryMask = input.geometryTokenMask;
        boolean[] valid = input.lightValid;
        int[] sky = input.skyLight;
        int[] block = input.blockLightRgb;
        int[] tint = input.tintRgb;

        boolean malformed = false;
        if (input.sourceAvailable()) {
            for (int token = 0; token < TOKEN_CAPACITY; token++) {
                if (!sourceMask[token]) {
                    continue;
                }
                int channel = token * CHANNEL_COUNT;
                int rgb = token * 3;
                malformed |= valid[channel] && sky[token] > 15;
                malformed |= valid[channel + 1]
                    && (block[rgb] > 15
                        || block[rgb + 1] > 15
                        || block[rgb + 2] > 15);
            }
        }

        boolean available = input.sourceAvailable()
            && !malformed
            && input.geometryAvailable()
            && input.actorValid()
            && Arrays.equals(sourceMask, geometryMask);
        boolean[] tokenMask = new boolean[TOKEN_CAPACITY];
        float[] values = new float[VALUE_COUNT];
        if (!available) {
            return new Result(false, values, tokenMask);
        }

        for (int token = 0; token < TOKEN_CAPACITY; token++) {
            boolean active = sourceMask[token];
            tokenMask[token] = active;
            if (!active) {
                continue;
            }
            int channel = token * CHANNEL_COUNT;
            int rgb = token * 3;
            int out = token * FEATURE_SIZE;
            boolean skyValid = valid[channel];
            boolean blockValid = valid[channel + 1];
            boolean tintValid = valid[channel + 2];
            values[out] = skyValid ? 1.0f : 0.0f;
            values[out + 1] = blockValid ? 1.0f : 0.0f;
            values[out + 2] = tintValid ? 1.0f : 0.0f;
            values[out + 3] = skyValid ? sky[token] / 15.0f : 0.0f;
            for (int component = 0; component < 3; component++) {
                values[out + 4 + component] = blockValid
                    ? block[rgb + component] / 15.0f : 0.0f;
                values[out + 7 + component] = tintValid
                    ? tint[rgb + component] / 255.0f : 0.0f;
            }
        }
        return new Result(true, values, tokenMask);
    }

    private static void require(boolean[] values, int expected, String name) {
        if (values == null || values.length != expected) {
            throw new IllegalArgumentException(
                name + " width drift: "
                    + (values == null ? "null" : values.length)
                    + " != " + expected);
        }
    }

    private static void require(int[] values, int expected, String name) {
        if (values == null || values.length != expected) {
            throw new IllegalArgumentException(
                name + " width drift: "
                    + (values == null ? "null" : values.length)
                    + " != " + expected);
        }
    }

    private static void require(float[] values, int expected, String name) {
        if (values == null || values.length != expected) {
            throw new IllegalArgumentException(
                name + " width drift: "
                    + (values == null ? "null" : values.length)
                    + " != " + expected);
        }
    }

    private static void requireUint8(int[] values, String name) {
        for (int index = 0; index < values.length; index++) {
            if (values[index] < 0 || values[index] > 255) {
                throw new IllegalArgumentException(
                    name + "[" + index + "] is outside uint8: " + values[index]);
            }
        }
    }
}
