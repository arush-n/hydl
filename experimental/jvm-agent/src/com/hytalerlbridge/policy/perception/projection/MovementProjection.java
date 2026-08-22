package com.hytalerlbridge.policy.perception.projection;

/** Decodes the native 23-bit {@code MovementStates} evidence row. */
public final class MovementProjection {

    public static final int SIZE = 23;
    public static final int ALL_BITS = (1 << SIZE) - 1;

    /** Values and per-field availability in learner contract order. */
    public record Result(float[] values, boolean[] mask, boolean invalid) {
        public Result {
            if (values.length != SIZE || mask.length != SIZE) {
                throw new IllegalArgumentException("movement projection width drift");
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

    private MovementProjection() {
    }

    /**
     * Decode the bridge's row-level availability plus native protocol bitset.
     *
     * <p>This matches {@code native_movement_state_evidence}: a missing row is
     * canonical only when its bitset is zero, and an out-of-range bit makes the
     * entire row invalid rather than silently aliasing another feature.
     */
    public static Result fromNativeBits(int bits, boolean available) {
        boolean inRange = bits >= 0 && bits <= ALL_BITS;
        boolean canonicalMissing = available || bits == 0;
        if (!inRange || !canonicalMissing) {
            return empty(true);
        }
        return fromBitMasks(bits, available ? ALL_BITS : 0);
    }

    /**
     * Decode feature-level value and availability masks.
     *
     * <p>The JAX NPC Walk model has feature-level availability while the live
     * bridge has row-level availability. Keeping this lower-level form lets the
     * same projection certify both without treating unavailable false as an
     * observed false.
     */
    public static Result fromBitMasks(int valueBits, int availableBits) {
        boolean inRange = valueBits >= 0 && valueBits <= ALL_BITS
            && availableBits >= 0 && availableBits <= ALL_BITS;
        boolean valuesHaveEvidence = (valueBits & ~availableBits) == 0;
        if (!inRange || !valuesHaveEvidence) {
            return empty(true);
        }
        float[] values = new float[SIZE];
        boolean[] mask = new boolean[SIZE];
        for (int index = 0; index < SIZE; index++) {
            int bit = 1 << index;
            mask[index] = (availableBits & bit) != 0;
            values[index] = mask[index] && (valueBits & bit) != 0 ? 1.0f : 0.0f;
        }
        return new Result(values, mask, false);
    }

    private static Result empty(boolean invalid) {
        return new Result(new float[SIZE], new boolean[SIZE], invalid);
    }
}
