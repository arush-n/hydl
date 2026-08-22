package com.hytalerlbridge.nativebackend.model;

import java.util.Arrays;

/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public final class DoubleArrayKey {
    private final long[] bits;

    public DoubleArrayKey(double[] values) {
        bits = new long[values.length];
        for (int index = 0; index < values.length; index++) {
            bits[index] = Double.doubleToLongBits(values[index]);
        }
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof DoubleArrayKey key
            && Arrays.equals(bits, key.bits);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(bits);
    }
}
