package com.hytalerlbridge.observation;

import com.hytalerlbridge.combat.CombatRuleset;

/**
 * Shared MessagePack observation encodings.
 *
 * <p>The wire scale is loaded from the same versioned resource as Java and
 * JAX combat mechanics. Keeping the conversion here prevents simulator tasks,
 * the native backend, and Python decoding from drifting through copied
 * literals.</p>
 */
public final class ObservationEncoding {

    public static final double NEARBY_ENTITY_FIXED_POINT_SCALE =
        CombatRuleset.defaultRules()
            .observationNormalization()
            .wireFixedPointScale();

    private ObservationEncoding() {}

    public static int encodeNearbyEntityScalar(double value) {
        long encoded = Math.round(value * NEARBY_ENTITY_FIXED_POINT_SCALE);
        if (encoded <= Integer.MIN_VALUE) return Integer.MIN_VALUE;
        if (encoded >= Integer.MAX_VALUE) return Integer.MAX_VALUE;
        return (int) encoded;
    }
}
