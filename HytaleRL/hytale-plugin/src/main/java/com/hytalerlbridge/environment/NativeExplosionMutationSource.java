package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeExplosionMutationProbe;

/**
 * Optional native-only block-damaging explosion surface.
 *
 * <p>Implementations execute and capture atomically on the World thread. V1
 * uses the server-owned synthetic source declared by
 * {@link NativeExplosionMutationProbe#SYNTHETIC_CONFIG_SOURCE_ID}; callers do
 * not provide or attest config provenance. An incomplete frame emitted after
 * execution must be empty and require resync. A preflight rejection is empty
 * but does not require resync.</p>
 */
public interface NativeExplosionMutationSource {

    NativeExplosionMutationProbe captureExplosionMutation(
        String worldEpoch,
        String fixtureKind,
        int cellCapacity,
        int dropCapacity
    );
}
