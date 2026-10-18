package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeExplosionCandidateProbe;

/** Optional native-only explosion entity-admission surface. */
public interface NativeExplosionProbeSource {

    NativeExplosionCandidateProbe captureExplosionCandidateProbe(
        double[] origin,
        int blockDamageRadius,
        float entityDamageRadius,
        boolean ignoreControlledActor,
        int capacity
    );
}
