package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeExplosionDynamicsProbe;

/** Optional native-only terminal fixture for entity explosion damage. */
public interface NativeExplosionDynamicsSource {

    NativeExplosionDynamicsProbe captureExplosionDynamics(
        String worldEpoch,
        String fixtureKind,
        int capacity
    );
}
