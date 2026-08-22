package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeLineOfSightEvidence;

/** Native-only dual-LOS and PositionCache timing evidence capability. */
public interface NativeLineOfSightSource {

    NativeLineOfSightEvidence captureLineOfSightEvidence();
}
