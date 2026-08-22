package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeDropProgramEvidence;

/** Native-only empirical block-drop evidence capability. */
public interface NativeDropProgramSource {

    NativeDropProgramEvidence captureDropProgramEvidence(
        String blockAssetId,
        String route,
        int sampleCount
    );
}
