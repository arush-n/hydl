package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeRegionManifest;
import com.hytalerlbridge.worldgen.NativeRegionBlockSemanticSection;
import com.hytalerlbridge.worldgen.NativeRegionFluidSemanticSection;
import com.hytalerlbridge.worldgen.NativeRegionLightSection;
import com.hytalerlbridge.worldgen.NativeRegionSection;

/** Optional capability exposed only by a native Hytale environment session. */
public interface NativeRegionSource {

    NativeRegionManifest regionManifest();

    NativeRegionManifest regionManifest(
        int coreMinChunkX,
        int coreMinChunkZ
    );

    NativeRegionSection captureRegionSection(
        int chunkX,
        int chunkZ,
        int sectionY
    );

    NativeRegionLightSection captureRegionLightSection(
        int chunkX,
        int chunkZ,
        int sectionY
    );

    NativeRegionBlockSemanticSection captureRegionBlockSemanticSection(
        int chunkX,
        int chunkZ,
        int sectionY
    );

    NativeRegionFluidSemanticSection captureRegionFluidSemanticSection(
        int chunkX,
        int chunkZ,
        int sectionY
    );
}
