package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.math.util.ChunkUtil;
import com.hytalerlbridge.worldgen.NativeRegionManifest;
import java.util.ArrayList;
import java.util.List;

/** Chunk coverage needed to calculate light for a complete Region capture. */
public final class RegionLightCoverage {

    private static final int LIGHT_HALO_CHUNKS = 1;

    private RegionLightCoverage() {}

    public static List<Long> indices(NativeRegionManifest manifest) {
        int minimum = ((Number) manifest.chunkApi().get(
            "min_chunk_coordinate"
        )).intValue();
        int maximum = ((Number) manifest.chunkApi().get(
            "max_chunk_coordinate"
        )).intValue();
        long startX = (long) manifest.captureMinChunkX() - LIGHT_HALO_CHUNKS;
        long startZ = (long) manifest.captureMinChunkZ() - LIGHT_HALO_CHUNKS;
        int chunksPerAxis = NativeRegionManifest.CAPTURE_CHUNKS_PER_AXIS
            + 2 * LIGHT_HALO_CHUNKS;
        long endX = startX + chunksPerAxis - 1L;
        long endZ = startZ + chunksPerAxis - 1L;
        if (
            startX < minimum
                || startZ < minimum
                || endX > maximum
                || endZ > maximum
        ) {
            throw new IllegalArgumentException(
                "Region lighting halo exceeds native chunk bounds"
            );
        }
        List<Long> result = new ArrayList<>(chunksPerAxis * chunksPerAxis);
        for (int dx = 0; dx < chunksPerAxis; dx++) {
            for (int dz = 0; dz < chunksPerAxis; dz++) {
                result.add(
                    ChunkUtil.indexChunk(
                        Math.toIntExact(startX + dx),
                        Math.toIntExact(startZ + dz)
                    )
                );
            }
        }
        return List.copyOf(result);
    }
}
