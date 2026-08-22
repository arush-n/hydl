package com.hytalerlbridge.worldgen;

import java.util.Map;

/** Metadata and fixed coordinates for one native Region v1 capture. */
public record NativeRegionManifest(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    int coreMinChunkX,
    int coreMinChunkZ,
    Map<String, Object> chunkApi
) {

    public static final String SCHEMA = "hytalerl_native_region_snapshot_v1";
    public static final int VERSION = 1;
    public static final String CAPTURE_MODE = "offline_complete";
    public static final String DYNAMIC_STATE = "static";
    public static final int CORE_CHUNKS_PER_AXIS = 3;
    public static final int HALO_CHUNKS = 1;
    public static final int CAPTURE_CHUNKS_PER_AXIS = 5;
    public static final int CAPTURE_SECTION_COUNT = 250;

    public NativeRegionManifest {
        if (!"0.5.7".equals(serverVersion)) {
            throw new IllegalArgumentException(
                "Region capture is certified only for Hytale 0.5.7"
            );
        }
        worldName = worldName == null ? "" : worldName;
        worldgenProvider = worldgenProvider == null ? "" : worldgenProvider;
        worldgenVersion = worldgenVersion == null ? "" : worldgenVersion;
        Map<String, Object> validatedChunkApi =
            chunkApi == null ? Map.of() : Map.copyOf(chunkApi);
        chunkApi = validatedChunkApi;
        for (
            Map.Entry<String, Object> entry
                : ChunkApiContract.load057().manifest(serverVersion).entrySet()
        ) {
            if (!entry.getValue().equals(validatedChunkApi.get(entry.getKey()))) {
                throw new IllegalArgumentException(
                    "Region manifest chunk API mismatch at " + entry.getKey()
                );
            }
        }
        int minimum = ((Number) validatedChunkApi.get(
            "min_chunk_coordinate"
        )).intValue();
        int maximum = ((Number) validatedChunkApi.get(
            "max_chunk_coordinate"
        )).intValue();
        if (
            (long) coreMinChunkX - HALO_CHUNKS < minimum
                || (long) coreMinChunkX + 3 > maximum
                || (long) coreMinChunkZ - HALO_CHUNKS < minimum
                || (long) coreMinChunkZ + 3 > maximum
        ) {
            throw new IllegalArgumentException(
                "Region capture exceeds native chunk bounds"
            );
        }
    }

    public int captureMinChunkX() {
        return coreMinChunkX - HALO_CHUNKS;
    }

    public int captureMinChunkZ() {
        return coreMinChunkZ - HALO_CHUNKS;
    }
}
