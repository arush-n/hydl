package com.hytalerlbridge.nativebackend.model;

import com.hypixel.hytale.math.util.ChunkUtil;
import com.hytalerlbridge.geometry.GeometryContract;
import com.hytalerlbridge.worldgen.NativeRegionManifest;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public final class GeometryChunkCoverage {

    public static final int CHUNK_SIZE = ChunkUtil.SIZE;
    private static final int HALO_RADIUS =
        (GeometryContract.RADIUS + CHUNK_SIZE - 1) / CHUNK_SIZE;
    public static final int MAX_PINNED_CHUNKS =
        NativeRegionManifest.CAPTURE_CHUNKS_PER_AXIS
            * NativeRegionManifest.CAPTURE_CHUNKS_PER_AXIS;

    public GeometryChunkCoverage() {}

    public static List<Long> indices(double blockX, double blockZ) {
        if (!Double.isFinite(blockX) || !Double.isFinite(blockZ)) {
            throw new IllegalArgumentException(
                "Geometry chunk-halo position must be finite"
            );
        }
        int centerX = ChunkUtil.chunkCoordinate(blockX);
        int centerZ = ChunkUtil.chunkCoordinate(blockZ);
        List<Long> indices = new ArrayList<>();
        for (int dx = -HALO_RADIUS; dx <= HALO_RADIUS; dx++) {
            for (int dz = -HALO_RADIUS; dz <= HALO_RADIUS; dz++) {
                indices.add(
                    ChunkUtil.indexChunk(centerX + dx, centerZ + dz)
                );
            }
        }
        return List.copyOf(indices);
    }

    public static void requirePinCapacity(
        Set<Long> loaded,
        List<Long> requested
    ) {
        Set<Long> combined = new HashSet<>(loaded);
        combined.addAll(requested);
        if (combined.size() > MAX_PINNED_CHUNKS) {
            throw new IllegalStateException(
                "Geometry chunk pin capacity exceeded: "
                    + combined.size()
                    + " > "
                    + MAX_PINNED_CHUNKS
            );
        }
    }
}
