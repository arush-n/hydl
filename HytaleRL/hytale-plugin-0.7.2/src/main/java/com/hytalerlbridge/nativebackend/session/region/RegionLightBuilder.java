package com.hytalerlbridge.nativebackend.session.region;

import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hytalerlbridge.nativebackend.support.NativeWorldAccess;
import com.hytalerlbridge.nativebackend.support.RegionLightCapture;
import com.hytalerlbridge.worldgen.NativeRegionLightSection;
import java.util.Set;
import org.joml.Vector3i;

/**
 * Region light section builder for the light capture path.
 */
public final class RegionLightBuilder {

    private RegionLightBuilder() {}

    public static NativeRegionLightSection buildRegionLightSection(
        Set<Long> loadedChunkIndices,
        World world,
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        long chunkIndex = ChunkUtil.indexChunk(chunkX, chunkZ);
        if (!loadedChunkIndices.contains(chunkIndex)) {
            throw new IllegalStateException(
                "Region light chunk was not pinned before capture"
            );
        }
        NativeWorldAccess.Column chunk = NativeWorldAccess.getChunkIfLoaded(world, chunkIndex);
        if (chunk == null) {
            throw new IllegalStateException(
                "Region light chunk is not loaded"
            );
        }
        NativeRegionLightSection result = RegionLightCapture.capture(
            chunk,
            chunkX,
            chunkZ,
            sectionY
        );
        if (!result.available()) {
            world.getChunkLighting().addToQueue(
                new Vector3i(chunkX, sectionY, chunkZ)
            );
        }
        return result;
    }
}
