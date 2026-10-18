package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.modules.blockhealth.BlockHealthModule;
import com.hypixel.hytale.server.core.modules.blockhealth.BlockHealthSection;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hytalerlbridge.action.NativeWorldVerbRequest;
import com.hytalerlbridge.nativebackend.model.WorldVerbCell;
import com.hytalerlbridge.nativebackend.support.NativeWorldAccess;
import org.joml.Vector3i;

/** Same-tick native block state used to admit and acknowledge World verbs. */
final class WorldVerbCapture {

    private WorldVerbCapture() {}

    static WorldVerbCell capture(
        World world,
        NativeWorldVerbRequest request
    ) {
        int x = request.targetX();
        int y = request.targetY();
        int z = request.targetZ();
        NativeWorldAccess.Column chunk = NativeWorldAccess.getChunkIfLoaded(
            world,
            ChunkUtil.indexChunkFromBlock(x, z)
        );
        if (chunk == null) return null;
        int localX = ChunkUtil.localCoordinate((long) x);
        int localZ = ChunkUtil.localCoordinate((long) z);
        int runtimeBlockId = NativeWorldAccess.getBlock(world, x, y, z);
        BlockType blockType = runtimeBlockId == 0
            ? null
            : NativeWorldAccess.getBlockType(world, x, y, z);
        String semanticBlockId = blockType == null
            ? BlockType.EMPTY_KEY
            : blockType.getId();
        int rotationIndex = chunk.getRotationIndex(localX, y, localZ);
        int filler = chunk.getFiller(localX, y, localZ);
        int fluidId = chunk.getFluidId(localX, y, localZ);
        int fluidLevel = fluidId == 0
            ? 0
            : Byte.toUnsignedInt(
                chunk.getFluidLevel(localX, y, localZ)
            );
        int supportValue = chunk.getSupportValue(localX, y, localZ);
        double blockHealth = runtimeBlockId == 0
            ? 0.0
            : blockHealth(world, new Vector3i(x, y, z));
        String canonical = String.join(
            "|",
            Integer.toString(x),
            Integer.toString(y),
            Integer.toString(z),
            semanticBlockId,
            Integer.toString(runtimeBlockId),
            Integer.toString(rotationIndex),
            Integer.toString(filler),
            Integer.toString(fluidId),
            Integer.toString(fluidLevel),
            Integer.toString(supportValue),
            Double.toHexString(blockHealth)
        );
        return new WorldVerbCell(
            semanticBlockId,
            runtimeBlockId,
            blockHealth,
            canonical
        );
    }

    static double blockHealth(World world, Vector3i position) {
        if (NativeWorldAccess.getBlock(world, position.x, position.y, position.z) == 0) {
            return 0.0;
        }
        NativeWorldAccess.Column chunk = NativeWorldAccess.getChunkIfLoaded(
            world,
            ChunkUtil.indexChunkFromBlock(position.x, position.z)
        );
        BlockHealthModule module = BlockHealthModule.get();
        // 0.7.2 keeps block health per section; 0.5 kept one BlockHealthChunk per column.
        Ref<ChunkStore> section = chunk == null || module == null
            ? null
            : world.getChunkStore().getChunkSectionReferenceAtBlock(
                position.x,
                position.y,
                position.z
            );
        BlockHealthSection health = section == null || !section.isValid()
            ? null
            : world.getChunkStore().getStore().getComponent(
                section,
                module.getBlockHealthSectionComponentType()
            );
        if (health == null) {
            throw new IllegalStateException(
                "World verb target has no native block-health state"
            );
        }
        // Undamaged blocks read as full health, as the 0.5 BlockHealthChunk did.
        return health.getHealth(position.x, position.y, position.z);
    }
}
