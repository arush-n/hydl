package com.hytalerlbridge.nativebackend.session;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.modules.blockhealth.BlockHealthModule;
import com.hypixel.hytale.server.core.modules.blockhealth.BlockHealthSection;
import com.hypixel.hytale.server.core.modules.time.WorldTimeResource;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.nativebackend.session.region.RegionGeometryCapture;
import com.hytalerlbridge.nativebackend.support.NativeCellSemanticsCache;
import com.hytalerlbridge.nativebackend.support.NativeWorldAccess;
import com.hytalerlbridge.worldgen.NativeExplosionMutationProbe;
import com.hytalerlbridge.worldgen.NativeMutableBlockCells;
import com.hytalerlbridge.worldgen.NativeMutableBlockEvidence;
import com.hytalerlbridge.worldgen.NativePerceptionChannels;
import java.util.ArrayList;
import java.util.List;
import org.joml.Vector3i;

/**
 * Mutable block evidence, mutable block cell and explosion mutation cell builders, with the
 * block health lookup they share and the controlled block setup used by the evidence builder.
 */
public final class MutableBlockBuilders {

    private MutableBlockBuilders() {}

    public static NativeMutableBlockEvidence buildMutableBlockEvidence(
        String NATIVE_SERVER_VERSION,
        NativeCellSemanticsCache nativeCellSemanticsCache,
        long seed,
        double spawnX,
        double spawnY,
        double spawnZ,
        World world,
        String worldName,
        String worldgenProvider,
        String worldgenVersion
    ) {
        int x = (int) Math.floor(spawnX) + 8;
        int y = (int) Math.floor(spawnY) + 1;
        int z = (int) Math.floor(spawnZ) - 8;
        Vector3i position = new Vector3i(x, y, z);
        int blockIndex = ChunkUtil.indexBlock(x, y, z);
        NativeWorldAccess.Column chunk = NativeWorldAccess.getChunkIfLoaded(
            world,
            ChunkUtil.indexChunkFromBlock(x, z)
        );
        if (chunk == null) {
            throw new IllegalStateException(
                "Mutable block evidence fixture chunk is not loaded"
            );
        }
        if (NativeWorldAccess.getBlock(world, x, y, z) != 0 || chunk.getFluidId(
            ChunkUtil.localCoordinate((long) x),
            y,
            ChunkUtil.localCoordinate((long) z)
        ) != 0) {
            throw new IllegalStateException(
                "Mutable block evidence fixture cell is not initially empty"
            );
        }

        BlockHealthModule module = BlockHealthModule.get();
        Store<ChunkStore> chunkStore = world.getChunkStore().getStore();
        BlockHealthSection health = module == null
            ? null
            : blockHealthSection(chunkStore, chunk, y);
        if (health == null) {
            throw new IllegalStateException(
                "Mutable block evidence chunk has no BlockHealthSection"
            );
        }
        Store<EntityStore> entityStore = world.getEntityStore().getStore();
        WorldTimeResource time = entityStore.getResource(
            WorldTimeResource.getResourceType()
        );
        if (time == null) {
            throw new IllegalStateException(
                "Mutable block evidence world has no game-time resource"
            );
        }

        List<NativeMutableBlockEvidence.Row> rows = new ArrayList<>(5);
        try {
            health.clearHealth(blockIndex);
            setControlledBlock(world, position, "Rock_Stone", true);
            rows.add(RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "initial",
                position,
                chunk,
                health,
                time.getGameTime()
            ));

            health.damage(x, y, z, 0.4f, time.getGameTime());
            rows.add(RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "damaged",
                position,
                chunk,
                health,
                time.getGameTime()
            ));

            // 0.5 repairBlock dropped the entry unless repaired health exceeded 1.0
            // (bytecode: dcmpl/ifle), so 0.6 + 0.1 read back as undamaged.
            // 0.7.2 has no repair call; clearing reproduces that result.
            health.clearHealth(blockIndex);
            rows.add(RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "repaired",
                position,
                chunk,
                health,
                time.getGameTime()
            ));

            setControlledBlock(world, position, BlockType.EMPTY_KEY, false);
            health.clearHealth(blockIndex);
            rows.add(RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "removed",
                position,
                chunk,
                health,
                time.getGameTime()
            ));

            setControlledBlock(world, position, "Rock_Stone", true);
            rows.add(RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "placed",
                position,
                chunk,
                health,
                time.getGameTime()
            ));
        } finally {
            health.clearHealth(blockIndex);
            NativeWorldAccess.setBlock(world, x, y, z, BlockType.EMPTY_KEY);
        }
        return new NativeMutableBlockEvidence(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            new int[] {x, y, z},
            rows
        );
    }

    public static NativeMutableBlockCells buildMutableBlockCells(
        String NATIVE_SERVER_VERSION,
        NativeCellSemanticsCache nativeCellSemanticsCache,
        long seed,
        World world,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        int[] positions
    ) {
        BlockHealthModule module = BlockHealthModule.get();
        if (module == null) {
            throw new IllegalStateException(
                "Native mutable block capture has no BlockHealthModule"
            );
        }
        Store<ChunkStore> chunkStore = world.getChunkStore().getStore();
        Store<EntityStore> entityStore = world.getEntityStore().getStore();
        WorldTimeResource time = entityStore.getResource(
            WorldTimeResource.getResourceType()
        );
        if (time == null) {
            throw new IllegalStateException(
                "Native mutable block capture has no game-time resource"
            );
        }
        List<NativeMutableBlockCells.Cell> cells = new ArrayList<>(
            positions.length / 3
        );
        for (int offset = 0; offset < positions.length; offset += 3) {
            int x = positions[offset];
            int y = positions[offset + 1];
            int z = positions[offset + 2];
            int[] position = {x, y, z};
            if (y < 0 || y >= NativePerceptionChannels.WORLD_HEIGHT) {
                cells.add(new NativeMutableBlockCells.Cell(
                    position,
                    false,
                    null
                ));
                continue;
            }
            NativeWorldAccess.Column chunk = NativeWorldAccess.getChunkIfLoaded(
                world,
                ChunkUtil.indexChunkFromBlock(x, z)
            );
            if (chunk == null) {
                cells.add(new NativeMutableBlockCells.Cell(
                    position,
                    false,
                    null
                ));
                continue;
            }
            BlockHealthSection health = blockHealthSection(chunkStore, chunk, y);
            if (health == null) {
                throw new IllegalStateException(
                    "Loaded mutable block cell has no BlockHealthSection"
                );
            }
            cells.add(new NativeMutableBlockCells.Cell(
                position,
                true,
                RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                    "snapshot",
                    new Vector3i(x, y, z),
                    chunk,
                    health,
                    time.getGameTime()
                )
            ));
        }
        return new NativeMutableBlockCells(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            cells
        );
    }

    private static void setControlledBlock(
        World world,
        Vector3i position,
        String blockAssetId,
        boolean expectedPresent
    ) {
        NativeWorldAccess.setBlock(
            world,
            position.x,
            position.y,
            position.z,
            blockAssetId
        );
        boolean present = NativeWorldAccess.getBlock(
            world,
            position.x,
            position.y,
            position.z
        ) != 0;
        if (present != expectedPresent) {
            throw new IllegalStateException(
                "Mutable block fixture transition was rejected: "
                    + blockAssetId
            );
        }
    }

    /** 0.7.2 keeps block health per section; 0.5 kept one BlockHealthChunk per column. */
    private static BlockHealthSection blockHealthSection(
        Store<ChunkStore> chunkStore,
        NativeWorldAccess.Column chunk,
        int y
    ) {
        Ref<ChunkStore> section = chunk.getSectionReference(ChunkUtil.chunkCoordinate(y));
        return section == null
            ? null
            : chunkStore.getComponent(section, BlockHealthSection.getComponentType());
    }

    public static NativeExplosionMutationProbe.CellState explosionMutationCellState(
        NativeCellSemanticsCache nativeCellSemanticsCache,
        World world,
        int x,
        int y,
        int z,
        BlockHealthModule healthModule,
        Store<ChunkStore> chunkStore,
        WorldTimeResource time
    ) {
        NativeWorldAccess.Column chunk = NativeWorldAccess.getChunkIfLoaded(
            world,
            ChunkUtil.indexChunkFromBlock(x, z)
        );
        if (chunk == null) {
            throw new IllegalStateException(
                "Explosion mutation scan chunk unloaded during capture"
            );
        }
        BlockHealthSection health = explosionMutationBlockHealth(world,
            x,
            y,
            z,
            healthModule,
            chunkStore
        );
        return NativeExplosionMutationProbe.CellState.fromMutableRow(
            RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "snapshot",
                new Vector3i(x, y, z),
                chunk,
                health,
                time.getGameTime()
            )
        );
    }

    public static BlockHealthSection explosionMutationBlockHealth(
        World world,
        int x,
        int y,
        int z,
        BlockHealthModule healthModule,
        Store<ChunkStore> chunkStore
    ) {
        if (y < 0 || y >= NativePerceptionChannels.WORLD_HEIGHT) {
            throw new IllegalArgumentException(
                "Explosion mutation scan escaped the native Y domain"
            );
        }
        NativeWorldAccess.Column chunk = NativeWorldAccess.getChunkIfLoaded(
            world,
            ChunkUtil.indexChunkFromBlock(x, z)
        );
        if (chunk == null) {
            throw new IllegalStateException(
                "Explosion mutation health chunk unloaded during capture"
            );
        }
        BlockHealthSection health = blockHealthSection(chunkStore, chunk, y);
        if (health == null) {
            throw new IllegalStateException(
                "Explosion mutation loaded chunk has no BlockHealthSection"
            );
        }
        return health;
    }
}
