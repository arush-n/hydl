package com.hytalerlbridge.policy.world;

import com.hypixel.hytale.assetstore.AssetRegistry;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.chunk.ChunkColumn;
import com.hypixel.hytale.server.core.universe.world.chunk.WorldChunk;
import com.hypixel.hytale.server.core.universe.world.chunk.section.BlockSection;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.GetChunkFlags;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;

/**
 * The 0.5.x {@code World} chunk and block accessors the agent uses, re-implemented on the 0.7.2
 * chunk store. Same helper as the bridge's {@code nativebackend/support/NativeWorldAccess}, copied
 * because the agent does not compile against the bridge.
 *
 * <p>0.7.2 moved block data onto per-section components and ticking onto the absence of
 * {@code NonTicking}. Reads hop to the world thread when called elsewhere.
 */
public final class NativeWorldAccess {
    private NativeWorldAccess() {}

    /** 0.5 {@code World.getChunk}: the column, loaded and ticking; null if it cannot load. */
    public static Column getChunk(World world, long index) {
        Column column = loadChunkIfInMemory(world, index);
        if (column != null) {
            return column;
        }
        CompletableFuture<Column> future = world.getChunkStore()
            .getChunkReferenceAsync(index, GetChunkFlags.SET_TICKING)
            .thenApplyAsync(reference -> {
                Column loaded = column(world, reference);
                if (loaded != null) {
                    loaded.setTicking(true);
                }
                return loaded;
            }, world);
        return waitForFutureWithoutLock(world, future);
    }

    /** 0.5 {@code World.loadChunkIfInMemory}: the column if in memory, started ticking. */
    public static Column loadChunkIfInMemory(World world, long index) {
        if (!world.isInThread()) {
            return onWorld(world, () -> loadChunkIfInMemory(world, index));
        }
        Column column = column(world, world.getChunkStore().getChunkReference(index));
        if (column != null) {
            column.setTicking(true);
        }
        return column;
    }

    /** 0.5 {@code World.getBlock}: the block id, loading the column; 0 if it cannot load. */
    public static int getBlock(World world, int x, int y, int z) {
        if (!world.isInThread()) {
            return onWorld(world, () -> getBlock(world, x, y, z));
        }
        Column column = getChunk(world, ChunkUtil.indexChunkFromBlock(x, z));
        return column == null ? 0 : column.getBlock(x, y, z);
    }

    /** 0.5 {@code World.getBlockType}: the block type, loading the column; null if it cannot load. */
    public static BlockType getBlockType(World world, int x, int y, int z) {
        if (!world.isInThread()) {
            return onWorld(world, () -> getBlockType(world, x, y, z));
        }
        Column column = getChunk(world, ChunkUtil.indexChunkFromBlock(x, z));
        return column == null ? null : BlockType.getAssetMap().getAsset(column.getBlock(x, y, z));
    }

    /**
     * 0.5 {@code IWorldChunks.waitForFutureWithoutLock}: on the world thread, releases the asset read
     * lock and drains the world task queue until the future completes; elsewhere, joins it.
     */
    public static <T> T waitForFutureWithoutLock(World world, CompletableFuture<T> future) {
        if (!world.isInThread()) {
            return future.join();
        }
        AssetRegistry.ASSET_LOCK.readLock().unlock();
        for (; !future.isDone(); Thread.yield()) {
            AssetRegistry.ASSET_LOCK.readLock().lock();
            try {
                world.consumeTaskQueue();
            } finally {
                AssetRegistry.ASSET_LOCK.readLock().unlock();
            }
        }
        AssetRegistry.ASSET_LOCK.readLock().lock();
        return future.join();
    }

    /** The column for a chunk reference, or null if the reference is missing or no longer valid. */
    public static Column column(World world, Ref<ChunkStore> reference) {
        if (reference == null || !reference.isValid()) {
            return null;
        }
        Store<ChunkStore> store = world.getChunkStore().getStore();
        WorldChunk worldChunk = store.getComponent(reference, WorldChunk.getComponentType());
        ChunkColumn chunkColumn = store.getComponent(reference, ChunkColumn.getComponentType());
        return worldChunk == null || chunkColumn == null
            ? null
            : new Column(world, store, reference, worldChunk, chunkColumn);
    }

    private static <T> T onWorld(World world, Supplier<T> read) {
        return CompletableFuture.supplyAsync(read, world).join();
    }

    /**
     * A chunk column: the 0.7.2 stand-in for a 0.5.x {@code WorldChunk}. Coordinates are world
     * coordinates. Use it within the tick that looked it up.
     */
    public static final class Column {
        private final World world;
        private final Store<ChunkStore> store;
        private final Ref<ChunkStore> reference;
        private final WorldChunk worldChunk;
        private final ChunkColumn chunkColumn;

        private Column(World world, Store<ChunkStore> store, Ref<ChunkStore> reference,
                       WorldChunk worldChunk, ChunkColumn chunkColumn) {
            this.world = world;
            this.store = store;
            this.reference = reference;
            this.worldChunk = worldChunk;
            this.chunkColumn = chunkColumn;
        }

        public WorldChunk worldChunk() {
            return worldChunk;
        }

        /** 0.5 {@code WorldChunk.is(ChunkFlag.TICKING)}. */
        public boolean isTicking() {
            if (!world.isInThread()) {
                return onWorld(world, this::isTicking);
            }
            return reference.isValid()
                && !store.getArchetype(reference).contains(ChunkStore.REGISTRY.getNonTickingComponentType());
        }

        /**
         * 0.5 {@code WorldChunk.setFlag(ChunkFlag.TICKING, ticking)}. 0.7.2 has no ticking flag, so the
         * column and each of its sections carry or lack the {@code NonTicking} marker.
         */
        public void setTicking(boolean ticking) {
            if (!world.isInThread()) {
                onWorld(world, () -> {
                    setTicking(ticking);
                    return null;
                });
                return;
            }
            if (isTicking() != ticking) {
                world.debugAssertInTickingThread();
                worldChunk.resetKeepAlive();
                setNonTicking(reference, !ticking);
            }
            for (int index = 0; index < ChunkUtil.HEIGHT_SECTIONS; index++) {
                Ref<ChunkStore> section = chunkColumn.getSection(index);
                if (section != null && section.isValid()) {
                    setNonTicking(section, !ticking);
                }
            }
        }

        private void setNonTicking(Ref<ChunkStore> entity, boolean nonTicking) {
            if (nonTicking) {
                store.ensureComponent(entity, ChunkStore.REGISTRY.getNonTickingComponentType());
            } else {
                store.tryRemoveComponent(entity, ChunkStore.REGISTRY.getNonTickingComponentType());
            }
        }

        /** 0.5 {@code WorldChunk.getBlock}: 0 outside 0..319. */
        public int getBlock(int x, int y, int z) {
            if (y < 0 || y >= ChunkUtil.HEIGHT) {
                return 0;
            }
            if (!world.isInThread()) {
                return onWorld(world, () -> getBlock(x, y, z));
            }
            Ref<ChunkStore> section = chunkColumn.getSection(ChunkUtil.indexSection(y));
            if (section == null) {
                throw new IllegalStateException("section of chunk " + worldChunk.getX() + "," + worldChunk.getZ()
                    + " is not loaded");
            }
            BlockSection blocks = store.getComponent(section, BlockSection.getComponentType());
            if (blocks == null) {
                throw new IllegalStateException("section has no block data");
            }
            return blocks.get(x, y, z);
        }
    }
}
