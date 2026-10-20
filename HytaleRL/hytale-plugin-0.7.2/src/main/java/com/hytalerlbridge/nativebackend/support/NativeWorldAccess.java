package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.assetstore.AssetRegistry;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.protocol.BlockPosition;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.blocktype.component.BlockPhysics;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.chunk.BlockChunk;
import com.hypixel.hytale.server.core.universe.world.chunk.BlockOperations;
import com.hypixel.hytale.server.core.universe.world.chunk.ChunkColumn;
import com.hypixel.hytale.server.core.universe.world.chunk.WorldChunk;
import com.hypixel.hytale.server.core.universe.world.chunk.heightmap.HeightmapColumn;
import com.hypixel.hytale.server.core.universe.world.chunk.section.BlockSection;
import com.hypixel.hytale.server.core.universe.world.chunk.section.EnvironmentSection;
import com.hypixel.hytale.server.core.universe.world.chunk.section.FluidSection;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.GetChunkFlags;
import com.hypixel.hytale.server.core.util.FillerBlockUtil;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import org.joml.Vector3i;

/**
 * The Server 0.5.x {@code World}, {@code WorldChunk} and {@code BlockChunk} block accessors the
 * bridge relied on, re-implemented on the 0.7.2 chunk store.
 *
 * <p>0.7.2 removed those accessors. Block, fluid, light, environment and support data now live on
 * per-section components, the heightmap on a per-column component, and a column or section is
 * "ticking" when it lacks the {@code NonTicking} component. Every method keeps the 0.5.10 behaviour
 * it replaces (synchronous loading, ticking rules, out-of-range defaults and exceptions); each
 * comment names the 0.5.10 method it mirrors.
 *
 * <p>Threads: 0.5 block reads were plain field reads, callable from any thread. 0.7.2 component
 * reads assert the world thread, so every public read here runs on the world thread, hopping to it
 * (and waiting) when called from elsewhere, as 0.5's {@code World} methods did for chunk lookups.
 */
public final class NativeWorldAccess {
    private NativeWorldAccess() {}

    // ---- column lookup ------------------------------------------------------------------

    /** 0.5 {@code IWorldChunks.getChunk}: the column, loading it and starting it ticking if needed. */
    @Nullable
    public static Column getChunk(@Nonnull World world, long index) {
        Column column = loadChunkIfInMemory(world, index);
        if (column != null) {
            return column;
        }
        CompletableFuture<Column> future = world.getChunkStore()
                .getChunkReferenceAsync(index, GetChunkFlags.SET_TICKING)
                .thenApplyAsync(reference -> {
                    Column loaded = column(world, reference);
                    if (loaded != null) {
                        // 0.7.2 ticks only the column on this path; 0.5 propagated it to the sections.
                        loaded.setTicking(true);
                    }
                    return loaded;
                }, world);
        return waitForFutureWithoutLock(world, future);
    }

    /** 0.5 {@code IWorldChunks.getNonTickingChunk}: the column, loading it without starting it ticking. */
    @Nullable
    public static Column getNonTickingChunk(@Nonnull World world, long index) {
        Column column = getChunkIfInMemory(world, index);
        if (column != null) {
            return column;
        }
        CompletableFuture<Column> future = world.getChunkStore()
                .getChunkReferenceAsync(index)
                .thenApplyAsync(reference -> column(world, reference), world);
        return waitForFutureWithoutLock(world, future);
    }

    /** 0.5 {@code World.loadChunkIfInMemory}: the column if it is in memory, after starting it ticking. */
    @Nullable
    public static Column loadChunkIfInMemory(@Nonnull World world, long index) {
        if (!world.isInThread()) {
            return CompletableFuture.supplyAsync(() -> loadChunkIfInMemory(world, index), world).join();
        }
        Column column = column(world, world.getChunkStore().getChunkReference(index));
        if (column != null) {
            column.setTicking(true);
        }
        return column;
    }

    /** 0.5 {@code World.getChunkIfInMemory}: the column if it is in memory, ticking or not. */
    @Nullable
    public static Column getChunkIfInMemory(@Nonnull World world, long index) {
        Ref<ChunkStore> reference = world.getChunkStore().getChunkReference(index);
        if (reference == null) {
            return null;
        }
        if (!world.isInThread()) {
            return CompletableFuture.supplyAsync(() -> getChunkIfInMemory(world, index), world).join();
        }
        return column(world, reference);
    }

    /** 0.5 {@code World.getChunkIfLoaded}: the column only if it is in memory and ticking. */
    @Nullable
    public static Column getChunkIfLoaded(@Nonnull World world, long index) {
        if (!world.isInThread()) {
            return CompletableFuture.supplyAsync(() -> getChunkIfLoaded(world, index), world).join();
        }
        Column column = column(world, world.getChunkStore().getChunkReference(index));
        return column != null && column.isTicking() ? column : null;
    }

    // ---- block access by world position (0.5 IChunkAccessorSync defaults) -----------------

    /** 0.5 {@code World.getBlock}: the block id, loading the column; 0 if it cannot load. */
    public static int getBlock(@Nonnull World world, int x, int y, int z) {
        if (!world.isInThread()) {
            return onWorld(world, () -> getBlock(world, x, y, z));
        }
        Column column = getChunk(world, ChunkUtil.indexChunkFromBlock(x, z));
        return column == null ? 0 : column.getBlock(x, y, z);
    }

    /** 0.5 {@code World.getBlockType}: the block type, loading the column; null if it cannot load. */
    @Nullable
    public static BlockType getBlockType(@Nonnull World world, int x, int y, int z) {
        if (!world.isInThread()) {
            return onWorld(world, () -> getBlockType(world, x, y, z));
        }
        Column column = getChunk(world, ChunkUtil.indexChunkFromBlock(x, z));
        return column == null ? null : BlockType.getAssetMap().getAsset(column.getBlock(x, y, z));
    }

    /** 0.5 {@code World.getBlockType(Vector3i)}. */
    @Nullable
    public static BlockType getBlockType(@Nonnull World world, @Nonnull Vector3i pos) {
        return getBlockType(world, pos.x(), pos.y(), pos.z());
    }

    /** 0.5 {@code World.getBlockRotationIndex}: like 0.5, fails if the column cannot load. */
    public static int getBlockRotationIndex(@Nonnull World world, int x, int y, int z) {
        if (!world.isInThread()) {
            return onWorld(world, () -> getBlockRotationIndex(world, x, y, z));
        }
        return requireChunk(world, x, z).getRotationIndex(x, y, z);
    }

    /** 0.5 {@code World.getBaseBlock}: the filler root, loading the column without ticking it. */
    @Nonnull
    public static BlockPosition getBaseBlock(@Nonnull World world, @Nonnull BlockPosition position) {
        if (!world.isInThread()) {
            return onWorld(world, () -> getBaseBlock(world, position));
        }
        Column column = Objects.requireNonNull(
                getNonTickingChunk(world, ChunkUtil.indexChunkFromBlock(position.x, position.z)),
                "chunk did not load");
        int filler = column.getFiller(position.x, position.y, position.z);
        return filler != 0
                ? new BlockPosition(
                        position.x - FillerBlockUtil.unpackX(filler),
                        position.y - FillerBlockUtil.unpackY(filler),
                        position.z - FillerBlockUtil.unpackZ(filler))
                : position;
    }

    /** 0.5 {@code World.setBlock(x, y, z, key)}: settings 0; unknown keys throw. */
    public static void setBlock(@Nonnull World world, int x, int y, int z, @Nonnull String blockTypeKey) {
        setBlock(world, x, y, z, blockTypeKey, 0);
    }

    /** 0.5 {@code World.setBlock(x, y, z, key, settings)}: loads the column, then sets the block. */
    public static void setBlock(@Nonnull World world, int x, int y, int z, @Nonnull String blockTypeKey, int settings) {
        if (!world.isInThread()) {
            onWorld(world, () -> {
                setBlock(world, x, y, z, blockTypeKey, settings);
                return null;
            });
            return;
        }
        requireChunk(world, x, z).setBlock(x, y, z, blockTypeKey, settings);
    }

    /**
     * 0.5 {@code IWorldChunks.waitForFutureWithoutLock}: on the world thread, releases the asset read
     * lock and drains the world task queue until the future completes; elsewhere, joins it.
     */
    public static <T> T waitForFutureWithoutLock(@Nonnull World world, @Nonnull CompletableFuture<T> future) {
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
    @Nullable
    public static Column column(@Nonnull World world, @Nullable Ref<ChunkStore> reference) {
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

    @Nonnull
    private static Column requireChunk(@Nonnull World world, int x, int z) {
        // 0.5 dereferenced getChunk() without a null check, so a column that cannot load fails here.
        return Objects.requireNonNull(getChunk(world, ChunkUtil.indexChunkFromBlock(x, z)), "chunk did not load");
    }

    private static <T> T onWorld(@Nonnull World world, @Nonnull Supplier<T> read) {
        return CompletableFuture.supplyAsync(read, world).join();
    }

    /**
     * A chunk column: the 0.7.2 stand-in for a 0.5.x {@code WorldChunk} together with its
     * {@code BlockChunk}. Coordinates are world coordinates, as in 0.5; section components mask
     * them to the section. Like the 0.5 objects it replaces, use it within the tick that looked it
     * up; section components are resolved once and reused.
     */
    public static final class Column {
        private final World world;
        private final Store<ChunkStore> store;
        private final Ref<ChunkStore> reference;
        private final WorldChunk worldChunk;
        private final ChunkColumn chunkColumn;
        private final BlockSection[] blockSections = new BlockSection[ChunkUtil.HEIGHT_SECTIONS];
        private final FluidSection[] fluidSections = new FluidSection[ChunkUtil.HEIGHT_SECTIONS];

        private Column(
                World world, Store<ChunkStore> store, Ref<ChunkStore> reference, WorldChunk worldChunk, ChunkColumn chunkColumn) {
            this.world = world;
            this.store = store;
            this.reference = reference;
            this.worldChunk = worldChunk;
            this.chunkColumn = chunkColumn;
        }

        @Nonnull
        public WorldChunk worldChunk() {
            return worldChunk;
        }

        @Nonnull
        public Ref<ChunkStore> reference() {
            return reference;
        }

        public int getX() {
            return worldChunk.getX();
        }

        public int getZ() {
            return worldChunk.getZ();
        }

        public long getIndex() {
            return worldChunk.getIndex();
        }

        /** 0.5 {@code WorldChunk.is(ChunkFlag.TICKING)}: false once the column is gone, like 0.5's flag read. */
        public boolean isTicking() {
            if (!world.isInThread()) {
                return onWorld(world, this::isTicking);
            }
            return reference.isValid()
                    && !store.getArchetype(reference).contains(ChunkStore.REGISTRY.getNonTickingComponentType());
        }

        /**
         * 0.5 {@code WorldChunk.setFlag(ChunkFlag.TICKING, ticking)}: on a change, resets keep-alive and
         * toggles the column's NonTicking marker. 0.5's {@code ChunkSystems.OnNonTicking} then copied the
         * marker to every section of the column; 0.7.2 dropped that system, so the sections are synced here,
         * whether or not the column changed (0.7.2's own load path can leave them behind).
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

        /** 0.5 {@code ChunkColumn.getSection} for this column: null outside 0..9 or when absent. */
        @Nullable
        public Ref<ChunkStore> getSectionReference(int index) {
            return chunkColumn.getSection(index);
        }

        /** 0.5 {@code BlockChunk.getSectionAtIndex}: throws outside 0..9, as in 0.5. */
        @Nonnull
        public BlockSection getSectionAtIndex(int index) {
            if (index < 0 || index >= ChunkUtil.HEIGHT_SECTIONS) {
                throw new IllegalArgumentException(
                        "Section index must >=0 and <" + ChunkUtil.HEIGHT_SECTIONS + " but was given " + index);
            }
            return blockSection(index);
        }

        /** 0.5 {@code BlockChunk.getSectionAtBlockY}: throws outside 0..319, as in 0.5. */
        @Nonnull
        public BlockSection getSectionAtBlockY(int y) {
            int index = ChunkUtil.indexSection(y);
            if (index < 0 || index >= ChunkUtil.HEIGHT_SECTIONS) {
                throw new IllegalArgumentException("Section y must >=0 and <320 but was given " + y);
            }
            return blockSection(index);
        }

        /** 0.5 {@code WorldChunk.getBlock}: 0 outside 0..319. */
        public int getBlock(int x, int y, int z) {
            return y >= 0 && y < ChunkUtil.HEIGHT ? getSectionAtBlockY(y).get(x, y, z) : 0;
        }

        /** 0.5 {@code WorldChunk.getRotationIndex}: 0 outside 0..319. */
        public int getRotationIndex(int x, int y, int z) {
            return y >= 0 && y < ChunkUtil.HEIGHT ? getSectionAtBlockY(y).getRotationIndex(x, y, z) : 0;
        }

        /** 0.5 {@code WorldChunk.getFiller}: 0 outside 0..319. */
        public int getFiller(int x, int y, int z) {
            return y >= 0 && y < ChunkUtil.HEIGHT ? getSectionAtBlockY(y).getFiller(x, y, z) : 0;
        }

        /** 0.5 {@code WorldChunk.getFluidId}: {@code Integer.MIN_VALUE} when the section is absent. */
        public int getFluidId(int x, int y, int z) {
            FluidSection fluid = fluidSection(ChunkUtil.chunkCoordinate(y));
            return fluid == null ? Integer.MIN_VALUE : fluid.getFluidId(x, y, z);
        }

        /** 0.5 {@code WorldChunk.getFluidLevel}: 0 when the section is absent. */
        public byte getFluidLevel(int x, int y, int z) {
            FluidSection fluid = fluidSection(ChunkUtil.chunkCoordinate(y));
            return fluid == null ? 0 : fluid.getFluidLevel(x, y, z);
        }

        /** 0.5 {@code WorldChunk.getSupportValue}: 0 when the section or its physics data is absent. */
        public int getSupportValue(int x, int y, int z) {
            if (!world.isInThread()) {
                return onWorld(world, () -> getSupportValue(x, y, z));
            }
            Ref<ChunkStore> section = chunkColumn.getSection(ChunkUtil.chunkCoordinate(y));
            if (section == null) {
                return 0;
            }
            BlockPhysics blockPhysics = store.getComponent(section, BlockPhysics.getComponentType());
            return blockPhysics != null ? blockPhysics.get(x, y, z) : 0;
        }

        /** 0.5 {@code WorldChunk/BlockChunk.getHeight}: the column heightmap; 0 where 0.7.2 has none, as 0.5 read. */
        public short getHeight(int x, int z) {
            if (!world.isInThread()) {
                return onWorld(world, () -> getHeight(x, z));
            }
            HeightmapColumn heightmap = store.getComponent(reference, HeightmapColumn.getComponentType());
            int height = heightmap == null ? HeightmapColumn.NO_HEIGHT : heightmap.getHeight(x, z);
            return height == HeightmapColumn.NO_HEIGHT ? 0 : (short) height;
        }

        /** True where the column's heightmap holds a height; NO_HEIGHT marks a column with none. */
        public boolean hasHeight(int x, int z) {
            if (!world.isInThread()) {
                return onWorld(world, () -> hasHeight(x, z));
            }
            HeightmapColumn heightmap = store.getComponent(reference, HeightmapColumn.getComponentType());
            return heightmap != null && hasHeightmapEntry(heightmap.getHeight(x, z));
        }

        static boolean hasHeightmapEntry(int engineHeight) {
            return engineHeight != HeightmapColumn.NO_HEIGHT;
        }

        /** 0.5 {@code WorldChunk/BlockChunk.getTint}: the column tint, still held by the 0.7.2 BlockChunk. */
        public int getTint(int x, int z) {
            if (!world.isInThread()) {
                return onWorld(world, () -> getTint(x, z));
            }
            return store.getComponent(reference, BlockChunk.getComponentType()).getTint(x, z);
        }

        /** 0.5 {@code BlockChunk.getEnvironment}: 0 outside 0..319. */
        public int getEnvironment(int x, int y, int z) {
            if (y < 0 || y >= ChunkUtil.HEIGHT) {
                return 0;
            }
            if (!world.isInThread()) {
                return onWorld(world, () -> getEnvironment(x, y, z));
            }
            Ref<ChunkStore> section = chunkColumn.getSection(ChunkUtil.chunkCoordinate(y));
            EnvironmentSection environment =
                    section == null ? null : store.getComponent(section, EnvironmentSection.getComponentType());
            return environment == null ? 0 : environment.get(x, y, z);
        }

        /** 0.5 {@code BlockChunk.getSkyLight}: propagated (global) light, 0 outside 0..319. */
        public byte getSkyLight(int x, int y, int z) {
            return y >= 0 && y < ChunkUtil.HEIGHT ? getSectionAtBlockY(y).getGlobalLight().getSkyLight(x, y, z) : 0;
        }

        /** 0.5 {@code BlockChunk.getBlockLightIntensity}. */
        public byte getBlockLightIntensity(int x, int y, int z) {
            return y >= 0 && y < ChunkUtil.HEIGHT
                    ? getSectionAtBlockY(y).getGlobalLight().getBlockLightIntensity(x, y, z)
                    : 0;
        }

        /** 0.5 {@code BlockChunk.getRedBlockLight}. */
        public byte getRedBlockLight(int x, int y, int z) {
            return y >= 0 && y < ChunkUtil.HEIGHT ? getSectionAtBlockY(y).getGlobalLight().getRedBlockLight(x, y, z) : 0;
        }

        /** 0.5 {@code BlockChunk.getGreenBlockLight}. */
        public byte getGreenBlockLight(int x, int y, int z) {
            return y >= 0 && y < ChunkUtil.HEIGHT ? getSectionAtBlockY(y).getGlobalLight().getGreenBlockLight(x, y, z) : 0;
        }

        /** 0.5 {@code BlockChunk.getBlueBlockLight}. */
        public byte getBlueBlockLight(int x, int y, int z) {
            return y >= 0 && y < ChunkUtil.HEIGHT ? getSectionAtBlockY(y).getGlobalLight().getBlueBlockLight(x, y, z) : 0;
        }

        /** 0.5 {@code BlockAccessor.setBlock(x, y, z, key, settings)}: unknown keys throw. */
        public boolean setBlock(int x, int y, int z, @Nonnull String blockTypeKey, int settings) {
            int id = BlockType.getAssetMap().getIndex(blockTypeKey);
            if (id == Integer.MIN_VALUE) {
                throw new IllegalArgumentException("Unknown key! " + blockTypeKey);
            }
            return setBlock(x, y, z, id, BlockType.getAssetMap().getAsset(id), 0, 0, settings);
        }

        /**
         * 0.5 {@code WorldChunk.setBlock(x, y, z, id, type, rotation, filler, settings)}: false outside
         * 0..319. 0.7.2 moved this body to {@code BlockOperations.setBlock} with the same settings bits.
         */
        public boolean setBlock(
                int x, int y, int z, int id, @Nonnull BlockType blockType, int rotation, int filler, int settings) {
            if (y < 0 || y >= ChunkUtil.HEIGHT) {
                return false;
            }
            if (!world.isInThread()) {
                return onWorld(world, () -> setBlock(x, y, z, id, blockType, rotation, filler, settings));
            }
            Ref<ChunkStore> section = requireSection(ChunkUtil.chunkCoordinate(y));
            return BlockOperations.setBlock(
                    world.getChunkStore(), section, x, y, z, id, blockType, rotation, filler, settings);
        }

        @Nonnull
        private BlockSection blockSection(int index) {
            BlockSection section = blockSections[index];
            if (section == null) {
                if (!world.isInThread()) {
                    return onWorld(world, () -> blockSection(index));
                }
                section = Objects.requireNonNull(
                        store.getComponent(requireSection(index), BlockSection.getComponentType()),
                        "section has no block data");
                blockSections[index] = section;
            }
            return section;
        }

        @Nullable
        private FluidSection fluidSection(int index) {
            if (index < 0 || index >= ChunkUtil.HEIGHT_SECTIONS || chunkColumn.getSection(index) == null) {
                return null;
            }
            FluidSection section = fluidSections[index];
            if (section == null) {
                if (!world.isInThread()) {
                    return onWorld(world, () -> fluidSection(index));
                }
                // Like 0.5, a section without fluid data fails here rather than reading as dry.
                section = Objects.requireNonNull(
                        store.getComponent(chunkColumn.getSection(index), FluidSection.getComponentType()),
                        "section has no fluid data");
                fluidSections[index] = section;
            }
            return section;
        }

        @Nonnull
        private Ref<ChunkStore> requireSection(int index) {
            // 0.5 columns always held all ten sections; a missing one means a cubic world the bridge never used.
            Ref<ChunkStore> section = chunkColumn.getSection(index);
            if (section == null) {
                throw new IllegalStateException(
                        "section " + index + " of chunk " + getX() + "," + getZ() + " is not loaded");
            }
            return section;
        }
    }
}
