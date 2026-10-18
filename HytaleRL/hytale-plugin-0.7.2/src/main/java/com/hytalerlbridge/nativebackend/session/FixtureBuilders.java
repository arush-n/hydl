package com.hytalerlbridge.nativebackend.session;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.server.core.asset.type.fluid.Fluid;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.chunk.ChunkColumn;
import com.hypixel.hytale.server.core.universe.world.chunk.WorldChunk;
import com.hypixel.hytale.server.core.universe.world.chunk.section.FluidSection;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hytalerlbridge.environment.EnvironmentOptions;
import com.hytalerlbridge.nativebackend.support.NativeWorldAccess;

/**
 * Fixture placement for the dive motion and fidelity descriptions.
 */
public final class FixtureBuilders {

    private static final int DIVE_POOL_HALF_WIDTH = 8;
    private static final int DIVE_POOL_MIN_Y = 65;
    private static final int DIVE_POOL_MAX_Y = 75;

    private FixtureBuilders() {}

    public static void createDiveMotionFixture(
        World world
    ) {
        Fluid water = Fluid.getAssetMap().getAsset("Water_Source");
        if (water == null || water == Fluid.EMPTY) {
            throw new IllegalStateException(
                "Installed assets do not provide Water_Source"
            );
        }
        int fluidId = Fluid.getAssetMap().getIndex(water.getId());
        byte fluidLevel = (byte) water.getMaxFluidLevel();
        Store<ChunkStore> store = world.getChunkStore().getStore();
        for (
            int x = -DIVE_POOL_HALF_WIDTH;
            x <= DIVE_POOL_HALF_WIDTH;
            x++
        ) {
            for (
                int z = -DIVE_POOL_HALF_WIDTH;
                z <= DIVE_POOL_HALF_WIDTH;
                z++
            ) {
                boolean wall = Math.abs(x) == DIVE_POOL_HALF_WIDTH
                    || Math.abs(z) == DIVE_POOL_HALF_WIDTH;
                for (int y = DIVE_POOL_MIN_Y; y <= DIVE_POOL_MAX_Y; y++) {
                    if (wall) {
                        NativeWorldAccess.setBlock(world, x, y, z, "Rock_Stone");
                    } else if (y < DIVE_POOL_MAX_Y) {
                        setFixtureFluid(world,
                            store,
                            x,
                            y,
                            z,
                            fluidId,
                            fluidLevel
                        );
                    }
                }
            }
        }
    }

    private static void setFixtureFluid(
        World world,
        Store<ChunkStore> store,
        int x,
        int y,
        int z,
        int fluidId,
        byte fluidLevel
    ) {
        NativeWorldAccess.Column loaded = NativeWorldAccess.getChunkIfLoaded(
            world,
            ChunkUtil.indexChunkFromBlock(x, z)
        );
        WorldChunk chunk = loaded == null ? null : loaded.worldChunk();
        if (chunk == null || chunk.getReference() == null) {
            throw new IllegalStateException(
                "Dive fixture chunk is not loaded"
            );
        }
        ChunkColumn column = store.getComponent(
            chunk.getReference(),
            ChunkColumn.getComponentType()
        );
        if (column == null) {
            throw new IllegalStateException(
                "Dive fixture chunk has no column"
            );
        }
        Ref<ChunkStore> section = column.getSection(
            ChunkUtil.chunkCoordinate(y)
        );
        FluidSection fluidSection = store.ensureAndGetComponent(
            section,
            FluidSection.getComponentType()
        );
        fluidSection.setFluid(x, y, z, fluidId, fluidLevel);
    }

    public static String fidelityFixtureDescription(
        EnvironmentOptions options,
        String taskId
    ) {
        if (options.world().equals("hytale")) return "native_hytale_worldgen";
        if (options.world().equals("hytale_generator")) {
            return "native_hytale_generator_worldgen";
        }
        return switch (options.fidelityFixture()) {
            case "ledge" -> "flat_floor;three_block_trench_z=-1..-5";
            case "ceiling" -> "flat_floor;stone_ceiling_y=67";
            case "half_block" -> "flat_floor;stone_half_block_z=-2";
            case "los_wall" -> "flat_floor;los_wall_z=-1";
            case "los_diagonal_graze" ->
                "flat_floor;diagonal_los_grazes_block_x=1,z=0";
            case "los_diagonal_block" ->
                "flat_floor;diagonal_los_crosses_block_x=1,z=-1";
            case EnvironmentOptions.TARGET_MEMORY_FIXTURE ->
                "flat_floor;natural_brawler;dynamic_sealed_occluder";
            case EnvironmentOptions.PATH_FOLLOWER_FIXTURE ->
                "flat_floor;chase_attack;stone_wall_z=-3,x=-2..2";
            case EnvironmentOptions.NATIVE_PROJECTILE_DUEL_FIXTURE ->
                "flat_floor;outlander_hunter_vs_kweebec;range=10";
            case EnvironmentOptions.DIVE_MOTION_FIXTURE ->
                "flat_floor;sealed_water_source_pool_xz=-7..7,y=65..74";
            case EnvironmentOptions.SYNTHETIC_WORLD_VERBS_FIXTURE ->
                "flat_floor;fixture_only_synthetic_adventure_player;"
                    + "place_x=1;break_x=-1";
            case EnvironmentOptions.SYNTHETIC_BLOCK_USE_FIXTURE ->
                "flat_floor;fixture_only_headless_npc;"
                    + "authored_use_block=Deco_Lantern;x=1";
            case EnvironmentOptions.FALLING_BLOCK_FIXTURE ->
                "flat_floor;native_falling_block_trace";
            default -> taskId.equals("native_fidelity") && !options.hasSpawn()
                ? "flat_floor;stone_wall_z=-4"
                : "flat_floor";
        };
    }
}
