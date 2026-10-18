package com.hytalerlbridge.nativebackend.session;

import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hytalerlbridge.geometry.GeometryContract;
import com.hytalerlbridge.geometry.GeometryFrame;
import com.hytalerlbridge.nativebackend.support.NativeWorldAccess;
import com.hytalerlbridge.nativebackend.support.RegionLightCapture;
import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import com.hytalerlbridge.worldgen.NativePerceptionChannels;
import java.util.HashMap;
import java.util.Map;
import org.joml.Vector3i;

/**
 * Perception channel builder and the geometry and actor helpers used by the NPC trace
 * capture. The trace light key is nested here because its only other user is this builder.
 */
public final class PerceptionBuilders {

    private PerceptionBuilders() {}

    public record TraceLightSection(int chunkX, int chunkZ, int sectionY) {}

    public static NativePerceptionChannels buildPerceptionChannels(
        String NATIVE_SERVER_VERSION,
        long seed,
        Map<TraceLightSection, Long> traceLightRequestTicks,
        World world,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        int[] positions
    ) {
        record Column(short height, int tint) {}

        int samples = positions.length / 3;
        byte[] available = new byte[samples];
        byte[] validity = new byte[samples];
        short[] heightmap = new short[samples];
        byte[] skyLight = new byte[samples];
        byte[] blockLightRgb = new byte[samples * 3];
        int[] environment = new int[samples];
        int[] tintArgb = new int[samples];
        Map<Long, NativeWorldAccess.Column> chunks = new HashMap<>();
        Map<Long, Column> columns = new HashMap<>();
        Map<TraceLightSection, Boolean> lightReady = new HashMap<>();
        long worldTick = world.getTick();

        for (int sample = 0; sample < samples; sample++) {
            int offset = sample * 3;
            int x = positions[offset];
            int y = positions[offset + 1];
            int z = positions[offset + 2];
            long chunkIndex = ChunkUtil.indexChunkFromBlock(x, z);
            NativeWorldAccess.Column chunk = chunks.get(chunkIndex);
            if (chunk == null) {
                chunk = NativeWorldAccess.getChunkIfLoaded(world, chunkIndex);
                if (chunk != null) chunks.put(chunkIndex, chunk);
            }
            if (chunk == null) continue;
            NativeWorldAccess.Column blocks = chunk;
            if (blocks == null) continue;

            available[sample] = 1;
            int localX = ChunkUtil.localCoordinate((long) x);
            int localZ = ChunkUtil.localCoordinate((long) z);
            int valid = NativePerceptionChannels.VALID_TINT_RGB;
            long columnKey = ((long) x << 32) ^ (z & 0xffffffffL);
            Column column = columns.get(columnKey);
            if (column == null) {
                column = new Column(
                    blocks.getHeight(localX, localZ),
                    blocks.getTint(localX, localZ)
                );
                columns.put(columnKey, column);
            }
            if (column.height() != NativePerceptionChannels.WORLD_HEIGHT) {
                heightmap[sample] = column.height();
                valid |= NativePerceptionChannels.VALID_HEIGHTMAP;
            }
            tintArgb[sample] = column.tint();

            if (y >= 0 && y < NativePerceptionChannels.WORLD_HEIGHT) {
                valid |= NativePerceptionChannels.VALID_ENVIRONMENT;
                environment[sample] = blocks.getEnvironment(localX, y, localZ);
                int sectionY = Math.floorDiv(y, ChunkUtil.SIZE);
                TraceLightSection lightSection = new TraceLightSection(
                    chunk.getX(), chunk.getZ(), sectionY
                );
                Boolean ready = lightReady.get(lightSection);
                if (ready == null) {
                    ready = RegionLightCapture.isAvailable(
                        blocks.getSectionAtBlockY(y)
                    );
                    lightReady.put(lightSection, ready);
                }
                if (ready) {
                    traceLightRequestTicks.remove(lightSection);
                    valid |= NativePerceptionChannels.VALID_SKY_LIGHT
                        | NativePerceptionChannels.VALID_BLOCK_LIGHT_RGB;
                    skyLight[sample] = blocks.getSkyLight(localX, y, localZ);
                    int rgb = sample * 3;
                    blockLightRgb[rgb] = blocks.getRedBlockLight(localX, y, localZ);
                    blockLightRgb[rgb + 1] = blocks.getGreenBlockLight(
                        localX,
                        y,
                        localZ
                    );
                    blockLightRgb[rgb + 2] = blocks.getBlueBlockLight(
                        localX,
                        y,
                        localZ
                    );
                } else {
                    Long requested = traceLightRequestTicks.get(lightSection);
                    if (requested == null || requested.longValue() != worldTick) {
                        RegionLightCapture.invalidateEmptyPlaceholders(
                            blocks.getSectionAtBlockY(y)
                        );
                        world.getChunkLighting().addToQueue(
                            new Vector3i(chunk.getX(), sectionY, chunk.getZ())
                        );
                        traceLightRequestTicks.put(lightSection, worldTick);
                    }
                }
            }
            validity[sample] = (byte) valid;
        }
        return new NativePerceptionChannels(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            positions,
            available,
            validity,
            heightmap,
            skyLight,
            blockLightRgb,
            environment,
            tintArgb
        );
    }

    public static NativeActorEvidenceFrame.Actor actorPerceptibility(
        NativeActorEvidenceFrame.Actor actor,
        boolean perceptible
    ) {
        if (!actor.present() || actor.perceptible() == perceptible) return actor;
        return new NativeActorEvidenceFrame.Actor(
            actor.entityId(),
            true,
            perceptible,
            actor.roleId(),
            actor.itemId(),
            actor.itemRuntimeIndex(),
            actor.activeAbilitySlot(),
            actor.position(),
            actor.velocity(),
            actor.motionForce(),
            actor.yawDegrees(),
            actor.pitchDegrees(),
            actor.headYawDegrees(),
            actor.health(),
            actor.maxHealth(),
            actor.resourceValues(),
            actor.resourceMaximums(),
            actor.resourceAvailable(),
            actor.defenseValues(),
            actor.defenseAvailable(),
            actor.statuses(),
            actor.actorWorldValues(),
            actor.actorWorldAvailable(),
            actor.movementStates()
        );
    }

    public static int[] geometryCubePositions(
        GeometryFrame geometry
    ) {
        return geometryCubePositions(
            geometry.originX(), geometry.originY(), geometry.originZ()
        );
    }

    public static int[] geometryCubePositions(
        int originX,
        int originY,
        int originZ
    ) {
        int[] result = new int[GeometryContract.CELL_COUNT * 3];
        for (int dx = -GeometryContract.RADIUS; dx <= GeometryContract.RADIUS; dx++) {
            for (int dy = -GeometryContract.RADIUS; dy <= GeometryContract.RADIUS; dy++) {
                for (int dz = -GeometryContract.RADIUS; dz <= GeometryContract.RADIUS; dz++) {
                    int index = GeometryContract.cellIndex(dx, dy, dz) * 3;
                    result[index] = originX + dx;
                    result[index + 1] = originY + dy;
                    result[index + 2] = originZ + dz;
                }
            }
        }
        return result;
    }
}
