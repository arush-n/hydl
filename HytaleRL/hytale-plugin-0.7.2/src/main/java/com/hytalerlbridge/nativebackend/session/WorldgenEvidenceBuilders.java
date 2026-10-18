package com.hytalerlbridge.nativebackend.session;

import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.query.Query;
import com.hypixel.hytale.math.vector.Rotation3f;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.entity.UUIDComponent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.modules.entity.component.FromPrefabInstance;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entity.component.WorldGenId;
import com.hypixel.hytale.server.core.modules.interaction.BlockHarvestUtils;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.spawning.spawnmarkers.SpawnMarkerEntity;
import com.hytalerlbridge.nativebackend.model.ResolvedDropRoute;
import com.hytalerlbridge.nativebackend.support.DropRouting;
import com.hytalerlbridge.worldgen.NativeDropProgramEvidence;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerQuery;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerRow;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerSnapshot;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.joml.Vector3d;

import static com.hytalerlbridge.nativebackend.support.DropRouting.canonicalDropOutcomeKey;
import static com.hytalerlbridge.nativebackend.support.DropRouting.canonicalDropStack;
import static com.hytalerlbridge.nativebackend.support.DropRouting.resolveDropRoute;

/**
 * Worldgen structure marker and drop program evidence builders. Each takes the worldgen
 * identity it reports as leading parameters.
 */
public final class WorldgenEvidenceBuilders {

    private WorldgenEvidenceBuilders() {}

    public static WorldgenStructureMarkerSnapshot buildWorldgenStructureMarkers(
        String NATIVE_SERVER_VERSION,
        long seed,
        World world,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        WorldgenStructureMarkerQuery query
    ) {
        Store<EntityStore> store = world.getEntityStore().getStore();
        Query<EntityStore> componentQuery = Query.and(
            UUIDComponent.getComponentType(),
            TransformComponent.getComponentType(),
            WorldGenId.getComponentType(),
            FromPrefabInstance.getComponentType(),
            SpawnMarkerEntity.getComponentType()
        );
        List<WorldgenStructureMarkerRow> matches = new ArrayList<>();
        store.forEachChunk(componentQuery, (chunk, commandBuffer) -> {
            for (int index = 0; index < chunk.size(); index++) {
                UUIDComponent identity = chunk.getComponent(
                    index,
                    UUIDComponent.getComponentType()
                );
                TransformComponent transform = chunk.getComponent(
                    index,
                    TransformComponent.getComponentType()
                );
                WorldGenId worldgenId = chunk.getComponent(
                    index,
                    WorldGenId.getComponentType()
                );
                FromPrefabInstance prefabInstance = chunk.getComponent(
                    index,
                    FromPrefabInstance.getComponentType()
                );
                SpawnMarkerEntity marker = chunk.getComponent(
                    index,
                    SpawnMarkerEntity.getComponentType()
                );
                if (
                    identity == null
                        || transform == null
                        || worldgenId == null
                        || prefabInstance == null
                        || marker == null
                        || !query.accepts(marker.getSpawnMarkerId())
                ) {
                    continue;
                }
                Vector3d position = transform.getPosition();
                if (!query.contains(position.x, position.y, position.z)) {
                    continue;
                }
                Rotation3f rotation = transform.getRotation();
                matches.add(WorldgenStructureMarkerRow.of(
                    identity.getUuid(),
                    marker.getSpawnMarkerId(),
                    position.x,
                    position.y,
                    position.z,
                    rotation.yaw(),
                    rotation.pitch(),
                    rotation.roll(),
                    worldgenId.getWorldGenId(),
                    prefabInstance.getPrefabInstanceId()
                ));
            }
        });
        return WorldgenStructureMarkerSnapshot.fromMatches(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            query,
            matches
        );
    }

    public static NativeDropProgramEvidence buildDropProgramEvidence(
        String NATIVE_SERVER_VERSION,
        long seed,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        String blockAssetId,
        String route,
        int sampleCount
    ) {
        if (blockAssetId == null || blockAssetId.isBlank()) {
            throw new IllegalArgumentException(
                "Drop-program block asset ID cannot be blank"
            );
        }
        if (
            sampleCount < 1
                || sampleCount > NativeDropProgramEvidence.MAX_SAMPLES
        ) {
            throw new IllegalArgumentException(
                "Drop-program sample count exceeds capacity"
            );
        }
        BlockType blockType = BlockType.getAssetMap().getAsset(blockAssetId);
        if (blockType == null || blockType == BlockType.EMPTY) {
            throw new IllegalArgumentException(
                "Unknown block asset ID: " + blockAssetId
            );
        }
        ResolvedDropRoute resolved = resolveDropRoute(blockType, route);
        Map<List<NativeDropProgramEvidence.Stack>, Integer> counts =
            new LinkedHashMap<>();
        for (int sample = 0; sample < sampleCount; sample++) {
            List<ItemStack> drops = BlockHarvestUtils.getDrops(
                blockType,
                resolved.quantity(),
                resolved.itemId(),
                resolved.dropListId()
            );
            if (drops == null) {
                throw new IllegalStateException(
                    "Native drop API returned null"
                );
            }
            List<NativeDropProgramEvidence.Stack> stacks = drops.stream()
                .filter(stack -> !ItemStack.isEmpty(stack))
                .map(DropRouting::canonicalDropStack)
                .toList();
            if (
                stacks.size()
                    > NativeDropProgramEvidence.MAX_STACKS_PER_OUTCOME
            ) {
                throw new IllegalStateException(
                    "Native drop stack capacity exceeded"
                );
            }
            counts.merge(stacks, 1, Integer::sum);
            if (
                counts.size()
                    > NativeDropProgramEvidence.MAX_OUTCOMES
            ) {
                throw new IllegalStateException(
                    "Native drop outcome capacity exceeded"
                );
            }
        }
        List<NativeDropProgramEvidence.Outcome> outcomes = counts.entrySet()
            .stream()
            .map(entry -> new NativeDropProgramEvidence.Outcome(
                entry.getValue(),
                entry.getKey()
            ))
            .sorted(Comparator.comparing(
                outcome -> canonicalDropOutcomeKey(outcome.stacks())
            ))
            .toList();
        return new NativeDropProgramEvidence(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            blockAssetId,
            route,
            resolved.quantity(),
            resolved.itemId(),
            resolved.dropListId(),
            sampleCount,
            outcomes
        );
    }
}
