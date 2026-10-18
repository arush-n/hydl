package com.hytalerlbridge.nativebackend.session;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.query.Query;
import com.hypixel.hytale.math.vector.Rotation3f;
import com.hypixel.hytale.server.core.entity.UUIDComponent;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.physics.component.Velocity;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.entity.PrivilegedEntityQuery;
import com.hytalerlbridge.entity.PrivilegedEntityRow;
import com.hytalerlbridge.entity.PrivilegedEntitySnapshot;
import com.hytalerlbridge.entity.PrivilegedNpcGeometryRow;
import com.hytalerlbridge.entity.PrivilegedNpcSnapshot;
import com.hytalerlbridge.entity.PrivilegedNpcUuidQuery;
import com.hytalerlbridge.nativebackend.support.PrivilegedNpcGeometryCapture;
import java.util.ArrayList;
import java.util.List;
import org.joml.Vector3d;

/**
 * Privileged entity and NPC snapshot builders. Each runs on the world thread and takes the world
 * state it reads as leading parameters; the session keeps the capture_* entry points.
 */
public final class PrivilegedSnapshotBuilders {

    private PrivilegedSnapshotBuilders() {}

    public static PrivilegedEntitySnapshot buildPrivilegedEntitySnapshot(
        String NATIVE_SERVER_VERSION,
        long seed,
        World world,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        PrivilegedEntityQuery query
    ) {
        Store<EntityStore> store = world.getEntityStore().getStore();
        Query<EntityStore> componentQuery = Query.and(
            UUIDComponent.getComponentType(),
            TransformComponent.getComponentType(),
            Velocity.getComponentType()
        );
        List<PrivilegedEntityRow> matches = new ArrayList<>();
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
                Velocity velocity = chunk.getComponent(
                    index,
                    Velocity.getComponentType()
                );
                if (
                    identity == null
                        || transform == null
                        || velocity == null
                ) {
                    continue;
                }
                Vector3d position = transform.getPosition();
                if (!query.contains(position.x, position.y, position.z)) {
                    continue;
                }
                Rotation3f rotation = transform.getRotation();
                matches.add(PrivilegedEntityRow.of(
                    identity.getUuid(),
                    position.x,
                    position.y,
                    position.z,
                    rotation.yaw(),
                    rotation.pitch(),
                    rotation.roll(),
                    velocity.getX(),
                    velocity.getY(),
                    velocity.getZ()
                ));
            }
        });
        return PrivilegedEntitySnapshot.fromMatches(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            query,
            matches
        );
    }

    public static PrivilegedNpcSnapshot buildPrivilegedNpcSnapshot(
        String NATIVE_SERVER_VERSION,
        long seed,
        World world,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        PrivilegedEntityQuery query
    ) {
        Store<EntityStore> store = world.getEntityStore().getStore();
        Query<EntityStore> componentQuery = Query.and(
            UUIDComponent.getComponentType(),
            TransformComponent.getComponentType(),
            Velocity.getComponentType(),
            NPCEntity.getComponentType()
        );
        List<PrivilegedNpcGeometryRow> matches = new ArrayList<>();
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
                Velocity velocity = chunk.getComponent(
                    index,
                    Velocity.getComponentType()
                );
                NPCEntity npc = chunk.getComponent(
                    index,
                    NPCEntity.getComponentType()
                );
                if (
                    identity == null
                        || transform == null
                        || velocity == null
                        || npc == null
                ) {
                    continue;
                }
                Vector3d position = transform.getPosition();
                if (!query.contains(position.x, position.y, position.z)) {
                    continue;
                }
                Rotation3f rotation = transform.getRotation();
                PrivilegedEntityRow spatial = PrivilegedEntityRow.ofNpc(
                    identity.getUuid(),
                    npc.getRoleName(),
                    position.x,
                    position.y,
                    position.z,
                    rotation.yaw(),
                    rotation.pitch(),
                    rotation.roll(),
                    velocity.getX(),
                    velocity.getY(),
                    velocity.getZ()
                );
                matches.add(PrivilegedNpcGeometryCapture.capture(
                    store,
                    chunk.getReferenceTo(index),
                    spatial
                ));
            }
        });
        return PrivilegedNpcSnapshot.fromMatches(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            query,
            matches
        );
    }

    public static PrivilegedNpcSnapshot buildPrivilegedNpcSnapshotByUuid(
        String NATIVE_SERVER_VERSION,
        long seed,
        World world,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        PrivilegedNpcUuidQuery uuidQuery
    ) {
        Store<EntityStore> store = world.getEntityStore().getStore();
        PrivilegedEntityQuery query = uuidQuery.spatialQuery();
        Ref<EntityStore> reference = world.getEntityStore().getRefFromUUID(
            uuidQuery.npcUuid()
        );
        List<PrivilegedNpcGeometryRow> matches = new ArrayList<>(1);
        if (reference != null && reference.isValid()) {
            UUIDComponent identity = store.getComponent(
                reference,
                UUIDComponent.getComponentType()
            );
            TransformComponent transform = store.getComponent(
                reference,
                TransformComponent.getComponentType()
            );
            Velocity velocity = store.getComponent(
                reference,
                Velocity.getComponentType()
            );
            NPCEntity npc = store.getComponent(
                reference,
                NPCEntity.getComponentType()
            );
            if (
                identity != null
                    && uuidQuery.npcUuid().equals(identity.getUuid())
                    && transform != null
                    && velocity != null
                    && npc != null
            ) {
                Vector3d position = transform.getPosition();
                if (query.contains(position.x, position.y, position.z)) {
                    Rotation3f rotation = transform.getRotation();
                    PrivilegedEntityRow spatial = PrivilegedEntityRow.ofNpc(
                        identity.getUuid(),
                        npc.getRoleName(),
                        position.x,
                        position.y,
                        position.z,
                        rotation.yaw(),
                        rotation.pitch(),
                        rotation.roll(),
                        velocity.getX(),
                        velocity.getY(),
                        velocity.getZ()
                    );
                    matches.add(PrivilegedNpcGeometryCapture.capture(
                        store,
                        reference,
                        spatial
                    ));
                }
            }
        }
        return PrivilegedNpcSnapshot.fromMatches(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            query,
            matches
        );
    }
}
