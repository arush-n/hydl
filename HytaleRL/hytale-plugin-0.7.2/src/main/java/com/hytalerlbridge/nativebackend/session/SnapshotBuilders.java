package com.hytalerlbridge.nativebackend.session;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.query.Query;
import com.hypixel.hytale.server.core.modules.entity.component.HeadRotation;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatMap;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatValue;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatsModule;
import com.hypixel.hytale.server.core.modules.physics.component.Velocity;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.support.StateSupport;
import com.hytalerlbridge.nativebackend.model.EntityReading;
import com.hytalerlbridge.nativebackend.model.TargetState;
import com.hytalerlbridge.observation.ObservationEncoding;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.joml.Vector3d;

import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.canonicalEntityType;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.captureStatuses;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.entityHealth;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.sameEntity;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.stat;
import static com.hytalerlbridge.nativebackend.support.SupportMath.normalizeDegrees;

/**
 * Nearby entity, target state and state text readers for the per-step snapshot.
 */
public final class SnapshotBuilders {

    private static final int MAX_NEARBY_ENTITIES = 20;

    private SnapshotBuilders() {}

    public static List<int[]> captureNearbyEntities(
        Ref<EntityStore> agentRef,
        double NEARBY_ENTITY_RADIUS,
        Store<EntityStore> store,
        Vector3d agentPosition
    ) {
        List<EntityReading> readings = new ArrayList<>();
        Query<EntityStore> query = Query.and(
            NPCEntity.getComponentType(),
            TransformComponent.getComponentType()
        );
        store.forEachChunk(query, (chunk, commandBuffer) -> {
            for (int index = 0; index < chunk.size(); index++) {
                Ref<EntityStore> reference = chunk.getReferenceTo(index);
                if (sameEntity(reference, agentRef)) continue;
                NPCEntity entity = chunk.getComponent(index, NPCEntity.getComponentType());
                TransformComponent entityTransform = chunk.getComponent(
                    index,
                    TransformComponent.getComponentType()
                );
                if (entity == null || entityTransform == null) continue;
                Vector3d entityPosition = entityTransform.getPosition();
                double dx = entityPosition.x - agentPosition.x;
                double dz = entityPosition.z - agentPosition.z;
                double distanceSquared = dx * dx + dz * dz;
                if (distanceSquared >= NEARBY_ENTITY_RADIUS * NEARBY_ENTITY_RADIUS) continue;
                double entityHealth = entityHealth(store, reference, entity);
                if (entityHealth <= 0.0) continue;
                readings.add(new EntityReading(
                    new int[] {
                        canonicalEntityType(entity.getRoleName()),
                        ObservationEncoding.encodeNearbyEntityScalar(dx),
                        ObservationEncoding.encodeNearbyEntityScalar(dz),
                        ObservationEncoding.encodeNearbyEntityScalar(entityHealth)
                    },
                    distanceSquared,
                    reference.getIndex()
                ));
            }
        });
        readings.sort(
            Comparator.comparingDouble(EntityReading::distanceSquared)
                .thenComparingInt(EntityReading::entityIndex)
        );
        List<int[]> encoded = new ArrayList<>(
            Math.min(MAX_NEARBY_ENTITIES, readings.size())
        );
        for (int i = 0; i < readings.size() && i < MAX_NEARBY_ENTITIES; i++) {
            encoded.add(readings.get(i).encoded());
        }
        return List.copyOf(encoded);
    }

    public static TargetState captureTargetState(
        Ref<EntityStore> targetRef,
        Store<EntityStore> store,
        Vector3d agentPosition
    ) {
        Ref<EntityStore> reference = targetRef;
        if (reference == null || !reference.isValid()) return TargetState.absent();
        NPCEntity entity = store.getComponent(reference, NPCEntity.getComponentType());
        TransformComponent transform = store.getComponent(
            reference,
            TransformComponent.getComponentType()
        );
        if (entity == null || transform == null) return TargetState.absent();
        double health = entityHealth(store, reference, entity);
        EntityStatMap stats = store.getComponent(
            reference,
            EntityStatsModule.get().getEntityStatMapComponentType()
        );
        EntityStatValue healthStat = stat(stats, "Health");
        Role role = entity.getRole();
        double maxHealth = healthStat == null
            ? (role == null ? health : role.getInitialMaxHealth())
            : healthStat.getMax();
        Vector3d targetPosition = transform.getPosition();
        Velocity velocity = store.getComponent(reference, Velocity.getComponentType());
        double distance = Math.hypot(
            targetPosition.x - agentPosition.x,
            targetPosition.z - agentPosition.z
        );
        double yawDegrees = normalizeDegrees(
            Math.toDegrees(transform.getRotation().yaw())
        );
        HeadRotation head = store.getComponent(
            reference,
            HeadRotation.getComponentType()
        );
        double headYawDegrees = head == null
            ? yawDegrees
            : normalizeDegrees(Math.toDegrees(head.getRotation().yaw()));
        double headPitchDegrees = head == null
            ? normalizeDegrees(Math.toDegrees(transform.getRotation().pitch()))
            : normalizeDegrees(Math.toDegrees(head.getRotation().pitch()));
        double agentBearing = Math.toDegrees(Math.atan2(
            -(agentPosition.x - targetPosition.x),
            -(agentPosition.z - targetPosition.z)
        ));
        double facingError = normalizeDegrees(agentBearing - yawDegrees);
        return new TargetState(
            true,
            entity.getRoleName(),
            health,
            maxHealth,
            distance,
            targetPosition.x,
            targetPosition.y,
            targetPosition.z,
            yawDegrees,
            headYawDegrees,
            headPitchDegrees,
            facingError,
            velocity == null ? 0.0 : velocity.getX(),
            velocity == null ? 0.0 : velocity.getZ()
        );
    }

    public static String stateName(
        Store<EntityStore> store,
        Ref<EntityStore> reference
    ) {
        if (reference == null || !reference.isValid()) return "";
        StateSupport state = store.getComponent(reference, StateSupport.getComponentType());
        return state == null ? "" : state.getStateName();
    }

    public static String statusText(
        Store<EntityStore> store,
        Ref<EntityStore> reference
    ) {
        if (reference == null || !reference.isValid()) return "";
        StringBuilder text = new StringBuilder();
        for (var status : captureStatuses(store, reference).statuses()) {
            if (text.length() > 0) text.append(';');
            text.append(status.effectId()).append('=')
                .append(String.format(java.util.Locale.ROOT, "%.4f", status.remainingDurationSeconds()));
        }
        return text.toString();
    }
}
