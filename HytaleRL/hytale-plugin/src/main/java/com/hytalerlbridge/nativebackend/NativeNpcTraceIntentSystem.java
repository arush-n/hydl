package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.ArchetypeChunk;
import com.hypixel.hytale.component.CommandBuffer;
import com.hypixel.hytale.component.ComponentType;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.dependency.Dependency;
import com.hypixel.hytale.component.dependency.Order;
import com.hypixel.hytale.component.dependency.SystemDependency;
import com.hypixel.hytale.component.query.Query;
import com.hypixel.hytale.component.system.tick.EntityTickingSystem;
import com.hypixel.hytale.server.core.modules.interaction.system.InteractionSystems;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.systems.AvoidanceSystem;
import com.hypixel.hytale.server.npc.systems.SteeringSystem;
import java.util.Set;

/** Captures native intent after avoidance and before bridge/physics control. */
final class NativeNpcTraceIntentSystem extends EntityTickingSystem<EntityStore> {

    private final ComponentType<EntityStore, NativeNpcTraceMarker> markerType;
    private final Query<EntityStore> query;

    NativeNpcTraceIntentSystem(
        ComponentType<EntityStore, NativeNpcTraceMarker> markerType
    ) {
        this.markerType = markerType;
        this.query = Query.and(markerType, NPCEntity.getComponentType());
    }

    @Override
    public Query<EntityStore> getQuery() {
        return query;
    }

    @Override
    public Set<Dependency<EntityStore>> getDependencies() {
        return orderingDependencies();
    }

    static Set<Dependency<EntityStore>> orderingDependencies() {
        return Set.of(
            new SystemDependency<>(Order.AFTER, AvoidanceSystem.class),
            new SystemDependency<>(
                Order.AFTER,
                InteractionSystems.TickInteractionManagerSystem.class
            ),
            new SystemDependency<>(Order.BEFORE, NativeControlSystem.class),
            new SystemDependency<>(Order.BEFORE, SteeringSystem.class)
        );
    }

    @Override
    public boolean isParallel(int entityCount, int chunkCount) {
        return false;
    }

    @Override
    public void tick(
        float deltaTime,
        int index,
        ArchetypeChunk<EntityStore> chunk,
        Store<EntityStore> store,
        CommandBuffer<EntityStore> commandBuffer
    ) {
        NativeNpcTraceMarker marker = chunk.getComponent(index, markerType);
        NPCEntity npc = chunk.getComponent(index, NPCEntity.getComponentType());
        if (marker != null && marker.recorder() != null && npc != null) {
            marker.recorder().captureIntent(
                chunk.getReferenceTo(index),
                npc,
                store
            );
        }
    }
}
