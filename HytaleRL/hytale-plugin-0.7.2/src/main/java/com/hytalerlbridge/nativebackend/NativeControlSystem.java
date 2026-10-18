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

/** Replaces role-authored steering after behavior/avoidance and before native physics. */
final class NativeControlSystem extends EntityTickingSystem<EntityStore> {

    private final ComponentType<EntityStore, NativeAgentMarker> markerType;
    private final Query<EntityStore> query;
    private final Set<Dependency<EntityStore>> dependencies;

    NativeControlSystem(ComponentType<EntityStore, NativeAgentMarker> markerType) {
        this.markerType = markerType;
        this.query = Query.and(markerType, NPCEntity.getComponentType());
        this.dependencies = orderingDependencies();
    }

    static Set<Dependency<EntityStore>> orderingDependencies() {
        return Set.of(
            new SystemDependency<>(Order.AFTER, AvoidanceSystem.class),
            // Queue player-equivalent inputs only after this tick's interaction
            // drain. Otherwise registering an unrelated pre-interaction helper
            // can make the same chain execute one engine tick early.
            new SystemDependency<>(
                Order.AFTER,
                InteractionSystems.TickInteractionManagerSystem.class
            ),
            new SystemDependency<>(Order.BEFORE, SteeringSystem.class)
        );
    }

    @Override
    public Query<EntityStore> getQuery() {
        return query;
    }

    @Override
    public Set<Dependency<EntityStore>> getDependencies() {
        return dependencies;
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
        NativeAgentMarker marker = chunk.getComponent(index, markerType);
        NPCEntity npc = chunk.getComponent(index, NPCEntity.getComponentType());
        if (marker != null && marker.session() != null && npc != null) {
            marker.session().applyControl(
                chunk.getReferenceTo(index),
                npc,
                store,
                deltaTime,
                marker.actorId()
            );
        }
    }
}
