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
import java.util.Set;

/** Supplies opt-in headless client sync before Hytale ticks interaction chains. */
final class NativeSyntheticClientSystem
    extends EntityTickingSystem<EntityStore> {

    private final ComponentType<EntityStore, NativeAgentMarker> markerType;
    private final Set<Dependency<EntityStore>> dependencies = Set.of(
        new SystemDependency<>(
            Order.BEFORE,
            InteractionSystems.TickInteractionManagerSystem.class
        )
    );

    NativeSyntheticClientSystem(
        ComponentType<EntityStore, NativeAgentMarker> markerType
    ) {
        this.markerType = markerType;
    }

    @Override
    public Query<EntityStore> getQuery() {
        return markerType;
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
        if (marker != null && marker.session() != null) {
            marker.session().prepareSyntheticClientTick(
                chunk.getReferenceTo(index),
                store,
                deltaTime,
                marker.actorId()
            );
        }
    }
}
