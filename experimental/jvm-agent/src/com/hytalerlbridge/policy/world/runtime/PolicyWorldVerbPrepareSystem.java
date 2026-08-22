package com.hytalerlbridge.policy.world.runtime;

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
import com.hytalerlbridge.policy.PolicyAgentMarker;
import java.util.Set;

/** Supplies synthetic client data before Hytale drains interaction chains. */
public final class PolicyWorldVerbPrepareSystem
    extends EntityTickingSystem<EntityStore> {

    private final ComponentType<EntityStore, PolicyAgentMarker> markerType;
    private final WorldVerbActionSink sink;
    private final Set<Dependency<EntityStore>> dependencies = Set.of(
        new SystemDependency<>(
            Order.BEFORE,
            InteractionSystems.TickInteractionManagerSystem.class
        )
    );

    public PolicyWorldVerbPrepareSystem(
        ComponentType<EntityStore, PolicyAgentMarker> markerType,
        WorldVerbActionSink sink
    ) {
        this.markerType = markerType;
        this.sink = sink;
    }

    @Override public Query<EntityStore> getQuery() { return markerType; }

    @Override
    public Set<Dependency<EntityStore>> getDependencies() {
        return dependencies;
    }

    @Override public boolean isParallel(int entityCount, int chunkCount) {
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
        PolicyAgentMarker marker = chunk.getComponent(index, markerType);
        if (marker != null && marker.active() && marker.worldVerb().active()) {
            sink.prepare(
                chunk.getReferenceTo(index), store, deltaTime, marker);
        }
    }
}
