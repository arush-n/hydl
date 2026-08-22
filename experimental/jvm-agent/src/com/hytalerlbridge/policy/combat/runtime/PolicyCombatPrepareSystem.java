package com.hytalerlbridge.policy.combat.runtime;

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

/** Supplies synthetic client rows immediately before native interaction drain. */
public final class PolicyCombatPrepareSystem
    extends EntityTickingSystem<EntityStore> {

    private final ComponentType<EntityStore, PolicyAgentMarker> markerType;
    private final CombatActionSink sink;
    private final Query<EntityStore> query;

    public PolicyCombatPrepareSystem(
        ComponentType<EntityStore, PolicyAgentMarker> markerType,
        CombatActionSink sink
    ) {
        this.markerType = markerType;
        this.sink = sink;
        query = Query.and(markerType);
    }

    @Override
    public Query<EntityStore> getQuery() {
        return query;
    }

    @Override
    public Set<Dependency<EntityStore>> getDependencies() {
        return Set.of(new SystemDependency<>(
            Order.BEFORE,
            InteractionSystems.TickInteractionManagerSystem.class
        ));
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
        PolicyAgentMarker marker = chunk.getComponent(index, markerType);
        if (marker == null || !marker.active()) return;
        sink.prepare(chunk.getReferenceTo(index), store, deltaTime, marker);
    }
}
