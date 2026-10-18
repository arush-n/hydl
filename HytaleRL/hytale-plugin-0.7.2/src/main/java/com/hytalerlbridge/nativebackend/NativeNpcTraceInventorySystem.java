package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.ArchetypeChunk;
import com.hypixel.hytale.component.CommandBuffer;
import com.hypixel.hytale.component.ComponentType;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.query.Query;
import com.hypixel.hytale.component.system.EntityEventSystem;
import com.hypixel.hytale.server.core.event.events.ecs.InventoryChangeEvent;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;

/** Copies exact server inventory transactions for actively traced NPCs. */
final class NativeNpcTraceInventorySystem
    extends EntityEventSystem<EntityStore, InventoryChangeEvent> {

    private final ComponentType<EntityStore, NativeNpcTraceMarker> markerType;

    NativeNpcTraceInventorySystem(
        ComponentType<EntityStore, NativeNpcTraceMarker> markerType
    ) {
        super(InventoryChangeEvent.class);
        this.markerType = markerType;
    }

    @Override
    public Query<EntityStore> getQuery() {
        return markerType;
    }

    @Override
    public void handle(
        int index,
        ArchetypeChunk<EntityStore> chunk,
        Store<EntityStore> store,
        CommandBuffer<EntityStore> commandBuffer,
        InventoryChangeEvent event
    ) {
        NativeNpcTraceMarker marker = chunk.getComponent(index, markerType);
        if (marker != null && marker.recorder() != null) {
            marker.recorder().captureInventory(
                chunk.getReferenceTo(index),
                store,
                event
            );
        }
    }
}
