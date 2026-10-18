package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.builtin.fallingblocks.FallingBlock;
import com.hypixel.hytale.builtin.fallingblocks.FallingBlockTickingSystem;
import com.hypixel.hytale.component.ArchetypeChunk;
import com.hypixel.hytale.component.CommandBuffer;
import com.hypixel.hytale.component.ComponentType;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.dependency.Dependency;
import com.hypixel.hytale.component.dependency.Order;
import com.hypixel.hytale.component.dependency.SystemDependency;
import com.hypixel.hytale.component.query.Query;
import com.hypixel.hytale.component.system.tick.EntityTickingSystem;
import com.hypixel.hytale.server.core.entity.entities.BlockEntity;
import com.hypixel.hytale.server.core.modules.entity.BlockEntitySystems;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.physics.component.PhysicsValues;
import com.hypixel.hytale.server.core.modules.physics.component.Velocity;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import java.util.Set;

/** Captures motion after block physics and before falling gravity/impact. */
final class NativeFallingBlockTraceSystem
    extends EntityTickingSystem<EntityStore> {

    private final ComponentType<EntityStore, NativeAgentMarker> markerType;
    private final Query<EntityStore> query;
    private final Set<Dependency<EntityStore>> dependencies = Set.of(
        new SystemDependency<>(Order.AFTER, BlockEntitySystems.Ticking.class),
        new SystemDependency<>(Order.BEFORE, FallingBlockTickingSystem.class)
    );

    NativeFallingBlockTraceSystem(
        ComponentType<EntityStore, NativeAgentMarker> markerType
    ) {
        this.markerType = markerType;
        this.query = Query.and(
            markerType,
            FallingBlock.getComponentType(),
            TransformComponent.getComponentType(),
            Velocity.getComponentType(),
            PhysicsValues.getComponentType(),
            BoundingBox.getComponentType(),
            BlockEntity.getComponentType()
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
        if (marker != null && marker.session() != null) {
            marker.session().captureFallingBlockBeforeGravity(
                chunk.getReferenceTo(index),
                chunk,
                index,
                deltaTime
            );
        }
    }
}
