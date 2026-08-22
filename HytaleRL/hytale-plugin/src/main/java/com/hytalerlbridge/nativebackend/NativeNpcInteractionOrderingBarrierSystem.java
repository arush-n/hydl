package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.dependency.Dependency;
import com.hypixel.hytale.component.dependency.Order;
import com.hypixel.hytale.component.dependency.SystemDependency;
import com.hypixel.hytale.component.system.tick.TickingSystem;
import com.hypixel.hytale.server.core.modules.interaction.system.InteractionSystems;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.systems.RoleSystems;
import java.util.Set;

/**
 * Keeps role-authored interaction requests behind the current manager drain.
 *
 * <p>The system intentionally has no per-entity work. Its two dependencies
 * close an otherwise unspecified ECS edge: NPC role behavior may queue an
 * interaction, so it must run after the manager has drained this tick's
 * queue. Registering unrelated bridge systems must not pull that queue into
 * the current tick.
 */
final class NativeNpcInteractionOrderingBarrierSystem
    extends TickingSystem<EntityStore> {

    private final Set<Dependency<EntityStore>> dependencies =
        orderingDependencies();

    static Set<Dependency<EntityStore>> orderingDependencies() {
        return Set.of(
            new SystemDependency<>(
                Order.AFTER,
                InteractionSystems.TickInteractionManagerSystem.class
            ),
            new SystemDependency<>(
                Order.BEFORE,
                RoleSystems.BehaviourTickSystem.class
            )
        );
    }

    @Override
    public Set<Dependency<EntityStore>> getDependencies() {
        return dependencies;
    }

    @Override
    public void tick(
        float deltaTime,
        int systemIndex,
        Store<EntityStore> store
    ) {
        // Ordering barrier only.
    }
}
