package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionContext;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;

/** Queues one authored root after applying native charge and interaction rules. */
public final class NativeInteractionQueue {

    private NativeInteractionQueue() {}

    public static InteractionChain queue(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        InteractionType type,
        RootInteraction root,
        double requestedChargeTime
    ) {
        InteractionManager manager =
            com.hytalerlbridge.nativebackend.support.InteractionSupport
                .interactionManager(ref, store);
        if (manager == null || type == null || root == null) return null;
        if (!NativeChargeSelection.apply(
            manager.getInteractionSimulationHandler(),
            requestedChargeTime
        )) {
            return null;
        }
        InteractionContext context = InteractionContext.forInteraction(
            manager,
            ref,
            type,
            store
        );
        InteractionChain chain = manager.initChain(
            type,
            context,
            root,
            false
        );
        /*
         * Match InteractionManager.tryStartChain while retaining the queued
         * execution boundary required by native evidence. applyRules examines
         * admitted chains and forks, deliberately excluding chainStartQueue;
         * it also performs authored interruption before this new root is
         * queued. Session-owned "combat busy" shortcuts cannot reproduce
         * Interrupting/InterruptedBy/Blocking or Dodge coexistence and are
         * therefore not an admission authority.
         */
        if (!manager.applyRules(context, chain.getChainData(), type, root)) {
            NativeChargeSelection.apply(
                manager.getInteractionSimulationHandler(),
                0.0
            );
            return null;
        }
        manager.queueExecuteChain(chain);
        return chain;
    }
}

