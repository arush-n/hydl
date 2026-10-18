package com.hytalerlbridge.policy.perception.acquisition;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.policy.perception.model.PerceptionFrame;

/** Acquires one coherent actor frame from the live server tick. */
@FunctionalInterface
public interface PerceptionEvidenceSource {

    /** Return {@code null} when required evidence is not atomically available. */
    PerceptionFrame capture(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        int slot,
        float deltaTime
    );

    /** Reason for the most recent null capture, for diagnostics only. */
    default String lastFailureReason() {
        return "unknown";
    }

    /** Whether the most recent null capture ends this actor's episode. */
    default boolean lastFailureTerminal() {
        return false;
    }
}
