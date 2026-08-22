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
}
