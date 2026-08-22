package com.hytalerlbridge.policy.perception.acquisition.synthetic;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.policy.perception.acquisition.CombatLifecycleEvidenceSource;

/** Immutable server-free lifecycle source for composition and failure tests. */
public final class SyntheticCombatLifecycleEvidenceSource
    implements CombatLifecycleEvidenceSource {

    private final Evidence evidence;

    public SyntheticCombatLifecycleEvidenceSource(Evidence evidence) {
        if (evidence == null) {
            throw new IllegalArgumentException(
                "synthetic lifecycle evidence is required");
        }
        this.evidence = evidence;
    }

    @Override
    public Evidence capture(
        Ref<EntityStore> agent,
        Ref<EntityStore> target,
        Store<EntityStore> store,
        int policySlot,
        float deltaTime
    ) {
        // Evidence is deeply immutable: its constructor and array accessors
        // both copy. Returning the fixed fixture cannot leak mutable state.
        return evidence;
    }
}
