package com.hytalerlbridge.policy;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;

/**
 * Applies one decoded action to one NPC.
 *
 * <p>An interface because the bridge owns the certified interaction
 * implementation and it should not be duplicated. The tempting existing entry
 * point, {@code NativeEnvironmentSession.applyControl}, is <em>not</em> a
 * generic actor API: it returns without that session's {@code PendingStep},
 * validates session-owned actor refs, and consumes session guard/dodge/ability
 * state. A role-claimed NPC has none of those. Wrapping that method would be
 * dead code; moving the NPC into a fake session would violate the standalone
 * design.
 *
 * <p>Bridge-owned ref-based facades are extracted from the same primitives and
 * called by native sessions and composed policy sinks. Fieldcraft and combat
 * follow this pattern. See
 * {@code SteeringActionSink} for the locomotion-only subset reachable with
 * public engine APIs. Do not re-port interaction state machines here.
 */
public interface PolicyActionSink {

    /** Whether this sink has a typed executor for this decoded World verb. */
    default boolean supportsWorldVerb(ActionDecoder.Decoded action) {
        return false;
    }

    /**
     * @param deltaTime the tick delta the system was invoked with
     * @param marker    the NPC's marker, holding the accumulated look pose that
     *                  the action's yaw/pitch deltas apply to
     * @param action    the decoded action; callers must honour
     *                  {@link ActionDecoder.Decoded#actionLegal()} and drop
     *                  anything for which
     *                  {@link ActionDecoder.Decoded#requestsWorldVerb()} is true
     *                  but no resolved verb request is available
     * @param evidence  privileged candidate identities captured with the
     *                  observation that produced {@code action}
     * @param firstControlTick true only on the first tick of a discrete
     *                  decision; held locomotion remains continuous, while
     *                  edge-triggered verbs must not be re-issued
     */
    void apply(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        float deltaTime,
        PolicyAgentMarker marker,
        ActionDecoder.Decoded action,
        WorldActionEvidence evidence,
        boolean firstControlTick
    );
}
