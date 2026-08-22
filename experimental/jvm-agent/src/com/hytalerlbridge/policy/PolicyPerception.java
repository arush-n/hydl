package com.hytalerlbridge.policy;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;

/**
 * Supplies one NPC's observation and action mask from live server state.
 *
 * <p>{@code perception.LivePolicyPerception} implements the composition step:
 * one coherent evidence frame becomes the 8,271-column learner row and its
 * 99-bit legality mask. Public Server 0.5.7 component readers acquire actor,
 * movement, status, resource, and inventory state. Exact combat-lifecycle and
 * terrain evidence remain explicit provider boundaries; a missing provider
 * skips the tick rather than pretending unavailable evidence is zero. The
 * lifecycle provider is a stateless bridge read-through; action admission is
 * reported later by the sink that issued the request.
 *
 * <p>Both arrays must be in exactly the layout the network was trained on. A
 * mislabelled column does not throw -- it silently feeds the wrong number to a
 * trained weight, so the implementation should be certified against the JAX
 * reference the way {@code AssembleTest} certifies the assembler. See
 * {@code perception/DEV.md} for the layer boundary and current live gate.
 */
public interface PolicyPerception {

    /**
     * One indivisible view of an actor at one engine tick.
     *
     * <p>Observation and legality must be captured together. Acquiring them in
     * separate calls can observe a cooldown, target, or interaction transition
     * between calls and produce a combination that never existed in the server.
     */
    record Sample(
        float[] observation,
        boolean[] actionMask,
        WorldActionEvidence actionEvidence
    ) {
        /** Compatibility constructor for captures with no privileged bindings. */
        public Sample(float[] observation, boolean[] actionMask) {
            this(observation, actionMask, WorldActionEvidence.empty());
        }

        public Sample {
            if (observation == null || actionMask == null
                || actionEvidence == null) {
                throw new IllegalArgumentException(
                    "perception sample values cannot be null");
            }
            observation = observation.clone();
            actionMask = actionMask.clone();
        }

        @Override
        public float[] observation() {
            return observation.clone();
        }

        @Override
        public boolean[] actionMask() {
            return actionMask.clone();
        }
    }

    /**
     * Return one observation and mask from the same server snapshot.
     *
     * <p>Returning {@code null} skips the tick, leaving the NPC to its authored
     * behaviour rather than feeding the policy a partial or zero vector. An
     * all-zero observation is a valid input that produces confident nonsense.
     */
    Sample sample(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        int slot,
        float deltaTime
    );
}
