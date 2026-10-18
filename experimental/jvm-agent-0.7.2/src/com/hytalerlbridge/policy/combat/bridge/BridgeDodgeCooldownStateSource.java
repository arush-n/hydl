package com.hytalerlbridge.policy.combat.bridge;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;

/** Read-only source for the engine-owned shared Dodge cooldown bit. */
public interface BridgeDodgeCooldownStateSource {

    BridgeDodgeCooldownStateSource NONE = (actor, store) ->
        State.unavailable("dodge_cooldown_state_source_not_configured");

    /** Capture current state without creating or advancing a cooldown. */
    State capture(Ref<EntityStore> actor, Store<EntityStore> store);

    record State(
        boolean available,
        boolean onCooldown,
        String unavailableReason
    ) {
        public State {
            unavailableReason = unavailableReason == null
                ? "" : unavailableReason.strip();
            if (available) unavailableReason = "";
        }

        public static State available(boolean onCooldown) {
            return new State(true, onCooldown, "");
        }

        public static State unavailable(String reason) {
            return new State(
                false,
                true,
                reason == null || reason.isBlank()
                    ? "dodge_cooldown_state_unavailable" : reason
            );
        }
    }
}
