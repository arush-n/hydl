package com.hytalerlbridge.policy.combat.bridge;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;

/** Typed policy-side view of the bridge-owned combat actor controller. */
public interface BridgeCombatFacade {

    Binding bind();

    Receipt initialize(
        Execution execution,
        Ref<EntityStore> actor,
        NPCEntity npc,
        Store<EntityStore> store
    );

    Receipt prepare(
        Execution execution,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        float deltaTime,
        boolean externalRemoteClientActive
    );

    Receipt apply(
        Execution execution,
        Ref<EntityStore> actor,
        NPCEntity npc,
        Store<EntityStore> store,
        float deltaTime,
        boolean firstControlTick,
        boolean attackRequested,
        int abilitySlot,
        boolean guardHeld,
        int dodgeDirection,
        double requestedChargeSeconds,
        boolean externalRootActive,
        boolean externalRemoteClientActive
    );

    void close(
        Execution execution,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        boolean externalRemoteClientActive
    );

    /** Checkpoint-derived charge for one authored slot; zero if unbound. */
    double requestedChargeSeconds(int abilitySlot);

    record Execution(Object nativeHandle) {
        public Execution {
            if (nativeHandle == null) {
                throw new IllegalArgumentException(
                    "native combat handle cannot be null");
            }
        }
    }

    record Binding(
        boolean accepted,
        String rejectReason,
        Execution execution
    ) {
        public Binding {
            rejectReason = rejectReason == null ? "" : rejectReason;
            if (accepted != (execution != null)) {
                throw new IllegalArgumentException(
                    "invalid native combat binding result");
            }
        }

        public static Binding rejected(String reason) {
            return new Binding(false, reason, null);
        }
    }

    record Receipt(
        boolean available,
        String unavailableReason,
        boolean attackRequested,
        boolean attackAccepted,
        String attackInteractionId,
        String attackRejectReason,
        boolean abilityRequested,
        boolean abilityAccepted,
        String abilityInteractionId,
        String abilityRejectReason,
        boolean dodgeRequested,
        boolean dodgeAccepted,
        String dodgeInteractionId,
        String dodgeRejectReason,
        boolean guardRequested,
        boolean guardAccepted,
        String guardInteractionId,
        String guardRejectReason,
        boolean abilityStarted,
        boolean abilityFinished,
        boolean abilityFailed,
        boolean abilityRejectedBeforeStart,
        int abilitySlot,
        boolean dodgeStarted,
        boolean dodgeFinished,
        boolean dodgeFailed,
        boolean dodgeRejectedBeforeStart,
        int dodgeDirection,
        boolean guardStarted,
        boolean guardFinished,
        boolean abilityActive,
        boolean dodgeActive,
        boolean guardChainActive,
        boolean guardWieldingActive,
        int syntheticClientChains
    ) {
        public Receipt {
            unavailableReason = clean(unavailableReason);
            attackInteractionId = clean(attackInteractionId);
            attackRejectReason = clean(attackRejectReason);
            abilityInteractionId = clean(abilityInteractionId);
            abilityRejectReason = clean(abilityRejectReason);
            dodgeInteractionId = clean(dodgeInteractionId);
            dodgeRejectReason = clean(dodgeRejectReason);
            guardInteractionId = clean(guardInteractionId);
            guardRejectReason = clean(guardRejectReason);
        }

        public static Receipt unavailable(String reason) {
            return new Receipt(
                false, reason,
                false, false, "", "",
                false, false, "", "",
                false, false, "", "",
                false, false, "", "",
                false, false, false, false, -1,
                false, false, false, false, 0,
                false, false,
                false, false, false, false, 0
            );
        }

        private static String clean(String value) {
            return value == null ? "" : value;
        }
    }
}
