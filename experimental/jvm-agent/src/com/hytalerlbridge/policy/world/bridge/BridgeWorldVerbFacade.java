package com.hytalerlbridge.policy.world.bridge;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.policy.world.model.WorldVerbBinding;

/** Typed policy-side view of the bridge's stateless World-verb facade. */
public interface BridgeWorldVerbFacade {

    /**
     * Whether this adapter can bind the bridge's mandatory atomic candidate
     * evidence to each request.  A transport-only adapter must return false so
     * callers do not advertise World verbs that will necessarily be rejected.
     */
    default boolean policyEvidenceSupported() {
        return true;
    }

    ContextResult installContext(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        PlayerRef packetAnchor
    );

    ContextResult uninstallContext(
        Ref<EntityStore> actor,
        Store<EntityStore> store
    );

    Admission startFieldcraft(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        long requestId,
        String recipeId,
        String worldEpoch
    );

    Admission startBoundWorldVerb(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        long requestId,
        WorldVerbBinding binding,
        String worldEpoch
    );

    Lifecycle prepare(
        Execution execution,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        float deltaTime
    );

    Lifecycle poll(
        Execution execution,
        Ref<EntityStore> actor,
        Store<EntityStore> store
    );

    record ContextResult(boolean accepted, String rejectReason) {
        public ContextResult {
            rejectReason = rejectReason == null ? "" : rejectReason;
        }
    }

    /** Opaque bridge-owned lifecycle handle retained only by the actor marker. */
    record Execution(Object nativeHandle) {
        public Execution {
            if (nativeHandle == null) {
                throw new IllegalArgumentException(
                    "native World-verb handle cannot be null");
            }
        }
    }

    record Admission(
        boolean accepted,
        String rejectReason,
        Execution execution,
        Lifecycle lifecycle
    ) {
        public Admission {
            rejectReason = rejectReason == null ? "" : rejectReason;
            if (lifecycle == null || (accepted && execution == null)) {
                throw new IllegalArgumentException(
                    "invalid World-verb admission result");
            }
        }
    }

    record Lifecycle(
        boolean accepted,
        boolean pending,
        boolean started,
        boolean finished,
        boolean failed,
        long requestedTick,
        long startedTick,
        long finishedTick,
        boolean geometryChanged,
        boolean inventoryChanged,
        String blockIdBefore,
        String blockIdAfter,
        int sourceQuantityBefore,
        int sourceQuantityAfter,
        String rejectReason
    ) {
        public Lifecycle {
            blockIdBefore = blockIdBefore == null ? "" : blockIdBefore;
            blockIdAfter = blockIdAfter == null ? "" : blockIdAfter;
            rejectReason = rejectReason == null ? "" : rejectReason;
        }

        public static Lifecycle rejected(String reason) {
            return new Lifecycle(
                false, false, false, false, true,
                -1L, -1L, -1L, false, false,
                "", "", -1, -1, reason
            );
        }
    }
}
