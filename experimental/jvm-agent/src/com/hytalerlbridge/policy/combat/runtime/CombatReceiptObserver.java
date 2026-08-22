package com.hytalerlbridge.policy.combat.runtime;

import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.combat.bridge.BridgeCombatFacade;
import com.hytalerlbridge.policy.combat.bridge.BridgeDodgeCooldownStateSource;

/** Optional, read-only observer of the receipts returned by the combat facade. */
public interface CombatReceiptObserver {

    CombatReceiptObserver NONE = new CombatReceiptObserver() {};

    enum Phase {
        PREPARE,
        APPLY
    }

    /**
     * Observe one production-facade receipt after caller-owned state has been
     * updated. Implementations must never affect admission or retain gameplay
     * state used by the sink.
     */
    default void observeCombatReceipt(
        Phase phase,
        long worldTick,
        int actorSlot,
        ActionDecoder.Decoded action,
        boolean firstControlTick,
        BridgeCombatFacade.Receipt receipt,
        BridgeDodgeCooldownStateSource.State dodgeCooldownBefore,
        BridgeDodgeCooldownStateSource.State dodgeCooldownAfter
    ) {
    }
}
