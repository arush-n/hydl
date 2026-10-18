package com.hytalerlbridge.policy.combat.runtime;

import com.hytalerlbridge.policy.combat.bridge.BridgeCombatFacade;

/** Caller-owned opaque handle and last native combat receipt for one NPC. */
public final class PolicyCombatState {

    private BridgeCombatFacade.Execution execution;
    private BridgeCombatFacade.Receipt receipt =
        BridgeCombatFacade.Receipt.unavailable("native_combat_not_bound");

    public BridgeCombatFacade.Execution execution() {
        return execution;
    }

    public BridgeCombatFacade.Receipt receipt() {
        return receipt;
    }

    public void begin(
        BridgeCombatFacade.Execution value,
        BridgeCombatFacade.Receipt initial
    ) {
        if (value == null || initial == null || !initial.available()) {
            throw new IllegalArgumentException(
                "native combat state requires an initialized handle");
        }
        execution = value;
        receipt = initial;
    }

    public void observe(BridgeCombatFacade.Receipt value) {
        if (value == null) {
            throw new IllegalArgumentException("native combat receipt is null");
        }
        receipt = value;
    }

    public void resetLocal() {
        execution = null;
        receipt = BridgeCombatFacade.Receipt.unavailable(
            "native_combat_not_bound");
    }
}
