package com.hytalerlbridge.combat;

/** Policy-facing phase of an authored melee interaction chain. */
public enum CombatPhase {
    IDLE(0),
    WINDUP(1),
    SWEEP(2),
    RECOVERY(3),
    COOLDOWN(4);

    private final int code;

    CombatPhase(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }
}
