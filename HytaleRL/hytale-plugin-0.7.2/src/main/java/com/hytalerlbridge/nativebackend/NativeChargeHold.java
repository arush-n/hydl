package com.hytalerlbridge.nativebackend;

/**
 * The learner's held charge level: which ability slot is charging and what a
 * tick's request does to it.
 *
 * <p>A charged melee slot is held by sending the same slot with
 * {@code charge_held} on consecutive ticks. The first held tick starts the
 * charge; later ticks continue it (a held request never re-queues it); the
 * first tick without the level releases it. The chain and the synthetic client
 * stay in {@link NativeEnvironmentSession}; this class only holds the slot and
 * the decision, so the bookkeeping can be tested without a server.</p>
 */
final class NativeChargeHold {

    /** What one tick's request does to the held charge. */
    enum Decision {
        /** No hold is active and this request holds a slot: queue the charge. */
        START,
        /** The hold continues; do nothing (no re-queue, no charge reset). */
        CONTINUE,
        /** The hold ends on this tick: release the charge once. */
        RELEASE,
        /** No held level involved: the ordinary request path applies. */
        NONE
    }

    static final int NO_SLOT = -1;

    private int heldSlot = NO_SLOT;

    /**
     * Decide one tick. {@code requestedSlot} is the tick's ability slot, or -1 on
     * a continuation tick, where the held level alone carries the hold.
     */
    Decision decide(boolean chargeHeld, int requestedSlot) {
        if (heldSlot == NO_SLOT) {
            return chargeHeld && requestedSlot >= 0 ? Decision.START : Decision.NONE;
        }
        boolean continues = chargeHeld
            && (requestedSlot < 0 || requestedSlot == heldSlot);
        return continues ? Decision.CONTINUE : Decision.RELEASE;
    }

    /** Whether a charge is held in this slot (the learner stays legal in it). */
    boolean holds(int slot) {
        return heldSlot != NO_SLOT && heldSlot == slot;
    }

    int heldSlot() {
        return heldSlot;
    }

    void begin(int slot) {
        heldSlot = slot;
    }

    void end() {
        heldSlot = NO_SLOT;
    }
}
