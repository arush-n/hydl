package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.protocol.InteractionState;

/** Pure admission/terminal classifier for a queued native interaction chain. */
public final class NativeQueuedInteractionLifecycle {

    private NativeQueuedInteractionLifecycle() {}

    public record Observation(
        boolean admitted,
        boolean terminal,
        boolean rejectedBeforeStart
    ) {}

    public static Observation classify(
        boolean registered,
        long chainId,
        InteractionState state
    ) {
        boolean admitted = registered || chainId < 0L;
        return new Observation(
            admitted,
            admitted
                && (state != InteractionState.NotFinished || !registered),
            !admitted && state == InteractionState.Failed
        );
    }
}
