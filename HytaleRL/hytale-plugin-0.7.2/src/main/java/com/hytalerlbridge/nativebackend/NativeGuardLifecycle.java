package com.hytalerlbridge.nativebackend;

/** Request-edge state for generation-qualified native guard telemetry. */
final class NativeGuardLifecycle {

    private boolean inputHeld;
    private boolean risingEdgePending;
    private boolean fallingEdgePending;
    private long requestGeneration;

    void observeInput(boolean held) {
        if (held == inputHeld) return;
        inputHeld = held;
        if (held) {
            requestGeneration = Math.incrementExact(requestGeneration);
            risingEdgePending = true;
            fallingEdgePending = false;
        } else {
            risingEdgePending = false;
            fallingEdgePending = true;
        }
    }

    boolean inputHeld() {
        return inputHeld;
    }

    long requestGeneration() {
        return requestGeneration;
    }

    boolean consumeRisingEdge() {
        boolean pending = risingEdgePending;
        risingEdgePending = false;
        return pending;
    }

    boolean consumeFallingEdge() {
        boolean pending = fallingEdgePending;
        fallingEdgePending = false;
        return pending;
    }

    void reset() {
        inputHeld = false;
        risingEdgePending = false;
        fallingEdgePending = false;
        requestGeneration = 0L;
    }

    static boolean terminalRetained(
        long requestGeneration,
        long interactionGeneration,
        long finishTick
    ) {
        return finishTick >= 0L
            && interactionGeneration >= 1L
            && interactionGeneration == requestGeneration;
    }
}
