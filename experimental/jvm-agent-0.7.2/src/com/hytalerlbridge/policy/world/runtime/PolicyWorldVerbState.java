package com.hytalerlbridge.policy.world.runtime;

import com.hytalerlbridge.policy.world.bridge.BridgeWorldVerbFacade;

/** Caller-owned lifecycle state for one policy actor's active World verb. */
public final class PolicyWorldVerbState {

    private BridgeWorldVerbFacade.Execution execution;
    private BridgeWorldVerbFacade.Lifecycle lastLifecycle;
    private String verb = "";

    public synchronized boolean active() {
        return execution != null;
    }

    public synchronized BridgeWorldVerbFacade.Execution execution() {
        return execution;
    }

    public synchronized BridgeWorldVerbFacade.Lifecycle lastLifecycle() {
        return lastLifecycle;
    }

    public synchronized String verb() {
        return verb;
    }

    public synchronized void begin(
        String selectedVerb,
        BridgeWorldVerbFacade.Execution selectedExecution,
        BridgeWorldVerbFacade.Lifecycle lifecycle
    ) {
        if (active() || selectedVerb == null || selectedVerb.isBlank()
            || selectedExecution == null || lifecycle == null) {
            throw new IllegalStateException(
                "invalid or overlapping policy World-verb lifecycle");
        }
        verb = selectedVerb;
        execution = selectedExecution;
        lastLifecycle = lifecycle;
    }

    public synchronized void observe(
        BridgeWorldVerbFacade.Lifecycle lifecycle
    ) {
        if (lifecycle == null) {
            throw new IllegalArgumentException(
                "World-verb lifecycle cannot be null");
        }
        lastLifecycle = lifecycle;
    }

    /** Clear the opaque handle after the native context is uninstalled. */
    public synchronized void finish(
        BridgeWorldVerbFacade.Lifecycle lifecycle
    ) {
        observe(lifecycle);
        execution = null;
        verb = "";
    }

    /** Test/reset seam; active native contexts must be closed by their sink. */
    public synchronized void resetLocal() {
        if (execution != null) {
            throw new IllegalStateException(
                "cannot reset an active native World verb locally");
        }
        lastLifecycle = null;
        verb = "";
    }
}
