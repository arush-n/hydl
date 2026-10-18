package com.hytalerlbridge.policy.perception.acquisition.trace;

import com.hytalerlbridge.policy.ActionDecoder;

/**
 * Optional observer of the standalone policy's native input/action boundary.
 *
 * <p>The observation and action mask are the coherent sample already acquired
 * by {@code PolicyPerception}. Implementations must not mutate them or throw
 * into the control system: tracing is evidence collection and must never
 * change the policy's behavior.</p>
 */
public interface PolicyTraceSink {

    /**
     * Record the action selected for this policy tick.
     *
     * @param inferredFactors raw greedy head indices from the current forward
     *     pass; on a held tick they describe the discarded fresh decision
     * @param action the action actually selected, which may be held
     */
    void observe(
        int policySlot,
        long worldTick,
        float deltaTime,
        float[] observation,
        boolean[] actionMask,
        int[] inferredFactors,
        ActionDecoder.Decoded action,
        boolean fresh,
        boolean constrained,
        String route
    );

    /** End a pending transition without pairing it with a later sample. */
    default void skip(
        int policySlot,
        long worldTick,
        String reason,
        boolean terminal
    ) {
    }

    /** Flush pending rows and release any recorder state. */
    default void close() {
    }

    /** Number of completed transition rows, or {@code -1} when unavailable. */
    default long transitionsWritten() {
        return -1L;
    }

    /** No-op production sink when tracing was not requested. */
    static PolicyTraceSink disabled() {
        return new PolicyTraceSink() {
            @Override
            public void observe(
                int policySlot,
                long worldTick,
                float deltaTime,
                float[] observation,
                boolean[] actionMask,
                int[] inferredFactors,
                ActionDecoder.Decoded action,
                boolean fresh,
                boolean constrained,
                String route
            ) {
            }
        };
    }
}
