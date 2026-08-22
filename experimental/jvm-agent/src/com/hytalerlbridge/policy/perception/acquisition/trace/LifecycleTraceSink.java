package com.hytalerlbridge.policy.perception.acquisition.trace;

import com.hytalerlbridge.policy.perception.acquisition.CombatLifecycleEvidenceSource;

/** Optional, observation-only consumer of one same-tick lifecycle row. */
@FunctionalInterface
public interface LifecycleTraceSink {

    /** Observe diagnostics without changing the row consumed by the policy. */
    void observe(
        int policySlot,
        CombatLifecycleEvidenceSource.Evidence evidence
    );

    /** Default production sink when no explicit trace was requested. */
    static LifecycleTraceSink disabled() {
        return (policySlot, evidence) -> {};
    }
}
