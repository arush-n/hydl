package com.hytalerlbridge.policy.action;

import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;

/**
 * Optional constraint applied after inference and before factored action decode.
 *
 * <p>The policy still produces every logit and advances its recurrent carry.
 * A control may only remove actions from the evidence-derived legality mask;
 * it can never advertise an action the observation did not. This makes the
 * seam suitable for narrow live diagnostics without creating a second action
 * decoder or bypassing the production sinks.</p>
 */
public interface ActionSelectionControl {

    ActionSelectionControl NONE = new ActionSelectionControl() {};

    /**
     * Constrain one policy decision. The returned mask must be a subset of
     * {@code legal} and retain at least one legal value in every head.
     */
    default Plan plan(
        int actorSlot,
        float[] logits,
        boolean[] legal,
        WorldActionEvidence evidence
    ) {
        return Plan.pass(legal);
    }

    /** Observe the decoded decision selected from {@code plan}. */
    default void observe(
        int actorSlot,
        Plan plan,
        ActionDecoder.Decoded action
    ) {
    }

    /** Human-readable, read-only diagnostic state. */
    default String describe() {
        return "disabled";
    }

    /** One immutable selection plan. */
    record Plan(boolean[] actionMask, boolean constrained, String route) {
        public Plan {
            if (actionMask == null) {
                throw new IllegalArgumentException(
                    "action-selection mask cannot be null");
            }
            actionMask = actionMask.clone();
            route = route == null ? "" : route;
            if (!constrained && !route.isEmpty()) {
                throw new IllegalArgumentException(
                    "an unconstrained plan cannot name a route");
            }
            if (constrained && route.isEmpty()) {
                throw new IllegalArgumentException(
                    "a constrained plan must name its route");
            }
        }

        @Override
        public boolean[] actionMask() {
            return actionMask.clone();
        }

        public static Plan pass(boolean[] legal) {
            return new Plan(legal, false, "");
        }
    }
}
