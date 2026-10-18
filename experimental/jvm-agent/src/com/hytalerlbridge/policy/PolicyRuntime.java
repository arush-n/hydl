package com.hytalerlbridge.policy;

import com.hytalerlbridge.policy.action.ActionSelectionControl;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Shared, immutable half of a policy-driven NPC: the weights and the decision.
 *
 * <p>Deliberately free of engine types so it can be exercised without a Hytale
 * server on the classpath. Everything that varies per NPC -- the GRU carry and
 * the accumulated look pose -- lives on {@link PolicyAgentMarker}, i.e. on the
 * entity, so two NPCs sharing one set of weights never share recurrent state.
 * Getting that wrong is silent: the network still runs and still emits
 * plausible actions, it just conditions each NPC on another NPC's history.
 *
 * <p>One instance can back every NPC of a role. It holds no mutable state, so
 * the fact that {@code PolicyControlSystem} runs single-threaded per world does
 * not have to be relied on here.
 */
public final class PolicyRuntime {

    private final Policy policy;
    private final boolean[] actionScope;

    private PolicyRuntime(Policy policy, boolean[] actionScope) {
        this.policy = policy;
        this.actionScope = actionScope.clone();
    }

    /** Load weights with the validated, bundle-declared action scope. */
    public static PolicyRuntime load(
        Path weights,
        boolean[] actionScope
    ) throws IOException {
        int expectedActions = java.util.Arrays.stream(Policy.HEAD_SIZES).sum();
        if (actionScope == null || actionScope.length != expectedActions) {
            throw new IllegalArgumentException(
                "policy_action_scope_bits width "
                    + (actionScope == null ? -1 : actionScope.length)
                    + " != supported action width " + expectedActions);
        }
        int offset = 0;
        for (int head = 0; head < Policy.HEAD_SIZES.length; head++) {
            int size = Policy.HEAD_SIZES[head];
            boolean anyAllowed = false;
            for (int choice = 0; choice < size; choice++) {
                anyAllowed |= actionScope[offset + choice];
            }
            if (!anyAllowed) {
                throw new IllegalArgumentException(
                    "policy_action_scope_bits leaves action head " + head
                        + " with no permitted choices");
            }
            offset += size;
        }
        Policy policy = Policy.load(weights);
        if (policy.actionSize != expectedActions) {
            throw new IllegalArgumentException(
                "policy_action_scope_bits width does not match loaded policy");
        }
        return new PolicyRuntime(policy, actionScope);
    }

    public int observationSize() {
        return policy.observationSize;
    }

    public int actionSize() {
        return policy.actionSize;
    }

    public int encoderSize() {
        return policy.encoderSize;
    }

    public int recurrentSize() {
        return policy.recurrentSize;
    }

    /** A zeroed GRU carry, the state an episode starts from. */
    public float[] newCarry() {
        return new float[policy.recurrentSize];
    }

    /** Intersect server legality with the policy choices fixed at training. */
    public boolean[] effectiveActionMask(boolean[] liveLegal) {
        if (liveLegal == null || liveLegal.length != actionScope.length) {
            throw new IllegalArgumentException(
                "live legal-action mask width does not match policy_action_scope_bits");
        }
        boolean[] effective = new boolean[actionScope.length];
        for (int index = 0; index < effective.length; index++) {
            effective[index] = liveLegal[index] && actionScope[index];
        }
        return effective;
    }

    /**
     * Run one tick of the policy and decode the result into an applicable
     * action. {@code carry} is advanced in place.
     *
     * @param observation the 8,272-column vector, assembled in the server's own
     *                    layout by {@link ObservationAssembler}
     * @param legal       the 124 per-logit legality bits derived from the same
     *                    evidence; masked-off logits are driven to -1e9 before
     *                    the argmax, and re-checked by {@link ActionDecoder}
     */
    public ActionDecoder.Decoded decide(
        float[] observation,
        float[] carry,
        boolean[] legal
    ) {
        return decide(
            observation,
            carry,
            legal,
            WorldActionEvidence.empty(),
            -1,
            ActionSelectionControl.NONE
        ).action();
    }

    /**
     * Run inference through an optional mask-narrowing control.
     *
     * <p>The effective mask is the intersection of the bundle's training-time
     * policy scope and the current evidence-derived server mask. A control can
     * remove further legal alternatives for a diagnostic decision, but
     * attempting to open a masked alternative fails loudly. The recurrent
     * policy still advances exactly once on the original observation.</p>
     */
    public ControlledDecision decide(
        float[] observation,
        float[] carry,
        boolean[] legal,
        WorldActionEvidence evidence,
        int actorSlot,
        ActionSelectionControl control
    ) {
        if (control == null || evidence == null) {
            throw new IllegalArgumentException(
                "action control and World evidence cannot be null");
        }
        boolean[] effectiveLegal = effectiveActionMask(legal);
        float[] logits = policy.step(observation, carry, effectiveLegal);
        ActionSelectionControl.Plan plan = control.plan(
            actorSlot, logits, effectiveLegal, evidence);
        boolean[] selectedMask = plan.actionMask();
        if (selectedMask.length != effectiveLegal.length) {
            throw new IllegalArgumentException(
                "action control changed the policy-mask width");
        }
        float[] selectedLogits = logits.clone();
        for (int index = 0; index < selectedMask.length; index++) {
            if (selectedMask[index] && !effectiveLegal[index]) {
                throw new IllegalArgumentException(
                    "action control opened a policy-scope-masked logit");
            }
            if (!selectedMask[index]) selectedLogits[index] = -1.0e9f;
        }
        int[] factors = Policy.factors(selectedLogits);
        ActionDecoder.Decoded action = ActionDecoder.decode(
            factors, selectedMask);
        control.observe(actorSlot, plan, action);
        return new ControlledDecision(
            action, plan.constrained(), plan.route(), factors, effectiveLegal);
    }

    /** The critic's estimate for the current carry; diagnostics only. */
    public float value(float[] carry) {
        return policy.value(carry);
    }

    /** One decoded action and whether a diagnostic narrowed its legal set. */
    public record ControlledDecision(
        ActionDecoder.Decoded action,
        boolean constrained,
        String route,
        int[] factors,
        boolean[] effectiveLegalMask
    ) {
        public ControlledDecision {
            if (action == null || route == null || factors == null
                || effectiveLegalMask == null
                || factors.length != Policy.HEAD_SIZES.length
                || (constrained && route.isEmpty())
                || (!constrained && !route.isEmpty())) {
                throw new IllegalArgumentException(
                    "invalid controlled policy decision");
            }
            factors = factors.clone();
            effectiveLegalMask = effectiveLegalMask.clone();
        }

        @Override
        public int[] factors() {
            return factors.clone();
        }

        @Override
        public boolean[] effectiveLegalMask() {
            return effectiveLegalMask.clone();
        }
    }
}
