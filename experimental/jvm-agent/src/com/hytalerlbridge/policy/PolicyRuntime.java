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

    private PolicyRuntime(Policy policy) {
        this.policy = policy;
    }

    /** Load weights exported by {@code export_case.py}. */
    public static PolicyRuntime load(Path weights) throws IOException {
        return new PolicyRuntime(Policy.load(weights));
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

    /**
     * Run one tick of the policy and decode the result into an applicable
     * action. {@code carry} is advanced in place.
     *
     * @param observation the 8,271-column vector, assembled in the server's own
     *                    layout by {@link ObservationAssembler}
     * @param legal       the 99 per-logit legality bits derived from the same
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
     * <p>The original evidence-derived mask remains authoritative. A control
     * can remove legal alternatives for a diagnostic decision, but attempting
     * to open a masked alternative fails loudly. The recurrent policy still
     * advances exactly once on the original observation.</p>
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
        float[] logits = policy.step(observation, carry, legal);
        ActionSelectionControl.Plan plan = control.plan(
            actorSlot, logits, legal, evidence);
        boolean[] selectedMask = plan.actionMask();
        if (selectedMask.length != legal.length) {
            throw new IllegalArgumentException(
                "action control changed the policy-mask width");
        }
        float[] selectedLogits = logits.clone();
        for (int index = 0; index < selectedMask.length; index++) {
            if (selectedMask[index] && !legal[index]) {
                throw new IllegalArgumentException(
                    "action control opened an evidence-masked logit");
            }
            if (!selectedMask[index]) selectedLogits[index] = -1.0e9f;
        }
        ActionDecoder.Decoded action = ActionDecoder.decode(
            Policy.factors(selectedLogits), selectedMask);
        control.observe(actorSlot, plan, action);
        return new ControlledDecision(action, plan.constrained(), plan.route());
    }

    /** The critic's estimate for the current carry; diagnostics only. */
    public float value(float[] carry) {
        return policy.value(carry);
    }

    /** One decoded action and whether a diagnostic narrowed its legal set. */
    public record ControlledDecision(
        ActionDecoder.Decoded action,
        boolean constrained,
        String route
    ) {
        public ControlledDecision {
            if (action == null || route == null
                || (constrained && route.isEmpty())
                || (!constrained && !route.isEmpty())) {
                throw new IllegalArgumentException(
                    "invalid controlled policy decision");
            }
        }
    }
}
