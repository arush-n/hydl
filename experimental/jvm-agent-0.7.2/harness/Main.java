import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.ObservationAssembler;
import com.hytalerlbridge.policy.Policy;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Proof of concept: the JAX-trained network running in the JVM, fed an
 * exported JAX profile observation, checked against JAX.
 *
 * <p>Three questions, in order:
 * <ol>
 *   <li>can the JVM load weights trained in JAX? (no pickle, no npz reader)</li>
 *   <li>fed the exported policy-contract observation, does it compute the
 *       same logits?</li>
 *   <li>does it decode to the same legal action the JAX policy would take?</li>
 * </ol>
 *
 * <p>Agreement is asserted numerically. A forward pass with the GRU gates
 * transposed, or the mask applied in the wrong place, still produces
 * plausible-looking logits -- so "it ran" is not the bar.
 */
public final class Main {

    /** float32 round-trip and reassociated summation; not exact equality. */
    private static final float LOGIT_TOLERANCE = 2.0e-3f;
    private static final float CARRY_TOLERANCE = 1.0e-5f;

    public static void main(String[] args) throws Exception {
        Path directory = Paths.get(args.length > 0 ? args[0] : "case");
        Policy policy = Policy.load(directory);

        float[] observation = Policy.read(directory, "observation");
        boolean[] legal = Policy.readMask(directory, "action_mask");
        float[] expectedLogits = Policy.read(directory, "expected_logits");
        float[] expectedCarry = Policy.read(directory, "expected_recurrent");

        int nonZero = 0;
        for (float value : observation) {
            if (value != 0.0f) {
                nonZero++;
            }
        }
        int legalBits = 0;
        for (boolean bit : legal) {
            if (bit) {
                legalBits++;
            }
        }

        System.out.printf(
                "loaded  observation=%d encoder=%d recurrent=%d actions=%d%n",
                policy.observationSize, policy.encoderSize,
                policy.recurrentSize, policy.actionSize);
        System.out.printf("input   %d/%d columns non-zero, %d/%d actions legal%n",
                nonZero, observation.length, legalBits, legal.length);

        float[] carry = new float[policy.recurrentSize];
        long start = System.nanoTime();
        float[] logits = policy.step(observation, carry, legal);
        long elapsed = System.nanoTime() - start;
        float value = policy.value(carry);
        int[] factors = Policy.factors(logits);

        float worstLogit = 0.0f;
        int worstIndex = -1;
        for (int i = 0; i < logits.length; i++) {
            if (!legal[i]) {
                continue;  // both sides drive illegal logits to -1e9
            }
            float delta = Math.abs(logits[i] - expectedLogits[i]);
            if (delta > worstLogit) {
                worstLogit = delta;
                worstIndex = i;
            }
        }
        float worstCarry = 0.0f;
        for (int i = 0; i < carry.length; i++) {
            worstCarry = Math.max(worstCarry, Math.abs(carry[i] - expectedCarry[i]));
        }

        int[] expectedFactors = Policy.factors(expectedLogits);
        boolean factorsMatch = java.util.Arrays.equals(factors, expectedFactors);

        System.out.printf("forward %.3f ms%n", elapsed / 1.0e6);
        System.out.printf("carry   max |java - jax| = %.3e%n", worstCarry);
        System.out.printf("logits  max |java - jax| = %.3e (legal bits%s)%n",
                worstLogit, worstIndex < 0 ? "" : ", worst at " + worstIndex);
        System.out.printf("value   java=%.6f%n", value);
        System.out.printf("action  java=%s%n", java.util.Arrays.toString(factors));
        System.out.printf("        jax =%s%n", java.util.Arrays.toString(expectedFactors));

        boolean chosenLegal = true;
        int offset = 0;
        for (int head = 0; head < Policy.HEAD_SIZES.length; head++) {
            if (!legal[offset + factors[head]]) {
                chosenLegal = false;
                System.out.printf("  head %d chose an ILLEGAL value %d%n",
                        head, factors[head]);
            }
            offset += Policy.HEAD_SIZES[head];
        }

        boolean ok = factorsMatch
                && chosenLegal
                && worstLogit <= LOGIT_TOLERANCE
                && worstCarry <= CARRY_TOLERANCE;
        System.out.println();
        System.out.println(ok
                ? "PASS - JVM reproduces the JAX policy on exported profile input"
                : "FAIL - see deltas above");
        if (!ok) {
            System.exit(1);
        }
    }
}
