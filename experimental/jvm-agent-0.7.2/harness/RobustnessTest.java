import com.hytalerlbridge.policy.Policy;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Pins what the network does with degenerate input, which was undefined.
 *
 * <p>Every other test feeds the policy well-formed observations JAX produced.
 * A live server will not always manage that: a division by zero in a perception
 * deriver, an uninitialised component read, a distance normalised by a zero
 * range. The interesting question is not whether that is *likely* but what
 * happens when it occurs, because the failure mode determines whether anyone
 * notices.
 *
 * <p>The network has no branch that can reject an input. Illegal logits are set
 * to {@code -1e9} and {@link Policy#factors} takes an argmax per head, so
 * *something* is always returned. The two properties worth pinning are
 * therefore:
 *
 * <ul>
 *   <li><b>Realistic inputs stay finite.</b> The observation is normalised to
 *       roughly [-1, 1]; a fully saturated one must not overflow. This is a
 *       genuine assertion and a failure here is a real defect.</li>
 *   <li><b>What a poisoned input actually does.</b> Characterised and printed
 *       rather than asserted, because the right *policy* response is a design
 *       decision for the perception layer, not something this class should
 *       silently impose. What matters is that the behaviour is written down and
 *       a future change to it is visible.</li>
 * </ul>
 */
public final class RobustnessTest {

    public static void main(String[] args) throws IOException {
        Path caseRoot = Path.of(args.length > 0 ? args[0] : "case");
        Policy policy = Policy.load(caseRoot);
        boolean[] legal = Policy.readMask(caseRoot, "action_mask");
        int n = policy.observationSize;
        List<String> failures = new ArrayList<>();

        // --- assertions: realistic inputs must stay finite ------------------
        check(policy, legal, "all zeros", new float[n], failures, true);
        check(policy, legal, "all +1 (saturated)", filled(n, 1.0f), failures, true);
        check(policy, legal, "all -1 (saturated)", filled(n, -1.0f), failures, true);
        check(policy, legal, "alternating +/-1", alternating(n), failures, true);
        check(policy, legal, "denormals", filled(n, Float.MIN_VALUE), failures, true);

        // --- characterisation: inputs that should never reach the network ---
        System.out.println();
        System.out.println("characterisation (not assertions -- these are inputs a");
        System.out.println("correct perception layer must never emit):");
        check(policy, legal, "one NaN column", oneOf(n, Float.NaN), failures, false);
        check(policy, legal, "one +Inf column", oneOf(n, Float.POSITIVE_INFINITY),
            failures, false);
        check(policy, legal, "all 1e6", filled(n, 1.0e6f), failures, false);
        check(policy, legal, "all FLT_MAX", filled(n, Float.MAX_VALUE), failures, false);

        // --- does a poisoned carry ever recover? ----------------------------
        System.out.println();
        float[] carry = new float[policy.recurrentSize];
        policy.step(oneOf(n, Float.NaN), carry, legal);
        int poisoned = countNonFinite(carry);
        float[] clean = new float[n];
        for (int step = 0; step < 200; step++) {
            policy.step(clean, carry, legal);
        }
        int stillPoisoned = countNonFinite(carry);
        System.out.printf(
            "carry poisoning  : %d/%d units non-finite after one NaN observation, "
            + "%d/%d still non-finite after 200 clean steps%n",
            poisoned, carry.length, stillPoisoned, carry.length);
        if (stillPoisoned > 0) {
            System.out.println(
                "                   -> permanent. NaN is absorbing in the GRU: the "
                + "carry never returns to a usable state, so one bad tick disables "
                + "that NPC for the rest of its life with no error raised.");
        }

        // --- what does an all-illegal head decode to? -----------------------
        boolean[] noneLegal = new boolean[policy.actionSize];
        float[] logits = policy.step(new float[n], new float[policy.recurrentSize],
            noneLegal);
        int[] factors = Policy.factors(logits);
        boolean allZero = true;
        for (int factor : factors) {
            if (factor != 0) {
                allZero = false;
                break;
            }
        }
        System.out.printf("all-illegal mask : decodes to %s%n",
            allZero ? "index 0 of every head" : java.util.Arrays.toString(factors));
        System.out.println(
            "                   -> an argmax over twelve equal -1e9 logits picks "
            + "index 0, which is a legal-looking factor tuple built entirely from "
            + "illegal choices. The mask contract (at least one legal logit per "
            + "head) is unenforced here.");

        System.out.println();
        if (!failures.isEmpty()) {
            failures.forEach(x -> System.out.println("  " + x));
            System.out.println("FAIL - a realistic input produced a non-finite result");
            System.exit(1);
        }
        System.out.println("PASS - realistic inputs stay finite; degenerate ones pinned");
    }

    /**
     * @param assertFinite when true a non-finite output is a failure; when false
     *                     the result is only reported, because the input is one
     *                     perception must never produce in the first place
     */
    private static void check(
        Policy policy, boolean[] legal, String label, float[] observation,
        List<String> failures, boolean assertFinite
    ) {
        float[] carry = new float[policy.recurrentSize];
        float[] logits = policy.step(observation, carry, legal);
        float value = policy.value(carry);
        int badLogits = countNonFinite(logits);
        int badCarry = countNonFinite(carry);
        boolean badValue = !Float.isFinite(value);

        System.out.printf("  %-22s logits %2d/%d non-finite, carry %2d/%d, value %s%n",
            label, badLogits, logits.length, badCarry, carry.length,
            badValue ? "non-finite" : String.format("%+.4f", value));

        if (assertFinite && (badLogits > 0 || badCarry > 0 || badValue)) {
            failures.add(label + ": produced non-finite output from a finite, "
                + "in-range observation -- " + badLogits + " logits and "
                + badCarry + " carry units");
        }
    }

    private static int countNonFinite(float[] values) {
        int count = 0;
        for (float value : values) {
            if (!Float.isFinite(value)) {
                count++;
            }
        }
        return count;
    }

    private static float[] filled(int size, float value) {
        float[] values = new float[size];
        java.util.Arrays.fill(values, value);
        return values;
    }

    private static float[] alternating(int size) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        }
        return values;
    }

    /** One poisoned column in an otherwise clean observation. */
    private static float[] oneOf(int size, float value) {
        float[] values = new float[size];
        values[size / 2] = value;
        return values;
    }
}
