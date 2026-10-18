import com.hytalerlbridge.policy.Policy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Runs the GRU for 900 steps and checks it has not drifted from JAX.
 *
 * <p><b>What every other test misses.</b> {@code Main} and {@code EndToEnd} step
 * once. All 52 {@code ParityTest} states are evaluated from a *fresh zero
 * carry*. {@code HookTest} advances a carry but only asserts that it is private
 * and bounded, never that it is correct. {@code Bench} measures throughput. So
 * the recurrent path has never been compared against JAX past step 1.
 *
 * <p>That is the regime a deployed NPC actually lives in. It runs for thousands
 * of ticks with a warm carry, and the carry feeds itself: a per-step difference
 * of 1.6e-09 is harmless once and unquantified after 900 ticks. The GRU gate
 * order here is also non-standard, and a transposed gate can be nearly
 * invisible from a zero carry -- where the reset gate has little to act on --
 * while diverging steadily once the state is warm.
 *
 * <p>The observations are the 52 real parity states cycled, not one state
 * repeated: a constant input drives the GRU to a fixed point where error stops
 * accumulating, which would make a long run prove less than a short one.
 *
 * <p>Passing is not just "under tolerance at step 900". The report bins worst
 * delta by decile so <em>growth</em> is visible even while every step is within
 * tolerance -- a trend that is flat is a different claim from one that is
 * climbing and merely has not arrived yet.
 */
public final class RecurrentTest {

    /** Same float32 tolerance the other derivation tests use. */
    private static final float TOLERANCE = 1.0e-6f;

    public static void main(String[] args) throws IOException {
        Path caseRoot = Path.of(args.length > 0 ? args[0] : "case");
        Path parityRoot = Path.of(args.length > 1 ? args[1] : "case/parity");
        Path recurrentRoot = Path.of(args.length > 2 ? args[2] : "case/recurrent");

        if (!Files.exists(recurrentRoot.resolve("steps.txt"))) {
            System.out.println(
                "SKIP - no recurrent fixture; run export_recurrent.py first");
            return;
        }

        Policy policy = Policy.load(caseRoot);
        boolean[] sharedLegal = Policy.readMask(caseRoot, "action_mask");

        List<String> order = new ArrayList<>();
        for (String line : Files.readAllLines(recurrentRoot.resolve("steps.txt"))) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] parts = line.split("\t");
            if (parts.length != 2) {
                System.out.println("FAIL - steps.txt row has " + parts.length
                    + " fields, expected 2");
                System.exit(1);
            }
            order.add(parts[1]);
        }

        int steps = order.size();
        int actions = policy.actionSize;
        int hidden = policy.recurrentSize;

        float[] jaxLogits = Policy.read(recurrentRoot, "logits");
        float[] jaxCarry = Policy.read(recurrentRoot, "carry");
        float[] jaxValue = Policy.read(recurrentRoot, "value");
        if (jaxLogits.length != steps * actions
            || jaxCarry.length != steps * hidden
            || jaxValue.length != steps) {
            System.out.println("FAIL - fixture shape mismatch: logits "
                + jaxLogits.length + " carry " + jaxCarry.length
                + " value " + jaxValue.length + " for " + steps + " steps");
            System.exit(1);
        }

        int buckets = 10;
        float[] worstByBucket = new float[buckets];
        float[] worstCarryByBucket = new float[buckets];
        List<String> failures = new ArrayList<>();

        float[] carry = new float[hidden];
        float worstLogit = 0.0f;
        int worstLogitStep = -1;
        float worstCarry = 0.0f;
        float worstValue = 0.0f;
        float carryExtreme = 0.0f;

        // steps.txt names the parity states this fixture was generated against.
        // A re-export that adds or drops ticks leaves those names dangling, and
        // reading straight through produced a bare NoSuchFileException with no
        // verdict line -- which reads like a broken test rather than a stale
        // fixture. Name the cause and the fix instead.
        List<String> dangling = new ArrayList<>();
        for (String name : order) {
            if (!Files.exists(parityRoot.resolve(name)
                    .resolve("expected_observation.bin"))) {
                dangling.add(name);
            }
        }
        if (!dangling.isEmpty()) {
            System.out.println("FAIL - the recurrent fixture references "
                + dangling.size() + " parity state(s) that no longer exist: "
                + dangling.subList(0, Math.min(6, dangling.size()))
                + (dangling.size() > 6 ? " ..." : "")
                + " -- case/recurrent was generated against a different parity "
                + "export. If one is still running, wait for it; then re-run "
                + "export_recurrent.py to regenerate against the current states");
            System.exit(1);
        }

        for (int step = 0; step < steps; step++) {
            Path state = parityRoot.resolve(order.get(step));
            float[] observation = Policy.read(state, "expected_observation");
            boolean[] legal = Files.exists(state.resolve("action_mask.bin"))
                ? Policy.readMask(state, "action_mask")
                : sharedLegal;

            float[] logits = policy.step(observation, carry, legal);
            float value = policy.value(carry);
            int bucket = Math.min(buckets - 1, step * buckets / steps);

            for (int index = 0; index < actions; index++) {
                // Both sides drive illegal logits to -1e9; comparing them
                // measures the sentinel, not the network.
                if (!legal[index]) {
                    continue;
                }
                float delta = Math.abs(logits[index] - jaxLogits[step * actions + index]);
                if (delta > worstByBucket[bucket]) {
                    worstByBucket[bucket] = delta;
                }
                if (delta > worstLogit) {
                    worstLogit = delta;
                    worstLogitStep = step;
                }
                // Scaled for the SAME reason `value` is, below: a logit is a
                // pre-softmax activation and is not bounded by anything. The
                // note below this loop claimed logits were "softmax-bounded" --
                // softmax is applied after them, not to them. On the current
                // contract they run |logit| ~33-43, where an absolute 1e-6 is
                // finer than float32 can represent: one ULP at 33 is 3.9e-06,
                // and the observed worst delta is 3.815e-06, i.e. ~1 ULP.
                //
                // The CARRY check below stays absolute, and still passes at
                // 7.7e-07. Carry really is tanh-bounded to [-1, 1], so it is
                // the control that shows this is a miscalibrated gate rather
                // than a tolerance loosened until the suite turned green.
                float logitScale = Math.max(1.0f, Math.abs(jaxLogits[step * actions + index]));
                if (delta > TOLERANCE * logitScale && failures.size() < 12) {
                    failures.add(String.format(
                        "step %d %s logit %d: expected %.9f got %.9f "
                            + "(delta %.3e, relative %.3e)",
                        step, order.get(step), index,
                        jaxLogits[step * actions + index], logits[index], delta,
                        delta / logitScale));
                }
            }
            for (int index = 0; index < hidden; index++) {
                float expected = jaxCarry[step * hidden + index];
                float delta = Math.abs(carry[index] - expected);
                if (delta > worstCarryByBucket[bucket]) {
                    worstCarryByBucket[bucket] = delta;
                }
                if (delta > worstCarry) {
                    worstCarry = delta;
                }
                carryExtreme = Math.max(carryExtreme, Math.abs(carry[index]));
                if (delta > TOLERANCE && failures.size() < 12) {
                    failures.add(String.format(
                        "step %d carry %d: expected %.9f got %.9f (delta %.3e)",
                        step, index, expected, carry[index], delta));
                }
            }
            float valueDelta = Math.abs(value - jaxValue[step]);
            worstValue = Math.max(worstValue, valueDelta);
            // Carry is tanh-bounded, so a fixed absolute tolerance is the right
            // check for it. Logits are NOT (see the scaling above). Value is
            // unbounded too: on a
            // trained checkpoint |value| runs ~4.5 against the shipped
            // fixture's ~0.157, and a float32 result at that magnitude breaches
            // an absolute 1e-6 while still being ~5e-07 RELATIVE -- the same
            // precision the passing quantities show. Scale by magnitude, with
            // the absolute tolerance kept as the floor for values near zero.
            float valueScale = Math.max(1.0f, Math.abs(jaxValue[step]));
            if (valueDelta > TOLERANCE * valueScale && failures.size() < 12) {
                failures.add(String.format(
                    "step %d value: expected %.9f got %.9f "
                        + "(delta %.3e, relative %.3e)",
                    step, jaxValue[step], value, valueDelta,
                    valueDelta / valueScale));
            }
        }

        System.out.printf("steps            : %d over %d distinct observations%n",
            steps, order.stream().distinct().count());
        System.out.printf("worst logit      : %.3e at step %d%n",
            worstLogit, worstLogitStep);
        System.out.printf("worst carry      : %.3e%n", worstCarry);
        System.out.printf("worst value      : %.3e%n", worstValue);
        System.out.printf("carry magnitude  : max |c| = %.4f%n", carryExtreme);
        System.out.printf("tolerance        : %.1e%n", TOLERANCE);
        System.out.println("drift by decile  (worst logit / worst carry):");
        for (int bucket = 0; bucket < buckets; bucket++) {
            System.out.printf("  steps %4d-%4d   %.3e   %.3e%n",
                bucket * steps / buckets, (bucket + 1) * steps / buckets - 1,
                worstByBucket[bucket], worstCarryByBucket[bucket]);
        }

        // Anti-vacuity. A carry pinned near zero would make late-step agreement
        // meaningless -- it would be testing that zero equals zero.
        if (carryExtreme < 1.0e-3f) {
            failures.add(String.format(
                "carry never left zero (max |c| = %.3e) -- the recurrent path "
                + "was not exercised, so agreement proves nothing", carryExtreme));
        }
        // The point of a long run is the late steps. If the fixture somehow
        // ended up all-zero, the comparison above would pass trivially.
        boolean lateSignal = false;
        for (int index = 0; index < hidden; index++) {
            if (Math.abs(jaxCarry[(steps - 1) * hidden + index]) > 1.0e-6f) {
                lateSignal = true;
                break;
            }
        }
        if (!lateSignal) {
            failures.add("JAX carry is all-zero at the final step -- the "
                + "reference itself is degenerate");
        }

        if (!failures.isEmpty()) {
            failures.forEach(x -> System.out.println("  " + x));
            System.out.println("FAIL - Java's GRU diverges from JAX over a rollout");
            System.exit(1);
        }
        System.out.println(
            "PASS - Java's GRU tracks JAX for " + steps + " steps without drift");
    }
}
