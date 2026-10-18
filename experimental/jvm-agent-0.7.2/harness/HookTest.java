import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.Policy;
import com.hytalerlbridge.policy.PolicyAgentMarker;
import com.hytalerlbridge.policy.PolicyRuntime;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;
import java.util.List;

/**
 * Tests the parts of the hook that do not need a running server.
 *
 * <p>The failure this is really guarding against is shared recurrent state. Two
 * NPCs backed by one {@link PolicyRuntime} must not share a GRU carry, and the
 * symptom of getting it wrong is invisible: the network still runs, still emits
 * legal actions, and simply conditions each NPC on another NPC's history. No
 * exception, no wrong shape, just subtly wrong behaviour that would be very hard
 * to attribute later.
 *
 * <p>Requires the Server jar on the classpath only because
 * {@link PolicyAgentMarker} implements the engine's {@code Component}; nothing
 * here starts a server or touches a world.
 */
public final class HookTest {

    public static void main(String[] args) throws IOException {
        Path weights = Path.of(args.length > 0 ? args[0] : "case");
        float[] observation = Policy.read(weights, "observation");
        boolean[] legal = Policy.readMask(weights, "action_mask");
        boolean[] scope = new boolean[legal.length];
        Arrays.fill(scope, true);
        PolicyRuntime runtime = PolicyRuntime.load(weights, scope);
        List<String> failures = new ArrayList<>();

        // -- 1. two NPCs, one runtime, independent memory --------------------
        PolicyAgentMarker first = new PolicyAgentMarker(runtime, 0);
        PolicyAgentMarker second = new PolicyAgentMarker(runtime, 1);
        if (first.carry() == second.carry()) {
            failures.add("two markers share the same carry array");
        }

        // A held candidate index must retain the identity it selected. Using
        // the next tick's table would silently reinterpret slot 0 as a new
        // recipe whenever inventory changes between decision ticks.
        PolicyAgentMarker bound = new PolicyAgentMarker(runtime, 10);
        String[] firstIds = new String[WorldActionEvidence.RECIPE_CAPACITY];
        firstIds[0] = "recipe-A";
        String[] nextIds = new String[WorldActionEvidence.RECIPE_CAPACITY];
        nextIds[0] = "recipe-B";
        ActionDecoder.Decoded candidateAction =
            runtime.decide(observation, bound.carry(), legal);
        var firstBound = bound.chooseDecision(
            4, candidateAction, new WorldActionEvidence(firstIds));
        var heldBound = bound.chooseDecision(
            4, candidateAction, new WorldActionEvidence(nextIds));
        if (!firstBound.fresh() || heldBound.fresh()
            || !heldBound.evidence().recipeId(0).equals("recipe-A")) {
            failures.add(
                "held decision reinterpreted its candidate against a newer table");
        }

        // Advance only the first, several times, so its carry is far from zero.
        for (int i = 0; i < 8; i++) {
            runtime.decide(observation, first.carry(), legal);
        }
        if (isZero(first.carry())) {
            failures.add("first marker's carry never moved; the test proves nothing");
        }
        if (!isZero(second.carry())) {
            failures.add("advancing one marker moved another marker's carry");
        }

        // Same input, same starting state => same decision. If the runtime held
        // state of its own, the second NPC's first tick would differ.
        PolicyAgentMarker fresh = new PolicyAgentMarker(runtime, 2);
        ActionDecoder.Decoded a = runtime.decide(observation, second.carry(), legal);
        ActionDecoder.Decoded b = runtime.decide(observation, fresh.carry(), legal);
        if (!a.equals(b)) {
            failures.add("two fresh markers disagreed on identical input: "
                + a + " vs " + b);
        }

        // -- 2. clone() must deep-copy the memory ----------------------------
        PolicyAgentMarker original = new PolicyAgentMarker(runtime, 3);
        runtime.decide(observation, original.carry(), legal);
        PolicyAgentMarker copy = original.clone();
        if (copy.carry() == original.carry()) {
            failures.add("clone() shared the carry array");
        }
        if (!Arrays.equals(copy.carry(), original.carry())) {
            failures.add("clone() did not preserve the carry contents");
        }
        runtime.decide(observation, copy.carry(), legal);
        if (Arrays.equals(copy.carry(), original.carry())) {
            failures.add("advancing a clone also advanced the original");
        }
        PolicyAgentMarker takeover = new PolicyAgentMarker(runtime, 30, true);
        if (!takeover.stepFrozenNpc() || !takeover.clone().stepFrozenNpc()) {
            failures.add("clone() lost the takeover one-tick motion flag");
        }
        if (new PolicyAgentMarker(runtime, 31).stepFrozenNpc()) {
            failures.add("ordinary markers unexpectedly step frozen NPCs");
        }
        takeover.terminate();
        if (takeover.active() || !takeover.terminated()
                || !takeover.clone().terminated()) {
            failures.add("terminal marker remained active or lost terminal state");
        }

        // -- 3. pose accumulation --------------------------------------------
        PolicyAgentMarker pose = new PolicyAgentMarker(runtime, 4);
        // Pitch must clamp: looking up forever must not roll the head over.
        for (int i = 0; i < 40; i++) {
            pose.applyLookDelta(0.0, 45.0);
        }
        if (pose.desiredPitch() > (float) (Math.PI / 2.0) + 1.0e-6f) {
            failures.add("pitch exceeded +pi/2: " + pose.desiredPitch());
        }
        for (int i = 0; i < 80; i++) {
            pose.applyLookDelta(0.0, -45.0);
        }
        if (pose.desiredPitch() < (float) (-Math.PI / 2.0) - 1.0e-6f) {
            failures.add("pitch exceeded -pi/2: " + pose.desiredPitch());
        }
        // Yaw must wrap rather than grow without bound.
        PolicyAgentMarker spin = new PolicyAgentMarker(runtime, 5);
        for (int i = 0; i < 100; i++) {
            spin.applyLookDelta(45.0, 0.0);
        }
        if (Math.abs(spin.desiredYaw()) > (float) Math.PI + 1.0e-6f) {
            failures.add("yaw did not wrap to (-pi, pi]: " + spin.desiredYaw());
        }
        // Eight 45-degree steps is exactly one turn, i.e. back to the start.
        PolicyAgentMarker turn = new PolicyAgentMarker(runtime, 6);
        for (int i = 0; i < 8; i++) {
            turn.applyLookDelta(45.0, 0.0);
        }
        if (Math.abs(turn.desiredYaw()) > 1.0e-5f) {
            failures.add("a full turn did not return to zero: " + turn.desiredYaw());
        }

        // -- 4. reset clears memory ------------------------------------------
        PolicyAgentMarker reused = new PolicyAgentMarker(runtime, 7);
        runtime.decide(observation, reused.carry(), legal);
        reused.applyLookDelta(30.0, 15.0);
        reused.reset();
        if (!isZero(reused.carry())) {
            failures.add("reset() left recurrent state behind");
        }
        if (reused.desiredYaw() != 0.0f || reused.desiredPitch() != 0.0f) {
            failures.add("reset() left pose behind");
        }

        // -- 5b. action repeat -----------------------------------------------
        // Two properties, and the second matters more than the first.
        //
        // The obvious one: with period n, a fresh decision is taken every nth
        // tick and repeated in between. Checked by reference identity rather
        // than equality, because two independently-produced decisions can be
        // equal by coincidence and that would hide a bug where nothing is
        // actually being held.
        //
        // The load-bearing one: the GRU carry must still advance on held ticks.
        // The policy was trained with no action repeat anywhere, so the memory
        // must see one step per environment tick regardless of how often the
        // action changes. Skipping the forward pass would be a cheap speed win
        // and a silent change to what the network is.
        for (int period : new int[] {1, 4, 7}) {
            PolicyAgentMarker held = new PolicyAgentMarker(runtime, 8);
            ActionDecoder.Decoded lastTaken = null;
            int carryAdvances = 0;
            int heldTicks = 0;
            for (int tick = 0; tick < 3 * period + 2; tick++) {
                float[] before = held.carry().clone();
                ActionDecoder.Decoded currentFresh =
                    runtime.decide(observation, held.carry(), legal);
                if (!Arrays.equals(before, held.carry())) {
                    carryAdvances++;
                }
                ActionDecoder.Decoded applied = held.chooseDecision(period, currentFresh);

                boolean decisionTick = tick % period == 0;
                if (decisionTick && applied != currentFresh) {
                    failures.add("period " + period + " tick " + tick
                        + ": expected the fresh decision, got a held one");
                }
                if (!decisionTick && applied != lastTaken) {
                    failures.add("period " + period + " tick " + tick
                        + ": expected the held decision, got a different one");
                }
                if (!decisionTick && applied == currentFresh && period > 1) {
                    failures.add("period " + period + " tick " + tick
                        + ": the fresh decision was applied on a held tick");
                }
                if (!decisionTick) {
                    heldTicks++;
                }
                lastTaken = applied;
            }
            int ticks = 3 * period + 2;
            // Without this the loop passes vacuously at any period by simply
            // never taking the hold branch.
            int expectedHeld = ticks - (ticks + period - 1) / period;
            if (heldTicks != expectedHeld) {
                failures.add("period " + period + ": exercised " + heldTicks
                    + " held ticks, expected " + expectedHeld);
            }
            if (carryAdvances != ticks) {
                failures.add("period " + period + ": carry advanced on "
                    + carryAdvances + " of " + ticks + " ticks; the network must "
                    + "run every tick even when its action is discarded");
            }
            if (held.holding(1)) {
                failures.add("period 1 must never report holding");
            }
        }

        // -- 5c. reset must clear the held action ----------------------------
        // Otherwise a reused NPC repeats the previous episode's last action for
        // up to period-1 ticks before its first fresh decision lands.
        PolicyAgentMarker recycled = new PolicyAgentMarker(runtime, 9);
        ActionDecoder.Decoded stale =
            runtime.decide(observation, recycled.carry(), legal);
        recycled.chooseDecision(8, stale);
        recycled.chooseDecision(8, stale);
        recycled.reset();
        ActionDecoder.Decoded afterReset =
            runtime.decide(observation, recycled.carry(), legal);
        if (recycled.chooseDecision(8, afterReset) != afterReset) {
            failures.add("reset() left a held action behind; the first tick of "
                + "the next episode repeated the previous one");
        }

        // -- 5. an inert marker must be inert --------------------------------
        PolicyAgentMarker inert = new PolicyAgentMarker();
        if (inert.active()) {
            failures.add("the registry's no-arg marker reported itself active");
        }

        System.out.printf("recurrent size : %d%n", runtime.recurrentSize());
        System.out.printf("observation    : %d%n", runtime.observationSize());
        System.out.printf("failures       : %d%n", failures.size());
        if (!failures.isEmpty()) {
            failures.forEach(f -> System.out.println("  " + f));
            System.out.println("FAIL");
            System.exit(1);
        }
        System.out.println("PASS - per-NPC state is private, poses bound, reset clears");
    }

    private static boolean isZero(float[] values) {
        for (float value : values) {
            if (value != 0.0f) {
                return false;
            }
        }
        return true;
    }
}
