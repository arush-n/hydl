import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.ObservationAssembler;
import com.hytalerlbridge.policy.Policy;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Steady-state inference cost, against the server's 30 TPS budget.
 *
 * <p>The production question is not whether the forward pass is fast in
 * isolation but whether it fits inside a tick alongside everything else the
 * server does. At 30 TPS a tick is 33.3 ms; this reports the warm per-call
 * cost and what fraction of one tick it consumes for a single NPC.
 */
public final class Bench {

    private static final double TICK_MS = 1000.0 / 30.0;

    public static void main(String[] args) throws Exception {
        Path directory = Paths.get(args.length > 0 ? args[0] : "case");
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : 20000;

        Policy policy = Policy.load(directory);
        float[] observation = Policy.read(directory, "observation");
        boolean[] legal = Policy.readMask(directory, "action_mask");
        float[] carry = new float[policy.recurrentSize];

        for (int i = 0; i < 5000; i++) {          // let the JIT settle
            policy.step(observation, carry, legal);
        }

        java.util.Arrays.fill(carry, 0.0f);
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            policy.step(observation, carry, legal);
        }
        double perCall = (System.nanoTime() - start) / 1.0e6 / iterations;

        System.out.printf("warm forward pass : %.4f ms%n", perCall);
        System.out.printf("one 30 TPS tick   : %.2f ms%n", TICK_MS);
        System.out.printf("cost per NPC      : %.3f%% of a tick%n",
                100.0 * perCall / TICK_MS);
        System.out.printf("NPCs per tick @50%%: ~%d%n",
                (int) (0.5 * TICK_MS / perCall));
    }
}
