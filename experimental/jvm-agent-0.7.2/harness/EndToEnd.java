import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.ObservationAssembler;
import com.hytalerlbridge.policy.Policy;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * The whole server-side chain, in Java: structured evidence to a chosen action.
 *
 * <p>Both halves are certified independently elsewhere -- {@link AssembleTest}
 * shows the assembler reproduces the 8,271-value observation exactly, and
 * {@link Main} shows the network reproduces JAX logits to ~1e-9 on a real
 * observation captured off the live server. This runs them back to back, which
 * is what the plugin would do once per NPC per tick.
 *
 * <p>What is still missing for a live NPC is upstream of here: turning raw
 * server state into the 37 structured fields. That is
 * {@code NativeActorEvidenceAssembler}, which still runs in Python.
 */
public final class EndToEnd {

    private static final String[] HEADS = {
        "base_action", "ability", "guard", "dodge", "jump", "world_move",
        "yaw", "pitch", "use", "block_trigger", "recipe", "block",
    };

    public static void main(String[] args) throws Exception {
        Path groups = Paths.get(args.length > 0 ? args[0] : "case/groups");
        Path weights = Paths.get(args.length > 1 ? args[1] : "case");

        long start = System.nanoTime();
        float[] observation = ObservationAssembler.assemble(groups);
        long assembled = System.nanoTime();

        Policy policy = Policy.load(weights);
        boolean[] legal = Policy.readMask(weights, "action_mask");
        float[] carry = new float[policy.recurrentSize];

        long inferStart = System.nanoTime();
        float[] logits = policy.step(observation, carry, legal);
        long inferEnd = System.nanoTime();

        int[] factors = Policy.factors(logits);
        float value = policy.value(carry);

        int nonZero = 0;
        for (float column : observation) {
            if (column != 0.0f) {
                nonZero++;
            }
        }

        System.out.printf("assembled  %d columns (%d non-zero) in %.3f ms%n",
                observation.length, nonZero, (assembled - start) / 1.0e6);
        System.out.printf("inference  %.3f ms%n", (inferEnd - inferStart) / 1.0e6);
        System.out.printf("value      %.6f%n%n", value);
        System.out.println("chosen action:");
        int offset = 0;
        for (int head = 0; head < HEADS.length; head++) {
            int size = Policy.HEAD_SIZES[head];
            System.out.printf("  %-14s %2d   (%d legal of %d)%n",
                    HEADS[head], factors[head], countLegal(legal, offset, size), size);
            offset += size;
        }
    }

    private static int countLegal(boolean[] mask, int offset, int size) {
        int legal = 0;
        for (int i = 0; i < size; i++) {
            if (mask[offset + i]) {
                legal++;
            }
        }
        return legal;
    }
}
