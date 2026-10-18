import com.hytalerlbridge.policy.Policy;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Ticks many NPCs through one shared {@link Policy} from several threads.
 *
 * <p><b>Why this is not covered already.</b> {@code HookTest} proves two markers
 * never share a carry array, which is a different claim: it says the *state* is
 * partitioned, not that the *code path* is safe to enter concurrently. The
 * observed live run claimed 77 NPCs in one world, and an ECS may tick them in
 * parallel, so the shared {@code Policy} is entered many times at once.
 *
 * <p>Today {@code Policy} is genuinely re-entrant: every weight field is
 * {@code private final} and read-only, {@code affine} allocates its output per
 * call, and the only mutation is the caller's own {@code carry}. So this test
 * passes on arrival. It is written anyway because that property is one commit
 * from being lost: allocating three arrays per tick per NPC is the most obvious
 * thing to optimise, and hoisting a scratch buffer into an instance field would
 * be silent. Two NPCs would interleave through the same scratch and still emit
 * perfectly legal actions -- each conditioned partly on the other's state.
 *
 * <p>The bar is bit-identity against a single-threaded reference, not "no
 * exception" and not "looks reasonable". Cross-talk through a shared buffer
 * corrupts values without ever throwing.
 *
 * <p>Each NPC is given a <em>different</em> observation sequence. Feeding them
 * all the same inputs would make cross-contamination invisible, because the
 * corrupted value and the correct one would be equal.
 */
public final class ConcurrencyTest {

    private static final int NPCS = 32;
    private static final int STEPS = 60;
    private static final int THREADS = 8;

    public static void main(String[] args) throws Exception {
        Path caseRoot = Path.of(args.length > 0 ? args[0] : "case");
        Path parityRoot = Path.of(args.length > 1 ? args[1] : "case/parity");

        Policy policy = Policy.load(caseRoot);
        boolean[] sharedLegal = Policy.readMask(caseRoot, "action_mask");

        List<Path> states = new ArrayList<>();
        try (var entries = Files.list(parityRoot)) {
            for (Path path : entries.sorted().toList()) {
                if (Files.isDirectory(path)
                    && Files.exists(path.resolve("expected_observation.bin"))) {
                    states.add(path);
                }
            }
        }
        if (states.size() < 2) {
            System.out.println("SKIP - need parity states; run export_parity.py");
            return;
        }

        // Preload so the threads do no I/O: a filesystem cache effect would
        // look like a timing-dependent result and muddy the diagnosis.
        int count = states.size();
        float[][] observations = new float[count][];
        boolean[][] masks = new boolean[count][];
        for (int index = 0; index < count; index++) {
            observations[index] = Policy.read(states.get(index), "expected_observation");
            masks[index] = Files.exists(states.get(index).resolve("action_mask.bin"))
                ? Policy.readMask(states.get(index), "action_mask")
                : sharedLegal;
        }

        List<String> failures = new ArrayList<>();

        float[][] reference = new float[NPCS][];
        float[][] referenceLogits = new float[NPCS][];
        for (int npc = 0; npc < NPCS; npc++) {
            Result single = run(policy, npc, observations, masks);
            reference[npc] = single.carry();
            referenceLogits[npc] = single.logits();
        }

        // Distinct sequences must produce distinct carries, or the comparison
        // below could not tell a corrupted NPC from a correct one.
        int distinct = 0;
        for (int npc = 1; npc < NPCS; npc++) {
            if (!java.util.Arrays.equals(reference[npc], reference[0])) {
                distinct++;
            }
        }
        if (distinct < NPCS - 1) {
            failures.add((NPCS - 1 - distinct) + " NPCs produced the same carry as "
                + "NPC 0 -- their inputs are not distinct enough for cross-talk "
                + "to be visible");
        }

        ExecutorService pool = Executors.newFixedThreadPool(THREADS);
        int rounds = 3;
        int mismatched = 0;
        try {
            for (int round = 0; round < rounds; round++) {
                List<Future<Result>> futures = new ArrayList<>();
                for (int npc = 0; npc < NPCS; npc++) {
                    final int id = npc;
                    Callable<Result> task = () -> run(policy, id, observations, masks);
                    futures.add(pool.submit(task));
                }
                for (int npc = 0; npc < NPCS; npc++) {
                    Result got = futures.get(npc).get(120, TimeUnit.SECONDS);
                    if (!java.util.Arrays.equals(got.carry(), reference[npc])) {
                        mismatched++;
                        if (failures.size() < 8) {
                            failures.add("round " + round + " NPC " + npc
                                + ": carry differs from the single-threaded "
                                + "reference -- state leaked between threads");
                        }
                    }
                    if (!java.util.Arrays.equals(got.logits(), referenceLogits[npc])) {
                        mismatched++;
                        if (failures.size() < 8) {
                            failures.add("round " + round + " NPC " + npc
                                + ": logits differ from the single-threaded "
                                + "reference");
                        }
                    }
                }
            }
        } finally {
            pool.shutdownNow();
        }

        System.out.printf("npcs             : %d x %d steps%n", NPCS, STEPS);
        System.out.printf("threads          : %d, %d rounds%n", THREADS, rounds);
        System.out.printf("distinct carries : %d/%d (inputs differ per NPC)%n",
            distinct + 1, NPCS);
        System.out.printf("mismatches       : %d%n", mismatched);

        if (!failures.isEmpty()) {
            failures.forEach(x -> System.out.println("  " + x));
            System.out.println(
                "FAIL - concurrent ticking does not match the single-threaded result");
            System.exit(1);
        }
        System.out.println(
            "PASS - one Policy ticks " + NPCS + " NPCs across " + THREADS
            + " threads bit-identically to single-threaded");
    }

    private record Result(float[] carry, float[] logits) { }

    /**
     * One NPC's whole rollout, returning the <em>entire</em> carry trajectory
     * rather than the final state.
     *
     * <p>Comparing only the final carry is not sensitive enough, and that is not
     * a hypothetical: {@code RecurrentTest} established that this GRU is
     * contractive, so a transient corruption decays toward the same attractor
     * and the endpoint converges back to the correct value. An early race would
     * be silently forgiven. Every intermediate step is therefore part of the
     * comparison.
     *
     * <p>The offset gives each NPC a different slice of the observation cycle,
     * so their carries diverge immediately.
     */
    private static Result run(
        Policy policy, int npc, float[][] observations, boolean[][] masks
    ) {
        float[] carry = new float[policy.recurrentSize];
        float[] trajectory = new float[STEPS * policy.recurrentSize];
        float[] logits = null;
        for (int step = 0; step < STEPS; step++) {
            int index = (npc * 7 + step) % observations.length;
            logits = policy.step(observations[index], carry, masks[index]);
            System.arraycopy(
                carry, 0, trajectory, step * policy.recurrentSize,
                policy.recurrentSize);
        }
        return new Result(trajectory, logits);
    }
}
