import com.hytalerlbridge.policy.Policy;
import com.hytalerlbridge.policy.PolicyPerception;
import com.hytalerlbridge.policy.perception.ActionMaskAssembler;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** Offline checks for perception composition that do not need a live server. */
public final class PerceptionTest {

    public static void main(String[] args) throws IOException {
        Path parity = Path.of(args.length > 0 ? args[0] : "case/parity");
        List<String> failures = new ArrayList<>();
        int states = 0;

        float[] sourceObservation = {1.0f, 2.0f};
        boolean[] sourceMask = {true, false};
        PolicyPerception.Sample sample =
            new PolicyPerception.Sample(sourceObservation, sourceMask);
        sourceObservation[0] = 9.0f;
        sourceMask[0] = false;
        if (sample.observation()[0] != 1.0f || !sample.actionMask()[0]) {
            failures.add("perception sample retained caller-owned arrays");
        }
        float[] exposedObservation = sample.observation();
        boolean[] exposedMask = sample.actionMask();
        exposedObservation[0] = 8.0f;
        exposedMask[0] = false;
        if (sample.observation()[0] != 1.0f || !sample.actionMask()[0]) {
            failures.add("perception sample exposed its mutable arrays");
        }

        List<Path> directories;
        try (var stream = Files.list(parity)) {
            directories = stream.filter(Files::isDirectory)
                .sorted(Comparator.comparing(path -> path.getFileName().toString()))
                .toList();
        }
        for (Path state : directories) {
            boolean valid = Policy.readMask(state, "valid")[0];
            boolean[] expected = Policy.readMask(state, "action_mask");
            boolean[] actual = ActionMaskAssembler.assemble(
                new ActionMaskAssembler.Input(
                    valid,
                    Policy.readMask(state, "skill_action_mask"),
                    new boolean[ActionMaskAssembler.DOOR_INTENT_COUNT],
                    agentRow(Policy.readMask(state, "ability_legal"), 16),
                    Policy.readMask(state, "guard_action_mask")[0],
                    Policy.readMask(state, "dodge_action_mask"),
                    Policy.readMask(state, "jump_action_mask")[0],
                    true,
                    false,
                    new boolean[2],
                    Policy.readMask(state, "block_available")[0],
                    Policy.readMask(state, "block_candidate_mask")
                )
            );
            if (!Arrays.equals(expected, actual)) {
                failures.add(state.getFileName() + " action mask differs at "
                    + firstDifference(expected, actual));
            }
            states++;
        }

        // Invalid evidence must skip inference. Producing even a neutral mask
        // would still feed a fabricated observation into the recurrent policy.
        try {
            ActionMaskAssembler.assemble(
                new ActionMaskAssembler.Input(
                    false,
                    filled(9), filled(3), filled(16), true, filled(4), true,
                    true, true, filled(2), true, filled(16)
                )
            );
            failures.add("invalid evidence produced an action mask instead of failing closed");
        } catch (IllegalStateException expected) {
            // Expected: PolicyPerception converts this boundary into a null sample.
        }

        System.out.printf("parity states  : %d%n", states);
        System.out.printf("failures       : %d%n", failures.size());
        if (!failures.isEmpty()) {
            failures.forEach(failure -> System.out.println("  " + failure));
            System.out.println("FAIL");
            System.exit(1);
        }
        System.out.println("PASS - Java composes every 124-bit JAX action mask exactly");
    }

    private static boolean[] agentRow(boolean[] values, int width) {
        if (values.length != 2 * width) {
            throw new IllegalArgumentException("entity-major mask width drift");
        }
        return Arrays.copyOfRange(values, 0, width);
    }

    private static int firstDifference(boolean[] expected, boolean[] actual) {
        for (int index = 0; index < expected.length; index++) {
            if (expected[index] != actual[index]) {
                return index;
            }
        }
        return -1;
    }

    private static boolean[] filled(int width) {
        boolean[] result = new boolean[width];
        Arrays.fill(result, true);
        return result;
    }
}
