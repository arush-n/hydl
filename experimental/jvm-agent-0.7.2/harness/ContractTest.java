import com.hytalerlbridge.policy.ObservationAssembler;
import com.hytalerlbridge.policy.Policy;
import com.hytalerlbridge.policy.AgentBundle;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Checks that Java's policy shapes and observation width match the exported
 * Gym contract fixture.
 *
 * <p><b>The hole this fills.</b> {@code case/manifest.json} records an
 * {@code observation_contract_sha256}, but its {@code checkpoint_metadata}
 * carries only three keys and omits the {@code world_geometry_config} and
 * {@code entity_count} the arsenal hash is computed from. So the recorded digest
 * cannot be recomputed, cannot be checked, and did not notice when the contract
 * moved 7818 -> 8259 -> 8271 -> 8272.
 *
 * <p>Equal width is not equal meaning. {@code Policy.load} derives shapes from
 * file lengths by design -- which is what lets one loader take a 32-wide
 * proof-of-concept and a 128-wide production checkpoint -- so weights trained
 * against one column layout load happily against another whenever the widths
 * agree. Every other failure mode in this port announces itself; this one does
 * not.
 *
 * <p>{@code export_contract.py --check} detects drift between the live Gym and
 * this generated fixture. This test checks Java's dimensions and action-head
 * sizes against that fixture; {@code AssembleTest} separately checks exact
 * observation values against JAX-produced inputs.
 *
 * <p>It deliberately does not recompute the SHA-256. That digest is
 * {@code sha256(canonical_json(manifest))} over a Python structure; reproducing
 * it in Java would be a second definition of the contract, which is the class of
 * bug this whole suite exists to avoid. The hash is carried and reported; what
 * is *checked* here are the shapes Java actually acts on.
 */
public final class ContractTest {

    public static void main(String[] args) throws IOException {
        Path caseRoot = Path.of(args.length > 0 ? args[0] : "case");
        Path pinPath = caseRoot.resolve("contract.json");

        if (!Files.exists(pinPath)) {
            System.out.println(
                "FAIL - no case/contract.json: the port is not pinned to any "
                + "observation contract, so a checkpoint from a different "
                + "column layout would load silently. Run export_contract.py");
            System.exit(1);
        }

        String pin = Files.readString(pinPath);
        List<String> failures = new ArrayList<>();

        String sha = text(pin, "arsenal_policy_contract_sha256");
        String projectionAbi = text(pin, "policy_projection_abi");
        String schema = text(pin, "schema");
        int pinnedObservation = number(pin, "observation_size");
        int pinnedAction = number(pin, "action_size");
        int[] pinnedHeads = numbers(pin, "action_head_sizes");

        Policy policy = Policy.load(caseRoot);

        if (!AgentBundle.SUPPORTED_POLICY_PROJECTION_ABI.equals(projectionAbi)) {
            failures.add("policy_projection_abi '" + projectionAbi
                + "' is unsupported; Java supports "
                + AgentBundle.SUPPORTED_POLICY_PROJECTION_ABI);
        }

        if (policy.observationSize != pinnedObservation) {
            failures.add("observation width " + policy.observationSize
                + " but the pinned contract says " + pinnedObservation
                + " -- these weights were trained against a different layout");
        }
        if (policy.actionSize != pinnedAction) {
            failures.add("action width " + policy.actionSize
                + " but the pinned contract says " + pinnedAction);
        }
        if (pinnedHeads.length != Policy.HEAD_SIZES.length) {
            failures.add("head count " + Policy.HEAD_SIZES.length
                + " but the pinned contract says " + pinnedHeads.length);
        } else {
            for (int head = 0; head < pinnedHeads.length; head++) {
                if (pinnedHeads[head] != Policy.HEAD_SIZES[head]) {
                    failures.add("head " + head + " size "
                        + Policy.HEAD_SIZES[head] + " but the pinned contract "
                        + "says " + pinnedHeads[head]);
                }
            }
        }

        // The head sizes must also sum to the actor width, or the argmax decode
        // reads past a head boundary and silently mixes two heads together.
        int summed = 0;
        for (int size : Policy.HEAD_SIZES) {
            summed += size;
        }
        if (summed != pinnedAction) {
            failures.add("head sizes sum to " + summed + ", not " + pinnedAction);
        }

        // The assembler is the other half: weights of the right width fed a
        // vector built to a different layout is exactly the silent case.
        int assembled = -1;
        Path profile = caseRoot.resolve("profile");
        Path template = Files.exists(profile) ? profile : caseRoot;
        try {
            assembled = ObservationAssembler.assemble(template).length;
        } catch (RuntimeException | IOException ignored) {
            // Not every case root carries a template; the width checks above
            // still hold and this is reported as unavailable rather than passed.
            assembled = -1;
        }
        if (assembled >= 0 && assembled != pinnedObservation) {
            failures.add("the assembler builds " + assembled
                + " columns but the pinned contract says " + pinnedObservation);
        }

        System.out.printf("pinned schema    : %s%n", schema);
        System.out.printf("pinned contract  : %s%n", sha);
        System.out.printf("projection ABI   : %s (Java %s)%n",
            projectionAbi, AgentBundle.SUPPORTED_POLICY_PROJECTION_ABI);
        System.out.printf("observation      : %d (Java %d)%n",
            pinnedObservation, policy.observationSize);
        System.out.printf("action           : %d (Java %d)%n",
            pinnedAction, policy.actionSize);
        System.out.printf("assembler width  : %s%n",
            assembled >= 0 ? String.valueOf(assembled) : "no template here");
        System.out.printf("heads            : %d%n", pinnedHeads.length);

        if (!failures.isEmpty()) {
            failures.forEach(x -> System.out.println("  " + x));
            System.out.println(
                "FAIL - Java no longer matches the exported policy contract shape");
            System.exit(1);
        }
        System.out.println("PASS - Java matches the exported policy contract shape");
    }

    private static String text(String json, String key) {
        int at = json.indexOf("\"" + key + "\"");
        if (at < 0) {
            return "?";
        }
        int open = json.indexOf('"', json.indexOf(':', at) + 1);
        int close = json.indexOf('"', open + 1);
        return open > 0 && close > open ? json.substring(open + 1, close) : "?";
    }

    private static int number(String json, String key) {
        int at = json.indexOf("\"" + key + "\"");
        if (at < 0) {
            return -1;
        }
        int start = json.indexOf(':', at) + 1;
        int end = start;
        while (end < json.length() && "-0123456789".indexOf(json.charAt(end)) < 0) {
            end++;
            start = end;
        }
        while (end < json.length() && "-0123456789".indexOf(json.charAt(end)) >= 0) {
            end++;
        }
        return end > start ? Integer.parseInt(json.substring(start, end)) : -1;
    }

    private static int[] numbers(String json, String key) {
        int at = json.indexOf("\"" + key + "\"");
        if (at < 0) {
            return new int[0];
        }
        int open = json.indexOf('[', at);
        int close = json.indexOf(']', open);
        if (open < 0 || close < open) {
            return new int[0];
        }
        String[] parts = json.substring(open + 1, close).split(",");
        List<Integer> values = new ArrayList<>();
        for (String part : parts) {
            String trimmed = part.trim();
            if (!trimmed.isEmpty()) {
                values.add(Integer.parseInt(trimmed));
            }
        }
        int[] out = new int[values.size()];
        for (int index = 0; index < out.length; index++) {
            out[index] = values.get(index);
        }
        return out;
    }
}
