package com.hytalerlbridge.policy;

import com.hytalerlbridge.policy.bundle.PolicyTensorIdentity;
import com.hytalerlbridge.policy.bundle.PolicyDeploymentCompatibility;
import com.hytalerlbridge.policy.bundle.BridgeTargetAttestation;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The contract a weights directory was trained against, and the check that it
 * matches the policy actually loaded.
 *
 * <p>Without this, a weights directory is just {@code .bin} files. A policy
 * trained on an older observation width loads cleanly, produces plausible
 * logits, and drives an NPC on a misread world -- no exception, no wrong shape,
 * nothing in a log. The only symptom is an NPC that behaves confidently and
 * wrongly, which is exactly the failure this project has spent the most effort
 * avoiding elsewhere.
 *
 * <p>Read from {@code contract.txt}, written by {@code adk.deploy.bundle}
 * beside the human-readable {@code agent.json}. Tab-separated rather than JSON
 * because this mod compiles against the server jar alone; pulling in a JSON
 * parser for this small scalar contract would be the only dependency in the
 * tree.
 *
 * <p>Old and hand-assembled directories fail closed.  A v1 manifest could
 * validate widths while same-shaped tensor bytes were swapped underneath it;
 * absence is therefore no longer treated as permission to run an unverified
 * policy. Export the bundle again to produce the v4 tensor, training, and
 * role/profile compatibility identity.
 */
public final class AgentBundle {

    public static final String SCHEMA = "hytalerl_policy_agent_bundle_v4";
    public static final String TRAINING_PROVENANCE_SCHEMA =
        "hytalerl_policy_training_provenance_v1";

    // Widths the current policy contract is pinned to. DERIVED, not copied:
    // both were hand-maintained literals and both went stale -- the action
    // width sat at 99 through two contract generations while `Policy.HEAD_SIZES`
    // had already moved to 124, so this gate would have refused every current
    // bundle for a reason that had nothing to do with the bundle.
    public static final int EXPECTED_OBSERVATION =
        ObservationAssembler.OBSERVATION_SIZE;
    public static final int EXPECTED_ACTIONS =
        java.util.Arrays.stream(Policy.HEAD_SIZES).sum();
    // Current Gym policy identity; regenerated with the Java projection.
    //
    // Re-keyed 2026-08-22 from E6D705C3... to the digest the Gym publishes now.
    // Unlike the widths above this cannot be derived on the Java side -- the
    // digest is computed by the Python contract -- so it is the one pin here
    // that must be updated by hand when the contract moves. Read the current
    // value from `current_combat_checkpoint_contract()["observation_contract_sha256"]`
    // or from any freshly exported bundle's `agent.json`; a stale value makes
    // the plugin refuse every current bundle and report itself inert.
    public static final String EXPECTED_POLICY_CONTRACT =
        "1F21B0488BFA1A14A217460A339FBAAB4665EF070AF0982BA35BBF6E64CE6497";

    private final Map<String, String> fields;
    private final boolean present;
    private final Path directory;

    private AgentBundle(
        Map<String, String> fields,
        boolean present,
        Path directory
    ) {
        this.fields = fields;
        this.present = present;
        this.directory = directory;
    }

    /** Read {@code contract.txt} if the bundle has one. Never throws on absence. */
    public static AgentBundle read(Path directory) throws IOException {
        Path path = directory.resolve("contract.txt");
        if (!Files.isRegularFile(path)) {
            return new AgentBundle(Map.of(), false, directory);
        }
        Map<String, String> fields = new LinkedHashMap<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            int tab = line.indexOf('\t');
            if (tab <= 0) {
                throw new IOException(
                    "contract.txt line is not 'key<TAB>value': " + line);
            }
            String key = line.substring(0, tab);
            String previous = fields.put(
                key, line.substring(tab + 1).trim());
            if (previous != null) {
                throw new IOException("contract.txt repeats key '" + key + "'");
            }
        }
        return new AgentBundle(fields, true, directory);
    }

    public boolean isPresent() {
        return present;
    }

    public String role() {
        return fields.getOrDefault("role", "");
    }

    public String schema() {
        return fields.getOrDefault("schema", "");
    }

    public String checkpointSha256() {
        return fields.getOrDefault("checkpoint_sha256", "");
    }

    private int intField(String key) {
        String value = fields.get(key);
        if (value == null) {
            return -1;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException error) {
            return -1;
        }
    }

    private long longField(String key) {
        String value = fields.get(key);
        if (value == null) {
            return -1L;
        }
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException error) {
            return -1L;
        }
    }

    /**
     * Every reason this bundle should not arm. Empty means safe to run.
     *
     * <p>Checks the declared contract against both the pinned widths and the
     * runtime that was actually loaded, because those can disagree in either
     * direction: a manifest can lie about weights, and weights can be swapped
     * under a correct manifest.
     */
    public List<String> problems(PolicyRuntime runtime) {
        List<String> problems = new ArrayList<>();
        if (!present) {
            problems.add(
                "bundle has no contract.txt; tensor content identity is unknown");
            return problems;
        }
        if (!schema().equals(SCHEMA)) {
            problems.add("unknown bundle schema '" + schema() + "'");
        }

        String tensorSchema = fields.getOrDefault(
            "policy_tensor_identity_schema", "");
        if (!tensorSchema.equals(PolicyTensorIdentity.SCHEMA)) {
            problems.add("unknown policy tensor identity schema '"
                + tensorSchema + "'");
        }
        int tensorCount = intField("policy_tensor_count");
        if (tensorCount != PolicyTensorIdentity.TENSOR_NAMES.size()) {
            problems.add("declared policy tensor count " + tensorCount + " != "
                + PolicyTensorIdentity.TENSOR_NAMES.size());
        }
        String declaredTensorSha = fields.getOrDefault(
            "policy_tensor_content_sha256", "");
        if (!isCanonicalSha256(declaredTensorSha)) {
            problems.add("policy tensor content SHA-256 is malformed");
        } else {
            try {
                String actualTensorSha = PolicyTensorIdentity.sha256(directory);
                if (!declaredTensorSha.equals(actualTensorSha)) {
                    problems.add("policy tensor content hash "
                        + declaredTensorSha + " != installed raw weights "
                        + actualTensorSha);
                }
            } catch (IOException | IllegalArgumentException error) {
                problems.add("policy tensor content cannot be verified: "
                    + error.getMessage());
            }
        }

        if (!isCanonicalSha256(checkpointSha256())) {
            problems.add("checkpoint SHA-256 is malformed");
        }

        String trainingSchema = fields.getOrDefault(
            "training_provenance_schema", "");
        if (!trainingSchema.equals(TRAINING_PROVENANCE_SCHEMA)) {
            problems.add("unknown training provenance schema '"
                + trainingSchema + "'");
        }
        long optimizerUpdates = longField("training_optimizer_updates");
        long environmentSteps = longField("training_environment_steps");
        long demonstrations = longField("training_demonstration_examples");
        if (optimizerUpdates < 0L || environmentSteps < 0L
                || demonstrations < 0L) {
            problems.add("training provenance counters must be nonnegative integers");
        }
        String sourceMetadataSha = fields.getOrDefault(
            "training_source_metadata_sha256", "");
        if (!isCanonicalSha256(sourceMetadataSha)) {
            problems.add("training source metadata SHA-256 is malformed");
        }
        String purpose = fields.getOrDefault("policy_purpose", "");
        if (purpose.equals("trained")) {
            if (!((optimizerUpdates > 0L && environmentSteps > 0L)
                    || demonstrations > 0L)) {
                problems.add("trained bundle has no positive training evidence");
            }
        } else if (purpose.equals("diagnostic")) {
            if (!Files.isRegularFile(directory.resolve("live-test-control.txt"))) {
                problems.add("diagnostic bundle requires live-test-control.txt "
                    + "and cannot arm as an ordinary policy");
            }
        } else {
            problems.add("unknown policy purpose '" + purpose + "'");
        }
        problems.addAll(PolicyDeploymentCompatibility.problems(
            fields, directory, purpose, role()));
        problems.addAll(BridgeTargetAttestation.bundleProblems(
            fields, directory));

        int observation = intField("observation_size");
        int actions = intField("action_mask_size");
        if (observation != EXPECTED_OBSERVATION) {
            problems.add("declared observation width " + observation + " != "
                + EXPECTED_OBSERVATION + "; trained against another contract");
        }
        if (actions != EXPECTED_ACTIONS) {
            problems.add("declared action width " + actions + " != "
                + EXPECTED_ACTIONS);
        }
        String observationContract = fields.get("observation_contract_sha256");
        String actionContract = fields.get("action_contract_sha256");
        if (observationContract == null || actionContract == null) {
            problems.add("bundle has no exact observation/action contract identity");
        } else {
            if (!observationContract.equals(EXPECTED_POLICY_CONTRACT)) {
                problems.add("observation contract " + observationContract
                    + " != " + EXPECTED_POLICY_CONTRACT);
            }
            if (!actionContract.equals(EXPECTED_POLICY_CONTRACT)) {
                problems.add("action contract " + actionContract
                    + " != " + EXPECTED_POLICY_CONTRACT);
            }
        }
        if (runtime != null) {
            if (observation > 0 && observation != runtime.observationSize()) {
                problems.add("manifest says observation " + observation
                    + " but the loaded weights are " + runtime.observationSize()
                    + "; manifest and weights disagree");
            }
            if (actions > 0 && actions != runtime.actionSize()) {
                problems.add("manifest says actions " + actions
                    + " but the loaded weights are " + runtime.actionSize());
            }
            int encoder = intField("encoder_size");
            int recurrent = intField("recurrent_size");
            if (encoder > 0 && encoder != runtime.encoderSize()) {
                problems.add("manifest encoder " + encoder + " != loaded "
                    + runtime.encoderSize());
            }
            if (recurrent > 0 && recurrent != runtime.recurrentSize()) {
                problems.add("manifest recurrent " + recurrent + " != loaded "
                    + runtime.recurrentSize());
            }
        }

        return problems;
    }

    /** One line for the arming log, so what ran is recoverable afterwards. */
    public String describe() {
        if (!present) {
            return "invalid bundle (no contract.txt)";
        }
        String sha = checkpointSha256();
        String tensors = fields.getOrDefault(
            "policy_tensor_content_sha256", "");
        return String.format(
            "role=%s purpose=%s updates=%d steps=%d obs=%d actions=%d "
                + "%d/%d std=%s ckpt=%s tensors=%s",
            role().isEmpty() ? "<default>" : role(),
            fields.getOrDefault("policy_purpose", "?"),
            longField("training_optimizer_updates"),
            longField("training_environment_steps"),
            intField("observation_size"), intField("action_mask_size"),
            intField("encoder_size"), intField("recurrent_size"),
            fields.getOrDefault("actor_kernel_std", "?"),
            sha.isEmpty() ? "?" : sha.substring(0, Math.min(16, sha.length())),
            tensors.isEmpty()
                ? "?" : tensors.substring(0, Math.min(16, tensors.length())));
    }

    private static boolean isCanonicalSha256(String value) {
        if (value.length() != 64) {
            return false;
        }
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            if (!((character >= '0' && character <= '9')
                    || (character >= 'A' && character <= 'F'))) {
                return false;
            }
        }
        return true;
    }
}
