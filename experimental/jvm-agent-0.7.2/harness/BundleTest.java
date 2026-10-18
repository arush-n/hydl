import com.hytalerlbridge.policy.AgentBundle;
import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.Policy;
import com.hytalerlbridge.policy.PolicyRuntime;
import com.hytalerlbridge.policy.bundle.PolicyDeploymentCompatibility;
import com.hytalerlbridge.policy.bundle.PolicyTensorIdentity;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.HexFormat;

/** Certifies a policy bundle gates contract, shape, and raw tensor content. */
public final class BundleTest {

    private static final String CHECKPOINT_SHA =
        "16D7930ED8E8177EE4D8CF23180114530E6B19BBBC3AA3F90BB406614C9F1DAC";
    private static final String TEST_POLICY_CONTRACT =
        digest("bundle-test policy contract");
    private static Path scratch;

    private BundleTest() {
    }

    public static void main(String[] arguments) throws IOException {
        scratch = Files.createTempDirectory("bundle-test");
        List<String> failures = new ArrayList<>();
        int checked = 0;

        // Backward/hand-built directories cannot prove their weight content.
        AgentBundle absent = AgentBundle.read(directory("absent"));
        checked++;
        if (absent.isPresent()
                || !mentions(absent.problems(null), "no contract.txt")) {
            failures.add("a no-contract bundle did not fail closed: "
                + absent.problems(null));
        }

        Path legacyPath = directory("legacy-v1");
        writeTensorFiles(legacyPath, new Random(1));
        AgentBundle legacy = writeRaw(legacyPath,
            "schema\thytalerl_policy_agent_bundle_v1",
            "checkpoint_sha256\t" + CHECKPOINT_SHA,
            "observation_size\t8271",
            "action_mask_size\t99",
            "observation_contract_sha256\t" + TEST_POLICY_CONTRACT,
            "action_contract_sha256\t" + TEST_POLICY_CONTRACT,
            "actor_kernel_std\t0.01");
        checked++;
        if (!mentions(legacy.problems(null), "unknown bundle schema")
                || !mentions(legacy.problems(null), "tensor identity schema")) {
            failures.add("a v1 bundle did not fail closed: "
                + legacy.problems(null));
        }

        AgentBundle retired = bundle("retired", Map.of(
            "observation_size", "22",
            "action_mask_size", "9"));
        checked++;
        if (!mentions(retired.problems(null), "observation width 22")
                || !mentions(retired.problems(null), "action width 9")) {
            failures.add("retired widths were not rejected: "
                + retired.problems(null));
        }

        AgentBundle widthOnly = bundle("width-only", Map.of(
            "observation_contract_sha256", "",
            "action_contract_sha256", ""));
        checked++;
        if (!mentions(widthOnly.problems(null), "observation contract")) {
            failures.add("a width-only bundle was accepted: "
                + widthOnly.problems(null));
        }

        String updatedPolicyContract = digest("updated policy contract");
        AgentBundle updatedContract = bundle("updated-contract", Map.of(
            "observation_contract_sha256", updatedPolicyContract,
            "action_contract_sha256", updatedPolicyContract,
            "policy_action_scope_sha256",
                actionScopeSha256(updatedPolicyContract, fullScopeBits())));
        checked++;
        if (!updatedContract.problems(null).isEmpty()) {
            failures.add("a current exported contract was rejected: "
                + updatedContract.problems(null));
        }

        AgentBundle unsupportedAbi = bundle("unsupported-abi", Map.of(
            "policy_projection_abi", "unsupported_projection_abi_v2"));
        checked++;
        if (!mentions(unsupportedAbi.problems(null), "policy_projection_abi")
                || !mentions(unsupportedAbi.problems(null),
                    AgentBundle.SUPPORTED_POLICY_PROJECTION_ABI)) {
            failures.add("an unsupported projection ABI was accepted or unclear: "
                + unsupportedAbi.problems(null));
        }

        AgentBundle unsupportedScope = bundle("unsupported-scope", Map.of(
            "policy_action_scope_schema", "future_action_scope_v2"));
        checked++;
        if (!mentions(unsupportedScope.problems(null),
                "policy_action_scope_schema")
                || !mentions(unsupportedScope.problems(null),
                    AgentBundle.POLICY_ACTION_SCOPE_SCHEMA)) {
            failures.add("an unsupported policy action scope was accepted or unclear: "
                + unsupportedScope.problems(null));
        }

        AgentBundle malformedScopeIdentity = bundle(
            "malformed-scope-identity",
            Map.of("policy_action_scope_sha256", "not-a-canonical-digest"));
        checked++;
        if (!mentions(malformedScopeIdentity.problems(null),
                "policy_action_scope_sha256 is malformed")) {
            failures.add("a malformed policy action-scope identity was accepted: "
                + malformedScopeIdentity.problems(null));
        }

        int jumpOffset = 0;
        for (int head = 0; head < ActionDecoder.JUMP; head++) {
            jumpOffset += Policy.HEAD_SIZES[head];
        }
        char[] narrowedScopeChars = fullScopeBits().toCharArray();
        narrowedScopeChars[jumpOffset + 1] = '0';
        String narrowedScopeBits = new String(narrowedScopeChars);
        AgentBundle narrowedScopeBundle = bundle("narrowed-scope", Map.of(
            "policy_action_scope_bits", narrowedScopeBits,
            "policy_action_scope_sha256",
                actionScopeSha256(TEST_POLICY_CONTRACT, narrowedScopeBits)));
        checked++;
        boolean[] narrowedMask = narrowedScopeBundle.actionScopeMask();
        if (!narrowedScopeBundle.problems(null).isEmpty()
                || narrowedMask[jumpOffset + 1]) {
            failures.add("a valid restricted action scope did not round-trip: "
                + narrowedScopeBundle.problems(null));
        }

        char[] emptyHeadChars = fullScopeBits().toCharArray();
        for (int choice = 0; choice < Policy.HEAD_SIZES[ActionDecoder.JUMP]; choice++) {
            emptyHeadChars[jumpOffset + choice] = '0';
        }
        String emptyHeadBits = new String(emptyHeadChars);
        AgentBundle emptyScopeHead = bundle("empty-scope-head", Map.of(
            "policy_action_scope_bits", emptyHeadBits,
            "policy_action_scope_sha256",
                actionScopeSha256(TEST_POLICY_CONTRACT, emptyHeadBits)));
        checked++;
        if (!mentions(emptyScopeHead.problems(null), "leaves action head")) {
            failures.add("an empty action-scope head was accepted: "
                + emptyScopeHead.problems(null));
        }

        AgentBundle changedScopeBits = bundle("changed-scope-bits", Map.of(
            "policy_action_scope_bits", narrowedScopeBits));
        checked++;
        if (!mentions(changedScopeBits.problems(null),
                "policy_action_scope_sha256 does not match")) {
            failures.add("changed action-scope bits were accepted: "
                + changedScopeBits.problems(null));
        }

        AgentBundle splitScopeContract = bundle("split-scope-contract", Map.of(
            "action_contract_sha256", digest("different action contract")));
        checked++;
        if (!mentions(splitScopeContract.problems(null),
                "policy_action_scope_sha256 does not match")) {
            failures.add("an action scope bound to another contract was accepted: "
                + splitScopeContract.problems(null));
        }

        AgentBundle splitContract = bundle("split-contract", Map.of(
            "observation_contract_sha256", digest("observation contract"),
            "action_contract_sha256", digest("action contract")));
        checked++;
        if (!mentions(splitContract.problems(null),
                "observation/action contract identities differ")) {
            failures.add("different observation/action identities were accepted: "
                + splitContract.problems(null));
        }
        AgentBundle malformedPolicyContract = bundle("malformed-policy-contract",
            Map.of("observation_contract_sha256", "not-a-canonical-digest"));
        checked++;
        if (!mentions(malformedPolicyContract.problems(null),
                "observation contract SHA-256 is malformed")) {
            failures.add("a malformed policy contract digest was accepted: "
                + malformedPolicyContract.problems(null));
        }

        AgentBundle untrained = bundle("untrained", Map.of(
            "training_optimizer_updates", "0",
            "training_environment_steps", "0"));
        checked++;
        if (!mentions(untrained.problems(null), "positive training evidence")) {
            failures.add("zero-evidence weights were accepted: "
                + untrained.problems(null));
        }

        // Weight dispersion is diagnostic only.  A genuinely trained policy
        // may stay narrow, while a synthetic policy can be initialized broad.
        AgentBundle narrowTrained = bundle("narrow-trained", Map.of(
            "actor_kernel_std", "0.0001"));
        checked++;
        if (!narrowTrained.problems(null).isEmpty()) {
            failures.add("low-dispersion trained weights were rejected: "
                + narrowTrained.problems(null));
        }

        AgentBundle broadDiagnostic = bundle("broad-diagnostic", Map.of(
            "policy_purpose", "diagnostic",
            "training_optimizer_updates", "0",
            "training_environment_steps", "0",
            "actor_kernel_std", "0.5"));
        checked++;
        if (!mentions(broadDiagnostic.problems(null), "live-test-control.txt")) {
            failures.add("an uncontrolled broad diagnostic was accepted: "
                + broadDiagnostic.problems(null));
        }
        Path controlledDiagnosticPath = directory("controlled-diagnostic");
        writeTensorFiles(controlledDiagnosticPath, new Random(0xD1A6));
        Files.writeString(
            controlledDiagnosticPath.resolve("live-test-control.txt"),
            "diagnostic\n",
            StandardCharsets.UTF_8);
        AgentBundle controlledDiagnostic = bundleAt(
            controlledDiagnosticPath,
            Map.of(
                "policy_purpose", "diagnostic",
                "training_optimizer_updates", "0",
                "training_environment_steps", "0",
                "actor_kernel_std", "0.5"));
        checked++;
        if (!controlledDiagnostic.problems(null).isEmpty()) {
            failures.add("an explicit controlled diagnostic was rejected: "
                + controlledDiagnostic.problems(null));
        }

        AgentBundle good = bundle("good", Map.of());
        checked++;
        if (!good.problems(null).isEmpty()) {
            failures.add("a current content-pinned bundle was rejected: "
                + good.problems(null));
        }
        AgentBundle unattestedFrozen = bundle("unattested-frozen", Map.of(
            "training_context_attestation_schema", "",
            "training_context_attestation_derivation", "",
            "training_context_runtime_config_content_sha256", "",
            "training_context_content_sha256", ""));
        checked++;
        if (!unattestedFrozen.problems(null).isEmpty()) {
            failures.add("an open-loop bundle made a live-profile claim: "
                + unattestedFrozen.problems(null));
        }
        if (!good.describe().contains("Kweebec_Razorleaf")
                || !good.describe().contains("tensors=")) {
            failures.add("describe() omitted role/tensor identity: "
                + good.describe());
        }

        Path knownContextPath = directory("known-context-vector");
        writeTensorFiles(knownContextPath, new Random(0xC07E87));
        AgentBundle knownContext = bundleAt(knownContextPath, Map.of(
            "training_context_policy_id", "7",
            "training_context_owner_0_entity_index", "2",
            "training_context_owner_0_team_id", "-3"));
        Map<String, String> knownFields = readFields(
            knownContextPath.resolve("contract.txt"));
        checked++;
        String knownDigest = knownFields.get(
            "deployment_compatibility_content_sha256");
        String knownContextDigest = knownFields.get(
            "training_context_content_sha256");
        if (!"52CAF80E2EB049DB10DCD98A2FD26C11A09F4E7E8B4E07975805EA3EB3607CEE"
                .equals(knownDigest)
                || !"456883F0A39A7A6A2E04EFCCD5A0EC9BA7EC6F0C919E30F0A7ADB0A6B6797F7B"
                    .equals(knownContextDigest)
                || !knownContext.problems(null).isEmpty()) {
            failures.add("cross-language deployment-context vector drifted: "
                + knownDigest + " context=" + knownContextDigest + " "
                + knownContext.problems(null));
        }

        AgentBundle malformed = bundle("malformed", Map.of(
            "policy_tensor_content_sha256", "not-a-canonical-digest"));
        checked++;
        if (!mentions(malformed.problems(null), "SHA-256 is malformed")) {
            failures.add("a malformed tensor digest was accepted: "
                + malformed.problems(null));
        }

        AgentBundle malformedCheckpoint = bundle("bad-checkpoint", Map.of(
            "checkpoint_sha256", CHECKPOINT_SHA.toLowerCase()));
        checked++;
        if (!mentions(malformedCheckpoint.problems(null),
                "checkpoint SHA-256 is malformed")) {
            failures.add("a non-canonical checkpoint digest was accepted: "
                + malformedCheckpoint.problems(null));
        }

        Path exactProfilePath = directory("exact-live-profile");
        writeTensorFiles(exactProfilePath, new Random(0xE7AC7));
        Path exactProfile = Files.createDirectories(
            exactProfilePath.resolve("profile"));
        Files.write(exactProfile.resolve("profile-values.bin"),
            new byte[] {1, 3, 3, 7});
        byte[] exactProfileManifest = ("{\"profile\":\"iron_sword\","
            + "\"arsenal_policy_contract_sha256\":\""
            + TEST_POLICY_CONTRACT + "\",\"policy_projection_abi\":\""
            + AgentBundle.SUPPORTED_POLICY_PROJECTION_ABI + "\"}\n")
            .getBytes(StandardCharsets.UTF_8);
        Files.write(exactProfile.resolve("profile.json"), exactProfileManifest);
        Files.createDirectory(exactProfile.resolve("unexpected-directory"));
        checked++;
        try {
            PolicyDeploymentCompatibility.profileContentSha256(exactProfile);
            failures.add("a nested profile directory was silently unbound");
        } catch (IOException expected) {
            // Every allowed profile entry is content-bound; nested trees fail.
        }
        Files.delete(exactProfile.resolve("unexpected-directory"));
        String exactProfileSha =
            PolicyDeploymentCompatibility.profileContentSha256(exactProfile);
        if (!"BF5A424EAD10D710C28E7AF33520E8BFA9FF8AADBD641B0A3C1F5020785EB0C1"
                .equals(exactProfileSha)) {
            failures.add("cross-language profile-content vector drifted: "
                + exactProfileSha);
        }
        AgentBundle exactLive = bundleAt(exactProfilePath, Map.of(
            "deployment_profile_mode", "live",
            "deployment_profile", "iron_sword",
            "deployment_profile_content_sha256", exactProfileSha));
        checked++;
        if (!exactLive.problems(null).isEmpty()) {
            failures.add("an exact training/live profile was rejected: "
                + exactLive.problems(null));
        }

        Path staleProfileIdentityPath = directory("stale-profile-identity");
        writeTensorFiles(staleProfileIdentityPath, new Random(0x51A1E));
        Path staleProfile = Files.createDirectories(
            staleProfileIdentityPath.resolve("profile"));
        Files.writeString(staleProfile.resolve("profile.json"),
            "{\"arsenal_policy_contract_sha256\":\""
                + digest("different profile policy")
                + "\",\"policy_projection_abi\":\""
                + AgentBundle.SUPPORTED_POLICY_PROJECTION_ABI + "\"}\n",
            StandardCharsets.UTF_8);
        String staleProfileSha =
            PolicyDeploymentCompatibility.profileContentSha256(staleProfile);
        AgentBundle staleProfileIdentity = bundleAt(staleProfileIdentityPath,
            Map.of(
                "deployment_profile_mode", "live",
                "deployment_profile", "iron_sword",
                "deployment_profile_content_sha256", staleProfileSha));
        checked++;
        if (!mentions(staleProfileIdentity.problems(null),
                "arsenal_policy_contract_sha256")) {
            failures.add("a mismatched live profile policy identity was accepted: "
                + staleProfileIdentity.problems(null));
        }

        Path staleProfileAbiPath = directory("stale-profile-abi");
        writeTensorFiles(staleProfileAbiPath, new Random(0x51A1F));
        Path staleProfileAbi = Files.createDirectories(
            staleProfileAbiPath.resolve("profile"));
        Files.writeString(staleProfileAbi.resolve("profile.json"),
            "{\"arsenal_policy_contract_sha256\":\""
                + TEST_POLICY_CONTRACT
                + "\",\"policy_projection_abi\":"
                + "\"unsupported_projection_abi_v2\"}\n",
            StandardCharsets.UTF_8);
        String staleProfileAbiSha =
            PolicyDeploymentCompatibility.profileContentSha256(staleProfileAbi);
        AgentBundle staleProfileAbiBundle = bundleAt(staleProfileAbiPath,
            Map.of(
                "deployment_profile_mode", "live",
                "deployment_profile", "iron_sword",
                "deployment_profile_content_sha256", staleProfileAbiSha));
        checked++;
        if (!mentions(staleProfileAbiBundle.problems(null),
                "live profile policy_projection_abi")) {
            failures.add("a live profile with unsupported policy_projection_abi "
                + "was accepted: " + staleProfileAbiBundle.problems(null));
        }
        AgentBundle unattestedLive = bundleAt(exactProfilePath, Map.of(
            "deployment_profile_mode", "live",
            "deployment_profile", "iron_sword",
            "deployment_profile_content_sha256", exactProfileSha,
            "training_context_attestation_schema", "",
            "training_context_attestation_derivation", "",
            "training_context_runtime_config_content_sha256", "",
            "training_context_content_sha256", ""));
        checked++;
        if (!mentions(unattestedLive.problems(null), "runtime-derived")) {
            failures.add("an unattested trained live profile was accepted: "
                + unattestedLive.problems(null));
        }
        AgentBundle staleContextAttestation = bundleAt(exactProfilePath, Map.of(
            "deployment_profile_mode", "live",
            "deployment_profile", "iron_sword",
            "deployment_profile_content_sha256", exactProfileSha,
            "training_context_content_sha256", CHECKPOINT_SHA));
        checked++;
        if (!mentions(staleContextAttestation.problems(null),
                "runtime-derived")) {
            failures.add("a stale training-context attestation was accepted: "
                + staleContextAttestation.problems(null));
        }
        AgentBundle malformedRuntimeAttestation = bundleAt(
            exactProfilePath, Map.of(
                "deployment_profile_mode", "live",
                "deployment_profile", "iron_sword",
                "deployment_profile_content_sha256", exactProfileSha,
                "training_context_runtime_config_content_sha256", "same-shape"));
        checked++;
        if (!mentions(malformedRuntimeAttestation.problems(null),
                "runtime-derived")) {
            failures.add("a malformed runtime-config attestation was accepted: "
                + malformedRuntimeAttestation.problems(null));
        }
        byte[] replacedManifest = exactProfileManifest.clone();
        replacedManifest[13] ^= 0x01;
        Files.write(exactProfile.resolve("profile.json"), replacedManifest);
        checked++;
        if (!mentions(exactLive.problems(null), "deployment profile content")) {
            failures.add("a same-size profile manifest replacement was accepted: "
                + exactLive.problems(null));
        }
        Files.write(exactProfile.resolve("profile.json"), exactProfileManifest);

        Path mappedProfilePath = directory("mapped-live-profile");
        writeTensorFiles(mappedProfilePath, new Random(0xC2055));
        Path mappedProfile = Files.createDirectories(
            mappedProfilePath.resolve("profile"));
        Files.write(mappedProfile.resolve("profile-values.bin"),
            new byte[] {2, 0, 2, 6});
        Files.writeString(mappedProfile.resolve("profile.json"),
            "{\"arsenal_policy_contract_sha256\":\""
                + TEST_POLICY_CONTRACT + "\",\"policy_projection_abi\":\""
                + AgentBundle.SUPPORTED_POLICY_PROJECTION_ABI + "\"}\n",
            StandardCharsets.UTF_8);
        String mappedProfileSha =
            PolicyDeploymentCompatibility.profileContentSha256(mappedProfile);
        AgentBundle mappedLive = bundleAt(mappedProfilePath, Map.of(
            "deployment_profile_mode", "live",
            "deployment_profile", "future_unique_profile",
            "deployment_profile_content_sha256", mappedProfileSha,
            "profile_transfer_source", "iron_sword",
            "transfer_reason", "declared cross-profile evaluation"));
        checked++;
        if (!mappedLive.problems(null).isEmpty()) {
            failures.add("an explicit generic profile mapping was rejected: "
                + mappedLive.problems(null));
        }

        AgentBundle unmappedLive = bundleAt(mappedProfilePath, Map.of(
            "deployment_profile_mode", "live",
            "deployment_profile", "future_unique_profile",
            "deployment_profile_content_sha256", mappedProfileSha));
        checked++;
        if (!mentions(unmappedLive.problems(null), "cross-profile deployment")) {
            failures.add("an unowned live profile was accepted: "
                + unmappedLive.problems(null));
        }

        AgentBundle roleMismatch = bundle("role-mismatch", Map.of(
            "deployment_role", "Future_Unique_Role"));
        checked++;
        if (!mentions(roleMismatch.problems(null), "cross-role deployment")) {
            failures.add("an unmapped role was accepted: "
                + roleMismatch.problems(null));
        }
        AgentBundle mappedRole = bundle("role-mapped", Map.of(
            "deployment_role", "Future_Unique_Role",
            "role_transfer_source", "Kweebec_Razorleaf",
            "transfer_reason", "declared role-transfer evaluation"));
        checked++;
        if (!mappedRole.problems(null).isEmpty()) {
            failures.add("an explicit generic role mapping was rejected: "
                + mappedRole.problems(null));
        }
        AgentBundle wildcardRole = bundle("wildcard-role", Map.of(
            "deployment_role", "*",
            "role_transfer_source", "Kweebec_Razorleaf",
            "transfer_reason", "would otherwise claim arbitrary roles"));
        checked++;
        if (!mentions(wildcardRole.problems(null), "cannot be wildcard")) {
            failures.add("a trained wildcard role was accepted: "
                + wildcardRole.problems(null));
        }

        Files.writeString(mappedProfilePath.resolve("role.txt"),
            "Another_Role\n", StandardCharsets.UTF_8);
        checked++;
        if (!mentions(unmappedLive.problems(null), "role.txt")) {
            failures.add("a post-export role.txt replacement was accepted: "
                + unmappedLive.problems(null));
        }

        // Same byte length and therefore the same Java tensor shape.  This is
        // the precise class a v1 width/shape check could not see.
        Path mutationPath = directory("same-shape-mutation");
        writeTensorFiles(mutationPath, new Random(0x51A9E));
        AgentBundle mutation = bundleAt(mutationPath, Map.of());
        Path actorBias = mutationPath.resolve("actor_bias.bin");
        byte[] actorBytes = Files.readAllBytes(actorBias);
        actorBytes[actorBytes.length / 2] ^= 0x55;
        Files.write(actorBias, actorBytes);
        checked++;
        if (!mentions(mutation.problems(null), "installed raw weights")) {
            failures.add("a same-shape byte mutation was accepted: "
                + mutation.problems(null));
        }

        Path swapPath = directory("same-shape-swap");
        writeTensorFiles(swapPath, new Random(0x5A9));
        AgentBundle swapped = bundleAt(swapPath, Map.of());
        Path inputBias = swapPath.resolve("encoder_input_bias.bin");
        Path hiddenBias = swapPath.resolve("encoder_hidden_bias.bin");
        byte[] inputBiasBytes = Files.readAllBytes(inputBias);
        byte[] hiddenBiasBytes = Files.readAllBytes(hiddenBias);
        Files.write(inputBias, hiddenBiasBytes);
        Files.write(hiddenBias, inputBiasBytes);
        checked++;
        if (!mentions(swapped.problems(null), "installed raw weights")) {
            failures.add("same-shaped tensor files were swapped undetected: "
                + swapped.problems(null));
        }

        // Compare declared sizes with the policy actually loaded by the suite.
        if (arguments.length > 0) {
            boolean verifyInstalled = arguments.length == 2
                && "--installed-bundle".equals(arguments[0]);
            if (arguments.length > 1 && !verifyInstalled) {
                throw new IllegalArgumentException(
                    "expected [weights] or --installed-bundle <directory>");
            }
            Path weights = Path.of(
                verifyInstalled ? arguments[1] : arguments[0]);
            AgentBundle installed = verifyInstalled
                ? AgentBundle.read(weights) : null;
            boolean[] scope = verifyInstalled
                ? installed.actionScopeMask() : fullScopeMask();
            PolicyRuntime runtime = PolicyRuntime.load(weights, scope);
            if (verifyInstalled) {
                checked++;
                List<String> installedProblems = installed.problems(runtime);
                if (!installedProblems.isEmpty()) {
                    failures.add("installed exported bundle was rejected: "
                        + installedProblems);
                }
            }

            boolean[] liveLegal = fullScopeMask();
            liveLegal[0] = false;
            boolean[] effectiveLegal = runtime.effectiveActionMask(liveLegal);
            for (int index = 0; index < effectiveLegal.length; index++) {
                if (effectiveLegal[index] != (scope[index] && liveLegal[index])) {
                    failures.add("runtime action-scope/live-mask intersection "
                        + "differs at index " + index);
                    break;
                }
            }

            boolean[] noLiveJumpChoices = fullScopeMask();
            int noLiveJumpOffset = 0;
            for (int head = 0; head < ActionDecoder.JUMP; head++) {
                noLiveJumpOffset += Policy.HEAD_SIZES[head];
            }
            for (int choice = 0;
                    choice < Policy.HEAD_SIZES[ActionDecoder.JUMP]; choice++) {
                noLiveJumpChoices[noLiveJumpOffset + choice] = false;
            }
            ActionDecoder.Decoded noLegalJump = runtime.decide(
                new float[runtime.observationSize()],
                runtime.newCarry(),
                noLiveJumpChoices);
            checked++;
            if (noLegalJump.actionLegal() || noLegalJump.jumpHeld()
                    || noLegalJump.attack()) {
                failures.add("an empty effective action head emitted a command");
            }

            boolean[] emptyScope = fullScopeMask();
            int runtimeJumpOffset = 0;
            for (int head = 0; head < ActionDecoder.JUMP; head++) {
                runtimeJumpOffset += Policy.HEAD_SIZES[head];
            }
            for (int choice = 0; choice < Policy.HEAD_SIZES[ActionDecoder.JUMP]; choice++) {
                emptyScope[runtimeJumpOffset + choice] = false;
            }
            checked++;
            try {
                PolicyRuntime.load(weights, emptyScope);
                failures.add("runtime loaded an action scope with an empty head");
            } catch (IllegalArgumentException expected) {
                if (!expected.getMessage().contains("policy_action_scope_bits")) {
                    failures.add("empty runtime action scope failed unclearly: "
                        + expected.getMessage());
                }
            }
            Path matchingPath = directory("runtime-matching");
            copyTensorFiles(weights, matchingPath);
            AgentBundle matching = bundleAt(matchingPath, Map.of(
                "observation_size", Integer.toString(runtime.observationSize()),
                "action_mask_size", Integer.toString(runtime.actionSize()),
                "encoder_size", Integer.toString(runtime.encoderSize()),
                "recurrent_size", Integer.toString(runtime.recurrentSize())));
            checked++;
            if (!matching.problems(runtime).isEmpty()) {
                failures.add("a bundle matching loaded weights was rejected: "
                    + matching.problems(runtime));
            }

            AgentBundle lying = bundle("runtime-lying", Map.of(
                "encoder_size", "999",
                "recurrent_size", Integer.toString(runtime.recurrentSize())));
            checked++;
            if (!mentions(lying.problems(runtime), "manifest encoder 999")) {
                failures.add("manifest/runtime shape disagreement was accepted: "
                    + lying.problems(runtime));
            }
        }

        // Keep the existing seeded semantic-contract breadth.  The separate
        // PolicyTensorIdentityTest owns 256 independently-oracled raw mutations.
        Random random = new Random(0xB00D1E5EEDL);
        Path generatedPath = directory("generated-corpus");
        writeTensorFiles(generatedPath, new Random(0x6E3E2A7EDL));
        for (int sample = 0; sample < 256; sample++) {
            int corruption = sample % 11;
            Map<String, String> overrides = new LinkedHashMap<>();
            String expected = null;
            switch (corruption) {
                case 1 -> {
                    // Strictly ABOVE the valid width. This was a literal 8272
                    // chosen when the contract was 8271; when the contract moved
                    // to 8272 a draw of 0 produced a VALID bundle that this case
                    // demanded be rejected. Anchor it to the constant so the
                    // negative control cannot generate the positive case.
                    overrides.put("observation_size", Integer.toString(
                        AgentBundle.EXPECTED_OBSERVATION + 1
                            + random.nextInt(4096)));
                    expected = "declared observation width";
                }
                case 2 -> {
                    overrides.put("action_mask_size",
                        Integer.toString(100 + random.nextInt(512)));
                    expected = "declared action width";
                }
                case 3 -> {
                    String corrupted = corruptDigest(TEST_POLICY_CONTRACT, random);
                    overrides.put("observation_contract_sha256", corrupted);
                    expected = corrupted.matches("[0-9A-F]{64}")
                        ? "observation/action contract identities differ"
                        : "observation contract SHA-256 is malformed";
                }
                case 4 -> {
                    String corrupted = corruptDigest(TEST_POLICY_CONTRACT, random);
                    overrides.put("action_contract_sha256", corrupted);
                    expected = corrupted.matches("[0-9A-F]{64}")
                        ? "observation/action contract identities differ"
                        : "action contract SHA-256 is malformed";
                }
                case 5 -> {
                    overrides.put("schema", "unknown_" + random.nextInt());
                    expected = "unknown bundle schema";
                }
                case 6 -> {
                    overrides.put("policy_tensor_count",
                        Integer.toString(12 + random.nextInt(32)));
                    expected = "declared policy tensor count";
                }
                case 7 -> {
                    overrides.put("training_optimizer_updates", "0");
                    overrides.put("training_environment_steps", "0");
                    expected = "positive training evidence";
                }
                case 8 -> {
                    overrides.put("policy_purpose", "unknown");
                    expected = "unknown policy purpose";
                }
                case 9 -> {
                    overrides.put("training_provenance_schema", "stale");
                    expected = "training provenance schema";
                }
                case 10 -> {
                    overrides.put("training_source_metadata_sha256", "bad");
                    expected = "training source metadata SHA-256";
                }
                default -> {
                    // Generated positive control.
                }
            }
            List<String> problems = bundleAt(
                generatedPath, overrides).problems(null);
            checked++;
            if (expected == null ? !problems.isEmpty() : !mentions(problems, expected)) {
                failures.add("generated bundle " + sample + " expected "
                    + expected + " but got " + problems);
            }
        }

        // Deployment compatibility needs its own generated corpus.  Profile
        // and role names are arbitrary strings, not a weapon/role catalog, so
        // this catches a solution tuned to the fixed iron-sword example.
        Random contextRandom = new Random(0xC07E87C0DEL);
        Path contextPath = directory("generated-context-corpus");
        writeTensorFiles(contextPath, new Random(0xA11CE));
        Path contextProfile = Files.createDirectories(
            contextPath.resolve("profile"));
        byte[] contextBytes = new byte[97];
        contextRandom.nextBytes(contextBytes);
        Files.write(contextProfile.resolve("generated-profile.bin"), contextBytes);
        Files.writeString(contextProfile.resolve("profile.json"),
            "{\"arsenal_policy_contract_sha256\":\""
                + TEST_POLICY_CONTRACT + "\",\"policy_projection_abi\":\""
                + AgentBundle.SUPPORTED_POLICY_PROJECTION_ABI + "\"}\n",
            StandardCharsets.UTF_8);
        String contextProfileSha =
            PolicyDeploymentCompatibility.profileContentSha256(contextProfile);
        for (int sample = 0; sample < 256; sample++) {
            String sourceProfile = "source_profile_"
                + Integer.toUnsignedString(contextRandom.nextInt(), 36);
            String targetProfile = sample % 2 == 0
                ? sourceProfile
                : "target_profile_"
                    + Integer.toUnsignedString(contextRandom.nextInt(), 36);
            String sourceRole = "SourceRole_"
                + Integer.toUnsignedString(contextRandom.nextInt(), 36);
            String targetRole = sample % 3 == 0
                ? "TargetRole_"
                    + Integer.toUnsignedString(contextRandom.nextInt(), 36)
                : sourceRole;
            Map<String, String> overrides = new LinkedHashMap<>();
            overrides.put("training_context_source_role", sourceRole);
            overrides.put("training_context_owner_0_profile", sourceProfile);
            overrides.put("deployment_role", targetRole);
            overrides.put("deployment_profile_mode", "live");
            overrides.put("deployment_profile", targetProfile);
            overrides.put("deployment_profile_content_sha256", contextProfileSha);
            if (!sourceRole.equals(targetRole)) {
                overrides.put("role_transfer_source", sourceRole);
                overrides.put("transfer_reason", "seeded context transfer");
            }
            if (!sourceProfile.equals(targetProfile)) {
                overrides.put("profile_transfer_source", sourceProfile);
                overrides.put("transfer_reason", "seeded context transfer");
            }
            // Every generated mapping is first required to pass.  The same
            // generated row is then corrupted below, so the corpus proves
            // both generic transfer support and 256 independent negative
            // substitutions instead of counting a mixture of positives and
            // negatives as "256 randomized cases".
            AgentBundle generatedControl = bundleAt(contextPath, overrides);
            List<String> controlProblems = generatedControl.problems(null);
            checked++;
            if (!controlProblems.isEmpty()) {
                failures.add("generated deployment control " + sample
                    + " was rejected: " + controlProblems);
            }
            String expected = null;
            boolean replaceRoleFile = false;
            boolean replaceProfileBytes = false;
            boolean replaceController = false;
            boolean replaceTeam = false;
            switch (sample % 12) {
                case 0 -> {
                    replaceController = true;
                    expected = "runtime-derived";
                }
                case 1 -> {
                    overrides.put("profile_transfer_source", "");
                    expected = "cross-profile deployment";
                }
                case 2 -> {
                    // sample % 12 == 2 is an exact role/profile pairing.
                    // A reason without a mapping is stale authority, not a
                    // harmless annotation.
                    overrides.put("transfer_reason", "invented exception");
                    expected = "unnecessary transfer reason";
                }
                case 3 -> {
                    overrides.put("deployment_profile_content_sha256",
                        corruptDigest(contextProfileSha, contextRandom));
                    expected = "deployment profile content";
                }
                case 4 -> {
                    overrides.put("deployment_compatibility_content_sha256",
                        corruptDigest(CHECKPOINT_SHA, contextRandom));
                    expected = "deployment compatibility content";
                }
                case 5 -> {
                    // Recomputed identity, but an owner profile was silently
                    // substituted.  The declared source mapping must still
                    // reject it independently of the checksum.
                    overrides.put("training_context_owner_0_profile",
                        sourceProfile + "_substituted");
                    expected = "cross-profile deployment";
                }
                case 6 -> {
                    // Same shape and recomputed identity: the source role no
                    // longer matches the explicit mapping (or exact target).
                    overrides.put("training_context_source_role",
                        sourceRole + "_substituted");
                    expected = "cross-role deployment";
                }
                case 7 -> {
                    replaceRoleFile = true;
                    expected = "role.txt";
                }
                case 8 -> {
                    replaceProfileBytes = true;
                    expected = "deployment profile content";
                }
                case 9 -> {
                    replaceController = true;
                    expected = "runtime-derived";
                }
                case 10 -> {
                    replaceTeam = true;
                    expected = "runtime-derived";
                }
                case 11 -> {
                    replaceTeam = true;
                    expected = "runtime-derived";
                }
                default -> throw new AssertionError("unreachable remainder");
            }
            AgentBundle generatedContext = bundleAt(contextPath, overrides);
            if (replaceRoleFile) {
                char last = targetRole.charAt(targetRole.length() - 1);
                String replacementRole = targetRole.substring(
                    0, targetRole.length() - 1) + (last == 'x' ? "y" : "x");
                Files.writeString(contextPath.resolve("role.txt"),
                    replacementRole + "\n", StandardCharsets.UTF_8);
            }
            if (replaceProfileBytes) {
                byte[] replacement = contextBytes.clone();
                replacement[sample % replacement.length] ^= 0x5A;
                Files.write(
                    contextProfile.resolve("generated-profile.bin"),
                    replacement);
            }
            if (replaceController) {
                replaceContextFieldAndRecomputeOuter(contextPath,
                    "training_context_owner_0_controller", "policz");
                generatedContext = AgentBundle.read(contextPath);
            }
            if (replaceTeam) {
                replaceContextFieldAndRecomputeOuter(contextPath,
                    "training_context_owner_0_team_id", "1");
                generatedContext = AgentBundle.read(contextPath);
            }
            List<String> problems = generatedContext.problems(null);
            if (replaceProfileBytes) {
                Files.write(
                    contextProfile.resolve("generated-profile.bin"),
                    contextBytes);
            }
            checked++;
            if (expected == null ? !problems.isEmpty()
                    : !mentions(problems, expected)) {
                failures.add("generated deployment context " + sample
                    + " expected " + expected + " but got " + problems);
            }
        }

        System.out.printf("cases         : %d%n", checked);
        if (!failures.isEmpty()) {
            System.out.println("failures      : " + failures.size());
            for (String failure : failures) {
                System.out.println("  " + failure);
            }
            System.out.println("FAIL - AgentBundle does not gate what it claims");
            System.exit(1);
        }
        System.out.println(
            "PASS - AgentBundle gates v6 ABI, action scope, training, tensors, and deployment context");
    }

    private static boolean mentions(List<String> problems, String fragment) {
        return problems.stream().anyMatch(problem -> problem.contains(fragment));
    }

    private static String digest(String value) {
        try {
            return HexFormat.of().withUpperCase().formatHex(
                MessageDigest.getInstance("SHA-256").digest(
                    value.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private static String fullScopeBits() {
        StringBuilder bits = new StringBuilder();
        for (int size : Policy.HEAD_SIZES) {
            bits.append("1".repeat(size));
        }
        return bits.toString();
    }

    private static boolean[] fullScopeMask() {
        boolean[] mask = new boolean[
            java.util.Arrays.stream(Policy.HEAD_SIZES).sum()];
        java.util.Arrays.fill(mask, true);
        return mask;
    }

    private static String actionScopeSha256(
        String actionContract,
        String maskBits
    ) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update("HYTALERL_POLICY_ACTION_SCOPE\0"
                .getBytes(StandardCharsets.UTF_8));
            digest.update(AgentBundle.POLICY_ACTION_SCOPE_SCHEMA
                .getBytes(StandardCharsets.UTF_8));
            digest.update((byte) 0);
            digest.update(actionContract.getBytes(StandardCharsets.US_ASCII));
            digest.update((byte) 0);
            digest.update(maskBits.getBytes(StandardCharsets.US_ASCII));
            return HexFormat.of().withUpperCase().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private static void replaceContextFieldAndRecomputeOuter(
        Path directory,
        String name,
        String replacement
    ) throws IOException {
        Path contract = directory.resolve("contract.txt");
        Map<String, String> fields = readFields(contract);
        if (!fields.containsKey(name)) {
            throw new IOException("generated contract has no field " + name);
        }
        fields.put(name, replacement);
        // Deliberately retain the old training_context_content_sha256 while
        // making the outer deployment digest internally consistent. This is
        // the same-shaped producer-context substitution the nested
        // attestation—not merely the outer checksum—must reject.
        fields.put("deployment_compatibility_content_sha256",
            compatibilitySha256(fields));
        List<String> rows = fields.entrySet().stream()
            .map(entry -> entry.getKey() + "\t" + entry.getValue())
            .toList();
        Files.writeString(contract, String.join("\n", rows) + "\n",
            StandardCharsets.UTF_8);
    }

    private static String corruptDigest(String digest, Random random) {
        int index = random.nextInt(digest.length());
        char original = digest.charAt(index);
        char replacement = original == 'F' ? '0' : (char) (original + 1);
        return digest.substring(0, index) + replacement
            + digest.substring(index + 1);
    }

    private static Path directory(String name) throws IOException {
        Path path = scratch.resolve(name);
        Files.createDirectories(path);
        return path;
    }

    private static AgentBundle bundle(String name, Map<String, String> overrides)
            throws IOException {
        Path path = directory(name);
        writeTensorFiles(path, new Random(name.hashCode()));
        return bundleAt(path, overrides);
    }

    private static AgentBundle bundleAt(
        Path path,
        Map<String, String> overrides
    ) throws IOException {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("schema", AgentBundle.SCHEMA);
        fields.put("role", "Kweebec_Razorleaf");
        fields.put("checkpoint_sha256", CHECKPOINT_SHA);
        fields.put("policy_tensor_identity_schema", PolicyTensorIdentity.SCHEMA);
        fields.put("policy_tensor_count",
            Integer.toString(PolicyTensorIdentity.TENSOR_NAMES.size()));
        fields.put("policy_tensor_content_sha256",
            PolicyTensorIdentity.sha256(path));
        fields.put("training_provenance_schema",
            AgentBundle.TRAINING_PROVENANCE_SCHEMA);
        fields.put("policy_purpose", "trained");
        fields.put("training_optimizer_updates", "4");
        fields.put("training_environment_steps", "1024");
        fields.put("training_demonstration_examples", "0");
        fields.put("training_source_metadata_sha256", CHECKPOINT_SHA);
        fields.put("observation_size",
            Integer.toString(AgentBundle.EXPECTED_OBSERVATION));
        fields.put("action_mask_size",
            Integer.toString(AgentBundle.EXPECTED_ACTIONS));
        fields.put("encoder_size", "64");
        fields.put("recurrent_size", "128");
        fields.put("observation_contract_sha256", TEST_POLICY_CONTRACT);
        fields.put("action_contract_sha256", TEST_POLICY_CONTRACT);
        fields.put("policy_projection_abi",
            AgentBundle.SUPPORTED_POLICY_PROJECTION_ABI);
        String scopeBits = fullScopeBits();
        fields.put("policy_action_scope_schema",
            AgentBundle.POLICY_ACTION_SCOPE_SCHEMA);
        fields.put("policy_action_scope_identity_schema",
            AgentBundle.POLICY_ACTION_SCOPE_IDENTITY_SCHEMA);
        fields.put("policy_action_scope_bits", scopeBits);
        fields.put("policy_action_scope_sha256",
            actionScopeSha256(TEST_POLICY_CONTRACT, scopeBits));
        fields.put("actor_kernel_std", "0.0036087273620069027");
        addCompatibilityFields(fields);
        fields.putAll(overrides);
        if (!overrides.containsKey("role")) {
            fields.put("role", fields.get("deployment_role"));
        }
        if (!overrides.containsKey("training_context_content_sha256")) {
            fields.put("training_context_content_sha256",
                trainingContextSha256(fields));
        }
        if (!overrides.containsKey("deployment_compatibility_content_sha256")) {
            fields.put("deployment_compatibility_content_sha256",
                compatibilitySha256(fields));
        }
        Files.writeString(path.resolve("role.txt"),
            fields.getOrDefault("deployment_role", "Kweebec_Razorleaf") + "\n",
            StandardCharsets.UTF_8);
        List<String> rows = fields.entrySet().stream()
            .map(entry -> entry.getKey() + "\t" + entry.getValue())
            .toList();
        Files.writeString(path.resolve("contract.txt"),
            String.join("\n", rows) + "\n", StandardCharsets.UTF_8);
        return AgentBundle.read(path);
    }

    private static void writeTensorFiles(Path path, Random random)
            throws IOException {
        for (String name : PolicyTensorIdentity.TENSOR_NAMES) {
            // Equal sizes make swaps shape-preserving. Contents remain unique.
            byte[] bytes = new byte[64];
            random.nextBytes(bytes);
            Files.write(path.resolve(name + ".bin"), bytes);
        }
    }

    private static void copyTensorFiles(Path source, Path destination)
            throws IOException {
        for (String name : PolicyTensorIdentity.TENSOR_NAMES) {
            Files.copy(
                source.resolve(name + ".bin"),
                destination.resolve(name + ".bin"));
        }
    }

    private static AgentBundle writeRaw(Path path, String... rows)
            throws IOException {
        Files.writeString(path.resolve("contract.txt"),
            String.join("\n", rows) + "\n", StandardCharsets.UTF_8);
        return AgentBundle.read(path);
    }

    private static Map<String, String> readFields(Path path) throws IOException {
        Map<String, String> result = new LinkedHashMap<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            int tab = line.indexOf('\t');
            if (tab > 0) {
                result.put(line.substring(0, tab), line.substring(tab + 1));
            }
        }
        return result;
    }

    private static void addCompatibilityFields(Map<String, String> fields) {
        fields.put("deployment_compatibility_schema",
            "hytalerl_policy_deployment_compatibility_v1");
        fields.put("training_context_schema",
            "hytalerl_policy_training_context_v1");
        fields.put("training_context_source_role", "Kweebec_Razorleaf");
        fields.put("training_context_policy_id", "0");
        fields.put("training_context_owner_count", "1");
        fields.put("training_context_owner_0_entity_index", "0");
        fields.put("training_context_owner_0_profile", "iron_sword");
        fields.put("training_context_owner_0_team_id", "0");
        fields.put("training_context_owner_0_controller", "policy");
        fields.put("training_context_owner_0_trainable", "true");
        fields.put("training_context_attestation_schema",
            PolicyDeploymentCompatibility.TRAINING_CONTEXT_ATTESTATION_SCHEMA);
        fields.put("training_context_attestation_derivation",
            PolicyDeploymentCompatibility
                .TRAINING_CONTEXT_ATTESTATION_DERIVATION);
        fields.put("training_context_runtime_config_content_sha256",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        fields.put("training_context_content_sha256", "");
        fields.put("deployment_role", "Kweebec_Razorleaf");
        fields.put("deployment_profile_mode", "frozen");
        fields.put("deployment_profile", "");
        fields.put("deployment_profile_content_sha256", "");
        fields.put("role_transfer_source", "");
        fields.put("profile_transfer_source", "");
        fields.put("transfer_reason", "");
    }

    /** Test-only independent implementation of the published v1 byte stream. */
    private static String compatibilitySha256(Map<String, String> fields) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update("HYTALERL_POLICY_DEPLOYMENT_COMPATIBILITY\0"
                .getBytes(StandardCharsets.UTF_8));
            writeInt(digest, 1);
            writeText(digest, fields.get("deployment_compatibility_schema"));
            writeText(digest, fields.get("training_context_schema"));
            writeText(digest, fields.get("training_context_source_role"));
            writeLong(digest,
                Long.parseLong(fields.get("training_context_policy_id")));
            int count = Integer.parseInt(
                fields.get("training_context_owner_count"));
            writeInt(digest, count);
            for (int index = 0; index < count; index++) {
                String prefix = "training_context_owner_" + index + "_";
                writeInt(digest,
                    Integer.parseInt(fields.get(prefix + "entity_index")));
                writeText(digest, fields.get(prefix + "profile"));
                writeInt(digest,
                    Integer.parseInt(fields.get(prefix + "team_id")));
                writeText(digest, fields.get(prefix + "controller"));
                digest.update((byte) (Boolean.parseBoolean(
                    fields.get(prefix + "trainable")) ? 1 : 0));
            }
            writeText(digest,
                fields.get("training_context_attestation_schema"));
            writeText(digest,
                fields.get("training_context_attestation_derivation"));
            writeText(digest,
                fields.get("training_context_runtime_config_content_sha256"));
            writeText(digest, fields.get("training_context_content_sha256"));
            writeText(digest, fields.get("deployment_role"));
            writeText(digest, fields.get("deployment_profile_mode"));
            writeText(digest, fields.get("deployment_profile"));
            writeText(digest,
                fields.get("deployment_profile_content_sha256"));
            writeText(digest, fields.get("role_transfer_source"));
            writeText(digest, fields.get("profile_transfer_source"));
            writeText(digest, fields.get("transfer_reason"));
            return HexFormat.of().withUpperCase()
                .formatHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    /** Independent test oracle for producer-attested selected owner rows. */
    private static String trainingContextSha256(Map<String, String> fields) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update("HYTALERL_POLICY_TRAINING_CONTEXT\0"
                .getBytes(StandardCharsets.UTF_8));
            writeInt(digest, 1);
            writeText(digest, fields.get("training_context_schema"));
            writeLong(digest,
                Long.parseLong(fields.get("training_context_policy_id")));
            int count = Integer.parseInt(
                fields.get("training_context_owner_count"));
            writeInt(digest, count);
            for (int index = 0; index < count; index++) {
                String prefix = "training_context_owner_" + index + "_";
                writeInt(digest,
                    Integer.parseInt(fields.get(prefix + "entity_index")));
                writeText(digest, fields.get(prefix + "profile"));
                writeInt(digest,
                    Integer.parseInt(fields.get(prefix + "team_id")));
                writeText(digest, fields.get(prefix + "controller"));
                digest.update((byte) (Boolean.parseBoolean(
                    fields.get(prefix + "trainable")) ? 1 : 0));
            }
            return HexFormat.of().withUpperCase().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private static void writeText(MessageDigest digest, String value) {
        byte[] encoded = value.getBytes(StandardCharsets.UTF_8);
        writeInt(digest, encoded.length);
        digest.update(encoded);
    }

    private static void writeInt(MessageDigest digest, int value) {
        digest.update(ByteBuffer.allocate(4).putInt(value).array());
    }

    private static void writeLong(MessageDigest digest, long value) {
        digest.update(ByteBuffer.allocate(8).putLong(value).array());
    }
}
