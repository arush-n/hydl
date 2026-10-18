package com.hytalerlbridge.policy.bundle;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Independent Java gate for a selected policy row's role/profile handoff.
 *
 * <p>Tensor shape and the Arsenal contract do not identify which profile a
 * policy learned to control.  Python writes the selected row's owner records;
 * Java reconstructs their canonical content identity, checks the separately
 * supplied {@code role.txt}, and hashes the exact installed live-profile
 * files. A trained live bundle also needs a producer attestation derived from
 * the actual JAX runtime config; caller-declared owner labels are insufficient.
 * A cross-context deployment is accepted only when the bundle names a
 * source-to-target mapping and a nonempty reason.
 */
public final class PolicyDeploymentCompatibility {

    public static final String SCHEMA =
        "hytalerl_policy_deployment_compatibility_v1";
    public static final String TRAINING_CONTEXT_SCHEMA =
        "hytalerl_policy_training_context_v1";
    public static final String TRAINING_CONTEXT_ATTESTATION_SCHEMA =
        "hytalerl_policy_training_context_attestation_v1";
    public static final String TRAINING_CONTEXT_ATTESTATION_DERIVATION =
        "runtime_config_entity_specs";
    private static final byte[] IDENTITY_DOMAIN =
        "HYTALERL_POLICY_DEPLOYMENT_COMPATIBILITY\0"
            .getBytes(StandardCharsets.UTF_8);
    private static final int IDENTITY_VERSION = 1;
    private static final byte[] TRAINING_CONTEXT_DOMAIN =
        "HYTALERL_POLICY_TRAINING_CONTEXT\0"
            .getBytes(StandardCharsets.UTF_8);
    private static final int TRAINING_CONTEXT_VERSION = 1;
    private static final int MAX_OWNERS = 4096;

    private PolicyDeploymentCompatibility() {
    }

    /** Return every reason the installed role/profile pairing must stay inert. */
    public static List<String> problems(
        Map<String, String> fields,
        Path directory,
        String purpose,
        String declaredRole
    ) {
        List<String> problems = new ArrayList<>();
        try {
            Parsed value = parse(fields);
            if (!SCHEMA.equals(value.schema())) {
                problems.add("unknown deployment compatibility schema '"
                    + value.schema() + "'");
            }
            if (!TRAINING_CONTEXT_SCHEMA.equals(value.trainingSchema())) {
                problems.add("unknown policy training context schema '"
                    + value.trainingSchema() + "'");
            }
            String actualIdentity = contentSha256(value);
            if (!isCanonicalSha256(value.contentSha256())) {
                problems.add(
                    "deployment compatibility content SHA-256 is malformed");
            } else if (!value.contentSha256().equals(actualIdentity)) {
                problems.add("deployment compatibility content hash "
                    + value.contentSha256() + " != reconstructed "
                    + actualIdentity);
            }

            if (!value.deploymentRole().equals(declaredRole)) {
                problems.add("bundle role differs from deployment role");
            }
            checkRoleFile(problems, directory, value.deploymentRole());

            if ("trained".equals(purpose) && value.owners().isEmpty()) {
                problems.add("trained deployment compatibility has no owner rows");
            }
            if ("trained".equals(purpose)
                    && "*".equals(value.deploymentRole())) {
                problems.add("trained deployment role cannot be wildcard '*'");
            }
            if (value.deploymentRole().equals(value.sourceRole())) {
                if (!value.roleTransferSource().isEmpty()) {
                    problems.add(
                        "exact role match carries an unnecessary transfer");
                }
            } else if (!value.roleTransferSource().equals(value.sourceRole())
                    || value.transferReason().isEmpty()) {
                problems.add(
                    "cross-role deployment has no explicit mapped reason");
            }

            if ("frozen".equals(value.profileMode())) {
                if (Files.isDirectory(directory.resolve("profile"))) {
                    problems.add(
                        "frozen compatibility unexpectedly has a profile");
                }
                if (!value.deploymentProfile().isEmpty()
                        || !value.profileContentSha256().isEmpty()
                        || !value.profileTransferSource().isEmpty()) {
                    problems.add(
                        "frozen compatibility carries live-profile fields");
                }
            } else if ("live".equals(value.profileMode())) {
                if ("trained".equals(purpose)) {
                    checkTrainingContextAttestation(problems, value);
                }
                checkLiveProfile(problems, directory, purpose, value);
            } else {
                problems.add("unknown deployment profile mode '"
                    + value.profileMode() + "'");
            }
            if (value.roleTransferSource().isEmpty()
                    && value.profileTransferSource().isEmpty()
                    && !value.transferReason().isEmpty()) {
                problems.add("exact context carries an unnecessary transfer reason");
            }
        } catch (IOException | IllegalArgumentException error) {
            problems.add("deployment compatibility is malformed: "
                + error.getMessage());
        }
        return problems;
    }

    private static void checkTrainingContextAttestation(
        List<String> problems,
        Parsed value
    ) {
        if (!TRAINING_CONTEXT_ATTESTATION_SCHEMA.equals(
                value.attestationSchema())
                || !TRAINING_CONTEXT_ATTESTATION_DERIVATION.equals(
                    value.attestationDerivation())
                || !isCanonicalSha256(value.runtimeConfigContentSha256())
                || !isCanonicalSha256(value.trainingContextContentSha256())
                || !value.trainingContextContentSha256().equals(
                    trainingContextContentSha256(value))) {
            problems.add("trained live compatibility has no valid "
                + "runtime-derived training-context attestation");
        }
    }

    private static void checkRoleFile(
        List<String> problems,
        Path directory,
        String expected
    ) throws IOException {
        Path rolePath = directory.resolve("role.txt");
        if (!Files.isRegularFile(rolePath)) {
            problems.add("deployment compatibility has no role.txt");
            return;
        }
        String actual = Files.readString(rolePath, StandardCharsets.UTF_8).trim();
        if (!actual.equals(expected)) {
            problems.add("role.txt '" + actual + "' != deployment role '"
                + expected + "'");
        }
    }

    private static void checkLiveProfile(
        List<String> problems,
        Path directory,
        String purpose,
        Parsed value
    ) throws IOException {
        Path profile = directory.resolve("profile");
        if (!Files.isDirectory(profile)) {
            problems.add("live compatibility has no profile directory");
            return;
        }
        if (value.deploymentProfile().isEmpty()) {
            problems.add("live compatibility has no profile name");
        }
        if (!isCanonicalSha256(value.profileContentSha256())) {
            problems.add("deployment profile content SHA-256 is malformed");
        } else {
            String actual = profileContentSha256(profile);
            if (!value.profileContentSha256().equals(actual)) {
                problems.add("deployment profile content hash "
                    + value.profileContentSha256() + " != installed " + actual);
            }
        }

        Set<String> sourceProfiles = new HashSet<>();
        for (Owner owner : value.owners()) {
            sourceProfiles.add(owner.profile());
        }
        if ("trained".equals(purpose)) {
            if (sourceProfiles.contains(value.deploymentProfile())) {
                if (!value.profileTransferSource().isEmpty()) {
                    problems.add(
                        "trained profile match carries an unnecessary transfer");
                }
            } else if (!sourceProfiles.contains(value.profileTransferSource())
                    || value.transferReason().isEmpty()) {
                problems.add(
                    "cross-profile deployment has no explicit mapped reason");
            }
        } else if (!value.profileTransferSource().isEmpty()) {
            problems.add("diagnostic profile carries a training transfer");
        }
    }

    /** Exact profile byte identity independently matching the Python exporter. */
    public static String profileContentSha256(Path profile) throws IOException {
        MessageDigest digest = sha256();
        List<Path> paths;
        try (var stream = Files.list(profile)) {
            paths = stream
                .sorted((left, right) -> compareUnsigned(
                    left.getFileName().toString().getBytes(StandardCharsets.UTF_8),
                    right.getFileName().toString().getBytes(StandardCharsets.UTF_8)))
                .toList();
        }
        for (Path path : paths) {
            if (Files.isSymbolicLink(path) || !Files.isRegularFile(path)) {
                throw new IOException(
                    "live profile must contain only flat regular files: "
                        + path.getFileName());
            }
            digest.update(path.getFileName().toString()
                .getBytes(StandardCharsets.UTF_8));
            digest.update((byte) 0);
            digest.update(Files.readAllBytes(path));
            digest.update((byte) 0);
        }
        return HexFormat.of().withUpperCase().formatHex(digest.digest());
    }

    private static int compareUnsigned(byte[] left, byte[] right) {
        int limit = Math.min(left.length, right.length);
        for (int index = 0; index < limit; index++) {
            int difference = Byte.toUnsignedInt(left[index])
                - Byte.toUnsignedInt(right[index]);
            if (difference != 0) {
                return difference;
            }
        }
        return Integer.compare(left.length, right.length);
    }

    private static Parsed parse(Map<String, String> fields) {
        int ownerCount = integer(
            fields, "training_context_owner_count", 0, MAX_OWNERS);
        long policyId = longInteger(
            fields, "training_context_policy_id", -1L, Long.MAX_VALUE);
        List<Owner> owners = new ArrayList<>(ownerCount);
        for (int index = 0; index < ownerCount; index++) {
            String prefix = "training_context_owner_" + index + "_";
            owners.add(new Owner(
                integer(fields, prefix + "entity_index", 0, Integer.MAX_VALUE),
                required(fields, prefix + "profile"),
                signedInteger(fields, prefix + "team_id"),
                required(fields, prefix + "controller"),
                bool(fields, prefix + "trainable")
            ));
        }
        List<Owner> sorted = owners.stream().sorted(OWNER_ORDER).toList();
        if (!owners.equals(sorted)) {
            throw new IllegalArgumentException(
                "policy training owners are not in canonical order");
        }
        if (new HashSet<>(owners).size() != owners.size()) {
            throw new IllegalArgumentException(
                "policy training owners contain duplicates");
        }
        return new Parsed(
            required(fields, "deployment_compatibility_schema"),
            required(fields, "deployment_compatibility_content_sha256"),
            required(fields, "training_context_schema"),
            required(fields, "training_context_source_role"),
            policyId,
            List.copyOf(owners),
            required(fields, "deployment_role"),
            required(fields, "deployment_profile_mode"),
            optional(fields, "deployment_profile"),
            optional(fields, "deployment_profile_content_sha256"),
            optional(fields, "training_context_attestation_schema"),
            optional(fields, "training_context_attestation_derivation"),
            optional(fields,
                "training_context_runtime_config_content_sha256"),
            optional(fields, "training_context_content_sha256"),
            optional(fields, "role_transfer_source"),
            optional(fields, "profile_transfer_source"),
            optional(fields, "transfer_reason")
        );
    }

    private static final Comparator<Owner> OWNER_ORDER = Comparator
        .comparingInt(Owner::entityIndex)
        .thenComparing(Owner::profile)
        .thenComparingInt(Owner::teamId)
        .thenComparing(Owner::controller)
        .thenComparing(Owner::trainable);

    private static String contentSha256(Parsed value) {
        MessageDigest digest = sha256();
        digest.update(IDENTITY_DOMAIN);
        writeInt(digest, IDENTITY_VERSION);
        writeText(digest, value.schema());
        writeText(digest, value.trainingSchema());
        writeText(digest, value.sourceRole());
        writeLong(digest, value.policyId());
        writeInt(digest, value.owners().size());
        for (Owner owner : value.owners()) {
            writeInt(digest, owner.entityIndex());
            writeText(digest, owner.profile());
            writeInt(digest, owner.teamId());
            writeText(digest, owner.controller());
            digest.update((byte) (owner.trainable() ? 1 : 0));
        }
        writeText(digest, value.attestationSchema());
        writeText(digest, value.attestationDerivation());
        writeText(digest, value.runtimeConfigContentSha256());
        writeText(digest, value.trainingContextContentSha256());
        writeText(digest, value.deploymentRole());
        writeText(digest, value.profileMode());
        writeText(digest, value.deploymentProfile());
        writeText(digest, value.profileContentSha256());
        writeText(digest, value.roleTransferSource());
        writeText(digest, value.profileTransferSource());
        writeText(digest, value.transferReason());
        return HexFormat.of().withUpperCase().formatHex(digest.digest());
    }

    private static String trainingContextContentSha256(Parsed value) {
        MessageDigest digest = sha256();
        digest.update(TRAINING_CONTEXT_DOMAIN);
        writeInt(digest, TRAINING_CONTEXT_VERSION);
        writeText(digest, value.trainingSchema());
        writeLong(digest, value.policyId());
        writeInt(digest, value.owners().size());
        for (Owner owner : value.owners()) {
            writeInt(digest, owner.entityIndex());
            writeText(digest, owner.profile());
            writeInt(digest, owner.teamId());
            writeText(digest, owner.controller());
            digest.update((byte) (owner.trainable() ? 1 : 0));
        }
        return HexFormat.of().withUpperCase().formatHex(digest.digest());
    }

    private static void writeText(MessageDigest digest, String value) {
        byte[] encoded = value.getBytes(StandardCharsets.UTF_8);
        writeInt(digest, encoded.length);
        digest.update(encoded);
    }

    private static void writeInt(MessageDigest digest, int value) {
        digest.update(ByteBuffer.allocate(Integer.BYTES).putInt(value).array());
    }

    private static void writeLong(MessageDigest digest, long value) {
        digest.update(ByteBuffer.allocate(Long.BYTES).putLong(value).array());
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError("Java runtime has no SHA-256", impossible);
        }
    }

    private static String required(Map<String, String> fields, String name) {
        String value = fields.get(name);
        if (value == null || value.isEmpty() || !value.equals(value.trim())
                || value.indexOf('\t') >= 0 || value.indexOf('\n') >= 0
                || value.indexOf('\r') >= 0) {
            throw new IllegalArgumentException(
                "missing or malformed compatibility field '" + name + "'");
        }
        return value;
    }

    private static String optional(Map<String, String> fields, String name) {
        String value = fields.get(name);
        if (value == null) {
            throw new IllegalArgumentException(
                "missing compatibility field '" + name + "'");
        }
        if (!value.equals(value.trim()) || value.indexOf('\t') >= 0
                || value.indexOf('\n') >= 0 || value.indexOf('\r') >= 0) {
            throw new IllegalArgumentException(
                "malformed compatibility field '" + name + "'");
        }
        return value;
    }

    private static int integer(
        Map<String, String> fields,
        String name,
        int minimum,
        int maximum
    ) {
        int value = signedInteger(fields, name);
        if (value < minimum || value > maximum) {
            throw new IllegalArgumentException(
                "compatibility field '" + name + "' is out of range");
        }
        return value;
    }

    private static int signedInteger(Map<String, String> fields, String name) {
        try {
            return Integer.parseInt(required(fields, name));
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException(
                "compatibility field '" + name + "' is not an integer");
        }
    }

    private static long longInteger(
        Map<String, String> fields,
        String name,
        long minimum,
        long maximum
    ) {
        try {
            long value = Long.parseLong(required(fields, name));
            if (value < minimum || value > maximum) {
                throw new IllegalArgumentException(
                    "compatibility field '" + name + "' is out of range");
            }
            return value;
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException(
                "compatibility field '" + name + "' is not an integer");
        }
    }

    private static boolean bool(Map<String, String> fields, String name) {
        String value = required(fields, name);
        if ("true".equals(value)) {
            return true;
        }
        if ("false".equals(value)) {
            return false;
        }
        throw new IllegalArgumentException(
            "compatibility field '" + name + "' is not Boolean");
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

    private record Owner(
        int entityIndex,
        String profile,
        int teamId,
        String controller,
        boolean trainable
    ) {
    }

    private record Parsed(
        String schema,
        String contentSha256,
        String trainingSchema,
        String sourceRole,
        long policyId,
        List<Owner> owners,
        String deploymentRole,
        String profileMode,
        String deploymentProfile,
        String profileContentSha256,
        String attestationSchema,
        String attestationDerivation,
        String runtimeConfigContentSha256,
        String trainingContextContentSha256,
        String roleTransferSource,
        String profileTransferSource,
        String transferReason
    ) {
    }
}
