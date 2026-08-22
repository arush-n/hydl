package com.hytalerlbridge.policy.bundle;

import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Data-driven attestation for the independently packaged bridge target.
 *
 * <p>The verifier contains class locations and a canonical tuple format, but
 * no release digest, role, profile, weapon, or actor-capacity special case.
 * A bundle supplies those values in {@code bridge-target.txt}; setup measures
 * the classes actually visible through the plugin classloader and fails before
 * policy weights are loaded or any actor-control system is registered.</p>
 */
public final class BridgeTargetAttestation {

    public static final String FILE_NAME = "bridge-target.txt";
    public static final String SCHEMA = "hytalerl_policy_bridge_target_v1";
    private static final byte[] DOMAIN =
        "HYTALERL_POLICY_BRIDGE_TARGET\0".getBytes(StandardCharsets.UTF_8);
    private static final int IDENTITY_VERSION = 1;

    private static final String COMBAT_FACADE =
        "com.hytalerlbridge.nativebackend.policy.combat."
            + "NativePolicyCombatFacade";
    private static final String WORLD_FACADE =
        "com.hytalerlbridge.nativebackend.NativePolicyWorldVerbFacade";
    private static final String WORLD_TRANSPORT =
        "com.hytalerlbridge.action.NativeWorldVerbRequest";
    private static final String ACTOR_CAPTURE =
        "com.hytalerlbridge.worldgen.policyactions.capture.model."
            + "NativePolicyWorldActionCapture";
    private static final String GROUP_ACTION =
        "com.hytalerlbridge.action.group.NativeGroupActionContract";
    private static final String DODGE_PROGRAM =
        "com.hytalerlbridge.combat.dodge.NativeDodgeProgram";

    private static final List<String> FILE_KEYS = List.of(
        "schema",
        "content_sha256",
        "bridge_jar_sha256",
        "combat_facade_schema",
        "combat_facade_version",
        "world_facade_schema",
        "world_facade_version",
        "world_transport_schema",
        "world_transport_version",
        "world_transport_sha256",
        "actor_capture_schema",
        "actor_capture_version",
        "actor_capture_sha256",
        "group_action_schema",
        "group_action_version",
        "group_action_sha256",
        "actor_capacity",
        "dodge_cooldown_id",
        "dodge_cooldown_seconds",
        "world_evidence_reject_reason"
    );

    private BridgeTargetAttestation() {}

    /** Offline bundle-to-flat-file binding; no bridge classes are loaded. */
    public static List<String> bundleProblems(
        Map<String, String> contract,
        Path directory
    ) {
        List<String> problems = new ArrayList<>();
        Path path = directory.resolve(FILE_NAME);
        boolean contractDeclared = contract.containsKey("bridge_target_schema")
            || contract.containsKey("bridge_target_content_sha256");
        boolean fileDeclared = Files.exists(path);
        if (!contractDeclared && !fileDeclared) {
            return problems;
        }
        if (!contractDeclared) {
            problems.add("bridge target file is not bound by contract.txt");
            return problems;
        }
        if (!Files.isRegularFile(path)) {
            problems.add("declared bridge target has no bridge-target.txt");
            return problems;
        }
        try {
            Target target = read(directory);
            String declaredSchema = contract.getOrDefault(
                "bridge_target_schema", "");
            String declaredContent = contract.getOrDefault(
                "bridge_target_content_sha256", "");
            if (!target.schema().equals(declaredSchema)) {
                problems.add("bridge target schema differs from contract.txt");
            }
            if (!target.contentSha256().equals(declaredContent)) {
                problems.add("bridge target content hash differs from contract.txt");
            }
        } catch (IOException | IllegalArgumentException error) {
            problems.add("bridge target attestation is malformed: "
                + error.getMessage());
        }
        return problems;
    }

    /** Runtime gate against the exact bridge classes visible to this plugin. */
    public static List<String> runtimeProblems(Path directory) {
        try {
            Target target = read(directory);
            ClassLoader loader = BridgeTargetAttestation.class.getClassLoader();
            return compare(target, capture(loader));
        } catch (ReflectiveOperationException | IOException
                | IllegalArgumentException error) {
            return List.of("bridge target cannot be attested: "
                + error.getMessage());
        }
    }

    public static Target read(Path directory) throws IOException {
        Path path = directory.resolve(FILE_NAME);
        if (!Files.isRegularFile(path)) {
            throw new IOException("missing " + FILE_NAME);
        }
        Map<String, String> fields = new LinkedHashMap<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            int tab = line.indexOf('\t');
            if (tab <= 0) {
                throw new IOException(
                    FILE_NAME + " line is not key<TAB>value: " + line);
            }
            String key = line.substring(0, tab);
            String value = line.substring(tab + 1);
            if (!value.equals(value.trim()) || value.indexOf('\0') >= 0) {
                throw new IOException("malformed bridge target field " + key);
            }
            if (fields.put(key, value) != null) {
                throw new IOException("repeated bridge target field " + key);
            }
        }
        if (!fields.keySet().equals(new java.util.LinkedHashSet<>(FILE_KEYS))) {
            throw new IOException("bridge target fields are not the exact v1 set");
        }
        Target target = new Target(
            fields.get("schema"),
            fields.get("content_sha256"),
            fields.get("bridge_jar_sha256"),
            fields.get("combat_facade_schema"),
            integer(fields, "combat_facade_version"),
            fields.get("world_facade_schema"),
            integer(fields, "world_facade_version"),
            fields.get("world_transport_schema"),
            integer(fields, "world_transport_version"),
            fields.get("world_transport_sha256"),
            fields.get("actor_capture_schema"),
            integer(fields, "actor_capture_version"),
            fields.get("actor_capture_sha256"),
            fields.get("group_action_schema"),
            integer(fields, "group_action_version"),
            fields.get("group_action_sha256"),
            integer(fields, "actor_capacity"),
            fields.get("dodge_cooldown_id"),
            decimal(fields, "dodge_cooldown_seconds"),
            fields.get("world_evidence_reject_reason")
        );
        String reconstructed = contentSha256(target);
        if (!target.contentSha256().equals(reconstructed)) {
            throw new IOException("bridge target content hash "
                + target.contentSha256() + " != reconstructed " + reconstructed);
        }
        return target;
    }

    static List<String> compare(Target expected, Actual actual) {
        List<String> problems = new ArrayList<>();
        mismatch(problems, "bridge JAR SHA-256",
            expected.bridgeJarSha256(), actual.bridgeJarSha256());
        mismatch(problems, "combat facade schema",
            expected.combatFacadeSchema(), actual.combatFacadeSchema());
        mismatch(problems, "combat facade version",
            expected.combatFacadeVersion(), actual.combatFacadeVersion());
        mismatch(problems, "World facade schema",
            expected.worldFacadeSchema(), actual.worldFacadeSchema());
        mismatch(problems, "World facade version",
            expected.worldFacadeVersion(), actual.worldFacadeVersion());
        mismatch(problems, "World transport schema",
            expected.worldTransportSchema(), actual.worldTransportSchema());
        mismatch(problems, "World transport version",
            expected.worldTransportVersion(), actual.worldTransportVersion());
        mismatch(problems, "World transport SHA-256",
            expected.worldTransportSha256(), actual.worldTransportSha256());
        mismatch(problems, "actor capture schema",
            expected.actorCaptureSchema(), actual.actorCaptureSchema());
        mismatch(problems, "actor capture version",
            expected.actorCaptureVersion(), actual.actorCaptureVersion());
        mismatch(problems, "actor capture SHA-256",
            expected.actorCaptureSha256(), actual.actorCaptureSha256());
        mismatch(problems, "group action schema",
            expected.groupActionSchema(), actual.groupActionSchema());
        mismatch(problems, "group action version",
            expected.groupActionVersion(), actual.groupActionVersion());
        mismatch(problems, "group action SHA-256",
            expected.groupActionSha256(), actual.groupActionSha256());
        mismatch(problems, "actor capacity",
            expected.actorCapacity(), actual.actorCapacity());
        mismatch(problems, "Dodge cooldown id",
            expected.dodgeCooldownId(), actual.dodgeCooldownId());
        if (Float.compare(expected.dodgeCooldownSeconds(),
                actual.dodgeCooldownSeconds()) != 0) {
            problems.add("bridge target Dodge cooldown seconds mismatch");
        }
        mismatch(problems, "World evidence reject reason",
            expected.worldEvidenceRejectReason(),
            actual.worldEvidenceRejectReason());
        return problems;
    }

    private static Actual capture(ClassLoader loader)
        throws ReflectiveOperationException, IOException {
        Class<?> combat = Class.forName(COMBAT_FACADE, true, loader);
        Class<?> world = Class.forName(WORLD_FACADE, true, loader);
        Class<?> transport = Class.forName(WORLD_TRANSPORT, true, loader);
        Class<?> actor = Class.forName(ACTOR_CAPTURE, true, loader);
        Class<?> group = Class.forName(GROUP_ACTION, true, loader);
        Class<?> dodge = Class.forName(DODGE_PROGRAM, true, loader);
        List<Class<?>> classes = List.of(
            combat, world, transport, actor, group, dodge);
        Path source = codeSource(classes.get(0));
        for (Class<?> type : classes.subList(1, classes.size())) {
            if (!source.equals(codeSource(type))) {
                throw new IOException(
                    "bridge target classes have different code sources");
            }
        }
        if (!Files.isRegularFile(source)) {
            throw new IOException("bridge code source is not a regular JAR");
        }
        return new Actual(
            sha256(source),
            (String) combat.getMethod("schema").invoke(null),
            (Integer) combat.getMethod("version").invoke(null),
            stringField(world, "SCHEMA"), intField(world, "VERSION"),
            stringField(transport, "SCHEMA"), intField(transport, "VERSION"),
            stringField(transport, "CONTRACT_SHA256"),
            stringField(actor, "SCHEMA"), intField(actor, "VERSION"),
            stringField(actor, "CONTRACT_SHA256"),
            stringField(group, "SCHEMA"), intField(group, "VERSION"),
            (String) group.getMethod("sha256").invoke(null),
            (Integer) group.getMethod("actorCapacity").invoke(null),
            stringField(dodge, "COOLDOWN_ID"),
            ((Number) dodge.getField("COOLDOWN_SECONDS").get(null)).floatValue(),
            stringField(transport, "POLICY_EVIDENCE_REQUIRED")
        );
    }

    static String contentSha256(Target target) {
        MessageDigest digest = digest();
        digest.update(DOMAIN);
        writeInt(digest, IDENTITY_VERSION);
        writeText(digest, target.schema());
        writeText(digest, target.bridgeJarSha256());
        writeText(digest, target.combatFacadeSchema());
        writeText(digest, Integer.toString(target.combatFacadeVersion()));
        writeText(digest, target.worldFacadeSchema());
        writeText(digest, Integer.toString(target.worldFacadeVersion()));
        writeText(digest, target.worldTransportSchema());
        writeText(digest, Integer.toString(target.worldTransportVersion()));
        writeText(digest, target.worldTransportSha256());
        writeText(digest, target.actorCaptureSchema());
        writeText(digest, Integer.toString(target.actorCaptureVersion()));
        writeText(digest, target.actorCaptureSha256());
        writeText(digest, target.groupActionSchema());
        writeText(digest, Integer.toString(target.groupActionVersion()));
        writeText(digest, target.groupActionSha256());
        writeText(digest, Integer.toString(target.actorCapacity()));
        writeText(digest, target.dodgeCooldownId());
        writeInt(digest, Float.floatToIntBits(target.dodgeCooldownSeconds()));
        writeText(digest, target.worldEvidenceRejectReason());
        return HexFormat.of().withUpperCase().formatHex(digest.digest());
    }

    static String sha256(Path path) throws IOException {
        MessageDigest digest = digest();
        try (var input = Files.newInputStream(path)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                if (read > 0) digest.update(buffer, 0, read);
            }
        }
        return HexFormat.of().withUpperCase().formatHex(digest.digest());
    }

    private static Path codeSource(Class<?> type) throws IOException {
        try {
            URI uri = type.getProtectionDomain().getCodeSource()
                .getLocation().toURI();
            return Path.of(uri).toRealPath();
        } catch (Exception error) {
            throw new IOException("cannot resolve code source for "
                + type.getName(), error);
        }
    }

    private static String stringField(Class<?> type, String name)
        throws ReflectiveOperationException {
        return (String) type.getField(name).get(null);
    }

    private static int intField(Class<?> type, String name)
        throws ReflectiveOperationException {
        return type.getField(name).getInt(null);
    }

    private static int integer(Map<String, String> fields, String name) {
        try {
            return Integer.parseInt(fields.get(name));
        } catch (RuntimeException error) {
            throw new IllegalArgumentException(name + " is not an integer");
        }
    }

    private static float decimal(Map<String, String> fields, String name) {
        try {
            return Float.parseFloat(fields.get(name));
        } catch (RuntimeException error) {
            throw new IllegalArgumentException(name + " is not a decimal");
        }
    }

    private static void mismatch(
        List<String> problems, String name, Object expected, Object actual
    ) {
        if (!expected.equals(actual)) {
            problems.add("bridge target " + name + " mismatch");
        }
    }

    private static void writeText(MessageDigest digest, String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        writeInt(digest, bytes.length);
        digest.update(bytes);
    }

    private static void writeInt(MessageDigest digest, int value) {
        digest.update(ByteBuffer.allocate(Integer.BYTES).putInt(value).array());
    }

    private static MessageDigest digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private static boolean canonicalSha256(String value) {
        return value != null && value.matches("[0-9A-F]{64}");
    }

    private static String required(String value, String name) {
        if (value == null || value.isEmpty() || !value.equals(value.trim())
                || value.indexOf('\0') >= 0) {
            throw new IllegalArgumentException(name + " is malformed");
        }
        return value;
    }

    public record Target(
        String schema,
        String contentSha256,
        String bridgeJarSha256,
        String combatFacadeSchema,
        int combatFacadeVersion,
        String worldFacadeSchema,
        int worldFacadeVersion,
        String worldTransportSchema,
        int worldTransportVersion,
        String worldTransportSha256,
        String actorCaptureSchema,
        int actorCaptureVersion,
        String actorCaptureSha256,
        String groupActionSchema,
        int groupActionVersion,
        String groupActionSha256,
        int actorCapacity,
        String dodgeCooldownId,
        float dodgeCooldownSeconds,
        String worldEvidenceRejectReason
    ) {
        public Target {
            if (!SCHEMA.equals(schema)) {
                throw new IllegalArgumentException(
                    "unsupported bridge target schema " + schema);
            }
            if (!canonicalSha256(contentSha256)
                    || !canonicalSha256(bridgeJarSha256)
                    || !canonicalSha256(worldTransportSha256)
                    || !canonicalSha256(actorCaptureSha256)
                    || !canonicalSha256(groupActionSha256)) {
                throw new IllegalArgumentException(
                    "bridge target SHA-256 is malformed");
            }
            combatFacadeSchema = required(
                combatFacadeSchema, "combat facade schema");
            worldFacadeSchema = required(worldFacadeSchema, "World facade schema");
            worldTransportSchema = required(
                worldTransportSchema, "World transport schema");
            actorCaptureSchema = required(
                actorCaptureSchema, "actor capture schema");
            groupActionSchema = required(groupActionSchema, "group action schema");
            dodgeCooldownId = required(dodgeCooldownId, "Dodge cooldown id");
            worldEvidenceRejectReason = required(
                worldEvidenceRejectReason, "World evidence reject reason");
            if (combatFacadeVersion < 1 || worldFacadeVersion < 1
                    || worldTransportVersion < 1 || actorCaptureVersion < 1
                    || groupActionVersion < 1 || actorCapacity < 1
                    || !Float.isFinite(dodgeCooldownSeconds)
                    || dodgeCooldownSeconds <= 0.0f) {
                throw new IllegalArgumentException(
                    "bridge target numeric boundary is invalid");
            }
        }

        public Target withComputedContent() {
            Target placeholder = this;
            return new Target(
                schema, BridgeTargetAttestation.contentSha256(placeholder),
                bridgeJarSha256,
                combatFacadeSchema, combatFacadeVersion,
                worldFacadeSchema, worldFacadeVersion,
                worldTransportSchema, worldTransportVersion,
                worldTransportSha256, actorCaptureSchema, actorCaptureVersion,
                actorCaptureSha256, groupActionSchema, groupActionVersion,
                groupActionSha256, actorCapacity, dodgeCooldownId,
                dodgeCooldownSeconds, worldEvidenceRejectReason);
        }
    }

    record Actual(
        String bridgeJarSha256,
        String combatFacadeSchema,
        int combatFacadeVersion,
        String worldFacadeSchema,
        int worldFacadeVersion,
        String worldTransportSchema,
        int worldTransportVersion,
        String worldTransportSha256,
        String actorCaptureSchema,
        int actorCaptureVersion,
        String actorCaptureSha256,
        String groupActionSchema,
        int groupActionVersion,
        String groupActionSha256,
        int actorCapacity,
        String dodgeCooldownId,
        float dodgeCooldownSeconds,
        String worldEvidenceRejectReason
    ) {}
}
