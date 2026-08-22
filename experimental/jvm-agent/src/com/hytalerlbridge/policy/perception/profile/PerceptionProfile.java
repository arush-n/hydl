package com.hytalerlbridge.policy.perception.profile;

import com.hytalerlbridge.policy.ObservationAssembler;
import com.hytalerlbridge.policy.Policy;
import com.hytalerlbridge.policy.Projection;
import com.hytalerlbridge.policy.perception.projection.AbilityProjection;
import com.hytalerlbridge.policy.perception.projection.ActionMaskProjection;
import com.hytalerlbridge.policy.perception.projection.MechanicsProjection;
import com.hytalerlbridge.policy.perception.projection.RecipeCandidateProjection;
import com.hytalerlbridge.policy.perception.acquisition.motion.DodgeMotionParameters;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Checkpoint-pinned constants needed to turn live server evidence into a row.
 *
 * <p>The live server owns changing facts such as health, cooldown remaining,
 * active interactions, and inventory contents. The checkpoint owns the
 * normalisers and authored loadout tensors used when that policy was trained.
 * Mixing either side with values copied from an arbitrary reset produces a
 * correctly shaped but semantically false observation, so this class loads
 * the static half once and exposes constructors that require the dynamic half
 * explicitly on every tick.
 *
 * <p>The on-disk format is intentionally the same exact-JAX profile directory
 * used by {@code ParityTest}. That makes the production loader testable against
 * the existing 52-state oracle rather than introducing a second serialization
 * format that could drift independently.
 */
public final class PerceptionProfile {

    private static final int RAW_ABILITY_COUNT = 868;
    private static final int RAW_DEFENSE_COUNT = 22;
    private static final int RAW_DODGE_COUNT = 8;

    private static final String[] IDENTITY_FILES = {
        "inputs.txt",
        "raw_resource_minimum.bin",
        "raw_resource_maximum.bin",
        "raw_defense.bin",
        "raw_dodge.bin",
        "raw_ability.bin",
        "status_programs.tsv",
        "status_omissions.txt",
        "ability_bindings.tsv",
        "guard_bindings.tsv",
        "recipe_encoder_contract.tsv",
        "recipe_encoder_input_kernel.bin",
        "recipe_encoder_input_bias.bin",
        "recipe_encoder_output_kernel.bin",
        "recipe_encoder_output_bias.bin",
        "recipe_encoder_requirement_kernel.bin",
        "recipe_encoder_requirement_bias.bin",
        "recipe_encoder_candidate_kernel.bin",
        "recipe_encoder_candidate_bias.bin",
    };
    private static final String[] OPTIONAL_IDENTITY_FILES = {
        "dodge_motion.tsv",
    };

    private final ObservationAssembler.Template template;
    private final Projection.Params params;
    private final float maximumDropHeight;
    private final float agentForcePerAxisDeadzone;
    private final float[] resourceMinimum;
    private final float[] resourceMaximum;
    private final float[] dodgeInvulnerabilityDuration;
    private final float[] dodgeForce;
    private final float dodgeCost;
    private final DodgeMotionParameters dodgeMotion;
    private final AbilityStatic ability;
    private final AbilityBinding[] agentAbilityBindings;
    private final AbilityBinding[] targetAbilityBindings;
    private final GuardBinding agentGuardBinding;
    private final GuardBinding targetGuardBinding;
    private final StatusProgramCatalog statusPrograms;
    private final boolean[] skillMask;
    private final boolean jumpConfigured;
    private final boolean guardConfigured;
    private final RecipeCandidateProjection.Parameters recipeCandidateEncoder;
    private final String sha256;

    private PerceptionProfile(Path root) throws IOException {
        template = ObservationAssembler.load(root);
        Map<String, Double> values = readValues(root.resolve("inputs.txt"));
        params = new Projection.Params(
            required(values, "agent_max_speed"),
            required(values, "vertical_speed_scale"),
            required(values, "agent_max_health"),
            required(values, "agent_attack_pause_max_seconds"),
            required(values, "regen_delay_ticks"),
            required(values, "agent_walk_max_fall_speed"),
            required(values, "loaded_dt"),
            required(values, "wire_fixed_point_scale"),
            required(values, "target_chase_speed"),
            required(values, "target_max_health"),
            required(values, "sensor_range"),
            required(values, "facing_error_degrees_scale"),
            required(values, "head_pitch_degrees_scale"),
            required(values, "target_attack_index_divisor")
        );
        double expectedAbilitySlotDivisor =
            AbilityProjection.ABILITY_CAPACITY - 1.0;
        if (params.targetAbilitySlotDivisor() != expectedAbilitySlotDivisor) {
            throw new IOException(
                "perception profile uses target ability-slot divisor "
                    + params.targetAbilitySlotDivisor() + ", expected "
                    + expectedAbilitySlotDivisor);
        }
        maximumDropHeight = (float) required(
            values, "agent_walk_max_drop_height");
        agentForcePerAxisDeadzone = (float) required(
            values, "agent_force_per_axis_deadzone");
        if (!Float.isFinite(agentForcePerAxisDeadzone)
            || agentForcePerAxisDeadzone < 0.0f) {
            throw new IOException(
                "agent force deadzone must be finite and nonnegative");
        }

        resourceMinimum = requireWidth(
            Policy.read(root, "raw_resource_minimum"),
            MechanicsProjection.RESOURCE_VALUES,
            "resource minimum"
        );
        resourceMaximum = requireWidth(
            Policy.read(root, "raw_resource_maximum"),
            MechanicsProjection.RESOURCE_VALUES,
            "resource maximum"
        );

        float[] defense = requireWidth(
            Policy.read(root, "raw_defense"), RAW_DEFENSE_COUNT, "raw defense");
        dodgeInvulnerabilityDuration = slice(defense, 6, 2);
        dodgeForce = slice(defense, 14, 2);

        float[] dodge = requireWidth(
            Policy.read(root, "raw_dodge"), RAW_DODGE_COUNT, "raw dodge");
        dodgeCost = dodge[7];
        dodgeMotion = DodgeMotionParameters.loadOptional(
            root.resolve("dodge_motion.tsv"));

        float[] rawAbility = requireWidth(
            Policy.read(root, "raw_ability"), RAW_ABILITY_COUNT, "raw ability");
        ability = new AbilityStatic(
            slice(rawAbility, 28, 32),
            slice(rawAbility, 60, 32),
            slice(rawAbility, 128, 224),
            integers(rawAbility, 352, 224),
            slice(rawAbility, 576, 224),
            booleans(rawAbility, 800, 32),
            booleans(rawAbility, 832, 2),
            booleans(rawAbility, 834, 2)
        );
        AbilityBinding[][] bindings = readAbilityBindings(
            root.resolve("ability_bindings.tsv"),
            ability.authoredMask()
        );
        agentAbilityBindings = bindings[0];
        targetAbilityBindings = bindings[1];
        GuardBinding[] guards = readGuardBindings(
            root.resolve("guard_bindings.tsv"));
        agentGuardBinding = guards[0];
        targetGuardBinding = guards[1];
        statusPrograms = StatusProgramCatalog.load(root);

        skillMask = Policy.readMask(root, "skill_action_mask");
        requireWidth(skillMask, 9, "skill action mask");
        boolean[] jump = Policy.readMask(root, "jump_action_mask");
        boolean[] guard = Policy.readMask(root, "guard_action_mask");
        requireWidth(jump, 1, "jump action mask");
        requireWidth(guard, 1, "guard action mask");
        jumpConfigured = jump[0];
        guardConfigured = guard[0];
        if (guardConfigured != (agentGuardBinding != null)) {
            throw new IOException(
                "agent guard mask and native guard binding disagree");
        }
        recipeCandidateEncoder = RecipeCandidateProjection.Parameters.load(root);
        sha256 = fingerprint(root);
    }

    public static PerceptionProfile load(Path root) throws IOException {
        return new PerceptionProfile(root);
    }

    public ObservationAssembler.Template template() {
        return template;
    }

    public Projection.Params params() {
        return params;
    }

    public float maximumDropHeight() {
        return maximumDropHeight;
    }

    public float agentForcePerAxisDeadzone() {
        return agentForcePerAxisDeadzone;
    }

    public AbilityBinding[] agentAbilityBindings() {
        return agentAbilityBindings.clone();
    }

    public AbilityBinding[] targetAbilityBindings() {
        return targetAbilityBindings.clone();
    }

    public String agentItemId() {
        return itemId(agentAbilityBindings, agentGuardBinding);
    }

    public String targetItemId() {
        return itemId(targetAbilityBindings, targetGuardBinding);
    }

    public GuardBinding agentGuardBinding() {
        return agentGuardBinding;
    }

    public GuardBinding targetGuardBinding() {
        return targetGuardBinding;
    }

    public float[] resourceMinimum() {
        return resourceMinimum.clone();
    }

    public float[] resourceMaximum() {
        return resourceMaximum.clone();
    }

    public float[] resourceSpan() {
        float[] result = new float[resourceMinimum.length];
        for (int index = 0; index < result.length; index++) {
            result[index] = resourceMaximum[index] - resourceMinimum[index];
        }
        return result;
    }

    public float[] dodgeInvulnerabilityDuration() {
        return dodgeInvulnerabilityDuration.clone();
    }

    public float[] dodgeForce() {
        return dodgeForce.clone();
    }

    public float dodgeCost() {
        return dodgeCost;
    }

    /** Static prospective-motion inputs; null means this older profile closes Dodge. */
    public DodgeMotionParameters dodgeMotion() {
        return dodgeMotion;
    }

    public boolean[] skillMask() {
        return skillMask.clone();
    }

    public boolean jumpConfigured() {
        return jumpConfigured;
    }

    public boolean guardConfigured() {
        return guardConfigured;
    }

    /** Exact fixed projection selected by the recipe-observation contract. */
    public RecipeCandidateProjection.Parameters recipeCandidateEncoder() {
        return recipeCandidateEncoder;
    }

    public StatusProgramCatalog statusPrograms() {
        return statusPrograms;
    }

    /** Content identity for every static scalar consumed by this loader. */
    public String sha256() {
        return sha256;
    }

    public MechanicsProjection.ResourceInput resources(float[] current) {
        return new MechanicsProjection.ResourceInput(
            current, resourceMinimum, resourceMaximum);
    }

    public MechanicsProjection.ResourceInput resources(
        float[] current, boolean[] available
    ) {
        return new MechanicsProjection.ResourceInput(
            current, resourceMinimum, resourceMaximum, available);
    }

    public ActionMaskProjection.DodgeInput dodge(
        boolean[] corridorClear,
        boolean movementEnabled,
        boolean alive,
        float stamina
    ) {
        return new ActionMaskProjection.DodgeInput(
            corridorClear, movementEnabled, alive, stamina, dodgeCost);
    }

    /** Combine live lifecycle state with checkpoint-pinned ability metadata. */
    public AbilityProjection.Input abilities(
        float[] resources,
        float[] remainingCooldownSeconds,
        int[] activeAbilitySlot,
        float[] abilityElapsedSeconds,
        boolean[] legal
    ) {
        return new AbilityProjection.Input(
            resources,
            resourceSpan(),
            ability.durationSeconds(),
            ability.cooldownSeconds(),
            remainingCooldownSeconds,
            activeAbilitySlot,
            abilityElapsedSeconds,
            ability.resourceCost(),
            ability.resourceCostKind(),
            ability.resourceMinimum(),
            ability.authoredMask(),
            ability.equipped(),
            ability.overflow(),
            legal
        );
    }

    /** Static authored ability data. Every accessor returns a private copy. */
    public record AbilityStatic(
        float[] durationSeconds,
        float[] cooldownSeconds,
        float[] resourceCost,
        int[] resourceCostKind,
        float[] resourceMinimum,
        boolean[] authoredMask,
        boolean[] equipped,
        boolean[] overflow
    ) {
        public AbilityStatic {
            durationSeconds = durationSeconds.clone();
            cooldownSeconds = cooldownSeconds.clone();
            resourceCost = resourceCost.clone();
            resourceCostKind = resourceCostKind.clone();
            resourceMinimum = resourceMinimum.clone();
            authoredMask = authoredMask.clone();
            equipped = equipped.clone();
            overflow = overflow.clone();
        }

        @Override public float[] durationSeconds() { return durationSeconds.clone(); }
        @Override public float[] cooldownSeconds() { return cooldownSeconds.clone(); }
        @Override public float[] resourceCost() { return resourceCost.clone(); }
        @Override public int[] resourceCostKind() { return resourceCostKind.clone(); }
        @Override public float[] resourceMinimum() { return resourceMinimum.clone(); }
        @Override public boolean[] authoredMask() { return authoredMask.clone(); }
        @Override public boolean[] equipped() { return equipped.clone(); }
        @Override public boolean[] overflow() { return overflow.clone(); }
    }

    /** Exact native interaction identity for one authored learner slot. */
    public record AbilityBinding(
        int slot,
        String itemId,
        String interactionId,
        String interactionType,
        double requestedChargeSeconds
    ) {
        public AbilityBinding {
            if (slot < 0 || slot >= AbilityProjection.ABILITY_CAPACITY
                || itemId == null || itemId.isBlank()
                || interactionId == null || interactionId.isBlank()
                || interactionType == null || interactionType.isBlank()
                || !Double.isFinite(requestedChargeSeconds)
                || requestedChargeSeconds < 0.0
                || requestedChargeSeconds > Float.MAX_VALUE) {
                throw new IllegalArgumentException(
                    "invalid native ability binding");
            }
        }
    }

    private static AbilityBinding[][] readAbilityBindings(
        Path path,
        boolean[] authoredMask
    ) throws IOException {
        @SuppressWarnings("unchecked")
        List<AbilityBinding>[] rows = new List[] {
            new ArrayList<AbilityBinding>(),
            new ArrayList<AbilityBinding>(),
        };
        @SuppressWarnings("unchecked")
        Set<Integer>[] seen = new Set[] {
            new HashSet<Integer>(),
            new HashSet<Integer>(),
        };
        String[] itemIds = {null, null};
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] parts = line.split("\\t", -1);
            if (parts.length != 6) {
                throw new IOException("invalid ability binding row: " + line);
            }
            int entity = switch (parts[0]) {
                case "agent" -> 0;
                case "target" -> 1;
                default -> throw new IOException(
                    "invalid ability binding actor: " + parts[0]);
            };
            int slot;
            try {
                slot = Integer.parseInt(parts[1]);
            } catch (NumberFormatException invalid) {
                throw new IOException("invalid ability binding slot: " + line,
                    invalid);
            }
            AbilityBinding binding;
            try {
                binding = new AbilityBinding(
                    slot,
                    parts[2],
                    parts[3],
                    parts[4],
                    Double.parseDouble(parts[5])
                );
            } catch (IllegalArgumentException invalid) {
                throw new IOException("invalid ability binding row: " + line,
                    invalid);
            }
            if (!seen[entity].add(slot)) {
                throw new IOException(
                    "duplicate ability binding slot " + parts[0] + "/" + slot);
            }
            String previousItem = itemIds[entity];
            if (previousItem != null && !previousItem.equals(binding.itemId())) {
                throw new IOException(
                    "one actor has multiple native ability items");
            }
            itemIds[entity] = binding.itemId();
            int authoredIndex = entity * AbilityProjection.ABILITY_CAPACITY
                + slot;
            if (authoredIndex >= authoredMask.length
                || !authoredMask[authoredIndex]) {
                throw new IOException(
                    "native binding points at an unauthored learner slot: "
                        + parts[0] + "/" + slot);
            }
            rows[entity].add(binding);
        }
        AbilityBinding[][] result = new AbilityBinding[2][];
        for (int entity = 0; entity < 2; entity++) {
            rows[entity].sort(Comparator.comparingInt(AbilityBinding::slot));
            result[entity] = rows[entity].toArray(AbilityBinding[]::new);
        }
        return result;
    }

    /** Exact native guard root for one actor, if the profile authors guard. */
    public record GuardBinding(
        String itemId,
        String interactionId,
        String interactionType
    ) {
        public GuardBinding {
            if (itemId == null || itemId.isBlank()
                || interactionId == null || interactionId.isBlank()
                || interactionType == null || interactionType.isBlank()) {
                throw new IllegalArgumentException(
                    "invalid native guard binding");
            }
        }
    }

    private static GuardBinding[] readGuardBindings(Path path)
        throws IOException {
        GuardBinding[] result = new GuardBinding[2];
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] parts = line.split("\\t", -1);
            if (parts.length != 4) {
                throw new IOException("invalid guard binding row: " + line);
            }
            int entity = switch (parts[0]) {
                case "agent" -> 0;
                case "target" -> 1;
                default -> throw new IOException(
                    "invalid guard binding actor: " + parts[0]);
            };
            if (result[entity] != null) {
                throw new IOException(
                    "duplicate guard binding actor: " + parts[0]);
            }
            try {
                result[entity] = new GuardBinding(
                    parts[1], parts[2], parts[3]);
            } catch (IllegalArgumentException invalid) {
                throw new IOException("invalid guard binding row: " + line,
                    invalid);
            }
        }
        return result;
    }

    private static String itemId(
        AbilityBinding[] bindings,
        GuardBinding guard
    ) {
        String abilityItem = bindings.length == 0 ? "" : bindings[0].itemId();
        if (guard == null) return abilityItem;
        if (!abilityItem.isEmpty() && !abilityItem.equals(guard.itemId())) {
            throw new IllegalStateException(
                "ability and guard bindings name different items");
        }
        return guard.itemId();
    }

    private static Map<String, Double> readValues(Path path) throws IOException {
        Map<String, Double> values = new HashMap<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank()) {
                continue;
            }
            String[] parts = line.split("\\t", -1);
            if (parts.length != 2 || parts[0].isBlank()) {
                throw new IOException("invalid perception scalar row: " + line);
            }
            double value;
            try {
                value = Double.parseDouble(parts[1]);
            } catch (NumberFormatException invalid) {
                throw new IOException(
                    "invalid perception scalar " + parts[0] + " at " + path,
                    invalid
                );
            }
            if (!Double.isFinite(value) || values.put(parts[0], value) != null) {
                throw new IOException(
                    "duplicate or non-finite perception scalar " + parts[0]);
            }
        }
        return values;
    }

    private static double required(Map<String, Double> values, String name)
        throws IOException {
        Double value = values.get(name);
        if (value == null) {
            throw new IOException("perception profile has no scalar " + name);
        }
        return value;
    }

    private static String fingerprint(Path root) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (String name : IDENTITY_FILES) {
                digest.update(name.getBytes(StandardCharsets.UTF_8));
                digest.update((byte) 0);
                digest.update(Files.readAllBytes(root.resolve(name)));
                digest.update((byte) 0);
            }
            for (String name : OPTIONAL_IDENTITY_FILES) {
                Path path = root.resolve(name);
                if (!Files.isRegularFile(path)) continue;
                digest.update(name.getBytes(StandardCharsets.UTF_8));
                digest.update((byte) 0);
                digest.update(Files.readAllBytes(path));
                digest.update((byte) 0);
            }
            return java.util.HexFormat.of().withUpperCase()
                .formatHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError("Java runtime has no SHA-256", impossible);
        }
    }

    private static float[] requireWidth(
        float[] values, int expected, String name
    ) {
        if (values.length != expected) {
            throw new IllegalArgumentException(
                name + " width drift: " + values.length + " != " + expected);
        }
        return values;
    }

    private static void requireWidth(
        boolean[] values, int expected, String name
    ) {
        if (values.length != expected) {
            throw new IllegalArgumentException(
                name + " width drift: " + values.length + " != " + expected);
        }
    }

    private static float[] slice(float[] values, int start, int length) {
        return Arrays.copyOfRange(values, start, start + length);
    }

    private static int[] integers(float[] values, int start, int length) {
        int[] result = new int[length];
        for (int index = 0; index < length; index++) {
            float value = values[start + index];
            int rounded = Math.round(value);
            if (!Float.isFinite(value) || value != rounded) {
                throw new IllegalArgumentException(
                    "non-integral ability metadata at " + (start + index));
            }
            result[index] = rounded;
        }
        return result;
    }

    private static boolean[] booleans(float[] values, int start, int length) {
        boolean[] result = new boolean[length];
        for (int index = 0; index < length; index++) {
            float value = values[start + index];
            if (value != 0.0f && value != 1.0f) {
                throw new IllegalArgumentException(
                    "non-boolean ability metadata at " + (start + index));
            }
            result[index] = value != 0.0f;
        }
        return result;
    }
}
