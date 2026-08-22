package com.hytalerlbridge.policy;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Builds the 8,271-value policy observation from structured evidence, in Java.
 *
 * <p>Port of {@code observation/v3/policy/layout.py:arsenal_policy_observation}.
 * Thirty-seven groups concatenated in a fixed order. Two semantics carry the
 * risk:
 *
 * <ul>
 *   <li><b>Agent selection.</b> Per-entity arrays keep an entity axis; only row
 *       {@code AGENT_ENTITY} enters the vector. Taking the wrong row still
 *       produces a well-formed 8,271-vector describing the opponent.</li>
 *   <li><b>Mask-before-append.</b> A masked-off value is multiplied to zero and
 *       the mask is appended <i>separately</i>. So a zero means "not
 *       applicable" only in combination with its mask bit; writing the raw
 *       value and trusting the mask downstream would silently change what the
 *       network sees.</li>
 * </ul>
 *
 * <p>Shapes are contract-pinned constants, asserted against the actual file
 * lengths on load, so a contract change fails loudly here instead of producing
 * a plausible but misaligned vector.
 */
public final class ObservationAssembler {

    public static final int AGENT_ENTITY = 0;
    public static final int OBSERVATION_SIZE = 8272;

    /** Which of the 24 combat floats reach the policy (18 of them). */
    private static final int[] COMBAT_FLOAT_INDICES = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23,
    };

    private static final int ENTITIES = 2;
    private static final int SELF = 16, TARGET = 16, COMBAT = 24;
    // DEFENSE moved 7 -> 8 on 2026-08-22. The eighth slot is
    // `locomotion_stamina_fraction`: the sprint budget over
    // PLAYER_STAMINA_MAXIMUM, which is a different bar from the guard/ability
    // stamina in the columns above. It is the whole 8271 -> 8272 observation
    // delta. The bridge already sends it (EvidenceCapture defenseValues[7]).
    private static final int RESOURCES = 7, DEFENSE = 8;
    private static final int STATUSES = 8, STATUS_FEATURES = 6;
    private static final int ABILITIES = 16, ABILITY_FEATURES = 11;
    private static final int ACTOR_WORLD = 5, ACTOR_WORLD_MASKS = 3;
    private static final int MOVEMENT = 23;
    private static final int GEOMETRY_TOKENS = 44, GEOMETRY_FEATURES = 131;
    private static final int LIGHT_FEATURES = 10;
    private static final int CONTAINERS = 6;
    private static final int INVENTORY_TOKENS = 76, INVENTORY_FEATURES = 7;
    private static final int SKILLS = 9, DODGES = 4;
    private static final int BLOCK_CANDIDATES = 16, BLOCK_FEATURES = 28;
    private static final int RECIPES = 16, RECIPE_FEATURES = 32;

    private final Template template;
    private final java.util.Map<String, float[]> overrides;
    private final java.util.Map<String, boolean[]> maskOverrides;
    private int cursor;
    private final float[] out = new float[OBSERVATION_SIZE];

    private ObservationAssembler(
        Template template,
        java.util.Map<String, float[]> overrides,
        java.util.Map<String, boolean[]> maskOverrides
    ) {
        this.template = template;
        this.overrides = overrides;
        this.maskOverrides = maskOverrides;
    }

    public static float[] assemble(Path groupDirectory) throws IOException {
        return assemble(load(groupDirectory), Map.of(), Map.of());
    }

    /**
     * Assemble with some float groups supplied in memory instead of read from
     * disk.
     *
     * <p>Exists for the end-to-end parity test, which derives
     * {@code base_self_f32}, {@code base_target_f32} and
     * {@code base_combat_f32} in Java and needs them to travel through the real
     * layout rather than a copy of it. Overrides are still length-checked
     * against the contract like any loaded group.
     */
    public static float[] assemble(
        Path groupDirectory, java.util.Map<String, float[]> overrides
    ) throws IOException {
        return assemble(load(groupDirectory), overrides, Map.of());
    }

    /** Assemble with both numeric and boolean evidence derived in memory. */
    public static float[] assemble(
        Path groupDirectory,
        java.util.Map<String, float[]> overrides,
        java.util.Map<String, boolean[]> maskOverrides
    ) throws IOException {
        return assemble(load(groupDirectory), overrides, maskOverrides);
    }

    /** Load and contract-check every static group once for per-tick reuse. */
    public static Template load(Path groupDirectory) throws IOException {
        return new Template(groupDirectory);
    }

    /** Assemble from a cached template without touching the filesystem. */
    public static float[] assemble(
        Template template,
        java.util.Map<String, float[]> overrides,
        java.util.Map<String, boolean[]> maskOverrides
    ) {
        return new ObservationAssembler(template, overrides, maskOverrides).build();
    }

    private float[] build() {
        float[] selfF32 = f32("base_self_f32", SELF);
        float[] targetF32 = f32("base_target_f32", TARGET);
        boolean[] targetMask = bool("base_target_mask", 1);
        float[] combatF32 = f32("base_combat_f32", COMBAT);

        float[] resourceF32 = agentRow(f32("resource_f32", ENTITIES * RESOURCES), RESOURCES);
        boolean[] resourceMask = agentRow(bool("resource_mask", ENTITIES * RESOURCES), RESOURCES);
        float[] defenseF32 = agentRow(f32("defense_f32", ENTITIES * DEFENSE), DEFENSE);

        float[] statusF32 = agentRow(
                f32("status_f32", ENTITIES * STATUSES * STATUS_FEATURES),
                STATUSES * STATUS_FEATURES);
        boolean[] statusMask = agentRow(bool("status_mask", ENTITIES * STATUSES), STATUSES);

        float[] abilityF32 = agentRow(
                f32("ability_f32", ENTITIES * ABILITIES * ABILITY_FEATURES),
                ABILITIES * ABILITY_FEATURES);
        boolean[] abilityMask = agentRow(bool("ability_mask", ENTITIES * ABILITIES), ABILITIES);
        boolean[] abilityLegal = agentRow(bool("ability_legal", ENTITIES * ABILITIES), ABILITIES);

        float[] actorWorldF32 = f32("actor_world_f32", ACTOR_WORLD);
        boolean[] actorWorldMask = bool("actor_world_mask", ACTOR_WORLD_MASKS);
        float[] movementF32 = f32("movement_state_f32", MOVEMENT);
        boolean[] movementMask = bool("movement_state_mask", MOVEMENT);

        float[] geometryF32 = f32("geometry_token_f32", GEOMETRY_TOKENS * GEOMETRY_FEATURES);
        boolean[] geometryMask = bool("geometry_token_mask", GEOMETRY_TOKENS);
        boolean[] geometryAvailable = bool("geometry_available", 1);
        float[] lightF32 = f32("light_token_f32", GEOMETRY_TOKENS * LIGHT_FEATURES);
        boolean[] lightAvailable = bool("light_available", 1);

        float[] containerF32 = f32("inventory_container_f32", CONTAINERS);
        boolean[] containerMask = bool("inventory_container_mask", CONTAINERS);
        float[] inventoryF32 = f32("inventory_token_f32", INVENTORY_TOKENS * INVENTORY_FEATURES);
        boolean[] inventoryMask = bool("inventory_token_mask", INVENTORY_TOKENS);
        boolean[] inventoryAvailable = bool("inventory_available", 1);

        boolean[] skillMask = bool("skill_action_mask", SKILLS);
        boolean[] jumpMask = bool("jump_action_mask", 1);
        boolean[] guardMask = bool("guard_action_mask", 1);
        boolean[] dodgeMask = bool("dodge_action_mask", DODGES);
        boolean[] valid = bool("valid", 1);

        float[] blockF32 = f32("block_candidate_f32", BLOCK_CANDIDATES * BLOCK_FEATURES);
        boolean[] blockMask = bool("block_candidate_mask", BLOCK_CANDIDATES);
        boolean[] blockAvailable = bool("block_available", 1);
        float[] recipeF32 = f32("recipe_embedding", RECIPES * RECIPE_FEATURES);
        boolean[] recipeMask = bool("recipe_mask", RECIPES);
        boolean[] recipeAvailable = bool("recipe_available", 1);

        // actor_world values are gated by 3 masks fanned out over 5 slots as
        // (m0, m1, m1, m2, m2) -- layout.py:303-312.
        boolean[] actorWorldValueMask = {
            actorWorldMask[0], actorWorldMask[1], actorWorldMask[1],
            actorWorldMask[2], actorWorldMask[2],
        };

        // Candidate rows are additionally gated by availability and validity,
        // and the gated mask is what zeroes the features (layout.py:363-368).
        boolean[] blockGated = and(blockMask, blockAvailable[0] && valid[0]);
        boolean[] recipeGated = and(recipeMask, recipeAvailable[0] && valid[0]);

        put(selfF32);
        put(targetF32);
        put(targetMask);
        put(take(combatF32, COMBAT_FLOAT_INDICES));
        put(scale(resourceF32, resourceMask, 1));
        put(resourceMask);
        put(defenseF32);
        put(scale(statusF32, statusMask, STATUS_FEATURES));
        put(statusMask);
        put(scale(abilityF32, abilityMask, ABILITY_FEATURES));
        put(abilityMask);
        put(abilityLegal);
        put(scale(actorWorldF32, actorWorldValueMask, 1));
        put(actorWorldMask);
        put(scale(movementF32, movementMask, 1));
        put(movementMask);
        put(geometryF32);
        put(geometryMask);
        put(geometryAvailable);
        put(lightF32);
        put(lightAvailable);
        put(containerF32);
        put(containerMask);
        put(inventoryF32);
        put(inventoryMask);
        put(inventoryAvailable);
        put(skillMask);
        put(jumpMask);
        put(guardMask);
        put(dodgeMask);
        put(valid);
        put(scale(blockF32, blockGated, BLOCK_FEATURES));
        put(blockGated);
        put(new boolean[] {blockAvailable[0] && valid[0]});
        put(scale(recipeF32, recipeGated, RECIPE_FEATURES));
        put(recipeGated);
        put(new boolean[] {recipeAvailable[0] && valid[0]});

        if (cursor != OBSERVATION_SIZE) {
            throw new IllegalStateException(
                    "assembled " + cursor + " columns, expected " + OBSERVATION_SIZE);
        }
        return out;
    }

    // -- group operations ----------------------------------------------------

    /** Row {@code AGENT_ENTITY} of an entity-major array. */
    private static float[] agentRow(float[] values, int stride) {
        float[] row = new float[stride];
        System.arraycopy(values, AGENT_ENTITY * stride, row, 0, stride);
        return row;
    }

    private static boolean[] agentRow(boolean[] values, int stride) {
        boolean[] row = new boolean[stride];
        System.arraycopy(values, AGENT_ENTITY * stride, row, 0, stride);
        return row;
    }

    /** Zero each value whose mask bit is false; {@code features} per mask bit. */
    private static float[] scale(float[] values, boolean[] mask, int features) {
        float[] scaled = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            scaled[i] = mask[i / features] ? values[i] : 0.0f;
        }
        return scaled;
    }

    private static float[] take(float[] values, int[] indices) {
        float[] taken = new float[indices.length];
        for (int i = 0; i < indices.length; i++) {
            taken[i] = values[indices[i]];
        }
        return taken;
    }

    private static boolean[] and(boolean[] mask, boolean gate) {
        boolean[] result = new boolean[mask.length];
        for (int i = 0; i < mask.length; i++) {
            result[i] = mask[i] && gate;
        }
        return result;
    }

    private void put(float[] values) {
        System.arraycopy(values, 0, out, cursor, values.length);
        cursor += values.length;
    }

    private void put(boolean[] values) {
        for (boolean value : values) {
            out[cursor++] = value ? 1.0f : 0.0f;
        }
    }

    // -- loading -------------------------------------------------------------

    private float[] f32(String name, int expected) {
        float[] values = overrides.get(name);
        if (values == null) {
            values = template.floats.get(name);
        }
        if (values == null) {
            throw new IllegalStateException("template has no float field " + name);
        }
        require(values.length == expected, name, expected, values.length);
        return values;
    }

    private boolean[] bool(String name, int expected) {
        boolean[] values = maskOverrides.get(name);
        if (values == null) {
            values = template.masks.get(name);
        }
        if (values == null) {
            throw new IllegalStateException("template has no boolean field " + name);
        }
        require(values.length == expected, name, expected, values.length);
        return values;
    }

    private static void require(boolean ok, String name, int expected, int actual) {
        if (!ok) {
            throw new IllegalStateException(
                    "field " + name + " has " + actual + " values, contract says "
                            + expected + "; the observation contract moved");
        }
    }

    /** Immutable, contract-checked source for groups that are not overridden. */
    public static final class Template {
        private final Map<String, float[]> floats;
        private final Map<String, boolean[]> masks;

        private Template(Path root) throws IOException {
            Map<String, float[]> loadedFloats = new LinkedHashMap<>();
            Map<String, boolean[]> loadedMasks = new LinkedHashMap<>();

            loadFloat(loadedFloats, root, "base_self_f32", SELF);
            loadFloat(loadedFloats, root, "base_target_f32", TARGET);
            loadMask(loadedMasks, root, "base_target_mask", 1);
            loadFloat(loadedFloats, root, "base_combat_f32", COMBAT);
            loadFloat(loadedFloats, root, "resource_f32", ENTITIES * RESOURCES);
            loadMask(loadedMasks, root, "resource_mask", ENTITIES * RESOURCES);
            loadFloat(loadedFloats, root, "defense_f32", ENTITIES * DEFENSE);
            loadFloat(loadedFloats, root, "status_f32",
                ENTITIES * STATUSES * STATUS_FEATURES);
            loadMask(loadedMasks, root, "status_mask", ENTITIES * STATUSES);
            loadFloat(loadedFloats, root, "ability_f32",
                ENTITIES * ABILITIES * ABILITY_FEATURES);
            loadMask(loadedMasks, root, "ability_mask", ENTITIES * ABILITIES);
            loadMask(loadedMasks, root, "ability_legal", ENTITIES * ABILITIES);
            loadFloat(loadedFloats, root, "actor_world_f32", ACTOR_WORLD);
            loadMask(loadedMasks, root, "actor_world_mask", ACTOR_WORLD_MASKS);
            loadFloat(loadedFloats, root, "movement_state_f32", MOVEMENT);
            loadMask(loadedMasks, root, "movement_state_mask", MOVEMENT);
            loadFloat(loadedFloats, root, "geometry_token_f32",
                GEOMETRY_TOKENS * GEOMETRY_FEATURES);
            loadMask(loadedMasks, root, "geometry_token_mask", GEOMETRY_TOKENS);
            loadMask(loadedMasks, root, "geometry_available", 1);
            loadFloat(loadedFloats, root, "light_token_f32",
                GEOMETRY_TOKENS * LIGHT_FEATURES);
            loadMask(loadedMasks, root, "light_available", 1);
            loadFloat(loadedFloats, root, "inventory_container_f32", CONTAINERS);
            loadMask(loadedMasks, root, "inventory_container_mask", CONTAINERS);
            loadFloat(loadedFloats, root, "inventory_token_f32",
                INVENTORY_TOKENS * INVENTORY_FEATURES);
            loadMask(loadedMasks, root, "inventory_token_mask", INVENTORY_TOKENS);
            loadMask(loadedMasks, root, "inventory_available", 1);
            loadMask(loadedMasks, root, "skill_action_mask", SKILLS);
            loadMask(loadedMasks, root, "jump_action_mask", 1);
            loadMask(loadedMasks, root, "guard_action_mask", 1);
            loadMask(loadedMasks, root, "dodge_action_mask", DODGES);
            loadMask(loadedMasks, root, "valid", 1);
            loadFloat(loadedFloats, root, "block_candidate_f32",
                BLOCK_CANDIDATES * BLOCK_FEATURES);
            loadMask(loadedMasks, root, "block_candidate_mask", BLOCK_CANDIDATES);
            loadMask(loadedMasks, root, "block_available", 1);
            loadFloat(loadedFloats, root, "recipe_embedding",
                RECIPES * RECIPE_FEATURES);
            loadMask(loadedMasks, root, "recipe_mask", RECIPES);
            loadMask(loadedMasks, root, "recipe_available", 1);

            floats = Map.copyOf(loadedFloats);
            masks = Map.copyOf(loadedMasks);
        }

        private static void loadFloat(
            Map<String, float[]> destination,
            Path root,
            String name,
            int expected
        ) throws IOException {
            float[] values = Policy.read(root, name);
            require(values.length == expected, name, expected, values.length);
            destination.put(name, values);
        }

        private static void loadMask(
            Map<String, boolean[]> destination,
            Path root,
            String name,
            int expected
        ) throws IOException {
            boolean[] values = Policy.readMask(root, name);
            require(values.length == expected, name, expected, values.length);
            destination.put(name, values);
        }
    }
}
