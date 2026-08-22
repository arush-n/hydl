package com.hytalerlbridge.action;

import java.util.Map;
import java.util.Set;
import org.msgpack.value.Value;

/**
 * Versioned, reset-scoped transport for one World verb.
 *
 * <p>The request uses stable asset identities and absolute coordinates. It
 * deliberately carries no execution claim; native acceptance is reported by
 * the step receipt.</p>
 */
public record NativeWorldVerbRequest(
    long requestId,
    String verb,
    int interactionType,
    String targetKind,
    String interactionId,
    String recipeId,
    String itemId,
    String blockId,
    int targetX,
    int targetY,
    int targetZ,
    int blockFace,
    int rotationYaw,
    int rotationPitch,
    int rotationRoll,
    String sourceContainer,
    int sourceSlot,
    int expectedSourceQuantity,
    String destinationContainer,
    String expectedBlockId,
    String worldEpoch,
    int quantity,
    String placementVariant,
    String craftingContext,
    int benchX,
    int benchY,
    int benchZ,
    String expectedBenchBlockId,
    String expectedBenchId,
    int expectedBenchType,
    int expectedBenchTier,
    String expectedCandidateGenerationSha256,
    String expectedSelectedSemanticSha256
) {
    public static final String ACTION_KEY = "native_world_verb_request";
    public static final String SCHEMA =
        "hytalerl_native_world_verb_transport_v4";
    public static final int VERSION = 4;
    public static final String CONTRACT_SHA256 =
        "3869A467733B057DD2C629243F3A6F0D5E7A116624E1413B48AFDEF664A8900C";
    public static final String POLICY_EVIDENCE_REQUIRED =
        "native_policy_world_action_evidence_required";

    private static final Set<String> BLOCK_VERBS = Set.of(
        "use",
        "place_block",
        "break_block"
    );
    private static final Set<String> ENCODED_FIELDS = Set.of(
        "schema",
        "version",
        "request_id",
        "verb",
        "interaction_type",
        "target_kind",
        "interaction_id",
        "recipe_id",
        "item_id",
        "block_id",
        "target_x",
        "target_y",
        "target_z",
        "block_face",
        "rotation_yaw",
        "rotation_pitch",
        "rotation_roll",
        "source_container",
        "source_slot",
        "expected_source_quantity",
        "destination_container",
        "expected_block_id",
        "world_epoch",
        "quantity",
        "placement_variant",
        "crafting_context",
        "bench_x",
        "bench_y",
        "bench_z",
        "expected_bench_block_id",
        "expected_bench_id",
        "expected_bench_type",
        "expected_bench_tier",
        "expected_candidate_generation_sha256",
        "expected_selected_semantic_sha256"
    );
    private static final NativeWorldVerbRequest NONE =
        new NativeWorldVerbRequest(
            -1L,
            "",
            -1,
            "",
            "",
            "",
            "",
            "",
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            "",
            -1,
            -1,
            "",
            "",
            "",
            0,
            "",
            "",
            0,
            0,
            0,
            "",
            "",
            -1,
            0,
            "",
            ""
        );

    /** Internal fixture constructor predating crafting context v2. */
    public NativeWorldVerbRequest(
        long requestId,
        String verb,
        int interactionType,
        String targetKind,
        String interactionId,
        String recipeId,
        String itemId,
        String blockId,
        int targetX,
        int targetY,
        int targetZ,
        int blockFace,
        int rotationYaw,
        int rotationPitch,
        int rotationRoll,
        String sourceContainer,
        int sourceSlot,
        int expectedSourceQuantity,
        String destinationContainer,
        String expectedBlockId,
        String worldEpoch,
        int quantity,
        String placementVariant
    ) {
        this(
            requestId,
            verb,
            interactionType,
            targetKind,
            interactionId,
            recipeId,
            itemId,
            blockId,
            targetX,
            targetY,
            targetZ,
            blockFace,
            rotationYaw,
            rotationPitch,
            rotationRoll,
            sourceContainer,
            sourceSlot,
            expectedSourceQuantity,
            destinationContainer,
            expectedBlockId,
            worldEpoch,
            quantity,
            placementVariant,
            "",
            0,
            0,
            0,
            "",
            "",
            -1,
            0,
            "",
            ""
        );
    }

    /** Internal fixture constructor without a policy-capture binding. */
    public NativeWorldVerbRequest(
        long requestId,
        String verb,
        int interactionType,
        String targetKind,
        String interactionId,
        String recipeId,
        String itemId,
        String blockId,
        int targetX,
        int targetY,
        int targetZ,
        int blockFace,
        int rotationYaw,
        int rotationPitch,
        int rotationRoll,
        String sourceContainer,
        int sourceSlot,
        int expectedSourceQuantity,
        String destinationContainer,
        String expectedBlockId,
        String worldEpoch,
        int quantity,
        String placementVariant,
        String craftingContext,
        int benchX,
        int benchY,
        int benchZ,
        String expectedBenchBlockId,
        String expectedBenchId,
        int expectedBenchType,
        int expectedBenchTier
    ) {
        this(
            requestId,
            verb,
            interactionType,
            targetKind,
            interactionId,
            recipeId,
            itemId,
            blockId,
            targetX,
            targetY,
            targetZ,
            blockFace,
            rotationYaw,
            rotationPitch,
            rotationRoll,
            sourceContainer,
            sourceSlot,
            expectedSourceQuantity,
            destinationContainer,
            expectedBlockId,
            worldEpoch,
            quantity,
            placementVariant,
            craftingContext,
            benchX,
            benchY,
            benchZ,
            expectedBenchBlockId,
            expectedBenchId,
            expectedBenchType,
            expectedBenchTier,
            "",
            ""
        );
    }

    public NativeWorldVerbRequest(
        long requestId,
        String verb,
        int interactionType,
        String targetKind,
        String interactionId,
        String recipeId,
        String itemId,
        String blockId,
        int targetX,
        int targetY,
        int targetZ,
        int blockFace,
        int rotationYaw,
        int rotationPitch,
        int rotationRoll,
        String sourceContainer,
        int sourceSlot,
        int expectedSourceQuantity,
        String destinationContainer,
        String expectedBlockId,
        String worldEpoch,
        int quantity,
        String placementVariant,
        String craftingContext,
        int benchX,
        int benchY,
        int benchZ,
        String expectedBenchBlockId,
        String expectedBenchId,
        int expectedBenchType,
        int expectedBenchTier,
        String expectedCandidateGenerationSha256,
        String expectedSelectedSemanticSha256
    ) {
        verb = bounded(verb, "verb", 32);
        targetKind = bounded(targetKind, "target_kind", 32);
        interactionId = bounded(interactionId, "interaction_id", 256);
        recipeId = bounded(recipeId, "recipe_id", 256);
        itemId = bounded(itemId, "item_id", 256);
        blockId = bounded(blockId, "block_id", 256);
        sourceContainer = bounded(sourceContainer, "source_container", 64);
        destinationContainer = bounded(
            destinationContainer,
            "destination_container",
            64
        );
        expectedBlockId = bounded(
            expectedBlockId,
            "expected_block_id",
            256
        );
        worldEpoch = bounded(worldEpoch, "world_epoch", 128);
        placementVariant = bounded(
            placementVariant,
            "placement_variant",
            128
        );
        craftingContext = bounded(
            craftingContext,
            "crafting_context",
            32
        );
        expectedBenchBlockId = bounded(
            expectedBenchBlockId,
            "expected_bench_block_id",
            256
        );
        expectedBenchId = bounded(
            expectedBenchId,
            "expected_bench_id",
            128
        );
        expectedCandidateGenerationSha256 = optionalSha256(
            expectedCandidateGenerationSha256,
            "expected_candidate_generation_sha256"
        );
        expectedSelectedSemanticSha256 = optionalSha256(
            expectedSelectedSemanticSha256,
            "expected_selected_semantic_sha256"
        );

        this.requestId = requestId;
        this.verb = verb;
        this.interactionType = interactionType;
        this.targetKind = targetKind;
        this.interactionId = interactionId;
        this.recipeId = recipeId;
        this.itemId = itemId;
        this.blockId = blockId;
        this.targetX = targetX;
        this.targetY = targetY;
        this.targetZ = targetZ;
        this.blockFace = blockFace;
        this.rotationYaw = rotationYaw;
        this.rotationPitch = rotationPitch;
        this.rotationRoll = rotationRoll;
        this.sourceContainer = sourceContainer;
        this.sourceSlot = sourceSlot;
        this.expectedSourceQuantity = expectedSourceQuantity;
        this.destinationContainer = destinationContainer;
        this.expectedBlockId = expectedBlockId;
        this.worldEpoch = worldEpoch;
        this.quantity = quantity;
        this.placementVariant = placementVariant;
        this.craftingContext = craftingContext;
        this.benchX = benchX;
        this.benchY = benchY;
        this.benchZ = benchZ;
        this.expectedBenchBlockId = expectedBenchBlockId;
        this.expectedBenchId = expectedBenchId;
        this.expectedBenchType = expectedBenchType;
        this.expectedBenchTier = expectedBenchTier;
        this.expectedCandidateGenerationSha256 =
            expectedCandidateGenerationSha256;
        this.expectedSelectedSemanticSha256 =
            expectedSelectedSemanticSha256;

        if (verb.isEmpty()) {
            if (
                requestId != -1L
                    || interactionType != -1
                    || !targetKind.isEmpty()
                    || !interactionId.isEmpty()
                    || !recipeId.isEmpty()
                    || !itemId.isEmpty()
                    || !blockId.isEmpty()
                    || targetX != 0
                    || targetY != 0
                    || targetZ != 0
                    || blockFace != 0
                    || rotationYaw != 0
                    || rotationPitch != 0
                    || rotationRoll != 0
                    || !sourceContainer.isEmpty()
                    || sourceSlot != -1
                    || expectedSourceQuantity != -1
                    || !destinationContainer.isEmpty()
                    || !expectedBlockId.isEmpty()
                    || !worldEpoch.isEmpty()
                    || quantity != 0
                    || !placementVariant.isEmpty()
                    || !craftingContext.isEmpty()
                    || benchX != 0
                    || benchY != 0
                    || benchZ != 0
                    || !expectedBenchBlockId.isEmpty()
                    || !expectedBenchId.isEmpty()
                    || expectedBenchType != -1
                    || expectedBenchTier != 0
                    || !expectedCandidateGenerationSha256.isEmpty()
                    || !expectedSelectedSemanticSha256.isEmpty()
            ) {
                throw new IllegalArgumentException(
                    "Absent native World verb must use the empty sentinel"
                );
            }
        } else {
            if (requestId < 0L) {
                throw new IllegalArgumentException(
                    "request_id must be nonnegative"
                );
            }
            if (worldEpoch.isEmpty()) {
                throw new IllegalArgumentException("world_epoch is required");
            }
            if (
                expectedCandidateGenerationSha256.isEmpty()
                    != expectedSelectedSemanticSha256.isEmpty()
            ) {
                throw new IllegalArgumentException(POLICY_EVIDENCE_REQUIRED);
            }
            if (quantity <= 0) {
                throw new IllegalArgumentException("quantity must be positive");
            }
            if (blockFace < 0 || blockFace > 6) {
                throw new IllegalArgumentException(
                    "block_face must be in [0, 6]"
                );
            }
            if (
                rotationYaw < 0
                    || rotationYaw > 3
                    || rotationPitch < 0
                    || rotationPitch > 3
                    || rotationRoll < 0
                    || rotationRoll > 3
            ) {
                throw new IllegalArgumentException(
                    "block rotations must be in [0, 3]"
                );
            }
            if (BLOCK_VERBS.contains(verb)) {
                validateBlockVerb();
            } else if (verb.equals("craft_recipe")) {
                validateCraft();
            } else {
                throw new IllegalArgumentException(
                    "Unknown native World verb"
                );
            }
        }
    }

    public static NativeWorldVerbRequest none() {
        return NONE;
    }

    public boolean present() {
        return !verb.isEmpty();
    }

    /** Whether this request is bound to one atomic policy candidate capture. */
    public boolean candidateEvidenceBound() {
        return !expectedCandidateGenerationSha256.isEmpty();
    }

    public static NativeWorldVerbRequest fromActionMap(
        Map<Value, Value> action
    ) {
        Value encoded = value(action, ACTION_KEY);
        if (encoded == null) return none();
        if (!encoded.isMapValue()) {
            throw new IllegalArgumentException(
                ACTION_KEY + " must be a map"
            );
        }
        Map<Value, Value> map = encoded.asMapValue().map();
        requireExactFields(map);
        String schema = requiredString(map, "schema");
        int version = requiredInt(map, "version");
        if (!SCHEMA.equals(schema) || version != VERSION) {
            throw new IllegalArgumentException(
                "native World-verb transport schema changed"
            );
        }
        NativeWorldVerbRequest request = new NativeWorldVerbRequest(
            requiredLong(map, "request_id"),
            requiredString(map, "verb"),
            requiredInt(map, "interaction_type"),
            requiredString(map, "target_kind"),
            requiredString(map, "interaction_id"),
            requiredString(map, "recipe_id"),
            requiredString(map, "item_id"),
            requiredString(map, "block_id"),
            requiredInt(map, "target_x"),
            requiredInt(map, "target_y"),
            requiredInt(map, "target_z"),
            requiredInt(map, "block_face"),
            requiredInt(map, "rotation_yaw"),
            requiredInt(map, "rotation_pitch"),
            requiredInt(map, "rotation_roll"),
            requiredString(map, "source_container"),
            requiredInt(map, "source_slot"),
            requiredInt(map, "expected_source_quantity"),
            requiredString(map, "destination_container"),
            requiredString(map, "expected_block_id"),
            requiredString(map, "world_epoch"),
            requiredInt(map, "quantity"),
            requiredString(map, "placement_variant"),
            requiredString(map, "crafting_context"),
            requiredInt(map, "bench_x"),
            requiredInt(map, "bench_y"),
            requiredInt(map, "bench_z"),
            requiredString(map, "expected_bench_block_id"),
            requiredString(map, "expected_bench_id"),
            requiredInt(map, "expected_bench_type"),
            requiredInt(map, "expected_bench_tier"),
            requiredString(map, "expected_candidate_generation_sha256"),
            requiredString(map, "expected_selected_semantic_sha256")
        );
        if (!request.candidateEvidenceBound()) {
            throw new IllegalArgumentException(POLICY_EVIDENCE_REQUIRED);
        }
        return request;
    }

    private void validateBlockVerb() {
        boolean unarmedUse =
            verb.equals("use") && sourceContainer.equals("unarmed");
        boolean contextUse =
            verb.equals("use")
                && sourceContainer.equals("interaction_context");
        if (!targetKind.equals("block")) {
            throw new IllegalArgumentException(
                "Block verbs require target_kind=block"
            );
        }
        if (targetY < 0 || targetY >= 320) {
            throw new IllegalArgumentException(
                "Block target Y must be in [0, 320)"
            );
        }
        if (
            interactionId.isEmpty()
                || itemId.isEmpty()
                || expectedBlockId.isEmpty()
                || !recipeId.isEmpty()
                || !destinationContainer.isEmpty()
                || !craftingContext.isEmpty()
                || benchX != 0
                || benchY != 0
                || benchZ != 0
                || !expectedBenchBlockId.isEmpty()
                || !expectedBenchId.isEmpty()
                || expectedBenchType != -1
                || expectedBenchTier != 0
        ) {
            throw new IllegalArgumentException(
                "Block verb request is incomplete"
            );
        }
        if (unarmedUse) {
            if (
                !itemId.equals("Empty")
                    || sourceSlot != -1
                    || expectedSourceQuantity != 0
            ) {
                throw new IllegalArgumentException(
                    "Unarmed Use requires the explicit Empty sentinel"
                );
            }
        } else if (contextUse) {
            if (
                itemId.equals("Empty")
                    || sourceSlot < 0
                    || expectedSourceQuantity < quantity
            ) {
                throw new IllegalArgumentException(
                    "Context Use requires an exact resolved item"
                );
            }
        } else if (
            sourceContainer.isEmpty()
                || sourceSlot < 0
                || expectedSourceQuantity < quantity
        ) {
            throw new IllegalArgumentException(
                "Block verb request is missing its source item"
            );
        }
        if (
            (verb.equals("use") && interactionType != 5)
                || (
                    !verb.equals("use")
                        && interactionType != 0
                        && interactionType != 1
                )
        ) {
            throw new IllegalArgumentException(
                "World verb request has an invalid InteractionType"
            );
        }
        if (verb.equals("place_block")) {
            if (blockId.isEmpty() || placementVariant.isEmpty()) {
                throw new IllegalArgumentException(
                    "Place request requires block_id and placement_variant"
                );
            }
        } else if (!blockId.isEmpty() || !placementVariant.isEmpty()) {
            throw new IllegalArgumentException(
                "Only place_block accepts block_id or placement_variant"
            );
        }
    }

    private void validateCraft() {
        boolean commonInvalid =
            interactionType != -1
                || !targetKind.equals("recipe")
                || recipeId.isEmpty()
                || sourceContainer.isEmpty()
                || sourceSlot != -1
                || expectedSourceQuantity != -1
                || destinationContainer.isEmpty()
                || !interactionId.isEmpty()
                || !itemId.isEmpty()
                || !blockId.isEmpty()
                || !expectedBlockId.isEmpty()
                || !placementVariant.isEmpty()
                || targetX != 0
                || targetY != 0
                || targetZ != 0
                || blockFace != 0
                || rotationYaw != 0
                || rotationPitch != 0
                || rotationRoll != 0
                || (
                    !craftingContext.equals("fieldcraft")
                        && !craftingContext.equals("bench")
                );
        if (commonInvalid) {
            throw new IllegalArgumentException(
                "Craft request is incomplete or carries block-only fields"
            );
        }
        if (craftingContext.equals("fieldcraft")) {
            if (
                benchX != 0
                    || benchY != 0
                    || benchZ != 0
                    || !expectedBenchBlockId.isEmpty()
                    || !expectedBenchId.equals("Fieldcraft")
                    || expectedBenchType != 0
                    || expectedBenchTier != 0
            ) {
                throw new IllegalArgumentException(
                    "Fieldcraft request has an invalid crafting context"
                );
            }
            return;
        }
        if (
            benchY < 0
                || benchY >= 320
                || expectedBenchBlockId.isEmpty()
                || expectedBenchId.isEmpty()
                || expectedBenchType < 0
                || expectedBenchType > 3
                || expectedBenchTier < 1
        ) {
            throw new IllegalArgumentException(
                "Bench craft request has an invalid crafting context"
            );
        }
    }

    private static void requireExactFields(Map<Value, Value> map) {
        if (map.size() != ENCODED_FIELDS.size()) {
            throw new IllegalArgumentException(
                "native World-verb request fields changed"
            );
        }
        for (Value key : map.keySet()) {
            if (
                !key.isStringValue()
                    || !ENCODED_FIELDS.contains(
                        key.asStringValue().asString()
                    )
            ) {
                throw new IllegalArgumentException(
                    "native World-verb request contains an unknown field"
                );
            }
        }
    }

    private static String bounded(String value, String name, int maximum) {
        if (value == null) {
            throw new IllegalArgumentException(name + " must be a string");
        }
        if (
            value.length() > maximum
                || value.indexOf('\0') >= 0
                || !value.equals(value.trim())
        ) {
            throw new IllegalArgumentException(name + " is invalid");
        }
        return value;
    }

    private static String optionalSha256(String value, String name) {
        value = bounded(value, name, 64);
        if (!value.isEmpty() && !value.matches("[0-9A-Fa-f]{64}")) {
            throw new IllegalArgumentException(name + " must be one SHA-256");
        }
        return value.toUpperCase(java.util.Locale.ROOT);
    }

    private static String requiredString(
        Map<Value, Value> map,
        String key
    ) {
        Value found = required(map, key);
        if (!found.isStringValue()) {
            throw new IllegalArgumentException(key + " must be a string");
        }
        return found.asStringValue().asString();
    }

    private static int requiredInt(Map<Value, Value> map, String key) {
        long decoded = requiredLong(map, key);
        if (decoded < Integer.MIN_VALUE || decoded > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(key + " is out of int range");
        }
        return (int) decoded;
    }

    private static long requiredLong(Map<Value, Value> map, String key) {
        Value found = required(map, key);
        if (!found.isIntegerValue()) {
            throw new IllegalArgumentException(key + " must be an integer");
        }
        return found.asIntegerValue().asLong();
    }

    private static Value required(Map<Value, Value> map, String key) {
        Value found = value(map, key);
        if (found == null) {
            throw new IllegalArgumentException(
                "Missing native World-verb field: " + key
            );
        }
        return found;
    }

    private static Value value(Map<Value, Value> map, String key) {
        for (Map.Entry<Value, Value> entry : map.entrySet()) {
            Value candidate = entry.getKey();
            if (
                candidate.isStringValue()
                    && candidate.asStringValue().asString().equals(key)
            ) {
                return entry.getValue();
            }
        }
        return null;
    }
}
