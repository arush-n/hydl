package com.hytalerlbridge.worldgen.policyactions.capture.semantic;

import com.hytalerlbridge.observation.NativeInventoryFrame;
import com.hytalerlbridge.worldgen.NativeCraftingCatalogEvidence;
import com.hytalerlbridge.worldgen.NativeMutableBlockEvidence;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionBinding;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionBlockSurface;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionCamera;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionRecipeSurface;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionUseSurface;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Content identities shared by the capture producer and commit verifier. */
public final class PolicyWorldActionSemanticHash {

    private PolicyWorldActionSemanticHash() {}

    public static String inventoryIdentity(NativeInventoryFrame inventory) {
        return CanonicalJson.sha256(inventoryManifest(inventory));
    }

    public static String recipeTableIdentity(
        NativeCraftingCatalogEvidence catalog
    ) {
        List<Object> recipes = new ArrayList<>(catalog.recipes().size());
        for (NativeCraftingCatalogEvidence.Recipe recipe : catalog.recipes()) {
            recipes.add(recipeManifest(recipe));
        }
        return CanonicalJson.sha256(Map.of(
            "schema",
            NativeCraftingCatalogEvidence.SCHEMA,
            "version",
            NativeCraftingCatalogEvidence.VERSION,
            "recipes",
            recipes
        ));
    }

    public static String generation(
        String bridgeSha256,
        String worldName,
        String worldEpoch,
        int actorSlot,
        String actorIdentity,
        String inventoryIdentity,
        NativeInventoryFrame inventory,
        String recipeTableIdentity,
        PolicyWorldActionCamera camera,
        PolicyWorldActionBlockSurface blocks,
        PolicyWorldActionRecipeSurface recipes,
        PolicyWorldActionUseSurface use
    ) {
        if (actorSlot < 0) {
            throw new IllegalArgumentException("actorSlot must be nonnegative");
        }
        Digest digest = new Digest("hytalerl_policy_world_action_generation_v2");
        digest.text("bridge_sha256", bridgeSha256);
        digest.text("world", worldName);
        digest.text("world_epoch", worldEpoch);
        digest.integer("actor_slot", actorSlot);
        digest.text("actor_identity", actorIdentity);
        digest.text("inventory_identity_sha256", inventoryIdentity);
        digest.text("inventory_payload", CanonicalJson.encode(
            inventoryManifest(inventory)
        ));
        digest.text("recipe_table_identity_sha256", recipeTableIdentity);
        putCamera(digest, camera);
        putBlocks(digest, blocks);
        putRecipes(digest, recipes);
        putUse(digest, use);
        return digest.finish();
    }

    public static String bindingIdentity(PolicyWorldActionBinding binding) {
        Digest digest = new Digest("hytalerl_policy_world_action_binding_v1");
        putBinding(digest, "binding", binding);
        return digest.finish();
    }

    public static String recipeCandidateIdentity(
        String recipeTableIdentity,
        PolicyWorldActionRecipeSurface.CraftingContext context,
        PolicyWorldActionRecipeSurface.Candidate candidate
    ) {
        Digest digest = new Digest(
            "hytalerl_policy_world_action_recipe_candidate_v1"
        );
        digest.text("recipe_table_identity_sha256", recipeTableIdentity);
        putContext(digest, context);
        putRecipeCandidate(digest, candidate);
        return digest.finish();
    }

    public static Map<String, Object> inventoryManifest(
        NativeInventoryFrame inventory
    ) {
        List<Object> containers = new ArrayList<>(inventory.containers().size());
        for (NativeInventoryFrame.Container container : inventory.containers()) {
            List<Object> occupied = new ArrayList<>(
                container.occupiedSlots().size()
            );
            for (NativeInventoryFrame.Slot slot : container.occupiedSlots()) {
                occupied.add(Map.of(
                    "slot", slot.slot(),
                    "item_id", slot.itemId(),
                    "item_runtime_index", slot.itemRuntimeIndex(),
                    "quantity", slot.quantity(),
                    "durability", slot.durability(),
                    "max_durability", slot.maxDurability(),
                    "metadata_present", slot.metadataPresent()
                ));
            }
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("name", container.name());
            row.put("section_id", container.sectionId());
            row.put("available", container.available());
            row.put("unavailable_reason", container.unavailableReason());
            row.put("capacity", container.capacity());
            row.put("occupied_slots", occupied);
            containers.add(row);
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("schema", NativeInventoryFrame.SCHEMA);
        result.put("version", NativeInventoryFrame.VERSION);
        result.put("contract_sha256", NativeInventoryFrame.CONTRACT_SHA256);
        result.put("available", inventory.available());
        result.put("unavailable_reason", inventory.unavailableReason());
        result.put("active_slots", Map.of(
            "hotbar", inventory.activeHotbarSlot(),
            "utility", inventory.activeUtilitySlot(),
            "tools", inventory.activeToolsSlot()
        ));
        result.put("containers", containers);
        return result;
    }

    public static Map<String, Object> recipeManifest(
        NativeCraftingCatalogEvidence.Recipe recipe
    ) {
        List<Object> inputs = recipe.inputs().stream()
            .map(PolicyWorldActionSemanticHash::materialManifest)
            .map(value -> (Object) value)
            .toList();
        List<Object> outputs = recipe.outputs().stream()
            .map(PolicyWorldActionSemanticHash::materialManifest)
            .map(value -> (Object) value)
            .toList();
        List<Object> benches = recipe.benchRequirements().stream()
            .map(bench -> (Object) Map.of(
                "type", bench.type(),
                "id", bench.id(),
                "required_tier_level", bench.requiredTierLevel()
            ))
            .toList();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("recipe_id", recipe.recipeId());
        result.put(
            "primary_output_item_asset_id",
            recipe.primaryOutputItemAssetId()
        );
        result.put("inputs", inputs);
        result.put("outputs", outputs);
        result.put("bench_requirements", benches);
        result.put("knowledge_required", recipe.knowledgeRequired());
        result.put(
            "required_memories_level",
            recipe.requiredMemoriesLevel()
        );
        result.put("time_seconds", (double) recipe.timeSeconds());
        return result;
    }

    private static Map<String, Object> materialManifest(
        NativeCraftingCatalogEvidence.Material material
    ) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("item_asset_id", material.itemAssetId());
        result.put("resource_type_id", material.resourceTypeId());
        result.put("quantity", material.quantity());
        result.put("metadata_json_sha256", material.metadataJsonSha256());
        result.put("item_tag_present", material.itemTagPresent());
        result.put(
            "excluded_item_asset_ids",
            material.excludedItemAssetIds()
        );
        return result;
    }

    private static void putCamera(
        Digest digest,
        PolicyWorldActionCamera camera
    ) {
        digest.bool("camera.available", camera.available());
        digest.text("camera.unavailable_reason", camera.unavailableReason());
        digest.doubles("camera.actor_position", camera.actorPosition());
        digest.doubles("camera.eye_position", camera.eyePosition());
        digest.doubles("camera.direction", camera.direction());
        digest.number("camera.maximum_distance", camera.maximumDistance());
        digest.ints("camera.raw_hit_cell", camera.rawHitCell());
        digest.ints(
            "camera.canonical_action_cell",
            camera.canonicalActionCell()
        );
        digest.integer("camera.block_face", camera.blockFace());
    }

    private static void putBlocks(
        Digest digest,
        PolicyWorldActionBlockSurface surface
    ) {
        digest.bool("blocks.available", surface.available());
        digest.text("blocks.unavailable_reason", surface.unavailableReason());
        digest.integer("blocks.source_count", surface.sourceCount());
        digest.integer("blocks.emitted_count", surface.emittedCount());
        digest.bool("blocks.capacity_exceeded", surface.capacityExceeded());
        digest.text(
            "blocks.primary_unavailable_reason",
            surface.primaryUnavailableReason()
        );
        digest.text(
            "blocks.secondary_unavailable_reason",
            surface.secondaryUnavailableReason()
        );
        digest.integer("blocks.candidates.count", surface.candidates().size());
        for (PolicyWorldActionBlockSurface.Candidate candidate
            : surface.candidates()) {
            String prefix = "blocks.candidate." + candidate.slot();
            digest.integer(prefix + ".slot", candidate.slot());
            digest.ints(prefix + ".visible_cell", candidate.visibleCell());
            digest.ints(prefix + ".action_cell", candidate.actionCell());
            digest.doubles(
                prefix + ".visible_relative_position",
                candidate.visibleRelativePosition()
            );
            putMutableRow(digest, prefix + ".semantics", candidate.semantics());
            putBinding(digest, prefix + ".primary", candidate.primary());
            putBinding(digest, prefix + ".secondary", candidate.secondary());
        }
    }

    private static void putRecipes(
        Digest digest,
        PolicyWorldActionRecipeSurface surface
    ) {
        digest.bool("recipes.available", surface.available());
        digest.text("recipes.unavailable_reason", surface.unavailableReason());
        putContext(digest, surface.craftingContext());
        digest.integer("recipes.memories_level", surface.memoriesLevel());
        digest.bool(
            "recipes.knowledge_available",
            surface.knowledgeAvailable()
        );
        digest.bool(
            "recipes.bench_context_available",
            surface.benchContextAvailable()
        );
        digest.bool("recipes.manager_available", surface.managerAvailable());
        digest.bool("recipes.manager_has_bench", surface.managerHasBench());
        digest.integer("recipes.manager_queue_size", surface.managerQueueSize());
        digest.text(
            "recipes.manager_queue_recipe_id",
            surface.managerQueueRecipeId()
        );
        digest.text(
            "recipes.manager_queue_identity_sha256",
            surface.managerQueueIdentitySha256()
        );
        digest.integer("recipes.source_count", surface.sourceCount());
        digest.integer("recipes.emitted_count", surface.emittedCount());
        digest.bool("recipes.capacity_exceeded", surface.capacityExceeded());
        digest.integer("recipes.candidates.count", surface.candidates().size());
        for (PolicyWorldActionRecipeSurface.Candidate candidate
            : surface.candidates()) {
            putRecipeCandidate(digest, candidate);
        }
    }

    private static void putContext(
        Digest digest,
        PolicyWorldActionRecipeSurface.CraftingContext context
    ) {
        digest.bool("recipes.context.present", context != null);
        if (context == null) return;
        digest.text("recipes.context.kind", context.kind());
        digest.ints("recipes.context.position", context.position());
        digest.text("recipes.context.block_id", context.blockId());
        digest.text("recipes.context.bench_id", context.benchId());
        digest.integer("recipes.context.bench_type", context.benchType());
        digest.integer("recipes.context.tier", context.tier());
    }

    private static void putRecipeCandidate(
        Digest digest,
        PolicyWorldActionRecipeSurface.Candidate candidate
    ) {
        String prefix = "recipes.candidate." + candidate.slot();
        digest.integer(prefix + ".slot", candidate.slot());
        digest.integer(
            prefix + ".native_recipe_index",
            candidate.nativeRecipeIndex()
        );
        digest.text(prefix + ".recipe_id", candidate.recipeId());
        digest.bytes(prefix + ".recipe_id_sha256", candidate.recipeIdSha256());
        digest.text(
            prefix + ".recipe_payload",
            CanonicalJson.encode(recipeManifest(candidate.recipe()))
        );
        digest.bool(
            prefix + ".knowledge_satisfied",
            candidate.knowledgeSatisfied()
        );
        digest.bool(
            prefix + ".memory_satisfied",
            candidate.memorySatisfied()
        );
        digest.bool(prefix + ".bench_satisfied", candidate.benchSatisfied());
        digest.text(prefix + ".knowledge_key", candidate.knowledgeKey());
    }

    private static void putUse(
        Digest digest,
        PolicyWorldActionUseSurface surface
    ) {
        digest.bool("use.available", surface.available());
        digest.text("use.unavailable_reason", surface.unavailableReason());
        putBinding(digest, "use.binding", surface.binding());
    }

    private static void putBinding(
        Digest digest,
        String prefix,
        PolicyWorldActionBinding binding
    ) {
        digest.bool(prefix + ".present", binding != null);
        if (binding == null) return;
        digest.text(prefix + ".verb", binding.verb());
        digest.integer(prefix + ".interaction_type", binding.interactionType());
        digest.text(prefix + ".interaction_id", binding.interactionId());
        digest.text(
            prefix + ".block_interaction_id",
            binding.blockInteractionId()
        );
        digest.ints(prefix + ".target", binding.target());
        digest.integer(prefix + ".block_face", binding.blockFace());
        digest.bool(
            prefix + ".rotation_applicable",
            binding.rotationApplicable()
        );
        digest.ints(prefix + ".rotation", binding.rotation());
        digest.text(prefix + ".source_container", binding.sourceContainer());
        digest.text(
            prefix + ".resolved_source_container",
            binding.resolvedSourceContainer()
        );
        digest.integer(prefix + ".source_slot", binding.sourceSlot());
        digest.integer(prefix + ".source_quantity", binding.sourceQuantity());
        digest.text(prefix + ".source_item_id", binding.sourceItemId());
        digest.text(prefix + ".source_block_id", binding.sourceBlockId());
        digest.text(
            prefix + ".expected_interaction_tool_id",
            binding.expectedInteractionToolId()
        );
        digest.text(prefix + ".expected_block_id", binding.expectedBlockId());
        digest.bytes(
            prefix + ".expected_semantic_key_sha256",
            binding.expectedSemanticKeySha256()
        );
    }

    private static void putMutableRow(
        Digest digest,
        String prefix,
        NativeMutableBlockEvidence.Row row
    ) {
        digest.text(prefix + ".phase", row.phase());
        digest.bool(prefix + ".block_present", row.blockPresent());
        digest.text(prefix + ".block_asset_id", row.blockAssetId());
        digest.integer(prefix + ".runtime_block_id", row.runtimeBlockId());
        digest.bytes(prefix + ".semantic_key", row.semanticKeySha256());
        digest.bool(prefix + ".semantic_key_valid", row.semanticKeyValid());
        digest.bool(prefix + ".affordance_valid", row.affordanceValid());
        digest.integer(prefix + ".affordance_tags", row.affordanceTags());
        digest.integer(prefix + ".gather_type_index", row.gatherTypeIndex());
        digest.integer(
            prefix + ".required_tool_quality",
            row.requiredToolQuality()
        );
        digest.integer(prefix + ".rotation_index", row.rotationIndex());
        digest.integer(prefix + ".flags", row.flags());
        digest.integer(prefix + ".fluid_level", row.fluidLevel());
        digest.number(prefix + ".fluid_fill_height", row.fluidFillHeight());
        digest.integer(prefix + ".support", row.supportValue());
        digest.integer(prefix + ".block_damage", row.blockDamage());
        digest.integer(prefix + ".fluid_damage", row.fluidDamage());
        digest.doubles(prefix + ".movement", row.movement());
        digest.doubles(prefix + ".fluid_movement", row.fluidMovement());
        digest.doubles(prefix + ".collision_boxes", row.collisionBoxes());
        digest.number(prefix + ".block_health", row.blockHealth());
        digest.bool(prefix + ".block_health_valid", row.blockHealthValid());
        digest.number(
            prefix + ".seconds_since_damage",
            row.secondsSinceDamage()
        );
        digest.bool(prefix + ".damage_age_valid", row.damageAgeValid());
        digest.integer(
            prefix + ".local_change_counter",
            row.localChangeCounter()
        );
        digest.integer(
            prefix + ".global_change_counter",
            row.globalChangeCounter()
        );
    }

    private static final class Digest {
        private final MessageDigest digest;

        private Digest(String domain) {
            try {
                digest = MessageDigest.getInstance("SHA-256");
            } catch (NoSuchAlgorithmException exception) {
                throw new IllegalStateException(
                    "SHA-256 is unavailable",
                    exception
                );
            }
            text("domain", domain);
        }

        private void bool(String name, boolean value) {
            field(name, new byte[] {(byte) (value ? 1 : 0)});
        }

        private void integer(String name, long value) {
            field(
                name,
                ByteBuffer.allocate(Long.BYTES)
                    .order(ByteOrder.BIG_ENDIAN)
                    .putLong(value)
                    .array()
            );
        }

        private void number(String name, double value) {
            field(
                name,
                ByteBuffer.allocate(Double.BYTES)
                    .order(ByteOrder.BIG_ENDIAN)
                    .putLong(Double.doubleToLongBits(value))
                    .array()
            );
        }

        private void text(String name, String value) {
            field(name, value.getBytes(StandardCharsets.UTF_8));
        }

        private void bytes(String name, byte[] value) {
            field(name, value);
        }

        private void ints(String name, int[] values) {
            ByteBuffer buffer = ByteBuffer.allocate(values.length * Integer.BYTES)
                .order(ByteOrder.BIG_ENDIAN);
            for (int value : values) buffer.putInt(value);
            field(name, buffer.array());
        }

        private void doubles(String name, double[] values) {
            ByteBuffer buffer = ByteBuffer.allocate(values.length * Double.BYTES)
                .order(ByteOrder.BIG_ENDIAN);
            for (double value : values) {
                buffer.putLong(Double.doubleToLongBits(value));
            }
            field(name, buffer.array());
        }

        private void field(String name, byte[] value) {
            byte[] key = name.getBytes(StandardCharsets.UTF_8);
            digest.update(int32(key.length));
            digest.update(key);
            digest.update(int32(value.length));
            digest.update(value);
        }

        private String finish() {
            return java.util.HexFormat.of().withUpperCase().formatHex(
                digest.digest()
            );
        }

        private static byte[] int32(int value) {
            return ByteBuffer.allocate(Integer.BYTES)
                .order(ByteOrder.BIG_ENDIAN)
                .putInt(value)
                .array();
        }
    }
}
