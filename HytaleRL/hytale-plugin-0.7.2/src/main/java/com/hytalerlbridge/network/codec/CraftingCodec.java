package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.worldgen.NativeCraftingCatalogEvidence;
import java.io.IOException;
import java.util.List;
import org.msgpack.core.MessagePacker;

/**
 * Extracted verbatim from {@code ClientHandler}.
 */
public final class CraftingCodec {

    private CraftingCodec() {}

    public static void packCraftingCatalogEvidence(
        MessagePacker packer,
        NativeCraftingCatalogEvidence evidence,
        String bridgeSha256
    ) throws IOException {
        if (
            bridgeSha256 == null
                || !bridgeSha256.matches("[0-9A-Fa-f]{64}")
        ) {
            throw new IllegalArgumentException(
                "Runtime bridge identity must be a SHA-256"
            );
        }
        packer.packMapHeader(22);
        packer.packString("type");
        packer.packString("crafting_catalog_evidence");
        packer.packString("schema");
        packer.packString(NativeCraftingCatalogEvidence.SCHEMA);
        packer.packString("version");
        packer.packInt(NativeCraftingCatalogEvidence.VERSION);
        packer.packString("bridge_sha256");
        packer.packString(bridgeSha256);
        packer.packString("server_version");
        packer.packString(evidence.serverVersion());
        packer.packString("world");
        packer.packString(evidence.worldName());
        packer.packString("worldgen_provider");
        packer.packString(evidence.worldgenProvider());
        packer.packString("worldgen_version");
        packer.packString(evidence.worldgenVersion());
        packer.packString("seed");
        packer.packLong(evidence.seed());
        packer.packString("expected_recipe_count");
        packer.packInt(NativeCraftingCatalogEvidence.RECIPE_CAPACITY);
        packer.packString("recipe_count");
        packer.packInt(evidence.recipes().size());
        packer.packString("recipes_without_bench_requirement");
        packer.packInt(evidence.recipesWithoutBenchRequirement());
        packer.packString("tagged_material_count");
        packer.packInt(evidence.taggedMaterialCount());
        packer.packString("excluded_material_count");
        packer.packInt(evidence.excludedMaterialCount());
        packer.packString("native_recipe_identity");
        packer.packString("CraftRecipeAction.recipeId_string");
        packer.packString("headless_npc_execution_supported");
        packer.packBoolean(false);
        packer.packString("headless_npc_execution_boundary");
        packer.packString("CraftingManager.craftItem_requires_Player");
        packer.packString("headless_server_actor_execution_supported");
        packer.packBoolean(true);
        packer.packString("headless_server_actor_execution_scope");
        packer.packString(
            "opt_in_Adventure_context_fieldcraft_and_Crafting_bench"
        );
        packer.packString("headless_server_actor_execution_requires");
        packer.packString(
            "native_world_verbs_true_exact_inventory_and_recipe_context"
        );
        packer.packString("public_player_execution_certified");
        packer.packBoolean(false);
        packer.packString("recipes");
        packer.packArrayHeader(evidence.recipes().size());
        for (NativeCraftingCatalogEvidence.Recipe recipe
            : evidence.recipes()) {
            packCraftingRecipe(packer, recipe);
        }
    }


    public static void packCraftingRecipe(
        MessagePacker packer,
        NativeCraftingCatalogEvidence.Recipe recipe
    ) throws IOException {
        packer.packMapHeader(8);
        packer.packString("recipe_id");
        packer.packString(recipe.recipeId());
        packer.packString("primary_output_item_asset_id");
        packer.packString(recipe.primaryOutputItemAssetId());
        packer.packString("inputs");
        packCraftingMaterials(packer, recipe.inputs());
        packer.packString("outputs");
        packCraftingMaterials(packer, recipe.outputs());
        packer.packString("bench_requirements");
        packer.packArrayHeader(recipe.benchRequirements().size());
        for (NativeCraftingCatalogEvidence.Bench bench
            : recipe.benchRequirements()) {
            packer.packMapHeader(3);
            packer.packString("type");
            packer.packInt(bench.type());
            packer.packString("id");
            packer.packString(bench.id());
            packer.packString("required_tier_level");
            packer.packInt(bench.requiredTierLevel());
        }
        packer.packString("knowledge_required");
        packer.packBoolean(recipe.knowledgeRequired());
        packer.packString("required_memories_level");
        packer.packInt(recipe.requiredMemoriesLevel());
        packer.packString("time_seconds");
        packer.packFloat(recipe.timeSeconds());
    }


    public static void packCraftingMaterials(
        MessagePacker packer,
        List<NativeCraftingCatalogEvidence.Material> materials
    ) throws IOException {
        packer.packArrayHeader(materials.size());
        for (NativeCraftingCatalogEvidence.Material material : materials) {
            packer.packMapHeader(6);
            packer.packString("item_asset_id");
            packer.packString(material.itemAssetId());
            packer.packString("resource_type_id");
            packer.packString(material.resourceTypeId());
            packer.packString("quantity");
            packer.packInt(material.quantity());
            packer.packString("metadata_json_sha256");
            packer.packString(material.metadataJsonSha256());
            packer.packString("item_tag_present");
            packer.packBoolean(material.itemTagPresent());
            packer.packString("excluded_item_asset_ids");
            packer.packArrayHeader(material.excludedItemAssetIds().size());
            for (String item : material.excludedItemAssetIds()) {
                packer.packString(item);
            }
        }
    }
}
