package com.hytalerlbridge.worldgen;

import com.hypixel.hytale.protocol.BenchRequirement;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.inventory.MaterialQuantity;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/** Complete runtime-resolved Hytale 0.5.7 crafting recipe catalog. */
public record NativeCraftingCatalogEvidence(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    List<Recipe> recipes
) {

    public static final String SCHEMA =
        "hytalerl_native_crafting_catalog_evidence_v2";
    public static final int VERSION = 2;
    public static final int RECIPE_CAPACITY = 1_947;
    public static final int INPUT_CAPACITY = 28;
    public static final int OUTPUT_CAPACITY = 4;
    public static final int BENCH_REQUIREMENT_CAPACITY = 3;

    public NativeCraftingCatalogEvidence {
        if (
            serverVersion == null
                || serverVersion.isBlank()
                || worldName == null
                || worldName.isBlank()
                || worldgenProvider == null
                || worldgenProvider.isBlank()
                || worldgenVersion == null
                || worldgenVersion.isBlank()
        ) {
            throw new IllegalArgumentException(
                "Native crafting provenance cannot be blank"
            );
        }
        recipes = List.copyOf(recipes);
        if (recipes.size() != RECIPE_CAPACITY) {
            throw new IllegalArgumentException(
                "Native crafting catalog must contain exactly "
                    + RECIPE_CAPACITY
                    + " recipes"
            );
        }
        String previous = null;
        for (Recipe recipe : recipes) {
            if (previous != null && previous.compareTo(recipe.recipeId()) >= 0) {
                throw new IllegalArgumentException(
                    "Native crafting recipes must have unique sorted IDs"
                );
            }
            previous = recipe.recipeId();
        }
    }

    public static NativeCraftingCatalogEvidence capture(
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed
    ) {
        Map<String, CraftingRecipe> assets = new TreeMap<>(
            CraftingRecipe.getAssetMap().getAssetMap()
        );
        List<Recipe> rows = new ArrayList<>(assets.size());
        for (Map.Entry<String, CraftingRecipe> entry : assets.entrySet()) {
            CraftingRecipe recipe = entry.getValue();
            if (recipe == null || !entry.getKey().equals(recipe.getId())) {
                throw new IllegalStateException(
                    "Native recipe map key does not resolve to the same String ID"
                );
            }
            rows.add(recipe(entry.getKey(), recipe));
        }
        return new NativeCraftingCatalogEvidence(
            serverVersion,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            rows
        );
    }

    public int recipesWithoutBenchRequirement() {
        return (int) recipes.stream()
            .filter(recipe -> recipe.benchRequirements().isEmpty())
            .count();
    }

    public int taggedMaterialCount() {
        return recipes.stream()
            .mapToInt(Recipe::taggedMaterialCount)
            .sum();
    }

    public int excludedMaterialCount() {
        return recipes.stream()
            .mapToInt(Recipe::excludedMaterialCount)
            .sum();
    }

    private static Recipe recipe(String id, CraftingRecipe recipe) {
        MaterialQuantity primary = recipe.getPrimaryOutput();
        return new Recipe(
            id,
            primary == null || primary.getItemId() == null
                ? ""
                : primary.getItemId(),
            materials(recipe.getInput()),
            materials(recipe.getOutputs()),
            requirements(recipe.getBenchRequirement()),
            recipe.isKnowledgeRequired(),
            recipe.getRequiredMemoriesLevel(),
            recipe.getTimeSeconds()
        );
    }

    private static List<Material> materials(MaterialQuantity[] values) {
        if (values == null) return List.of();
        List<Material> rows = new ArrayList<>(values.length);
        for (MaterialQuantity value : values) {
            if (value == null) {
                throw new IllegalStateException(
                    "Native crafting catalog contains a null material"
                );
            }
            var metadata = value.getMetadata();
            var excluded = value.getExcludedItemIds();
            rows.add(new Material(
                value.getItemId() == null ? "" : value.getItemId(),
                value.getResourceTypeId() == null
                    ? ""
                    : value.getResourceTypeId(),
                value.getQuantity(),
                metadata == null || metadata.isEmpty()
                    ? ""
                    : sha256Hex(metadata.toJson()),
                value.getTagIndex() != Integer.MIN_VALUE,
                excluded == null
                    ? List.of()
                    : excluded.stream().sorted().toList()
            ));
        }
        return rows;
    }

    private static List<Bench> requirements(BenchRequirement[] values) {
        if (values == null) return List.of();
        List<Bench> rows = new ArrayList<>(values.length);
        for (BenchRequirement value : values) {
            if (value == null || value.type == null) {
                throw new IllegalStateException(
                    "Native crafting catalog contains a null bench requirement"
                );
            }
            rows.add(new Bench(
                value.type.getValue(),
                value.id,
                value.requiredTierLevel
            ));
        }
        return rows;
    }

    private static String sha256Hex(String value) {
        try {
            return HexFormat.of().formatHex(
                MessageDigest.getInstance("SHA-256").digest(
                    value.getBytes(StandardCharsets.UTF_8)
                )
            );
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    public record Recipe(
        String recipeId,
        String primaryOutputItemAssetId,
        List<Material> inputs,
        List<Material> outputs,
        List<Bench> benchRequirements,
        boolean knowledgeRequired,
        int requiredMemoriesLevel,
        float timeSeconds
    ) {

        public Recipe {
            if (recipeId == null || recipeId.isBlank()) {
                throw new IllegalArgumentException("Recipe ID cannot be blank");
            }
            primaryOutputItemAssetId = nonNull(primaryOutputItemAssetId);
            inputs = List.copyOf(inputs);
            outputs = List.copyOf(outputs);
            benchRequirements = List.copyOf(benchRequirements);
            if (
                inputs.isEmpty()
                    || inputs.size() > INPUT_CAPACITY
                    || outputs.isEmpty()
                    || outputs.size() > OUTPUT_CAPACITY
                    || benchRequirements.size() > BENCH_REQUIREMENT_CAPACITY
                    || requiredMemoriesLevel < 1
                    || !Float.isFinite(timeSeconds)
                    || timeSeconds < 0.0f
            ) {
                throw new IllegalArgumentException(
                    "Native crafting recipe is outside the v1 contract"
                );
            }
        }

        int taggedMaterialCount() {
            return (int) java.util.stream.Stream.concat(
                inputs.stream(),
                outputs.stream()
            ).filter(Material::itemTagPresent).count();
        }

        int excludedMaterialCount() {
            return (int) inputs.stream()
                .filter(value -> !value.excludedItemAssetIds().isEmpty())
                .count();
        }
    }

    public record Material(
        String itemAssetId,
        String resourceTypeId,
        int quantity,
        String metadataJsonSha256,
        boolean itemTagPresent,
        List<String> excludedItemAssetIds
    ) {

        public Material {
            itemAssetId = nonNull(itemAssetId);
            resourceTypeId = nonNull(resourceTypeId);
            metadataJsonSha256 = nonNull(metadataJsonSha256);
            excludedItemAssetIds = List.copyOf(excludedItemAssetIds);
            if (
                (itemAssetId.isEmpty() && resourceTypeId.isEmpty()
                    && !itemTagPresent)
                    || quantity < 1
                    || (!metadataJsonSha256.isEmpty()
                        && !metadataJsonSha256.matches("[0-9a-f]{64}"))
                    || !isStrictlySorted(excludedItemAssetIds)
            ) {
                throw new IllegalArgumentException(
                    "Native crafting material is outside the v1 contract"
                );
            }
        }
    }

    public record Bench(int type, String id, int requiredTierLevel) {

        public Bench {
            if (
                type < 0
                    || type > 3
                    || id == null
                    || id.isBlank()
                    || requiredTierLevel < 0
            ) {
                throw new IllegalArgumentException(
                    "Native crafting bench is outside the v1 contract"
                );
            }
        }
    }

    private static String nonNull(String value) {
        return value == null ? "" : value;
    }

    private static boolean isStrictlySorted(List<String> values) {
        String previous = null;
        for (String value : values) {
            if (
                value == null
                    || value.isBlank()
                    || (previous != null && previous.compareTo(value) >= 0)
            ) {
                return false;
            }
            previous = value;
        }
        return true;
    }
}
