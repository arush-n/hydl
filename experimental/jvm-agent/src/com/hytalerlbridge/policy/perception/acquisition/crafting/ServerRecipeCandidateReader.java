package com.hytalerlbridge.policy.perception.acquisition.crafting;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.BenchRequirement;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.MaterialQuantity;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.policy.perception.profile.StatusProgramCatalog;
import com.hytalerlbridge.policy.perception.projection.RecipeCandidateProjection;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Produces exact, bounded Fieldcraft candidates from public server state.
 *
 * <p>The native catalog remains the authority. This reader admits only the
 * same losslessly representable Fieldcraft subset as JAX's
 * {@code make_region_fieldcraft_candidate_provider}: concrete item inputs and
 * outputs, no metadata/tags/resources/exclusions, no knowledge gate, memories
 * level one, zero authored duration, and an exact Crafting/Fieldcraft/tier-0
 * requirement. Unsupported recipes are omitted; an incomplete native catalog,
 * semantic-ID collision, unavailable inventory, or more than 16 legal rows
 * fails the complete actor row closed.
 *
 * <p>Candidate order is the native recipe String-ID order. Native recipe IDs
 * are retained only in {@link Capture#recipeIds()} for commit-time resolution;
 * the policy receives {@link Capture#policy()} and therefore never sees a
 * global recipe identity.
 */
public final class ServerRecipeCandidateReader {

    public static final int NATIVE_RECIPE_CAPACITY = 1_947;

    private static final int BENCH_TYPE_CRAFTING = 0;
    private static final String FIELDCRAFT_ID = "Fieldcraft";
    private static final int ABSENT_ID = -1;

    private final List<RecipeRow> recipes;
    private final boolean catalogAvailable;
    private final String catalogReason;

    private ServerRecipeCandidateReader(
        List<RecipeRow> recipes,
        boolean catalogAvailable,
        String catalogReason
    ) {
        this.recipes = List.copyOf(recipes);
        this.catalogAvailable = catalogAvailable;
        this.catalogReason = catalogReason == null ? "" : catalogReason;
    }

    /** Resolve and validate the complete current server catalog once. */
    public static ServerRecipeCandidateReader load() {
        try {
            Map<String, CraftingRecipe> assets = new TreeMap<>(
                CraftingRecipe.getAssetMap().getAssetMap());
            if (assets.size() != NATIVE_RECIPE_CAPACITY) {
                return unavailableReader("native_recipe_catalog_incomplete");
            }
            List<RecipeRow> supported = new ArrayList<>();
            Map<Integer, String> semanticIds = new HashMap<>();
            for (Map.Entry<String, CraftingRecipe> entry : assets.entrySet()) {
                CraftingRecipe recipe = entry.getValue();
                if (recipe == null || !entry.getKey().equals(recipe.getId())) {
                    return unavailableReader("native_recipe_identity_mismatch");
                }
                RecipeRow row = fromNative(entry.getKey(), recipe);
                if (row == null) continue;
                for (MaterialRow material : row.inputs()) {
                    if (!registerSemanticId(semanticIds, material.itemId())) {
                        return unavailableReader("recipe_semantic_id_collision");
                    }
                }
                for (MaterialRow material : row.outputs()) {
                    if (!registerSemanticId(semanticIds, material.itemId())) {
                        return unavailableReader("recipe_semantic_id_collision");
                    }
                }
                supported.add(row);
            }
            return new ServerRecipeCandidateReader(supported, true, "");
        } catch (RuntimeException unavailable) {
            return unavailableReader("native_recipe_catalog_unavailable");
        }
    }

    /** Capture one actor row from the exact containers native crafting reads. */
    public Capture capture(Ref<EntityStore> ref, Store<EntityStore> store) {
        if (!catalogAvailable) {
            return Capture.unavailable(catalogReason);
        }
        InventoryRows inventory = inventory(ref, store);
        if (!inventory.available()) {
            return Capture.unavailable(inventory.reason());
        }
        return captureRows(recipes, inventory.quantityByItem());
    }

    /** Pure selection seam used by the server-free differential tests. */
    static Capture captureRows(
        List<RecipeRow> recipes,
        Map<String, Integer> quantityByItem
    ) {
        if (recipes == null || quantityByItem == null) {
            return Capture.unavailable("recipe_evidence_unavailable");
        }
        List<RecipeRow> legal = new ArrayList<>();
        String previous = null;
        for (RecipeRow recipe : recipes) {
            if (recipe == null || (previous != null
                && previous.compareTo(recipe.recipeId()) >= 0)) {
                return Capture.unavailable("recipe_catalog_order_invalid");
            }
            previous = recipe.recipeId();
            if (satisfiable(recipe, quantityByItem)) legal.add(recipe);
        }
        int sourceCount = legal.size();
        if (sourceCount > RecipeCandidateProjection.CANDIDATE_CAPACITY) {
            return Capture.capacityExceeded(sourceCount);
        }
        return Capture.available(legal, sourceCount);
    }

    private static InventoryRows inventory(
        Ref<EntityStore> ref,
        Store<EntityStore> store
    ) {
        if (ref == null || !ref.isValid() || store == null) {
            return InventoryRows.unavailable("crafting_inventory_unavailable");
        }
        InventoryComponent.Backpack backpack = store.getComponent(
            ref, InventoryComponent.Backpack.getComponentType());
        InventoryComponent.Storage storage = store.getComponent(
            ref, InventoryComponent.Storage.getComponentType());
        InventoryComponent.Hotbar hotbar = store.getComponent(
            ref, InventoryComponent.Hotbar.getComponentType());
        ItemContainer[] containers = {
            inventory(backpack), inventory(storage), inventory(hotbar),
        };
        if (containers[0] == null && containers[1] == null
            && containers[2] == null) {
            return InventoryRows.unavailable("crafting_inventory_unavailable");
        }
        Map<String, Integer> quantities = new HashMap<>();
        try {
            for (ItemContainer container : containers) {
                if (container == null) continue;
                int capacity = Short.toUnsignedInt(container.getCapacity());
                for (int slot = 0; slot < capacity; slot++) {
                    ItemStack stack = container.getItemStack((short) slot);
                    if (ItemStack.isEmpty(stack)) continue;
                    String itemId = stack.getItemId();
                    int quantity = stack.getQuantity();
                    if (itemId == null || itemId.isBlank() || quantity <= 0) {
                        return InventoryRows.unavailable(
                            "crafting_inventory_invalid");
                    }
                    quantities.merge(itemId, quantity, Math::addExact);
                }
            }
        } catch (RuntimeException invalid) {
            return InventoryRows.unavailable("crafting_inventory_invalid");
        }
        return new InventoryRows(true, Map.copyOf(quantities), "");
    }

    private static ItemContainer inventory(InventoryComponent component) {
        return component == null ? null : component.getInventory();
    }

    private static boolean satisfiable(
        RecipeRow recipe,
        Map<String, Integer> quantityByItem
    ) {
        Map<String, Integer> remaining = new HashMap<>(quantityByItem);
        for (MaterialRow input : recipe.inputs()) {
            int available = remaining.getOrDefault(input.itemId(), 0);
            if (available < input.quantity()) return false;
            remaining.put(input.itemId(), available - input.quantity());
        }
        return true;
    }

    private static RecipeRow fromNative(String recipeId, CraftingRecipe recipe) {
        MaterialQuantity[] inputs = recipe.getInput();
        MaterialQuantity[] outputs = recipe.getOutputs();
        BenchRequirement[] requirements = recipe.getBenchRequirement();
        if (recipe.isKnowledgeRequired()
            || recipe.getRequiredMemoriesLevel() != 1
            || Float.compare(recipe.getTimeSeconds(), 0.0f) != 0
            || inputs == null || inputs.length == 0
            || inputs.length > RecipeCandidateProjection.INGREDIENT_CAPACITY
            || outputs == null || outputs.length == 0
            || outputs.length > RecipeCandidateProjection.OUTPUT_CAPACITY
            || requirements == null || requirements.length == 0
            || requirements.length > RecipeCandidateProjection.REQUIREMENT_CAPACITY
            || !hasFieldcraft(requirements)) {
            return null;
        }
        List<MaterialRow> inputRows = materials(inputs);
        List<MaterialRow> outputRows = materials(outputs);
        List<RequirementRow> requirementRows = requirements(requirements);
        if (inputRows == null || outputRows == null
            || requirementRows == null) {
            return null;
        }
        return new RecipeRow(
            recipeId,
            inputRows,
            outputRows,
            requirementRows,
            false,
            1,
            0.0f
        );
    }

    private static List<MaterialRow> materials(MaterialQuantity[] values) {
        List<MaterialRow> rows = new ArrayList<>(values.length);
        for (MaterialQuantity value : values) {
            if (!concrete(value)) return null;
            rows.add(new MaterialRow(value.getItemId(), value.getQuantity()));
        }
        return List.copyOf(rows);
    }

    private static List<RequirementRow> requirements(
        BenchRequirement[] values
    ) {
        List<RequirementRow> rows = new ArrayList<>(values.length);
        for (BenchRequirement value : values) {
            if (value == null || value.type == null || value.id == null
                || value.id.isBlank() || value.requiredTierLevel < 0) {
                return null;
            }
            rows.add(new RequirementRow(
                value.type.getValue(), value.id, value.requiredTierLevel));
        }
        return List.copyOf(rows);
    }

    private static boolean concrete(MaterialQuantity value) {
        if (value == null || value.getItemId() == null
            || value.getItemId().isBlank() || value.getQuantity() <= 0
            || (value.getResourceTypeId() != null
                && !value.getResourceTypeId().isBlank())
            || value.getTagIndex() != Integer.MIN_VALUE) {
            return false;
        }
        var metadata = value.getMetadata();
        var excluded = value.getExcludedItemIds();
        return (metadata == null || metadata.isEmpty())
            && (excluded == null || excluded.isEmpty());
    }

    private static boolean hasFieldcraft(BenchRequirement[] requirements) {
        for (BenchRequirement requirement : requirements) {
            if (requirement != null && requirement.type != null
                && requirement.type.getValue() == BENCH_TYPE_CRAFTING
                && FIELDCRAFT_ID.equals(requirement.id)
                && requirement.requiredTierLevel == 0) {
                return true;
            }
        }
        return false;
    }

    private static boolean registerSemanticId(
        Map<Integer, String> ids,
        String itemId
    ) {
        int semantic = StatusProgramCatalog.semanticId(itemId);
        String previous = ids.putIfAbsent(semantic, itemId);
        return previous == null || previous.equals(itemId);
    }

    private static int[] identityWords(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(
                value.getBytes(StandardCharsets.UTF_8));
            int[] result = new int[
                RecipeCandidateProjection.IDENTITY_HASH_WORDS];
            ByteBuffer buffer = ByteBuffer.wrap(digest);
            for (int index = 0; index < result.length; index++) {
                result[index] = buffer.getInt();
            }
            return result;
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError("Java runtime has no SHA-256", impossible);
        }
    }

    private static ServerRecipeCandidateReader unavailableReader(String reason) {
        return new ServerRecipeCandidateReader(List.of(), false, reason);
    }

    /** Immutable policy and privileged binding halves of one actor row. */
    public record Capture(
        RecipeCandidateProjection.Input policy,
        String[] recipeIds,
        int sourceCount,
        boolean capacityExceeded,
        String reason
    ) {
        public Capture {
            if (policy == null || recipeIds == null || reason == null
                || recipeIds.length
                    != RecipeCandidateProjection.CANDIDATE_CAPACITY
                || sourceCount < 0) {
                throw new IllegalArgumentException("invalid recipe capture");
            }
            recipeIds = recipeIds.clone();
        }

        @Override public String[] recipeIds() { return recipeIds.clone(); }

        static Capture unavailable(String reason) {
            return new Capture(
                RecipeCandidateProjection.Input.unavailable(),
                new String[RecipeCandidateProjection.CANDIDATE_CAPACITY],
                0,
                false,
                reason
            );
        }

        static Capture capacityExceeded(int sourceCount) {
            return new Capture(
                RecipeCandidateProjection.Input.unavailable(),
                new String[RecipeCandidateProjection.CANDIDATE_CAPACITY],
                sourceCount,
                true,
                "recipe_candidate_capacity_exceeded"
            );
        }

        static Capture available(List<RecipeRow> legal, int sourceCount) {
            int candidates = RecipeCandidateProjection.CANDIDATE_CAPACITY;
            int inputCapacity = RecipeCandidateProjection.INGREDIENT_CAPACITY;
            int outputCapacity = RecipeCandidateProjection.OUTPUT_CAPACITY;
            int requirementCapacity =
                RecipeCandidateProjection.REQUIREMENT_CAPACITY;
            boolean[] candidateMask = new boolean[candidates];
            boolean[] inputMask = new boolean[candidates * inputCapacity];
            int[] inputItemId = new int[candidates * inputCapacity];
            int[] inputResourceTypeId = new int[candidates * inputCapacity];
            java.util.Arrays.fill(inputResourceTypeId, ABSENT_ID);
            int[] inputQuantity = new int[candidates * inputCapacity];
            boolean[] inputMetadataRequired =
                new boolean[candidates * inputCapacity];
            int[] inputMetadataHash = new int[
                candidates * inputCapacity
                    * RecipeCandidateProjection.METADATA_HASH_WORDS];
            boolean[] outputMask = new boolean[candidates * outputCapacity];
            int[] outputItemId = new int[candidates * outputCapacity];
            int[] outputQuantity = new int[candidates * outputCapacity];
            int[] outputMetadataHash = new int[
                candidates * outputCapacity
                    * RecipeCandidateProjection.METADATA_HASH_WORDS];
            boolean[] requirementMask =
                new boolean[candidates * requirementCapacity];
            int[] requirementBenchType =
                new int[candidates * requirementCapacity];
            int[] requirementBenchIdHash = new int[
                candidates * requirementCapacity
                    * RecipeCandidateProjection.IDENTITY_HASH_WORDS];
            int[] requirementTierLevel =
                new int[candidates * requirementCapacity];
            boolean[] knowledgeRequired = new boolean[candidates];
            int[] requiredMemoriesLevel = new int[candidates];
            float[] timeSeconds = new float[candidates];
            String[] recipeIds = new String[candidates];

            for (int candidate = 0; candidate < legal.size(); candidate++) {
                RecipeRow recipe = legal.get(candidate);
                candidateMask[candidate] = true;
                recipeIds[candidate] = recipe.recipeId();
                knowledgeRequired[candidate] = recipe.knowledgeRequired();
                requiredMemoriesLevel[candidate] =
                    recipe.requiredMemoriesLevel();
                timeSeconds[candidate] = recipe.timeSeconds();
                for (int slot = 0; slot < recipe.inputs().size(); slot++) {
                    MaterialRow material = recipe.inputs().get(slot);
                    int index = candidate * inputCapacity + slot;
                    inputMask[index] = true;
                    inputItemId[index] = StatusProgramCatalog.semanticId(
                        material.itemId());
                    inputQuantity[index] = material.quantity();
                }
                for (int slot = 0; slot < recipe.outputs().size(); slot++) {
                    MaterialRow material = recipe.outputs().get(slot);
                    int index = candidate * outputCapacity + slot;
                    outputMask[index] = true;
                    outputItemId[index] = StatusProgramCatalog.semanticId(
                        material.itemId());
                    outputQuantity[index] = material.quantity();
                }
                for (int slot = 0;
                    slot < recipe.requirements().size(); slot++) {
                    RequirementRow requirement = recipe.requirements().get(slot);
                    int index = candidate * requirementCapacity + slot;
                    requirementMask[index] = true;
                    requirementBenchType[index] = requirement.benchType();
                    requirementTierLevel[index] = requirement.tierLevel();
                    int[] words = identityWords(requirement.benchId());
                    System.arraycopy(
                        words,
                        0,
                        requirementBenchIdHash,
                        index * RecipeCandidateProjection.IDENTITY_HASH_WORDS,
                        words.length
                    );
                }
            }
            RecipeCandidateProjection.Input policy =
                new RecipeCandidateProjection.Input(
                    true,
                    candidateMask,
                    inputMask,
                    inputItemId,
                    inputResourceTypeId,
                    inputQuantity,
                    inputMetadataRequired,
                    inputMetadataHash,
                    outputMask,
                    outputItemId,
                    outputQuantity,
                    outputMetadataHash,
                    requirementMask,
                    requirementBenchType,
                    requirementBenchIdHash,
                    requirementTierLevel,
                    knowledgeRequired,
                    requiredMemoriesLevel,
                    timeSeconds
                );
            return new Capture(policy, recipeIds, sourceCount, false, "");
        }
    }

    /** Pure catalog rows are package-visible so tests need no live assets. */
    record RecipeRow(
        String recipeId,
        List<MaterialRow> inputs,
        List<MaterialRow> outputs,
        List<RequirementRow> requirements,
        boolean knowledgeRequired,
        int requiredMemoriesLevel,
        float timeSeconds
    ) {
        RecipeRow {
            inputs = List.copyOf(inputs);
            outputs = List.copyOf(outputs);
            requirements = List.copyOf(requirements);
        }
    }

    record MaterialRow(String itemId, int quantity) {}

    record RequirementRow(int benchType, String benchId, int tierLevel) {}

    private record InventoryRows(
        boolean available,
        Map<String, Integer> quantityByItem,
        String reason
    ) {
        private InventoryRows {
            quantityByItem = Map.copyOf(quantityByItem);
        }

        static InventoryRows unavailable(String reason) {
            return new InventoryRows(false, Map.of(), reason);
        }
    }
}
