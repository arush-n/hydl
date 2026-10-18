package com.hytalerlbridge.nativebackend.worldactions.capture.provider;

import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.nativebackend.worldactions.capture.NativeFieldcraftLegality;
import com.hytalerlbridge.worldgen.NativeCraftingCatalogEvidence;
import com.hytalerlbridge.worldgen.policyactions.capture.model.NativePolicyWorldActionCapture;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionRecipeSurface;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;

/** Produces the bounded actor-legal idle-Fieldcraft recipe surface. */
public final class NativePolicyRecipeSurfaceProvider {

    private NativePolicyRecipeSurfaceProvider() {}

    public static PolicyWorldActionRecipeSurface capture(Context context) {
        if (
            context.actor() == null
                || !context.actor().isValid()
                || context.store() == null
                || context.world() == null
        ) {
            return PolicyWorldActionRecipeSurface.unavailable(
                "controlled_actor_unavailable",
                false,
                false,
                0
            );
        }
        CraftingManager manager = context.store().getComponent(
            context.actor(),
            CraftingManager.getComponentType()
        );
        NativeFieldcraftLegality.ActorContext actor =
            NativeFieldcraftLegality.capture(
                context.actor(),
                context.store(),
                context.world()
            );
        boolean managerAvailable = manager != null;
        boolean managerHasBench = managerAvailable && manager.hasBenchSet();
        int managerQueueSize = managerAvailable
            ? manager.getRemainingQueueSize()
            : 0;
        boolean fieldcraftContext = managerAvailable && !managerHasBench;
        PolicyWorldActionRecipeSurface.CraftingContext craftingContext =
            fieldcraftContext
                ? PolicyWorldActionRecipeSurface.CraftingContext.fieldcraft()
                : null;
        int memoriesLevel = actor.available() ? actor.memoriesLevel() : 0;
        boolean knowledgeAvailable = actor.available();

        String preflightReason = "";
        if (!managerAvailable) {
            preflightReason = "headless_crafting_manager_unavailable";
        } else if (managerHasBench) {
            preflightReason = "ordinary_bench_context_not_supported_v1";
        } else if (managerQueueSize != 0) {
            preflightReason = "headless_crafting_manager_busy";
        } else if (!actor.available()) {
            preflightReason = actor.unavailableReason();
        } else if (context.catalog() == null) {
            preflightReason = "native_recipe_catalog_unavailable";
        }
        if (!preflightReason.isEmpty()) {
            return PolicyWorldActionRecipeSurface.unavailable(
                preflightReason,
                craftingContext,
                memoriesLevel,
                knowledgeAvailable,
                fieldcraftContext,
                managerAvailable,
                managerHasBench,
                managerQueueSize,
                0
            );
        }

        List<PolicyWorldActionRecipeSurface.Candidate> legal =
            new ArrayList<>();
        int sourceCount = 0;
        List<NativeCraftingCatalogEvidence.Recipe> recipes =
            context.catalog().recipes();
        for (int nativeIndex = 0; nativeIndex < recipes.size(); nativeIndex++) {
            NativeCraftingCatalogEvidence.Recipe evidence = recipes.get(
                nativeIndex
            );
            CraftingRecipe recipe = CraftingRecipe.getAssetMap().getAsset(
                evidence.recipeId()
            );
            if (recipe == null || !recipe.getId().equals(evidence.recipeId())) {
                return PolicyWorldActionRecipeSurface.unavailable(
                    "native_recipe_catalog_identity_changed",
                    craftingContext,
                    memoriesLevel,
                    true,
                    true,
                    true,
                    false,
                    0,
                    0
                );
            }
            String reason = NativeFieldcraftLegality.rejectionReason(
                recipe,
                actor,
                1
            );
            if (!reason.isEmpty()) continue;
            int slot = sourceCount++;
            if (
                slot
                    < NativePolicyWorldActionCapture.RECIPE_CANDIDATE_CAPACITY
            ) {
                legal.add(new PolicyWorldActionRecipeSurface.Candidate(
                    slot,
                    nativeIndex,
                    evidence.recipeId(),
                    sha256(evidence.recipeId()),
                    evidence,
                    true,
                    true,
                    true,
                    NativeFieldcraftLegality.knowledgeKey(recipe)
                ));
            }
        }
        if (sourceCount == 0) {
            return PolicyWorldActionRecipeSurface.unavailable(
                "no_actor_legal_fieldcraft_recipes",
                craftingContext,
                memoriesLevel,
                true,
                true,
                true,
                false,
                0,
                0
            );
        }
        if (
            sourceCount
                > NativePolicyWorldActionCapture.RECIPE_CANDIDATE_CAPACITY
        ) {
            return PolicyWorldActionRecipeSurface.unavailable(
                "recipe_candidate_capacity_exceeded",
                craftingContext,
                memoriesLevel,
                true,
                true,
                true,
                false,
                0,
                sourceCount
            );
        }
        return new PolicyWorldActionRecipeSurface(
            true,
            "",
            craftingContext,
            memoriesLevel,
            true,
            true,
            true,
            false,
            0,
            "",
            "",
            sourceCount,
            sourceCount,
            false,
            legal
        );
    }

    private static byte[] sha256(String value) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(
                value.getBytes(StandardCharsets.UTF_8)
            );
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    public record Context(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        World world,
        NativeCraftingCatalogEvidence catalog
    ) {}
}
