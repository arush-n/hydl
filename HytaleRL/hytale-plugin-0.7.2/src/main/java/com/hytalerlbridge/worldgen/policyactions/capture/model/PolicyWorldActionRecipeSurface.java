package com.hytalerlbridge.worldgen.policyactions.capture.model;

import com.hytalerlbridge.worldgen.NativeCraftingCatalogEvidence;
import java.util.List;

/** Complete recipe legality context and bounded native candidates. */
public record PolicyWorldActionRecipeSurface(
    boolean available,
    String unavailableReason,
    CraftingContext craftingContext,
    int memoriesLevel,
    boolean knowledgeAvailable,
    boolean benchContextAvailable,
    boolean managerAvailable,
    boolean managerHasBench,
    int managerQueueSize,
    String managerQueueRecipeId,
    String managerQueueIdentitySha256,
    int sourceCount,
    int emittedCount,
    boolean capacityExceeded,
    List<Candidate> candidates
) {
    public PolicyWorldActionRecipeSurface {
        unavailableReason = CaptureValidation.optionalText(unavailableReason);
        managerQueueRecipeId = CaptureValidation.optionalText(
            managerQueueRecipeId
        );
        managerQueueIdentitySha256 = CaptureValidation.optionalSha256(
            managerQueueIdentitySha256,
            "managerQueueIdentitySha256"
        );
        candidates = candidates == null ? List.of() : List.copyOf(candidates);
        if (
            memoriesLevel < 0
                || managerQueueSize < 0
                || sourceCount < 0
                || emittedCount < 0
                || emittedCount
                    > NativePolicyWorldActionCapture.RECIPE_CANDIDATE_CAPACITY
                || emittedCount != candidates.size()
                || capacityExceeded != (
                    sourceCount
                        > NativePolicyWorldActionCapture.RECIPE_CANDIDATE_CAPACITY
                )
                || (managerQueueSize == 0
                    && (!managerQueueRecipeId.isEmpty()
                        || !managerQueueIdentitySha256.isEmpty()))
                || (benchContextAvailable != (craftingContext != null))
        ) {
            throw new IllegalArgumentException(
                "Recipe context or candidate counts are inconsistent"
            );
        }
        if (available) {
            if (
                !unavailableReason.isEmpty()
                    || craftingContext == null
                    || !knowledgeAvailable
                    || !benchContextAvailable
                    || !managerAvailable
                    || managerHasBench
                    || managerQueueSize != 0
                    || capacityExceeded
                    || sourceCount == 0
                    || sourceCount != emittedCount
            ) {
                throw new IllegalArgumentException(
                    "Available recipe surface is incomplete"
                );
            }
        } else if (
            unavailableReason.isEmpty()
                || emittedCount != 0
                || !candidates.isEmpty()
        ) {
            throw new IllegalArgumentException(
                "Unavailable recipe surface must fail closed"
            );
        }
        for (int index = 0; index < candidates.size(); index++) {
            Candidate candidate = candidates.get(index);
            if (candidate.slot() != index || !candidate.actorLegal()) {
                throw new IllegalArgumentException(
                    "Recipe candidates must be actor-legal and slot ordered"
                );
            }
        }
    }

    public static PolicyWorldActionRecipeSurface unavailable(
        String reason,
        boolean managerAvailable,
        boolean managerHasBench,
        int managerQueueSize
    ) {
        return new PolicyWorldActionRecipeSurface(
            false,
            CaptureValidation.text(reason, "recipe unavailable reason"),
            null,
            0,
            false,
            false,
            managerAvailable,
            managerHasBench,
            managerQueueSize,
            "",
            "",
            0,
            0,
            false,
            List.of()
        );
    }

    public static PolicyWorldActionRecipeSurface unavailable(
        String reason,
        CraftingContext context,
        int memoriesLevel,
        boolean knowledgeAvailable,
        boolean benchContextAvailable,
        boolean managerAvailable,
        boolean managerHasBench,
        int managerQueueSize,
        int sourceCount
    ) {
        return new PolicyWorldActionRecipeSurface(
            false,
            CaptureValidation.text(reason, "recipe unavailable reason"),
            context,
            memoriesLevel,
            knowledgeAvailable,
            benchContextAvailable,
            managerAvailable,
            managerHasBench,
            managerQueueSize,
            "",
            "",
            sourceCount,
            0,
            sourceCount
                > NativePolicyWorldActionCapture.RECIPE_CANDIDATE_CAPACITY,
            List.of()
        );
    }

    public record CraftingContext(
        String kind,
        int[] position,
        String blockId,
        String benchId,
        int benchType,
        int tier
    ) {
        public CraftingContext {
            kind = CaptureValidation.choice(
                kind,
                "crafting context kind",
                List.of("fieldcraft", "bench")
            );
            position = CaptureValidation.ints(position);
            blockId = CaptureValidation.optionalText(blockId);
            benchId = CaptureValidation.text(benchId, "benchId");
            if (kind.equals("fieldcraft")) {
                if (
                    position.length != 0
                        || !blockId.isEmpty()
                        || !benchId.equals("Fieldcraft")
                        || benchType != 0
                        || tier != 0
                ) {
                    throw new IllegalArgumentException(
                        "Fieldcraft context must have canonical empty geometry"
                    );
                }
            } else if (
                position.length != 3
                    || blockId.isEmpty()
                    || benchType < 0
                    || benchType > 3
                    || tier < 1
            ) {
                throw new IllegalArgumentException(
                    "Ordinary bench context is incomplete"
                );
            }
        }

        @Override public int[] position() { return position.clone(); }

        public static CraftingContext fieldcraft() {
            return new CraftingContext(
                "fieldcraft",
                new int[0],
                "",
                "Fieldcraft",
                0,
                0
            );
        }
    }

    public record Candidate(
        int slot,
        int nativeRecipeIndex,
        String recipeId,
        byte[] recipeIdSha256,
        NativeCraftingCatalogEvidence.Recipe recipe,
        boolean knowledgeSatisfied,
        boolean memorySatisfied,
        boolean benchSatisfied,
        String knowledgeKey
    ) {
        public Candidate {
            recipeId = CaptureValidation.text(recipeId, "recipeId");
            recipeIdSha256 = CaptureValidation.bytes(recipeIdSha256);
            knowledgeKey = CaptureValidation.optionalText(knowledgeKey);
            if (
                slot < 0
                    || slot
                        >= NativePolicyWorldActionCapture.RECIPE_CANDIDATE_CAPACITY
                    || nativeRecipeIndex < 0
                    || nativeRecipeIndex
                        >= NativeCraftingCatalogEvidence.RECIPE_CAPACITY
                    || recipeIdSha256.length != 32
                    || recipe == null
                    || !recipe.recipeId().equals(recipeId)
                    || (recipe.knowledgeRequired()
                        != !knowledgeKey.isEmpty())
                    || (recipe.knowledgeRequired()
                        && !knowledgeKey.equals(
                            recipe.primaryOutputItemAssetId()
                        ))
            ) {
                throw new IllegalArgumentException(
                    "Recipe candidate identity is incomplete"
                );
            }
        }

        public boolean actorLegal() {
            return knowledgeSatisfied && memorySatisfied && benchSatisfied;
        }

        @Override public byte[] recipeIdSha256() {
            return recipeIdSha256.clone();
        }
    }
}
