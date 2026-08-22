package com.hytalerlbridge.policy.world.model;

import java.util.Arrays;

/**
 * Privileged action bindings captured beside one policy observation.
 *
 * <p>The learner sees only bounded candidate features and indices. Native
 * execution needs the corresponding stable asset identity, but that identity
 * must never leak into the observation. Keeping the two halves in this
 * immutable value lets a held decision retain the binding it was sampled
 * against. Commit-time code must still recapture server evidence and require
 * the selected binding to match.</p>
 */
public record WorldActionEvidence(
    String[] recipeIds,
    BlockCandidateBinding[] blockBindings
) {

    public static final int RECIPE_CAPACITY = 16;
    public static final int BLOCK_CAPACITY = 16;
    private static final WorldActionEvidence EMPTY =
        new WorldActionEvidence(
            new String[RECIPE_CAPACITY], emptyBlockBindings());

    /** Compatibility constructor for recipe-only evidence producers. */
    public WorldActionEvidence(String[] recipeIds) {
        this(recipeIds, emptyBlockBindings());
    }

    public WorldActionEvidence {
        if (recipeIds == null || recipeIds.length != RECIPE_CAPACITY) {
            throw new IllegalArgumentException(
                "recipe binding width must be " + RECIPE_CAPACITY);
        }
        if (blockBindings == null || blockBindings.length != BLOCK_CAPACITY) {
            throw new IllegalArgumentException(
                "block binding width must be " + BLOCK_CAPACITY);
        }
        recipeIds = recipeIds.clone();
        blockBindings = blockBindings.clone();
        for (int index = 0; index < recipeIds.length; index++) {
            if (recipeIds[index] == null) {
                recipeIds[index] = "";
            } else if (!recipeIds[index].equals(recipeIds[index].trim())
                || recipeIds[index].indexOf('\0') >= 0) {
                throw new IllegalArgumentException(
                    "recipe binding contains an invalid identity");
            }
        }
        for (BlockCandidateBinding binding : blockBindings) {
            if (binding == null) {
                throw new IllegalArgumentException(
                    "block bindings cannot contain null");
            }
        }
    }

    @Override
    public String[] recipeIds() {
        return recipeIds.clone();
    }

    @Override
    public BlockCandidateBinding[] blockBindings() {
        return blockBindings.clone();
    }

    /** Stable recipe identity sampled at {@code index}, or empty if absent. */
    public String recipeId(int index) {
        return index < 0 || index >= recipeIds.length ? "" : recipeIds[index];
    }

    public boolean hasRecipe(int index) {
        return !recipeId(index).isEmpty();
    }

    public BlockCandidateBinding blockBinding(int index) {
        return index < 0 || index >= blockBindings.length
            ? BlockCandidateBinding.empty() : blockBindings[index];
    }

    public static WorldActionEvidence empty() {
        return EMPTY;
    }

    public boolean isEmpty() {
        return Arrays.stream(recipeIds).allMatch(String::isEmpty)
            && Arrays.stream(blockBindings).noneMatch(
                BlockCandidateBinding::available);
    }

    private static BlockCandidateBinding[] emptyBlockBindings() {
        BlockCandidateBinding[] result =
            new BlockCandidateBinding[BLOCK_CAPACITY];
        Arrays.fill(result, BlockCandidateBinding.empty());
        return result;
    }
}
