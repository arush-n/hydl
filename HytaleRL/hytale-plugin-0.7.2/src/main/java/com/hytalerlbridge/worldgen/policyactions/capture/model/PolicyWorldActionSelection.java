package com.hytalerlbridge.worldgen.policyactions.capture.model;

/** Bounded learner selection carried by a commit capture request. */
public record PolicyWorldActionSelection(
    boolean useRequested,
    int blockInteractionTrigger,
    int blockCandidateIndex,
    int recipeCandidateIndex
) {
    public PolicyWorldActionSelection {
        if (
            blockInteractionTrigger < 0
                || blockInteractionTrigger > 2
                || blockCandidateIndex < -1
                || blockCandidateIndex
                    >= NativePolicyWorldActionCapture.BLOCK_CANDIDATE_CAPACITY
                || recipeCandidateIndex < -1
                || recipeCandidateIndex
                    >= NativePolicyWorldActionCapture.RECIPE_CANDIDATE_CAPACITY
        ) {
            throw new IllegalArgumentException(
                "Policy World-action selection is outside its bounded domain"
            );
        }
        int selectedCount = (useRequested ? 1 : 0)
            + (!useRequested && blockCandidateIndex >= 0 ? 1 : 0)
            + (recipeCandidateIndex >= 0 ? 1 : 0);
        if (selectedCount != 1) {
            throw new IllegalArgumentException(
                "Commit must select exactly one World verb"
            );
        }
        if (
            !useRequested
                && blockCandidateIndex >= 0
                && blockInteractionTrigger != 1
                && blockInteractionTrigger != 2
        ) {
            throw new IllegalArgumentException(
                "Selected block action requires Primary or Secondary"
            );
        }
    }

    public int selectedCount() {
        return (useRequested ? 1 : 0)
            + (!useRequested && blockCandidateIndex >= 0 ? 1 : 0)
            + (recipeCandidateIndex >= 0 ? 1 : 0);
    }

    public boolean selectsBlock() {
        return !useRequested && blockCandidateIndex >= 0;
    }

    public boolean selectsRecipe() {
        return recipeCandidateIndex >= 0;
    }
}
