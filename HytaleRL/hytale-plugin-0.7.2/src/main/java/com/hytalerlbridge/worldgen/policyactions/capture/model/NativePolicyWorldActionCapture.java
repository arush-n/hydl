package com.hytalerlbridge.worldgen.policyactions.capture.model;

import com.hytalerlbridge.observation.NativeInventoryFrame;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence;
import com.hytalerlbridge.worldgen.NativeMutableBlockCells;
import java.util.List;

/** One World-thread atomic actor-safe candidate capture and commit receipt. */
public record NativePolicyWorldActionCapture(
    String phase,
    String bridgeSha256,
    String worldName,
    String worldEpoch,
    long worldTick,
    int environmentStep,
    int actorSlot,
    String actorIdentity,
    String inventoryIdentitySha256,
    String recipeTableIdentitySha256,
    String candidateGenerationSha256,
    String expectedCandidateGenerationSha256,
    boolean expectedGenerationMatched,
    NativeInventoryFrame inventory,
    PolicyWorldActionCamera camera,
    PolicyWorldActionBlockSurface blocks,
    PolicyWorldActionRecipeSurface recipes,
    PolicyWorldActionUseSurface use,
    PolicyWorldActionSelection selection,
    NativeMutableBlockCells commitBlockCells,
    NativeItemInteractionEvidence commitItemInteractions
) {
    public static final String TYPE = "policy_world_action_capture";
    public static final String SCHEMA =
        "hytalerl_native_policy_world_action_capture_v2";
    public static final int VERSION = 2;
    public static final String CONTRACT_SHA256 =
        "5E6672AD838CB9D7E705EC196FF84D3D9992FCAFD5D5256504B3C76007289F3D";
    public static final int BLOCK_CANDIDATE_CAPACITY = 16;
    public static final int RECIPE_CANDIDATE_CAPACITY = 16;

    public NativePolicyWorldActionCapture {
        phase = CaptureValidation.choice(
            phase,
            "phase",
            List.of("observe", "commit")
        );
        bridgeSha256 = CaptureValidation.sha256(
            bridgeSha256,
            "bridgeSha256"
        );
        worldName = CaptureValidation.text(worldName, "worldName");
        worldEpoch = CaptureValidation.text(worldEpoch, "worldEpoch");
        actorIdentity = CaptureValidation.text(
            actorIdentity,
            "actorIdentity"
        );
        inventoryIdentitySha256 = CaptureValidation.sha256(
            inventoryIdentitySha256,
            "inventoryIdentitySha256"
        );
        recipeTableIdentitySha256 = CaptureValidation.sha256(
            recipeTableIdentitySha256,
            "recipeTableIdentitySha256"
        );
        candidateGenerationSha256 = CaptureValidation.sha256(
            candidateGenerationSha256,
            "candidateGenerationSha256"
        );
        expectedCandidateGenerationSha256 =
            CaptureValidation.optionalSha256(
                expectedCandidateGenerationSha256,
                "expectedCandidateGenerationSha256"
            );
        if (
            worldTick < 0L
                || environmentStep < 0
                || actorSlot < 0
                || inventory == null
                || camera == null
                || blocks == null
                || recipes == null
                || use == null
        ) {
            throw new IllegalArgumentException(
                "Policy World-action capture provenance is incomplete"
            );
        }
        if (phase.equals("observe")) {
            if (
                !expectedCandidateGenerationSha256.isEmpty()
                    || !expectedGenerationMatched
                    || selection != null
                    || commitBlockCells != null
                    || commitItemInteractions != null
            ) {
                throw new IllegalArgumentException(
                    "Observe capture cannot carry commit-only evidence"
                );
            }
        } else {
            if (
                expectedCandidateGenerationSha256.isEmpty()
                    || selection == null
                    || expectedGenerationMatched
                        != candidateGenerationSha256.equalsIgnoreCase(
                            expectedCandidateGenerationSha256
                        )
            ) {
                throw new IllegalArgumentException(
                    "Commit generation or selection is inconsistent"
                );
            }
            boolean needsBlockEvidence = expectedGenerationMatched
                && selectedBindingExists(
                    selection,
                    blocks,
                    use
                );
            if (
                needsBlockEvidence
                    != (commitBlockCells != null
                        && commitItemInteractions != null)
            ) {
                throw new IllegalArgumentException(
                    "Use/Break commit evidence must be complete and verb-scoped"
                );
            }
        }
    }

    private static boolean selectedBindingExists(
        PolicyWorldActionSelection selection,
        PolicyWorldActionBlockSurface blocks,
        PolicyWorldActionUseSurface use
    ) {
        if (selection.useRequested()) return use.available();
        if (!selection.selectsBlock() || !blocks.available()) return false;
        int slot = selection.blockCandidateIndex();
        if (slot < 0 || slot >= blocks.candidates().size()) return false;
        PolicyWorldActionBlockSurface.Candidate candidate =
            blocks.candidates().get(slot);
        return selection.blockInteractionTrigger() == 1
            ? candidate.primary() != null
            : candidate.secondary() != null;
    }
}
