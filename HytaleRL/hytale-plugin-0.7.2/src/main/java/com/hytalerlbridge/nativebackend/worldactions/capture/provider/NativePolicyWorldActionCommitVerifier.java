package com.hytalerlbridge.nativebackend.worldactions.capture.provider;

import com.hytalerlbridge.action.NativeWorldVerbRequest;
import com.hytalerlbridge.worldgen.policyactions.capture.model.NativePolicyWorldActionCapture;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionBinding;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionBlockSurface;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionRecipeSurface;
import com.hytalerlbridge.worldgen.policyactions.capture.semantic.PolicyWorldActionSemanticHash;
import java.util.Arrays;

/** Verifies a typed request against a fresh, World-thread atomic capture. */
public final class NativePolicyWorldActionCommitVerifier {

    public static final String STALE_GENERATION =
        "stale_policy_world_action_generation";
    public static final String STALE_SELECTION =
        "stale_policy_world_action_selection";
    public static final String PLACE_ROTATION_UNAVAILABLE =
        "native_policy_place_rotation_zero_destination_geometry_unavailable";

    private NativePolicyWorldActionCommitVerifier() {}

    public static Verification verify(
        NativeWorldVerbRequest request,
        NativePolicyWorldActionCapture fresh
    ) {
        if (request == null || !request.present()) {
            return Verification.rejected("native_world_verb_request_absent");
        }
        if (!request.candidateEvidenceBound()) {
            return Verification.rejected(
                NativeWorldVerbRequest.POLICY_EVIDENCE_REQUIRED
            );
        }
        if (fresh == null || !fresh.phase().equals("observe")) {
            return Verification.rejected(
                "native_policy_world_action_capture_unavailable"
            );
        }
        if (!request.worldEpoch().equals(fresh.worldEpoch())) {
            return Verification.rejected("stale_world_epoch");
        }
        if (!request.expectedCandidateGenerationSha256().equalsIgnoreCase(
            fresh.candidateGenerationSha256()
        )) {
            return Verification.rejected(STALE_GENERATION);
        }
        if (request.verb().equals("place_block")) {
            return Verification.rejected(PLACE_ROTATION_UNAVAILABLE);
        }
        boolean selected = switch (request.verb()) {
            case "use" -> useMatches(request, fresh);
            case "break_block" -> blockMatches(request, fresh);
            case "craft_recipe" -> recipeMatches(request, fresh);
            default -> false;
        };
        return selected
            ? Verification.certified()
            : Verification.rejected(STALE_SELECTION);
    }

    private static boolean useMatches(
        NativeWorldVerbRequest request,
        NativePolicyWorldActionCapture fresh
    ) {
        PolicyWorldActionBinding binding = fresh.use().binding();
        return fresh.use().available()
            && binding != null
            && selectedIdentityMatches(request, binding)
            && bindingMatches(request, binding);
    }

    private static boolean blockMatches(
        NativeWorldVerbRequest request,
        NativePolicyWorldActionCapture fresh
    ) {
        if (!fresh.blocks().available()) return false;
        for (PolicyWorldActionBlockSurface.Candidate candidate
            : fresh.blocks().candidates()) {
            if (
                candidate.primary() != null
                    && selectedIdentityMatches(request, candidate.primary())
            ) {
                return bindingMatches(request, candidate.primary());
            }
            if (
                candidate.secondary() != null
                    && selectedIdentityMatches(request, candidate.secondary())
            ) {
                return bindingMatches(request, candidate.secondary());
            }
        }
        return false;
    }

    private static boolean recipeMatches(
        NativeWorldVerbRequest request,
        NativePolicyWorldActionCapture fresh
    ) {
        PolicyWorldActionRecipeSurface surface = fresh.recipes();
        if (!surface.available() || surface.craftingContext() == null) {
            return false;
        }
        for (PolicyWorldActionRecipeSurface.Candidate candidate
            : surface.candidates()) {
            String identity = PolicyWorldActionSemanticHash
                .recipeCandidateIdentity(
                    fresh.recipeTableIdentitySha256(),
                    surface.craftingContext(),
                    candidate
                );
            if (
                identity.equalsIgnoreCase(
                    request.expectedSelectedSemanticSha256()
                )
            ) {
                return recipeRequestMatches(request, surface, candidate);
            }
        }
        return false;
    }

    private static boolean selectedIdentityMatches(
        NativeWorldVerbRequest request,
        PolicyWorldActionBinding binding
    ) {
        return PolicyWorldActionSemanticHash.bindingIdentity(binding)
            .equalsIgnoreCase(request.expectedSelectedSemanticSha256());
    }

    private static boolean bindingMatches(
        NativeWorldVerbRequest request,
        PolicyWorldActionBinding binding
    ) {
        int[] target = binding.target();
        int[] rotation = binding.rotation();
        return request.verb().equals(binding.verb())
            && request.interactionType() == binding.interactionType()
            && request.targetKind().equals("block")
            && request.interactionId().equals(binding.interactionId())
            && request.recipeId().isEmpty()
            && request.itemId().equals(binding.sourceItemId())
            && request.blockId().isEmpty()
            && Arrays.equals(
                target,
                new int[] {
                    request.targetX(),
                    request.targetY(),
                    request.targetZ(),
                }
            )
            && request.blockFace() == binding.blockFace()
            && Arrays.equals(
                rotation,
                new int[] {
                    request.rotationYaw(),
                    request.rotationPitch(),
                    request.rotationRoll(),
                }
            )
            && request.sourceContainer().equals(binding.sourceContainer())
            && request.sourceSlot() == binding.sourceSlot()
            && request.expectedSourceQuantity() == binding.sourceQuantity()
            && request.destinationContainer().isEmpty()
            && request.expectedBlockId().equals(binding.expectedBlockId())
            && request.quantity() == 1
            && request.placementVariant().isEmpty()
            && request.craftingContext().isEmpty()
            && request.benchX() == 0
            && request.benchY() == 0
            && request.benchZ() == 0
            && request.expectedBenchBlockId().isEmpty()
            && request.expectedBenchId().isEmpty()
            && request.expectedBenchType() == -1
            && request.expectedBenchTier() == 0;
    }

    private static boolean recipeRequestMatches(
        NativeWorldVerbRequest request,
        PolicyWorldActionRecipeSurface surface,
        PolicyWorldActionRecipeSurface.Candidate candidate
    ) {
        PolicyWorldActionRecipeSurface.CraftingContext context =
            surface.craftingContext();
        int[] position = context.position();
        int[] expectedPosition = position.length == 0
            ? new int[] {0, 0, 0}
            : position;
        return request.interactionType() == -1
            && request.targetKind().equals("recipe")
            && request.interactionId().isEmpty()
            && request.recipeId().equals(candidate.recipeId())
            && request.itemId().isEmpty()
            && request.blockId().isEmpty()
            && request.targetX() == 0
            && request.targetY() == 0
            && request.targetZ() == 0
            && request.blockFace() == 0
            && request.rotationYaw() == 0
            && request.rotationPitch() == 0
            && request.rotationRoll() == 0
            && request.sourceContainer().equals("player_inventory")
            && request.sourceSlot() == -1
            && request.expectedSourceQuantity() == -1
            && request.destinationContainer().equals(
                "player_inventory_or_world_drop"
            )
            && request.expectedBlockId().isEmpty()
            && request.quantity() == 1
            && request.placementVariant().isEmpty()
            && request.craftingContext().equals(context.kind())
            && request.benchX() == expectedPosition[0]
            && request.benchY() == expectedPosition[1]
            && request.benchZ() == expectedPosition[2]
            && request.expectedBenchBlockId().equals(context.blockId())
            && request.expectedBenchId().equals(context.benchId())
            && request.expectedBenchType() == context.benchType()
            && request.expectedBenchTier() == context.tier();
    }

    /** Every accepted result is bound to fresh candidate evidence. */
    public record Verification(
        boolean accepted,
        boolean candidateEvidenceCertified,
        String rejectReason
    ) {
        private static Verification certified() {
            return new Verification(true, true, "");
        }

        private static Verification rejected(String reason) {
            return new Verification(false, false, reason);
        }

        public Verification {
            if (
                accepted == !rejectReason.isEmpty()
                    || accepted != candidateEvidenceCertified
            ) {
                throw new IllegalArgumentException(
                    "World-action commit verification is inconsistent"
                );
            }
        }
    }
}
