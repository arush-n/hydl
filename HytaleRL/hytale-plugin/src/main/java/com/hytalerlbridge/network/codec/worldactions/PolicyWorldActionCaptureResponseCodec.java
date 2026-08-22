package com.hytalerlbridge.network.codec.worldactions;

import static com.hytalerlbridge.network.codec.CraftingCodec.packCraftingRecipe;
import static com.hytalerlbridge.network.codec.EvidenceCodec.packMutableBlockCells;
import static com.hytalerlbridge.network.codec.EvidenceCodec.packMutableBlockRow;
import static com.hytalerlbridge.network.codec.ItemInteractionCodec.packItemInteractionEvidence;
import static com.hytalerlbridge.network.codec.ObservationCodec.packNativeInventory;
import static com.hytalerlbridge.network.wire.BinaryCodec.packBinary;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;

import com.hytalerlbridge.worldgen.policyactions.capture.model.NativePolicyWorldActionCapture;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionBinding;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionBlockSurface;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionCamera;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionRecipeSurface;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionSelection;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionUseSurface;
import java.io.DataOutputStream;
import java.io.IOException;
import org.msgpack.core.MessagePacker;

/** Lossless MessagePack response for one atomic native capture. */
public final class PolicyWorldActionCaptureResponseCodec {

    private PolicyWorldActionCaptureResponseCodec() {}

    public static void send(
        DataOutputStream output,
        NativePolicyWorldActionCapture capture
    ) throws IOException {
        sendFrame(output, packer -> pack(packer, capture));
    }

    public static void pack(
        MessagePacker packer,
        NativePolicyWorldActionCapture capture
    ) throws IOException {
        packer.packMapHeader(25);
        string(packer, "type", NativePolicyWorldActionCapture.TYPE);
        string(packer, "schema", NativePolicyWorldActionCapture.SCHEMA);
        integer(packer, "version", NativePolicyWorldActionCapture.VERSION);
        string(
            packer,
            "contract_sha256",
            NativePolicyWorldActionCapture.CONTRACT_SHA256
        );
        string(packer, "phase", capture.phase());
        string(packer, "bridge_sha256", capture.bridgeSha256());
        string(packer, "world", capture.worldName());
        string(packer, "world_epoch", capture.worldEpoch());
        longValue(packer, "world_tick", capture.worldTick());
        integer(packer, "environment_step", capture.environmentStep());
        integer(packer, "actor_slot", capture.actorSlot());
        string(packer, "actor_identity", capture.actorIdentity());
        string(
            packer,
            "inventory_identity_sha256",
            capture.inventoryIdentitySha256()
        );
        string(
            packer,
            "recipe_table_identity_sha256",
            capture.recipeTableIdentitySha256()
        );
        string(
            packer,
            "candidate_generation_sha256",
            capture.candidateGenerationSha256()
        );
        string(
            packer,
            "expected_candidate_generation_sha256",
            capture.expectedCandidateGenerationSha256()
        );
        bool(
            packer,
            "expected_generation_matched",
            capture.expectedGenerationMatched()
        );
        packer.packString("inventory");
        packNativeInventory(packer, capture.inventory());
        packer.packString("camera");
        packCamera(packer, capture.camera());
        packer.packString("blocks");
        packBlocks(packer, capture.blocks());
        packer.packString("recipes");
        packRecipes(
            packer,
            capture.recipeTableIdentitySha256(),
            capture.recipes()
        );
        packer.packString("use");
        packUse(packer, capture.use());
        packer.packString("selection");
        packSelection(packer, capture.selection());
        packer.packString("commit_block_cells");
        if (capture.commitBlockCells() == null) {
            packer.packNil();
        } else {
            packMutableBlockCells(
                packer,
                capture.commitBlockCells(),
                capture.bridgeSha256()
            );
        }
        packer.packString("commit_item_interactions");
        if (capture.commitItemInteractions() == null) {
            packer.packNil();
        } else {
            packItemInteractionEvidence(
                packer,
                capture.commitItemInteractions(),
                capture.bridgeSha256()
            );
        }
    }

    private static void packCamera(
        MessagePacker packer,
        PolicyWorldActionCamera camera
    ) throws IOException {
        packer.packMapHeader(9);
        bool(packer, "available", camera.available());
        string(packer, "unavailable_reason", camera.unavailableReason());
        doubles(packer, "actor_position", camera.actorPosition());
        doubles(packer, "eye_position", camera.eyePosition());
        doubles(packer, "direction", camera.direction());
        doubleValue(packer, "maximum_distance", camera.maximumDistance());
        ints(packer, "raw_hit_cell", camera.rawHitCell());
        ints(
            packer,
            "canonical_action_cell",
            camera.canonicalActionCell()
        );
        integer(packer, "block_face", camera.blockFace());
    }

    private static void packBlocks(
        MessagePacker packer,
        PolicyWorldActionBlockSurface surface
    ) throws IOException {
        packer.packMapHeader(8);
        bool(packer, "available", surface.available());
        string(packer, "unavailable_reason", surface.unavailableReason());
        integer(packer, "source_count", surface.sourceCount());
        integer(packer, "emitted_count", surface.emittedCount());
        bool(packer, "capacity_exceeded", surface.capacityExceeded());
        string(
            packer,
            "primary_unavailable_reason",
            surface.primaryUnavailableReason()
        );
        string(
            packer,
            "secondary_unavailable_reason",
            surface.secondaryUnavailableReason()
        );
        packer.packString("candidates");
        packer.packArrayHeader(surface.candidates().size());
        for (PolicyWorldActionBlockSurface.Candidate candidate
            : surface.candidates()) {
            packer.packMapHeader(7);
            integer(packer, "slot", candidate.slot());
            ints(packer, "visible_cell", candidate.visibleCell());
            ints(packer, "action_cell", candidate.actionCell());
            doubles(
                packer,
                "visible_relative_position",
                candidate.visibleRelativePosition()
            );
            packer.packString("semantics");
            packMutableBlockRow(packer, candidate.semantics());
            packer.packString("primary");
            nullableBinding(packer, candidate.primary());
            packer.packString("secondary");
            nullableBinding(packer, candidate.secondary());
        }
    }

    private static void packRecipes(
        MessagePacker packer,
        String recipeTableIdentity,
        PolicyWorldActionRecipeSurface surface
    ) throws IOException {
        packer.packMapHeader(15);
        bool(packer, "available", surface.available());
        string(packer, "unavailable_reason", surface.unavailableReason());
        packer.packString("crafting_context");
        packContext(packer, surface.craftingContext());
        integer(packer, "memories_level", surface.memoriesLevel());
        bool(
            packer,
            "knowledge_available",
            surface.knowledgeAvailable()
        );
        bool(
            packer,
            "bench_context_available",
            surface.benchContextAvailable()
        );
        bool(packer, "manager_available", surface.managerAvailable());
        bool(packer, "manager_has_bench", surface.managerHasBench());
        integer(packer, "manager_queue_size", surface.managerQueueSize());
        string(
            packer,
            "manager_queue_recipe_id",
            surface.managerQueueRecipeId()
        );
        string(
            packer,
            "manager_queue_identity_sha256",
            surface.managerQueueIdentitySha256()
        );
        integer(packer, "source_count", surface.sourceCount());
        integer(packer, "emitted_count", surface.emittedCount());
        bool(packer, "capacity_exceeded", surface.capacityExceeded());
        packer.packString("candidates");
        packer.packArrayHeader(surface.candidates().size());
        for (PolicyWorldActionRecipeSurface.Candidate candidate
            : surface.candidates()) {
            packer.packMapHeader(10);
            integer(packer, "slot", candidate.slot());
            integer(
                packer,
                "native_recipe_index",
                candidate.nativeRecipeIndex()
            );
            string(packer, "recipe_id", candidate.recipeId());
            packBinary(
                packer,
                "recipe_id_sha256",
                candidate.recipeIdSha256()
            );
            packer.packString("recipe");
            packCraftingRecipe(packer, candidate.recipe());
            bool(
                packer,
                "knowledge_satisfied",
                candidate.knowledgeSatisfied()
            );
            bool(
                packer,
                "memory_satisfied",
                candidate.memorySatisfied()
            );
            bool(
                packer,
                "bench_satisfied",
                candidate.benchSatisfied()
            );
            string(packer, "knowledge_key", candidate.knowledgeKey());
            string(
                packer,
                "candidate_semantic_sha256",
                com.hytalerlbridge.worldgen.policyactions.capture.semantic
                    .PolicyWorldActionSemanticHash.recipeCandidateIdentity(
                        recipeTableIdentity,
                        surface.craftingContext(),
                        candidate
                    )
            );
        }
    }

    private static void packContext(
        MessagePacker packer,
        PolicyWorldActionRecipeSurface.CraftingContext context
    ) throws IOException {
        if (context == null) {
            packer.packNil();
            return;
        }
        if (context.kind().equals("fieldcraft")) {
            packer.packMapHeader(1);
            string(packer, "kind", "fieldcraft");
            return;
        }
        packer.packMapHeader(6);
        string(packer, "kind", "bench");
        ints(packer, "position", context.position());
        string(packer, "block_id", context.blockId());
        string(packer, "bench_id", context.benchId());
        integer(packer, "bench_type", context.benchType());
        integer(packer, "tier", context.tier());
    }

    private static void packUse(
        MessagePacker packer,
        PolicyWorldActionUseSurface use
    ) throws IOException {
        packer.packMapHeader(3);
        bool(packer, "available", use.available());
        string(packer, "unavailable_reason", use.unavailableReason());
        packer.packString("binding");
        nullableBinding(packer, use.binding());
    }

    private static void packSelection(
        MessagePacker packer,
        PolicyWorldActionSelection selection
    ) throws IOException {
        if (selection == null) {
            packer.packNil();
            return;
        }
        packer.packMapHeader(4);
        bool(packer, "use_requested", selection.useRequested());
        integer(
            packer,
            "block_interaction_trigger",
            selection.blockInteractionTrigger()
        );
        integer(
            packer,
            "block_candidate_index",
            selection.blockCandidateIndex()
        );
        integer(
            packer,
            "recipe_candidate_index",
            selection.recipeCandidateIndex()
        );
    }

    private static void nullableBinding(
        MessagePacker packer,
        PolicyWorldActionBinding binding
    ) throws IOException {
        if (binding == null) {
            packer.packNil();
            return;
        }
        packer.packMapHeader(18);
        string(packer, "verb", binding.verb());
        integer(packer, "interaction_type", binding.interactionType());
        string(packer, "interaction_id", binding.interactionId());
        string(
            packer,
            "block_interaction_id",
            binding.blockInteractionId()
        );
        ints(packer, "target", binding.target());
        integer(packer, "block_face", binding.blockFace());
        bool(
            packer,
            "rotation_applicable",
            binding.rotationApplicable()
        );
        ints(packer, "rotation", binding.rotation());
        string(packer, "source_container", binding.sourceContainer());
        string(
            packer,
            "resolved_source_container",
            binding.resolvedSourceContainer()
        );
        integer(packer, "source_slot", binding.sourceSlot());
        integer(packer, "source_quantity", binding.sourceQuantity());
        string(packer, "source_item_id", binding.sourceItemId());
        string(packer, "source_block_id", binding.sourceBlockId());
        string(
            packer,
            "expected_interaction_tool_id",
            binding.expectedInteractionToolId()
        );
        string(packer, "expected_block_id", binding.expectedBlockId());
        packBinary(
            packer,
            "expected_semantic_key_sha256",
            binding.expectedSemanticKeySha256()
        );
        string(
            packer,
            "binding_semantic_sha256",
            com.hytalerlbridge.worldgen.policyactions.capture.semantic
                .PolicyWorldActionSemanticHash.bindingIdentity(binding)
        );
    }

    private static void string(
        MessagePacker packer,
        String name,
        String value
    ) throws IOException {
        packer.packString(name);
        packer.packString(value);
    }

    private static void integer(
        MessagePacker packer,
        String name,
        int value
    ) throws IOException {
        packer.packString(name);
        packer.packInt(value);
    }

    private static void longValue(
        MessagePacker packer,
        String name,
        long value
    ) throws IOException {
        packer.packString(name);
        packer.packLong(value);
    }

    private static void bool(
        MessagePacker packer,
        String name,
        boolean value
    ) throws IOException {
        packer.packString(name);
        packer.packBoolean(value);
    }

    private static void doubleValue(
        MessagePacker packer,
        String name,
        double value
    ) throws IOException {
        packer.packString(name);
        packer.packDouble(value);
    }

    private static void ints(
        MessagePacker packer,
        String name,
        int[] values
    ) throws IOException {
        packer.packString(name);
        packer.packArrayHeader(values.length);
        for (int value : values) packer.packInt(value);
    }

    private static void doubles(
        MessagePacker packer,
        String name,
        double[] values
    ) throws IOException {
        packer.packString(name);
        packer.packArrayHeader(values.length);
        for (double value : values) packer.packDouble(value);
    }
}
