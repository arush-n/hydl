package com.hytalerlbridge.nativebackend.worldactions.capture.provider;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.entity.UUIDComponent;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.observation.NativeInventoryCapture;
import com.hytalerlbridge.observation.NativeInventoryFrame;
import com.hytalerlbridge.worldgen.NativeCraftingCatalogEvidence;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence;
import com.hytalerlbridge.worldgen.NativeMutableBlockCells;
import com.hytalerlbridge.worldgen.policyactions.capture.PolicyWorldActionCaptureRequest;
import com.hytalerlbridge.worldgen.policyactions.capture.model.NativePolicyWorldActionCapture;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionBlockSurface;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionSelection;
import com.hytalerlbridge.worldgen.policyactions.capture.semantic.PolicyWorldActionSemanticHash;
import java.util.Map;
import java.util.function.Function;

/** Orchestrates one complete World-thread observe or commit capture. */
public final class NativePolicyWorldActionCaptureProducer {

    private NativePolicyWorldActionCaptureProducer() {}

    public static NativePolicyWorldActionCapture capture(Context context) {
        requireContext(context);
        UUIDComponent uuid = context.store().getComponent(
            context.actor(),
            UUIDComponent.getComponentType()
        );
        if (uuid == null || uuid.getUuid() == null) {
            throw new IllegalStateException(
                "Policy actor has no authoritative UUID component"
            );
        }
        String actorIdentity = uuid.getUuid().toString();
        if (!actorIdentity.equals(context.request().expectedActorIdentity())) {
            throw new IllegalStateException(
                "Policy actor identity changed before atomic capture"
            );
        }
        NativeInventoryFrame inventory = NativeInventoryCapture.capture(
            context.npc().getInventory(),
            context.itemIds()
        );
        String inventoryIdentity =
            PolicyWorldActionSemanticHash.inventoryIdentity(inventory);

        NativePolicyBlockSurfaceProvider.Result block =
            NativePolicyBlockSurfaceProvider.capture(
                new NativePolicyBlockSurfaceProvider.Context(
                    context.serverVersion(),
                    context.worldName(),
                    context.worldgenProvider(),
                    context.worldgenVersion(),
                    context.seed(),
                    context.actor(),
                    context.store(),
                    context.world(),
                    context.npc(),
                    context.manager(),
                    context.desiredPitch(),
                    context.desiredYaw(),
                    context.allowHeadlessPlayerContext(),
                    context.mutableCellsCapture()
                )
            );
        var recipes = NativePolicyRecipeSurfaceProvider.capture(
            new NativePolicyRecipeSurfaceProvider.Context(
                context.actor(),
                context.store(),
                context.world(),
                context.catalog()
            )
        );
        String generation = PolicyWorldActionSemanticHash.generation(
            context.bridgeSha256(),
            context.worldName(),
            context.worldEpoch(),
            context.actorSlot(),
            actorIdentity,
            inventoryIdentity,
            inventory,
            context.recipeTableIdentitySha256(),
            block.camera(),
            block.blocks(),
            recipes,
            block.use()
        );
        PolicyWorldActionCaptureRequest request = context.request();
        boolean observe = request.phase().equals("observe");
        boolean matched = observe || generation.equalsIgnoreCase(
            request.expectedCandidateGenerationSha256()
        );
        NativeMutableBlockCells commitCells = null;
        NativeItemInteractionEvidence commitInteractions = null;
        if (
            !observe
                && matched
                && selectedBlockEvidenceExists(
                    request.selection(),
                    block
                )
        ) {
            commitCells = block.cells();
            commitInteractions = block.interactions();
            if (commitCells == null || commitInteractions == null) {
                throw new IllegalStateException(
                    "Selected native World action lacks fresh commit evidence"
                );
            }
        }
        return new NativePolicyWorldActionCapture(
            request.phase(),
            context.bridgeSha256(),
            context.worldName(),
            context.worldEpoch(),
            context.world().getTick(),
            context.environmentStep(),
            context.actorSlot(),
            actorIdentity,
            inventoryIdentity,
            context.recipeTableIdentitySha256(),
            generation,
            observe ? "" : request.expectedCandidateGenerationSha256(),
            matched,
            inventory,
            block.camera(),
            block.blocks(),
            recipes,
            block.use(),
            observe ? null : request.selection(),
            commitCells,
            commitInteractions
        );
    }

    private static boolean selectedBlockEvidenceExists(
        PolicyWorldActionSelection selection,
        NativePolicyBlockSurfaceProvider.Result result
    ) {
        if (selection == null) return false;
        if (selection.useRequested()) return result.use().available();
        if (!selection.selectsBlock() || !result.blocks().available()) {
            return false;
        }
        int slot = selection.blockCandidateIndex();
        if (slot < 0 || slot >= result.blocks().candidates().size()) {
            return false;
        }
        PolicyWorldActionBlockSurface.Candidate candidate =
            result.blocks().candidates().get(slot);
        return selection.blockInteractionTrigger() == 1
            ? candidate.primary() != null
            : candidate.secondary() != null;
    }

    private static void requireContext(Context context) {
        if (
            context == null
                || context.bridgeSha256() == null
                || !context.bridgeSha256().matches("[0-9A-Fa-f]{64}")
                || context.serverVersion() == null
                || context.serverVersion().isBlank()
                || context.worldName() == null
                || context.worldName().isBlank()
                || context.worldgenProvider() == null
                || context.worldgenProvider().isBlank()
                || context.worldgenVersion() == null
                || context.worldgenVersion().isBlank()
                || context.worldEpoch() == null
                || context.worldEpoch().isBlank()
                || context.environmentStep() < 0
                || context.actorSlot() < 0
                || context.world() == null
                || context.actor() == null
                || !context.actor().isValid()
                || context.store() == null
                || context.npc() == null
                || context.manager() == null
                || context.itemIds() == null
                || context.catalog() == null
                || context.recipeTableIdentitySha256() == null
                || !context.recipeTableIdentitySha256().matches(
                    "[0-9A-Fa-f]{64}"
                )
                || context.mutableCellsCapture() == null
                || context.request() == null
        ) {
            throw new IllegalArgumentException(
                "Policy World-action capture context is incomplete"
            );
        }
    }

    public record Context(
        String bridgeSha256,
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        String worldEpoch,
        int environmentStep,
        int actorSlot,
        World world,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        NPCEntity npc,
        InteractionManager manager,
        float desiredPitch,
        float desiredYaw,
        boolean allowHeadlessPlayerContext,
        Map<String, Integer> itemIds,
        NativeCraftingCatalogEvidence catalog,
        String recipeTableIdentitySha256,
        Function<int[], NativeMutableBlockCells> mutableCellsCapture,
        PolicyWorldActionCaptureRequest request
    ) {
        public Context {
            itemIds = itemIds == null ? null : Map.copyOf(itemIds);
        }
    }
}
