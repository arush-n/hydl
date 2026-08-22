package com.hytalerlbridge.nativebackend.worldactions.capture.provider;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.BlockPosition;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.entity.InteractionContext;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.nativebackend.NativeBlockUse;
import com.hytalerlbridge.nativebackend.worldactions.capture.NativeBlockRayFace;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence;
import com.hytalerlbridge.worldgen.NativeMutableBlockCells;
import com.hytalerlbridge.worldgen.NativeMutableBlockEvidence;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionBinding;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionBlockSurface;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionCamera;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionUseSurface;
import java.util.List;
import java.util.function.Function;
import org.joml.Vector3d;
import org.joml.Vector3i;

/** Captures the exact current camera-hit Use/Break surface. */
public final class NativePolicyBlockSurfaceProvider {

    private NativePolicyBlockSurfaceProvider() {}

    public static Result capture(Context context) {
        if (
            context.actor() == null
                || !context.actor().isValid()
                || context.store() == null
                || context.world() == null
                || context.npc() == null
                || context.manager() == null
        ) {
            return Result.unavailable("controlled_actor_unavailable");
        }
        InteractionContext useContext = InteractionContext.forInteraction(
            context.manager(),
            context.actor(),
            InteractionType.Use,
            context.store()
        );
        NativeBlockUse.CameraTarget cameraTarget =
            NativeBlockUse.inspectCamera(
                context.actor(),
                context.store(),
                context.world(),
                useContext.getOriginalItemType(),
                context.desiredPitch(),
                context.desiredYaw()
            );
        if (!cameraTarget.available()) {
            return Result.unavailable(cameraTarget.rejectReason());
        }
        BlockPosition raw = cameraTarget.rawTarget();
        BlockPosition action = cameraTarget.target();
        int face = NativeBlockRayFace.faceForTarget(
            cameraTarget.eyePosition(),
            cameraTarget.direction(),
            cameraTarget.maximumDistance(),
            new Vector3i(raw.x, raw.y, raw.z)
        );
        if (face == 0) {
            return Result.unavailable("camera_block_entry_face_ambiguous");
        }
        PolicyWorldActionCamera camera = new PolicyWorldActionCamera(
            true,
            "",
            vector(cameraTarget.actorPosition()),
            vector(cameraTarget.eyePosition()),
            vector(cameraTarget.direction()),
            cameraTarget.maximumDistance(),
            cell(raw),
            cell(action),
            face
        );

        NativeMutableBlockCells cells = context.mutableCellsCapture().apply(
            cell(action)
        );
        NativeMutableBlockEvidence.Row semantics = exactRow(cells, action);
        NativeItemInteractionEvidence interactions =
            NativeItemInteractionEvidence.capture(
                context.serverVersion(),
                context.worldName(),
                context.worldgenProvider(),
                context.worldgenVersion(),
                context.seed(),
                context.actor(),
                context.store(),
                context.manager(),
                0
            );
        if (semantics == null) {
            return new Result(
                camera,
                PolicyWorldActionBlockSurface.unavailable(
                    "camera_action_cell_semantics_unavailable",
                    1
                ),
                PolicyWorldActionUseSurface.unavailable(
                    "camera_action_cell_semantics_unavailable"
                ),
                cells,
                interactions
            );
        }
        if (!context.allowHeadlessPlayerContext()) {
            return new Result(
                camera,
                PolicyWorldActionBlockSurface.unavailable(
                    "native_world_verb_execution_context_unavailable",
                    1
                ),
                PolicyWorldActionUseSurface.unavailable(
                    "native_world_verb_execution_context_unavailable"
                ),
                cells,
                interactions
            );
        }

        PolicyWorldActionBinding breakBinding = breakBinding(
            context,
            interactions,
            semantics,
            action,
            face
        );
        PolicyWorldActionBlockSurface blocks = breakBinding == null
            ? PolicyWorldActionBlockSurface.unavailable(
                breakUnavailableReason(context, interactions),
                1
            )
            : new PolicyWorldActionBlockSurface(
                true,
                "",
                1,
                1,
                false,
                "",
                "rotation_zero_destination_geometry_unavailable",
                List.of(new PolicyWorldActionBlockSurface.Candidate(
                    0,
                    cell(raw),
                    cell(action),
                    relative(raw, cameraTarget.actorPosition()),
                    semantics,
                    breakBinding,
                    null
                ))
            );

        NativeBlockUse.Availability useAvailability = NativeBlockUse.inspect(
            context.actor(),
            context.store(),
            context.world(),
            context.manager(),
            context.desiredPitch(),
            context.desiredYaw(),
            context.allowHeadlessPlayerContext()
        );
        PolicyWorldActionBinding useBinding = useBinding(
            context,
            useAvailability,
            semantics,
            cameraTarget,
            face
        );
        PolicyWorldActionUseSurface use = useBinding == null
            ? PolicyWorldActionUseSurface.unavailable(
                useAvailability.available()
                    ? "use_capture_identity_mismatch"
                    : useAvailability.rejectReason()
            )
            : new PolicyWorldActionUseSurface(true, "", useBinding);
        return new Result(camera, blocks, use, cells, interactions);
    }

    private static PolicyWorldActionBinding breakBinding(
        Context context,
        NativeItemInteractionEvidence evidence,
        NativeMutableBlockEvidence.Row semantics,
        BlockPosition target,
        int face
    ) {
        InteractionContext primaryContext = InteractionContext.forInteraction(
            context.manager(),
            context.actor(),
            InteractionType.Primary,
            context.store()
        );
        ItemStack source = primaryContext.getHeldItem();
        int sourceSlot = Byte.toUnsignedInt(primaryContext.getHeldItemSlot());
        if (
            primaryContext.getHeldItemSectionId() != -1
                || ItemStack.isEmpty(source)
                || sourceSlot < 0
        ) {
            return null;
        }
        NativeItemInteractionEvidence.Trigger trigger = evidence.triggers().get(
            InteractionType.Primary.getValue()
        );
        if (!trigger.rootPresent() || trigger.root() == null) return null;
        List<NativeItemInteractionEvidence.Node> breaking = trigger.nodes()
            .stream()
            .filter(node -> node.payload()
                instanceof NativeItemInteractionEvidence.BreakBlockPayload)
            .toList();
        if (breaking.size() != 1) return null;
        NativeItemInteractionEvidence.Node node = breaking.get(0);
        NativeItemInteractionEvidence.BreakBlockPayload payload =
            (NativeItemInteractionEvidence.BreakBlockPayload) node.payload();
        if (payload.toolId().isBlank()) return null;
        String sourceBlockId = source.getBlockKey();
        return new PolicyWorldActionBinding(
            "break_block",
            InteractionType.Primary.getValue(),
            trigger.root().rootId(),
            node.interactionId(),
            cell(target),
            face,
            false,
            new int[] {0, 0, 0},
            "hotbar",
            "hotbar",
            sourceSlot,
            source.getQuantity(),
            source.getItemId(),
            sourceBlockId == null ? "" : sourceBlockId,
            payload.toolId(),
            semantics.blockAssetId(),
            semantics.semanticKeySha256()
        );
    }

    private static String breakUnavailableReason(
        Context context,
        NativeItemInteractionEvidence evidence
    ) {
        InteractionContext primary = InteractionContext.forInteraction(
            context.manager(),
            context.actor(),
            InteractionType.Primary,
            context.store()
        );
        if (primary.getHeldItemSectionId() != -1) {
            return "break_source_container_not_hotbar_v1";
        }
        if (ItemStack.isEmpty(primary.getHeldItem())) {
            return "break_source_item_unavailable";
        }
        NativeItemInteractionEvidence.Trigger trigger = evidence.triggers().get(
            InteractionType.Primary.getValue()
        );
        if (!trigger.rootPresent()) return "break_primary_root_unavailable";
        long breakNodes = trigger.nodes().stream().filter(
            node -> node.payload()
                instanceof NativeItemInteractionEvidence.BreakBlockPayload
        ).count();
        if (breakNodes != 1L) return "break_program_not_uniquely_bound";
        return "break_authored_tool_unavailable";
    }

    private static PolicyWorldActionBinding useBinding(
        Context context,
        NativeBlockUse.Availability availability,
        NativeMutableBlockEvidence.Row semantics,
        NativeBlockUse.CameraTarget camera,
        int face
    ) {
        if (!availability.available()) return null;
        if (
            !same(availability.rawTarget(), camera.rawTarget())
                || !same(availability.target(), camera.target())
                || availability.maximumDistance() != camera.maximumDistance()
                || !availability.actorPosition().equals(camera.actorPosition())
                || !availability.eyePosition().equals(camera.eyePosition())
                || !availability.direction().equals(camera.direction())
        ) {
            return null;
        }
        boolean unarmed = availability.sourceContainer().equals("unarmed");
        String resolvedContainer = resolvedUseContainer(context.npc());
        if (
            resolvedContainer == null
                || !useInventoryMatches(
                    context.npc(),
                    availability,
                    unarmed,
                    resolvedContainer
                )
        ) {
            return null;
        }
        return new PolicyWorldActionBinding(
            "use",
            InteractionType.Use.getValue(),
            availability.interactionId(),
            availability.blockInteractionId(),
            cell(availability.target()),
            face,
            false,
            new int[] {0, 0, 0},
            availability.sourceContainer(),
            resolvedContainer,
            availability.sourceSlot(),
            availability.sourceQuantity(),
            availability.itemId(),
            "",
            "",
            semantics.blockAssetId(),
            semantics.semanticKeySha256()
        );
    }

    private static boolean useInventoryMatches(
        NPCEntity npc,
        NativeBlockUse.Availability availability,
        boolean unarmed,
        String resolvedContainer
    ) {
        if (npc.getInventory() == null) return false;
        ItemContainer container;
        int active;
        if (resolvedContainer.equals("tools")) {
            container = npc.getInventory().getTools();
            active = Byte.toUnsignedInt(
                npc.getInventory().getActiveToolsSlot()
            );
        } else if (resolvedContainer.equals("hotbar")) {
            container = npc.getInventory().getHotbar();
            active = Byte.toUnsignedInt(
                npc.getInventory().getActiveHotbarSlot()
            );
        } else {
            return false;
        }
        if (container == null) return false;
        int capacity = Short.toUnsignedInt(
            container.getCapacity()
        );
        if (active >= capacity) return false;
        ItemStack stack = container.getItemStack((short) active);
        if (unarmed) return ItemStack.isEmpty(stack);
        return availability.sourceSlot() == active
            && !ItemStack.isEmpty(stack)
            && availability.itemId().equals(stack.getItemId())
            && availability.sourceQuantity() == stack.getQuantity();
    }

    private static String resolvedUseContainer(NPCEntity npc) {
        if (npc == null || npc.getInventory() == null) return null;
        return npc.getInventory().usingToolsItem() ? "tools" : "hotbar";
    }

    private static NativeMutableBlockEvidence.Row exactRow(
        NativeMutableBlockCells cells,
        BlockPosition target
    ) {
        if (cells == null || cells.cells().size() != 1) return null;
        NativeMutableBlockCells.Cell cell = cells.cells().get(0);
        int[] position = cell.position();
        if (
            position[0] != target.x
                || position[1] != target.y
                || position[2] != target.z
                || !cell.available()
        ) {
            return null;
        }
        NativeMutableBlockEvidence.Row row = cell.row();
        return row != null
                && row.blockPresent()
                && row.semanticKeyValid()
                && row.affordanceValid()
            ? row
            : null;
    }

    private static boolean same(BlockPosition left, BlockPosition right) {
        return left != null && right != null
            && left.x == right.x && left.y == right.y && left.z == right.z;
    }

    private static int[] cell(BlockPosition value) {
        return new int[] {value.x, value.y, value.z};
    }

    private static double[] vector(Vector3d value) {
        return new double[] {value.x, value.y, value.z};
    }

    private static double[] relative(BlockPosition cell, Vector3d actor) {
        return new double[] {
            cell.x + 0.5 - actor.x,
            cell.y + 0.5 - actor.y,
            cell.z + 0.5 - actor.z
        };
    }

    public record Context(
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        World world,
        NPCEntity npc,
        InteractionManager manager,
        float desiredPitch,
        float desiredYaw,
        boolean allowHeadlessPlayerContext,
        Function<int[], NativeMutableBlockCells> mutableCellsCapture
    ) {
        public Context {
            if (
                serverVersion == null || serverVersion.isBlank()
                    || worldName == null || worldName.isBlank()
                    || worldgenProvider == null || worldgenProvider.isBlank()
                    || worldgenVersion == null || worldgenVersion.isBlank()
                    || mutableCellsCapture == null
            ) {
                throw new IllegalArgumentException(
                    "Block-surface capture context is incomplete"
                );
            }
        }
    }

    public record Result(
        PolicyWorldActionCamera camera,
        PolicyWorldActionBlockSurface blocks,
        PolicyWorldActionUseSurface use,
        NativeMutableBlockCells cells,
        NativeItemInteractionEvidence interactions
    ) {
        static Result unavailable(String reason) {
            return new Result(
                PolicyWorldActionCamera.unavailable(reason),
                PolicyWorldActionBlockSurface.unavailable(reason, 0),
                PolicyWorldActionUseSurface.unavailable(reason),
                null,
                null
            );
        }
    }
}
