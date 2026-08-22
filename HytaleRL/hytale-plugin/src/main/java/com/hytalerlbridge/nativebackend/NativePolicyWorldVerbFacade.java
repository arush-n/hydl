package com.hytalerlbridge.nativebackend;

import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.sameEntity;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.interactionManager;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.canonicalWorldVerbInventoryState;

import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.BlockFace;
import com.hypixel.hytale.protocol.BlockPosition;
import com.hypixel.hytale.protocol.BlockRotation;
import com.hypixel.hytale.protocol.InteractionState;
import com.hypixel.hytale.protocol.InteractionSyncData;
import com.hypixel.hytale.protocol.Rotation;
import com.hypixel.hytale.protocol.WaitForDataFrom;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionEntry;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.inventory.Inventory;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.ChainingInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.operation.Operation;
import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.action.NativeWorldVerbRequest;
import com.hytalerlbridge.nativebackend.model.WorldVerbCell;
import org.joml.Vector3i;

/**
 * Public, reflection-safe native World-verb admission for autonomous NPCs.
 *
 * <p>The facade retains the originating actor's stable ECS identity only in
 * the caller-owned opaque execution handle and owns no global lifecycle map.
 * A caller installs the bridge's exact headless Adventure context, calls
 * {@link #start}, and retains the returned opaque handle. The caller then
 * invokes {@link #prepare} before Hytale's interaction-manager tick and
 * {@link #poll} after it. Both lifecycle calls require the same ECS store and
 * entity index as {@code start}; refreshed {@link Ref} wrappers are accepted.
 * Native sessions and this facade share the same
 * {@code NativeWorldVerbUse/Craft/Block} admission primitives.
 *
 * <p>Every request is revalidated against current inventory, block identity,
 * authored interaction roots, and the caller-owned world epoch at commit time.
 * The policy mod may access this class reflectively without compiling against
 * the bridge jar.
 */
public final class NativePolicyWorldVerbFacade {

    public static final String SCHEMA =
        "hytalerl_native_policy_world_verb_facade_v3";
    public static final int VERSION = 3;

    private NativePolicyWorldVerbFacade() {}

    public static String schema() {
        return SCHEMA;
    }

    public static int version() {
        return VERSION;
    }

    /** Install the exact bridge-owned synthetic Player context. */
    public static ContextResult installContext(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        PlayerRef packetAnchor
    ) {
        if (actor == null || !actor.isValid() || store == null
            || packetAnchor == null) {
            return ContextResult.rejectedResult("invalid_context_request");
        }
        if (NativeHeadlessWorldVerbContext.installed(actor, store)) {
            return ContextResult.acceptedResult();
        }
        try {
            NativeHeadlessWorldVerbContext.install(actor, store, packetAnchor);
            return NativeHeadlessWorldVerbContext.installed(actor, store)
                ? ContextResult.acceptedResult()
                : ContextResult.rejectedResult("headless_context_not_installed");
        } catch (RuntimeException unavailable) {
            return ContextResult.rejectedResult(
                "headless_context_install_failed");
        }
    }

    /** Remove only the bridge-owned synthetic context, never a real player. */
    public static ContextResult uninstallContext(
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        if (actor == null || !actor.isValid() || store == null) {
            return ContextResult.rejectedResult("invalid_context_request");
        }
        if (!NativeHeadlessWorldVerbContext.installed(actor, store)) {
            return ContextResult.rejectedResult(
                "headless_context_not_installed");
        }
        try {
            NativeHeadlessWorldVerbContext.uninstall(actor, store);
            return ContextResult.acceptedResult();
        } catch (RuntimeException unavailable) {
            return ContextResult.rejectedResult(
                "headless_context_uninstall_failed");
        }
    }

    /** Commit one exact typed request and return caller-owned lifecycle state. */
    public static Admission start(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        NativeWorldVerbRequest request,
        String expectedWorldEpoch
    ) {
        if (request == null || !request.present()) {
            return Admission.rejected("native_world_verb_request_absent");
        }
        if (!request.candidateEvidenceBound()) {
            return Admission.rejected(
                NativeWorldVerbRequest.POLICY_EVIDENCE_REQUIRED
            );
        }
        if (expectedWorldEpoch == null
            || !request.worldEpoch().equals(expectedWorldEpoch)) {
            return Admission.rejected("stale_world_epoch");
        }
        if (actor == null || !actor.isValid() || store == null) {
            return Admission.rejected("controlled_entity_unavailable");
        }
        NPCEntity npc = store.getComponent(
            actor, NPCEntity.getComponentType());
        World world = store.getExternalData().getWorld();
        if (npc == null || world == null) {
            return Admission.rejected("controlled_entity_unavailable");
        }
        boolean headless = NativeHeadlessWorldVerbContext.installed(
            actor, store);
        long tick = world.getTick();
        return switch (request.verb()) {
            case "use" -> startUse(
                actor, npc, store, world, request, headless, tick);
            case "craft_recipe" -> startCraft(
                actor, store, world, request, headless, tick);
            case "place_block", "break_block" -> startBlock(
                actor, npc, store, world, request, headless, tick);
            default -> Admission.rejected(
                "typed_" + request.verb() + "_execution_not_implemented");
        };
    }

    /** Supply client-wait operation data before InteractionManager ticks. */
    public static Lifecycle prepare(
        Object opaqueHandle,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        float deltaTime
    ) {
        Execution execution = handle(opaqueHandle);
        if (execution == null) {
            return Lifecycle.rejected("native_world_verb_handle_invalid");
        }
        return execution.prepare(actor, store, deltaTime);
    }

    /** Observe start/finish/failure and same-tick world/inventory deltas. */
    public static Lifecycle poll(
        Object opaqueHandle,
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        Execution execution = handle(opaqueHandle);
        if (execution == null) {
            return Lifecycle.rejected("native_world_verb_handle_invalid");
        }
        return execution.poll(actor, store);
    }

    private static Admission startUse(
        Ref<EntityStore> actor,
        NPCEntity npc,
        Store<EntityStore> store,
        World world,
        NativeWorldVerbRequest request,
        boolean headless,
        long tick
    ) {
        NativeWorldVerbUse.StartResult result = NativeWorldVerbUse.start(
            actor, npc, store, world, request, headless);
        if (!result.accepted()) {
            return Admission.rejected(result.rejectReason());
        }
        ItemStack source = result.resolvedSource();
        String inventoryBefore = result.unarmed()
            ? "unarmed"
            : canonicalWorldVerbInventoryState(source);
        String resolvedSourceContainer = result.unarmed()
            ? "unarmed"
            : npc.getInventory() != null && npc.getInventory().usingToolsItem()
                ? "tools"
                : "hotbar";
        Execution execution = Execution.interaction(
            actor,
            request,
            world,
            result.chain(),
            result.before(),
            inventoryBefore,
            resolvedSourceContainer,
            result.before().runtimeBlockId(),
            tick
        );
        return Admission.accepted(execution);
    }

    private static Admission startCraft(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        World world,
        NativeWorldVerbRequest request,
        boolean headless,
        long tick
    ) {
        NativeWorldVerbCraft.StartResult result = NativeWorldVerbCraft.start(
            actor, store, world, request, headless);
        if (!result.accepted()) {
            return Admission.rejected(result.rejectReason());
        }
        Execution execution = Execution.craft(
            actor,
            request,
            world,
            result.manager(),
            result.recipe(),
            result.inventoryBefore(),
            result.inventoryAfter(),
            result.outputQuantityBefore(),
            result.outputQuantityAfter(),
            result.queued(),
            tick
        );
        return Admission.accepted(execution);
    }

    private static Admission startBlock(
        Ref<EntityStore> actor,
        NPCEntity npc,
        Store<EntityStore> store,
        World world,
        NativeWorldVerbRequest request,
        boolean headless,
        long tick
    ) {
        Vector3i target = new Vector3i(
            request.targetX(), request.targetY(), request.targetZ());
        NativeWorldVerbBlock.StartResult result = NativeWorldVerbBlock.start(
            actor,
            npc,
            store,
            world,
            request,
            headless,
            new NativeWorldVerbBlock.Plan(
                false,
                target,
                target,
                request.interactionId(),
                request.itemId(),
                request.sourceSlot(),
                request.interactionType(),
                request.expectedBlockId(),
                request.blockId()
            )
        );
        if (!result.accepted()) {
            return Admission.rejected(result.rejectReason());
        }
        Execution execution = Execution.interaction(
            actor,
            request,
            world,
            result.chain(),
            result.before(),
            canonicalWorldVerbInventoryState(result.source()),
            "hotbar",
            result.placedRuntimeBlockId(),
            tick
        );
        return Admission.accepted(execution);
    }

    private static Execution handle(Object value) {
        return value instanceof Execution execution ? execution : null;
    }

    static boolean sameActorBinding(
        Ref<EntityStore> originatingActor,
        Ref<EntityStore> callerActor
    ) {
        return sameEntity(originatingActor, callerActor);
    }

    public record ContextResult(boolean accepted, String rejectReason) {
        private static ContextResult acceptedResult() {
            return new ContextResult(true, "");
        }

        private static ContextResult rejectedResult(String reason) {
            return new ContextResult(false, reason);
        }
    }

    public record Admission(
        boolean accepted,
        String rejectReason,
        Object handle,
        Lifecycle lifecycle
    ) {
        private static Admission accepted(Execution execution) {
            return new Admission(true, "", execution, execution.lifecycle());
        }

        private static Admission rejected(String reason) {
            return new Admission(
                false, reason, null, Lifecycle.rejected(reason));
        }
    }

    /** Immutable lifecycle receipt returned on every prepare/poll call. */
    public record Lifecycle(
        boolean accepted,
        boolean pending,
        boolean started,
        boolean finished,
        boolean failed,
        long requestedTick,
        long startedTick,
        long finishedTick,
        boolean geometryChanged,
        boolean inventoryChanged,
        String blockIdBefore,
        String blockIdAfter,
        int sourceQuantityBefore,
        int sourceQuantityAfter,
        String rejectReason
    ) {
        private static Lifecycle rejected(String reason) {
            return new Lifecycle(
                false, false, false, false, true,
                -1L, -1L, -1L, false, false,
                "", "", -1, -1, reason
            );
        }
    }

    /** Mutable only through the caller-held opaque instance. */
    private static final class Execution {
        private final Ref<EntityStore> originatingActor;
        private final NativeWorldVerbRequest request;
        private final World world;
        private final InteractionChain chain;
        private final CraftingManager crafting;
        private final CraftingRecipe recipe;
        private final String geometryBefore;
        private final String inventoryBefore;
        private final String resolvedSourceContainer;
        private final String blockIdBefore;
        private final int sourceQuantityBefore;
        private final int placedRuntimeBlockId;
        private final long requestedTick;
        private int operationIndex = -1;
        private float clientElapsedSeconds;
        private boolean started;
        private boolean finished;
        private boolean failed;
        private long startedTick = -1L;
        private long finishedTick = -1L;
        private boolean geometryChanged;
        private boolean inventoryChanged;
        private String blockIdAfter = "";
        private int sourceQuantityAfter = -1;
        private String rejectReason = "";

        private Execution(
            Ref<EntityStore> originatingActor,
            NativeWorldVerbRequest request,
            World world,
            InteractionChain chain,
            CraftingManager crafting,
            CraftingRecipe recipe,
            String geometryBefore,
            String inventoryBefore,
            String resolvedSourceContainer,
            String blockIdBefore,
            int sourceQuantityBefore,
            int placedRuntimeBlockId,
            long requestedTick
        ) {
            this.originatingActor = originatingActor;
            this.request = request;
            this.world = world;
            this.chain = chain;
            this.crafting = crafting;
            this.recipe = recipe;
            this.geometryBefore = geometryBefore;
            this.inventoryBefore = inventoryBefore;
            this.resolvedSourceContainer = resolvedSourceContainer;
            this.blockIdBefore = blockIdBefore;
            this.sourceQuantityBefore = sourceQuantityBefore;
            this.placedRuntimeBlockId = placedRuntimeBlockId;
            this.requestedTick = requestedTick;
        }

        static Execution interaction(
            Ref<EntityStore> originatingActor,
            NativeWorldVerbRequest request,
            World world,
            InteractionChain chain,
            WorldVerbCell before,
            String inventoryBefore,
            String resolvedSourceContainer,
            int placedRuntimeBlockId,
            long tick
        ) {
            int quantity = request.sourceContainer().equals("unarmed")
                ? 0
                : request.expectedSourceQuantity();
            return new Execution(
                originatingActor,
                request,
                world,
                chain,
                null,
                null,
                before.canonical(),
                inventoryBefore,
                resolvedSourceContainer,
                before.semanticBlockId(),
                quantity,
                placedRuntimeBlockId,
                tick
            );
        }

        static Execution craft(
            Ref<EntityStore> originatingActor,
            NativeWorldVerbRequest request,
            World world,
            CraftingManager crafting,
            CraftingRecipe recipe,
            String inventoryBefore,
            String inventoryAfter,
            int quantityBefore,
            int quantityAfter,
            boolean queued,
            long tick
        ) {
            Execution result = new Execution(
                originatingActor,
                request,
                world,
                null,
                crafting,
                recipe,
                "craft_recipe",
                inventoryBefore,
                "player_inventory",
                "not_applicable",
                quantityBefore,
                0,
                tick
            );
            if (!queued) {
                result.started = true;
                result.finished = true;
                result.startedTick = tick;
                result.finishedTick = tick;
                result.inventoryChanged = !inventoryBefore.equals(
                    inventoryAfter);
                result.blockIdAfter = "not_applicable";
                result.sourceQuantityAfter = quantityAfter;
            }
            return result;
        }

        synchronized Lifecycle prepare(
            Ref<EntityStore> actor,
            Store<EntityStore> store,
            float deltaTime
        ) {
            if (!validActorWorld(actor, store)) {
                fail("native_world_verb_actor_or_world_changed");
                return lifecycle();
            }
            if (finished || chain == null) return lifecycle();
            if (!Float.isFinite(deltaTime) || deltaTime < 0.0f) {
                fail("native_world_verb_delta_time_invalid");
                return lifecycle();
            }
            if (chain.getServerState() != InteractionState.NotFinished) {
                return lifecycle();
            }
            RootInteraction root = chain.getRootInteraction();
            int operationCounter = chain.getOperationCounter();
            if (root == null || operationCounter < 0
                || operationCounter >= root.getOperationMax()) {
                return lifecycle();
            }
            Operation operation = root.getOperation(operationCounter);
            if (operation == null
                || operation.getWaitForDataFrom() != WaitForDataFrom.Client) {
                return lifecycle();
            }
            int currentOperationIndex = chain.getOperationIndex();
            if (currentOperationIndex != operationIndex) {
                operationIndex = currentOperationIndex;
                clientElapsedSeconds = 0.0f;
            }
            Operation inner = operation.getInnerOperation();
            float runTime = inner instanceof Interaction interaction
                ? Math.max(0.0f, interaction.getRunTime())
                : 0.0f;
            boolean operationFinished = clientElapsedSeconds >= runTime;
            InteractionSyncData data = new InteractionSyncData();
            data.state = operationFinished
                ? InteractionState.Finished
                : InteractionState.NotFinished;
            data.progress = clientElapsedSeconds;
            data.operationCounter = operationCounter;
            data.rootInteraction =
                RootInteraction.getRootInteractionIdOrUnknown(root.getId());
            data.blockPosition = new BlockPosition(
                request.targetX(), request.targetY(), request.targetZ());
            data.blockFace = BlockFace.fromValue(request.blockFace());
            data.blockRotation = new BlockRotation(
                Rotation.fromValue(request.rotationYaw()),
                Rotation.fromValue(request.rotationPitch()),
                Rotation.fromValue(request.rotationRoll())
            );
            data.placedBlockId = request.verb().equals("place_block")
                ? placedRuntimeBlockId
                : world.getBlock(
                    request.targetX(), request.targetY(), request.targetZ());
            if (inner instanceof ChainingInteraction) data.chainingIndex = 0;
            InteractionEntry entry = chain.getInteraction(currentOperationIndex);
            if (entry == null) {
                chain.putInteractionSyncData(currentOperationIndex, data);
            } else if (!entry.setClientState(data)) {
                fail("native_world_verb_client_sync_rejected");
                return lifecycle();
            }
            if (operationFinished && chain.getCallDepth() == 0
                && operationCounter + 1 >= root.getOperationMax()) {
                chain.setClientState(InteractionState.Finished);
            }
            clientElapsedSeconds += deltaTime;
            return lifecycle();
        }

        synchronized Lifecycle poll(
            Ref<EntityStore> actor,
            Store<EntityStore> store
        ) {
            if (!validActorWorld(actor, store)) {
                fail("native_world_verb_actor_or_world_changed");
                return lifecycle();
            }
            if (finished) return lifecycle();
            long tick = world.getTick();
            if (crafting != null) {
                if (!started) {
                    started = true;
                    startedTick = tick;
                }
                if (crafting.getRemainingQueueSize() != 0) {
                    return lifecycle();
                }
                if (recipe == null) {
                    fail("native_craft_recipe_lifecycle_unavailable");
                    return lifecycle();
                }
                String after = NativeHeadlessCrafting.canonicalInventory(
                    actor, store);
                sourceQuantityAfter = NativeHeadlessCrafting.outputQuantity(
                    actor, store, recipe);
                inventoryChanged = !inventoryBefore.equals(after);
                blockIdAfter = "not_applicable";
                finish(false, tick);
                if (crafting.hasBenchSet()) crafting.clearBench(actor, store);
                return lifecycle();
            }

            InteractionManager manager = interactionManager(actor, store);
            boolean registered = manager != null
                && manager.getChains().values().contains(chain);
            InteractionState state = chain.getServerState();
            boolean nowStarted = registered || chain.getChainId() < 0
                || state != InteractionState.NotFinished;
            if (nowStarted && !started) {
                started = true;
                startedTick = tick;
            }
            boolean nowFinished = nowStarted
                && (state != InteractionState.NotFinished || !registered);
            if (!nowFinished) return lifecycle();

            WorldVerbCell after = WorldVerbCapture.capture(world, request);
            if (after == null) {
                fail("native_world_verb_acknowledgement_unavailable");
                return lifecycle();
            }
            String inventoryAfter = sourceInventory(actor, store);
            blockIdAfter = after.semanticBlockId();
            geometryChanged = !geometryBefore.equals(after.canonical());
            inventoryChanged = !inventoryBefore.equals(inventoryAfter);
            finish(state == InteractionState.Failed, tick);
            return lifecycle();
        }

        private String sourceInventory(
            Ref<EntityStore> actor,
            Store<EntityStore> store
        ) {
            if (request.sourceContainer().equals("unarmed")) {
                sourceQuantityAfter = 0;
                return "unarmed";
            }
            NPCEntity npc = store.getComponent(
                actor, NPCEntity.getComponentType());
            Inventory inventory = npc == null ? null : npc.getInventory();
            ItemContainer source = inventory == null
                ? null
                : resolvedSourceContainer.equals("tools")
                    ? inventory.getTools()
                    : inventory.getHotbar();
            ItemStack stack = source == null
                || request.sourceSlot() < 0
                || request.sourceSlot()
                    >= Short.toUnsignedInt(source.getCapacity())
                ? null
                : source.getItemStack((short) request.sourceSlot());
            sourceQuantityAfter = ItemStack.isEmpty(stack)
                ? 0
                : stack.getQuantity();
            return canonicalWorldVerbInventoryState(stack);
        }

        private boolean validActorWorld(
            Ref<EntityStore> actor,
            Store<EntityStore> store
        ) {
            return sameActorBinding(originatingActor, actor)
                && actor.isValid() && store != null
                && actor.getStore() == store
                && store.getExternalData().getWorld() == world;
        }

        private void finish(boolean failed, long tick) {
            this.finished = true;
            this.failed = failed;
            this.finishedTick = tick;
        }

        private void fail(String reason) {
            if (!started) {
                started = true;
                startedTick = world == null ? -1L : world.getTick();
            }
            finished = true;
            failed = true;
            finishedTick = world == null ? -1L : world.getTick();
            rejectReason = reason;
        }

        synchronized Lifecycle lifecycle() {
            return new Lifecycle(
                true,
                !finished,
                started,
                finished,
                failed,
                requestedTick,
                startedTick,
                finishedTick,
                geometryChanged,
                inventoryChanged,
                blockIdBefore,
                blockIdAfter,
                sourceQuantityBefore,
                sourceQuantityAfter,
                rejectReason
            );
        }
    }
}
