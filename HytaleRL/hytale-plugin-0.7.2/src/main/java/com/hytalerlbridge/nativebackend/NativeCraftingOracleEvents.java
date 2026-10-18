package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.builtin.crafting.component.BenchBlock;
import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.builtin.crafting.window.SimpleCraftingWindow;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.protocol.packets.window.CraftRecipeAction;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.entity.entities.Player;
import com.hypixel.hytale.server.core.entity.entities.player.windows.WindowManager;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entity.damage.Damage;
import com.hypixel.hytale.server.core.modules.entity.damage.DamageCause;
import com.hypixel.hytale.server.core.modules.entity.damage.DamageSystems;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.CraftRequest;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.ProbeEvent;
import com.hytalerlbridge.nativebackend.NativeCraftingOracle.BenchPlacement;
import com.hytalerlbridge.nativebackend.support.EvidenceCapture;
import com.hytalerlbridge.nativebackend.support.NativeWorldAccess;
import org.joml.Vector3d;

/**
 * Native actions behind a crafting-oracle probe's scripted events. Each one calls the
 * server's own entry point, so the consequences are the server's: a window closing
 * when the actor is moved out of its reach (TransformComponent.teleportPosition, then
 * WindowManager.validateWindows, the check Player.moveTo and TeleportSystems run after
 * a position change; BlockWindow.validate's 7-block reach), a second CraftRecipeAction
 * through the open bench window (SimpleCraftingWindow.handleAction queues another job),
 * and a physical hit (DamageSystems.executeDamage). Every method runs on the world thread.
 */
final class NativeCraftingOracleEvents {

    /** The reach move: the actor steps this far east, past the 7-block bench reach. */
    static final double REACH_OUT_BLOCKS = 9.0;
    /** The damage event's physical amount. */
    static final float DAMAGE_AMOUNT = 1.0F;

    static final String REACH_CANCEL_REASON = "bench_window_closed_out_of_reach";

    private NativeCraftingOracleEvents() {}

    /**
     * Opens the placed bench's window in the actor's WindowManager, as the native bench
     * interaction does, with the same block, rotation and BenchBlock the bridge uses.
     */
    static SimpleCraftingWindow openBenchWindow(
        World world,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        BenchPlacement placement
    ) {
        Player player = store.getComponent(actor, Player.getComponentType());
        if (player == null) {
            throw new IllegalStateException("bench window probe needs the headless Player");
        }
        BlockType blockType = NativeWorldAccess.getBlockType(
            world,
            placement.x(),
            placement.y(),
            placement.z()
        );
        BenchBlock benchBlock = NativeCraftingOracle.benchComponent(
            world,
            placement.x(),
            placement.y(),
            placement.z()
        );
        NativeWorldAccess.Column column = NativeWorldAccess.getChunk(
            world,
            ChunkUtil.indexChunkFromBlock(placement.x(), placement.z())
        );
        if (blockType == null || benchBlock == null || column == null) {
            throw new IllegalStateException("bench window probe needs the placed bench and its BenchBlock");
        }
        int rotation = column.getRotationIndex(placement.x(), placement.y(), placement.z());
        SimpleCraftingWindow window = new SimpleCraftingWindow(
            placement.x(),
            placement.y(),
            placement.z(),
            rotation,
            blockType,
            benchBlock
        );
        player.getWindowManager().openWindow(actor, window, store);
        if (player.getWindowManager().getWindow(window.getId()) != window) {
            throw new IllegalStateException("bench window probe: the window did not open");
        }
        return window;
    }

    /** Closes the bench window if it is still registered; a window closed by an event is not closed twice. */
    static void closeBenchWindow(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        SimpleCraftingWindow window
    ) {
        Player player = store.getComponent(actor, Player.getComponentType());
        if (player == null) {
            return;
        }
        WindowManager windows = player.getWindowManager();
        if (windows.getWindow(window.getId()) == window) {
            windows.closeWindow(actor, window.getId(), store);
        }
    }

    /** Applies one scripted event at the world tick, recording it on the oracle's ledger. */
    static void apply(
        NativeCraftingOracle oracle,
        CraftRequest probe,
        ProbeEvent event,
        SimpleCraftingWindow window,
        long tick
    ) {
        Ref<EntityStore> actor = oracle.actor();
        Store<EntityStore> store = oracle.store();
        Player player = store.getComponent(actor, Player.getComponentType());
        if (player == null) {
            oracle.event(tick, event.kind(), false, "no_headless_player");
            return;
        }
        WindowManager windows = player.getWindowManager();
        boolean windowOpen = window != null && windows.getWindow(window.getId()) == window;
        switch (event.kind()) {
            case ProbeEvent.REACH_OUT -> {
                if (!windowOpen) {
                    oracle.event(tick, event.kind(), false, "bench_window_not_open");
                    return;
                }
                int before = windows.getWindows().size();
                TransformComponent transform = store.getComponent(actor, TransformComponent.getComponentType());
                Vector3d position = transform.getPosition();
                transform.teleportPosition(new Vector3d(
                    position.x + REACH_OUT_BLOCKS,
                    position.y,
                    position.z
                ));
                // The window validation Player.moveTo and TeleportSystems run after a position change.
                windows.validateWindows(actor, store);
                boolean closed = windows.getWindow(window.getId()) != window;
                int queue = queueSize(store, actor);
                oracle.event(tick, event.kind(), true,
                    "windows_before=" + before
                        + ";windows_after=" + windows.getWindows().size()
                        + ";window_closed=" + closed
                        + ";queue_after=" + queue
                        + ";moved_blocks=" + REACH_OUT_BLOCKS + ";method=teleportPosition+validateWindows");
                if (closed && queue == 0 && oracle.pending()) {
                    oracle.cancel(tick, REACH_CANCEL_REASON);
                }
            }
            case ProbeEvent.SECOND_CRAFT -> {
                if (!windowOpen) {
                    oracle.event(tick, event.kind(), false, "bench_window_not_open");
                    return;
                }
                int queueBefore = queueSize(store, actor);
                window.handleAction(
                    actor,
                    store,
                    new CraftRecipeAction(probe.recipeId(), probe.quantity())
                );
                int queueAfter = queueSize(store, actor);
                oracle.event(tick, event.kind(), queueAfter > queueBefore,
                    "queue_before=" + queueBefore + ";queue_after=" + queueAfter);
            }
            case ProbeEvent.DAMAGE -> {
                NPCEntity npc = store.getComponent(actor, NPCEntity.getComponentType());
                double before = npc == null ? -1.0 : EvidenceCapture.entityHealth(store, actor, npc);
                DamageSystems.executeDamage(
                    actor,
                    store,
                    new Damage(Damage.NULL_SOURCE, DamageCause.PHYSICAL, DAMAGE_AMOUNT)
                );
                double after = npc == null ? -1.0 : EvidenceCapture.entityHealth(store, actor, npc);
                oracle.event(tick, event.kind(), true,
                    "amount=" + DAMAGE_AMOUNT + ";cause=PHYSICAL;health_before=" + before
                        + ";health_after=" + after);
            }
            default -> throw new IllegalArgumentException("unknown probe event " + event.kind());
        }
    }

    private static int queueSize(Store<EntityStore> store, Ref<EntityStore> actor) {
        CraftingManager manager = store.getComponent(actor, CraftingManager.getComponentType());
        return manager == null ? -1 : manager.getRemainingQueueSize();
    }
}
