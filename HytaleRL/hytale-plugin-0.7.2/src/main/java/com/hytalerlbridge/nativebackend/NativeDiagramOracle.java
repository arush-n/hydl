package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.builtin.crafting.component.BenchBlock;
import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.builtin.crafting.window.DiagramCraftingWindow;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.protocol.packets.window.CraftItemAction;
import com.hypixel.hytale.protocol.packets.window.UpdateCategoryAction;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.asset.type.item.config.Item;
import com.hypixel.hytale.server.core.entity.entities.Player;
import com.hypixel.hytale.server.core.entity.entities.player.windows.WindowManager;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.MaterialQuantity;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.CraftRequest;
import com.hytalerlbridge.nativebackend.NativeCraftingOracle.ResetReadout;
import com.hytalerlbridge.nativebackend.NativeCraftingOracleLedger.Slot;
import com.hytalerlbridge.nativebackend.support.NativeWorldAccess;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

/**
 * Diagram-bench probe (stage 5): the route an Armory (DiagramCrafting bench) takes instead of
 * craft_recipe. The probe opens the placed bench's DiagramCraftingWindow, selects the probe's
 * category (UpdateCategoryAction), places the recipe's inputs into the window's input slots
 * positionally (input i in slot i; slot 0 is the window's primary slot), and sends
 * CraftItemAction, which the window answers with CraftingManager.queueCraft (ORDERED removal).
 * The ledger is the craft oracle's own, so the record has the same shape as the craft probes,
 * plus a {@code diagram} object and the route name.
 *
 * <p>Every method that touches world state runs on the world thread through the
 * {@code onWorld} hook; {@code tick} advances the world one step.</p>
 */
final class NativeDiagramOracle {

    static final String ROUTE = "diagram_craft_item_action";
    static final String SCOPE = NativeCraftingOracle.UNBOUND_SCOPE_PREFIX + "diagram_window";
    /** Upper bound on the probe's world ticks, as for the craft probes. */
    private static final int PROBE_TICK_LIMIT = 1200;

    private NativeDiagramOracle() {}

    /** Runs one diagram probe at the placed Armory and returns its oracle. */
    static NativeCraftingOracle run(
        World world,
        Ref<EntityStore> agent,
        CraftRequest probe,
        ResetReadout readout,
        Consumer<Runnable> onWorld,
        Runnable tick
    ) {
        if (readout == null || readout.bench() == null) {
            throw new IllegalStateException("diagram probe needs a placed Armory bench");
        }
        Session session = new Session(world, agent, probe, readout);
        onWorld.accept(session::open);
        onWorld.accept(session::craft);
        int elapsed = 0;
        while (session.pending() && elapsed < PROBE_TICK_LIMIT) {
            tick.run();
            elapsed++;
            onWorld.accept(session::observe);
        }
        onWorld.accept(session::close);
        return session.oracle;
    }

    /** The probe's window and oracle, and what the window held when the craft was sent. */
    private static final class Session {

        private final World world;
        private final Ref<EntityStore> agent;
        private final CraftRequest probe;
        private final ResetReadout readout;
        private final List<String> placed = new ArrayList<>();
        private NativeCraftingOracle oracle;
        private DiagramCraftingWindow window;
        private CraftingRecipe recipe;
        private ItemContainer inputs;
        private int capacity;
        private String diagramJson = "";
        private final List<String> contents = new ArrayList<>();
        private String outputBefore = "";
        private int queueBefore = -1;
        private int queueAfter = -1;
        private boolean queued;

        Session(World world, Ref<EntityStore> agent, CraftRequest probe, ResetReadout readout) {
            this.world = world;
            this.agent = agent;
            this.probe = probe;
            this.readout = readout;
        }

        boolean pending() {
            return oracle != null && oracle.pending();
        }

        /** Begins the record, opens the window, selects the category, and places the recipe inputs. */
        void open() {
            Store<EntityStore> store = world.getEntityStore().getStore();
            recipe = CraftingRecipe.getAssetMap().getAsset(probe.recipeId());
            oracle = NativeCraftingOracle.begin(
                NativeCraftingOracle.probeRequest(probe, readout),
                world,
                agent,
                store,
                world.getTick()
            );
            if (recipe == null) {
                oracle.refuse("recipe_unknown", world.getTick());
                return;
            }
            Player player = store.getComponent(agent, Player.getComponentType());
            if (player == null) {
                throw new IllegalStateException("diagram probe needs the headless Player");
            }
            window = openWindow(world, agent, store, readout.bench().x(), readout.bench().y(), readout.bench().z());
            window.handleAction(agent, store, new UpdateCategoryAction(probe.category(), probe.itemCategory()));
            inputs = innerInput(window);
            capacity = inputs.getCapacity();
            MaterialQuantity[] wanted = recipe.getInput();
            if (wanted.length > capacity) {
                oracle.refuse("diagram_input_count_mismatch;recipe_inputs=" + wanted.length
                    + ";window_capacity=" + capacity, world.getTick());
                return;
            }
            for (int slot = 0; slot < wanted.length; slot++) {
                String itemId = itemFor(wanted[slot]);
                boolean accepted = inputs.setItemStackForSlot(
                    (short) slot,
                    new ItemStack(itemId, wanted[slot].getQuantity()),
                    true
                ).succeeded();
                refreshInputs(window, inputs, agent, store);
                placed.add("{\"slot\":" + slot + ",\"item_id\":\"" + NativeCraftingOracleLedger.escape(itemId)
                    + "\",\"quantity\":" + wanted[slot].getQuantity() + ",\"accepted\":" + accepted + "}");
            }
            oracle.accepted(inputs, SCOPE);
        }

        /** Sends CraftItemAction; the craft is pending when the window queued it, refused otherwise. */
        void craft() {
            if (!pending()) {
                return;
            }
            Store<EntityStore> store = world.getEntityStore().getStore();
            outputBefore = outputItemId();
            queueBefore = queueSize(store);
            window.handleAction(agent, store, new CraftItemAction());
            queueAfter = queueSize(store);
            queued = queueAfter > queueBefore;
            if (queued) {
                oracle.observe(world.getTick(), true);
            } else {
                oracle.refuse("diagram_craft_item_not_queued", world.getTick());
            }
            record(world.getTick());
            diagramJson = ",\"route\":\"" + ROUTE + "\",\"diagram\":{\"category\":\""
                + NativeCraftingOracleLedger.escape(probe.category()) + "\",\"item_category\":\""
                + NativeCraftingOracleLedger.escape(probe.itemCategory())
                + "\",\"window_input_capacity\":" + capacity
                + ",\"inputs\":[" + String.join(",", placed) + "]"
                + ",\"output_slot_before_craft\":\"" + NativeCraftingOracleLedger.escape(outputBefore) + "\""
                + ",\"queue_before\":" + queueBefore
                + ",\"queue_after\":" + queueAfter
                + ",\"craft_item_queued\":" + queued
                + ",\"knowledge_required\":" + recipe.isKnowledgeRequired()
                + ",\"recipe_time_seconds\":" + String.format(Locale.ROOT, "%.6f", recipe.getTimeSeconds())
                + "}";
        }

        void observe() {
            if (!pending()) {
                return;
            }
            Store<EntityStore> store = world.getEntityStore().getStore();
            oracle.observe(world.getTick(), queueSize(store) > 0);
            record(world.getTick());
        }

        /** Records the window's input and output slots and the actor's inventory at this tick, when non-empty. */
        private void record(long tick) {
            List<Slot> now = new ArrayList<>();
            Store<EntityStore> store = world.getEntityStore().getStore();
            if (inputs != null) {
                now.addAll(NativeCraftingOracle.containerSlots("bench_input", inputs));
            }
            if (window != null) {
                ItemStack output = window.getItemContainer().getItemStack((short) capacity);
                if (output != null && !output.isEmpty()) {
                    now.add(new Slot("bench_output", 0, output.getItemId(), output.getQuantity()));
                }
            }
            ItemContainer inventory = InventoryComponent.getCombined(store, agent, InventoryComponent.EVERYTHING);
            if (inventory != null) {
                now.addAll(NativeCraftingOracle.containerSlots("player_inventory", inventory));
            }
            StringBuilder out = new StringBuilder("{\"tick\":" + tick + ",\"slots\":[");
            boolean first = true;
            for (Slot slot : now) {
                if (slot.itemId().isEmpty()) {
                    continue;
                }
                if (!first) {
                    out.append(',');
                }
                first = false;
                out.append("{\"container\":\"").append(slot.container()).append("\",\"slot\":").append(slot.slot())
                    .append(",\"item_id\":\"").append(NativeCraftingOracleLedger.escape(slot.itemId()))
                    .append("\",\"quantity\":").append(slot.quantity()).append('}');
            }
            contents.add(out.append("]}").toString());
        }

        /** Closes the probe's window if it is still open, and sets the record's diagram members and per-tick contents. */
        void close() {
            oracle.extraFields(diagramJson + ",\"contents_by_tick\":[" + String.join(",", contents) + "]");
            if (window == null) {
                return;
            }
            Store<EntityStore> store = world.getEntityStore().getStore();
            Player player = store.getComponent(agent, Player.getComponentType());
            if (player == null) {
                return;
            }
            WindowManager windows = player.getWindowManager();
            if (windows.getWindow(window.getId()) == window) {
                windows.closeWindow(agent, window.getId(), store);
            }
        }

        private String outputItemId() {
            ItemStack output = window.getItemContainer().getItemStack((short) capacity);
            return output == null || output.isEmpty() ? "" : output.getItemId();
        }

        private int queueSize(Store<EntityStore> store) {
            CraftingManager manager = store.getComponent(agent, CraftingManager.getComponentType());
            return manager == null ? -1 : manager.getRemainingQueueSize();
        }
    }

    /** The item a recipe input takes: the named item, or the first item (by id) the input accepts. */
    static String itemFor(MaterialQuantity input) {
        if (input.getItemId() != null && !input.getItemId().isEmpty()) {
            return input.getItemId();
        }
        List<String> ids = new ArrayList<>(Item.getAssetMap().getAssetMap().keySet());
        Collections.sort(ids);
        for (String id : ids) {
            if (CraftingManager.matches(input, new ItemStack(id, input.getQuantity()))) {
                return id;
            }
        }
        throw new IllegalStateException("no item matches the diagram input " + input.getResourceTypeId());
    }

    /**
     * The window's own input refresh, the method its open runs and its change listener calls. The
     * listener returns early for a window without a PlayerRef, which the headless actor's window
     * does not get, so the probe calls the refresh after each write, as the listener would.
     */
    private static void refreshInputs(
        DiagramCraftingWindow window,
        ItemContainer inputs,
        Ref<EntityStore> agent,
        Store<EntityStore> store
    ) {
        try {
            Method update = DiagramCraftingWindow.class.getDeclaredMethod(
                "updateInput",
                ItemContainer.class,
                Ref.class,
                Store.class
            );
            update.setAccessible(true);
            update.invoke(window, inputs, agent, store);
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException("diagram probe cannot refresh the window's inputs", error);
        }
    }

    /** The window's input container (primary and secondary slots), which the window's own listener watches. */
    private static ItemContainer innerInput(DiagramCraftingWindow window) {
        try {
            Field field = DiagramCraftingWindow.class.getDeclaredField("combinedInputItemContainer");
            field.setAccessible(true);
            return (ItemContainer) field.get(window);
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException("diagram probe cannot read the window's input container", error);
        }
    }

    private static DiagramCraftingWindow openWindow(
        World world,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        int x,
        int y,
        int z
    ) {
        Player player = store.getComponent(actor, Player.getComponentType());
        BlockType blockType = NativeWorldAccess.getBlockType(world, x, y, z);
        BenchBlock benchBlock = NativeCraftingOracle.benchComponent(world, x, y, z);
        NativeWorldAccess.Column column = NativeWorldAccess.getChunk(
            world,
            ChunkUtil.indexChunkFromBlock(x, z)
        );
        if (player == null || blockType == null || benchBlock == null || column == null) {
            throw new IllegalStateException("diagram probe needs the headless Player and the placed Armory");
        }
        int rotation = column.getRotationIndex(x, y, z);
        DiagramCraftingWindow window = new DiagramCraftingWindow(actor, store, x, y, z, rotation, blockType, benchBlock);
        player.getWindowManager().openWindow(actor, window, store);
        if (player.getWindowManager().getWindow(window.getId()) != window) {
            throw new IllegalStateException("diagram probe: the window did not open");
        }
        return window;
    }
}
