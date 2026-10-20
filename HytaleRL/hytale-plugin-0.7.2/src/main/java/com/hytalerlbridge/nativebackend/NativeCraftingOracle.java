package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.builtin.adventure.memories.MemoriesGameplayConfig;
import com.hypixel.hytale.builtin.adventure.memories.MemoriesPlugin;
import com.hypixel.hytale.builtin.crafting.component.BenchBlock;
import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.builtin.crafting.window.SimpleCraftingWindow;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.bench.Bench;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.modules.block.BlockModule;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.action.NativeWorldVerbRequest;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.CraftRequest;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.ProbeEvent;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.ProcessingProbe;
import com.hytalerlbridge.nativebackend.NativeCraftingOracleLedger.BenchReadout;
import com.hytalerlbridge.nativebackend.NativeCraftingOracleLedger.Slot;
import com.hytalerlbridge.nativebackend.support.NativeWorldAccess;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.function.Consumer;
import java.util.function.Supplier;
import org.joml.Vector3d;

/**
 * Crafting oracle for one {@code craft_recipe} request: snapshots the actor's
 * inventory and the bench's input slot each tick, reads the bench readout, and
 * places the reset fixture. The bookkeeping is in {@link NativeCraftingOracleLedger}.
 *
 * <p>Every method runs on the world thread, as the craft path does.</p>
 */
final class NativeCraftingOracle {

    /** Where a reset fixture placed its bench, in absolute block coordinates. */
    record BenchPlacement(int x, int y, int z, String blockId) {}

    /** Scope prefix for probe executions: no policy-capture candidate is bound. */
    static final String UNBOUND_SCOPE_PREFIX = "oracle_unbound_";

    private final NativeCraftingOracleLedger ledger;
    private final Ref<EntityStore> actor;
    private final Store<EntityStore> store;
    private ItemContainer benchInput;
    private SimpleCraftingWindow benchWindow;

    private NativeCraftingOracle(
        NativeCraftingOracleLedger ledger,
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        this.ledger = ledger;
        this.actor = actor;
        this.store = store;
    }

    /** Snapshot taken before the request starts; the bench readout is the request's own. */
    static NativeCraftingOracle begin(
        NativeWorldVerbRequest request,
        World world,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        long requestedTick
    ) {
        BenchReadout bench = "bench".equals(request.craftingContext())
            ? readBench(
                world,
                request.benchX(),
                request.benchY(),
                request.benchZ()
            )
            : null;
        List<Slot> before = actorSlots(actor, store);
        NativeCraftingOracleLedger ledger = new NativeCraftingOracleLedger(
            request.recipeId(),
            request.craftingContext(),
            bench,
            requestedTick,
            before
        );
        CraftingRecipe recipe = bench == null ? null : NativeHeadlessCrafting.recipe(request.recipeId());
        if (recipe != null) {
            ledger.augmentVerdict(NativeHeadlessCrafting.augmentVerdict(
                recipe.getBenchRequirement(),
                bench.type(),
                bench.benchId(),
                bench.tier(),
                bench.augmentTags()
            ));
        }
        return new NativeCraftingOracle(ledger, actor, store);
    }

    void refuse(String reason, long tick) {
        ledger.refuse(reason, tick);
    }

    /** The craft was accepted: its structural input slot (if any) and its execution scope. */
    void accepted(ItemContainer container, String scope) {
        this.benchInput = container;
        ledger.executionScope(scope);
    }

    String json() {
        return ledger.toJson();
    }

    /** Appends pre-encoded JSON members to the record: a diagram probe's route and window readout. */
    void extraFields(String members) {
        ledger.extraFields(members);
    }

    boolean pending() {
        return "pending".equals(ledger.status());
    }

    /** Records one tick while the craft is pending; {@code active} false marks it finished. */
    void observe(long tick, boolean active) {
        if (!"pending".equals(ledger.status())) {
            return;
        }
        ledger.observe(tick, currentSlots(), active, progressSeconds());
    }

    /** Cancels the pending craft on this tick, recording the slots it changed. */
    void cancel(long tick, String reason) {
        ledger.cancel(reason, tick, currentSlots(), progressSeconds());
    }

    void event(long tick, String kind, boolean applied, String detail) {
        ledger.event(tick, kind, applied, detail);
    }

    Ref<EntityStore> actor() {
        return actor;
    }

    Store<EntityStore> store() {
        return store;
    }

    /** The probe's open bench window, opened before its craft request (or null). */
    void attachWindow(SimpleCraftingWindow window) {
        benchWindow = window;
    }

    /** Applies the events due at this probe tick, in the order the fixture lists them. */
    void applyEvents(CraftRequest probe, List<ProbeEvent> due, long tick) {
        for (ProbeEvent event : due) {
            if (!pending()) {
                ledger.event(tick, event.kind(), false, "craft_not_pending");
                continue;
            }
            NativeCraftingOracleEvents.apply(this, probe, event, benchWindow, tick);
        }
    }

    /** Records events the probe never reached because its craft finished first. */
    void recordUnreached(List<ProbeEvent> late, long tick) {
        for (ProbeEvent event : late) {
            ledger.event(tick, event.kind(), false, "craft_ended_before_event;scheduled_tick=" + event.tick());
        }
    }

    /** Closes the probe's bench window if an event has not closed it already. */
    void closeWindow() {
        if (benchWindow != null) {
            NativeCraftingOracleEvents.closeBenchWindow(actor, store, benchWindow);
            benchWindow = null;
        }
    }

    private List<Slot> currentSlots() {
        List<Slot> now = actorSlots(actor, store);
        if (benchInput != null) {
            now.addAll(containerSlots("bench_input", benchInput));
        }
        return now;
    }

    /**
     * The native craft progress: the queued head job's timeSecondsCompleted, read by
     * reflection (CraftingManager and CraftingJob keep it private). -1 when no job is
     * queued, -2 when the field cannot be read. Observation only; nothing branches on it.
     */
    private double progressSeconds() {
        CraftingManager manager = store.getComponent(actor, CraftingManager.getComponentType());
        if (manager == null) {
            return -1.0;
        }
        try {
            Field queueField = CraftingManager.class.getDeclaredField("queuedCraftingJobs");
            queueField.setAccessible(true);
            Object head = ((BlockingQueue<?>) queueField.get(manager)).peek();
            if (head == null) {
                return -1.0;
            }
            Field seconds = head.getClass().getDeclaredField("timeSecondsCompleted");
            seconds.setAccessible(true);
            return seconds.getFloat(head);
        } catch (ReflectiveOperationException | ClassCastException error) {
            return -2.0;
        }
    }

    static void putInfo(Map<String, Object> info, NativeCraftingOracle oracle) {
        info.put("native_craft_oracle_present", oracle != null);
        if (oracle == null) {
            return;
        }
        info.put("native_craft_oracle_status", oracle.ledger.status());
        info.put("native_craft_oracle_reject_reason", oracle.ledger.rejectReason());
        info.put("native_craft_oracle_json", oracle.ledger.toJson());
    }

    /** Reset readout, captured on the world thread: the native memories level and the placed bench. */
    record ResetReadout(int memoriesLevel, int recordedCount, BenchReadout bench) {}

    static ResetReadout readReset(World world, BenchPlacement placement) {
        MemoriesPlugin memories = MemoriesPlugin.get();
        boolean levelReadable = memories != null
            && world.getGameplayConfig() != null;
        int level = levelReadable
            ? memories.getMemoriesLevel(world.getGameplayConfig())
            : -1;
        int count = memories == null
            ? -1
            : memories.getRecordedMemories().size();
        BenchReadout bench = placement == null
            ? null
            : readBench(world, placement.x(), placement.y(), placement.z());
        return new ResetReadout(level, count, bench);
    }

    /**
     * Reset output: the readout captured after the fixtures, each craft or diagram probe's oracle in
     * request order, and each processing probe's record (JSON) in request order.
     */
    record ResetOutcome(
        ResetReadout readout,
        List<NativeCraftingOracle> records,
        List<String> processingRecords
    ) {}

    /** Upper bound on one probe's world ticks; a timed craft with a second job needs several seconds. */
    private static final int PROBE_TICK_LIMIT = 1200;

    /**
     * The reset-time sequence, in order: the bench fixture (placement, one tick, tier,
     * one tick), the memories fixture, the native readout, then each requested
     * craft_recipe probe. Work on world state runs through onWorld; tick advances the
     * world one step; craft issues one probe; current returns the latest probe's oracle.
     */
    static ResetOutcome resetSequence(
        World world,
        Ref<EntityStore> agent,
        NativeCraftingOracleFixture fixture,
        Consumer<Runnable> onWorld,
        Runnable tick,
        Consumer<NativeWorldVerbRequest> craft,
        Supplier<NativeCraftingOracle> current
    ) {
        BenchPlacement[] placement = new BenchPlacement[1];
        if (fixture != null && fixture.hasBench()) {
            onWorld.accept(() -> placement[0] = placeBench(world, agent, fixture));
            tick.run();
            onWorld.accept(() -> settleBenchTier(world, placement[0], fixture.benchTier()));
            tick.run();
        }
        if (fixture != null && fixture.setsMemories()) {
            onWorld.accept(() -> setMemoriesLevel(world, fixture.memoriesLevel()));
        }
        ResetReadout[] readout = new ResetReadout[1];
        onWorld.accept(() -> readout[0] = readReset(world, placement[0]));
        List<NativeCraftingOracle> records = new ArrayList<>();
        if (fixture != null) {
            for (CraftRequest probe : fixture.requests()) {
                if (NativeCraftingOracleFixture.DIAGRAM_CONTEXT.equals(probe.craftingContext())) {
                    records.add(NativeDiagramOracle.run(world, agent, probe, readout[0], onWorld, tick));
                    continue;
                }
                SimpleCraftingWindow[] window = new SimpleCraftingWindow[1];
                if (probe.openWindow()) {
                    onWorld.accept(() -> window[0] = NativeCraftingOracleEvents.openBenchWindow(
                        world,
                        agent,
                        world.getEntityStore().getStore(),
                        placement[0]
                    ));
                }
                craft.accept(probeRequest(probe, readout[0]));
                NativeCraftingOracle oracle = current.get();
                oracle.attachWindow(window[0]);
                runProbe(world, probe, oracle, onWorld, tick);
                onWorld.accept(oracle::closeWindow);
                records.add(oracle);
            }
        }
        List<String> processingRecords = new ArrayList<>();
        if (fixture != null) {
            for (ProcessingProbe probe : fixture.processing()) {
                processingRecords.add(NativeProcessingOracle.run(world, agent, probe, onWorld));
            }
        }
        return new ResetOutcome(readout[0], records, processingRecords);
    }

    /**
     * Ticks one probe until its craft is no longer pending. The events whose tick is k are
     * dispatched to the world thread after k loop ticks; the world-thread dispatch runs them
     * before the next world tick, so each event's recorded world tick is k + 1 past the request
     * (measured in the reach, second-craft and damage runs). The recorded tick is authoritative.
     */
    private static void runProbe(
        World world,
        CraftRequest probe,
        NativeCraftingOracle oracle,
        Consumer<Runnable> onWorld,
        Runnable tick
    ) {
        int elapsed = 0;
        int checkedThrough = -1;
        while (oracle.pending()) {
            if (elapsed >= PROBE_TICK_LIMIT) {
                throw new IllegalStateException(
                    "crafting oracle probe did not finish in " + PROBE_TICK_LIMIT
                        + " ticks: " + probe.recipeId()
                );
            }
            int at = elapsed;
            checkedThrough = at;
            List<ProbeEvent> due = probe.events().stream()
                .filter(event -> event.tick() == at)
                .toList();
            if (!due.isEmpty()) {
                onWorld.accept(() -> oracle.applyEvents(probe, due, world.getTick()));
            }
            if (!oracle.pending()) {
                break;
            }
            tick.run();
            elapsed++;
        }
        int checked = checkedThrough;
        List<ProbeEvent> late = probe.events().stream()
            .filter(event -> event.tick() > checked)
            .toList();
        if (!late.isEmpty()) {
            onWorld.accept(() -> oracle.recordUnreached(late, world.getTick()));
        }
    }

    /**
     * The request for one probe. A bench probe names the placed bench's own block, id,
     * type and tier, so only the recipe's requirements can refuse it. A fieldcraft probe
     * names the Fieldcraft context.
     */
    static NativeWorldVerbRequest probeRequest(CraftRequest probe, ResetReadout readout) {
        // A diagram probe runs in the Armory's window; its carrier request names the placed bench.
        String context = NativeCraftingOracleFixture.DIAGRAM_CONTEXT.equals(probe.craftingContext())
            ? NativeCraftingOracleFixture.BENCH_CONTEXT
            : probe.craftingContext();
        boolean bench = NativeCraftingOracleFixture.BENCH_CONTEXT.equals(context);
        BenchReadout placed = readout == null ? null : readout.bench();
        if (bench && placed == null) {
            throw new IllegalStateException("bench crafting probe needs a placed bench");
        }
        return new NativeWorldVerbRequest(
            1L,
            "craft_recipe",
            -1,
            "recipe",
            "",
            probe.recipeId(),
            "",
            "",
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            "player_inventory",
            -1,
            -1,
            "player_inventory_or_world_drop",
            "",
            "oracle-probe",
            probe.quantity(),
            "",
            context,
            bench ? placed.x() : 0,
            bench ? placed.y() : 0,
            bench ? placed.z() : 0,
            bench ? placed.blockId() : "",
            bench ? placed.benchId() : "Fieldcraft",
            bench ? placed.type() : 0,
            bench ? placed.tier() : 0,
            "",
            ""
        );
    }

    /** Reset info: the readout, the probe count, and each probe's record as one JSON array. */
    static void putResetInfo(Map<String, Object> info, ResetOutcome outcome) {
        putReadout(info, outcome == null ? null : outcome.readout());
        List<String> processing = outcome == null ? List.of() : outcome.processingRecords();
        info.put("native_processing_oracle_count", processing.size());
        if (!processing.isEmpty()) {
            info.put("native_processing_oracle_records_json", "[" + String.join(",", processing) + "]");
        }
        List<NativeCraftingOracle> records = outcome == null ? List.of() : outcome.records();
        info.put("native_craft_oracle_count", records.size());
        if (records.isEmpty()) {
            return;
        }
        StringBuilder json = new StringBuilder("[");
        for (int index = 0; index < records.size(); index++) {
            if (index > 0) {
                json.append(',');
            }
            json.append(records.get(index).json());
        }
        info.put("native_craft_oracle_records_json", json.append(']').toString());
    }

    /** Reads only the cached readout; the ECS and chunk stores are world-thread state. */
    static void putReadout(Map<String, Object> info, ResetReadout readout) {
        info.put(
            "native_memories_level",
            readout == null ? -1 : readout.memoriesLevel()
        );
        info.put(
            "native_memories_recorded_count",
            readout == null ? -1 : readout.recordedCount()
        );
        BenchReadout bench = readout == null ? null : readout.bench();
        if (bench == null) {
            return;
        }
        info.put("native_crafting_bench_block_id", bench.blockId());
        info.put("native_crafting_bench_id", bench.benchId());
        info.put("native_crafting_bench_type", bench.type());
        info.put("native_crafting_bench_tier", bench.tier());
        info.put("native_crafting_bench_x", bench.x());
        info.put("native_crafting_bench_y", bench.y());
        info.put("native_crafting_bench_z", bench.z());
        info.put("native_crafting_bench_augment_tags", bench.augmentTags());
    }

    /**
     * Places the bench two blocks west of the settled actor, on the floor level
     * the actor stands on, and returns where it went.
     */
    static BenchPlacement placeBench(
        World world,
        Ref<EntityStore> agent,
        NativeCraftingOracleFixture fixture
    ) {
        return placeBench(world, agent, fixture.benchBlockId());
    }

    /** Places the named bench block two blocks west of the settled actor and returns where it went. */
    static BenchPlacement placeBench(World world, Ref<EntityStore> agent, String benchBlockId) {
        TransformComponent transform = world.getEntityStore().getStore()
            .getComponent(agent, TransformComponent.getComponentType());
        if (transform == null) {
            throw new IllegalStateException(
                "crafting bench fixture has no settled actor transform"
            );
        }
        Vector3d position = transform.getPosition();
        int x = (int) Math.floor(position.x) - 2;
        int y = (int) Math.floor(position.y);
        int z = (int) Math.floor(position.z);
        NativeWorldAccess.setBlock(world, x, y, z, benchBlockId);
        BlockType placed = NativeWorldAccess.getBlockType(world, x, y, z);
        if (
            placed == null
                || placed.getBench() == null
                || !benchBlockId.equals(placed.getId())
        ) {
            throw new IllegalStateException(
                "crafting bench fixture did not place a bench: "
                    + benchBlockId
            );
        }
        return new BenchPlacement(x, y, z, placed.getId());
    }

    /** Sets the placed bench's BenchBlock.tierLevel, the component a native upgrade writes. */
    static void settleBenchTier(
        World world,
        BenchPlacement placement,
        int tier
    ) {
        BenchBlock bench = benchComponent(
            world,
            placement.x(),
            placement.y(),
            placement.z()
        );
        if (bench == null) {
            throw new IllegalStateException(
                "crafting bench fixture has no BenchBlock component"
            );
        }
        bench.setTierLevel(tier);
    }

    /** Sets the universe's recorded-memories count to the smallest count that reaches {@code level}. */
    static void setMemoriesLevel(World world, int level) {
        MemoriesPlugin memories = MemoriesPlugin.get();
        MemoriesGameplayConfig config = world.getGameplayConfig() == null
            ? null
            : MemoriesGameplayConfig.get(world.getGameplayConfig());
        if (memories == null || config == null) {
            throw new IllegalStateException(
                "memories fixture needs the memories plugin and its gameplay config"
            );
        }
        memories.setRecordedMemoriesCount(
            NativeCraftingOracleFixture.memoriesCountForLevel(
                level,
                config.getMemoriesAmountPerLevel()
            )
        );
    }

    static BenchReadout readBench(World world, int x, int y, int z) {
        BlockType block = NativeWorldAccess.getBlockType(world, x, y, z);
        if (block == null) {
            return new BenchReadout("", "", -1, -1, x, y, z, List.of());
        }
        Bench asset = block.getBench();
        String benchId = asset == null ? "" : asset.getId();
        int type = asset == null ? -1 : asset.getType().ordinal();
        BenchBlock component = benchComponent(world, x, y, z);
        int tier = component == null ? -1 : component.getTierLevel();
        return new BenchReadout(
            block.getId(),
            benchId,
            type,
            tier,
            x,
            y,
            z,
            NativeHeadlessCrafting.grantedAugmentTags(world, x, y, z)
        );
    }

    static BenchBlock benchComponent(World world, int x, int y, int z) {
        Ref<ChunkStore> reference = BlockModule.getBlockEntity(world, x, y, z);
        if (reference == null || !reference.isValid()) {
            return null;
        }
        return world.getChunkStore().getStore().getComponent(
            reference,
            BenchBlock.getComponentType()
        );
    }

    private static List<Slot> actorSlots(
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        ItemContainer inventory = InventoryComponent.getCombined(
            store,
            actor,
            InventoryComponent.EVERYTHING
        );
        return inventory == null
            ? new ArrayList<>()
            : containerSlots("player_inventory", inventory);
    }

    static List<Slot> containerSlots(String name, ItemContainer container) {
        List<Slot> slots = new ArrayList<>();
        int capacity = Short.toUnsignedInt(container.getCapacity());
        for (int index = 0; index < capacity; index++) {
            ItemStack stack = container.getItemStack((short) index);
            boolean empty = ItemStack.isEmpty(stack);
            slots.add(new Slot(
                name,
                index,
                empty ? "" : stack.getItemId(),
                empty ? 0 : stack.getQuantity()
            ));
        }
        return slots;
    }
}
