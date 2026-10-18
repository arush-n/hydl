package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.builtin.crafting.component.BenchBlock;
import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.builtin.crafting.component.ProcessingBenchBlock;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.bench.Bench;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.inventory.transaction.ItemStackTransaction;
import com.hypixel.hytale.server.core.modules.block.BlockModule;
import com.hypixel.hytale.server.core.modules.time.WorldTimeResource;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.WorldConfig;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.ItemQuantity;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.ProcessingProbe;
import com.hytalerlbridge.nativebackend.NativeCraftingOracle.BenchPlacement;
import com.hytalerlbridge.nativebackend.NativeCraftingOracleLedger.Slot;
import com.hytalerlbridge.nativebackend.support.NativeWorldAccess;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Processing-bench oracle (stage 5): the route a Processing bench (Campfire, Furnace, Tannery,
 * Salvagebench) takes instead of craft_recipe. The probe places the bench in reach, puts the
 * input and fuel stacks into the bench's own containers, activates it with the block's
 * {@code setActive} (the call ProcessingBenchWindow makes for SetActiveAction), and then lets
 * the world run unpaused with its game clock running, as it does for a player. It observes once
 * per world tick on the world thread and records the input, fuel and output containers, the
 * active flag, the selected recipe, the input progress, the fuel time and the processing gates.
 *
 * <p>The probe does not step the world. The bridge's steps pause the world inside the last
 * entity tick of each step, so the chunk store (where the processing tick runs) skips that
 * world tick; a one-tick step would never run the processing tick.</p>
 *
 * <p>Every method that touches world state runs on the world thread through the
 * {@code onWorld} hook.</p>
 */
final class NativeProcessingOracle {

    static final String SCHEMA = "hytalerl_processing_oracle_v1";
    static final String INPUT = "bench_input";
    static final String FUEL = "bench_fuel";
    static final String OUTPUT = "bench_output";
    static final String SCOPE = NativeCraftingOracle.UNBOUND_SCOPE_PREFIX + "processing_bench";
    /** Bench.Type ordinal of a Processing bench (Crafting 0, Processing 1, DiagramCrafting 2, StructuralCrafting 3). */
    static final int PROCESSING_TYPE = 1;

    private NativeProcessingOracle() {}

    /** Runs one probe and returns its JSON record. */
    static String run(
        World world,
        Ref<EntityStore> agent,
        ProcessingProbe probe,
        Consumer<Runnable> onWorld
    ) {
        Session session = new Session(world, probe);
        onWorld.accept(() -> session.place(agent));
        onWorld.accept(session::start);
        while (session.running()) {
            onWorld.accept(session::observeNewTick);
        }
        onWorld.accept(session::finish);
        return session.json();
    }

    /** One probe's world-thread state, its per-tick lifecycle and its slot deltas. */
    private static final class Session {

        private final World world;
        private final ProcessingProbe probe;
        private final Map<String, Slot> previous = new LinkedHashMap<>();
        private final Map<String, Tracked> tracked = new LinkedHashMap<>();
        private final List<String> inserted = new ArrayList<>();
        private final List<String> lifecycle = new ArrayList<>();
        private BenchPlacement placement;
        private ProcessingBenchBlock processing;
        private BenchBlock bench;
        private BlockModule.BlockStateInfo blockState;
        private String benchAssetId = "";
        private int benchType = -1;
        private long requestedTick = -1L;
        private Instant gameStart;
        private String gameStartText = "";
        private String gameEndText = "";
        private boolean pausedBefore;
        private boolean worldPausedBefore;
        private long lastObservedTick = -1L;
        private String recipeId = "";
        private String expectedOutputId = "";
        private String gatesAtPlacement = "";
        private String gatesAtStart = "";
        private int observed;
        private boolean inputSeen;
        private long firstRemovedTick = -1L;
        private long inputConsumedTick = -1L;
        private long firstOutputTick = -1L;
        private long firstExtraOutputTick = -1L;
        private long firstFuelConsumedTick = -1L;
        private int fuelItemsConsumed;
        private int fuelBurnTicks;
        private float lastFuelSeconds;
        private boolean setActiveAccepted;
        private boolean activeAfterActivation;
        private boolean done;
        private String status = "pending";
        private String rejectReason = "";

        /** A slot that changed: the first and last tick it changed, its state before the first change and after the last. */
        private record Tracked(long firstTick, long lastTick, Slot before, Slot after) {}

        Session(World world, ProcessingProbe probe) {
            this.world = world;
            this.probe = probe;
        }

        boolean running() {
            return !done && observed < probe.maxTicks();
        }

        /** Places the probe's bench and gives its column the ticking state a loaded chunk has. */
        void place(Ref<EntityStore> agent) {
            placement = NativeCraftingOracle.placeBench(world, agent, probe.benchBlockId());
            NativeWorldAccess.getChunk(world, ChunkUtil.indexChunkFromBlock(placement.x(), placement.z()));
            int x = placement.x();
            int y = placement.y();
            int z = placement.z();
            BlockType block = NativeWorldAccess.getBlockType(world, x, y, z);
            Bench asset = block == null ? null : block.getBench();
            benchAssetId = asset == null ? "" : asset.getId();
            benchType = asset == null ? -1 : asset.getType().ordinal();
            processing = processingComponent(world, x, y, z);
            bench = NativeCraftingOracle.benchComponent(world, x, y, z);
            blockState = blockStateInfo(world, x, y, z);
            gatesAtPlacement = gates();
        }

        /** Clears the game clock for the probe, inserts the inputs and fuel, and activates the bench. */
        void start() {
            requestedTick = world.getTick();
            gameStart = resourceGameTime(world);
            gameStartText = String.valueOf(gameStart);
            if (benchType != PROCESSING_TYPE || processing == null || bench == null || blockState == null) {
                refuse("bench_not_processing");
                return;
            }
            gatesAtStart = gates();
            WorldConfig config = world.getWorldConfig();
            pausedBefore = config.isGameTimePaused();
            if (pausedBefore) {
                config.setGameTimePaused(false);
                config.markChanged();
            }
            worldPausedBefore = world.isPaused();
            if (worldPausedBefore) {
                world.setPaused(false);
            }
            boolean inputsAccepted = true;
            for (ItemQuantity stack : probe.inputs()) {
                inputsAccepted &= insert(INPUT, processing.getInputContainer(), stack);
            }
            boolean fuelAccepted = true;
            for (ItemQuantity stack : probe.fuel()) {
                fuelAccepted &= insert(FUEL, processing.getFuelContainer(), stack);
            }
            if (!inputsAccepted) {
                refuse("input_rejected");
                return;
            }
            if (!fuelAccepted) {
                refuse("fuel_rejected");
                return;
            }
            setActiveAccepted = processing.setActive(true, bench, blockState);
            activeAfterActivation = processing.isActive();
            if (!setActiveAccepted) {
                refuse("set_active_refused");
                return;
            }
            observe();
        }

        /** Observes once per world tick: the world thread runs this between two ticks, so a new tick number is a new observation. */
        void observeNewTick() {
            if (world.getTick() != lastObservedTick) {
                lastObservedTick = world.getTick();
                observe();
            }
        }

        /** Records the current world tick: the containers, the flags, the recipe, the progress and the gates. */
        void observe() {
            if (done) {
                return;
            }
            observed++;
            long tick = world.getTick();
            lastObservedTick = tick;
            List<Slot> now = new ArrayList<>();
            now.addAll(NativeCraftingOracle.containerSlots(INPUT, processing.getInputContainer()));
            now.addAll(NativeCraftingOracle.containerSlots(FUEL, processing.getFuelContainer()));
            now.addAll(NativeCraftingOracle.containerSlots(OUTPUT, processing.getOutputContainer()));
            List<String> changes = new ArrayList<>();
            for (Slot slot : now) {
                String key = slot.key();
                Slot prior = previous.getOrDefault(key, Slot.empty(slot.container(), slot.slot()));
                if (prior.equals(slot)) {
                    continue;
                }
                changes.add(change(prior, slot));
                noteChange(tick, prior, slot);
                previous.put(key, slot);
                Tracked known = tracked.get(key);
                tracked.put(key, known == null
                    ? new Tracked(tick, tick, prior, slot)
                    : new Tracked(known.firstTick(), tick, known.before(), slot));
            }
            CraftingRecipe recipe = processing.getRecipe();
            String recipeNow = recipe == null ? "" : recipe.getId();
            if (recipeId.isEmpty() && recipe != null) {
                recipeId = recipeNow;
                List<ItemStack> outputs = CraftingManager.getOutputItemStacks(recipe);
                expectedOutputId = outputs.isEmpty() || outputs.get(0) == null ? "" : outputs.get(0).getItemId();
            }
            boolean active = processing.isActive();
            boolean inputPresent = anyItem(now, INPUT);
            if (inputPresent) {
                inputSeen = true;
            }
            if (inputSeen && !inputPresent && inputConsumedTick < 0L) {
                inputConsumedTick = tick;
            }
            if (!expectedOutputId.isEmpty() && hasItem(now, OUTPUT, expectedOutputId) && firstOutputTick < 0L) {
                firstOutputTick = tick;
            }
            float fuelNow = processing.getFuelTime();
            if (fuelNow < lastFuelSeconds) {
                fuelBurnTicks++;
            }
            lastFuelSeconds = fuelNow;
            if (firstOutputTick >= 0L && !inputPresent) {
                done = true;
                status = "finished";
            } else if (!active && firstOutputTick < 0L && observed > 2) {
                done = true;
                status = "stalled";
            } else if (observed >= probe.maxTicks()) {
                done = true;
                status = "timed_out";
            }
            Instant gameNow = resourceGameTime(world);
            double gameSeconds = Duration.between(gameStart, gameNow).toMillis() / 1000.0;
            lifecycle.add("{\"tick\":" + tick
                + ",\"active\":" + active
                + ",\"finished\":" + "finished".equals(status)
                + ",\"cancelled\":false"
                + ",\"elapsed_ticks\":" + (tick - requestedTick)
                + ",\"progress_seconds\":" + fmt(processing.getInputProgress())
                + ",\"fuel_seconds\":" + fmt(fuelNow)
                + ",\"recipe_id\":\"" + esc(recipeNow) + "\""
                + ",\"game_time\":\"" + esc(String.valueOf(gameNow)) + "\""
                + ",\"game_seconds_elapsed\":" + fmt(gameSeconds)
                + ",\"input\":" + nonEmpty(now, INPUT)
                + ",\"fuel\":" + nonEmpty(now, FUEL)
                + ",\"output\":" + nonEmpty(now, OUTPUT)
                + ",\"changes\":[" + String.join(",", changes) + "]"
                + ",\"gates\":" + gates()
                + "}");
        }

        /** Restores the game clock's and the world's pause state the server had before the probe. */
        void finish() {
            if (pausedBefore) {
                WorldConfig config = world.getWorldConfig();
                config.setGameTimePaused(true);
                config.markChanged();
            }
            if (worldPausedBefore) {
                world.setPaused(true);
            }
            gameEndText = String.valueOf(resourceGameTime(world));
            if (!done) {
                done = true;
                status = "timed_out";
            }
        }

        private void refuse(String reason) {
            status = "refused";
            rejectReason = reason;
            done = true;
        }

        private boolean insert(String container, ItemContainer target, ItemQuantity stack) {
            ItemStackTransaction placed = target.addItemStack(new ItemStack(stack.itemId(), stack.quantity()));
            boolean accepted = placed.succeeded();
            inserted.add("{\"container\":\"" + container + "\",\"item_id\":\"" + esc(stack.itemId())
                + "\",\"quantity\":" + stack.quantity() + ",\"accepted\":" + accepted + "}");
            return accepted;
        }

        private void noteChange(long tick, Slot prior, Slot slot) {
            boolean removed = !prior.itemId().isEmpty()
                && (!prior.itemId().equals(slot.itemId()) || slot.quantity() < prior.quantity());
            if (removed && (INPUT.equals(slot.container()) || FUEL.equals(slot.container()))) {
                if (firstRemovedTick < 0L) {
                    firstRemovedTick = tick;
                }
            }
            if (removed && FUEL.equals(slot.container())) {
                if (firstFuelConsumedTick < 0L) {
                    firstFuelConsumedTick = tick;
                }
                if (prior.itemId().equals(slot.itemId())) {
                    fuelItemsConsumed += prior.quantity() - slot.quantity();
                } else {
                    fuelItemsConsumed += prior.quantity();
                }
            }
            boolean added = !slot.itemId().isEmpty()
                && (!prior.itemId().equals(slot.itemId()) || slot.quantity() > prior.quantity());
            if (added && OUTPUT.equals(slot.container()) && !slot.itemId().equals(expectedOutputId)) {
                if (firstExtraOutputTick < 0L) {
                    firstExtraOutputTick = tick;
                }
            }
        }

        private String slotsJson() {
            StringBuilder out = new StringBuilder();
            for (Map.Entry<String, Tracked> entry : tracked.entrySet()) {
                Tracked known = entry.getValue();
                if (out.length() > 0) {
                    out.append(',');
                }
                out.append("{\"container\":\"").append(known.before().container())
                    .append("\",\"slot\":").append(known.before().slot())
                    .append(",\"before_item_id\":\"").append(esc(known.before().itemId()))
                    .append("\",\"before_quantity\":").append(known.before().quantity())
                    .append(",\"after_item_id\":\"").append(esc(known.after().itemId()))
                    .append("\",\"after_quantity\":").append(known.after().quantity())
                    .append(",\"first_changed_tick\":").append(known.firstTick())
                    .append(",\"last_changed_tick\":").append(known.lastTick())
                    .append('}');
            }
            return out.toString();
        }

        String json() {
            StringBuilder out = new StringBuilder(4096);
            out.append("{\"schema\":\"").append(SCHEMA).append('"');
            out.append(",\"status\":\"").append(status).append('"');
            out.append(",\"reject_reason\":\"").append(esc(rejectReason)).append('"');
            out.append(",\"cancel_reason\":\"\"");
            out.append(",\"recipe_id\":\"").append(esc(recipeId)).append('"');
            out.append(",\"crafting_context\":\"processing\"");
            out.append(",\"execution_scope\":\"").append(SCOPE).append('"');
            out.append(",\"requested_tick\":").append(requestedTick);
            out.append(",\"bench\":{\"block_id\":\"").append(esc(placement.blockId()))
                .append("\",\"bench_id\":\"").append(esc(benchAssetId))
                .append("\",\"type\":").append(benchType)
                .append(",\"tier\":").append(bench == null ? -1 : bench.getTierLevel())
                .append(",\"x\":").append(placement.x())
                .append(",\"y\":").append(placement.y())
                .append(",\"z\":").append(placement.z())
                .append('}');
            out.append(",\"expected_output_item_id\":\"").append(esc(expectedOutputId)).append('"');
            out.append(",\"first_removed_tick\":").append(firstRemovedTick);
            out.append(",\"first_added_tick\":").append(firstOutputTick);
            out.append(",\"first_output_tick\":").append(firstOutputTick);
            out.append(",\"first_extra_output_tick\":").append(firstExtraOutputTick);
            out.append(",\"input_consumed_tick\":").append(inputConsumedTick);
            out.append(",\"first_fuel_consumed_tick\":").append(firstFuelConsumedTick);
            out.append(",\"fuel_items_consumed\":").append(fuelItemsConsumed);
            out.append(",\"fuel_burn_ticks\":").append(fuelBurnTicks);
            out.append(",\"output_elapsed_ticks\":").append(firstOutputTick < 0L ? -1L : firstOutputTick - requestedTick);
            out.append(",\"clock\":{\"paused_before\":").append(pausedBefore)
                .append(",\"game_time_start\":\"").append(esc(gameStartText))
                .append("\",\"game_time_end\":\"").append(esc(gameEndText))
                .append("\",\"seconds_per_tick\":").append(fmt(WorldTimeResource.getSecondsPerTick(world)))
                .append(",\"durations\":\"").append(world.getDaytimeDurationSeconds()).append('/')
                .append(world.getNighttimeDurationSeconds()).append("\"}");
            out.append(",\"set_active_accepted\":").append(setActiveAccepted);
            out.append(",\"active_after_activation\":").append(activeAfterActivation);
            out.append(",\"gates_at_placement\":").append(gatesAtPlacement);
            out.append(",\"gates_at_start\":").append(gatesAtStart.isEmpty() ? "null" : gatesAtStart);
            out.append(",\"inserted\":[").append(String.join(",", inserted)).append(']');
            out.append(",\"slots\":[").append(slotsJson()).append(']');
            out.append(",\"lifecycle\":[").append(String.join(",", lifecycle)).append(']');
            out.append(",\"events\":[]");
            out.append('}');
            return out.toString();
        }

        /** The gates that decide whether the bench's processing tick runs and advances, as one JSON object. */
        private String gates() {
            WorldConfig config = world.getWorldConfig();
            ChunkStore chunks = world.getChunkStore();
            int x = placement.x();
            int y = placement.y();
            int z = placement.z();
            Ref<ChunkStore> section = chunks.getChunkSectionReferenceAtBlock(x, y, z);
            Ref<ChunkStore> block = BlockModule.getBlockEntity(world, x, y, z);
            boolean sectionNonTicking = section != null && section.isValid()
                && chunks.getStore().getArchetype(section).contains(ChunkStore.REGISTRY.getNonTickingComponentType());
            boolean blockNonTicking = block != null && block.isValid()
                && chunks.getStore().getArchetype(block).contains(ChunkStore.REGISTRY.getNonTickingComponentType());
            boolean lastTickSet = processing != null && processing.getLastTickGameTime() != null;
            return "{\"world_paused\":" + world.isPaused()
                + ",\"game_time_paused\":" + config.isGameTimePaused()
                + ",\"last_tick_game_time_set\":" + lastTickSet
                + ",\"section_non_ticking\":" + sectionNonTicking
                + ",\"block_non_ticking\":" + blockNonTicking
                + ",\"block_ref_valid\":" + (block != null && block.isValid())
                + "}";
        }
    }

    private static boolean anyItem(List<Slot> slots, String container) {
        for (Slot slot : slots) {
            if (container.equals(slot.container()) && !slot.itemId().isEmpty()) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasItem(List<Slot> slots, String container, String itemId) {
        for (Slot slot : slots) {
            if (container.equals(slot.container()) && itemId.equals(slot.itemId())) {
                return true;
            }
        }
        return false;
    }

    private static String nonEmpty(List<Slot> slots, String container) {
        StringBuilder out = new StringBuilder("[");
        boolean first = true;
        for (Slot slot : slots) {
            if (!container.equals(slot.container()) || slot.itemId().isEmpty()) {
                continue;
            }
            if (!first) {
                out.append(',');
            }
            first = false;
            out.append("{\"slot\":").append(slot.slot())
                .append(",\"item_id\":\"").append(esc(slot.itemId()))
                .append("\",\"quantity\":").append(slot.quantity()).append('}');
        }
        return out.append(']').toString();
    }

    private static String change(Slot before, Slot after) {
        return "{\"container\":\"" + after.container() + "\",\"slot\":" + after.slot()
            + ",\"before_item_id\":\"" + esc(before.itemId()) + "\",\"before_quantity\":" + before.quantity()
            + ",\"after_item_id\":\"" + esc(after.itemId()) + "\",\"after_quantity\":" + after.quantity() + "}";
    }

    private static String fmt(double value) {
        return String.format(Locale.ROOT, "%.6f", value);
    }

    private static String esc(String value) {
        return NativeCraftingOracleLedger.escape(value);
    }

    static Instant resourceGameTime(World world) {
        return world.getEntityStore().getStore()
            .getResource(WorldTimeResource.getResourceType())
            .getGameTime();
    }

    static ProcessingBenchBlock processingComponent(World world, int x, int y, int z) {
        Ref<ChunkStore> reference = BlockModule.getBlockEntity(world, x, y, z);
        if (reference == null || !reference.isValid()) {
            return null;
        }
        return world.getChunkStore().getStore().getComponent(
            reference,
            ProcessingBenchBlock.getComponentType()
        );
    }

    static BlockModule.BlockStateInfo blockStateInfo(World world, int x, int y, int z) {
        Ref<ChunkStore> reference = BlockModule.getBlockEntity(world, x, y, z);
        if (reference == null || !reference.isValid()) {
            return null;
        }
        return world.getChunkStore().getStore().getComponent(
            reference,
            BlockModule.BlockStateInfo.getComponentType()
        );
    }
}
