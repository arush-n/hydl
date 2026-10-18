package com.hytalerlbridge.environment.fidelity;

import java.util.List;

/**
 * Reset-time fixture for the crafting oracle: an optional bench placed in reach at
 * a native tier, the native memories level, an ordered list of craft_recipe probes that run
 * at reset after those fixtures apply, and an ordered list of processing-bench probes.
 *
 * <p>A bench tier is the {@code BenchBlock.tierLevel} component of the placed block
 * entity. A fresh bench starts at 1 and the native upgrade table reaches 8, so 1..8
 * is the native range. The memories level is
 * {@code MemoriesPlugin.getMemoriesLevel}: {@code i + 2} for the highest index
 * {@code i} with {@code count >= MemoriesAmountPerLevel[i]}, otherwise 1. The count
 * is the universe's recorded-memories set, so the level is not per actor.</p>
 *
 * <p>A craft probe is an ordinary craft_recipe request run through the legacy craft path
 * without a policy-capture candidate. Its execution scope is labelled
 * {@code oracle_unbound_*}. The production step path and its evidence gates are
 * unchanged. A diagram probe ({@code crafting_context diagram}) is not a craft_recipe: it
 * drives the Armory's DiagramCraftingWindow (see {@code NativeDiagramOracle}). A processing
 * probe places its own processing bench and cannot be combined with a crafting bench.</p>
 *
 * @param benchBlockId block id of the bench to place, or null for no bench
 * @param benchTier tier to set on the placed bench (1..8), or 0 when none
 * @param memoriesLevel level to set (1..6), or 0 to leave the native level
 * @param requests craft and diagram probes to run at reset, in order; empty for none
 * @param processing processing-bench probes to run after the craft probes, in order; empty for none
 */
public record NativeCraftingOracleFixture(
    String benchBlockId,
    int benchTier,
    int memoriesLevel,
    List<CraftRequest> requests,
    List<ProcessingProbe> processing
) {
    public static final int MIN_BENCH_TIER = 1;
    public static final int MAX_BENCH_TIER = 8;
    public static final int MAX_MEMORIES_LEVEL = 6;
    public static final int MAX_REQUESTS = 16;
    public static final String BENCH_CONTEXT = "bench";
    public static final String FIELDCRAFT_CONTEXT = "fieldcraft";
    public static final String DIAGRAM_CONTEXT = "diagram";

    /**
     * One probe: the recipe, its quantity, and its crafting context (bench, fieldcraft or
     * diagram). A bench probe with {@code openWindow} opens the placed bench's window in the
     * actor's WindowManager once the craft is queued, as the native client does while it
     * crafts; {@code reach_out} and {@code second_craft} events need that open window. A
     * diagram probe names its window's {@code category} and {@code itemCategory}.
     */
    public record CraftRequest(
        String recipeId,
        int quantity,
        String craftingContext,
        boolean openWindow,
        List<ProbeEvent> events,
        String category,
        String itemCategory
    ) {

        public CraftRequest(String recipeId, int quantity, String craftingContext) {
            this(recipeId, quantity, craftingContext, false, List.of(), "", "");
        }

        public CraftRequest(
            String recipeId,
            int quantity,
            String craftingContext,
            boolean openWindow,
            List<ProbeEvent> events
        ) {
            this(recipeId, quantity, craftingContext, openWindow, events, "", "");
        }

        public CraftRequest {
            if (recipeId == null || recipeId.isBlank()) {
                throw new IllegalArgumentException(
                    "crafting_oracle_requests recipe_id must be nonblank"
                );
            }
            recipeId = recipeId.trim();
            if (quantity < 1) {
                throw new IllegalArgumentException(
                    "crafting_oracle_requests quantity must be positive"
                );
            }
            if (
                !BENCH_CONTEXT.equals(craftingContext)
                    && !FIELDCRAFT_CONTEXT.equals(craftingContext)
                    && !DIAGRAM_CONTEXT.equals(craftingContext)
            ) {
                throw new IllegalArgumentException(
                    "crafting_oracle_requests crafting_context must be bench, fieldcraft or diagram"
                );
            }
            events = events == null ? List.of() : List.copyOf(events);
            category = category == null ? "" : category.trim();
            itemCategory = itemCategory == null ? "" : itemCategory.trim();
            if (openWindow && !BENCH_CONTEXT.equals(craftingContext)) {
                throw new IllegalArgumentException(
                    "crafting_oracle_requests open_window needs crafting_context bench"
                );
            }
            for (ProbeEvent event : events) {
                boolean needsWindow = ProbeEvent.REACH_OUT.equals(event.kind())
                    || ProbeEvent.SECOND_CRAFT.equals(event.kind());
                if (needsWindow && !openWindow) {
                    throw new IllegalArgumentException(
                        "crafting_oracle_requests event " + event.kind() + " needs open_window"
                    );
                }
            }
            if (DIAGRAM_CONTEXT.equals(craftingContext)) {
                if (category.isEmpty() || itemCategory.isEmpty()) {
                    throw new IllegalArgumentException(
                        "crafting_oracle_requests diagram probes need category and item_category"
                    );
                }
                if (openWindow || !events.isEmpty()) {
                    throw new IllegalArgumentException(
                        "crafting_oracle_requests diagram probes take no open_window or events"
                    );
                }
            } else if (!category.isEmpty() || !itemCategory.isEmpty()) {
                throw new IllegalArgumentException(
                    "crafting_oracle_requests category and item_category are for diagram probes"
                );
            }
        }
    }

    /**
     * A scripted event of a probe. {@code tick} counts world ticks from the request: the
     * event is applied when the world is {@code tick} ticks past the request tick, before the
     * next tick runs. {@code reach_out} moves the actor past the open bench window's reach,
     * {@code second_craft} sends a second CraftRecipeAction through that window, and
     * {@code damage} applies one native physical hit to the actor.
     */
    public record ProbeEvent(int tick, String kind) {
        public static final String REACH_OUT = "reach_out";
        public static final String SECOND_CRAFT = "second_craft";
        public static final String DAMAGE = "damage";

        public ProbeEvent {
            if (tick < 0) {
                throw new IllegalArgumentException(
                    "crafting_oracle_requests event tick must be nonnegative"
                );
            }
            if (
                !REACH_OUT.equals(kind)
                    && !SECOND_CRAFT.equals(kind)
                    && !DAMAGE.equals(kind)
            ) {
                throw new IllegalArgumentException(
                    "crafting_oracle_requests event kind must be reach_out, second_craft or damage"
                );
            }
        }
    }

    /** One stack of an item: the item asset id and a positive quantity. */
    public record ItemQuantity(String itemId, int quantity) {

        public ItemQuantity {
            if (itemId == null || itemId.isBlank()) {
                throw new IllegalArgumentException(
                    "processing_oracle_requests item_id must be nonblank"
                );
            }
            itemId = itemId.trim();
            if (quantity < 1) {
                throw new IllegalArgumentException(
                    "processing_oracle_requests quantity must be positive"
                );
            }
        }
    }

    /**
     * A processing probe: the processing bench to place, the input and fuel stacks inserted
     * into its containers once it is placed and the bench is activated, and how many world
     * ticks to observe at most.
     */
    public record ProcessingProbe(
        String benchBlockId,
        List<ItemQuantity> inputs,
        List<ItemQuantity> fuel,
        int maxTicks
    ) {
        public static final int DEFAULT_MAX_TICKS = 1200;
        public static final int MAX_TICKS = 3600;

        public ProcessingProbe {
            if (benchBlockId == null || benchBlockId.isBlank()) {
                throw new IllegalArgumentException(
                    "processing_oracle_requests bench_block_id must be nonblank"
                );
            }
            benchBlockId = benchBlockId.trim();
            inputs = inputs == null ? List.of() : List.copyOf(inputs);
            fuel = fuel == null ? List.of() : List.copyOf(fuel);
            if (inputs.isEmpty()) {
                throw new IllegalArgumentException(
                    "processing_oracle_requests needs at least one input"
                );
            }
            if (maxTicks < 1 || maxTicks > MAX_TICKS) {
                throw new IllegalArgumentException(
                    "processing_oracle_requests max_ticks must be 1..3600"
                );
            }
        }
    }

    public NativeCraftingOracleFixture(
        String benchBlockId,
        int benchTier,
        int memoriesLevel
    ) {
        this(benchBlockId, benchTier, memoriesLevel, List.of(), List.of());
    }

    public NativeCraftingOracleFixture(
        String benchBlockId,
        int benchTier,
        int memoriesLevel,
        List<CraftRequest> requests
    ) {
        this(benchBlockId, benchTier, memoriesLevel, requests, List.of());
    }

    public NativeCraftingOracleFixture {
        benchBlockId = benchBlockId == null || benchBlockId.isBlank()
            ? null
            : benchBlockId.trim();
        requests = requests == null ? List.of() : List.copyOf(requests);
        processing = processing == null ? List.of() : List.copyOf(processing);
        if (requests.size() > MAX_REQUESTS) {
            throw new IllegalArgumentException(
                "at most 16 crafting_oracle_requests per reset"
            );
        }
        if (processing.size() > MAX_REQUESTS) {
            throw new IllegalArgumentException(
                "at most 16 processing_oracle_requests per reset"
            );
        }
        if (benchBlockId == null) {
            if (benchTier != 0) {
                throw new IllegalArgumentException(
                    "crafting_bench_tier requires crafting_bench_block_id"
                );
            }
        } else if (
            benchTier < MIN_BENCH_TIER || benchTier > MAX_BENCH_TIER
        ) {
            throw new IllegalArgumentException(
                "crafting_bench_tier must be 1..8"
            );
        }
        if (memoriesLevel < 0 || memoriesLevel > MAX_MEMORIES_LEVEL) {
            throw new IllegalArgumentException(
                "memories_level must be 1..6, or absent"
            );
        }
        if (
            benchBlockId == null
                && requests.stream().anyMatch(
                    probe -> !FIELDCRAFT_CONTEXT.equals(probe.craftingContext())
                )
        ) {
            throw new IllegalArgumentException(
                "bench crafting_oracle_requests need crafting_bench_block_id"
            );
        }
        if (!processing.isEmpty() && (benchBlockId != null || !requests.isEmpty())) {
            throw new IllegalArgumentException(
                "processing_oracle_requests place their own bench; do not combine them with"
                    + " crafting_bench_block_id or crafting_oracle_requests"
            );
        }
    }

    public boolean hasBench() {
        return benchBlockId != null;
    }

    public boolean setsMemories() {
        return memoriesLevel > 0;
    }

    public boolean hasRequests() {
        return !requests.isEmpty();
    }

    public boolean hasProcessing() {
        return !processing.isEmpty();
    }

    /** Mirrors MemoriesPlugin.getMemoriesLevel for a recorded-memories count. */
    public static int memoriesLevelForCount(int count, int[] amountsPerLevel) {
        if (count < 0) {
            throw new IllegalArgumentException(
                "recorded memories count must be nonnegative"
            );
        }
        for (int index = amountsPerLevel.length - 1; index >= 0; index--) {
            if (count >= amountsPerLevel[index]) {
                return index + 2;
            }
        }
        return 1;
    }

    /** The smallest recorded-memories count at which the native level is {@code level}. */
    public static int memoriesCountForLevel(int level, int[] amountsPerLevel) {
        if (level < 1 || level > amountsPerLevel.length + 1) {
            throw new IllegalArgumentException(
                "memories level " + level + " is not reachable with "
                    + amountsPerLevel.length + " thresholds"
            );
        }
        return level == 1 ? 0 : amountsPerLevel[level - 2];
    }
}
