package com.hytalerlbridge.environment.fidelity;

import java.util.List;

/**
 * Reset-time fixture for the crafting oracle: an optional bench placed in reach at
 * a native tier, the native memories level, and an ordered list of craft_recipe
 * probes that run at reset after those fixtures apply.
 *
 * <p>A bench tier is the {@code BenchBlock.tierLevel} component of the placed block
 * entity. A fresh bench starts at 1 and the native upgrade table reaches 8, so 1..8
 * is the native range. The memories level is
 * {@code MemoriesPlugin.getMemoriesLevel}: {@code i + 2} for the highest index
 * {@code i} with {@code count >= MemoriesAmountPerLevel[i]}, otherwise 1. The count
 * is the universe's recorded-memories set, so the level is not per actor.</p>
 *
 * <p>A probe is an ordinary craft_recipe request run through the legacy craft path
 * without a policy-capture candidate. Its execution scope is labelled
 * {@code oracle_unbound_*}. The production step path and its evidence gates are
 * unchanged.</p>
 *
 * @param benchBlockId block id of the bench to place, or null for no bench
 * @param benchTier tier to set on the placed bench (1..8), or 0 when none
 * @param memoriesLevel level to set (1..6), or 0 to leave the native level
 * @param requests probes to run at reset, in order; empty for none
 */
public record NativeCraftingOracleFixture(
    String benchBlockId,
    int benchTier,
    int memoriesLevel,
    List<CraftRequest> requests
) {
    public static final int MIN_BENCH_TIER = 1;
    public static final int MAX_BENCH_TIER = 8;
    public static final int MAX_MEMORIES_LEVEL = 6;
    public static final int MAX_REQUESTS = 16;
    public static final String BENCH_CONTEXT = "bench";
    public static final String FIELDCRAFT_CONTEXT = "fieldcraft";

    /** One probe: the recipe, its quantity, and its crafting context (bench or fieldcraft). */
    public record CraftRequest(String recipeId, int quantity, String craftingContext) {

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
            ) {
                throw new IllegalArgumentException(
                    "crafting_oracle_requests crafting_context must be bench or fieldcraft"
                );
            }
        }
    }

    public NativeCraftingOracleFixture(
        String benchBlockId,
        int benchTier,
        int memoriesLevel
    ) {
        this(benchBlockId, benchTier, memoriesLevel, List.of());
    }

    public NativeCraftingOracleFixture {
        benchBlockId = benchBlockId == null || benchBlockId.isBlank()
            ? null
            : benchBlockId.trim();
        requests = requests == null ? List.of() : List.copyOf(requests);
        if (requests.size() > MAX_REQUESTS) {
            throw new IllegalArgumentException(
                "at most 16 crafting_oracle_requests per reset"
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
                    probe -> BENCH_CONTEXT.equals(probe.craftingContext())
                )
        ) {
            throw new IllegalArgumentException(
                "bench crafting_oracle_requests need crafting_bench_block_id"
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
