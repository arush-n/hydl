package com.hytalerlbridge.network.wire;

import static com.hytalerlbridge.network.wire.MessageFields.checkedInt;
import static com.hytalerlbridge.network.wire.MessageFields.getField;
import static com.hytalerlbridge.network.wire.MessageFields.getLongField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalStringField;
import static com.hytalerlbridge.network.wire.MessageFields.getStringField;

import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.CraftRequest;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.msgpack.value.Value;

/** Decodes the optional crafting-oracle reset fixture; absent keys mean no fixture. */
public final class CraftingOracleOptions {

    private CraftingOracleOptions() {}

    /**
     * Reads {@code crafting_bench_block_id}, {@code crafting_bench_tier} (default 1
     * when a bench is named), {@code memories_level}, and
     * {@code crafting_oracle_requests}: an array of maps with {@code recipe_id}
     * (required), {@code quantity} (default 1) and {@code crafting_context}
     * ({@code bench} or {@code fieldcraft}, default {@code bench}). Returns null when
     * none is present, so resets without these keys are unchanged.
     */
    public static NativeCraftingOracleFixture decode(Map<Value, Value> options) {
        String benchBlockId = getOptionalStringField(
            options,
            "crafting_bench_block_id"
        );
        long benchTier = getLongField(
            options,
            "crafting_bench_tier",
            benchBlockId == null ? 0L : 1L
        );
        long memoriesLevel = getLongField(options, "memories_level", 0L);
        List<CraftRequest> requests = decodeRequests(
            getField(options, "crafting_oracle_requests")
        );
        if (
            benchBlockId == null
                && benchTier == 0L
                && memoriesLevel == 0L
                && requests.isEmpty()
        ) {
            return null;
        }
        return new NativeCraftingOracleFixture(
            benchBlockId,
            checkedInt(benchTier, "crafting_bench_tier"),
            checkedInt(memoriesLevel, "memories_level"),
            requests
        );
    }

    private static List<CraftRequest> decodeRequests(Value raw) {
        if (raw == null || raw.isNilValue()) {
            return List.of();
        }
        if (!raw.isArrayValue()) {
            throw new IllegalArgumentException(
                "crafting_oracle_requests must be an array"
            );
        }
        List<CraftRequest> requests = new ArrayList<>();
        for (Value entry : raw.asArrayValue().list()) {
            if (!entry.isMapValue()) {
                throw new IllegalArgumentException(
                    "crafting_oracle_requests entries must be maps"
                );
            }
            Map<Value, Value> probe = entry.asMapValue().map();
            String context = getStringField(probe, "crafting_context");
            requests.add(new CraftRequest(
                getStringField(probe, "recipe_id"),
                checkedInt(getLongField(probe, "quantity", 1L), "quantity"),
                context.isEmpty() ? NativeCraftingOracleFixture.BENCH_CONTEXT : context
            ));
        }
        return requests;
    }
}
