package com.hytalerlbridge.network.wire;

import static com.hytalerlbridge.network.wire.MessageFields.checkedInt;
import static com.hytalerlbridge.network.wire.MessageFields.getBooleanField;
import static com.hytalerlbridge.network.wire.MessageFields.getField;
import static com.hytalerlbridge.network.wire.MessageFields.getLongField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalStringField;
import static com.hytalerlbridge.network.wire.MessageFields.getStringField;

import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.CraftRequest;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.ItemQuantity;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.ProbeEvent;
import com.hytalerlbridge.environment.fidelity.NativeCraftingOracleFixture.ProcessingProbe;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.msgpack.value.Value;

/** Decodes the optional crafting-oracle reset fixture; absent keys mean no fixture. */
public final class CraftingOracleOptions {

    private CraftingOracleOptions() {}

    /**
     * Reads {@code crafting_bench_block_id}, {@code crafting_bench_tier} (default 1
     * when a bench is named), {@code memories_level}, {@code crafting_oracle_requests}
     * and {@code processing_oracle_requests}. A craft request is a map with {@code recipe_id}
     * (required), {@code quantity} (default 1), {@code crafting_context} ({@code bench},
     * {@code fieldcraft} or {@code diagram}, default {@code bench}), {@code open_window}
     * (default false), {@code events} (maps with {@code tick}, world ticks from the request,
     * and {@code kind}: {@code reach_out}, {@code second_craft} or {@code damage}) and, for a
     * diagram probe, {@code category} and {@code item_category}. A processing request is a
     * map with {@code bench_block_id} (required), {@code inputs} and {@code fuel} (arrays of
     * {@code item_id}, {@code quantity}) and {@code max_ticks} (default 1200). Returns null
     * when none is present, so resets without these keys are unchanged.
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
        List<ProcessingProbe> processing = decodeProcessing(
            getField(options, "processing_oracle_requests")
        );
        if (
            benchBlockId == null
                && benchTier == 0L
                && memoriesLevel == 0L
                && requests.isEmpty()
                && processing.isEmpty()
        ) {
            return null;
        }
        return new NativeCraftingOracleFixture(
            benchBlockId,
            checkedInt(benchTier, "crafting_bench_tier"),
            checkedInt(memoriesLevel, "memories_level"),
            requests,
            processing
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
                context.isEmpty() ? NativeCraftingOracleFixture.BENCH_CONTEXT : context,
                getBooleanField(probe, "open_window", false),
                decodeEvents(getField(probe, "events")),
                getStringField(probe, "category"),
                getStringField(probe, "item_category")
            ));
        }
        return requests;
    }

    private static List<ProbeEvent> decodeEvents(Value raw) {
        if (raw == null || raw.isNilValue()) {
            return List.of();
        }
        if (!raw.isArrayValue()) {
            throw new IllegalArgumentException("crafting_oracle_requests events must be an array");
        }
        List<ProbeEvent> events = new ArrayList<>();
        for (Value entry : raw.asArrayValue().list()) {
            if (!entry.isMapValue()) {
                throw new IllegalArgumentException("crafting_oracle_requests events entries must be maps");
            }
            Map<Value, Value> event = entry.asMapValue().map();
            events.add(new ProbeEvent(
                checkedInt(getLongField(event, "tick", -1L), "event tick"),
                getStringField(event, "kind")
            ));
        }
        return events;
    }

    private static List<ProcessingProbe> decodeProcessing(Value raw) {
        if (raw == null || raw.isNilValue()) {
            return List.of();
        }
        if (!raw.isArrayValue()) {
            throw new IllegalArgumentException("processing_oracle_requests must be an array");
        }
        List<ProcessingProbe> probes = new ArrayList<>();
        for (Value entry : raw.asArrayValue().list()) {
            if (!entry.isMapValue()) {
                throw new IllegalArgumentException("processing_oracle_requests entries must be maps");
            }
            Map<Value, Value> probe = entry.asMapValue().map();
            probes.add(new ProcessingProbe(
                getStringField(probe, "bench_block_id"),
                decodeItems(getField(probe, "inputs"), "processing_oracle_requests inputs"),
                decodeItems(getField(probe, "fuel"), "processing_oracle_requests fuel"),
                checkedInt(
                    getLongField(probe, "max_ticks", ProcessingProbe.DEFAULT_MAX_TICKS),
                    "max_ticks"
                )
            ));
        }
        return probes;
    }

    private static List<ItemQuantity> decodeItems(Value raw, String name) {
        if (raw == null || raw.isNilValue()) {
            return List.of();
        }
        if (!raw.isArrayValue()) {
            throw new IllegalArgumentException(name + " must be an array");
        }
        List<ItemQuantity> items = new ArrayList<>();
        for (Value entry : raw.asArrayValue().list()) {
            if (!entry.isMapValue()) {
                throw new IllegalArgumentException(name + " entries must be maps");
            }
            Map<Value, Value> item = entry.asMapValue().map();
            items.add(new ItemQuantity(
                getStringField(item, "item_id"),
                checkedInt(getLongField(item, "quantity", 1L), "quantity")
            ));
        }
        return items;
    }
}
