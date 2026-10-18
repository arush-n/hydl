package com.hytalerlbridge.nativebackend;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Per-request ledger for the crafting oracle: the container slots that changed
 * with the tick each change first and last lands, the per-tick lifecycle flags,
 * and the JSON record. It takes plain values so it can be tested without the
 * server; the server-side snapshots live in {@link NativeCraftingOracle}.
 */
final class NativeCraftingOracleLedger {

    static final String SCHEMA = "hytalerl_crafting_oracle_v1";

    /** One container slot: the item asset id ("" when empty) and its quantity. */
    record Slot(String container, int slot, String itemId, int quantity) {

        Slot {
            if (container == null || container.isEmpty()) {
                throw new IllegalArgumentException("slot container must be named");
            }
            if (slot < 0) {
                throw new IllegalArgumentException("slot index must be nonnegative");
            }
            if (itemId == null || quantity < 0) {
                throw new IllegalArgumentException("slot item must be a string with a nonnegative quantity");
            }
        }

        static Slot empty(String container, int slot) {
            return new Slot(container, slot, "", 0);
        }

        String key() {
            return container + ":" + slot;
        }
    }

    /** The bench a request resolved against; the ids are native asset ids. */
    record BenchReadout(String blockId, String benchId, int type, int tier, int x, int y, int z) {}

    /** One observed tick: whether the craft was still active, and whether it had finished. */
    record Lifecycle(long tick, boolean active, boolean finished) {}

    private final String recipeId;
    private final String craftingContext;
    private final BenchReadout bench;
    private final long requestedTick;
    private final Map<String, Slot> before = new LinkedHashMap<>();
    private final Map<String, Slot> latest = new LinkedHashMap<>();
    private final Map<String, Slot> shape = new LinkedHashMap<>();
    private final Map<String, Long> firstChanged = new LinkedHashMap<>();
    private final Map<String, Long> lastChanged = new LinkedHashMap<>();
    private final List<Lifecycle> lifecycle = new ArrayList<>();
    private String status = "pending";
    private String rejectReason = "";
    private String executionScope = "";

    NativeCraftingOracleLedger(
        String recipeId,
        String craftingContext,
        BenchReadout bench,
        long requestedTick,
        List<Slot> beforeSlots
    ) {
        this.recipeId = recipeId == null ? "" : recipeId;
        this.craftingContext = craftingContext == null ? "" : craftingContext;
        this.bench = bench;
        this.requestedTick = requestedTick;
        for (Slot slot : beforeSlots) {
            before.put(slot.key(), slot);
            latest.put(slot.key(), slot);
            shape.put(slot.key(), slot);
        }
    }

    /** The request was refused before any native craft began. */
    void refuse(String reason, long tick) {
        if (!"pending".equals(status)) {
            throw new IllegalStateException("crafting oracle already " + status);
        }
        status = "refused";
        rejectReason = reason == null ? "" : reason;
        recordLifecycle(new Lifecycle(tick, false, false));
    }

    /**
     * Records one tick. A slot's first change tick is the first tick it differs
     * from its previous observed state; its last change tick is the latest such tick.
     * Observations after the craft finished or was refused are ignored.
     */
    void observe(long tick, List<Slot> now, boolean active) {
        if (!"pending".equals(status)) {
            return;
        }
        for (Slot slot : now) {
            String key = slot.key();
            Slot previous = latest.getOrDefault(
                key,
                Slot.empty(slot.container(), slot.slot())
            );
            shape.putIfAbsent(key, slot);
            if (!previous.equals(slot)) {
                firstChanged.putIfAbsent(key, tick);
                lastChanged.put(key, tick);
            }
            latest.put(key, slot);
        }
        recordLifecycle(new Lifecycle(tick, active, !active));
        if (!active) {
            status = "finished";
        }
    }

    /** The execution scope the craft was accepted under; empty when refused. */
    void executionScope(String scope) {
        executionScope = scope == null ? "" : scope;
    }

    String status() {
        return status;
    }

    String rejectReason() {
        return rejectReason;
    }

    long requestedTick() {
        return requestedTick;
    }

    private void recordLifecycle(Lifecycle entry) {
        int last = lifecycle.size() - 1;
        if (last >= 0 && lifecycle.get(last).tick() == entry.tick()) {
            lifecycle.set(last, entry);
        } else {
            lifecycle.add(entry);
        }
    }

    String toJson() {
        StringBuilder out = new StringBuilder(256);
        out.append('{');
        appendField(out, "schema", SCHEMA);
        out.append(',');
        appendField(out, "status", status);
        out.append(',');
        appendField(out, "reject_reason", rejectReason);
        out.append(',');
        appendField(out, "recipe_id", recipeId);
        out.append(',');
        appendField(out, "crafting_context", craftingContext);
        out.append(',');
        appendField(out, "execution_scope", executionScope);
        out.append(",\"requested_tick\":").append(requestedTick);
        out.append(",\"bench\":");
        if (bench == null) {
            out.append("null");
        } else {
            out.append('{');
            appendField(out, "block_id", bench.blockId());
            out.append(',');
            appendField(out, "bench_id", bench.benchId());
            out.append(",\"type\":").append(bench.type());
            out.append(",\"tier\":").append(bench.tier());
            out.append(",\"x\":").append(bench.x());
            out.append(",\"y\":").append(bench.y());
            out.append(",\"z\":").append(bench.z());
            out.append('}');
        }
        out.append(",\"lifecycle\":[");
        for (int index = 0; index < lifecycle.size(); index++) {
            Lifecycle entry = lifecycle.get(index);
            if (index > 0) out.append(',');
            out.append("{\"tick\":").append(entry.tick())
                .append(",\"active\":").append(entry.active())
                .append(",\"finished\":").append(entry.finished())
                .append('}');
        }
        out.append("],\"slots\":[");
        boolean first = true;
        for (Map.Entry<String, Long> changed : firstChanged.entrySet()) {
            String key = changed.getKey();
            Slot start = before.getOrDefault(key, Slot.empty(shape.get(key).container(), shape.get(key).slot()));
            Slot end = latest.get(key);
            if (!first) out.append(',');
            first = false;
            out.append('{');
            appendField(out, "container", start.container());
            out.append(",\"slot\":").append(start.slot());
            out.append(',');
            appendField(out, "before_item_id", start.itemId());
            out.append(",\"before_quantity\":").append(start.quantity());
            out.append(',');
            appendField(out, "after_item_id", end.itemId());
            out.append(",\"after_quantity\":").append(end.quantity());
            out.append(",\"first_changed_tick\":").append(changed.getValue());
            out.append(",\"last_changed_tick\":").append(lastChanged.get(key));
            out.append('}');
        }
        out.append("]}");
        return out.toString();
    }

    private static void appendField(StringBuilder out, String name, String value) {
        out.append('"').append(name).append("\":\"").append(escape(value)).append('"');
    }

    static String escape(String value) {
        StringBuilder out = new StringBuilder(value.length() + 8);
        for (int index = 0; index < value.length(); index++) {
            char c = value.charAt(index);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        return out.toString();
    }
}
