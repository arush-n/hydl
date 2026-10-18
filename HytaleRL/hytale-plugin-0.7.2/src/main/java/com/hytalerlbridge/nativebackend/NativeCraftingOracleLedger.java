package com.hytalerlbridge.nativebackend;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Per-request ledger for the crafting oracle: the container slots that changed
 * with the tick each change first and last lands, the per-tick lifecycle (flags,
 * elapsed ticks, native craft progress and that tick's slot changes), the cancel
 * and the applied scripted events, and the JSON record. It takes plain values so it
 * can be tested without the server; the server-side snapshots live in
 * {@link NativeCraftingOracle}.
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

    /** One slot whose item or quantity changed on a tick, with its state before and after. */
    record Change(
        String container,
        int slot,
        String beforeItemId,
        int beforeQuantity,
        String afterItemId,
        int afterQuantity
    ) {}

    /**
     * One observed tick: whether the craft was still active, whether it had finished or
     * been cancelled, the native craft progress in seconds (-1 when no timed craft is
     * queued, -2 when the native job cannot be read), and the slot changes of that tick.
     */
    record Lifecycle(
        long tick,
        boolean active,
        boolean finished,
        boolean cancelled,
        double progressSeconds,
        List<Change> changes
    ) {}

    /** One scripted event as it was applied, or not, at a world tick. */
    record Event(long tick, String kind, boolean applied, String detail) {}

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
    private final List<Event> events = new ArrayList<>();
    private String status = "pending";
    private String rejectReason = "";
    private String cancelReason = "";
    private String executionScope = "";
    private long firstRemovedTick = -1L;
    private long firstAddedTick = -1L;
    private String extraFields = "";

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
        recordLifecycle(new Lifecycle(tick, false, false, false, -1.0, List.of()));
    }

    void observe(long tick, List<Slot> now, boolean active) {
        observe(tick, now, active, -1.0);
    }

    /**
     * Records one tick. A slot's first change tick is the first tick it differs
     * from its previous observed state; its last change tick is the latest such tick.
     * Observations after the craft finished or was cancelled are ignored.
     */
    void observe(long tick, List<Slot> now, boolean active, double progressSeconds) {
        if (!"pending".equals(status)) {
            return;
        }
        List<Change> changes = applySlots(tick, now);
        recordLifecycle(new Lifecycle(tick, active, !active, false, progressSeconds, changes));
        if (!active) {
            status = "finished";
        }
    }

    /** The craft was cancelled on this tick; the slot changes of the cancel (a refund) land here. */
    void cancel(String reason, long tick, List<Slot> now, double progressSeconds) {
        if (!"pending".equals(status)) {
            throw new IllegalStateException("crafting oracle already " + status);
        }
        List<Change> changes = applySlots(tick, now);
        status = "cancelled";
        cancelReason = reason == null ? "" : reason;
        recordLifecycle(new Lifecycle(tick, false, false, true, progressSeconds, changes));
    }

    /** Records one scripted event, applied or not, at a world tick. */
    void event(long tick, String kind, boolean applied, String detail) {
        events.add(new Event(tick, kind, applied, detail == null ? "" : detail));
    }

    /** The execution scope the craft was accepted under; empty when refused. */
    void executionScope(String scope) {
        executionScope = scope == null ? "" : scope;
    }

    /** Pre-encoded JSON members, each starting with a comma, appended to the record; empty by default. */
    void extraFields(String members) {
        extraFields = members == null ? "" : members;
    }

    String status() {
        return status;
    }

    String rejectReason() {
        return rejectReason;
    }

    String cancelReason() {
        return cancelReason;
    }

    long requestedTick() {
        return requestedTick;
    }

    private List<Change> applySlots(long tick, List<Slot> now) {
        List<Change> changes = new ArrayList<>();
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
                changes.add(new Change(
                    slot.container(),
                    slot.slot(),
                    previous.itemId(),
                    previous.quantity(),
                    slot.itemId(),
                    slot.quantity()
                ));
                boolean removed = !previous.itemId().isEmpty()
                    && (!previous.itemId().equals(slot.itemId())
                        || slot.quantity() < previous.quantity());
                boolean added = !slot.itemId().isEmpty()
                    && (!previous.itemId().equals(slot.itemId())
                        || slot.quantity() > previous.quantity());
                if (removed && firstRemovedTick < 0L) {
                    firstRemovedTick = tick;
                }
                if (added && firstAddedTick < 0L) {
                    firstAddedTick = tick;
                }
            }
            latest.put(key, slot);
        }
        return changes;
    }

    private void recordLifecycle(Lifecycle entry) {
        int last = lifecycle.size() - 1;
        if (last >= 0 && lifecycle.get(last).tick() == entry.tick()) {
            // Two observations on one tick (the ticking system, then the telemetry
            // finish): keep the changes of both, and the later flags.
            List<Change> merged = new ArrayList<>(lifecycle.get(last).changes());
            merged.addAll(entry.changes());
            lifecycle.set(last, new Lifecycle(
                entry.tick(),
                entry.active(),
                entry.finished(),
                entry.cancelled(),
                entry.progressSeconds(),
                merged
            ));
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
        appendField(out, "cancel_reason", cancelReason);
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
        out.append(",\"first_removed_tick\":").append(firstRemovedTick);
        out.append(",\"first_added_tick\":").append(firstAddedTick);
        out.append(",\"lifecycle\":[");
        for (int index = 0; index < lifecycle.size(); index++) {
            Lifecycle entry = lifecycle.get(index);
            if (index > 0) out.append(',');
            out.append("{\"tick\":").append(entry.tick())
                .append(",\"active\":").append(entry.active())
                .append(",\"finished\":").append(entry.finished())
                .append(",\"cancelled\":").append(entry.cancelled())
                .append(",\"elapsed_ticks\":").append(entry.tick() - requestedTick)
                .append(",\"progress_seconds\":")
                .append(String.format(Locale.ROOT, "%.6f", entry.progressSeconds()))
                .append(",\"changes\":[");
            for (int change = 0; change < entry.changes().size(); change++) {
                Change c = entry.changes().get(change);
                if (change > 0) out.append(',');
                out.append('{');
                appendField(out, "container", c.container());
                out.append(",\"slot\":").append(c.slot()).append(',');
                appendField(out, "before_item_id", c.beforeItemId());
                out.append(",\"before_quantity\":").append(c.beforeQuantity()).append(',');
                appendField(out, "after_item_id", c.afterItemId());
                out.append(",\"after_quantity\":").append(c.afterQuantity()).append('}');
            }
            out.append("]}");
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
        out.append("],\"events\":[");
        for (int index = 0; index < events.size(); index++) {
            Event event = events.get(index);
            if (index > 0) out.append(',');
            out.append("{\"tick\":").append(event.tick()).append(',');
            appendField(out, "kind", event.kind());
            out.append(",\"applied\":").append(event.applied()).append(',');
            appendField(out, "detail", event.detail());
            out.append('}');
        }
        out.append(']').append(extraFields).append('}');
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
