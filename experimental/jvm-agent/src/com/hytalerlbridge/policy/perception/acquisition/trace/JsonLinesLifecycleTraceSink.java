package com.hytalerlbridge.policy.perception.acquisition.trace;

import com.hytalerlbridge.policy.perception.acquisition.CombatLifecycleEvidenceSource;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;

/**
 * Bounded, change-only JSONL trace for the native attack lifecycle.
 *
 * <p>This recorder is deliberately outside the lifecycle source. It retains
 * only diagnostic history needed to de-duplicate rows and measure a cursor
 * interval; none of its state is read by perception, legality, or the action
 * sink. I/O failures disable the trace and never suppress a policy tick.</p>
 */
public final class JsonLinesLifecycleTraceSink implements LifecycleTraceSink {

    public static final String SCHEMA = "hytale_policy_lifecycle_trace_v1";

    private static final System.Logger LOGGER =
        System.getLogger(JsonLinesLifecycleTraceSink.class.getName());

    private final Path path;
    private final int maximumEvents;
    private final Map<Integer, SlotState> slots = new HashMap<>();
    private int eventsWritten;
    private boolean disabled;

    public JsonLinesLifecycleTraceSink(Path directory, int maximumEvents)
        throws IOException {
        if (directory == null || maximumEvents < 1) {
            throw new IllegalArgumentException(
                "lifecycle trace requires a directory and positive limit");
        }
        Files.createDirectories(directory);
        path = directory.resolve(
            "lifecycle-trace-" + System.currentTimeMillis() + ".jsonl");
        this.maximumEvents = maximumEvents;
    }

    /** Unique trace file for this process start. */
    public Path path() {
        return path;
    }

    /** Number of change events successfully written. */
    public synchronized int eventsWritten() {
        return eventsWritten;
    }

    @Override
    public synchronized void observe(
        int policySlot,
        CombatLifecycleEvidenceSource.Evidence evidence
    ) {
        if (disabled || eventsWritten >= maximumEvents || evidence == null) {
            return;
        }
        SlotState state = slots.computeIfAbsent(
            policySlot, ignored -> new SlotState());
        long sample = ++state.samples;
        TraceKey current = TraceKey.from(evidence);
        TraceKey previous = state.previous;
        if (previous != null && current.sameDiscreteState(previous)) {
            // The pause clock changes every tick. Retain its newest value for
            // the next transition, but do not turn a 1-2 second pause into
            // 30-60 duplicate file writes.
            state.previous = current;
            return;
        }

        boolean enteringCursor = current.cursorSource()
            && (previous == null || !previous.cursorSource());
        boolean leavingCursor = previous != null
            && previous.cursorSource()
            && !current.cursorSource();
        if (enteringCursor) {
            state.cursorStartSample = sample;
        }
        long completedCursorTicks = leavingCursor
            && state.cursorStartSample >= 0
            ? sample - state.cursorStartSample
            : -1;
        boolean activeToCursorSameSlot = previous != null
            && CombatLifecycleEvidenceSource
                .TARGET_ATTACK_SOURCE_ACTIVE_CHAIN.equals(previous.source())
            && current.cursorSource()
            && previous.reportedSlot() == current.reportedSlot();

        String transition = (previous == null ? "start" : previous.source())
            + "->" + current.source();
        String line = "{"
            + "\"schema\":\"" + SCHEMA + "\","
            + "\"event\":" + (eventsWritten + 1) + ","
            + "\"policy_slot\":" + policySlot + ","
            + "\"sample\":" + sample + ","
            + "\"transition\":\"" + transition + "\","
            + "\"source\":\"" + current.source() + "\","
            + "\"reported_ability_slot\":" + current.reportedSlot() + ","
            + "\"cursor\":" + current.cursor() + ","
            + "\"sequence_count\":" + current.count() + ","
            + "\"pause_seconds\":" + Float.toString(current.pauseSeconds())
            + ",\"active_to_cursor_same_slot\":"
            + activeToCursorSameSlot
            + ",\"completed_cursor_ticks\":" + completedCursorTicks
            + "}\n";
        try {
            Files.writeString(
                path,
                line,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND
            );
            eventsWritten++;
            state.previous = current;
            if (leavingCursor) {
                state.cursorStartSample = -1;
            }
        } catch (IOException failure) {
            disabled = true;
            LOGGER.log(
                System.Logger.Level.WARNING,
                "lifecycle trace disabled after write failure: " + failure
            );
        }
    }

    private static final class SlotState {
        private long samples;
        private long cursorStartSample = -1;
        private TraceKey previous;
    }

    private record TraceKey(
        String source,
        int reportedSlot,
        int cursor,
        int count,
        float pauseSeconds
    ) {
        private static TraceKey from(
            CombatLifecycleEvidenceSource.Evidence evidence
        ) {
            return new TraceKey(
                evidence.targetAttackEvidenceSource(),
                evidence.targetReportedAbilitySlot(),
                evidence.targetAttackSequenceCursor(),
                evidence.targetAttackSequenceCount(),
                evidence.targetAttackPauseSeconds()
            );
        }

        private boolean cursorSource() {
            return CombatLifecycleEvidenceSource
                .TARGET_ATTACK_SOURCE_ACTION_LIST_CURSOR.equals(source);
        }

        private boolean sameDiscreteState(TraceKey other) {
            return source.equals(other.source)
                && reportedSlot == other.reportedSlot
                && cursor == other.cursor
                && count == other.count;
        }
    }
}
