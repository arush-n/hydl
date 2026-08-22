package com.hytalerlbridge.policy.diagnostics.action.dodge;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Bounded JSONL writer for the opt-in live Dodge cooldown diagnostic. */
public final class DodgeCooldownTraceSink {

    public static final String SCHEMA = "hytalerl_dodge_cooldown_trace_v1";

    private static final System.Logger LOGGER =
        System.getLogger(DodgeCooldownTraceSink.class.getName());

    private final Path path;
    private final int maximumEvents;
    private int eventsWritten;
    private boolean disabled;
    private String failure = "";

    private DodgeCooldownTraceSink(Path path, int maximumEvents) {
        this.path = path;
        this.maximumEvents = maximumEvents;
    }

    public static DodgeCooldownTraceSink live(
        Path directory,
        int actorSlot,
        long seed,
        int maximumEvents
    ) throws IOException {
        if (directory == null || actorSlot < 1 || maximumEvents < 1) {
            throw new IllegalArgumentException(
                "Dodge trace requires a directory, nonzero actor, and limit");
        }
        Files.createDirectories(directory);
        Path path = directory.resolve(
            "dodge-cooldown-trace-" + System.currentTimeMillis()
                + "-slot" + actorSlot + "-seed" + seed + ".jsonl"
        );
        return new DodgeCooldownTraceSink(path, maximumEvents);
    }

    public static DodgeCooldownTraceSink disabled() {
        return new DodgeCooldownTraceSink(null, Integer.MAX_VALUE);
    }

    public synchronized Path path() {
        return path;
    }

    public synchronized int eventsWritten() {
        return eventsWritten;
    }

    public synchronized String failure() {
        return failure;
    }

    public synchronized void write(String fields) {
        if (path == null || disabled || eventsWritten >= maximumEvents) return;
        String line = "{\"schema\":\"" + SCHEMA + "\",\"event\":"
            + (eventsWritten + 1) + "," + fields + "}\n";
        try {
            Files.writeString(
                path,
                line,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND
            );
            eventsWritten++;
        } catch (IOException unavailable) {
            disabled = true;
            failure = unavailable.toString();
            LOGGER.log(
                System.Logger.Level.WARNING,
                "Dodge cooldown trace disabled after write failure: "
                    + unavailable
            );
        }
    }

    static String json(String value) {
        if (value == null) return "";
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r");
    }
}
