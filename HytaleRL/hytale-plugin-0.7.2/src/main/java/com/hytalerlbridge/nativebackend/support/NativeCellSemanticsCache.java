package com.hytalerlbridge.nativebackend.support;

import com.hytalerlbridge.nativebackend.model.NativeCellSemantics;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/** Bounded session cache for immutable block/fluid observation semantics. */
public final class NativeCellSemanticsCache {
    public static final String SCHEMA = "hytalerl_native_cell_semantics_cache_v1";
    public static final int DEFAULT_CAPACITY = 4096;

    private final int capacity;
    private final Map<Key, NativeCellSemantics> entries = new HashMap<>();
    private long hits;
    private long misses;
    private long generationResets;

    public NativeCellSemanticsCache() {
        this(DEFAULT_CAPACITY);
    }

    public NativeCellSemanticsCache(int capacity) {
        if (capacity < 1) throw new IllegalArgumentException("capacity must be positive");
        this.capacity = capacity;
    }

    public synchronized NativeCellSemantics lookup(Key key) {
        NativeCellSemantics value = entries.get(Objects.requireNonNull(key));
        if (value == null) misses++;
        else hits++;
        return value;
    }

    public synchronized NativeCellSemantics store(
        Key key,
        NativeCellSemantics value
    ) {
        Objects.requireNonNull(key);
        Objects.requireNonNull(value);
        NativeCellSemantics existing = entries.get(key);
        if (existing != null) return existing;
        if (entries.size() >= capacity) {
            entries.clear();
            generationResets++;
        }
        entries.put(key, value);
        return value;
    }

    public synchronized Stats stats() {
        return new Stats(
            capacity,
            entries.size(),
            hits,
            misses,
            generationResets
        );
    }

    public record Key(
        int runtimeBlockId,
        int runtimeFluidId,
        int rotationIndex,
        int fluidLevel,
        int supportValue
    ) {}

    public record Stats(
        int capacity,
        int entries,
        long hits,
        long misses,
        long generationResets
    ) {
        public double hitRate() {
            long requests = hits + misses;
            return requests == 0L ? 0.0 : (double) hits / requests;
        }
    }
}
