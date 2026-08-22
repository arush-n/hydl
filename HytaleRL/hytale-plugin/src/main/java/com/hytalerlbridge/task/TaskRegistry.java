package com.hytalerlbridge.task;

import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Registry of available RL tasks.
 */
public class TaskRegistry {

    private final ConcurrentHashMap<String, Supplier<? extends RLTask>> tasks = new ConcurrentHashMap<>();

    public void register(String id, Supplier<? extends RLTask> taskFactory) {
        tasks.put(id, taskFactory);
    }

    public RLTask create(String id) {
        Supplier<? extends RLTask> factory = tasks.get(id);
        return factory == null ? null : factory.get();
    }

    public boolean has(String id) {
        return tasks.containsKey(id);
    }

    public int getTaskCount() {
        return tasks.size();
    }

    public java.util.Set<String> getTaskIds() {
        return tasks.keySet();
    }
}
