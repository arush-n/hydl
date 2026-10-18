package com.hytalerlbridge.task.builtin;

import com.hytalerlbridge.task.TaskRegistry;

/** Registers a fresh task factory for every built-in environment. */
public final class BuiltinTasks {

    private BuiltinTasks() {
    }

    public static void registerAll(TaskRegistry registry) {
        registry.register("survive", SurviveTask::new);
        registry.register("mine_adamantite", MineAdamantiteTask::new);
        registry.register("kill_trork", KillTrorkTask::new);
        registry.register("build_house", BuildHouseTask::new);
        registry.register("navigate", NavigateTask::new);
        registry.register("base_builder", BaseBuilderTask::new);
    }
}
