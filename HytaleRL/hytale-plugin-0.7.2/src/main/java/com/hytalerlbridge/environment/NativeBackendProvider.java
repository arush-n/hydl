package com.hytalerlbridge.environment;

/** Optional native backend, supplied only by the in-server plugin entry point. */
public interface NativeBackendProvider {

    EnvironmentSession create(
        String taskId,
        long seed,
        int curriculumPhase,
        int ticksPerStep,
        int maxEpisodeSteps,
        EnvironmentOptions options
    );
}
