package com.hytalerlbridge.task;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.environment.EnvironmentOptions;
import com.hytalerlbridge.observation.Observation;
import java.util.Map;

/**
 * Interface for RL task definitions.
 * Each task defines the reward function, termination conditions,
 * and how actions affect the simulated world state.
 */
public interface RLTask {

    /** Human-readable task description. */
    String description();

    /** Reset the task state for a new episode. */
    void reset(long seed);

    /** Reset with an optional curriculum phase; -1 means the full task. */
    default void reset(long seed, int curriculumPhase) {
        reset(seed);
    }

    /** Configure backend-neutral reset options before reset is called. */
    default void configure(EnvironmentOptions options) {
        // Most simulator tasks do not consume environment options.
    }

    /** Called once before the engine ticks belonging to one RL step. */
    default void beginStep(AgentAction action) {
        // Most tasks have no step-scoped bookkeeping.
    }

    /** Apply an agent action to the current state. */
    void applyAction(AgentAction action);

    /** Advance the simulation by one tick. */
    void tick();

    /** Build the current observation. */
    Observation observe();

    /** Compute the reward for the current step. */
    double computeReward();

    /** Whether the episode has terminated (success or failure). */
    boolean isTerminated();

    /** Optional task-specific diagnostics returned in Gymnasium's info mapping. */
    default Map<String, Object> info() {
        return Map.of();
    }
}
