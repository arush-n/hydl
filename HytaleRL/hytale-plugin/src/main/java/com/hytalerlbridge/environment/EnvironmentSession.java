package com.hytalerlbridge.environment;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.action.group.NativeGroupAction;

/** Backend-neutral lifecycle for one isolated RL environment. */
public interface EnvironmentSession extends AutoCloseable {

    void reset();

    StepResult observe();

    StepResult step(AgentAction action);

    /**
     * Apply several reset-pinned actors in one shared engine step.
     *
     * <p>Backends that have not published group control remain explicitly
     * entity-zero-only instead of silently discarding extra actors.</p>
     */
    default StepResult step(NativeGroupAction actions) {
        if (
            actions.actors().size() != 1
                || actions.actors().getFirst().entityId() != 0
        ) {
            throw new IllegalStateException(
                "environment backend does not support native group actions"
            );
        }
        return step(actions.primaryAction());
    }

    @Override
    void close();
}
