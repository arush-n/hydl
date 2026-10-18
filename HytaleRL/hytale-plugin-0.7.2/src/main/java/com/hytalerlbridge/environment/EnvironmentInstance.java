package com.hytalerlbridge.environment;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.task.RLTask;

/**
 * One isolated simulator environment for one connected RL client.
 *
 * This intentionally remains simulator-backed. A future live-world adapter can
 * implement RLTask without changing the network or Python-facing API.
 */
public final class EnvironmentInstance implements EnvironmentSession {

    private final RLTask task;
    private final int ticksPerStep;
    private final int maxEpisodeSteps;
    private final long seed;
    private final int curriculumPhase;
    private int currentTick;
    private int currentStep;
    private boolean done;

    public EnvironmentInstance(
        RLTask task,
        long seed,
        int curriculumPhase,
        int ticksPerStep,
        int maxEpisodeSteps
    ) {
        this.task = task;
        this.seed = seed;
        this.curriculumPhase = curriculumPhase;
        this.ticksPerStep = ticksPerStep;
        this.maxEpisodeSteps = maxEpisodeSteps;
    }

    @Override
    public void reset() {
        currentTick = 0;
        currentStep = 0;
        done = false;
        task.reset(seed, curriculumPhase);
    }

    @Override
    public StepResult observe() {
        return new StepResult(
            task.observe(), 0.0, false, false, currentTick, currentStep, task.info()
        );
    }

    @Override
    public StepResult step(AgentAction action) {
        if (done) {
            return new StepResult(
                task.observe(), 0.0, true, false, currentTick, currentStep, task.info()
            );
        }

        task.beginStep(action);
        AgentAction continuation = action.continuation();
        for (int i = 0; i < ticksPerStep; i++) {
            task.applyAction(i == 0 ? action : continuation);
            task.tick();
            currentTick++;
            if (task.isTerminated()) {
                break;
            }
        }
        currentStep++;

        double reward = task.computeReward();
        boolean terminated = task.isTerminated();
        boolean truncated = !terminated && maxEpisodeSteps > 0 && currentStep >= maxEpisodeSteps;
        done = terminated || truncated;

        return new StepResult(
            task.observe(), reward, terminated, truncated, currentTick, currentStep, task.info()
        );
    }

    @Override
    public void close() {
        // Simulator state is owned by this instance and is garbage-collected.
    }
}
