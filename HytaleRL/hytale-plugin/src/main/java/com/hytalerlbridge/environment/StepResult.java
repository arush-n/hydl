package com.hytalerlbridge.environment;

import com.hytalerlbridge.observation.Observation;
import java.util.Map;

/** Result of one Gymnasium step, including bridge-side counters for diagnostics. */
public record StepResult(
    Observation observation,
    double reward,
    boolean terminated,
    boolean truncated,
    int tick,
    int step,
    Map<String, Object> info
) {
    public static StepResult initial(Observation observation) {
        return new StepResult(observation, 0.0, false, false, 0, 0, Map.of());
    }
}
