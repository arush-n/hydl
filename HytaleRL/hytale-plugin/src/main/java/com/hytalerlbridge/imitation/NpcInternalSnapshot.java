package com.hytalerlbridge.imitation;

import java.util.List;
import java.util.UUID;

/** Mutable NPC support state that can influence the next authored decision. */
public record NpcInternalSnapshot(
    String stateName,
    int stateIndex,
    int subStateIndex,
    boolean busy,
    boolean transitioning,
    boolean roleChangeRequested,
    boolean terminalAction,
    boolean backingAway,
    String steeringMotion,
    boolean motionControllerPresent,
    boolean motionInProgress,
    boolean obstructed,
    double currentSpeed,
    double maximumSpeed,
    double[] avoidanceSteering,
    double[] separationSteering,
    List<MarkedTarget> markedTargets
) {
    public static final String LAYOUT =
        "state_name:utf8,state_index:i32,substate_index:i32,busy:bool,"
            + "transitioning:bool,role_change_requested:bool,terminal_action:bool,"
            + "backing_away:bool,steering_motion:utf8,motion_controller_present:bool,"
            + "motion_in_progress:bool,obstructed:bool,current_speed:f64,"
            + "maximum_speed:f64,avoidance_steering:f64[3],"
            + "separation_steering:f64[3],marked_targets:marked_target[]";

    public NpcInternalSnapshot {
        stateName = text(stateName);
        steeringMotion = text(steeringMotion);
        avoidanceSteering = vector(avoidanceSteering, "avoidanceSteering");
        separationSteering = vector(separationSteering, "separationSteering");
        markedTargets = markedTargets == null ? List.of() : List.copyOf(markedTargets);
        if (
            !Double.isFinite(currentSpeed)
                || !Double.isFinite(maximumSpeed)
                || currentSpeed < 0.0
                || maximumSpeed < 0.0
        ) {
            throw new IllegalArgumentException("motion speeds must be finite and nonnegative");
        }
        if (!motionControllerPresent && (
            motionInProgress
                || obstructed
                || currentSpeed != 0.0
                || maximumSpeed != 0.0
                || !steeringMotion.isEmpty()
        )) {
            throw new IllegalArgumentException(
                "missing motion controller cannot carry motion state"
            );
        }
    }

    @Override
    public double[] avoidanceSteering() {
        return avoidanceSteering.clone();
    }

    @Override
    public double[] separationSteering() {
        return separationSteering.clone();
    }

    public record MarkedTarget(int slot, String name, UUID uuid) {
        public MarkedTarget {
            if (slot < 0 || uuid == null) {
                throw new IllegalArgumentException("marked target identity is incomplete");
            }
            name = text(name);
        }
    }

    private static double[] vector(double[] value, String name) {
        if (value == null || value.length != 3) {
            throw new IllegalArgumentException(name + " must have width 3");
        }
        double[] copy = value.clone();
        for (double scalar : copy) {
            if (!Double.isFinite(scalar)) {
                throw new IllegalArgumentException(name + " must be finite");
            }
        }
        return copy;
    }

    private static String text(String value) {
        return value == null ? "" : value;
    }
}
