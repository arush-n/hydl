package com.hytalerlbridge.imitation;

import java.util.List;
import java.util.UUID;
import com.hytalerlbridge.observation.NativeActorEvidenceFrame;

/** One native NPC decision paired with authoritative pre/post tick state. */
public record NpcTraceFrame(
    long tick,
    float deltaSeconds,
    double[] state,
    double[] control,
    int controlMask,
    double[] nextState,
    boolean nativeControl,
    boolean attackActionActive,
    boolean attackActivation,
    NpcAttackExecutionCause attackExecutionCause,
    boolean combatAttack,
    boolean nextCombatAttack,
    float attackPauseSeconds,
    float nextAttackPauseSeconds,
    UUID targetUuid,
    UUID nextTargetUuid,
    UUID decisionTargetUuid,
    String stateName,
    String nextStateName,
    NpcInternalSnapshot internalState,
    NpcInternalSnapshot decisionInternalState,
    NpcInternalSnapshot nextInternalState,
    String bodyInstruction,
    String bodyDecisionPath,
    String headInstruction,
    String headDecisionPath,
    List<String> activeActions,
    List<String> activeActionPaths,
    List<NpcAttackActionSnapshot> attackActions,
    List<NpcAttackActionSnapshot> nextAttackActions,
    List<NpcInteractionSnapshot> interactions,
    List<NpcInteractionSnapshot> nextInteractions,
    List<NpcDamageEventSnapshot> damageEvents,
    int damageEventCount,
    boolean damageEventOverflow,
    List<NpcLifecycleEventSnapshot> lifecycleEvents,
    int lifecycleEventCount,
    boolean lifecycleEventOverflow,
    int lifecycleSourceAvailableBits,
    int lifecycleSourcePartialBits,
    NpcObservationSnapshot observation,
    NpcObservationSnapshot nextObservation,
    NpcWorldSnapshot worldview,
    NpcWorldSnapshot nextWorldview
) {
    public static final int STATE_WIDTH = 14;
    public static final int CONTROL_WIDTH = 9;
    public static final String STATE_LAYOUT =
        "x,y,z,vx,vy,vz,body_yaw,body_pitch,body_roll,head_yaw,head_pitch,health,max_health,on_ground";
    public static final String CONTROL_LAYOUT =
        "body_x,body_y,body_z,body_yaw,body_pitch,head_yaw,head_pitch,body_turn_speed,head_turn_speed";
    public static final String CONTROL_MASK_LAYOUT =
        "bit0=body_translation,bit1=body_yaw,bit2=body_pitch,bit3=head_yaw,bit4=head_pitch,bit5=body_steering_present,bit6=head_steering_present";

    public NpcTraceFrame {
        state = finiteRow(state, STATE_WIDTH, "state");
        control = finiteRow(control, CONTROL_WIDTH, "control");
        nextState = finiteRow(nextState, STATE_WIDTH, "nextState");
        if (!Float.isFinite(deltaSeconds) || deltaSeconds < 0.0f) {
            throw new IllegalArgumentException(
                "deltaSeconds must be finite and nonnegative"
            );
        }
        if (
            !Float.isFinite(attackPauseSeconds)
                || attackPauseSeconds < 0.0f
                || !Float.isFinite(nextAttackPauseSeconds)
                || nextAttackPauseSeconds < 0.0f
        ) {
            throw new IllegalArgumentException(
                "attack pause times must be finite and nonnegative"
            );
        }
        if ((controlMask & ~0x7f) != 0) {
            throw new IllegalArgumentException("controlMask has unknown bits");
        }
        if (attackExecutionCause == null) {
            throw new IllegalArgumentException("attack execution cause is required");
        }
        stateName = text(stateName);
        nextStateName = text(nextStateName);
        if (
            internalState == null
                || decisionInternalState == null
                || nextInternalState == null
        ) {
            throw new IllegalArgumentException("internal state must cover all boundaries");
        }
        bodyInstruction = text(bodyInstruction);
        bodyDecisionPath = text(bodyDecisionPath);
        headInstruction = text(headInstruction);
        headDecisionPath = text(headDecisionPath);
        activeActions = activeActions == null
            ? List.of()
            : activeActions.stream().map(NpcTraceFrame::text).distinct().toList();
        activeActionPaths = activeActionPaths == null
            ? List.of()
            : activeActionPaths.stream().map(NpcTraceFrame::text).distinct().toList();
        attackActions = attackActions == null ? List.of() : List.copyOf(attackActions);
        nextAttackActions = nextAttackActions == null
            ? List.of()
            : List.copyOf(nextAttackActions);
        interactions = interactions == null ? List.of() : List.copyOf(interactions);
        nextInteractions = nextInteractions == null
            ? List.of()
            : List.copyOf(nextInteractions);
        damageEvents = damageEvents == null ? List.of() : List.copyOf(damageEvents);
        if (
            damageEvents.size() > NpcDamageEventSnapshot.CAPACITY
                || damageEventCount < damageEvents.size()
                || damageEventOverflow != (damageEventCount > damageEvents.size())
        ) {
            throw new IllegalArgumentException("damage event count/overflow is invalid");
        }
        lifecycleEvents = lifecycleEvents == null
            ? List.of()
            : List.copyOf(lifecycleEvents);
        if (
            lifecycleEvents.size() > NpcLifecycleEventSnapshot.CAPACITY
                || lifecycleEventCount < lifecycleEvents.size()
                || lifecycleEventOverflow
                    != (lifecycleEventCount > lifecycleEvents.size())
                || (lifecycleSourceAvailableBits
                    & ~NpcLifecycleEventSnapshot.ALL_SOURCE_BITS) != 0
                || (lifecycleSourcePartialBits
                    & ~lifecycleSourceAvailableBits) != 0
        ) {
            throw new IllegalArgumentException(
                "lifecycle event count, overflow, or source bits are invalid"
            );
        }
        if (observation == null || nextObservation == null) {
            throw new IllegalArgumentException("observation evidence must be stateful");
        }
        if (worldview == null || nextWorldview == null) {
            throw new IllegalArgumentException("worldview must be stateful");
        }
        if (
            observation.worldTick() != tick
                || worldview.worldTick() != tick
                || nextObservation.worldTick() != nextWorldview.worldTick()
        ) {
            throw new IllegalArgumentException("observation boundaries disagree on tick");
        }
        if (!stateName.equals(internalState.stateName())) {
            throw new IllegalArgumentException("stateName must describe pre-decision state");
        }
        if (!nextStateName.equals(nextInternalState.stateName())) {
            throw new IllegalArgumentException("nextStateName must describe post-tick state");
        }
    }

    public NativeActorEvidenceFrame.Actor actorEvidence() {
        return observation.actorEvidence();
    }

    public NativeActorEvidenceFrame.Actor nextActorEvidence() {
        return nextObservation.actorEvidence();
    }

    public NativeActorEvidenceFrame.Actor targetActorEvidence() {
        return observation.targetActorEvidence();
    }

    public NativeActorEvidenceFrame.Actor nextTargetActorEvidence() {
        return nextObservation.targetActorEvidence();
    }

    private static double[] finiteRow(double[] value, int width, String name) {
        if (value == null || value.length != width) {
            throw new IllegalArgumentException(name + " must have width " + width);
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
