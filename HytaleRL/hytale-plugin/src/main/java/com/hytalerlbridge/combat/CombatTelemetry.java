package com.hytalerlbridge.combat;

import java.util.Map;

/** Stable scalar telemetry consumed as a policy observation by Python. */
public record CombatTelemetry(
    boolean agentAttackExecuting,
    boolean targetVisible,
    CombatPhase targetAttackPhase,
    double targetAttackProgress,
    int targetAttackIndex,
    int targetAttackElapsedTicks,
    String targetAttackId,
    double targetFacingErrorDegrees,
    double targetYawDegrees,
    double targetHeadYawDegrees,
    double targetHeadPitchDegrees,
    double targetVelocityX,
    double targetVelocityZ
) {
    public static final String AGENT_ATTACK_EXECUTING = "combat_agent_attack_executing";
    public static final String TARGET_VISIBLE = "combat_target_visible";
    public static final String TARGET_ATTACK_PHASE = "combat_target_attack_phase";
    public static final String TARGET_ATTACK_PROGRESS = "combat_target_attack_progress";
    public static final String TARGET_ATTACK_INDEX = "combat_target_attack_index";
    public static final String TARGET_ATTACK_ELAPSED_TICKS = "combat_target_attack_elapsed_ticks";
    public static final String TARGET_ATTACK_ID = "combat_target_attack_id";
    public static final String TARGET_FACING_ERROR_DEGREES =
        "combat_target_facing_error_degrees";
    public static final String TARGET_YAW_DEGREES = "combat_target_yaw_degrees";
    public static final String TARGET_HEAD_YAW_DEGREES =
        "combat_target_head_yaw_degrees";
    public static final String TARGET_HEAD_PITCH_DEGREES =
        "combat_target_head_pitch_degrees";
    public static final String TARGET_VELOCITY_X = "combat_target_velocity_x";
    public static final String TARGET_VELOCITY_Z = "combat_target_velocity_z";

    public CombatTelemetry {
        targetAttackPhase = targetAttackPhase == null
            ? CombatPhase.IDLE
            : targetAttackPhase;
        targetAttackProgress = clamp(targetAttackProgress, 0.0, 1.0);
        targetAttackId = targetAttackId == null ? "" : targetAttackId;
    }

    public void putInto(Map<String, Object> info) {
        info.put(AGENT_ATTACK_EXECUTING, agentAttackExecuting);
        info.put(TARGET_VISIBLE, targetVisible);
        info.put(TARGET_ATTACK_PHASE, targetAttackPhase.code());
        info.put(TARGET_ATTACK_PROGRESS, targetAttackProgress);
        info.put(TARGET_ATTACK_INDEX, targetAttackIndex);
        info.put(TARGET_ATTACK_ELAPSED_TICKS, targetAttackElapsedTicks);
        info.put(TARGET_ATTACK_ID, targetAttackId);
        info.put(TARGET_FACING_ERROR_DEGREES, targetFacingErrorDegrees);
        info.put(TARGET_YAW_DEGREES, targetYawDegrees);
        info.put(TARGET_HEAD_YAW_DEGREES, targetHeadYawDegrees);
        info.put(TARGET_HEAD_PITCH_DEGREES, targetHeadPitchDegrees);
        info.put(TARGET_VELOCITY_X, targetVelocityX);
        info.put(TARGET_VELOCITY_Z, targetVelocityZ);
    }

    public static CombatTelemetry idle(boolean targetVisible) {
        return new CombatTelemetry(
            false,
            targetVisible,
            CombatPhase.IDLE,
            0.0,
            -1,
            0,
            "",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        );
    }

    private static double clamp(double value, double minimum, double maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }
}
