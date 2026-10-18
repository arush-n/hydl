package com.hytalerlbridge.policy.perception;

import com.hytalerlbridge.policy.Projection;

/** Opt-in mirror of Arena's omniscient pursuit context, preserving visibility. */
public final class PursuitTargetContext {
    private PursuitTargetContext() {}

    public static void apply(float[] target, Projection.TargetState state,
                             Projection.Params params) {
        if (target.length != 16) throw new IllegalArgumentException("target width");
        float x = (float) (state.targetX() - state.agentX());
        float y = (float) (state.targetY() - state.agentY());
        float z = (float) (state.targetZ() - state.agentZ());
        float vx = (float) (state.targetVelocityX() - state.agentVelocityX());
        float vy = (float) (state.targetVelocityY() - state.agentVelocityY());
        float vz = (float) (state.targetVelocityZ() - state.agentVelocityZ());
        float yaw = (float) Math.toRadians(state.agentYawDegrees());
        float sin = (float) Math.sin(yaw), cos = (float) Math.cos(yaw);
        float forward = -x * sin - z * cos, right = x * cos - z * sin;
        float planar = (float) Math.hypot(forward, right);
        float radius = Projection.NEARBY_ENTITY_RADIUS_BLOCKS;
        float speed = Math.max(1.0e-6f, (float) params.targetChaseSpeed());
        float vertical = Math.max(1.0e-6f, (float) params.verticalSpeedScale());
        int[] indices = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10};
        float[] values = {
            forward / radius, right / radius, y / radius,
            (-vx * sin - vz * cos) / speed,
            (vx * cos - vz * sin) / speed, vy / vertical,
            planar / radius, (float) Math.sqrt(x*x + y*y + z*z) / radius,
            right / Math.max(planar, 1.0e-6f),
            forward / Math.max(planar, 1.0e-6f)
        };
        for (int i = 0; i < indices.length; i++) {
            target[indices[i]] = Math.max(-1.0f, Math.min(1.0f, values[i]));
        }
    }
}
