package com.hytalerlbridge.task.builtin;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.observation.Observation;
import com.hytalerlbridge.observation.ObservationEncoding;
import com.hytalerlbridge.task.RLTask;

import java.util.List;
import java.util.Random;

/**
 * Navigation task: reach a target coordinate.
 * Dense reward: negative distance to target (closer = higher reward).
 */
public class NavigateTask implements RLTask {

    private double x, y, z, yaw, pitch;
    private double targetX, targetZ;
    private double prevDist;
    private double pendingReward;
    private boolean completionAwarded;
    private double health;
    private int tickCount;
    private Random rng;

    @Override
    public String description() {
        return "Navigate to a target location. Dense reward based on distance.";
    }

    @Override
    public void reset(long seed) {
        rng = new Random(seed);
        x = 0; y = 64; z = 0;
        yaw = rng.nextDouble() * 360; pitch = 0;
        health = 100;
        tickCount = 0;

        // Random target 20-50 blocks away
        double angle = rng.nextDouble() * Math.PI * 2;
        double dist = 20 + rng.nextDouble() * 30;
        targetX = Math.cos(angle) * dist;
        targetZ = Math.sin(angle) * dist;
        prevDist = distToTarget();
        pendingReward = 0;
        completionAwarded = false;
    }

    @Override
    public void applyAction(AgentAction action) {
        double speed = 0.15;
        double rad = Math.toRadians(yaw);
        if (action.forward()) { x -= Math.sin(rad) * speed; z -= Math.cos(rad) * speed; }
        if (action.back())    { x += Math.sin(rad) * speed; z += Math.cos(rad) * speed; }
        if (action.left())    { x -= Math.cos(rad) * speed; z += Math.sin(rad) * speed; }
        if (action.right())   { x += Math.cos(rad) * speed; z -= Math.sin(rad) * speed; }

        yaw += action.cameraDeltaYaw();
        pitch = Math.max(-90, Math.min(90, pitch + action.cameraDeltaPitch()));
    }

    @Override
    public void tick() {
        double currentDist = distToTarget();
        pendingReward += prevDist - currentDist;
        prevDist = currentDist;
        if (currentDist < 2.0 && !completionAwarded) {
            completionAwarded = true;
            pendingReward += 100.0;
        }
        tickCount++;
    }

    @Override
    public Observation observe() {
        List<int[]> targetMarker = List.of(new int[]{
            4,
            ObservationEncoding.encodeNearbyEntityScalar(targetX - x),
            ObservationEncoding.encodeNearbyEntityScalar(targetZ - z),
            0
        });
        return new Observation(x, y, z, 0, 0, 0, yaw, pitch, health, 100, 0, 10.0, 100.0,
            new int[36], List.of(), targetMarker, 6000, 0);
    }

    @Override
    public double computeReward() {
        double reward = pendingReward;
        pendingReward = 0;
        return reward;
    }

    @Override
    public boolean isTerminated() {
        return distToTarget() < 2.0;
    }

    private double distToTarget() {
        double dx = x - targetX;
        double dz = z - targetZ;
        return Math.sqrt(dx * dx + dz * dz);
    }
}
