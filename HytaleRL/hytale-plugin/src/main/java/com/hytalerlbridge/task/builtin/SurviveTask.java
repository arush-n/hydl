package com.hytalerlbridge.task.builtin;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.observation.Observation;
import com.hytalerlbridge.task.RLTask;

import java.util.List;
import java.util.Random;

/**
 * Survival task: stay alive as long as possible.
 * Reward: +0.1 per tick survived. Terminates when health reaches 0.
 */
public class SurviveTask implements RLTask {

    private double x, y, z;
    private double vx, vy, vz;
    private double yaw, pitch;
    private double health;
    private int tickCount;
    private double pendingReward;
    private Random rng;

    @Override
    public String description() {
        return "Stay alive as long as possible. +0.1 reward per tick survived.";
    }

    @Override
    public void reset(long seed) {
        rng = new Random(seed);
        x = rng.nextDouble() * 100 - 50;
        y = 64;
        z = rng.nextDouble() * 100 - 50;
        vx = vy = vz = 0;
        yaw = rng.nextDouble() * 360;
        pitch = 0;
        health = 100.0;
        tickCount = 0;
        pendingReward = 0;
    }

    @Override
    public void applyAction(AgentAction action) {
        double speed = 0.1;
        double rad = Math.toRadians(yaw);
        if (action.forward()) { x -= Math.sin(rad) * speed; z -= Math.cos(rad) * speed; }
        if (action.back())    { x += Math.sin(rad) * speed; z += Math.cos(rad) * speed; }
        if (action.left())    { x -= Math.cos(rad) * speed; z += Math.sin(rad) * speed; }
        if (action.right())   { x += Math.cos(rad) * speed; z -= Math.sin(rad) * speed; }
        if (action.jump() && y <= 64.0) { vy = 0.4; }

        yaw += action.cameraDeltaYaw();
        pitch = Math.max(-90, Math.min(90, pitch + action.cameraDeltaPitch()));
    }

    @Override
    public void tick() {
        // Simple physics
        vy -= 0.02; // gravity
        y += vy;
        if (y < 64.0) { y = 64.0; vy = 0; }

        // Random environmental damage
        if (rng.nextDouble() < 0.005) {
            health -= rng.nextDouble() * 4;
        }

        // Natural health regeneration (Hytale: slow passive regen)
        if (health > 0 && health < 100 && tickCount % 100 == 0) {
            health = Math.min(100, health + 0.5);
        }

        health = Math.max(0, health);
        if (health > 0) pendingReward += 0.1;
        tickCount++;
    }

    @Override
    public Observation observe() {
        return new Observation(x, y, z, vx, vy, vz, yaw, pitch, health, 100, 0, 10.0, 100.0,
            new int[36], List.of(), List.of(), 6000, 0);
    }

    @Override
    public double computeReward() {
        double reward = pendingReward;
        pendingReward = 0;
        return reward;
    }

    @Override
    public boolean isTerminated() {
        return health <= 0;
    }
}
