package com.hytalerlbridge.task.builtin;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.observation.Observation;
import com.hytalerlbridge.task.RLTask;

import java.util.List;
import java.util.Random;

/**
 * Mine adamantite task: find and mine an adamantite ore.
 * Sparse reward: +100 on success. Agent must navigate underground.
 */
public class MineAdamantiteTask implements RLTask {

    private double x, y, z;
    private double yaw, pitch;
    private double health;
    private double adamantiteX, adamantiteY, adamantiteZ;
    private boolean found;
    private int tickCount;
    private Random rng;

    @Override
    public String description() {
        return "Find and mine an adamantite ore. Sparse reward of +100 on success.";
    }

    @Override
    public void reset(long seed) {
        rng = new Random(seed);
        x = 0; y = 64; z = 0;
        yaw = 0; pitch = 0;
        health = 100;
        found = false;
        tickCount = 0;

        // Place adamantite at random underground location
        adamantiteX = rng.nextDouble() * 60 - 30;
        adamantiteY = 10 + rng.nextDouble() * 20;
        adamantiteZ = rng.nextDouble() * 60 - 30;
    }

    @Override
    public void applyAction(AgentAction action) {
        double speed = 0.15;
        double rad = Math.toRadians(yaw);
        if (action.forward()) { x -= Math.sin(rad) * speed; z -= Math.cos(rad) * speed; }
        if (action.back())    { x += Math.sin(rad) * speed; z += Math.cos(rad) * speed; }
        if (action.left())    { x -= Math.cos(rad) * speed; z += Math.sin(rad) * speed; }
        if (action.right())   { x += Math.cos(rad) * speed; z -= Math.sin(rad) * speed; }

        // Allow vertical movement (mining down)
        if (action.attack()) { y -= 0.2; }
        if (action.jump())   { y += 0.2; }

        yaw += action.cameraDeltaYaw();
        pitch = Math.max(-90, Math.min(90, pitch + action.cameraDeltaPitch()));
    }

    @Override
    public void tick() {
        // Check if agent is near the adamantite
        double dist = Math.sqrt(
            Math.pow(x - adamantiteX, 2) + Math.pow(y - adamantiteY, 2) + Math.pow(z - adamantiteZ, 2)
        );
        if (dist < 2.0) {
            found = true;
        }

        tickCount++;
    }

    @Override
    public Observation observe() {
        return new Observation(x, y, z, 0, 0, 0, yaw, pitch, health, 100, 0, 10.0, 100.0,
            new int[36], List.of(), List.of(), 6000, 0);
    }

    @Override
    public double computeReward() {
        return found ? 100.0 : 0.0;
    }

    @Override
    public boolean isTerminated() {
        return found;
    }
}
