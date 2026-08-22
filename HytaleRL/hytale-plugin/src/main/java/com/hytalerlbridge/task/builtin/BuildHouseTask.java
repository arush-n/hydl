package com.hytalerlbridge.task.builtin;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.observation.Observation;
import com.hytalerlbridge.task.RLTask;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Build task: construct a simple house structure.
 * Shaped reward: +1 per block placed in a valid position, +50 on completion.
 */
public class BuildHouseTask implements RLTask {

    private static final int REQUIRED_BLOCKS = 20; // minimum for a basic house

    private double x, y, z, yaw, pitch;
    private double health;
    private Set<String> placedBlocks;
    private double pendingReward;
    private boolean completionAwarded;
    private int tickCount;
    private Random rng;

    @Override
    public String description() {
        return "Build a simple house structure. Shaped reward per block placed.";
    }

    @Override
    public void reset(long seed) {
        rng = new Random(seed);
        x = 0; y = 64; z = 0;
        yaw = 0; pitch = 0;
        health = 100;
        placedBlocks = new HashSet<>();
        pendingReward = 0;
        completionAwarded = false;
        tickCount = 0;
    }

    @Override
    public void applyAction(AgentAction action) {
        double speed = 0.1;
        double rad = Math.toRadians(yaw);
        if (action.forward()) { x -= Math.sin(rad) * speed; z -= Math.cos(rad) * speed; }
        if (action.back())    { x += Math.sin(rad) * speed; z += Math.cos(rad) * speed; }
        if (action.left())    { x -= Math.cos(rad) * speed; z += Math.sin(rad) * speed; }
        if (action.right())   { x += Math.cos(rad) * speed; z -= Math.sin(rad) * speed; }
        if (action.jump())    { y += 0.2; }

        yaw += action.cameraDeltaYaw();
        pitch = Math.max(-90, Math.min(90, pitch + action.cameraDeltaPitch()));

        // Place block in the direction the agent is looking
        if (action.use()) {
            double lookRad = Math.toRadians(yaw);
            double pitchRad = Math.toRadians(pitch);
            int bx = (int) Math.round(x - Math.sin(lookRad) * 2);
            int by = (int) Math.round(y - Math.sin(pitchRad) * 2);
            int bz = (int) Math.round(z + Math.cos(lookRad) * 2);
            if (placedBlocks.add(bx + "," + by + "," + bz)) {
                pendingReward += 1.0;
            }
        }
    }

    @Override
    public void tick() {
        if (placedBlocks.size() >= REQUIRED_BLOCKS && !completionAwarded) {
            completionAwarded = true;
            pendingReward += 50.0;
        }
        tickCount++;
    }

    @Override
    public Observation observe() {
        // Encode placed block count in first inventory slot
        int[] inventory = new int[36];
        inventory[0] = placedBlocks.size();
        return new Observation(x, y, z, 0, 0, 0, yaw, pitch, health, 100, 0, 10.0, 100.0,
            inventory, List.of(), List.of(), 6000, 0);
    }

    @Override
    public double computeReward() {
        double reward = pendingReward;
        pendingReward = 0;
        return reward;
    }

    @Override
    public boolean isTerminated() {
        return placedBlocks.size() >= REQUIRED_BLOCKS;
    }
}
