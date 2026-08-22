package com.hytalerlbridge.world;

import java.util.Random;

/**
 * A simulated creature in the Hytale world.
 *
 * Hostile creatures pursue the player and attack on cooldown.
 * Passive creatures (Kweebecs) wander randomly.
 *
 * HP values based on Hytale Early Access data:
 *   Trork (Brawler/Guard/Hunter/Mauler/Sentry): 61 HP
 *   Outlander Stalker: 61 HP
 *   Scarak Seeker: 61 HP, Scarak Fighter: 81 HP
 *   Kweebec Razorleaf: 105 HP
 */
public class SimEntity {

    private final EntityType type;
    private double x, y, z;
    private double health;
    private double maxHealth;
    private double speed;
    private boolean alive;
    private int attackCooldown;

    public SimEntity(EntityType type, double x, double y, double z, double maxHealth, double speed) {
        this.type = type;
        this.x = x;
        this.y = y;
        this.z = z;
        this.health = maxHealth;
        this.maxHealth = maxHealth;
        this.speed = speed;
        this.alive = true;
        this.attackCooldown = 0;
    }

    /** Create an entity with default HP for its type. */
    public static SimEntity withDefaultHP(EntityType type, double x, double y, double z, double speed) {
        return new SimEntity(type, x, y, z, defaultHP(type), speed);
    }

    /** Default HP per creature type, based on Hytale Early Access values. */
    public static double defaultHP(EntityType type) {
        return switch (type) {
            case TRORK -> 61;        // Trork Brawler/Guard/Hunter/Mauler/Sentry
            case OUTLANDER -> 61;    // Outlander Stalker
            case SCARAK -> 61;       // Scarak Seeker (simplified; Fighters have 81)
            case KWEEBEC -> 105;     // Kweebec Razorleaf
            case ITEM_DROP -> 1;
        };
    }

    /** Tick AI: hostile creatures pursue the player, passive Kweebecs wander. */
    public void tick(double targetX, double targetZ, Random rng) {
        if (!alive) return;

        if (attackCooldown > 0) attackCooldown--;

        if (type.isHostile()) {
            double dx = targetX - x;
            double dz = targetZ - z;
            double dist = Math.sqrt(dx * dx + dz * dz);
            if (dist > 1.5 && dist < 32) {
                x += (dx / dist) * speed;
                z += (dz / dist) * speed;
            }
        } else {
            // Kweebecs: random wander
            if (rng.nextDouble() < 0.1) {
                x += (rng.nextDouble() - 0.5) * speed * 2;
                z += (rng.nextDouble() - 0.5) * speed * 2;
            }
        }
    }

    /** Try to attack. Returns damage dealt, 0 if on cooldown or out of range. */
    public double tryAttack(double targetX, double targetZ, Random rng) {
        if (!alive || attackCooldown > 0) return 0;
        double dist = distanceTo(targetX, targetZ);
        if (dist < 2.0 && type.isHostile()) {
            attackCooldown = 15; // ticks
            return switch (type) {
                case TRORK -> 6.0 + rng.nextDouble() * 4;        // maul/club strike (6-10 dmg)
                case OUTLANDER -> 5.0 + rng.nextDouble() * 5.0;  // ranged bolt (5-10 dmg)
                case SCARAK -> 3.0 + rng.nextDouble() * 3;       // venomous bite (3-6 dmg)
                default -> 0;
            };
        }
        return 0;
    }

    public void takeDamage(double amount) {
        health -= amount;
        if (health <= 0) {
            health = 0;
            alive = false;
        }
    }

    public double distanceTo(double tx, double tz) {
        double dx = x - tx;
        double dz = z - tz;
        return Math.sqrt(dx * dx + dz * dz);
    }

    // Getters
    public EntityType getType() { return type; }
    public double getX() { return x; }
    public double getY() { return y; }
    public double getZ() { return z; }
    public double getHealth() { return health; }
    public double getMaxHealth() { return maxHealth; }
    public boolean isAlive() { return alive; }
}
