package com.hytalerlbridge.observation;

import com.hytalerlbridge.geometry.GeometryFrame;
import java.util.List;

/**
 * Immutable observation of the current game state.
 *
 * Reflects Hytale's player stats:
 *   - Health (0-100): Regenerates naturally over time. Death loses 10% gear durability.
 *   - Food buff (0-100): Hytale has no hunger bar; eating food grants temporary
 *     buffs (health regen, stamina regen, damage bonus). This value represents
 *     the remaining strength of the active food buff.
 *   - Stamina (0-10): Fuels sprinting, charged attacks, and dodge rolls.
 *     Regenerates at ~1/sec. Base value is 10, not 100.
 *   - Mana (0-100): Placeholder — not yet functional in Hytale Early Access.
 *     Staves currently use stamina for charged attacks.
 */
public record Observation(
    double x, double y, double z,
    double vx, double vy, double vz,
    double yaw, double pitch,
    double health, double maxHealth,
    int foodBuffTimer,
    double stamina,
    double mana,
    int[] inventory,
    // Nearby blocks: list of [dx, dy, dz, blockTypeId]
    List<int[]> nearbyBlocks,
    // Nearby entities: [type, relX, relZ, health] using ObservationEncoding.
    List<int[]> nearbyEntities,
    // Time of day (0=dawn, 6000=noon, 12000=dusk, 18000=midnight)
    int timeOfDay,
    // Crafting: number of available recipes the agent can craft right now
    int craftableRecipeCount,
    // Versioned, model-agnostic local collision/support/LOS state.
    GeometryFrame geometry,
    // Bounded events heard during the step ending at this observation.
    AudioFrame audio,
    // Negotiated native inputs for the shared learner-v3 actor projector.
    NativeActorEvidenceFrame nativeActorEvidence,
    // Lossless native container capacities, occupied stacks, and quantities.
    NativeInventoryFrame nativeInventory
) {
    /** Compatibility constructor for callers predating native actor evidence. */
    public Observation(
        double x, double y, double z,
        double vx, double vy, double vz,
        double yaw, double pitch,
        double health, double maxHealth,
        int foodBuffTimer,
        double stamina,
        double mana,
        int[] inventory,
        List<int[]> nearbyBlocks,
        List<int[]> nearbyEntities,
        int timeOfDay,
        int craftableRecipeCount,
        GeometryFrame geometry,
        AudioFrame audio
    ) {
        this(
            x, y, z, vx, vy, vz, yaw, pitch, health, maxHealth,
            foodBuffTimer, stamina, mana, inventory, nearbyBlocks,
            nearbyEntities, timeOfDay, craftableRecipeCount, geometry, audio,
            NativeActorEvidenceFrame.unavailable("not_negotiated"),
            NativeInventoryFrame.unavailable("not_native")
        );
    }

    /** Compatibility constructor for callers predating auditory observations. */
    public Observation(
        double x, double y, double z,
        double vx, double vy, double vz,
        double yaw, double pitch,
        double health, double maxHealth,
        int foodBuffTimer,
        double stamina,
        double mana,
        int[] inventory,
        List<int[]> nearbyBlocks,
        List<int[]> nearbyEntities,
        int timeOfDay,
        int craftableRecipeCount,
        GeometryFrame geometry
    ) {
        this(
            x, y, z, vx, vy, vz, yaw, pitch, health, maxHealth,
            foodBuffTimer, stamina, mana, inventory, nearbyBlocks,
            nearbyEntities, timeOfDay, craftableRecipeCount, geometry,
            AudioFrame.unavailable()
        );
    }

    /** Compatibility constructor for simulator tasks that expose sparse blocks. */
    public Observation(
        double x, double y, double z,
        double vx, double vy, double vz,
        double yaw, double pitch,
        double health, double maxHealth,
        int foodBuffTimer,
        double stamina,
        double mana,
        int[] inventory,
        List<int[]> nearbyBlocks,
        List<int[]> nearbyEntities,
        int timeOfDay,
        int craftableRecipeCount
    ) {
        this(
            x, y, z, vx, vy, vz, yaw, pitch, health, maxHealth,
            foodBuffTimer, stamina, mana, inventory, nearbyBlocks,
            nearbyEntities, timeOfDay, craftableRecipeCount,
            GeometryFrame.fromSimulatorBlocks(x, y, z, nearbyBlocks)
        );
    }

    public Observation {
        geometry = geometry == null ? GeometryFrame.empty() : geometry;
        audio = audio == null ? AudioFrame.unavailable() : audio;
        nativeActorEvidence = nativeActorEvidence == null
            ? NativeActorEvidenceFrame.unavailable("not_negotiated")
            : nativeActorEvidence;
        nativeInventory = nativeInventory == null
            ? NativeInventoryFrame.unavailable("not_native")
            : nativeInventory;
    }

    public Observation withAudio(AudioFrame frame) {
        return new Observation(
            x, y, z, vx, vy, vz, yaw, pitch, health, maxHealth,
            foodBuffTimer, stamina, mana, inventory, nearbyBlocks,
            nearbyEntities, timeOfDay, craftableRecipeCount, geometry, frame,
            nativeActorEvidence, nativeInventory
        );
    }

    public Observation withNativeActorEvidence(NativeActorEvidenceFrame frame) {
        return new Observation(
            x, y, z, vx, vy, vz, yaw, pitch, health, maxHealth,
            foodBuffTimer, stamina, mana, inventory, nearbyBlocks,
            nearbyEntities, timeOfDay, craftableRecipeCount, geometry, audio,
            frame, nativeInventory
        );
    }

    public Observation withNativeInventory(NativeInventoryFrame frame) {
        return new Observation(
            x, y, z, vx, vy, vz, yaw, pitch, health, maxHealth,
            foodBuffTimer, stamina, mana, inventory, nearbyBlocks,
            nearbyEntities, timeOfDay, craftableRecipeCount, geometry, audio,
            nativeActorEvidence, frame
        );
    }

    public static Observation empty() {
        return new Observation(
            0, 64, 0, 0, 0, 0, 0, 0, 100, 100, 0, 10.0, 100.0,
            new int[36], List.of(), List.of(), 6000, 0, GeometryFrame.empty()
        );
    }
}
