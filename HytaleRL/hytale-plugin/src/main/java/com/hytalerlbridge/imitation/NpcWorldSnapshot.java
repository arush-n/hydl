package com.hytalerlbridge.imitation;

import java.util.List;
import java.util.UUID;

/** Bounded omniscient NPC arena state at one exact native tick boundary. */
public record NpcWorldSnapshot(
    long worldTick,
    long gameTimeEpochSecond,
    int gameTimeNano,
    float dayProgress,
    float sunlightFactor,
    int moonPhase,
    int npcCount,
    boolean overflow,
    List<Actor> actors,
    int entityCount,
    boolean entityOverflow,
    List<Entity> entities,
    long simulationTimeEpochSecond,
    int simulationTimeNano
) {
    public static final int ACTOR_CAPACITY = 256;
    public static final int ENTITY_CAPACITY = 512;
    public static final String SCHEMA = "hytalerl_npc_world_snapshot_v4";
    public static final int VERSION = 4;
    public static final String ACTOR_LAYOUT =
        "uuid:uuid,role:utf8,state:f64[14],state_name:utf8,target_uuid:uuid?,"
            + "combat_attack:bool,attack_pause_seconds:f32";
    public static final String ENTITY_LAYOUT =
        "entity_index:i32,uuid:uuid?,flags:i32,asset_id:utf8,model_asset_id:utf8,"
            + "position:f64[3],rotation:f64[3],velocity_available:bool,"
            + "velocity:f64[3],bounds_available:bool,bounds:f64[6],"
            + "owner_uuid:uuid?,physics_state:utf8,lifecycle_timing_available:bool,"
            + "lifecycle_start_epoch_second:i64,lifecycle_start_nano:i32,"
            + "lifecycle_end_epoch_second:i64,lifecycle_end_nano:i32";

    public static final int ENTITY_NPC = 1;
    public static final int ENTITY_LEGACY_PROJECTILE = 1 << 1;
    public static final int ENTITY_STANDARD_PROJECTILE = 1 << 2;
    public static final int ENTITY_PREDICTED_PROJECTILE = 1 << 3;
    public static final int ENTITY_ON_GROUND = 1 << 4;
    public static final int ENTITY_IN_FLUID = 1 << 5;
    public static final int ENTITY_IMPACTED_OR_BOUNCED = 1 << 6;
    public static final int ENTITY_RESTING_OR_SLIDING = 1 << 7;
    public static final int ENTITY_DEPLOYABLE = 1 << 8;
    public static final int ENTITY_ALL_FLAGS = (1 << 9) - 1;

    public NpcWorldSnapshot {
        actors = actors == null ? List.of() : List.copyOf(actors);
        entities = entities == null ? List.of() : List.copyOf(entities);
        if (
            gameTimeNano < 0
                || gameTimeNano >= 1_000_000_000
                || simulationTimeNano < 0
                || simulationTimeNano >= 1_000_000_000
                || !Float.isFinite(dayProgress)
                || dayProgress < 0.0f
                || dayProgress > 1.0f
                || !Float.isFinite(sunlightFactor)
                || npcCount < actors.size()
                || actors.size() > ACTOR_CAPACITY
                || overflow != (npcCount > actors.size())
                || entityCount < entities.size()
                || entities.size() > ENTITY_CAPACITY
                || entityOverflow != (entityCount > entities.size())
        ) {
            throw new IllegalArgumentException("invalid NPC worldview metadata");
        }
        if (entities.stream().map(Entity::entityIndex).distinct().count()
            != entities.size()) {
            throw new IllegalArgumentException("world entities must have unique indexes");
        }
    }

    public record Actor(
        UUID uuid,
        String role,
        double[] state,
        String stateName,
        UUID targetUuid,
        boolean combatAttack,
        float attackPauseSeconds
    ) {
        public Actor {
            if (uuid == null || role == null || role.isBlank()) {
                throw new IllegalArgumentException("world actor identity is incomplete");
            }
            stateName = stateName == null ? "" : stateName;
            if (state == null || state.length != NpcTraceFrame.STATE_WIDTH) {
                throw new IllegalArgumentException("world actor state has wrong width");
            }
            state = state.clone();
            for (double scalar : state) {
                if (!Double.isFinite(scalar)) {
                    throw new IllegalArgumentException("world actor state must be finite");
                }
            }
            if (!Float.isFinite(attackPauseSeconds) || attackPauseSeconds < 0.0f) {
                throw new IllegalArgumentException("attack pause must be finite and nonnegative");
            }
        }

        @Override
        public double[] state() {
            return state.clone();
        }
    }

    /** Omniscient raw state for every bounded transform-bearing ECS entity. */
    public record Entity(
        int entityIndex,
        UUID uuid,
        int flags,
        String assetId,
        String modelAssetId,
        double[] position,
        double[] rotation,
        boolean velocityAvailable,
        double[] velocity,
        boolean boundsAvailable,
        double[] bounds,
        UUID ownerUuid,
        String physicsState,
        boolean lifecycleTimingAvailable,
        long lifecycleStartEpochSecond,
        int lifecycleStartNano,
        long lifecycleEndEpochSecond,
        int lifecycleEndNano
    ) {
        public Entity(
            int entityIndex,
            UUID uuid,
            int flags,
            String assetId,
            String modelAssetId,
            double[] position,
            double[] rotation,
            boolean velocityAvailable,
            double[] velocity,
            boolean boundsAvailable,
            double[] bounds,
            UUID ownerUuid,
            String physicsState
        ) {
            this(
                entityIndex,
                uuid,
                flags,
                assetId,
                modelAssetId,
                position,
                rotation,
                velocityAvailable,
                velocity,
                boundsAvailable,
                bounds,
                ownerUuid,
                physicsState,
                false,
                0L,
                0,
                0L,
                0
            );
        }

        public Entity {
            assetId = text(assetId);
            modelAssetId = text(modelAssetId);
            physicsState = text(physicsState);
            position = finite(position, 3, "position");
            rotation = finite(rotation, 3, "rotation");
            velocity = finite(velocity, 3, "velocity");
            bounds = finite(bounds, 6, "bounds");
            if (entityIndex < 0 || (flags & ~ENTITY_ALL_FLAGS) != 0) {
                throw new IllegalArgumentException("invalid world entity identity");
            }
            if (!velocityAvailable && any(velocity)) {
                throw new IllegalArgumentException(
                    "unavailable entity velocity must be zero"
                );
            }
            if (!boundsAvailable && any(bounds)) {
                throw new IllegalArgumentException(
                    "unavailable entity bounds must be zero"
                );
            }
            if (
                lifecycleStartNano < 0
                    || lifecycleStartNano >= 1_000_000_000
                    || lifecycleEndNano < 0
                    || lifecycleEndNano >= 1_000_000_000
            ) {
                throw new IllegalArgumentException(
                    "entity lifecycle nanoseconds are outside Instant bounds"
                );
            }
            if (!lifecycleTimingAvailable && (
                lifecycleStartEpochSecond != 0L
                    || lifecycleStartNano != 0
                    || lifecycleEndEpochSecond != 0L
                    || lifecycleEndNano != 0
            )) {
                throw new IllegalArgumentException(
                    "unavailable entity lifecycle timing must be zero"
                );
            }
            if (lifecycleTimingAvailable && instantBefore(
                lifecycleEndEpochSecond,
                lifecycleEndNano,
                lifecycleStartEpochSecond,
                lifecycleStartNano
            )) {
                throw new IllegalArgumentException(
                    "entity lifecycle end precedes its start"
                );
            }
        }

        public double lifecycleDurationSeconds() {
            if (!lifecycleTimingAvailable) return 0.0;
            return (double) lifecycleEndEpochSecond
                - (double) lifecycleStartEpochSecond
                + (double) (lifecycleEndNano - lifecycleStartNano) / 1_000_000_000.0;
        }

        @Override
        public double[] position() { return position.clone(); }

        @Override
        public double[] rotation() { return rotation.clone(); }

        @Override
        public double[] velocity() { return velocity.clone(); }

        @Override
        public double[] bounds() { return bounds.clone(); }
    }

    private static double[] finite(double[] value, int width, String name) {
        if (value == null || value.length != width) {
            throw new IllegalArgumentException(name + " has wrong width");
        }
        double[] result = value.clone();
        for (double scalar : result) {
            if (!Double.isFinite(scalar)) {
                throw new IllegalArgumentException(name + " must be finite");
            }
        }
        return result;
    }

    private static boolean any(double[] values) {
        for (double value : values) if (value != 0.0) return true;
        return false;
    }

    private static boolean instantBefore(
        long leftSecond,
        int leftNano,
        long rightSecond,
        int rightNano
    ) {
        return leftSecond < rightSecond
            || (leftSecond == rightSecond && leftNano < rightNano);
    }

    private static String text(String value) {
        return value == null ? "" : value;
    }
}
