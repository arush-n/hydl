package com.hytalerlbridge.imitation;

import com.hytalerlbridge.geometry.GeometryFrame;
import com.hytalerlbridge.geometry.GeometryContract;
import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import com.hytalerlbridge.observation.NativeInventoryFrame;
import com.hytalerlbridge.worldgen.NativePerceptionChannels;
import java.util.List;
import java.util.UUID;

/** Raw, model-neutral Java evidence at one NPC policy boundary. */
public record NpcObservationSnapshot(
    long worldTick,
    NativeActorEvidenceFrame.Actor actorEvidence,
    NativeActorEvidenceFrame.Actor targetActorEvidence,
    NativeActorEvidenceFrame actionCapabilityEvidence,
    GeometryFrame geometry,
    NativeInventoryFrame inventory,
    boolean roleOpaqueCellMaskAvailable,
    boolean[] roleOpaqueCellMask,
    NativePerceptionChannels localPerception,
    boolean combatLifecycleAvailable,
    boolean combatAttackExecuting,
    float attackPauseSeconds,
    List<NpcAttackActionSnapshot> attackCandidates,
    int attackCandidateCount,
    boolean attackCandidateOverflow,
    List<NpcAttackActionSnapshot> targetAttackCandidates,
    int targetAttackCandidateCount,
    boolean targetAttackCandidateOverflow,
    boolean perceptionAvailable,
    boolean targetPresent,
    boolean targetPerceptible,
    double targetDistance,
    List<UUID> perceptibleNpcUuids,
    int perceptibleNpcCount,
    boolean perceptibleNpcOverflow,
    List<Integer> perceptibleEntityIndices,
    int perceptibleEntityCount,
    boolean perceptibleEntityOverflow,
    int statusFailureBits
) {
    public static final String SCHEMA = "hytalerl_native_npc_observation_v5";
    public static final int VERSION = 5;
    public static final String CONTRACT_SHA256 =
        "5A48A2FC675C75EFECE5E0BCFD2CA567AF04C7CAFAED63EE2C9E4DC8DA15CFA0";
    public static final int ATTACK_CANDIDATE_CAPACITY = 64;
    public static final int PERCEPTIBLE_NPC_CAPACITY = 256;
    public static final int PERCEPTIBLE_ENTITY_CAPACITY = 512;
    public static final String GROUP_LAYOUT =
        "bit0=actor_core,bit1=target_core,bit2=combat,bit3=resources,"
            + "bit4=weapon,bit5=defense,bit6=status,bit7=ability,"
            + "bit8=actor_world,bit9=movement,bit10=geometry,bit11=collision,"
            + "bit12=perception,bit13=nearby_npcs,bit14=inventory,"
            + "bit15=world_clock,bit16=decision_state,bit17=active_interactions,"
            + "bit18=interaction_candidates,bit19=traversal,bit20=projectiles,"
            + "bit21=dynamic_hazards,bit22=audio,bit23=non_npc_entities,"
            + "bit24=role_opacity,bit25=light,bit26=environment_channels";

    private static final long ACTOR_CORE = 1L;
    private static final long TARGET_CORE = 1L << 1;
    private static final long COMBAT = 1L << 2;
    private static final long RESOURCES = 1L << 3;
    private static final long WEAPON = 1L << 4;
    private static final long DEFENSE = 1L << 5;
    private static final long STATUS = 1L << 6;
    private static final long ABILITY = 1L << 7;
    private static final long ACTOR_WORLD = 1L << 8;
    private static final long MOVEMENT = 1L << 9;
    private static final long GEOMETRY = 1L << 10;
    private static final long COLLISION = 1L << 11;
    private static final long PERCEPTION = 1L << 12;
    private static final long NEARBY_NPCS = 1L << 13;
    private static final long INVENTORY = 1L << 14;
    private static final long WORLD_CLOCK = 1L << 15;
    private static final long DECISION_STATE = 1L << 16;
    private static final long ACTIVE_INTERACTIONS = 1L << 17;
    private static final long INTERACTION_CANDIDATES = 1L << 18;
    private static final long TRAVERSAL = 1L << 19;
    private static final long PROJECTILES = 1L << 20;
    private static final long DYNAMIC_HAZARDS = 1L << 21;
    private static final long AUDIO = 1L << 22;
    private static final long NON_NPC_ENTITIES = 1L << 23;
    private static final long ROLE_OPACITY = 1L << 24;
    private static final long LIGHT = 1L << 25;
    private static final long ENVIRONMENT_CHANNELS = 1L << 26;
    public static final long ALL_GROUP_BITS = (1L << 27) - 1L;

    public static final int STATUS_ACTOR_OVERFLOW = 1;
    public static final int STATUS_TARGET_OVERFLOW = 1 << 1;
    public static final int STATUS_ACTOR_INVALID = 1 << 2;
    public static final int STATUS_TARGET_INVALID = 1 << 3;

    public NpcObservationSnapshot {
        if (worldTick < 0L) {
            throw new IllegalArgumentException("worldTick must be nonnegative");
        }
        if (actorEvidence == null || !actorEvidence.present()) {
            throw new IllegalArgumentException("observation requires the traced actor");
        }
        if (targetActorEvidence == null) {
            throw new IllegalArgumentException("target actor evidence is required");
        }
        actionCapabilityEvidence = actionCapabilityEvidence == null
            ? NativeActorEvidenceFrame.unavailable("not_captured")
            : actionCapabilityEvidence;
        if (actionCapabilityEvidence.available()) {
            NativeActorEvidenceFrame.Actor capabilityActor =
                actionCapabilityEvidence.actors().get(0);
            if (
                actionCapabilityEvidence.worldTick() != worldTick
                    || !capabilityActor.present()
                    || !capabilityActor.roleId().equals(actorEvidence.roleId())
                    || !capabilityActor.itemId().equals(actorEvidence.itemId())
            ) {
                throw new IllegalArgumentException(
                    "action capability evidence is not aligned to the traced actor"
                );
            }
        }
        geometry = geometry == null ? GeometryFrame.empty() : geometry;
        inventory = inventory == null
            ? NativeInventoryFrame.unavailable("not_captured")
            : inventory;
        roleOpaqueCellMask = roleOpaqueCellMask == null
            ? new boolean[GeometryContract.CELL_COUNT]
            : roleOpaqueCellMask.clone();
        if (roleOpaqueCellMask.length != GeometryContract.CELL_COUNT) {
            throw new IllegalArgumentException("role opacity mask has wrong width");
        }
        if (!roleOpaqueCellMaskAvailable && any(roleOpaqueCellMask)) {
            throw new IllegalArgumentException(
                "unavailable role opacity must be zero-masked"
            );
        }
        if (localPerception == null) {
            throw new IllegalArgumentException("local perception channels are required");
        }
        validateLocalPerception(geometry, localPerception);
        if (
            !Float.isFinite(attackPauseSeconds)
                || attackPauseSeconds < 0.0f
                || (!combatLifecycleAvailable
                    && (combatAttackExecuting || attackPauseSeconds != 0.0f))
        ) {
            throw new IllegalArgumentException(
                "combat lifecycle must be finite and zero-masked when unavailable"
            );
        }
        attackCandidates = attackCandidates == null
            ? List.of()
            : List.copyOf(attackCandidates);
        targetAttackCandidates = targetAttackCandidates == null
            ? List.of()
            : List.copyOf(targetAttackCandidates);
        perceptibleNpcUuids = perceptibleNpcUuids == null
            ? List.of()
            : List.copyOf(perceptibleNpcUuids);
        perceptibleEntityIndices = perceptibleEntityIndices == null
            ? List.of()
            : List.copyOf(perceptibleEntityIndices);
        bounded(
            attackCandidates.size(),
            attackCandidateCount,
            attackCandidateOverflow,
            ATTACK_CANDIDATE_CAPACITY,
            "attack candidates"
        );
        bounded(
            targetAttackCandidates.size(),
            targetAttackCandidateCount,
            targetAttackCandidateOverflow,
            ATTACK_CANDIDATE_CAPACITY,
            "target attack candidates"
        );
        bounded(
            perceptibleNpcUuids.size(),
            perceptibleNpcCount,
            perceptibleNpcOverflow,
            PERCEPTIBLE_NPC_CAPACITY,
            "perceptible NPCs"
        );
        bounded(
            perceptibleEntityIndices.size(),
            perceptibleEntityCount,
            perceptibleEntityOverflow,
            PERCEPTIBLE_ENTITY_CAPACITY,
            "perceptible entities"
        );
        if (perceptibleNpcUuids.stream().distinct().count()
            != perceptibleNpcUuids.size()) {
            throw new IllegalArgumentException("perceptible NPC UUIDs must be unique");
        }
        if (
            perceptibleEntityIndices.stream().anyMatch(index -> index == null || index < 0)
                || perceptibleEntityIndices.stream().distinct().count()
                    != perceptibleEntityIndices.size()
                || !perceptibleEntityIndices.equals(
                    perceptibleEntityIndices.stream().sorted().toList()
                )
        ) {
            throw new IllegalArgumentException(
                "perceptible entity indices must be nonnegative, unique, and sorted"
            );
        }
        if (!Double.isFinite(targetDistance) || targetDistance < 0.0) {
            throw new IllegalArgumentException("targetDistance must be finite and nonnegative");
        }
        if (!targetPresent && (
            targetPerceptible
                || targetDistance != 0.0
                || targetActorEvidence.present()
                || targetAttackCandidateCount != 0
        )) {
            throw new IllegalArgumentException("absent target carries observation evidence");
        }
        if (targetPerceptible && !perceptionAvailable) {
            throw new IllegalArgumentException("unavailable perception cannot see a target");
        }
        if (!perceptionAvailable && (
            !perceptibleNpcUuids.isEmpty()
                || perceptibleNpcCount != 0
                || perceptibleNpcOverflow
                || !perceptibleEntityIndices.isEmpty()
                || perceptibleEntityCount != 0
                || perceptibleEntityOverflow
        )) {
            throw new IllegalArgumentException(
                "unavailable perception cannot carry perceived entities"
            );
        }
        if (targetPresent != targetActorEvidence.present()
            || targetPerceptible != targetActorEvidence.perceptible()) {
            throw new IllegalArgumentException("target evidence disagrees with perception");
        }
        if ((statusFailureBits & ~0xf) != 0) {
            throw new IllegalArgumentException("statusFailureBits has unknown bits");
        }
    }

    /** Families complete from Java facts, without learner-specific projection. */
    public long completeGroupBits() {
        long result = ACTOR_CORE | TARGET_CORE | COMBAT | RESOURCES | WEAPON
            | DEFENSE | MOVEMENT | WORLD_CLOCK | ACTIVE_INTERACTIONS;
        if (geometry.available()) result |= GEOMETRY;
        if (geometry.available() && geometry.exactCollisionShapes()) {
            result |= COLLISION;
        }
        if (perceptionAvailable) {
            result |= PERCEPTION;
            if (!perceptibleNpcOverflow) result |= NEARBY_NPCS;
            if (!perceptibleEntityOverflow) result |= PROJECTILES;
        }
        if (inventoryComplete()) result |= INVENTORY;
        if (roleOpaqueCellMaskAvailable) result |= ROLE_OPACITY;
        if (channelComplete(
            NativePerceptionChannels.VALID_SKY_LIGHT
                | NativePerceptionChannels.VALID_BLOCK_LIGHT_RGB
                | NativePerceptionChannels.VALID_TINT_RGB
        )) result |= LIGHT;
        if (channelComplete(
            NativePerceptionChannels.VALID_ENVIRONMENT
                | NativePerceptionChannels.VALID_TINT_RGB
        )) result |= ENVIRONMENT_CHANNELS;
        return result;
    }

    /** Families with useful raw evidence plus an explicitly missing semantic. */
    public long partialGroupBits() {
        long result = STATUS | ABILITY | ACTOR_WORLD | DECISION_STATE
            | TRAVERSAL | DYNAMIC_HAZARDS;
        if (perceptionAvailable && perceptibleNpcOverflow) result |= NEARBY_NPCS;
        if (perceptionAvailable && perceptibleEntityOverflow) result |= PROJECTILES;
        if (geometry.available() && !geometry.exactCollisionShapes()) {
            result |= COLLISION;
        }
        if (inventory.available() && !inventoryComplete()) result |= INVENTORY;
        if (channelPartial(
            NativePerceptionChannels.VALID_SKY_LIGHT
                | NativePerceptionChannels.VALID_BLOCK_LIGHT_RGB
                | NativePerceptionChannels.VALID_TINT_RGB
        )) result |= LIGHT;
        if (channelPartial(
            NativePerceptionChannels.VALID_ENVIRONMENT
                | NativePerceptionChannels.VALID_TINT_RGB
        )) result |= ENVIRONMENT_CHANNELS;
        return result;
    }

    /** Families for which this server path publishes no usable evidence. */
    public long unavailableGroupBits() {
        return ALL_GROUP_BITS & ~(completeGroupBits() | partialGroupBits());
    }

    private boolean inventoryComplete() {
        return inventory.available()
            && inventory.containers().stream().allMatch(
                NativeInventoryFrame.Container::available
            );
    }

    @Override
    public boolean[] roleOpaqueCellMask() {
        return roleOpaqueCellMask.clone();
    }

    private boolean channelComplete(int requiredBits) {
        byte[] available = localPerception.available();
        byte[] validity = localPerception.channelValidity();
        for (int index = 0; index < available.length; index++) {
            if (
                available[index] == 0
                    || (Byte.toUnsignedInt(validity[index]) & requiredBits)
                        != requiredBits
            ) return false;
        }
        return true;
    }

    private boolean channelPartial(int requiredBits) {
        if (channelComplete(requiredBits)) return false;
        byte[] available = localPerception.available();
        byte[] validity = localPerception.channelValidity();
        for (int index = 0; index < available.length; index++) {
            if (
                available[index] != 0
                    && (Byte.toUnsignedInt(validity[index]) & requiredBits) != 0
            ) return true;
        }
        return false;
    }

    private static void validateLocalPerception(
        GeometryFrame geometry,
        NativePerceptionChannels channels
    ) {
        if (channels.sampleCount() != GeometryContract.CELL_COUNT) {
            throw new IllegalArgumentException("local perception has wrong width");
        }
        int[] positions = channels.positions();
        for (int dx = -GeometryContract.RADIUS; dx <= GeometryContract.RADIUS; dx++) {
            for (int dy = -GeometryContract.RADIUS; dy <= GeometryContract.RADIUS; dy++) {
                for (int dz = -GeometryContract.RADIUS; dz <= GeometryContract.RADIUS; dz++) {
                    int index = GeometryContract.cellIndex(dx, dy, dz) * 3;
                    if (
                        positions[index] != geometry.originX() + dx
                            || positions[index + 1] != geometry.originY() + dy
                            || positions[index + 2] != geometry.originZ() + dz
                    ) {
                        throw new IllegalArgumentException(
                            "local perception does not align with geometry"
                        );
                    }
                }
            }
        }
    }

    private static boolean any(boolean[] values) {
        for (boolean value : values) if (value) return true;
        return false;
    }

    private static void bounded(
        int emitted,
        int total,
        boolean overflow,
        int capacity,
        String label
    ) {
        if (emitted > capacity || total < emitted || overflow != (total > emitted)) {
            throw new IllegalArgumentException(label + " metadata is inconsistent");
        }
    }
}
