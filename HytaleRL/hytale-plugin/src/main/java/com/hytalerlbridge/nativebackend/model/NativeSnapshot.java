package com.hytalerlbridge.nativebackend.model;

import com.hytalerlbridge.combat.CombatTelemetry;
import com.hytalerlbridge.observation.MovementStateFrame;
import com.hytalerlbridge.observation.Observation;

/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record NativeSnapshot(
    Observation observation,
    boolean agentPresent,
    boolean onGround,
    String roleName,
    int blockBelowId,
    long worldTick,
    int activeHotbarSlot,
    boolean hasHealthStat,
    boolean hasStaminaStat,
    boolean hasManaStat,
    boolean targetPresent,
    String targetRoleName,
    double targetHealth,
    double targetMaxHealth,
    double targetDistance,
    double targetX,
    double targetY,
    double targetZ,
    int nativeNpcCount,
    EntityModelEvidence agentModelEvidence,
    EntityModelEvidence targetModelEvidence,
    MovementStateFrame agentMovementStates,
    MovementStateFrame targetMovementStates,
    CombatTelemetry combatTelemetry
) {}
