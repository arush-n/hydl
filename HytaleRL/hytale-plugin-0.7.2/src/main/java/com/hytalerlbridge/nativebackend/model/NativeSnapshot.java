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
    CombatTelemetry combatTelemetry,
    /** Agent Mana stat maximum (0 when absent); the value is {@code observation().mana()}. */
    double agentManaMaximum,
    /** Server StateSupport state names (phase 4b, hit interrupt readout); empty when the entity or state is absent. */
    String agentStateName,
    String targetStateName,
    /** Active entity effects as {@code effectId=remainingSeconds;...} (phase 4b, flinch and stun readout). */
    String agentStatusText,
    String targetStatusText
) {}
