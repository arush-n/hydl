package com.hytalerlbridge.policy.perception.acquisition;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.policy.perception.projection.AbilityProjection;

/**
 * Read-only boundary for combat facts owned by the native interaction runtime.
 *
 * <p>The production bridge implementation reads the engine's per-entity
 * interaction manager, status effects, motion controller and knockback
 * component on the current call. It retains no gameplay state. Reimplementing
 * those rules in this mod would create a second combat definition, so a live
 * policy tick is skipped unless one coherent lifecycle row is supplied.
 * Interaction admission is deliberately absent: the action sink owns that
 * caller-side fact at the point it issues a request.
 */
@FunctionalInterface
public interface CombatLifecycleEvidenceSource {
    String TARGET_ATTACK_SOURCE_IDLE = "idle";
    String TARGET_ATTACK_SOURCE_ACTIVE_CHAIN = "active_interaction_chain";
    String TARGET_ATTACK_SOURCE_ACTION_LIST_CURSOR =
        "role_action_list_cursor";

    /** Capture both actors at the same engine tick as the public evidence. */
    Evidence capture(
        Ref<EntityStore> agent,
        Ref<EntityStore> target,
        Store<EntityStore> store,
        int policySlot,
        float deltaTime
    );

    /**
     * Entity-major lifecycle state. Arrays are copied on construction and read.
     *
     * @param appliedVelocity six floats: agent xyz, then target xyz
     * @param dodgeInvulnerabilityRemaining one clock per entity
     * @param targetReportedAbilitySlot authored learner ability slot, or -1
     * @param targetAttackEvidenceSource native source for the reported slot
     * @param targetAttackSequenceCursor current role cursor, or -1
     * @param targetAttackSequenceCount authored sequence width, or 0
     * @param targetAttackPauseSeconds engine-owned target pause clock
     * @param abilityCooldownSeconds 16 slots per entity
     * @param activeAbilitySlot one slot per entity, or -1
     * @param abilityElapsedSeconds one elapsed clock per entity
     * @param abilityLegal 16 slots per entity
     */
    record Evidence(
        boolean agentAttackExecuting,
        float agentAttackCooldownSeconds,
        boolean agentKnockbackControlLock,
        float[] appliedVelocity,
        float[] dodgeInvulnerabilityRemaining,
        int targetAttackPhase,
        float targetAttackProgress,
        int targetReportedAbilitySlot,
        String targetAttackEvidenceSource,
        int targetAttackSequenceCursor,
        int targetAttackSequenceCount,
        float targetAttackPauseSeconds,
        float[] abilityCooldownSeconds,
        int[] activeAbilitySlot,
        float[] abilityElapsedSeconds,
        boolean[] abilityLegal
    ) {
        public Evidence {
            targetAttackEvidenceSource = targetAttackEvidenceSource == null
                ? ""
                : targetAttackEvidenceSource;
            require(appliedVelocity, 6, "applied velocity");
            require(
                dodgeInvulnerabilityRemaining, 2,
                "dodge invulnerability remaining");
            require(abilityCooldownSeconds, 32, "ability cooldown");
            require(activeAbilitySlot, 2, "active ability slot");
            require(abilityElapsedSeconds, 2, "ability elapsed");
            require(abilityLegal, 32, "ability legality");
            if (targetAttackPhase < 0 || targetAttackPhase > 4) {
                throw new IllegalArgumentException(
                    "target attack phase must be in [0, 4]");
            }
            if (targetReportedAbilitySlot < -1
                || targetReportedAbilitySlot
                    >= AbilityProjection.ABILITY_CAPACITY) {
                throw new IllegalArgumentException(
                    "target ability slot must be -1 or in learner capacity");
            }
            switch (targetAttackEvidenceSource) {
                case TARGET_ATTACK_SOURCE_IDLE,
                     TARGET_ATTACK_SOURCE_ACTIVE_CHAIN -> {
                    if (targetAttackSequenceCursor != -1
                        || targetAttackSequenceCount != 0) {
                        throw new IllegalArgumentException(
                            "non-cursor target attack source has cursor state");
                    }
                }
                case TARGET_ATTACK_SOURCE_ACTION_LIST_CURSOR -> {
                    if (targetAttackSequenceCount != 5
                        || targetAttackSequenceCursor < 0
                        || targetAttackSequenceCursor
                            > targetAttackSequenceCount) {
                        throw new IllegalArgumentException(
                            "target attack cursor diagnostics are invalid");
                    }
                }
                default -> throw new IllegalArgumentException(
                    "unsupported target attack evidence source");
            }
            if (!Float.isFinite(agentAttackCooldownSeconds)
                || agentAttackCooldownSeconds < 0.0f
                || !Float.isFinite(targetAttackProgress)
                || targetAttackProgress < 0.0f
                || !Float.isFinite(targetAttackPauseSeconds)
                || targetAttackPauseSeconds < 0.0f) {
                throw new IllegalArgumentException(
                    "lifecycle clocks must be finite and nonnegative");
            }
            for (float value : abilityCooldownSeconds) {
                if (!Float.isFinite(value) || value < 0.0f) {
                    throw new IllegalArgumentException(
                        "ability cooldowns must be finite and nonnegative");
                }
            }
            for (float value : abilityElapsedSeconds) {
                if (!Float.isFinite(value) || value < 0.0f) {
                    throw new IllegalArgumentException(
                        "ability elapsed clocks must be finite and nonnegative");
                }
            }
            for (int value : activeAbilitySlot) {
                if (value < -1 || value >= AbilityProjection.ABILITY_CAPACITY) {
                    throw new IllegalArgumentException(
                        "active ability slot is outside learner capacity");
                }
            }
            for (float value : appliedVelocity) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException(
                        "applied velocity must be finite");
                }
            }
            for (float value : dodgeInvulnerabilityRemaining) {
                if (!Float.isFinite(value) || value < 0.0f) {
                    throw new IllegalArgumentException(
                        "dodge clocks must be finite and nonnegative");
                }
            }
            appliedVelocity = appliedVelocity.clone();
            dodgeInvulnerabilityRemaining =
                dodgeInvulnerabilityRemaining.clone();
            abilityCooldownSeconds = abilityCooldownSeconds.clone();
            activeAbilitySlot = activeAbilitySlot.clone();
            abilityElapsedSeconds = abilityElapsedSeconds.clone();
            abilityLegal = abilityLegal.clone();
        }

        @Override public float[] appliedVelocity() {
            return appliedVelocity.clone();
        }

        @Override public float[] dodgeInvulnerabilityRemaining() {
            return dodgeInvulnerabilityRemaining.clone();
        }

        @Override public float[] abilityCooldownSeconds() {
            return abilityCooldownSeconds.clone();
        }

        @Override public int[] activeAbilitySlot() {
            return activeAbilitySlot.clone();
        }

        @Override public float[] abilityElapsedSeconds() {
            return abilityElapsedSeconds.clone();
        }

        @Override public boolean[] abilityLegal() {
            return abilityLegal.clone();
        }
    }

    private static void require(float[] values, int width, String name) {
        if (values == null || values.length != width) {
            throw new IllegalArgumentException(name + " width drift");
        }
    }

    private static void require(int[] values, int width, String name) {
        if (values == null || values.length != width) {
            throw new IllegalArgumentException(name + " width drift");
        }
    }

    private static void require(boolean[] values, int width, String name) {
        if (values == null || values.length != width) {
            throw new IllegalArgumentException(name + " width drift");
        }
    }
}
