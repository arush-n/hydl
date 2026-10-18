package com.hytalerlbridge.nativebackend.policy.lifecycle;

import java.util.Arrays;

/**
 * Immutable, reflection-friendly lifecycle row for one policy actor pair.
 *
 * <p>The standalone policy mod deliberately does not link against the bridge
 * jar. It reads this record through reflection, so the schema/version and
 * primitive accessor surface are part of the bridge boundary.</p>
 */
public record NativePolicyLifecycleSnapshot(
    boolean available,
    String unavailableReason,
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
    public static final int ENTITY_COUNT = 2;
    public static final int ABILITY_CAPACITY = 16;
    public static final String TARGET_ATTACK_SOURCE_IDLE = "idle";
    public static final String TARGET_ATTACK_SOURCE_ACTIVE_CHAIN =
        "active_interaction_chain";
    public static final String TARGET_ATTACK_SOURCE_ACTION_LIST_CURSOR =
        "role_action_list_cursor";

    public NativePolicyLifecycleSnapshot {
        unavailableReason = unavailableReason == null ? "" : unavailableReason;
        targetAttackEvidenceSource = targetAttackEvidenceSource == null
            ? ""
            : targetAttackEvidenceSource;
        requireWidth(appliedVelocity, ENTITY_COUNT * 3, "appliedVelocity");
        requireWidth(
            dodgeInvulnerabilityRemaining,
            ENTITY_COUNT,
            "dodgeInvulnerabilityRemaining"
        );
        requireWidth(
            abilityCooldownSeconds,
            ENTITY_COUNT * ABILITY_CAPACITY,
            "abilityCooldownSeconds"
        );
        requireWidth(activeAbilitySlot, ENTITY_COUNT, "activeAbilitySlot");
        requireWidth(
            abilityElapsedSeconds,
            ENTITY_COUNT,
            "abilityElapsedSeconds"
        );
        requireWidth(
            abilityLegal,
            ENTITY_COUNT * ABILITY_CAPACITY,
            "abilityLegal"
        );
        if (available != unavailableReason.isBlank()) {
            throw new IllegalArgumentException(
                "exactly one of available and unavailableReason is required"
            );
        }
        if (targetAttackPhase < 0 || targetAttackPhase > 4) {
            throw new IllegalArgumentException("targetAttackPhase is out of range");
        }
        if (
            targetReportedAbilitySlot < -1
                || targetReportedAbilitySlot >= ABILITY_CAPACITY
        ) {
            throw new IllegalArgumentException(
                "targetReportedAbilitySlot is out of range"
            );
        }
        switch (targetAttackEvidenceSource) {
            case TARGET_ATTACK_SOURCE_IDLE,
                 TARGET_ATTACK_SOURCE_ACTIVE_CHAIN -> {
                if (
                    targetAttackSequenceCursor != -1
                        || targetAttackSequenceCount != 0
                ) {
                    throw new IllegalArgumentException(
                        "non-cursor target attack source has cursor state"
                    );
                }
            }
            case TARGET_ATTACK_SOURCE_ACTION_LIST_CURSOR -> {
                if (
                    targetAttackSequenceCount != 5
                        || targetAttackSequenceCursor < 0
                        || targetAttackSequenceCursor
                            > targetAttackSequenceCount
                ) {
                    throw new IllegalArgumentException(
                        "target attack cursor diagnostics are out of range"
                    );
                }
            }
            default -> throw new IllegalArgumentException(
                "targetAttackEvidenceSource is unsupported"
            );
        }
        requireFiniteNonnegative(
            agentAttackCooldownSeconds,
            "agentAttackCooldownSeconds"
        );
        requireFiniteNonnegative(targetAttackProgress, "targetAttackProgress");
        requireFiniteNonnegative(
            targetAttackPauseSeconds,
            "targetAttackPauseSeconds"
        );
        requireFinite(appliedVelocity, "appliedVelocity");
        requireFiniteNonnegative(
            dodgeInvulnerabilityRemaining,
            "dodgeInvulnerabilityRemaining"
        );
        requireFiniteNonnegative(
            abilityCooldownSeconds,
            "abilityCooldownSeconds"
        );
        requireFiniteNonnegative(
            abilityElapsedSeconds,
            "abilityElapsedSeconds"
        );
        for (int slot : activeAbilitySlot) {
            if (slot < -1 || slot >= ABILITY_CAPACITY) {
                throw new IllegalArgumentException(
                    "activeAbilitySlot is out of range"
                );
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

    public static NativePolicyLifecycleSnapshot unavailable(String reason) {
        String normalized = reason == null || reason.isBlank()
            ? "unavailable"
            : reason;
        int[] active = new int[ENTITY_COUNT];
        Arrays.fill(active, -1);
        return new NativePolicyLifecycleSnapshot(
            false,
            normalized,
            false,
            0.0f,
            false,
            new float[ENTITY_COUNT * 3],
            new float[ENTITY_COUNT],
            0,
            0.0f,
            -1,
            TARGET_ATTACK_SOURCE_IDLE,
            -1,
            0,
            0.0f,
            new float[ENTITY_COUNT * ABILITY_CAPACITY],
            active,
            new float[ENTITY_COUNT],
            new boolean[ENTITY_COUNT * ABILITY_CAPACITY]
        );
    }

    @Override
    public float[] appliedVelocity() {
        return appliedVelocity.clone();
    }

    @Override
    public float[] dodgeInvulnerabilityRemaining() {
        return dodgeInvulnerabilityRemaining.clone();
    }

    @Override
    public float[] abilityCooldownSeconds() {
        return abilityCooldownSeconds.clone();
    }

    @Override
    public int[] activeAbilitySlot() {
        return activeAbilitySlot.clone();
    }

    @Override
    public float[] abilityElapsedSeconds() {
        return abilityElapsedSeconds.clone();
    }

    @Override
    public boolean[] abilityLegal() {
        return abilityLegal.clone();
    }

    private static void requireWidth(float[] value, int width, String name) {
        if (value == null || value.length != width) {
            throw new IllegalArgumentException(name + " width drift");
        }
    }

    private static void requireWidth(int[] value, int width, String name) {
        if (value == null || value.length != width) {
            throw new IllegalArgumentException(name + " width drift");
        }
    }

    private static void requireWidth(boolean[] value, int width, String name) {
        if (value == null || value.length != width) {
            throw new IllegalArgumentException(name + " width drift");
        }
    }

    private static void requireFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must be finite");
            }
        }
    }

    private static void requireFiniteNonnegative(float value, String name) {
        if (!Float.isFinite(value) || value < 0.0f) {
            throw new IllegalArgumentException(
                name + " must be finite and nonnegative"
            );
        }
    }

    private static void requireFiniteNonnegative(float[] values, String name) {
        requireFinite(values, name);
        for (float value : values) {
            if (value < 0.0f) {
                throw new IllegalArgumentException(name + " must be nonnegative");
            }
        }
    }
}

