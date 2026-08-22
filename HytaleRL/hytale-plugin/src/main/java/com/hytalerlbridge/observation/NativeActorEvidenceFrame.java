package com.hytalerlbridge.observation;

import com.hytalerlbridge.geometry.GeometryContract;
import java.util.Arrays;
import java.util.List;

/**
 * Versioned native inputs for the shared learner-v3 actor-evidence projector.
 *
 * <p>This is deliberately not a Java reimplementation of the JAX encoder.
 * Native supplies copied server facts and independent availability; the host
 * combines them with episode-pinned loadout/ruleset data and invokes the one
 * shared encoder. Unsupported evidence is zero plus {@code available=false},
 * never a guessed value.</p>
 */
public record NativeActorEvidenceFrame(
    boolean available,
    String unavailableReason,
    long worldTick,
    List<Actor> actors,
    List<Ability> abilities,
    boolean worldGeometryAvailable,
    boolean roleOpaqueCellMaskAvailable,
    boolean[] roleOpaqueCellMask,
    boolean[] skillActionMask,
    boolean jumpActionAvailable,
    boolean guardActionAvailable,
    boolean[] dodgeActionMask,
    boolean[][] doorActionMask,
    int loadoutFailure,
    int mechanicsFailureBits,
    int arsenalFailureBits
) {
    public static final String SCHEMA =
        "hytalerl_native_learner_v3_actor_evidence_inputs_v4";
    public static final int VERSION = 4;
    public static final String CONTRACT_SHA256 =
        "1A7CFF4EE96218C4AA48F10E5F65BA0F92B8372C508C66F1EE28D2A811723CD5";
    public static final int ENTITY_COUNT = 2;
    public static final int RESOURCE_COUNT = 7;
    public static final int STATUS_CAPACITY = 8;
    public static final int ABILITY_CAPACITY = 16;
    public static final int SKILL_COUNT = 9;
    public static final int DODGE_DIRECTION_COUNT = 4;
    public static final int DOOR_CANDIDATE_CAPACITY = 8;
    public static final int DOOR_INTENT_COUNT = 3;
    // 8, not 7. The learner contract publishes eight defense columns
    // (DEFENSE_FLOAT_FEATURES in observation/v3/schema/contract.py) and the
    // native decoder validates the incoming row against DEFENSE_FLOAT_SIZE == 8
    // (native/codec/decode.py). At 7 the bridge could not fill the eighth
    // column -- `locomotion_stamina_fraction`, added with locomotion stamina --
    // so the array was one short of the shape the decoder demands.
    public static final int DEFENSE_VALUE_COUNT = 8;
    public static final int ACTOR_WORLD_VALUE_COUNT = 5;
    public static final int ACTOR_WORLD_MASK_COUNT = 3;
    public static final int GEOMETRY_CELL_COUNT = GeometryContract.CELL_COUNT;

    public NativeActorEvidenceFrame {
        unavailableReason = unavailableReason == null ? "" : unavailableReason;
        actors = actors == null ? List.of() : List.copyOf(actors);
        abilities = abilities == null ? List.of() : List.copyOf(abilities);
        roleOpaqueCellMask = copy(
            roleOpaqueCellMask,
            GEOMETRY_CELL_COUNT,
            "roleOpaqueCellMask"
        );
        skillActionMask = copy(skillActionMask, SKILL_COUNT, "skillActionMask");
        dodgeActionMask = copy(
            dodgeActionMask,
            DODGE_DIRECTION_COUNT,
            "dodgeActionMask"
        );
        doorActionMask = copyDoorMask(doorActionMask);
        if (available) {
            if (!unavailableReason.isEmpty()) {
                throw new IllegalArgumentException(
                    "available actor evidence cannot carry an unavailable reason"
                );
            }
            if (actors.size() != ENTITY_COUNT) {
                throw new IllegalArgumentException(
                    "available actor evidence must contain exactly "
                        + ENTITY_COUNT
                        + " actors"
                );
            }
            if (abilities.size() != ABILITY_CAPACITY) {
                throw new IllegalArgumentException(
                    "available actor evidence must contain exactly "
                        + ABILITY_CAPACITY
                        + " ability slots"
                );
            }
        } else if (
            !actors.isEmpty()
                || !abilities.isEmpty()
                || any(skillActionMask)
                || jumpActionAvailable
                || guardActionAvailable
                || any(dodgeActionMask)
                || any(doorActionMask)
                || worldGeometryAvailable
                || roleOpaqueCellMaskAvailable
                || any(roleOpaqueCellMask)
        ) {
            throw new IllegalArgumentException(
                "unavailable actor evidence must not carry usable rows"
            );
        }
    }

    public static NativeActorEvidenceFrame unavailable(String reason) {
        String value = reason == null || reason.isBlank()
            ? "not_negotiated"
            : reason.trim();
        return new NativeActorEvidenceFrame(
            false,
            value,
            0L,
            List.of(),
            List.of(),
            false,
            false,
            new boolean[GEOMETRY_CELL_COUNT],
            new boolean[SKILL_COUNT],
            false,
            false,
            new boolean[DODGE_DIRECTION_COUNT],
            new boolean[DOOR_CANDIDATE_CAPACITY][DOOR_INTENT_COUNT],
            0,
            0,
            0
        );
    }

    @Override
    public boolean[] roleOpaqueCellMask() {
        return roleOpaqueCellMask.clone();
    }

    @Override
    public boolean[] skillActionMask() {
        return skillActionMask.clone();
    }

    @Override
    public boolean[] dodgeActionMask() {
        return dodgeActionMask.clone();
    }

    @Override
    public boolean[][] doorActionMask() {
        return copyDoorMask(doorActionMask);
    }

    /**
     * Copied per-entity server facts. Stable IDs are host-resolved from the
     * reset-pinned asset strings; runtime asset indices are diagnostic only.
     */
    public record Actor(
        int entityId,
        boolean present,
        boolean perceptible,
        String roleId,
        String itemId,
        int itemRuntimeIndex,
        int activeAbilitySlot,
        double[] position,
        double[] velocity,
        MotionForce motionForce,
        double yawDegrees,
        double pitchDegrees,
        double health,
        double maxHealth,
        double[] resourceValues,
        double[] resourceMaximums,
        boolean[] resourceAvailable,
        double[] defenseValues,
        boolean[] defenseAvailable,
        List<Status> statuses,
        double[] actorWorldValues,
        boolean[] actorWorldAvailable,
        MovementStateFrame movementStates
    ) {
        public Actor {
            roleId = roleId == null ? "" : roleId;
            itemId = itemId == null ? "" : itemId;
            position = finiteCopy(position, 3, "position");
            velocity = finiteCopy(velocity, 3, "velocity");
            motionForce = motionForce == null
                ? MotionForce.unavailable()
                : motionForce;
            resourceValues = finiteCopy(
                resourceValues,
                RESOURCE_COUNT,
                "resourceValues"
            );
            resourceMaximums = finiteCopy(
                resourceMaximums,
                RESOURCE_COUNT,
                "resourceMaximums"
            );
            resourceAvailable = copy(
                resourceAvailable,
                RESOURCE_COUNT,
                "resourceAvailable"
            );
            defenseValues = finiteCopy(
                defenseValues,
                DEFENSE_VALUE_COUNT,
                "defenseValues"
            );
            defenseAvailable = copy(
                defenseAvailable,
                DEFENSE_VALUE_COUNT,
                "defenseAvailable"
            );
            statuses = statuses == null ? List.of() : List.copyOf(statuses);
            actorWorldValues = finiteCopy(
                actorWorldValues,
                ACTOR_WORLD_VALUE_COUNT,
                "actorWorldValues"
            );
            actorWorldAvailable = copy(
                actorWorldAvailable,
                ACTOR_WORLD_MASK_COUNT,
                "actorWorldAvailable"
            );
            movementStates = movementStates == null
                ? MovementStateFrame.unavailable()
                : movementStates;
            requireFinite(yawDegrees, "yawDegrees");
            requireFinite(pitchDegrees, "pitchDegrees");
            requireFinite(health, "health");
            requireFinite(maxHealth, "maxHealth");
            if (statuses.size() > STATUS_CAPACITY) {
                throw new IllegalArgumentException(
                    "status evidence exceeds fixed capacity"
                );
            }
            if (!present && (
                perceptible
                    || any(resourceAvailable)
                    || motionForce.carriesEvidence()
            )) {
                throw new IllegalArgumentException(
                    "absent actor cannot carry perceptible/resource/force evidence"
                );
            }
        }

        @Override
        public double[] position() {
            return position.clone();
        }

        @Override
        public double[] velocity() {
            return velocity.clone();
        }

        @Override
        public double[] resourceValues() {
            return resourceValues.clone();
        }

        @Override
        public double[] resourceMaximums() {
            return resourceMaximums.clone();
        }

        @Override
        public boolean[] resourceAvailable() {
            return resourceAvailable.clone();
        }

        @Override
        public double[] defenseValues() {
            return defenseValues.clone();
        }

        @Override
        public boolean[] defenseAvailable() {
            return defenseAvailable.clone();
        }

        @Override
        public double[] actorWorldValues() {
            return actorWorldValues.clone();
        }

        @Override
        public boolean[] actorWorldAvailable() {
            return actorWorldAvailable.clone();
        }
    }

    /**
     * Lossless force channels captured at the policy-visible server boundary.
     *
     * <p>Legacy controller velocity, configured split velocity, and the
     * staged knockback component are distinct server facts.  The projected
     * vector is present only when those facts have one unambiguous mapping to
     * the learner mechanics state; otherwise the raw channels remain on the
     * wire and {@code projectedAvailable=false} fails the actor adapter
     * closed.</p>
     */
    public record MotionForce(
        boolean legacyExternalAvailable,
        double[] legacyExternalVelocity,
        boolean configuredAppliedAvailable,
        double[] configuredAppliedVelocity,
        int configuredAppliedCount,
        boolean pendingKnockbackAvailable,
        double[] pendingKnockbackVelocity,
        boolean projectedAvailable,
        double[] projectedVelocity,
        int projectionSource
    ) {
        public static final int SOURCE_NONE = 0;
        public static final int SOURCE_LEGACY_EXTERNAL = 1;
        public static final int SOURCE_CONFIGURED_APPLIED = 2;
        public static final int SOURCE_COMBINED_CURRENT = 3;
        public static final int SOURCE_PENDING_KNOCKBACK = 4;
        public static final int SOURCE_AMBIGUOUS_PENDING = 5;
        public static final int SOURCE_UNAVAILABLE = 6;

        public MotionForce {
            legacyExternalVelocity = finiteCopy(
                legacyExternalVelocity,
                3,
                "legacyExternalVelocity"
            );
            configuredAppliedVelocity = finiteCopy(
                configuredAppliedVelocity,
                3,
                "configuredAppliedVelocity"
            );
            pendingKnockbackVelocity = finiteCopy(
                pendingKnockbackVelocity,
                3,
                "pendingKnockbackVelocity"
            );
            projectedVelocity = finiteCopy(
                projectedVelocity,
                3,
                "projectedVelocity"
            );
            if (configuredAppliedCount < 0) {
                throw new IllegalArgumentException(
                    "configuredAppliedCount must be nonnegative"
                );
            }
            if (
                (!legacyExternalAvailable && any(legacyExternalVelocity))
                    || (
                        !configuredAppliedAvailable
                            && (
                                configuredAppliedCount != 0
                                    || any(configuredAppliedVelocity)
                            )
                    )
                    || (
                        !pendingKnockbackAvailable
                            && any(pendingKnockbackVelocity)
                    )
                    || (!projectedAvailable && any(projectedVelocity))
            ) {
                throw new IllegalArgumentException(
                    "unavailable force channels must be structurally empty"
                );
            }
            if (
                projectionSource < SOURCE_NONE
                    || projectionSource > SOURCE_UNAVAILABLE
            ) {
                throw new IllegalArgumentException(
                    "force projection source is out of range"
                );
            }
            if (
                projectedAvailable
                    == (
                        projectionSource == SOURCE_AMBIGUOUS_PENDING
                            || projectionSource == SOURCE_UNAVAILABLE
                    )
            ) {
                throw new IllegalArgumentException(
                    "force projection availability disagrees with source"
                );
            }
        }

        public static MotionForce unavailable() {
            return new MotionForce(
                false,
                new double[3],
                false,
                new double[3],
                0,
                false,
                new double[3],
                false,
                new double[3],
                SOURCE_UNAVAILABLE
            );
        }

        public static MotionForce resolve(
            boolean legacyAvailable,
            double[] legacyVelocity,
            boolean configuredAvailable,
            double[] configuredVelocity,
            int configuredCount,
            boolean pendingAvailable,
            double[] pendingVelocity
        ) {
            double[] legacy = finiteCopy(
                legacyVelocity,
                3,
                "legacyVelocity"
            );
            double[] configured = finiteCopy(
                configuredVelocity,
                3,
                "configuredVelocity"
            );
            double[] pending = finiteCopy(
                pendingVelocity,
                3,
                "pendingVelocity"
            );
            if (!legacyAvailable || !configuredAvailable || !pendingAvailable) {
                return new MotionForce(
                    legacyAvailable,
                    legacyAvailable ? legacy : new double[3],
                    configuredAvailable,
                    configuredAvailable ? configured : new double[3],
                    configuredAvailable ? configuredCount : 0,
                    pendingAvailable,
                    pendingAvailable ? pending : new double[3],
                    false,
                    new double[3],
                    SOURCE_UNAVAILABLE
                );
            }
            double[] current = new double[] {
                legacy[0] + configured[0],
                legacy[1] + configured[1],
                legacy[2] + configured[2]
            };
            boolean legacyActive = any(legacy);
            boolean configuredActive = configuredCount > 0 || any(configured);
            boolean currentActive = any(current);
            boolean pendingActive = any(pending);
            if (currentActive && pendingActive) {
                return new MotionForce(
                    true,
                    legacy,
                    true,
                    configured,
                    configuredCount,
                    true,
                    pending,
                    false,
                    new double[3],
                    SOURCE_AMBIGUOUS_PENDING
                );
            }
            int source;
            double[] projected;
            if (currentActive) {
                source = legacyActive && configuredActive
                    ? SOURCE_COMBINED_CURRENT
                    : (
                        configuredActive
                            ? SOURCE_CONFIGURED_APPLIED
                            : SOURCE_LEGACY_EXTERNAL
                    );
                projected = current;
            } else if (pendingActive) {
                source = SOURCE_PENDING_KNOCKBACK;
                projected = pending;
            } else {
                source = SOURCE_NONE;
                projected = new double[3];
            }
            return new MotionForce(
                true,
                legacy,
                true,
                configured,
                configuredCount,
                true,
                pending,
                true,
                projected,
                source
            );
        }

        @Override
        public double[] legacyExternalVelocity() {
            return legacyExternalVelocity.clone();
        }

        @Override
        public double[] configuredAppliedVelocity() {
            return configuredAppliedVelocity.clone();
        }

        @Override
        public double[] pendingKnockbackVelocity() {
            return pendingKnockbackVelocity.clone();
        }

        @Override
        public double[] projectedVelocity() {
            return projectedVelocity.clone();
        }

        boolean carriesEvidence() {
            return legacyExternalAvailable
                || configuredAppliedAvailable
                || configuredAppliedCount != 0
                || pendingKnockbackAvailable
                || projectedAvailable
                || any(legacyExternalVelocity)
                || any(configuredAppliedVelocity)
                || any(pendingKnockbackVelocity)
                || any(projectedVelocity);
        }
    }

    /** Raw active-effect facts; the host owns semantic ID/value projection. */
    public record Status(
        int runtimeEffectIndex,
        String effectId,
        double initialDurationSeconds,
        double remainingDurationSeconds,
        boolean infinite,
        boolean debuff,
        boolean invulnerable
    ) {
        public Status {
            effectId = effectId == null ? "" : effectId;
            if (effectId.isBlank()) {
                throw new IllegalArgumentException(
                    "active status evidence requires a stable effect ID"
                );
            }
            requireFinite(initialDurationSeconds, "initialDurationSeconds");
            requireFinite(remainingDurationSeconds, "remainingDurationSeconds");
            if (initialDurationSeconds < 0.0 || remainingDurationSeconds < 0.0) {
                throw new IllegalArgumentException(
                    "status durations must be nonnegative"
                );
            }
        }
    }

    /** One reset-pinned generic interaction slot, never a weapon branch. */
    public record Ability(
        int slot,
        String interactionId,
        String interactionType,
        boolean authored,
        boolean hostLegal,
        boolean active,
        long startWorldTick,
        long finishWorldTick
    ) {
        public Ability {
            interactionId = interactionId == null ? "" : interactionId;
            interactionType = interactionType == null ? "" : interactionType;
            if (slot < 0 || slot >= ABILITY_CAPACITY) {
                throw new IllegalArgumentException("ability slot is out of range");
            }
            if (!authored && (
                !interactionId.isEmpty()
                    || !interactionType.isEmpty()
                    || hostLegal
                    || active
            )) {
                throw new IllegalArgumentException(
                    "unauthored ability slot cannot carry interaction evidence"
                );
            }
        }
    }

    private static boolean[] copy(
        boolean[] values,
        int expected,
        String label
    ) {
        boolean[] result = values == null ? new boolean[expected] : values.clone();
        if (result.length != expected) {
            throw new IllegalArgumentException(
                label + " must contain exactly " + expected + " values"
            );
        }
        return result;
    }

    private static double[] finiteCopy(
        double[] values,
        int expected,
        String label
    ) {
        double[] result = values == null ? new double[expected] : values.clone();
        if (result.length != expected) {
            throw new IllegalArgumentException(
                label + " must contain exactly " + expected + " values"
            );
        }
        for (double value : result) requireFinite(value, label);
        return result;
    }

    private static boolean[][] copyDoorMask(boolean[][] values) {
        boolean[][] result = new boolean[DOOR_CANDIDATE_CAPACITY][
            DOOR_INTENT_COUNT
        ];
        if (values == null) return result;
        if (values.length != DOOR_CANDIDATE_CAPACITY) {
            throw new IllegalArgumentException(
                "doorActionMask must contain exactly "
                    + DOOR_CANDIDATE_CAPACITY
                    + " candidates"
            );
        }
        for (int index = 0; index < values.length; index++) {
            if (values[index] == null || values[index].length != DOOR_INTENT_COUNT) {
                throw new IllegalArgumentException(
                    "each doorActionMask row must contain exactly "
                        + DOOR_INTENT_COUNT
                        + " intents"
                );
            }
            result[index] = values[index].clone();
        }
        return result;
    }

    private static boolean any(boolean[] values) {
        for (boolean value : values) if (value) return true;
        return false;
    }

    private static boolean any(double[] values) {
        for (double value : values) if (value != 0.0) return true;
        return false;
    }

    private static boolean any(boolean[][] values) {
        return Arrays.stream(values).anyMatch(NativeActorEvidenceFrame::any);
    }

    private static void requireFinite(double value, String label) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(label + " must be finite");
        }
    }
}
