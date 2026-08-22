package com.hytalerlbridge.policy;

import com.hytalerlbridge.policy.perception.projection.MovementProjection;
import com.hytalerlbridge.policy.perception.projection.ActionMaskProjection;
import com.hytalerlbridge.policy.perception.projection.AbilityProjection;
import com.hytalerlbridge.policy.perception.projection.InventoryProjection;
import com.hytalerlbridge.policy.perception.projection.GeometryProjection;
import com.hytalerlbridge.policy.perception.projection.LightProjection;
import com.hytalerlbridge.policy.perception.projection.MechanicsProjection;
import com.hytalerlbridge.policy.perception.projection.WorldProjection;

/**
 * Derives observation fields from raw simulation quantities.
 *
 * <p>This is the step above {@link ObservationAssembler}. That class is
 * certified to turn 37 pre-computed structured fields into the 8,271-column
 * vector; this one computes those fields from things a server can read straight
 * off its components -- velocity, orientation, health, attack state.
 *
 * <p>Ported from {@code observation/encoder.py}, which is the same code the
 * native path runs: {@code NativeActorEvidenceAssembler} rebuilds a JAX state
 * from the wire and calls the shared projector, so there is one definition to
 * match rather than a separate native one.
 *
 * <p><b>Tolerance, unlike the assembler.</b> {@code AssembleTest} demands exact
 * equality because every operation there is order-preserving on values JAX had
 * already produced. Here real arithmetic happens: JAX evaluates in float32
 * while Java's {@code Math.sin}/{@code cos} are double. The results differ by
 * float32 rounding, so the test uses a small absolute tolerance rather than
 * pretending to bit-identity it cannot have.
 */
public final class Projection {

    /** Column order of {@code self_f32}; see {@code SELF_FLOAT_FEATURES}. */
    public static final String[] SELF_FEATURES = {
        "forward_velocity", "right_velocity", "vertical_velocity",
        "health_fraction", "yaw_sin", "yaw_cos", "pitch", "grounded",
        "attack_executing", "attack_cooldown", "knockback_control_lock",
        "damage_recovery", "applied_vertical_velocity", "fall_speed",
        "motion_delta", "alive",
    };

    public static final int SELF_SIZE = 16;

    /** Column order of {@code target_f32}; see {@code TARGET_FLOAT_FEATURES}. */
    public static final String[] TARGET_FEATURES = {
        "relative_forward", "relative_right", "relative_up",
        "relative_velocity_forward", "relative_velocity_right",
        "relative_velocity_up", "planar_distance", "distance",
        "health_fraction", "bearing_sin", "bearing_cos", "facing_error",
        "head_facing_error", "head_pitch", "attack_progress", "visible",
    };

    public static final int TARGET_SIZE = 16;

    /**
     * Column order of {@code combat_f32} — the 24-value actor observation.
     *
     * <p>Only 18 of these reach the arsenal policy; {@code ObservationAssembler}
     * drops the five {@code target_phase_*} columns and
     * {@code target_attack_progress}. They are still derived here so this class
     * matches the JAX array one-for-one and can be certified against it
     * directly; selecting the subset is the assembler's job, not this one's.
     */
    public static final String[] COMBAT_FEATURES = {
        "agent_forward_velocity", "agent_right_velocity",
        "agent_vertical_velocity", "agent_health_fraction",
        "agent_yaw_sin", "agent_yaw_cos",
        "visible_target_forward", "visible_target_right",
        "visible_target_planar_distance", "visible_target_health_fraction",
        "target_visible", "agent_attack_executing",
        "target_phase_idle", "target_phase_windup", "target_phase_sweep",
        "target_phase_recovery", "target_phase_cooldown",
        "target_attack_progress", "target_facing_error",
        "target_forward_velocity", "target_right_velocity",
        "target_attack_index", "target_head_facing_error", "target_head_pitch",
    };

    public static final int COMBAT_SIZE = 24;

    /** Exact protocol order of the native movement-state bitset. */
    public static final String[] MOVEMENT_STATE_FEATURES = {
        "idle", "horizontal_idle", "jumping", "flying", "walking",
        "running", "sprinting", "crouching", "forced_crouching",
        "falling", "falling_far", "climbing", "in_fluid", "swimming",
        "swim_jumping", "on_ground", "mantling", "sliding", "mounting",
        "rolling", "sitting", "gliding", "sleeping",
    };

    public static final int MOVEMENT_STATE_SIZE = MovementProjection.SIZE;

    /** Attack phases, in the order the one-hot block expects. */
    public static final int PHASE_IDLE = 0;
    public static final int PHASE_COOLDOWN = 4;

    /** {@code NEARBY_ENTITY_RADIUS_BLOCKS} — the distance normaliser. */
    public static final float NEARBY_ENTITY_RADIUS_BLOCKS = 24.0f;

    /**
     * Scalars from {@code CombatParams} that the derivation divides by.
     *
     * <p>Note {@code sensorRange} is <em>not</em> the same normaliser as
     * {@link #NEARBY_ENTITY_RADIUS_BLOCKS}: the combat row divides target
     * offsets by the sensor range while the target row divides by the nearby
     * radius. They are different numbers and swapping them silently rescales
     * every distance the policy reads.
     */
    public record Params(
        double agentMaxSpeed,
        double verticalSpeedScale,
        double agentMaxHealth,
        double agentAttackPauseMaxSeconds,
        double regenDelayTicks,
        double agentWalkMaxFallSpeed,
        double loadedDt,
        double wireFixedPointScale,
        double targetChaseSpeed,
        double targetMaxHealth,
        double sensorRange,
        double facingErrorDegreesScale,
        double headPitchDegreesScale,
        double targetAbilitySlotDivisor
    ) {}

    /**
     * Both entities plus the four combat-row values this derivation copies.
     *
     * <p>{@code facingError}, {@code headFacingError}, {@code headPitch} and
     * {@code attackProgress} are produced by a separate derivation (the combat
     * row) and are inputs here rather than being recomputed, which keeps the
     * two portable independently.
     */
    public record TargetState(
        double agentX, double agentY, double agentZ,
        double agentVelocityX, double agentVelocityY, double agentVelocityZ,
        double agentYawDegrees,
        double targetX, double targetY, double targetZ,
        double targetVelocityX, double targetVelocityY, double targetVelocityZ,
        double targetHealth,
        double visible,
        double facingError,
        double headFacingError,
        double headPitch,
        double attackProgress
    ) {}

    /** The agent's own raw state for one tick, as read from the server. */
    public record SelfState(
        double velocityX,
        double velocityY,
        double velocityZ,
        double yawDegrees,
        double pitchDegrees,
        double health,
        boolean grounded,
        double attackExecuting,
        double attackCooldownSeconds,
        boolean knockbackControlLock,
        double ticksSinceDamage,
        double appliedVerticalVelocity,
        double fallSpeed,
        double motionDeltaSeconds
    ) {}

    private Projection() {
    }

    /** Decode the live bridge's row-level movement-state evidence. */
    public static MovementProjection.Result movementFeatures(
        int bits, boolean available
    ) {
        return MovementProjection.fromNativeBits(bits, available);
    }

    /** Decode the simulator's feature-level movement-state evidence. */
    public static MovementProjection.Result movementFeatures(
        int valueBits, int availableBits
    ) {
        return MovementProjection.fromBitMasks(valueBits, availableBits);
    }

    /** Project fluid, submersion and drop evidence into learner-v3 order. */
    public static WorldProjection.ActorWorldResult actorWorldFeatures(
        WorldProjection.ActorWorldInput input
    ) {
        return WorldProjection.actorWorld(input);
    }

    /** Normalize entity-major mechanics resources and publish their masks. */
    public static MechanicsProjection.ResourceResult resourceFeatures(
        MechanicsProjection.ResourceInput input
    ) {
        return MechanicsProjection.resources(input);
    }

    /** Project per-entity guard, dodge, force and liveness evidence. */
    public static MechanicsProjection.DefenseResult defenseFeatures(
        MechanicsProjection.DefenseInput input
    ) {
        return MechanicsProjection.defense(input);
    }

    /** Project active status slots and their independent availability mask. */
    public static MechanicsProjection.StatusResult statusFeatures(
        MechanicsProjection.StatusInput input
    ) {
        return MechanicsProjection.statuses(input);
    }

    /** Resolve the four native-NPC directional dodge legality bits. */
    public static boolean[] dodgeActionMask(
        ActionMaskProjection.DodgeInput input
    ) {
        return ActionMaskProjection.dodge(input);
    }

    /** Project dynamic ability features and the authoritative legality row. */
    public static AbilityProjection.Result abilityFeatures(
        AbilityProjection.Input input
    ) {
        return AbilityProjection.project(input);
    }

    /** Project actor-safe semantic inventory containers and stack tokens. */
    public static InventoryProjection.Result inventoryFeatures(
        InventoryProjection.Input input
    ) {
        return InventoryProjection.project(input);
    }

    /** Normalize and geometry-align native sky, block-light and tint rows. */
    public static LightProjection.Result lightFeatures(
        LightProjection.Input input
    ) {
        return LightProjection.project(input);
    }

    /** Normalize an already actor-legal World geometry-token row. */
    public static GeometryProjection.Result geometryFeatures(
        GeometryProjection.Input input
    ) {
        return GeometryProjection.project(input);
    }

    /**
     * The 16 columns describing the agent's own body.
     *
     * <p>Velocity is projected into the body frame, which is why yaw enters
     * twice -- once to build the basis and once as raw sin/cos. The basis is
     * {@code forward = (-sin, -cos)} and {@code right = (cos, -sin)} in XZ;
     * getting a sign wrong here produces a plausible-looking vector that is
     * simply rotated, so it is pinned by exported cases rather than reasoned
     * about.
     *
     * <p>Every column is clipped to [-1, 1] last, exactly as JAX does.
     */
    public static float[] selfFeatures(SelfState state, Params params) {
        // Every step is float32, because JAX is float32 and this expression
        // cancels hard: forward/right velocity subtracts two products of
        // similar magnitude (2.59 against 2.00 to yield 0.588 in one measured
        // case), so any precision mismatch is amplified ~20x. Computing in
        // double looks more accurate and is *further* from JAX -- measured
        // 1.17e-06 against float32's 1.49e-09 on the same row. Do not
        // "improve" this to double.
        float yawDegrees = (float) state.yawDegrees();
        float yaw = yawDegrees * (float) (Math.PI / 180.0);
        float sinYaw = (float) Math.sin(yaw);
        float cosYaw = (float) Math.cos(yaw);

        float forwardX = -sinYaw;
        float forwardZ = -cosYaw;
        float rightX = cosYaw;
        float rightZ = -sinYaw;

        float velocityX = (float) state.velocityX();
        float velocityY = (float) state.velocityY();
        float velocityZ = (float) state.velocityZ();
        float health = (float) state.health();

        float maximumSpeed = positive((float) params.agentMaxSpeed());
        float verticalScale = positive((float) params.verticalSpeedScale());

        float[] out = new float[SELF_SIZE];
        out[0] = (velocityX * forwardX + velocityZ * forwardZ) / maximumSpeed;
        out[1] = (velocityX * rightX + velocityZ * rightZ) / maximumSpeed;
        out[2] = velocityY / verticalScale;
        out[3] = health / positive((float) params.agentMaxHealth());
        out[4] = sinYaw;
        out[5] = cosYaw;
        out[6] = (float) state.pitchDegrees() / 90.0f;
        out[7] = state.grounded() ? 1.0f : 0.0f;
        out[8] = (float) state.attackExecuting();
        out[9] = (float) state.attackCooldownSeconds()
            / positive((float) params.agentAttackPauseMaxSeconds());
        out[10] = state.knockbackControlLock() ? 1.0f : 0.0f;
        out[11] = (float) state.ticksSinceDamage()
            / positive((float) params.regenDelayTicks());
        out[12] = (float) state.appliedVerticalVelocity() / verticalScale;
        out[13] = (float) state.fallSpeed()
            / positive((float) params.agentWalkMaxFallSpeed());
        out[14] = (float) state.motionDeltaSeconds()
            / positive((float) params.loadedDt());
        out[15] = health > 0.0f ? 1.0f : 0.0f;

        for (int i = 0; i < out.length; i++) {
            out[i] = Math.max(-1.0f, Math.min(1.0f, out[i]));
        }
        return out;
    }

    /**
     * The 16 columns describing the opponent, relative to the agent.
     *
     * <p>Two details are easy to miss and both are load-bearing:
     *
     * <ul>
     *   <li><b>The offset is quantised through the wire fixed-point scale</b>
     *       before anything is derived from it. That is deliberate on the JAX
     *       side — it projects privileged float positions through the same
     *       {@code Math.round}-compatible boundary Java publishes, so the
     *       learner does not get a second, slightly different distance to the
     *       same target. In-server this is the native quantisation, so keeping
     *       it is what makes the two agree.
     *   <li><b>Everything is masked by visibility first.</b> Offset, relative
     *       velocity and health fraction are zeroed when the target is not
     *       perceptible, so a zero here means "no reading", not "at my
     *       position". Deriving from unmasked values leaks privileged state.
     * </ul>
     *
     * <p>Note {@code bearing_sin} is {@code right / planar} and
     * {@code bearing_cos} is {@code forward / planar} — that ordering looks
     * transposed and is not.
     *
     * <p><b>Target health is NOT quantised here, and that is not an
     * oversight.</b> {@link #targetEvidence} (feeding the combat row) rounds
     * target health through the wire scale before taking the fraction; this
     * derivation divides the raw value. The two really do differ — checked
     * against {@code _target_observation_evidence}, which reads
     * {@code target_health / _positive(target_max_health)} with no rounding,
     * while {@code _reset.target_evidence} rounds first. Making them agree
     * would look like a tidy-up and would silently move this column.
     */
    public static float[] targetFeatures(TargetState state, Params params) {
        boolean perceptible = state.visible() > 0.5;

        float scale = (float) params.wireFixedPointScale();
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        float offsetZ = 0.0f;
        float velocityX = 0.0f;
        float velocityY = 0.0f;
        float velocityZ = 0.0f;
        float healthFraction = 0.0f;
        if (perceptible) {
            offsetX = roundToScale((float) (state.targetX() - state.agentX()), scale);
            offsetY = roundToScale((float) (state.targetY() - state.agentY()), scale);
            offsetZ = roundToScale((float) (state.targetZ() - state.agentZ()), scale);
            velocityX = (float) (state.targetVelocityX() - state.agentVelocityX());
            velocityY = (float) (state.targetVelocityY() - state.agentVelocityY());
            velocityZ = (float) (state.targetVelocityZ() - state.agentVelocityZ());
            healthFraction = (float) state.targetHealth()
                / positive((float) params.targetMaxHealth());
        }

        float yaw = (float) state.agentYawDegrees() * (float) (Math.PI / 180.0);
        float sinYaw = (float) Math.sin(yaw);
        float cosYaw = (float) Math.cos(yaw);
        float forwardX = -sinYaw;
        float forwardZ = -cosYaw;
        float rightX = cosYaw;
        float rightZ = -sinYaw;

        float forward = offsetX * forwardX + offsetZ * forwardZ;
        float right = offsetX * rightX + offsetZ * rightZ;
        float velocityForward = velocityX * forwardX + velocityZ * forwardZ;
        float velocityRight = velocityX * rightX + velocityZ * rightZ;

        float planar = (float) Math.sqrt(forward * forward + right * right);
        float distance = (float) Math.sqrt(
            offsetX * offsetX + offsetY * offsetY + offsetZ * offsetZ);
        float safePlanar = Math.max(planar, 1.0e-6f);
        float radius = NEARBY_ENTITY_RADIUS_BLOCKS;
        float targetSpeed = positive((float) params.targetChaseSpeed());
        float inverseRadius = 1.0f / radius;
        float inverseTargetSpeed = 1.0f / targetSpeed;
        float inverseVerticalSpeed =
            1.0f / positive((float) params.verticalSpeedScale());

        float[] out = new float[TARGET_SIZE];
        // XLA lowers these fixed-scale divisions to reciprocal multiplies.
        // Spell that association explicitly so Java reproduces the current
        // CUDA learner row instead of landing one ULP below it.
        out[0] = forward * inverseRadius;
        out[1] = right * inverseRadius;
        out[2] = offsetY * inverseRadius;
        out[3] = velocityForward * inverseTargetSpeed;
        out[4] = velocityRight * inverseTargetSpeed;
        out[5] = velocityY * inverseVerticalSpeed;
        out[6] = planar * inverseRadius;
        out[7] = distance * inverseRadius;
        out[8] = healthFraction;
        out[9] = right / safePlanar;
        out[10] = forward / safePlanar;
        out[11] = (float) state.facingError();
        out[12] = (float) state.headFacingError();
        out[13] = (float) state.headPitch();
        out[14] = (float) state.attackProgress();
        out[15] = perceptible ? 1.0f : 0.0f;

        for (int i = 0; i < out.length; i++) {
            out[i] = Math.max(-1.0f, Math.min(1.0f, out[i]));
        }
        return out;
    }

    /** The agent's own contribution to the combat row. */
    public record CombatSelf(
        double velocityX,
        double velocityY,
        double velocityZ,
        double yawDegrees,
        double healthFraction,
        boolean attackExecuting
    ) {}

    /**
     * Legal target evidence — what the encoder is allowed to see.
     *
     * <p>This is the boundary the JAX side draws too: {@code target_evidence}
     * filters privileged state down to this, and {@code _encode_actor_observation}
     * runs with nothing else in scope. Everything here is already masked and
     * quantised; {@link #targetEvidence} produces it.
     */
    public record TargetEvidence(
        boolean perceptible,
        float offsetX, float offsetY, float offsetZ,
        float planarDistance,
        float healthFraction,
        float[] attackPhaseOneHot,
        float attackProgress,
        float facingErrorDegrees,
        float velocityX, float velocityY, float velocityZ,
        float normalizedAttackIndex,
        float headFacingErrorDegrees,
        float headPitchDegrees
    ) {}

    /** Raw target state, before masking and quantisation. */
    public record TargetRaw(
        double agentX, double agentY, double agentZ,
        double targetX, double targetY, double targetZ,
        double targetVelocityX, double targetVelocityY, double targetVelocityZ,
        double targetHealth,
        double targetYawDegrees,
        double targetHeadYawDegrees,
        double targetHeadPitchDegrees,
        boolean perceptible,
        boolean primaryTarget,
        int attackPhase,
        double attackProgress,
        int reportedAbilitySlot
    ) {}

    /**
     * The 24-column combat row — {@code _encode_actor_observation}.
     *
     * <p>Structurally this is the same body-frame projection as
     * {@link #selfFeatures}, applied to the agent's velocity, the target's
     * offset and the target's velocity in turn. The three divide by
     * <em>different</em> scales ({@code agentMaxSpeed}, {@code sensorRange},
     * {@code targetChaseSpeed}); they are not interchangeable.
     *
     * <p>Unlike the other two derivations this one divides directly rather than
     * through the {@code positive()} guard, matching the JAX source.
     */
    public static float[] combatFeatures(
        CombatSelf self, TargetEvidence target, Params params
    ) {
        float yaw = (float) self.yawDegrees() * (float) (Math.PI / 180.0);
        float sinYaw = (float) Math.sin(yaw);
        float cosYaw = (float) Math.cos(yaw);
        float forwardX = -sinYaw;
        float forwardZ = -cosYaw;
        float rightX = cosYaw;
        float rightZ = -sinYaw;

        float velocityX = (float) self.velocityX();
        float velocityZ = (float) self.velocityZ();
        float maximumSpeed = (float) params.agentMaxSpeed();
        float sensorRange = (float) params.sensorRange();
        float chaseSpeed = (float) params.targetChaseSpeed();
        float facingScale = (float) params.facingErrorDegreesScale();

        float[] out = new float[COMBAT_SIZE];
        out[0] = (velocityX * forwardX + velocityZ * forwardZ) / maximumSpeed;
        out[1] = (velocityX * rightX + velocityZ * rightZ) / maximumSpeed;
        out[2] = (float) self.velocityY() / (float) params.verticalSpeedScale();
        out[3] = (float) self.healthFraction();
        out[4] = sinYaw;
        out[5] = cosYaw;
        out[6] = (target.offsetX() * forwardX + target.offsetZ() * forwardZ)
            / sensorRange;
        out[7] = (target.offsetX() * rightX + target.offsetZ() * rightZ)
            / sensorRange;
        out[8] = target.planarDistance() / sensorRange;
        out[9] = target.healthFraction();
        out[10] = target.perceptible() ? 1.0f : 0.0f;
        out[11] = self.attackExecuting() ? 1.0f : 0.0f;
        System.arraycopy(target.attackPhaseOneHot(), 0, out, 12, 5);
        out[17] = target.attackProgress();
        out[18] = target.facingErrorDegrees() / facingScale;
        out[19] = (target.velocityX() * forwardX + target.velocityZ() * forwardZ)
            / chaseSpeed;
        out[20] = (target.velocityX() * rightX + target.velocityZ() * rightZ)
            / chaseSpeed;
        out[21] = target.normalizedAttackIndex();
        out[22] = target.headFacingErrorDegrees() / facingScale;
        out[23] = target.headPitchDegrees() / (float) params.headPitchDegreesScale();

        for (int i = 0; i < out.length; i++) {
            out[i] = Math.max(-1.0f, Math.min(1.0f, out[i]));
        }
        return out;
    }

    /**
     * Filter raw target state into legal evidence — {@code target_evidence}.
     *
     * <p>Three details here are load-bearing and none of them are guessable:
     *
     * <ul>
     *   <li><b>Two different offsets are in play.</b> Position and planar
     *       distance are quantised through the wire scale, but the bearing used
     *       for {@code facing_error} is computed from the <em>unquantised</em>
     *       offset. Using one for both is a plausible-looking error worth about
     *       a tenth of a block of bearing noise.
     *   <li><b>Target health is quantised too</b>, before the fraction is taken.
     *   <li><b>Masking is not uniform.</b> Most fields go to zero when the
     *       target is imperceptible, but {@code normalizedAttackIndex} carries a
     *       third state: {@code -1} means "visible, but not attacking", while
     *       {@code 0} means "cannot see it". A nonnegative value is the authored
     *       Arsenal ability slot divided by the fixed learner-slot divisor
     *       (15), not a legacy attack index divided by the number of attacks.
     *       Collapsing or rescaling those states loses information the policy
     *       was trained on.
     * </ul>
     */
    public static TargetEvidence targetEvidence(TargetRaw raw, Params params) {
        float scale = (float) params.wireFixedPointScale();
        float rawOffsetX = (float) (raw.targetX() - raw.agentX());
        float rawOffsetY = (float) (raw.targetY() - raw.agentY());
        float rawOffsetZ = (float) (raw.targetZ() - raw.agentZ());

        float wireX = roundToScale(rawOffsetX, scale);
        float wireY = roundToScale(rawOffsetY, scale);
        float wireZ = roundToScale(rawOffsetZ, scale);

        boolean perceptible = raw.perceptible();
        float legal = perceptible ? 1.0f : 0.0f;

        float offsetX = perceptible ? wireX : 0.0f;
        float offsetY = perceptible ? wireY : 0.0f;
        float offsetZ = perceptible ? wireZ : 0.0f;
        float planarDistance = perceptible
            ? (float) Math.sqrt(wireX * wireX + wireZ * wireZ)
            : 0.0f;
        float healthFraction = perceptible
            ? roundToScale((float) raw.targetHealth(), scale)
                / (float) params.targetMaxHealth()
            : 0.0f;

        // A non-primary target reports no attack at all, regardless of what the
        // shared attack state machine happens to hold.
        int phase = raw.primaryTarget() ? raw.attackPhase() : PHASE_IDLE;
        float progress = raw.primaryTarget() ? (float) raw.attackProgress() : 0.0f;
        int reportedSlot = raw.primaryTarget() ? raw.reportedAbilitySlot() : -1;

        float[] oneHot = new float[5];
        oneHot[Math.max(PHASE_IDLE, Math.min(PHASE_COOLDOWN, phase))] = legal;

        float normalizedIndex;
        if (!perceptible) {
            normalizedIndex = 0.0f;
        } else if (reportedSlot < 0) {
            normalizedIndex = -1.0f;
        } else {
            normalizedIndex = reportedSlot
                / (float) params.targetAbilitySlotDivisor();
        }

        // Bearing runs off the *unquantised* offset, unlike everything above.
        float toAgentX = -rawOffsetX;
        float toAgentZ = -rawOffsetZ;
        float facingDistance = (float) Math.sqrt(
            toAgentX * toAgentX + toAgentZ * toAgentZ);
        float bearing = canonicalBearing(toAgentX, toAgentZ);
        float facingError = facingDistance <= 1.0e-9f
            ? 0.0f
            : normalizeDegrees(bearing - (float) raw.targetYawDegrees());
        float headYaw = raw.primaryTarget()
            ? (float) raw.targetHeadYawDegrees()
            : (float) raw.targetYawDegrees();
        float headFacingError = facingDistance <= 1.0e-9f
            ? 0.0f
            : normalizeDegrees(bearing - headYaw);

        return new TargetEvidence(
            perceptible,
            offsetX, offsetY, offsetZ,
            planarDistance,
            healthFraction,
            oneHot,
            perceptible ? progress : 0.0f,
            perceptible ? facingError : 0.0f,
            perceptible ? (float) raw.targetVelocityX() : 0.0f,
            perceptible ? (float) raw.targetVelocityY() : 0.0f,
            perceptible ? (float) raw.targetVelocityZ() : 0.0f,
            normalizedIndex,
            perceptible ? headFacingError : 0.0f,
            // Head pitch is gated by visibility AND primacy, unlike head
            // *facing* error immediately above it, which is gated by visibility
            // alone and falls back to the body yaw. The asymmetry is real: a
            // non-primary target reports pitch 0 even when plainly visible.
            perceptible && raw.primaryTarget()
                ? (float) raw.targetHeadPitchDegrees()
                : 0.0f);
    }

    /**
     * {@code rad2deg(atan2(-dx, -dz))} with the source's snap at the wrap point.
     *
     * <p>The snap matters: without it a target directly behind lands on -180
     * about half the time, which flips the sign of every derived facing error.
     */
    private static float canonicalBearing(float dx, float dz) {
        float bearing = (float) Math.toDegrees(Math.atan2(-dx, -dz));
        if (Math.abs(Math.abs(bearing) - 180.0f) < 1.0e-5f) {
            return 180.0f;
        }
        return bearing;
    }

    /**
     * Wrap to [-180, 180). Uses floored modulo because the source is
     * {@code jnp.mod}, which follows Python and takes the sign of the divisor —
     * Java's {@code %} takes the sign of the dividend and is wrong here for
     * every negative input.
     */
    private static float normalizeDegrees(float value) {
        float shifted = value + 180.0f;
        float wrapped = shifted - 360.0f * (float) Math.floor(shifted / 360.0f);
        return wrapped - 180.0f;
    }

    /**
     * {@code floor(v * scale + 0.5) / scale} — the wire's fixed-point
     * quantisation, matching Java's {@code Math.round} half-up rule.
     */
    private static float roundToScale(float value, float scale) {
        return (float) Math.floor(value * scale + 0.5f) / scale;
    }

    /**
     * Divide-guard matching {@code encoder._positive}. Note it takes the
     * absolute value, so a negative parameter divides as its magnitude --
     * `agent_walk_max_fall_speed` is negative and relies on this.
     */
    private static float positive(float value) {
        return Math.max(Math.abs(value), 1.0e-6f);
    }

}
