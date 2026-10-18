package com.hytalerlbridge.combat.simulation;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.combat.CombatPhase;
import com.hytalerlbridge.combat.CombatRuleset;
import com.hytalerlbridge.combat.CombatTelemetry;
import com.hytalerlbridge.combat.HytaleCombatAssets;
import com.hytalerlbridge.combat.MeleeAttackProfile;
import com.hytalerlbridge.combat.knockback.DirectionalKnockbackModel;
import com.hytalerlbridge.combat.selector.HorizontalSelectorGeometry;
import com.hytalerlbridge.environment.EnvironmentOptions;
import com.hytalerlbridge.observation.Observation;
import com.hytalerlbridge.observation.ObservationEncoding;
import com.hytalerlbridge.task.RLTask;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import com.hytalerlbridge.combat.ruleset.AgentAttack;
import com.hytalerlbridge.combat.ruleset.Chase;
import com.hytalerlbridge.combat.ruleset.MaintainDistance;

/**
 * High-throughput combat approximation calibrated against the native 0.5.7
 * Kweebec_Razorleaf versus Trork_Brawler oracle scenario.
 *
 * <p>The model preserves the policy-facing invariants that matter for transfer:
 * Hytale axes, 30 Hz timing, role health/speed/acceleration, edge-triggered
 * attacks, authored attack sequence, wind-up, exact flat-fixture head aim and
 * selector projection, cooldown, damage, exact flat-fixture directional
 * knockback, retaliation, measured motion-delta variation, and reward deltas.
 * Native Hytale remains the final authority for arbitrary-terrain
 * LOS/navigation, collision, animation, and interaction-chain side effects.</p>
 */
public class ActiveMeleeSimulator implements RLTask {

    private static final CombatRuleset RULES = HytaleCombatAssets.RULESET;
    private static final double NOMINAL_DT =
        RULES.engine().nominalDeltaSeconds();
    private static final double LOADED_DT =
        RULES.engine().loadedDeltaSeconds();
    private static final int MOTION_TIMING_PROFILE_COUNT =
        RULES.engine().motionTimingProfileCount();

    private static final double AGENT_MAX_HEALTH = RULES.agent().maxHealth();
    private static final double AGENT_MAX_SPEED = RULES.agent().maxSpeed();
    private static final double AGENT_ACCELERATION = RULES.agent().acceleration();
    private static final double AGENT_JUMP_VELOCITY = Math.sqrt(
        2.0
            * RULES.agent().jumpVelocityGravityFloor()
            * RULES.agent().jumpHeightParameter()
    );
    private static final double WORLD_GRAVITY = RULES.engine().gravity();
    private static final double FLOOR_Y = RULES.fixture().floorY();
    private static final double LANDING_VELOCITY_SCALE =
        RULES.agent().landingVelocityScale();
    private static final double AGENT_TURN_SPEED_DEGREES =
        RULES.agent().turnDegreesPerSecond();
    private static final double AGENT_DAMAGE = RULES.agent().damage();
    private static final double AGENT_ATTACK_PAUSE_MIN_SECONDS =
        RULES.agent().attackPauseMinSeconds();
    private static final double AGENT_ATTACK_PAUSE_MAX_SECONDS =
        RULES.agent().attackPauseMaxSeconds();

    private static final double TRORK_MAX_HEALTH = RULES.target().maxHealth();
    private static final double TRORK_MAX_SPEED =
        RULES.target().assetMaxWalkSpeed();
    private static final double TRORK_CHASE_SPEED = RULES.target().chaseSpeed();
    private static final double TRORK_ACCELERATION = RULES.target().acceleration();
    private static final double TRORK_STEERING_SLOWDOWN_FALLOFF =
        RULES.engine().steeringSlowdownFalloff();
    private static final Chase TRORK_CHASE =
        RULES.target().chase();
    private static final MaintainDistance TRORK_MAINTAIN =
        RULES.target().maintainDistance();
    private static final int TRORK_CHASE_REACTION_TICKS =
        RULES.target().chaseReactionTicks();
    private static final double TRORK_TURN_SPEED_DEGREES =
        HytaleCombatAssets.TRORK_BRAWLER_TURN_DEGREES_PER_SECOND;
    // Native role initialization is asynchronous relative to the isolated
    // world's first requested tick. Fresh and warmed 0.5.7 worlds produced a
    // 36-44 tick activation window; varying it by episode trains policies not
    // to depend on process warm-up timing.
    private static final int TARGET_AI_ACTIVATION_MIN_TICK =
        RULES.target().activationMinTick();
    private static final int TARGET_AI_ACTIVATION_MAX_TICK =
        RULES.target().activationMaxTick();
    private static final int TARGET_DECISION_DELAY_TICKS =
        RULES.target().decisionDelayTicks();
    private static final double SENSOR_RANGE = RULES.target().sensorRange();

    private static final double TARGET_DAMAGE_REWARD_SCALE =
        RULES.reward().targetDamageScale();
    private static final double AGENT_DAMAGE_REWARD_SCALE =
        RULES.reward().agentDamageScale();
    private static final double COMPLETION_REWARD = RULES.reward().completion();
    private static final double DEATH_REWARD = RULES.reward().death();

    // 0.5.7 Kweebec Razorleaf role order: left swing, right swing, stab.
    // Live oracle hit ticks at the default 2.5-block target distance are
    // approximately 15, 11, and 20 ticks from the accepted request.
    private static final int[] AGENT_HIT_DELAYS = RULES.agent().attacks().stream()
        .mapToInt(AgentAttack::hitDelayTicks)
        .toArray();
    private static final double[] AGENT_ATTACK_RANGES =
        RULES.agent().attacks().stream()
            .mapToDouble(AgentAttack::range)
            .toArray();
    private static final double[] AGENT_HALF_ANGLES =
        RULES.agent().attacks().stream()
            .mapToDouble(AgentAttack::halfAngleDegrees)
            .toArray();

    private double x;
    private double y;
    private double z;
    private double vx;
    private double vy;
    private double vz;
    private double yaw;
    private double desiredYaw;
    private double pitch;
    private double desiredPitch;
    private double health;

    private double trorkX;
    private double trorkZ;
    private double trorkVx;
    private double trorkVz;
    private double trorkYaw;
    private double trorkHeadYaw;
    private double trorkHeadPitch;
    private double trorkHealth;

    private double desiredVx;
    private double desiredVz;
    private double agentMoveSpeed;
    private int tickCount;
    private int motionTimingProfile;
    private double lastMotionDeltaSeconds;
    private int attackSequenceIndex;
    private double agentAttackCooldownSeconds;
    private int agentHitDelay;
    private int pendingAgentAttackIndex;
    private int targetAttackSequenceIndex;
    private int targetAiActivationTick;
    private int targetAttackIndex;
    private int lastTargetAttackIndex;
    private int targetAttackElapsedTicks;
    private double targetAttackCooldownSeconds;
    private int targetNextAttackQueueTick;
    private boolean targetAttackQueued;
    private boolean targetAttackHitApplied;
    private boolean targetDamagePending;
    private double pendingKnockbackVx;
    private double pendingKnockbackVy;
    private double pendingKnockbackVz;
    private boolean stagedKnockbackPending;
    private double stagedKnockbackVx;
    private double stagedKnockbackVy;
    private double stagedKnockbackVz;
    private boolean targetDamageAppliedThisTick;
    private int targetOutOfRangeTicks;
    private boolean targetMaintainApproaching;
    private boolean targetMaintainMovingAway;
    private double targetStrafeDelaySeconds;
    private boolean targetStrafePaused;
    private int targetStrafeDirection;
    private int ticksSinceAgentDamage;
    private double pendingReward;
    private boolean completionAwarded;
    private boolean deathPenaltyAwarded;
    private boolean attackRequestedThisStep;
    private boolean attackAcceptedThisStep;
    private boolean verticalImpulseApplied;
    private double agentAppliedVerticalVelocity;
    private boolean groundedWithResidualVelocity;
    private boolean knockbackControlLock;
    private boolean targetActive = true;
    private Random rng;
    private Random targetMotionRng;

    @Override
    public String description() {
        return "Defeat a native-calibrated "
            + RULES.matchup().targetRole()
            + " as "
            + RULES.matchup().agentRole()
            + ".";
    }

    @Override
    public void configure(EnvironmentOptions options) {
        targetActive = options == null || options.combatTargetActive();
    }

    @Override
    public void reset(long seed) {
        rng = new Random(seed);
        // Navigation timing has a dedicated deterministic stream so enabling
        // authored strafing does not perturb attack-pause draws.
        targetMotionRng = new Random(~seed);
        double[] spawn = RULES.fixture().agentSpawn();
        double[] targetOffset = RULES.fixture().targetOffset();
        x = spawn[0];
        y = spawn[1];
        z = spawn[2];
        vx = 0.0;
        vy = 0.0;
        vz = 0.0;
        yaw = 0.0;
        desiredYaw = 0.0;
        pitch = 0.0;
        desiredPitch = 0.0;
        health = AGENT_MAX_HEALTH;

        trorkX = x + targetOffset[0];
        trorkZ = z + targetOffset[2];
        trorkVx = 0.0;
        trorkVz = 0.0;
        trorkYaw = 0.0;
        trorkHeadYaw = 0.0;
        trorkHeadPitch = 0.0;
        trorkHealth = TRORK_MAX_HEALTH;

        desiredVx = 0.0;
        desiredVz = 0.0;
        agentMoveSpeed = 0.0;
        tickCount = 0;
        motionTimingProfile = (int) Math.floorMod(seed, MOTION_TIMING_PROFILE_COUNT);
        lastMotionDeltaSeconds = NOMINAL_DT;
        attackSequenceIndex = 0;
        agentAttackCooldownSeconds = 0.0;
        agentHitDelay = 0;
        pendingAgentAttackIndex = -1;
        targetAiActivationTick = TARGET_AI_ACTIVATION_MIN_TICK
            + (int) Math.floorMod(
                seed + 2L,
                TARGET_AI_ACTIVATION_MAX_TICK
                    - TARGET_AI_ACTIVATION_MIN_TICK + 1L
            );
        targetAttackSequenceIndex = 0;
        targetAttackIndex = -1;
        lastTargetAttackIndex = -1;
        targetAttackElapsedTicks = 0;
        targetAttackCooldownSeconds = 0.0;
        targetNextAttackQueueTick = targetAiActivationTick + 1;
        targetAttackQueued = false;
        targetAttackHitApplied = false;
        targetDamagePending = false;
        pendingKnockbackVx = 0.0;
        pendingKnockbackVy = 0.0;
        pendingKnockbackVz = 0.0;
        stagedKnockbackPending = false;
        stagedKnockbackVx = 0.0;
        stagedKnockbackVy = 0.0;
        stagedKnockbackVz = 0.0;
        targetDamageAppliedThisTick = false;
        targetOutOfRangeTicks = 0;
        targetMaintainApproaching = false;
        targetMaintainMovingAway = false;
        targetStrafeDelaySeconds = 0.0;
        targetStrafePaused = false;
        targetStrafeDirection = 1;
        ticksSinceAgentDamage = 0;
        pendingReward = 0.0;
        completionAwarded = false;
        deathPenaltyAwarded = false;
        attackRequestedThisStep = false;
        attackAcceptedThisStep = false;
        verticalImpulseApplied = false;
        agentAppliedVerticalVelocity = 0.0;
        groundedWithResidualVelocity = false;
        knockbackControlLock = false;
    }

    @Override
    public void beginStep(AgentAction action) {
        attackRequestedThisStep = action != null && action.attack();
        attackAcceptedThisStep = false;
    }

    @Override
    public void applyAction(AgentAction action) {
        desiredYaw = normalizeDegrees(desiredYaw + action.cameraDeltaYaw());
        desiredPitch = clamp(
            desiredPitch + action.cameraDeltaPitch(), -90.0, 90.0
        );

        double forward = (action.forward() ? 1.0 : 0.0)
            - (action.back() ? 1.0 : 0.0);
        double strafe = (action.right() ? 1.0 : 0.0)
            - (action.left() ? 1.0 : 0.0);
        double length = Math.hypot(forward, strafe);
        if (length > 1.0e-9) {
            forward /= length;
            strafe /= length;
            // The native controller steers translation against its requested
            // heading while the visible body rotation converges separately.
            double radians = Math.toRadians(desiredYaw);
            desiredVx = AGENT_MAX_SPEED
                * (forward * -Math.sin(radians) + strafe * Math.cos(radians));
            desiredVz = AGENT_MAX_SPEED
                * (forward * -Math.cos(radians) - strafe * Math.sin(radians));
        } else {
            desiredVx = 0.0;
            desiredVz = 0.0;
        }

        if (action.jump() && y <= FLOOR_Y + 1.0e-9) {
            vy = AGENT_JUMP_VELOCITY;
            agentAppliedVerticalVelocity = AGENT_JUMP_VELOCITY;
            verticalImpulseApplied = true;
            groundedWithResidualVelocity = false;
            knockbackControlLock = false;
        }

        if (action.attack()
            && trorkHealth > 0.0
            && agentAttackCooldownSeconds <= 0.0) {
            pendingAgentAttackIndex = attackSequenceIndex;
            agentHitDelay = AGENT_HIT_DELAYS[pendingAgentAttackIndex];
            attackSequenceIndex = (attackSequenceIndex + 1) % AGENT_HIT_DELAYS.length;
            agentAttackCooldownSeconds = uniformRange(
                AGENT_ATTACK_PAUSE_MIN_SECONDS,
                AGENT_ATTACK_PAUSE_MAX_SECONDS
            );
            attackAcceptedThisStep = true;
        }
    }

    @Override
    public void tick() {
        tickCount++;
        double motionDeltaSeconds = motionDeltaSeconds();
        lastMotionDeltaSeconds = motionDeltaSeconds;
        targetDamageAppliedThisTick = false;
        boolean knockbackAppliedThisTick = applyStagedTargetKnockback();

        yaw = approachAngle(
            yaw,
            desiredYaw,
            AGENT_TURN_SPEED_DEGREES * motionDeltaSeconds
        );
        pitch = approach(
            pitch,
            desiredPitch,
            AGENT_TURN_SPEED_DEGREES * motionDeltaSeconds
        );

        if (knockbackAppliedThisTick) {
            // The NPC forced-push branch already produced this tick's complete
            // horizontal velocity from external force plus reduced walk speed.
        } else if (knockbackControlLock) {
            vx = 0.0;
            vz = 0.0;
        } else if (Math.abs(desiredVx) <= 1.0e-9
            && Math.abs(desiredVz) <= 1.0e-9) {
            // Native Walk steering stops immediately when its translation is
            // cleared; acceleration applies while acquiring movement speed.
            agentMoveSpeed = 0.0;
            vx = 0.0;
            vz = 0.0;
        } else {
            double desiredSpeed = Math.hypot(desiredVx, desiredVz);
            agentMoveSpeed = Math.min(
                desiredSpeed,
                agentMoveSpeed + AGENT_ACCELERATION * motionDeltaSeconds
            );
            vx = desiredVx / desiredSpeed * agentMoveSpeed;
            vz = desiredVz / desiredSpeed * agentMoveSpeed;
        }
        x += vx * motionDeltaSeconds;
        z += vz * motionDeltaSeconds;
        tickAgentVerticalMotion(motionDeltaSeconds);

        // DamageEntityInteraction writes KnockbackComponent through a command
        // buffer after this tick's motion consumer. Health changes now, while
        // the separate ApplyKnockback system consumes the staged force on the
        // following tick.
        applyPendingTargetDamage();
        tickAgentAttack();
        if (targetActive) {
            double targetAttackQueueDistance = distanceToTrork();
            TargetSelectorReference selectorReference =
                new TargetSelectorReference(
                    trorkX,
                    trorkZ,
                    trorkHeadYaw,
                    trorkHeadPitch
                );
            tickTrorkMovement(motionDeltaSeconds);
            tickTrorkAttack(targetAttackQueueDistance, selectorReference);
        }

        agentAttackCooldownSeconds = Math.max(
            0.0,
            agentAttackCooldownSeconds - motionDeltaSeconds
        );
        tickAgentHealthRegeneration();

        if (trorkHealth <= 0.0 && !completionAwarded) {
            completionAwarded = true;
            pendingReward += COMPLETION_REWARD;
        }
        if (health <= 0.0 && !deathPenaltyAwarded) {
            deathPenaltyAwarded = true;
            pendingReward += DEATH_REWARD;
        }
    }

    private void tickAgentAttack() {
        if (agentHitDelay <= 0) return;
        agentHitDelay--;
        if (agentHitDelay > 0 || pendingAgentAttackIndex < 0 || trorkHealth <= 0.0) return;

        int attackIndex = pendingAgentAttackIndex;
        pendingAgentAttackIndex = -1;
        double distance = distanceToTrork();
        double targetBearing = Math.toDegrees(Math.atan2(-(trorkX - x), -(trorkZ - z)));
        double facingError = Math.abs(normalizeDegrees(targetBearing - yaw));
        if (distance <= AGENT_ATTACK_RANGES[attackIndex]
            && facingError <= AGENT_HALF_ANGLES[attackIndex]) {
            double before = trorkHealth;
            trorkHealth = Math.max(0.0, trorkHealth - AGENT_DAMAGE);
            pendingReward +=
                (before - trorkHealth) * TARGET_DAMAGE_REWARD_SCALE;
        }
    }

    private void tickAgentVerticalMotion(double motionDeltaSeconds) {
        if (verticalImpulseApplied) {
            y += vy * motionDeltaSeconds;
            verticalImpulseApplied = false;
            return;
        }
        if (groundedWithResidualVelocity) {
            vy = 0.0;
            agentAppliedVerticalVelocity = 0.0;
            groundedWithResidualVelocity = false;
            knockbackControlLock = false;
            return;
        }
        if (y <= FLOOR_Y && vy <= 0.0) {
            y = FLOOR_Y;
            vy = 0.0;
            agentAppliedVerticalVelocity = 0.0;
            return;
        }
        vy -= WORLD_GRAVITY * motionDeltaSeconds;
        y += vy * motionDeltaSeconds;
        if (y <= FLOOR_Y) {
            y = FLOOR_Y;
            vy *= LANDING_VELOCITY_SCALE;
            agentAppliedVerticalVelocity = 0.0;
            groundedWithResidualVelocity = true;
        }
    }

    private void tickTrorkMovement(double motionDeltaSeconds) {
        if (trorkHealth <= 0.0 || health <= 0.0) {
            trorkVx = 0.0;
            trorkVz = 0.0;
            return;
        }
        if (tickCount < targetAiActivationTick) {
            trorkVx = 0.0;
            trorkVz = 0.0;
            return;
        }
        double dx = x - trorkX;
        double dz = z - trorkZ;
        double distance = Math.hypot(dx, dz);
        double agentBearing = distance > 1.0e-9
            ? canonicalBearing(dx, dz)
            : trorkYaw;
        boolean maintainContext =
            distance <= TRORK_MAINTAIN.activationRange();
        boolean maintain = maintainContext;

        double desiredMin = TRORK_MAINTAIN.desiredDistanceMin();
        double desiredMax = TRORK_MAINTAIN.desiredDistanceMax();
        double lowerThreshold = Math.max(
            0.0,
            desiredMin - TRORK_MAINTAIN.moveThreshold()
        );
        double upperThreshold =
            desiredMax + TRORK_MAINTAIN.moveThreshold();
        double approachTarget = lerp(
            desiredMin,
            desiredMax,
            1.0 - TRORK_MAINTAIN.targetDistanceFactor()
        );
        double awayTarget = lerp(
            desiredMin,
            desiredMax,
            TRORK_MAINTAIN.targetDistanceFactor()
        );

        if (maintain) {
            boolean approaching =
                distance > upperThreshold
                    || (
                        targetMaintainApproaching
                            && distance > approachTarget
                    );
            boolean movingAway = !approaching
                && (
                    distance < lowerThreshold
                        || (
                            targetMaintainMovingAway
                                && distance < awayTarget
                        )
                );
            targetMaintainApproaching = approaching;
            targetMaintainMovingAway = movingAway;
            tickTargetStrafing(motionDeltaSeconds);
        }

        boolean strafing = maintain && !targetStrafePaused;
        double requestedRelativeSpeed;
        double translationYaw;
        double bodyYaw;
        if (maintainContext) {
            if (targetMaintainApproaching) {
                requestedRelativeSpeed =
                    TRORK_MAINTAIN.relativeForwardSpeed()
                        * pursueSpeedScale(
                            distance,
                            approachTarget,
                            approachTarget
                                + TRORK_MAINTAIN
                                    .moveTowardsSlowdownDistance()
                        );
                translationYaw = agentBearing;
            } else if (targetMaintainMovingAway) {
                requestedRelativeSpeed =
                    TRORK_MAINTAIN.relativeBackwardSpeed();
                translationYaw = normalizeDegrees(agentBearing + 180.0);
            } else if (strafing) {
                requestedRelativeSpeed =
                    TRORK_MAINTAIN.relativeForwardSpeed();
                translationYaw = normalizeDegrees(
                    agentBearing
                        + targetStrafeDirection
                            * TRORK_MAINTAIN
                                .strafingTranslationOffsetDegrees()
                );
            } else {
                requestedRelativeSpeed = 0.0;
                translationYaw = agentBearing;
            }

            if (strafing
                && (
                    targetMaintainApproaching
                        || targetMaintainMovingAway
                )) {
                double direction = targetMaintainMovingAway ? -1.0 : 1.0;
                translationYaw = normalizeDegrees(
                    translationYaw
                        + targetStrafeDirection
                            * direction
                            * TRORK_MAINTAIN.strafingYawOffsetDegrees()
                );
            }
            bodyYaw = normalizeDegrees(
                agentBearing
                    + (
                        strafing
                            ? targetStrafeDirection
                                * TRORK_MAINTAIN
                                    .strafingYawOffsetDegrees()
                            : 0.0
                    )
            );
        } else {
            requestedRelativeSpeed =
                (TRORK_CHASE_SPEED / TRORK_MAX_SPEED)
                    * pursueSpeedScale(
                        distance,
                        TRORK_CHASE.stopDistance(),
                        TRORK_CHASE.slowdownDistance()
                    );
            bodyYaw = agentBearing;
            // Seek without an explicit body-yaw instruction moves along the
            // walk controller's converging heading.
            translationYaw = trorkYaw;
        }

        if (distance > 1.0e-9) {
            trorkYaw = approachTargetAngle(
                trorkYaw,
                bodyYaw,
                TRORK_TURN_SPEED_DEGREES * motionDeltaSeconds
            );
        }

        double[] agentBounds = RULES.agent().boundingBox();
        double targetEyeY =
            FLOOR_Y + RULES.target().effectiveEyeHeight();
        double aimY = clamp(
            targetEyeY,
            y + agentBounds[1],
            y + agentBounds[4]
        );
        double desiredHeadPitch = Math.toDegrees(Math.atan2(
            aimY - targetEyeY,
            Math.max(distance, 1.0e-9)
        ));
        boolean headAiming = distance <= maximumTargetAttackRange();
        double requestedHeadYaw = headAiming ? agentBearing : trorkYaw;
        double requestedHeadPitch = headAiming ? desiredHeadPitch : 0.0;
        double relativeHeadSpeed = headAiming
            ? RULES.target().headAimRelativeTurnSpeed()
            : RULES.target().headDefaultRelativeTurnSpeed();
        double headTurnStep =
            RULES.target().maxHeadRotationDegreesPerSecond()
                * relativeHeadSpeed
                * motionDeltaSeconds;
        double turnedHeadYaw = approachTargetAngle(
            trorkHeadYaw,
            requestedHeadYaw,
            headTurnStep
        );
        double turnedHeadPitch = approach(
            trorkHeadPitch,
            requestedHeadPitch,
            headTurnStep
        );
        double relativeHeadYaw = clamp(
            normalizeDegrees(turnedHeadYaw - trorkYaw),
            RULES.target().headYawMinDegrees(),
            RULES.target().headYawMaxDegrees()
        );
        trorkHeadYaw = normalizeDegrees(trorkYaw + relativeHeadYaw);
        trorkHeadPitch = clamp(
            turnedHeadPitch,
            RULES.target().headPitchMinDegrees(),
            RULES.target().headPitchMaxDegrees()
        );

        if (!maintain) translationYaw = trorkYaw;

        boolean motionRequested = requestedRelativeSpeed > 0.0;
        if (motionRequested) {
            targetOutOfRangeTicks++;
        } else {
            targetOutOfRangeTicks = 0;
        }
        double speed = Math.hypot(trorkVx, trorkVz);
        double desiredSpeed = TRORK_MAX_SPEED * requestedRelativeSpeed;
        speed = motionRequested
                && targetOutOfRangeTicks >= TRORK_CHASE_REACTION_TICKS
            ? Math.min(
                desiredSpeed,
                speed + TRORK_ACCELERATION * motionDeltaSeconds
            )
            : 0.0;
        double yawRadians = Math.toRadians(translationYaw);
        trorkVx = -Math.sin(yawRadians) * speed;
        trorkVz = -Math.cos(yawRadians) * speed;
        trorkX += trorkVx * motionDeltaSeconds;
        trorkZ += trorkVz * motionDeltaSeconds;
    }

    private void tickTargetStrafing(double motionDeltaSeconds) {
        if (targetStrafeDelaySeconds > 0.0) {
            targetStrafeDelaySeconds -= motionDeltaSeconds;
            return;
        }
        if (targetStrafePaused) {
            targetStrafeDelaySeconds = uniformRange(
                targetMotionRng,
                TRORK_MAINTAIN.strafingDurationMinSeconds(),
                TRORK_MAINTAIN.strafingDurationMaxSeconds()
            );
            targetStrafeDirection = targetMotionRng.nextBoolean() ? 1 : -1;
            targetStrafePaused = false;
        } else {
            targetStrafeDelaySeconds = uniformRange(
                targetMotionRng,
                TRORK_MAINTAIN.strafingFrequencyMinSeconds(),
                TRORK_MAINTAIN.strafingFrequencyMaxSeconds()
            );
            targetStrafePaused = true;
        }
    }

    private static double pursueSpeedScale(
        double distance,
        double stopDistance,
        double slowdownDistance
    ) {
        if (distance <= stopDistance) return 0.0;
        if (distance >= slowdownDistance) return 1.0;
        double ratio = (distance - stopDistance)
            / (slowdownDistance - stopDistance);
        return Math.pow(
            ratio,
            1.0 / TRORK_STEERING_SLOWDOWN_FALLOFF
        );
    }

    private static double lerp(double minimum, double maximum, double amount) {
        return minimum + (maximum - minimum) * amount;
    }

    private void tickTrorkAttack(
        double queueDistance,
        TargetSelectorReference selectorReference
    ) {
        if (trorkHealth <= 0.0
            || (health <= 0.0 && !targetDamageAppliedThisTick)) {
            targetAttackIndex = -1;
            targetAttackElapsedTicks = 0;
            targetAttackQueued = false;
            targetDamagePending = false;
            pendingKnockbackVx = 0.0;
            pendingKnockbackVy = 0.0;
            pendingKnockbackVz = 0.0;
            return;
        }

        targetAttackCooldownSeconds = Math.max(
            0.0,
            targetAttackCooldownSeconds - lastMotionDeltaSeconds
        );
        if (targetAttackIndex >= 0) {
            MeleeAttackProfile completedProfile =
                HytaleCombatAssets.TRORK_BRAWLER_ATTACKS.get(targetAttackIndex);
            if (targetAttackElapsedTicks >= completedProfile.chainTicks()) {
                targetAttackIndex = -1;
                targetAttackElapsedTicks = 0;
            }
        }

        if (targetAttackQueued) {
            targetAttackQueued = false;
            targetAttackIndex = targetAttackSequenceIndex;
            lastTargetAttackIndex = targetAttackIndex;
            targetAttackSequenceIndex = (
                targetAttackSequenceIndex + 1
            ) % HytaleCombatAssets.TRORK_BRAWLER_ATTACKS.size();
            targetAttackElapsedTicks = 0;
            targetAttackHitApplied = false;
        } else if (targetAttackIndex < 0
            && targetAttackCooldownSeconds <= 0.0
            && tickCount >= targetAiActivationTick) {
            if (queueDistance > maximumTargetAttackRange()) {
                targetNextAttackQueueTick =
                    tickCount + TARGET_DECISION_DELAY_TICKS;
                return;
            }
            if (targetNextAttackQueueTick == Integer.MAX_VALUE) {
                targetNextAttackQueueTick =
                    tickCount + TARGET_DECISION_DELAY_TICKS;
            } else if (tickCount >= targetNextAttackQueueTick
                && queueDistance <= maximumTargetAttackRange()) {
                targetAttackQueued = true;
                targetAttackCooldownSeconds = uniformRange(
                    HytaleCombatAssets.TARGET_ATTACK_PAUSE_MIN_SECONDS,
                    HytaleCombatAssets.TARGET_ATTACK_PAUSE_MAX_SECONDS
                );
                targetNextAttackQueueTick = Integer.MAX_VALUE;
                return;
            }
        }

        if (targetAttackIndex < 0) return;
        MeleeAttackProfile profile =
            HytaleCombatAssets.TRORK_BRAWLER_ATTACKS.get(targetAttackIndex);
        targetAttackElapsedTicks++;
        if (profile.phaseAt(targetAttackElapsedTicks) == CombatPhase.SWEEP
            && !targetAttackHitApplied
            && targetSelectorIntersectsAgent(profile, selectorReference)) {
            targetAttackHitApplied = true;
            targetDamagePending = true;
            DirectionalKnockbackModel.Velocity external =
                DirectionalKnockbackModel.pendingExternalVelocity(
                    selectorReference.x(),
                    FLOOR_Y,
                    selectorReference.z(),
                    x,
                    y,
                    z,
                    selectorReference.headYaw(),
                    RULES
                );
            pendingKnockbackVx = external.x();
            pendingKnockbackVy = external.y();
            pendingKnockbackVz = external.z();
        }
    }

    private void applyPendingTargetDamage() {
        if (!targetDamagePending) return;
        targetDamagePending = false;
        targetDamageAppliedThisTick = true;
        double before = health;
        MeleeAttackProfile lastProfile =
            HytaleCombatAssets.TRORK_BRAWLER_ATTACKS.get(lastTargetAttackIndex);
        health = Math.max(0.0, health - lastProfile.damage());
        ticksSinceAgentDamage = 0;
        pendingReward += (before - health) * AGENT_DAMAGE_REWARD_SCALE;

        stagedKnockbackPending = true;
        stagedKnockbackVx = pendingKnockbackVx;
        stagedKnockbackVy = pendingKnockbackVy;
        stagedKnockbackVz = pendingKnockbackVz;
        pendingKnockbackVx = 0.0;
        pendingKnockbackVy = 0.0;
        pendingKnockbackVz = 0.0;
    }

    private boolean applyStagedTargetKnockback() {
        if (!stagedKnockbackPending) return false;
        stagedKnockbackPending = false;
        DirectionalKnockbackModel.ForcedPush forced =
            DirectionalKnockbackModel.forcePushedMotion(
                new DirectionalKnockbackModel.Velocity(
                    stagedKnockbackVx,
                    stagedKnockbackVy,
                    stagedKnockbackVz
                ),
                agentMoveSpeed,
                yaw,
                RULES
            );
        agentMoveSpeed = forced.moveSpeed();
        vx = forced.velocity().x();
        vy = forced.velocity().y() + agentAppliedVerticalVelocity;
        vz = forced.velocity().z();
        stagedKnockbackVx = 0.0;
        stagedKnockbackVy = 0.0;
        stagedKnockbackVz = 0.0;
        verticalImpulseApplied = true;
        groundedWithResidualVelocity = false;
        knockbackControlLock = true;
        return true;
    }

    private void tickAgentHealthRegeneration() {
        if (health <= 0.0 || health >= AGENT_MAX_HEALTH) return;
        if (targetDamageAppliedThisTick) return;
        ticksSinceAgentDamage++;
        int delay = HytaleCombatAssets.NPC_HEALTH_REGEN_DELAY_TICKS;
        int interval = HytaleCombatAssets.NPC_HEALTH_REGEN_INTERVAL_TICKS;
        if (ticksSinceAgentDamage >= delay
            && (ticksSinceAgentDamage - delay) % interval == 0) {
            health = Math.min(
                AGENT_MAX_HEALTH,
                health + AGENT_MAX_HEALTH
                    * HytaleCombatAssets.NPC_HEALTH_REGEN_FRACTION
            );
        }
    }

    private boolean targetSelectorIntersectsAgent(
        MeleeAttackProfile profile,
        TargetSelectorReference reference
    ) {
        double[] relativeBounds = RULES.agent().boundingBox();
        double[] worldBounds = new double[] {
            x + relativeBounds[0],
            y + relativeBounds[1],
            z + relativeBounds[2],
            x + relativeBounds[3],
            y + relativeBounds[4],
            z + relativeBounds[5]
        };
        double[] selectorOrigin = new double[] {
            reference.x(),
            FLOOR_Y + RULES.target().effectiveEyeHeight(),
            reference.z()
        };
        return HorizontalSelectorGeometry.intersectsAabb(
            profile,
            targetAttackElapsedTicks,
            NOMINAL_DT,
            RULES.engine().horizontalSelectorPi(),
            selectorOrigin,
            reference.headYaw(),
            reference.headPitch(),
            worldBounds
        );
    }

    private static double maximumTargetAttackRange() {
        double maximum = 0.0;
        for (MeleeAttackProfile profile : HytaleCombatAssets.TRORK_BRAWLER_ATTACKS) {
            maximum = Math.max(maximum, profile.endDistance());
        }
        return maximum;
    }

    @Override
    public Observation observe() {
        // Match the bridge's native nearby-NPC sensor radius. The task info
        // can retain privileged diagnostics, but the policy observation may
        // not see a target that native Hytale would omit.
        List<int[]> nearbyEntities =
            trorkHealth > 0.0 && distanceToTrork() <= SENSOR_RANGE
            ? List.of(new int[] {
                0,
                ObservationEncoding.encodeNearbyEntityScalar(trorkX - x),
                ObservationEncoding.encodeNearbyEntityScalar(trorkZ - z),
                ObservationEncoding.encodeNearbyEntityScalar(trorkHealth)
            })
            : List.of();
        return new Observation(
            x, y, z,
            vx, vy, vz,
            yaw, pitch,
            health, AGENT_MAX_HEALTH,
            0, 0.0, 0.0,
            new int[36],
            List.of(),
            nearbyEntities,
            0,
            0
        );
    }

    @Override
    public double computeReward() {
        double reward = pendingReward;
        pendingReward = 0.0;
        return reward;
    }

    @Override
    public boolean isTerminated() {
        return trorkHealth <= 0.0 || health <= 0.0;
    }

    @Override
    public Map<String, Object> info() {
        LinkedHashMap<String, Object> info = new LinkedHashMap<>();
        CombatTelemetry telemetry = combatTelemetry();
        info.put("backend", "simulator");
        info.put(
            "fidelity",
            "native_"
                + RULES.hytaleServerVersion()
                + "_authored_active_melee_model"
        );
        info.put("combat_model_version", RULES.version());
        info.put("agent_profile", RULES.matchup().agentRole());
        info.put("target_role", RULES.matchup().targetRole());
        info.put("target_present", trorkHealth > 0.0);
        info.put("target_is_native_entity", false);
        info.put("target_health", trorkHealth);
        info.put("target_max_health", TRORK_MAX_HEALTH);
        info.put("target_x", trorkX);
        info.put("target_y", FLOOR_Y);
        info.put("target_z", trorkZ);
        info.put("target_distance", distanceToTrork());
        info.put("native_attack_requested", attackRequestedThisStep);
        info.put("native_attack_accepted", attackAcceptedThisStep);
        info.put("native_attack_executing", telemetry.agentAttackExecuting());
        info.put("attack_sequence_index", attackSequenceIndex);
        info.put("attack_cooldown_seconds", agentAttackCooldownSeconds);
        info.put(
            "target_attack_cooldown_seconds",
            targetAttackCooldownSeconds
        );
        info.put("target_attack_sequence_index", targetAttackSequenceIndex);
        info.put("target_ai_activation_tick", targetAiActivationTick);
        info.put("simulation_motion_timing_profile", motionTimingProfile);
        info.put("simulation_motion_delta_seconds", lastMotionDeltaSeconds);
        info.put("combat_tick_hz", RULES.engine().ticksPerSecond());
        info.put("target_forced_combat_state", targetActive ? "Chase.Attack" : "disabled");
        telemetry.putInto(info);
        info.put(
            "fidelity_limit",
            "arbitrary_terrain_selector_los,knockback_collision,navigation,"
                + "and_non_brawler_role_ai_are_not_authoritative;"
                + "use_native_backend_for_final_validation"
        );
        return Map.copyOf(info);
    }

    private CombatTelemetry combatTelemetry() {
        boolean targetVisible = trorkHealth > 0.0;
        CombatPhase phase = CombatPhase.IDLE;
        double progress = 0.0;
        int reportedIndex = -1;
        int elapsedTicks = 0;
        String attackId = "";
        if (targetActive && targetVisible && targetAttackIndex >= 0) {
            MeleeAttackProfile profile =
                HytaleCombatAssets.TRORK_BRAWLER_ATTACKS.get(targetAttackIndex);
            phase = profile.phaseAt(targetAttackElapsedTicks);
            progress = profile.phaseProgressAt(targetAttackElapsedTicks);
            reportedIndex = targetAttackIndex;
            elapsedTicks = targetAttackElapsedTicks;
            attackId = profile.interactionId();
        } else if (targetActive && targetVisible && targetAttackQueued) {
            phase = CombatPhase.COOLDOWN;
            progress = 1.0;
        } else if (targetActive
            && targetVisible
            && targetAttackCooldownSeconds > 0.0
            && lastTargetAttackIndex >= 0) {
            phase = CombatPhase.COOLDOWN;
            progress = 1.0;
            reportedIndex = lastTargetAttackIndex;
            attackId = HytaleCombatAssets.TRORK_BRAWLER_ATTACKS
                .get(lastTargetAttackIndex)
                .interactionId();
        }
        return new CombatTelemetry(
            agentHitDelay > 0 || agentAttackCooldownSeconds > 0.0,
            targetVisible,
            phase,
            progress,
            reportedIndex,
            elapsedTicks,
            attackId,
            targetFacingErrorDegrees(),
            trorkYaw,
            trorkHeadYaw,
            trorkHeadPitch,
            trorkVx,
            trorkVz
        );
    }

    private double targetFacingErrorDegrees() {
        double dx = x - trorkX;
        double dz = z - trorkZ;
        if (Math.hypot(dx, dz) <= 1.0e-9) return 0.0;
        double agentBearing = canonicalBearing(dx, dz);
        return normalizeDegrees(agentBearing - trorkYaw);
    }

    private double distanceToTrork() {
        return Math.hypot(x - trorkX, z - trorkZ);
    }

    private double motionDeltaSeconds() {
        if (motionTimingProfile == 0) return NOMINAL_DT;
        boolean loadedTick = motionTimingProfile == 1
            ? tickCount % 2 == 1
            : tickCount % 2 == 0;
        return loadedTick ? LOADED_DT : NOMINAL_DT;
    }

    private double uniformRange(double minimum, double maximum) {
        return uniformRange(rng, minimum, maximum);
    }

    private static double uniformRange(
        Random source,
        double minimum,
        double maximum
    ) {
        if (maximum <= minimum) return minimum;
        return minimum + source.nextDouble() * (maximum - minimum);
    }

    private static double approach(double value, double target, double amount) {
        if (value < target) return Math.min(value + amount, target);
        return Math.max(value - amount, target);
    }

    private static double approachAngle(double value, double target, double amount) {
        double difference = normalizeDegrees(target - value);
        if (Math.abs(difference) <= amount) return target;
        return normalizeDegrees(value + Math.copySign(amount, difference));
    }

    private static double approachTargetAngle(
        double value,
        double target,
        double amount
    ) {
        double difference = normalizeDegrees(target - value);
        if (Math.abs(difference + 180.0) < 1.0e-9) difference = 180.0;
        if (Math.abs(difference) <= amount) return target;
        return value + Math.copySign(amount, difference);
    }

    private static double canonicalBearing(double dx, double dz) {
        double bearing = Math.toDegrees(Math.atan2(-dx, -dz));
        return Math.abs(Math.abs(bearing) - 180.0) < 1.0e-9
            ? 180.0
            : bearing;
    }

    private static double normalizeDegrees(double value) {
        double normalized = (value + 180.0) % 360.0;
        if (normalized < 0.0) normalized += 360.0;
        return normalized - 180.0;
    }

    private static double clamp(double value, double minimum, double maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private record TargetSelectorReference(
        double x,
        double z,
        double headYaw,
        double headPitch
    ) {}

}
