package com.hytalerlbridge.policy;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.modules.splitvelocity.VelocityConfig;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.movement.MotionKind;
import com.hypixel.hytale.server.npc.movement.NavState;
import com.hypixel.hytale.server.npc.movement.Steering;
import com.hypixel.hytale.server.npc.movement.controllers.MotionController;
import com.hypixel.hytale.server.npc.movement.controllers.MotionControllerBase;
import com.hypixel.hytale.server.npc.role.Role;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;
import org.joml.Vector3d;

/**
 * Applies the locomotion and look half of a decoded action.
 *
 * <p>This covers exactly the part of the bridge's {@code applyControl} that is
 * reachable with public engine APIs: body and head steering, motion kind, nav
 * state, and the jump impulse. It deliberately does <b>not</b> attempt attack,
 * ability, dodge, guard or world verbs -- those route through machinery that is
 * package-private to the bridge, and a second implementation of them would be a
 * second definition that could drift from the one the fidelity suite covers.
 *
 * <p>Standalone, an NPC driven by this sink is inert on every interaction
 * head, and {@link #skippedInteractions()} makes that limitation visible.
 * Production composes this leaf beneath bridge-backed World and combat sinks;
 * the five-argument constructor disables that standalone counter when another
 * sink owns those requests.
 *
 * <p>The body-relative direction projection and gait-to-relative-speed mapping
 * are exercised by the dependency-free {@link PlayerLocomotionProjection}
 * contract test. The surrounding NPC seam was executed against the isolated
 * Hytale Server 0.5.7 fixture in {@code adk/runs/live-port-v6}; that run
 * verified that this sink is live, but predates the direction/gait correction
 * and is not itself a transfer pass.
 *
 * <p>This is a player-policy-to-NPC public-API surrogate. Hytale's player gait
 * transition and stamina drain are client-side and are not exposed by the
 * public NPC steering API, so the sink preserves their authored directional
 * scale as NPC-relative steering magnitude while reporting those residual
 * differences in the port status documentation.
 */
public final class SteeringActionSink implements PolicyActionSink {

    private static final VelocityConfig VELOCITY_CONFIG = new VelocityConfig();

    private final double steeringRelativeTurnSpeed;
    private final double jumpVelocityGravityFloor;
    private final double jumpHeightParameter;
    private final boolean allowJump;
    private final boolean combatInteractionsSupported;

    private static final System.Logger LOGGER =
        System.getLogger(SteeringActionSink.class.getName());

    private long skippedInteractions;

    /**
     * One-shot description of the seams this sink writes through.
     *
     * <p>Every one of them is a silent skip on null: no body steering, no
     * motion controller, or a controller that is not a {@code
     * MotionControllerBase} all produce an NPC that is ticked, throws nothing,
     * and does not move. Logging the seam state once is the difference between
     * "steering is ignored" and knowing which seam was absent.
     */
    private volatile boolean described;

    /**
     * @param steeringRelativeTurnSpeed from {@code ruleset.agent().steeringRelativeTurnSpeed()}
     * @param jumpVelocityGravityFloor  from {@code ruleset.agent().jumpVelocityGravityFloor()}
     * @param jumpHeightParameter       from {@code ruleset.agent().jumpHeightParameter()}
     */
    public SteeringActionSink(
        double steeringRelativeTurnSpeed,
        double jumpVelocityGravityFloor,
        double jumpHeightParameter
    ) {
        this(steeringRelativeTurnSpeed, jumpVelocityGravityFloor,
            jumpHeightParameter, true, false);
    }

    /**
     * @param allowJump when false, the jump head is ignored. This exists
     *     because a policy that commands jump every tick keeps the NPC
     *     airborne, and {@code MotionControllerWalk} applies no horizontal
     *     translation off the ground -- so continuous jumping is
     *     indistinguishable from steering being broken. Suppressing jump
     *     separates the two.
     */
    public SteeringActionSink(
        double steeringRelativeTurnSpeed,
        double jumpVelocityGravityFloor,
        double jumpHeightParameter,
        boolean allowJump
    ) {
        this(
            steeringRelativeTurnSpeed,
            jumpVelocityGravityFloor,
            jumpHeightParameter,
            allowJump,
            false
        );
    }

    public SteeringActionSink(
        double steeringRelativeTurnSpeed,
        double jumpVelocityGravityFloor,
        double jumpHeightParameter,
        boolean allowJump,
        boolean combatInteractionsSupported
    ) {
        this.steeringRelativeTurnSpeed = steeringRelativeTurnSpeed;
        this.jumpVelocityGravityFloor = jumpVelocityGravityFloor;
        this.jumpHeightParameter = jumpHeightParameter;
        this.allowJump = allowJump;
        this.combatInteractionsSupported = combatInteractionsSupported;
    }

    /** How many ticks asked for an interaction this sink cannot apply. */
    public long skippedInteractions() {
        return skippedInteractions;
    }

    private void describeOnce(String detail) {
        if (described) {
            return;
        }
        described = true;
        LOGGER.log(System.Logger.Level.INFO,
            "SteeringActionSink seams: " + detail);
    }

    @Override
    public void apply(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        float deltaTime,
        PolicyAgentMarker marker,
        ActionDecoder.Decoded action,
        WorldActionEvidence evidence,
        boolean firstControlTick
    ) {
        Role role = npc.getRole();
        if (role == null) {
            describeOnce("role=null");
            return;
        }
        MotionController controller = role.getActiveMotionController();
        if (controller == null) {
            describeOnce("role=" + role.getRoleName() + " controller=null");
            return;
        }
        if (!described) {
            Steering probeBody = role.getBodySteering();
            describeOnce(String.format(
                "role=%s controller=%s base=%b bodySteering=%s headSteering=%s "
                    + "onGround=%b",
                role.getRoleName(),
                controller.getClass().getSimpleName(),
                controller instanceof MotionControllerBase,
                probeBody == null ? "null" : probeBody.getClass().getSimpleName(),
                role.getHeadSteering() == null ? "null" : "present",
                role.isOnGround()));
        }
        if (!combatInteractionsSupported && (action.attack()
            || action.guardHeld()
            || action.abilitySlot() >= 0
            || action.dodgeDirection() > 0
            || action.useRequested())) {
            skippedInteractions++;
        }

        float yaw = marker.desiredBodyYaw();
        PlayerLocomotionProjection.Vector movement =
            PlayerLocomotionProjection.project(action, yaw);
        boolean moving = movement.moving();

        Steering body = role.getBodySteering();
        if (body != null) {
            body.clear();
            if (moving) {
                // Steering.getTranslation().length() is the relative speed
                // consumed by MotionControllerWalk. Do not call
                // setTranslationRelativeSpeed(1.0) here: that API normalizes
                // the vector and would erase the gait projection.
                body.setTranslation(movement.x(), 0.0, movement.z());
            }
            body.setYaw(yaw);
            body.setRelativeTurnSpeed(steeringRelativeTurnSpeed);
        }

        Steering head = role.getHeadSteering();
        if (head != null) {
            head.clear();
            head.setYaw(marker.desiredYaw());
            head.setPitch(marker.desiredPitch());
            head.setRelativeTurnSpeed(steeringRelativeTurnSpeed);
        }

        if (controller instanceof MotionControllerBase base) {
            base.setMotionKind(moving ? MotionKind.MOVING : MotionKind.STANDING);
        }
        if (moving) {
            controller.setNavState(NavState.PROGRESSING, 0.0, 0.0);
        }

        if (allowJump && action.jumpHeld() && role.isOnGround()) {
            double gravity = controller instanceof MotionControllerBase base
                ? Math.abs(base.getGravity())
                : jumpVelocityGravityFloor;
            if (!Double.isFinite(gravity)) {
                gravity = jumpVelocityGravityFloor;
            }
            gravity = Math.max(jumpVelocityGravityFloor, gravity);
            double jumpVelocity = Math.sqrt(2.0 * gravity * jumpHeightParameter);
            // setVelocity, not addVelocity: addVelocity is the knockback path and
            // scales by the role's knockback multiplier, which in 0.5.7 turns a
            // jump into roughly 0.17 blocks for Trork_Unarmed.
            controller.setVelocity(new Vector3d(0.0, jumpVelocity, 0.0),
                VELOCITY_CONFIG, true);
        }
    }
}
