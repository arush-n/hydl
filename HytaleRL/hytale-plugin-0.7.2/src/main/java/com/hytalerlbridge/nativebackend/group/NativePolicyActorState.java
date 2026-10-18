package com.hytalerlbridge.nativebackend.group;

import com.hypixel.hytale.server.npc.corecomponents.combat.ActionAttack;
import java.util.List;

/**
 * Mutable control state owned by one reset-pinned policy actor.
 *
 * <p>Pose and role-attack cursors must not be shared between entities.  The
 * native session owns one instance per actor slot and clears every instance at
 * reset/close boundaries.</p>
 */
public final class NativePolicyActorState {

    private final int actorId;
    private float desiredYaw;
    /**
     * The commanded BODY heading, steered separately from the camera.
     *
     * <p>{@code MotionControllerBase.calculateYaw} takes a body steer and a head
     * steer and only bounds the head into the model's authored window around the
     * body afterwards.  This session used to hand {@code desiredYaw} to both
     * seams, so an actor could not walk one way and look another -- and a policy
     * trained against the split simulator would have had its body snapped round
     * to its aim on every tick.</p>
     */
    private float desiredBodyYaw;
    private float desiredPitch;
    private List<ActionAttack> attackActions = List.of();
    private int attackActionIndex;
    private boolean manualAttackWindow;

    public NativePolicyActorState(int actorId) {
        if (actorId < 0) {
            throw new IllegalArgumentException("actorId must be nonnegative");
        }
        this.actorId = actorId;
    }

    public int actorId() {
        return actorId;
    }

    public float desiredYaw() {
        return desiredYaw;
    }

    public float desiredBodyYaw() {
        return desiredBodyYaw;
    }

    public float desiredPitch() {
        return desiredPitch;
    }

    /** Steer the body on its own channel; zero means "no body steer". */
    public void applyBodyYawDelta(double yawDegrees) {
        desiredBodyYaw = normalizeRadians(
            desiredBodyYaw + (float) Math.toRadians(yawDegrees)
        );
    }

    public void applyLookDelta(double yawDegrees, double pitchDegrees) {
        desiredYaw = normalizeRadians(
            desiredYaw + (float) Math.toRadians(yawDegrees)
        );
        desiredPitch = clamp(
            desiredPitch + (float) Math.toRadians(pitchDegrees),
            (float) (-Math.PI / 2.0),
            (float) (Math.PI / 2.0)
        );
    }

    public void initialize(
        float yaw,
        float pitch,
        List<ActionAttack> discoveredAttackActions
    ) {
        setPose(yaw, pitch);
        attackActions = discoveredAttackActions == null
            ? List.of()
            : List.copyOf(discoveredAttackActions);
        attackActionIndex = 0;
        manualAttackWindow = false;
    }

    public void setPose(float yaw, float pitch) {
        desiredYaw = normalizeRadians(yaw);
        // The body starts on the same heading the camera does; calculateYaw
        // 609-611 does exactly this when no head steering has been supplied.
        desiredBodyYaw = normalizeRadians(yaw);
        desiredPitch = clamp(
            pitch,
            (float) (-Math.PI / 2.0),
            (float) (Math.PI / 2.0)
        );
    }

    public boolean hasRoleAttack() {
        return !attackActions.isEmpty();
    }

    public int roleAttackCount() {
        return attackActions.size();
    }

    public List<ActionAttack> roleAttacks() {
        return attackActions;
    }

    public ActionAttack nextRoleAttack() {
        if (attackActions.isEmpty()) return null;
        int index = Math.floorMod(attackActionIndex, attackActions.size());
        ActionAttack result = attackActions.get(index);
        attackActionIndex = (index + 1) % attackActions.size();
        return result;
    }

    public boolean manualAttackWindow() {
        return manualAttackWindow;
    }

    public void setManualAttackWindow(boolean active) {
        manualAttackWindow = active;
    }

    public void clear() {
        desiredYaw = 0.0f;
        desiredBodyYaw = 0.0f;
        desiredPitch = 0.0f;
        attackActions = List.of();
        attackActionIndex = 0;
        manualAttackWindow = false;
    }

    private static float clamp(float value, float low, float high) {
        return Math.max(low, Math.min(high, value));
    }

    private static float normalizeRadians(float angle) {
        float twoPi = (float) (Math.PI * 2.0);
        float normalized = angle % twoPi;
        if (normalized > Math.PI) normalized -= twoPi;
        if (normalized < -Math.PI) normalized += twoPi;
        return normalized;
    }
}
