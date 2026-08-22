package com.hytalerlbridge.policy;

import com.hypixel.hytale.component.Component;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.policy.combat.runtime.PolicyCombatState;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;
import com.hytalerlbridge.policy.world.runtime.PolicyWorldVerbState;

/**
 * Binds one NPC to a policy, and carries that NPC's private recurrent state.
 *
 * <p>Mirrors the bridge's {@code NativeAgentMarker}: a non-serialised marker
 * component whose presence is what {@link PolicyControlSystem} queries for. The
 * difference is that this one also owns state, because a recurrent policy is
 * not a pure function of the current observation -- the GRU carry is the NPC's
 * memory, and it must live on the entity rather than in the shared runtime.
 *
 * <p>The look pose is accumulated the same way, and for the same reason the
 * bridge does it in {@code NativePolicyActorState}: the policy emits a yaw/pitch
 * <i>delta</i> per tick, so somebody has to hold the absolute pose it is a delta
 * against. Yaw wraps to (-pi, pi]; pitch clamps to +/- pi/2 so an NPC cannot
 * roll over backwards by looking up for long enough.
 */
public final class PolicyAgentMarker implements Component<EntityStore> {

    private final PolicyRuntime runtime;
    private final int slot;
    /** Re-enable post-behaviour native motion for one tick while AI stays frozen. */
    private final boolean stepFrozenNpc;
    private final float[] carry;
    private float desiredYaw;
    private float desiredPitch;

    // Motion evidence. The sink is called every tick and throws nothing, which
    // says only that the call happened -- steering can be silently overwritten
    // downstream and look identical. These record whether the entity actually
    // went anywhere.
    // Action repeat. `step` counts ticks this NPC has been driven for; `held`
    // is the decision being repeated between fresh ones. The GRU carry above is
    // deliberately NOT part of this -- see PolicyControlSystem for why it must
    // keep advancing on held ticks.
    private long step;
    private ActionDecoder.Decoded held;
    private WorldActionEvidence heldEvidence = WorldActionEvidence.empty();
    private boolean freshDecision;
    private final PolicyWorldVerbState worldVerb = new PolicyWorldVerbState();
    private final PolicyCombatState combat = new PolicyCombatState();

    private boolean havePosition;
    private double firstX;
    private double firstY;
    private double firstZ;
    private double lastX;
    private double lastY;
    private double lastZ;
    private double pathLength;
    private double horizontalPathLength;
    private long positionSamples;

    /** Required by the component registry; produces an inert marker. */
    public PolicyAgentMarker() {
        this(null, -1);
    }

    public PolicyAgentMarker(PolicyRuntime runtime, int slot) {
        this(runtime, slot, false);
    }

    public PolicyAgentMarker(
        PolicyRuntime runtime,
        int slot,
        boolean stepFrozenNpc
    ) {
        this.runtime = runtime;
        this.slot = slot;
        this.stepFrozenNpc = stepFrozenNpc;
        this.carry = runtime == null ? new float[0] : runtime.newCarry();
    }

    private PolicyAgentMarker(
        PolicyRuntime runtime,
        int slot,
        boolean stepFrozenNpc,
        float[] carry,
        float desiredYaw,
        float desiredPitch
    ) {
        this.runtime = runtime;
        this.slot = slot;
        this.stepFrozenNpc = stepFrozenNpc;
        this.carry = carry.clone();
        this.desiredYaw = desiredYaw;
        this.desiredPitch = desiredPitch;
    }

    public PolicyRuntime runtime() {
        return runtime;
    }

    public int slot() {
        return slot;
    }

    public boolean stepFrozenNpc() {
        return stepFrozenNpc;
    }

    /** This NPC's own GRU carry. Advanced in place by the runtime. */
    public float[] carry() {
        return carry;
    }

    public boolean active() {
        return runtime != null && slot >= 0;
    }

    public float desiredYaw() {
        return desiredYaw;
    }

    public float desiredPitch() {
        return desiredPitch;
    }

    /** Seed the pose from the NPC's spawn orientation, in radians. */
    public void setPose(float yaw, float pitch) {
        desiredYaw = normalizeRadians(yaw);
        desiredPitch = clamp(pitch, (float) (-Math.PI / 2.0), (float) (Math.PI / 2.0));
    }

    /** Integrate one tick of the policy's look deltas, given in degrees. */
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

    /**
     * Choose between a fresh decision and the one being held.
     *
     * <p>Called once per driven tick, after the network has already run. The
     * counter is this NPC's own, so a server full of policy NPCs does not
     * re-decide in lockstep on the same tick.
     *
     * @param period ticks between fresh decisions; 1 means never hold
     * @param fresh  the decision the network just produced
     * @return {@code fresh} on a decision tick, otherwise the held decision
     */
    public ActionDecoder.Decoded chooseDecision(int period, ActionDecoder.Decoded fresh) {
        return chooseDecision(
            period,
            fresh,
            WorldActionEvidence.empty()
        ).action();
    }

    /**
     * Hold the privileged candidate binding with the decision that selected it.
     * A current-tick candidate table must never reinterpret an older held index.
     */
    public ChosenDecision chooseDecision(
        int period,
        ActionDecoder.Decoded fresh,
        WorldActionEvidence freshEvidence
    ) {
        if (fresh == null || freshEvidence == null) {
            throw new IllegalArgumentException(
                "a policy decision and its evidence cannot be null");
        }
        boolean take = held == null || step % period == 0;
        step++;
        freshDecision = take;
        if (take) {
            held = fresh;
            heldEvidence = freshEvidence;
        }
        return new ChosenDecision(held, heldEvidence, take);
    }

    /** True when the last {@link #chooseDecision} repeated an earlier action. */
    public boolean holding(int period) {
        return period > 1 && step > 0 && (step - 1) % period != 0;
    }

    /** Whether the last chosen action came from this control tick. */
    public boolean freshDecision() {
        return freshDecision;
    }

    /** Caller-owned lifecycle state for this NPC's one active World verb. */
    public PolicyWorldVerbState worldVerb() {
        return worldVerb;
    }

    /** Caller-owned bridge combat handle and last lifecycle receipt. */
    public PolicyCombatState combat() {
        return combat;
    }

    /**
     * Record where this NPC is, once per controlled tick.
     *
     * <p>Path length and net displacement answer different questions and both
     * are needed. An NPC walking a circle has near-zero displacement while
     * clearly moving; one being carried by something else has displacement
     * without the policy causing it. Horizontal is tracked apart from total so
     * that falling -- which happens regardless of the policy -- cannot be
     * mistaken for locomotion.
     */
    public void observePosition(double x, double y, double z) {
        if (!havePosition) {
            havePosition = true;
            firstX = x;
            firstY = y;
            firstZ = z;
        } else {
            double dx = x - lastX;
            double dy = y - lastY;
            double dz = z - lastZ;
            pathLength += Math.sqrt(dx * dx + dy * dy + dz * dz);
            horizontalPathLength += Math.sqrt(dx * dx + dz * dz);
        }
        lastX = x;
        lastY = y;
        lastZ = z;
        positionSamples++;
    }

    public boolean havePosition() {
        return havePosition;
    }

    public long positionSamples() {
        return positionSamples;
    }

    /** Total distance travelled along the path, including vertical. */
    public double pathLength() {
        return pathLength;
    }

    /** Distance travelled ignoring the Y axis, so falling does not count. */
    public double horizontalPathLength() {
        return horizontalPathLength;
    }

    /** Straight-line distance from the first observed position, XZ only. */
    public double horizontalDisplacement() {
        if (!havePosition) {
            return 0.0;
        }
        double dx = lastX - firstX;
        double dz = lastZ - firstZ;
        return Math.sqrt(dx * dx + dz * dz);
    }

    /** Signed change on each axis since the first sample: {dx, dy, dz}. */
    public double[] displacement() {
        if (!havePosition) {
            return new double[] {0.0, 0.0, 0.0};
        }
        return new double[] {lastX - firstX, lastY - firstY, lastZ - firstZ};
    }

    /** Current position: {x, y, z}. */
    public double[] position() {
        return new double[] {lastX, lastY, lastZ};
    }

    /** Forget the episode: zero the memory and the pose. */
    public void reset() {
        java.util.Arrays.fill(carry, 0.0f);
        desiredYaw = 0.0f;
        desiredPitch = 0.0f;
        havePosition = false;
        pathLength = 0.0;
        horizontalPathLength = 0.0;
        positionSamples = 0;
        step = 0;
        held = null;
        heldEvidence = WorldActionEvidence.empty();
        freshDecision = false;
        worldVerb.resetLocal();
        combat.resetLocal();
    }

    /**
     * Deep-copies the carry. A shallow copy would silently entangle the memory
     * of the original NPC and the copy.
     */
    @Override
    public PolicyAgentMarker clone() {
        return new PolicyAgentMarker(
            runtime,
            slot,
            stepFrozenNpc,
            carry,
            desiredYaw,
            desiredPitch
        );
    }

    private static float clamp(float value, float low, float high) {
        return Math.max(low, Math.min(high, value));
    }

    private static float normalizeRadians(float angle) {
        float twoPi = (float) (Math.PI * 2.0);
        float normalized = angle % twoPi;
        if (normalized > Math.PI) {
            normalized -= twoPi;
        }
        if (normalized < -Math.PI) {
            normalized += twoPi;
        }
        return normalized;
    }

    /** One selected action and the exact bindings visible when it was chosen. */
    public record ChosenDecision(
        ActionDecoder.Decoded action,
        WorldActionEvidence evidence,
        boolean fresh
    ) {
        public ChosenDecision {
            if (action == null || evidence == null) {
                throw new IllegalArgumentException(
                    "chosen policy decision cannot contain null values");
            }
        }
    }
}
