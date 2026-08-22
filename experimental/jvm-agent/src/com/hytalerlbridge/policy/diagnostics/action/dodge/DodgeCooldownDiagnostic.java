package com.hytalerlbridge.policy.diagnostics.action.dodge;

import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.Policy;
import com.hytalerlbridge.policy.action.ActionSelectionControl;
import com.hytalerlbridge.policy.combat.bridge.BridgeCombatFacade;
import com.hytalerlbridge.policy.combat.bridge.BridgeDodgeCooldownStateSource;
import com.hytalerlbridge.policy.combat.runtime.CombatReceiptObserver;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Deterministic, actor-scoped probe for cross-direction native Dodge cooldown.
 *
 * <p>Directions are selected from the same non-neutral legality row consumed
 * by the policy. The seed chooses their order; no direction, weapon, role,
 * material, interaction id, or cooldown scalar is encoded here.</p>
 */
public final class DodgeCooldownDiagnostic
    implements ActionSelectionControl, CombatReceiptObserver {

    public enum Stage {
        WAITING_FOR_DIRECTIONS,
        REQUEST_FIRST,
        WAIT_FIRST_LIFECYCLE,
        REQUEST_SECOND,
        WAIT_SECOND_LIFECYCLE,
        WAIT_SECOND_FINISH,
        COMPLETE
    }

    private final int actorSlot;
    private final long seed;
    private final DodgeCooldownContract contract;
    private final DodgeCooldownTraceSink trace;

    private Stage stage = Stage.WAITING_FOR_DIRECTIONS;
    private long policySamples;
    private long waitingSamples;
    private int firstDirection;
    private int secondDirection;
    private int pendingDirection;
    private String lastDirectionMask = "";
    private long firstAdmissionTick = -1;
    private long firstStartedTick = -1;
    private long firstFinishedTick = -1;
    private long secondAdmissionTick = -1;
    private long secondStartedTick = -1;
    private long secondFinishedTick = -1;
    private long rejectedAdmissions;
    private long rejectedBeforeStart;
    private long oppositeRejectedAdmissions;
    private long oppositeRejectedBeforeStart;
    private long firstAttemptFailures;
    private long selectedDirectionMaskClosedSamples;
    private long bothDirectionMaskClosedSamples;
    private long unavailableReceipts;
    private long nativeStateAvailableSamples;
    private long nativeStateUnavailableSamples;
    private long nativeInactiveBeforeFirstSamples;
    private long nativeBothClosedSamples;
    private long nativeBothOpenAtBoundarySamples;
    private long nativePrematureOpenSamples;
    private long nativeActiveAtFirstStartSamples;
    private long nativeActiveAtSecondStartSamples;
    private final Set<Long> nativeBoundaryTicksObserved = new HashSet<>();
    private String firstInteractionId = "";
    private String secondInteractionId = "";

    public DodgeCooldownDiagnostic(
        int actorSlot,
        long seed,
        DodgeCooldownContract contract,
        DodgeCooldownTraceSink trace
    ) {
        if (actorSlot < 1 || contract == null || trace == null) {
            throw new IllegalArgumentException(
                "Dodge cooldown diagnostic requires a nonzero actor, contract, and trace");
        }
        this.actorSlot = actorSlot;
        this.seed = seed;
        this.contract = contract;
        this.trace = trace;
        trace.write(
            "\"kind\":\"contract\","
                + "\"actor_slot\":" + actorSlot + ","
                + "\"seed\":" + seed + ","
                + "\"cooldown_id\":\""
                    + DodgeCooldownTraceSink.json(contract.cooldownId()) + "\","
                + "\"cooldown_seconds\":" + contract.cooldownSeconds() + ","
                + "\"ticks_per_second\":" + contract.ticksPerSecond() + ","
                + "\"expected_boundary_ticks\":" + contract.boundaryTicks() + ","
                + "\"native_cooldown_state_required\":true,"
                + "\"native_cooldown_state_source\":"
                    + "\"bridge_native_dodge_cooldown_view\""
        );
    }

    @Override
    public synchronized Plan plan(
        int slot,
        float[] logits,
        boolean[] legal,
        WorldActionEvidence evidence
    ) {
        validateInputs(logits, legal, evidence);
        if (slot != actorSlot) return Plan.pass(legal);
        policySamples++;
        lastDirectionMask = directionMask(legal);

        if (stage == Stage.WAITING_FOR_DIRECTIONS) {
            int[] directions = availableDirections(legal);
            if (directions.length >= 2) {
                int selected = seededIndex(seed, directions.length);
                firstDirection = directions[selected];
                secondDirection = directions[(selected + 1) % directions.length];
                stage = Stage.REQUEST_FIRST;
            } else {
                waitingSamples++;
            }
        }

        if (firstStartedTick >= 0 && secondStartedTick < 0
                && secondDirection > 0
                && !directionLegal(legal, secondDirection)) {
            selectedDirectionMaskClosedSamples++;
            if (!directionLegal(legal, firstDirection)) {
                bothDirectionMaskClosedSamples++;
            }
        }

        int requested = switch (stage) {
            case REQUEST_FIRST -> firstDirection;
            case REQUEST_SECOND -> secondDirection;
            default -> 0;
        };
        Plan result = requested > 0 && directionLegal(legal, requested)
            ? dodgePlan(legal, requested)
            : neutralHold(legal);
        trace.write(
            "\"kind\":\"policy_mask\","
                + "\"actor_slot\":" + actorSlot + ","
                + "\"seed\":" + seed + ","
                + "\"policy_sample\":" + policySamples + ","
                + "\"stage\":\"" + stage.name().toLowerCase() + "\","
                + "\"direction_mask\":\"" + lastDirectionMask + "\","
                + "\"planned_direction\":" + requested
        );
        return result;
    }

    @Override
    public synchronized void observe(
        int slot,
        Plan plan,
        ActionDecoder.Decoded action
    ) {
        if (slot != actorSlot || !plan.constrained()) return;
        if (action == null || !action.actionLegal()) {
            throw new IllegalStateException(
                "Dodge cooldown diagnostic emitted an illegal action");
        }
        if (plan.route().equals("dodge_probe")) {
            int direction = action.dodgeDirection();
            int expected = stage == Stage.REQUEST_FIRST
                ? firstDirection
                : stage == Stage.REQUEST_SECOND ? secondDirection : 0;
            if (expected <= 0 || direction != expected) {
                throw new IllegalStateException(
                    "Dodge cooldown diagnostic direction drift");
            }
            pendingDirection = direction;
            stage = direction == firstDirection
                ? Stage.WAIT_FIRST_LIFECYCLE
                : Stage.WAIT_SECOND_LIFECYCLE;
            return;
        }
        if (!plan.route().equals("hold_dodge_cooldown")
                || !isNeutral(action)) {
            throw new IllegalStateException(
                "Dodge cooldown diagnostic failed to hold neutral");
        }
    }

    @Override
    public synchronized void observeCombatReceipt(
        Phase phase,
        long worldTick,
        int slot,
        ActionDecoder.Decoded action,
        boolean firstControlTick,
        BridgeCombatFacade.Receipt receipt,
        BridgeDodgeCooldownStateSource.State dodgeCooldownBefore,
        BridgeDodgeCooldownStateSource.State dodgeCooldownAfter
    ) {
        if (slot != actorSlot || receipt == null) return;
        int requestedDirection = action == null ? 0 : action.dodgeDirection();
        observeNativeCooldownState(
            phase,
            worldTick,
            dodgeCooldownBefore
        );
        trace.write(receiptFields(
            phase,
            worldTick,
            requestedDirection,
            firstControlTick,
            receipt,
            dodgeCooldownBefore,
            dodgeCooldownAfter
        ));
        if (!receipt.available()) {
            unavailableReceipts++;
            observeNativeCooldownState(
                phase,
                worldTick,
                dodgeCooldownAfter
            );
            if (phase == Phase.APPLY && requestedDirection == pendingDirection) {
                pendingDirection = 0;
                retryCurrentStage();
            }
            return;
        }

        // Poll events belong to an earlier queued request, while admission in
        // the same receipt belongs to the edge issued now. Clear the terminal
        // old request first so a same-direction retry is not erased.
        if (receipt.dodgeRejectedBeforeStart() || receipt.dodgeFailed()) {
            if (receipt.dodgeRejectedBeforeStart()) rejectedBeforeStart++;
            int direction = receipt.dodgeDirection();
            if (direction == firstDirection && firstStartedTick < 0) {
                firstAttemptFailures++;
                firstAdmissionTick = -1;
                firstInteractionId = "";
            } else if (direction == secondDirection && secondStartedTick < 0) {
                if (receipt.dodgeRejectedBeforeStart()) {
                    oppositeRejectedBeforeStart++;
                }
                secondAdmissionTick = -1;
                secondInteractionId = "";
            }
            if (direction == pendingDirection || direction == 0) {
                pendingDirection = 0;
                retryCurrentStage();
            }
        }

        if (phase == Phase.APPLY && receipt.dodgeRequested()) {
            if (receipt.dodgeAccepted()) {
                if (requestedDirection == firstDirection) {
                    firstAdmissionTick = worldTick;
                    firstInteractionId = clean(receipt.dodgeInteractionId());
                    pendingDirection = firstDirection;
                    stage = Stage.WAIT_FIRST_LIFECYCLE;
                } else if (requestedDirection == secondDirection) {
                    secondAdmissionTick = worldTick;
                    secondInteractionId = clean(receipt.dodgeInteractionId());
                    pendingDirection = secondDirection;
                    stage = Stage.WAIT_SECOND_LIFECYCLE;
                }
            } else {
                rejectedAdmissions++;
                if (requestedDirection == firstDirection) {
                    firstAttemptFailures++;
                } else if (requestedDirection == secondDirection) {
                    oppositeRejectedAdmissions++;
                }
                if (requestedDirection == pendingDirection) {
                    pendingDirection = 0;
                    retryCurrentStage();
                }
            }
        }

        if (receipt.dodgeStarted()) {
            int direction = receipt.dodgeDirection();
            if (direction == firstDirection && firstStartedTick < 0) {
                firstStartedTick = worldTick;
                pendingDirection = 0;
                stage = Stage.WAIT_FIRST_LIFECYCLE;
            } else if (direction == secondDirection && secondStartedTick < 0) {
                secondStartedTick = worldTick;
                pendingDirection = 0;
                stage = Stage.WAIT_SECOND_FINISH;
            }
        }
        if (receipt.dodgeFinished()) {
            int direction = receipt.dodgeDirection();
            if (direction == firstDirection && firstFinishedTick < 0) {
                firstFinishedTick = worldTick;
                if (firstStartedTick >= 0) stage = Stage.REQUEST_SECOND;
            } else if (direction == secondDirection && secondFinishedTick < 0) {
                secondFinishedTick = worldTick;
                if (secondStartedTick >= 0) stage = Stage.COMPLETE;
            }
        }
        observeNativeCooldownState(
            phase,
            worldTick,
            dodgeCooldownAfter
        );
    }

    public synchronized boolean complete() {
        return stage == Stage.COMPLETE;
    }

    /** A complete result is certified only at the bridge-derived boundary. */
    public synchronized boolean certified() {
        return complete()
            && firstAdmissionTick >= 0
            && firstStartedTick >= firstAdmissionTick
            && firstFinishedTick >= firstStartedTick
            && secondAdmissionTick - firstAdmissionTick
                == contract.boundaryTicks()
            && secondStartedTick >= secondAdmissionTick
            && secondFinishedTick >= secondStartedTick
            && nativeStateAvailableSamples > 0
            && nativeStateUnavailableSamples == 0
            && firstAttemptFailures == 0
            && nativeInactiveBeforeFirstSamples > 0
            && nativeBothClosedSamples > 0
            && nativeBothOpenAtBoundarySamples > 0
            && nativePrematureOpenSamples == 0
            && nativeActiveAtFirstStartSamples > 0
            && nativeActiveAtSecondStartSamples > 0
            && nativeBoundaryTicksObserved.size()
                == contract.boundaryTicks() + 1
            && (contract.boundaryTicks() == 1
                || oppositeRejectedAdmissions > 0
                || oppositeRejectedBeforeStart > 0
                || bothDirectionMaskClosedSamples > 0)
            && !firstInteractionId.isEmpty()
            && !secondInteractionId.isEmpty()
            && !firstInteractionId.equals(secondInteractionId)
            && unavailableReceipts == 0;
    }

    public synchronized long observedBoundaryTicks() {
        return firstAdmissionTick < 0 || secondAdmissionTick < 0
            ? -1 : secondAdmissionTick - firstAdmissionTick;
    }

    public synchronized Stage stage() {
        return stage;
    }

    public synchronized int firstDirection() {
        return firstDirection;
    }

    public synchronized int secondDirection() {
        return secondDirection;
    }

    public synchronized long rejectedBeforeStart() {
        return rejectedBeforeStart;
    }

    public synchronized long rejectedAdmissions() {
        return rejectedAdmissions;
    }

    public synchronized long oppositeRejections() {
        return oppositeRejectedAdmissions + oppositeRejectedBeforeStart;
    }

    public synchronized String describe() {
        String verdict = certified() ? "passed" : complete() ? "failed" : "pending";
        return "dodge_cooldown actor_slot=" + actorSlot
            + " seed=" + seed
            + " cooldown_id=" + contract.cooldownId()
            + " cooldown_seconds=" + contract.cooldownSeconds()
            + " expected_boundary_ticks=" + contract.boundaryTicks()
            + " stage=" + stage.name().toLowerCase()
            + " first_direction=" + firstDirection
            + " second_direction=" + secondDirection
            + " first_admission_tick=" + firstAdmissionTick
            + " first_started_tick=" + firstStartedTick
            + " first_finished_tick=" + firstFinishedTick
            + " second_admission_tick=" + secondAdmissionTick
            + " second_started_tick=" + secondStartedTick
            + " second_finished_tick=" + secondFinishedTick
            + " observed_boundary_ticks=" + observedBoundaryTicks()
            + " admission_rejections=" + rejectedAdmissions
            + " rejected_before_start=" + rejectedBeforeStart
            + " opposite_admission_rejections="
                + oppositeRejectedAdmissions
            + " opposite_rejected_before_start="
                + oppositeRejectedBeforeStart
            + " first_attempt_failures=" + firstAttemptFailures
            + " selected_mask_closed_samples="
                + selectedDirectionMaskClosedSamples
            + " both_mask_closed_samples=" + bothDirectionMaskClosedSamples
            + " unavailable_receipts=" + unavailableReceipts
            + " native_state_available_samples=" + nativeStateAvailableSamples
            + " native_state_unavailable_samples="
                + nativeStateUnavailableSamples
            + " native_inactive_before_first_samples="
                + nativeInactiveBeforeFirstSamples
            + " native_both_closed_samples=" + nativeBothClosedSamples
            + " native_both_open_at_boundary_samples="
                + nativeBothOpenAtBoundarySamples
            + " native_premature_open_samples=" + nativePrematureOpenSamples
            + " native_active_at_first_start_samples="
                + nativeActiveAtFirstStartSamples
            + " native_active_at_second_start_samples="
                + nativeActiveAtSecondStartSamples
            + " native_boundary_ticks_observed="
                + nativeBoundaryTicksObserved.size()
            + " policy_samples=" + policySamples
            + " waiting_samples=" + waitingSamples
            + " trace_events=" + trace.eventsWritten()
            + " trace_path=" + (trace.path() == null ? "" : trace.path())
            + " trace_error=" + trace.failure()
            + " verdict=" + verdict;
    }

    private void retryCurrentStage() {
        if (firstStartedTick < 0) {
            stage = Stage.REQUEST_FIRST;
        } else if (firstFinishedTick < 0) {
            stage = Stage.WAIT_FIRST_LIFECYCLE;
        } else if (secondStartedTick < 0) {
            stage = Stage.REQUEST_SECOND;
        } else if (secondFinishedTick < 0) {
            stage = Stage.WAIT_SECOND_FINISH;
        }
    }

    private String receiptFields(
        Phase phase,
        long worldTick,
        int requestedDirection,
        boolean firstControlTick,
        BridgeCombatFacade.Receipt receipt,
        BridgeDodgeCooldownStateSource.State dodgeCooldownBefore,
        BridgeDodgeCooldownStateSource.State dodgeCooldownAfter
    ) {
        return "\"kind\":\"combat_receipt\","
            + "\"phase\":\"" + phase.name().toLowerCase() + "\","
            + "\"world_tick\":" + worldTick + ","
            + "\"actor_slot\":" + actorSlot + ","
            + "\"policy_sample\":" + policySamples + ","
            + "\"first_control_tick\":" + firstControlTick + ","
            + "\"direction_mask\":\"" + lastDirectionMask + "\","
            + "\"requested_direction\":" + requestedDirection + ","
            + "\"available\":" + receipt.available() + ","
            + "\"unavailable_reason\":\""
                + DodgeCooldownTraceSink.json(receipt.unavailableReason()) + "\","
            + "\"dodge_requested\":" + receipt.dodgeRequested() + ","
            + "\"dodge_accepted\":" + receipt.dodgeAccepted() + ","
            + "\"interaction_id\":\""
                + DodgeCooldownTraceSink.json(receipt.dodgeInteractionId()) + "\","
            + "\"reject_reason\":\""
                + DodgeCooldownTraceSink.json(receipt.dodgeRejectReason()) + "\","
            + "\"started\":" + receipt.dodgeStarted() + ","
            + "\"finished\":" + receipt.dodgeFinished() + ","
            + "\"failed\":" + receipt.dodgeFailed() + ","
            + "\"rejected_before_start\":"
                + receipt.dodgeRejectedBeforeStart() + ","
            + "\"event_direction\":" + receipt.dodgeDirection() + ","
            + "\"active\":" + receipt.dodgeActive() + ","
            + nativeStateFields("before", dodgeCooldownBefore) + ","
            + nativeStateFields("after", dodgeCooldownAfter);
    }

    private String nativeStateFields(
        String position,
        BridgeDodgeCooldownStateSource.State state
    ) {
        return "\"native_cooldown_" + position + "_available\":"
            + (state != null && state.available()) + ","
            + "\"native_cooldown_" + position + "_active\":"
            + (state == null || !state.available() || state.onCooldown()) + ","
            + "\"native_shared_cooldown_" + position + "_mask\":\""
            + nativeSharedCooldownMask(state) + "\","
            + "\"native_cooldown_" + position
                + "_unavailable_reason\":\""
            + DodgeCooldownTraceSink.json(
                state == null
                    ? "dodge_cooldown_state_missing"
                    : state.unavailableReason())
            + "\"";
    }

    private void observeNativeCooldownState(
        Phase phase,
        long worldTick,
        BridgeDodgeCooldownStateSource.State state
    ) {
        if (state == null || !state.available()) {
            nativeStateUnavailableSamples++;
            return;
        }
        nativeStateAvailableSamples++;
        if (firstStartedTick < 0 && !state.onCooldown()) {
            nativeInactiveBeforeFirstSamples++;
        }
        if (firstDirection > 0 && secondDirection > 0
                && state.onCooldown()) {
            nativeBothClosedSamples++;
        }
        if (firstAdmissionTick >= 0 && secondAdmissionTick < 0) {
            long elapsed = worldTick - firstAdmissionTick;
            if (elapsed >= 0 && elapsed <= contract.boundaryTicks()) {
                nativeBoundaryTicksObserved.add(elapsed);
            }
        }
        if (firstStartedTick >= 0 && worldTick == firstStartedTick
                && state.onCooldown()) {
            nativeActiveAtFirstStartSamples++;
        }
        if (secondStartedTick >= 0 && worldTick == secondStartedTick
                && state.onCooldown()) {
            nativeActiveAtSecondStartSamples++;
        }
        if (firstAdmissionTick < 0 || secondAdmissionTick >= 0
                || state.onCooldown()) {
            return;
        }
        long elapsed = worldTick - firstAdmissionTick;
        if (elapsed > 0 && elapsed < contract.boundaryTicks()) {
            nativePrematureOpenSamples++;
        }
        if (phase == Phase.PREPARE
                && elapsed == contract.boundaryTicks()) {
            nativeBothOpenAtBoundarySamples++;
        }
    }

    private String nativeSharedCooldownMask(
        BridgeDodgeCooldownStateSource.State state
    ) {
        if (state == null || !state.available()
                || firstDirection <= 0 || secondDirection <= 0) {
            return "unknown";
        }
        return state.onCooldown() ? "00" : "11";
    }

    private static Plan dodgePlan(boolean[] legal, int direction) {
        boolean[] mask = legal.clone();
        for (int head = 0; head < Policy.HEAD_SIZES.length; head++) {
            if (head == ActionDecoder.LOCOMOTION) continue;
            retainOnly(mask, head, neutralChoice(head));
        }
        retainOnly(mask, ActionDecoder.LOCOMOTION,
            ActionDecoder.locomotionDodgeChoice(direction));
        return viable(mask)
            ? new Plan(mask, true, "dodge_probe")
            : Plan.pass(legal);
    }

    private static Plan neutralHold(boolean[] legal) {
        boolean[] mask = neutralMask(legal);
        return viable(mask)
            ? new Plan(mask, true, "hold_dodge_cooldown")
            : Plan.pass(legal);
    }

    private static boolean[] neutralMask(boolean[] legal) {
        boolean[] mask = legal.clone();
        for (int head = 0; head < Policy.HEAD_SIZES.length; head++) {
            retainOnly(mask, head, neutralChoice(head));
        }
        return mask;
    }

    /**
     * The do-nothing choice for a head.
     *
     * <p>Signed-delta heads are odd-width with zero in the middle; every other
     * head, including locomotion, uses index 0. Locomotion 0 is idle, which is
     * also no dodge -- one head now pins what used to take three.
     */
    private static int neutralChoice(int head) {
        return head == ActionDecoder.YAW_BINS
                || head == ActionDecoder.BODY_YAW_BINS
                || head == ActionDecoder.PITCH_BINS
            ? Policy.HEAD_SIZES[head] / 2
            : 0;
    }

    private static int[] availableDirections(boolean[] legal) {
        List<Integer> values = new ArrayList<>();
        int start = offset(ActionDecoder.LOCOMOTION);
        for (int direction = 1;
                direction <= ActionDecoder.dodgeDirectionCount(); direction++) {
            if (legal[start + ActionDecoder.locomotionDodgeChoice(direction)]) {
                values.add(direction);
            }
        }
        return values.stream().mapToInt(Integer::intValue).toArray();
    }

    private static boolean directionLegal(boolean[] legal, int direction) {
        return direction > 0
            && direction <= ActionDecoder.dodgeDirectionCount()
            && legal[offset(ActionDecoder.LOCOMOTION)
                + ActionDecoder.locomotionDodgeChoice(direction)];
    }

    private static String directionMask(boolean[] legal) {
        StringBuilder value = new StringBuilder();
        int start = offset(ActionDecoder.LOCOMOTION);
        for (int direction = 1;
                direction <= ActionDecoder.dodgeDirectionCount(); direction++) {
            value.append(
                legal[start + ActionDecoder.locomotionDodgeChoice(direction)]
                    ? '1' : '0');
        }
        return value.toString();
    }

    private static int seededIndex(long seed, int size) {
        long value = seed + 0x9E3779B97F4A7C15L;
        value = (value ^ (value >>> 30)) * 0xBF58476D1CE4E5B9L;
        value = (value ^ (value >>> 27)) * 0x94D049BB133111EBL;
        value ^= value >>> 31;
        return (int) Long.remainderUnsigned(value, size);
    }

    private static boolean isNeutral(ActionDecoder.Decoded action) {
        return !action.attack()
            && action.abilitySlot() < 0
            && !action.guardHeld()
            && action.dodgeDirection() == 0
            && !action.jumpHeld()
            && action.worldMoveDirection() == 0
            && !action.forward() && !action.back()
            && !action.left() && !action.right()
            && !action.requestsWorldVerb()
            && action.yawDeltaDegrees() == 0.0
            && action.pitchDeltaDegrees() == 0.0;
    }

    private static void retainOnly(boolean[] mask, int head, int selected) {
        int start = offset(head);
        for (int index = 0; index < Policy.HEAD_SIZES[head]; index++) {
            if (index != selected) mask[start + index] = false;
        }
    }

    private static boolean viable(boolean[] mask) {
        for (int head = 0; head < Policy.HEAD_SIZES.length; head++) {
            boolean any = false;
            int start = offset(head);
            for (int index = 0; index < Policy.HEAD_SIZES[head]; index++) {
                any |= mask[start + index];
            }
            if (!any) return false;
        }
        return true;
    }

    private static int offset(int head) {
        int result = 0;
        for (int index = 0; index < head; index++) {
            result += Policy.HEAD_SIZES[index];
        }
        return result;
    }

    private static void validateInputs(
        float[] logits,
        boolean[] legal,
        WorldActionEvidence evidence
    ) {
        int width = 0;
        for (int size : Policy.HEAD_SIZES) width += size;
        if (logits == null || logits.length != width
                || legal == null || legal.length != width || evidence == null) {
            throw new IllegalArgumentException(
                "Dodge cooldown diagnostic input width drift");
        }
    }

    private static String clean(String value) {
        return value == null ? "" : value;
    }
}
