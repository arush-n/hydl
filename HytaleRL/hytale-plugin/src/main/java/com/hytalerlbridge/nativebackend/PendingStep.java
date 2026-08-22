package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.server.core.universe.world.World;
import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.action.NativeWorldVerbRequest;
import com.hytalerlbridge.action.group.NativeGroupAction;
import com.hytalerlbridge.action.group.NativeGroupActionContract;
import com.hytalerlbridge.environment.StepResult;
import com.hytalerlbridge.nativebackend.group.NativeAttackAdmission;
import com.hytalerlbridge.nativebackend.policy.combat.NativePolicyCombatFacade;
import com.hytalerlbridge.observation.AudioFrame;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;

/**
 * The per-step buffer the world thread fills while a step is in flight.
 *
 * <p>Extracted from {@code NativeEnvironmentSession}, where it was a 614-line
 * private nested class - about 6% of that file. It references nothing from the
 * session (zero outer fields, zero outer methods), so it was always a separate
 * type living in the wrong place.
 *
 * <p>Deliberately package-private, and its members are widened from
 * {@code private} only where the session actually reads them. Java subpackages
 * do not see a parent package's package-private members, so moving this into
 * {@code session/} would have forced the class and ~87 members to become
 * {@code public} - a real widening of the bridge's API surface to satisfy a
 * file move. Staying in {@code nativebackend} keeps the public surface
 * unchanged: nothing outside this package can see it.
 *
 * <p>This is a mutable buffer, not a value type: the session writes its fields
 * individually as native callbacks land during the step, then reads them back
 * when composing the step's info map.
 */
final class PendingStep {
    final NativeGroupAction actions;
    final AgentAction action;
    final int requestedTicks;
    final boolean episodeStep;
    final CompletableFuture<StepResult> future = new CompletableFuture<>();
    final int[] controlTicks = new int[
        NativeGroupActionContract.actorCapacity()
    ];
    private final boolean[] attackRequested = new boolean[
        NativeGroupActionContract.actorCapacity()
    ];
    final boolean[] attackAcceptedByActor = new boolean[
        NativeGroupActionContract.actorCapacity()
    ];
    private final String[] attackRejectReasonByActor = new String[
        NativeGroupActionContract.actorCapacity()
    ];
    final boolean[] abilityRequestedByActor = actorBits();
    final boolean[] abilityAcceptedByActor = actorBits();
    final boolean[] abilityStartedByActor = actorBits();
    final boolean[] abilityFinishedByActor = actorBits();
    final boolean[] abilityFailedByActor = actorBits();
    final boolean[] abilityActiveByActor = actorBits();
    private final int[] abilitySlotByActor = actorInts(-1);
    private final String[] abilityRejectReasonByActor = actorStrings();
    private final String[] abilityInteractionIdByActor = actorStrings();
    final boolean[] guardRequestedByActor = actorBits();
    final boolean[] guardAcceptedByActor = actorBits();
    final boolean[] guardStartedByActor = actorBits();
    final boolean[] guardFinishedByActor = actorBits();
    final boolean[] guardActiveByActor = actorBits();
    private final String[] guardRejectReasonByActor = actorStrings();
    final boolean[] dodgeRequestedByActor = actorBits();
    final boolean[] dodgeAcceptedByActor = actorBits();
    final boolean[] dodgeStartedByActor = actorBits();
    final boolean[] dodgeFinishedByActor = actorBits();
    final boolean[] dodgeFailedByActor = actorBits();
    final boolean[] dodgeActiveByActor = actorBits();
    private final int[] dodgeDirectionByActor = actorInts(0);
    private final String[] dodgeRejectReasonByActor = actorStrings();
    private final String[] dodgeInteractionIdByActor = actorStrings();
    final boolean[] worldVerbRequestedByActor = actorBits();
    final boolean[] worldVerbAcceptedByActor = actorBits();
    final boolean[] worldVerbStartedByActor = actorBits();
    final boolean[] worldVerbFinishedByActor = actorBits();
    final boolean[] worldVerbFailedByActor = actorBits();
    final boolean[] worldVerbActiveByActor = actorBits();
    private final String[] worldVerbRejectReasonByActor = actorStrings();
    private final String[] worldVerbByActor = actorStrings();
    private final String[] worldVerbGenerationByActor = actorStrings();
    private final String[] worldVerbSemanticByActor = actorStrings();
    boolean worldVerbBatchStarted;
    int executedTicks;
    double simulatedSeconds;
    double minimumDeltaSeconds = Double.POSITIVE_INFINITY;
    double maximumDeltaSeconds;
    boolean attackAccepted;
    boolean guardAccepted;
    boolean guardStarted;
    boolean guardFinished;
    String guardRejectReason = "";
    boolean dodgeAccepted;
    boolean dodgeStarted;
    boolean dodgeFinished;
    boolean dodgeFailed;
    String dodgeRejectReason = "";
    String dodgeInteractionId = "";
    boolean abilityAccepted;
    int abilityAcceptedSlot = -1;
    boolean abilityStarted;
    boolean abilityFinished;
    boolean abilityFailed;
    String abilityRejectReason = "";
    String abilityInteractionId = "";
    String abilityInteractionType = "";
    boolean useAccepted;
    boolean useStarted;
    boolean useFinished;
    boolean useFailed;
    String useRejectReason = "";
    /** Deprecated actor-zero projection; v5 actor-major fields are authoritative. */
    WorldVerbTelemetry.Execution legacyActorZeroWorldVerbExecution;
    SyntheticWorldInteractionEvidence syntheticWorldInteraction;
    NativeDiveTrace diveTrace;
    final AudioFrame.StepBuffer audio = new AudioFrame.StepBuffer();

    PendingStep(
        NativeGroupAction actions,
        int requestedTicks,
        boolean episodeStep
    ) {
        this.actions = actions;
        this.action = actions.primaryAction();
        this.requestedTicks = requestedTicks;
        this.episodeStep = episodeStep;
        for (int actorId = 0; actorId < attackRejectReasonByActor.length; actorId++) {
            attackRejectReasonByActor[actorId] = "";
            AgentAction actorAction = actions.actionForEntity(actorId);
            attackRequested[actorId] = actorAction != null
                && actorAction.attack();
            abilityRequestedByActor[actorId] = actorAction != null
                && actorAction.hasAbility();
            abilitySlotByActor[actorId] = actorAction == null
                ? -1
                : actorAction.abilitySlot();
            guardRequestedByActor[actorId] = actorAction != null
                && actorAction.guardHeld();
            dodgeRequestedByActor[actorId] = actorAction != null
                && actorAction.hasDodge();
            dodgeDirectionByActor[actorId] = actorAction == null
                ? 0
                : actorAction.dodgeDirection();
            NativeWorldVerbRequest worldVerb = actorAction == null
                ? NativeWorldVerbRequest.none()
                : actorAction.nativeWorldVerbRequest();
            worldVerbRequestedByActor[actorId] = worldVerb.present();
            if (worldVerb.present()) {
                worldVerbByActor[actorId] = worldVerb.verb();
                worldVerbGenerationByActor[actorId] =
                    worldVerb.expectedCandidateGenerationSha256();
                worldVerbSemanticByActor[actorId] =
                    worldVerb.expectedSelectedSemanticSha256();
            }
        }
    }

    AgentAction actionForEntity(int entityId) {
        return actions.actionForEntity(entityId);
    }

    int controlTicks(int entityId) {
        return controlTicks[entityId];
    }

    void incrementControlTicks(int entityId) {
        controlTicks[entityId]++;
    }

    void observeAttack(
        int entityId,
        NativeAttackAdmission admission
    ) {
        attackAcceptedByActor[entityId] = admission.accepted();
        attackRejectReasonByActor[entityId] = admission.rejectReason();
        if (entityId == 0) attackAccepted = admission.accepted();
    }

    void observePolicyCombat(
        int entityId,
        NativePolicyCombatFacade.Receipt receipt
    ) {
        if (receipt == null || !receipt.available()) {
            String reason = receipt == null
                ? "native_policy_combat_receipt_unavailable"
                : receipt.unavailableReason();
            if (attackRequested[entityId]) {
                putReason(attackRejectReasonByActor, entityId, reason);
            }
            if (abilityRequestedByActor[entityId]) {
                putReason(abilityRejectReasonByActor, entityId, reason);
            }
            if (guardRequestedByActor[entityId]) {
                putReason(guardRejectReasonByActor, entityId, reason);
            }
            if (dodgeRequestedByActor[entityId]) {
                putReason(dodgeRejectReasonByActor, entityId, reason);
            }
            return;
        }
        attackAcceptedByActor[entityId] |= receipt.attackAccepted();
        if (receipt.attackRequested() && !receipt.attackAccepted()) {
            putReason(
                attackRejectReasonByActor,
                entityId,
                receipt.attackRejectReason()
            );
        }

        abilityAcceptedByActor[entityId] |= receipt.abilityAccepted();
        abilityStartedByActor[entityId] |= receipt.abilityStarted();
        abilityFinishedByActor[entityId] |= receipt.abilityFinished();
        abilityFailedByActor[entityId] |= receipt.abilityFailed()
            || receipt.abilityRejectedBeforeStart();
        abilityActiveByActor[entityId] = receipt.abilityActive();
        if (receipt.abilitySlot() >= 0) {
            abilitySlotByActor[entityId] = receipt.abilitySlot();
        }
        if (!receipt.abilityInteractionId().isEmpty()) {
            abilityInteractionIdByActor[entityId] =
                receipt.abilityInteractionId();
        }
        if (receipt.abilityRequested() && !receipt.abilityAccepted()) {
            putReason(
                abilityRejectReasonByActor,
                entityId,
                receipt.abilityRejectReason()
            );
        } else if (receipt.abilityRejectedBeforeStart()) {
            putReason(
                abilityRejectReasonByActor,
                entityId,
                "native_interaction_cooldown_or_rules"
            );
        }

        guardAcceptedByActor[entityId] |= receipt.guardAccepted();
        guardStartedByActor[entityId] |= receipt.guardStarted();
        guardFinishedByActor[entityId] |= receipt.guardFinished();
        guardActiveByActor[entityId] = receipt.guardWieldingActive();
        if (receipt.guardRequested() && !receipt.guardAccepted()) {
            putReason(
                guardRejectReasonByActor,
                entityId,
                receipt.guardRejectReason()
            );
        }

        dodgeAcceptedByActor[entityId] |= receipt.dodgeAccepted();
        dodgeStartedByActor[entityId] |= receipt.dodgeStarted();
        dodgeFinishedByActor[entityId] |= receipt.dodgeFinished();
        dodgeFailedByActor[entityId] |= receipt.dodgeFailed()
            || receipt.dodgeRejectedBeforeStart();
        dodgeActiveByActor[entityId] = receipt.dodgeActive();
        if (receipt.dodgeDirection() > 0) {
            dodgeDirectionByActor[entityId] = receipt.dodgeDirection();
        }
        if (!receipt.dodgeInteractionId().isEmpty()) {
            dodgeInteractionIdByActor[entityId] =
                receipt.dodgeInteractionId();
        }
        if (receipt.dodgeRequested() && !receipt.dodgeAccepted()) {
            putReason(
                dodgeRejectReasonByActor,
                entityId,
                receipt.dodgeRejectReason()
            );
        } else if (receipt.dodgeRejectedBeforeStart()) {
            putReason(
                dodgeRejectReasonByActor,
                entityId,
                "native_interaction_cooldown_or_rules"
            );
        }
    }

    void observeUnboundPolicyCombat(
        int entityId,
        AgentAction actorAction
    ) {
        if (actorAction.hasAbility()) {
            putReason(
                abilityRejectReasonByActor,
                entityId,
                "native_policy_combat_binding_unavailable"
            );
        }
        if (actorAction.guardHeld()) {
            putReason(
                guardRejectReasonByActor,
                entityId,
                "native_policy_combat_binding_unavailable"
            );
        }
        if (actorAction.hasDodge()) {
            putReason(
                dodgeRejectReasonByActor,
                entityId,
                "native_policy_combat_binding_unavailable"
            );
        }
    }

    void rejectWorldVerb(
        int entityId,
        NativeWorldVerbRequest request,
        String reason
    ) {
        if (
            request == null
                || !request.present()
                || reason == null
                || reason.isBlank()
        ) {
            throw new IllegalArgumentException(
                "Rejected actor World verb requires request and reason"
            );
        }
        worldVerbRequestedByActor[entityId] = true;
        worldVerbFailedByActor[entityId] = true;
        worldVerbActiveByActor[entityId] = false;
        worldVerbRejectReasonByActor[entityId] = reason;
        worldVerbByActor[entityId] = request.verb();
        worldVerbGenerationByActor[entityId] =
            request.expectedCandidateGenerationSha256();
        worldVerbSemanticByActor[entityId] =
            request.expectedSelectedSemanticSha256();
        if (entityId == 0) {
            legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.rejected(
                    request,
                    reason,
                    -1L
                );
        }
    }

    void rejectUnprocessedWorldVerbs() {
        for (
            int entityId = 0;
            entityId < worldVerbRequestedByActor.length;
            entityId++
        ) {
            if (
                !worldVerbRequestedByActor[entityId]
                    || worldVerbAcceptedByActor[entityId]
                    || worldVerbFailedByActor[entityId]
            ) {
                continue;
            }
            AgentAction actorAction = actions.actionForEntity(entityId);
            NativeWorldVerbRequest request = actorAction == null
                ? NativeWorldVerbRequest.none()
                : actorAction.nativeWorldVerbRequest();
            rejectWorldVerb(
                entityId,
                request,
                "native_world_verb_actor_control_not_observed"
            );
        }
    }

    void observeWorldVerb(
        int entityId,
        NativeWorldVerbRequest request,
        NativePolicyWorldVerbFacade.Lifecycle lifecycle
    ) {
        if (request == null || !request.present() || lifecycle == null) {
            throw new IllegalArgumentException(
                "Actor World-verb lifecycle is incomplete"
            );
        }
        worldVerbAcceptedByActor[entityId] |= lifecycle.accepted();
        worldVerbStartedByActor[entityId] |= lifecycle.started();
        worldVerbFinishedByActor[entityId] |= lifecycle.finished();
        worldVerbFailedByActor[entityId] |= lifecycle.failed();
        worldVerbActiveByActor[entityId] = lifecycle.pending();
        if (!lifecycle.rejectReason().isEmpty()) {
            worldVerbRejectReasonByActor[entityId] =
                lifecycle.rejectReason();
        }
        worldVerbByActor[entityId] = request.verb();
        worldVerbGenerationByActor[entityId] =
            request.expectedCandidateGenerationSha256();
        worldVerbSemanticByActor[entityId] =
            request.expectedSelectedSemanticSha256();
        if (entityId == 0) {
            legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.actorMajorProjection(
                    request,
                    lifecycle
                );
        }
    }

    void observeLegacyPrimaryCombat(
        boolean guardActive,
        boolean abilityActive,
        boolean dodgeActive
    ) {
        abilityAcceptedByActor[0] |= abilityAccepted;
        abilityStartedByActor[0] |= abilityStarted;
        abilityFinishedByActor[0] |= abilityFinished;
        abilityFailedByActor[0] |= abilityFailed;
        abilityActiveByActor[0] = abilityActive;
        if (!abilityInteractionId.isEmpty()) {
            abilityInteractionIdByActor[0] = abilityInteractionId;
        }
        putReason(abilityRejectReasonByActor, 0, abilityRejectReason);

        guardAcceptedByActor[0] |= guardAccepted;
        guardStartedByActor[0] |= guardStarted;
        guardFinishedByActor[0] |= guardFinished;
        guardActiveByActor[0] = guardActive;
        putReason(guardRejectReasonByActor, 0, guardRejectReason);

        dodgeAcceptedByActor[0] |= dodgeAccepted;
        dodgeStartedByActor[0] |= dodgeStarted;
        dodgeFinishedByActor[0] |= dodgeFinished;
        dodgeFailedByActor[0] |= dodgeFailed;
        dodgeActiveByActor[0] = dodgeActive;
        if (!dodgeInteractionId.isEmpty()) {
            dodgeInteractionIdByActor[0] = dodgeInteractionId;
        }
        putReason(dodgeRejectReasonByActor, 0, dodgeRejectReason);
    }

    String requestedEntityIds() {
        return actions.actors().stream()
            .map(actor -> Integer.toString(actor.entityId()))
            .reduce((left, right) -> left + "," + right)
            .orElse("");
    }

    String controlTicksByActor() {
        return joinInts(controlTicks);
    }

    String attackRequestedByActor() {
        return joinBooleans(attackRequested);
    }

    String attackAcceptedByActor() {
        return joinBooleans(attackAcceptedByActor);
    }

    String attackRejectReasonsByActor() {
        return String.join(",", attackRejectReasonByActor);
    }

    String abilityRequestedByActor() {
        return joinBooleans(abilityRequestedByActor);
    }

    String abilityAcceptedByActor() {
        return joinBooleans(abilityAcceptedByActor);
    }

    String abilityStartedByActor() {
        return joinBooleans(abilityStartedByActor);
    }

    String abilityFinishedByActor() {
        return joinBooleans(abilityFinishedByActor);
    }

    String abilityFailedByActor() {
        return joinBooleans(abilityFailedByActor);
    }

    String abilityActiveByActor() {
        return joinBooleans(abilityActiveByActor);
    }

    String abilitySlotsByActor() {
        return joinInts(abilitySlotByActor);
    }

    String abilityRejectReasonsByActor() {
        return String.join(",", abilityRejectReasonByActor);
    }

    String abilityInteractionIdsByActor() {
        return String.join(",", abilityInteractionIdByActor);
    }

    String guardRequestedByActor() {
        return joinBooleans(guardRequestedByActor);
    }

    String guardAcceptedByActor() {
        return joinBooleans(guardAcceptedByActor);
    }

    String guardStartedByActor() {
        return joinBooleans(guardStartedByActor);
    }

    String guardFinishedByActor() {
        return joinBooleans(guardFinishedByActor);
    }

    String guardActiveByActor() {
        return joinBooleans(guardActiveByActor);
    }

    String guardRejectReasonsByActor() {
        return String.join(",", guardRejectReasonByActor);
    }

    String dodgeRequestedByActor() {
        return joinBooleans(dodgeRequestedByActor);
    }

    String dodgeAcceptedByActor() {
        return joinBooleans(dodgeAcceptedByActor);
    }

    String dodgeStartedByActor() {
        return joinBooleans(dodgeStartedByActor);
    }

    String dodgeFinishedByActor() {
        return joinBooleans(dodgeFinishedByActor);
    }

    String dodgeFailedByActor() {
        return joinBooleans(dodgeFailedByActor);
    }

    String dodgeActiveByActor() {
        return joinBooleans(dodgeActiveByActor);
    }

    String dodgeDirectionsByActor() {
        return joinInts(dodgeDirectionByActor);
    }

    String dodgeRejectReasonsByActor() {
        return String.join(",", dodgeRejectReasonByActor);
    }

    String dodgeInteractionIdsByActor() {
        return String.join(",", dodgeInteractionIdByActor);
    }

    String worldVerbRequestedByActor() {
        return joinBooleans(worldVerbRequestedByActor);
    }

    String worldVerbAcceptedByActor() {
        return joinBooleans(worldVerbAcceptedByActor);
    }

    String worldVerbStartedByActor() {
        return joinBooleans(worldVerbStartedByActor);
    }

    String worldVerbFinishedByActor() {
        return joinBooleans(worldVerbFinishedByActor);
    }

    String worldVerbFailedByActor() {
        return joinBooleans(worldVerbFailedByActor);
    }

    String worldVerbActiveByActor() {
        return joinBooleans(worldVerbActiveByActor);
    }

    String worldVerbRejectReasonsByActor() {
        return String.join(",", worldVerbRejectReasonByActor);
    }

    String worldVerbsByActor() {
        return String.join(",", worldVerbByActor);
    }

    String worldVerbGenerationsByActor() {
        return String.join(",", worldVerbGenerationByActor);
    }

    String worldVerbSemanticsByActor() {
        return String.join(",", worldVerbSemanticByActor);
    }

    private static String joinInts(int[] values) {
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < values.length; index++) {
            if (index > 0) result.append(',');
            result.append(values[index]);
        }
        return result.toString();
    }

    private static String joinBooleans(boolean[] values) {
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < values.length; index++) {
            if (index > 0) result.append(',');
            result.append(values[index] ? '1' : '0');
        }
        return result.toString();
    }

    private static boolean[] actorBits() {
        return new boolean[NativeGroupActionContract.actorCapacity()];
    }

    private static int[] actorInts(int fill) {
        int[] values = new int[
            NativeGroupActionContract.actorCapacity()
        ];
        java.util.Arrays.fill(values, fill);
        return values;
    }

    private static String[] actorStrings() {
        String[] values = new String[
            NativeGroupActionContract.actorCapacity()
        ];
        java.util.Arrays.fill(values, "");
        return values;
    }

    private static void putReason(
        String[] reasons,
        int actorId,
        String reason
    ) {
        if (
            reasons[actorId].isEmpty()
                && reason != null
                && !reason.isEmpty()
        ) {
            reasons[actorId] = reason;
        }
    }

    double minimumDeltaSeconds() {
        return executedTicks == 0 || !Double.isFinite(minimumDeltaSeconds)
            ? 0.0
            : minimumDeltaSeconds;
    }
}
