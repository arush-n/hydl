package com.hytalerlbridge.nativebackend.policy.combat;

import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.logger.HytaleLogger;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.ChargingInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.operation.Operation;
import com.hytalerlbridge.nativebackend.model.NativeInteractionBinding;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;

/**
 * Stateless discovery and caller-owned tracking for authored Guard forks.
 *
 * <p>The held Wielding interaction remains the engine lifecycle authority.
 * This helper reads its compiled packet/operation graph to discover which
 * input type reaches an ability; no weapon name or per-item branch is kept in
 * the bridge.</p>
 */
public final class NativeGuardFork {

    private NativeGuardFork() {}

    /** Return the authored fork input, or {@code null} for an ordinary root. */
    public static InteractionType resolveType(
        NativeInteractionBinding guard,
        NativeInteractionBinding ability
    ) {
        if (guard == null || guard.root() == null || ability == null
            || ability.id() == null || ability.id().isBlank()) {
            return null;
        }
        InteractionType resolved = resolveType(
            guard.root(),
            ability,
            new HashSet<>()
        );
        if (resolved != null) return resolved;
        HytaleLogger.getLogger().at(Level.WARNING).log(
            "Native Guard fork did not reach requested ability: %s",
            describeResolution(guard, ability)
        );
        return null;
    }

    private static InteractionType resolveType(
        RootInteraction root,
        NativeInteractionBinding ability,
        Set<String> visitedRootIds
    ) {
        if (root == null || !visitedRootIds.add(root.getId())) return null;
        for (int index = 0; index < root.getOperationMax(); index++) {
            Operation operation = root.getOperation(index);
            if (operation == null) continue;
            Object inner = operation.getInnerOperation();
            if (inner instanceof ChargingInteraction charging
                && charging.toPacket()
                    instanceof com.hypixel.hytale.protocol.ChargingInteraction packet
                && packet.forks != null) {
                for (Map.Entry<InteractionType, Integer> fork
                    : packet.forks.entrySet()) {
                    RootInteraction forkRoot =
                        RootInteraction.getAssetMap().getAsset(fork.getValue());
                    if (reachesAbility(forkRoot, ability)) {
                        return fork.getKey();
                    }
                }
            }
            if (inner instanceof Interaction interaction
                && interaction.toPacket()
                    instanceof com.hypixel.hytale.protocol.ReplaceInteraction replace) {
                InteractionType nested = resolveType(
                    RootInteraction.getAssetMap().getAsset(
                        replace.defaultValue
                    ),
                    ability,
                    visitedRootIds
                );
                if (nested != null) return nested;
            }
        }
        return null;
    }

    /** Live-only diagnostic for a fail-closed authored graph mismatch. */
    static String describeResolution(
        NativeInteractionBinding guard,
        NativeInteractionBinding ability
    ) {
        StringBuilder result = new StringBuilder(512);
        RootInteraction guardRoot = guard == null ? null : guard.root();
        RootInteraction abilityRoot = ability == null ? null : ability.root();
        result.append("guard=").append(guard == null ? null : guard.id())
            .append(" guardRoot=").append(rootId(guardRoot))
            .append(" ability=").append(ability == null ? null : ability.id())
            .append(" abilityType=").append(ability == null ? null : ability.type())
            .append(" abilityRoot=").append(rootId(abilityRoot))
            .append(" abilityIdIndex=").append(rootIndex(
                ability == null ? null : ability.id()
            ))
            .append(" abilityRootIndex=").append(rootIndex(rootId(abilityRoot)));
        appendOperations(result, " guardOp", guardRoot, new HashSet<>());
        return result.toString();
    }

    private static void appendOperations(
        StringBuilder result,
        String label,
        RootInteraction root,
        Set<String> visitedRootIds
    ) {
        if (root == null || !visitedRootIds.add(root.getId())) return;
        for (int index = 0; index < root.getOperationMax(); index++) {
            Operation operation = root.getOperation(index);
            Object inner = operation == null ? null : operation.getInnerOperation();
            result.append(' ').append(label).append('[').append(index)
                .append("]=").append(inner == null
                    ? null
                    : inner.getClass().getSimpleName());
            if (!(inner instanceof Interaction interaction)) continue;
            result.append(':').append(interaction.getId());
            Object packet = interaction.toPacket();
            if (packet instanceof com.hypixel.hytale.protocol.ReplaceInteraction replace) {
                RootInteraction replaceRoot =
                    RootInteraction.getAssetMap().getAsset(
                        replace.defaultValue
                    );
                result.append(" replace=").append(replace.defaultValue)
                    .append(':').append(rootId(replaceRoot));
                appendOperations(
                    result,
                    " replaceOp",
                    replaceRoot,
                    visitedRootIds
                );
            }
            if (packet instanceof com.hypixel.hytale.protocol.ChargingInteraction charging
                && charging.forks != null) {
                result.append(" forks=").append(charging.forks);
                for (Map.Entry<InteractionType, Integer> fork
                    : charging.forks.entrySet()) {
                    RootInteraction forkRoot =
                        RootInteraction.getAssetMap().getAsset(fork.getValue());
                    result.append(" forkRoot[").append(fork.getKey())
                        .append("]=").append(fork.getValue()).append(':')
                        .append(rootId(forkRoot));
                    appendOperations(
                        result,
                        " forkOp",
                        forkRoot,
                        visitedRootIds
                    );
                }
            }
        }
    }

    private static int rootIndex(String rootId) {
        return rootId == null
            ? Integer.MIN_VALUE
            : RootInteraction.getAssetMap().getIndex(rootId);
    }

    private static String rootId(RootInteraction root) {
        return root == null ? null : root.getId();
    }

    /** Snapshot the parent's current children before requesting a new fork. */
    public static Request capture(
        InteractionChain parent,
        NativeInteractionBinding ability,
        int abilitySlot,
        InteractionType forkType,
        double requestedChargeSeconds
    ) {
        if (parent == null || ability == null || forkType == null) {
            throw new IllegalArgumentException(
                "Guard fork parent, ability, and input type are required"
            );
        }
        return new Request(
            parent,
            ability,
            abilitySlot,
            forkType,
            new ArrayList<>(parent.getForkedChains().values()),
            requestedChargeSeconds
        );
    }

    /** Find only the child introduced after {@link #capture}. */
    public static InteractionChain findSpawned(Request request) {
        if (request == null) return null;
        for (InteractionChain child
            : request.parent().getForkedChains().values()) {
            if (!request.existingChildren().contains(child)
                && reachesAbility(
                    child.getRootInteraction(),
                    request.ability()
                )) {
                return child;
            }
        }
        return null;
    }

    public static boolean containsChild(
        InteractionChain parent,
        InteractionChain child
    ) {
        return parent != null && child != null
            && parent.getForkedChains().values().contains(child);
    }

    static boolean containsInteraction(
        RootInteraction root,
        String interactionId
    ) {
        int interactionRootIndex = interactionId == null
            ? Integer.MIN_VALUE
            : RootInteraction.getAssetMap().getIndex(interactionId);
        return containsInteraction(
            root,
            interactionId,
            interactionRootIndex,
            null,
            new HashSet<>()
        );
    }

    private static boolean reachesAbility(
        RootInteraction root,
        NativeInteractionBinding ability
    ) {
        if (root == null || ability == null) return false;
        RootInteraction abilityRoot = ability.root();
        if (abilityRoot != null && root.getId().equals(abilityRoot.getId())) {
            return true;
        }
        int abilityRootIndex = abilityRoot == null
            ? Integer.MIN_VALUE
            : RootInteraction.getAssetMap().getIndex(abilityRoot.getId());
        return containsInteraction(
            root,
            ability.id(),
            abilityRootIndex,
            abilityRoot,
            new HashSet<>()
        );
    }

    private static boolean containsInteraction(
        RootInteraction root,
        String interactionId,
        int interactionRootIndex,
        RootInteraction interactionRoot,
        Set<String> visitedRootIds
    ) {
        if (root == null || interactionId == null) return false;
        // ReplaceInteraction points at a RootInteraction. Reaching that root
        // is already a match; its operation ids describe the implementation
        // below the requested ability and need not repeat the root id.
        if (matchesRootId(root.getId(), interactionId)) return true;
        if (interactionRoot != null
            && sameCompiledProgram(root, interactionRoot)) return true;
        if (!visitedRootIds.add(root.getId())) return false;
        for (int index = 0; index < root.getOperationMax(); index++) {
            Operation operation = root.getOperation(index);
            if (operation == null) continue;
            Operation inner = operation.getInnerOperation();
            if (inner instanceof Interaction interaction
                && interactionId.equals(interaction.getId())) {
                return true;
            }
            if (inner instanceof Interaction interaction
                && interaction.toPacket()
                    instanceof com.hypixel.hytale.protocol.ReplaceInteraction replace
            ) {
                if (matchesRootIndex(
                    replace.defaultValue,
                    interactionRootIndex
                )) return true;
                if (containsInteraction(
                        RootInteraction.getAssetMap().getAsset(
                            replace.defaultValue
                        ),
                        interactionId,
                        interactionRootIndex,
                        interactionRoot,
                        visitedRootIds
                    )) {
                    return true;
                }
            }
        }
        return false;
    }

    static boolean matchesRootId(String rootId, String interactionId) {
        return interactionId != null && interactionId.equals(rootId);
    }

    static boolean matchesRootIndex(int rootIndex, int interactionRootIndex) {
        return interactionRootIndex != Integer.MIN_VALUE
            && rootIndex == interactionRootIndex;
    }

    /** Exact compiled operation identity, independent of wrapper root names. */
    static boolean sameCompiledProgram(
        RootInteraction candidate,
        RootInteraction requested
    ) {
        if (candidate == null || requested == null
            || candidate.getOperationMax() != requested.getOperationMax()) {
            return false;
        }
        for (int index = 0; index < candidate.getOperationMax(); index++) {
            Operation left = candidate.getOperation(index);
            Operation right = requested.getOperation(index);
            if (left == null || right == null) {
                if (left != right) return false;
                continue;
            }
            if (!left.getClass().equals(right.getClass())) return false;
            Object leftInner = left.getInnerOperation();
            Object rightInner = right.getInnerOperation();
            if (leftInner == null || rightInner == null) {
                if (leftInner != rightInner) return false;
                continue;
            }
            if (!leftInner.getClass().equals(rightInner.getClass())) {
                return false;
            }
            if (leftInner instanceof Interaction leftInteraction
                && rightInner instanceof Interaction rightInteraction
                && !leftInteraction.getId().equals(rightInteraction.getId())) {
                return false;
            }
        }
        return true;
    }

    /** Input-edge state only; every gameplay transition remains engine-owned. */
    public record Request(
        InteractionChain parent,
        NativeInteractionBinding ability,
        int abilitySlot,
        InteractionType forkType,
        List<InteractionChain> existingChildren,
        double requestedChargeSeconds
    ) {
        public Request {
            if (abilitySlot < 0) {
                throw new IllegalArgumentException(
                    "Guard fork ability slot must be non-negative"
                );
            }
            if (!NativeSyntheticCombatClient.validRequestedChargeSeconds(
                requestedChargeSeconds
            )) {
                throw new IllegalArgumentException(
                    "Guard fork charge duration must be a non-negative float"
                );
            }
            existingChildren = List.copyOf(existingChildren);
        }
    }
}

