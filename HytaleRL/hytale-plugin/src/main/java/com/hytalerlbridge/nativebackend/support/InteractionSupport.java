package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.assetstore.AssetExtraInfo;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.InteractionState;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.modules.interaction.InteractionModule;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.operation.Operation;
import com.hypixel.hytale.server.core.asset.type.item.config.Item;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.corecomponents.combat.ActionAttack;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.support.CombatSupport;
import com.hypixel.hytale.server.npc.util.IAnnotatedComponent;
import com.hypixel.hytale.server.npc.util.IAnnotatedComponentCollection;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import com.hytalerlbridge.nativebackend.model.NativeInteractionBinding;

/**
 * Extracted verbatim from {@code NativeEnvironmentSession}.
 *
 * <p>Every method here touches no session instance field, so the move
 * required no state surgery.
 */
public final class InteractionSupport {

    private InteractionSupport() {}

    public static final String NATIVE_COMBAT_PROBE_ASSET_PACK =
        "HytaleRL:HytaleRLBridge";
    public static final String NATIVE_COMBAT_PROBE_ROOT_PREFIX =
        "HytaleRL_ArsenalProbe_";
    public static final String NATIVE_WORLD_PROBE_ROOT_PREFIX =
        "HytaleRL_WorldProbe_";

    public static NativeInteractionBinding resolveNativeInteractionBinding(
        String interactionId,
        String interactionType
    ) {
        RootInteraction root = RootInteraction.getAssetMap().getAsset(
            interactionId
        );
        if (root == null) {
            if (Interaction.getAssetMap().getAsset(interactionId) == null) {
                throw new IllegalArgumentException(
                    "Native interaction asset not found: " + interactionId
                );
            }
            root = registeredCombatProbeRoot(interactionId);
        }
        return new NativeInteractionBinding(
            interactionId,
            InteractionType.valueOf(interactionType),
            root
        );
    }

    /**
     * Resolve an exact ability through its authored item root when doing so
     * cannot change which child executes.
     *
     * <p>Item-level roots own cooldowns, charge banks, rules, and operations.
     * Executing a synthetic root around the child bypasses all of them.  A
     * single-child item root is unambiguous; multi-child roots stay on the
     * explicitly requested child because selecting a policy slot by guessing
     * the server selector branch would be incorrect.
     */
    public static NativeInteractionBinding resolveNativeAbilityBinding(
        String itemId,
        String interactionId,
        String interactionType
    ) {
        return resolveNativeAbilityBinding(
            itemId,
            interactionId,
            InteractionType.valueOf(interactionType),
            false
        );
    }

    /**
     * Resolve a complete compact ability row against one equipped item.
     *
     * <p>When exactly one policy-selectable ability owns an authored
     * {@link InteractionType}, that ability can be routed through the
     * equipped item's root without ambiguity. This preserves outer inventory
     * transactions, cooldowns, charge banks, rules, and failure branches.
     * Types with multiple selectable children retain explicit child binding;
     * the caller-selected slot is the only authority for that choice.</p>
     */
    public static NativeInteractionBinding[] resolveNativeAbilityBindings(
        String itemId,
        String[] interactionIds,
        String[] interactionTypes
    ) {
        if (interactionIds == null || interactionTypes == null) {
            throw new IllegalArgumentException(
                "Native ability interaction rows must be present"
            );
        }
        if (interactionIds.length != interactionTypes.length) {
            throw new IllegalArgumentException(
                "Native ability interaction rows must have equal widths"
            );
        }

        InteractionType[] types = new InteractionType[interactionTypes.length];
        Map<InteractionType, Integer> typeCounts = new EnumMap<>(
            InteractionType.class
        );
        for (int index = 0; index < interactionTypes.length; index++) {
            InteractionType type = InteractionType.valueOf(
                interactionTypes[index]
            );
            types[index] = type;
            typeCounts.merge(type, 1, Integer::sum);
        }

        NativeInteractionBinding[] bindings =
            new NativeInteractionBinding[interactionIds.length];
        for (int index = 0; index < interactionIds.length; index++) {
            bindings[index] = resolveNativeAbilityBinding(
                itemId,
                interactionIds[index],
                types[index],
                typeCounts.get(types[index]) == 1
            );
        }
        return bindings;
    }

    private static NativeInteractionBinding resolveNativeAbilityBinding(
        String itemId,
        String interactionId,
        InteractionType type,
        boolean soleBindingForType
    ) {
        NativeInteractionBinding direct = resolveNativeInteractionBinding(
            interactionId,
            type.name()
        );
        Item item = Item.getAssetMap().getAsset(itemId);
        if (item == null) {
            throw new IllegalArgumentException(
                "Native combat item asset not found: " + itemId
            );
        }
        String itemRootId = item.getInteractions().get(type);
        if (itemRootId == null) return direct;

        RootInteraction itemRoot = RootInteraction.getAssetMap().getAsset(
            itemRootId
        );
        if (itemRoot == null) return direct;
        String[] children = itemRoot.getInteractionIds();
        if (soleBindingForType) {
            Set<RootInteraction> visited = Collections.newSetFromMap(
                new IdentityHashMap<>()
            );
            if (!rootReachesInteraction(itemRoot, interactionId, visited)) {
                return direct;
            }
        } else if (
            children.length != 1 || !interactionId.equals(children[0])
        ) {
            return direct;
        }
        return new NativeInteractionBinding(
            interactionId,
            type,
            itemRoot
        );
    }

    private static boolean rootReachesInteraction(
        RootInteraction root,
        String interactionId,
        Set<RootInteraction> visited
    ) {
        if (root == null || !visited.add(root)) return false;
        if (interactionId.equals(root.getId())) return true;
        for (String childId : root.getInteractionIds()) {
            if (interactionId.equals(childId)) return true;
            RootInteraction child = RootInteraction.getAssetMap().getAsset(
                childId
            );
            if (rootReachesInteraction(child, interactionId, visited)) {
                return true;
            }
        }
        int requestedRootIndex = RootInteraction.getAssetMap().getIndex(
            interactionId
        );
        for (int index = 0; index < root.getOperationMax(); index++) {
            Operation operation = root.getOperation(index);
            if (operation == null) continue;
            Operation inner = operation.getInnerOperation();
            if (!(inner instanceof Interaction interaction)) continue;
            if (interactionInheritsFrom(interaction, interactionId)) {
                return true;
            }
            if (
                interaction.toPacket()
                    instanceof com.hypixel.hytale.protocol.ReplaceInteraction replace
            ) {
                if (
                    requestedRootIndex != Integer.MIN_VALUE
                        && replace.defaultValue == requestedRootIndex
                ) {
                    return true;
                }
                if (
                    rootReachesInteraction(
                        RootInteraction.getAssetMap().getAsset(
                            replace.defaultValue
                        ),
                        interactionId,
                        visited
                    )
                ) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean interactionInheritsFrom(
        Interaction interaction,
        String requestedId
    ) {
        Set<String> visitedParentIds = new HashSet<>();
        Interaction current = interaction;
        while (current != null) {
            if (requestedId.equals(current.getId())) return true;
            Object parentKey = interactionParentKey(current);
            if (
                !(parentKey instanceof String parentId)
                    || parentId.isBlank()
                    || !visitedParentIds.add(parentId)
            ) {
                return false;
            }
            if (requestedId.equals(parentId)) return true;
            current = Interaction.getAssetMap().getAsset(parentId);
        }
        return false;
    }

    private static Object interactionParentKey(Interaction interaction) {
        try {
            var field = Interaction.class.getDeclaredField("data");
            field.setAccessible(true);
            Object raw = field.get(interaction);
            return raw instanceof AssetExtraInfo.Data data
                ? data.getParentKey()
                : null;
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException(
                "Pinned Interaction parent metadata is unavailable",
                error
            );
        }
    }

    public static synchronized RootInteraction registeredCombatProbeRoot(
        String interactionId
    ) {
        String rootId = NATIVE_COMBAT_PROBE_ROOT_PREFIX + interactionId;
        RootInteraction existing = RootInteraction.getAssetMap().getAsset(
            rootId
        );
        if (existing != null) return existing;

        var result = RootInteraction.getAssetStore().loadAssets(
            NATIVE_COMBAT_PROBE_ASSET_PACK,
            List.of(new RootInteraction(rootId, interactionId))
        );
        RootInteraction loaded = RootInteraction.getAssetMap().getAsset(rootId);
        if (result.hasFailed() || loaded == null) {
            throw new IllegalStateException(
                "Failed to register native combat probe root: " + rootId
            );
        }
        return loaded;
    }

    public static synchronized RootInteraction registeredWorldProbeRoot(
        String interactionId
    ) {
        if (Interaction.getAssetMap().getAsset(interactionId) == null) {
            throw new IllegalStateException(
                "Synthetic World interaction asset is missing: "
                    + interactionId
            );
        }
        String rootId = NATIVE_WORLD_PROBE_ROOT_PREFIX + interactionId;
        RootInteraction existing = RootInteraction.getAssetMap().getAsset(
            rootId
        );
        if (existing != null) return existing;

        var result = RootInteraction.getAssetStore().loadAssets(
            NATIVE_COMBAT_PROBE_ASSET_PACK,
            List.of(new RootInteraction(rootId, interactionId))
        );
        RootInteraction loaded = RootInteraction.getAssetMap().getAsset(rootId);
        if (result.hasFailed() || loaded == null) {
            throw new IllegalStateException(
                "Failed to register synthetic World probe root: " + rootId
            );
        }
        return loaded;
    }

    public static RootInteraction resolvedWorldProbeRoot(
        String rootOrInteractionId
    ) {
        RootInteraction root = RootInteraction.getAssetMap().getAsset(
            rootOrInteractionId
        );
        return root == null
            ? registeredWorldProbeRoot(rootOrInteractionId)
            : root;
    }

    public static void collectAttackActions(
        IAnnotatedComponent component,
        List<ActionAttack> attacks,
        java.util.Set<IAnnotatedComponent> visited
    ) {
        if (component == null || !visited.add(component)) return;
        if (component instanceof ActionAttack attack) attacks.add(attack);
        if (component instanceof IAnnotatedComponentCollection collection) {
            for (int i = 0; i < collection.componentCount(); i++) {
                collectAttackActions(collection.getComponent(i), attacks, visited);
            }
        }
    }

    public static List<ActionAttack> discoverAttackActions(Role role) {
        if (role == null) return List.of();
        List<ActionAttack> attacks = new ArrayList<>();
        var visited = Collections.newSetFromMap(
            new IdentityHashMap<IAnnotatedComponent, Boolean>()
        );
        collectAttackActions(role, attacks, visited);
        return List.copyOf(attacks);
    }

    public static InteractionManager interactionManager(
        Ref<EntityStore> reference,
        Store<EntityStore> store
    ) {
        InteractionModule module = InteractionModule.get();
        return module == null
            ? null
            : store.getComponent(reference, module.getInteractionManagerComponent());
    }

    public static void clearNativeCombat(
        InteractionManager manager,
        CombatSupport combat
    ) {
        manager.clear();
        combat.setExecutingAttack(null, false, 0.0);
    }

    /**
     * Return whether an engine-owned interaction tree still has server or
     * client work in flight.
     *
     * <p>A parent root may finish immediately after forking predicted child
     * chains (for example a generic {@code ParallelInteraction}). NPC
     * {@link CombatSupport} then drops its parent-only active-attack pointer,
     * while the children remain authoritative members of the manager tree.
     * Clearing the manager at that boundary cancels valid child work.</p>
     */
    public static boolean hasUnfinishedInteractionTree(
        InteractionManager manager
    ) {
        if (manager == null) return false;
        for (InteractionChain chain : manager.getChains().values()) {
            if (hasUnfinishedInteractionTree(chain)) return true;
        }
        return false;
    }

    static boolean hasUnfinishedInteractionTree(InteractionChain chain) {
        if (chain == null) return false;
        return anyTreeNodeMatches(
            chain,
            node -> interactionStatesUnfinished(
                node.getServerState(),
                node.getClientState()
            ),
            node -> node.getForkedChains().values()
        );
    }

    static boolean interactionStatesUnfinished(
        InteractionState server,
        InteractionState client
    ) {
        return server == InteractionState.NotFinished
            || client == InteractionState.NotFinished;
    }

    static <T> boolean anyTreeNodeMatches(
        T node,
        Predicate<T> matches,
        Function<T, ? extends Iterable<T>> children
    ) {
        if (matches.test(node)) return true;
        for (T child : children.apply(node)) {
            if (anyTreeNodeMatches(child, matches, children)) return true;
        }
        return false;
    }

    public static void clearMarkedTargets(Role role) {
        var marked = role.getMarkedEntitySupport();
        if (marked == null) return;
        for (int slot = 0; slot < marked.getMarkedEntitySlotCount(); slot++) {
            marked.clearMarkedEntity(slot);
        }
    }
}
