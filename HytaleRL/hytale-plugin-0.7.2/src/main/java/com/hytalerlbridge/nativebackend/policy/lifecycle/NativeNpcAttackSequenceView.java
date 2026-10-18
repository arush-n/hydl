package com.hytalerlbridge.nativebackend.policy.lifecycle;

import com.hypixel.hytale.server.npc.corecomponents.combat.ActionAttack;
import com.hypixel.hytale.server.npc.instructions.ActionList;
import com.hypixel.hytale.server.npc.instructions.Instruction;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.util.IAnnotatedComponent;
import com.hypixel.hytale.server.npc.util.IAnnotatedComponentCollection;
import com.hytalerlbridge.combat.HytaleCombatAssets;
import java.lang.reflect.Field;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

/**
 * Read-only view of the engine-owned NPC attack-sequence cursor.
 *
 * <p>The interaction manager removes a completed chain before the authored
 * one-to-two-second Brawler attack pause ends. The role's blocking
 * {@link ActionList}, however, advances its cursor immediately after launching
 * that chain and keeps the cursor for the next action. The previous cursor
 * position therefore names the completed root throughout the pause. This view
 * traverses the current role tree on every call and caches only immutable
 * reflection metadata; it retains no actor or previous-tick state.</p>
 */
public final class NativeNpcAttackSequenceView {
    private static final Field INSTRUCTION_ACTIONS = field(
        Instruction.class,
        "actions"
    );
    private static final Field ACTION_INDEX = field(
        ActionList.class,
        "actionIndex"
    );
    private static final Field CONFIGURED_ATTACK = field(
        ActionAttack.class,
        "attack"
    );
    private static final Field RESOLVED_ATTACK = field(
        ActionAttack.class,
        "attackInteraction"
    );

    private NativeNpcAttackSequenceView() {}

    public record State(
        boolean available,
        String reason,
        int attackIndex,
        int cursor,
        int actionCount
    ) {
        public State {
            reason = reason == null ? "" : reason;
            if (available != reason.isBlank()) {
                throw new IllegalArgumentException(
                    "exactly one of available and reason is required"
                );
            }
            if (available && (attackIndex < 0 || attackIndex >= 5)) {
                throw new IllegalArgumentException("attackIndex is out of range");
            }
            if (
                available
                    && (actionCount != 5 || cursor < 0 || cursor > actionCount)
            ) {
                throw new IllegalArgumentException(
                    "attack sequence cursor is out of range"
                );
            }
            if (
                !available
                    && (attackIndex != -1 || cursor != -1 || actionCount != 0)
            ) {
                throw new IllegalArgumentException(
                    "unavailable attack sequence must be structurally empty"
                );
            }
        }

        private static State unavailable(String reason) {
            return new State(false, reason, -1, -1, 0);
        }
    }

    /** Source-layout gate used before a bridge candidate is deployed. */
    public static boolean layoutAvailable() {
        return INSTRUCTION_ACTIONS != null
            && ACTION_INDEX != null
            && CONFIGURED_ATTACK != null
            && RESOLVED_ATTACK != null;
    }

    /** Resolve the last launched Trork Brawler root from the current role. */
    public static State capture(Role role) {
        if (role == null || !layoutAvailable()) {
            return State.unavailable("target_attack_sequence_layout_unavailable");
        }
        try {
            ArrayDeque<IAnnotatedComponent> pending = new ArrayDeque<>();
            Set<IAnnotatedComponent> visited = Collections.newSetFromMap(
                new IdentityHashMap<>()
            );
            pending.add(role);
            State found = null;
            while (!pending.isEmpty()) {
                IAnnotatedComponent component = pending.removeFirst();
                if (component == null || !visited.add(component)) continue;
                if (component instanceof Instruction instruction) {
                    State candidate = brawlerSequence(instruction);
                    if (candidate != null) {
                        if (!candidate.available()) return candidate;
                        if (found != null) {
                            return State.unavailable(
                                "target_attack_sequence_ambiguous"
                            );
                        }
                        found = candidate;
                    }
                }
                if (component instanceof IAnnotatedComponentCollection group) {
                    int count = group.componentCount();
                    if (count < 0 || count > 16_384) {
                        return State.unavailable(
                            "target_attack_sequence_tree_invalid"
                        );
                    }
                    for (int index = 0; index < count; index++) {
                        IAnnotatedComponent child = group.getComponent(index);
                        if (child != null) pending.addLast(child);
                    }
                }
            }
            return found == null
                ? State.unavailable("target_attack_sequence_unavailable")
                : found;
        } catch (IllegalAccessException | RuntimeException unavailable) {
            return State.unavailable("target_attack_sequence_read_failed");
        }
    }

    private static State brawlerSequence(Instruction instruction)
        throws IllegalAccessException {
        Object raw = INSTRUCTION_ACTIONS.get(instruction);
        if (!(raw instanceof ActionList actions)) return null;
        int count = actions.actionCount();
        int expected = HytaleCombatAssets.TRORK_BRAWLER_ATTACKS.size();
        if (count != expected) return null;

        ActionAttack[] sequence = new ActionAttack[count];
        boolean[] seen = new boolean[count];
        for (int index = 0; index < count; index++) {
            if (!(actions.getComponent(index) instanceof ActionAttack attack)) {
                return null;
            }
            Object configured = CONFIGURED_ATTACK.get(attack);
            if (!(configured instanceof String id)) return null;
            int attackIndex = HytaleCombatAssets.brawlerAttackIndex(id);
            if (attackIndex < 0 || attackIndex >= count || seen[attackIndex]) {
                return null;
            }
            seen[attackIndex] = true;
            sequence[index] = attack;
        }

        int cursor = ACTION_INDEX.getInt(actions);
        if (cursor < 0 || cursor > count) {
            return State.unavailable("target_attack_sequence_cursor_invalid");
        }
        Object resolved = RESOLVED_ATTACK.get(
            sequence[previousIndex(cursor, count)]
        );
        if (!(resolved instanceof String id)) {
            return State.unavailable("target_attack_sequence_root_unavailable");
        }
        int attackIndex = HytaleCombatAssets.brawlerAttackIndex(id);
        return attackIndex < 0
            ? State.unavailable("target_attack_sequence_root_unknown")
            : new State(true, "", attackIndex, cursor, count);
    }

    /** Package-visible arithmetic gate for the five-to-zero wrap boundary. */
    static int previousIndex(int cursor, int count) {
        if (count <= 0 || cursor < 0 || cursor > count) {
            throw new IllegalArgumentException("invalid action sequence cursor");
        }
        return Math.floorMod(cursor - 1, count);
    }

    private static Field field(Class<?> owner, String name) {
        try {
            Field value = owner.getDeclaredField(name);
            return value.trySetAccessible() ? value : null;
        } catch (NoSuchFieldException | RuntimeException unavailable) {
            return null;
        }
    }
}

