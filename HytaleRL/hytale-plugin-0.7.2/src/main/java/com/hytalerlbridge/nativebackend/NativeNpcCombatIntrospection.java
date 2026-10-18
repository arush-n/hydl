package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.npc.corecomponents.ActionBase;
import com.hypixel.hytale.server.npc.corecomponents.combat.ActionAttack;
import com.hypixel.hytale.server.npc.role.support.CombatSupport;
import com.hytalerlbridge.imitation.NpcAttackActionSnapshot;
import java.lang.reflect.Field;

/** Pinned 0.5.7 fields that have no public lifecycle accessors. */
final class NativeNpcCombatIntrospection {

    private static final Field ACTIVE_ATTACK = field(CombatSupport.class, "activeAttack");
    private static final Field ATTACK_PAUSE = field(CombatSupport.class, "attackPause");
    private static final Field ATTACK_READY = field(ActionAttack.class, "attackReady");
    private static final Field AIMING_REMAINING = field(
        ActionAttack.class,
        "aimingTimeRemaining"
    );
    private static final Field CHARGE_FOR = field(ActionAttack.class, "chargeFor");
    private static final Field INTERACTION_TYPE = field(
        ActionAttack.class,
        "interactionType"
    );
    private static final Field INTERACTION_ID = field(
        ActionAttack.class,
        "attackInteraction"
    );

    private NativeNpcCombatIntrospection() {}

    static InteractionChain activeAttack(CombatSupport support) {
        return support == null ? null : value(ACTIVE_ATTACK, support, InteractionChain.class);
    }

    static float attackPauseSeconds(CombatSupport support) {
        return support == null
            ? 0.0f
            : (float) Math.max(0.0, number(ATTACK_PAUSE, support).doubleValue());
    }

    static NpcAttackActionSnapshot attack(ActionAttack attack) {
        Object type = value(INTERACTION_TYPE, attack, Object.class);
        return new NpcAttackActionSnapshot(
            attack.getLabel(),
            attack.isActivated(),
            ((ActionBase) attack).isTriggered(),
            bool(ATTACK_READY, attack),
            number(AIMING_REMAINING, attack).floatValue(),
            number(CHARGE_FOR, attack).floatValue(),
            type instanceof Enum<?> enumeration
                ? enumeration.name()
                : (type == null ? "" : type.toString()),
            value(INTERACTION_ID, attack, String.class),
            attack.getBreadCrumbs()
        );
    }

    private static Field field(Class<?> owner, String name) {
        try {
            Field field = owner.getDeclaredField(name);
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException exception) {
            throw new ExceptionInInitializerError(exception);
        }
    }

    private static boolean bool(Field field, Object owner) {
        try {
            return field.getBoolean(owner);
        } catch (IllegalAccessException exception) {
            throw inaccessible(field, exception);
        }
    }

    private static Number number(Field field, Object owner) {
        try {
            return (Number) field.get(owner);
        } catch (IllegalAccessException exception) {
            throw inaccessible(field, exception);
        }
    }

    private static <T> T value(Field field, Object owner, Class<T> type) {
        try {
            Object value = field.get(owner);
            return value == null ? null : type.cast(value);
        } catch (IllegalAccessException exception) {
            throw inaccessible(field, exception);
        }
    }

    private static IllegalStateException inaccessible(
        Field field,
        IllegalAccessException cause
    ) {
        return new IllegalStateException(
            "Pinned native combat field is inaccessible: " + field,
            cause
        );
    }
}
