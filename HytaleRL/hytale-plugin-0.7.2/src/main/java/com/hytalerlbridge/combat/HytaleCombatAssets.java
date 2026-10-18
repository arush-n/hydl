package com.hytalerlbridge.combat;

import java.util.List;
import com.hytalerlbridge.combat.ruleset.TargetAttack;

/** Typed compatibility facade over the shared versioned combat ruleset. */
public final class HytaleCombatAssets {

    public static final CombatRuleset RULESET = CombatRuleset.defaultRules();
    public static final int TICKS_PER_SECOND =
        RULESET.engine().ticksPerSecond();
    public static final double TRORK_BRAWLER_TURN_DEGREES_PER_SECOND =
        RULESET.target().turnDegreesPerSecond();
    public static final double TARGET_ATTACK_PAUSE_MIN_SECONDS =
        RULESET.target().attackPauseMinSeconds();
    public static final double TARGET_ATTACK_PAUSE_MAX_SECONDS =
        RULESET.target().attackPauseMaxSeconds();
    public static final int NPC_HEALTH_REGEN_DELAY_TICKS =
        RULESET.regeneration().delayTicks();
    public static final int NPC_HEALTH_REGEN_INTERVAL_TICKS =
        RULESET.regeneration().intervalTicks();
    public static final double NPC_HEALTH_REGEN_FRACTION =
        RULESET.regeneration().fraction();

    public static final List<MeleeAttackProfile> TRORK_BRAWLER_ATTACKS =
        RULESET.target().attacks().stream()
            .map(TargetAttack::profile)
            .toList();

    private HytaleCombatAssets() {}

    public static int brawlerAttackIndex(String rootInteractionId) {
        if (rootInteractionId == null || rootInteractionId.isBlank()) return -1;
        for (int index = 0; index < TRORK_BRAWLER_ATTACKS.size(); index++) {
            String expected = TRORK_BRAWLER_ATTACKS.get(index).interactionId();
            if (rootInteractionId.equals(expected) || rootInteractionId.endsWith(expected)) {
                return index;
            }
        }
        return -1;
    }
}
