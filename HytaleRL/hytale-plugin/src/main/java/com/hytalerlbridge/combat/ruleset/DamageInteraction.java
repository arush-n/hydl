package com.hytalerlbridge.combat.ruleset;

import static com.hytalerlbridge.combat.ruleset.RulesetChecks.require;

/** Extracted verbatim from {@code CombatRuleset}. */
public record DamageInteraction(
    String interactionId,
    DirectionalKnockback knockback
) {
    public DamageInteraction {
        require(
            "NPC_Attack_Melee_Damage".equals(interactionId),
            "damage interaction"
        );
        require(knockback != null, "damage interaction knockback");
    }
}
