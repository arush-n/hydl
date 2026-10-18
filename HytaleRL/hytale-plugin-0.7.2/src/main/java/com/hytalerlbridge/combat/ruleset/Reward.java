package com.hytalerlbridge.combat.ruleset;

/** Extracted verbatim from {@code CombatRuleset}. */
public record Reward(
    double targetDamageScale,
    double agentDamageScale,
    double completion,
    double death
) {}
