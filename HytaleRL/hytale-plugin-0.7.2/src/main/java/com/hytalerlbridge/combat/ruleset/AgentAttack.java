package com.hytalerlbridge.combat.ruleset;

/** Extracted verbatim from {@code CombatRuleset}. */
public record AgentAttack(
    int hitDelayTicks,
    double range,
    double halfAngleDegrees
) {}
