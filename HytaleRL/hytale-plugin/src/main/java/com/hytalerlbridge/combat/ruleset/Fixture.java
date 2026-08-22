package com.hytalerlbridge.combat.ruleset;

import static com.hytalerlbridge.combat.ruleset.RulesetChecks.checkedVector;

/** Extracted verbatim from {@code CombatRuleset}. */
public record Fixture(double[] agentSpawn, double[] targetOffset, double floorY) {
    public Fixture {
        agentSpawn = checkedVector(agentSpawn, "agentSpawn");
        targetOffset = checkedVector(targetOffset, "targetOffset");
    }

    @Override
    public double[] agentSpawn() {
        return agentSpawn.clone();
    }

    @Override
    public double[] targetOffset() {
        return targetOffset.clone();
    }
}
