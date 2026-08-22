package com.hytalerlbridge.combat.ruleset;

/** Extracted verbatim from {@code CombatRuleset}. */
public record ObservationNormalization(
    double wireFixedPointScale,
    double verticalSpeedScale,
    double facingErrorDegreesScale,
    double headPitchDegreesScale
) {}
