package com.hytalerlbridge.nativebackend.support;

/** Stateless coordinate rules for native combat-fixture spawns. */
public final class CombatSpawnSupport {
    private CombatSpawnSupport() {}

    /**
     * Place a flat-fixture target relative to the agent's settled feet.
     *
     * <p>The requested agent spawn may be above the floor and is consumed
     * before the warm-up ticks settle the NPC. Reusing that requested Y for
     * a later target spawn starts the target airborne and exposes hidden
     * Walk-controller fall state in the reset observation. The observation
     * captured after warm-up is the authoritative settled coordinate.
     */
    public static double flatTargetY(
        double settledAgentY,
        double authoredTargetOffsetY
    ) {
        if (
            !Double.isFinite(settledAgentY)
                || !Double.isFinite(authoredTargetOffsetY)
        ) {
            throw new IllegalArgumentException(
                "Flat combat spawn coordinates must be finite"
            );
        }
        return settledAgentY + authoredTargetOffsetY;
    }
}
