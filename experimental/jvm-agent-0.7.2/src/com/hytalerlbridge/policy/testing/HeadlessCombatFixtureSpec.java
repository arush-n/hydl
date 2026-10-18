package com.hytalerlbridge.policy.testing;

/** Operator-authored bounds for the explicit two-actor combat fixture. */
public record HeadlessCombatFixtureSpec(
    String agentRole,
    String targetRole,
    Origin origin,
    double x,
    double y,
    double z,
    double separation,
    int searchRadius
) {

    public enum Origin {
        /** Resolve through the current world's authored spawn provider. */
        SERVER_SPAWN,
        /** Use the operator-authored coordinates as the search origin. */
        EXPLICIT,
    }

    public HeadlessCombatFixtureSpec {
        if (agentRole == null || agentRole.isBlank()
            || targetRole == null || targetRole.isBlank() || origin == null) {
            throw new IllegalArgumentException(
                "fixture roles and origin must be non-empty");
        }
        if (!Double.isFinite(separation) || separation <= 0.0
            || separation > 16.0 || searchRadius < 0 || searchRadius > 64) {
            throw new IllegalArgumentException(
                "separation must be in (0, 16] and search radius in [0, 64]"
            );
        }
        if (origin == Origin.EXPLICIT
            && (!Double.isFinite(x) || !Double.isFinite(y)
                || !Double.isFinite(z))) {
            throw new IllegalArgumentException(
                "explicit fixture coordinates must be finite");
        }
        if (origin == Origin.SERVER_SPAWN) {
            x = Double.NaN;
            y = Double.NaN;
            z = Double.NaN;
        }
    }

    /**
     * Parse one of:
     *
     * <ul>
     *   <li>{@code AgentRole TargetRole spawn separation searchRadius}</li>
     *   <li>{@code AgentRole TargetRole exact x y z separation searchRadius}</li>
     *   <li>legacy {@code AgentRole TargetRole x y z separation}, retained as
     *       an exact zero-radius fail-closed fixture.</li>
     * </ul>
     */
    public static HeadlessCombatFixtureSpec parse(String text) {
        if (text == null) {
            throw new IllegalArgumentException("fixture text is missing");
        }
        String[] fields = text.trim().split("\\s+");
        try {
            if (fields.length == 5 && fields[2].equalsIgnoreCase("spawn")) {
                return new HeadlessCombatFixtureSpec(
                    fields[0],
                    fields[1],
                    Origin.SERVER_SPAWN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.parseDouble(fields[3]),
                    Integer.parseInt(fields[4])
                );
            }
            if (fields.length == 8 && fields[2].equalsIgnoreCase("exact")) {
                return new HeadlessCombatFixtureSpec(
                    fields[0],
                    fields[1],
                    Origin.EXPLICIT,
                    Double.parseDouble(fields[3]),
                    Double.parseDouble(fields[4]),
                    Double.parseDouble(fields[5]),
                    Double.parseDouble(fields[6]),
                    Integer.parseInt(fields[7])
                );
            }
            if (fields.length == 6) {
                return new HeadlessCombatFixtureSpec(
                    fields[0],
                    fields[1],
                    Origin.EXPLICIT,
                    Double.parseDouble(fields[2]),
                    Double.parseDouble(fields[3]),
                    Double.parseDouble(fields[4]),
                    Double.parseDouble(fields[5]),
                    0
                );
            }
        } catch (NumberFormatException invalid) {
            throw new IllegalArgumentException(
                "combat fixture numeric fields are malformed", invalid);
        }
        throw new IllegalArgumentException(
            "combat-fixture.txt must be 'AgentRole TargetRole spawn "
                + "separation searchRadius' or 'AgentRole TargetRole exact "
                + "x y z separation searchRadius'"
        );
    }
}
