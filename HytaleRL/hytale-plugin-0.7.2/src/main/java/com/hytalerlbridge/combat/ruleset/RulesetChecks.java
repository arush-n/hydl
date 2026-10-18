package com.hytalerlbridge.combat.ruleset;

/**
 * Validators shared by every ruleset value type.
 *
 * <p>Lifted verbatim out of {@code CombatRuleset}. The nested records
 * reached these as enclosing-class privates; as separate files they need a
 * package-visible home, and {@code CombatRuleset.load} still uses them too.
 */
public final class RulesetChecks {

    private RulesetChecks() {}

    public static double[] checkedVector(double[] values, String name) {
        if (values == null || values.length != 3) {
            throw new IllegalArgumentException(name + " must contain three values");
        }
        double[] copy = values.clone();
        for (double value : copy) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(name + " must be finite");
            }
        }
        return copy;
    }

    public static double[] checkedBounds(double[] values, String name) {
        if (values == null || values.length != 6) {
            throw new IllegalArgumentException(name + " must contain six values");
        }
        double[] copy = values.clone();
        for (double value : copy) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(name + " must be finite");
            }
        }
        for (int axis = 0; axis < 3; axis++) {
            if (copy[axis + 3] <= copy[axis]) {
                throw new IllegalArgumentException(
                    name + " maximum must exceed minimum on every axis"
                );
            }
        }
        return copy;
    }

    public static void require(boolean condition, String name) {
        if (!condition) throw new IllegalStateException("Invalid " + name);
    }
}
