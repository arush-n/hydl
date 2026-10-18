package com.hytalerlbridge.policy.diagnostics.action.dodge;

/** Bridge-published Dodge cooldown metadata used by the live diagnostic. */
public record DodgeCooldownContract(
    String cooldownId,
    float cooldownSeconds,
    int ticksPerSecond
) {
    public DodgeCooldownContract {
        cooldownId = cooldownId == null ? "" : cooldownId.strip();
        if (cooldownId.isEmpty()) {
            throw new IllegalArgumentException("Dodge cooldown id is empty");
        }
        if (!Float.isFinite(cooldownSeconds) || cooldownSeconds <= 0.0f) {
            throw new IllegalArgumentException(
                "Dodge cooldown seconds must be finite and positive");
        }
        if (ticksPerSecond < 1) {
            throw new IllegalArgumentException(
                "ticks per second must be positive");
        }
    }

    /** Duration boundaries round up; continuous rates do not use this rule. */
    public int boundaryTicks() {
        return (int) Math.ceil(cooldownSeconds * ticksPerSecond);
    }
}
