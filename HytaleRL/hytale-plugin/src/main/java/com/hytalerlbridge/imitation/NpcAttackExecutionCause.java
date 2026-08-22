package com.hytalerlbridge.imitation;

/** Exact role-authored cause of a CombatSupport attack-chain start. */
public record NpcAttackExecutionCause(
    boolean available,
    boolean executed,
    int candidateIndex,
    String interactionType,
    String interactionId,
    String path
) {
    public static final String LAYOUT =
        "available:bool,executed:bool,candidate_index:i32,interaction_type:utf8,"
            + "interaction_id:utf8,path:utf8";

    public NpcAttackExecutionCause {
        interactionType = text(interactionType);
        interactionId = text(interactionId);
        path = text(path);
        if (!available || !executed) {
            if (executed || candidateIndex != -1 || !interactionType.isEmpty()
                || !interactionId.isEmpty() || !path.isEmpty()) {
                throw new IllegalArgumentException(
                    "absent attack execution cause must be zero-masked"
                );
            }
        } else if (
            candidateIndex < 0 || interactionType.isEmpty()
                || interactionId.isEmpty() || path.isEmpty()
        ) {
            throw new IllegalArgumentException(
                "executed attack cause requires exact candidate identity"
            );
        }
    }

    public static NpcAttackExecutionCause none() {
        return new NpcAttackExecutionCause(true, false, -1, "", "", "");
    }

    public static NpcAttackExecutionCause unavailable() {
        return new NpcAttackExecutionCause(false, false, -1, "", "", "");
    }

    private static String text(String value) {
        return value == null ? "" : value;
    }
}
