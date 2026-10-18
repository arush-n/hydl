package com.hytalerlbridge.nativebackend.worldactions.execution;

import com.hytalerlbridge.action.NativeWorldVerbRequest;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** Deterministic actor-major arbitration for simultaneous World verbs. */
public final class PolicyWorldVerbArbitrator {

    public static final String SAME_TARGET_REJECTION =
        "native_world_verb_same_target_conflict_lost";

    private PolicyWorldVerbArbitrator() {}

    /**
     * Attempt actor-local admission in ascending actor-slot order. A spatial
     * target is reserved only after its earlier actor was actually admitted;
     * a stale, busy, or otherwise rejected actor cannot suppress the next
     * eligible actor. Input order is therefore irrelevant, as is ECS chunk
     * iteration order.
     */
    public static Map<Integer, Decision> arbitrate(
        List<Entry> entries,
        AdmissionAttempt attempt
    ) {
        if (entries == null || attempt == null) {
            throw new IllegalArgumentException(
                "World-verb entries and admission attempt are required"
            );
        }
        ArrayList<Entry> ordered = new ArrayList<>(entries);
        HashSet<Integer> slots = new HashSet<>();
        HashSet<String> identities = new HashSet<>();
        for (Entry entry : ordered) requireEntry(entry, slots, identities);
        ordered.sort(Comparator.comparingInt(Entry::actorSlot));

        LinkedHashMap<Integer, Decision> result = new LinkedHashMap<>();
        HashSet<Target> reserved = new HashSet<>();
        for (Entry entry : ordered) {
            if (!entry.request().candidateEvidenceBound()) {
                result.put(
                    entry.actorSlot(),
                    new Decision(
                        false,
                        NativeWorldVerbRequest.POLICY_EVIDENCE_REQUIRED
                    )
                );
                continue;
            }
            Target target = spatialTarget(entry.request());
            if (target != null && reserved.contains(target)) {
                result.put(
                    entry.actorSlot(),
                    new Decision(false, SAME_TARGET_REJECTION)
                );
                continue;
            }
            Admission admission = attempt.attempt(entry);
            if (admission == null) {
                throw new IllegalStateException(
                    "World-verb admission attempt returned no result"
                );
            }
            if (!admission.admitted()) {
                if (
                    admission.rejectReason() == null
                        || admission.rejectReason().isBlank()
                ) {
                    throw new IllegalStateException(
                        "Rejected World verb has no typed reason"
                    );
                }
                result.put(
                    entry.actorSlot(),
                    new Decision(false, admission.rejectReason())
                );
                continue;
            }
            if (
                admission.rejectReason() != null
                    && !admission.rejectReason().isEmpty()
            ) {
                throw new IllegalStateException(
                    "Admitted World verb carries a rejection reason"
                );
            }
            result.put(entry.actorSlot(), new Decision(true, ""));
            if (target != null) reserved.add(target);
        }
        return Map.copyOf(result);
    }

    private static Target spatialTarget(NativeWorldVerbRequest request) {
        return switch (request.verb()) {
            case "use", "break_block", "place_block" -> new Target(
                request.targetX(), request.targetY(), request.targetZ()
            );
            default -> null;
        };
    }

    private static void requireEntry(
        Entry entry,
        HashSet<Integer> slots,
        HashSet<String> identities
    ) {
        if (entry == null || entry.actorSlot() < 0) {
            throw new IllegalArgumentException(
                "World-verb arbitration entry is incomplete"
            );
        }
        String canonical;
        try {
            canonical = UUID.fromString(entry.actorIdentity()).toString();
        } catch (RuntimeException invalid) {
            throw new IllegalArgumentException(
                "World-verb arbitration actor identity must be a UUID",
                invalid
            );
        }
        if (!canonical.equals(entry.actorIdentity())) {
            throw new IllegalArgumentException(
                "World-verb arbitration actor identity must be canonical"
            );
        }
        if (entry.request() == null || !entry.request().present()) {
            throw new IllegalArgumentException(
                "World-verb arbitration request is absent"
            );
        }
        if (!slots.add(entry.actorSlot())) {
            throw new IllegalArgumentException(
                "World-verb arbitration actor slots must be unique"
            );
        }
        if (!identities.add(entry.actorIdentity())) {
            throw new IllegalArgumentException(
                "World-verb arbitration actor identities must be unique"
            );
        }
    }

    @FunctionalInterface
    public interface AdmissionAttempt {
        Admission attempt(Entry entry);
    }

    public record Entry(
        int actorSlot,
        String actorIdentity,
        NativeWorldVerbRequest request
    ) {}

    public record Admission(boolean admitted, String rejectReason) {
        public static Admission admittedResult() {
            return new Admission(true, "");
        }

        public static Admission rejected(String reason) {
            return new Admission(false, reason);
        }
    }

    public record Decision(boolean admitted, String rejectReason) {}

    private record Target(int x, int y, int z) {}
}
