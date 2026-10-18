package com.hytalerlbridge.network.codec.worldactions;

import static com.hytalerlbridge.network.wire.MessageFields.checkedInt;
import static com.hytalerlbridge.network.wire.MessageFields.getField;
import static com.hytalerlbridge.network.wire.MessageFields.getLongField;
import static com.hytalerlbridge.network.wire.MessageFields.getMapField;
import static com.hytalerlbridge.network.wire.MessageFields.getStringField;

import com.hytalerlbridge.worldgen.policyactions.capture.PolicyWorldActionCaptureRequest;
import com.hytalerlbridge.worldgen.policyactions.capture.model.NativePolicyWorldActionCapture;
import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionSelection;
import java.util.Map;
import java.util.Set;
import org.msgpack.value.Value;

/** Strict decoder for the versioned policy World-action endpoint. */
public final class PolicyWorldActionCaptureRequestCodec {

    private static final Set<String> OBSERVE_FIELDS = Set.of(
        "type",
        "schema",
        "version",
        "contract_sha256",
        "phase",
        "actor_slot",
        "expected_actor_identity"
    );
    private static final Set<String> COMMIT_FIELDS = Set.of(
        "type",
        "schema",
        "version",
        "contract_sha256",
        "phase",
        "actor_slot",
        "expected_actor_identity",
        "expected_candidate_generation_sha256",
        "selection"
    );
    private static final Set<String> SELECTION_FIELDS = Set.of(
        "use_requested",
        "block_interaction_trigger",
        "block_candidate_index",
        "recipe_candidate_index"
    );

    private PolicyWorldActionCaptureRequestCodec() {}

    public static PolicyWorldActionCaptureRequest decode(
        Map<Value, Value> message
    ) {
        requireExactFields(message, phaseFields(message), "capture request");
        requireText(
            getStringField(message, "type"),
            NativePolicyWorldActionCapture.TYPE,
            "type"
        );
        requireText(
            getStringField(message, "schema"),
            NativePolicyWorldActionCapture.SCHEMA,
            "schema"
        );
        if (
            getLongField(message, "version", Long.MIN_VALUE)
                != NativePolicyWorldActionCapture.VERSION
        ) {
            throw new IllegalArgumentException(
                "Unsupported policy World-action capture version"
            );
        }
        requireText(
            getStringField(message, "contract_sha256").toUpperCase(),
            NativePolicyWorldActionCapture.CONTRACT_SHA256,
            "contract_sha256"
        );
        String phase = getStringField(message, "phase");
        int actorSlot = checkedInt(
            getLongField(message, "actor_slot", Long.MIN_VALUE),
            "actor_slot"
        );
        String expectedActorIdentity = getStringField(
            message,
            "expected_actor_identity"
        );
        if (phase.equals("observe")) {
            return PolicyWorldActionCaptureRequest.observe(
                actorSlot,
                expectedActorIdentity
            );
        }
        if (!phase.equals("commit")) {
            throw new IllegalArgumentException(
                "World-action capture phase must be observe or commit"
            );
        }
        Map<Value, Value> selection = getMapField(message, "selection");
        requireExactFields(selection, SELECTION_FIELDS, "selection");
        return new PolicyWorldActionCaptureRequest(
            phase,
            actorSlot,
            expectedActorIdentity,
            getStringField(
                message,
                "expected_candidate_generation_sha256"
            ),
            new PolicyWorldActionSelection(
                requiredBoolean(selection, "use_requested"),
                checkedInt(
                    getLongField(
                        selection,
                        "block_interaction_trigger",
                        Long.MIN_VALUE
                    ),
                    "block_interaction_trigger"
                ),
                checkedInt(
                    getLongField(
                        selection,
                        "block_candidate_index",
                        Long.MIN_VALUE
                    ),
                    "block_candidate_index"
                ),
                checkedInt(
                    getLongField(
                        selection,
                        "recipe_candidate_index",
                        Long.MIN_VALUE
                    ),
                    "recipe_candidate_index"
                )
            )
        );
    }

    private static Set<String> phaseFields(Map<Value, Value> message) {
        String phase = getStringField(message, "phase");
        return phase.equals("observe") ? OBSERVE_FIELDS : COMMIT_FIELDS;
    }

    private static void requireText(
        String actual,
        String expected,
        String name
    ) {
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException(
                name + " does not match the policy World-action contract"
            );
        }
    }

    private static void requireExactFields(
        Map<Value, Value> values,
        Set<String> expected,
        String name
    ) {
        if (values.size() != expected.size()) {
            throw new IllegalArgumentException(
                name + " has missing or unknown fields"
            );
        }
        for (String field : expected) {
            if (getField(values, field) == null) {
                throw new IllegalArgumentException(
                    name + " is missing " + field
                );
            }
        }
        for (Value key : values.keySet()) {
            if (
                !key.isStringValue()
                    || !expected.contains(key.asStringValue().asString())
            ) {
                throw new IllegalArgumentException(
                    name + " has an unknown field"
                );
            }
        }
    }

    private static boolean requiredBoolean(
        Map<Value, Value> values,
        String name
    ) {
        Value value = getField(values, name);
        if (value == null || !value.isBooleanValue()) {
            throw new IllegalArgumentException(name + " must be boolean");
        }
        return value.asBooleanValue().getBoolean();
    }
}
