package com.hytalerlbridge.action.group;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** Identity and bounded semantics for simultaneous actor actions. */
public final class NativeGroupActionContract {

    public static final String SCHEMA = "hytalerl_native_group_action_v5";
    public static final int VERSION = 5;
    public static final String EMPTY_ACTOR_STEP =
        "native_ai_unoverridden";
    public static final String NONZERO_ACTOR_PRECONDITION =
        "kill_trork_and_combat_target_active_false";
    public static final String NONZERO_ACTOR_SURFACE =
        "movement_look_jump_hotbar_role_attack_policy_item_ability_guard_dodge_charge_typed_world_verbs";
    public static final String WORLD_VERB_ARBITRATION =
        "entity_id_ascending_first_admitted_per_exact_target";
    public static final String TYPED_WORLD_VERB_EVIDENCE =
        "candidate_generation_and_selected_semantic_sha256_required";
    public static final String ENTITY_ZERO_ONLY_SURFACE =
        "legacy_use_place_break_craft";

    private NativeGroupActionContract() {}

    /** Group transport cannot name actors absent from native evidence. */
    public static int actorCapacity() {
        return NativeActorEvidenceFrame.ENTITY_COUNT;
    }

    public static String canonicalJson() {
        return "{\"actor_capacity\":" + actorCapacity()
            + ",\"actor_order\":\"entity_id_ascending\""
            + ",\"atomicity\":\"one_shared_engine_tick_actor_major\""
            + ",\"binding_schema\":\""
            + NativePolicyCombatBindingSpec.SCHEMA + "\""
            + ",\"binding_version\":"
            + NativePolicyCombatBindingSpec.VERSION
            + ",\"empty_actor_step\":\"" + EMPTY_ACTOR_STEP + "\""
            + ",\"entity_zero_only_surface\":\"" + ENTITY_ZERO_ONLY_SURFACE + "\""
            + ",\"nonzero_actor_precondition\":\"" + NONZERO_ACTOR_PRECONDITION + "\""
            + ",\"nonzero_actor_surface\":\"" + NONZERO_ACTOR_SURFACE + "\""
            + ",\"schema\":\"" + SCHEMA + "\""
            + ",\"typed_world_verb_evidence\":\""
            + TYPED_WORLD_VERB_EVIDENCE + "\""
            + ",\"version\":" + VERSION
            + ",\"world_verb_arbitration\":\""
            + WORLD_VERB_ARBITRATION + "\"}";
    }

    public static String sha256() {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(
                canonicalJson().getBytes(StandardCharsets.UTF_8)
            );
            return java.util.HexFormat.of().withUpperCase().formatHex(digest);
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    /** Empty when the action is inside the nonzero-actor v1 surface. */
    public static String nonzeroActorRejectReason(AgentAction action) {
        if (action == null) return "action_missing";
        if (
            action.use()
                || action.hasPlacement()
                || action.hasBreak()
                || action.hasCraft()
        ) {
            return "legacy_world_or_use_verb_entity_zero_only";
        }
        return "";
    }

    /** Empty only when a typed policy World verb has complete evidence. */
    public static String worldVerbEvidenceRejectReason(AgentAction action) {
        if (action == null) return "action_missing";
        var request = action.nativeWorldVerbRequest();
        if (
            request != null
                && request.present()
                && !request.candidateEvidenceBound()
        ) {
            return com.hytalerlbridge.action.NativeWorldVerbRequest
                .POLICY_EVIDENCE_REQUIRED;
        }
        return "";
    }
}
