package com.hytalerlbridge.worldgen;

import java.util.Map;

/** Build-scoped native World-action transport capabilities. */
public record NativeWorldActionCapabilities(
    boolean useRequestAvailable,
    boolean placeBlockRequestAvailable,
    boolean breakBlockRequestAvailable,
    boolean craftRecipeRequestAvailable,
    boolean liveMutationAcknowledgementAvailable,
    boolean mutableBlockEvidenceAvailable,
    boolean dropProgramEvidenceAvailable,
    boolean craftingCatalogEvidenceAvailable,
    boolean headlessServerActorAcceptanceAvailable,
    boolean publicPlayerAcceptanceCertified
) {

    public static final String SCHEMA =
        "hytalerl_native_world_action_capabilities_v1";
    public static final int VERSION = 1;
    public static final NativeWorldActionCapabilities CURRENT =
        new NativeWorldActionCapabilities(
            true,
            true,
            true,
            true,
            false,
            true,
            true,
            true,
            false,
            false
        );

    public static NativeWorldActionCapabilities forSession(
        boolean headlessWorldVerbs
    ) {
        return new NativeWorldActionCapabilities(
            true,
            true,
            true,
            true,
            headlessWorldVerbs,
            true,
            true,
            true,
            headlessWorldVerbs,
            false
        );
    }

    /** Add only scalar values supported by the bridge info encoder. */
    public void putInto(Map<String, Object> info) {
        info.put("world_action_capabilities_schema", SCHEMA);
        info.put("world_action_capabilities_version", VERSION);
        info.put("world_action_use_request_available", useRequestAvailable);
        info.put(
            "world_action_place_block_request_available",
            placeBlockRequestAvailable
        );
        info.put(
            "world_action_break_block_request_available",
            breakBlockRequestAvailable
        );
        info.put(
            "world_action_craft_recipe_request_available",
            craftRecipeRequestAvailable
        );
        info.put(
            "world_action_live_mutation_acknowledgement_available",
            liveMutationAcknowledgementAvailable
        );
        info.put(
            "world_action_mutable_block_evidence_available",
            mutableBlockEvidenceAvailable
        );
        info.put(
            "world_action_drop_program_evidence_available",
            dropProgramEvidenceAvailable
        );
        info.put(
            "world_action_crafting_catalog_evidence_available",
            craftingCatalogEvidenceAvailable
        );
        info.put(
            "world_action_headless_server_actor_acceptance_available",
            headlessServerActorAcceptanceAvailable
        );
        info.put(
            "world_action_public_player_acceptance_certified",
            publicPlayerAcceptanceCertified
        );
        info.put(
            "world_action_unavailable_reason",
            headlessServerActorAcceptanceAvailable
                ? "authenticated_public_player_acceptance_not_certified"
                : "headless_world_verbs_not_enabled_for_this_reset"
        );
    }
}
