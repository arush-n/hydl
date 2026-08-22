package com.hytalerlbridge.environment;

import com.hytalerlbridge.action.group.NativePolicyCombatBindingSpec;
import com.hytalerlbridge.environment.fidelity.NativeResourceOverride;
import com.hytalerlbridge.observation.NativeActorEvidenceRequest;

/** Options decoded from a reset request without depending on Hytale classes. */
public record EnvironmentOptions(
    String backend,
    String world,
    String npcRole,
    Double spawnX,
    Double spawnY,
    Double spawnZ,
    boolean combatTargetActive,
    String fidelityFixture,
    String nativeCombatItemId,
    String nativeCombatInteractionId,
    String nativeCombatInteractionType,
    java.util.List<String> nativeAgentArmorItemIds,
    boolean nativeNavigationTrace,
    java.util.List<String> nativeCombatAbilityInteractionIds,
    java.util.List<String> nativeCombatAbilityInteractionTypes,
    String nativeGuardInteractionId,
    String nativeGuardInteractionType,
    NativeActorEvidenceRequest nativeActorEvidenceRequest,
    boolean nativeWorldVerbs,
    java.util.List<Integer> nativeWorldHotbarSlots,
    java.util.List<String> nativeWorldHotbarItemIds,
    java.util.List<Integer> nativeWorldHotbarQuantities,
    String worldgenStructure,
    java.util.List<NativePolicyCombatBindingSpec> nativePolicyCombatBindings,
    int nativeTickRate,
    float nativeTimeDilation,
    java.util.List<NativeResourceOverride> nativeFidelityResourceOverrides,
    String combatTargetRole
) {
    public static final int DEFAULT_NATIVE_TICK_RATE = 30;
    public static final float DEFAULT_NATIVE_TIME_DILATION = 1.0f;
    public static final String STATIC_REGION_FIXTURE = "static_region";
    public static final String TARGET_MEMORY_FIXTURE = "target_memory";
    public static final String PATH_FOLLOWER_FIXTURE = "path_follower";
    public static final String NATIVE_DUEL_FIXTURE = "native_duel";
    public static final String NATIVE_PROJECTILE_DUEL_FIXTURE =
        "native_projectile_duel";
    public static final String DIVE_MOTION_FIXTURE = "dive_motion";
    public static final String SYNTHETIC_WORLD_VERBS_FIXTURE =
        "synthetic_world_verbs";
    public static final String SYNTHETIC_BLOCK_USE_FIXTURE =
        "synthetic_block_use";
    public static final String FALLING_BLOCK_FIXTURE = "falling_block";
    private static final java.util.Set<String> NATIVE_COMBAT_INTERACTION_TYPES =
        java.util.Set.of(
            "Primary",
            "Secondary",
            "Ability1",
            "Ability2",
            "Ability3"
        );
    private static final int MAX_NATIVE_WORLD_HOTBAR_ENTRIES = 16;
    private static final int MAX_NATIVE_WORLD_HOTBAR_SLOT = 31;

    public static EnvironmentOptions simulator() {
        return new EnvironmentOptions(
            "simulator", "flat", "Trork_Unarmed", null, null, null, true,
            "default", null, null, null, null, false
        );
    }

    /** Compatibility constructor for callers predating custom target roles. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType,
        java.util.List<String> nativeAgentArmorItemIds,
        boolean nativeNavigationTrace,
        java.util.List<String> nativeCombatAbilityInteractionIds,
        java.util.List<String> nativeCombatAbilityInteractionTypes,
        String nativeGuardInteractionId,
        String nativeGuardInteractionType,
        NativeActorEvidenceRequest nativeActorEvidenceRequest,
        boolean nativeWorldVerbs,
        java.util.List<Integer> nativeWorldHotbarSlots,
        java.util.List<String> nativeWorldHotbarItemIds,
        java.util.List<Integer> nativeWorldHotbarQuantities,
        String worldgenStructure,
        java.util.List<NativePolicyCombatBindingSpec> nativePolicyCombatBindings,
        int nativeTickRate,
        float nativeTimeDilation,
        java.util.List<NativeResourceOverride> nativeFidelityResourceOverrides
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            nativeAgentArmorItemIds,
            nativeNavigationTrace,
            nativeCombatAbilityInteractionIds,
            nativeCombatAbilityInteractionTypes,
            nativeGuardInteractionId,
            nativeGuardInteractionType,
            nativeActorEvidenceRequest,
            nativeWorldVerbs,
            nativeWorldHotbarSlots,
            nativeWorldHotbarItemIds,
            nativeWorldHotbarQuantities,
            worldgenStructure,
            nativePolicyCombatBindings,
            nativeTickRate,
            nativeTimeDilation,
            nativeFidelityResourceOverrides,
            null
        );
    }

    /** Compatibility constructor for callers predating fidelity resource setup. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType,
        java.util.List<String> nativeAgentArmorItemIds,
        boolean nativeNavigationTrace,
        java.util.List<String> nativeCombatAbilityInteractionIds,
        java.util.List<String> nativeCombatAbilityInteractionTypes,
        String nativeGuardInteractionId,
        String nativeGuardInteractionType,
        NativeActorEvidenceRequest nativeActorEvidenceRequest,
        boolean nativeWorldVerbs,
        java.util.List<Integer> nativeWorldHotbarSlots,
        java.util.List<String> nativeWorldHotbarItemIds,
        java.util.List<Integer> nativeWorldHotbarQuantities,
        String worldgenStructure,
        java.util.List<NativePolicyCombatBindingSpec> nativePolicyCombatBindings,
        int nativeTickRate,
        float nativeTimeDilation
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            nativeAgentArmorItemIds,
            nativeNavigationTrace,
            nativeCombatAbilityInteractionIds,
            nativeCombatAbilityInteractionTypes,
            nativeGuardInteractionId,
            nativeGuardInteractionType,
            nativeActorEvidenceRequest,
            nativeWorldVerbs,
            nativeWorldHotbarSlots,
            nativeWorldHotbarItemIds,
            nativeWorldHotbarQuantities,
            worldgenStructure,
            nativePolicyCombatBindings,
            nativeTickRate,
            nativeTimeDilation,
            null,
            null
        );
    }

    /** Compatibility constructor for callers predating headless World verbs. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType,
        java.util.List<String> nativeAgentArmorItemIds,
        boolean nativeNavigationTrace,
        java.util.List<String> nativeCombatAbilityInteractionIds,
        java.util.List<String> nativeCombatAbilityInteractionTypes,
        String nativeGuardInteractionId,
        String nativeGuardInteractionType,
        NativeActorEvidenceRequest nativeActorEvidenceRequest
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            nativeAgentArmorItemIds,
            nativeNavigationTrace,
            nativeCombatAbilityInteractionIds,
            nativeCombatAbilityInteractionTypes,
            nativeGuardInteractionId,
            nativeGuardInteractionType,
            nativeActorEvidenceRequest,
            false,
            null,
            null,
            null
        );
    }

    /** Compatibility constructor for callers predating explicit World loadouts. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType,
        java.util.List<String> nativeAgentArmorItemIds,
        boolean nativeNavigationTrace,
        java.util.List<String> nativeCombatAbilityInteractionIds,
        java.util.List<String> nativeCombatAbilityInteractionTypes,
        String nativeGuardInteractionId,
        String nativeGuardInteractionType,
        NativeActorEvidenceRequest nativeActorEvidenceRequest,
        boolean nativeWorldVerbs
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            nativeAgentArmorItemIds,
            nativeNavigationTrace,
            nativeCombatAbilityInteractionIds,
            nativeCombatAbilityInteractionTypes,
            nativeGuardInteractionId,
            nativeGuardInteractionType,
            nativeActorEvidenceRequest,
            nativeWorldVerbs,
            null,
            null,
            null
        );
    }

    /** Compatibility constructor for callers predating actor-evidence negotiation. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType,
        java.util.List<String> nativeAgentArmorItemIds,
        boolean nativeNavigationTrace,
        java.util.List<String> nativeCombatAbilityInteractionIds,
        java.util.List<String> nativeCombatAbilityInteractionTypes,
        String nativeGuardInteractionId,
        String nativeGuardInteractionType
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            nativeAgentArmorItemIds,
            nativeNavigationTrace,
            nativeCombatAbilityInteractionIds,
            nativeCombatAbilityInteractionTypes,
            nativeGuardInteractionId,
            nativeGuardInteractionType,
            null,
            false
        );
    }

    /** Compatibility constructor for callers predating policy combat verbs. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType,
        java.util.List<String> nativeAgentArmorItemIds,
        boolean nativeNavigationTrace
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            nativeAgentArmorItemIds,
            nativeNavigationTrace,
            null,
            null,
            null,
            null,
            null,
            false
        );
    }

    /** Compatibility constructor for callers predating native navigation traces. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType,
        java.util.List<String> nativeAgentArmorItemIds
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            nativeAgentArmorItemIds,
            false
        );
    }

    /** Compatibility constructor for callers predating native armor overrides. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            null
        );
    }

    /** Compatibility constructor for callers predating native item interactions. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            null,
            null,
            null
        );
    }

    /** Compatibility constructor for callers predating geometry fixtures. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            "default",
            null,
            null,
            null,
            null
        );
    }

    /** Compatibility constructor for callers predating V2 world structures. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType,
        java.util.List<String> nativeAgentArmorItemIds,
        boolean nativeNavigationTrace,
        java.util.List<String> nativeCombatAbilityInteractionIds,
        java.util.List<String> nativeCombatAbilityInteractionTypes,
        String nativeGuardInteractionId,
        String nativeGuardInteractionType,
        NativeActorEvidenceRequest nativeActorEvidenceRequest,
        boolean nativeWorldVerbs,
        java.util.List<Integer> nativeWorldHotbarSlots,
        java.util.List<String> nativeWorldHotbarItemIds,
        java.util.List<Integer> nativeWorldHotbarQuantities
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            nativeAgentArmorItemIds,
            nativeNavigationTrace,
            nativeCombatAbilityInteractionIds,
            nativeCombatAbilityInteractionTypes,
            nativeGuardInteractionId,
            nativeGuardInteractionType,
            nativeActorEvidenceRequest,
            nativeWorldVerbs,
            nativeWorldHotbarSlots,
            nativeWorldHotbarItemIds,
            nativeWorldHotbarQuantities,
            "Default",
            null
        );
    }

    /** Compatibility constructor for callers predating actor-major combat. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType,
        java.util.List<String> nativeAgentArmorItemIds,
        boolean nativeNavigationTrace,
        java.util.List<String> nativeCombatAbilityInteractionIds,
        java.util.List<String> nativeCombatAbilityInteractionTypes,
        String nativeGuardInteractionId,
        String nativeGuardInteractionType,
        NativeActorEvidenceRequest nativeActorEvidenceRequest,
        boolean nativeWorldVerbs,
        java.util.List<Integer> nativeWorldHotbarSlots,
        java.util.List<String> nativeWorldHotbarItemIds,
        java.util.List<Integer> nativeWorldHotbarQuantities,
        String worldgenStructure
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            nativeAgentArmorItemIds,
            nativeNavigationTrace,
            nativeCombatAbilityInteractionIds,
            nativeCombatAbilityInteractionTypes,
            nativeGuardInteractionId,
            nativeGuardInteractionType,
            nativeActorEvidenceRequest,
            nativeWorldVerbs,
            nativeWorldHotbarSlots,
            nativeWorldHotbarItemIds,
            nativeWorldHotbarQuantities,
            worldgenStructure,
            null
        );
    }

    /** Compatibility constructor for callers predating native fast-forward. */
    public EnvironmentOptions(
        String backend,
        String world,
        String npcRole,
        Double spawnX,
        Double spawnY,
        Double spawnZ,
        boolean combatTargetActive,
        String fidelityFixture,
        String nativeCombatItemId,
        String nativeCombatInteractionId,
        String nativeCombatInteractionType,
        java.util.List<String> nativeAgentArmorItemIds,
        boolean nativeNavigationTrace,
        java.util.List<String> nativeCombatAbilityInteractionIds,
        java.util.List<String> nativeCombatAbilityInteractionTypes,
        String nativeGuardInteractionId,
        String nativeGuardInteractionType,
        NativeActorEvidenceRequest nativeActorEvidenceRequest,
        boolean nativeWorldVerbs,
        java.util.List<Integer> nativeWorldHotbarSlots,
        java.util.List<String> nativeWorldHotbarItemIds,
        java.util.List<Integer> nativeWorldHotbarQuantities,
        String worldgenStructure,
        java.util.List<NativePolicyCombatBindingSpec> nativePolicyCombatBindings
    ) {
        this(
            backend,
            world,
            npcRole,
            spawnX,
            spawnY,
            spawnZ,
            combatTargetActive,
            fidelityFixture,
            nativeCombatItemId,
            nativeCombatInteractionId,
            nativeCombatInteractionType,
            nativeAgentArmorItemIds,
            nativeNavigationTrace,
            nativeCombatAbilityInteractionIds,
            nativeCombatAbilityInteractionTypes,
            nativeGuardInteractionId,
            nativeGuardInteractionType,
            nativeActorEvidenceRequest,
            nativeWorldVerbs,
            nativeWorldHotbarSlots,
            nativeWorldHotbarItemIds,
            nativeWorldHotbarQuantities,
            worldgenStructure,
            nativePolicyCombatBindings,
            DEFAULT_NATIVE_TICK_RATE,
            DEFAULT_NATIVE_TIME_DILATION
        );
    }

    public EnvironmentOptions {
        backend = normalized(backend, "simulator");
        world = normalized(world, "flat");
        worldgenStructure = normalized(worldgenStructure, "Default");
        npcRole = normalized(npcRole, "Trork_Unarmed");
        combatTargetRole = optional(combatTargetRole);
        fidelityFixture = normalized(fidelityFixture, "default");
        nativeCombatItemId = optional(nativeCombatItemId);
        nativeCombatInteractionId = optional(
            nativeCombatInteractionId
        );
        nativeCombatInteractionType = optional(nativeCombatInteractionType);
        nativeAgentArmorItemIds = optionalItemIds(
            nativeAgentArmorItemIds,
            "native_agent_armor_item_ids"
        );
        nativeCombatAbilityInteractionIds = optionalItemIds(
            nativeCombatAbilityInteractionIds,
            "native_combat_ability_interaction_ids"
        );
        nativeCombatAbilityInteractionTypes = optionalInteractionTypes(
            nativeCombatAbilityInteractionTypes,
            "native_combat_ability_interaction_types"
        );
        nativeGuardInteractionId = optional(nativeGuardInteractionId);
        nativeGuardInteractionType = optional(nativeGuardInteractionType);
        nativeWorldHotbarSlots = optionalHotbarSlots(
            nativeWorldHotbarSlots,
            "native_world_hotbar_slots"
        );
        nativeWorldHotbarItemIds = optionalHotbarItemIds(
            nativeWorldHotbarItemIds,
            "native_world_hotbar_item_ids"
        );
        nativeWorldHotbarQuantities = optionalPositiveIntegers(
            nativeWorldHotbarQuantities,
            "native_world_hotbar_quantities"
        );
        nativePolicyCombatBindings = optionalPolicyCombatBindings(
            nativePolicyCombatBindings
        );
        nativeFidelityResourceOverrides = optionalResourceOverrides(
            nativeFidelityResourceOverrides
        );
        if (nativeTickRate < 1 || nativeTickRate > 2048) {
            throw new IllegalArgumentException(
                "native_tick_rate must be 1..2048"
            );
        }
        if (!Float.isFinite(nativeTimeDilation)
            || nativeTimeDilation <= 0.01f
            || nativeTimeDilation > 4.0f) {
            throw new IllegalArgumentException(
                "native_time_dilation must be finite and in (0.01, 4]"
            );
        }
        if ((!backend.equals("native") && !backend.equals("headless"))
            && (nativeTickRate != DEFAULT_NATIVE_TICK_RATE
                || nativeTimeDilation != DEFAULT_NATIVE_TIME_DILATION)) {
            throw new IllegalArgumentException(
                "native tick controls require the native or headless backend"
            );
        }
        if (combatTargetRole != null
            && !backend.equals("native")
            && !backend.equals("headless")) {
            throw new IllegalArgumentException(
                "combat_target_role requires the native or headless backend"
            );
        }
        if (combatTargetRole != null
            && fidelityFixture.equals(NATIVE_PROJECTILE_DUEL_FIXTURE)) {
            throw new IllegalArgumentException(
                "native_projectile_duel owns its pinned target role"
            );
        }
        if (
            !world.equals("hytale_generator")
                && !worldgenStructure.equals("Default")
        ) {
            throw new IllegalArgumentException(
                "worldgen_structure is only selectable for hytale_generator"
            );
        }
        if (nativeActorEvidenceRequest != null) {
            nativeActorEvidenceRequest.requireSupported();
        }
        if (
            nativeWorldVerbs
                && !backend.equals("native")
                && !backend.equals("headless")
        ) {
            throw new IllegalArgumentException(
                "native_world_verbs requires the native or headless backend"
            );
        }
        boolean anyWorldLoadout = nativeWorldHotbarSlots != null
            || nativeWorldHotbarItemIds != null
            || nativeWorldHotbarQuantities != null;
        boolean allWorldLoadout = nativeWorldHotbarSlots != null
            && nativeWorldHotbarItemIds != null
            && nativeWorldHotbarQuantities != null;
        if (anyWorldLoadout && !allWorldLoadout) {
            throw new IllegalArgumentException(
                "native_world_hotbar_slots, native_world_hotbar_item_ids, "
                    + "and native_world_hotbar_quantities must be supplied together"
            );
        }
        if (allWorldLoadout && !nativeWorldVerbs) {
            throw new IllegalArgumentException(
                "native World hotbar entries require native_world_verbs=true"
            );
        }
        if (allWorldLoadout
            && fidelityFixture.equals(SYNTHETIC_WORLD_VERBS_FIXTURE)) {
            throw new IllegalArgumentException(
                "native World hotbar entries cannot replace the pinned "
                    + "synthetic_world_verbs fixture loadout"
            );
        }
        if (allWorldLoadout
            && (nativeWorldHotbarSlots.isEmpty()
                || nativeWorldHotbarSlots.size()
                    != nativeWorldHotbarItemIds.size()
                || nativeWorldHotbarSlots.size()
                    != nativeWorldHotbarQuantities.size())) {
            throw new IllegalArgumentException(
                "native World hotbar entries must contain 1.."
                    + MAX_NATIVE_WORLD_HOTBAR_ENTRIES
                    + " aligned entries"
            );
        }
        if (!java.util.Set.of(
            "default",
            "los_wall",
            "los_diagonal_graze",
            "los_diagonal_block",
            "ledge",
            "ceiling",
            "half_block",
            TARGET_MEMORY_FIXTURE,
            PATH_FOLLOWER_FIXTURE,
            NATIVE_DUEL_FIXTURE,
            NATIVE_PROJECTILE_DUEL_FIXTURE,
            DIVE_MOTION_FIXTURE,
            SYNTHETIC_WORLD_VERBS_FIXTURE,
            SYNTHETIC_BLOCK_USE_FIXTURE,
            FALLING_BLOCK_FIXTURE,
            STATIC_REGION_FIXTURE
        ).contains(fidelityFixture)) {
            throw new IllegalArgumentException(
                "Unknown fidelity_fixture: " + fidelityFixture
            );
        }
        boolean anySpawn = spawnX != null || spawnY != null || spawnZ != null;
        boolean allSpawn = spawnX != null && spawnY != null && spawnZ != null;
        if (anySpawn && !allSpawn) {
            throw new IllegalArgumentException(
                "spawn_x, spawn_y, and spawn_z must be supplied together"
            );
        }
        if (allSpawn && (!Double.isFinite(spawnX)
            || !Double.isFinite(spawnY)
            || !Double.isFinite(spawnZ))) {
            throw new IllegalArgumentException("spawn coordinates must be finite");
        }
        if (
            fidelityFixture.equals(TARGET_MEMORY_FIXTURE)
                && combatTargetActive
        ) {
            throw new IllegalArgumentException(
                "target_memory requires combat_target_active=false"
            );
        }
        if (fidelityFixture.equals(PATH_FOLLOWER_FIXTURE)
            && (!combatTargetActive || !nativeNavigationTrace)) {
            throw new IllegalArgumentException(
                "path_follower requires combat_target_active=true and "
                    + "native_navigation_trace=true"
            );
        }
        if ((fidelityFixture.equals(NATIVE_DUEL_FIXTURE)
                || fidelityFixture.equals(NATIVE_PROJECTILE_DUEL_FIXTURE))
            && (!combatTargetActive
                || (!backend.equals("native") && !backend.equals("headless")))) {
            throw new IllegalArgumentException(
                fidelityFixture
                    + " requires combat_target_active=true and a native backend"
            );
        }
        if (fidelityFixture.equals(DIVE_MOTION_FIXTURE)
            && (!backend.equals("native") && !backend.equals("headless"))) {
            throw new IllegalArgumentException(
                "dive_motion requires the native or headless backend"
            );
        }
        if (
            (fidelityFixture.equals(SYNTHETIC_WORLD_VERBS_FIXTURE)
                || fidelityFixture.equals(SYNTHETIC_BLOCK_USE_FIXTURE)
                || fidelityFixture.equals(FALLING_BLOCK_FIXTURE))
                && !backend.equals("native")
                && !backend.equals("headless")
        ) {
            throw new IllegalArgumentException(
                fidelityFixture
                    + " requires the native or headless backend"
            );
        }
        boolean anyNativeCombat = nativeCombatInteractionId != null
            || nativeCombatInteractionType != null;
        boolean allNativeCombat = nativeCombatInteractionId != null
            && nativeCombatInteractionType != null;
        if (anyNativeCombat && !allNativeCombat) {
            throw new IllegalArgumentException(
                "native_combat_item_id, native_combat_interaction_id, "
                    + "and native_combat_interaction_type must be supplied together"
            );
        }
        if (allNativeCombat
            && !backend.equals("native")
            && !backend.equals("headless")) {
            throw new IllegalArgumentException(
                "native combat interactions require the native or headless backend"
            );
        }
        if (allNativeCombat
            && !NATIVE_COMBAT_INTERACTION_TYPES.contains(
                nativeCombatInteractionType
            )) {
            throw new IllegalArgumentException(
                "Unknown native_combat_interaction_type: "
                    + nativeCombatInteractionType
            );
        }
        boolean anyAbilities = nativeCombatAbilityInteractionIds != null
            || nativeCombatAbilityInteractionTypes != null;
        boolean allAbilities = nativeCombatAbilityInteractionIds != null
            && nativeCombatAbilityInteractionTypes != null;
        if (anyAbilities && !allAbilities) {
            throw new IllegalArgumentException(
                "native_combat_ability_interaction_ids and "
                    + "native_combat_ability_interaction_types must be supplied together"
            );
        }
        if (allAbilities
            && (nativeCombatAbilityInteractionIds.isEmpty()
                || nativeCombatAbilityInteractionIds.size() > 16
                || nativeCombatAbilityInteractionIds.size()
                    != nativeCombatAbilityInteractionTypes.size())) {
            throw new IllegalArgumentException(
                "native combat ability bindings must contain 1..16 paired entries"
            );
        }
        boolean anyGuard = nativeGuardInteractionId != null
            || nativeGuardInteractionType != null;
        boolean allGuard = nativeGuardInteractionId != null
            && nativeGuardInteractionType != null;
        if (anyGuard && !allGuard) {
            throw new IllegalArgumentException(
                "native_guard_interaction_id and native_guard_interaction_type "
                    + "must be supplied together"
            );
        }
        if (allGuard
            && !NATIVE_COMBAT_INTERACTION_TYPES.contains(
                nativeGuardInteractionType
            )) {
            throw new IllegalArgumentException(
                "Unknown native_guard_interaction_type: "
                    + nativeGuardInteractionType
            );
        }
        boolean anyPolicyCombat = allAbilities || allGuard;
        if (anyPolicyCombat && nativeCombatItemId == null) {
            throw new IllegalArgumentException(
                "native policy combat bindings require native_combat_item_id"
            );
        }
        if (anyPolicyCombat
            && !backend.equals("native")
            && !backend.equals("headless")) {
            throw new IllegalArgumentException(
                "native policy combat bindings require the native or headless backend"
            );
        }
        if (allNativeCombat && nativeCombatItemId == null) {
            throw new IllegalArgumentException(
                "native combat interactions require native_combat_item_id"
            );
        }
        if (nativeCombatItemId != null && !allNativeCombat && !anyPolicyCombat) {
            throw new IllegalArgumentException(
                "native_combat_item_id requires an attack, ability, or guard binding"
            );
        }
        if (nativeAgentArmorItemIds != null
            && !backend.equals("native")
            && !backend.equals("headless")) {
            throw new IllegalArgumentException(
                "native_agent_armor_item_ids requires the native or headless backend"
            );
        }
        if (nativeNavigationTrace
            && !backend.equals("native")
            && !backend.equals("headless")) {
            throw new IllegalArgumentException(
                "native_navigation_trace requires the native or headless backend"
            );
        }
        if (nativePolicyCombatBindings != null
            && !backend.equals("native")
            && !backend.equals("headless")) {
            throw new IllegalArgumentException(
                "native group policy-combat bindings require the native or "
                    + "headless backend"
            );
        }
        if (nativeFidelityResourceOverrides != null
            && !fidelityFixture.equals(NATIVE_DUEL_FIXTURE)
            && !fidelityFixture.equals(NATIVE_PROJECTILE_DUEL_FIXTURE)) {
            throw new IllegalArgumentException(
                "native fidelity resource overrides require native_duel or "
                    + "native_projectile_duel"
            );
        }
    }

    public boolean isNative() {
        return backend.equals("native") || backend.equals("headless");
    }

    public boolean hasSpawn() {
        return spawnX != null;
    }

    public boolean hasNativeCombatInteraction() {
        return nativeCombatInteractionId != null;
    }

    public boolean hasNativeCombatItem() {
        return nativeCombatItemId != null;
    }

    public boolean hasNativeCombatAbilities() {
        return nativeCombatAbilityInteractionIds != null;
    }

    public boolean hasNativeGuardInteraction() {
        return nativeGuardInteractionId != null;
    }

    public boolean hasNativeActorEvidenceRequest() {
        return nativeActorEvidenceRequest != null;
    }

    public boolean hasNativeWorldHotbarOverride() {
        return nativeWorldHotbarSlots != null;
    }

    public boolean hasNativePolicyCombatBindings() {
        return nativePolicyCombatBindings != null;
    }

    public boolean hasNativeFidelityResourceOverrides() {
        return nativeFidelityResourceOverrides != null;
    }

    public boolean hasCombatTargetRole() {
        return combatTargetRole != null;
    }

    public java.util.List<NativeResourceOverride> nativeResourceOverrides(
        int entityId
    ) {
        if (nativeFidelityResourceOverrides == null) return java.util.List.of();
        java.util.ArrayList<NativeResourceOverride> rows =
            new java.util.ArrayList<>();
        for (NativeResourceOverride row : nativeFidelityResourceOverrides) {
            if (row.entityId() == entityId) rows.add(row);
        }
        return java.util.List.copyOf(rows);
    }

    public NativePolicyCombatBindingSpec nativePolicyCombatBinding(
        int entityId
    ) {
        if (nativePolicyCombatBindings == null) return null;
        for (NativePolicyCombatBindingSpec binding : nativePolicyCombatBindings) {
            if (binding.entityId() == entityId) return binding;
        }
        return null;
    }

    /** Whether autonomous native block ticking is in the fixture's scope. */
    public boolean blockTickingEnabled() {
        return !fidelityFixture.equals(STATIC_REGION_FIXTURE)
            && !fidelityFixture.equals(DIVE_MOTION_FIXTURE);
    }

    /**
     * Whether reset replaces the role-authored agent armor exactly.
     *
     * <p>An explicitly empty list is a meaningful unarmored override. A
     * missing field remains {@code null} and preserves the role loadout.</p>
     */
    public boolean hasNativeAgentArmorOverride() {
        return nativeAgentArmorItemIds != null;
    }

    private static String normalized(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value.trim();
    }

    private static String optional(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }

    private static java.util.List<NativePolicyCombatBindingSpec>
        optionalPolicyCombatBindings(
            java.util.List<NativePolicyCombatBindingSpec> values
        ) {
        if (values == null) return null;
        if (values.isEmpty()) {
            throw new IllegalArgumentException(
                "native group policy-combat bindings must be nonempty"
            );
        }
        java.util.HashSet<Integer> seen = new java.util.HashSet<>();
        java.util.ArrayList<NativePolicyCombatBindingSpec> copy =
            new java.util.ArrayList<>(values.size());
        for (NativePolicyCombatBindingSpec value : values) {
            if (value == null || value.entityId() == 0) {
                throw new IllegalArgumentException(
                    "native group policy-combat bindings are for nonzero "
                        + "actors; entity 0 uses native_combat_* options"
                );
            }
            if (!seen.add(value.entityId())) {
                throw new IllegalArgumentException(
                    "native group policy-combat bindings require unique actors"
                );
            }
            copy.add(value);
        }
        return java.util.List.copyOf(copy);
    }

    private static java.util.List<NativeResourceOverride>
        optionalResourceOverrides(
            java.util.List<NativeResourceOverride> values
        ) {
        if (values == null) return null;
        if (values.isEmpty()) {
            throw new IllegalArgumentException(
                "native fidelity resource overrides must be nonempty"
            );
        }
        java.util.HashSet<String> seen = new java.util.HashSet<>();
        java.util.ArrayList<NativeResourceOverride> copy =
            new java.util.ArrayList<>(values.size());
        for (NativeResourceOverride value : values) {
            if (value == null) {
                throw new IllegalArgumentException(
                    "native fidelity resource overrides contain null"
                );
            }
            String key = value.entityId() + ":" + value.resourceId();
            if (!seen.add(key)) {
                throw new IllegalArgumentException(
                    "duplicate native fidelity resource override: " + key
                );
            }
            copy.add(value);
        }
        return java.util.List.copyOf(copy);
    }

    private static java.util.List<String> optionalItemIds(
        java.util.List<String> values,
        String fieldName
    ) {
        if (values == null) return null;
        java.util.ArrayList<String> normalized = new java.util.ArrayList<>(
            values.size()
        );
        java.util.HashSet<String> seen = new java.util.HashSet<>();
        for (String value : values) {
            String itemId = optional(value);
            if (itemId == null) {
                throw new IllegalArgumentException(
                    fieldName + " entries must be nonblank strings"
                );
            }
            if (!seen.add(itemId)) {
                throw new IllegalArgumentException(
                    fieldName + " contains duplicate item id: " + itemId
                );
            }
            normalized.add(itemId);
        }
        return java.util.List.copyOf(normalized);
    }

    private static java.util.List<String> optionalHotbarItemIds(
        java.util.List<String> values,
        String fieldName
    ) {
        if (values == null) return null;
        requireHotbarEntryCount(values.size(), fieldName);
        java.util.ArrayList<String> normalized = new java.util.ArrayList<>(
            values.size()
        );
        for (String value : values) {
            String itemId = optional(value);
            if (itemId == null) {
                throw new IllegalArgumentException(
                    fieldName + " entries must be nonblank strings"
                );
            }
            normalized.add(itemId);
        }
        return java.util.List.copyOf(normalized);
    }

    private static java.util.List<Integer> optionalHotbarSlots(
        java.util.List<Integer> values,
        String fieldName
    ) {
        if (values == null) return null;
        requireHotbarEntryCount(values.size(), fieldName);
        java.util.HashSet<Integer> seen = new java.util.HashSet<>();
        for (Integer value : values) {
            if (value == null || value < 0 || value > MAX_NATIVE_WORLD_HOTBAR_SLOT) {
                throw new IllegalArgumentException(
                    fieldName + " entries must be integers in [0, "
                        + MAX_NATIVE_WORLD_HOTBAR_SLOT + "]"
                );
            }
            if (!seen.add(value)) {
                throw new IllegalArgumentException(
                    fieldName + " contains duplicate slot: " + value
                );
            }
        }
        return java.util.List.copyOf(values);
    }

    private static java.util.List<Integer> optionalPositiveIntegers(
        java.util.List<Integer> values,
        String fieldName
    ) {
        if (values == null) return null;
        requireHotbarEntryCount(values.size(), fieldName);
        for (Integer value : values) {
            if (value == null || value <= 0) {
                throw new IllegalArgumentException(
                    fieldName + " entries must be positive integers"
                );
            }
        }
        return java.util.List.copyOf(values);
    }

    private static void requireHotbarEntryCount(int size, String fieldName) {
        if (size < 1 || size > MAX_NATIVE_WORLD_HOTBAR_ENTRIES) {
            throw new IllegalArgumentException(
                fieldName + " must contain 1.."
                    + MAX_NATIVE_WORLD_HOTBAR_ENTRIES + " entries"
            );
        }
    }

    private static java.util.List<String> optionalInteractionTypes(
        java.util.List<String> values,
        String fieldName
    ) {
        if (values == null) return null;
        java.util.ArrayList<String> normalized = new java.util.ArrayList<>(
            values.size()
        );
        for (String value : values) {
            String type = optional(value);
            if (type == null || !NATIVE_COMBAT_INTERACTION_TYPES.contains(type)) {
                throw new IllegalArgumentException(
                    fieldName + " contains unknown interaction type: " + value
                );
            }
            normalized.add(type);
        }
        return java.util.List.copyOf(normalized);
    }
}
