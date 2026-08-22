package com.hytalerlbridge.policy.perception;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.policy.ObservationAssembler;
import com.hytalerlbridge.policy.PolicyPerception;
import com.hytalerlbridge.policy.Projection;
import com.hytalerlbridge.policy.perception.acquisition.PerceptionEvidenceSource;
import com.hytalerlbridge.policy.perception.model.PerceptionFrame;
import com.hytalerlbridge.policy.perception.profile.PerceptionProfile;
import com.hytalerlbridge.policy.perception.projection.AbilityProjection;
import com.hytalerlbridge.policy.perception.projection.InventoryProjection;
import com.hytalerlbridge.policy.perception.projection.GeometryProjection;
import com.hytalerlbridge.policy.perception.projection.LightProjection;
import com.hytalerlbridge.policy.perception.projection.MechanicsProjection;
import com.hytalerlbridge.policy.perception.projection.RecipeCandidateProjection;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Arrays;

/** Production composition: same-tick server evidence to a learner-v3 sample. */
public final class LivePolicyPerception implements PolicyPerception {

    private final PerceptionProfile profile;
    private final PerceptionEvidenceSource source;

    public LivePolicyPerception(
        PerceptionProfile profile,
        PerceptionEvidenceSource source
    ) {
        if (profile == null || source == null) {
            throw new IllegalArgumentException(
                "live perception requires a profile and evidence source");
        }
        this.profile = profile;
        this.source = source;
    }

    @Override
    public Sample sample(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        int slot,
        float deltaTime
    ) {
        PerceptionFrame frame = source.capture(
            ref, npc, store, slot, deltaTime);
        if (frame == null || !frame.valid()) {
            return null;
        }

        var movement = Projection.movementFeatures(
            frame.movementValueBits(), frame.movementAvailableBits());
        if (movement.invalid()) {
            return null;
        }
        var actorWorld = Projection.actorWorldFeatures(frame.actorWorld());
        var resources = Projection.resourceFeatures(
            profile.resources(
                frame.mechanics().resources(),
                frame.mechanics().resourceAvailable()));

        PerceptionFrame.Defense defenseRaw = frame.mechanics().defense();
        var defense = Projection.defenseFeatures(
            new MechanicsProjection.DefenseInput(
                defenseRaw.guardActive(),
                defenseRaw.staminaBroken(),
                defenseRaw.dodgeInvulnerabilityRemaining(),
                profile.dodgeInvulnerabilityDuration(),
                defenseRaw.appliedVelocity(),
                profile.dodgeForce(),
                defenseRaw.staminaRegenDelay(),
                defenseRaw.controlImmunity(),
                defenseRaw.health()
            )
        );

        PerceptionFrame.Status statusRaw = frame.mechanics().status();
        var statuses = Projection.statusFeatures(
            new MechanicsProjection.StatusInput(
                new float[] {
                    (float) profile.params().agentMaxHealth(),
                    (float) profile.params().targetMaxHealth(),
                },
                profile.resourceSpan(),
                statusRaw.remainingSeconds(),
                statusRaw.cycleElapsedSeconds(),
                statusRaw.cycleCooldownSeconds(),
                statusRaw.damagePerCycle(),
                statusRaw.healingPerCycle(),
                statusRaw.resourceId(),
                statusRaw.resourceDeltaPerCycle(),
                statusRaw.speedMultiplier(),
                statusRaw.active()
            )
        );

        PerceptionFrame.Ability abilityRaw = frame.mechanics().ability();
        AbilityProjection.Result abilities = Projection.abilityFeatures(
            profile.abilities(
                frame.mechanics().resources(),
                abilityRaw.remainingCooldownSeconds(),
                abilityRaw.activeSlot(),
                abilityRaw.elapsedSeconds(),
                abilityRaw.legal()
            )
        );
        boolean[] dodge = Projection.dodgeActionMask(profile.dodge(
            frame.mechanics().dodgeCorridorClear(),
            frame.mechanics().movementEnabled(),
            defenseRaw.health()[0] > 0.0f,
            frame.mechanics().stamina()
        ));
        InventoryProjection.Result inventory =
            Projection.inventoryFeatures(frame.inventory());

        GeometryProjection.Result geometry = Projection.geometryFeatures(
            frame.world().geometry());
        LightProjection.Input rawLight = frame.world().light();
        LightProjection.Result light = Projection.lightFeatures(
            new LightProjection.Input(
                rawLight.sourceAvailable(), rawLight.sourceTokenMask(),
                rawLight.lightValid(), rawLight.skyLight(),
                rawLight.blockLightRgb(), rawLight.tintRgb(),
                geometry.tokenMask(), geometry.available(), frame.valid()));

        float[] self = Projection.selfFeatures(frame.self(), profile.params());
        float[] target = Projection.targetFeatures(
            frame.target(), profile.params());
        float[] combat = Projection.combatFeatures(
            frame.combatSelf(),
            Projection.targetEvidence(frame.targetRaw(), profile.params()),
            profile.params()
        );

        PerceptionFrame.WorldGroups world = frame.world();
        RecipeCandidateProjection.Result recipes = RecipeCandidateProjection.project(
            profile.recipeCandidateEncoder(), world.recipeCandidates());
        PerceptionFrame.Actions actions = frame.actions();
        Map<String, float[]> floats = new LinkedHashMap<>();
        floats.put("base_self_f32", self);
        floats.put("base_target_f32", target);
        floats.put("base_combat_f32", combat);
        floats.put("resource_f32", resources.values());
        floats.put("defense_f32", defense.values());
        floats.put("status_f32", statuses.values());
        floats.put("ability_f32", abilities.values());
        floats.put("actor_world_f32", actorWorld.values());
        floats.put("movement_state_f32", movement.values());
        floats.put("geometry_token_f32", geometry.values());
        floats.put("light_token_f32", light.values());
        floats.put("inventory_container_f32", inventory.containerValues());
        floats.put("inventory_token_f32", inventory.tokenValues());
        floats.put("block_candidate_f32", world.blockCandidates());
        floats.put("recipe_embedding", recipes.candidateEmbedding());

        Map<String, boolean[]> masks = new LinkedHashMap<>();
        masks.put("base_target_mask", new boolean[] {frame.targetPresent()});
        masks.put("resource_mask", resources.mask());
        masks.put("status_mask", statuses.mask());
        masks.put("ability_mask", abilities.mask());
        masks.put("ability_legal", abilities.legal());
        masks.put("actor_world_mask", actorWorld.mask());
        masks.put("movement_state_mask", movement.mask());
        masks.put("geometry_token_mask", geometry.tokenMask());
        masks.put("geometry_available", new boolean[] {
            geometry.available()});
        masks.put("light_available", new boolean[] {light.available()});
        masks.put("inventory_container_mask", inventory.containerMask());
        masks.put("inventory_token_mask", inventory.tokenMask());
        masks.put("inventory_available", new boolean[] {inventory.available()});
        masks.put("skill_action_mask", actions.skill());
        masks.put("jump_action_mask", new boolean[] {actions.jump()});
        masks.put("guard_action_mask", new boolean[] {actions.guard()});
        masks.put("dodge_action_mask", dodge);
        masks.put("valid", new boolean[] {true});
        masks.put("block_candidate_mask", world.blockMask());
        masks.put("block_available", new boolean[] {world.blockAvailable()});
        masks.put("recipe_mask", recipes.candidateMask());
        masks.put("recipe_available", new boolean[] {recipes.available()});

        float[] observation = ObservationAssembler.assemble(
            profile.template(), floats, masks);
        boolean[] actionMask = ActionMaskAssembler.assemble(
            new ActionMaskAssembler.Input(
                true,
                actions.skill(),
                actions.door(),
                Arrays.copyOfRange(
                    abilities.legal(), 0, AbilityProjection.ABILITY_CAPACITY),
                actions.guard(),
                dodge,
                actions.jump(),
                actions.movement(),
                actions.look(),
                actions.use(),
                actions.blockTrigger(),
                world.blockAvailable(),
                world.blockMask(),
                actions.craft() && recipes.available(),
                actions.craft()
                    ? recipes.candidateMask()
                    : new boolean[RecipeCandidateProjection.CANDIDATE_CAPACITY]
            )
        );
        return new Sample(
            observation,
            actionMask,
            new WorldActionEvidence(
                world.recipeIds(), world.blockBindings())
        );
    }
}
