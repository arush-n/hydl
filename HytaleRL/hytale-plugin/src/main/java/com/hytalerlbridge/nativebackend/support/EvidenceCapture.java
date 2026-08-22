package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.asset.type.entityeffect.config.EntityEffect;
import com.hypixel.hytale.server.core.entity.effect.ActiveEntityEffect;
import com.hypixel.hytale.server.core.entity.effect.EffectControllerComponent;
import com.hypixel.hytale.server.core.inventory.Inventory;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.modules.entity.component.EntityScaleComponent;
import com.hypixel.hytale.server.core.modules.entity.component.ModelComponent;
import com.hypixel.hytale.server.core.entity.movement.MovementStatesComponent;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatMap;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatValue;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatsModule;
import com.hypixel.hytale.server.core.modules.blockset.BlockSetModule;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.NPCPlugin;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.builders.BuilderRole;
import com.hypixel.hytale.server.npc.role.builders.BuilderRoleVariant;
import com.hytalerlbridge.geometry.GeometryCell;
import com.hytalerlbridge.geometry.GeometryContract;
import com.hytalerlbridge.geometry.GeometryFrame;
import com.hytalerlbridge.observation.MovementStateFrame;
import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import com.hytalerlbridge.nativebackend.model.CapturedRoleOpacity;
import com.hytalerlbridge.nativebackend.model.CapturedStatuses;
import com.hytalerlbridge.nativebackend.model.EntityModelEvidence;
import com.hypixel.hytale.server.core.entity.damage.DamageDataComponent;
import com.hypixel.hytale.server.core.entity.knockback.KnockbackComponent;
import com.hypixel.hytale.server.core.modules.entity.component.HeadRotation;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.physics.component.Velocity;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.npc.movement.controllers.MotionController;
import com.hytalerlbridge.combat.dodge.NativeDodgeProgram;
import com.hytalerlbridge.nativebackend.model.CapturedActorEvidence;
import org.joml.Vector3d;
import static com.hytalerlbridge.nativebackend.support.SupportMath.normalizeDegrees;

/**
 * Extracted verbatim from {@code NativeEnvironmentSession}.
 *
 * <p>Every method here touches no session instance field, so the move
 * required no state surgery.
 */
public final class EvidenceCapture {

    private EvidenceCapture() {}

    /**
     * Capture the role-specific half of PositionCache's opacity predicate.
     *
     * <p>{@link GeometryCell#flags()} already carries the independent base
     * opacity test. This row contains only membership in the reset-pinned
     * role's authored opaque BlockSet, which is the second term used by
     * {@code PositionCache.hasLineOfSightInternal}. Missing builder or
     * BlockSet evidence is unavailable rather than approximated.</p>
     */
    public static CapturedRoleOpacity captureRoleOpacity(
        Store<EntityStore> store,
        Ref<EntityStore> reference,
        GeometryFrame geometry
    ) {
        boolean[] cellMask = new boolean[GeometryContract.CELL_COUNT];
        if (
            !geometry.available()
                || reference == null
                || !reference.isValid()
        ) {
            return new CapturedRoleOpacity(false, cellMask);
        }
        NPCEntity npc = store.getComponent(
            reference,
            NPCEntity.getComponentType()
        );
        if (npc == null) return new CapturedRoleOpacity(false, cellMask);
        try {
            var plugin = NPCPlugin.get();
            var manager = plugin.getBuilderManager();
            Object rawBuilder = plugin.tryGetCachedValidRole(npc.getRoleIndex());
            for (
                int depth = 0;
                rawBuilder instanceof BuilderRoleVariant variant && depth < 16;
                depth++
            ) {
                rawBuilder = manager.getCachedBuilder(
                    variant.getReferenceIndex(),
                    Role.class
                );
            }
            if (!(rawBuilder instanceof BuilderRole builder)) {
                return new CapturedRoleOpacity(false, cellMask);
            }
            int blockSetIndex = builder.getOpaqueBlockSet();
            IntSet opaqueSet = null;
            if (blockSetIndex >= 0) {
                var blockSets = BlockSetModule.getInstance().getBlockSets();
                opaqueSet = blockSets.get(blockSetIndex);
                if (opaqueSet == null) {
                    return new CapturedRoleOpacity(false, cellMask);
                }
            }
            for (GeometryCell cell : geometry.cells()) {
                if (
                    opaqueSet != null
                        && cell.runtimeBlockId() != 0
                        && opaqueSet.contains(cell.runtimeBlockId())
                ) {
                    cellMask[GeometryContract.cellIndex(
                        cell.dx(),
                        cell.dy(),
                        cell.dz()
                    )] = true;
                }
            }
            return new CapturedRoleOpacity(true, cellMask);
        } catch (RuntimeException unavailable) {
            return new CapturedRoleOpacity(false, cellMask);
        }
    }

    public static CapturedStatuses captureStatuses(
        Store<EntityStore> store,
        Ref<EntityStore> ref
    ) {
        EffectControllerComponent controller = store.getComponent(
            ref,
            EffectControllerComponent.getComponentType()
        );
        ActiveEntityEffect[] active = controller == null
            ? null
            : controller.getAllActiveEntityEffects();
        if (active == null || active.length == 0) {
            return new CapturedStatuses(List.of(), false, false);
        }
        Arrays.sort(
            active,
            Comparator.comparingInt(ActiveEntityEffect::getEntityEffectIndex)
        );
        boolean overflow =
            active.length > NativeActorEvidenceFrame.STATUS_CAPACITY;
        boolean invalid = false;
        List<NativeActorEvidenceFrame.Status> statuses = new ArrayList<>(
            Math.min(active.length, NativeActorEvidenceFrame.STATUS_CAPACITY)
        );
        for (
            int index = 0;
            index < active.length
                && statuses.size() < NativeActorEvidenceFrame.STATUS_CAPACITY;
            index++
        ) {
            ActiveEntityEffect value = active[index];
            EntityEffect asset = EntityEffect.getAssetMap().getAsset(
                value.getEntityEffectIndex()
            );
            if (asset == null || asset.getId() == null || asset.getId().isBlank()) {
                invalid = true;
                continue;
            }
            statuses.add(new NativeActorEvidenceFrame.Status(
                value.getEntityEffectIndex(),
                asset.getId(),
                Math.max(0.0, value.getInitialDuration()),
                Math.max(0.0, value.getRemainingDuration()),
                value.isInfinite(),
                value.isDebuff(),
                value.isInvulnerable()
            ));
        }
        return new CapturedStatuses(List.copyOf(statuses), overflow, invalid);
    }

    public static EntityModelEvidence captureEntityModelEvidence(
        Store<EntityStore> store,
        Ref<EntityStore> reference
    ) {
        if (reference == null || !reference.isValid()) {
            return EntityModelEvidence.absent();
        }
        ModelComponent modelComponent = store.getComponent(
            reference,
            ModelComponent.getComponentType()
        );
        var model = modelComponent == null ? null : modelComponent.getModel();
        BoundingBox bounds = store.getComponent(
            reference,
            BoundingBox.getComponentType()
        );
        EntityScaleComponent entityScale = store.getComponent(
            reference,
            EntityScaleComponent.getComponentType()
        );
        String modelAssetId = model == null || model.getModelAssetId() == null
            ? ""
            : model.getModelAssetId();
        return new EntityModelEvidence(
            model != null,
            modelAssetId,
            model == null ? 0.0 : model.getScale(),
            model == null ? 0.0 : model.getEyeHeight(),
            bounds != null && bounds.getBoundingBox() != null,
            entityScale != null,
            entityScale == null ? 0.0 : entityScale.getScale()
        );
    }

    public static NativeActorEvidenceFrame.Actor absentActor(int entityId) {
        return new NativeActorEvidenceFrame.Actor(
            entityId,
            false,
            false,
            "",
            "",
            0,
            -1,
            new double[3],
            new double[3],
            NativeActorEvidenceFrame.MotionForce.unavailable(),
            0.0,
            0.0,
            0.0,
            0.0,
            new double[NativeActorEvidenceFrame.RESOURCE_COUNT],
            new double[NativeActorEvidenceFrame.RESOURCE_COUNT],
            new boolean[NativeActorEvidenceFrame.RESOURCE_COUNT],
            new double[NativeActorEvidenceFrame.DEFENSE_VALUE_COUNT],
            new boolean[NativeActorEvidenceFrame.DEFENSE_VALUE_COUNT],
            List.of(),
            new double[NativeActorEvidenceFrame.ACTOR_WORLD_VALUE_COUNT],
            new boolean[NativeActorEvidenceFrame.ACTOR_WORLD_MASK_COUNT],
            MovementStateFrame.unavailable()
        );
    }

    public static MovementStateFrame captureMovementStates(
        Store<EntityStore> store,
        Ref<EntityStore> ref
    ) {
        MovementStatesComponent component = store.getComponent(
            ref,
            MovementStatesComponent.getComponentType()
        );
        return MovementStateFrame.from(
            component == null ? null : component.getMovementStates()
        );
    }

    public static double entityHealth(
        Store<EntityStore> store,
        Ref<EntityStore> reference,
        NPCEntity entity
    ) {
        EntityStatMap stats = store.getComponent(
            reference,
            EntityStatsModule.get().getEntityStatMapComponentType()
        );
        EntityStatValue value = stat(stats, "Health");
        Role role = entity.getRole();
        return value == null
            ? (role == null ? 0.0 : role.getInitialMaxHealth())
            : value.get();
    }

    public static EntityStatValue stat(EntityStatMap map, String name) {
        return map == null ? null : map.get(name);
    }

    public static String activeItemId(Inventory inventory) {
        if (inventory == null || inventory.getHotbar() == null) return "";
        int slot = Byte.toUnsignedInt(inventory.getActiveHotbarSlot());
        int capacity = Short.toUnsignedInt(inventory.getHotbar().getCapacity());
        if (slot < 0 || slot >= capacity) return "";
        ItemStack stack = inventory.getHotbar().getItemStack((short) slot);
        return ItemStack.isEmpty(stack) ? "" : stack.getItemId();
    }

    public static int canonicalEntityType(String roleName) {
        if (roleName == null) return 999;
        if (roleName.startsWith("Trork")) return 0;
        if (roleName.startsWith("Outlander")) return 1;
        if (roleName.startsWith("Scarak")) return 2;
        if (roleName.startsWith("Kweebec")) return 3;
        return 5 + Math.floorMod(roleName.hashCode(), 995);
    }

    public static boolean sameEntity(Ref<EntityStore> left, Ref<EntityStore> right) {
        return left != null && right != null
            && left.getStore() == right.getStore()
            && left.getIndex() == right.getIndex();
    }

    public static CapturedActorEvidence captureActorEvidence(
        Store<EntityStore> store,
        Ref<EntityStore> ref,
        int entityId,
        boolean perceptible,
        MovementStateFrame movementStates,
        int activeAbilitySlot,
        Map<String, Integer> itemIds,
        List<String> resourceStatIds,
        // True only for the actor the learner controls. Used solely to decide
        // whether `locomotion_stamina_fraction` may carry a real value; see the
        // defense slot 7 comment below.
        boolean learnerControlled
    ) {
        if (ref == null || !ref.isValid() || !perceptible) {
            return new CapturedActorEvidence(absentActor(entityId), false, false);
        }
        TransformComponent transform = store.getComponent(
            ref,
            TransformComponent.getComponentType()
        );
        NPCEntity npc = store.getComponent(ref, NPCEntity.getComponentType());
        if (transform == null || npc == null) {
            return new CapturedActorEvidence(absentActor(entityId), false, true);
        }
        Velocity velocity = store.getComponent(ref, Velocity.getComponentType());
        HeadRotation head = store.getComponent(
            ref,
            HeadRotation.getComponentType()
        );
        EntityStatMap stats = store.getComponent(
            ref,
            EntityStatsModule.get().getEntityStatMapComponentType()
        );
        EntityStatValue healthStat = stat(stats, "Health");
        Role role = npc.getRole();
        double health = healthStat == null
            ? (role == null ? 0.0 : role.getInitialMaxHealth())
            : healthStat.get();
        double maxHealth = healthStat == null
            ? (role == null ? Math.max(0.0, health) : role.getInitialMaxHealth())
            : healthStat.getMax();
        double[] resourceValues = new double[
            NativeActorEvidenceFrame.RESOURCE_COUNT
        ];
        double[] resourceMaximums = new double[
            NativeActorEvidenceFrame.RESOURCE_COUNT
        ];
        boolean[] resourceAvailable = new boolean[
            NativeActorEvidenceFrame.RESOURCE_COUNT
        ];
        for (int index = 0; index < resourceStatIds.size(); index++) {
            String statName = resourceStatIds.get(index);
            EntityStatValue value = statName == null ? null : stat(stats, statName);
            if (value == null) continue;
            resourceValues[index] = value.get();
            resourceMaximums[index] = value.getMax();
            resourceAvailable[index] = true;
        }

        CapturedStatuses capturedStatuses = captureStatuses(store, ref);
        double[] defenseValues = new double[
            NativeActorEvidenceFrame.DEFENSE_VALUE_COUNT
        ];
        boolean[] defenseAvailable = new boolean[
            NativeActorEvidenceFrame.DEFENSE_VALUE_COUNT
        ];
        DamageDataComponent damage = store.getComponent(
            ref,
            DamageDataComponent.getComponentType()
        );
        defenseValues[0] = damage != null && damage.getCurrentWielding() != null
            ? 1.0
            : 0.0;
        defenseAvailable[0] = true;
        EntityStatValue stamina = stat(stats, "Stamina");
        defenseValues[1] = stamina != null && stamina.get() <= 0.0 ? 1.0 : 0.0;
        // A missing stamina stat cannot enter the stamina-broken state.
        defenseAvailable[1] = true;
        double dodgeInvulnerabilityRemainingSeconds = 0.0;
        for (NativeActorEvidenceFrame.Status status :
            capturedStatuses.statuses()) {
            if (
                NativeDodgeProgram.INVULNERABILITY_EFFECT_ID.equals(
                    status.effectId()
                )
            ) {
                dodgeInvulnerabilityRemainingSeconds = Math.max(
                    dodgeInvulnerabilityRemainingSeconds,
                    status.remainingDurationSeconds()
                );
            }
        }
        defenseValues[2] = dodgeInvulnerabilityRemainingSeconds;
        defenseAvailable[2] = true;
        KnockbackComponent knockback = store.getComponent(
            ref,
            KnockbackComponent.getComponentType()
        );
        MotionController controller = role == null
            ? null
            : role.getActiveMotionController();
        NativeActorEvidenceFrame.MotionForce motionForce =
            NativeMotionForceCapture.capture(controller, knockback);
        double[] projectedForce = motionForce.projectedVelocity();
        defenseValues[3] = Math.sqrt(
            projectedForce[0] * projectedForce[0]
                + projectedForce[1] * projectedForce[1]
                + projectedForce[2] * projectedForce[2]
        );
        defenseAvailable[3] = motionForce.projectedAvailable();
        EntityStatValue staminaDelay = stat(stats, "StaminaRegenDelay");
        defenseValues[4] = staminaDelay == null ? 0.0 : staminaDelay.get();
        defenseAvailable[4] = true;
        EntityStatValue immunity = stat(stats, "Immunity");
        defenseValues[5] = immunity == null ? 0.0 : immunity.get();
        defenseAvailable[5] = true;
        defenseValues[6] = health > 0.0 ? 1.0 : 0.0;
        defenseAvailable[6] = true;
        // `locomotion_stamina_fraction`, mirroring observation/v3/encoding/
        // encoder.py 240-249 exactly:
        //   self  -> clip(locomotion_stamina / PLAYER_STAMINA_MAXIMUM, -1, 1)
        //   other -> 1.0
        // The constant is PLAYER_STAMINA_MAXIMUM = 10.0 (combat/types.py 187),
        // and the range is signed because stamina floors at -4, not 0.
        //
        // The 1.0 for non-learner actors is NOT a placeholder and must not be
        // "improved" into the real value: the sim withholds it deliberately
        // because an opponent's sprint budget is not observable in Hytale, and
        // publishing it here would hand the policy privileged information. The
        // native decoder (native/codec/decode.py) passes this row straight
        // through without re-masking, so this is the only place that rule can
        // be enforced on the native path.
        if (learnerControlled) {
            double locomotionStamina = stamina == null ? 0.0 : stamina.get();
            defenseValues[7] = Math.max(
                -1.0,
                Math.min(1.0, locomotionStamina / 10.0)
            );
        } else {
            defenseValues[7] = 1.0;
        }
        defenseAvailable[7] = true;

        double[] actorWorldValues = new double[
            NativeActorEvidenceFrame.ACTOR_WORLD_VALUE_COUNT
        ];
        boolean[] actorWorldAvailable = new boolean[
            NativeActorEvidenceFrame.ACTOR_WORLD_MASK_COUNT
        ];
        actorWorldValues[0] = movementStates.state(12) ? 1.0 : 0.0;
        actorWorldAvailable[0] = movementStates.available();
        // Exact feet/eyes submersion and drop support remain World-produced.
        actorWorldAvailable[1] = false;
        actorWorldAvailable[2] = false;

        Vector3d position = transform.getPosition();
        String itemId = activeItemId(npc.getInventory());
        int runtimeItemIndex = itemId.isEmpty()
            ? 0
            : itemIds.getOrDefault(itemId, 0);
        double yawDegrees = normalizeDegrees(
            Math.toDegrees(transform.getRotation().yaw())
        );
        double pitchDegrees = Math.toDegrees(
            head == null
                ? transform.getRotation().pitch()
                : head.getRotation().pitch()
        );
        return new CapturedActorEvidence(
            new NativeActorEvidenceFrame.Actor(
                entityId,
                true,
                true,
                npc.getRoleName(),
                itemId,
                runtimeItemIndex,
                activeAbilitySlot,
                new double[] {position.x, position.y, position.z},
                new double[] {
                    velocity == null ? 0.0 : velocity.getX(),
                    velocity == null ? 0.0 : velocity.getY(),
                    velocity == null ? 0.0 : velocity.getZ()
                },
                motionForce,
                yawDegrees,
                pitchDegrees,
                health,
                maxHealth,
                resourceValues,
                resourceMaximums,
                resourceAvailable,
                defenseValues,
                defenseAvailable,
                capturedStatuses.statuses(),
                actorWorldValues,
                actorWorldAvailable,
                movementStates
            ),
            capturedStatuses.overflow(),
            capturedStatuses.invalid()
        );
    }

}
