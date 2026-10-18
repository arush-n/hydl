package com.hytalerlbridge.policy.perception.acquisition;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.MovementStates;
import com.hypixel.hytale.server.core.entity.damage.DamageDataComponent;
import com.hypixel.hytale.server.core.entity.entities.Player;
import com.hypixel.hytale.server.core.entity.movement.MovementStatesComponent;
import com.hypixel.hytale.server.core.modules.entity.component.HeadRotation;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatMap;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatValue;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatsModule;
import com.hypixel.hytale.server.core.modules.entitystats.asset.EntityStatType;
import com.hypixel.hytale.server.core.modules.physics.component.Velocity;
import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.Universe;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.support.MarkedEntitySupport;
import com.hypixel.hytale.server.npc.role.support.PositionCache;
import com.hytalerlbridge.policy.perception.projection.MovementProjection;
import org.joml.Vector3d;

/** Public 0.5.7 component reads for one policy actor or target. */
public final class ServerActorReader {

    public static final float PERCEPTION_RADIUS = 24.0f;
    public static final String[] RESOURCE_STAT_NAMES = {
        "Stamina",
        "Mana",
        "MagicCharges",
        "SignatureEnergy",
        "SignatureCharges",
        "Ammo",
        "Oxygen",
    };

    /** Raw public component state, before checkpoint-specific scaling. */
    public record Actor(
        Ref<EntityStore> ref,
        NPCEntity npc,
        double x,
        double y,
        double z,
        double velocityX,
        double velocityY,
        double velocityZ,
        double yawDegrees,
        double pitchDegrees,
        double headYawDegrees,
        double headPitchDegrees,
        float health,
        float maximumHealth,
        float[] resourceValues,
        float[] resourceMaximums,
        boolean[] resourceAvailable,
        int movementBits,
        boolean guardActive,
        boolean staminaBroken,
        float staminaRegenDelay,
        float controlImmunity
    ) {
        public Actor {
            if (ref == null
                || resourceValues == null || resourceValues.length != 7
                || resourceMaximums == null || resourceMaximums.length != 7
                || resourceAvailable == null
                || resourceAvailable.length != 7
                || movementBits < 0
                || movementBits > MovementProjection.ALL_BITS) {
                throw new IllegalArgumentException("invalid actor evidence");
            }
            resourceValues = resourceValues.clone();
            resourceMaximums = resourceMaximums.clone();
            resourceAvailable = resourceAvailable.clone();
        }

        @Override public float[] resourceValues() {
            return resourceValues.clone();
        }

        @Override public float[] resourceMaximums() {
            return resourceMaximums.clone();
        }

        @Override public boolean[] resourceAvailable() {
            return resourceAvailable.clone();
        }

        public boolean movement(int index) {
            return (movementBits & (1 << index)) != 0;
        }
    }

    private ServerActorReader() {
    }

    /** Return {@code null} if a load-bearing public component is unavailable. */
    public static Actor capture(
        Ref<EntityStore> ref,
        Store<EntityStore> store
    ) {
        if (ref == null || !ref.isValid() || store == null) {
            return null;
        }
        NPCEntity npc = store.getComponent(ref, NPCEntity.getComponentType());
        Player player = store.getComponent(ref, Player.getComponentType());
        TransformComponent transform = store.getComponent(
            ref, TransformComponent.getComponentType());
        MovementStatesComponent movement = store.getComponent(
            ref, MovementStatesComponent.getComponentType());
        MovementStates states = movement == null
            ? null
            : movement.getMovementStates();
        if ((npc == null && player == null)
            || transform == null || states == null
            || transform.getPosition() == null
            || transform.getRotation() == null) {
            return null;
        }

        Velocity velocity = store.getComponent(ref, Velocity.getComponentType());
        HeadRotation head = store.getComponent(
            ref, HeadRotation.getComponentType());
        EntityStatMap stats = store.getComponent(
            ref, EntityStatsModule.get().getEntityStatMapComponentType());
        EntityStatValue healthStat = stat(stats, "Health");
        Role role = npc == null ? null : npc.getRole();
        float fallbackMaximum = role == null
            ? 0.0f
            : Math.max(0.0f, role.getInitialMaxHealth());
        float health = healthStat == null ? fallbackMaximum : healthStat.get();
        float maximumHealth = healthStat == null
            ? Math.max(health, fallbackMaximum)
            : healthStat.getMax();
        if (!Float.isFinite(health) || !Float.isFinite(maximumHealth)
            || health < 0.0f || maximumHealth <= 0.0f) {
            return null;
        }

        float[] resources = new float[RESOURCE_STAT_NAMES.length];
        float[] resourceMaximums = new float[RESOURCE_STAT_NAMES.length];
        boolean[] resourceAvailable = new boolean[RESOURCE_STAT_NAMES.length];
        for (int index = 0; index < RESOURCE_STAT_NAMES.length; index++) {
            EntityStatValue value = stat(stats, RESOURCE_STAT_NAMES[index]);
            if (value == null || !Float.isFinite(value.get())
                || !Float.isFinite(value.getMax())) {
                continue;
            }
            resources[index] = value.get();
            resourceMaximums[index] = value.getMax();
            resourceAvailable[index] = true;
        }

        Vector3d position = transform.getPosition();
        double yaw = normalizeDegrees(Math.toDegrees(
            transform.getRotation().yaw()));
        double pitch = normalizeDegrees(Math.toDegrees(
            transform.getRotation().pitch()));
        double headYaw = head == null || head.getRotation() == null
            ? yaw
            : normalizeDegrees(Math.toDegrees(head.getRotation().yaw()));
        double headPitch = head == null || head.getRotation() == null
            ? pitch
            : normalizeDegrees(Math.toDegrees(head.getRotation().pitch()));
        DamageDataComponent damage = store.getComponent(
            ref, DamageDataComponent.getComponentType());
        EntityStatValue stamina = stat(stats, "Stamina");
        EntityStatValue regenDelay = stat(stats, "StaminaRegenDelay");
        EntityStatValue immunity = stat(stats, "Immunity");

        return new Actor(
            ref,
            npc,
            position.x,
            position.y,
            position.z,
            velocity == null ? 0.0 : velocity.getX(),
            velocity == null ? 0.0 : velocity.getY(),
            velocity == null ? 0.0 : velocity.getZ(),
            yaw,
            pitch,
            headYaw,
            headPitch,
            health,
            maximumHealth,
            resources,
            resourceMaximums,
            resourceAvailable,
            movementBits(states),
            damage != null && damage.getCurrentWielding() != null,
            stamina != null && stamina.get() <= 0.0f,
            regenDelay == null ? 0.0f : finiteOrZero(regenDelay.get()),
            immunity == null ? 0.0f : finiteOrZero(immunity.get())
        );
    }

    /** Prefer the authored default NPC target slot, then the first valid slot. */
    public static Ref<EntityStore> selectTarget(
        Ref<EntityStore> self,
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        return selectTarget(self, npc, store, false);
    }

    /**
     * Select the policy target, optionally preferring the nearest connected
     * player in the same world store.
     *
     * <p>The player mode is deliberately explicit. It does not fall back to
     * native NPC target memory, because doing so would make a validation run
     * appear to follow the player while actually following a marked NPC.</p>
     */
    public static Ref<EntityStore> selectTarget(
        Ref<EntityStore> self,
        NPCEntity npc,
        Store<EntityStore> store,
        boolean followPlayer
    ) {
        if (followPlayer) {
            return nearestConnectedPlayer(self, store);
        }
        Role role = npc == null ? null : npc.getRole();
        MarkedEntitySupport marked = role == null
            ? null
            : store.getComponent(self, MarkedEntitySupport.getComponentType());
        if (marked == null) {
            return null;
        }
        Ref<EntityStore> preferred = marked.getMarkedEntityRef(
            MarkedEntitySupport.DEFAULT_TARGET_SLOT);
        if (usableTarget(self, preferred, store, false)) {
            return preferred;
        }
        Ref<EntityStore>[] targets = marked.getEntityTargets();
        if (targets == null) {
            return null;
        }
        for (Ref<EntityStore> candidate : targets) {
            if (usableTarget(self, candidate, store, false)) {
                return candidate;
            }
        }
        return null;
    }

    /** Native perception boundary: alive, within 24 blocks, and role LOS. */
    public static boolean perceptible(
        Actor self,
        Actor target,
        Store<EntityStore> store
    ) {
        if (self == null || target == null || target.health() <= 0.0f) {
            return false;
        }
        double dx = target.x() - self.x();
        double dz = target.z() - self.z();
        if (dx * dx + dz * dz
            > PERCEPTION_RADIUS * PERCEPTION_RADIUS) {
            return false;
        }
        if (self.npc() == null) {
            return false;
        }
        Role role = self.npc().getRole();
        PositionCache cache = role == null
            ? null
            : store.getComponent(self.ref(), PositionCache.getComponentType());
        if (cache == null) {
            return false;
        }
        try {
            return cache.hasLineOfSight(
                self.ref(), target.ref(), store);
        } catch (RuntimeException unavailable) {
            return false;
        }
    }

    private static boolean usableTarget(
        Ref<EntityStore> self,
        Ref<EntityStore> target,
        Store<EntityStore> store,
        boolean allowPlayer
    ) {
        if (target == null || !target.isValid() || store == null
            || sameEntity(self, target)) {
            return false;
        }
        boolean npc = target != null
            && store.getComponent(target, NPCEntity.getComponentType()) != null;
        boolean player = allowPlayer
            && target != null
            && store.getComponent(target, Player.getComponentType()) != null;
        return (npc || player)
            && store.getComponent(
                target, TransformComponent.getComponentType()) != null;
    }

    /** Select the nearest live player represented by the current world store. */
    private static Ref<EntityStore> nearestConnectedPlayer(
        Ref<EntityStore> self,
        Store<EntityStore> store
    ) {
        if (self == null || !self.isValid() || store == null) {
            return null;
        }
        TransformComponent selfTransform = store.getComponent(
            self, TransformComponent.getComponentType());
        if (selfTransform == null || selfTransform.getPosition() == null) {
            return null;
        }
        try {
            Universe universe = Universe.get();
            if (universe == null || universe.getPlayers() == null) {
                return null;
            }
            Ref<EntityStore> nearest = null;
            double nearestDistance = Double.POSITIVE_INFINITY;
            String nearestKey = null;
            for (PlayerRef playerRef : universe.getPlayers()) {
                if (playerRef == null || !playerRef.isValid()) {
                    continue;
                }
                Ref<EntityStore> candidate = playerRef.getReference();
                if (!usableTarget(self, candidate, store, true)) {
                    continue;
                }
                TransformComponent transform = store.getComponent(
                    candidate, TransformComponent.getComponentType());
                if (transform == null || transform.getPosition() == null) {
                    continue;
                }
                double dx = transform.getPosition().x
                    - selfTransform.getPosition().x;
                double dy = transform.getPosition().y
                    - selfTransform.getPosition().y;
                double dz = transform.getPosition().z
                    - selfTransform.getPosition().z;
                double distance = dx * dx + dy * dy + dz * dz;
                if (!Double.isFinite(distance)) {
                    continue;
                }
                String key = String.valueOf(playerRef.getUuid());
                boolean closer = distance < nearestDistance;
                boolean tied = distance == nearestDistance
                    && (nearestKey == null || key.compareTo(nearestKey) < 0);
                if (closer || tied) {
                    nearest = candidate;
                    nearestDistance = distance;
                    nearestKey = key;
                }
            }
            return nearest;
        } catch (RuntimeException unavailable) {
            // Target acquisition is an optional validation accommodation. A
            // transient Universe/player registry read must not kill the tick.
            return null;
        }
    }

    private static boolean sameEntity(
        Ref<EntityStore> left,
        Ref<EntityStore> right
    ) {
        return left != null && right != null
            && left.getStore() == right.getStore()
            && left.getIndex() == right.getIndex();
    }

    private static EntityStatValue stat(EntityStatMap map, String name) {
        if (map == null) {
            return null;
        }
        int index = EntityStatType.getAssetMap().getIndex(name);
        return index < 0 ? null : map.get(index);
    }

    private static float finiteOrZero(float value) {
        return Float.isFinite(value) ? value : 0.0f;
    }

    private static double normalizeDegrees(double value) {
        double shifted = value + 180.0;
        return shifted - 360.0 * Math.floor(shifted / 360.0) - 180.0;
    }

    static int movementBits(MovementStates value) {
        boolean[] fields = {
            value.idle,
            value.horizontalIdle,
            value.jumping,
            value.flying,
            value.walking,
            value.running,
            value.sprinting,
            value.crouching,
            value.forcedCrouching,
            value.falling,
            value.fallingFar,
            value.climbing,
            value.inFluid,
            value.swimming,
            value.swimJumping,
            value.onGround,
            value.mantling,
            value.sliding,
            value.mounting,
            value.rolling,
            value.sitting,
            value.gliding,
            value.sleeping,
        };
        int bits = 0;
        for (int index = 0; index < fields.length; index++) {
            if (fields[index]) {
                bits |= 1 << index;
            }
        }
        return bits;
    }
}
