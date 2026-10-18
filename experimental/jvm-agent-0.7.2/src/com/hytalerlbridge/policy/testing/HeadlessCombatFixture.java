package com.hytalerlbridge.policy.testing;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.math.vector.Rotation3f;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.NPCPlugin;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.support.MarkedEntitySupport;
import com.hytalerlbridge.policy.PolicyActorLoadout;
import com.hytalerlbridge.policy.perception.acquisition.motion.DodgeMotionParameters;
import com.hytalerlbridge.policy.testing.placement.ServerCombatFixturePlacement;
import com.hytalerlbridge.policy.world.NativeWorldAccess;
import it.unimi.dsi.fastutil.Pair;
import org.joml.Vector3d;

/**
 * Creates the explicit playerless two-actor fixture used by live policy gates.
 *
 * <p>This is not production loadout logic. It runs only when the operator puts
 * {@code combat-fixture.txt} beside the checkpoint. Item identities come from
 * the checkpoint profile, both held items are verified after installation, and
 * the policy actor's native target slot is verified before control can begin.</p>
 */
public final class HeadlessCombatFixture {

    private HeadlessCombatFixture() {}

    public static Result spawn(
        World world,
        Store<EntityStore> store,
        HeadlessCombatFixtureSpec spec,
        String agentItemId,
        String targetItemId,
        DodgeMotionParameters dodgeMotion
    ) {
        requireItem(agentItemId, "agent");
        requireItem(targetItemId, "target");
        if (dodgeMotion == null) {
            throw new IllegalArgumentException(
                "combat fixture requires checkpoint Dodge motion");
        }
        ServerCombatFixturePlacement.Placement placement =
            ServerCombatFixturePlacement.find(
                world, store, spec, dodgeMotion);
        if (!placement.ready()) {
            throw new IllegalStateException(
                "combat fixture placement failed: " + placement.receipt());
        }
        Vector3d agentPosition = placement.agentPosition();
        Vector3d targetPosition = placement.targetPosition();
        long agentChunk = ChunkUtil.indexChunkFromBlock(
            agentPosition.x, agentPosition.z);
        long targetChunk = ChunkUtil.indexChunkFromBlock(
            targetPosition.x, targetPosition.z);
        if (NativeWorldAccess.getChunk(world, agentChunk) == null
            || NativeWorldAccess.getChunk(world, targetChunk) == null) {
            throw new IllegalStateException("combat fixture chunks did not load");
        }

        Ref<EntityStore> agent = spawnActor(
            store,
            spec.agentRole(),
            agentPosition,
            0.0f
        );
        Ref<EntityStore> target = spawnActor(
            store,
            spec.targetRole(),
            targetPosition,
            (float) Math.PI
        );

        equip(store, agent, agentItemId, "agent");
        equip(store, target, targetItemId, "target");
        String verification = ServerCombatFixturePlacement.verifySpawned(
            world,
            store,
            agent,
            target,
            agentPosition,
            targetPosition,
            dodgeMotion
        );
        if (!verification.isEmpty()) {
            throw new IllegalStateException(
                "spawned combat fixture failed exact validation: "
                    + verification);
        }
        linkTarget(store, agent, target);
        return new Result(
            agent,
            target,
            agentChunk,
            targetChunk,
            new Vector3d(agentPosition),
            new Vector3d(targetPosition),
            placement.receipt()
        );
    }

    private static Ref<EntityStore> spawnActor(
        Store<EntityStore> store,
        String role,
        Vector3d position,
        float yaw
    ) {
        Pair<Ref<EntityStore>, ?> spawned = NPCPlugin.get().spawnNPC(
            store,
            role,
            null,
            position,
            new Rotation3f(0.0f, yaw, 0.0f)
        );
        if (spawned == null || spawned.left() == null) {
            throw new IllegalStateException("server rejected fixture role " + role);
        }
        Ref<EntityStore> ref = spawned.left();
        NPCEntity npc = store.getComponent(ref, NPCEntity.getComponentType());
        if (npc == null) {
            throw new IllegalStateException("fixture role has no NPCEntity: " + role);
        }
        npc.setDespawnTime(Float.POSITIVE_INFINITY);
        return ref;
    }

    private static void equip(
        Store<EntityStore> store,
        Ref<EntityStore> ref,
        String itemId,
        String actor
    ) {
        try {
            PolicyActorLoadout.install(
                ref,
                store,
                itemId,
                com.hytalerlbridge.policy.perception.acquisition.resources
                    .CatalogBackedActorStats.Boundary.FIXTURE_EQUIP
            );
        } catch (RuntimeException failure) {
            throw new IllegalStateException(
                "fixture " + actor + " loadout failed: " + failure,
                failure
            );
        }
        /*
         * Role stat maps can be sparse. PolicyActorLoadout settles the active
         * item's requirements through Hytale's catalog-backed clone rather
         * than special-casing a resource used by this diagnostic.
         */
    }

    private static void linkTarget(
        Store<EntityStore> store,
        Ref<EntityStore> agent,
        Ref<EntityStore> target
    ) {
        NPCEntity npc = store.getComponent(agent, NPCEntity.getComponentType());
        Role role = npc == null ? null : npc.getRole();
        MarkedEntitySupport marked = role == null
            ? null
            : store.getComponent(agent, MarkedEntitySupport.getComponentType());
        if (role == null || marked == null) {
            throw new IllegalStateException(
                "fixture agent role has no native target memory"
            );
        }
        role.setMarkedTarget(agent, store, MarkedEntitySupport.DEFAULT_TARGET_SLOT, target);
        Ref<EntityStore> selected = marked.getMarkedEntityRef(
            MarkedEntitySupport.DEFAULT_TARGET_SLOT
        );
        if (!sameEntity(selected, target)) {
            throw new IllegalStateException("fixture target link was not retained");
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

    private static void requireItem(String itemId, String actor) {
        if (itemId == null || itemId.isBlank()) {
            throw new IllegalArgumentException(
                "checkpoint profile has no " + actor + " item binding"
            );
        }
    }

    public record Result(
        Ref<EntityStore> agent,
        Ref<EntityStore> target,
        long agentChunkKey,
        long targetChunkKey,
        Vector3d agentPosition,
        Vector3d targetPosition,
        String placementReceipt
    ) {
        public Result {
            agentPosition = new Vector3d(agentPosition);
            targetPosition = new Vector3d(targetPosition);
        }

        @Override public Vector3d agentPosition() {
            return new Vector3d(agentPosition);
        }

        @Override public Vector3d targetPosition() {
            return new Vector3d(targetPosition);
        }
    }
}
