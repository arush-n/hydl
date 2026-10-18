package com.hytalerlbridge.policy;

import com.hypixel.hytale.component.ArchetypeChunk;
import com.hypixel.hytale.component.CommandBuffer;
import com.hypixel.hytale.component.ComponentType;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.dependency.Dependency;
import com.hypixel.hytale.component.query.Query;
import com.hypixel.hytale.component.system.tick.EntityTickingSystem;
import com.hypixel.hytale.server.core.entity.Frozen;
import com.hypixel.hytale.server.core.modules.entity.component.HeadRotation;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.role.Role;
import com.hytalerlbridge.policy.perception.acquisition.resources.CatalogBackedActorStats;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

/**
 * Gives every NPC of a chosen role a {@link PolicyAgentMarker}, once.
 *
 * <p>This is what makes the hook apply to NPCs the server spawned by itself,
 * rather than only to entities an RL episode created. It queries NPCs that do
 * <i>not</i> already carry a marker, so each entity is claimed exactly once and
 * the system costs nothing after the population is marked.
 *
 * <p>The new marker's pose is seeded from the NPC's current orientation rather
 * than from zero. The policy emits look <i>deltas</i>, so a marker starting at
 * yaw 0 would make the NPC snap to facing north on its first controlled tick --
 * self-correcting, but visible, and easy to misread as the policy behaving
 * badly.
 *
 * <p>Slots are handed out in claim order and are only an identifier for logging
 * and for {@link PolicyPerception}; recurrent state lives on the marker, so two
 * NPCs never share memory regardless of slot.
 */
public final class PolicyAttachmentSystem extends EntityTickingSystem<EntityStore> {

    private static final System.Logger LOGGER =
        System.getLogger(PolicyAttachmentSystem.class.getName());

    private final ComponentType<EntityStore, PolicyAgentMarker> markerType;
    private final PolicyRuntime runtime;
    private final String roleName;
    private final Query<EntityStore> query;
    private final AtomicInteger nextSlot = new AtomicInteger();
    private final Map<String, Integer> seen = new ConcurrentHashMap<>();
    private final BooleanSupplier readyToClaim;

    /**
     * @param roleName the exact {@code Role.getRoleName()} to control, e.g.
     *                 {@code "Example_Role"}; {@code null} claims every NPC
     */
    /**
     * Suppress the role's own decision-making on claim.
     *
     * <p>{@code Frozen} suppresses the authored behaviour tick, but Hytale's
     * {@code SteeringSystem}, {@code ComputeVelocitySystem}, and other motion
     * systems are {@code SteppableTickingSystem}s and therefore stop too. The
     * marker records this mode so {@link PolicyControlSystem} can add one
     * {@code StepComponent} only after behaviour and avoidance have already
     * been skipped. Hytale's last-set {@code StepCleanupSystem} removes it at
     * the end of that tick. The result is policy-owned steering through the
     * native motion pipeline without letting the authored role brain compete.
     */
    private final boolean takeover;

    public PolicyAttachmentSystem(
        ComponentType<EntityStore, PolicyAgentMarker> markerType,
        PolicyRuntime runtime,
        String roleName
    ) {
        this(markerType, runtime, roleName, false);
    }

    public PolicyAttachmentSystem(
        ComponentType<EntityStore, PolicyAgentMarker> markerType,
        PolicyRuntime runtime,
        String roleName,
        boolean takeover
    ) {
        this(markerType, runtime, roleName, takeover, () -> true);
    }

    PolicyAttachmentSystem(
        ComponentType<EntityStore, PolicyAgentMarker> markerType,
        PolicyRuntime runtime,
        String roleName,
        boolean takeover,
        BooleanSupplier readyToClaim
    ) {
        if (readyToClaim == null) {
            throw new IllegalArgumentException("attachment readiness is required");
        }
        this.markerType = markerType;
        this.runtime = runtime;
        this.roleName = roleName;
        this.takeover = takeover;
        this.readyToClaim = readyToClaim;
        this.query = Query.and(
            NPCEntity.getComponentType(),
            Query.not(markerType)
        );
    }

    @Override
    public Query<EntityStore> getQuery() {
        return query;
    }

    @Override
    public Set<Dependency<EntityStore>> getDependencies() {
        return Set.of();
    }

    @Override
    public boolean isParallel(int entityCount, int chunkCount) {
        return false;
    }

    @Override
    public void tick(
        float deltaTime,
        int index,
        ArchetypeChunk<EntityStore> chunk,
        Store<EntityStore> store,
        CommandBuffer<EntityStore> commandBuffer
    ) {
        // Live profiles resolve interaction assets off the world thread. A
        // first claim before that warm-up completes lets perception or the
        // combat sink perform the first resolution from inside the world tick,
        // where the asset-store read-to-write lock upgrade deadlocks forever.
        if (!readyToClaim()) {
            return;
        }
        NPCEntity npc = chunk.getComponent(index, NPCEntity.getComponentType());
        if (npc == null) {
            return;
        }
        Role role = npc.getRole();
        if (role == null) {
            return;
        }
        String actual = role.getRoleName();
        if (!acceptsRole(roleName, actual)) {
            seen.merge(actual == null ? "(null)" : actual, 1, Integer::sum);
            return;
        }

        Ref<EntityStore> ref = chunk.getReferenceTo(index);
        // Claiming transfers control to this policy. Settle the actor's sparse
        // role stat map at that explicit boundary, before perception can
        // advertise item/resource-gated actions. A missing stat map is a loud
        // fail-closed capability error; it must not become a claimed actor
        // whose legal mask is silently incomplete.
        CatalogBackedActorStats.Settlement stats =
            CatalogBackedActorStats.settle(
                ref,
                store,
                CatalogBackedActorStats.Boundary.CONTROLLED_ACTOR_CLAIM
            );
        PolicyAgentMarker marker = new PolicyAgentMarker(
            runtime,
            allocateSlot(nextSlot),
            takeover
        );

        TransformComponent transform =
            store.getComponent(ref, TransformComponent.getComponentType());
        HeadRotation head = store.getComponent(ref, HeadRotation.getComponentType());
        marker.setPose(
            transform == null ? 0.0f : transform.getRotation().yaw(),
            head == null ? 0.0f : head.getRotation().pitch()
        );

        commandBuffer.addComponent(ref, markerType, marker);
        if (takeover) {
            commandBuffer.ensureComponent(ref, Frozen.getComponentType());
        }
        LOGGER.log(System.Logger.Level.INFO, () -> String.format(
            "PolicyAgent claimed NPC role=%s slot=%d yaw=%+.3f pitch=%+.3f%s",
            actual, marker.slot(), marker.desiredYaw(), marker.desiredPitch(),
            (takeover ? " [takeover: role AI frozen]" : "")
                + (stats.replaced()
                    ? String.format(" [catalog stats +%d]", stats.missingCatalogRows())
                    : " [catalog stats complete]")));
    }

    /** How many NPCs have been claimed so far. */
    public int claimed() {
        return nextSlot.get();
    }

    boolean readyToClaim() {
        return readiness(readyToClaim);
    }

    static boolean readiness(BooleanSupplier gate) {
        return gate.getAsBoolean();
    }

    /** Exact role predicate used by the live query and fixture preflight. */
    static boolean acceptsRole(String configuredRole, String actualRole) {
        return configuredRole == null || configuredRole.equals(actualRole);
    }

    /**
     * Allocate one process-local policy slot in serialized claim order.
     *
     * <p>The ECS system is deliberately non-parallel, so two matching actors
     * seen in one claim pass receive distinct consecutive slots without any
     * actor-, role-, item-, or profile-specific branch.</p>
     */
    static int allocateSlot(AtomicInteger nextSlot) {
        if (nextSlot == null) {
            throw new IllegalArgumentException("policy slot counter is required");
        }
        return nextSlot.getAndIncrement();
    }

    /** Roles this system saw but did not claim, for diagnosing an empty run. */
    public Map<String, Integer> unclaimedRoles() {
        return Map.copyOf(seen);
    }
}
