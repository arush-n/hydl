package com.hytalerlbridge.policy.testing.placement;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.shape.Box;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.math.vector.Transform;
import com.hypixel.hytale.server.core.asset.type.model.config.Model;
import com.hypixel.hytale.server.core.modules.collision.CollisionModule;
import com.hypixel.hytale.server.core.modules.collision.CollisionResult;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.spawn.ISpawnProvider;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.NPCPlugin;
import com.hypixel.hytale.server.npc.asset.builder.Builder;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.spawning.ISpawnableWithModel;
import com.hypixel.hytale.server.spawning.SpawnTestResult;
import com.hypixel.hytale.server.spawning.SpawningContext;
import com.hytalerlbridge.policy.perception.acquisition.motion.DodgeCorridorEvaluator;
import com.hytalerlbridge.policy.perception.acquisition.motion.DodgeMotionParameters;
import com.hytalerlbridge.policy.perception.acquisition.motion.ServerDodgeCorridorReader;
import com.hytalerlbridge.policy.testing.HeadlessCombatFixtureSpec;
import com.hytalerlbridge.policy.world.NativeWorldAccess;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;
import java.nio.charset.StandardCharsets;
import java.util.UUID;
import org.joml.Vector3d;

/** Public-engine safe-placement adapter for the explicit combat fixture. */
public final class ServerCombatFixturePlacement {

    private static final int EXPECTED_TIMING_PROFILE = 0;

    public enum Reason {
        READY,
        WORLD_SPAWN_UNAVAILABLE,
        ROLE_UNAVAILABLE,
        MODEL_UNAVAILABLE,
        COLLISION_UNAVAILABLE,
        SEARCH_EXHAUSTED,
        ACTUAL_ACTOR_INVALID,
    }

    public record Placement(
        Reason reason,
        Vector3d agentPosition,
        Vector3d targetPosition,
        int attempts,
        int offsetX,
        int offsetZ,
        String detail
    ) {
        public Placement {
            if (reason == null || agentPosition == null
                || targetPosition == null || attempts < 0 || detail == null) {
                throw new IllegalArgumentException(
                    "invalid combat fixture placement");
            }
        }

        public boolean ready() {
            return reason == Reason.READY;
        }

        public String receipt() {
            return String.format(
                Locale.ROOT,
                "schema=hytalerl_fixture_placement_v1 reason=%s attempts=%d "
                    + "offset=%d,%d agent=%.3f,%.3f,%.3f "
                    + "target=%.3f,%.3f,%.3f detail=%s",
                reason.name().toLowerCase(Locale.ROOT),
                attempts,
                offsetX,
                offsetZ,
                agentPosition.x,
                agentPosition.y,
                agentPosition.z,
                targetPosition.x,
                targetPosition.y,
                targetPosition.z,
                detail.isBlank() ? "none" : detail.replace(' ', '_')
            );
        }
    }

    private ServerCombatFixturePlacement() {}

    /** Resolve a deterministic operator/spawn-provider origin for chunk pinning. */
    public static Vector3d previewOrigin(
        World world,
        HeadlessCombatFixtureSpec spec
    ) {
        if (world == null || spec == null) {
            throw new IllegalArgumentException(
                "world and fixture specification are required");
        }
        if (spec.origin() == HeadlessCombatFixtureSpec.Origin.EXPLICIT) {
            return new Vector3d(spec.x(), spec.y(), spec.z());
        }
        ISpawnProvider provider = world.getWorldConfig().getSpawnProvider();
        if (provider == null) {
            throw new IllegalStateException(
                "fixture world has no authored spawn provider");
        }
        Transform point = NativeWorldAccess.waitForFutureWithoutLock(
            world,
            provider.getSpawnPointAsync(world, fixtureUuid(spec))
        );
        if (point == null || !valid(point.getPosition())) {
            throw new IllegalStateException(
                "fixture spawn provider exposes no finite spawn point");
        }
        return new Vector3d(point.getPosition());
    }

    /**
     * Chunks covering the bounded ring, target separation and full Dodge path.
     * A one-chunk collision halo avoids a model/sweep lookup crossing into an
     * unpinned neighboring chunk; exact acceptance still comes from collision
     * evidence after loading.
     */
    public static long[] requiredChunkKeys(
        Vector3d origin,
        HeadlessCombatFixtureSpec spec,
        DodgeMotionParameters parameters
    ) {
        if (!valid(origin) || spec == null || parameters == null) {
            throw new IllegalArgumentException(
                "fixture chunk bounds require origin/spec/motion");
        }
        DodgeCorridorEvaluator.MotionPath path =
            DodgeCorridorEvaluator.integrate(0.0, 0L, parameters, true);
        if (!path.complete()) {
            throw new IllegalStateException(
                "fixture Dodge path did not complete before chunk pinning");
        }
        double corridor = Math.max(
            Math.abs(path.displacement().x()),
            Math.abs(path.displacement().z()));
        int reach = spec.searchRadius() + (int) Math.ceil(corridor);
        int minBlockX = (int) Math.floor(origin.x) - reach;
        int maxBlockX = (int) Math.floor(origin.x) + reach;
        int minBlockZ = (int) Math.floor(
            origin.z - spec.separation()) - reach;
        int maxBlockZ = (int) Math.floor(origin.z) + reach;
        int minChunkX = ChunkUtil.chunkCoordinate(minBlockX) - 1;
        int maxChunkX = ChunkUtil.chunkCoordinate(maxBlockX) + 1;
        int minChunkZ = ChunkUtil.chunkCoordinate(minBlockZ) - 1;
        int maxChunkZ = ChunkUtil.chunkCoordinate(maxBlockZ) + 1;
        Set<Long> keys = new LinkedHashSet<>();
        for (int chunkX = minChunkX; chunkX <= maxChunkX; chunkX++) {
            for (int chunkZ = minChunkZ; chunkZ <= maxChunkZ; chunkZ++) {
                keys.add(ChunkUtil.indexChunk(chunkX, chunkZ));
            }
        }
        return keys.stream().mapToLong(Long::longValue).toArray();
    }

    /** Search and validate both maximum-scale role AABBs before spawning. */
    public static Placement find(
        World world,
        Store<EntityStore> store,
        HeadlessCombatFixtureSpec spec,
        DodgeMotionParameters parameters
    ) {
        if (world == null || store == null || spec == null
            || parameters == null) {
            return failed(
                Reason.COLLISION_UNAVAILABLE,
                "placement_input_unavailable");
        }
        if (parameters.motionTimingProfile() != EXPECTED_TIMING_PROFILE) {
            return failed(
                Reason.COLLISION_UNAVAILABLE,
                "timing_profile_requires_reset_local_clock");
        }
        CollisionModule collision = CollisionModule.get();
        if (collision == null || collision.isDisabled()) {
            return failed(
                Reason.COLLISION_UNAVAILABLE,
                "collision_module_unavailable_or_disabled");
        }
        Vector3d origin;
        try {
            origin = previewOrigin(world, spec);
        } catch (RuntimeException | LinkageError unavailable) {
            return failed(
                Reason.WORLD_SPAWN_UNAVAILABLE,
                unavailable.getClass().getSimpleName());
        }
        SpawningContext agent = context(spec.agentRole());
        SpawningContext target = context(spec.targetRole());
        if (agent == null || target == null) {
            release(agent);
            release(target);
            return failed(Reason.ROLE_UNAVAILABLE, "role_not_spawnable");
        }
        try {
            FixturePlacementPlanner.Result search =
                FixturePlacementPlanner.search(
                    spec.searchRadius(),
                    (offsetX, offsetZ) -> evaluate(
                        world,
                        store,
                        collision,
                        agent,
                        target,
                        origin,
                        spec,
                        parameters,
                        offsetX,
                        offsetZ
                    )
                );
            if (search.reason()
                    != FixturePlacementPlanner.Reason.READY) {
                return new Placement(
                    Reason.SEARCH_EXHAUSTED,
                    new Vector3d(Double.NaN),
                    new Vector3d(Double.NaN),
                    search.attempts(),
                    0,
                    0,
                    search.lastRejection()
                );
            }
            FixturePlacementPlanner.Candidate value = search.candidate();
            return new Placement(
                Reason.READY,
                new Vector3d(
                    value.agentX(), value.agentY(), value.agentZ()),
                new Vector3d(
                    value.targetX(), value.targetY(), value.targetZ()),
                search.attempts(),
                search.offsetX(),
                search.offsetZ(),
                value.detail()
            );
        } finally {
            agent.releaseFull();
            target.releaseFull();
        }
    }

    /** Verify the random-scale spawned boxes before the attachment gate opens. */
    public static String verifySpawned(
        World world,
        Store<EntityStore> store,
        Ref<EntityStore> agentRef,
        Ref<EntityStore> targetRef,
        Vector3d agentPosition,
        Vector3d targetPosition,
        DodgeMotionParameters parameters
    ) {
        if (world == null || store == null || agentRef == null
            || targetRef == null || !agentRef.isValid() || !targetRef.isValid()) {
            return "spawned_reference_unavailable";
        }
        BoundingBox agentComponent = store.getComponent(
            agentRef, BoundingBox.getComponentType());
        BoundingBox targetComponent = store.getComponent(
            targetRef, BoundingBox.getComponentType());
        if (agentComponent == null || targetComponent == null) {
            return "spawned_bounding_box_unavailable";
        }
        return validatePair(
            store,
            CollisionModule.get(),
            new Box(agentComponent.getBoundingBox()),
            agentPosition,
            0.0,
            new Box(targetComponent.getBoundingBox()),
            targetPosition,
            180.0,
            parameters
        );
    }

    private static FixturePlacementPlanner.Candidate evaluate(
        World world,
        Store<EntityStore> store,
        CollisionModule collision,
        SpawningContext agent,
        SpawningContext target,
        Vector3d origin,
        HeadlessCombatFixtureSpec spec,
        DodgeMotionParameters parameters,
        int offsetX,
        int offsetZ
    ) {
        double x = Math.floor(origin.x) + 0.5 + offsetX;
        double z = Math.floor(origin.z) + 0.5 + offsetZ;
        if (!agent.set(world, x, origin.y, z)) {
            return FixturePlacementPlanner.Candidate.reject(
                "agent_column_unavailable");
        }
        SpawnTestResult agentSpawn = agent.canSpawn(true, false);
        if (agentSpawn != SpawnTestResult.TEST_OK) {
            return FixturePlacementPlanner.Candidate.reject(
                "agent_" + agentSpawn.name().toLowerCase(Locale.ROOT));
        }
        double targetZ = z - spec.separation();
        if (!target.set(world, x, origin.y, targetZ)) {
            return FixturePlacementPlanner.Candidate.reject(
                "target_column_unavailable");
        }
        SpawnTestResult targetSpawn = target.canSpawn(true, false);
        if (targetSpawn != SpawnTestResult.TEST_OK) {
            return FixturePlacementPlanner.Candidate.reject(
                "target_" + targetSpawn.name().toLowerCase(Locale.ROOT));
        }
        Model agentModel = agent.getModel();
        Model targetModel = target.getModel();
        if (agentModel == null || targetModel == null
            || agentModel.getBoundingBox() == null
            || targetModel.getBoundingBox() == null) {
            return FixturePlacementPlanner.Candidate.reject(
                "model_bounding_box_unavailable");
        }
        Vector3d agentPosition = agent.newPosition();
        Vector3d targetPosition = target.newPosition();
        // SpawningContext probes a block column at its center. Preserve the
        // operator-authored continuous separation and use the probe only for
        // native ground height; the exact shifted AABB is validated below.
        targetPosition.x = x;
        targetPosition.z = targetZ;
        String rejected = validatePair(
            store,
            collision,
            new Box(agentModel.getBoundingBox()),
            agentPosition,
            0.0,
            new Box(targetModel.getBoundingBox()),
            targetPosition,
            180.0,
            parameters
        );
        if (!rejected.isEmpty()) {
            return FixturePlacementPlanner.Candidate.reject(rejected);
        }
        return new FixturePlacementPlanner.Candidate(
            true,
            "grounded_aabbs_and_lateral_corridors_clear",
            agentPosition.x,
            agentPosition.y,
            agentPosition.z,
            targetPosition.x,
            targetPosition.y,
            targetPosition.z
        );
    }

    private static String validatePair(
        Store<EntityStore> store,
        CollisionModule collision,
        Box agentBox,
        Vector3d agentPosition,
        double agentYaw,
        Box targetBox,
        Vector3d targetPosition,
        double targetYaw,
        DodgeMotionParameters parameters
    ) {
        if (collision == null || collision.isDisabled()) {
            return "collision_module_unavailable_or_disabled";
        }
        World world = store.getExternalData().getWorld();
        int agentCode = collision.validatePosition(
            world, agentBox, agentPosition, new CollisionResult());
        if (agentCode != CollisionModule.VALIDATE_ON_GROUND) {
            return "agent_position_code_" + agentCode;
        }
        int targetCode = collision.validatePosition(
            world, targetBox, targetPosition, new CollisionResult());
        if (targetCode != CollisionModule.VALIDATE_ON_GROUND) {
            return "target_position_code_" + targetCode;
        }
        if (overlaps(
                agentBox, agentPosition, targetBox, targetPosition)) {
            return "actor_aabbs_overlap";
        }
        String agentCorridor = corridor(
            store, agentBox, agentPosition, agentYaw, parameters);
        if (!agentCorridor.isEmpty()) return "agent_" + agentCorridor;
        String targetCorridor = corridor(
            store, targetBox, targetPosition, targetYaw, parameters);
        if (!targetCorridor.isEmpty()) return "target_" + targetCorridor;
        return "";
    }

    private static String corridor(
        Store<EntityStore> store,
        Box box,
        Vector3d position,
        double yaw,
        DodgeMotionParameters parameters
    ) {
        DodgeCorridorEvaluator.Evaluation evaluation =
            DodgeCorridorEvaluator.evaluateDetailed(
                yaw,
                0L,
                true,
                parameters,
                (direction, displacement) ->
                    ServerDodgeCorridorReader.sweep(
                        store,
                        box,
                        position,
                        direction,
                        displacement
                    ).result()
            );
        if (evaluation.reason() != DodgeCorridorEvaluator.EvaluationReason.READY) {
            return "corridor_" + evaluation.reason().name().toLowerCase(Locale.ROOT);
        }
        boolean[] clear = evaluation.clear();
        if (!clear[DodgeCorridorEvaluator.Direction.LEFT.ordinal()]) {
            return "left_corridor_blocked";
        }
        if (!clear[DodgeCorridorEvaluator.Direction.RIGHT.ordinal()]) {
            return "right_corridor_blocked";
        }
        return "";
    }

    private static SpawningContext context(String roleName) {
        NPCPlugin plugin = NPCPlugin.get();
        int index = plugin.getIndex(roleName);
        Builder<Role> builder = index < 0
            ? null
            : plugin.tryGetCachedValidRole(index);
        if (builder == null || !builder.isSpawnable()
            || !(builder instanceof ISpawnableWithModel spawnable)) {
            return null;
        }
        SpawningContext context = new SpawningContext();
        // Validate the maximum authored model scale so the later random-scale
        // spawn cannot be larger than the preflight envelope.
        if (!context.setSpawnable(spawnable, true)) {
            context.releaseFull();
            return null;
        }
        return context;
    }

    private static boolean overlaps(
        Box left,
        Vector3d leftPosition,
        Box right,
        Vector3d rightPosition
    ) {
        return leftPosition.x + left.max.x > rightPosition.x + right.min.x
            && leftPosition.x + left.min.x < rightPosition.x + right.max.x
            && leftPosition.y + left.max.y > rightPosition.y + right.min.y
            && leftPosition.y + left.min.y < rightPosition.y + right.max.y
            && leftPosition.z + left.max.z > rightPosition.z + right.min.z
            && leftPosition.z + left.min.z < rightPosition.z + right.max.z;
    }

    private static Placement failed(Reason reason, String detail) {
        return new Placement(
            reason,
            new Vector3d(Double.NaN),
            new Vector3d(Double.NaN),
            0,
            0,
            0,
            detail
        );
    }

    private static void release(SpawningContext context) {
        if (context != null) context.releaseFull();
    }

    private static boolean valid(Vector3d position) {
        return position != null
            && Double.isFinite(position.x)
            && Double.isFinite(position.y)
            && Double.isFinite(position.z);
    }

    private static UUID fixtureUuid(HeadlessCombatFixtureSpec spec) {
        String identity = String.join("\u001f",
            "hytalerl-safe-combat-fixture-v1",
            spec.agentRole(),
            spec.targetRole(),
            Double.toString(spec.separation()),
            Integer.toString(spec.searchRadius())
        );
        return UUID.nameUUIDFromBytes(
            identity.getBytes(StandardCharsets.UTF_8));
    }
}
