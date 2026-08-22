package com.hytalerlbridge.nativebackend.policy.combat.raycast;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.shape.Box;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.math.vector.Transform;
import com.hypixel.hytale.protocol.BlockMaterial;
import com.hypixel.hytale.server.core.asset.type.blockhitbox.BlockBoundingBoxes;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.RotationTuple;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.core.util.TargetUtil;
import com.hytalerlbridge.geometry.GeometryContract;
import java.util.ArrayList;
import java.util.List;
import org.joml.Vector3d;

/**
 * Exact zero-width point-ray contact for bridge-driven client operations.
 *
 * <p>The arithmetic mirrors the Gym's
 * {@code jax.world.geometry.point_raycast} contract: normalize once, split the
 * ray into one-block segments, intersect closed native collision boxes, and
 * select the first contact. Missing chunks, excessive distance, malformed
 * hitboxes, and excessive candidate volume fail closed.</p>
 */
public final class NativePointRaycast {

    public static final int SEGMENT_CAPACITY = 64;
    public static final int WORLD_CANDIDATE_CELL_CAPACITY = 1_000_000;
    private static final float INPUT_EPSILON = 1.0e-6f;
    private static final float CONTACT_EPSILON = 1.0e-7f;

    private NativePointRaycast() {}

    /** World-space collision box plus stable native source identity. */
    public record RayBox(
        float minimumX,
        float minimumY,
        float minimumZ,
        float maximumX,
        float maximumY,
        float maximumZ,
        int blockX,
        int blockY,
        int blockZ,
        int detailBoxIndex
    ) {}

    /** Point-ray outcome with explicit evidence availability. */
    public record Result(
        boolean available,
        boolean hit,
        float hitX,
        float hitY,
        float hitZ,
        float normalX,
        float normalY,
        float normalZ,
        float distance,
        int segmentCount,
        int blockX,
        int blockY,
        int blockZ,
        int detailBoxIndex,
        String unavailableReason
    ) {
        static Result unavailable(String reason) {
            return unavailable(reason, 0.0f);
        }

        static Result unavailable(String reason, float distance) {
            return new Result(
                false,
                false,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                distance,
                0,
                Integer.MIN_VALUE,
                Integer.MIN_VALUE,
                Integer.MIN_VALUE,
                -1,
                reason
            );
        }
    }

    /** Trace from the actor's native eye position and head rotation. */
    public static Result traceActorLook(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        float maximumDistance
    ) {
        if (actor == null || !actor.isValid() || store == null) {
            return Result.unavailable("actor_or_store_unavailable");
        }
        World world = store.getExternalData().getWorld();
        if (world == null) return Result.unavailable("world_unavailable");
        Transform look = TargetUtil.getLook(actor, store);
        Vector3d origin = look.getPosition();
        Vector3d direction = look.getDirection();
        return traceWorld(
            world,
            (float) origin.x,
            (float) origin.y,
            (float) origin.z,
            (float) direction.x,
            (float) direction.y,
            (float) direction.z,
            maximumDistance
        );
    }

    /** Trace through currently loaded native world collision boxes. */
    static Result traceWorld(
        World world,
        float originX,
        float originY,
        float originZ,
        float directionX,
        float directionY,
        float directionZ,
        float maximumDistance
    ) {
        Result invalid = validateInputs(
            originX,
            originY,
            originZ,
            directionX,
            directionY,
            directionZ,
            maximumDistance
        );
        if (invalid != null) return invalid;

        float length = vectorLength(directionX, directionY, directionZ);
        float unitX = directionX / length;
        float unitY = directionY / length;
        float unitZ = directionZ / length;
        float endX = originX + unitX * maximumDistance;
        float endY = originY + unitY * maximumDistance;
        float endZ = originZ + unitZ * maximumDistance;
        int protrusion = maximumProtrusionCells();
        if (protrusion < 0) {
            return Result.unavailable(
                "block_hitbox_assets_unavailable",
                maximumDistance
            );
        }

        int rawMinimumX;
        int rawMinimumY;
        int rawMinimumZ;
        int rawMaximumX;
        int rawMaximumY;
        int rawMaximumZ;
        try {
            rawMinimumX = floorToInt(Math.min(originX, endX));
            rawMinimumY = floorToInt(Math.min(originY, endY));
            rawMinimumZ = floorToInt(Math.min(originZ, endZ));
            rawMaximumX = floorToInt(Math.max(originX, endX));
            rawMaximumY = floorToInt(Math.max(originY, endY));
            rawMaximumZ = floorToInt(Math.max(originZ, endZ));
        } catch (IllegalArgumentException exception) {
            return Result.unavailable("coordinate_out_of_range", maximumDistance);
        }
        if (
            rawMinimumX < Integer.MIN_VALUE + protrusion
                || rawMinimumY < Integer.MIN_VALUE + protrusion
                || rawMinimumZ < Integer.MIN_VALUE + protrusion
                || rawMaximumX > Integer.MAX_VALUE - protrusion
                || rawMaximumY > Integer.MAX_VALUE - protrusion
                || rawMaximumZ > Integer.MAX_VALUE - protrusion
        ) {
            return Result.unavailable("coordinate_out_of_range", maximumDistance);
        }
        int minimumX = rawMinimumX - protrusion;
        int minimumY = rawMinimumY - protrusion;
        int minimumZ = rawMinimumZ - protrusion;
        int maximumX = rawMaximumX + protrusion;
        int maximumY = rawMaximumY + protrusion;
        int maximumZ = rawMaximumZ + protrusion;

        List<RayBox> boxes = new ArrayList<>();
        long candidateCells = 0L;
        RayBox broadphase = null;
        for (int x = minimumX; x <= maximumX; x++) {
            for (int y = minimumY; y <= maximumY; y++) {
                if (y < 0 || y >= 320) continue;
                for (int z = minimumZ; z <= maximumZ; z++) {
                    broadphase = new RayBox(
                        x - protrusion,
                        y - protrusion,
                        z - protrusion,
                        x + 1.0f + protrusion,
                        y + 1.0f + protrusion,
                        z + 1.0f + protrusion,
                        x,
                        y,
                        z,
                        -1
                    );
                    if (!rayMayTouchBox(
                        originX,
                        originY,
                        originZ,
                        unitX * maximumDistance,
                        unitY * maximumDistance,
                        unitZ * maximumDistance,
                        broadphase
                    )) {
                        continue;
                    }
                    candidateCells++;
                    if (candidateCells > WORLD_CANDIDATE_CELL_CAPACITY) {
                        return Result.unavailable(
                            "candidate_capacity_exceeded",
                            maximumDistance
                        );
                    }
                    if (world.getChunkIfLoaded(
                        ChunkUtil.indexChunkFromBlock(x, z)
                    ) == null) {
                        return Result.unavailable(
                            "ray_chunk_unavailable",
                            maximumDistance
                        );
                    }
                    int runtimeBlockId = world.getBlock(x, y, z);
                    if (runtimeBlockId == 0) continue;
                    BlockType blockType = world.getBlockType(x, y, z);
                    if (
                        blockType == null
                            || blockType.getMaterial() != BlockMaterial.Solid
                    ) {
                        continue;
                    }
                    BlockBoundingBoxes hitbox = BlockBoundingBoxes.getAssetMap()
                        .getAssetOrDefault(
                            blockType.getHitboxTypeIndex(),
                            BlockBoundingBoxes.UNIT_BOX
                        );
                    if (hitbox == null) {
                        return Result.unavailable(
                            "block_hitbox_unavailable",
                            maximumDistance
                        );
                    }
                    int rotation = world.getBlockRotationIndex(x, y, z);
                    if (
                        rotation < 0
                            || rotation >= RotationTuple.VALUES.length
                    ) {
                        return Result.unavailable(
                            "block_rotation_invalid",
                            maximumDistance
                        );
                    }
                    BlockBoundingBoxes.RotatedVariantBoxes variant =
                        hitbox.get(rotation);
                    Box[] details = variant.hasDetailBoxes()
                        ? variant.getDetailBoxes()
                        : new Box[] {variant.getBoundingBox()};
                    if (
                        details.length
                            > GeometryContract.MAX_DETAIL_BOXES_0_5_7
                    ) {
                        return Result.unavailable(
                            "detail_box_capacity_exceeded",
                            maximumDistance
                        );
                    }
                    for (int detail = 0; detail < details.length; detail++) {
                        Box box = details[detail];
                        boxes.add(new RayBox(
                            (float) (x + box.min.x),
                            (float) (y + box.min.y),
                            (float) (z + box.min.z),
                            (float) (x + box.max.x),
                            (float) (y + box.max.y),
                            (float) (z + box.max.z),
                            x,
                            y,
                            z,
                            detail
                        ));
                    }
                }
            }
        }
        return traceBoxes(
            originX,
            originY,
            originZ,
            directionX,
            directionY,
            directionZ,
            maximumDistance,
            boxes
        );
    }

    /** Pure arithmetic entry point used by the offline differential tests. */
    static Result traceBoxes(
        float originX,
        float originY,
        float originZ,
        float directionX,
        float directionY,
        float directionZ,
        float maximumDistance,
        List<RayBox> boxes
    ) {
        Result invalid = validateInputs(
            originX,
            originY,
            originZ,
            directionX,
            directionY,
            directionZ,
            maximumDistance
        );
        if (invalid != null) return invalid;
        if (boxes == null) {
            return Result.unavailable("boxes_unavailable", maximumDistance);
        }

        float length = vectorLength(directionX, directionY, directionZ);
        float unitX = directionX / length;
        float unitY = directionY / length;
        float unitZ = directionZ / length;
        int segmentCount = 0;
        for (
            int segment = 0;
            segment < SEGMENT_CAPACITY && segment < maximumDistance;
            segment++
        ) {
            float start = segment;
            float segmentLength = Math.min(1.0f, maximumDistance - start);
            if (segmentLength <= INPUT_EPSILON) break;
            segmentCount++;
            float startX = originX + unitX * start;
            float startY = originY + unitY * start;
            float startZ = originZ + unitZ * start;
            float deltaX = unitX * segmentLength;
            float deltaY = unitY * segmentLength;
            float deltaZ = unitZ * segmentLength;
            BoxHit nearest = null;
            RayBox nearestBox = null;
            for (RayBox box : boxes) {
                BoxHit hit = intersect(
                    startX,
                    startY,
                    startZ,
                    deltaX,
                    deltaY,
                    deltaZ,
                    box
                );
                if (hit == null) continue;
                if (nearest == null || hit.fraction < nearest.fraction) {
                    nearest = hit;
                    nearestBox = box;
                }
            }
            if (nearest != null && nearestBox != null) {
                float distance = start + nearest.fraction * segmentLength;
                return new Result(
                    true,
                    true,
                    startX + deltaX * nearest.fraction,
                    startY + deltaY * nearest.fraction,
                    startZ + deltaZ * nearest.fraction,
                    nearest.normalX,
                    nearest.normalY,
                    nearest.normalZ,
                    distance,
                    segmentCount,
                    nearestBox.blockX,
                    nearestBox.blockY,
                    nearestBox.blockZ,
                    nearestBox.detailBoxIndex,
                    ""
                );
            }
        }
        return new Result(
            true,
            false,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            maximumDistance,
            segmentCount,
            Integer.MIN_VALUE,
            Integer.MIN_VALUE,
            Integer.MIN_VALUE,
            -1,
            ""
        );
    }

    private static Result validateInputs(
        float originX,
        float originY,
        float originZ,
        float directionX,
        float directionY,
        float directionZ,
        float maximumDistance
    ) {
        if (
            !Float.isFinite(originX)
                || !Float.isFinite(originY)
                || !Float.isFinite(originZ)
                || !Float.isFinite(directionX)
                || !Float.isFinite(directionY)
                || !Float.isFinite(directionZ)
                || !Float.isFinite(maximumDistance)
        ) {
            return Result.unavailable("non_finite_input");
        }
        if (vectorLength(directionX, directionY, directionZ) <= INPUT_EPSILON) {
            return Result.unavailable("direction_zero");
        }
        if (maximumDistance <= 0.0f) {
            return Result.unavailable("distance_non_positive");
        }
        if (maximumDistance > SEGMENT_CAPACITY) {
            return Result.unavailable(
                "segment_capacity_exceeded",
                maximumDistance
            );
        }
        return null;
    }

    private static int maximumProtrusionCells() {
        var assets = BlockBoundingBoxes.getAssetMap();
        if (assets == null) return -1;
        double maximum = 0.0;
        for (int assetIndex = 0; assetIndex < assets.getNextIndex(); assetIndex++) {
            BlockBoundingBoxes asset = assets.getAsset(assetIndex);
            if (asset == null) continue;
            for (
                int rotation = 0;
                rotation < RotationTuple.VALUES.length;
                rotation++
            ) {
                BlockBoundingBoxes.RotatedVariantBoxes variant =
                    asset.get(rotation);
                if (variant == null) return -1;
                maximum = Math.max(
                    maximum,
                    variant.getBoundingBox().getMaximumExtent()
                );
            }
        }
        if (!Double.isFinite(maximum) || maximum > Integer.MAX_VALUE - 1.0) {
            return -1;
        }
        return (int) Math.ceil(maximum);
    }

    private static boolean rayMayTouchBox(
        float originX,
        float originY,
        float originZ,
        float deltaX,
        float deltaY,
        float deltaZ,
        RayBox box
    ) {
        float[] origin = {originX, originY, originZ};
        float[] delta = {deltaX, deltaY, deltaZ};
        float[] minimum = {
            box.minimumX,
            box.minimumY,
            box.minimumZ
        };
        float[] maximum = {
            box.maximumX,
            box.maximumY,
            box.maximumZ
        };
        float intervalStart = 0.0f;
        float intervalEnd = 1.0f;
        for (int axis = 0; axis < 3; axis++) {
            if (Math.abs(delta[axis]) <= CONTACT_EPSILON) {
                if (
                    origin[axis] < minimum[axis] - CONTACT_EPSILON
                        || origin[axis] > maximum[axis] + CONTACT_EPSILON
                ) {
                    return false;
                }
                continue;
            }
            float first = (minimum[axis] - origin[axis]) / delta[axis];
            float second = (maximum[axis] - origin[axis]) / delta[axis];
            float near = Math.min(first, second);
            float far = Math.max(first, second);
            intervalStart = Math.max(intervalStart, near);
            intervalEnd = Math.min(intervalEnd, far);
            if (intervalStart > intervalEnd + CONTACT_EPSILON) return false;
        }
        return intervalEnd >= -CONTACT_EPSILON
            && intervalStart <= 1.0f + CONTACT_EPSILON;
    }

    private static BoxHit intersect(
        float originX,
        float originY,
        float originZ,
        float deltaX,
        float deltaY,
        float deltaZ,
        RayBox box
    ) {
        float[] origin = {originX, originY, originZ};
        float[] delta = {deltaX, deltaY, deltaZ};
        float[] minimum = {
            box.minimumX,
            box.minimumY,
            box.minimumZ
        };
        float[] maximum = {
            box.maximumX,
            box.maximumY,
            box.maximumZ
        };
        float[] entry = new float[3];
        float[] exit = new float[3];
        for (int axis = 0; axis < 3; axis++) {
            if (delta[axis] > CONTACT_EPSILON) {
                entry[axis] = (minimum[axis] - origin[axis]) / delta[axis];
                exit[axis] = (maximum[axis] - origin[axis]) / delta[axis];
            } else if (delta[axis] < -CONTACT_EPSILON) {
                entry[axis] = (maximum[axis] - origin[axis]) / delta[axis];
                exit[axis] = (minimum[axis] - origin[axis]) / delta[axis];
            } else {
                boolean overlap =
                    origin[axis] > minimum[axis] + CONTACT_EPSILON
                        && origin[axis] < maximum[axis] - CONTACT_EPSILON;
                entry[axis] = overlap
                    ? Float.NEGATIVE_INFINITY
                    : Float.POSITIVE_INFINITY;
                exit[axis] = overlap
                    ? Float.POSITIVE_INFINITY
                    : Float.NEGATIVE_INFINITY;
            }
        }
        float intervalStart = Math.max(entry[0], Math.max(entry[1], entry[2]));
        float intervalEnd = Math.min(exit[0], Math.min(exit[1], exit[2]));
        boolean hit = intervalStart <= intervalEnd + CONTACT_EPSILON
            && intervalEnd >= -CONTACT_EPSILON
            && intervalStart >= -CONTACT_EPSILON
            && intervalStart <= 1.0f + CONTACT_EPSILON;
        if (!hit) return null;
        float fraction = Math.max(0.0f, Math.min(1.0f, intervalStart));
        int normalAxis = 0;
        if (entry[1] > entry[normalAxis]) normalAxis = 1;
        if (entry[2] > entry[normalAxis]) normalAxis = 2;
        float sign = Math.signum(delta[normalAxis]);
        return new BoxHit(
            fraction,
            normalAxis == 0 ? -sign : 0.0f,
            normalAxis == 1 ? -sign : 0.0f,
            normalAxis == 2 ? -sign : 0.0f
        );
    }

    private static float vectorLength(float x, float y, float z) {
        return (float) Math.sqrt((double) x * x + (double) y * y + (double) z * z);
    }

    private static int floorToInt(float value) {
        double floor = Math.floor(value);
        if (floor < Integer.MIN_VALUE || floor > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("ray coordinate is outside int range");
        }
        return (int) floor;
    }

    private record BoxHit(
        float fraction,
        float normalX,
        float normalY,
        float normalZ
    ) {}
}
