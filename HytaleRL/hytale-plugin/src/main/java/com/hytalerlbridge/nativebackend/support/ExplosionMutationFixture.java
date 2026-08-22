package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.server.core.asset.type.blockhitbox.BlockBoundingBoxes;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.modules.interaction.BlockHarvestUtils;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.chunk.WorldChunk;
import com.hypixel.hytale.server.core.universe.world.chunk.section.BlockSection;
import com.hypixel.hytale.server.core.util.FillerBlockUtil;
import com.hytalerlbridge.nativebackend.model.ResolvedDropRoute;
import com.hytalerlbridge.worldgen.NativeExplosionMutationProbe;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.joml.Vector3i;

/** Dry-volume preflight, finite scanning, and exact changed-cell diffing. */
final class ExplosionMutationFixture {

    private ExplosionMutationFixture() {}

    static Vector3i centeredOriginCell(
        double originX,
        double originY,
        double originZ
    ) {
        int x = centeredCoordinate(originX, "originX");
        int y = centeredCoordinate(originY, "originY");
        int z = centeredCoordinate(originZ, "originZ");
        int scanRadius = ExplosionMutationCapture.SCAN_RADIUS;
        if (y < scanRadius || y >= 320 - scanRadius) {
            throw new IllegalArgumentException(
                "server-selected explosion scan must remain inside y=[0,320)"
            );
        }
        return new Vector3i(x, y, z);
    }

    static List<Vector3i> scanPositions(Vector3i originCell) {
        int radius = ExplosionMutationCapture.SCAN_RADIUS;
        List<Vector3i> positions = new ArrayList<>(
            ExplosionMutationCapture.SCAN_CELL_COUNT
        );
        for (int x = originCell.x - radius; x <= originCell.x + radius; x++) {
            for (int y = originCell.y - radius;
                y <= originCell.y + radius;
                y++) {
                for (int z = originCell.z - radius;
                    z <= originCell.z + radius;
                    z++) {
                    positions.add(new Vector3i(x, y, z));
                }
            }
        }
        return List.copyOf(positions);
    }

    static Preflight preflightEmptyDryVolume(
        World world,
        List<Vector3i> positions
    ) {
        int unsupported = 0;
        for (Vector3i position : positions) {
            WorldChunk chunk = loadedChunk(world, position);
            BlockSection section = chunk == null
                ? null
                : chunk.getBlockChunk().getSectionAtBlockY(position.y);
            if (chunk == null || section == null || !section.loaded) {
                return new Preflight(false, 0, "preflight chunk is not loaded");
            }
            RawCell raw = rawCell(chunk, position);
            if (raw.runtimeBlockId() != 0) {
                return new Preflight(false, 0, "preflight volume is not air");
            }
            if (raw.fluid()) {
                unsupported |= NativeExplosionMutationProbe
                    .UNSUPPORTED_FLUID_MUTATION;
            }
            if (raw.filler() != 0) {
                unsupported |= NativeExplosionMutationProbe
                    .UNSUPPORTED_FILLER_MUTATION;
            }
            if (raw.supportValue() != 0) {
                unsupported |= NativeExplosionMutationProbe
                    .UNSUPPORTED_SUPPORT_CASCADE;
            }
        }
        return unsupported == 0
            ? new Preflight(true, 0, "")
            : new Preflight(
                false,
                unsupported,
                "preflight volume is outside the controlled dry scope"
            );
    }

    static boolean placeFixture(
        World world,
        Vector3i target,
        String blockAssetId
    ) {
        WorldChunk chunk = loadedChunk(world, target);
        int blockId = BlockType.getAssetMap().getIndex(blockAssetId);
        BlockType block = blockId == Integer.MIN_VALUE
            ? null
            : BlockType.getAssetMap().getAsset(blockId);
        return chunk != null
            && block != null
            && chunk.setBlock(
                target.x,
                target.y,
                target.z,
                blockId,
                block,
                0,
                FillerBlockUtil.NO_FILLER,
                4
            )
            && chunk.getBlock(target.x, target.y, target.z) == blockId;
    }

    static boolean clearFixture(World world, Vector3i target) {
        WorldChunk chunk = loadedChunk(world, target);
        if (chunk == null) return false;
        chunk.setBlock(
            target.x,
            target.y,
            target.z,
            BlockType.EMPTY_ID,
            BlockType.EMPTY,
            0,
            FillerBlockUtil.NO_FILLER,
            4
        );
        return chunk.getBlock(target.x, target.y, target.z) == 0;
    }

    static String validatePlacedFixture(
        World world,
        Vector3i target,
        ExplosionMutationCapture.Fixture fixture
    ) {
        WorldChunk chunk = loadedChunk(world, target);
        BlockType block = chunk == null
            ? null
            : BlockType.getAssetMap().getAsset(
                chunk.getBlock(target.x, target.y, target.z)
            );
        if (
            chunk == null
                || block == null
                || !fixture.blockAssetId().equals(block.getId())
                || chunk.getBlock(target.x, target.y, target.z) == 0
        ) {
            return "fixture block placement was rejected";
        }
        RawCell raw = rawCell(chunk, target);
        if (
            raw.fluid()
                || raw.filler() != 0
                || raw.supportValue() != 0
                || block.hasSupport()
                || block.getFallingBlockSettings() != null
                || block.getConnectedBlockRuleSet() != null
                || block.getBlockEntity() != null
                || block.isState()
                || !block.getSupporting(0).isEmpty()
        ) {
            return "fixture block is outside the dry no-support scope";
        }
        if (block.getGathering() == null || !block.getGathering().isSoft()) {
            return "fixture block has no native soft-gather route";
        }
        BlockBoundingBoxes boxes = BlockBoundingBoxes.getAssetMap().getAsset(
            block.getHitboxTypeIndex()
        );
        if (boxes == null) {
            return "fixture block collision boxes are unavailable";
        }
        boolean[] externalFiller = {false};
        FillerBlockUtil.forEachFillerBlock(boxes.get(0), (x, y, z) -> {
            if (x != 0 || y != 0 || z != 0) externalFiller[0] = true;
        });
        if (externalFiller[0]) {
            return "fixture block requires external filler cells";
        }
        ResolvedDropRoute route = DropRouting.resolveDropRoute(block, "soft");
        List<ItemStack> expected = BlockHarvestUtils.getDrops(
            block,
            route.quantity(),
            route.itemId(),
            route.dropListId()
        );
        if (expected.size() != fixture.expectedDropCount()) {
            return "direct-drop fixture no longer resolves exactly one stack";
        }
        if (
            fixture == ExplosionMutationCapture.Fixture.DIRECT_DROP
                && !expected.getFirst().getItemId().equals(
                    fixture.blockAssetId()
                )
        ) {
            return "direct-drop fixture resolves another item";
        }
        return "";
    }

    static Map<Position, ObservedCell> snapshotVolume(
        World world,
        List<Vector3i> positions,
        ExplosionMutationCapture.CellStateAccess cells
    ) {
        Map<Position, ObservedCell> result = new HashMap<>(positions.size());
        for (Vector3i position : positions) {
            WorldChunk chunk = loadedChunk(world, position);
            if (chunk == null) {
                throw new IllegalStateException(
                    "scan chunk unloaded during atomic capture"
                );
            }
            Position key = new Position(position.x, position.y, position.z);
            NativeExplosionMutationProbe.CellState scalar = Objects.requireNonNull(
                cells.snapshot(position.x, position.y, position.z),
                "cell-state snapshot"
            );
            result.put(key, new ObservedCell(rawCell(chunk, position), scalar));
        }
        return Map.copyOf(result);
    }

    static Diff diff(
        Map<Position, ObservedCell> before,
        Map<Position, ObservedCell> after,
        Vector3i target
    ) {
        List<Position> changed = new ArrayList<>();
        int unsupported = 0;
        for (Map.Entry<Position, ObservedCell> entry : before.entrySet()) {
            Position position = entry.getKey();
            ObservedCell left = entry.getValue();
            ObservedCell right = Objects.requireNonNull(after.get(position));
            if (
                left.raw().fluid()
                    || right.raw().fluid()
                    || !left.scalar().dry()
                    || !right.scalar().dry()
            ) {
                unsupported |= NativeExplosionMutationProbe
                    .UNSUPPORTED_FLUID_MUTATION;
            }
            if (left.raw().filler() != 0 || right.raw().filler() != 0) {
                unsupported |= NativeExplosionMutationProbe
                    .UNSUPPORTED_FILLER_MUTATION;
            }
            boolean differs = !left.raw().equals(right.raw())
                || left.scalar().substantivelyDiffersFrom(right.scalar());
            if (differs) {
                changed.add(position);
                if (!position.matches(target)) {
                    unsupported |= NativeExplosionMutationProbe
                        .UNSUPPORTED_SUPPORT_CASCADE;
                }
            }
        }
        changed.sort(Position.ORDER);
        List<NativeExplosionMutationProbe.ChangedCell> rows = new ArrayList<>();
        for (Position position : changed) {
            ObservedCell left = before.get(position);
            ObservedCell right = after.get(position);
            if (!left.scalar().substantivelyDiffersFrom(right.scalar())) {
                unsupported |= NativeExplosionMutationProbe
                    .UNSUPPORTED_FILLER_MUTATION;
                continue;
            }
            rows.add(new NativeExplosionMutationProbe.ChangedCell(
                rows.size(),
                position.x(),
                position.y(),
                position.z(),
                left.scalar(),
                right.scalar()
            ));
        }
        return new Diff(changed.size(), unsupported, List.copyOf(rows));
    }

    static String validateOutcome(
        ExplosionMutationCapture.Fixture fixture,
        Vector3i target,
        List<NativeExplosionMutationProbe.ChangedCell> cells,
        List<NativeExplosionMutationProbe.ResolvedDrop> drops
    ) {
        if (cells.size() != 1) {
            return "controlled fixture must change exactly one cell";
        }
        NativeExplosionMutationProbe.ChangedCell cell = cells.getFirst();
        if (
            cell.x() != target.x
                || cell.y() != target.y
                || cell.z() != target.z
                || !cell.before().blockPresent()
                || cell.after().blockPresent()
        ) {
            return "controlled fixture did not destroy its target block";
        }
        if (drops.size() != fixture.expectedDropCount()) {
            return "controlled fixture produced an unexpected drop count";
        }
        if (
            fixture == ExplosionMutationCapture.Fixture.DIRECT_DROP
                && !drops.getFirst().itemAssetId().equals(
                    fixture.blockAssetId()
                )
        ) {
            return "controlled fixture produced an unexpected drop item";
        }
        return "";
    }

    private static WorldChunk loadedChunk(World world, Vector3i position) {
        return world.getChunkIfLoaded(
            ChunkUtil.indexChunkFromBlock(position.x, position.z)
        );
    }

    private static RawCell rawCell(WorldChunk chunk, Vector3i position) {
        int localX = ChunkUtil.localCoordinate((long) position.x);
        int localZ = ChunkUtil.localCoordinate((long) position.z);
        return new RawCell(
            chunk.getBlock(localX, position.y, localZ),
            chunk.getRotationIndex(localX, position.y, localZ),
            chunk.getFiller(localX, position.y, localZ),
            chunk.getFluidId(localX, position.y, localZ),
            Byte.toUnsignedInt(chunk.getFluidLevel(localX, position.y, localZ)),
            chunk.getSupportValue(localX, position.y, localZ)
        );
    }

    private static int centeredCoordinate(double value, String name) {
        int radius = ExplosionMutationCapture.SCAN_RADIUS;
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
        double floor = Math.floor(value);
        if (
            value != floor + 0.5
                || floor < Integer.MIN_VALUE + radius
                || floor > Integer.MAX_VALUE - radius
        ) {
            throw new IllegalArgumentException(
                name + " must be a bounded air-cell centre"
            );
        }
        return (int) floor;
    }

    record Preflight(
        boolean accepted,
        int unsupportedScopeMask,
        String failureReason
    ) {}

    record Position(int x, int y, int z) {
        private static final Comparator<Position> ORDER = Comparator
            .comparingInt(Position::x)
            .thenComparingInt(Position::y)
            .thenComparingInt(Position::z);

        private boolean matches(Vector3i value) {
            return x == value.x && y == value.y && z == value.z;
        }
    }

    private record RawCell(
        int runtimeBlockId,
        int rotationIndex,
        int filler,
        int fluidId,
        int fluidLevel,
        int supportValue
    ) {
        private boolean fluid() {
            // Integer.MIN_VALUE is native "section unavailable", not dry.
            return fluidId != 0 || fluidLevel != 0;
        }
    }

    record ObservedCell(
        RawCell raw,
        NativeExplosionMutationProbe.CellState scalar
    ) {}

    record Diff(
        int totalChangedCells,
        int unsupportedScopeMask,
        List<NativeExplosionMutationProbe.ChangedCell> changedCells
    ) {}
}
