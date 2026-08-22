package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.entity.ExplosionConfig;
import com.hypixel.hytale.server.core.entity.ExplosionUtils;
import com.hypixel.hytale.server.core.modules.entity.damage.Damage;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.worldgen.NativeExplosionMutationProbe;
import java.util.List;
import java.util.Objects;
import org.joml.Vector3d;
import org.joml.Vector3i;

/** Executes the narrow, server-owned dry-block explosion fixture. */
public final class ExplosionMutationCapture {

    public static final int BLOCK_DAMAGE_RADIUS = 2;
    public static final float BLOCK_DAMAGE_FALLOFF = 1.0f;
    public static final float BLOCK_DROP_CHANCE = 1.0f;
    public static final int SCAN_HALO = 1;
    public static final int SCAN_RADIUS = BLOCK_DAMAGE_RADIUS + SCAN_HALO;
    public static final int SCAN_CELL_COUNT = 343;

    private ExplosionMutationCapture() {}

    public enum Fixture {
        DIRECT_DROP(
            NativeExplosionMutationProbe.DIRECT_DROP_FIXTURE_KIND,
            NativeExplosionMutationProbe.DIRECT_DROP_FIXTURE_BLOCK_ASSET_ID,
            1
        );

        private final String kind;
        private final String blockAssetId;
        private final int expectedDropCount;

        Fixture(String kind, String blockAssetId, int expectedDropCount) {
            this.kind = kind;
            this.blockAssetId = blockAssetId;
            this.expectedDropCount = expectedDropCount;
        }

        public String kind() {
            return kind;
        }

        public String blockAssetId() {
            return blockAssetId;
        }

        public int expectedDropCount() {
            return expectedDropCount;
        }

        public static Fixture fromKind(String kind) {
            return switch (kind) {
                case NativeExplosionMutationProbe.DIRECT_DROP_FIXTURE_KIND ->
                    DIRECT_DROP;
                default -> throw new IllegalArgumentException(
                    "fixtureKind must be direct_drop"
                );
            };
        }
    }

    public record Context(
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        String worldEpoch
    ) {}

    /** Exact adapter to the scalar subset of NativeMutableBlockEvidence.Row. */
    public interface CellStateAccess {
        NativeExplosionMutationProbe.CellState snapshot(int x, int y, int z);

        /** Remove any fixture-owned partial-health entry during cleanup. */
        void clearBlockHealth(int x, int y, int z);
    }

    /**
     * Run on the owning World thread. The origin is selected by the server and
     * must be an air-cell centre; the target centre is exactly origin+[1,0,0].
     */
    public static NativeExplosionMutationProbe capture(
        World world,
        Context context,
        String fixtureKind,
        double originX,
        double originY,
        double originZ,
        int cellCapacity,
        int dropCapacity,
        CellStateAccess cells
    ) {
        try {
            return captureWithCleanup(
                world,
                context,
                fixtureKind,
                originX,
                originY,
                originZ,
                cellCapacity,
                dropCapacity,
                cells
            );
        } catch (CleanupFailure exception) {
            return cleanupFailureReceipt(
                context,
                fixtureKind,
                originX,
                originY,
                originZ,
                cellCapacity,
                dropCapacity,
                exception.cleanupCause()
            );
        }
    }

    private static NativeExplosionMutationProbe captureWithCleanup(
        World world,
        Context context,
        String fixtureKind,
        double originX,
        double originY,
        double originZ,
        int cellCapacity,
        int dropCapacity,
        CellStateAccess cells
    ) {
        Objects.requireNonNull(world, "world");
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(cells, "cells");
        Fixture fixture = Fixture.fromKind(fixtureKind);
        requireCapacity(
            cellCapacity,
            NativeExplosionMutationProbe.MAX_CELL_CAPACITY,
            "cellCapacity"
        );
        requireCapacity(
            dropCapacity,
            NativeExplosionMutationProbe.MAX_DROP_CAPACITY,
            "dropCapacity"
        );

        Vector3i originCell = ExplosionMutationFixture.centeredOriginCell(
            originX,
            originY,
            originZ
        );
        Vector3i target = new Vector3i(originCell).add(1, 0, 0);
        List<Vector3i> scan = ExplosionMutationFixture.scanPositions(originCell);
        Store<EntityStore> entities = world.getEntityStore().getStore();
        Store<ChunkStore> chunks = world.getChunkStore().getStore();
        ExplosionMutationAccessor recordingAccessor = null;
        boolean targetPlaced = false;
        boolean executionStarted = false;

        try {
            ExplosionMutationFixture.Preflight preflight =
                ExplosionMutationFixture.preflightEmptyDryVolume(world, scan);
            if (!preflight.accepted()) {
                return failure(
                    context,
                    fixture,
                    originX,
                    originY,
                    originZ,
                    cellCapacity,
                    dropCapacity,
                    preflight.unsupportedScopeMask(),
                    false,
                    0,
                    0,
                    preflight.failureReason()
                );
            }

            executionStarted = true;
            targetPlaced = true;
            if (!ExplosionMutationFixture.placeFixture(
                world,
                target,
                fixture.blockAssetId()
            )) {
                return failure(
                    context,
                    fixture,
                    originX,
                    originY,
                    originZ,
                    cellCapacity,
                    dropCapacity,
                    0,
                    true,
                    0,
                    0,
                    "fixture block placement was rejected"
                );
            }
            String fixtureFailure = ExplosionMutationFixture
                .validatePlacedFixture(world, target, fixture);
            if (!fixtureFailure.isEmpty()) {
                return failure(
                    context,
                    fixture,
                    originX,
                    originY,
                    originZ,
                    cellCapacity,
                    dropCapacity,
                    NativeExplosionMutationProbe.UNSUPPORTED_SUPPORT_CASCADE,
                    true,
                    0,
                    0,
                    fixtureFailure
                );
            }

            var before = ExplosionMutationFixture.snapshotVolume(
                world,
                scan,
                cells
            );
            recordingAccessor = new ExplosionMutationAccessor(
                entities,
                target
            );
            ExplosionUtils.performExplosion(
                Damage.NULL_SOURCE,
                new Vector3d(originX, originY, originZ),
                nativeConfig(),
                null,
                recordingAccessor.accessor(),
                chunks
            );

            var after = ExplosionMutationFixture.snapshotVolume(
                world,
                scan,
                cells
            );
            ExplosionMutationFixture.Diff diff =
                ExplosionMutationFixture.diff(before, after, target);
            var drops = ExplosionMutationDrops.resolvedDrops(
                recordingAccessor.createdItems(),
                target
            );
            String outcomeFailure = ExplosionMutationFixture.validateOutcome(
                fixture,
                target,
                diff.changedCells(),
                drops
            );
            if (recordingAccessor.redirected()) {
                outcomeFailure = "native listener redirected the fixture target";
            }
            boolean cellOverflow = diff.totalChangedCells() > cellCapacity;
            boolean dropOverflow = drops.size() > dropCapacity;
            if (
                diff.unsupportedScopeMask() != 0
                    || cellOverflow
                    || dropOverflow
                    || !outcomeFailure.isEmpty()
            ) {
                String reason = !outcomeFailure.isEmpty()
                    ? outcomeFailure
                    : cellOverflow
                        ? "changed-cell capacity exceeded"
                        : dropOverflow
                            ? "resolved-drop capacity exceeded"
                            : "mutation left the controlled dry scope";
                return failure(
                    context,
                    fixture,
                    originX,
                    originY,
                    originZ,
                    cellCapacity,
                    dropCapacity,
                    diff.unsupportedScopeMask(),
                    true,
                    diff.totalChangedCells(),
                    drops.size(),
                    reason
                );
            }
            return complete(
                context,
                fixture,
                originX,
                originY,
                originZ,
                cellCapacity,
                dropCapacity,
                diff.changedCells(),
                drops
            );
        } catch (RuntimeException exception) {
            return failure(
                context,
                fixture,
                originX,
                originY,
                originZ,
                cellCapacity,
                dropCapacity,
                0,
                executionStarted,
                0,
                0,
                failureText(exception)
            );
        } finally {
            RuntimeException cleanupFailure = null;
            if (recordingAccessor != null) {
                try {
                    ExplosionMutationDrops.removeCreatedItems(
                        entities,
                        recordingAccessor.createdItems()
                    );
                } catch (RuntimeException exception) {
                    cleanupFailure = exception;
                }
            }
            if (targetPlaced) {
                try {
                    cells.clearBlockHealth(target.x, target.y, target.z);
                } catch (RuntimeException exception) {
                    if (cleanupFailure == null) cleanupFailure = exception;
                    else cleanupFailure.addSuppressed(exception);
                }
                try {
                    if (!ExplosionMutationFixture.clearFixture(world, target)) {
                        throw new IllegalStateException(
                            "explosion fixture cleanup did not leave air"
                        );
                    }
                } catch (RuntimeException exception) {
                    if (cleanupFailure == null) cleanupFailure = exception;
                    else cleanupFailure.addSuppressed(exception);
                }
            }
            if (cleanupFailure != null) {
                throw new CleanupFailure(cleanupFailure);
            }
        }
    }

    static ExplosionConfig nativeConfig() {
        return new ControlledExplosionConfig();
    }

    static Vector3i centeredOriginCell(
        double originX,
        double originY,
        double originZ
    ) {
        return ExplosionMutationFixture.centeredOriginCell(
            originX,
            originY,
            originZ
        );
    }

    static List<Vector3i> scanPositions(Vector3i originCell) {
        return ExplosionMutationFixture.scanPositions(originCell);
    }

    static NativeExplosionMutationProbe cleanupFailureReceipt(
        Context context,
        String fixtureKind,
        double originX,
        double originY,
        double originZ,
        int cellCapacity,
        int dropCapacity,
        RuntimeException cleanupFailure
    ) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(cleanupFailure, "cleanupFailure");
        String reason = "fixture cleanup failed: "
            + failureText(cleanupFailure);
        if (reason.length() > 256) reason = reason.substring(0, 256);
        return failure(
            context,
            Fixture.fromKind(fixtureKind),
            originX,
            originY,
            originZ,
            cellCapacity,
            dropCapacity,
            0,
            true,
            0,
            0,
            reason
        );
    }

    private static NativeExplosionMutationProbe complete(
        Context context,
        Fixture fixture,
        double originX,
        double originY,
        double originZ,
        int cellCapacity,
        int dropCapacity,
        List<NativeExplosionMutationProbe.ChangedCell> cells,
        List<NativeExplosionMutationProbe.ResolvedDrop> drops
    ) {
        return probe(
            context,
            fixture,
            originX,
            originY,
            originZ,
            cellCapacity,
            dropCapacity,
            0,
            true,
            true,
            false,
            "",
            cells.size(),
            drops.size(),
            cells,
            drops
        );
    }

    private static NativeExplosionMutationProbe failure(
        Context context,
        Fixture fixture,
        double originX,
        double originY,
        double originZ,
        int cellCapacity,
        int dropCapacity,
        int unsupported,
        boolean executionStarted,
        int totalCells,
        int totalDrops,
        String reason
    ) {
        return probe(
            context,
            fixture,
            originX,
            originY,
            originZ,
            cellCapacity,
            dropCapacity,
            unsupported,
            executionStarted,
            false,
            executionStarted,
            reason,
            totalCells,
            totalDrops,
            List.of(),
            List.of()
        );
    }

    private static NativeExplosionMutationProbe probe(
        Context context,
        Fixture fixture,
        double originX,
        double originY,
        double originZ,
        int cellCapacity,
        int dropCapacity,
        int unsupported,
        boolean executionStarted,
        boolean complete,
        boolean resyncRequired,
        String failureReason,
        int totalCells,
        int totalDrops,
        List<NativeExplosionMutationProbe.ChangedCell> cells,
        List<NativeExplosionMutationProbe.ResolvedDrop> drops
    ) {
        return new NativeExplosionMutationProbe(
            context.serverVersion(),
            context.worldName(),
            context.worldgenProvider(),
            context.worldgenVersion(),
            context.seed(),
            context.worldEpoch(),
            NativeExplosionMutationProbe.SYNTHETIC_CONFIG_SOURCE_ID,
            fixture.kind(),
            fixture.blockAssetId(),
            NativeExplosionMutationProbe.syntheticConfigSemanticSha256(
                fixture.kind(),
                fixture.blockAssetId(),
                BLOCK_DAMAGE_RADIUS,
                BLOCK_DAMAGE_FALLOFF,
                BLOCK_DROP_CHANCE
            ),
            originX,
            originY,
            originZ,
            BLOCK_DAMAGE_RADIUS,
            BLOCK_DAMAGE_FALLOFF,
            BLOCK_DROP_CHANCE,
            false,
            0.0f,
            0.0f,
            0.0f,
            false,
            NativeExplosionMutationProbe.V1_ENTITY_CAPACITY,
            0,
            false,
            cellCapacity,
            totalCells,
            totalCells > cellCapacity,
            dropCapacity,
            totalDrops,
            totalDrops > dropCapacity,
            unsupported,
            executionStarted,
            complete,
            resyncRequired,
            failureReason,
            List.of(),
            cells,
            drops
        );
    }

    private static void requireCapacity(int value, int maximum, String name) {
        if (value < 1 || value > maximum) {
            throw new IllegalArgumentException(
                name + " must be in [1," + maximum + "]"
            );
        }
    }

    private static String failureText(RuntimeException exception) {
        String message = exception.getMessage();
        String value = exception.getClass().getSimpleName()
            + (message == null || message.isBlank() ? "" : ": " + message);
        return value.length() <= 256 ? value : value.substring(0, 256);
    }

    private static final class ControlledExplosionConfig
        extends ExplosionConfig {

        private ControlledExplosionConfig() {
            damageBlocks = true;
            damageEntities = false;
            blockDamageRadius = BLOCK_DAMAGE_RADIUS;
            blockDamageFalloff = BLOCK_DAMAGE_FALLOFF;
            blockDropChance = BLOCK_DROP_CHANCE;
            entityDamageRadius = 0.0f;
            entityDamage = 0.0f;
            entityDamageFalloff = 0.0f;
            itemTool = null;
            knockback = null;
            particles = null;
            soundEventId = null;
            soundEventIndex = 0;
        }
    }

    private static final class CleanupFailure extends RuntimeException {

        private CleanupFailure(RuntimeException cleanupCause) {
            super(cleanupCause);
        }

        private RuntimeException cleanupCause() {
            return (RuntimeException) getCause();
        }
    }
}
