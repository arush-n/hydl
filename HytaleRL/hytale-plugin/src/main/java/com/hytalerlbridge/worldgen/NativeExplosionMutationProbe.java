package com.hytalerlbridge.worldgen;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;

/**
 * Complete bounded result of one controlled native block-damaging explosion.
 *
 * <p>The v1 scope is deliberately dry and direct. Fluid mutation, filler-root
 * resolution, and support cascades make the frame incomplete. Every
 * incomplete frame carries no rows; one that started execution requires a
 * full resync, while a preflight rejection does not.</p>
 */
public record NativeExplosionMutationProbe(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    String worldEpoch,
    String explosionConfigSourceId,
    String fixtureKind,
    String fixtureBlockAssetId,
    String explosionConfigSemanticSha256,
    double originX,
    double originY,
    double originZ,
    int blockDamageRadius,
    float blockDamageFalloff,
    float blockDropChance,
    boolean damageEntities,
    float entityDamageRadius,
    float entityDamage,
    float entityDamageFalloff,
    boolean ignoredControlledActor,
    int entityCapacity,
    int totalEntityAdmissions,
    boolean entityOverflow,
    int cellCapacity,
    int totalChangedCells,
    boolean cellOverflow,
    int dropCapacity,
    int totalResolvedDrops,
    boolean dropOverflow,
    int unsupportedScopeMask,
    boolean executionStarted,
    boolean complete,
    boolean resyncRequired,
    String failureReason,
    List<EntityAdmission> entityAdmissions,
    List<ChangedCell> changedCells,
    List<ResolvedDrop> resolvedDrops
) {
    public static final String SCHEMA =
        "hytalerl_native_explosion_mutation_probe_v1";
    public static final int VERSION = 1;
    public static final int MAX_BLOCK_RADIUS = 5;
    public static final int MAX_ENTITY_CAPACITY = 64;
    public static final int V1_ENTITY_CAPACITY = 1;
    public static final int MAX_CELL_CAPACITY = 64;
    public static final int MAX_DROP_CAPACITY = 4_096;

    /**
     * Server-owned synthetic source. It is not an asset-store identifier.
     * The semantic SHA is derived from the applied scalar configuration.
     */
    public static final String SYNTHETIC_CONFIG_SOURCE_ID =
        "bridge_synthetic_dry_block_explosion_v1";
    public static final String DIRECT_DROP_FIXTURE_KIND = "direct_drop";
    public static final String DIRECT_DROP_FIXTURE_BLOCK_ASSET_ID =
        "Recipe_Book_Magic_Air";

    public static final int UNSUPPORTED_FLUID_MUTATION = 1;
    public static final int UNSUPPORTED_FILLER_MUTATION = 1 << 1;
    public static final int UNSUPPORTED_SUPPORT_CASCADE = 1 << 2;
    public static final int KNOWN_UNSUPPORTED_SCOPE_MASK =
        UNSUPPORTED_FLUID_MUTATION
            | UNSUPPORTED_FILLER_MUTATION
            | UNSUPPORTED_SUPPORT_CASCADE;

    public static final String CONTROLLED_SCOPE =
        "loaded_dry_direct_blocks_no_filler_no_support_cascade";
    public static final String CELL_SCALAR_COMPATIBILITY =
        "NativeMutableBlockEvidence.Row_scalar_fields_v1";
    public static final String CHANGED_CELL_ORDERING =
        "lexicographic_xyz_ascending";
    public static final String RESOLVED_DROP_ORDERING =
        "source_xyz_item_quantity_durability_metadata_spawn_xyz_ascending";

    public NativeExplosionMutationProbe {
        serverVersion = requireText(serverVersion, "serverVersion", 128);
        worldName = requireText(worldName, "worldName", 256);
        worldgenProvider = requireText(
            worldgenProvider,
            "worldgenProvider",
            256
        );
        worldgenVersion = requireText(
            worldgenVersion,
            "worldgenVersion",
            256
        );
        worldEpoch = requireText(worldEpoch, "worldEpoch", 128);
        explosionConfigSourceId = requireText(
            explosionConfigSourceId,
            "explosionConfigSourceId",
            512
        );
        fixtureKind = requireText(fixtureKind, "fixtureKind", 32);
        fixtureBlockAssetId = requireText(
            fixtureBlockAssetId,
            "fixtureBlockAssetId",
            512
        );
        explosionConfigSemanticSha256 = requireSha256(
            explosionConfigSemanticSha256,
            "explosionConfigSemanticSha256"
        );
        failureReason = failureReason == null ? "" : failureReason.trim();
        entityAdmissions = copy(entityAdmissions);
        changedCells = copy(changedCells);
        resolvedDrops = copy(resolvedDrops);

        requireFinite(originX, "originX");
        requireFinite(originY, "originY");
        requireFinite(originZ, "originZ");
        if (blockDamageRadius < 1 || blockDamageRadius > MAX_BLOCK_RADIUS) {
            throw new IllegalArgumentException(
                "blockDamageRadius must be 1.." + MAX_BLOCK_RADIUS
            );
        }
        requirePositive(blockDamageFalloff, "blockDamageFalloff");
        requireProbability(blockDropChance, "blockDropChance");
        if (!SYNTHETIC_CONFIG_SOURCE_ID.equals(explosionConfigSourceId)) {
            throw new IllegalArgumentException(
                "v1 requires the server-owned synthetic explosion source"
            );
        }
        String expectedFixtureBlockAssetId = fixtureBlockAssetId(fixtureKind);
        if (!expectedFixtureBlockAssetId.equals(fixtureBlockAssetId)) {
            throw new IllegalArgumentException(
                "fixture kind and block asset ID do not match"
            );
        }
        String expectedConfigSha256 = syntheticConfigSemanticSha256(
            fixtureKind,
            fixtureBlockAssetId,
            blockDamageRadius,
            blockDamageFalloff,
            blockDropChance
        );
        if (!expectedConfigSha256.equals(explosionConfigSemanticSha256)) {
            throw new IllegalArgumentException(
                "synthetic explosion semantic SHA does not match its scalar configuration"
            );
        }
        if (
            damageEntities
                || entityDamageRadius != 0.0f
                || entityDamage != 0.0f
                || entityDamageFalloff != 0.0f
                || ignoredControlledActor
        ) {
            throw new IllegalArgumentException(
                "v1 disables entity damage and requires zero entity fields"
            );
        }

        requireCapacity(entityCapacity, MAX_ENTITY_CAPACITY, "entityCapacity");
        requireCapacity(cellCapacity, MAX_CELL_CAPACITY, "cellCapacity");
        requireCapacity(dropCapacity, MAX_DROP_CAPACITY, "dropCapacity");
        requireTotal(totalEntityAdmissions, "totalEntityAdmissions");
        requireTotal(totalChangedCells, "totalChangedCells");
        requireTotal(totalResolvedDrops, "totalResolvedDrops");
        if (
            entityCapacity != V1_ENTITY_CAPACITY
                || totalEntityAdmissions != 0
                || entityOverflow
                || !entityAdmissions.isEmpty()
        ) {
            throw new IllegalArgumentException(
                "v1 reserves one entity slot and cannot emit admissions"
            );
        }
        requireExactOverflow(
            entityOverflow,
            totalEntityAdmissions,
            entityCapacity,
            "entityOverflow"
        );
        requireExactOverflow(
            cellOverflow,
            totalChangedCells,
            cellCapacity,
            "cellOverflow"
        );
        requireExactOverflow(
            dropOverflow,
            totalResolvedDrops,
            dropCapacity,
            "dropOverflow"
        );
        if ((unsupportedScopeMask & ~KNOWN_UNSUPPORTED_SCOPE_MASK) != 0) {
            throw new IllegalArgumentException(
                "unsupportedScopeMask contains an unknown bit"
            );
        }

        boolean bounded = !entityOverflow && !cellOverflow && !dropOverflow;
        boolean supported = unsupportedScopeMask == 0;
        if (complete) {
            if (!executionStarted || !bounded || !supported || resyncRequired) {
                throw new IllegalArgumentException(
                    "complete mutation frames must execute and be bounded, supported, and synced"
                );
            }
            if (!failureReason.isEmpty()) {
                throw new IllegalArgumentException(
                    "complete mutation frames cannot carry a failure reason"
                );
            }
            if (
                entityAdmissions.size() != totalEntityAdmissions
                    || changedCells.size() != totalChangedCells
                    || resolvedDrops.size() != totalResolvedDrops
            ) {
                throw new IllegalArgumentException(
                    "complete mutation frames must emit every row"
                );
            }
            validateCompleteRows(entityAdmissions, changedCells, resolvedDrops);
        } else {
            if (failureReason.isEmpty()) {
                throw new IllegalArgumentException(
                    "incomplete mutation frames must explain their failure"
                );
            }
            if (failureReason.length() > 256) {
                throw new IllegalArgumentException("failureReason is too long");
            }
            if (
                !entityAdmissions.isEmpty()
                    || !changedCells.isEmpty()
                    || !resolvedDrops.isEmpty()
            ) {
                throw new IllegalArgumentException(
                    "incomplete mutation frames must not emit partial rows"
                );
            }
            if (resyncRequired != executionStarted) {
                throw new IllegalArgumentException(
                    "only an incomplete executed capture requires resync"
                );
            }
        }
    }

    /**
     * Canonical semantic payload hashed by the server for the synthetic v1
     * config. This attests the applied configuration, not RNG determinism;
     * native chance zero still admits the exact {@code nextFloat() == 0} edge.
     */
    public static String syntheticConfigSemantics(
        String fixtureKind,
        String fixtureBlockAssetId,
        int blockDamageRadius,
        float blockDamageFalloff,
        float blockDropChance
    ) {
        if (blockDamageRadius < 1 || blockDamageRadius > MAX_BLOCK_RADIUS) {
            throw new IllegalArgumentException(
                "blockDamageRadius must be 1.." + MAX_BLOCK_RADIUS
            );
        }
        requirePositive(blockDamageFalloff, "blockDamageFalloff");
        requireProbability(blockDropChance, "blockDropChance");
        String expectedFixtureBlockAssetId = fixtureBlockAssetId(fixtureKind);
        if (!expectedFixtureBlockAssetId.equals(fixtureBlockAssetId)) {
            throw new IllegalArgumentException(
                "fixture kind and block asset ID do not match"
            );
        }
        return String.join(
            "\n",
            "source_id=" + SYNTHETIC_CONFIG_SOURCE_ID,
            "fixture_kind=" + fixtureKind,
            "fixture_block_asset_id=" + fixtureBlockAssetId,
            "damage_blocks=true",
            "damage_entities=false",
            "block_damage_radius=" + blockDamageRadius,
            "block_damage_falloff=" + Float.toHexString(blockDamageFalloff),
            "block_drop_chance=" + Float.toHexString(blockDropChance),
            "item_tool=null",
            "knockback=null",
            "particles=null",
            "sound=null"
        );
    }

    public static String syntheticConfigSemanticSha256(
        String fixtureKind,
        String fixtureBlockAssetId,
        int blockDamageRadius,
        float blockDamageFalloff,
        float blockDropChance
    ) {
        return sha256Hex(syntheticConfigSemantics(
            fixtureKind,
            fixtureBlockAssetId,
            blockDamageRadius,
            blockDamageFalloff,
            blockDropChance
        ));
    }

    public static String fixtureBlockAssetId(String fixtureKind) {
        return switch (fixtureKind) {
            case DIRECT_DROP_FIXTURE_KIND ->
                DIRECT_DROP_FIXTURE_BLOCK_ASSET_ID;
            default -> throw new IllegalArgumentException(
                "fixtureKind must be direct_drop"
            );
        };
    }

    public boolean controlledDryScope() {
        return unsupportedScopeMask == 0;
    }

    public boolean unsupportedFluidMutation() {
        return (unsupportedScopeMask & UNSUPPORTED_FLUID_MUTATION) != 0;
    }

    public boolean unsupportedFillerMutation() {
        return (unsupportedScopeMask & UNSUPPORTED_FILLER_MUTATION) != 0;
    }

    public boolean unsupportedSupportCascade() {
        return (unsupportedScopeMask & UNSUPPORTED_SUPPORT_CASCADE) != 0;
    }

    /** Reserved for a future schema; v1 disables entity damage and emits none. */
    public record EntityAdmission(
        int ordinal,
        UUID uuid,
        double x,
        double y,
        double z,
        double distance,
        float damage
    ) {
        public EntityAdmission {
            if (ordinal < 0 || uuid == null) {
                throw new IllegalArgumentException(
                    "entity admission requires an ordinal and UUID"
                );
            }
            requireFinite(x, "entity.x");
            requireFinite(y, "entity.y");
            requireFinite(z, "entity.z");
            if (!Double.isFinite(distance) || distance < 0.0) {
                throw new IllegalArgumentException(
                    "entity admission distance must be finite and non-negative"
                );
            }
            requireNonNegative(damage, "entity.damage");
        }
    }

    /** One direct changed cell in canonical ascending XYZ order. */
    public record ChangedCell(
        int ordinal,
        int x,
        int y,
        int z,
        CellState before,
        CellState after
    ) {
        public ChangedCell {
            if (ordinal < 0 || before == null || after == null) {
                throw new IllegalArgumentException(
                    "changed cell requires an ordinal and both states"
                );
            }
            if (!before.substantivelyDiffersFrom(after)) {
                throw new IllegalArgumentException(
                    "changed cell requires a substantive block-state delta"
                );
            }
        }

        public boolean partialBlockHealth() {
            return after.blockHealthValid()
                && after.blockHealth() > 0.0f
                && after.blockHealth() < 1.0f;
        }

        public boolean dropEligibleTransition() {
            return before.blockPresent()
                && (
                    !after.blockPresent()
                        || before.runtimeBlockId() != after.runtimeBlockId()
                        || !before.blockAssetId().equals(after.blockAssetId())
                        || !before.semanticKeySha256().equals(
                            after.semanticKeySha256()
                        )
                );
        }
    }

    /**
     * Immutable scalar subset of {@link NativeMutableBlockEvidence.Row}.
     * Vector movement and collision boxes are refreshed through the existing
     * mutable-cell query after the mutation frame is accepted.
     */
    public record CellState(
        boolean blockPresent,
        String blockAssetId,
        int runtimeBlockId,
        String semanticKeySha256,
        boolean semanticKeyValid,
        boolean affordanceValid,
        int affordanceTags,
        int gatherTypeIndex,
        int requiredToolQuality,
        int rotationIndex,
        int flags,
        int fluidLevel,
        double fluidFillHeight,
        int supportValue,
        int blockDamage,
        int fluidDamage,
        float blockHealth,
        boolean blockHealthValid,
        double secondsSinceDamage,
        boolean damageAgeValid,
        short localChangeCounter,
        short globalChangeCounter
    ) {
        public CellState {
            blockAssetId = blockAssetId == null ? "" : blockAssetId;
            semanticKeySha256 = semanticKeySha256 == null
                ? ""
                : semanticKeySha256.toLowerCase(Locale.ROOT);
            if (
                semanticKeyValid != blockPresent
                    || affordanceValid != blockPresent
                    || blockHealthValid != blockPresent
            ) {
                throw new IllegalArgumentException(
                    "cell identity, affordance, and health validity follow presence"
                );
            }
            if (
                semanticKeySha256.length() != (semanticKeyValid ? 64 : 0)
                    || (
                        semanticKeyValid
                            && !semanticKeySha256.matches("[0-9a-f]{64}")
                    )
            ) {
                throw new IllegalArgumentException(
                    "cell semantic identity must be one SHA-256"
                );
            }
            if (
                (affordanceTags & ~BlockAffordanceContract.KNOWN_TAG_MASK) != 0
                    || gatherTypeIndex < 0
                    || gatherTypeIndex
                        >= BlockAffordanceContract.GATHER_TYPES.size()
                    || requiredToolQuality < 0
                    || requiredToolQuality > Short.MAX_VALUE
                    || (!affordanceValid && (
                        affordanceTags != 0
                            || gatherTypeIndex != 0
                            || requiredToolQuality != 0
                    ))
            ) {
                throw new IllegalArgumentException(
                    "cell affordance value is outside its dictionary"
                );
            }
            if (
                !Double.isFinite(fluidFillHeight)
                    || fluidFillHeight < 0.0
                    || fluidFillHeight > 1.0
                    || !Float.isFinite(blockHealth)
                    || blockHealth < 0.0f
                    || blockHealth > 1.0f
                    || (!blockPresent && blockHealth != 0.0f)
                    || !Double.isFinite(secondsSinceDamage)
                    || secondsSinceDamage < 0.0
                    || (!damageAgeValid && secondsSinceDamage != 0.0)
            ) {
                throw new IllegalArgumentException(
                    "cell scalar value is outside the mutable-block domain"
                );
            }
        }

        public boolean dry() {
            return fluidLevel == 0
                && fluidFillHeight == 0.0
                && fluidDamage == 0;
        }

        public boolean substantivelyDiffersFrom(CellState other) {
            return blockPresent != other.blockPresent
                || !blockAssetId.equals(other.blockAssetId)
                || runtimeBlockId != other.runtimeBlockId
                || !semanticKeySha256.equals(other.semanticKeySha256)
                || semanticKeyValid != other.semanticKeyValid
                || affordanceValid != other.affordanceValid
                || affordanceTags != other.affordanceTags
                || gatherTypeIndex != other.gatherTypeIndex
                || requiredToolQuality != other.requiredToolQuality
                || rotationIndex != other.rotationIndex
                || flags != other.flags
                || fluidLevel != other.fluidLevel
                || Double.compare(fluidFillHeight, other.fluidFillHeight) != 0
                || supportValue != other.supportValue
                || blockDamage != other.blockDamage
                || fluidDamage != other.fluidDamage
                || Float.compare(blockHealth, other.blockHealth) != 0
                || blockHealthValid != other.blockHealthValid;
        }

        /** Lossless scalar projection of the established mutable-cell row. */
        public static CellState fromMutableRow(
            NativeMutableBlockEvidence.Row row
        ) {
            if (row == null) {
                throw new IllegalArgumentException(
                    "mutable block row must be present"
                );
            }
            return new CellState(
                row.blockPresent(),
                row.blockAssetId(),
                row.runtimeBlockId(),
                HexFormat.of().formatHex(row.semanticKeySha256()),
                row.semanticKeyValid(),
                row.affordanceValid(),
                row.affordanceTags(),
                row.gatherTypeIndex(),
                row.requiredToolQuality(),
                row.rotationIndex(),
                row.flags(),
                row.fluidLevel(),
                row.fluidFillHeight(),
                row.supportValue(),
                row.blockDamage(),
                row.fluidDamage(),
                row.blockHealth(),
                row.blockHealthValid(),
                row.secondsSinceDamage(),
                row.damageAgeValid(),
                row.localChangeCounter(),
                row.globalChangeCounter()
            );
        }
    }

    /** One resolved native drop stack in canonical scalar order. */
    public record ResolvedDrop(
        int ordinal,
        int sourceX,
        int sourceY,
        int sourceZ,
        String itemAssetId,
        int quantity,
        double durability,
        double maxDurability,
        String metadataJsonSha256,
        double spawnX,
        double spawnY,
        double spawnZ
    ) {
        public ResolvedDrop {
            itemAssetId = requireText(itemAssetId, "itemAssetId", 512);
            metadataJsonSha256 = metadataJsonSha256 == null
                ? ""
                : metadataJsonSha256.toLowerCase(Locale.ROOT);
            if (
                ordinal < 0
                    || quantity < 1
                    || !Double.isFinite(durability)
                    || !Double.isFinite(maxDurability)
                    || durability < 0.0
                    || maxDurability < 0.0
                    || durability > maxDurability
                    || (
                        !metadataJsonSha256.isEmpty()
                            && !metadataJsonSha256.matches("[0-9a-f]{64}")
                    )
            ) {
                throw new IllegalArgumentException(
                    "resolved drop is outside its scalar domain"
                );
            }
            requireFinite(spawnX, "drop.spawnX");
            requireFinite(spawnY, "drop.spawnY");
            requireFinite(spawnZ, "drop.spawnZ");
        }
    }

    private static void validateCompleteRows(
        List<EntityAdmission> entities,
        List<ChangedCell> cells,
        List<ResolvedDrop> drops
    ) {
        Set<UUID> entityIds = new HashSet<>();
        for (int index = 0; index < entities.size(); index++) {
            EntityAdmission entity = requireElement(
                entities.get(index),
                "entity admission"
            );
            if (entity.ordinal() != index || !entityIds.add(entity.uuid())) {
                throw new IllegalArgumentException(
                    "entity admissions must be uniquely ordered"
                );
            }
        }

        Set<String> positions = new HashSet<>();
        Set<String> dropEligiblePositions = new HashSet<>();
        ChangedCell previousCell = null;
        for (int index = 0; index < cells.size(); index++) {
            ChangedCell cell = requireElement(cells.get(index), "changed cell");
            String position = positionKey(cell.x(), cell.y(), cell.z());
            if (
                cell.ordinal() != index
                    || !positions.add(position)
                    || (
                        previousCell != null
                            && compareCellPosition(previousCell, cell) >= 0
                    )
            ) {
                throw new IllegalArgumentException(
                    "changed cells must use canonical ascending XYZ order"
                );
            }
            previousCell = cell;
            if (!cell.before().dry() || !cell.after().dry()) {
                throw new IllegalArgumentException(
                    "complete v1 mutation cells must remain dry"
                );
            }
            if (cell.dropEligibleTransition()) {
                dropEligiblePositions.add(position);
            }
        }

        ResolvedDrop previousDrop = null;
        for (int index = 0; index < drops.size(); index++) {
            ResolvedDrop drop = requireElement(drops.get(index), "resolved drop");
            if (
                drop.ordinal() != index
                    || (
                        previousDrop != null
                            && compareResolvedDrop(previousDrop, drop) > 0
                    )
                    || !dropEligiblePositions.contains(positionKey(
                        drop.sourceX(),
                        drop.sourceY(),
                        drop.sourceZ()
                    ))
            ) {
                throw new IllegalArgumentException(
                    "resolved drops must be canonical and originate at a destroyed or replaced cell"
                );
            }
            previousDrop = drop;
        }
    }

    private static int compareCellPosition(ChangedCell left, ChangedCell right) {
        int comparison = Integer.compare(left.x(), right.x());
        if (comparison != 0) return comparison;
        comparison = Integer.compare(left.y(), right.y());
        return comparison != 0
            ? comparison
            : Integer.compare(left.z(), right.z());
    }

    private static int compareResolvedDrop(ResolvedDrop left, ResolvedDrop right) {
        int comparison = Integer.compare(left.sourceX(), right.sourceX());
        if (comparison != 0) return comparison;
        comparison = Integer.compare(left.sourceY(), right.sourceY());
        if (comparison != 0) return comparison;
        comparison = Integer.compare(left.sourceZ(), right.sourceZ());
        if (comparison != 0) return comparison;
        comparison = left.itemAssetId().compareTo(right.itemAssetId());
        if (comparison != 0) return comparison;
        comparison = Integer.compare(left.quantity(), right.quantity());
        if (comparison != 0) return comparison;
        comparison = Double.compare(left.durability(), right.durability());
        if (comparison != 0) return comparison;
        comparison = Double.compare(left.maxDurability(), right.maxDurability());
        if (comparison != 0) return comparison;
        comparison = left.metadataJsonSha256().compareTo(
            right.metadataJsonSha256()
        );
        if (comparison != 0) return comparison;
        comparison = Double.compare(left.spawnX(), right.spawnX());
        if (comparison != 0) return comparison;
        comparison = Double.compare(left.spawnY(), right.spawnY());
        return comparison != 0
            ? comparison
            : Double.compare(left.spawnZ(), right.spawnZ());
    }

    private static String positionKey(int x, int y, int z) {
        return x + ":" + y + ":" + z;
    }

    private static <T> List<T> copy(List<T> values) {
        return values == null ? List.of() : List.copyOf(values);
    }

    private static <T> T requireElement(T value, String name) {
        if (value == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
        return value;
    }

    private static String requireText(String value, String name, int maximum) {
        if (value == null || value.isBlank() || value.length() > maximum) {
            throw new IllegalArgumentException(name + " is blank or too long");
        }
        return value;
    }

    private static String requireSha256(String value, String name) {
        if (value == null || !value.matches("[0-9A-Fa-f]{64}")) {
            throw new IllegalArgumentException(name + " must be a SHA-256");
        }
        return value.toLowerCase(Locale.ROOT);
    }

    private static String sha256Hex(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(
                digest.digest(value.getBytes(StandardCharsets.UTF_8))
            );
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
    }

    private static void requirePositive(float value, String name) {
        if (!Float.isFinite(value) || value <= 0.0f) {
            throw new IllegalArgumentException(name + " must be positive");
        }
    }

    private static void requireNonNegative(float value, String name) {
        if (!Float.isFinite(value) || value < 0.0f) {
            throw new IllegalArgumentException(name + " must be non-negative");
        }
    }

    private static void requireProbability(float value, String name) {
        if (!Float.isFinite(value) || value < 0.0f || value > 1.0f) {
            throw new IllegalArgumentException(name + " must be in [0,1]");
        }
    }

    private static void requireCapacity(int value, int maximum, String name) {
        if (value < 1 || value > maximum) {
            throw new IllegalArgumentException(
                name + " must be in [1," + maximum + "]"
            );
        }
    }

    private static void requireTotal(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative");
        }
    }

    private static void requireExactOverflow(
        boolean overflow,
        int total,
        int capacity,
        String name
    ) {
        if (overflow != (total > capacity)) {
            throw new IllegalArgumentException(
                name + " must exactly reflect total > capacity"
            );
        }
    }
}
