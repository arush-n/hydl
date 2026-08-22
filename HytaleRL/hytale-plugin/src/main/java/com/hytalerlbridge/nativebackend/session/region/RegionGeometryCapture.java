package com.hytalerlbridge.nativebackend.session.region;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.shape.Box;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.protocol.BlockMaterial;
import com.hypixel.hytale.protocol.FluidFXMovementSettings;
import com.hypixel.hytale.server.core.asset.type.blockhitbox.BlockBoundingBoxes;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockMovementSettings;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.RotationTuple;
import com.hypixel.hytale.server.core.asset.type.fluid.Fluid;
import com.hypixel.hytale.server.core.asset.type.fluidfx.config.FluidFX;
import com.hypixel.hytale.server.core.modules.blockhealth.BlockHealth;
import com.hypixel.hytale.server.core.modules.blockhealth.BlockHealthChunk;
import com.hypixel.hytale.server.core.modules.collision.BlockCollisionData;
import com.hypixel.hytale.server.core.modules.collision.CollisionResult;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.modules.entity.component.CollisionResultComponent;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.chunk.WorldChunk;
import com.hypixel.hytale.server.core.universe.world.chunk.section.BlockSection;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.core.util.FillerBlockUtil;
import com.hypixel.hytale.server.npc.role.Role;
import com.hytalerlbridge.geometry.GeometryCell;
import com.hytalerlbridge.geometry.GeometryContact;
import com.hytalerlbridge.geometry.GeometryContract;
import com.hytalerlbridge.geometry.GeometryFrame;
import com.hytalerlbridge.nativebackend.model.DoubleArrayKey;
import com.hytalerlbridge.nativebackend.model.NativeCellSemantics;
import com.hytalerlbridge.nativebackend.model.RegionBlockSemanticKey;
import com.hytalerlbridge.nativebackend.model.RegionCellKey;
import com.hytalerlbridge.nativebackend.support.NativeCellSemanticsCache;
import com.hytalerlbridge.worldgen.BlockAffordanceContract;
import com.hytalerlbridge.worldgen.NativeMutableBlockEvidence;
import com.hytalerlbridge.worldgen.NativeRegionBlockSemanticSection;
import com.hytalerlbridge.worldgen.NativeRegionFluidSemanticSection;
import com.hytalerlbridge.worldgen.NativeRegionSection;
import com.hytalerlbridge.worldgen.RegionCellPaletteEntry;
import com.hytalerlbridge.worldgen.RegionShapePaletteEntry;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.joml.Vector3d;
import org.joml.Vector3i;
import static com.hytalerlbridge.nativebackend.support.GeometryContacts.appendTouchingGeometryContacts;
import static com.hytalerlbridge.nativebackend.support.GeometryContacts.blocksDefaultNpcLineOfSight;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.canonicalWorldVerbInventoryState;
import static com.hytalerlbridge.nativebackend.support.GeometryContacts.collisionBoxes;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.implementationVersion;
import static com.hytalerlbridge.nativebackend.support.GeometryContacts.isDiagonalLosFixture;
import static com.hytalerlbridge.nativebackend.support.GeometryContacts.lineOfSightOffset;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.mutableBlockAssetKey;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.mutableBlockSemanticKey;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.sha256Hex;

/**
 * Region, cell and block-semantic capture lifted out of NativeEnvironmentSession.
 *
 * <p>These six methods were 765 lines of that class and referenced only three of
 * its 152 fields -- {@code world}, {@code loadedChunkIndices} and the
 * {@code BLOCK_RADIUS} constant -- and called none of its private helpers. They
 * are pure readers of world state, so they are static and take what they need
 * explicitly rather than reaching into session state.
 */
public final class RegionGeometryCapture {

    private RegionGeometryCapture() {}

    private static final int BLOCK_RADIUS = GeometryContract.RADIUS;

    private static final boolean NATIVE_CELL_SEMANTICS_CACHE =
        Boolean.parseBoolean(System.getProperty(
            "hytalerl.native.cell_semantics_cache",
            "true"
        ));

    public static GeometryFrame captureGeometry(
        NativeCellSemanticsCache nativeCellSemanticsCache,
        World world,
        Store<EntityStore> store,
        Ref<EntityStore> actorReference,
        Ref<EntityStore> targetReference,
        Role role,
        int originX,
        int originY,
        int originZ,
        Vector3d agentPosition,
        boolean targetPresent
    ) {
        List<GeometryCell> cells = new ArrayList<>();
        Map<Long, WorldChunk> chunks = new HashMap<>();
        for (int dx = -BLOCK_RADIUS; dx <= BLOCK_RADIUS; dx++) {
            for (int dy = -BLOCK_RADIUS; dy <= BLOCK_RADIUS; dy++) {
                for (int dz = -BLOCK_RADIUS; dz <= BLOCK_RADIUS; dz++) {
                    int x = originX + dx;
                    int y = originY + dy;
                    int z = originZ + dz;
                    long chunkIndex = ChunkUtil.indexChunkFromBlock(x, z);
                    WorldChunk chunk = chunks.get(chunkIndex);
                    if (chunk == null) {
                        chunk = world.getChunkIfLoaded(chunkIndex);
                        if (chunk != null) chunks.put(chunkIndex, chunk);
                    }
                    if (chunk == null) {
                        return GeometryFrame.empty();
                    }
                    NativeCellSemantics semantics;
                    try {
                        semantics = captureNativeCell(nativeCellSemanticsCache, chunk, x, y, z);
                    } catch (IllegalStateException error) {
                        String message = error.getMessage();
                        if (
                            message == null
                                || !message.startsWith("Native fluid")
                        ) {
                            throw error;
                        }
                        // Fresh generated sections can expose a runtime fluid
                        // ID before its versioned asset is resolvable. The
                        // observation contract has an explicit geometry-
                        // availability bit, so abstain instead of killing the
                        // world thread or fabricating an empty fluid cell.
                        return GeometryFrame.empty();
                    }
                    if (semantics != null) {
                        cells.add(semantics.geometryCell(dx, dy, dz));
                    }
                }
            }
        }

        BoundingBox boundingBoxComponent = store.getComponent(
            actorReference,
            BoundingBox.getComponentType()
        );
        Box agentBox = boundingBoxComponent == null
            ? null
            : boundingBoxComponent.getBoundingBox();
        double[] agentBounds = agentBox == null
            ? new double[6]
            : new double[] {
                agentBox.min.x,
                agentBox.min.y,
                agentBox.min.z,
                agentBox.max.x,
                agentBox.max.y,
                agentBox.max.z
            };
        BoundingBox targetBoundingBoxComponent =
            targetReference == null || !targetReference.isValid()
                ? null
                : store.getComponent(
                    targetReference,
                    BoundingBox.getComponentType()
                );
        Box targetBox = targetBoundingBoxComponent == null
            ? null
            : targetBoundingBoxComponent.getBoundingBox();
        double[] targetBounds = targetBox == null
            ? new double[6]
            : new double[] {
                targetBox.min.x,
                targetBox.min.y,
                targetBox.min.z,
                targetBox.max.x,
                targetBox.max.y,
                targetBox.max.z
            };

        List<GeometryContact> contacts = new ArrayList<>();
        boolean ceilingContact = false;
        CollisionResultComponent collisionComponent = store.getComponent(
            actorReference,
            CollisionResultComponent.getComponentType()
        );
        CollisionResult collisionResult = collisionComponent == null
            ? null
            : collisionComponent.getCollisionResult();
        if (collisionResult != null) {
            int count = collisionResult.getBlockCollisionCount();
            for (int index = 0; index < count; index++) {
                BlockCollisionData collision = collisionResult.getBlockCollision(index);
                if (collision == null) continue;
                boolean touchingCeiling = collision.touching
                    && collision.collisionNormal.y < -0.5;
                ceilingContact |= touchingCeiling;
                contacts.add(new GeometryContact(
                    collision.x - originX,
                    collision.y - originY,
                    collision.z - originZ,
                    collision.detailBoxIndex,
                    collision.collisionNormal.x,
                    collision.collisionNormal.y,
                    collision.collisionNormal.z,
                    collision.collisionPoint.x - originX,
                    collision.collisionPoint.y - originY,
                    collision.collisionPoint.z - originZ,
                    collision.collisionStart,
                    collision.collisionEnd,
                    collision.touching,
                    collision.overlapping
                ));
            }
        }
        if (agentBox != null) {
            int beforeCanonicalContacts = contacts.size();
            appendTouchingGeometryContacts(
                cells,
                agentPosition,
                agentBox,
                originX,
                originY,
                originZ,
                contacts
            );
            for (int index = beforeCanonicalContacts; index < contacts.size(); index++) {
                ceilingContact |= contacts.get(index).normalY() < -0.5;
            }
        }

        boolean targetLineOfSightValid = targetPresent
            && targetReference != null
            && targetReference.isValid()
            && role != null
            && role.getPositionCache() != null;
        boolean targetLineOfSight = targetLineOfSightValid
            && role.getPositionCache().hasLineOfSight(
                actorReference,
                targetReference,
                store
            );
        double[] agentLosOffset = lineOfSightOffset(
            store,
            actorReference,
            true
        );
        double[] targetLosOffset = targetLineOfSightValid
            ? lineOfSightOffset(store, targetReference, false)
            : new double[3];
        return new GeometryFrame(
            true,
            agentBox != null,
            originX,
            originY,
            originZ,
            cells,
            agentBounds,
            targetBounds,
            agentLosOffset,
            targetLosOffset,
            contacts,
            role != null && role.isOnGround(),
            ceilingContact,
            targetLineOfSight,
            targetLineOfSightValid
        );
    }

    public static NativeCellSemantics captureNativeCell(
        NativeCellSemanticsCache nativeCellSemanticsCache,
        WorldChunk chunk,
        int x,
        int y,
        int z
    ) {
        int localX = ChunkUtil.localCoordinate((long) x);
        int localZ = ChunkUtil.localCoordinate((long) z);
        int runtimeBlockId = chunk.getBlock(x, y, z);
        int fluidId = chunk.getFluidId(localX, y, localZ);
        if (runtimeBlockId == 0 && fluidId == 0) return null;

        int rotationIndex = chunk.getRotationIndex(localX, y, localZ);
        int fluidLevel = fluidId == 0
            ? 0
            : Byte.toUnsignedInt(chunk.getFluidLevel(localX, y, localZ));
        int supportValue = chunk.getSupportValue(localX, y, localZ);
        NativeCellSemanticsCache.Key semanticKey = null;
        if (NATIVE_CELL_SEMANTICS_CACHE) {
            semanticKey = new NativeCellSemanticsCache.Key(
                runtimeBlockId,
                fluidId,
                rotationIndex,
                fluidLevel,
                supportValue
            );
            NativeCellSemantics cached = nativeCellSemanticsCache.lookup(
                semanticKey
            );
            if (cached != null) return cached;
        }
        BlockType blockType = BlockType.getAssetMap().getAsset(runtimeBlockId);
        Fluid fluid = fluidId == 0
            ? null
            : Fluid.getAssetMap().getAsset(fluidId);
        double fluidFillHeight = 0.0;
        if (fluidId != 0) {
            if (fluid == null || fluid == Fluid.EMPTY) {
                throw new IllegalStateException(
                    "Native fluid cell has no versioned Fluid asset"
                );
            }
            int maximumFluidLevel = fluid.getMaxFluidLevel();
            if (maximumFluidLevel <= 0 || fluidLevel > maximumFluidLevel) {
                throw new IllegalStateException(
                    "Native fluid fill level is outside its asset range"
                );
            }
            fluidFillHeight = (double) fluidLevel / maximumFluidLevel;
        }
        FluidFXMovementSettings fluidMovement = null;
        if (fluid != null && fluid != Fluid.EMPTY) {
            FluidFX fluidFx = FluidFX.getAssetMap().getAsset(
                fluid.getFluidFXIndex()
            );
            if (fluidFx != null) {
                fluidMovement = fluidFx.getMovementSettings();
            }
        }
        boolean solid = blockType != null
            && blockType.getMaterial() == BlockMaterial.Solid;

        int flags = 0;
        if (solid) flags |= GeometryContract.FLAG_SOLID;
        if (blocksDefaultNpcLineOfSight(runtimeBlockId, blockType)) {
            flags |= GeometryContract.FLAG_OPAQUE;
        }
        if (fluidId != 0) flags |= GeometryContract.FLAG_FLUID;
        int blockDamage = blockType == null
            ? 0
            : blockType.getDamageToEntities();
        int fluidDamage = fluid == null
            ? 0
            : fluid.getDamageToEntities();
        if (blockDamage != 0 || fluidDamage != 0) {
            flags |= GeometryContract.FLAG_DAMAGING;
        }
        if (blockType != null && blockType.isTrigger()) {
            flags |= GeometryContract.FLAG_TRIGGER;
        }

        BlockMovementSettings movement = blockType == null
            ? null
            : blockType.getMovementSettings();
        if (movement != null) {
            flags |= GeometryContract.FLAG_HAS_MOVEMENT_SETTINGS;
            if (movement.isClimbable()) {
                flags |= GeometryContract.FLAG_CLIMBABLE;
            }
            if (movement.isBouncy()) {
                flags |= GeometryContract.FLAG_BOUNCY;
            }
        }
        if (fluidMovement != null) {
            flags |= GeometryContract.FLAG_HAS_FLUID_MOVEMENT_SETTINGS;
        }

        int hitboxTypeIndex = blockType == null
            ? BlockBoundingBoxes.DEFAULT_ID
            : blockType.getHitboxTypeIndex();
        BlockBoundingBoxes hitboxAsset = BlockBoundingBoxes.getAssetMap()
            .getAssetOrDefault(
                hitboxTypeIndex,
                BlockBoundingBoxes.UNIT_BOX
            );
        if (hitboxAsset.protrudesUnitBox()) {
            flags |= GeometryContract.FLAG_PROTRUDES_CELL;
        }
        int shapeId = 1
            + Math.max(0, hitboxTypeIndex) * RotationTuple.VALUES.length
            + Math.max(0, rotationIndex);
        double[] collisionBoxes = solid
            ? collisionBoxes(hitboxAsset, rotationIndex)
            : new double[0];
        double[] movementValues = {
            movement == null ? 0.0 : movement.getFriction(),
            movement == null ? 0.0 : movement.getDrag(),
            movement == null ? 0.0 : movement.getHorizontalSpeedMultiplier(),
            movement == null ? 0.0 : movement.jumpForceMultiplier(),
            movement == null ? 0.0 : movement.getClimbUpSpeedMultiplier(),
            movement == null ? 0.0 : movement.getClimbDownSpeedMultiplier(),
            movement == null ? 0.0 : movement.getClimbLateralSpeedMultiplier(),
            movement == null ? 0.0 : movement.getTerminalVelocityModifier(),
            movement == null ? 0.0 : movement.getBounceVelocity()
        };
        double[] fluidMovementValues = {
            fluidMovement == null ? 0.0 : fluidMovement.swimUpSpeed,
            fluidMovement == null ? 0.0 : fluidMovement.swimDownSpeed,
            fluidMovement == null ? 0.0 : fluidMovement.sinkSpeed,
            fluidId == 0
                ? 0.0
                : fluidMovement == null
                    ? 1.0
                    : fluidMovement.horizontalSpeedMultiplier,
            fluidId == 0
                ? 0.0
                : fluidMovement == null
                    ? 1.0
                    : fluidMovement.fieldOfViewMultiplier,
            fluidId == 0
                ? 0.0
                : fluidMovement == null
                    ? 1.0
                    : fluidMovement.entryVelocityMultiplier
        };
        NativeCellSemantics semantics = new NativeCellSemantics(
            runtimeBlockId,
            fluidId,
            shapeId,
            rotationIndex,
            flags,
            fluidLevel,
            fluidFillHeight,
            supportValue,
            blockDamage,
            fluidDamage,
            movementValues,
            fluidMovementValues,
            collisionBoxes
        );
        return NATIVE_CELL_SEMANTICS_CACHE
            ? nativeCellSemanticsCache.store(semanticKey, semantics)
            : semantics;
    }

    public static NativeRegionSection buildRegionSection(
        NativeCellSemanticsCache nativeCellSemanticsCache,
        World world,
        Set<Long> loadedChunkIndices,
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        long chunkIndex = ChunkUtil.indexChunk(chunkX, chunkZ);
        if (!loadedChunkIndices.contains(chunkIndex)) {
            throw new IllegalStateException(
                "Region section chunk was not pinned before capture"
            );
        }
        WorldChunk chunk = world.getChunkIfLoaded(chunkIndex);
        if (chunk == null) {
            throw new IllegalStateException(
                "Region section chunk is not loaded"
            );
        }

        byte[] codes = new byte[NativeRegionSection.CODE_BYTES];
        byte[] fillerRootOffsets = new byte[
            NativeRegionSection.FILLER_ROOT_BYTES
        ];
        List<RegionShapePaletteEntry> shapes = new ArrayList<>();
        shapes.add(new RegionShapePaletteEntry(new double[0]));
        Map<DoubleArrayKey, Integer> shapeIndices = new HashMap<>();
        shapeIndices.put(new DoubleArrayKey(new double[0]), 0);

        List<RegionCellPaletteEntry> cells = new ArrayList<>();
        RegionCellPaletteEntry air = new RegionCellPaletteEntry(
            0,
            0,
            0,
            0.0,
            0,
            0,
            0,
            new double[RegionCellPaletteEntry.MOVEMENT_VALUES],
            new double[RegionCellPaletteEntry.FLUID_MOVEMENT_VALUES]
        );
        cells.add(air);
        Map<RegionCellKey, Integer> cellIndices = new HashMap<>();
        cellIndices.put(RegionCellKey.from(air), 0);

        int baseX = chunkX * ChunkUtil.SIZE;
        int baseY = sectionY * ChunkUtil.SIZE;
        int baseZ = chunkZ * ChunkUtil.SIZE;
        BlockSection blockSection = chunk
            .getBlockChunk()
            .getSectionAtIndex(sectionY);
        for (int localY = 0; localY < ChunkUtil.SIZE; localY++) {
            for (int localZ = 0; localZ < ChunkUtil.SIZE; localZ++) {
                for (int localX = 0; localX < ChunkUtil.SIZE; localX++) {
                    int worldX = baseX + localX;
                    int worldY = baseY + localY;
                    int worldZ = baseZ + localZ;
                    int filler = blockSection.getFiller(
                        worldX,
                        worldY,
                        worldZ
                    );
                    if ((filler & ~0x7fff) != 0) {
                        throw new IllegalStateException(
                            "Native filler-root value exceeds packed 15-bit capacity"
                        );
                    }
                    NativeCellSemantics semantics = captureNativeCell(
                        nativeCellSemanticsCache,
                        chunk,
                        worldX,
                        worldY,
                        worldZ
                    );
                    if (semantics == null && filler != FillerBlockUtil.NO_FILLER) {
                        throw new IllegalStateException(
                            "Native air cell carries a filler-root offset"
                        );
                    }

                    int cellIndex = 0;
                    if (semantics != null) {
                        DoubleArrayKey shapeKey = new DoubleArrayKey(
                            semantics.collisionBoxes()
                        );
                        Integer shapeIndex = shapeIndices.get(shapeKey);
                        if (shapeIndex == null) {
                            if (shapes.size() >= NativeRegionSection.MAX_SHAPES) {
                                throw new IllegalStateException(
                                    "Native section exceeds Region v1 shape capacity"
                                );
                            }
                            shapeIndex = shapes.size();
                            shapeIndices.put(shapeKey, shapeIndex);
                            shapes.add(new RegionShapePaletteEntry(
                                semantics.collisionBoxes()
                            ));
                        }

                        RegionCellPaletteEntry cell = semantics.regionEntry(
                            shapeIndex
                        );
                        RegionCellKey cellKey = RegionCellKey.from(cell);
                        Integer existingCellIndex = cellIndices.get(cellKey);
                        if (existingCellIndex == null) {
                            if (cells.size() >= NativeRegionSection.MAX_CELLS) {
                                throw new IllegalStateException(
                                    "Native section exceeds Region v1 cell capacity"
                                );
                            }
                            existingCellIndex = cells.size();
                            cellIndices.put(cellKey, existingCellIndex);
                            cells.add(cell);
                        }
                        cellIndex = existingCellIndex;
                    }

                    int index = localY * ChunkUtil.SIZE_2
                        + localZ * ChunkUtil.SIZE
                        + localX;
                    int offset = index * 2;
                    codes[offset] = (byte) (cellIndex & 0xff);
                    codes[offset + 1] = (byte) ((cellIndex >>> 8) & 0xff);
                    fillerRootOffsets[offset] = (byte) (filler & 0xff);
                    fillerRootOffsets[offset + 1] = (byte) (
                        (filler >>> 8) & 0xff
                    );
                }
            }
        }
        return new NativeRegionSection(
            chunkX,
            chunkZ,
            sectionY,
            codes,
            fillerRootOffsets,
            cells,
            shapes
        );
    }

    public static NativeRegionBlockSemanticSection buildRegionBlockSemanticSection(
        World world,
        Set<Long> loadedChunkIndices,
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        long chunkIndex = ChunkUtil.indexChunk(chunkX, chunkZ);
        if (!loadedChunkIndices.contains(chunkIndex)) {
            throw new IllegalStateException(
                "Region block-semantic chunk was not pinned before capture"
            );
        }
        WorldChunk chunk = world.getChunkIfLoaded(chunkIndex);
        if (chunk == null) {
            throw new IllegalStateException(
                "Region block-semantic chunk is not loaded"
            );
        }
        byte[] codes = new byte[
            NativeRegionBlockSemanticSection.CODE_BYTES
        ];
        List<NativeRegionBlockSemanticSection.Entry> palette =
            new ArrayList<>();
        palette.add(NativeRegionBlockSemanticSection.Entry.air());
        Map<RegionBlockSemanticKey, Integer> indices = new HashMap<>();
        int baseX = chunkX * ChunkUtil.SIZE;
        int baseY = sectionY * ChunkUtil.SIZE;
        int baseZ = chunkZ * ChunkUtil.SIZE;
        BlockSection section = chunk
            .getBlockChunk()
            .getSectionAtIndex(sectionY);
        for (int localY = 0; localY < ChunkUtil.SIZE; localY++) {
            for (int localZ = 0; localZ < ChunkUtil.SIZE; localZ++) {
                for (int localX = 0; localX < ChunkUtil.SIZE; localX++) {
                    int worldX = baseX + localX;
                    int worldY = baseY + localY;
                    int worldZ = baseZ + localZ;
                    int runtimeBlockId = world.getBlock(
                        worldX,
                        worldY,
                        worldZ
                    );
                    int paletteIndex = 0;
                    if (runtimeBlockId != 0) {
                        BlockType blockType = world.getBlockType(
                            worldX,
                            worldY,
                            worldZ
                        );
                        if (
                            blockType == null
                                || blockType == BlockType.EMPTY
                        ) {
                            throw new IllegalStateException(
                                "Native block has no versioned BlockType"
                            );
                        }
                        int rotation = section.getRotationIndex(
                            worldX,
                            worldY,
                            worldZ
                        );
                        BlockAffordanceContract.Value affordance =
                            BlockAffordanceContract.resolve(blockType);
                        if (!affordance.valid()) {
                            throw new IllegalStateException(
                                "Native block affordance is unavailable for "
                                    + blockType.getId()
                            );
                        }
                        RegionBlockSemanticKey key =
                            new RegionBlockSemanticKey(
                                blockType.getId(),
                                rotation,
                                affordance.tags(),
                                affordance.gatherTypeIndex(),
                                affordance.requiredToolQuality()
                            );
                        Integer existing = indices.get(key);
                        if (existing == null) {
                            if (
                                palette.size()
                                    >= NativeRegionBlockSemanticSection
                                        .MAX_ENTRIES
                            ) {
                                throw new IllegalStateException(
                                    "Native section exceeds uint16 block-"
                                        + "semantic palette capacity"
                                );
                            }
                            existing = palette.size();
                            indices.put(key, existing);
                            palette.add(
                                new NativeRegionBlockSemanticSection.Entry(
                                    mutableBlockSemanticKey(
                                        key.assetId(),
                                        key.rotationIndex()
                                    ),
                                    mutableBlockAssetKey(key.assetId()),
                                    true,
                                    key.affordanceTags(),
                                    key.gatherTypeIndex(),
                                    key.requiredToolQuality(),
                                    key.rotationIndex()
                                )
                            );
                        }
                        paletteIndex = existing;
                    }
                    int index = localY * ChunkUtil.SIZE_2
                        + localZ * ChunkUtil.SIZE
                        + localX;
                    int offset = index * 2;
                    codes[offset] = (byte) (paletteIndex & 0xff);
                    codes[offset + 1] = (byte) (
                        (paletteIndex >>> 8) & 0xff
                    );
                }
            }
        }
        return new NativeRegionBlockSemanticSection(
            chunkX,
            chunkZ,
            sectionY,
            codes,
            palette
        );
    }

    public static NativeRegionFluidSemanticSection buildRegionFluidSemanticSection(
        World world,
        Set<Long> loadedChunkIndices,
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        long chunkIndex = ChunkUtil.indexChunk(chunkX, chunkZ);
        if (!loadedChunkIndices.contains(chunkIndex)) {
            throw new IllegalStateException(
                "Region fluid-semantic chunk was not pinned before capture"
            );
        }
        WorldChunk chunk = world.getChunkIfLoaded(chunkIndex);
        if (chunk == null) {
            throw new IllegalStateException(
                "Region fluid-semantic chunk is not loaded"
            );
        }
        byte[] codes = new byte[NativeRegionFluidSemanticSection.CODE_BYTES];
        List<NativeRegionFluidSemanticSection.Entry> palette =
            new ArrayList<>();
        palette.add(NativeRegionFluidSemanticSection.Entry.empty());
        Map<String, Integer> indices = new HashMap<>();
        int baseY = sectionY * ChunkUtil.SIZE;
        for (int localY = 0; localY < ChunkUtil.SIZE; localY++) {
            for (int localZ = 0; localZ < ChunkUtil.SIZE; localZ++) {
                for (int localX = 0; localX < ChunkUtil.SIZE; localX++) {
                    int worldY = baseY + localY;
                    int runtimeFluidId = chunk.getFluidId(
                        localX,
                        worldY,
                        localZ
                    );
                    int paletteIndex = 0;
                    if (runtimeFluidId != 0) {
                        Fluid fluid = Fluid.getAssetMap().getAsset(
                            runtimeFluidId
                        );
                        if (fluid == null || fluid == Fluid.EMPTY) {
                            throw new IllegalStateException(
                                "Native fluid has no versioned Fluid asset"
                            );
                        }
                        String assetId = fluid.getId();
                        if (assetId == null || assetId.isBlank()) {
                            throw new IllegalStateException(
                                "Native fluid has no stable asset ID"
                            );
                        }
                        Integer existing = indices.get(assetId);
                        if (existing == null) {
                            if (
                                palette.size()
                                    >= NativeRegionFluidSemanticSection
                                        .MAX_ENTRIES
                            ) {
                                throw new IllegalStateException(
                                    "Native section exceeds uint16 fluid-"
                                        + "semantic palette capacity"
                                );
                            }
                            existing = palette.size();
                            indices.put(assetId, existing);
                            palette.add(
                                new NativeRegionFluidSemanticSection.Entry(
                                    assetId
                                )
                            );
                        }
                        paletteIndex = existing;
                    }
                    int index = localY * ChunkUtil.SIZE_2
                        + localZ * ChunkUtil.SIZE
                        + localX;
                    int offset = index * 2;
                    codes[offset] = (byte) (paletteIndex & 0xff);
                    codes[offset + 1] = (byte) (
                        (paletteIndex >>> 8) & 0xff
                    );
                }
            }
        }
        return new NativeRegionFluidSemanticSection(
            chunkX,
            chunkZ,
            sectionY,
            codes,
            palette
        );
    }

    public static NativeMutableBlockEvidence.Row mutableBlockRow(
        NativeCellSemanticsCache nativeCellSemanticsCache,
        World world,
        String phase,
        Vector3i position,
        WorldChunk chunk,
        BlockHealthChunk health,
        Instant gameTime
    ) {
        NativeCellSemantics cell = captureNativeCell(
            nativeCellSemanticsCache,
            chunk,
            position.x,
            position.y,
            position.z
        );
        boolean present = cell != null && cell.runtimeBlockId() != 0;
        BlockType blockType = present
            ? world.getBlockType(position.x, position.y, position.z)
            : null;
        String blockAssetId = blockType == null ? "" : blockType.getId();
        BlockAffordanceContract.Value affordance =
            BlockAffordanceContract.resolve(blockType);
        BlockHealth stored = present
            ? health.getBlockHealthMap().get(position)
            : null;
        double age = stored == null
            ? 0.0
            : Math.max(
                0.0,
                Duration.between(
                    stored.getLastDamageGameTime(),
                    gameTime
                ).toNanos() / 1_000_000_000.0
            );
        BlockSection section = chunk.getBlockChunk().getSectionAtBlockY(
            position.y
        );
        return new NativeMutableBlockEvidence.Row(
            phase,
            present,
            blockAssetId,
            present ? cell.runtimeBlockId() : 0,
            present
                ? mutableBlockSemanticKey(
                    blockAssetId,
                    cell.rotationIndex()
                )
                : new byte[0],
            present,
            affordance.valid(),
            affordance.tags(),
            affordance.gatherTypeIndex(),
            affordance.requiredToolQuality(),
            present ? cell.rotationIndex() : 0,
            cell == null ? 0 : cell.flags(),
            cell == null ? 0 : cell.fluidLevel(),
            cell == null ? 0.0 : cell.fluidFillHeight(),
            cell == null ? 0 : cell.supportValue(),
            cell == null ? 0 : cell.blockDamage(),
            cell == null ? 0 : cell.fluidDamage(),
            cell == null ? new double[9] : cell.movement(),
            cell == null ? new double[6] : cell.fluidMovement(),
            cell == null ? new double[0] : cell.collisionBoxes(),
            present ? health.getBlockHealth(position) : 0.0f,
            present,
            age,
            stored != null,
            section.getLocalChangeCounter(),
            section.getGlobalChangeCounter()
        );
    }
}
