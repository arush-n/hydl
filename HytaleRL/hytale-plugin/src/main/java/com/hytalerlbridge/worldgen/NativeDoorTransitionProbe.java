package com.hytalerlbridge.worldgen;

import com.hypixel.hytale.component.CommandBuffer;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.BlockPosition;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.math.util.MathUtil;
import com.hypixel.hytale.server.core.asset.type.blockhitbox.BlockBoundingBoxes;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.Rotation;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.RotationTuple;
import com.hypixel.hytale.server.core.entity.InteractionContext;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.interaction.InteractionModule;
import com.hypixel.hytale.server.core.modules.interaction.interaction.CooldownHandler;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.server.DoorInteraction;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.chunk.WorldChunk;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import java.util.ArrayList;
import java.util.List;
import org.joml.Vector3d;
import org.joml.Vector3i;

/** Executes native vertical-door state logic on one isolated world. */
public final class NativeDoorTransitionProbe {

    public static final String DOOR_ASSET_ID = "Furniture_Kweebec_Door";
    public static final String ROOT_INTERACTION_ID = "Door";
    private static final int CLEAR_RADIUS = 3;
    private static final int CLEAR_HEIGHT = 4;
    private static final VerticalDoorInvoker VERTICAL_DOOR_INVOKER =
        new VerticalDoorInvoker();

    private NativeDoorTransitionProbe() {
    }

    public static NativeDoorTransitionEvidence capture(
        World world,
        Store<EntityStore> store,
        Ref<EntityStore> agentRef,
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed
    ) {
        if (world == null || store == null || agentRef == null || !agentRef.isValid()) {
            throw new IllegalStateException(
                "Native door evidence requires a live world and agent"
            );
        }
        TransformComponent transform = store.getComponent(
            agentRef,
            TransformComponent.getComponentType()
        );
        if (transform == null) {
            throw new IllegalStateException(
                "Native door evidence agent has no TransformComponent"
            );
        }
        RootInteraction root = RootInteraction.getAssetMap().getAsset(
            ROOT_INTERACTION_ID
        );
        if (
            root == null
                || root.getInteractionIds().length != 1
                || !ROOT_INTERACTION_ID.equals(root.getInteractionIds()[0])
                || !(Interaction.getAssetMap().getAsset(ROOT_INTERACTION_ID)
                    instanceof DoorInteraction)
        ) {
            throw new IllegalStateException(
                "Installed Door root no longer resolves to DoorInteraction"
            );
        }

        BlockType baseDoor = BlockType.getAssetMap().getAsset(DOOR_ASSET_ID);
        if (baseDoor == null || !baseDoor.isDoor() || baseDoor.getItem() == null) {
            throw new IllegalStateException(
                "Installed vertical-door fixture asset is unavailable"
            );
        }
        BlockType canonicalDoor = BlockType.getAssetMap().getAsset(
            baseDoor.getItem().getId()
        );
        BlockBoundingBoxes boxes = canonicalDoor == null
            ? null
            : BlockBoundingBoxes.getAssetMap().getAsset(
                canonicalDoor.getHitboxTypeIndex()
            );
        if (boxes == null) {
            throw new IllegalStateException(
                "Installed vertical-door fixture has no base hitbox"
            );
        }
        int partnerDistance = (int) boxes.get(
            Rotation.None,
            Rotation.None,
            Rotation.None
        ).getBoundingBox().getMax().x * 2 - 1;

        Vector3d originalPosition = new Vector3d(transform.getPosition());
        int rootX = (int) Math.floor(originalPosition.x) + 4;
        int rootY = (int) Math.floor(originalPosition.y);
        int rootZ = (int) Math.floor(originalPosition.z) - 8;
        Vector3i rootPosition = new Vector3i(rootX, rootY, rootZ);
        List<NativeDoorTransitionEvidence.Row> rows = new ArrayList<>(
            NativeDoorTransitionEvidence.YAW_COUNT
        );
        try {
            for (Rotation yaw : Rotation.NORMAL) {
                rows.add(
                    captureYaw(
                        world,
                        store,
                        agentRef,
                        transform,
                        rootPosition,
                        yaw,
                        partnerDistance
                    )
                );
            }
        } finally {
            transform.setPosition(originalPosition);
            clearControlledVolume(world, rootPosition);
        }
        return new NativeDoorTransitionEvidence(
            serverVersion,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            DOOR_ASSET_ID,
            ROOT_INTERACTION_ID,
            rows
        );
    }

    private static NativeDoorTransitionEvidence.Row captureYaw(
        World world,
        Store<EntityStore> store,
        Ref<EntityStore> agentRef,
        TransformComponent transform,
        Vector3i rootPosition,
        Rotation yaw,
        int partnerDistance
    ) {
        Vector3i normal = MathUtil.rotateVectorYAxis(
            new Vector3i(0, 0, 1),
            yaw.getDegrees(),
            false
        );
        Vector3i partnerOffset = MathUtil.rotateVectorYAxis(
            new Vector3i(partnerDistance, 0, 0),
            yaw.getDegrees(),
            false
        );
        Vector3i partnerPosition = new Vector3i(rootPosition).add(partnerOffset);

        clearControlledVolume(world, rootPosition);
        placeDoor(world, rootPosition, yaw);
        setActorSide(transform, rootPosition, normal, -2.0);
        useDoor(world, store, agentRef, rootPosition);
        DoorSnapshot singleFront = snapshot(world, rootPosition);
        setActorSide(transform, rootPosition, normal, -2.0);
        useDoor(world, store, agentRef, rootPosition);
        boolean singleClosedAfterFront = snapshot(
            world,
            rootPosition
        ).closed();
        setActorSide(transform, rootPosition, normal, 2.0);
        useDoor(world, store, agentRef, rootPosition);
        DoorSnapshot singleBack = snapshot(world, rootPosition);
        setActorSide(transform, rootPosition, normal, 2.0);
        useDoor(world, store, agentRef, rootPosition);
        boolean singleClosedAfterBack = snapshot(
            world,
            rootPosition
        ).closed();

        clearControlledVolume(world, rootPosition);
        placeDoor(world, rootPosition, yaw);
        placeDoor(world, partnerPosition, yaw.flip());
        setActorSide(transform, rootPosition, normal, -2.0);
        useDoor(world, store, agentRef, rootPosition);
        DoorSnapshot doubleRoot = snapshot(world, rootPosition);
        DoorSnapshot doublePartner = snapshot(world, partnerPosition);
        setActorSide(transform, rootPosition, normal, -2.0);
        useDoor(world, store, agentRef, rootPosition);
        boolean doubleRootClosed = snapshot(world, rootPosition).closed();
        boolean doublePartnerClosed = snapshot(
            world,
            partnerPosition
        ).closed();

        return new NativeDoorTransitionEvidence.Row(
            yaw.getDegrees(),
            new int[] {
                partnerOffset.x,
                partnerOffset.y,
                partnerOffset.z
            },
            singleFront.state(),
            singleFront.hitboxType(),
            singleClosedAfterFront,
            singleBack.state(),
            singleBack.hitboxType(),
            singleClosedAfterBack,
            doubleRoot.state(),
            doubleRoot.hitboxType(),
            doublePartner.state(),
            doublePartner.hitboxType(),
            doubleRootClosed,
            doublePartnerClosed
        );
    }

    private static void placeDoor(
        World world,
        Vector3i position,
        Rotation yaw
    ) {
        WorldChunk chunk = world.getChunkIfLoaded(
            ChunkUtil.indexChunkFromBlock(position.x, position.z)
        );
        boolean placedAccepted = chunk != null && chunk.placeBlock(
            position.x,
            position.y,
            position.z,
            DOOR_ASSET_ID,
            yaw,
            Rotation.None,
            Rotation.None,
            0
        );
        BlockType placed = world.getBlockType(position);
        int expectedRotation = RotationTuple.index(
            yaw,
            Rotation.None,
            Rotation.None
        );
        int actualRotation = world.getBlockRotationIndex(
            position.x,
            position.y,
            position.z
        );
        if (
            !placedAccepted
                || placed == null
                || !placed.isDoor()
                || actualRotation != expectedRotation
                || DoorInteraction.getDoorAtPosition(
                    world.getChunkStore(),
                    position.x,
                    position.y,
                    position.z,
                    yaw
                ) == null
        ) {
            throw new IllegalStateException(
                "Native door fixture placement or rotation was rejected: "
                    + "accepted=" + placedAccepted
                    + ", expectedRotation=" + expectedRotation
                    + ", actualRotation=" + actualRotation
            );
        }
    }

    private static void useDoor(
        World world,
        Store<EntityStore> store,
        Ref<EntityStore> agentRef,
        Vector3i target
    ) {
        InteractionManager manager = interactionManager(agentRef, store);
        if (manager == null) {
            throw new IllegalStateException(
                "Native door evidence agent has no InteractionManager"
            );
        }
        InteractionContext context = InteractionContext.forInteraction(
            manager,
            agentRef,
            InteractionType.Use,
            store
        );
        BlockPosition block = new BlockPosition(target.x, target.y, target.z);
        context.getMetaStore().putMetaObject(Interaction.TARGET_BLOCK, block);
        context.getMetaStore().putMetaObject(
            Interaction.TARGET_BLOCK_RAW,
            block
        );
        ProbeCommandBuffer buffer = new ProbeCommandBuffer(store);
        VERTICAL_DOOR_INVOKER.apply(
            world,
            buffer,
            context,
            target
        );
    }

    private static InteractionManager interactionManager(
        Ref<EntityStore> agentRef,
        Store<EntityStore> store
    ) {
        InteractionModule module = InteractionModule.get();
        return module == null
            ? null
            : store.getComponent(
                agentRef,
                module.getInteractionManagerComponent()
            );
    }

    private static DoorSnapshot snapshot(World world, Vector3i position) {
        BlockType block = world.getBlockType(position);
        if (block == null || !block.isDoor()) {
            throw new IllegalStateException(
                "Native door fixture lost its root block"
            );
        }
        String state = block.getStateForBlock(block);
        return new DoorSnapshot(
            state == null ? "closed" : state,
            block.getHitboxType()
        );
    }

    private static void setActorSide(
        TransformComponent transform,
        Vector3i door,
        Vector3i normal,
        double distance
    ) {
        transform.setPosition(
            new Vector3d(
                door.x + 0.5 + normal.x * distance,
                door.y,
                door.z + 0.5 + normal.z * distance
            )
        );
    }

    private static void clearControlledVolume(
        World world,
        Vector3i root
    ) {
        for (int y = root.y; y < root.y + CLEAR_HEIGHT; y++) {
            for (int z = root.z - CLEAR_RADIUS; z <= root.z + CLEAR_RADIUS; z++) {
                for (
                    int x = root.x - CLEAR_RADIUS;
                    x <= root.x + CLEAR_RADIUS;
                    x++
                ) {
                    world.setBlock(x, y, z, BlockType.EMPTY_KEY);
                }
            }
        }
    }

    private record DoorSnapshot(String state, String hitboxType) {
        private boolean closed() {
            return !state.equals("OpenDoorIn")
                && !state.equals("OpenDoorOut");
        }
    }

    private static final class ProbeCommandBuffer
        extends CommandBuffer<EntityStore> {
        private ProbeCommandBuffer(Store<EntityStore> store) {
            super(store);
        }
    }

    private static final class VerticalDoorInvoker extends DoorInteraction {
        private void apply(
            World world,
            CommandBuffer<EntityStore> buffer,
            InteractionContext context,
            Vector3i target
        ) {
            interactWithBlock(
                world,
                buffer,
                InteractionType.Use,
                context,
                context.getHeldItem(),
                target,
                new CooldownHandler()
            );
        }
    }
}
