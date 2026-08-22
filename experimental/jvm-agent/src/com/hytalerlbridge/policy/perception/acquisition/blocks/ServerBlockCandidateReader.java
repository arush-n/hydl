package com.hytalerlbridge.policy.perception.acquisition.blocks;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.vector.Transform;
import com.hypixel.hytale.protocol.BlockFace;
import com.hypixel.hytale.protocol.GameMode;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.item.config.Item;
import com.hypixel.hytale.server.core.entity.InteractionContext;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.inventory.ActiveSlotInventoryComponent;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.interaction.InteractionModule;
import com.hypixel.hytale.server.core.modules.interaction.interaction.UnarmedInteractions;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.InteractionConfiguration;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.BreakBlockInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.PlaceBlockInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.data.ListCollector;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.core.util.TargetUtil;
import com.hytalerlbridge.policy.world.model.BlockCandidateBinding;
import com.hytalerlbridge.policy.world.model.WorldVerbBinding;
import java.util.Arrays;
import org.joml.Vector3d;
import org.joml.Vector3i;

/** Live one-slot camera candidate matching Hytale 0.5.7 block targeting. */
public final class ServerBlockCandidateReader {

    public static final int CANDIDATE_CAPACITY = 16;
    public static final int FEATURE_SIZE = 28;
    public static final double CAMERA_RAY_DISTANCE = 8.0;
    public static final double INTERACTION_DISTANCE_BUFFER = 2.0;

    private ServerBlockCandidateReader() {}

    public static Capture capture(
        Ref<EntityStore> ref,
        Store<EntityStore> store
    ) {
        if (ref == null || !ref.isValid() || store == null) {
            return Capture.unavailable("controlled_actor_unavailable");
        }
        World world = store.getExternalData().getWorld();
        TransformComponent body = store.getComponent(
            ref, TransformComponent.getComponentType());
        if (world == null || body == null) {
            return Capture.unavailable("world_or_transform_unavailable");
        }

        Transform look;
        Vector3i raw;
        Vector3i action;
        try {
            look = TargetUtil.getLook(ref, store);
            raw = TargetUtil.getTargetBlock(ref, CAMERA_RAY_DISTANCE, store);
            action = TargetUtil.getTargetBlockOrigin(
                ref, CAMERA_RAY_DISTANCE, store);
        } catch (RuntimeException unavailable) {
            return Capture.unavailable("native_camera_target_unavailable");
        }
        if (raw == null && action == null) return Capture.empty();
        if (raw == null || action == null) {
            return Capture.unavailable("camera_target_canonicalization_failed");
        }
        if (world.getBlock(raw.x, raw.y, raw.z) == 0
            || world.getBlock(action.x, action.y, action.z) == 0) {
            return Capture.unavailable("camera_target_changed_during_capture");
        }
        BlockType type = world.getBlockType(action.x, action.y, action.z);
        BlockAffordanceProjection.Value affordance =
            BlockAffordanceProjection.resolve(type);
        if (!affordance.valid() || type == null || type.getId() == null
            || type.getId().isBlank()) {
            return Capture.unavailable("block_affordance_unavailable");
        }

        double maximum = interactionDistance(ref, store);
        Vector3d eye = look.getPosition();
        double dx = action.x + 0.5 - eye.x;
        double dy = action.y + 0.5 - eye.y;
        double dz = action.z + 0.5 - eye.z;
        if (!Double.isFinite(maximum) || maximum <= 0.0) {
            return Capture.unavailable("interaction_distance_invalid");
        }
        if (dx * dx + dy * dy + dz * dz > maximum * maximum) {
            return Capture.empty();
        }

        Vector3d origin = body.getPosition();
        float[] features = new float[CANDIDATE_CAPACITY * FEATURE_SIZE];
        features[0] = clip((raw.x + 0.5 - origin.x) / maximum);
        features[1] = clip((raw.y + 0.5 - origin.y) / maximum);
        features[2] = clip((raw.z + 0.5 - origin.z) / maximum);
        float[] tail = BlockAffordanceProjection.features(affordance);
        System.arraycopy(tail, 0, features, 3, tail.length);
        boolean[] mask = new boolean[CANDIDATE_CAPACITY];
        mask[0] = true;
        int face = aimFace(look, raw);
        BlockCandidateBinding[] bindings = emptyBindings();
        bindings[0] = new BlockCandidateBinding(
            true,
            raw.x, raw.y, raw.z,
            action.x, action.y, action.z,
            type.getId(),
            blockBinding(
                ref, store, world, look, raw, action, face, maximum,
                InteractionType.Primary),
            blockBinding(
                ref, store, world, look, raw, action, face, maximum,
                InteractionType.Secondary),
            useBinding(ref, store, world, look, action, type)
        );
        return new Capture(features, mask, true, bindings, maximum, "");
    }

    private static WorldVerbBinding blockBinding(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        Transform look,
        Vector3i rawTarget,
        Vector3i actionTarget,
        int face,
        double maximum,
        InteractionType interactionType
    ) {
        ItemStack source = activeHotbarItem(ref, store);
        InventoryComponent.Hotbar hotbar = store.getComponent(
            ref, InventoryComponent.Hotbar.getComponentType());
        int slot = activeSlot(hotbar);
        InteractionManager manager = store.getComponent(
            ref, InteractionModule.get().getInteractionManagerComponent());
        if (ItemStack.isEmpty(source) || source.getItem() == null || slot < 0
            || manager == null) {
            return WorldVerbBinding.empty();
        }
        InteractionContext context;
        try {
            context = InteractionContext.forInteraction(
                manager, ref, interactionType, store);
        } catch (RuntimeException unavailable) {
            return WorldVerbBinding.empty();
        }
        ItemStack selected = context.getHeldItem();
        if (ItemStack.isEmpty(selected)
            || !source.getItemId().equals(selected.getItemId())
            || Byte.toUnsignedInt(context.getHeldItemSlot()) != slot) {
            return WorldVerbBinding.empty();
        }
        Item item = source.getItem();
        String interactionId = item.getInteractions().get(interactionType);
        RootInteraction root = interactionId == null ? null
            : RootInteraction.getAssetMap().getAsset(interactionId);
        InteractionKinds kinds = root == null ? InteractionKinds.empty()
            : interactionKinds(root, context, interactionType);
        boolean place = kinds.places() == 1 && kinds.breaks() == 0;
        boolean breaking = kinds.breaks() == 1 && kinds.places() == 0;
        if (root == null || (!place && !breaking)) {
            return WorldVerbBinding.empty();
        }

        Vector3i commitTarget = place ? rawTarget : actionTarget;
        String blockId = "";
        String expectedBlockId;
        String placementVariant = "";
        if (place) {
            if (face == BlockFace.None.getValue()) {
                return WorldVerbBinding.empty();
            }
            Vector3i direction = faceDirection(face);
            commitTarget = new Vector3i(rawTarget).add(direction);
            blockId = source.getBlockKey();
            if (blockId == null || blockId.isBlank()
                || commitTarget.y < 0 || commitTarget.y >= 320
                || world.getBlock(
                    commitTarget.x, commitTarget.y, commitTarget.z) != 0
                || !withinReach(look.getPosition(), commitTarget, maximum)) {
                return WorldVerbBinding.empty();
            }
            expectedBlockId = "Empty";
            placementVariant = "default";
        } else {
            BlockType current = world.getBlockType(
                actionTarget.x, actionTarget.y, actionTarget.z);
            if (current == null || current == BlockType.EMPTY
                || current.getId() == null || current.getId().isBlank()) {
                return WorldVerbBinding.empty();
            }
            expectedBlockId = current.getId();
        }
        return new WorldVerbBinding(
            true,
            place ? "place_block" : "break_block",
            interactionType.getValue(), interactionId, "",
            source.getItemId(), blockId,
            commitTarget.x, commitTarget.y, commitTarget.z,
            face, 0, 0, 0,
            "hotbar", slot, source.getQuantity(),
            expectedBlockId, placementVariant);
    }

    private static WorldVerbBinding useBinding(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        Transform look,
        Vector3i action,
        BlockType targetType
    ) {
        InteractionManager manager = store.getComponent(
            ref, InteractionModule.get().getInteractionManagerComponent());
        if (manager == null || targetType == null) {
            return WorldVerbBinding.empty();
        }
        InteractionContext context;
        try {
            context = InteractionContext.forInteraction(
                manager, ref, InteractionType.Use, store);
        } catch (RuntimeException unavailable) {
            return WorldVerbBinding.empty();
        }
        Item held = context.getOriginalItemType();
        ItemStack stack = context.getHeldItem();
        String interactionId;
        String itemId;
        String sourceContainer;
        int sourceSlot;
        int sourceQuantity;
        if (held == null) {
            UnarmedInteractions unarmed =
                UnarmedInteractions.getAssetMap().getAsset("Empty");
            interactionId = unarmed == null ? null
                : unarmed.getInteractions().get(InteractionType.Use);
            itemId = "Empty";
            sourceContainer = "unarmed";
            sourceSlot = -1;
            sourceQuantity = 0;
        } else {
            if (ItemStack.isEmpty(stack)) return WorldVerbBinding.empty();
            interactionId = held.getInteractions().get(InteractionType.Use);
            itemId = stack.getItemId();
            sourceContainer = "interaction_context";
            sourceSlot = Byte.toUnsignedInt(context.getHeldItemSlot());
            sourceQuantity = stack.getQuantity();
        }
        String blockInteractionId =
            targetType.getInteractions().get(InteractionType.Use);
        if (interactionId == null || blockInteractionId == null
            || RootInteraction.getAssetMap().getAsset(interactionId) == null
            || RootInteraction.getAssetMap().getAsset(blockInteractionId) == null
            || !withinReach(
                look.getPosition(),
                action,
                useDistance(held))) {
            return WorldVerbBinding.empty();
        }
        return new WorldVerbBinding(
            true, "use", InteractionType.Use.getValue(), interactionId,
            blockInteractionId, itemId, "",
            action.x, action.y, action.z,
            0, 0, 0, 0,
            sourceContainer, sourceSlot, sourceQuantity,
            targetType.getId(), "");
    }

    private static InteractionKinds interactionKinds(
        RootInteraction root,
        InteractionContext context,
        InteractionType type
    ) {
        ListCollector<Object> collector = new ListCollector<>(
            (_tag, _context, interaction) -> interaction);
        try {
            InteractionManager.walkChain(collector, type, context, root);
        } catch (RuntimeException unavailable) {
            return InteractionKinds.empty();
        }
        int breaks = 0;
        int places = 0;
        for (Object interaction : collector.getList()) {
            if (interaction instanceof BreakBlockInteraction) breaks++;
            if (interaction instanceof PlaceBlockInteraction) places++;
        }
        return new InteractionKinds(breaks, places);
    }

    private static int aimFace(Transform look, Vector3i block) {
        Vector3d origin = look.getPosition();
        Vector3d direction = look.getDirection();
        double[] o = {origin.x, origin.y, origin.z};
        double[] d = {direction.x, direction.y, direction.z};
        double[] minimum = {block.x, block.y, block.z};
        double[] entry = new double[3];
        double exit = Double.POSITIVE_INFINITY;
        for (int axis = 0; axis < 3; axis++) {
            if (Math.abs(d[axis]) <= 1.0e-7) {
                if (o[axis] < minimum[axis]
                    || o[axis] >= minimum[axis] + 1.0) {
                    return BlockFace.None.getValue();
                }
                entry[axis] = Double.NEGATIVE_INFINITY;
                continue;
            }
            double first = (minimum[axis] - o[axis]) / d[axis];
            double second = (minimum[axis] + 1.0 - o[axis]) / d[axis];
            entry[axis] = Math.min(first, second);
            exit = Math.min(exit, Math.max(first, second));
        }
        int axis = 0;
        for (int candidate = 1; candidate < 3; candidate++) {
            if (entry[candidate] > entry[axis]) axis = candidate;
        }
        double selected = entry[axis];
        int ties = 0;
        for (double value : entry) {
            if (Math.abs(value - selected) <= 1.0e-6) ties++;
        }
        if (ties != 1 || selected < 0.0 || selected > exit) {
            return BlockFace.None.getValue();
        }
        int[] positive = {
            BlockFace.West.getValue(), BlockFace.Down.getValue(),
            BlockFace.North.getValue()};
        int[] negative = {
            BlockFace.East.getValue(), BlockFace.Up.getValue(),
            BlockFace.South.getValue()};
        return d[axis] > 0.0 ? positive[axis] : negative[axis];
    }

    private static Vector3i faceDirection(int face) {
        return switch (face) {
            case 1 -> new Vector3i(0, 1, 0);
            case 2 -> new Vector3i(0, -1, 0);
            case 3 -> new Vector3i(0, 0, -1);
            case 4 -> new Vector3i(0, 0, 1);
            case 5 -> new Vector3i(1, 0, 0);
            case 6 -> new Vector3i(-1, 0, 0);
            default -> new Vector3i();
        };
    }

    private static boolean withinReach(
        Vector3d eye,
        Vector3i target,
        double maximum
    ) {
        double dx = target.x + 0.5 - eye.x;
        double dy = target.y + 0.5 - eye.y;
        double dz = target.z + 0.5 - eye.z;
        return Double.isFinite(maximum) && maximum > 0.0
            && dx * dx + dy * dy + dz * dz <= maximum * maximum;
    }

    private static double useDistance(Item item) {
        InteractionConfiguration configuration = item == null
            ? null : item.getInteractionConfig();
        if (configuration == null) {
            configuration = InteractionConfiguration.DEFAULT;
        }
        return configuration.getUseDistance(GameMode.Adventure)
            + INTERACTION_DISTANCE_BUFFER;
    }

    private record InteractionKinds(int breaks, int places) {
        private static InteractionKinds empty() {
            return new InteractionKinds(0, 0);
        }
    }

    static double interactionDistance(
        Ref<EntityStore> ref,
        Store<EntityStore> store
    ) {
        ItemStack held = activeHotbarItem(ref, store);
        InteractionConfiguration configuration = held == null
            || held.getItem() == null
            || held.getItem().getInteractionConfig() == null
            ? InteractionConfiguration.DEFAULT
            : held.getItem().getInteractionConfig();
        return configuration.getUseDistance(GameMode.Adventure)
            + INTERACTION_DISTANCE_BUFFER;
    }

    static ItemStack activeHotbarItem(
        Ref<EntityStore> ref,
        Store<EntityStore> store
    ) {
        InventoryComponent.Hotbar component = store.getComponent(
            ref, InventoryComponent.Hotbar.getComponentType());
        ItemContainer hotbar = component == null
            ? null : component.getInventory();
        int slot = activeSlot(component);
        if (hotbar == null || slot < 0
            || slot >= Short.toUnsignedInt(hotbar.getCapacity())) {
            return null;
        }
        ItemStack stack = hotbar.getItemStack((short) slot);
        return ItemStack.isEmpty(stack) ? null : stack;
    }

    static int activeSlot(ActiveSlotInventoryComponent component) {
        if (component == null) return -1;
        byte value = component.getActiveSlot();
        return value < 0 ? -1 : Byte.toUnsignedInt(value);
    }

    private static float clip(double value) {
        return (float) Math.max(-1.0, Math.min(1.0, value));
    }

    private static BlockCandidateBinding[] emptyBindings() {
        BlockCandidateBinding[] result =
            new BlockCandidateBinding[CANDIDATE_CAPACITY];
        Arrays.fill(result, BlockCandidateBinding.empty());
        return result;
    }

    public record Capture(
        float[] candidateFeatures,
        boolean[] candidateMask,
        boolean available,
        BlockCandidateBinding[] bindings,
        double maximumDistance,
        String reason
    ) {
        public Capture {
            if (candidateFeatures == null
                || candidateFeatures.length != CANDIDATE_CAPACITY * FEATURE_SIZE
                || candidateMask == null
                || candidateMask.length != CANDIDATE_CAPACITY
                || bindings == null
                || bindings.length != CANDIDATE_CAPACITY) {
                throw new IllegalArgumentException("block candidate width drift");
            }
            candidateFeatures = candidateFeatures.clone();
            candidateMask = candidateMask.clone();
            bindings = bindings.clone();
            reason = reason == null ? "" : reason;
        }

        @Override public float[] candidateFeatures() {
            return candidateFeatures.clone();
        }
        @Override public boolean[] candidateMask() {
            return candidateMask.clone();
        }
        @Override public BlockCandidateBinding[] bindings() {
            return bindings.clone();
        }

        public boolean primaryAvailable() {
            return Arrays.stream(bindings).anyMatch(
                binding -> binding.primary().available());
        }

        public boolean secondaryAvailable() {
            return Arrays.stream(bindings).anyMatch(
                binding -> binding.secondary().available());
        }

        public boolean useAvailable() {
            return Arrays.stream(bindings).anyMatch(
                binding -> binding.use().available());
        }

        public static Capture empty() {
            return new Capture(
                new float[CANDIDATE_CAPACITY * FEATURE_SIZE],
                new boolean[CANDIDATE_CAPACITY], true, emptyBindings(), 0.0, "");
        }

        public static Capture unavailable(String reason) {
            return new Capture(
                new float[CANDIDATE_CAPACITY * FEATURE_SIZE],
                new boolean[CANDIDATE_CAPACITY], false, emptyBindings(),
                0.0, reason);
        }
    }
}
