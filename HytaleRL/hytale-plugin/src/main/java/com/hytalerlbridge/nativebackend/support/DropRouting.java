package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockBreakingDropType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockGathering;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.HarvestingDropType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.SoftBlockDropType;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hytalerlbridge.worldgen.NativeDropProgramEvidence;
import java.util.List;
import com.hytalerlbridge.nativebackend.model.ResolvedDropRoute;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.sha256Hex;

/**
 * Extracted verbatim from {@code NativeEnvironmentSession}.
 *
 * <p>Every method here touches no session instance field, so the move
 * required no state surgery.
 */
public final class DropRouting {

    private DropRouting() {}

    public static ResolvedDropRoute resolveDropRoute(
        BlockType blockType,
        String route
    ) {
        if (!NativeDropProgramEvidence.ROUTES.contains(route)) {
            throw new IllegalArgumentException(
                "Unsupported native drop route: " + route
            );
        }
        BlockGathering gathering = blockType.getGathering();
        if (gathering == null) {
            throw new IllegalArgumentException(
                "Block has no gathering routes: " + blockType.getId()
            );
        }
        return switch (route) {
            case "breaking" -> {
                BlockBreakingDropType value = gathering.getBreaking();
                if (value == null) {
                    throw missingDropRoute(blockType, route);
                }
                yield new ResolvedDropRoute(
                    value.getQuantity(),
                    value.getItemId(),
                    value.getDropListId()
                );
            }
            case "soft" -> {
                SoftBlockDropType value = gathering.getSoft();
                if (value == null) {
                    throw missingDropRoute(blockType, route);
                }
                yield new ResolvedDropRoute(
                    1,
                    value.getItemId(),
                    value.getDropListId()
                );
            }
            case "harvest" -> {
                HarvestingDropType value = gathering.getHarvest();
                if (value == null) {
                    throw missingDropRoute(blockType, route);
                }
                yield new ResolvedDropRoute(
                    1,
                    value.getItemId(),
                    value.getDropListId()
                );
            }
            default -> throw new AssertionError(route);
        };
    }

    public static String canonicalDropOutcomeKey(
        List<NativeDropProgramEvidence.Stack> stacks
    ) {
        StringBuilder key = new StringBuilder();
        for (NativeDropProgramEvidence.Stack stack : stacks) {
            String item = stack.itemAssetId();
            key.append(item.length()).append(':').append(item)
                .append('|').append(stack.quantity())
                .append('|').append(Double.toHexString(stack.durability()))
                .append('|').append(
                    Double.toHexString(stack.maxDurability())
                )
                .append('|').append(stack.metadataJsonSha256())
                .append(';');
        }
        return key.toString();
    }

    public static NativeDropProgramEvidence.Stack canonicalDropStack(
        ItemStack stack
    ) {
        var metadata = stack.getMetadata();
        String metadataSha256 = metadata == null || metadata.isEmpty()
            ? ""
            : sha256Hex(metadata.toJson());
        return new NativeDropProgramEvidence.Stack(
            stack.getItemId(),
            stack.getQuantity(),
            stack.getDurability(),
            stack.getMaxDurability(),
            metadataSha256
        );
    }

    public static IllegalArgumentException missingDropRoute(
        BlockType blockType,
        String route
    ) {
        return new IllegalArgumentException(
            "Block " + blockType.getId() + " has no " + route + " route"
        );
    }
}
