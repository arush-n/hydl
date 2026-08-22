package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.RemoveReason;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.worldgen.NativeDropProgramEvidence;
import com.hytalerlbridge.worldgen.NativeExplosionMutationProbe;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.joml.Vector3i;

/** Captures and removes fixture-owned native item-drop entities. */
final class ExplosionMutationDrops {

    private ExplosionMutationDrops() {}

    static List<NativeExplosionMutationProbe.ResolvedDrop> resolvedDrops(
        List<ItemReading> items,
        Vector3i source
    ) {
        List<DropValue> values = new ArrayList<>(items.size());
        for (ItemReading item : items) {
            if (item.stack() == null) {
                throw new IllegalStateException(
                    "fixture-created item entity has no ItemStack"
                );
            }
            NativeDropProgramEvidence.Stack stack =
                DropRouting.canonicalDropStack(item.stack());
            values.add(new DropValue(
                source.x,
                source.y,
                source.z,
                stack.itemAssetId(),
                stack.quantity(),
                stack.durability(),
                stack.maxDurability(),
                stack.metadataJsonSha256(),
                item.x(),
                item.y(),
                item.z()
            ));
        }
        values.sort(DropValue.ORDER);
        List<NativeExplosionMutationProbe.ResolvedDrop> result =
            new ArrayList<>(values.size());
        for (DropValue value : values) {
            result.add(value.toProbe(result.size()));
        }
        return List.copyOf(result);
    }

    static void removeCreatedItems(
        Store<EntityStore> store,
        List<ItemReading> items
    ) {
        RuntimeException cleanupFailure = null;
        for (ItemReading item : items) {
            try {
                if (item.reference().isValid()) {
                    store.removeEntity(item.reference(), RemoveReason.REMOVE);
                }
                if (item.reference().isValid()) {
                    throw new IllegalStateException(
                        "fixture-created item reference remained valid"
                    );
                }
            } catch (RuntimeException exception) {
                if (cleanupFailure == null) cleanupFailure = exception;
                else cleanupFailure.addSuppressed(exception);
            }
        }
        if (cleanupFailure != null) {
            throw new IllegalStateException(
                "fixture-created item cleanup failed",
                cleanupFailure
            );
        }
    }

    record ItemReading(
        Ref<EntityStore> reference,
        ItemStack stack,
        double x,
        double y,
        double z
    ) {}

    private record DropValue(
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
        private static final Comparator<DropValue> ORDER = Comparator
            .comparingInt(DropValue::sourceX)
            .thenComparingInt(DropValue::sourceY)
            .thenComparingInt(DropValue::sourceZ)
            .thenComparing(DropValue::itemAssetId)
            .thenComparingInt(DropValue::quantity)
            .thenComparingDouble(DropValue::durability)
            .thenComparingDouble(DropValue::maxDurability)
            .thenComparing(DropValue::metadataJsonSha256)
            .thenComparingDouble(DropValue::spawnX)
            .thenComparingDouble(DropValue::spawnY)
            .thenComparingDouble(DropValue::spawnZ);

        private NativeExplosionMutationProbe.ResolvedDrop toProbe(int ordinal) {
            return new NativeExplosionMutationProbe.ResolvedDrop(
                ordinal,
                sourceX,
                sourceY,
                sourceZ,
                itemAssetId,
                quantity,
                durability,
                maxDurability,
                metadataJsonSha256,
                spawnX,
                spawnY,
                spawnZ
            );
        }
    }
}
