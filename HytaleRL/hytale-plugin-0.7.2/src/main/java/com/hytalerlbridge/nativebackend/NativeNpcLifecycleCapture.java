package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.entity.UUIDComponent;
import com.hypixel.hytale.server.core.entity.effect.ActiveEntityEffect;
import com.hypixel.hytale.server.core.entity.effect.EffectControllerComponent;
import com.hypixel.hytale.server.core.asset.type.entityeffect.config.EntityEffect;
import com.hypixel.hytale.server.core.event.events.ecs.InventoryChangeEvent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.transaction.ItemStackTransaction;
import com.hypixel.hytale.server.core.inventory.transaction.ListTransaction;
import com.hypixel.hytale.server.core.inventory.transaction.MoveTransaction;
import com.hypixel.hytale.server.core.inventory.transaction.SlotTransaction;
import com.hypixel.hytale.server.core.inventory.transaction.Transaction;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entity.damage.Damage;
import com.hypixel.hytale.server.core.modules.entity.damage.DeathComponent;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatMap;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatValue;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.imitation.NpcLifecycleEventSnapshot;
import com.hytalerlbridge.imitation.NpcWorldSnapshot;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.UUID;
import org.joml.Vector3d;

/** Extracts generic lifecycle facts without assigning task reward or termination. */
final class NativeNpcLifecycleCapture {

    private static final String DELTA = "server_boundary_state_delta";
    private static final String INTER_ROW = "server_inter_row_state_delta";
    private static final String INVENTORY = "server_inventory_event";
    private static final String DEATH = "server_death_component";
    private static final int PROJECTILE_FLAGS =
        NpcWorldSnapshot.ENTITY_LEGACY_PROJECTILE
            | NpcWorldSnapshot.ENTITY_STANDARD_PROJECTILE
            | NpcWorldSnapshot.ENTITY_PREDICTED_PROJECTILE;

    private NativeNpcLifecycleCapture() {}

    static Boundary boundary(Ref<EntityStore> ref, Store<EntityStore> store) {
        Identity actor = identity(ref, store);
        EntityStatMap stats = store.getComponent(ref, EntityStatMap.getComponentType());
        EffectControllerComponent effects = store.getComponent(
            ref,
            EffectControllerComponent.getComponentType()
        );
        ActiveEntityEffect[] activeEffects = effects == null
            ? null
            : effects.getAllActiveEntityEffects();
        DeathComponent death = store.getComponent(ref, DeathComponent.getComponentType());
        return new Boundary(
            stats != null,
            statValues(stats),
            activeEffects != null,
            statusValues(activeEffects),
            deathValue(death),
            actor
        );
    }

    static InventoryCapture inventory(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        InventoryChangeEvent event
    ) {
        if (event == null || event.getTransaction() == null) {
            return new InventoryCapture(List.of(), 0, true);
        }
        String subject = event.getComponentType().getTypeClass().getSimpleName();
        Transaction transaction = event.getTransaction();
        Identity actor = identity(ref, store);
        List<NpcLifecycleEventSnapshot> slots = new ArrayList<>();
        boolean complete = flatten(
            transaction,
            subject,
            actor,
            slots,
            new IdentityHashMap<>()
        );
        NpcLifecycleEventSnapshot summary = event(
            "inventory_transaction",
            INVENTORY,
            subject,
            transaction.getClass().getSimpleName(),
            -1,
            0.0,
            0.0,
            0.0,
            0.0,
            "",
            "",
            transaction.succeeded(),
            complete,
            actor,
            0
        );
        List<NpcLifecycleEventSnapshot> events = new ArrayList<>(slots.size() + 1);
        events.add(summary);
        events.addAll(slots);
        return new InventoryCapture(List.copyOf(events), events.size(), !complete);
    }

    static Events derive(
        Boundary before,
        Boundary after,
        NpcWorldSnapshot beforeWorld,
        NpcWorldSnapshot afterWorld,
        List<NpcLifecycleEventSnapshot> inventoryEvents,
        int inventoryEventCount,
        boolean inventoryPartial
    ) {
        Accumulator result = new Accumulator(inventoryEvents, inventoryEventCount);
        int available = NpcLifecycleEventSnapshot.SOURCE_INVENTORY
            | NpcLifecycleEventSnapshot.SOURCE_ACTOR_LIFECYCLE
            | NpcLifecycleEventSnapshot.SOURCE_PROJECTILES;
        int partial = inventoryPartial
            ? NpcLifecycleEventSnapshot.SOURCE_INVENTORY
            : 0;

        if (before.statsAvailable() && after.statsAvailable()) {
            available |= NpcLifecycleEventSnapshot.SOURCE_STATS;
            statDeltas(before, after, result, DELTA);
        }
        if (before.statusAvailable() && after.statusAvailable()) {
            available |= NpcLifecycleEventSnapshot.SOURCE_STATUS;
            statusDeltas(before, after, result, DELTA);
        }
        if (beforeWorld.entityOverflow() || afterWorld.entityOverflow()) {
            partial |= NpcLifecycleEventSnapshot.SOURCE_PROJECTILES;
        } else {
            projectileDeltas(beforeWorld, afterWorld, result, DELTA);
        }
        deathDelta(before, after, result, DEATH);
        if (result.overflow()) partial |= available;
        return new Events(
            result.events(),
            result.count(),
            result.overflow(),
            available,
            partial
        );
    }

    static Events betweenRows(
        Boundary before,
        Boundary after,
        NpcWorldSnapshot beforeWorld,
        NpcWorldSnapshot afterWorld
    ) {
        Accumulator result = new Accumulator(List.of(), 0);
        int available = NpcLifecycleEventSnapshot.SOURCE_INVENTORY
            | NpcLifecycleEventSnapshot.SOURCE_PROJECTILES
            | NpcLifecycleEventSnapshot.SOURCE_ACTOR_LIFECYCLE;
        int partial = 0;
        if (before.statsAvailable() && after.statsAvailable()) {
            available |= NpcLifecycleEventSnapshot.SOURCE_STATS;
            statDeltas(before, after, result, INTER_ROW);
        }
        if (before.statusAvailable() && after.statusAvailable()) {
            available |= NpcLifecycleEventSnapshot.SOURCE_STATUS;
            statusDeltas(before, after, result, INTER_ROW);
        }
        if (beforeWorld.entityOverflow() || afterWorld.entityOverflow()) {
            partial |= NpcLifecycleEventSnapshot.SOURCE_PROJECTILES;
        } else {
            projectileDeltas(beforeWorld, afterWorld, result, INTER_ROW);
        }
        deathDelta(before, after, result, INTER_ROW);
        if (result.overflow()) partial |= available;
        return new Events(
            result.events(), result.count(), result.overflow(), available, partial
        );
    }

    static Events combine(Events first, Events second) {
        if (
            first.count() == 0
                && first.availableBits() == 0
                && first.partialBits() == 0
        ) {
            return second;
        }
        List<NpcLifecycleEventSnapshot> events = new ArrayList<>(
            NpcLifecycleEventSnapshot.CAPACITY
        );
        events.addAll(first.events());
        int remaining = NpcLifecycleEventSnapshot.CAPACITY - events.size();
        events.addAll(second.events().subList(
            0, Math.min(remaining, second.events().size())
        ));
        int count = first.count() + second.count();
        int available = first.availableBits() | second.availableBits();
        int partial = first.partialBits()
            | second.partialBits()
            | (first.availableBits() ^ second.availableBits());
        boolean overflow = count > events.size();
        if (overflow) partial |= available;
        return new Events(List.copyOf(events), count, overflow, available, partial);
    }

    static Events emptyEvents() {
        return new Events(List.of(), 0, false, 0, 0);
    }

    private static void statDeltas(
        Boundary before,
        Boundary after,
        Accumulator output,
        String origin
    ) {
        TreeSet<Integer> indexes = new TreeSet<>(before.stats().keySet());
        indexes.addAll(after.stats().keySet());
        for (int index : indexes) {
            StatValue previous = before.stats().get(index);
            StatValue next = after.stats().get(index);
            double oldValue = previous == null ? 0.0 : previous.value();
            double newValue = next == null ? 0.0 : next.value();
            if (Double.compare(oldValue, newValue) == 0) continue;
            StatValue descriptor = next == null ? previous : next;
            output.add(event(
                "stat_changed",
                origin,
                "actor",
                descriptor.id(),
                index,
                oldValue,
                newValue,
                descriptor.minimum(),
                descriptor.maximum(),
                "",
                "",
                true,
                true,
                after.actor(),
                0
            ));
        }
    }

    private static void statusDeltas(
        Boundary before,
        Boundary after,
        Accumulator output,
        String origin
    ) {
        TreeSet<Integer> indexes = new TreeSet<>(before.statuses().keySet());
        indexes.addAll(after.statuses().keySet());
        for (int index : indexes) {
            StatusValue previous = before.statuses().get(index);
            StatusValue next = after.statuses().get(index);
            if (previous == null && next != null) {
                output.add(statusEvent("status_added", null, next, after.actor(), origin));
            } else if (previous != null && next == null) {
                output.add(statusEvent("status_removed", previous, null, before.actor(), origin));
            } else if (
                previous != null
                    && next.remaining() > previous.remaining() + 1.0e-6f
            ) {
                output.add(statusEvent(
                    "status_refreshed",
                    previous,
                    next,
                    after.actor(),
                    origin
                ));
            }
        }
    }

    private static NpcLifecycleEventSnapshot statusEvent(
        String kind,
        StatusValue before,
        StatusValue after,
        Identity actor,
        String origin
    ) {
        StatusValue value = after == null ? before : after;
        return event(
            kind,
            origin,
            "actor",
            value.id(),
            value.index(),
            before == null ? 0.0 : before.remaining(),
            after == null ? 0.0 : after.remaining(),
            value.initial(),
            0.0,
            "",
            "",
            true,
            true,
            actor,
            value.flags()
        );
    }

    private static void projectileDeltas(
        NpcWorldSnapshot before,
        NpcWorldSnapshot after,
        Accumulator output,
        String origin
    ) {
        Map<String, NpcWorldSnapshot.Entity> previous = projectiles(before);
        Map<String, NpcWorldSnapshot.Entity> next = projectiles(after);
        TreeSet<String> keys = new TreeSet<>(previous.keySet());
        keys.addAll(next.keySet());
        for (String key : keys) {
            NpcWorldSnapshot.Entity oldEntity = previous.get(key);
            NpcWorldSnapshot.Entity newEntity = next.get(key);
            if (oldEntity == null) {
                output.add(projectileEvent("projectile_spawned", newEntity, origin));
            } else if (newEntity == null) {
                output.add(projectileEvent("projectile_despawned", oldEntity, origin));
            } else if (
                (oldEntity.flags() & NpcWorldSnapshot.ENTITY_IMPACTED_OR_BOUNCED) == 0
                    && (newEntity.flags()
                        & NpcWorldSnapshot.ENTITY_IMPACTED_OR_BOUNCED) != 0
            ) {
                output.add(projectileEvent("projectile_impacted", newEntity, origin));
            }
        }
    }

    private static NpcLifecycleEventSnapshot projectileEvent(
        String kind,
        NpcWorldSnapshot.Entity entity,
        String origin
    ) {
        return new NpcLifecycleEventSnapshot(
            kind,
            origin,
            entity.physicsState(),
            entity.assetId(),
            -1,
            kind.equals("projectile_despawned") ? 1.0 : 0.0,
            kind.equals("projectile_despawned") ? 0.0 : 1.0,
            0.0,
            0.0,
            "",
            "",
            true,
            true,
            entity.entityIndex(),
            entity.uuid(),
            entity.ownerUuid(),
            true,
            entity.position(),
            entity.flags()
        );
    }

    private static void deathDelta(
        Boundary before,
        Boundary after,
        Accumulator output,
        String origin
    ) {
        if (!before.death().present() && after.death().present()) {
            output.add(deathEvent("actor_death", after, origin));
        } else if (before.death().present() && !after.death().present()) {
            output.add(deathEvent("actor_revived", before, origin));
        }
    }

    private static NpcLifecycleEventSnapshot deathEvent(
        String kind,
        Boundary boundary,
        String origin
    ) {
        DeathValue death = boundary.death();
        return event(
            kind,
            origin,
            "actor",
            death.causeId(),
            death.causeIndex(),
            kind.equals("actor_death") ? 0.0 : 1.0,
            kind.equals("actor_death") ? 1.0 : 0.0,
            0.0,
            0.0,
            "",
            "",
            true,
            death.complete(),
            boundary.actor(),
            0
        );
    }

    private static boolean flatten(
        Transaction transaction,
        String subject,
        Identity actor,
        List<NpcLifecycleEventSnapshot> output,
        IdentityHashMap<Transaction, Boolean> seen
    ) {
        if (transaction == null || seen.put(transaction, Boolean.TRUE) != null) {
            return false;
        }
        if (transaction instanceof SlotTransaction slot) {
            output.add(slotEvent(subject, slot, actor));
            return true;
        }
        if (transaction instanceof ItemStackTransaction itemStack) {
            boolean complete = true;
            for (SlotTransaction slot : itemStack.getSlotTransactions()) {
                complete &= flatten(slot, subject, actor, output, seen);
            }
            return complete;
        }
        if (transaction instanceof ListTransaction<?> list) {
            boolean complete = true;
            for (Transaction child : list.getList()) {
                complete &= flatten(child, subject, actor, output, seen);
            }
            return complete;
        }
        if (transaction instanceof MoveTransaction<?> move) {
            return flatten(move.getRemoveTransaction(), subject, actor, output, seen)
                & flatten(move.getAddTransaction(), subject, actor, output, seen);
        }
        return false;
    }

    private static NpcLifecycleEventSnapshot slotEvent(
        String subject,
        SlotTransaction transaction,
        Identity actor
    ) {
        ItemStack before = transaction.getSlotBefore();
        ItemStack after = transaction.getSlotAfter();
        return event(
            "inventory_slot",
            INVENTORY,
            subject,
            transaction.getAction() == null ? "" : transaction.getAction().name(),
            transaction.getSlot(),
            quantity(before),
            quantity(after),
            durability(before),
            durability(after),
            itemId(before),
            itemId(after),
            transaction.succeeded(),
            true,
            actor,
            0
        );
    }

    private static NpcLifecycleEventSnapshot event(
        String kind,
        String origin,
        String subject,
        String key,
        int index,
        double before,
        double after,
        double auxiliary0,
        double auxiliary1,
        String textBefore,
        String textAfter,
        boolean successful,
        boolean complete,
        Identity actor,
        int flags
    ) {
        return new NpcLifecycleEventSnapshot(
            kind,
            origin,
            subject,
            key,
            index,
            before,
            after,
            auxiliary0,
            auxiliary1,
            textBefore,
            textAfter,
            successful,
            complete,
            actor.entityIndex(),
            actor.uuid(),
            null,
            actor.positionAvailable(),
            actor.position(),
            flags
        );
    }

    private static Map<Integer, StatValue> statValues(EntityStatMap values) {
        if (values == null) return Map.of();
        Map<Integer, StatValue> result = new TreeMap<>();
        for (int index = 0; index < values.size(); index++) {
            EntityStatValue value = values.get(index);
            if (value == null) continue;
            result.put(
                value.getIndex(),
                new StatValue(
                    value.getId(),
                    value.get(),
                    value.getMin(),
                    value.getMax()
                )
            );
        }
        return Map.copyOf(result);
    }

    private static Map<Integer, StatusValue> statusValues(
        ActiveEntityEffect[] activeEffects
    ) {
        if (activeEffects == null) return Map.of();
        Map<Integer, StatusValue> result = new TreeMap<>();
        for (ActiveEntityEffect active : activeEffects) {
            if (active == null) continue;
            int index = active.getEntityEffectIndex();
            EntityEffect asset = EntityEffect.getAssetMap().getAsset(index);
            int flags = (active.isInfinite() ? 1 : 0)
                | (active.isDebuff() ? 1 << 1 : 0)
                | (active.isInvulnerable() ? 1 << 2 : 0);
            result.put(
                index,
                new StatusValue(
                    asset == null ? "" : asset.getId(),
                    index,
                    active.getInitialDuration(),
                    active.getRemainingDuration(),
                    flags
                )
            );
        }
        return Map.copyOf(result);
    }

    private static DeathValue deathValue(DeathComponent death) {
        if (death == null) return DeathValue.ALIVE;
        Damage info = death.getDeathInfo();
        String cause = death.getDeathCause() == null
            ? ""
            : death.getDeathCause().getId();
        return new DeathValue(
            true,
            cause,
            info == null ? -1 : info.getDamageCauseIndex(),
            info != null && !cause.isBlank()
        );
    }

    private static Map<String, NpcWorldSnapshot.Entity> projectiles(
        NpcWorldSnapshot world
    ) {
        Map<String, NpcWorldSnapshot.Entity> result = new TreeMap<>();
        for (NpcWorldSnapshot.Entity entity : world.entities()) {
            if ((entity.flags() & PROJECTILE_FLAGS) == 0) continue;
            String key = entity.uuid() == null
                ? "index:" + entity.entityIndex()
                : "uuid:" + entity.uuid();
            result.put(key, entity);
        }
        return result;
    }

    private static Identity identity(Ref<EntityStore> ref, Store<EntityStore> store) {
        UUIDComponent uuid = store.getComponent(ref, UUIDComponent.getComponentType());
        TransformComponent transform = store.getComponent(
            ref,
            TransformComponent.getComponentType()
        );
        if (transform == null) {
            return new Identity(
                ref.getIndex(),
                uuid == null ? null : uuid.getUuid(),
                false,
                new double[3]
            );
        }
        Vector3d value = transform.getPosition();
        return new Identity(
            ref.getIndex(),
            uuid == null ? null : uuid.getUuid(),
            true,
            new double[] {value.x, value.y, value.z}
        );
    }

    private static int quantity(ItemStack value) {
        return ItemStack.isEmpty(value) ? 0 : value.getQuantity();
    }

    private static double durability(ItemStack value) {
        return ItemStack.isEmpty(value) ? 0.0 : value.getDurability();
    }

    private static String itemId(ItemStack value) {
        return ItemStack.isEmpty(value) ? "" : value.getItemId();
    }

    record Boundary(
        boolean statsAvailable,
        Map<Integer, StatValue> stats,
        boolean statusAvailable,
        Map<Integer, StatusValue> statuses,
        DeathValue death,
        Identity actor
    ) {}

    record InventoryCapture(
        List<NpcLifecycleEventSnapshot> events,
        int count,
        boolean partial
    ) {}

    record Events(
        List<NpcLifecycleEventSnapshot> events,
        int count,
        boolean overflow,
        int availableBits,
        int partialBits
    ) {}

    private record StatValue(String id, float value, float minimum, float maximum) {}

    private record StatusValue(
        String id,
        int index,
        float initial,
        float remaining,
        int flags
    ) {}

    private record DeathValue(
        boolean present,
        String causeId,
        int causeIndex,
        boolean complete
    ) {
        private static final DeathValue ALIVE = new DeathValue(false, "", -1, true);
    }

    private record Identity(
        int entityIndex,
        UUID uuid,
        boolean positionAvailable,
        double[] position
    ) {
        private Identity {
            position = position.clone();
        }

        @Override
        public double[] position() {
            return position.clone();
        }
    }

    private static final class Accumulator {
        private final List<NpcLifecycleEventSnapshot> events = new ArrayList<>();
        private int count;

        private Accumulator(
            List<NpcLifecycleEventSnapshot> initial,
            int initialCount
        ) {
            events.addAll(initial);
            count = initialCount;
        }

        private void add(NpcLifecycleEventSnapshot value) {
            count++;
            if (events.size() < NpcLifecycleEventSnapshot.CAPACITY) events.add(value);
        }

        private List<NpcLifecycleEventSnapshot> events() {
            return List.copyOf(events);
        }

        private int count() {
            return count;
        }

        private boolean overflow() {
            return count > events.size();
        }
    }
}
