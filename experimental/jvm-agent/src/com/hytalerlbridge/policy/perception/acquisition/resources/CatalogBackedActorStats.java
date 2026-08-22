package com.hytalerlbridge.policy.perception.acquisition.resources;

import com.hypixel.hytale.component.ComponentType;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatMap;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatsModule;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;

/**
 * Settles a policy-controlled actor's sparse stat map against Hytale's catalog.
 *
 * <p>The operation is allowed only at an explicit policy ownership boundary:
 * when a policy claims an actor, or immediately after a diagnostic fixture
 * equips one. Ordinary perception stays read-only. No resource, role, item,
 * or weapon is named here.</p>
 */
public final class CatalogBackedActorStats {

    public enum Boundary {
        CONTROLLED_ACTOR_CLAIM,
        FIXTURE_EQUIP,
    }

    public record Settlement(
        Boundary boundary,
        boolean replaced,
        int sourceSize,
        int catalogSize,
        int missingCatalogRows
    ) {
        public Settlement {
            if (boundary == null || sourceSize < 0 || catalogSize < 0
                || missingCatalogRows < 0) {
                throw new IllegalArgumentException("invalid actor-stat settlement");
            }
        }
    }

    private CatalogBackedActorStats() {}

    /** Settle once at an explicit policy claim or fixture-equip boundary. */
    public static Settlement settle(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        Boundary boundary
    ) {
        if (reference == null || !reference.isValid() || store == null
            || boundary == null) {
            throw new IllegalArgumentException(
                "actor-stat settlement requires a live actor, store, and boundary");
        }
        ComponentType<EntityStore, EntityStatMap> statMapType =
            EntityStatsModule.get().getEntityStatMapComponentType();
        EntityStatMap source = store.getComponent(reference, statMapType);
        if (source == null) {
            throw new IllegalStateException(
                "policy-controlled actor has no native entity-stat map");
        }

        // EntityStatMap.clone() starts from the current asset catalog and then
        // copies the role's exact values and modifiers. This is the same public
        // native primitive used by the bridge session path.
        EntityStatMap catalogBacked = source.clone();
        boolean[] sourceRows = rowPresence(source, catalogBacked.size());
        boolean[] catalogRows = rowPresence(catalogBacked, catalogBacked.size());
        int missing = missingCatalogRows(sourceRows, catalogRows);
        EntityStatMap effective = source;
        if (missing > 0) {
            // EntityStatMap is already in the archetype, so this replaces the
            // value without changing the actor's component shape.
            store.replaceComponent(reference, statMapType, catalogBacked);
            effective = catalogBacked;
        }
        effective.getStatModifiersManager().scheduleRecalculate();
        effective.getStatModifiersManager().recalculateEntityStatModifiers(
            reference, effective, store);
        return new Settlement(
            boundary, missing > 0, source.size(), catalogBacked.size(), missing);
    }

    /** Pure shape predicate used by fixed and generated sparse-map tests. */
    public static int missingCatalogRows(
        boolean[] sourceRows,
        boolean[] catalogRows
    ) {
        if (sourceRows == null || catalogRows == null
            || sourceRows.length != catalogRows.length) {
            throw new IllegalArgumentException(
                "stat-row presence arrays must have equal width");
        }
        int missing = 0;
        for (int index = 0; index < sourceRows.length; index++) {
            if (catalogRows[index] && !sourceRows[index]) missing++;
        }
        return missing;
    }

    /** The same catalog rule applies at both controlled-actor boundaries. */
    public static int missingCatalogRows(
        Boundary boundary,
        boolean[] sourceRows,
        boolean[] catalogRows
    ) {
        if (boundary == null) {
            throw new IllegalArgumentException("settlement boundary is required");
        }
        return missingCatalogRows(sourceRows, catalogRows);
    }

    private static boolean[] rowPresence(EntityStatMap map, int width) {
        boolean[] result = new boolean[width];
        for (int index = 0; index < width; index++) {
            result[index] = index < map.size() && map.get(index) != null;
        }
        return result;
    }
}
