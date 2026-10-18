package com.hytalerlbridge.imitation;

import java.util.Set;
import java.util.UUID;

/** One model-neutral actor/world lifecycle fact aligned to a native transition. */
public record NpcLifecycleEventSnapshot(
    String kind,
    String origin,
    String subject,
    String key,
    int index,
    double valueBefore,
    double valueAfter,
    double auxiliary0,
    double auxiliary1,
    String textBefore,
    String textAfter,
    boolean successful,
    boolean complete,
    int entityIndex,
    UUID entityUuid,
    UUID ownerUuid,
    boolean positionAvailable,
    double[] position,
    int flags
) {
    public static final int CAPACITY = 64;
    public static final int WIDTH = 19;
    public static final String LAYOUT =
        "kind:utf8,origin:utf8,subject:utf8,key:utf8,index:i32,"
            + "value_before:f64,value_after:f64,auxiliary_0:f64,auxiliary_1:f64,"
            + "text_before:utf8,text_after:utf8,successful:bool,complete:bool,"
            + "entity_index:i32,entity_uuid:uuid?,owner_uuid:uuid?,"
            + "position_available:bool,position:f64[3],flags:i32";
    public static final int SOURCE_STATS = 1;
    public static final int SOURCE_STATUS = 1 << 1;
    public static final int SOURCE_INVENTORY = 1 << 2;
    public static final int SOURCE_PROJECTILES = 1 << 3;
    public static final int SOURCE_ACTOR_LIFECYCLE = 1 << 4;
    public static final int ALL_SOURCE_BITS = (1 << 5) - 1;
    public static final String SOURCE_LAYOUT =
        "bit0=stats,bit1=status,bit2=inventory,bit3=projectiles,"
            + "bit4=actor_lifecycle";

    private static final Set<String> KINDS = Set.of(
        "stat_changed",
        "status_added",
        "status_removed",
        "status_refreshed",
        "inventory_transaction",
        "inventory_slot",
        "projectile_spawned",
        "projectile_despawned",
        "projectile_impacted",
        "actor_death",
        "actor_revived"
    );
    private static final Set<String> ORIGINS = Set.of(
        "server_boundary_state_delta",
        "server_inter_row_state_delta",
        "server_inventory_event",
        "server_death_component"
    );

    public NpcLifecycleEventSnapshot {
        kind = text(kind);
        origin = text(origin);
        subject = text(subject);
        key = text(key);
        textBefore = text(textBefore);
        textAfter = text(textAfter);
        if (!KINDS.contains(kind) || !ORIGINS.contains(origin)) {
            throw new IllegalArgumentException("unknown lifecycle event identity");
        }
        if (index < -1 || entityIndex < -1 || flags < 0) {
            throw new IllegalArgumentException("invalid lifecycle event integer");
        }
        if (
            !Double.isFinite(valueBefore)
                || !Double.isFinite(valueAfter)
                || !Double.isFinite(auxiliary0)
                || !Double.isFinite(auxiliary1)
        ) {
            throw new IllegalArgumentException("lifecycle event values must be finite");
        }
        position = finite(position, 3);
        if (!positionAvailable && any(position)) {
            throw new IllegalArgumentException(
                "unavailable lifecycle position must be zero"
            );
        }
    }

    @Override
    public double[] position() {
        return position.clone();
    }

    private static double[] finite(double[] value, int width) {
        if (value == null || value.length != width) {
            throw new IllegalArgumentException("lifecycle position has wrong width");
        }
        double[] copy = value.clone();
        for (double scalar : copy) {
            if (!Double.isFinite(scalar)) {
                throw new IllegalArgumentException(
                    "lifecycle position must be finite"
                );
            }
        }
        return copy;
    }

    private static boolean any(double[] values) {
        for (double value : values) if (value != 0.0) return true;
        return false;
    }

    private static String text(String value) {
        return value == null ? "" : value;
    }
}
