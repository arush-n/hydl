package com.hytalerlbridge.action;

import java.util.Map;
import org.msgpack.value.Value;

/**
 * Represents an action taken by the RL agent in a single step.
 * Supports movement, combat, building, crafting, and world interaction.
 */
public record AgentAction(
    boolean forward,
    boolean back,
    boolean left,
    boolean right,
    boolean jump,
    boolean attack,
    boolean use,
    double cameraDeltaYaw,
    double cameraDeltaPitch,
    int hotbarSlot,
    // --- Extended NPC actions ---
    int placeBlockX,    // relative X offset for targeted block placement (-3..3)
    int placeBlockY,    // relative Y offset (-3..3)
    int placeBlockZ,    // relative Z offset (-3..3)
    int placeBlockType, // block type ID to place (0 = no placement)
    int breakBlockX,    // relative X offset for targeted block break (-3..3)
    int breakBlockY,    // relative Y offset (-3..3)
    int breakBlockZ,    // relative Z offset (-3..3)
    boolean breakBlock, // whether to break the targeted block
    int craftRecipeId,  // recipe ID to craft (-1 = no crafting)
    double requestedChargeTime, // NPC interaction charge selection, seconds
    boolean guardHeld,  // continuation-held level
    int dodgeDirection, // edge: 0=none, 1=forward, 2=back, 3=left, 4=right
    int abilitySlot,    // edge: -1=none, otherwise reset-pinned local slot
    int worldMoveDirection, // level: 0=none, 1..8=N,NE,E,SE,S,SW,W,NW
    NativeWorldVerbRequest nativeWorldVerbRequest,
    // Body steer, degrees, independent of the camera. MotionControllerBase
    // .calculateYaw takes a body steer and a head steer and only bounds the head
    // into the model's authored window around the body afterwards; this session
    // used to hand one yaw to both, so an actor could not walk one way and look
    // another. Zero means "no body steer this tick".
    double bodyDeltaYaw
) {
    /** Backward-compatible constructor for callers predating typed World verbs. */
    public AgentAction(
        boolean forward,
        boolean back,
        boolean left,
        boolean right,
        boolean jump,
        boolean attack,
        boolean use,
        double cameraDeltaYaw,
        double cameraDeltaPitch,
        int hotbarSlot,
        int placeBlockX,
        int placeBlockY,
        int placeBlockZ,
        int placeBlockType,
        int breakBlockX,
        int breakBlockY,
        int breakBlockZ,
        boolean breakBlock,
        int craftRecipeId,
        double requestedChargeTime,
        boolean guardHeld,
        int dodgeDirection,
        int abilitySlot,
        int worldMoveDirection
    ) {
        this(
            forward,
            back,
            left,
            right,
            jump,
            attack,
            use,
            cameraDeltaYaw,
            cameraDeltaPitch,
            hotbarSlot,
            placeBlockX,
            placeBlockY,
            placeBlockZ,
            placeBlockType,
            breakBlockX,
            breakBlockY,
            breakBlockZ,
            breakBlock,
            craftRecipeId,
            requestedChargeTime,
            guardHeld,
            dodgeDirection,
            abilitySlot,
            worldMoveDirection,
            NativeWorldVerbRequest.none(),
            0.0
        );
    }

    /** Backward-compatible constructor for callers predating policy verbs. */
    public AgentAction(
        boolean forward,
        boolean back,
        boolean left,
        boolean right,
        boolean jump,
        boolean attack,
        boolean use,
        double cameraDeltaYaw,
        double cameraDeltaPitch,
        int hotbarSlot,
        int placeBlockX,
        int placeBlockY,
        int placeBlockZ,
        int placeBlockType,
        int breakBlockX,
        int breakBlockY,
        int breakBlockZ,
        boolean breakBlock,
        int craftRecipeId,
        double requestedChargeTime
    ) {
        this(
            forward,
            back,
            left,
            right,
            jump,
            attack,
            use,
            cameraDeltaYaw,
            cameraDeltaPitch,
            hotbarSlot,
            placeBlockX,
            placeBlockY,
            placeBlockZ,
            placeBlockType,
            breakBlockX,
            breakBlockY,
            breakBlockZ,
            breakBlock,
            craftRecipeId,
            requestedChargeTime,
            false,
            0,
            -1,
            0
        );
    }

    /** Backward-compatible constructor for callers predating charge selection. */
    public AgentAction(
        boolean forward,
        boolean back,
        boolean left,
        boolean right,
        boolean jump,
        boolean attack,
        boolean use,
        double cameraDeltaYaw,
        double cameraDeltaPitch,
        int hotbarSlot,
        int placeBlockX,
        int placeBlockY,
        int placeBlockZ,
        int placeBlockType,
        int breakBlockX,
        int breakBlockY,
        int breakBlockZ,
        boolean breakBlock,
        int craftRecipeId
    ) {
        this(
            forward,
            back,
            left,
            right,
            jump,
            attack,
            use,
            cameraDeltaYaw,
            cameraDeltaPitch,
            hotbarSlot,
            placeBlockX,
            placeBlockY,
            placeBlockZ,
            placeBlockType,
            breakBlockX,
            breakBlockY,
            breakBlockZ,
            breakBlock,
            craftRecipeId,
            0.0
        );
    }

    public AgentAction {
        if (
            !Double.isFinite(requestedChargeTime)
                || requestedChargeTime < 0.0
                || requestedChargeTime > Float.MAX_VALUE
        ) {
            throw new IllegalArgumentException(
                "requestedChargeTime must be a finite nonnegative float"
            );
        }
        if (!Double.isFinite(bodyDeltaYaw)) {
            throw new IllegalArgumentException("bodyDeltaYaw must be finite");
        }
        if (dodgeDirection < 0 || dodgeDirection > 4) {
            throw new IllegalArgumentException("dodgeDirection must be in [0, 4]");
        }
        if (abilitySlot < -1 || abilitySlot > 15) {
            throw new IllegalArgumentException("abilitySlot must be in [-1, 15]");
        }
        if (worldMoveDirection < 0 || worldMoveDirection > 8) {
            throw new IllegalArgumentException(
                "worldMoveDirection must be in [0, 8]"
            );
        }
        if (nativeWorldVerbRequest == null) {
            nativeWorldVerbRequest = NativeWorldVerbRequest.none();
        }
        if (
            nativeWorldVerbRequest.present()
                && (use || placeBlockType > 0 || breakBlock || craftRecipeId >= 0)
        ) {
            throw new IllegalArgumentException(
                "Typed and legacy World-verb requests cannot be combined"
            );
        }
        int standardInputRoots = (attack ? 1 : 0)
            + (use ? 1 : 0)
            + (placeBlockType > 0 ? 1 : 0)
            + (breakBlock ? 1 : 0)
            + (craftRecipeId >= 0 ? 1 : 0)
            + (guardHeld ? 1 : 0)
            + (abilitySlot >= 0 ? 1 : 0)
            + (nativeWorldVerbRequest.present() ? 1 : 0);
        if (standardInputRoots > 1) {
            throw new IllegalArgumentException(
                "At most one standard-input interaction root may start per action"
            );
        }
    }

    public static AgentAction noop() {
        return new AgentAction(
            false, false, false, false, false, false, false,
            0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, false,
            -1
        );
    }

    /**
     * Action state for later engine ticks in the same RL step. Movement remains
     * held, while edge-triggered inputs fire only on the first tick. This
     * matches the native backend and prevents ticks_per_step from multiplying
     * camera turns, jumps, attacks, dodges, abilities, block edits, or
     * crafting requests. Guard is a held level and therefore persists.
     */
    public AgentAction continuation() {
        return new AgentAction(
            forward, back, left, right,
            false, false, false,
            0.0, 0.0, hotbarSlot,
            0, 0, 0, 0,
            0, 0, 0, false,
            -1, 0.0,
            guardHeld, 0, -1,
            worldMoveDirection,
            NativeWorldVerbRequest.none(),
            0.0
        );
    }

    public static AgentAction fromMap(Map<Value, Value> map) {
        return new AgentAction(
            getBool(map, "forward"),
            getBool(map, "back"),
            getBool(map, "left"),
            getBool(map, "right"),
            getBool(map, "jump"),
            getBool(map, "attack"),
            getBool(map, "use"),
            getDouble(map, "camera_delta_yaw"),
            getDouble(map, "camera_delta_pitch"),
            // -1, not 0: a missing key must mean "no switch". The default was 0,
            // so any sender that omitted the field re-selected slot 0 on every
            // tick -- invisible only because slot 0 is the loadout slot.
            getInt(map, "hotbar_slot", -1),
            getInt(map, "place_block_x"),
            getInt(map, "place_block_y"),
            getInt(map, "place_block_z"),
            getInt(map, "place_block_type"),
            getInt(map, "break_block_x"),
            getInt(map, "break_block_y"),
            getInt(map, "break_block_z"),
            getBool(map, "break_block"),
            getInt(map, "craft_recipe_id", -1),
            getDouble(map, "requested_charge_time"),
            getBool(map, "guard_held"),
            getInt(map, "dodge_direction"),
            getInt(map, "ability_slot", -1),
            getInt(map, "world_move_direction"),
            NativeWorldVerbRequest.fromActionMap(map),
            // getDouble returns 0.0 for a missing key, so a sender predating the
            // body/head split still means "no body steer" and behaves as before.
            getDouble(map, "body_delta_yaw")
        );
    }

    /** Whether this action includes a targeted block placement. */
    public boolean hasPlacement() {
        return placeBlockType > 0;
    }

    /** Whether this action includes a targeted block break. */
    public boolean hasBreak() {
        return breakBlock;
    }

    /** Whether this action includes a crafting request. */
    public boolean hasCraft() {
        return craftRecipeId >= 0;
    }

    /** Whether this action requests a first-control-tick dodge edge. */
    public boolean hasDodge() {
        return dodgeDirection > 0;
    }

    /** Whether this action requests a reset-pinned profile-local ability. */
    public boolean hasAbility() {
        return abilitySlot >= 0;
    }

    private static boolean getBool(Map<Value, Value> map, String key) {
        Value value = getValue(map, key);
        if (value == null) return false;
        if (value.isBooleanValue()) return value.asBooleanValue().getBoolean();
        if (value.isIntegerValue()) return value.asIntegerValue().asLong() != 0;
        return false;
    }

    private static double getDouble(Map<Value, Value> map, String key) {
        Value value = getValue(map, key);
        if (value == null) return 0.0;
        if (value.isFloatValue()) return value.asFloatValue().toDouble();
        if (value.isIntegerValue()) return value.asIntegerValue().asLong();
        return 0.0;
    }

    private static int getInt(Map<Value, Value> map, String key) {
        return getInt(map, key, 0);
    }

    private static int getInt(Map<Value, Value> map, String key, int defaultValue) {
        Value value = getValue(map, key);
        if (value == null || !value.isIntegerValue()) return defaultValue;
        long decoded = value.asIntegerValue().asLong();
        if (decoded < Integer.MIN_VALUE || decoded > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Action field out of integer range: " + key);
        }
        return (int) decoded;
    }

    private static Value getValue(Map<Value, Value> map, String key) {
        for (Map.Entry<Value, Value> entry : map.entrySet()) {
            Value candidate = entry.getKey();
            if (candidate.isStringValue()
                && candidate.asStringValue().asString().equals(key)) {
                return entry.getValue();
            }
        }
        return null;
    }
}
