package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.protocol.io.ProtocolException;
import java.util.List;
import javax.annotation.Nonnull;

/**
 * The Server 0.5.x numbering of {@link InteractionType}, which the bridge's wire messages and
 * evidence use; the Python side fixes these 25 names at these indices.
 *
 * <p>0.7.2 inserted {@code Ability4} at value 5, shifting {@code Use} through {@code GameModeSwap}
 * up by one, and appended {@code OnBreak} and {@code OnBreakImpact}. Those three have no 0.5 index:
 * evidence leaves them out and no wire index decodes to them.
 */
public final class NativeInteractionTypes {

    /** The 0.7.2 constants at their 0.5.10 values 0..24. */
    public static final List<InteractionType> WIRE_TYPES = List.of(
        InteractionType.Primary,
        InteractionType.Secondary,
        InteractionType.Ability1,
        InteractionType.Ability2,
        InteractionType.Ability3,
        InteractionType.Use,
        InteractionType.Pick,
        InteractionType.Pickup,
        InteractionType.CollisionEnter,
        InteractionType.CollisionLeave,
        InteractionType.Collision,
        InteractionType.EntityStatEffect,
        InteractionType.SwapTo,
        InteractionType.SwapFrom,
        InteractionType.Death,
        InteractionType.Wielding,
        InteractionType.ProjectileSpawn,
        InteractionType.ProjectileHit,
        InteractionType.ProjectileMiss,
        InteractionType.ProjectileBounce,
        InteractionType.Held,
        InteractionType.HeldOffhand,
        InteractionType.Equipped,
        InteractionType.Dodge,
        InteractionType.GameModeSwap
    );

    private NativeInteractionTypes() {}

    /** The 0.5 index of {@code type} (its 0.5 {@code getValue()}), or -1 for a type 0.5 did not have. */
    public static int wireIndex(@Nonnull InteractionType type) {
        return WIRE_TYPES.indexOf(type);
    }

    /** 0.5 {@code InteractionType.fromValue}: the type at a 0.5 index; throws for any other index, as 0.5 did. */
    @Nonnull
    public static InteractionType fromWireIndex(int index) {
        if (index >= 0 && index < WIRE_TYPES.size()) {
            return WIRE_TYPES.get(index);
        }
        throw ProtocolException.invalidEnumValue("InteractionType", index);
    }
}
