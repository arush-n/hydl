package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.protocol.io.ProtocolException;
import java.util.List;
import javax.annotation.Nonnull;

/**
 * The wire numbering of {@link InteractionType}, which the bridge's wire messages and evidence
 * use: the Server 0.5.x indices 0 to 24, then the three types 0.7.2 added, appended at 25 to 27.
 * The Python side fixes these names at these indices (25 on 0.5, 28 on 0.7.2).
 *
 * <p>0.7.2 inserted {@code Ability4} at value 5 and appended {@code OnBreak} and
 * {@code OnBreakImpact}. The bridge does not follow that server order: inserting {@code Ability4}
 * would move {@code Use} through {@code GameModeSwap} on the wire, so the three are appended.
 */
public final class NativeInteractionTypes {

    /** The 0.5.10 values 0..24, then the three 0.7.2-only types at 25 to 27. */
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
        InteractionType.GameModeSwap,
        InteractionType.Ability4,
        InteractionType.OnBreak,
        InteractionType.OnBreakImpact
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
