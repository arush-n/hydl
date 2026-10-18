package com.hytalerlbridge.observation;

import com.hypixel.hytale.protocol.MovementStates;
import java.util.Map;

/** Versioned, availability-masked transport for native entity movement state. */
public record MovementStateFrame(boolean available, int bits) {
    public static final String SCHEMA = "hytalerl_native_movement_states_v1";
    public static final int VERSION = 1;
    public static final int STATE_COUNT = 23;
    public static final String ORDER =
        "idle,horizontal_idle,jumping,flying,walking,running,sprinting,crouching,"
            + "forced_crouching,falling,falling_far,climbing,in_fluid,swimming,"
            + "swim_jumping,on_ground,mantling,sliding,mounting,rolling,sitting,"
            + "gliding,sleeping";

    public static MovementStateFrame unavailable() {
        return new MovementStateFrame(false, 0);
    }

    public static MovementStateFrame from(MovementStates states) {
        if (states == null) return unavailable();
        int bits = 0;
        bits = set(bits, 0, states.idle);
        bits = set(bits, 1, states.horizontalIdle);
        bits = set(bits, 2, states.jumping);
        bits = set(bits, 3, states.flying);
        bits = set(bits, 4, states.walking);
        bits = set(bits, 5, states.running);
        bits = set(bits, 6, states.sprinting);
        bits = set(bits, 7, states.crouching);
        bits = set(bits, 8, states.forcedCrouching);
        bits = set(bits, 9, states.falling);
        bits = set(bits, 10, states.fallingFar);
        bits = set(bits, 11, states.climbing);
        bits = set(bits, 12, states.inFluid);
        bits = set(bits, 13, states.swimming);
        bits = set(bits, 14, states.swimJumping);
        bits = set(bits, 15, states.onGround);
        bits = set(bits, 16, states.mantling);
        bits = set(bits, 17, states.sliding);
        bits = set(bits, 18, states.mounting);
        bits = set(bits, 19, states.rolling);
        bits = set(bits, 20, states.sitting);
        bits = set(bits, 21, states.gliding);
        bits = set(bits, 22, states.sleeping);
        return new MovementStateFrame(true, bits);
    }

    public boolean state(int index) {
        if (index < 0 || index >= STATE_COUNT) {
            throw new IndexOutOfBoundsException("movement-state index is out of range");
        }
        return available && (bits & (1 << index)) != 0;
    }

    public void putInto(Map<String, Object> info, String subject) {
        info.put(subject + "_movement_states_available", available);
        info.put(subject + "_movement_states_bits", bits);
    }

    private static int set(int bits, int index, boolean enabled) {
        return enabled ? bits | (1 << index) : bits;
    }
}
