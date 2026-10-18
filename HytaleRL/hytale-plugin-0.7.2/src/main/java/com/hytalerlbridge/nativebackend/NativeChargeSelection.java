package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.server.core.modules.interaction.IInteractionSimulationHandler;
import com.hypixel.hytale.server.npc.interactions.NPCInteractionSimulationHandler;

public final class NativeChargeSelection {

    private NativeChargeSelection() {}

    public static boolean apply(
        IInteractionSimulationHandler simulation,
        double requestedChargeTime
    ) {
        if (simulation instanceof NPCInteractionSimulationHandler npcSimulation) {
            npcSimulation.requestChargeTime((float) requestedChargeTime);
            return true;
        }
        return requestedChargeTime == 0.0;
    }
}
