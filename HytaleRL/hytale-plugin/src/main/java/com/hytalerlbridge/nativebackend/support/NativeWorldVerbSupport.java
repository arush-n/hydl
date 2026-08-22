package com.hytalerlbridge.nativebackend.support;

import com.hytalerlbridge.action.AgentAction;
import java.util.ArrayList;
import java.util.List;

/** Pure availability projection for legacy action diagnostics. */
public final class NativeWorldVerbSupport {

    private NativeWorldVerbSupport() {}

    public static int requiredHotbarCapacity(
        List<Integer> requestedSlots,
        int currentCapacity
    ) {
        int requiredCapacity = currentCapacity;
        for (int slot : requestedSlots) {
            requiredCapacity = Math.max(requiredCapacity, slot + 1);
        }
        return requiredCapacity;
    }

    public static List<String> requestedUnsupportedWorldVerbs(
        AgentAction action,
        boolean headlessWorldVerbsEnabled
    ) {
        List<String> requested = new ArrayList<>(4);
        // Legacy relative/integer requests are never executed by the native
        // path. Opt-in support applies only to the typed v2 envelope.
        if (action.hasPlacement()) requested.add("place_block");
        if (action.hasBreak()) requested.add("break_block");
        if (action.hasCraft()) requested.add("craft_recipe");
        if (
            action.nativeWorldVerbRequest().present()
                && !headlessWorldVerbsEnabled
                && !action.nativeWorldVerbRequest().verb().equals("use")
        ) {
            requested.add(action.nativeWorldVerbRequest().verb());
        }
        return List.copyOf(requested);
    }
}
