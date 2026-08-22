package com.hytalerlbridge.nativebackend.fixture;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.support.StateSupport;
import java.util.Map;

/** Extracted verbatim from {@code NativeEnvironmentSession}. */
/**
 * Native-only measurement fixture for the legacy Brawler's target memory.
 *
 * <p>The target begins ten blocks away with its authored AI untouched. Once
 * it marks the controlled agent, this fixture erects a sealed stone
 * occluder between them. The enclosure is removed when the target returns
 * home/idle or after a bounded censoring interval. Per-tick work is one
 * bounded marked-slot scan and one native LOS query; block writes happen
 * only at the two phase transitions.</p>
 */
public final class TargetMemoryEvaluationFixture {

    private static final int VERSION = 1;
    private static final int HALF_WIDTH = 16;
    private static final int DEPTH = 12;
    private static final int WALL_HEIGHT = 4;
    private static final int MAX_OCCLUDED_TICKS = 450;

    private final int minimumX;
    private final int maximumX;
    private final int minimumY;
    private final int frontZ;
    private final int backZ;

    private String phase = "acquiring";
    private String targetState = "unavailable";
    private int targetStateIndex = -1;
    private int targetSubStateIndex = -1;
    private boolean targetMarksAgent;
    private boolean targetLineOfSight;
    private boolean targetLineOfSightValid;
    private boolean enclosurePresent;
    private boolean occlusionCensored;
    private boolean wasSearching;
    private int acquisitionTick = -1;
    private int enclosureTick = -1;
    private int lastVisibleTick = -1;
    private int firstOccludedTick = -1;
    private int searchStartTick = -1;
    private int searchEndTick = -1;
    private int reopenTick = -1;
    private int visibilityRestoredTick = -1;

    public TargetMemoryEvaluationFixture(
        int centerX,
        int floorY,
        int frontZ
    ) {
        this.minimumX = centerX - HALF_WIDTH;
        this.maximumX = centerX + HALF_WIDTH;
        this.minimumY = floorY;
        this.frontZ = frontZ;
        this.backZ = frontZ - DEPTH;
    }

    public void afterTick(
        World world,
        Store<EntityStore> store,
        Ref<EntityStore> targetReference,
        Ref<EntityStore> agentReference,
        int tick
    ) {
        if (targetReference == null
            || !targetReference.isValid()
            || agentReference == null
            || !agentReference.isValid()) return;
        NPCEntity target = store.getComponent(
            targetReference,
            NPCEntity.getComponentType()
        );
        Role role = target == null ? null : target.getRole();
        if (role == null) return;

        StateSupport state = role.getStateSupport();
        if (state != null) {
            String name = state.getStateName();
            targetState = name == null ? "unavailable" : name;
            targetStateIndex = state.getStateIndex();
            targetSubStateIndex = state.getSubStateIndex();
        }
        targetMarksAgent = marks(role, agentReference);
        targetLineOfSightValid = role.getPositionCache() != null;
        targetLineOfSight = targetLineOfSightValid
            && role.getPositionCache().hasLineOfSight(
                targetReference,
                agentReference,
                store
            );
        if (targetLineOfSight) {
            lastVisibleTick = tick;
        } else if (enclosurePresent && firstOccludedTick < 0) {
            firstOccludedTick = tick;
        }

        boolean searching = targetState.equals("Search")
            || targetState.startsWith("Search.");
        if (searching && searchStartTick < 0) {
            searchStartTick = tick;
        } else if (!searching && wasSearching && searchEndTick < 0) {
            searchEndTick = tick;
        }
        wasSearching = searching;

        if (phase.equals("acquiring") && targetMarksAgent) {
            acquisitionTick = tick;
            setEnclosure(world, true);
            enclosurePresent = true;
            enclosureTick = tick;
            phase = "occluded";
            return;
        }
        if (phase.equals("occluded")) {
            boolean finishedSearch = firstOccludedTick >= 0
                && (targetState.equals("Idle")
                    || targetState.startsWith("ReturnHome"));
            boolean timedOut =
                tick - enclosureTick >= MAX_OCCLUDED_TICKS;
            if (finishedSearch || timedOut) {
                occlusionCensored = timedOut && !finishedSearch;
                setEnclosure(world, false);
                enclosurePresent = false;
                reopenTick = tick;
                phase = "reopened";
            }
            return;
        }
        if (phase.equals("reopened") && targetLineOfSight) {
            visibilityRestoredTick = tick;
            phase = "complete";
        }
    }

    public void putInto(Map<String, Object> info) {
        info.put("target_memory_fixture_version", VERSION);
        info.put("target_memory_phase", phase);
        info.put("target_ai_state", targetState);
        info.put("target_ai_state_index", targetStateIndex);
        info.put("target_ai_substate_index", targetSubStateIndex);
        info.put("target_marks_agent", targetMarksAgent);
        info.put("target_line_of_sight", targetLineOfSight);
        info.put(
            "target_line_of_sight_valid",
            targetLineOfSightValid
        );
        info.put(
            "target_memory_enclosure_present",
            enclosurePresent
        );
        info.put(
            "target_memory_occlusion_censored",
            occlusionCensored
        );
        info.put("target_memory_acquisition_tick", acquisitionTick);
        info.put("target_memory_enclosure_tick", enclosureTick);
        info.put("target_memory_last_visible_tick", lastVisibleTick);
        info.put(
            "target_memory_first_occluded_tick",
            firstOccludedTick
        );
        info.put("target_memory_search_start_tick", searchStartTick);
        info.put("target_memory_search_end_tick", searchEndTick);
        info.put("target_memory_reopen_tick", reopenTick);
        info.put(
            "target_memory_visibility_restored_tick",
            visibilityRestoredTick
        );
        info.put("target_memory_enclosure_min_x", minimumX);
        info.put("target_memory_enclosure_max_x", maximumX);
        info.put("target_memory_enclosure_front_z", frontZ);
        info.put("target_memory_enclosure_back_z", backZ);
        info.put(
            "target_memory_max_occluded_ticks",
            MAX_OCCLUDED_TICKS
        );
    }

    public void setEnclosure(World world, boolean present) {
        String block = present ? "Rock_Stone" : BlockType.EMPTY_KEY;
        for (int x = minimumX; x <= maximumX; x++) {
            for (int y = minimumY; y < minimumY + WALL_HEIGHT; y++) {
                world.setBlock(x, y, frontZ, block);
                world.setBlock(x, y, backZ, block);
            }
        }
        for (int z = backZ + 1; z < frontZ; z++) {
            for (int y = minimumY; y < minimumY + WALL_HEIGHT; y++) {
                world.setBlock(minimumX, y, z, block);
                world.setBlock(maximumX, y, z, block);
            }
        }
    }

    public static boolean marks(
        Role role,
        Ref<EntityStore> agentReference
    ) {
        var marked = role.getMarkedEntitySupport();
        if (marked == null || marked.getEntityTargets() == null) {
            return false;
        }
        for (Ref<EntityStore> reference : marked.getEntityTargets()) {
            if (agentReference.equals(reference)) return true;
        }
        return false;
    }
}
