package com.hytalerlbridge.policy;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Replays one observation captured from a live server, for every NPC.
 *
 * <p><b>This is a harness, not perception.</b> It exists to answer one question
 * -- can a trained network run inside the server, consume an observation in the
 * trained layout, and drive an NPC -- without waiting on the shared projector
 * port that real {@link PolicyPerception} needs.
 *
 * <p>What it does prove: the mod loads, the systems register in the right order,
 * the network evaluates inside the server tick, the decoder produces a legal
 * action, and the sink moves the NPC. What it cannot prove: that a real
 * observation would be assembled correctly, or that the NPC responds to its
 * surroundings. The observation is frozen, so the policy sees the same world
 * every tick and its behaviour is necessarily open-loop.
 *
 * <p>The recurrent carry still advances, so the action may change from tick to
 * tick even though the input does not. That is the GRU's memory, not perception.
 */
public final class CapturedObservationPerception implements PolicyPerception {

    private final float[] observation;
    private final boolean[] legal;

    private CapturedObservationPerception(float[] observation, boolean[] legal) {
        this.observation = observation;
        this.legal = legal;
    }

    /**
     * Load {@code observation.bin} and {@code action_mask.bin} as written by
     * {@code export_case.py} from a live bridge capture.
     */
    public static CapturedObservationPerception load(Path directory)
        throws IOException {
        return new CapturedObservationPerception(
            Policy.read(directory, "observation"),
            Policy.readMask(directory, "action_mask")
        );
    }

    /** Columns that are non-zero, so a silently empty capture is visible. */
    public int nonZeroColumns() {
        int count = 0;
        for (float value : observation) {
            if (value != 0.0f) {
                count++;
            }
        }
        return count;
    }

    public int legalActions() {
        int count = 0;
        for (boolean bit : legal) {
            if (bit) {
                count++;
            }
        }
        return count;
    }

    @Override
    public Sample sample(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        int slot,
        float deltaTime
    ) {
        return new Sample(observation, legal);
    }
}
