package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeMutableBlockEvidence;
import com.hytalerlbridge.worldgen.NativeMutableBlockCells;

/** Native-only controlled mutable-block evidence capability. */
public interface NativeMutableBlockSource {

    NativeMutableBlockEvidence captureMutableBlockEvidence();

    NativeMutableBlockCells captureMutableBlockCells(int[] positions);

    default NativeMutableBlockCells captureMutableBlockCells(
        int[] positions,
        String expectedWorldEpoch
    ) {
        if (expectedWorldEpoch != null) {
            throw new IllegalStateException(
                "Epoch-bound mutable block capture is unavailable"
            );
        }
        return captureMutableBlockCells(positions);
    }
}
