package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.WorldgenStructureMarkerQuery;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerSnapshot;

/** Optional native-only capture of authored WorldGen V2 structure markers. */
public interface NativeWorldgenStructureMarkerSource {

    WorldgenStructureMarkerSnapshot captureWorldgenStructureMarkers(
        WorldgenStructureMarkerQuery query
    );
}
