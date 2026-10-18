package com.hytalerlbridge.environment;

import com.hytalerlbridge.entity.PrivilegedEntityQuery;
import com.hytalerlbridge.entity.PrivilegedEntitySnapshot;
import com.hytalerlbridge.entity.PrivilegedNpcSnapshot;
import com.hytalerlbridge.entity.PrivilegedNpcUuidQuery;

/** Optional native-only, privileged entity enumeration surface. */
public interface NativePrivilegedEntitySource {

    PrivilegedEntitySnapshot capturePrivilegedEntitySnapshot(
        PrivilegedEntityQuery query
    );

    PrivilegedNpcSnapshot capturePrivilegedNpcSnapshot(
        PrivilegedEntityQuery query
    );

    PrivilegedNpcSnapshot capturePrivilegedNpcSnapshotByUuid(
        PrivilegedNpcUuidQuery query
    );
}
