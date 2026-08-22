package com.hytalerlbridge.nativebackend.session.info;

import java.util.Map;
import com.hytalerlbridge.geometry.GeometryContract;
import com.hytalerlbridge.geometry.GeometryFrame;
import com.hytalerlbridge.nativebackend.model.GeometryChunkCoverage;
import java.util.Set;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.implementationVersion;

/**
 * Geometry frame counts, chunk pinning and shape-id scope.
 *
 * <p>One section of the info map built each step by
 * {@code NativeEnvironmentSession.buildInfo}. See the package README for
 * why these sections are split out and what the ordering contract is.
 */
public final class GeometryInfo {

    private static final String NATIVE_SERVER_VERSION = implementationVersion();

    private GeometryInfo() {}

    public static void contribute(
        Map<String, Object> info,
        GeometryFrame geometry,
        Set<Long> geometryChunkIndices,
        Set<Long> loadedChunkIndices
    ) {
        info.put("geometry_schema", GeometryContract.SCHEMA);
        info.put("geometry_available", geometry.available());
        info.put("geometry_exact_collision_shapes", geometry.exactCollisionShapes());
        info.put("geometry_cell_count", geometry.cells().size());
        info.put("geometry_contact_count", geometry.contacts().size());
        info.put("geometry_contact_capacity", GeometryContract.MAX_CONTACTS);
        info.put(
            "geometry_chunk_streaming",
            "preload_and_pin_one_chunk_halo_before_each_chunk_transition"
        );
        info.put(
            "geometry_pinned_chunk_count",
            geometryChunkIndices.size()
        );
        info.put(
            "geometry_pinned_chunk_capacity",
            GeometryChunkCoverage.MAX_PINNED_CHUNKS
        );
        info.put("native_pinned_chunk_count", loadedChunkIndices.size());
        info.put(
            "geometry_los_endpoints",
            "native_model_eye_height;target_fallback=bounding_box_center"
        );
        info.put(
            "geometry_shape_id_scope",
            "runtime_"
                + NATIVE_SERVER_VERSION
                + "_hitbox_index_and_rotation;boxes_are_portable_source_of_truth"
        );
    }
}
