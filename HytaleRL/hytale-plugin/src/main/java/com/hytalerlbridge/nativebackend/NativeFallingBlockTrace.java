package com.hytalerlbridge.nativebackend;

import java.util.Map;
import org.joml.Vector3d;

/** Latest fixture-only falling-block sample at a pinned native tick stage. */
final class NativeFallingBlockTrace {
    static final String SCHEMA = "hytalerl_native_falling_block_trace_v2";
    static final int VERSION = 2;
    static final String CONTRACT_SHA256 =
        "AB46D1D6D8974375199A749895E2C3E07A0C58949BEC85C26D25BB89AC713432";

    private final boolean fixtureAvailable;
    private final boolean sampleAvailable;
    private final String stage;
    private final String blockAssetId;
    private final int runtimeBlockId;
    private final String impactType;
    private final long sampleIndex;
    private final long worldTick;
    private final double deltaSeconds;
    private final Vector3d position;
    private final Vector3d velocity;
    private final boolean onGround;
    private final boolean entityPresent;
    private final boolean impactObserved;
    private final int impactX;
    private final int impactY;
    private final int impactZ;
    private final double mass;
    private final double dragCoefficient;
    private final boolean invertedGravity;
    private final double boxWidth;
    private final double boxDepth;

    private NativeFallingBlockTrace(
        boolean fixtureAvailable,
        boolean sampleAvailable,
        String stage,
        String blockAssetId,
        int runtimeBlockId,
        String impactType,
        long sampleIndex,
        long worldTick,
        double deltaSeconds,
        Vector3d position,
        Vector3d velocity,
        boolean onGround,
        boolean entityPresent,
        boolean impactObserved,
        int impactX,
        int impactY,
        int impactZ,
        double mass,
        double dragCoefficient,
        boolean invertedGravity,
        double boxWidth,
        double boxDepth
    ) {
        this.fixtureAvailable = fixtureAvailable;
        this.sampleAvailable = sampleAvailable;
        this.stage = stage;
        this.blockAssetId = blockAssetId;
        this.runtimeBlockId = runtimeBlockId;
        this.impactType = impactType;
        this.sampleIndex = sampleIndex;
        this.worldTick = worldTick;
        this.deltaSeconds = deltaSeconds;
        this.position = new Vector3d(position);
        this.velocity = new Vector3d(velocity);
        this.onGround = onGround;
        this.entityPresent = entityPresent;
        this.impactObserved = impactObserved;
        this.impactX = impactX;
        this.impactY = impactY;
        this.impactZ = impactZ;
        this.mass = mass;
        this.dragCoefficient = dragCoefficient;
        this.invertedGravity = invertedGravity;
        this.boxWidth = boxWidth;
        this.boxDepth = boxDepth;
    }

    static NativeFallingBlockTrace unavailable(boolean fixtureAvailable) {
        return new NativeFallingBlockTrace(
            fixtureAvailable,
            false,
            "unavailable",
            "",
            0,
            "",
            -1L,
            -1L,
            0.0,
            new Vector3d(),
            new Vector3d(),
            false,
            false,
            false,
            0,
            0,
            0,
            0.0,
            0.0,
            false,
            0.0,
            0.0
        );
    }

    static NativeFallingBlockTrace sample(
        String stage,
        String blockAssetId,
        int runtimeBlockId,
        String impactType,
        long sampleIndex,
        long worldTick,
        double deltaSeconds,
        Vector3d position,
        Vector3d velocity,
        boolean onGround,
        double mass,
        double dragCoefficient,
        boolean invertedGravity,
        double boxWidth,
        double boxDepth
    ) {
        return new NativeFallingBlockTrace(
            true,
            true,
            stage,
            blockAssetId,
            runtimeBlockId,
            impactType,
            sampleIndex,
            worldTick,
            deltaSeconds,
            position,
            velocity,
            onGround,
            true,
            false,
            0,
            0,
            0,
            mass,
            dragCoefficient,
            invertedGravity,
            boxWidth,
            boxDepth
        );
    }

    NativeFallingBlockTrace impact() {
        if (!sampleAvailable || !onGround) return this;
        return new NativeFallingBlockTrace(
            true,
            true,
            "ground_impact_after_falling_ticker",
            blockAssetId,
            runtimeBlockId,
            impactType,
            sampleIndex,
            worldTick,
            deltaSeconds,
            position,
            velocity,
            true,
            false,
            true,
            (int) Math.floor(position.x),
            (int) Math.floor(position.y),
            (int) Math.floor(position.z),
            mass,
            dragCoefficient,
            invertedGravity,
            boxWidth,
            boxDepth
        );
    }

    boolean onGround() {
        return onGround;
    }

    void putInto(Map<String, Object> info) {
        info.put("native_falling_block_trace_schema", SCHEMA);
        info.put("native_falling_block_trace_version", VERSION);
        info.put("native_falling_block_trace_contract_sha256", CONTRACT_SHA256);
        info.put("native_falling_block_fixture_available", fixtureAvailable);
        info.put("native_falling_block_sample_available", sampleAvailable);
        info.put("native_falling_block_stage", stage);
        info.put("native_falling_block_asset_id", blockAssetId);
        info.put("native_falling_block_runtime_id", runtimeBlockId);
        info.put("native_falling_block_impact_type", impactType);
        info.put("native_falling_block_sample_index", sampleIndex);
        info.put("native_falling_block_world_tick", worldTick);
        info.put("native_falling_block_delta_seconds", deltaSeconds);
        info.put("native_falling_block_position_x", position.x);
        info.put("native_falling_block_position_y", position.y);
        info.put("native_falling_block_position_z", position.z);
        info.put("native_falling_block_velocity_x", velocity.x);
        info.put("native_falling_block_velocity_y", velocity.y);
        info.put("native_falling_block_velocity_z", velocity.z);
        info.put("native_falling_block_on_ground", onGround);
        info.put("native_falling_block_entity_present", entityPresent);
        info.put("native_falling_block_impact_observed", impactObserved);
        info.put("native_falling_block_impact_x", impactX);
        info.put("native_falling_block_impact_y", impactY);
        info.put("native_falling_block_impact_z", impactZ);
        info.put("native_falling_block_mass", mass);
        info.put("native_falling_block_drag_coefficient", dragCoefficient);
        info.put("native_falling_block_inverted_gravity", invertedGravity);
        info.put("native_falling_block_box_width", boxWidth);
        info.put("native_falling_block_box_depth", boxDepth);
        info.put("native_falling_block_dry_static_fixture", fixtureAvailable);
        info.put("native_falling_block_public_cascade_certified", false);
    }
}
