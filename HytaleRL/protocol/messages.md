# HytaleRL Protocol Specification

## Transport

- TCP socket connection on configurable port (default: 5556)
- MessagePack binary serialization
- Request-response pattern: Python sends a request, Java responds
- Every MessagePack payload is prefixed with a 4-byte unsigned big-endian length
- Maximum inbound or outbound payload size: 16 MiB

## Message Format

All messages are MessagePack maps with a `type` field.

### Client → Server (Python → Java)

#### ResetRequest
```json
{
  "type": "reset",
  "task_id": "kill_trork",
  "seed": 42,
  "options": {
    "curriculum_phase": -1,
    "backend": "native",
    "world": "hytale",
    "npc_role": "Kweebec_Razorleaf",
    "combat_target_active": false,
    "fidelity_fixture": "default",
    "native_combat_item_id": "Weapon_Sword_Iron",
    "native_combat_interaction_id": "Weapon_Sword_Primary_Swing_Left",
    "native_combat_interaction_type": "Primary"
  }
}
```

For `base_builder`, `curriculum_phase` may be 0–3. The value `-1` selects the
full task and is the default when the option is absent.

`backend` defaults to `simulator`. `native` and its `headless` alias require the
bridge to be running as a Hytale Server `0.5.7` plugin; they are unavailable in
the standalone JAR. Native mode accepts `flat` or real 0.5.7 `hytale` world
generation and supports `native_fidelity`, `navigate`, `survive`, and
`kill_trork`. `spawn_x`, `spawn_y`, and `spawn_z` are optional but must be
supplied together. `npc_role` names a loaded, spawnable Hytale NPC role asset.
For `kill_trork`, `combat_target_active` controls whether the native
`Trork_Brawler` is forced into `Chase.Attack` or held passive. The option also
configures the calibrated simulator task. The evaluation-only
`fidelity_fixture="target_memory"` requires
`combat_target_active=false` and selects a third, natural-AI mode: the bridge
neither forces `Chase.Attack` nor clears the Brawler's authored decisions.

`fidelity_fixture` defaults to `default`. Controlled native differential
fixtures are `half_block`, `ceiling`, and `ledge` for `native_fidelity`, plus
`los_wall`, `los_diagonal_graze`, and `los_diagonal_block` for passive
`kill_trork`. `target_memory` is a native/headless `kill_trork` measurement
fixture: it starts the Brawler ten blocks away, waits for authored lock-on,
erects a sealed occluder, and reopens it after the search exits or a bounded
15-second censoring interval. The diagonal pair distinguishes a cell touched
only at a ray corner from the tied-axis destination cell. Those controlled
fixtures require `backend=native`/`headless` and `world=flat`. `static_region`
instead requires native `world=hytale`, disables block ticking before world
creation, and is the only fixture allowed to publish a Region v1 manifest.
Invalid combinations are rejected rather than approximated.

The three `native_combat_*` fields are an optional, native/headless-only
evidence binding and must be supplied together. Interaction type must be one
of `Primary`, `Secondary`, `Ability1`, `Ability2`, or `Ability3`. Reset
validates the shipped Item and internal Interaction assets, equips the Item in
hotbar slot zero, and pins that one internal program. A later `attack` request
queues the program through Hytale's real `InteractionManager`; invalid or
partial bindings fail reset. This bounded probe does not claim the item's
outer root selection, combos/cooldowns, inventory mutation, or projectile
lifecycle.

#### StepRequest
```json
{
  "type": "step",
  "action": {
    "forward": 1,
    "back": 0,
    "left": 0,
    "right": 0,
    "jump": 0,
    "attack": 0,
    "use": 0,
    "camera_delta_yaw": 0.0,
    "camera_delta_pitch": 0.0,
    "hotbar_slot": 0,
    "place_block_x": 0,
    "place_block_y": 0,
    "place_block_z": 1,
    "place_block_type": 9,
    "break_block_x": 0,
    "break_block_y": 1,
    "break_block_z": 0,
    "break_block": 0,
    "craft_recipe_id": -1
  }
}
```

#### CloseRequest
```json
{
  "type": "close"
}
```

#### ConfigRequest
```json
{
  "type": "config",
  "tick_rate": 4,
  "max_episode_steps": 6000
}
```

The server replies with an `ack` before closing the connection.

Configuration is scoped to the current TCP client and is acknowledged with
`{"type": "ack"}`. `tick_rate` must be 1–1000 and `max_episode_steps` may be
zero for no server-side limit.

#### RegionManifestRequest

```json
{"type": "region_manifest"}
```

This read-only request requires an active native Hytale reset with
`fidelity_fixture="static_region"`. Dynamic sessions are rejected. With no
coordinates, it loads and pins the complete 5x5 chunk capture around the
spawn-selected 3x3 core. A client may instead select another core in the same
active world:

```json
{
  "type": "region_manifest",
  "core_min_chunk_x": 3,
  "core_min_chunk_z": -4
}
```

The two signed int32 core coordinates are optional as a pair and rejected when
only one is present. The selected manifest remains active for subsequent
`region_section` requests. Selecting another core pins its missing chunks in
the same native world; captures do not silently create a second same-seed
world.

#### RegionSectionRequest

```json
{
  "type": "region_section",
  "chunk_x": 12,
  "chunk_z": -7,
  "section_y": 2
}
```

The requested section must be one of the manifest's 250 fixed capture
sections. Requests outside the halo or Hytale's ten vertical sections fail.

#### RegionLightSectionRequest

```json
{
  "type": "region_light_section",
  "chunk_x": 12,
  "chunk_z": -7,
  "section_y": 2
}
```

The bounds are identical to `region_section`. An unready request queues that
exact native section for lighting and returns unavailable for the current
response; it never returns initialization zeros as darkness.

#### TraversalProbeRequest

```text
type = "traversal_probe"
positions_f64_le_xyz = binary[N * 3 * 8]
upward_limits_f64_le = binary[N * 8]
```

This evidence-only request requires an active native `static_region` session
and a selected Region manifest. `1 <= N <= 4096`; positions are actor
transform coordinates and each upward limit is finite and non-negative. The
bridge rejects unequal row counts before entering the native world thread.

#### PrivilegedEntitySnapshotRequest

```text
type = "privileged_entity_snapshot"
component_filter = "uuid_transform_velocity_v1"
bounds_f64_le_min_max_xyz = binary[6 * 8]
capacity = 1..256
```

This is a physically separate native-only diagnostic request. It requires an
active native/headless environment and is unavailable from the simulator.
Bounds are finite world coordinates ordered
`[min_x,min_y,min_z,max_x,max_y,max_z]`, with each minimum inclusive and each
maximum exclusive. The only v1 filter requires initialized
`UUIDComponent + TransformComponent + Velocity` rows. Arbitrary Java class
names are rejected.

The request does not alter reset/step observations and is never evaluated on
the normal actor path.

#### PrivilegedNpcSnapshotByUuidRequest

```text
type = "privileged_npc_snapshot_by_uuid"
component_filter = "npc_uuid_transform_velocity_role_geometry_v2"
npc_uuid_bytes = binary[16]
bounds_f64_le_min_max_xyz = binary[6 * 8]
```

This native-only detail request resolves the RFC-4122 UUID through Hytale's
entity index instead of enumerating an AABB. The current entity must still be
an NPC with initialized transform and velocity components, and its position
must remain inside the supplied half-open AABB. The server fixes response
capacity to one; a missing, despawned, non-NPC, or out-of-bounds UUID returns a
complete empty snapshot.

### Server → Client (Java → Python)

#### RegionManifest

`hytalerl_native_region_snapshot_v1` records the core/capture coordinates,
seed/provider identity, `offline_complete`/static/exact capability, and the
runtime-reflected `hytalerl_chunk_api_v1` map. The chunk map fixes the local
0.5.7 dimensions (`32^3`, ten sections, Y `[0,320)`) and the `y_z_x` index
order. Any mismatch is rejected before Python allocates a region artifact.
`static` means that the emitted artifact is a frozen snapshot. It does not
promise that mutable native source fields such as fluid fronts are determined
by the seed alone or remain unchanged after native block ticks; those require
a future delta protocol.

The offline Region client requests all 250 sections twice, first forward and
then in reverse order. It compares complete expanded semantics, not section
palette codes, and accepts only the later pass when all 8,192,000 cells match.
This is a host-side certification gate and does not change the wire schema.
Cross-reset same-seed reproducibility is a separate exact-semantic gate.

#### RegionSection

Each `hytalerl_native_region_section_v2` response contains:

```text
chunk_x, chunk_z, section_y
code_encoding = "uint16_le_y_z_x"
cell_codes                             65,536-byte MessagePack binary
cell_palette[C]                        C <= 65,536
shape_palette[S]                       S <= 16,384
```

`cell_palette` entries are:

```text
[flags, shape_index, fluid_level, fluid_fill_height,
 support, block_damage, fluid_damage,
 movement[9], fluid_movement[6]]
```

Each shape entry is a flat sequence of zero to nine exact six-value collision
boxes. Active box order is part of the semantic identity because compiled
collision resolution consumes that sequence. Entry zero in both palettes is
canonical air/empty. Palettes are local to the section; Python validates and
merges their semantics into one bounded Region v1 palette. Codes, palette
references, numeric ranges, boxes, duplicate sections, and total merged
capacity all fail closed. `fluid_fill_height` is the native cell level divided
by that fluid asset's `MaxFluidLevel`, in `[0,1]`; legacy section-v1 artifacts
remain readable with fluid fill explicitly unavailable.

Section protocols v1/v2 do not transmit `BlockSection.getFiller()` offsets.
Their detail boxes are exact asset/rotation shapes, but they are insufficient
for native standability because `CollisionConfig` translates filler-cell box
origins back to the root.

#### RegionLightSection

`hytalerl_native_region_light_section_v1` returns:

```text
chunk_x, chunk_z, section_y
status = ready | global_light_not_ready | changed_during_capture
available
global_change_counter, global_light_change_id
light_encoding = uint16_le_y_z_x_rgbs_nibbles
light_data                              65,536-byte binary when ready; empty otherwise
```

Each uint16 contains red, green, blue, and sky light in successive four-bit
nibbles. Ready requires counter/change-ID equality and a global-light object
other than `ChunkLightData.EMPTY`; the latter matters because a fresh section
starts at counter zero and the shared empty object also has change ID zero.
The bridge checks the counter and object identity again after copying all
32^3 cells. This is a static native-light sidecar, not Region geometry
identity or a dynamic relighting stream.

#### TraversalProbe

`hytalerl_native_traversal_probe_v1` returns:

```text
server_version, world, worldgen_provider, worldgen_version, seed
sample_count = N
positions_f64_le_xyz                    binary[N * 3 * 8]
upward_limits_f64_le                    binary[N * 8]
actor_bounds_f64_le                     binary[6 * 8]
validation_codes_i8                     binary[N]
upward_collision_distances_f64_le       binary[N * 8]
```

Validation codes are native `CollisionModule.validatePosition` results:
`-1=invalid overlap`, `0=clear`, `1=on ground`, `2=touching ceiling`, and
`3=ground+ceiling`. Upward distance is the first native collision distance,
bounded by the requested limit. Invalid rows publish zero distance. The
response is a certification fixture, not a graph or learner observation.

#### PrivilegedEntitySnapshot

`hytalerl_privileged_entity_snapshot_v1` returns:

```text
bridge_sha256, server_version, world, worldgen_provider, worldgen_version, seed
component_filter = "uuid_transform_velocity_v1"
bounds_semantics = "half_open_min_inclusive_max_exclusive"
bounds_f64_le_min_max_xyz                binary[6 * 8]
capacity, total_matching, emitted_count, overflow
uuid_encoding = "rfc4122_network_order_16_bytes_per_row"
uuid_bytes                               binary[emitted_count * 16]
positions_f64_le_xyz                     binary[emitted_count * 3 * 8]
rotation_units = "radians_yaw_pitch_roll"
rotations_f64_le_yaw_pitch_roll          binary[emitted_count * 3 * 8]
velocities_f64_le_xyz                    binary[emitted_count * 3 * 8]
```

Complete rows are sorted lexicographically by unsigned UUID bytes, not by
distance or transient ECS slot. UUID values remain the authoritative 128
bits. When `total_matching > capacity`, `overflow=true`,
`emitted_count=0`, and every row column is empty; no nearest-N or arbitrary
partial result is returned. The response fails closed when the runtime JAR
identity is unavailable.

This payload is privileged diagnostics/critic evidence. It must not be copied
into an actor observation. The deployed three-seed wire/reset-lifecycle smoke
is frozen under
`artifacts/native_privileged_entity_snapshot_rev33_8daef871/`; World's
neutral-scene and in-episode replacement differentials remain separate.

`privileged_npc_snapshot_by_uuid` returns the existing
`hytalerl_privileged_npc_snapshot_v2` payload with `capacity=1`. Exact clients
must additionally reject overflow, more than one row, or any emitted UUID that
differs from the requested 16 bytes.

#### Observation
```json
{
  "type": "observation",
  "obs": {
    "position": [100.5, 64.0, -200.3],
    "velocity": [0.0, 0.0, 0.0],
    "yaw": 45.0,
    "pitch": 0.0,
    "health": 100.0,
    "food_buff_timer": 0,
    "stamina": 10.0,
    "mana": 100.0,
    "inventory": [0, 0, 0, "..."],
    "nearby_blocks": [[1, 0, -1, 9], [-1, 0, 0, 18], "..."],
    "nearby_entities": [[0, 50, -30, 200], "..."],
    "time_of_day": 6000,
    "craftable_count": 3,
    "audio": {
      "schema": "hytalerl_audio_frame_v1",
      "version": 1,
      "available": true,
      "capture_mask": [true, false, true],
      "events": [[0, 41, 2, false, 0.0, 0.0, 0.0, 1.0, 1.0, 0.033]]
    }
  },
  "reward": 1.0,
  "terminated": false,
  "truncated": false,
  "info": {
    "tick": 1234,
    "server_step": 309,
    "bridge_sha256": "64 uppercase hexadecimal digits",
    "native_server_process_uptime_seconds": 123.456
  }
}
```

`bridge_sha256` is the SHA-256 of the JAR loaded by the connected process. It
is omitted for exploded development classes, never inferred from a client-side
build path. `native_server_process_uptime_seconds` is sampled from the
connected JVM and is nonnegative. Evidence-producing clients must fail closed
when the loaded identity is absent or differs from their expected canonical
bridge.

Each `nearby_entities` row is
`[canonical_family, relative_x_scaled, relative_z_scaled, health_scaled]`.
The three scalar fields use the versioned
`observation_normalization.wire_fixed_point_scale` from the packaged combat
ruleset (currently `10`); Java encoding and Python decoding both load that
value rather than duplicating it.

The `base_builder` task additionally reports `base_score`, `curriculum_phase`,
`nights_survived`, and `shelter_built` in `info`.

Native responses additionally identify the authoritative execution surface and
timing. Important fields include:

```json
{
  "backend": "native",
  "native_server_version": "0.5.7",
  "world_template": "hytale",
  "worldgen_provider": "Hytale",
  "worldgen_version": "0.0.0",
  "world": "hytalerl_<unique-id>",
  "npc_role": "Trork_Unarmed",
  "engine_ticks_executed": 4,
  "episode_engine_ticks": 20,
  "engine_simulated_seconds": 0.134,
  "engine_mean_delta_seconds": 0.0335,
  "native_tick_rate": 30,
  "world_paused_between_steps": true,
  "geometry_chunk_streaming": "preload_and_pin_one_chunk_halo_before_each_chunk_transition",
  "geometry_pinned_chunk_count": 12,
  "geometry_pinned_chunk_capacity": 25,
  "native_pinned_chunk_count": 12,
  "supported_actions": "forward,back,left,right,jump,camera_delta_yaw,camera_delta_pitch,hotbar_slot,attack",
  "unsupported_actions": "use,place_block,break_block,craft_recipe",
  "requested_unsupported_actions": "",
  "native_attack_requested": true,
  "native_attack_accepted": true,
  "native_attack_executing": true,
  "native_attack_action_count": 1,
  "native_attack_source": "reset_pinned_item_interaction",
  "native_combat_item_id": "Weapon_Sword_Iron",
  "native_combat_interaction_id": "Weapon_Sword_Primary_Swing_Left",
  "native_combat_interaction_type": "Primary",
  "target_role": "Trork_Brawler",
  "target_health": 61.0,
  "target_max_health": 61.0,
  "target_distance": 2.5,
  "target_x": 100.5,
  "target_y": 64.0,
  "target_z": -202.8,
  "combat_agent_attack_executing": true,
  "combat_target_visible": true,
  "combat_target_attack_phase": 1,
  "combat_target_attack_progress": 0.5,
  "combat_target_attack_index": 0,
  "combat_target_attack_elapsed_ticks": 6,
  "combat_target_attack_id": "Trork_Warrior_Battleaxe_Swing_Left",
  "combat_target_facing_error_degrees": 2.1,
  "combat_target_yaw_degrees": 180.0,
  "combat_target_velocity_x": 0.0,
  "combat_target_velocity_z": 0.0,
  "combat_target_head_yaw_degrees": 180.0,
  "combat_target_head_pitch_degrees": 0.0
}
```

`attack` appears in `supported_actions` when either the reset pins an Item
interaction or the selected NPC role exposes one or more discoverable native
`ActionAttack` components. `native_attack_source` distinguishes those paths.
A request may still have `native_attack_accepted=false` while a prior
chain/cooldown is active.
Hytale owns selector, line-of-sight, facing, damage, and knockback behavior.
Geometry pin count is cumulative within one native session. Capacity 25 is the
fixed 3x3 traversal core plus one-chunk halo; exceeding it fails before
loading rather than evicting or fabricating geometry. `native_pinned_chunk_count`
may be larger during bounded Region capture because it also includes
non-geometry capture pins.

The `combat_*` fields form the stable policy-facing combat contract and are
emitted by both the calibrated simulator and native backend. Attack phase codes
are `0=idle`, `1=windup`, `2=sweep`, `3=recovery`, and `4=cooldown`. Attack
index is `-1` while no authored root is active; the Brawler's five roots use
indices 0-4.

The `target_memory` fixture additionally emits `target_ai_state`, its numeric
state/substate indices, whether the Brawler marks the agent, native target LOS,
the occluder phase, and acquisition/occlusion/search/reopen/reacquisition tick
markers. These are evaluation telemetry only; they are not added to
`combat_state` or used to control the Brawler.

`HytaleEnv` converts the numeric fields into a fixed `combat_state` observation
with this layout:

```text
[agent_attack_executing, target_visible, target_attack_phase,
 target_attack_progress, target_attack_index, target_attack_elapsed_ticks,
 target_facing_error_degrees, target_yaw_degrees,
 target_velocity_x, target_velocity_z,
 target_head_yaw_degrees, target_head_pitch_degrees]
```

This vector is derived client-side from response `info`; it is not duplicated
inside the wire-level `obs` map. Missing combat telemetry becomes a valid idle
vector so older responses and non-combat tasks remain parseable.

Combat-ruleset version 8 simulator responses also expose diagnostic-only
`target_ai_activation_tick`, `simulation_motion_timing_profile`, and
`simulation_motion_delta_seconds`. They identify the deterministic
role-initialization and motion-timing domain selected by the episode seed; they
are not included in the compact policy observation.

Version 7 and later simulator diagnostics use `attack_cooldown_seconds` and
`target_attack_cooldown_seconds`. Hytale assets author `AttackPauseRange` in
seconds, so legality advances by `simulation_motion_delta_seconds`; it is not a
fixed 30-60-response counter.

For the version-8 Razorleaf/Brawler contract, target velocity is aligned with
the Brawler's current facing while scalar speed accelerates toward the authored
chase speed. One scalar speed budget is projected onto the requested direction,
so diagonal input cannot exceed straight-line acceleration or maximum speed.
This NPC pair has non-blocking physical contact; no separation impulse is
encoded in the observation or transition.

### Audio v1

Every observation carries `obs.audio`. Legacy responses that omit it are
parsed as explicitly unavailable, not as silence.

`capture_mask` is ordered `[2d, 3d, entity]`. `available=true` means the frame
is structurally valid; it does not mean every transport is captured. The
current native headless backend reports `[true, false, true]`: its tracked
tick anchor observes world-broadcast `PlaySoundEvent2D` and
`PlaySoundEventEntity` packets, but it is not a player entity and therefore is
not selected by the server's 3-D player spatial query. Simulator observations
report audio unavailable until they have an explicit event producer.

Each sparse event uses this versioned ten-field layout:

```text
[kind, sound_event_index, category, spatial_valid,
 relative_x, relative_y, relative_z,
 volume_modifier, pitch_modifier, age_seconds]
```

Kinds are `0=2d`, `1=3d`, and `2=entity`; entity events use category `-1`.
Events are chronological oldest-to-newest, so nonnegative `age_seconds` is
nonincreasing. Every age must also be no greater than native
`info.engine_simulated_seconds`; `HytaleEnv` validates that cross-field
boundary. The bridge retains at most 16 events from the current step.
`HytaleEnv` expands the sparse list into padding-first fixed arrays and a
`valid` mask.

The policy surface intentionally excludes absolute world position, entity
network IDs, source max-distance internals, and reward shaping. A spatial
vector is meaningful only where `spatial_valid` is set. Packet kind, category,
and sound-event identity are categorical features; consumers should embed or
one-hot them rather than treating their integer values as magnitudes.

Native `info` additionally reports `audio_capture_complete`,
`audio_captures_2d`, `audio_captures_3d`, `audio_captures_entity`,
`audio_event_count`, `audio_overwritten_count`, and `audio_invalid_count`.
Warmup and fixture-setup events are discarded before the reset observation.
Repeated `observe` calls return the same immutable frame and do not drain a
live queue.

### Geometry v5

Every observation carries `obs.geometry`. Legacy responses that omit it are
parsed as explicitly unavailable, not as an empty world.

```json
{
  "schema": "hytale_geometry_v5",
  "version": 5,
  "available": true,
  "exact_collision_shapes": true,
  "origin": [0, 65, 0],
  "cells": [
    [
      0, -1, 0,
      42, 0, 101, 0, 3, 0, 0.0, 255, 0, 0,
      0.18, 0.82, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    ]
  ],
  "agent_bounds": [-0.5, 0.0, -0.5, 0.5, 1.9, 0.5],
  "target_bounds": [-0.475, 0.0, -0.475, 0.475, 1.805, 0.475],
  "agent_los_offset": [0.0, 1.0, 0.0],
  "target_los_offset": [0.0, 1.5675, 0.0],
  "contacts": [
    [0, -1, 0, 0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 1]
  ],
  "grounded": true,
  "ceiling_contact": false,
  "target_los": true,
  "target_los_valid": true
}
```

The origin is the floored agent position and cell offsets span `[-4,4]` on
each axis. Each cell uses this 29-field positional layout:

```text
[dx, dy, dz,
 runtime_block_id, runtime_fluid_id, shape_id, rotation, flags,
 fluid_level, fluid_fill_height, support, block_damage, fluid_damage,
 block_friction, block_drag, block_horizontal_speed_multiplier,
 block_jump_force_multiplier, block_climb_up_speed_multiplier,
 block_climb_down_speed_multiplier, block_climb_lateral_speed_multiplier,
 block_terminal_velocity_modifier, block_bounce_velocity,
 fluid_swim_up_speed, fluid_swim_down_speed, fluid_sink_speed,
 fluid_horizontal_speed_multiplier, fluid_field_of_view_multiplier,
 fluid_entry_velocity_multiplier,
 flattened_collision_boxes]
```

Collision boxes are cell-local six-tuples. The bundled 0.5.7 asset audit
certifies at most nine boxes per cell. Flag bits are:

```text
0 solid, 1 blocks-base-NPC-LOS, 2 fluid, 3 damaging, 4 climbable,
5 bouncy, 6 trigger, 7 protrudes-cell, 8 has-block-movement-settings,
9 has-FluidFX-movement-settings
```

Bit 1 is the role-independent non-transparent predicate. An NPC role's
effective `OpaqueBlockSet` is additional; `target_los` carries the native
effective-role result.

Contacts use 13 fields:

```text
[dx, dy, dz, detail_box_index,
 normal_x, normal_y, normal_z,
 point_x, point_y, point_z,
 collision_start, collision_end, flags]
```

Contact flag bit 0 is touching and bit 1 is overlapping. The Python contract
materializes fixed `[729,9,6]` boxes and `[64,13]` contacts. Neither side
silently truncates: the client raises if a native frame exceeds its certified
capacity. Runtime block, fluid, and shape IDs are diagnostic and scoped to the
running 0.5.7 asset registry; explicit boxes, movement values, and semantic
flags are portable source-of-truth fields. Compiled state preserves block
support/damage, all nine block movement values, and all six FluidFX values.
`cell_environment_result()` exposes them inside JIT. The active walk-controller
transition applies the FluidFX horizontal-speed multiplier sampled at the NPC
feet cell; other captured settings are not yet transition-certified.

Native NPC trace v12 carries the same geometry-v5 contract using
`transport: "hytale_geometry_binary_v1"`. Its `origin_i32_le`,
`cell_i32_le` (12 integer fields per cell), `cell_f64_le` (fill plus 15
movement fields), collision-box columns, 18-value bounds/offset column,
13-value contact rows, counts, and four state-flag bits materialize to the
arrays above. This representation is trace-only; ordinary observations retain
the readable map. The decoder validates lengths, ranges, uniqueness,
finiteness, and capacities before exposing a frame.

LOS offsets are relative to entity transforms and come from the runtime model,
including per-spawn scale. A target without a model uses its bounding-box
center, matching native `PositionCache`. Clients must not substitute nominal
asset eye heights.

`fluid_fill_height` is `0` for non-fluid cells and native
`level / Fluid.MaxFluidLevel` for fluid cells. This is distinct from the raw,
asset-dependent `fluid_level` and is required for exact submersion tests.

See `docs/fidelity/geometry-v5.md` for the Gym/JAX shapes, exhaustion behavior,
native differential fixtures, and current simulation boundaries.

Hytale uses measured wall-clock tick deltas, so `engine_simulated_seconds` is
telemetry rather than an exact `engine_ticks_executed / 30` identity. Clients
should inspect `requested_unsupported_actions`; unsupported requests are not
silently simulated by the native backend.

Native `nearby_blocks` values are runtime block asset indices and native
inventory values use a sorted 0.5.7 item-asset table; neither should be treated
as portable IDs across server versions. Native `nearby_entities` uses canonical
families (`0=trork`, `1=outlander`, `2=scarak`, `3=kweebec`, `4=item_drop`) and
stable role hashes for other NPC families.

#### ErrorResponse
```json
{
  "type": "error",
  "message": "Task not found: invalid_task"
}
```

Errors do not normally close the connection, so a client may correct the
request and continue. Invalid frame lengths are treated as fatal protocol
errors and close the connection after the error response.
