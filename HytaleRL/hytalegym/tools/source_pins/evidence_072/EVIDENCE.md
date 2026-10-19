# EVIDENCE: World source-pin audits on 0.7.2 (lane audits_classify, 2026-10-18/19)

Per audit, one block per claim: verdict A (pin only), B (misaligned, false on 0.7.2), C (open). Inventory was verified by the previous lane and is not re-classified here.

Jars: control 0.5.10 (install/v0.5), installed 0.7.2 (install/release, sha256 23f71f03...). Locator convention: 'offset N' = raw javap bytecode offset; 'dump index N' = instruction index in keybody/window output.

## inventory (previous lane, unchanged)
claim: containers/item reads/bridge | verdict: A | InventoryComponent, ItemStack getters, ItemContainer.getCapacity | verified in the previous pass (see source_pin_baselines_072.json, verified_claims and caveats); caveat: transport.schema v1 text vs v2 emitted by both bridges


## perception
# perception (lead: Claude, partial; control 0.5.10, installed 0.7.2). The analysis agent stopped on the weekly API limit; claims not listed here are C by top-level key. The audit has no SOURCE_CONTRACT dict; its claims are the payload.
Rule result: B (projected_distance_declarations). Locator convention: 'dump index N' = keybody index; 'offset N' = raw javap offset.

claim: projected_distance_declarations (the authored set of UseProjectedDistance declarations, expected 11 by EXPECTED_PROJECTED_DISTANCE_DECLARATIONS) | verdict: B | Assets.zip 0.7.2 via the audit's own _projected_distance_declarations | 17 live declarations against 11 expected: 8 in common, 3 expected declarations absent (CAE_Goblin_Ogre.json, CAE_Skeleton_Praetorian.json under Balancing/Intelligent), 9 new (CAE_Bear_Voidtaken.json, Goblin_Dungeon/CAE_Dungeon_Feastmaster_Boss.json, Goblin_New/CAE_Goblin_Glutton.json, Undead/CAE_Skeleton_Elite_Phase_2.json, Undead/CAE_Skeleton_Praetorian.json, Void/CAE_Void_Spawn_Entombed.json, Void/CAE_Void_Spawn_Surge.json); no value changes on the common paths | status: confirmed (computed live) | 0.7.2 behaviour: the set of authored projected-distance uses changed | construct: none
claim: projected_distance_values_boolean | verdict: C | payload check | the live values in the 17 entries were shown as True in the sample; the boolean test over all entries was not printed in this pass
claim: effective_component_selectors (default_and_walk, fly, dive, use_projected_distance_false) | verdict: C | MotionControllerBase, MotionControllerWalk, MotionControllerFly, MotionControllerDive component selectors | not re-derived
claim: component_selector_contract.* | verdict: C | MotionController selector path | not re-derived
claim: los_contract.perception.unloaded_chunk (blocked) | verdict: C | PositionCache lambda$hasLineOfSightInternal$0 (0.7.2) | the previous pass confirmed this by reading the lambda; not re-read in this pass
claim: los_contract.hit_confirmation.* and los_contract.cache.* | verdict: C | HorizontalSelector.RuntimeSelector.tick, HitDetectionExecutor.LineOfSightProvider, PositionCache | not re-derived
claim: chunk_perception_channels.* (heightmap getHeight, environment getEnvironment, tint getTint, sky light, block light RGB) | verdict: C | ChunkStore, BlockSection, ChunkLightData, ChunkLightingManager, FloodLightCalculation | the previous pass records 103 changed bodies in the light and channel classes; not re-derived
claim: native_perception_channel_capture, native_region_light_capture, native_los_evidence_capture, section_aggregates, region_delta | verdict: C | repo-side contract functions (worldgen) | not re-derived; repo-side

## entity_geometry
# entity_geometry (lead: Claude; control 0.5.10, installed 0.7.2)

Rule result: B (scale_boundary and collision_eligibility.excluded are false on 0.7.2). Claims: 14 entries (see SOURCE_CONTRACT lines 42-70).
Locator convention: 'offset N' is a raw javap bytecode offset; 'dump index N' is an instruction index in keybody.py output.

claim: physical_bounds.semantics (current rotation-adjusted local AABB per entity) | verdict: A | ModelSystems$UpdateBoundingBox | inner class ModelSystems$UpdateBoundingBox: 12 methods, identical normalised bodies on 0.5.10 and 0.7.2 (dump) | confirmed (rotation part); the scale part is decided under scale_boundary
claim: physical_bounds.producer (ModelSystems.UpdateBoundingBox) | verdict: A | ModelSystems$UpdateBoundingBox | same evidence as above | confirmed
claim: collision_eligibility.system (TangiableEntitySpatialSystem) | verdict: A | TangiableEntitySpatialSystem | class present on both jars, 4 methods, bodies identical; query built in <clinit> | confirmed
claim: collision_eligibility.required (TransformComponent, BoundingBox) | verdict: A | TangiableEntitySpatialSystem.<clinit> | <clinit> calls TransformComponent.getComponentType and BoundingBox.getComponentType while building the query (dump index list) | confirmed
claim: collision_eligibility.excluded (Intangible, DeathComponent) | verdict: B | TangiableEntitySpatialSystem.<clinit> | 0.7.2 query excludes Intangible (Intangible.getComponentType, Query.not, Query.and); the class has 0 references to DeathComponent (raw javap count 0 on 0.7.2; 0 in the dump on 0.5.10, whose bodies are identical) | status: confirmed for this class; inferred that no other system removes dead entities from the spatial store (not checked) | 0.7.2 behaviour: the query does not exclude DeathComponent | construct: none. The claim is also false on the 0.5.10 control, so it may be a 0.5.7-era statement; 0.5.7 bytes are not on this machine.
claim: perception_endpoint.owner (PositionCache) | verdict: C | PositionCache.testLineOfSightEntity | the entity LOS path changed on 0.7.2 (previous pass: hasLineOfSight reads a cached roleIndex field); not re-derived in this pass (analysis agent stopped on the weekly API limit)
claim: perception_endpoint.model_present (Model.eyeHeight) | verdict: C | PositionCache.testLineOfSightEntity | same method as above; not re-derived
claim: perception_endpoint.model_absent (BoundingBox.center) | verdict: C | PositionCache.testLineOfSightEntity | same method as above; not re-derived
claim: perception_endpoint.entity_body_blocks_los (false) | verdict: C | PositionCache.testLineOfSightEntity | same method as above; not re-derived
claim: scale_boundary.EntityScaleComponent (diagnostic_display_network_scale) | verdict: B | EntityScaleBoundingBoxSystem.tick | 0.7.2 tick: EntityScaleComponent.getScale() (offset 32), resolveBoxScale(scale, ModelComponent) (39), BoundingBox.setScale(F) (40), each tick; the component changes the physical box | status: confirmed (dump of EntityScaleBoundingBoxSystem.tick; class absent on 0.5.10 per the previous pass, not re-checked here) | 0.7.2 behaviour: EntityScaleComponent is a physical input to BoundingBox | construct: none (mechanics_072.md has no entity-scale row; GEN7-PLAN.md line 370 records the same finding)
claim: scale_boundary.physical_scale_application (already_reflected_in_BoundingBox) | verdict: B | EntityScaleBoundingBoxSystem.tick | the box scale is set from the entity scale component by this system, so the scale is applied here as well as in the model path (a second application is not ruled out, see the next line) | status: confirmed for setScale; inferred for the double application | construct: none
claim: scale_boundary.double_application_forbidden (true) | verdict: C | EntityScaleBoundingBoxSystem.resolveBoxScale | whether resolveBoxScale plus ModelSystems applies the model scale twice is not decided in this pass
claim: transport.schema (hytalerl_privileged_npc_snapshot_v2) | verdict: A | hytale-plugin-0.7.2 PrivilegedNpcSnapshot.java:17 | the bridge emits "hytalerl_privileged_npc_snapshot_v2" (grep of the 0.7.2 bridge source) | confirmed
claim: transport.ordering (uuid_lexicographic_aligned_spatial_and_geometry_rows) | verdict: A | hytale-plugin-0.7.2 PrivilegedNpcSnapshot.java:79-82 | spatial rows sorted by PrivilegedEntityRow.UUID_BYTE_ORDER (PrivilegedEntityRow.java:23-35: Long.compareUnsigned on the most, then least, significant bits, equal to canonical UUID text order); the "aligned geometry rows" part (line 49 message) was not read | status: confirmed for the spatial sort; inferred for the geometry alignment
claim: transport.missing_or_invalid (physical_scene_unavailable_fail_closed) | verdict: C | hytale-plugin-0.7.2 PrivilegedNpcSnapshot.java, SnapshotCodec.java | the label "physical_scene_unavailable_fail_closed" is not emitted anywhere in the 0.7.2 bridge (grep); the error-frame path for missing input was not read end to end
payload: class_sha256 (5 classes) | verdict: C | (pins) | the BoundingBox, EntityScaleComponent, ModelSystems, TangiableEntitySpatialSystem and PositionCache digests; their use is covered by the claims above

## actor_world_state
# actor_world_state (lead: no claim re-derived; the analysis agent stopped on the weekly API limit). Every claim is C in this pass.
Rule result: C. Claims are grouped by SOURCE_CONTRACT top-level key (lines 81-152); per-leaf classification was not completed.

claim: fall_state.* (owner, accumulate, reset, npc_landing_gate, npc_landing_speed, player_boundary) | verdict: C | LivingEntity.currentFallDistance; MotionControllerWalk.computeNewFallSpeed (changed on 0.7.2, previous pass) | not re-derived
claim: breathing_state.* (sample interval, point TargetUtil.getLook_position, default_can_breathe, head_occlusion, event_override, classification, component_fields) | verdict: C | TargetUtil.getLook (changed on 0.7.2, previous pass), BreathingCheckEvent, DamageSystems$BreathingSystem | not re-derived
claim: compiled_provider_boundary.* | verdict: C | repo-side provider | not re-derived
claim: controller_medium.* (SensorInWater to activeMotionController.inWater; floor sample; predicate fluid id not empty) | verdict: C | SensorInWater, MotionControllerBase.inWater | not re-derived
claim: submersion.* (WorldUtil.getPackedMaterialAndFluidAtPosition; fill height Fluid.getMaxFluidLevel; boundary) | verdict: C | WorldUtil (now takes ChunkStore; 0.5.7 took Ref and ComponentAccessor, previous pass) | the signature change is recorded; the semantics were not re-derived
claim: drop.* (MotionControllerWalk.findDropBlockCollision; entity AABB; negative y; first collision times max distance) | verdict: C | MotionControllerWalk.findDropBlockCollision (changed on 0.7.2, previous pass) | not re-derived
claim: stationary_traversal.* (CollisionModule.validatePosition, BoxBlockIntersectionEvaluator.intersectBoxComputeOnGround, detail boxes, filler root offset, clearance) | verdict: C | CollisionConfig.canCollide, CollisionModule (changed on 0.7.2, previous pass) | not re-derived

## locomotion
# locomotion (lead: Claude, partial; control 0.5.10, installed 0.7.2). The analysis agent stopped on the weekly API limit; claims not listed here are C by top-level key.
Rule result: B (movement_state_observation.protocol). Locator convention: 'dump index N' = keybody index; 'offset N' = raw javap offset.

claim: movement_state_observation.protocol (MovementStates_23_ordered_boolean_fields) | verdict: B | com.hypixel.hytale.protocol.MovementStates | exact instance-field count (javap -p): 0.5.10 has 23 boolean fields; 0.7.2 has the same 23 boolean fields plus `public byte extraJumpsUsed;` (a 24th instance field, not boolean); the five static int constants are on both | status: confirmed | 0.7.2 behaviour: the protocol object carries an extra byte after the 23 booleans, and the bridge's 23-bit mask does not carry it | construct: none (no extra-jump row in mechanics_072.md or DESIGNS-072.md)
claim: movement_state_observation.component / npc_update / bridge_transport / availability / client_transition_boundary | verdict: C | MovementStatesComponent, MovementStatesSystem, bridge codec | not re-derived in this pass
claim: npc_walk_movement_state_producer.* (publication, source_order, run_threshold, configuration, player_sprint_boundary, external_fields, invalid) | verdict: C | repo producer (jax/world) citing Role.updateMovementState and MovementStatesSystem | not re-derived; repo-side
claim: walk_steering.* (source, turn, maximum_turn, epsilon, otherwise, movement_stop, builder_defaults_degrees) | verdict: C | MotionControllerWalk.computeHeading; BuilderMotionControllerWalk | computeHeading changed on 0.7.2 (previous pass); not re-derived
claim: sprint.installed_gameplay_config (Default) | verdict: A | Assets.zip 0.7.2 Server/GameplayConfigs/Default.json | "MovementConfig": "Default" (python zipfile read) | confirmed
claim: sprint.installed_forward_sprint_multiplier (1.273) | verdict: A | Assets.zip 0.7.2 Server/Entity/MovementConfig/Default.json | "ForwardSprintSpeedMultiplier": 1.273 (python zipfile read) | confirmed
claim: sprint.master_fallback_forward_sprint_multiplier (1.65) | verdict: C | MovementConfig class default (not a float literal in the 0.7.2 server bytecode, per the analysis agent) | the source of 1.65 was not located in the assets read here
claim: sprint.npc_walk_controller / player_config_selection / server_stamina_scope / transition_physics_location / native_npc_oracle (and the keys after native_npc_oracle) | verdict: C | not re-derived in this pass
payload: pinned assets (ASSET_SHA256, 7 assets, dive roles) | verdict: C | previous pass: 6 of 7 differ from the 0.5.7 pins; not re-derived here

## navigation
# Audit: navigation (tests/fidelity/surrogate_navigation_source_audit.py, SCHEMA hytalerl_navigation_source_audit_v6)

Claims: 53 (astar 32, path_follower 16, path_lifecycle 2, region_adapter_boundary 3). Jar: 0.7.2 HytaleServer.jar (server sha 23f71f03...); control 0.5.10.
Counts: A 50, B 0, C 3. Rule result: C (no B; not all A).
Reading notes: "limits" = the values BodyMotionFindBase installs at runtime from BuilderBodyMotionFindBase defaults. "replan_triggers" = the triggers the claim asserts (non-exhaustive reading, see that block). Bridge-side claims are marked unmodelled.
Tools: nav_craft/off.py (single-class javap -c -p with offsets), nav_craft/asset_keys.py, nav_craft/trace_assets.py; keybody.py / an.py for the normalised bodies.

## astar

claim: astar.source | verdict: A | AStarBase.computePath | evidence: computePath invokes AStarNode.close, getLength, getPosition, getSuccessors and buildPath(AStarNode):boolean (off.py J72 offsets 31-143) | status: confirmed

claim: astar.limits.maximum_path_length | verdict: A | BuilderBodyMotionFindBase.readConfig; AStarBase.computePath | evidence: readConfig 'MaxPathLength' default sipush 200 (readConfig offsets 494-501); BodyMotionFindBase setMaxPathLength (offsets 315-324); computePath offsets 143-150 compare getLength with maxPathLength | status: confirmed

claim: astar.limits.open_nodes | verdict: A | BuilderBodyMotionFindBase.readConfig; AStarBase.computePath | evidence: readConfig 'MaxOpenNodes' default sipush 200 (readConfig offsets 520-527); computePath offsets 507-526 test openNodes.size() >= openNodesLimit | status: confirmed

claim: astar.limits.total_nodes | verdict: A | BuilderBodyMotionFindBase.readConfig; AStarBase.computePath | evidence: readConfig 'MaxTotalNodes' default sipush 900 (readConfig offsets 546-553); computePath offsets 538-557 test visitedBlocks.size() >= totalNodesLimit | status: confirmed

claim: astar.limits.limit_check (after_all_eight_successors_of_one_node) | verdict: A | AStarBase.computePath | evidence: direction loop back-edge goto 180 (offset 503), loop exit if_icmpge 506 (offset 184) falls into the open-node limit test at 506-510; limit tests run once per expanded node | status: confirmed

claim: astar.limits.storage_headroom_per_limit (8) | verdict: A | SearchDirections.create / createDirections / countSteps; MotionControllerWalk.is2D | evidence: countSteps returns 3^n-1 when diagonal moves are on (n = axes not projected out); MotionControllerWalk.is2D() is iconst_1;ireturn; so a Walk NPC has 3^2-1 = 8 successors per expansion, the overshoot bound | status: inferred (flag-to-axis mapping of create args at AStarBase.initComputePath offsets 362-381 not fully traced; would confirm by dumping those pushes)

claim: astar.astar_base_field_limits.open_nodes (80) | verdict: A | AStarBase.<init>() | evidence: offset 12 bipush 80 -> openNodesLimit (constructor default; BodyMotionFindBase overwrites it at runtime via setOpenNodesLimit) | status: confirmed

claim: astar.astar_base_field_limits.total_nodes (400) | verdict: A | AStarBase.<init>() | evidence: offset 18 sipush 400 -> totalNodesLimit (constructor default) | status: confirmed

claim: astar.runtime_default_nodes_per_tick (50) | verdict: A | BuilderBodyMotionFindBase.readConfig; BodyMotionFindBase.continuePathFinder | evidence: readConfig 'StepsPerTick' default bipush 50 (readConfig offsets 469-476) -> nodesPerTick; continuePathFinder passes this.nodesPerTick as computePath's 4th argument (offsets 1-15) | status: confirmed (ctor copy of nodesPerTick from the builder not traced)

claim: astar.runtime_default_build_optimised_path (True) | verdict: A | BuilderBodyMotionFindBase.readConfig; AStarBase.<init>() | evidence: AStarBase.<init> offset 30 iconst_1 (putfield at 31) -> optimizedBuildPath; readConfig 'BuildOptimisedPath' default iconst_1 (readConfig offsets 150-156) | status: confirmed

claim: astar.installed_asset_limit_overrides (0) | verdict: A | 0.7.2 Assets.zip JSON members | evidence: nav_craft/asset_keys.py: MaxPathLength 0, MaxOpenNodes 0, MaxTotalNodes 0, StepsPerTick 0 matching members (BuildOptimisedPath matches 1, Test_Walk_Leave.json, not a limit key) | status: confirmed

claim: astar.goal_estimate (Vector3d.distance_double_then_float) | verdict: A | BodyMotionFind.estimateToGoal(AStarBase,Vector3d,MotionController) | evidence: checkcast AStarWithTarget; getTargetPosition; Vector3d.distance(Vector3dc); d2f; freturn. Identical on J05 and J72 | status: confirmed

claim: astar.grid.fractional_bits (1) | verdict: A | AStarBase.INDEX_FRACTIONAL_BITS; positionToIndex | evidence: javap -constants gives 1; positionToIndex multiplies each axis by ldc2_w 2.0 before MathUtil.fastFloor | status: confirmed

claim: astar.grid.position_bits (11) | verdict: A | AStarBase.POSITION_BITS; indexFromXYZ | evidence: javap -constants gives 11; indexFromXYZ shifts x by 22 and y by 11 (bipush 22, bipush 11) | status: confirmed

claim: astar.grid.position_offset (1024) | verdict: A | AStarBase.POSITION_OFFSET; initComputePath | evidence: javap -constants gives 1024; initComputePath stores ldc2_w 1024 into indexToPositionOffsetX/Y/Z (offsets 143-171) | status: confirmed

claim: astar.grid.position_mask (2047) | verdict: A | AStarBase.POSITION_MASK; indexFromXYZ | evidence: javap -constants gives 2047; indexFromXYZ range-checks each coordinate against ldc2_w 2047 (offsets 6-41) | status: confirmed

claim: astar.successor (MotionController.probeMove) | verdict: A | AStarBase.computePath | evidence: computePath invokeinterface MotionController.probeMove (normalised index 106, keybody.py J72); MotionController changed only addVelocity (an.py changed) | status: confirmed

claim: astar.open (descending_total_cost_pop_end) | verdict: A | AStarBase.addOpenNode(AStarNode,long); AStarBase.computePath | evidence: addOpenNode scans from the end while openNodes[j].getTotalCost() < cost (fcmpg, ifge 53) and inserts at j+1, so the list is descending; computePath pops remove(size-1) (offsets 120-130) | status: confirmed

claim: astar.equal_total_cost (strict_less_insert_scan_lifo) | verdict: A | AStarBase.addOpenNode(AStarNode,long) | evidence: the scan moves only past strictly smaller costs (fcmpg, ifge stops on equal), so an equal-cost node lands after existing equals, nearer the pop end, and is popped first (LIFO) | status: confirmed

claim: astar.cost (travel_cost_plus_evaluator_estimate) | verdict: A | AStarNode.initWithPredecessor; AStarBase.addOpenNode(6-arg) | evidence: initWithPredecessor offsets 32-42 totalCost = travelCost + estimateToGoal; estimate from AStarEvaluator.estimateToGoal at addOpenNode offsets 27-34 | status: confirmed

claim: astar.step_cost (AStarBase.measureWalkCost -> MotionControllerBase.waypointDistance) | verdict: A | AStarBase.addOrUpdateNode; measureWalkCost; MotionControllerBase.waypointDistance | evidence: addOrUpdateNode offset 104 calls measureWalkCost; measureWalkCost invokes MotionController.waypointDistance(Vector3dc,Vector3dc); MotionControllerWalk declares no override (javap -p), so MotionControllerBase.waypointDistance = Math.sqrt(waypointDistanceSquared) (offsets 0-9) | status: confirmed

claim: astar.step_cost_projection (runtime_component_selector_axis_mask) | verdict: A | MotionControllerBase.waypointDistanceSquared(Vector3dc,Vector3dc) | evidence: each axis delta is multiplied by getComponentSelector().x/y/z before squaring (offsets 0-68); identical on J05 and J72 | status: confirmed

claim: astar.native_edge_capture (MotionControllerWalk.probeMove_return_distance) | verdict: A | MotionControllerWalk.probeMove; AStarBase.computePath | evidence: probeMove is public double with dreturn at offsets 45, 356, 406 (values from waypointDistance, getLastDistance, projected length); computePath compares the returned distance with 0.49999995 times the step (ldc2_w 0.49999995 at offset 238, dmul 241, dcmpg 248) | status: confirmed

claim: astar.body_motion_find_estimate (euclidean_distance_to_target) | verdict: A | BodyMotionFind.estimateToGoal | evidence: estimate is org.joml.Vector3d.distance to the target position (Euclidean by JOML's API; JOML body not dumped) | status: confirmed for the call; inferred for JOML semantics

claim: astar.limit_order (after_all_eight_successors_of_one_node) | verdict: A | AStarBase.computePath | evidence: same as astar.limits.limit_check (offsets 184 -> 506) | status: confirmed

claim: astar.progress (7 values in this order) | verdict: A | AStarBase$Progress | evidence: javap -p lists UNSTARTED, ABORTED, COMPUTING, ACCOMPLISHED, TERMINATED, TERMINATED_OPEN_NODE_LIMIT_EXCEEDED, TERMINATED_TOTAL_NODE_LIMIT_EXCEEDED (declaration order equals the claim order; no extra constants) | status: confirmed

claim: astar.native_probe.pathfinder (AStarWithTarget) | verdict: A [unmodelled] | bridge NavigationProbeCapture | evidence: NavigationProbeCapture.java:455 new AStarWithTarget(); native_navigation.py contract "pathfinder": "native_AStarWithTarget" | status: confirmed

claim: astar.native_probe.goal (exact_half_cell_position_index) | verdict: A [unmodelled] | NavigationProbeCapture.java:426; AStarWithTarget.initComputePath | evidence: isGoalReached compares node.getPositionIndex() with getTargetPositionIndex(); AStarWithTarget.initComputePath offsets 33-36 store positionToIndex(target) (half-cell index, both jars) | status: confirmed

claim: astar.native_probe.optimized_build_path (False) | verdict: A [unmodelled] | NavigationProbeCapture.java:460 | evidence: pathfinder.setOptimizedBuildPath(false) | status: confirmed

claim: astar.native_probe.result (full_predecessor_chain) | verdict: A [unmodelled] | NavigationProbeCapture.java (result loop after the sample loop) | evidence: walks node.getNextPathNode() until null, copies positions and travel costs, and throws past capacity | status: inferred (the walk is over the complete path chain; the link direction, predecessor or successor, is not traced in bytecode)

claim: astar.native_probe.bounded_samples (32) | verdict: A [unmodelled] | NativeNavigationPathProbe.java:35 | evidence: public static final int MAX_SAMPLES = 32; count > MAX_SAMPLES rejected (line 61) | status: confirmed

claim: astar.native_probe.ordinary_rollout_overhead (none_opt_in_request_only) | verdict: A [unmodelled] | ClientHandler.java:250, 785-807 | evidence: captureNavigationPathProbe is reached only from ClientHandler.handleNavigationPathProbe (network message); no rollout code references it (grep of the bridge tree) | status: confirmed

## path_follower

claim: path_follower.rejection_weight_default (3.0) | verdict: A | PathFollower.<init>(); BuilderBodyMotionFindBase.readConfig | evidence: PathFollower.<init> offset 71 ldc2_w 3.0; readConfig 'RejectionWeight' default ldc2_w 3.0 (readConfig offsets 327-333) | status: confirmed

claim: path_follower.exact_reached_distance_squared (1.0000000000000002e-10) | verdict: A | PathFollower.updateCurrentTarget | evidence: offsets 29-37 dcmpg against ldc2_w 1.0000000000000002E-10; d2 <= threshold sets reached | status: confirmed

claim: path_follower.near_reached_distance_squared (0.01) | verdict: A | PathFollower.updateCurrentTarget | evidence: offsets 41-57 d2 < 0.01 and d2 > currentWaypointDistanceSquared sets reached | status: confirmed

claim: path_follower.advance (negative_projected_dot_or_distance_short_circuit) | verdict: A | PathFollower.updateCurrentTarget | evidence: offsets 60-63 short-circuit to reached on the distance tests; offsets 66-95 dot = NPCPhysicsMath.dotProduct(..., getComponentSelector), reached when dot < 0 | status: confirmed

claim: path_follower.smoothing (ProbeMoveData.canMoveTo) | verdict: A | PathFollower.smoothPath -> PathFollower.canMoveTo -> ProbeMoveData.canMoveTo | evidence: smoothPath invokes canMoveTo (offsets 98, 142); PathFollower.canMoveTo offset 24 invokes ProbeMoveData.canMoveTo; ProbeMoveData.canMoveTo not in the changed list | status: confirmed

claim: path_follower.native_trace.active_motion (Role.getLastBodySteeringMotion -> BodyMotion.findWrappedBodyMotion(BodyMotionFindBase)) | verdict: A | Role.getLastBodySteeringMotion; Instruction.getBodyMotion; BodyMotion.getSteeringMotion; BodyMotion.findWrappedBodyMotion | evidence: getLastBodySteeringMotion returns lastBodyMotionStep.getBodyMotion().getSteeringMotion(); getSteeringMotion default returns this; findWrappedBodyMotion is isInstance then cast; bridge NativePathFollowerTrace.java:81-84 calls it with BodyMotionFindBase.class | status: confirmed

claim: path_follower.native_trace.waypoints (IWaypoint position and length) | verdict: A | IWaypoint | evidence: javap -p: getLength():int, getPosition():Vector3d, advance(int), next() | status: confirmed

claim: path_follower.native_trace.steering (translation and max distance) | verdict: A | Steering | evidence: javap -p: fields translation (Vector3d), maxDistance (double); getTranslation(), getMaxDistance() | status: confirmed

claim: path_follower.native_trace.component_selector (Role.getActiveMotionController().getComponentSelector) | verdict: A | Role.getActiveMotionController; MotionController.getComponentSelector | evidence: getActiveMotionController returns field activeMotionController; PathFollower.updateCurrentTarget offset 73 invokeinterface MotionController.getComponentSelector | status: confirmed

claim: path_follower.native_trace.stage (after_avoidance_before_steering) | verdict: C | MotionControllerBase.steer/steer0/computeMove; Role.blendAvoidance; BodyMotionFindBase.computeSteering | evidence: steer -> steer0 -> computeMove (abstract; MotionControllerWalk.computeMove) -> executeMove; no pinned class invokes Role.blendAvoidance (only computeSummedDistancePush from Role.java); PathFollower.executePath called from BodyMotionFindBase (offset 1608); the avoidance caller is outside the pinned set and deeper than two levels | status: inferred (needs the blendAvoidance caller and the computeMove order)

claim: path_follower.fixture.role (Trork_Brawler) | verdict: A | 0.7.2 Assets.zip Server/NPC/Roles/Intelligent/Faction/Trork/Trork_Brawler.json | evidence: 843 bytes, sha 1566e5739850 (equals pin); a Variant of Template_Trork_Melee | status: confirmed

claim: path_follower.fixture.state (Chase.Attack) | verdict: A | Template_Trork_Melee.json | evidence: lines 355-369: Sensor State "Chase", nested Sensor State ".Attack" | status: confirmed

claim: path_follower.fixture.motion (Intelligent_Chase_BodyMotion_Seek) | verdict: A | Component_Instruction_Intelligent_Chase.json | evidence: no literal id in Assets.zip (0 hits); lines 247-256 BodyMotion Type "Seek" under the ReadPosition sensor, which the label names | status: inferred (label to BodyMotion mapping)

claim: path_follower.fixture.sight_range (15) | verdict: A | Template_Trork_Melee.json | evidence: Parameters ViewRange Value 15 (lines 16-19); Chase component Parameters ViewRange Value 15 | status: confirmed

claim: path_follower.fixture.hearing_range (8) | verdict: A | Template_Trork_Melee.json | evidence: Parameters HearingRange Value 8 (lines 24-27) | status: confirmed

claim: path_follower.fixture.attack_target_range (4) | verdict: A | Template_Trork_Melee.json | evidence: line 398 "TargetRange": 4 inside the Chase/.Attack branch (line 393 Component_Instruction_Target_Adjusted_Attack_Melee) | status: confirmed

## path_lifecycle

claim: path_lifecycle.owner (BodyMotionFindBase_and_BodyMotionFindWithTarget) | verdict: A | BodyMotionFindWithTarget.mustRecomputePath; BodyMotionFindBase.mustRecomputePath | evidence: WithTarget.mustRecomputePath invokespecial BodyMotionFindBase.mustRecomputePath (offset 4); both classes own the trigger code | status: confirmed

claim: path_lifecycle.replan_triggers (controller_obstructed, avoid_block_damage_changed, force_recompute_flag, target_moved_beyond_authored_distance, target_cone_condition) | verdict: A | BodyMotionFindBase.mustRecomputePath; BodyMotionFindBase.computeSteering; BodyMotionFindWithTarget.mustRecomputePath | evidence: controller_obstructed: isObstructed and !isComputing -> onBlockedPath -> true; avoid_block_damage_changed: isAvoidingBlockDamage != wasAvoidingBlockDamage -> true; force_recompute_flag: computeSteering offsets 718-779 isForceRecomputePath -> forceRecomputePath; target_moved: targetDeltaSquared > minMoveDistanceRecomputeSquared -> reset throttle, true; target_cone_condition: isInViewCone fails -> "Target left (2D) cone" -> true. 0.7.2 onBlockedPath changed (recordRetry + armDelay; 0.5.10 was a no-op) but still leaves the trigger true | status: confirmed for the five; inferred reading. A sixth trigger, "Recovered breathing" (couldBreatheCached), exists on BOTH 0.5.10 and 0.7.2, so the list was already non-exhaustive before 0.7.2 and this is not a 0.7.2 misalignment. If the claim is read as exhaustive, it is false on both jars (pin-era gap, not 0.7.2)

## region_adapter_boundary

claim: region_adapter_boundary.accepted_edges (native_MotionControllerWalk_probeMove) | verdict: A [unmodelled] | MotionControllerWalk.probeMove; repo region/traversal.py:614 | evidence: probeMove is public double (0.7.2) and is the edge authority; traversal.py:614 publishes only edges accepted by native probe evidence | status: confirmed for the 0.7.2 method; repo side confirmed by source

claim: region_adapter_boundary.rejected_probe_positions_stored (False) | verdict: C [unmodelled] | repo artifact writer (region/traversal.py) | evidence: a storage property of the repo's artifact; no 0.7.2 bytecode decides what the writer keeps; source only states accepted-only publication (traversal.py:614) | status: inferred (needs the artifact writer's field list)

claim: region_adapter_boundary.full_astar_equivalence (requires_T8_node_for_node_fixture) | verdict: C [unmodelled] | native_navigation.py contract scope | evidence: a verification requirement (node-for-node fixture), decidable only by a live run; the repo contract scopes it to T8 and leaves the runtime follower and replan to T11 | status: inferred (runtime-only)

## doors
# doors (lead: Claude; control 0.5.10, installed 0.7.2)
Locator convention: 'offset N' is a raw javap bytecode offset; 'dump index N' is an instruction index in keybody.py/window.py output (normalised dump), not a bytecode offset.

Rule result: B (census: 33 vertical door assets on 0.7.2 against the audit's own expectation of 31; no construct). Other claims A or C as listed. 13 claim entries (+1 payload census entry).
Payload hashed: class_sha256 (6 classes; BlockAccessor is absent on 0.7.2 and named by no claim), asset_sha256 (9 door assets, all byte-identical to the pins), vertical_door_asset_sha256 and horizontal_door_asset_sha256 (live census dicts), fixture (live), source_contract (SOURCE_CONTRACT lines 77-106).

claim: vertical_transition.side_test (dot(block_to_actor, yaw_normal) < 0) | verdict: A | 0.7.2 DoorBlockUtils.isInFrontOfDoor(Vector3i,Rotation,Vector3d)Z; 0.5.10 DoorInteraction.isInFrontOfDoor; identical instruction sequence (sin/cos of yaw, directionTo(blockCenter, actor), dot, dcmpg, ifge) | confirmed
claim: vertical_transition.closed_actor_front (OpenDoorIn) | verdict: A | DoorInteraction.interactWithBlock | interactWithBlock on both jars: if isInFrontOfDoor is true the chosen state is OPENED_OUT (0.7.2 dump index 100-102 (isInFrontOfDoor call at 100, ifeq at 101); 0.5.10 dump index 115-117), and OPENED_OUT maps to OpenDoorIn (claim below) | confirmed (selection, mapping); the getInteractionState argument order is the same code on both jars (inferred from identity)
claim: vertical_transition.closed_actor_behind (OpenDoorOut) | verdict: A | DoorInteraction.interactWithBlock | same branch: false gives OPENED_IN (0.7.2 dump index 105; 0.5.10 120), which maps to OpenDoorOut | confirmed
claim: vertical_transition.open (classified as closed) | verdict: A [inferred reading] | DoorBlockUtils$DoorState.fromBlockState: null gives CLOSED; any string other than OpenDoorOut or OpenDoorIn falls to CLOSED (raw offsets 4 and 7 for the null case; the default CLOSED getstatic is dump index 33-34) | confirmed (fromBlockState), inferred (reading of "open: closed")
claim: vertical_transition.enum_to_block_state (OPENED_OUT -> OpenDoorIn; OPENED_IN -> OpenDoorOut) | verdict: A | fromBlockState on both jars: lookupswitch keys 0 (OpenDoorOut, ldc 45) and 1 (OpenDoorIn, ldc 59) go to offsets 96 (OPENED_IN) and 102 (OPENED_OUT); the inverse mapping holds | confirmed
claim: vertical_transition.closed_classification (any other state is closed) | verdict: A | fromBlockState default branch gives CLOSED on both jars | confirmed
claim: double_door.partner_offset (java_int(base_hitbox_max_x)*2-1) | verdict: A | DoorInteraction.getDoubleDoor, identical instructions on both jars: BlockBoundingBoxes ... Box.getMax().x, d2i, iconst_2, imul, iconst_1, isub (dump index 40-45) | confirmed
claim: double_door.partner_yaw (yaw + 180) | verdict: A | Rotation.flip | getDoubleDoor calls Rotation.flip() (dump index 73 on both jars); Rotation.flip() is add(Rotation.OneEighty) on both jars (dump index 0-3) | confirmed
claim: double_door.partner_requirements (same base hitbox; root not filler; opposite physical state) | verdict: A | DoorInteraction.getDoubleDoor | getDoubleDoor on both jars: hitbox index comparison (if_icmpne), DoorInfo.filler test (ifeq), doorState comparison (if_acmpeq), dump index 77-102; the only text difference is the moved DoorState type name | confirmed
claim: double_door.fallback (both panels try opposite direction) | verdict: C | DoorInteraction.checkForDoubleDoor, DoorInteraction.checkDoor | checkForDoubleDoor (one getDoubleDoor call, dump index 5-20) and checkDoor (one getDoubleDoor call) both changed shape; how the two panels fall back to each other spans both methods and was not read end to end on both jars
claim: unsupported_surrogate_mechanics (horizontal door, soft block break and drop, animation and sound, lock and permission) | verdict: A (unmodelled) | the surrogate does not model these; 0.7.2 still has the horizontal door (Door_Horizontal, 15 assets), soft-block break (lambda breakSoftBlocksInHitbox in DoorInteraction), and canOpenDoor; the list states only what the surrogate omits | confirmed (existence), unmodelled
payload: vertical door census (31 expected by the audit's matches flag; live count on 0.7.2 = 33) | verdict: B | (no method: asset census) | Assets.zip 0.7.2: Server/Item/Items/* with BlockType.IsDoor true and Interactions.Use == "Door": 33 (python zipfile read with the audit's own selection rule); Door_Horizontal: 15 (matches). The audit's literal 31 (0.5.7) is false on 0.7.2. The two added vertical door assets were not identified (no 0.5.7 member list is recorded in the repo). construct: none (Constructs.doors = world_v0_5, on demand, mechanics_072.md section 9). Note: pinning the 0.7.2 semantic hash would silently re-pin the 33-entry census, so this audit cannot be (A).
Caveat (not a claim): class BlockAccessor is pinned but absent on 0.7.2; Rotation.add(Rotation,Rotation) removed (instance add kept) per the previous pass; neither is named by a claim.

## mutable_blocks
# mutable_blocks (lead: Claude; control 0.5.10, installed 0.7.2). PARTIAL: the analysis agent stopped on the weekly API limit; this file covers the claims checked in the lead pass. Top-level keys not re-derived are marked C.

Rule result: B (mutation.world, the embedded census, the falling-block list, and the item-interaction vocabulary are false on 0.7.2). Per-leaf classification is incomplete: grouped C entries name the top-level key.
Locator convention: 'offset N' is a raw javap bytecode offset; 'dump index N' is an instruction index in keybody.py output.

claim: health.storage (BlockHealthChunk sparse absolute-position map; missing entry = full health) | verdict: B | BlockHarvestUtils.damageSingleBlock | 0.7.2 damage path reads and writes BlockHealthSection (getHealth(III) dump index 532, damage(III,F,Instant) 684, isFragile(III) 695) and never references BlockHealthChunk; 0.5.10 full body uses BlockHealthChunk.getBlockHealth and damageBlock (see explosion_mutation.md) | status: confirmed (call targets) | 0.7.2 behaviour: per-section BlockHealthSection component | construct: none
claim: health.destroyed (health < 0 or closeToZero) | verdict: A | BlockHealth.isDestroyed(float) (0.7.2) | closeToZero(health) or health < 0, same predicate as 0.5.10 BlockHealth.isDestroyed() | confirmed
claim: health.damage (current minus resolved normalised damage) | verdict: C | BlockHealthSection.damage(III,F,Instant) | body not read in this pass
claim: health.repair (delay 5.0 s, rate 0.1 per s, threshold, tick amount, full removes entry) | verdict: C | BlockHealthSystems$Tick (0.7.2) | not dumped; the regeneration reset at health 1.0 is described in DESIGNS-072.md section 7 (design note, not verified here)
claim: mutation.world (WorldChunk.setBlock) | verdict: B | WorldChunk (0.7.2) and BlockOperations.setBlock | WorldChunk has 0 setBlock and 0 breakBlock overloads on 0.7.2 (dump; the 0.5.10 control has setBlock and setBlockPhysics only); the 0.7.2 write path is BlockOperations.setBlock, which calls BlockAnimationModule.onBlockReplaced (dump index 153) and BlockHealthModule.onBlockReplaced (dump index 158) | status: confirmed | 0.7.2 behaviour: block writes go through BlockOperations.setBlock, and a write clears block health through BlockHealthModule.onBlockReplaced | construct: block_health_reset_on_write (mechanics_072.md row 45) is a design stub: mechanics/block_health.py on_block_write raises NotImplementedError; sim/gen7/profile.py:52 sets the switch True
claim: mutation.break (BlockHarvestUtils) | verdict: C | BlockHarvestUtils.performBlockBreak, naturallyRemoveBlock | changed on 0.7.2 (previous pass); not re-derived
claim: mutation.place (BlockPlaceUtils) | verdict: C | BlockPlaceUtils | not re-derived
claim: mutation.first_run (side effect before interaction runtime wait) | verdict: C | BlockHarvestUtils / BlockPlaceUtils | not re-derived
claim: mutation_outcome_certifier (player_free_effects, bridge_helper, synthetic_player_fixture, certifies, does_not_certify, give_item, run_root_interaction) | verdict: C | bridge and repo contracts | bridge-side and repo-side; the synthetic fixture hash is repo-side; not re-derived
claim: resolved_item_interactions.trigger_vocabulary (all_25_InteractionType_values) | verdict: B | com.hypixel.hytale.protocol.InteractionType | the enum has 29 constants on 0.7.2 (javap -constants, fields of the enum type) and 26 on the 0.5.10 control; the claim says 25, so it was stale on the control too | status: confirmed (count) | 0.7.2 behaviour: 29 interaction types | construct: none
claim: resolved_item_interactions (active_item_resolution, equipped_resolution, game_modes, root_rules, chain, contract_sha256) | verdict: C | repo and bridge contracts | not re-derived
payload: asset_census (EXPECTED_BLOCK_SUPPORT_CENSUS, embedded in the hashed contract) | verdict: B | Assets.zip 0.7.2 via the audit's own _asset_block_support_census (72 s) | live 0.7.2 census differs from the embedded 0.5.7 values in 6 fields: resolved_block_rows 5823 -> 6216; implicit_bottom_support_rows 88 -> 144; implicit_full_supporting_rows 617 -> 665; player_placement_deco_rows 571 -> 535; support_dependent_rows 1883 -> 2065; support_drop_counts {none 3940, break 1881, destroy 0, fall 2} -> {none 4151, break 2055, destroy 0, fall 10} | status: confirmed (computed live) | construct: none
payload: installed_falling_blocks (EXPECTED_FALLING_BLOCKS, embedded) | verdict: B | _asset_falling_blocks(Assets.zip 0.7.2) | 11 live entries against 2 expected | status: confirmed (computed live) | construct: none
payload: installed_resolved_procedures (EXPECTED_BLOCK_TICK_PROCEDURES, embedded) | verdict: C | _asset_block_tick_procedures | not computed in this pass
claim: tick procedures and the remaining health-API claims (BlockHealthSystems Tick, ReplicateChanges, PlaceBlockEventSystem, EnsureSection, LoadPacket, MigrateBlockHealthChunk) | verdict: C | BlockHealthSystems (not dumped) | not re-derived

## explosion_mutation
# explosion_mutation (lead: Claude; jars: control 0.5.10, installed 0.7.2)

Rule result: B (the health storage claim is false on 0.7.2; see below). 27 claim entries (the four host_publication items after the contract share one line).
Locator convention: 'offset N' is a raw javap bytecode offset; 'dump index N' is an instruction index in keybody.py/window.py output (normalised dump), not a bytecode offset.
Payload hashed: class_sha256 (12 classes), asset_sha256 (Recipe_Book_Magic_Air.json), fixture facts (live), host_contract_sha256 (repo), source_contract (SOURCE_CONTRACT lines 94-135).

claim: entrypoint.method (performExplosion/processTargetBlocks) | verdict: A | ExplosionUtils.performExplosion | 0.7.2 calls processTargetBlocks at bytecode offset 130; 0.5.10 at 76 | confirmed (raw_J72/raw_J05)
claim: entrypoint.origin (floor each axis + 0.5) | verdict: A | ExplosionUtils.performExplosion | 0.7.2 offsets 77-114 floor(x,y,z)+0.5; 0.5.10 offsets 24-61 identical | confirmed
claim: entrypoint.entity_path (disabled) | verdict: A | performExplosion | processTargetEntities runs only if damageEntities (0.7.2 offsets 133-137; 0.5.10 79-83); the fixture's false value is repo-side, not re-derived | confirmed (code), inferred (fixture value)
claim: entrypoint.block_path (enabled) | verdict: A |  ExplosionUtils.processTargetBlocks | called unconditionally from performExplosion on both jars; block work gated by damageBlocks inside (0.7.2 processTargetBlocks, gate near offset 587-591) | confirmed
claim: targeting.candidates (BlockSphereUtil.forEachBlock integer radius) | verdict: A | BlockSphereUtil | 9 common methods, 0 changed; processTargetBlocks calls forEachBlock on both | confirmed
claim: targeting.occlusion (TargetUtil.getTargetBlock first valid block on ray) | verdict: A (mechanism); the predicate changed | TargetUtil.getTargetBlock(ChunkStore,...) 0.7.2 = TargetBuffer + BlockIterator.iterate with accept predicate, same structure as 0.5.10 World overload | status: confirmed (structure); the "valid" predicate ExplosionUtils.isValidTargetBlock(int,boolean) gained one rule in 0.7.2: getExplosionReactionState() != null returns true (offsets 36-40), not named by the claim, not exercised by the fixture (inferred: fixture blocks carry no reaction state; not checked)
claim: targeting.loaded_chunk (chunk reference present and valid) | verdict: C |  ExplosionUtils.processTargetBlocks | 0.5.10 uses ChunkStore.getChunkReference(J) with ChunkUtil.indexChunkFromBlock; 0.7.2 uses ChunkStore.getChunkSectionReferenceAtBlock(III) and Ref.isValid; whether a loaded chunk can lack a section that holds blocks is not decided here (needs the section lifecycle)
claim: targeting.duplicate_failed_hits (avoid set) | verdict: A |  ExplosionUtils.processTargetBlocks | 0.7.2 skips positions in the ObjectOpenHashSet (Set.contains) and adds a position when performBlockDamage returns false (offset ~618 ifne 678 then Set.add); 0.7.2 also adds non-breakable positions (isBlockBreakingAllowed, new, unclaimed) | confirmed
claim: damage.scale (one_minus_pow(distance/radius, falloff)) | verdict: A | ExplosionUtils.calculateBlockDamageScale(FFF)F | 19 instructions, byte-identical normalised body on both jars (raw offsets 8-21: fdiv (10), Math.pow (17), fsub (21)) | confirmed
claim: damage.radius_boundary (zero scale at distance >= radius) | verdict: A | calculateBlockDamageScale | raw offsets 2-7: fcmpl(distance, radius) at 2, iflt 8 at 3, fconst_0 and freturn at 6-7, so distance >= radius returns 0; identical on both jars | confirmed
claim: damage.application (BlockHarvestUtils.performBlockDamage) | verdict: A (name and wrapper shape hold); caveat | 0.5.10: 8-arg wrapper (passes iconst_0) into the 11-arg full body; 0.7.2: 9-arg (explosion passes iconst_1) -> 13-arg -> damageSingleBlock (982 instructions); the forwarded boolean's meaning was not re-derived | confirmed (structure), inferred (flag meaning)
claim: damage.soft_block (damage one, force scale one) | verdict: C |  BlockHarvestUtils.damageSingleBlock | 0.7.2 has an isSoft branch (dump index 382 (keybody)) with fconst_1 pushes (397, 405), but the forced-scale path and the flag that reaches it were not decided in this pass
claim: damage.health (BlockHealthChunk sparse absolute-position map; destroyed = health < 0 or closeToZero) | verdict: B (storage); destroyed predicate A | 0.5.10 full body calls BlockHealthModule.get().getBlockHealthChunkComponentType, BlockHealthChunk.getBlockHealth(Vector3i), damageBlock(Instant,...), isBlockFragile(Vector3i), BlockHealth.setHealth (dump index 455-615 (keybody)). 0.7.2 damageSingleBlock calls BlockHealthSection.getHealth(III), damage(III,F,Instant) and isFragile(III) (dump index 513-698 (keybody)) and never references BlockHealthChunk. BlockHealth.isDestroyed keeps the same predicate (closeToZero or health < 0; 0.5.10 dump index 0-12 (keybody); 0.7.2 isDestroyed(float) 0-10) | status: confirmed | 0.7.2 behaviour: explosion damage is a per-section BlockHealthSection component (no absolute-position BlockHealthChunk map on the damage path) | construct: none for storage and damage; block_health_reset_on_write (mechanics_072.md row 45) is a design stub: mechanics/block_health.py on_block_write raises NotImplementedError, its switch is True in sim/gen7/profile.py:52 and False in sim/common/profile.py:103
claim: damage.filler (filler offset resolved by FillerBlockUtil) | verdict: A | damage body | FillerBlockUtil.unpackX/Y/Z used on both jars (0.5.10 dump index 155-161 (keybody); 0.7.2 157-166 after BlockSection.getFiller(III)) | confirmed
claim: mutation.destroy (remove block health, then WorldChunk.breakBlock) | verdict: A (renamed mechanism, same end state) | 0.7.2 destroy is performBlockBreak then BlockOperations.setBlock(EMPTY) (damageSingleBlock dump index 726-773 (keybody)); BlockOperations.setBlock calls BlockHealthModule.onBlockReplaced (dump index 158 (keybody)), which clears the entry (per DESIGNS-072 section 7, the clear chain still to confirm in the mutable_blocks evidence). WorldChunk.breakBlock is absent on 0.7.2 and on the 0.5.10 control (WorldChunk has only setBlock and setBlockPhysics on 0.5.10), so the named method was already gone from the control | status: confirmed (setBlock path, hook call), inferred (clear chain) | construct: block_health_reset_on_write is a design stub (mechanics/block_health.py raises NotImplementedError); the write-clears-health rule is not modelled in code
claim: mutation.partial (retained sparse block health) | verdict: C |  BlockHarvestUtils.damageSingleBlock | retained health is written back via BlockHealthSection.damage and kept unless isDestroyed; the sparse storage form was not re-derived
claim: mutation.controlled_scope (loaded dry direct blocks, no filler, no support cascade) | verdict: C | repo fixture scope plus performBlockBreak and naturallyRemoveBlock (changed on 0.7.2); whether the 0.7.2 path cascades is not decided here
claim: drops.settings (1028 + 2048 when random float exceeds drop chance) | verdict: A |  ExplosionUtils.processTargetBlocks | 0.7.2 offsets 259-279 (sipush 1028; nextFloat; blockDropChance; fcmpl; ifle; sipush 2048); 0.5.10 offsets 236-256 identical | confirmed
claim: drops.drop_chance_one (no suppression bit when chance is 1) | verdict: A |  ExplosionUtils.processTargetBlocks | the bit is added only when nextFloat() > chance; nextFloat is in [0,1) (Java API), so the bit is never added at chance 1 | confirmed (structure), Java spec (range)
claim: drops.resolution (BlockHarvestUtils.getDrops then ItemComponent.generateItemDrops) | verdict: A (0.7.2 path) | damageSingleBlock calls getDrops(BlockType,int,String,String) then BlockHarvestUtils.spawnDrops, which calls ItemComponent.generateItemDrops (dump index 793-870 (keybody) and the spawnDrops show) | confirmed for 0.7.2; the 0.5.10 drop path not located in this pass
claim: drops.stack (ItemStack item, quantity, durability, metadata) | verdict: C | getDrops builds ItemStack(String,int) which delegates to ItemStack(String,int,BsonDocument null) on both jars (identical 6-instruction delegate); the 3-argument constructor changed on 0.7.2 (ItemStack constructors changed, per the crafting gap); durability and metadata initialisation not re-derived
claim: drops.fixture (soft ItemId is Recipe_Book_Magic_Air) | verdict: A | Server/Item/Items/Recipe/Recipe_Book_Magic_Air.json in 0.7.2 Assets.zip | BlockType.Gathering.Soft = {"ItemId": "Recipe_Book_Magic_Air"}, Breaking null (python zipfile read) | confirmed
claim: host_publication.contract (native_explosion_mutation_host_contract_sha256) | verdict: C | repo identity, not a server claim; no bytecode carries it | status: not decidable from the server build
claim: host_publication.evidence_phase / geometry_refresh / incomplete_after_execution / unsupported_or_overflow | verdict: C | bridge: ExplosionMutationCodec.java:136-137 packs resync_required from the probe; NativeExplosionMutationProbe.java:798-803 checks overflow == (total > capacity); the empty-frame, same-connection requery and fail-closed paths were not read end to end

## explosion_dynamics
# explosion_dynamics (lead: Claude; control 0.5.10, installed 0.7.2)

Rule result: A (all five contract claims hold on 0.7.2; the audit's matches flag stays false by design, see caveat 2). 5 claim entries.
Payload hashed: class_sha256 (ExplosionUtils, DamageSystems), host_contract_sha256 (repo native_explosion_dynamics_contract), source_contract (5 claims).
Recorded: EXPECTED_SEMANTIC_SHA256 = d03a747e... by rebaseline.py accept in this lane (observed semantic equals the accepted value; verifier not yet re-run at this point).

claim: phase_order (blocks, then entities, then visual effects) | verdict: A | ExplosionUtils.performExplosion | 0.7.2 calls processTargetBlocks (offset 130), processTargetEntities when damageEntities (133-149), performExplosionEffects (157); a debugShapes gate (15-67) precedes the blocks phase; 0.5.10 same order | confirmed
claim: within_phase_order (hash_set_iteration_unspecified) | verdict: A | processTargetBlocks and processTargetEntities | both iterate Set.iterator over fastutil hash sets (ObjectOpenHashSet in blocks, ReferenceOpenHashSet from performExplosion) on both jars | confirmed
claim: raw_damage (entityDamage*pow(1-distance/entityDamageRadius,entityDamageFalloff)) | verdict: A | ExplosionUtils.processTargetEntity | 0.7.2 offsets 117-135 compute entityDamage * pow(1 - d/entityDamageRadius, entityDamageFalloff), then the raw > 0 test (138-142); same formula on 0.5.10 | confirmed
claim: health_application (Math.round(raw_damage) before health subtraction) | verdict: A | DamageSystems$ApplyDamage.handle (inner class, not hashed by the audit) | setAmount((float) Math.round(getAmount())) at offsets 76-87, then EntityStatMap.subtractStatValue at 90-99; identical offsets on 0.5.10 (round 83, subtract 99) | confirmed
claim: wire_order (UUID_text_ascending_serialization_only) | verdict: A | hytale-plugin-0.7.2 NativeExplosionCandidateProbe.java | fromMatches sorts by Comparator.comparing(Candidate::uuid, UUID_TEXT_ORDER) (line 111; comparator UUID::toString, lines 149-150); the constructor rejects non-ascending or duplicate UUIDs (76-86); overflow emits an empty list (128) | confirmed (bridge source; the server's execution order is the hash-set order above)

Caveats (not claims of this audit; recorded, not blocking):
1. processTargetEntity changed beyond the five claims. On 0.7.2 the damage cause comes from ExplosionConfig.entityDamageCauseId (0.7.2 ExplosionConfig.<init> sets the string "Environment", offsets 41-44; a null cause skips the hit at 145-162); 0.5.10 used getstatic DamageCause.ENVIRONMENT. A zero-length knockback direction falls back to (0,1,0) (offsets 240-290). The fixture sets knockback false (repo host contract), and no bridge code sets entityDamageCauseId (grep of hytale-plugin-0.7.2/src/main), so the default applies. Inferred: the string resolves to DamageCause.ENVIRONMENT (asset Server/Entity/Damage/Environment.json exists in the 0.7.2 Assets.zip; the static binding site was not read).
2. The host contract (worldgen/native_explosion_dynamics.py) still names the 0.5.7 ExplosionUtils and DamageSystems digests as source identity. EXPECTED_HOST_CONTRACT_SHA256 is not recorded for 0.7.2 (not re-keyed in this lane), so the audit's matches_0_5_7_baseline stays false and the audit exits 1. The gate passes because it compares only the semantic hash when the exit is non-zero (source_pins.py:133).

## crafting
# crafting (lead: no claim re-derived; the analysis agent stopped on the weekly API limit). Every claim is C in this pass.
Rule result: C. Claims are grouped by SOURCE_CONTRACT top-level key (lines 135-249).

note: the previous pass names CraftingManager.isValidBenchForRecipe as changed. In 0.7.2 that method (package com.hypixel.hytale.builtin.crafting.component.CraftingManager) checks the player's known recipes (Player.getPlayerConfigData, KnownRecipes, isKnowledgeRequired), not the bench match, so the bench-match predicate the audit cites (old CraftingManager.java:372-403) was not located in this pass.
claim: recipe_identity.* (String asset key; BenchRecipeRegistry and CraftRecipeAction java file:line citations) | verdict: C | old decompile file:line citations, not bytecode | not checkable from the 0.7.2 bytecode
claim: bench_gate.* (window attachment; execution_predicate; fieldcraft; page types) | verdict: C | bench-match predicate not located (see note) | not re-derived
claim: execution_boundary.* (player_requirement assert; non_player_output; certificate) | verdict: C | CraftingManager.craftItem and giveOutput | not re-derived
claim: learning.* (PlayerConfigData codecs; execution key; interaction key; unverified consistency) | verdict: C | PlayerConfigData, LearnRecipeInteraction | not re-derived
claim: craftable_count.* (simulator counts; native writes zero) | verdict: C | repo-side simulator (BaseBuilderTask.java and NativeEnvironmentSession.java file:line) | repo and old citations; not re-derived
claim: inventory_boundary.* (getInputMaterials clone; resource selectors; simple input priority; crafted stack; pickup priority; overflow) | verdict: C | CraftingManager.getInputMaterials, getOutputItemStack; InventoryUtils.moveItem, smartMoveItem, getSectionById (changed, previous pass); SimpleItemContainer.addOrDropItemStack | not re-derived
payload: recipe corpus expectation (EXPECTED_ASSET_METRICS, 1947 recipes) | verdict: C | asset metrics | the previous pass records 2,047 recipes on 0.7.2 (FIDELITY (b)); not recounted in this pass, so a B cannot be stated here
