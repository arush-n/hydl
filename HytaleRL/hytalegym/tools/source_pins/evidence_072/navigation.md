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
