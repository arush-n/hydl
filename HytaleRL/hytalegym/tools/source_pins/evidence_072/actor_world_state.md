# actor_world_state (lead: no claim re-derived; the analysis agent stopped on the weekly API limit). Every claim is C in this pass.
Rule result: C. Claims are grouped by SOURCE_CONTRACT top-level key (lines 81-152); per-leaf classification was not completed.

claim: fall_state.* (owner, accumulate, reset, npc_landing_gate, npc_landing_speed, player_boundary) | verdict: C | LivingEntity.currentFallDistance; MotionControllerWalk.computeNewFallSpeed (changed on 0.7.2, previous pass) | not re-derived
claim: breathing_state.* (sample interval, point TargetUtil.getLook_position, default_can_breathe, head_occlusion, event_override, classification, component_fields) | verdict: C | TargetUtil.getLook (changed on 0.7.2, previous pass), BreathingCheckEvent, DamageSystems$BreathingSystem | not re-derived
claim: compiled_provider_boundary.* | verdict: C | repo-side provider | not re-derived
claim: controller_medium.* (SensorInWater to activeMotionController.inWater; floor sample; predicate fluid id not empty) | verdict: C | SensorInWater, MotionControllerBase.inWater | not re-derived
claim: submersion.* (WorldUtil.getPackedMaterialAndFluidAtPosition; fill height Fluid.getMaxFluidLevel; boundary) | verdict: C | WorldUtil (now takes ChunkStore; 0.5.7 took Ref and ComponentAccessor, previous pass) | the signature change is recorded; the semantics were not re-derived
claim: drop.* (MotionControllerWalk.findDropBlockCollision; entity AABB; negative y; first collision times max distance) | verdict: C | MotionControllerWalk.findDropBlockCollision (changed on 0.7.2, previous pass) | not re-derived
claim: stationary_traversal.* (CollisionModule.validatePosition, BoxBlockIntersectionEvaluator.intersectBoxComputeOnGround, detail boxes, filler root offset, clearance) | verdict: C | CollisionConfig.canCollide, CollisionModule (changed on 0.7.2, previous pass) | not re-derived
