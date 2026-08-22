package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.builtin.fallingblocks.FallingBlock;
import com.hypixel.hytale.builtin.hytalegenerator.plugin.HandleProvider;
import com.hypixel.hytale.builtin.hytalegenerator.plugin.HytaleGenerator;
import com.hypixel.hytale.component.AddReason;
import com.hypixel.hytale.component.ArchetypeChunk;
import com.hypixel.hytale.component.ComponentType;
import com.hypixel.hytale.component.ComponentAccessor;
import com.hypixel.hytale.component.Component;
import com.hypixel.hytale.component.Holder;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.query.Query;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.math.shape.Box;
import com.hypixel.hytale.math.vector.Rotation3f;
import com.hypixel.hytale.math.vector.Transform;
import com.hypixel.hytale.protocol.BlockMaterial;
import com.hypixel.hytale.protocol.BlockFace;
import com.hypixel.hytale.protocol.BlockPosition;
import com.hypixel.hytale.protocol.BlockRotation;
import com.hypixel.hytale.protocol.FluidFXMovementSettings;
import com.hypixel.hytale.protocol.GameMode;
import com.hypixel.hytale.protocol.InteractionSyncData;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.protocol.InteractionState;
import com.hypixel.hytale.protocol.Rotation;
import com.hypixel.hytale.protocol.WaitForDataFrom;
import com.hypixel.hytale.protocol.ToClientPacket;
import com.hypixel.hytale.server.core.asset.type.attitude.Attitude;
import com.hypixel.hytale.protocol.packets.world.PlaySoundEvent2D;
import com.hypixel.hytale.protocol.packets.world.PlaySoundEventEntity;
import com.hypixel.hytale.server.core.asset.type.blockhitbox.BlockBoundingBoxes;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockMovementSettings;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.RotationTuple;
import com.hypixel.hytale.server.core.asset.type.fluid.Fluid;
import com.hypixel.hytale.server.core.asset.type.fluidfx.config.FluidFX;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.asset.type.item.config.Item;
import com.hypixel.hytale.server.core.asset.type.item.config.ItemArmor;
import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionContext;
import com.hypixel.hytale.server.core.entity.InteractionEntry;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.entity.UUIDComponent;
import com.hypixel.hytale.server.core.entity.entities.Player;
import com.hypixel.hytale.server.core.entity.entities.BlockEntity;
import com.hypixel.hytale.server.core.entity.entities.ProjectileComponent;
import com.hypixel.hytale.server.core.entity.damage.DamageDataComponent;
import com.hypixel.hytale.server.core.entity.knockback.KnockbackComponent;
import com.hypixel.hytale.server.core.inventory.Inventory;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.modules.entity.component.HeadRotation;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.modules.entity.component.CollisionResultComponent;
import com.hypixel.hytale.server.core.modules.entity.component.ModelComponent;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entity.component.FromPrefabInstance;
import com.hypixel.hytale.server.core.modules.entity.component.WorldGenId;
import com.hypixel.hytale.server.core.modules.entity.damage.Damage;
import com.hypixel.hytale.server.core.modules.entity.tracker.EntityTrackerSystems;
import com.hypixel.hytale.server.core.modules.entity.player.ChunkTracker;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatMap;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatValue;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatsModule;
import com.hypixel.hytale.server.core.modules.interaction.BlockHarvestUtils;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.ChargingInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.ChainingInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.operation.Operation;
import com.hypixel.hytale.server.core.modules.blockhealth.BlockHealth;
import com.hypixel.hytale.server.core.modules.blockhealth.BlockHealthChunk;
import com.hypixel.hytale.server.core.modules.blockhealth.BlockHealthModule;
import com.hypixel.hytale.server.core.modules.physics.component.Velocity;
import com.hypixel.hytale.server.core.modules.physics.component.PhysicsValues;
import com.hypixel.hytale.server.core.modules.projectile.component.PredictedProjectile;
import com.hypixel.hytale.server.core.modules.collision.BlockCollisionData;
import com.hypixel.hytale.server.core.modules.collision.CollisionModule;
import com.hypixel.hytale.server.core.modules.collision.CollisionResult;
import com.hypixel.hytale.server.core.modules.splitvelocity.VelocityConfig;
import com.hypixel.hytale.server.core.modules.time.WorldTimeResource;
import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.Universe;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.WorldConfig;
import com.hypixel.hytale.server.core.universe.world.chunk.ChunkFlag;
import com.hypixel.hytale.server.core.universe.world.chunk.BlockChunk;
import com.hypixel.hytale.server.core.universe.world.chunk.ChunkColumn;
import com.hypixel.hytale.server.core.universe.world.chunk.WorldChunk;
import com.hypixel.hytale.server.core.universe.world.chunk.section.BlockSection;
import com.hypixel.hytale.server.core.universe.world.chunk.section.FluidSection;
import com.hypixel.hytale.server.core.universe.world.spawn.GlobalSpawnProvider;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.core.universe.world.storage.provider.EmptyChunkStorageProvider;
import com.hypixel.hytale.server.core.universe.world.storage.resources.EmptyResourceStorageProvider;
import com.hypixel.hytale.server.core.universe.world.worldgen.provider.FlatWorldGenProvider;
import com.hypixel.hytale.server.core.universe.world.worldgen.provider.FlatWorldGenProvider.Layer;
import com.hypixel.hytale.server.core.universe.world.worldmap.provider.DisabledWorldMapProvider;
import com.hypixel.hytale.server.core.util.FillerBlockUtil;
import com.hypixel.hytale.server.npc.NPCPlugin;
import com.hypixel.hytale.server.npc.corecomponents.combat.ActionAttack;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.movement.MotionKind;
import com.hypixel.hytale.server.npc.movement.NavState;
import com.hypixel.hytale.server.npc.movement.Steering;
import com.hypixel.hytale.server.npc.movement.controllers.MotionController;
import com.hypixel.hytale.server.npc.movement.controllers.MotionControllerBase;
import com.hypixel.hytale.server.npc.movement.controllers.MotionControllerWalk;
import com.hypixel.hytale.server.npc.movement.controllers.ProbeMoveData;
import com.hypixel.hytale.server.npc.navigation.AStarBase;
import com.hypixel.hytale.server.npc.navigation.AStarEvaluator;
import com.hypixel.hytale.server.npc.navigation.AStarNode;
import com.hypixel.hytale.server.npc.navigation.AStarNodePoolProviderSimple;
import com.hypixel.hytale.server.npc.navigation.AStarWithTarget;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.support.MarkedEntitySupport;
import com.hypixel.hytale.server.npc.role.support.CombatSupport;
import com.hypixel.hytale.server.npc.role.support.StateSupport;
import com.hypixel.hytale.server.npc.util.NPCPhysicsMath;
import com.hypixel.hytale.server.spawning.spawnmarkers.SpawnMarkerEntity;
import com.hypixel.hytale.server.worldgen.HytaleWorldGenProvider;
import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.action.group.NativeGroupAction;
import com.hytalerlbridge.action.group.NativeGroupActionContract;
import com.hytalerlbridge.action.group.NativePolicyCombatBindingSpec;
import com.hytalerlbridge.action.NativeWorldVerbRequest;
import com.hytalerlbridge.combat.CombatPhase;
import com.hytalerlbridge.combat.CombatRuleset;
import com.hytalerlbridge.combat.CombatTelemetry;
import com.hytalerlbridge.combat.HytaleCombatAssets;
import com.hytalerlbridge.combat.MeleeAttackProfile;
import com.hytalerlbridge.combat.dodge.NativeDodgeBindingResolver;
import com.hytalerlbridge.combat.dodge.NativeDodgeProgram;
import com.hytalerlbridge.combat.guard.NativeGuardProgram;
import com.hytalerlbridge.entity.PrivilegedEntityQuery;
import com.hytalerlbridge.entity.PrivilegedEntityRow;
import com.hytalerlbridge.entity.PrivilegedEntitySnapshot;
import com.hytalerlbridge.entity.PrivilegedNpcGeometryRow;
import com.hytalerlbridge.entity.PrivilegedNpcSnapshot;
import com.hytalerlbridge.entity.PrivilegedNpcUuidQuery;
import com.hytalerlbridge.environment.EnvironmentOptions;
import com.hytalerlbridge.environment.fidelity.NativeResourceOverride;
import com.hytalerlbridge.environment.NativeDoorTransitionSource;
import com.hytalerlbridge.environment.NativeCraftingCatalogSource;
import com.hytalerlbridge.environment.NativePolicyWorldActionCaptureSource;
import com.hytalerlbridge.environment.NativeDropProgramSource;
import com.hytalerlbridge.environment.NativeExplosionMutationSource;
import com.hytalerlbridge.environment.NativeExplosionDynamicsSource;
import com.hytalerlbridge.environment.NativeExplosionProbeSource;
import com.hytalerlbridge.environment.NativeItemInteractionSource;
import com.hytalerlbridge.environment.NativeLineOfSightSource;
import com.hytalerlbridge.environment.NativeMutableBlockSource;
import com.hytalerlbridge.environment.NativePerceptionChannelSource;
import com.hytalerlbridge.environment.NativePrivilegedEntitySource;
import com.hytalerlbridge.environment.NativeNpcTraceSource;
import com.hytalerlbridge.environment.NativeWorldgenStructureMarkerSource;
import com.hytalerlbridge.environment.NativeRegionSource;
import com.hytalerlbridge.environment.NativeTraversalProbeSource;
import com.hytalerlbridge.environment.EnvironmentSession;
import com.hytalerlbridge.environment.StepResult;
import com.hytalerlbridge.geometry.GeometryCell;
import com.hytalerlbridge.geometry.GeometryContact;
import com.hytalerlbridge.geometry.GeometryContract;
import com.hytalerlbridge.geometry.GeometryFrame;
import com.hytalerlbridge.observation.AudioFrame;
import com.hytalerlbridge.observation.MovementStateFrame;
import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import com.hytalerlbridge.observation.NativeActorEvidenceRequest;
import com.hytalerlbridge.observation.NativeInventoryFrame;
import com.hytalerlbridge.observation.Observation;
import com.hytalerlbridge.observation.ObservationEncoding;
import com.hytalerlbridge.imitation.NpcAttackActionSnapshot;
import com.hytalerlbridge.imitation.NpcObservationSnapshot;
import com.hytalerlbridge.imitation.NpcTraceBatch;
import com.hytalerlbridge.nativebackend.support.CombatSpawnSupport;
import com.hytalerlbridge.worldgen.ChunkApiContract;
import com.hytalerlbridge.worldgen.BlockAffordanceContract;
import com.hytalerlbridge.worldgen.NativeDoorTransitionEvidence;
import com.hytalerlbridge.worldgen.NativeExplosionCandidateProbe;
import com.hytalerlbridge.worldgen.NativeExplosionMutationProbe;
import com.hytalerlbridge.worldgen.NativeExplosionDynamicsProbe;
import com.hytalerlbridge.worldgen.NativeCraftingCatalogEvidence;
import com.hytalerlbridge.worldgen.NativeDropProgramEvidence;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence;
import com.hytalerlbridge.worldgen.NativeDoorTransitionProbe;
import com.hytalerlbridge.worldgen.NativePerceptionChannels;
import com.hytalerlbridge.worldgen.NativeLineOfSightEvidence;
import com.hytalerlbridge.worldgen.NativeLineOfSightProbe;
import com.hytalerlbridge.worldgen.NativeMutableBlockCells;
import com.hytalerlbridge.worldgen.NativeMutableBlockEvidence;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerQuery;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerRow;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerSnapshot;
import com.hytalerlbridge.worldgen.NativeWorldActionCapabilities;
import com.hytalerlbridge.worldgen.NativeRegionManifest;
import com.hytalerlbridge.worldgen.NativeRegionBlockSemanticSection;
import com.hytalerlbridge.worldgen.NativeRegionFluidSemanticSection;
import com.hytalerlbridge.worldgen.NativeRegionLightSection;
import com.hytalerlbridge.worldgen.NativeRegionSection;
import com.hytalerlbridge.worldgen.NativeTraversalProbe;
import com.hytalerlbridge.worldgen.NativeTraversalEdgeProbe;
import com.hytalerlbridge.worldgen.NativeNavigationPathProbe;
import com.hytalerlbridge.worldgen.NativeNavigationSuccessorProbe;
import com.hytalerlbridge.worldgen.RegionCellPaletteEntry;
import com.hytalerlbridge.worldgen.RegionShapePaletteEntry;
import com.hytalerlbridge.worldgen.policyactions.capture.PolicyWorldActionCaptureRequest;
import com.hytalerlbridge.worldgen.policyactions.capture.model.NativePolicyWorldActionCapture;
import com.hytalerlbridge.worldgen.policyactions.capture.semantic.PolicyWorldActionSemanticHash;
import com.hytalerlbridge.nativebackend.worldactions.capture.provider.NativePolicyWorldActionCommitVerifier;
import com.hytalerlbridge.nativebackend.worldactions.capture.provider.NativePolicyWorldActionCaptureProducer;
import com.hytalerlbridge.nativebackend.worldactions.capture.provider.PolicyActorCaptureRoute;
import com.hytalerlbridge.nativebackend.worldactions.execution.PolicyWorldVerbArbitrator;
import it.unimi.dsi.fastutil.Pair;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.bson.BsonDocument;
import org.bson.BsonString;
import org.joml.Vector3d;
import org.joml.Vector3i;
import com.hytalerlbridge.nativebackend.model.CapturedActorEvidence;
import com.hytalerlbridge.nativebackend.session.info.EngineTimingInfo;
import com.hytalerlbridge.nativebackend.session.info.GeometryInfo;
import com.hytalerlbridge.nativebackend.session.info.SyntheticClientInfo;
import com.hytalerlbridge.nativebackend.session.navigation.NavigationProbeCapture;
import com.hytalerlbridge.nativebackend.session.navigation.NavigationProbeContext;
import com.hytalerlbridge.nativebackend.session.region.RegionGeometryCapture;
import com.hytalerlbridge.nativebackend.model.CapturedRoleOpacity;
import com.hytalerlbridge.nativebackend.model.CapturedStatuses;
import com.hytalerlbridge.nativebackend.model.CheckedRunnable;
import com.hytalerlbridge.nativebackend.model.DoubleArrayKey;
import com.hytalerlbridge.nativebackend.model.EntityModelEvidence;
import com.hytalerlbridge.nativebackend.model.EntityReading;
import com.hytalerlbridge.nativebackend.model.GeometryChunkCoverage;
import com.hytalerlbridge.nativebackend.model.NativeArmorEntry;
import com.hytalerlbridge.nativebackend.model.NativeCellSemantics;
import com.hytalerlbridge.nativebackend.model.NativeInteractionBinding;
import com.hytalerlbridge.nativebackend.model.NativeSnapshot;
import com.hytalerlbridge.nativebackend.policy.combat.NativeSyntheticCombatClient;
import com.hytalerlbridge.nativebackend.policy.combat.NativePolicyCombatFacade;
import com.hytalerlbridge.nativebackend.policy.combat.NativeGuardFork;
import com.hytalerlbridge.nativebackend.model.RegionBlockSemanticKey;
import com.hytalerlbridge.nativebackend.model.RegionCellKey;
import com.hytalerlbridge.nativebackend.model.ResolvedDropRoute;
import com.hytalerlbridge.nativebackend.model.TargetState;
import com.hytalerlbridge.nativebackend.model.WorldVerbCell;
import com.hytalerlbridge.nativebackend.fixture.TargetMemoryEvaluationFixture;
import com.hytalerlbridge.nativebackend.group.NativeAttackAdmission;
import com.hytalerlbridge.nativebackend.group.NativePolicyActorState;
import com.hytalerlbridge.nativebackend.support.RegionLightCapture;
import com.hytalerlbridge.nativebackend.support.RegionLightCoverage;
import com.hytalerlbridge.nativebackend.support.NativeMotionForceCapture;
import com.hytalerlbridge.nativebackend.support.NativeCellSemanticsCache;
import com.hytalerlbridge.nativebackend.support.PrivilegedNpcGeometryCapture;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.absentActor;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.captureActorEvidence;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.activeItemId;
import static com.hytalerlbridge.nativebackend.support.GeometryContacts.appendTouchingGeometryContacts;
import static com.hytalerlbridge.nativebackend.support.GeometryContacts.blocksDefaultNpcLineOfSight;
import static com.hytalerlbridge.nativebackend.support.DropRouting.canonicalDropOutcomeKey;
import static com.hytalerlbridge.nativebackend.support.DropRouting.canonicalDropStack;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.canonicalEntityType;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.canonicalWorldVerbInventoryState;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.captureEntityModelEvidence;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.captureMovementStates;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.captureRoleOpacity;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.captureStatuses;
import static com.hytalerlbridge.nativebackend.support.SupportMath.clamp;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.clearMarkedTargets;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.clearNativeCombat;
import static com.hytalerlbridge.nativebackend.support.GeometryContacts.collisionBoxes;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.discoverAttackActions;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.hasUnfinishedInteractionTree;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.resolvedWorldProbeRoot;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.entityHealth;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.implementationVersion;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.interactionManager;
import static com.hytalerlbridge.nativebackend.support.GeometryContacts.isDiagonalLosFixture;
import static com.hytalerlbridge.nativebackend.support.GeometryContacts.lineOfSightOffset;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.mutableBlockAssetKey;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.mutableBlockSemanticKey;
import static com.hytalerlbridge.nativebackend.support.SupportMath.navigationSearchDirections;
import static com.hytalerlbridge.nativebackend.support.SupportMath.normalizeDegrees;
import static com.hytalerlbridge.nativebackend.support.SupportMath.normalizeRadians;
import static com.hytalerlbridge.nativebackend.support.SupportMath.requireFinite;
import static com.hytalerlbridge.nativebackend.support.SupportMath.requireNavigationCapacity;
import static com.hytalerlbridge.nativebackend.support.DropRouting.resolveDropRoute;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.resolveNativeInteractionBinding;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.resolveNativeAbilityBindings;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.resolvedWorldProbeRoot;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.sameEntity;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.sha256Hex;
import static com.hytalerlbridge.nativebackend.support.NativeWorldVerbSupport.requestedUnsupportedWorldVerbs;
import static com.hytalerlbridge.nativebackend.support.NativeWorldVerbSupport.requiredHotbarCapacity;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.stat;
import com.hytalerlbridge.nativebackend.support.DropRouting;
import com.hytalerlbridge.nativebackend.support.ExplosionAdmissionProbe;
import com.hytalerlbridge.nativebackend.support.ExplosionMutationCapture;
import com.hytalerlbridge.nativebackend.support.ExplosionDynamicsCapture;
import com.hytalerlbridge.nativebackend.support.MutableBlockEpochGuard;

/** One isolated, server-authoritative Hytale 0.5.7 headless environment. */
final class NativeEnvironmentSession
    implements
        EnvironmentSession,
        NativeRegionSource,
        NativePerceptionChannelSource,
        NativeDoorTransitionSource,
        NativeLineOfSightSource,
        NativeMutableBlockSource,
        NativeDropProgramSource,
        NativeCraftingCatalogSource,
        NativePolicyWorldActionCaptureSource,
        NativeItemInteractionSource,
        NativeExplosionDynamicsSource,
        NativeExplosionMutationSource,
        NativeExplosionProbeSource,
        NativeTraversalProbeSource,
        NativePrivilegedEntitySource,
        NativeNpcTraceSource,
        NativeWorldgenStructureMarkerSource {

    private static final System.Logger LOGGER =
        System.getLogger(NativeEnvironmentSession.class.getName());
    private static final Duration INITIALIZATION_TIMEOUT = Duration.ofSeconds(30);
    private static final CombatRuleset COMBAT_RULES =
        HytaleCombatAssets.RULESET;
    private static final int WARMUP_TICKS = 16;
    private static final int BLOCK_RADIUS = GeometryContract.RADIUS;
    private static final int INVENTORY_SIZE = 36;
    private static final double NAVIGATE_SUCCESS_RADIUS = 1.5;
    private static final double NEARBY_ENTITY_RADIUS =
        COMBAT_RULES.target().sensorRange();
    private static final boolean NATIVE_CELL_SEMANTICS_CACHE =
        Boolean.parseBoolean(System.getProperty(
            "hytalerl.native.cell_semantics_cache",
            "true"
        ));
    private static final boolean TRACE_PERCEPTION_CANDIDATE_CACHE =
        Boolean.parseBoolean(System.getProperty(
            "hytalerl.trace.perception_candidate_cache",
            "true"
        ));
    private static final boolean TRACE_PERCEPTION_WORLDVIEW_REUSE =
        Boolean.parseBoolean(System.getProperty(
            "hytalerl.trace.perception_worldview_reuse",
            "true"
        ));
    private static final boolean TRACE_PERCEPTION_PROJECTILES =
        Boolean.parseBoolean(System.getProperty(
            "hytalerl.trace.perception_projectiles",
            "true"
        ));
    private static final Query<EntityStore> TRACE_NPC_PERCEPTION_QUERY =
        Query.and(
            UUIDComponent.getComponentType(),
            TransformComponent.getComponentType(),
            NPCEntity.getComponentType()
        );
    private static final Query<EntityStore> TRACE_PERCEPTION_QUERY = Query.and(
        TransformComponent.getComponentType(),
        Query.or(
            NPCEntity.getComponentType(),
            ProjectileComponent.getComponentType(),
            com.hypixel.hytale.server.core.modules.projectile.component.Projectile
                .getComponentType(),
            PredictedProjectile.getComponentType()
        )
    );
    private static final int MAX_NEARBY_ENTITIES = 20;
    private static final double AGENT_STEERING_RELATIVE_TURN_SPEED =
        COMBAT_RULES.agent().steeringRelativeTurnSpeed();
    private static final double AGENT_JUMP_VELOCITY_GRAVITY_FLOOR =
        COMBAT_RULES.agent().jumpVelocityGravityFloor();
    private static final double AGENT_JUMP_HEIGHT_PARAMETER =
        COMBAT_RULES.agent().jumpHeightParameter();
    private static final String KILL_TRORK_TARGET_ROLE =
        COMBAT_RULES.matchup().targetRole();
    private static final String NATIVE_PROJECTILE_TARGET_ROLE =
        COMBAT_RULES.matchup().agentRole();
    private static final String SYNTHETIC_PLACE_INTERACTION_ID =
        "Block_Secondary";
    private static final String SYNTHETIC_BREAK_INTERACTION_ID =
        "Pickaxe_Attack";
    private static final String SYNTHETIC_PLACE_ITEM_ID =
        "Rock_Stone";
    private static final String SYNTHETIC_BREAK_TOOL_ITEM_ID =
        "Tool_Pickaxe_Iron";
    private static final String SYNTHETIC_PLACE_BLOCK_ID =
        "Rock_Stone";
    private static final String SYNTHETIC_BLOCK_USE_BLOCK_ID =
        "Deco_Lantern";
    private static final String FALLING_BLOCK_FIXTURE_ASSET_ID =
        "Debug_Falling_Barrel";
    private static final String FALLING_BLOCK_FIXTURE_IMPACT_TYPE =
        "Break";
    /*
     * NPCInteractionSimulationHandler represents a held Charging/Wielding
     * interaction as `time < requestedChargeTime`. Wielding sets
     * allowIndefiniteHold, so +infinity is the server-native NPC equivalent
     * of a held input. The release edge changes both the NPC simulation
     * duration and the synthetic client cursor to a finished state so the
     * authored Wielding.Next chain still executes.
     */
    private static final double NATIVE_INDEFINITE_HOLD_SECONDS =
        Double.POSITIVE_INFINITY;
    private static final double KILL_TRORK_TARGET_OFFSET_X =
        COMBAT_RULES.fixture().targetOffset()[0];
    private static final double KILL_TRORK_TARGET_OFFSET_Y =
        COMBAT_RULES.fixture().targetOffset()[1];
    private static final double KILL_TRORK_TARGET_OFFSET_Z =
        COMBAT_RULES.fixture().targetOffset()[2];
    private static final double TARGET_MEMORY_TARGET_OFFSET_Z = -10.0;
    private static final double PATH_FOLLOWER_TARGET_OFFSET_Z = -7.0;
    private static final double NATIVE_DUEL_TARGET_OFFSET_Z = -10.0;
    private static final double NATIVE_PROJECTILE_TARGET_OFFSET_Z = 10.0;
    private static final int PATH_FOLLOWER_WALL_Z = -3;
    private static final int PATH_FOLLOWER_WALL_HALF_WIDTH = 2;
    private static final int DIVE_POOL_HALF_WIDTH = 8;
    private static final int DIVE_POOL_MIN_Y = 65;
    private static final int DIVE_POOL_MAX_Y = 75;
    private static final double DIVE_POOL_SPAWN_Y = 69.0;
    // These controlled fixtures cross the X/Z cell corner at (1, 0). One
    // block is only grazed; the other occupies the tied-axis destination.
    private static final double DIAGONAL_LOS_TARGET_OFFSET_X = 2.0;
    private static final double DIAGONAL_LOS_TARGET_OFFSET_Z = -2.0;
    private static final VelocityConfig AGENT_VELOCITY_CONFIG = new VelocityConfig();
    private static final String NATIVE_SERVER_VERSION = implementationVersion();
    private static final AtomicInteger HYTALE_GENERATOR_WORLD_COUNTER =
        new AtomicInteger(Integer.MIN_VALUE);

    private record TraceLightSection(int chunkX, int chunkZ, int sectionY) {}

    private final HytaleNativeBackendProvider provider;
    private final ComponentType<EntityStore, NativeAgentMarker> markerType;
    private final NativeNpcTraceRecorder npcTraceRecorder;
    private final String taskId;
    private final long seed;
    private final int curriculumPhase;
    private final int ticksPerStep;
    private final int maxEpisodeSteps;
    private final EnvironmentOptions options;
    private final Map<String, Integer> itemIds;
    private final UUID reservationId = UUID.randomUUID();
    private final AtomicReference<PendingStep> pending = new AtomicReference<>();
    private final NativePolicyActorState[] policyActorStates =
        createPolicyActorStates();
    private final String[] policyActorIdentities = new String[
        NativeActorEvidenceFrame.ENTITY_COUNT
    ];
    private final Object[] nativePolicyCombatHandles = new Object[
        NativeActorEvidenceFrame.ENTITY_COUNT
    ];
    private final NativePolicyCombatBindingSpec[] nativePolicyCombatSpecs =
        new NativePolicyCombatBindingSpec[
            NativeActorEvidenceFrame.ENTITY_COUNT
        ];
    private final int[] nativePolicyCombatActiveAbilitySlots =
        createInactivePolicyCombatSlots();
    private final List<Ref<ChunkStore>> loadedChunks = new ArrayList<>();
    private final Set<Long> loadedChunkIndices = new HashSet<>();
    private final Set<Long> geometryChunkIndices = new HashSet<>();
    private final Map<TraceLightSection, Long> traceLightRequestTicks =
        new HashMap<>();
    private final NativeCellSemanticsCache nativeCellSemanticsCache =
        new NativeCellSemanticsCache();
    private List<Ref<EntityStore>> tracePerceptionCandidates = List.of();
    private long tracePerceptionCandidateTick = Long.MIN_VALUE;
    private final TargetMemoryEvaluationFixture targetMemoryFixture;

    private volatile World world;
    private volatile Ref<EntityStore> agentRef;
    private volatile Ref<EntityStore> targetRef;
    private volatile PlayerRef tickAnchor;
    private volatile PlayerRef syntheticPlayerRef;
    private final PlayerRef[] syntheticPlayerRefsByActor = new PlayerRef[
        NativeGroupActionContract.actorCapacity()
    ];
    private volatile NativeSnapshot lastSnapshot;
    private volatile StepResult lastResult = StepResult.initial(Observation.empty());
    private String worldgenProvider = "uninitialized";
    private String worldgenVersion = "uninitialized";
    private String worldgenWorldStructure = "";
    private String worldName;
    private NativeRegionManifest nativeRegionManifest;
    private NativeCraftingCatalogEvidence policyWorldActionRecipeCatalog;
    private String policyWorldActionRecipeTableIdentity = "";
    private NativeRegionManifest regionLightHaloManifest;
    private double spawnX;
    private double spawnY;
    private double spawnZ;
    private boolean attached;
    private boolean initialized;
    private boolean closed;
    private boolean done;
    private boolean nativeDuelAutonomous;
    private String nativeActorTargetAttitude = "";
    private String nativeTargetActorAttitude = "";
    private boolean nativeMutualHostility;
    private int episodeEngineTicks;
    private int episodeSteps;
    private double targetX;
    private double targetY;
    private double targetZ;
    private double previousTargetDistance;
    private double previousHealth;
    private double previousTargetHealth;
    private RootInteraction nativeCombatRootInteraction;
    private InteractionType nativeCombatInteractionType;
    private List<NativeInteractionBinding> nativeCombatAbilities = List.of();
    private NativeInteractionBinding nativeGuardInteraction;
    private NativeInteractionBinding nativeDodgeLeftInteraction;
    private NativeInteractionBinding nativeDodgeRightInteraction;
    private RootInteraction syntheticPlaceRoot;
    private RootInteraction syntheticBreakRoot;
    private int syntheticWorldFixtureBlockId;
    private int syntheticWorldFixturePlaceSourceQuantity;
    private int syntheticWorldFixtureBreakSourceQuantity;
    private Vector3i syntheticWorldFixturePlaceTarget;
    private Vector3i syntheticWorldFixtureBreakTarget;
    private InteractionChain syntheticWorldChain;
    private SyntheticWorldInteractionEvidence.Row syntheticWorldEvidenceRow;
    private String syntheticWorldVerb = "";
    private Vector3i syntheticWorldTarget;
    private int syntheticWorldBlockBefore;
    private double syntheticWorldHealthBefore;
    private int syntheticWorldExpectedBlockId;
    private BlockFace syntheticWorldBlockFace = BlockFace.Up;
    private Rotation syntheticWorldRotationYaw = Rotation.None;
    private Rotation syntheticWorldRotationPitch = Rotation.None;
    private Rotation syntheticWorldRotationRoll = Rotation.None;
    private int syntheticWorldOperationIndex = -1;
    private double syntheticWorldClientElapsedSeconds;
    private boolean syntheticWorldStarted;
    private final NativeSyntheticCombatClient syntheticCombatClient =
        new NativeSyntheticCombatClient();
    private Ref<EntityStore> fallingBlockRef;
    private long fallingBlockSampleIndex = -1L;
    private NativeFallingBlockTrace fallingBlockTrace;
    private InteractionChain nativeGuardChain;
    private boolean nativeGuardActive;
    private final NativeGuardLifecycle nativeGuardLifecycle =
        new NativeGuardLifecycle();
    private long trackedNativeGuardInteractionGeneration = -1L;
    private String trackedNativeGuardInteractionId = "";
    private long trackedNativeGuardStartTick = -1L;
    private long trackedNativeGuardFinishTick = -1L;
    private NativeGuardFork.Request pendingAgentAbilityFork;
    private InteractionChain trackedAgentAbilityChain;
    private InteractionChain trackedAgentAbilityParentChain;
    private int trackedAgentAbilitySlot = -1;
    private String trackedAgentAbilityId = "";
    private String trackedAgentAbilityType = "";
    private long trackedAgentAbilityStartTick = -1L;
    private long trackedAgentAbilityFinishTick = -1L;
    private InteractionChain trackedAgentDodgeChain;
    private int trackedAgentDodgeDirection;
    private String trackedAgentDodgeInteractionId = "";
    private long trackedAgentDodgeStartTick = -1L;
    private long trackedAgentDodgeFinishTick = -1L;
    private InteractionChain trackedAgentUseChain;
    private String trackedAgentUseInteractionId = "";
    private String trackedAgentUseBlockInteractionId = "";
    private BlockPosition trackedAgentUseTarget;
    private double trackedAgentUseMaximumDistance;
    private long trackedAgentUseStartTick = -1L;
    private long trackedAgentUseFinishTick = -1L;
    // Actor-zero compatibility state for the retired scalar typed-verb path.
    // The authoritative policy path is actor-major below.
    private InteractionChain legacyActorZeroTypedWorldVerbChain;
    private CraftingManager legacyActorZeroTypedWorldVerbCraftingManager;
    private CraftingRecipe legacyActorZeroTypedWorldVerbCraftingRecipe;
    private WorldVerbTelemetry.Execution legacyActorZeroTypedWorldVerbExecution =
        WorldVerbTelemetry.Execution.unrequested();
    private final Object[] nativePolicyWorldVerbHandlesByActor = new Object[
        NativeGroupActionContract.actorCapacity()
    ];
    private final NativeWorldVerbRequest[] nativePolicyWorldVerbRequestsByActor =
        new NativeWorldVerbRequest[
            NativeGroupActionContract.actorCapacity()
        ];
    private String legacyActorZeroTypedWorldVerbGeometryBefore = "";
    private String legacyActorZeroTypedWorldVerbInventoryBefore = "";
    private boolean nativeUseAvailable;
    private String nativeUseAvailableInteractionId = "";
    private String nativeUseAvailableBlockInteractionId = "";
    private String nativeUseAvailableItemId = "";
    private String nativeUseAvailableSourceContainer = "";
    private int nativeUseAvailableSourceSlot = -1;
    private int nativeUseAvailableSourceQuantity = -1;
    private BlockPosition nativeUseAvailableTarget;
    private double nativeUseAvailableMaximumDistance;
    private String nativeUseUnavailableReason = "not_sampled";
    private List<NativeArmorEntry> nativeAgentArmorOverride;
    private InteractionChain trackedTargetAttackChain;
    private CombatPhase targetAttackPhase = CombatPhase.IDLE;
    private double targetAttackProgress;
    private int targetAttackIndex = -1;
    private int targetAttackElapsedTicks;
    private String targetAttackId = "";
    private NativePathFollowerTrace targetNavigationTrace;
    private long geometryChunkHaloCenter = Long.MIN_VALUE;

    NativeEnvironmentSession(
        HytaleNativeBackendProvider provider,
        ComponentType<EntityStore, NativeAgentMarker> markerType,
        ComponentType<EntityStore, NativeNpcTraceMarker> traceMarkerType,
        String taskId,
        long seed,
        int curriculumPhase,
        int ticksPerStep,
        int maxEpisodeSteps,
        EnvironmentOptions options,
        Map<String, Integer> itemIds
    ) {
        this.provider = provider;
        this.markerType = markerType;
        this.npcTraceRecorder = new NativeNpcTraceRecorder(
            traceMarkerType,
            this::captureTraceObservation,
            options.fidelityFixture().equals(
                EnvironmentOptions.NATIVE_PROJECTILE_DUEL_FIXTURE
            )
        );
        this.taskId = taskId;
        this.seed = seed;
        this.curriculumPhase = curriculumPhase;
        this.ticksPerStep = ticksPerStep;
        this.maxEpisodeSteps = maxEpisodeSteps;
        this.options = options;
        this.itemIds = itemIds;
        this.fallingBlockTrace = NativeFallingBlockTrace.unavailable(
            options.fidelityFixture().equals(
                EnvironmentOptions.FALLING_BLOCK_FIXTURE
            )
        );
        if (options.nativeNavigationTrace()) {
            NativePathFollowerTrace.requireSupported();
            this.targetNavigationTrace = NativePathFollowerTrace.pending();
        }
        if (
            options.fidelityFixture().equals(
                EnvironmentOptions.DIVE_MOTION_FIXTURE
            )
        ) {
            NativeDiveTrace.requireSupported();
        }
        double[] fixtureSpawn = COMBAT_RULES.fixture().agentSpawn();
        this.spawnX = options.hasSpawn() ? options.spawnX() : fixtureSpawn[0];
        this.spawnY = options.hasSpawn()
            ? options.spawnY()
            : COMBAT_RULES.fixture().floorY() + 1.0;
        this.spawnZ = options.hasSpawn() ? options.spawnZ() : fixtureSpawn[2];
        if (
            !options.hasSpawn()
                && options.fidelityFixture().equals(
                    EnvironmentOptions.DIVE_MOTION_FIXTURE
                )
        ) {
            this.spawnY = DIVE_POOL_SPAWN_Y;
        }
        this.targetMemoryFixture = options.fidelityFixture().equals(
            EnvironmentOptions.TARGET_MEMORY_FIXTURE
        )
            ? new TargetMemoryEvaluationFixture(
                (int) Math.floor(spawnX),
                (int) Math.floor(spawnY),
                (int) Math.floor(spawnZ - 5.0)
            )
            : null;
    }

    @Override
    public synchronized void reset() {
        if (closed) throw new IllegalStateException("Native environment is closed");
        if (initialized) throw new IllegalStateException("Native environment was already reset");
        if (options.hasCombatTargetRole() && !taskId.equals("kill_trork")) {
            throw new IllegalArgumentException(
                "combat_target_role requires the kill_trork task"
            );
        }

        NPCPlugin npcPlugin = NPCPlugin.get();
        if (npcPlugin == null) {
            throw new IllegalStateException("Hytale NPC plugin is not available");
        }
        if (!npcPlugin.hasRoleName(options.npcRole())) {
            throw new IllegalArgumentException("Native NPC role not found: " + options.npcRole());
        }
        npcPlugin.validateSpawnableRole(options.npcRole());
        if (taskId.equals("kill_trork") && options.hasCombatTargetRole()) {
            if (!npcPlugin.hasRoleName(options.combatTargetRole())) {
                throw new IllegalArgumentException(
                    "Native target role not found: " + options.combatTargetRole()
                );
            }
            npcPlugin.validateSpawnableRole(options.combatTargetRole());
        }
        resolveNativeCombatInteraction();
        resolveSyntheticWorldInteractions();
        resolveNativeAgentArmorOverride();

        createWorld();
        pauseWorld();
        installHeadlessTickAnchor();
        loadAndPinSpawnChunks();
        resolveGeneratedSpawnHeight();
        createFidelityFixture();
        spawnAgent(npcPlugin);
        provider.attach(world, this);
        attached = true;

        advance(AgentAction.noop(), WARMUP_TICKS, false);
        if (syntheticWorldInteractionsEnabled()) {
            bindSyntheticWorldFixture();
            advance(AgentAction.noop(), 1, false);
        }
        if (syntheticBlockUseFixtureEnabled()) {
            bindSyntheticBlockUseFixture();
            advance(AgentAction.noop(), 1, false);
        }
        if (taskId.equals("kill_trork")) {
            spawnTaskEntities(npcPlugin);
            advance(AgentAction.noop(), 1, false);
        }
        boolean itemStatsSettled = options.hasNativeCombatItem()
            || options.hasNativePolicyCombatBindings();
        if (itemStatsSettled) {
            settleNativeCombatItemStats();
        }
        if (options.hasNativeFidelityResourceOverrides()) {
            applyNativeFidelityResourceOverrides();
        }
        if (itemStatsSettled
            || options.hasNativeFidelityResourceOverrides()) {
            advance(AgentAction.noop(), 1, false);
        }
        NativeSnapshot snapshot = lastSnapshot;
        if (snapshot == null) {
            throw new IllegalStateException("Native warmup completed without an observation");
        }

        initializeTask(snapshot.observation());
        activateTaskEntities();
        spawnFallingBlockFixture();
        episodeEngineTicks = 0;
        episodeSteps = 0;
        done = false;
        initialized = true;
        lastResult = new StepResult(
            snapshot.observation().withAudio(AudioFrame.headlessPartial(List.of())),
            0.0,
            false,
            false,
            0,
            0,
            buildResetInfo(snapshot)
        );
    }

    @Override
    public StepResult observe() {
        return lastResult;
    }

    @Override
    public synchronized NativeRegionManifest regionManifest() {
        requireWorldCaptureReady();
        int centerX = ChunkUtil.chunkCoordinate(spawnX);
        int centerZ = ChunkUtil.chunkCoordinate(spawnZ);
        return regionManifest(centerX - 1, centerZ - 1);
    }

    @Override
    public synchronized NativeRegionManifest regionManifest(
        int coreMinChunkX,
        int coreMinChunkZ
    ) {
        requireWorldCaptureReady();
        if (
            !options.fidelityFixture().equals(
                EnvironmentOptions.STATIC_REGION_FIXTURE
            )
        ) {
            throw new IllegalStateException(
                "Region v1 requires fidelity_fixture=static_region"
            );
        }
        NativeRegionManifest cached = nativeRegionManifest;
        if (
            cached != null
                && cached.coreMinChunkX() == coreMinChunkX
                && cached.coreMinChunkZ() == coreMinChunkZ
        ) {
            return cached;
        }

        ChunkApiContract chunkApi = ChunkApiContract.load057();
        cached = new NativeRegionManifest(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            coreMinChunkX,
            coreMinChunkZ,
            chunkApi.manifest(NATIVE_SERVER_VERSION)
        );
        loadAndPinRegion(cached);
        nativeRegionManifest = cached;
        return cached;
    }

    @Override
    public synchronized NativeRegionSection captureRegionSection(
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        requireRegionSectionRequest(chunkX, chunkZ, sectionY);
        AtomicReference<NativeRegionSection> captured = new AtomicReference<>();
        runOnWorld(
            () -> captured.set(RegionGeometryCapture.buildRegionSection(nativeCellSemanticsCache, world, loadedChunkIndices, chunkX, chunkZ, sectionY)),
            INITIALIZATION_TIMEOUT
        );
        NativeRegionSection result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no region section"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeRegionBlockSemanticSection
        captureRegionBlockSemanticSection(
            int chunkX,
            int chunkZ,
            int sectionY
        ) {
        requireRegionSectionRequest(chunkX, chunkZ, sectionY);
        AtomicReference<NativeRegionBlockSemanticSection> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(
                RegionGeometryCapture.buildRegionBlockSemanticSection(world, loadedChunkIndices, 
                    chunkX,
                    chunkZ,
                    sectionY
                )
            ),
            INITIALIZATION_TIMEOUT
        );
        NativeRegionBlockSemanticSection result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no Region block semantics"
            );
        }
        return result;
    }

    private void requireRegionSectionRequest(
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        NativeRegionManifest manifest = nativeRegionManifest;
        if (manifest == null) manifest = regionManifest();
        int minimumX = manifest.captureMinChunkX();
        int minimumZ = manifest.captureMinChunkZ();
        int maximumX = minimumX
            + NativeRegionManifest.CAPTURE_CHUNKS_PER_AXIS;
        int maximumZ = minimumZ
            + NativeRegionManifest.CAPTURE_CHUNKS_PER_AXIS;
        if (
            chunkX < minimumX
                || chunkX >= maximumX
                || chunkZ < minimumZ
                || chunkZ >= maximumZ
        ) {
            throw new IllegalArgumentException(
                "Requested section lies outside the Region v1 capture halo"
            );
        }
        int heightSections = ((Number) manifest.chunkApi().get(
            "height_sections"
        )).intValue();
        if (sectionY < 0 || sectionY >= heightSections) {
            throw new IllegalArgumentException(
                "Requested section Y lies outside the Hytale world"
            );
        }
    }

    @Override
    public synchronized NativePerceptionChannels capturePerceptionChannels(
        int[] requestedPositions
    ) {
        requireWorldCaptureReady();
        int[] positions = requestedPositions == null
            ? new int[0]
            : requestedPositions.clone();
        if (
            positions.length == 0
                || positions.length % 3 != 0
                || positions.length / 3 > NativePerceptionChannels.MAX_SAMPLES
        ) {
            throw new IllegalArgumentException(
                "Perception positions must contain 1.."
                    + NativePerceptionChannels.MAX_SAMPLES
                    + " absolute XYZ triples"
            );
        }
        AtomicReference<NativePerceptionChannels> captured = new AtomicReference<>();
        runOnWorld(
            () -> captured.set(buildPerceptionChannels(positions)),
            INITIALIZATION_TIMEOUT
        );
        NativePerceptionChannels result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no perception channels"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeLineOfSightEvidence captureLineOfSightEvidence() {
        requireWorldCaptureReady();
        AtomicReference<NativeLineOfSightEvidence> captured =
            new AtomicReference<>();
        runOnWorld(() -> {
            Ref<EntityStore> agent = agentRef;
            Ref<EntityStore> target = targetRef;
            Store<EntityStore> store = world.getEntityStore().getStore();
            NPCEntity npc = agent == null
                ? null
                : store.getComponent(agent, NPCEntity.getComponentType());
            Role role = npc == null ? null : npc.getRole();
            captured.set(
                NativeLineOfSightProbe.capture(
                    world,
                    store,
                    agent,
                    target,
                    role,
                    NATIVE_SERVER_VERSION,
                    worldName,
                    worldgenProvider,
                    worldgenVersion,
                    seed
                )
            );
        }, INITIALIZATION_TIMEOUT);
        NativeLineOfSightEvidence result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no LOS evidence"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeMutableBlockEvidence
        captureMutableBlockEvidence() {
        requireWorldCaptureReady();
        if (!options.world().equals("flat")) {
            throw new IllegalStateException(
                "Mutable block evidence requires the isolated flat fixture"
            );
        }
        AtomicReference<NativeMutableBlockEvidence> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(buildMutableBlockEvidence()),
            INITIALIZATION_TIMEOUT
        );
        NativeMutableBlockEvidence result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no mutable block evidence"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeRegionFluidSemanticSection
        captureRegionFluidSemanticSection(
            int chunkX,
            int chunkZ,
            int sectionY
        ) {
        requireRegionSectionRequest(chunkX, chunkZ, sectionY);
        AtomicReference<NativeRegionFluidSemanticSection> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(
                RegionGeometryCapture.buildRegionFluidSemanticSection(world, loadedChunkIndices, chunkX, chunkZ, sectionY)
            ),
            INITIALIZATION_TIMEOUT
        );
        NativeRegionFluidSemanticSection result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no Region fluid semantics"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeRegionLightSection captureRegionLightSection(
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        requireRegionSectionRequest(chunkX, chunkZ, sectionY);
        loadAndPinRegionLightHalo(nativeRegionManifest);
        AtomicReference<NativeRegionLightSection> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(
                buildRegionLightSection(chunkX, chunkZ, sectionY)
            ),
            INITIALIZATION_TIMEOUT
        );
        NativeRegionLightSection result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no Region light section"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeMutableBlockCells captureMutableBlockCells(
        int[] requestedPositions
    ) {
        return captureMutableBlockCells(requestedPositions, null);
    }

    @Override
    public synchronized NativeMutableBlockCells captureMutableBlockCells(
        int[] requestedPositions,
        String expectedWorldEpoch
    ) {
        requireWorldCaptureReady();
        MutableBlockEpochGuard.require(
            reservationId,
            expectedWorldEpoch,
            "before scheduling"
        );
        int[] positions = requestedPositions == null
            ? new int[0]
            : requestedPositions.clone();
        if (
            positions.length == 0
                || positions.length % 3 != 0
                || positions.length / 3
                    > NativeMutableBlockCells.MAX_SAMPLES
        ) {
            throw new IllegalArgumentException(
                "Mutable block positions must contain 1.."
                    + NativeMutableBlockCells.MAX_SAMPLES
                    + " absolute XYZ triples"
            );
        }
        AtomicReference<NativeMutableBlockCells> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> {
                MutableBlockEpochGuard.require(
                    reservationId,
                    expectedWorldEpoch,
                    "on world thread"
                );
                captured.set(buildMutableBlockCells(positions));
            },
            INITIALIZATION_TIMEOUT
        );
        NativeMutableBlockCells result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no mutable block cells"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeExplosionMutationProbe captureExplosionMutation(
        String worldEpoch,
        String fixtureKind,
        int cellCapacity,
        int dropCapacity
    ) {
        requireWorldCaptureReady();
        if (!options.world().equals("flat")) {
            throw new IllegalStateException(
                "Explosion mutation requires the isolated flat fixture"
            );
        }
        String currentWorldEpoch = reservationId.toString();
        if (!currentWorldEpoch.equals(worldEpoch)) {
            throw new IllegalArgumentException(
                "Explosion mutation world_epoch is stale"
            );
        }

        AtomicReference<NativeExplosionMutationProbe> captured =
            new AtomicReference<>();
        runOnWorld(() -> {
            if (!reservationId.toString().equals(worldEpoch)) {
                throw new IllegalArgumentException(
                    "Explosion mutation world_epoch changed before execution"
                );
            }
            BlockHealthModule healthModule = BlockHealthModule.get();
            if (healthModule == null) {
                throw new IllegalStateException(
                    "Explosion mutation has no BlockHealthModule"
                );
            }
            Store<ChunkStore> chunkStore = world.getChunkStore().getStore();
            Store<EntityStore> entityStore = world.getEntityStore().getStore();
            WorldTimeResource time = entityStore.getResource(
                WorldTimeResource.getResourceType()
            );
            if (time == null) {
                throw new IllegalStateException(
                    "Explosion mutation has no game-time resource"
                );
            }

            int originCellX = (int) Math.floor(spawnX) + 4;
            int originCellY = (int) Math.floor(spawnY) + 7;
            int originCellZ = (int) Math.floor(spawnZ);
            ExplosionMutationCapture.CellStateAccess cells =
                new ExplosionMutationCapture.CellStateAccess() {
                    @Override
                    public NativeExplosionMutationProbe.CellState snapshot(
                        int x,
                        int y,
                        int z
                    ) {
                        return explosionMutationCellState(
                            x,
                            y,
                            z,
                            healthModule,
                            chunkStore,
                            time
                        );
                    }

                    @Override
                    public void clearBlockHealth(int x, int y, int z) {
                        Vector3i position = new Vector3i(x, y, z);
                        BlockHealthChunk health = explosionMutationBlockHealth(
                            x,
                            y,
                            z,
                            healthModule,
                            chunkStore
                        );
                        health.removeBlock(world, position);
                        health.getBlockFragilityMap().remove(position);
                    }
                };
            captured.set(ExplosionMutationCapture.capture(
                world,
                new ExplosionMutationCapture.Context(
                    NATIVE_SERVER_VERSION,
                    worldName,
                    worldgenProvider,
                    worldgenVersion,
                    seed,
                    worldEpoch
                ),
                fixtureKind,
                originCellX + 0.5,
                originCellY + 0.5,
                originCellZ + 0.5,
                cellCapacity,
                dropCapacity,
                cells
            ));
        }, INITIALIZATION_TIMEOUT);
        NativeExplosionMutationProbe result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no explosion mutation"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeExplosionDynamicsProbe captureExplosionDynamics(
        String worldEpoch,
        String fixtureKind,
        int capacity
    ) {
        requireWorldCaptureReady();
        if (!options.world().equals("flat")) {
            throw new IllegalStateException(
                "Explosion dynamics requires the isolated flat fixture"
            );
        }
        if (!NativeExplosionDynamicsProbe.FIXTURE_KIND.equals(fixtureKind)) {
            throw new IllegalArgumentException(
                "Unsupported explosion dynamics fixture: " + fixtureKind
            );
        }
        if (!reservationId.toString().equals(worldEpoch)) {
            throw new IllegalArgumentException(
                "Explosion dynamics world_epoch is stale"
            );
        }
        AtomicReference<NativeExplosionDynamicsProbe> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(ExplosionDynamicsCapture.capture(
                world,
                agentRef,
                targetRef,
                new ExplosionDynamicsCapture.Context(
                    NATIVE_SERVER_VERSION,
                    worldName,
                    worldgenProvider,
                    worldgenVersion,
                    seed,
                    worldEpoch
                ),
                capacity
            )),
            INITIALIZATION_TIMEOUT
        );
        NativeExplosionDynamicsProbe result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no explosion dynamics"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeDropProgramEvidence
        captureDropProgramEvidence(
            String blockAssetId,
            String route,
            int sampleCount
        ) {
        requireWorldCaptureReady();
        AtomicReference<NativeDropProgramEvidence> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(buildDropProgramEvidence(
                blockAssetId,
                route,
                sampleCount
            )),
            INITIALIZATION_TIMEOUT
        );
        NativeDropProgramEvidence result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no drop-program evidence"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeCraftingCatalogEvidence
        captureCraftingCatalogEvidence() {
        requireWorldCaptureReady();
        AtomicReference<NativeCraftingCatalogEvidence> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(NativeCraftingCatalogEvidence.capture(
                NATIVE_SERVER_VERSION,
                worldName,
                worldgenProvider,
                worldgenVersion,
                seed
            )),
            INITIALIZATION_TIMEOUT
        );
        NativeCraftingCatalogEvidence result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no crafting catalog evidence"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeItemInteractionEvidence
        captureItemInteractionEvidence(int equippedSlot) {
        requireWorldCaptureReady();
        AtomicReference<NativeItemInteractionEvidence> captured =
            new AtomicReference<>();
        runOnWorld(() -> {
            Ref<EntityStore> reference = agentRef;
            Store<EntityStore> store = world.getEntityStore().getStore();
            captured.set(NativeItemInteractionEvidence.capture(
                NATIVE_SERVER_VERSION,
                worldName,
                worldgenProvider,
                worldgenVersion,
                seed,
                reference,
                store,
                interactionManager(reference, store),
                equippedSlot
            ));
        }, INITIALIZATION_TIMEOUT);
        NativeItemInteractionEvidence result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no item interaction evidence"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeDoorTransitionEvidence
        captureDoorTransitionEvidence() {
        requireWorldCaptureReady();
        if (!options.world().equals("flat")) {
            throw new IllegalStateException(
                "Native door evidence requires the isolated flat fixture"
            );
        }
        AtomicReference<NativeDoorTransitionEvidence> captured =
            new AtomicReference<>();
        runOnWorld(() -> captured.set(
            NativeDoorTransitionProbe.capture(
                world,
                world.getEntityStore().getStore(),
                agentRef,
                NATIVE_SERVER_VERSION,
                worldName,
                worldgenProvider,
                worldgenVersion,
                seed
            )
        ), INITIALIZATION_TIMEOUT);
        NativeDoorTransitionEvidence result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no door evidence"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeTraversalProbe captureTraversalProbe(
        double[] requestedPositions,
        double[] requestedUpwardLimits
    ) {
        requireWorldCaptureReady();
        if (nativeRegionManifest == null) {
            throw new IllegalStateException(
                "Traversal probe requires a selected static Region manifest"
            );
        }
        double[] positions = requestedPositions == null
            ? new double[0]
            : requestedPositions.clone();
        double[] upwardLimits = requestedUpwardLimits == null
            ? new double[0]
            : requestedUpwardLimits.clone();
        int count = upwardLimits.length;
        if (
            count < 1
                || count > NativeTraversalProbe.MAX_SAMPLES
                || positions.length != count * 3
        ) {
            throw new IllegalArgumentException(
                "Traversal probe requires 1.."
                    + NativeTraversalProbe.MAX_SAMPLES
                    + " positions and matching upward limits"
            );
        }
        for (double value : positions) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                    "Traversal probe positions must be finite"
                );
            }
        }
        for (double limit : upwardLimits) {
            if (!Double.isFinite(limit) || limit < 0.0) {
                throw new IllegalArgumentException(
                    "Traversal upward limits must be finite and non-negative"
                );
            }
        }
        AtomicReference<NativeTraversalProbe> captured = new AtomicReference<>();
        runOnWorld(
            () -> captured.set(NavigationProbeCapture.buildTraversalProbe(
                    new NavigationProbeContext(world, agentRef, worldName, worldgenProvider, worldgenVersion, seed),
                    positions, upwardLimits)),
            INITIALIZATION_TIMEOUT
        );
        NativeTraversalProbe result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no traversal probe"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeTraversalEdgeProbe captureTraversalEdgeProbe(
        double[] requestedStartPositions,
        double[] requestedTargetPositions,
        double[] requestedHorizontalArrivalTolerances,
        double[] requestedVerticalArrivalTolerances
    ) {
        requireWorldCaptureReady();
        if (nativeRegionManifest == null) {
            throw new IllegalStateException(
                "Traversal edge probe requires a selected static Region manifest"
            );
        }
        double[] starts = requestedStartPositions == null
            ? new double[0]
            : requestedStartPositions.clone();
        double[] targets = requestedTargetPositions == null
            ? new double[0]
            : requestedTargetPositions.clone();
        double[] horizontalTolerances =
            requestedHorizontalArrivalTolerances == null
            ? new double[0]
            : requestedHorizontalArrivalTolerances.clone();
        double[] verticalTolerances =
            requestedVerticalArrivalTolerances == null
            ? new double[0]
            : requestedVerticalArrivalTolerances.clone();
        int count = horizontalTolerances.length;
        if (
            count < 1
                || count > NativeTraversalEdgeProbe.MAX_SAMPLES
                || starts.length != count * 3
                || targets.length != count * 3
                || verticalTolerances.length != count
        ) {
            throw new IllegalArgumentException(
                "Traversal edge probe requires 1.."
                    + NativeTraversalEdgeProbe.MAX_SAMPLES
                    + " matching start, target, and tolerance rows"
            );
        }
        requireFinite(starts, "Traversal edge starts");
        requireFinite(targets, "Traversal edge targets");
        requireFinite(
            horizontalTolerances,
            "Traversal edge horizontal tolerances"
        );
        requireFinite(
            verticalTolerances,
            "Traversal edge vertical tolerances"
        );
        for (int index = 0; index < count; index++) {
            if (
                horizontalTolerances[index] < 0.0
                    || verticalTolerances[index] < 0.0
            ) {
                throw new IllegalArgumentException(
                    "Traversal edge tolerances must be non-negative"
                );
            }
        }
        AtomicReference<NativeTraversalEdgeProbe> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(
                NavigationProbeCapture.buildTraversalEdgeProbe(
                    new NavigationProbeContext(world, agentRef, worldName, worldgenProvider, worldgenVersion, seed),
                    
                    starts,
                    targets,
                    horizontalTolerances,
                    verticalTolerances
                )
            ),
            INITIALIZATION_TIMEOUT
        );
        NativeTraversalEdgeProbe result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no traversal edge probe"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeNavigationPathProbe captureNavigationPathProbe(
        double[] requestedStartPositions,
        double[] requestedTargetPositions,
        int maximumPathLength,
        int openNodesLimit,
        int totalNodesLimit,
        int nodesPerIteration
    ) {
        requireWorldCaptureReady();
        if (nativeRegionManifest == null) {
            throw new IllegalStateException(
                "Navigation path probe requires a selected static Region manifest"
            );
        }
        double[] starts = requestedStartPositions == null
            ? new double[0]
            : requestedStartPositions.clone();
        double[] targets = requestedTargetPositions == null
            ? new double[0]
            : requestedTargetPositions.clone();
        int count = starts.length / 3;
        if (
            count < 1
                || count > NativeNavigationPathProbe.MAX_SAMPLES
                || starts.length != count * 3
                || targets.length != starts.length
        ) {
            throw new IllegalArgumentException(
                "Navigation path probe requires 1.."
                    + NativeNavigationPathProbe.MAX_SAMPLES
                    + " matching start and target rows"
            );
        }
        requireFinite(starts, "Navigation path starts");
        requireFinite(targets, "Navigation path targets");
        requireNavigationCapacity(
            maximumPathLength,
            1,
            NativeNavigationPathProbe.MAXIMUM_PATH_LENGTH,
            "maximum_path_length"
        );
        requireNavigationCapacity(
            openNodesLimit,
            1,
            NativeNavigationPathProbe.MAX_OPEN_NODES,
            "open_nodes_limit"
        );
        requireNavigationCapacity(
            totalNodesLimit,
            1,
            NativeNavigationPathProbe.MAX_TOTAL_NODES,
            "total_nodes_limit"
        );
        requireNavigationCapacity(
            nodesPerIteration,
            1,
            NativeNavigationPathProbe.MAX_NODES_PER_ITERATION,
            "nodes_per_iteration"
        );
        AtomicReference<NativeNavigationPathProbe> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(
                NavigationProbeCapture.buildNavigationPathProbe(
                    new NavigationProbeContext(world, agentRef, worldName, worldgenProvider, worldgenVersion, seed),
                    
                    starts,
                    targets,
                    maximumPathLength,
                    openNodesLimit,
                    totalNodesLimit,
                    nodesPerIteration
                )
            ),
            INITIALIZATION_TIMEOUT
        );
        NativeNavigationPathProbe result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no navigation path probe"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeNavigationSuccessorProbe
        captureNavigationSuccessorProbe(
            double[] requestedStartPositions,
            byte[] requestedDirectionIndices
        ) {
        requireWorldCaptureReady();
        if (nativeRegionManifest == null) {
            throw new IllegalStateException(
                "Navigation successor probe requires a selected static Region manifest"
            );
        }
        double[] starts = requestedStartPositions == null
            ? new double[0]
            : requestedStartPositions.clone();
        byte[] directions = requestedDirectionIndices == null
            ? new byte[0]
            : requestedDirectionIndices.clone();
        int count = directions.length;
        if (
            count < 1
                || count > NativeNavigationSuccessorProbe.MAX_SAMPLES
                || starts.length != count * 3
        ) {
            throw new IllegalArgumentException(
                "Navigation successor probe requires 1.."
                    + NativeNavigationSuccessorProbe.MAX_SAMPLES
                    + " matching start and direction rows"
            );
        }
        requireFinite(starts, "Navigation successor starts");
        AtomicReference<NativeNavigationSuccessorProbe> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(
                NavigationProbeCapture.buildNavigationSuccessorProbe(
                    new NavigationProbeContext(world, agentRef, worldName, worldgenProvider, worldgenVersion, seed),
                    starts, directions)
            ),
            INITIALIZATION_TIMEOUT
        );
        NativeNavigationSuccessorProbe result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no navigation successor probe"
            );
        }
        return result;
    }

    @Override
    public synchronized PrivilegedEntitySnapshot
        capturePrivilegedEntitySnapshot(PrivilegedEntityQuery query) {
        requireWorldCaptureReady();
        if (query == null) {
            throw new IllegalArgumentException(
                "Privileged entity query must be present"
            );
        }
        AtomicReference<PrivilegedEntitySnapshot> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(buildPrivilegedEntitySnapshot(query)),
            INITIALIZATION_TIMEOUT
        );
        PrivilegedEntitySnapshot result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no privileged entity snapshot"
            );
        }
        return result;
    }

    @Override
    public synchronized PrivilegedNpcSnapshot
        capturePrivilegedNpcSnapshot(PrivilegedEntityQuery query) {
        requireWorldCaptureReady();
        if (query == null) {
            throw new IllegalArgumentException(
                "Privileged NPC query must be present"
            );
        }
        AtomicReference<PrivilegedNpcSnapshot> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(buildPrivilegedNpcSnapshot(query)),
            INITIALIZATION_TIMEOUT
        );
        PrivilegedNpcSnapshot result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no privileged NPC snapshot"
            );
        }
        return result;
    }

    @Override
    public synchronized NativePolicyWorldActionCapture
        capturePolicyWorldActions(PolicyWorldActionCaptureRequest request) {
        requireWorldCaptureReady();
        AtomicReference<NativePolicyWorldActionCapture> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(buildPolicyWorldActionCapture(request)),
            INITIALIZATION_TIMEOUT
        );
        NativePolicyWorldActionCapture result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native World thread returned no policy action capture"
            );
        }
        return result;
    }

    /** Build one capture while already executing on the owning World thread. */
    private NativePolicyWorldActionCapture buildPolicyWorldActionCapture(
        PolicyWorldActionCaptureRequest request
    ) {
        if (request == null) {
            throw new IllegalArgumentException(
                "Policy World-action capture request is required"
            );
        }
        int actorSlot = request.actorSlot();
        if (actorSlot >= policyActorStates.length) {
            throw new IllegalArgumentException(
                "Policy World-action capture actor is outside negotiated capacity"
            );
        }
        Ref<EntityStore> actor = policyActorRef(actorSlot);
        Store<EntityStore> store = world.getEntityStore().getStore();
        if (actor == null || !actor.isValid()) {
            throw new IllegalStateException(
                "Requested policy actor is unavailable"
            );
        }
        UUIDComponent actorUuid = store.getComponent(
            actor,
            UUIDComponent.getComponentType()
        );
        String liveActorIdentity = actorUuid == null
                || actorUuid.getUuid() == null
            ? ""
            : actorUuid.getUuid().toString();
        PolicyActorCaptureRoute.Route route = PolicyActorCaptureRoute.require(
            actorSlot,
            request.expectedActorIdentity(),
            policyActorStates.length,
            liveActorIdentity
        );
        NPCEntity npc = actor == null || !actor.isValid()
            ? null
            : store.getComponent(actor, NPCEntity.getComponentType());
        InteractionManager manager = actor == null || !actor.isValid()
            ? null
            : interactionManager(actor, store);
        if (policyWorldActionRecipeCatalog == null) {
            policyWorldActionRecipeCatalog = NativeCraftingCatalogEvidence.capture(
                NATIVE_SERVER_VERSION,
                worldName,
                worldgenProvider,
                worldgenVersion,
                seed
            );
            policyWorldActionRecipeTableIdentity =
                PolicyWorldActionSemanticHash.recipeTableIdentity(
                    policyWorldActionRecipeCatalog
                );
        }
        String bridgeSha256 =
            com.hytalerlbridge.network.wire.BridgeIdentity.BRIDGE_SHA256
                .orElseThrow(() -> new IllegalStateException(
                    "Native policy World actions require bridge identity"
                ));
        NativePolicyActorState actorState = policyActorState(route.actorSlot());
        return NativePolicyWorldActionCaptureProducer.capture(
            new NativePolicyWorldActionCaptureProducer.Context(
                bridgeSha256,
                NATIVE_SERVER_VERSION,
                worldName,
                worldgenProvider,
                worldgenVersion,
                seed,
                reservationId.toString(),
                episodeSteps,
                route.actorSlot(),
                world,
                actor,
                store,
                npc,
                manager,
                actorState.desiredPitch(),
                actorState.desiredYaw(),
                NativeHeadlessWorldVerbContext.installed(actor, store),
                itemIds,
                policyWorldActionRecipeCatalog,
                policyWorldActionRecipeTableIdentity,
                this::buildMutableBlockCells,
                request
            )
        );
    }

    @Override
    public synchronized PrivilegedNpcSnapshot
        capturePrivilegedNpcSnapshotByUuid(PrivilegedNpcUuidQuery query) {
        requireWorldCaptureReady();
        if (query == null) {
            throw new IllegalArgumentException(
                "Privileged NPC UUID query must be present"
            );
        }
        AtomicReference<PrivilegedNpcSnapshot> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(buildPrivilegedNpcSnapshotByUuid(query)),
            INITIALIZATION_TIMEOUT
        );
        PrivilegedNpcSnapshot result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no privileged NPC snapshot"
            );
        }
        return result;
    }

    @Override
    public synchronized NpcTraceBatch startNpcTrace(
        UUID npcUuid,
        String expectedRole,
        int capacity
    ) {
        requireWorldCaptureReady();
        AtomicReference<NpcTraceBatch> captured = new AtomicReference<>();
        runOnWorld(
            () -> captured.set(npcTraceRecorder.start(
                world,
                NATIVE_SERVER_VERSION,
                worldgenProvider,
                worldgenVersion,
                seed,
                npcTraceEnvironmentParameters(),
                npcUuid,
                expectedRole,
                capacity
            )),
            INITIALIZATION_TIMEOUT
        );
        NpcTraceBatch result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no NPC trace identity"
            );
        }
        return result;
    }

    private Map<String, String> npcTraceEnvironmentParameters() {
        Map<String, String> parameters = new LinkedHashMap<>();
        parameters.put("task_id", taskId);
        parameters.put("world_template", options.world());
        parameters.put("requested_npc_role", options.npcRole());
        parameters.put("combat_target_role", combatTargetRole());
        parameters.put("native_actor_target_attitude", nativeActorTargetAttitude);
        parameters.put("native_target_actor_attitude", nativeTargetActorAttitude);
        parameters.put(
            "native_mutual_hostility",
            Boolean.toString(nativeMutualHostility)
        );
        parameters.put(
            "native_hostility_authority",
            "WorldSupport.getAttitude"
        );
        parameters.put("curriculum_phase", Integer.toString(curriculumPhase));
        parameters.put("ticks_per_step", Integer.toString(ticksPerStep));
        parameters.put("max_episode_steps", Integer.toString(maxEpisodeSteps));
        parameters.put("fidelity_fixture", options.fidelityFixture());
        parameters.put("combat_target_active", Boolean.toString(options.combatTargetActive()));
        parameters.put("worldgen_structure", worldgenWorldStructure);
        parameters.put("spawn_x", Double.toString(spawnX));
        parameters.put("spawn_y", Double.toString(spawnY));
        parameters.put("spawn_z", Double.toString(spawnZ));
        parameters.put("native_tick_rate", Integer.toString(world.getTps()));
        parameters.put(
            "native_time_dilation",
            Float.toString(options.nativeTimeDilation())
        );
        parameters.put("native_control", "authored_java_ai");
        parameters.put("block_ticking", Boolean.toString(options.blockTickingEnabled()));
        parameters.put(
            "native_navigation_trace",
            Boolean.toString(options.nativeNavigationTrace())
        );
        parameters.put("native_world_verbs", Boolean.toString(options.nativeWorldVerbs()));
        parameters.put(
            "native_combat_item_id",
            options.hasNativeCombatItem() ? options.nativeCombatItemId() : ""
        );
        parameters.put(
            "native_combat_interaction_id",
            options.hasNativeCombatInteraction()
                ? options.nativeCombatInteractionId()
                : ""
        );
        parameters.put(
            "native_combat_interaction_type",
            options.hasNativeCombatInteraction()
                ? options.nativeCombatInteractionType()
                : ""
        );
        parameters.put(
            "native_combat_ability_ids",
            options.hasNativeCombatAbilities()
                ? String.join(",", options.nativeCombatAbilityInteractionIds())
                : ""
        );
        parameters.put(
            "native_combat_ability_types",
            options.hasNativeCombatAbilities()
                ? String.join(",", options.nativeCombatAbilityInteractionTypes())
                : ""
        );
        parameters.put(
            "native_guard_interaction_id",
            options.hasNativeGuardInteraction() ? options.nativeGuardInteractionId() : ""
        );
        parameters.put(
            "native_guard_interaction_type",
            options.hasNativeGuardInteraction()
                ? options.nativeGuardInteractionType()
                : ""
        );
        parameters.put(
            "native_actor_evidence_resource_stat_ids",
            String.join(",", nativeActorEvidenceResourceStatIds())
        );
        parameters.put(
            "native_agent_armor_item_ids",
            options.hasNativeAgentArmorOverride()
                ? String.join(",", options.nativeAgentArmorItemIds())
                : ""
        );
        parameters.put(
            "native_policy_combat_binding_count",
            Integer.toString(
                options.hasNativePolicyCombatBindings()
                    ? options.nativePolicyCombatBindings().size()
                    : 0
            )
        );
        return Map.copyOf(parameters);
    }

    @Override
    public synchronized NpcTraceBatch drainNpcTrace(
        UUID traceUuid,
        int maxFrames,
        boolean stop
    ) {
        requireWorldCaptureReady();
        AtomicReference<NpcTraceBatch> captured = new AtomicReference<>();
        runOnWorld(
            () -> captured.set(npcTraceRecorder.drain(
                world.getEntityStore().getStore(),
                traceUuid,
                maxFrames,
                stop
            )),
            INITIALIZATION_TIMEOUT
        );
        NpcTraceBatch result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no NPC trace batch"
            );
        }
        return result;
    }

    @Override
    public synchronized WorldgenStructureMarkerSnapshot
        captureWorldgenStructureMarkers(WorldgenStructureMarkerQuery query) {
        requireWorldCaptureReady();
        if (query == null) {
            throw new IllegalArgumentException(
                "WorldGen structure marker query must be present"
            );
        }
        AtomicReference<WorldgenStructureMarkerSnapshot> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(buildWorldgenStructureMarkers(query)),
            INITIALIZATION_TIMEOUT
        );
        WorldgenStructureMarkerSnapshot result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no WorldGen structure markers"
            );
        }
        return result;
    }

    @Override
    public synchronized NativeExplosionCandidateProbe
        captureExplosionCandidateProbe(
            double[] requestedOrigin,
            int blockDamageRadius,
            float entityDamageRadius,
            boolean ignoreControlledActor,
            int capacity
        ) {
        requireWorldCaptureReady();
        double[] origin = requestedOrigin == null
            ? null
            : requestedOrigin.clone();
        if (
            origin == null
                || origin.length != 3
                || !Double.isFinite(origin[0])
                || !Double.isFinite(origin[1])
                || !Double.isFinite(origin[2])
        ) {
            throw new IllegalArgumentException(
                "Explosion origin must be one finite XYZ row"
            );
        }
        AtomicReference<NativeExplosionCandidateProbe> captured =
            new AtomicReference<>();
        runOnWorld(
            () -> captured.set(ExplosionAdmissionProbe.capture(
                world,
                agentRef,
                NATIVE_SERVER_VERSION,
                worldName,
                worldgenProvider,
                worldgenVersion,
                seed,
                origin,
                blockDamageRadius,
                entityDamageRadius,
                ignoreControlledActor,
                capacity
            )),
            INITIALIZATION_TIMEOUT
        );
        NativeExplosionCandidateProbe result = captured.get();
        if (result == null) {
            throw new IllegalStateException(
                "Native world thread returned no explosion candidate probe"
            );
        }
        return result;
    }

    private PrivilegedEntitySnapshot buildPrivilegedEntitySnapshot(
        PrivilegedEntityQuery query
    ) {
        Store<EntityStore> store = world.getEntityStore().getStore();
        Query<EntityStore> componentQuery = Query.and(
            UUIDComponent.getComponentType(),
            TransformComponent.getComponentType(),
            Velocity.getComponentType()
        );
        List<PrivilegedEntityRow> matches = new ArrayList<>();
        store.forEachChunk(componentQuery, (chunk, commandBuffer) -> {
            for (int index = 0; index < chunk.size(); index++) {
                UUIDComponent identity = chunk.getComponent(
                    index,
                    UUIDComponent.getComponentType()
                );
                TransformComponent transform = chunk.getComponent(
                    index,
                    TransformComponent.getComponentType()
                );
                Velocity velocity = chunk.getComponent(
                    index,
                    Velocity.getComponentType()
                );
                if (
                    identity == null
                        || transform == null
                        || velocity == null
                ) {
                    continue;
                }
                Vector3d position = transform.getPosition();
                if (!query.contains(position.x, position.y, position.z)) {
                    continue;
                }
                Rotation3f rotation = transform.getRotation();
                matches.add(PrivilegedEntityRow.of(
                    identity.getUuid(),
                    position.x,
                    position.y,
                    position.z,
                    rotation.yaw(),
                    rotation.pitch(),
                    rotation.roll(),
                    velocity.getX(),
                    velocity.getY(),
                    velocity.getZ()
                ));
            }
        });
        return PrivilegedEntitySnapshot.fromMatches(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            query,
            matches
        );
    }

    private PrivilegedNpcSnapshot buildPrivilegedNpcSnapshot(
        PrivilegedEntityQuery query
    ) {
        Store<EntityStore> store = world.getEntityStore().getStore();
        Query<EntityStore> componentQuery = Query.and(
            UUIDComponent.getComponentType(),
            TransformComponent.getComponentType(),
            Velocity.getComponentType(),
            NPCEntity.getComponentType()
        );
        List<PrivilegedNpcGeometryRow> matches = new ArrayList<>();
        store.forEachChunk(componentQuery, (chunk, commandBuffer) -> {
            for (int index = 0; index < chunk.size(); index++) {
                UUIDComponent identity = chunk.getComponent(
                    index,
                    UUIDComponent.getComponentType()
                );
                TransformComponent transform = chunk.getComponent(
                    index,
                    TransformComponent.getComponentType()
                );
                Velocity velocity = chunk.getComponent(
                    index,
                    Velocity.getComponentType()
                );
                NPCEntity npc = chunk.getComponent(
                    index,
                    NPCEntity.getComponentType()
                );
                if (
                    identity == null
                        || transform == null
                        || velocity == null
                        || npc == null
                ) {
                    continue;
                }
                Vector3d position = transform.getPosition();
                if (!query.contains(position.x, position.y, position.z)) {
                    continue;
                }
                Rotation3f rotation = transform.getRotation();
                PrivilegedEntityRow spatial = PrivilegedEntityRow.ofNpc(
                    identity.getUuid(),
                    npc.getRoleName(),
                    position.x,
                    position.y,
                    position.z,
                    rotation.yaw(),
                    rotation.pitch(),
                    rotation.roll(),
                    velocity.getX(),
                    velocity.getY(),
                    velocity.getZ()
                );
                matches.add(PrivilegedNpcGeometryCapture.capture(
                    store,
                    chunk.getReferenceTo(index),
                    spatial
                ));
            }
        });
        return PrivilegedNpcSnapshot.fromMatches(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            query,
            matches
        );
    }

    private PrivilegedNpcSnapshot buildPrivilegedNpcSnapshotByUuid(
        PrivilegedNpcUuidQuery uuidQuery
    ) {
        Store<EntityStore> store = world.getEntityStore().getStore();
        PrivilegedEntityQuery query = uuidQuery.spatialQuery();
        Ref<EntityStore> reference = world.getEntityStore().getRefFromUUID(
            uuidQuery.npcUuid()
        );
        List<PrivilegedNpcGeometryRow> matches = new ArrayList<>(1);
        if (reference != null && reference.isValid()) {
            UUIDComponent identity = store.getComponent(
                reference,
                UUIDComponent.getComponentType()
            );
            TransformComponent transform = store.getComponent(
                reference,
                TransformComponent.getComponentType()
            );
            Velocity velocity = store.getComponent(
                reference,
                Velocity.getComponentType()
            );
            NPCEntity npc = store.getComponent(
                reference,
                NPCEntity.getComponentType()
            );
            if (
                identity != null
                    && uuidQuery.npcUuid().equals(identity.getUuid())
                    && transform != null
                    && velocity != null
                    && npc != null
            ) {
                Vector3d position = transform.getPosition();
                if (query.contains(position.x, position.y, position.z)) {
                    Rotation3f rotation = transform.getRotation();
                    PrivilegedEntityRow spatial = PrivilegedEntityRow.ofNpc(
                        identity.getUuid(),
                        npc.getRoleName(),
                        position.x,
                        position.y,
                        position.z,
                        rotation.yaw(),
                        rotation.pitch(),
                        rotation.roll(),
                        velocity.getX(),
                        velocity.getY(),
                        velocity.getZ()
                    );
                    matches.add(PrivilegedNpcGeometryCapture.capture(
                        store,
                        reference,
                        spatial
                    ));
                }
            }
        }
        return PrivilegedNpcSnapshot.fromMatches(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            query,
            matches
        );
    }

    private NativePerceptionChannels buildPerceptionChannels(int[] positions) {
        record Column(short height, int tint) {}

        int samples = positions.length / 3;
        byte[] available = new byte[samples];
        byte[] validity = new byte[samples];
        short[] heightmap = new short[samples];
        byte[] skyLight = new byte[samples];
        byte[] blockLightRgb = new byte[samples * 3];
        int[] environment = new int[samples];
        int[] tintArgb = new int[samples];
        Map<Long, WorldChunk> chunks = new HashMap<>();
        Map<Long, Column> columns = new HashMap<>();
        Map<TraceLightSection, Boolean> lightReady = new HashMap<>();
        long worldTick = world.getTick();

        for (int sample = 0; sample < samples; sample++) {
            int offset = sample * 3;
            int x = positions[offset];
            int y = positions[offset + 1];
            int z = positions[offset + 2];
            long chunkIndex = ChunkUtil.indexChunkFromBlock(x, z);
            WorldChunk chunk = chunks.get(chunkIndex);
            if (chunk == null) {
                chunk = world.getChunkIfLoaded(chunkIndex);
                if (chunk != null) chunks.put(chunkIndex, chunk);
            }
            if (chunk == null) continue;
            BlockChunk blocks = chunk.getBlockChunk();
            if (blocks == null) continue;

            available[sample] = 1;
            int localX = ChunkUtil.localCoordinate((long) x);
            int localZ = ChunkUtil.localCoordinate((long) z);
            int valid = NativePerceptionChannels.VALID_TINT_RGB;
            long columnKey = ((long) x << 32) ^ (z & 0xffffffffL);
            Column column = columns.get(columnKey);
            if (column == null) {
                column = new Column(
                    blocks.getHeight(localX, localZ),
                    blocks.getTint(localX, localZ)
                );
                columns.put(columnKey, column);
            }
            if (column.height() != NativePerceptionChannels.WORLD_HEIGHT) {
                heightmap[sample] = column.height();
                valid |= NativePerceptionChannels.VALID_HEIGHTMAP;
            }
            tintArgb[sample] = column.tint();

            if (y >= 0 && y < NativePerceptionChannels.WORLD_HEIGHT) {
                valid |= NativePerceptionChannels.VALID_ENVIRONMENT;
                environment[sample] = blocks.getEnvironment(localX, y, localZ);
                int sectionY = Math.floorDiv(y, ChunkUtil.SIZE);
                TraceLightSection lightSection = new TraceLightSection(
                    chunk.getX(), chunk.getZ(), sectionY
                );
                Boolean ready = lightReady.get(lightSection);
                if (ready == null) {
                    ready = RegionLightCapture.isAvailable(
                        blocks.getSectionAtBlockY(y)
                    );
                    lightReady.put(lightSection, ready);
                }
                if (ready) {
                    traceLightRequestTicks.remove(lightSection);
                    valid |= NativePerceptionChannels.VALID_SKY_LIGHT
                        | NativePerceptionChannels.VALID_BLOCK_LIGHT_RGB;
                    skyLight[sample] = blocks.getSkyLight(localX, y, localZ);
                    int rgb = sample * 3;
                    blockLightRgb[rgb] = blocks.getRedBlockLight(localX, y, localZ);
                    blockLightRgb[rgb + 1] = blocks.getGreenBlockLight(
                        localX,
                        y,
                        localZ
                    );
                    blockLightRgb[rgb + 2] = blocks.getBlueBlockLight(
                        localX,
                        y,
                        localZ
                    );
                } else {
                    Long requested = traceLightRequestTicks.get(lightSection);
                    if (requested == null || requested.longValue() != worldTick) {
                        RegionLightCapture.invalidateEmptyPlaceholders(
                            blocks.getSectionAtBlockY(y)
                        );
                        world.getChunkLighting().addToQueue(
                            new Vector3i(chunk.getX(), sectionY, chunk.getZ())
                        );
                        traceLightRequestTicks.put(lightSection, worldTick);
                    }
                }
            }
            validity[sample] = (byte) valid;
        }
        return new NativePerceptionChannels(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            positions,
            available,
            validity,
            heightmap,
            skyLight,
            blockLightRgb,
            environment,
            tintArgb
        );
    }

    private NativeMutableBlockEvidence buildMutableBlockEvidence() {
        int x = (int) Math.floor(spawnX) + 8;
        int y = (int) Math.floor(spawnY) + 1;
        int z = (int) Math.floor(spawnZ) - 8;
        Vector3i position = new Vector3i(x, y, z);
        WorldChunk chunk = world.getChunkIfLoaded(
            ChunkUtil.indexChunkFromBlock(x, z)
        );
        if (chunk == null) {
            throw new IllegalStateException(
                "Mutable block evidence fixture chunk is not loaded"
            );
        }
        if (world.getBlock(x, y, z) != 0 || chunk.getFluidId(
            ChunkUtil.localCoordinate((long) x),
            y,
            ChunkUtil.localCoordinate((long) z)
        ) != 0) {
            throw new IllegalStateException(
                "Mutable block evidence fixture cell is not initially empty"
            );
        }

        BlockHealthModule module = BlockHealthModule.get();
        Store<ChunkStore> chunkStore = world.getChunkStore().getStore();
        BlockHealthChunk health = module == null
            ? null
            : chunkStore.getComponent(
                chunk.getReference(),
                module.getBlockHealthChunkComponentType()
            );
        if (health == null) {
            throw new IllegalStateException(
                "Mutable block evidence chunk has no BlockHealthChunk"
            );
        }
        Store<EntityStore> entityStore = world.getEntityStore().getStore();
        WorldTimeResource time = entityStore.getResource(
            WorldTimeResource.getResourceType()
        );
        if (time == null) {
            throw new IllegalStateException(
                "Mutable block evidence world has no game-time resource"
            );
        }

        List<NativeMutableBlockEvidence.Row> rows = new ArrayList<>(5);
        try {
            health.removeBlock(world, position);
            setControlledBlock(position, "Rock_Stone", true);
            rows.add(RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "initial",
                position,
                chunk,
                health,
                time.getGameTime()
            ));

            health.damageBlock(time.getGameTime(), world, position, 0.4f);
            rows.add(RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "damaged",
                position,
                chunk,
                health,
                time.getGameTime()
            ));

            health.repairBlock(world, position, 0.1f);
            rows.add(RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "repaired",
                position,
                chunk,
                health,
                time.getGameTime()
            ));

            setControlledBlock(position, BlockType.EMPTY_KEY, false);
            health.removeBlock(world, position);
            rows.add(RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "removed",
                position,
                chunk,
                health,
                time.getGameTime()
            ));

            setControlledBlock(position, "Rock_Stone", true);
            rows.add(RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "placed",
                position,
                chunk,
                health,
                time.getGameTime()
            ));
        } finally {
            health.removeBlock(world, position);
            world.setBlock(x, y, z, BlockType.EMPTY_KEY);
        }
        return new NativeMutableBlockEvidence(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            new int[] {x, y, z},
            rows
        );
    }

    private WorldgenStructureMarkerSnapshot buildWorldgenStructureMarkers(
        WorldgenStructureMarkerQuery query
    ) {
        Store<EntityStore> store = world.getEntityStore().getStore();
        Query<EntityStore> componentQuery = Query.and(
            UUIDComponent.getComponentType(),
            TransformComponent.getComponentType(),
            WorldGenId.getComponentType(),
            FromPrefabInstance.getComponentType(),
            SpawnMarkerEntity.getComponentType()
        );
        List<WorldgenStructureMarkerRow> matches = new ArrayList<>();
        store.forEachChunk(componentQuery, (chunk, commandBuffer) -> {
            for (int index = 0; index < chunk.size(); index++) {
                UUIDComponent identity = chunk.getComponent(
                    index,
                    UUIDComponent.getComponentType()
                );
                TransformComponent transform = chunk.getComponent(
                    index,
                    TransformComponent.getComponentType()
                );
                WorldGenId worldgenId = chunk.getComponent(
                    index,
                    WorldGenId.getComponentType()
                );
                FromPrefabInstance prefabInstance = chunk.getComponent(
                    index,
                    FromPrefabInstance.getComponentType()
                );
                SpawnMarkerEntity marker = chunk.getComponent(
                    index,
                    SpawnMarkerEntity.getComponentType()
                );
                if (
                    identity == null
                        || transform == null
                        || worldgenId == null
                        || prefabInstance == null
                        || marker == null
                        || !query.accepts(marker.getSpawnMarkerId())
                ) {
                    continue;
                }
                Vector3d position = transform.getPosition();
                if (!query.contains(position.x, position.y, position.z)) {
                    continue;
                }
                Rotation3f rotation = transform.getRotation();
                matches.add(WorldgenStructureMarkerRow.of(
                    identity.getUuid(),
                    marker.getSpawnMarkerId(),
                    position.x,
                    position.y,
                    position.z,
                    rotation.yaw(),
                    rotation.pitch(),
                    rotation.roll(),
                    worldgenId.getWorldGenId(),
                    prefabInstance.getPrefabInstanceId()
                ));
            }
        });
        return WorldgenStructureMarkerSnapshot.fromMatches(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            query,
            matches
        );
    }

    private NativeMutableBlockCells buildMutableBlockCells(int[] positions) {
        BlockHealthModule module = BlockHealthModule.get();
        if (module == null) {
            throw new IllegalStateException(
                "Native mutable block capture has no BlockHealthModule"
            );
        }
        Store<ChunkStore> chunkStore = world.getChunkStore().getStore();
        Store<EntityStore> entityStore = world.getEntityStore().getStore();
        WorldTimeResource time = entityStore.getResource(
            WorldTimeResource.getResourceType()
        );
        if (time == null) {
            throw new IllegalStateException(
                "Native mutable block capture has no game-time resource"
            );
        }
        List<NativeMutableBlockCells.Cell> cells = new ArrayList<>(
            positions.length / 3
        );
        for (int offset = 0; offset < positions.length; offset += 3) {
            int x = positions[offset];
            int y = positions[offset + 1];
            int z = positions[offset + 2];
            int[] position = {x, y, z};
            if (y < 0 || y >= NativePerceptionChannels.WORLD_HEIGHT) {
                cells.add(new NativeMutableBlockCells.Cell(
                    position,
                    false,
                    null
                ));
                continue;
            }
            WorldChunk chunk = world.getChunkIfLoaded(
                ChunkUtil.indexChunkFromBlock(x, z)
            );
            if (chunk == null) {
                cells.add(new NativeMutableBlockCells.Cell(
                    position,
                    false,
                    null
                ));
                continue;
            }
            BlockHealthChunk health = chunkStore.getComponent(
                chunk.getReference(),
                module.getBlockHealthChunkComponentType()
            );
            if (health == null) {
                throw new IllegalStateException(
                    "Loaded mutable block cell has no BlockHealthChunk"
                );
            }
            cells.add(new NativeMutableBlockCells.Cell(
                position,
                true,
                RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                    "snapshot",
                    new Vector3i(x, y, z),
                    chunk,
                    health,
                    time.getGameTime()
                )
            ));
        }
        return new NativeMutableBlockCells(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            cells
        );
    }

    private NativeDropProgramEvidence buildDropProgramEvidence(
        String blockAssetId,
        String route,
        int sampleCount
    ) {
        if (blockAssetId == null || blockAssetId.isBlank()) {
            throw new IllegalArgumentException(
                "Drop-program block asset ID cannot be blank"
            );
        }
        if (
            sampleCount < 1
                || sampleCount > NativeDropProgramEvidence.MAX_SAMPLES
        ) {
            throw new IllegalArgumentException(
                "Drop-program sample count exceeds capacity"
            );
        }
        BlockType blockType = BlockType.getAssetMap().getAsset(blockAssetId);
        if (blockType == null || blockType == BlockType.EMPTY) {
            throw new IllegalArgumentException(
                "Unknown block asset ID: " + blockAssetId
            );
        }
        ResolvedDropRoute resolved = resolveDropRoute(blockType, route);
        Map<List<NativeDropProgramEvidence.Stack>, Integer> counts =
            new LinkedHashMap<>();
        for (int sample = 0; sample < sampleCount; sample++) {
            List<ItemStack> drops = BlockHarvestUtils.getDrops(
                blockType,
                resolved.quantity(),
                resolved.itemId(),
                resolved.dropListId()
            );
            if (drops == null) {
                throw new IllegalStateException(
                    "Native drop API returned null"
                );
            }
            List<NativeDropProgramEvidence.Stack> stacks = drops.stream()
                .filter(stack -> !ItemStack.isEmpty(stack))
                .map(DropRouting::canonicalDropStack)
                .toList();
            if (
                stacks.size()
                    > NativeDropProgramEvidence.MAX_STACKS_PER_OUTCOME
            ) {
                throw new IllegalStateException(
                    "Native drop stack capacity exceeded"
                );
            }
            counts.merge(stacks, 1, Integer::sum);
            if (
                counts.size()
                    > NativeDropProgramEvidence.MAX_OUTCOMES
            ) {
                throw new IllegalStateException(
                    "Native drop outcome capacity exceeded"
                );
            }
        }
        List<NativeDropProgramEvidence.Outcome> outcomes = counts.entrySet()
            .stream()
            .map(entry -> new NativeDropProgramEvidence.Outcome(
                entry.getValue(),
                entry.getKey()
            ))
            .sorted(Comparator.comparing(
                outcome -> canonicalDropOutcomeKey(outcome.stacks())
            ))
            .toList();
        return new NativeDropProgramEvidence(
            NATIVE_SERVER_VERSION,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            blockAssetId,
            route,
            resolved.quantity(),
            resolved.itemId(),
            resolved.dropListId(),
            sampleCount,
            outcomes
        );
    }

    private void setControlledBlock(
        Vector3i position,
        String blockAssetId,
        boolean expectedPresent
    ) {
        world.setBlock(
            position.x,
            position.y,
            position.z,
            blockAssetId
        );
        boolean present = world.getBlock(
            position.x,
            position.y,
            position.z
        ) != 0;
        if (present != expectedPresent) {
            throw new IllegalStateException(
                "Mutable block fixture transition was rejected: "
                    + blockAssetId
            );
        }
    }


    private NativeExplosionMutationProbe.CellState explosionMutationCellState(
        int x,
        int y,
        int z,
        BlockHealthModule healthModule,
        Store<ChunkStore> chunkStore,
        WorldTimeResource time
    ) {
        WorldChunk chunk = world.getChunkIfLoaded(
            ChunkUtil.indexChunkFromBlock(x, z)
        );
        if (chunk == null) {
            throw new IllegalStateException(
                "Explosion mutation scan chunk unloaded during capture"
            );
        }
        BlockHealthChunk health = explosionMutationBlockHealth(
            x,
            y,
            z,
            healthModule,
            chunkStore
        );
        return NativeExplosionMutationProbe.CellState.fromMutableRow(
            RegionGeometryCapture.mutableBlockRow(nativeCellSemanticsCache, world, 
                "snapshot",
                new Vector3i(x, y, z),
                chunk,
                health,
                time.getGameTime()
            )
        );
    }

    private BlockHealthChunk explosionMutationBlockHealth(
        int x,
        int y,
        int z,
        BlockHealthModule healthModule,
        Store<ChunkStore> chunkStore
    ) {
        if (y < 0 || y >= NativePerceptionChannels.WORLD_HEIGHT) {
            throw new IllegalArgumentException(
                "Explosion mutation scan escaped the native Y domain"
            );
        }
        WorldChunk chunk = world.getChunkIfLoaded(
            ChunkUtil.indexChunkFromBlock(x, z)
        );
        if (chunk == null) {
            throw new IllegalStateException(
                "Explosion mutation health chunk unloaded during capture"
            );
        }
        BlockHealthChunk health = chunkStore.getComponent(
            chunk.getReference(),
            healthModule.getBlockHealthChunkComponentType()
        );
        if (health == null) {
            throw new IllegalStateException(
                "Explosion mutation loaded chunk has no BlockHealthChunk"
            );
        }
        return health;
    }

    private void requireWorldCaptureReady() {
        if (closed) {
            throw new IllegalStateException("Native environment is closed");
        }
        if (!initialized || world == null || !world.isAlive()) {
            throw new IllegalStateException(
                "Reset a live native environment before world capture"
            );
        }
    }




    @Override
    public synchronized StepResult step(AgentAction action) {
        if (closed) throw new IllegalStateException("Native environment is closed");
        if (!initialized) throw new IllegalStateException("Native environment is not reset");
        AgentAction actualAction = action == null ? AgentAction.noop() : action;
        if (done) return lastResult;
        return advance(actualAction, ticksPerStep, true);
    }

    @Override
    public synchronized StepResult step(NativeGroupAction actions) {
        if (closed) throw new IllegalStateException("Native environment is closed");
        if (!initialized) {
            throw new IllegalStateException("Native environment is not reset");
        }
        if (actions == null) {
            throw new IllegalArgumentException("native group action is required");
        }
        if (done) return lastResult;
        return advance(actions, ticksPerStep, true);
    }

    void applyControl(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        float deltaTime,
        int actorId
    ) {
        PendingStep step = pending.get();
        if (step == null || actorId < 0) return;
        Ref<EntityStore> expected = policyActorRef(actorId);
        if (ref != expected) return;
        AgentAction action = step.actionForEntity(actorId);
        if (action == null) return;
        npcTraceRecorder.markBridgeOverride(ref);
        NativePolicyActorState actorState = policyActorState(actorId);
        if (actorId == 0 && options.nativeNavigationTrace()) {
            captureTargetNavigationTrace(store);
        }

        Object policyCombatHandle = nativePolicyCombatHandles[actorId];
        boolean firstControlTick = step.controlTicks(actorId) == 0;
        InteractionType requestedGuardFork = null;
        boolean retainGuardForFork = false;
        if (
            policyCombatHandle == null
                && actorId == 0
                && firstControlTick
                && action.hasAbility()
                && action.abilitySlot() >= 0
                && action.abilitySlot() < nativeCombatAbilities.size()
        ) {
            requestedGuardFork = NativeGuardFork.resolveType(
                nativeGuardInteraction,
                nativeCombatAbilities.get(action.abilitySlot())
            );
            retainGuardForFork = requestedGuardFork != null
                && nativeGuardLifecycle.inputHeld()
                && nativeGuardActive;
        }
        boolean effectiveGuardHeld = action.guardHeld() || retainGuardForFork;
        if (policyCombatHandle == null && actorId == 0) {
            // Request identity must supersede retained terminal evidence even
            // when this tick cannot reach actor control components.
            nativeGuardLifecycle.observeInput(effectiveGuardHeld);
        }

        Role role = npc.getRole();
        if (role == null) return;
        MotionController controller = role.getActiveMotionController();
        if (controller == null) return;
        if (policyCombatHandle == null) {
            suppressAutonomousActorCombat(
                ref,
                role,
                store,
                actorState,
                actorId == 0 && action.guardHeld()
            );
            if (actorId == 0) {
                updateNativeGuard(
                    ref,
                    store,
                    effectiveGuardHeld,
                    step
                );
                if (retainGuardForFork && !action.guardHeld()) {
                    step.guardAccepted = false;
                }
            }
        }
        if (firstControlTick) {
            actorState.applyLookDelta(
                action.cameraDeltaYaw(),
                action.cameraDeltaPitch()
            );
            // Two levers, matching the simulator the policy was trained in.
            actorState.applyBodyYawDelta(action.bodyDeltaYaw());
            Inventory inventory = npc.getInventory();
            if (inventory != null && action.hotbarSlot() >= 0 && action.hotbarSlot() < 9) {
                inventory.setActiveHotbarSlot(ref, (byte) action.hotbarSlot(), store);
            }
            if (actorId == 0) {
                requestSyntheticWorldInteraction(ref, npc, store, action, step);
                if (action.use()) {
                    requestNativeBlockUse(ref, store, step);
                }
            }
            if (policyCombatHandle == null && action.attack()) {
                NativeAttackAdmission admission = requestNativeAttack(
                    ref,
                    npc,
                    store,
                    Float.isFinite(deltaTime) ? deltaTime : 0.0f,
                    action.requestedChargeTime(),
                    actorId,
                    actorState
                );
                step.observeAttack(actorId, admission);
            }
            if (
                policyCombatHandle == null
                    && actorId == 0
                    && action.hasAbility()
            ) {
                requestNativeAbility(
                    ref,
                    npc,
                    store,
                    action.abilitySlot(),
                    action.requestedChargeTime(),
                    requestedGuardFork,
                    step
                );
            }
            if (
                policyCombatHandle == null
                    && actorId == 0
                    && action.hasDodge()
            ) {
                requestNativeDodge(
                    ref,
                    store,
                    action.dodgeDirection(),
                    step
                );
            }
        }
        if (policyCombatHandle != null) {
            NativePolicyCombatFacade.Receipt receipt =
                NativePolicyCombatFacade.apply(
                    policyCombatHandle,
                    ref,
                    npc,
                    store,
                    Float.isFinite(deltaTime) ? deltaTime : 0.0f,
                    firstControlTick,
                    action.attack(),
                    action.abilitySlot(),
                    action.guardHeld(),
                    action.dodgeDirection(),
                    action.requestedChargeTime(),
                    false,
                    false
            );
            step.observePolicyCombat(actorId, receipt);
            nativePolicyCombatActiveAbilitySlots[actorId] =
                receipt.available() && receipt.abilityActive()
                    ? receipt.activeAbilitySlot()
                    : -1;
        } else if (actorId != 0) {
            step.observeUnboundPolicyCombat(actorId, action);
            nativePolicyCombatActiveAbilitySlots[actorId] = -1;
        }

        double forward = (action.forward() ? 1.0 : 0.0)
            - (action.back() ? 1.0 : 0.0);
        double strafe = (action.right() ? 1.0 : 0.0)
            - (action.left() ? 1.0 : 0.0);
        double length = Math.hypot(forward, strafe);
        int worldMoveDirection = action.worldMoveDirection();
        boolean worldMoveRequested = worldMoveDirection != 0;
        boolean moving = worldMoveRequested || length > 1.0e-9;
        boolean diveFixture = options.fidelityFixture().equals(
            EnvironmentOptions.DIVE_MOTION_FIXTURE
        );

        Steering body = role.getBodySteering();
        if (body != null) {
            body.clear();
            if (moving) {
                double dx;
                double dy;
                double dz;
                if (worldMoveRequested) {
                    // BODY-relative, matching the simulator. Choice one is the
                    // body's own forward, three its right, and so on -- the
                    // same thing the forward/back/left/right branch below
                    // expresses, so both rotate by `desiredBodyYaw` here.
                    //
                    // This used to be an ABSOLUTE compass (dx = sin(angle),
                    // dz = -cos(angle), no body rotation), which disagreed with
                    // the branch below and, more importantly, asked the policy
                    // to name a world bearing while every bearing it observes
                    // is measured from its own head.
                    double bodyAngle = (worldMoveDirection - 1)
                        * (Math.PI / 4.0);
                    double compassYaw = actorState.desiredBodyYaw();
                    double longitudinal = Math.cos(bodyAngle);
                    double lateral = Math.sin(bodyAngle);
                    dx = longitudinal * -Math.sin(compassYaw)
                        + lateral * Math.cos(compassYaw);
                    dy = 0.0;
                    dz = longitudinal * -Math.cos(compassYaw)
                        - lateral * Math.sin(compassYaw);
                } else {
                    forward /= length;
                    strafe /= length;
                    double pitchScale = diveFixture
                        ? Math.cos(actorState.desiredPitch())
                        : 1.0;
                    // Body-relative, like the simulator's compass channels.
                    double travelYaw = actorState.desiredBodyYaw();
                    dx = forward * pitchScale * -Math.sin(travelYaw)
                        + strafe * Math.cos(travelYaw);
                    dy = diveFixture
                        ? forward * Math.sin(actorState.desiredPitch())
                        : 0.0;
                    dz = forward * pitchScale * -Math.cos(travelYaw)
                        - strafe * Math.sin(travelYaw);
                }
                double vectorLength = Math.sqrt(
                    dx * dx + dy * dy + dz * dz
                );
                if (vectorLength > 1.0) {
                    dx /= vectorLength;
                    dy /= vectorLength;
                    dz /= vectorLength;
                }
                body.setTranslation(dx, dy, dz);
                body.setTranslationRelativeSpeed(1.0);
            }
            // The body carries its OWN commanded heading. This used to be
            // desiredYaw -- the camera -- so the chest was dragged round by every
            // look and the actor could not strafe while tracking a target.
            if (!diveFixture) body.setYaw(actorState.desiredBodyYaw());
            body.setRelativeTurnSpeed(AGENT_STEERING_RELATIVE_TURN_SPEED);
        }

        Steering head = role.getHeadSteering();
        if (head != null) {
            head.clear();
            head.setYaw(actorState.desiredYaw());
            head.setPitch(actorState.desiredPitch());
            head.setRelativeTurnSpeed(AGENT_STEERING_RELATIVE_TURN_SPEED);
        }

        if (controller instanceof MotionControllerBase base) {
            base.setMotionKind(moving ? MotionKind.MOVING : MotionKind.STANDING);
        }
        if (moving) controller.setNavState(NavState.PROGRESSING, 0.0, 0.0);
        if (diveFixture) {
            step.diveTrace = NativeDiveTrace.captureBefore(
                controller,
                body,
                role,
                ref,
                store
            );
            if (!step.diveTrace.availableBefore()) {
                throw new IllegalStateException(
                    "dive_motion fixture did not activate MotionControllerDive"
                );
            }
        }

        if (firstControlTick && action.jump() && role.isOnGround()) {
            double gravity = controller instanceof MotionControllerBase base
                ? Math.abs(base.getGravity())
                : AGENT_JUMP_VELOCITY_GRAVITY_FLOOR;
            // 0.5.7 walk controllers may expose their per-tick gravity scale here,
            // while addVelocity consumes world-units/second. Ten is the native
            // world gravity used by Hytale's jump-height conversion.
            if (!Double.isFinite(gravity)) {
                gravity = AGENT_JUMP_VELOCITY_GRAVITY_FLOOR;
            }
            gravity = Math.max(AGENT_JUMP_VELOCITY_GRAVITY_FLOOR, gravity);
            double jumpVelocity = Math.sqrt(
                2.0 * gravity * AGENT_JUMP_HEIGHT_PARAMETER
            );
            // setVelocity is intentional: addVelocity is the knockback path and
            // scales by the role's knockback multiplier, which turns a jump into
            // only ~0.17 blocks for Trork_Unarmed in 0.5.7.
            controller.setVelocity(
                new Vector3d(0.0, jumpVelocity, 0.0),
                AGENT_VELOCITY_CONFIG,
                true
            );
        }
        step.incrementControlTicks(actorId);
        if (firstControlTick) {
            requestPolicyNativeWorldVerbsWhenReady(step, store);
        }
    }

    void afterNativeTick(Store<EntityStore> store, float deltaTime) {
        npcTraceRecorder.afterTick(store);
        PendingStep step = pending.get();
        if (step == null) return;

        try {
            if (
                options.fidelityFixture().equals(
                    EnvironmentOptions.DIVE_MOTION_FIXTURE
                )
            ) {
                captureDiveTraceAfter(step, store);
            }
            if (step.actionForEntity(0) != null) {
                clearAgentDecisionTargets(store);
            }
            suppressPassiveTargetCombat(step, store);
            if (targetMemoryFixture != null && initialized) {
                targetMemoryFixture.afterTick(
                    world,
                    store,
                    targetRef,
                    agentRef,
                    episodeEngineTicks + step.executedTicks + 1
                );
            }
            updateAgentPolicyVerbTelemetry(step, store);
            updatePolicyWorldVerbTelemetry(step, store);
            updateLegacyActorZeroTypedWorldVerbTelemetry(step, store);
            updateSyntheticWorldInteractionTelemetry(store);
            if (
                fallingBlockTrace != null
                    && fallingBlockTrace.onGround()
            ) {
                fallingBlockTrace = fallingBlockTrace.impact();
                fallingBlockRef = null;
            }
            updateTargetCombatTelemetry(store);
            step.executedTicks++;
            if (Float.isFinite(deltaTime) && deltaTime >= 0.0f) {
                step.simulatedSeconds += deltaTime;
                step.minimumDeltaSeconds = Math.min(step.minimumDeltaSeconds, deltaTime);
                step.maximumDeltaSeconds = Math.max(step.maximumDeltaSeconds, deltaTime);
            }
            boolean agentMissing = agentRef == null || !agentRef.isValid();
            if (!agentMissing && step.executedTicks < step.requestedTicks) return;

            step.rejectUnprocessedWorldVerbs();

            NativeSnapshot snapshot = agentMissing
                ? deadSnapshot()
                : captureSnapshot(store);
            setWorldPausedWithoutAnchorBroadcast(true);
            StepResult result = finishStep(step, snapshot);
            if (pending.compareAndSet(step, null)) {
                step.future.complete(result);
            }
        } catch (Throwable throwable) {
            setWorldPausedWithoutAnchorBroadcast(true);
            if (pending.compareAndSet(step, null)) {
                step.future.completeExceptionally(throwable);
            }
        }
    }

    private void captureDiveTraceAfter(
        PendingStep step,
        Store<EntityStore> store
    ) {
        NativeDiveTrace trace = step.diveTrace;
        Ref<EntityStore> reference = agentRef;
        if (
            trace == null
                || reference == null
                || !reference.isValid()
        ) {
            throw new IllegalStateException(
                "dive_motion trace lost its controlled native entity"
            );
        }
        NPCEntity npc = store.getComponent(
            reference,
            NPCEntity.getComponentType()
        );
        Role role = npc == null ? null : npc.getRole();
        MotionController controller = role == null
            ? null
            : role.getActiveMotionController();
        step.diveTrace = trace.captureAfter(
            controller,
            role,
            reference,
            store
        );
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;

        PendingStep active = pending.getAndSet(null);
        if (active != null) {
            active.future.completeExceptionally(
                new IllegalStateException("Native environment closed during a step")
            );
        }
        clearSyntheticWorldInteractionRun();
        fallingBlockRef = null;

        World closingWorld = world;
        if (closingWorld != null && closingWorld.isAlive()) {
            try {
                runOnWorld(() -> {
                    PlayerRef anchor = tickAnchor;
                    if (anchor != null) closingWorld.untrackPlayerRef(anchor);
                    for (PlayerRef fixturePlayer : syntheticPlayerRefsByActor) {
                        if (fixturePlayer != null) {
                            closingWorld.untrackPlayerRef(fixturePlayer);
                        }
                    }
                    Store<EntityStore> entityStore = closingWorld.getEntityStore().getStore();
                    for (
                        int actorId = 0;
                        actorId < nativePolicyCombatHandles.length;
                        actorId++
                    ) {
                        Object handle = nativePolicyCombatHandles[actorId];
                        Ref<EntityStore> actor = policyActorRef(actorId);
                        if (handle != null && actor != null && actor.isValid()) {
                            NativePolicyCombatFacade.close(
                                handle,
                                actor,
                                entityStore,
                                false
                            );
                        }
                    }
                    for (
                        int actorId = 0;
                        actorId < policyActorStates.length;
                        actorId++
                    ) {
                        Ref<EntityStore> actor = policyActorRef(actorId);
                        if (actor == null || !actor.isValid()) continue;
                        if (NativeHeadlessWorldVerbContext.installed(
                            actor,
                            entityStore
                        )) {
                            NativeHeadlessWorldVerbContext.uninstall(
                                actor,
                                entityStore
                            );
                        }
                        NPCEntity npc = entityStore.getComponent(
                            actor,
                            NPCEntity.getComponentType()
                        );
                        if (npc != null) npc.removeReservation(reservationId);
                    }
                    Store<ChunkStore> chunkStore = closingWorld.getChunkStore().getStore();
                    for (Ref<ChunkStore> chunkRef : loadedChunks) {
                        if (chunkRef != null && chunkRef.isValid()) {
                            WorldChunk chunk = chunkStore.getComponent(
                                chunkRef,
                                WorldChunk.getComponentType()
                            );
                            if (chunk != null && chunk.shouldKeepLoaded()) chunk.removeKeepLoaded();
                        }
                    }
                    closingWorld.setPaused(true);
                }, Duration.ofSeconds(5));
            } catch (RuntimeException exception) {
                LOGGER.log(
                    System.Logger.Level.WARNING,
                    "Native environment cleanup task failed: " + exception.getMessage()
                );
            }
        }

        if (attached) {
            provider.detach(closingWorld, this);
            attached = false;
        }
        if (worldName != null && Universe.get() != null) {
            Universe.get().removeWorld(worldName);
        }
        loadedChunks.clear();
        loadedChunkIndices.clear();
        geometryChunkIndices.clear();
        traceLightRequestTicks.clear();
        tracePerceptionCandidates = List.of();
        tracePerceptionCandidateTick = Long.MIN_VALUE;
        nativeRegionManifest = null;
        regionLightHaloManifest = null;
        tickAnchor = null;
        syntheticPlayerRef = null;
        agentRef = null;
        targetRef = null;
        for (NativePolicyActorState state : policyActorStates) state.clear();
        for (int actorId = 0; actorId < nativePolicyCombatHandles.length; actorId++) {
            nativePolicyCombatHandles[actorId] = null;
            nativePolicyCombatSpecs[actorId] = null;
            nativePolicyCombatActiveAbilitySlots[actorId] = -1;
        }
        nativeCombatRootInteraction = null;
        nativeCombatInteractionType = null;
        nativeCombatAbilities = List.of();
        nativeGuardInteraction = null;
        nativeDodgeLeftInteraction = null;
        nativeDodgeRightInteraction = null;
        nativeGuardChain = null;
        nativeGuardActive = false;
        nativeGuardLifecycle.reset();
        trackedNativeGuardInteractionGeneration = -1L;
        trackedNativeGuardInteractionId = "";
        trackedNativeGuardStartTick = -1L;
        trackedNativeGuardFinishTick = -1L;
        pendingAgentAbilityFork = null;
        trackedAgentAbilityChain = null;
        trackedAgentAbilityParentChain = null;
        trackedAgentAbilitySlot = -1;
        trackedAgentAbilityId = "";
        trackedAgentAbilityType = "";
        trackedAgentAbilityStartTick = -1L;
        trackedAgentAbilityFinishTick = -1L;
        trackedAgentDodgeChain = null;
        trackedAgentDodgeDirection = 0;
        trackedAgentDodgeInteractionId = "";
        trackedAgentDodgeStartTick = -1L;
        trackedAgentDodgeFinishTick = -1L;
        trackedAgentUseChain = null;
        trackedAgentUseInteractionId = "";
        trackedAgentUseBlockInteractionId = "";
        trackedAgentUseTarget = null;
        trackedAgentUseMaximumDistance = 0.0;
        trackedAgentUseStartTick = -1L;
        trackedAgentUseFinishTick = -1L;
        syntheticCombatClient.reset();
        legacyActorZeroTypedWorldVerbChain = null;
        legacyActorZeroTypedWorldVerbCraftingManager = null;
        legacyActorZeroTypedWorldVerbCraftingRecipe = null;
        legacyActorZeroTypedWorldVerbExecution =
            WorldVerbTelemetry.Execution.unrequested();
        for (int actorId = 0; actorId < nativePolicyWorldVerbHandlesByActor.length; actorId++) {
            nativePolicyWorldVerbHandlesByActor[actorId] = null;
            nativePolicyWorldVerbRequestsByActor[actorId] = null;
        }
        legacyActorZeroTypedWorldVerbGeometryBefore = "";
        legacyActorZeroTypedWorldVerbInventoryBefore = "";
        nativeUseAvailable = false;
        nativeUseAvailableInteractionId = "";
        nativeUseAvailableBlockInteractionId = "";
        nativeUseAvailableItemId = "";
        nativeUseAvailableSourceContainer = "";
        nativeUseAvailableSourceSlot = -1;
        nativeUseAvailableSourceQuantity = -1;
        nativeUseAvailableTarget = null;
        nativeUseAvailableMaximumDistance = 0.0;
        nativeUseUnavailableReason = "not_sampled";
        nativeAgentArmorOverride = null;
        resetTargetAttackTelemetry();
        world = null;
    }

    private void createWorld() {
        worldName = "hytalerl_" + UUID.randomUUID().toString().replace("-", "");
        WorldConfig config = new WorldConfig();
        config.setUuid(UUID.randomUUID());
        config.setDisplayName("HytaleRL " + worldName);
        config.setSeed(seed);
        config.setSpawnProvider(new GlobalSpawnProvider(
            new Transform(spawnX, spawnY, spawnZ, 0.0f, 0.0f, 0.0f)
        ));
        if (options.world().equals("hytale")) {
            HytaleWorldGenProvider worldgen = new HytaleWorldGenProvider();
            config.setWorldGenProvider(worldgen);
            worldgenProvider = HytaleWorldGenProvider.ID;
            worldgenVersion = worldgen.getVersion().toString();
        } else if (options.world().equals("hytale_generator")) {
            HytaleGenerator plugin = HytaleGenerator.get();
            if (plugin == null || plugin.getAssetManager() == null) {
                throw new IllegalStateException(
                    "HytaleGenerator plugin is not ready"
                );
            }
            String worldStructure = options.worldgenStructure();
            if (
                plugin.getAssetManager().getWorldStructureAsset(worldStructure)
                    == null
            ) {
                throw new IllegalArgumentException(
                    "HytaleGenerator world structure not found: "
                        + worldStructure
                );
            }
            HandleProvider worldgen = new HandleProvider(
                plugin,
                HYTALE_GENERATOR_WORLD_COUNTER.getAndIncrement()
            );
            worldgen.setWorldStructureName(worldStructure);
            config.setWorldGenProvider(worldgen);
            worldgenProvider = HandleProvider.ID;
            worldgenVersion = NATIVE_SERVER_VERSION;
            worldgenWorldStructure = worldStructure;
        } else {
            worldgenProvider = "Flat";
            worldgenVersion = "fixture-v1";
            config.setWorldGenProvider(new FlatWorldGenProvider(
                FlatWorldGenProvider.DEFAULT_TINT,
                new Layer[] {
                    new Layer(0, 64, "Default", "Rock_Stone"),
                    new Layer(64, 65, "Default", "Soil_Grass")
                }
            ));
        }
        config.setWorldMapProvider(new DisabledWorldMapProvider());
        config.setChunkStorageProvider(EmptyChunkStorageProvider.INSTANCE);
        config.setResourceStorageProvider(EmptyResourceStorageProvider.INSTANCE);
        config.setTicking(true);
        config.setBlockTicking(options.blockTickingEnabled());
        config.setPvpEnabled(false);
        config.setFallDamageEnabled(true);
        config.setGameTimePaused(true);
        config.setSpawningNPC(false);
        config.setIsSpawnMarkersEnabled(false);
        config.setIsAllNPCFrozen(false);
        config.setCompassUpdating(false);
        config.setObjectiveMarkersEnabled(false);
        config.setSavingPlayers(false);
        config.setSavingConfig(false);
        config.setCanUnloadChunks(false);
        config.setCanSaveChunks(false);
        config.setSaveNewChunks(false);
        // Empty storage plus savingConfig=false creates no world directory.
        config.setDeleteOnUniverseStart(false);
        config.setDeleteOnRemove(false);

        try {
            world = Universe.get().makeWorld(
                worldName,
                Universe.get().getWorldsPath().resolve(worldName),
                config
            ).get(INITIALIZATION_TIMEOUT.toMillis(), TimeUnit.MILLISECONDS);
            runOnWorld(
                () -> {
                    world.setTps(options.nativeTickRate());
                    World.setTimeDilation(
                        options.nativeTimeDilation(),
                        world.getEntityStore().getStore()
                    );
                },
                INITIALIZATION_TIMEOUT
            );
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while creating native world", exception);
        } catch (ExecutionException | TimeoutException exception) {
            throw new IllegalStateException("Failed to create native Hytale world", exception);
        }
    }

    private void pauseWorld() {
        runOnWorld(() -> world.setPaused(true), INITIALIZATION_TIMEOUT);
    }

    private void installHeadlessTickAnchor() {
        // World.isIdle() is true when no PlayerRef is tracked. In 0.5.7 an idle
        // TickingThread skips its 30 TPS limiter and supplies tiny wall-clock
        // deltas to ECS systems, unlike a live world. This reference is not an
        // entity/player and is untracked before the ephemeral world is removed.
        tickAnchor = new PlayerRef(
            null,
            UUID.randomUUID(),
            "HytaleRLHeadlessTickAnchor",
            "en-US",
            new HeadlessPacketHandler(this::captureHeadlessPacket),
            new ChunkTracker()
        );
        runOnWorld(() -> world.trackPlayerRef(tickAnchor), INITIALIZATION_TIMEOUT);
    }

    private void captureHeadlessPacket(ToClientPacket packet) {
        PendingStep step = pending.get();
        if (step == null) return;
        double offsetSeconds = step.simulatedSeconds;
        if (packet instanceof PlaySoundEvent2D sound) {
            int category = sound.category == null ? Integer.MIN_VALUE : sound.category.getValue();
            step.audio.captureNonSpatial(
                AudioFrame.KIND_2D,
                sound.soundEventIndex,
                category,
                sound.volumeModifier,
                sound.pitchModifier,
                offsetSeconds
            );
        } else if (packet instanceof PlaySoundEventEntity sound) {
            step.audio.captureNonSpatial(
                AudioFrame.KIND_ENTITY,
                sound.soundEventIndex,
                AudioFrame.CATEGORY_NONE,
                sound.volumeModifier,
                sound.pitchModifier,
                offsetSeconds
            );
        }
    }

    private void setWorldPausedWithoutAnchorBroadcast(boolean paused) {
        PlayerRef anchor = tickAnchor;
        if (anchor == null) {
            world.setPaused(paused);
            return;
        }
        world.untrackPlayerRef(anchor);
        try {
            world.setPaused(paused);
        } finally {
            world.trackPlayerRef(anchor);
        }
    }

    private void loadAndPinSpawnChunks() {
        loadAndPinGeometryChunkHalo(spawnX, spawnZ);
    }

    private void loadAndPinGeometryChunkHalo(double blockX, double blockZ) {
        int centerX = ChunkUtil.chunkCoordinate(blockX);
        int centerZ = ChunkUtil.chunkCoordinate(blockZ);
        long center = ChunkUtil.indexChunk(centerX, centerZ);
        if (geometryChunkHaloCenter == center) {
            return;
        }
        List<Long> indices = GeometryChunkCoverage.indices(blockX, blockZ);
        GeometryChunkCoverage.requirePinCapacity(
            geometryChunkIndices,
            indices
        );
        loadAndPinChunkIndices(indices);
        geometryChunkIndices.addAll(indices);
        geometryChunkHaloCenter = center;
    }

    private void loadAndPinRegion(NativeRegionManifest manifest) {
        List<Long> indices = new ArrayList<>(
            NativeRegionManifest.CAPTURE_CHUNKS_PER_AXIS
                * NativeRegionManifest.CAPTURE_CHUNKS_PER_AXIS
        );
        for (
            int dx = 0;
            dx < NativeRegionManifest.CAPTURE_CHUNKS_PER_AXIS;
            dx++
        ) {
            for (
                int dz = 0;
                dz < NativeRegionManifest.CAPTURE_CHUNKS_PER_AXIS;
                dz++
            ) {
                indices.add(
                    ChunkUtil.indexChunk(
                        manifest.captureMinChunkX() + dx,
                        manifest.captureMinChunkZ() + dz
                    )
                );
            }
        }
        loadAndPinChunkIndices(indices);
    }

    void captureNpcTraceDamage(
        Ref<EntityStore> target,
        Store<EntityStore> store,
        Damage damage
    ) {
        npcTraceRecorder.captureDamage(target, store, damage);
    }

    private void loadAndPinRegionLightHalo(NativeRegionManifest manifest) {
        if (manifest == null) {
            throw new IllegalStateException(
                "Region manifest is unavailable for native light capture"
            );
        }
        if (regionLightHaloManifest == manifest) return;
        List<Long> indices = RegionLightCoverage.indices(manifest);
        loadAndPinChunkIndices(indices);
        runOnWorld(() -> {
            for (long index : indices) {
                int chunkX = ChunkUtil.xOfChunkIndex(index);
                int chunkZ = ChunkUtil.zOfChunkIndex(index);
                WorldChunk chunk = world.getChunkIfLoaded(index);
                if (chunk == null) {
                    throw new IllegalStateException(
                        "Pinned Region lighting halo chunk is not loaded"
                    );
                }
                for (int sectionY = 0; sectionY < 10; sectionY++) {
                    RegionLightCapture.invalidateEmptyPlaceholders(
                        chunk.getBlockChunk().getSectionAtIndex(sectionY)
                    );
                    world.getChunkLighting().addToQueue(
                        new Vector3i(chunkX, sectionY, chunkZ)
                    );
                }
            }
        }, INITIALIZATION_TIMEOUT);
        regionLightHaloManifest = manifest;
    }

    private void loadAndPinChunkIndices(List<Long> indices) {
        LinkedHashMap<Long, CompletableFuture<Ref<ChunkStore>>> futures =
            new LinkedHashMap<>();
        try {
            for (long index : indices) {
                if (loadedChunkIndices.contains(index)) continue;
                CompletableFuture<Ref<ChunkStore>> future = world
                    .getChunkStore()
                    .getChunkReferenceAsync(index);
                futures.put(index, future);
            }
            if (futures.isEmpty()) return;
            CompletableFuture.allOf(
                futures.values().toArray(CompletableFuture[]::new)
            )
                .get(
                    INITIALIZATION_TIMEOUT.toMillis(),
                    TimeUnit.MILLISECONDS
                );
            for (
                Map.Entry<Long, CompletableFuture<Ref<ChunkStore>>> entry
                    : futures.entrySet()
            ) {
                loadedChunkIndices.add(entry.getKey());
                loadedChunks.add(entry.getValue().join());
            }
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while loading native chunks", exception);
        } catch (ExecutionException | TimeoutException exception) {
            throw new IllegalStateException(
                "Failed to generate native Hytale chunks",
                exception
            );
        }

        runOnWorld(() -> {
            Store<ChunkStore> store = world.getChunkStore().getStore();
            for (CompletableFuture<Ref<ChunkStore>> future : futures.values()) {
                Ref<ChunkStore> ref = future.join();
                WorldChunk chunk = store.getComponent(ref, WorldChunk.getComponentType());
                if (chunk == null) throw new IllegalStateException("Loaded chunk has no WorldChunk");
                chunk.addKeepLoaded();
                chunk.setFlag(ChunkFlag.TICKING, true);
            }
        }, INITIALIZATION_TIMEOUT);
    }

    private void resolveGeneratedSpawnHeight() {
        if (!usesGeneratedWorld() || options.hasSpawn()) return;
        runOnWorld(() -> {
            int originX = (int) Math.floor(spawnX);
            int originZ = (int) Math.floor(spawnZ);
            for (int radius = 0; radius <= 24; radius++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    for (int dz = -radius; dz <= radius; dz++) {
                        if (radius > 0 && Math.abs(dx) != radius && Math.abs(dz) != radius) {
                            continue;
                        }
                        double candidateX = originX + dx + 0.5;
                        double candidateZ = originZ + dz + 0.5;
                        double candidateY = generatedSurfaceHeight(candidateX, candidateZ);
                        if (!isClearGeneratedSpawn(candidateX, candidateY, candidateZ)) continue;
                        spawnX = candidateX;
                        spawnY = candidateY;
                        spawnZ = candidateZ;
                        return;
                    }
                }
            }
            throw new IllegalStateException(
                "No clear 3x3 native-generated spawn surface found within 24 blocks"
            );
        }, INITIALIZATION_TIMEOUT);
    }

    private boolean isClearGeneratedSpawn(double x, double y, double z) {
        int baseX = (int) Math.floor(x);
        int baseY = (int) Math.floor(y);
        int baseZ = (int) Math.floor(z);
        for (int dx = -1; dx <= 1; dx++) {
            for (int dz = -1; dz <= 1; dz++) {
                if ((int) generatedSurfaceHeight(baseX + dx + 0.5, baseZ + dz + 0.5)
                    != baseY) return false;
                if (world.getBlock(baseX + dx, baseY - 1, baseZ + dz) == 0) return false;
                for (int dy = 0; dy <= 2; dy++) {
                    if (world.getBlock(baseX + dx, baseY + dy, baseZ + dz) != 0) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    private double generatedSurfaceHeight(double x, double z) {
        WorldChunk chunk = world.getChunkIfLoaded(ChunkUtil.indexChunkFromBlock(x, z));
        if (chunk == null) {
            throw new IllegalStateException("Generated surface chunk is not loaded");
        }
        return chunk.getHeight(
            ChunkUtil.localCoordinate((long) Math.floor(x)),
            ChunkUtil.localCoordinate((long) Math.floor(z))
        ) + 1.0;
    }

    private void spawnAgent(NPCPlugin npcPlugin) {
        runOnWorld(() -> {
            Store<EntityStore> store = world.getEntityStore().getStore();
            Pair<Ref<EntityStore>, ?> spawned = npcPlugin.spawnNPC(
                store,
                options.npcRole(),
                null,
                new Vector3d(spawnX, spawnY, spawnZ),
                new Rotation3f(0.0f, 0.0f, 0.0f)
            );
            if (spawned == null || spawned.left() == null) {
                throw new IllegalStateException("Hytale rejected native NPC spawn");
            }
            agentRef = spawned.left();
            NPCEntity npc = store.getComponent(agentRef, NPCEntity.getComponentType());
            if (npc == null) throw new IllegalStateException("Spawned NPC has no NPCEntity component");
            npc.addReservation(reservationId);
            npc.setDespawnTime(Float.POSITIVE_INFINITY);
            store.addComponent(
                agentRef,
                markerType,
                new NativeAgentMarker(this, 0)
            );
            policyActorIdentities[0] = policyActorIdentity(0, store);
            replaceNativeAgentArmor(npc);
            equipNativeCombatItem(npc, store);
            if (options.hasNativeWorldHotbarOverride()) {
                equipNativeWorldHotbar(npc, store);
            }
            if (syntheticWorldInteractionsEnabled()) {
                equipSyntheticWorldItems(npc, store);
            }
            if (installHeadlessWorldVerbContextAtReset()) {
                installSyntheticPlayer(0, agentRef, store);
            }

            TransformComponent transform = store.getComponent(
                agentRef,
                TransformComponent.getComponentType()
            );
            primeNpcTracePerception(transform);
            HeadRotation head = store.getComponent(agentRef, HeadRotation.getComponentType());
            policyActorState(0).initialize(
                transform == null ? 0.0f : transform.getRotation().yaw(),
                head == null ? 0.0f : head.getRotation().pitch(),
                discoverAttackActions(npc.getRole())
            );
            resetTargetAttackTelemetry();
        }, INITIALIZATION_TIMEOUT);
    }

    private void spawnTaskEntities(NPCPlugin npcPlugin) {
        if (!taskId.equals("kill_trork")) return;
        String targetRole = combatTargetRole();
        if (!npcPlugin.hasRoleName(targetRole)) {
            throw new IllegalStateException(
                "Native target role not found: " + targetRole
            );
        }
        npcPlugin.validateSpawnableRole(targetRole);

        NativeSnapshot settledAgentSnapshot = lastSnapshot;
        if (settledAgentSnapshot == null) {
            throw new IllegalStateException(
                "Cannot spawn a combat target before agent warm-up completes"
            );
        }
        double settledAgentY = settledAgentSnapshot.observation().y();
        boolean diagonalLos = isDiagonalLosFixture(options.fidelityFixture());
        boolean targetMemory = targetMemoryFixture != null;
        double targetSpawnX = spawnX + (
            diagonalLos
                ? DIAGONAL_LOS_TARGET_OFFSET_X
                : KILL_TRORK_TARGET_OFFSET_X
        );
        double targetSpawnZ = spawnZ + (
            diagonalLos
                ? DIAGONAL_LOS_TARGET_OFFSET_Z
                : targetMemory
                    ? TARGET_MEMORY_TARGET_OFFSET_Z
                    : nativeProjectileDuelFixture()
                        ? NATIVE_PROJECTILE_TARGET_OFFSET_Z
                        : nativeDuelFixture()
                            ? NATIVE_DUEL_TARGET_OFFSET_Z
                        : options.fidelityFixture().equals(
                            EnvironmentOptions.PATH_FOLLOWER_FIXTURE
                        )
                            ? PATH_FOLLOWER_TARGET_OFFSET_Z
                            : KILL_TRORK_TARGET_OFFSET_Z
        );
        boolean faceAgent = targetMemory
            || nativeBehaviorMatchup()
            || options.fidelityFixture().equals(
                EnvironmentOptions.PATH_FOLLOWER_FIXTURE
            );
        loadAndPinGeometryChunkHalo(targetSpawnX, targetSpawnZ);

        runOnWorld(() -> {
            Store<EntityStore> store = world.getEntityStore().getStore();
            double targetSpawnY = usesGeneratedWorld()
                ? generatedSurfaceHeight(targetSpawnX, targetSpawnZ)
                : CombatSpawnSupport.flatTargetY(
                    settledAgentY,
                    KILL_TRORK_TARGET_OFFSET_Y
                );
            Pair<Ref<EntityStore>, ?> spawned = npcPlugin.spawnNPC(
                store,
                targetRole,
                null,
                new Vector3d(targetSpawnX, targetSpawnY, targetSpawnZ),
                faceAgent
                    ? Rotation3f.lookAt(
                        new Vector3d(
                            spawnX - targetSpawnX,
                            0.0,
                            spawnZ - targetSpawnZ
                        )
                    )
                    : new Rotation3f(0.0f, 0.0f, 0.0f)
            );
            if (spawned == null || spawned.left() == null) {
                throw new IllegalStateException("Hytale rejected native combat target spawn");
            }
            targetRef = spawned.left();
            NPCEntity target = store.getComponent(targetRef, NPCEntity.getComponentType());
            if (target == null) {
                throw new IllegalStateException("Spawned combat target has no NPCEntity component");
            }
            target.addReservation(reservationId);
            target.setDespawnTime(Float.POSITIVE_INFINITY);
            store.addComponent(
                targetRef,
                markerType,
                new NativeAgentMarker(this, 1)
            );
            policyActorIdentities[1] = policyActorIdentity(1, store);
            NativePolicyCombatBindingSpec combatBinding =
                nativePolicyCombatSpecs[1];
            if (combatBinding != null) {
                equipNativeCombatItem(
                    targetRef,
                    target,
                    store,
                    combatBinding.itemId()
                );
            }
            TransformComponent transform = store.getComponent(
                targetRef,
                TransformComponent.getComponentType()
            );
            primeNpcTracePerception(transform);
            HeadRotation head = store.getComponent(
                targetRef,
                HeadRotation.getComponentType()
            );
            policyActorState(1).initialize(
                transform == null ? 0.0f : transform.getRotation().yaw(),
                head == null ? 0.0f : head.getRotation().pitch(),
                discoverAttackActions(target.getRole())
            );
            if (installHeadlessWorldVerbContextAtReset()) {
                installSyntheticPlayer(1, targetRef, store);
            }
            initializeNativePolicyCombat(1, targetRef, target, store);
        }, INITIALIZATION_TIMEOUT);
    }

    private void activateTaskEntities() {
        if (!taskId.equals("kill_trork") || !options.combatTargetActive()) return;
        runOnWorld(() -> {
            Ref<EntityStore> targetReference = targetRef;
            Ref<EntityStore> agentReference = agentRef;
            if (targetReference == null || !targetReference.isValid()
                || agentReference == null || !agentReference.isValid()) return;
            Store<EntityStore> store = world.getEntityStore().getStore();
            NPCEntity target = store.getComponent(targetReference, NPCEntity.getComponentType());
            Role role = target == null ? null : target.getRole();
            if (role == null) return;
            if (!nativeProjectileDuelFixture()) {
                role.setMarkedTarget("LockedTarget", agentReference);
            }
            pinNativeDuelTargets(store);
            if (options.hasCombatTargetRole()) return;
            StateSupport stateSupport = role.getStateSupport();
            if (stateSupport != null) {
                if (nativeProjectileDuelFixture()) {
                    stateSupport.setState(targetReference, "Idle", null, store);
                } else {
                    stateSupport.setState(
                        targetReference,
                        "Chase",
                        "Attack",
                        store
                    );
                }
            }
        }, INITIALIZATION_TIMEOUT);
    }

    private void initializeNativePolicyCombat(
        int actorId,
        Ref<EntityStore> reference,
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        Object handle = nativePolicyCombatHandles[actorId];
        if (handle == null) return;
        NativePolicyCombatFacade.Receipt receipt =
            NativePolicyCombatFacade.initialize(
                handle,
                reference,
                npc,
                store
            );
        if (!receipt.available()) {
            throw new IllegalStateException(
                "Native policy-combat actor " + actorId
                    + " failed initialization: "
                    + receipt.unavailableReason()
            );
        }
    }

    private void requestNativeBlockUse(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        PendingStep step
    ) {
        NativePolicyActorState actorState = policyActorState(0);
        if (
            trackedAgentUseChain != null
                && trackedAgentUseChain.getServerState()
                    == InteractionState.NotFinished
        ) {
            step.useRejectReason = "native_use_chain_active";
            return;
        }
        NativeBlockUse.StartResult result = NativeBlockUse.start(
            ref,
            store,
            world,
            interactionManager(ref, store),
            actorState.desiredPitch(),
            actorState.desiredYaw(),
            headlessWorldVerbsEnabled()
        );
        if (!result.accepted()) {
            step.useRejectReason = result.rejectReason();
            return;
        }

        trackedAgentUseChain = result.chain();
        trackedAgentUseInteractionId = result.interactionId();
        trackedAgentUseBlockInteractionId = result.blockInteractionId();
        trackedAgentUseTarget = result.target();
        trackedAgentUseMaximumDistance = result.maximumDistance();
        trackedAgentUseStartTick = -1L;
        trackedAgentUseFinishTick = -1L;
        step.useAccepted = true;
    }

    private void requestLegacyActorZeroNativeWorldVerb(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        AgentAction action,
        PendingStep step
    ) {
        NativeWorldVerbRequest request = action.nativeWorldVerbRequest();
        if (!request.present()) return;
        long requestedTick = world.getTick();
        if (!request.candidateEvidenceBound()) {
            step.legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.rejected(
                    request,
                    NativeWorldVerbRequest.POLICY_EVIDENCE_REQUIRED,
                    requestedTick
                );
            return;
        }
        if (!request.worldEpoch().equals(reservationId.toString())) {
            step.legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.rejected(
                    request,
                    "stale_world_epoch",
                    requestedTick
                );
            return;
        }
        if (legacyActorZeroTypedWorldVerbExecution.pending()) {
            step.legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.rejected(
                    request,
                    "native_world_verb_busy",
                    requestedTick
                );
            return;
        }
        if (request.candidateEvidenceBound()) {
            NativePolicyWorldActionCommitVerifier.Verification verification;
            try {
                NativePolicyWorldActionCapture fresh =
                    buildPolicyWorldActionCapture(
                        PolicyWorldActionCaptureRequest.observe(
                            0,
                            policyActorIdentity(0, store)
                        )
                    );
                verification = NativePolicyWorldActionCommitVerifier.verify(
                    request,
                    fresh
                );
            } catch (RuntimeException captureFailure) {
                step.legacyActorZeroWorldVerbExecution =
                    WorldVerbTelemetry.Execution.rejected(
                        request,
                        "native_policy_world_action_capture_unavailable",
                        requestedTick
                    );
                return;
            }
            if (!verification.accepted()) {
                step.legacyActorZeroWorldVerbExecution =
                    WorldVerbTelemetry.Execution.rejected(
                        request,
                        verification.rejectReason(),
                        requestedTick
                    );
                return;
            }
        }
        if (
            request.verb().equals("place_block")
                || request.verb().equals("break_block")
        ) {
            requestLegacyActorZeroTypedBlockWorldInteraction(
                ref,
                npc,
                store,
                request,
                step,
                requestedTick
            );
            return;
        }
        if (request.verb().equals("craft_recipe")) {
            requestLegacyActorZeroTypedCraft(
                ref,
                store,
                request,
                step,
                requestedTick
            );
            return;
        }
        if (!request.verb().equals("use")) {
            step.legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.rejected(
                    request,
                    "typed_" + request.verb()
                        + "_execution_not_implemented",
                    requestedTick
                );
            return;
        }
        NativeWorldVerbUse.StartResult result = NativeWorldVerbUse.start(
            ref,
            npc,
            store,
            world,
            request,
            headlessWorldVerbsEnabled()
        );
        if (!result.accepted()) {
            step.legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.rejected(
                    request,
                    result.rejectReason(),
                    requestedTick
                );
            return;
        }
        boolean unarmed = result.unarmed();
        WorldVerbCell before = result.before();
        ItemStack resolvedSource = result.resolvedSource();

        WorldVerbTelemetry.Execution execution =
            WorldVerbTelemetry.Execution.accepted(
                request,
                candidateExecutionScope(
                    request,
                    "headless_native_target_relative_use_chain"
                ),
                requestedTick,
                before.semanticBlockId(),
                before.runtimeBlockId(),
                unarmed ? 0 : resolvedSource.getQuantity(),
                -1,
                options.nativeWorldVerbs()
            );
        legacyActorZeroTypedWorldVerbExecution = execution;
        step.legacyActorZeroWorldVerbExecution = execution;
        legacyActorZeroTypedWorldVerbChain = result.chain();
        legacyActorZeroTypedWorldVerbGeometryBefore = before.canonical();
        legacyActorZeroTypedWorldVerbInventoryBefore =
            unarmed
                ? "unarmed"
                : canonicalWorldVerbInventoryState(resolvedSource);
        if (headlessWorldVerbsEnabled()) {
            syntheticWorldChain = result.chain();
            syntheticWorldEvidenceRow = null;
            syntheticWorldVerb = "use";
            syntheticWorldTarget = new Vector3i(
                request.targetX(),
                request.targetY(),
                request.targetZ()
            );
            syntheticWorldBlockBefore = before.runtimeBlockId();
            syntheticWorldHealthBefore = before.blockHealth();
            syntheticWorldExpectedBlockId = before.runtimeBlockId();
            syntheticWorldBlockFace = BlockFace.fromValue(
                request.blockFace()
            );
            syntheticWorldRotationYaw = Rotation.fromValue(
                request.rotationYaw()
            );
            syntheticWorldRotationPitch = Rotation.fromValue(
                request.rotationPitch()
            );
            syntheticWorldRotationRoll = Rotation.fromValue(
                request.rotationRoll()
            );
            syntheticWorldOperationIndex = -1;
            syntheticWorldClientElapsedSeconds = 0.0;
            syntheticWorldStarted = false;
        }
    }

    /**
     * Start the whole typed World-verb batch once every requesting actor has
     * received its first control callback. ECS chunk iteration order is not
     * an action-order contract, so actor-local admission and target
     * reservation happen here in ascending policy-actor order.
     */
    private void requestPolicyNativeWorldVerbsWhenReady(
        PendingStep step,
        Store<EntityStore> store
    ) {
        if (step.worldVerbBatchStarted) return;
        List<PolicyWorldVerbArbitrator.Entry> entries = new ArrayList<>();
        for (var actorAction : step.actions.actors()) {
            NativeWorldVerbRequest request =
                actorAction.action().nativeWorldVerbRequest();
            if (!request.present()) continue;
            int actorId = actorAction.entityId();
            if (step.controlTicks(actorId) == 0) return;
            entries.add(new PolicyWorldVerbArbitrator.Entry(
                actorId,
                policyActorIdentity(actorId, store),
                request
            ));
        }
        if (entries.isEmpty()) return;
        step.worldVerbBatchStarted = true;
        Map<Integer, PolicyWorldVerbArbitrator.Decision> decisions =
            PolicyWorldVerbArbitrator.arbitrate(
                entries,
                entry -> attemptPolicyNativeWorldVerb(entry, step, store)
            );
        for (PolicyWorldVerbArbitrator.Entry entry : entries) {
            PolicyWorldVerbArbitrator.Decision decision = decisions.get(
                entry.actorSlot()
            );
            if (decision == null) {
                throw new IllegalStateException(
                    "Typed World-verb actor is absent from arbitration result"
                );
            }
            if (!decision.admitted()) {
                step.rejectWorldVerb(
                    entry.actorSlot(),
                    entry.request(),
                    decision.rejectReason()
                );
            }
        }
    }

    private PolicyWorldVerbArbitrator.Admission attemptPolicyNativeWorldVerb(
        PolicyWorldVerbArbitrator.Entry entry,
        PendingStep step,
        Store<EntityStore> store
    ) {
        int actorId = entry.actorSlot();
        NativeWorldVerbRequest request = entry.request();
        if (!request.candidateEvidenceBound()) {
            return PolicyWorldVerbArbitrator.Admission.rejected(
                NativeWorldVerbRequest.POLICY_EVIDENCE_REQUIRED
            );
        }
        if (!request.worldEpoch().equals(reservationId.toString())) {
            return PolicyWorldVerbArbitrator.Admission.rejected(
                "stale_world_epoch"
            );
        }
        if (nativePolicyWorldVerbHandlesByActor[actorId] != null) {
            return PolicyWorldVerbArbitrator.Admission.rejected(
                "native_world_verb_actor_busy"
            );
        }
        NativePolicyWorldActionCommitVerifier.Verification verification;
        try {
            NativePolicyWorldActionCapture fresh =
                buildPolicyWorldActionCapture(
                    PolicyWorldActionCaptureRequest.observe(
                        actorId,
                        entry.actorIdentity()
                    )
                );
            verification = NativePolicyWorldActionCommitVerifier.verify(
                request,
                fresh
            );
        } catch (RuntimeException captureFailure) {
            return PolicyWorldVerbArbitrator.Admission.rejected(
                "native_policy_world_action_capture_unavailable"
            );
        }
        if (!verification.accepted()) {
            return PolicyWorldVerbArbitrator.Admission.rejected(
                verification.rejectReason()
            );
        }
        Ref<EntityStore> actor = policyActorRef(actorId);
        NativePolicyWorldVerbFacade.Admission admission =
            NativePolicyWorldVerbFacade.start(
                actor,
                store,
                request,
                reservationId.toString()
            );
        if (!admission.accepted()) {
            return PolicyWorldVerbArbitrator.Admission.rejected(
                admission.rejectReason()
            );
        }
        nativePolicyWorldVerbHandlesByActor[actorId] = admission.handle();
        nativePolicyWorldVerbRequestsByActor[actorId] = request;
        step.observeWorldVerb(actorId, request, admission.lifecycle());
        if (
            admission.lifecycle().finished()
                || admission.lifecycle().failed()
        ) {
            nativePolicyWorldVerbHandlesByActor[actorId] = null;
            nativePolicyWorldVerbRequestsByActor[actorId] = null;
        }
        return PolicyWorldVerbArbitrator.Admission.admittedResult();
    }

    private String policyActorIdentity(
        int actorId,
        Store<EntityStore> store
    ) {
        Ref<EntityStore> actor = policyActorRef(actorId);
        UUIDComponent uuid = actor == null || !actor.isValid()
            ? null
            : store.getComponent(actor, UUIDComponent.getComponentType());
        if (uuid == null || uuid.getUuid() == null) {
            throw new IllegalStateException(
                "Policy actor has no authoritative UUID: " + actorId
            );
        }
        return uuid.getUuid().toString();
    }

    private NativeRegionLightSection buildRegionLightSection(
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        long chunkIndex = ChunkUtil.indexChunk(chunkX, chunkZ);
        if (!loadedChunkIndices.contains(chunkIndex)) {
            throw new IllegalStateException(
                "Region light chunk was not pinned before capture"
            );
        }
        WorldChunk chunk = world.getChunkIfLoaded(chunkIndex);
        if (chunk == null) {
            throw new IllegalStateException(
                "Region light chunk is not loaded"
            );
        }
        NativeRegionLightSection result = RegionLightCapture.capture(
            chunk,
            chunkX,
            chunkZ,
            sectionY
        );
        if (!result.available()) {
            world.getChunkLighting().addToQueue(
                new Vector3i(chunkX, sectionY, chunkZ)
            );
        }
        return result;
    }

    private void requestLegacyActorZeroTypedCraft(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        NativeWorldVerbRequest request,
        PendingStep step,
        long requestedTick
    ) {
        NativeWorldVerbCraft.StartResult result =
            NativeWorldVerbCraft.start(
                ref,
                store,
                world,
                request,
                options.nativeWorldVerbs()
            );
        if (!result.accepted()) {
            step.legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.rejected(
                    request,
                    result.rejectReason(),
                    requestedTick
                );
            return;
        }

        WorldVerbTelemetry.Execution execution =
            WorldVerbTelemetry.Execution.accepted(
                request,
                candidateExecutionScope(
                    request,
                    "opt_in_headless_player_native_crafting_manager"
                ),
                requestedTick,
                "not_applicable",
                0,
                0,
                result.outputQuantityBefore(),
                true
            );
        legacyActorZeroTypedWorldVerbExecution = execution;
        step.legacyActorZeroWorldVerbExecution = execution;
        legacyActorZeroTypedWorldVerbGeometryBefore = "craft_recipe";
        legacyActorZeroTypedWorldVerbInventoryBefore = result.inventoryBefore();

        if (result.queued()) {
            legacyActorZeroTypedWorldVerbCraftingManager = result.manager();
            legacyActorZeroTypedWorldVerbCraftingRecipe = result.recipe();
            return;
        }

        execution.observeStarted(requestedTick);
        String geometrySha256 = sha256Hex(
            "hytalerl_world_verb_geometry_delta_v1\n"
                + legacyActorZeroTypedWorldVerbGeometryBefore
                + "\n"
                + legacyActorZeroTypedWorldVerbGeometryBefore
        );
        String inventorySha256 = sha256Hex(
            "hytalerl_world_verb_inventory_delta_v1\n"
                + request.sourceContainer()
                + "\n-1\n"
                + result.inventoryBefore()
                + "\n"
                + result.inventoryAfter()
        );
        execution.observeFinished(
            false,
            requestedTick,
            "not_applicable",
            0,
            0,
            result.outputQuantityAfter(),
            false,
            !result.inventoryBefore().equals(result.inventoryAfter()),
            geometrySha256,
            inventorySha256,
            ""
        );
        legacyActorZeroTypedWorldVerbGeometryBefore = "";
        legacyActorZeroTypedWorldVerbInventoryBefore = "";
    }

    private static String candidateExecutionScope(
        NativeWorldVerbRequest request,
        String scope
    ) {
        if (!request.candidateEvidenceBound()) {
            throw new IllegalArgumentException(
                NativeWorldVerbRequest.POLICY_EVIDENCE_REQUIRED
            );
        }
        return "candidate_bound_" + scope;
    }

    private void requestLegacyActorZeroTypedBlockWorldInteraction(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        NativeWorldVerbRequest request,
        PendingStep step,
        long requestedTick
    ) {
        boolean place = request.verb().equals("place_block");
        boolean fixture = syntheticWorldInteractionsEnabled();
        /*
         * Preserve the original admission order: a session without the
         * explicit headless context rejects before it inspects or clears any
         * synthetic-chain lifecycle state. NativeWorldVerbBlock repeats this
         * check because autonomous callers must also fail closed.
         */
        if (!headlessWorldVerbsEnabled()) {
            step.legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.rejected(
                    request,
                    "typed_" + request.verb()
                        + "_requires_headless_world_verb_context",
                    requestedTick
                );
            return;
        }
        if (
            syntheticWorldChain != null
                && syntheticWorldChain.getServerState()
                    == InteractionState.NotFinished
        ) {
            step.legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.rejected(
                    request,
                    "synthetic_world_interaction_busy",
                    requestedTick
                );
            return;
        }
        clearSyntheticWorldInteractionRun();

        Vector3i target = new Vector3i(
            request.targetX(),
            request.targetY(),
            request.targetZ()
        );
        Vector3i expectedTarget = fixture
            ? syntheticFixtureTarget(place)
            : target;
        String expectedInteraction = fixture
            ? place
                ? SYNTHETIC_PLACE_INTERACTION_ID
                : SYNTHETIC_BREAK_INTERACTION_ID
            : request.interactionId();
        String expectedItem = fixture
            ? place
                ? SYNTHETIC_PLACE_ITEM_ID
                : SYNTHETIC_BREAK_TOOL_ITEM_ID
            : request.itemId();
        int expectedSlot = fixture
            ? place ? 0 : 1
            : request.sourceSlot();
        int expectedInteractionType = fixture
            ? place ? 1 : 0
            : request.interactionType();
        String expectedBlockBefore = fixture
            ? place
                ? BlockType.EMPTY_KEY
                : SYNTHETIC_PLACE_BLOCK_ID
            : request.expectedBlockId();
        NativeWorldVerbBlock.StartResult result =
            NativeWorldVerbBlock.start(
                ref,
                npc,
                store,
                world,
                request,
                headlessWorldVerbsEnabled(),
                new NativeWorldVerbBlock.Plan(
                    fixture,
                    target,
                    expectedTarget,
                    expectedInteraction,
                    expectedItem,
                    expectedSlot,
                    expectedInteractionType,
                    expectedBlockBefore,
                    SYNTHETIC_PLACE_BLOCK_ID
                )
            );
        if (!result.accepted()) {
            step.legacyActorZeroWorldVerbExecution =
                WorldVerbTelemetry.Execution.rejected(
                    request,
                    result.rejectReason(),
                    requestedTick
                );
            return;
        }
        InteractionChain chain = result.chain();
        ItemStack source = result.source();
        WorldVerbCell before = result.before();
        int placedRuntimeBlockId = result.placedRuntimeBlockId();

        WorldVerbTelemetry.Execution execution =
            WorldVerbTelemetry.Execution.accepted(
                request,
                candidateExecutionScope(
                    request,
                    (fixture ? "fixture_" : "opt_in_")
                        + "headless_player_native_"
                        + (place ? "place" : "break")
                        + "_chain"
                ),
                requestedTick,
                before.semanticBlockId(),
                before.runtimeBlockId(),
                source.getQuantity(),
                -1,
                options.nativeWorldVerbs()
            );
        legacyActorZeroTypedWorldVerbExecution = execution;
        step.legacyActorZeroWorldVerbExecution = execution;
        legacyActorZeroTypedWorldVerbChain = chain;
        legacyActorZeroTypedWorldVerbGeometryBefore = before.canonical();
        legacyActorZeroTypedWorldVerbInventoryBefore =
            canonicalWorldVerbInventoryState(source);

        syntheticWorldChain = chain;
        syntheticWorldEvidenceRow = null;
        syntheticWorldVerb = request.verb();
        syntheticWorldTarget = target;
        syntheticWorldBlockBefore = before.runtimeBlockId();
        syntheticWorldHealthBefore = before.blockHealth();
        syntheticWorldExpectedBlockId = place
            ? placedRuntimeBlockId
            : before.runtimeBlockId();
        syntheticWorldBlockFace = BlockFace.fromValue(request.blockFace());
        syntheticWorldRotationYaw = Rotation.fromValue(
            request.rotationYaw()
        );
        syntheticWorldRotationPitch = Rotation.fromValue(
            request.rotationPitch()
        );
        syntheticWorldRotationRoll = Rotation.fromValue(
            request.rotationRoll()
        );
        syntheticWorldOperationIndex = -1;
        syntheticWorldClientElapsedSeconds = 0.0;
        syntheticWorldStarted = false;
    }

    private NativeAttackAdmission requestNativeAttack(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        double deltaTime,
        double requestedChargeTime,
        int actorId,
        NativePolicyActorState actorState
    ) {
        Role role = npc.getRole();
        CombatSupport combat = role == null ? null : role.getCombatSupport();
        if (role == null || combat == null) {
            return NativeAttackAdmission.rejected("native_combat_busy_or_unavailable");
        }

        if (actorId == 0 && nativeCombatRootInteraction != null) {
            InteractionChain chain = queueNativeInteraction(
                ref,
                store,
                nativeCombatInteractionType,
                nativeCombatRootInteraction,
                requestedChargeTime
            );
            if (chain == null) {
                return NativeAttackAdmission.rejected(
                    "native_interaction_rejected_charge"
                );
            }
            bindSyntheticCombatClientForExplicitRoot(
                ref,
                store,
                chain,
                requestedChargeTime
            );
            combat.setExecutingAttack(chain, false, 0.0);
            actorState.setManualAttackWindow(true);
            return NativeAttackAdmission.admitted();
        }

        if (combat.isExecutingAttack()) {
            return NativeAttackAdmission.rejected("native_combat_busy_or_unavailable");
        }

        if (requestedChargeTime != 0.0) {
            return NativeAttackAdmission.rejected(
                "role_attack_does_not_accept_explicit_charge"
            );
        }
        ActionAttack attack = actorState.nextRoleAttack();
        if (attack == null) {
            return NativeAttackAdmission.rejected("role_attack_unavailable");
        }
        boolean accepted = attack.execute(ref, role, null, deltaTime, store);
        if (accepted) actorState.setManualAttackWindow(true);
        return accepted
            ? NativeAttackAdmission.admitted()
            : NativeAttackAdmission.rejected("role_attack_not_ready");
    }

    private boolean requestNativeAbility(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        int abilitySlot,
        double requestedChargeTime,
        InteractionType guardForkType,
        PendingStep step
    ) {
        if (abilitySlot < 0 || abilitySlot >= nativeCombatAbilities.size()) {
            step.abilityRejectReason = "ability_slot_not_reset_bound";
            return false;
        }
        Role role = npc.getRole();
        CombatSupport combat = role == null ? null : role.getCombatSupport();
        if (combat == null) {
            step.abilityRejectReason = "native_combat_busy";
            return false;
        }

        NativeInteractionBinding binding = nativeCombatAbilities.get(abilitySlot);
        if (trackedAgentAbilityChain != null || pendingAgentAbilityFork != null) {
            step.abilityRejectReason = "native_combat_busy";
            return false;
        }
        if (guardForkType != null) {
            if (
                !nativeGuardLifecycle.inputHeld()
                    || !nativeGuardActive
                    || nativeGuardChain == null
                    || nativeGuardChain.getServerState()
                        != InteractionState.NotFinished
            ) {
                step.abilityRejectReason =
                    "native_guard_fork_requires_held_parent";
                return false;
            }
            NativeGuardFork.Request request = NativeGuardFork.capture(
                nativeGuardChain,
                binding,
                abilitySlot,
                guardForkType,
                requestedChargeTime
            );
            if (!syntheticCombatClient.requestFork(
                    nativeGuardChain,
                    guardForkType
                )) {
                step.abilityRejectReason =
                    "native_guard_fork_client_unavailable";
                return false;
            }
            pendingAgentAbilityFork = request;
            policyActorState(0).setManualAttackWindow(true);
            trackedAgentAbilitySlot = abilitySlot;
            trackedAgentAbilityId = binding.id();
            trackedAgentAbilityType = guardForkType.name();
            trackedAgentAbilityStartTick = -1L;
            trackedAgentAbilityFinishTick = -1L;
            step.abilityAccepted = true;
            step.abilityAcceptedSlot = abilitySlot;
            step.abilityInteractionId = binding.id();
            step.abilityInteractionType = guardForkType.name();
            return true;
        }
        InteractionChain chain = queueNativeInteraction(
            ref,
            store,
            binding.type(),
            binding.root(),
            requestedChargeTime
        );
        if (chain == null) {
            step.abilityRejectReason = "native_interaction_rejected_charge";
            return false;
        }
        bindSyntheticCombatClientForExplicitRoot(
            ref,
            store,
            chain,
            requestedChargeTime
        );
        // This is an explicit policy/item-root edge, not the autonomous
        // AbilityCombatAction self-target branch.  The latter owns its
        // authored two-second controller pause; direct roots are governed by
        // InteractionManager's cooldown/charge state and the chain lifetime.
        combat.setExecutingAttack(chain, false, 0.0);
        policyActorState(0).setManualAttackWindow(true);
        trackedAgentAbilityChain = chain;
        trackedAgentAbilitySlot = abilitySlot;
        trackedAgentAbilityId = binding.id();
        trackedAgentAbilityType = binding.type().name();
        trackedAgentAbilityStartTick = -1L;
        trackedAgentAbilityFinishTick = -1L;
        step.abilityInteractionId = binding.id();
        step.abilityInteractionType = binding.type().name();
        return true;
    }

    private void requestNativeDodge(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        int direction,
        PendingStep step
    ) {
        NativeInteractionBinding binding = switch (direction) {
            case NativeDodgeProgram.LEFT_DIRECTION ->
                nativeDodgeLeftInteraction;
            case NativeDodgeProgram.RIGHT_DIRECTION ->
                nativeDodgeRightInteraction;
            default -> null;
        };
        if (binding == null) {
            step.dodgeRejectReason =
                "dodge_direction_has_no_authored_payload_hytale_0_5_7";
            return;
        }
        InteractionChain chain = queueNativeInteraction(
            ref,
            store,
            binding.type(),
            binding.root(),
            0.0
        );
        if (chain == null) {
            step.dodgeRejectReason = "native_dodge_chain_rejected";
            return;
        }
        bindSyntheticCombatClientForExplicitRoot(
            ref,
            store,
            chain,
            0.0
        );
        trackedAgentDodgeChain = chain;
        trackedAgentDodgeDirection = direction;
        trackedAgentDodgeInteractionId = binding.id();
        trackedAgentDodgeStartTick = -1L;
        trackedAgentDodgeFinishTick = -1L;
        step.dodgeInteractionId = binding.id();
    }

    private void updateNativeGuard(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        boolean held,
        PendingStep step
    ) {
        InteractionManager manager = interactionManager(ref, store);
        boolean risingEdge = nativeGuardLifecycle.consumeRisingEdge();
        boolean fallingEdge = nativeGuardLifecycle.consumeFallingEdge();

        if (!held) {
            if (fallingEdge && manager != null && nativeGuardChain != null) {
                syntheticCombatClient.release(
                    ref,
                    store,
                    nativeGuardChain
                );
            }
            if (
                nativeGuardChain != null
                    && nativeGuardChain.getServerState()
                        != InteractionState.NotFinished
                    && !hasActiveWielding(ref, store)
            ) {
                nativeGuardChain = null;
                restoreNpcInteractionSimulationIfIdle(ref, store);
            }
            return;
        }
        if (nativeGuardInteraction == null) {
            step.guardRejectReason = "guard_interaction_not_reset_bound";
            return;
        }
        if (manager == null) {
            step.guardRejectReason = "interaction_manager_unavailable";
            return;
        }
        if (nativeGuardChain != null) {
            if (
                nativeGuardChain.getServerState()
                    != InteractionState.NotFinished
                    && !hasActiveWielding(ref, store)
            ) {
                nativeGuardChain = null;
                restoreNpcInteractionSimulationIfIdle(ref, store);
            } else {
                step.guardAccepted = !risingEdge;
                if (risingEdge) {
                    step.guardRejectReason = "native_guard_release_pending";
                }
                return;
            }
        }
        if (!risingEdge) {
            step.guardRejectReason = "native_guard_requires_new_press";
            return;
        }
        nativeGuardChain = queueNativeInteraction(
            ref,
            store,
            nativeGuardInteraction.type(),
            nativeGuardInteraction.root(),
            NATIVE_INDEFINITE_HOLD_SECONDS
        );
        if (nativeGuardChain == null) {
            step.guardRejectReason = "native_guard_chain_rejected";
            return;
        }
        trackedNativeGuardInteractionGeneration =
            nativeGuardLifecycle.requestGeneration();
        trackedNativeGuardInteractionId = nativeGuardInteraction.id();
        trackedNativeGuardStartTick = -1L;
        trackedNativeGuardFinishTick = -1L;
        /*
         * Guard is always a continuation-held level, so its release must
         * arrive as client sync. Other policy-controlled roots bind a cursor
         * only when the explicit headless Player has already selected remote
         * client mode for the actor's InteractionManager.
         */
        bindSyntheticCombatClient(
            ref,
            store,
            nativeGuardChain,
            NATIVE_INDEFINITE_HOLD_SECONDS
        );
        step.guardAccepted = true;
    }

    private static boolean hasActiveWielding(
        Ref<EntityStore> ref,
        Store<EntityStore> store
    ) {
        DamageDataComponent damage = store.getComponent(
            ref,
            DamageDataComponent.getComponentType()
        );
        return damage != null && damage.getCurrentWielding() != null;
    }

    static boolean guardLifecycleVisible(
        boolean active,
        boolean accepted,
        boolean started,
        boolean finished,
        long requestGeneration,
        long interactionGeneration,
        long finishTick
    ) {
        return active
            || accepted
            || started
            || finished
            || NativeGuardLifecycle.terminalRetained(
                requestGeneration,
                interactionGeneration,
                finishTick
            );
    }

    private InteractionChain queueNativeInteraction(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        InteractionType type,
        RootInteraction root,
        double requestedChargeTime
    ) {
        return NativeInteractionQueue.queue(
            ref,
            store,
            type,
            root,
            requestedChargeTime
        );
    }

    private Vector3i syntheticFixtureTarget(boolean placement) {
        Vector3i target = placement
            ? syntheticWorldFixturePlaceTarget
            : syntheticWorldFixtureBreakTarget;
        if (target == null) {
            throw new IllegalStateException(
                "Synthetic World fixture was not bound to the settled actor"
            );
        }
        return new Vector3i(target);
    }

    private void bindSyntheticWorldFixture() {
        runOnWorld(() -> {
            Store<EntityStore> store = world.getEntityStore().getStore();
            TransformComponent transform = store.getComponent(
                agentRef,
                TransformComponent.getComponentType()
            );
            if (transform == null) {
                throw new IllegalStateException(
                    "Synthetic World fixture has no settled actor transform"
                );
            }
            Vector3d position = transform.getPosition();
            int originX = (int) Math.floor(position.x);
            int originY = (int) Math.floor(position.y);
            int originZ = (int) Math.floor(position.z);
            Vector3i place = new Vector3i(originX + 1, originY, originZ);
            Vector3i broken = new Vector3i(originX - 1, originY, originZ);
            world.setBlock(place.x, place.y, place.z, BlockType.EMPTY_KEY);
            world.setBlock(
                broken.x,
                broken.y,
                broken.z,
                SYNTHETIC_PLACE_BLOCK_ID
            );
            syntheticWorldFixturePlaceTarget = place;
            syntheticWorldFixtureBreakTarget = broken;
        }, INITIALIZATION_TIMEOUT);
    }

    private void bindSyntheticBlockUseFixture() {
        runOnWorld(() -> {
            Store<EntityStore> store = world.getEntityStore().getStore();
            TransformComponent transform = store.getComponent(
                agentRef,
                TransformComponent.getComponentType()
            );
            ModelComponent model = store.getComponent(
                agentRef,
                ModelComponent.getComponentType()
            );
            if (transform == null || model == null) {
                throw new IllegalStateException(
                    "Synthetic block-Use fixture has no settled actor model"
                );
            }
            Vector3d position = transform.getPosition();
            int targetX = (int) Math.floor(position.x) + 1;
            int targetY = (int) Math.floor(position.y);
            int targetZ = (int) Math.floor(position.z);
            world.setBlock(
                targetX,
                targetY,
                targetZ,
                SYNTHETIC_BLOCK_USE_BLOCK_ID
            );
            BlockType block = world.getBlockType(
                new Vector3i(targetX, targetY, targetZ)
            );
            String interactionId = block == null
                ? null
                : block.getInteractions().get(InteractionType.Use);
            if (
                interactionId == null
                    || RootInteraction.getAssetMap().getAsset(
                        interactionId
                    ) == null
            ) {
                throw new IllegalStateException(
                    "Synthetic block-Use asset has no authored Use root: "
                        + SYNTHETIC_BLOCK_USE_BLOCK_ID
                );
            }

            double eyeY = position.y
                + model.getModel().getEyeHeight(agentRef, store);
            double dx = targetX + 0.5 - position.x;
            double dy = targetY + 0.5 - eyeY;
            double dz = targetZ + 0.5 - position.z;
            policyActorState(0).setPose(
                (float) Math.atan2(-dx, -dz),
                (float) Math.atan2(dy, Math.hypot(dx, dz))
            );
        }, INITIALIZATION_TIMEOUT);
    }

    private void requestSyntheticWorldInteraction(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        AgentAction action,
        PendingStep step
    ) {
        boolean place = action.hasPlacement();
        boolean broken = action.hasBreak();
        if (!place && !broken) return;
        SyntheticWorldInteractionEvidence.Row placeRow =
            step.syntheticWorldInteraction.placeBlock();
        SyntheticWorldInteractionEvidence.Row breakRow =
            step.syntheticWorldInteraction.breakBlock();
        if (!syntheticWorldInteractionsEnabled()) {
            if (place) placeRow.reject("synthetic_fixture_not_selected");
            if (broken) breakRow.reject("synthetic_fixture_not_selected");
            return;
        }
        if (place && broken) {
            placeRow.reject("conflicting_world_verb_requests");
            breakRow.reject("conflicting_world_verb_requests");
            return;
        }
        if (
            syntheticWorldChain != null
                && syntheticWorldChain.getServerState()
                    == InteractionState.NotFinished
        ) {
            (place ? placeRow : breakRow).reject(
                "synthetic_world_interaction_busy"
            );
            return;
        }
        clearSyntheticWorldInteractionRun();

        TransformComponent transform = store.getComponent(
            ref,
            TransformComponent.getComponentType()
        );
        if (transform == null) {
            (place ? placeRow : breakRow).reject(
                "controlled_transform_unavailable"
            );
            return;
        }
        Vector3d position = transform.getPosition();
        Vector3i target = new Vector3i(
            (int) Math.floor(position.x)
                + (place ? action.placeBlockX() : action.breakBlockX()),
            (int) Math.floor(position.y)
                + (place ? action.placeBlockY() : action.breakBlockY()),
            (int) Math.floor(position.z)
                + (place ? action.placeBlockZ() : action.breakBlockZ())
        );
        Vector3i expectedTarget = syntheticFixtureTarget(place);
        if (!target.equals(expectedTarget)) {
            (place ? placeRow : breakRow).reject(
                "target_outside_synthetic_fixture"
            );
            return;
        }
        int expectedBlockId = syntheticWorldFixtureBlockId;
        if (place && action.placeBlockType() != expectedBlockId) {
            placeRow.reject("placed_block_id_not_fixture_bound");
            return;
        }
        int blockBefore = world.getBlock(target.x, target.y, target.z);
        if (place && blockBefore != 0) {
            placeRow.reject("placement_target_not_empty");
            return;
        }
        if (broken && blockBefore != expectedBlockId) {
            breakRow.reject("break_target_not_fixture_block");
            return;
        }
        Inventory inventory = npc.getInventory();
        int activeSlot = place ? 0 : 1;
        inventory.setActiveHotbarSlot(ref, (byte) activeSlot, store);
        Player player = store.getComponent(ref, Player.getComponentType());
        if (player == null || player.getGameMode() != GameMode.Adventure) {
            (place ? placeRow : breakRow).reject(
                "synthetic_adventure_player_unavailable"
            );
            return;
        }
        String expectedHeldItem = place
            ? SYNTHETIC_PLACE_ITEM_ID
            : SYNTHETIC_BREAK_TOOL_ITEM_ID;
        String heldItem = activeItemId(player.getInventory());
        if (
            !expectedHeldItem.equals(heldItem)
                || !expectedHeldItem.equals(activeItemId(inventory))
        ) {
            (place ? placeRow : breakRow).reject(
                "synthetic_fixture_item_not_held"
            );
            return;
        }
        double healthBefore = nativeBlockHealth(target);
        RootInteraction root = place
            ? syntheticPlaceRoot
            : syntheticBreakRoot;
        InteractionChain chain = queueNativeInteraction(
            ref,
            store,
            place ? InteractionType.Secondary : InteractionType.Primary,
            root,
            0.0
        );
        if (chain == null) {
            (place ? placeRow : breakRow).reject(
                "native_interaction_chain_rejected"
            );
            return;
        }
        bindSyntheticWorldTarget(chain, target);

        SyntheticWorldInteractionEvidence.Row row = place
            ? placeRow
            : breakRow;
        row.accept(
            place
                ? SYNTHETIC_PLACE_INTERACTION_ID
                : SYNTHETIC_BREAK_INTERACTION_ID,
            heldItem,
            target,
            blockBefore,
            healthBefore,
            world.getTick()
        );
        syntheticWorldChain = chain;
        syntheticWorldEvidenceRow = row;
        syntheticWorldVerb = place ? "place_block" : "break_block";
        syntheticWorldTarget = new Vector3i(target);
        syntheticWorldBlockBefore = blockBefore;
        syntheticWorldHealthBefore = healthBefore;
        syntheticWorldExpectedBlockId = expectedBlockId;
        syntheticWorldBlockFace = BlockFace.Up;
        syntheticWorldRotationYaw = Rotation.None;
        syntheticWorldRotationPitch = Rotation.None;
        syntheticWorldRotationRoll = Rotation.None;
        syntheticWorldOperationIndex = -1;
        syntheticWorldClientElapsedSeconds = 0.0;
        syntheticWorldStarted = false;
    }

    void prepareSyntheticClientTick(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        float deltaTime,
        int actorId
    ) {
        prepareSyntheticCombatClientTick(ref, store, deltaTime, actorId);
        if (actorId >= 0 && actorId < nativePolicyWorldVerbHandlesByActor.length) {
            Object handle = nativePolicyWorldVerbHandlesByActor[actorId];
            if (handle != null) {
                NativePolicyWorldVerbFacade.Lifecycle lifecycle =
                    NativePolicyWorldVerbFacade.prepare(
                        handle,
                        ref,
                        store,
                        deltaTime
                    );
                PendingStep step = pending.get();
                if (step != null) {
                    step.observeWorldVerb(
                        actorId,
                        nativePolicyWorldVerbRequestsByActor[actorId],
                        lifecycle
                    );
                }
            }
        }
        InteractionChain chain = syntheticWorldChain;
        Vector3i target = syntheticWorldTarget;
        if (
            chain == null
                || target == null
                || ref != agentRef
                || chain.getServerState() != InteractionState.NotFinished
        ) {
            return;
        }
        RootInteraction root = chain.getRootInteraction();
        int operationCounter = chain.getOperationCounter();
        if (
            root == null
                || operationCounter < 0
                || operationCounter >= root.getOperationMax()
        ) {
            return;
        }
        Operation operation = root.getOperation(operationCounter);
        if (
            operation == null
                || operation.getWaitForDataFrom() != WaitForDataFrom.Client
        ) {
            return;
        }
        int operationIndex = chain.getOperationIndex();
        if (operationIndex != syntheticWorldOperationIndex) {
            syntheticWorldOperationIndex = operationIndex;
            syntheticWorldClientElapsedSeconds = 0.0;
        }
        Operation inner = operation.getInnerOperation();
        float runTime = inner instanceof Interaction interaction
            ? Math.max(0.0f, interaction.getRunTime())
            : 0.0f;
        boolean finished = syntheticWorldClientElapsedSeconds >= runTime;

        InteractionSyncData data = new InteractionSyncData();
        data.state = finished
            ? InteractionState.Finished
            : InteractionState.NotFinished;
        data.progress = (float) Math.min(
            Float.MAX_VALUE,
            syntheticWorldClientElapsedSeconds
        );
        data.operationCounter = operationCounter;
        data.rootInteraction =
            RootInteraction.getRootInteractionIdOrUnknown(root.getId());
        data.blockPosition = new BlockPosition(
            target.x,
            target.y,
            target.z
        );
        data.blockFace = syntheticWorldBlockFace;
        data.blockRotation = new BlockRotation(
            syntheticWorldRotationYaw,
            syntheticWorldRotationPitch,
            syntheticWorldRotationRoll
        );
        data.placedBlockId = syntheticWorldVerb.equals("place_block")
            ? syntheticWorldExpectedBlockId
            : world.getBlock(target.x, target.y, target.z);
        if (inner instanceof ChainingInteraction) {
            /*
             * The real client chooses the first authored chain entry on a
             * fresh click. Leaving the protocol sentinel (-1) means "no
             * selection yet" and eventually times out the server chain.
             */
            data.chainingIndex = 0;
        }

        InteractionEntry entry = chain.getInteraction(operationIndex);
        if (entry == null) {
            chain.putInteractionSyncData(operationIndex, data);
        } else if (!entry.setClientState(data)) {
            throw new IllegalStateException(
                "Synthetic client interaction sync desynchronized"
            );
        }
        if (
            finished
                && chain.getCallDepth() == 0
                && operationCounter + 1 >= root.getOperationMax()
        ) {
            chain.setClientState(InteractionState.Finished);
        }
        if (Float.isFinite(deltaTime) && deltaTime >= 0.0f) {
            syntheticWorldClientElapsedSeconds += deltaTime;
        }
    }

    private void bindSyntheticCombatClient(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        InteractionChain chain,
        double requestedChargeTime
    ) {
        bindSyntheticCombatClient(
            ref,
            store,
            chain,
            requestedChargeTime,
            null
        );
    }

    private void bindSyntheticCombatClient(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        InteractionChain chain,
        double requestedChargeTime,
        Ref<EntityStore> selectedTarget
    ) {
        /*
         * An NPC InteractionManager normally runs both serverTick and
         * simulationTick.  That is correct for autonomous NPC input, but it
         * executes predictable SimpleInstant mutations (for example guard's
         * ChangeStat) once on each side when the bridge is supplying the
         * client state itself.  Lease remote-client mode for the lifetime of
         * every explicitly policy-controlled chain.  NativeSyntheticClientSystem
         * supplies its client rows before Hytale's interaction tick.
         */
        syntheticCombatClient.bind(
            ref,
            store,
            chain,
            requestedChargeTime,
            selectedTarget
        );
    }

    private void bindSyntheticCombatClientForExplicitRoot(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        InteractionChain chain,
        double requestedChargeTime
    ) {
        /*
         * Explicit policy control owns the client lifecycle for the entire
         * engine chain, including roots selected or replaced after admission.
         * Looking only at the admitted root's direct operations is unsound:
         * authored selectors may introduce Charging, ApplyForce, Select, or
         * Projectile interactions later. Bind from execution ownership, never
         * from a weapon name or a shallow snapshot of the current root graph.
         */
        bindSyntheticCombatClient(
            ref,
            store,
            chain,
            requestedChargeTime,
            targetRef
        );
    }

    private void restoreNpcInteractionSimulationIfIdle(
        Ref<EntityStore> ref,
        Store<EntityStore> store
    ) {
        syntheticCombatClient.restoreNpcSimulationIfIdle(
            ref,
            store,
            syntheticPlayerRef != null
        );
    }

    private void prepareSyntheticCombatClientTick(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        float deltaTime,
        int actorId
    ) {
        if (actorId >= 0 && actorId < nativePolicyCombatHandles.length) {
            Object handle = nativePolicyCombatHandles[actorId];
            if (handle != null) {
                NativePolicyCombatFacade.Receipt receipt =
                    NativePolicyCombatFacade.prepare(
                        handle,
                        ref,
                        store,
                        deltaTime,
                        false
                    );
                if (!receipt.available()) {
                    throw new IllegalStateException(
                        "Native policy-combat synthetic client is unavailable "
                            + "for actor " + actorId + ": "
                            + receipt.unavailableReason()
                    );
                }
                return;
            }
        }
        if (ref != agentRef) return;
        syntheticCombatClient.prepare(
            ref,
            store,
            deltaTime,
            syntheticPlayerRef != null
        );
    }

    private void bindSyntheticWorldTarget(
        InteractionChain chain,
        Vector3i target
    ) {
        BlockPosition raw = new BlockPosition(target.x, target.y, target.z);
        BlockPosition base = world.getBaseBlock(raw);
        InteractionContext context = chain.getContext();
        context.getMetaStore().putMetaObject(Interaction.TARGET_BLOCK, base);
        context.getMetaStore().putMetaObject(
            Interaction.TARGET_BLOCK_RAW,
            raw
        );
    }

    private void updateSyntheticWorldInteractionTelemetry(
        Store<EntityStore> store
    ) {
        InteractionChain chain = syntheticWorldChain;
        SyntheticWorldInteractionEvidence.Row row =
            syntheticWorldEvidenceRow;
        Vector3i target = syntheticWorldTarget;
        if (chain == null || row == null || target == null) return;

        InteractionManager manager = interactionManager(agentRef, store);
        boolean registered = manager != null
            && manager.getChains().values().contains(chain);
        InteractionState state = chain.getServerState();
        syntheticWorldStarted |= registered
            || chain.getChainId() < 0
            || state != InteractionState.NotFinished;
        int blockAfter = world.getBlock(target.x, target.y, target.z);
        double healthAfter = nativeBlockHealth(target);
        boolean mutationApplied = syntheticWorldVerb.equals("place_block")
            ? blockAfter == syntheticWorldExpectedBlockId
                && blockAfter != syntheticWorldBlockBefore
            : blockAfter == 0
                || healthAfter < syntheticWorldHealthBefore;
        row.observe(
            state.name(),
            registered || syntheticWorldStarted,
            blockAfter,
            healthAfter,
            mutationApplied,
            world.getTick()
        );
        if (
            state != InteractionState.NotFinished
                || (syntheticWorldStarted && !registered)
        ) {
            clearSyntheticWorldInteractionRun();
        }
    }

    private double nativeBlockHealth(Vector3i position) {
        return WorldVerbCapture.blockHealth(world, position);
    }

    private WorldVerbCell captureWorldVerbCell(
        NativeWorldVerbRequest request
    ) {
        return WorldVerbCapture.capture(world, request);
    }

    private ItemStack worldVerbSourceStack(
        NativeWorldVerbRequest request,
        InteractionChain chain,
        Store<EntityStore> store
    ) {
        if (request.sourceContainer().equals("unarmed")) {
            return ItemStack.EMPTY;
        }
        if (request.sourceContainer().equals("interaction_context")) {
            return chain.getContext().getHeldItem();
        }
        if (!request.sourceContainer().equals("hotbar")) {
            return ItemStack.EMPTY;
        }
        NPCEntity npc = store.getComponent(
            agentRef,
            NPCEntity.getComponentType()
        );
        ItemContainer hotbar = npc == null || npc.getInventory() == null
            ? null
            : npc.getInventory().getHotbar();
        if (
            hotbar == null
                || request.sourceSlot() < 0
                || request.sourceSlot()
                    >= Short.toUnsignedInt(hotbar.getCapacity())
        ) {
            return ItemStack.EMPTY;
        }
        return hotbar.getItemStack((short) request.sourceSlot());
    }

    private void clearSyntheticWorldInteractionRun() {
        syntheticWorldChain = null;
        syntheticWorldEvidenceRow = null;
        syntheticWorldVerb = "";
        syntheticWorldTarget = null;
        syntheticWorldBlockBefore = 0;
        syntheticWorldHealthBefore = 0.0;
        syntheticWorldExpectedBlockId = 0;
        syntheticWorldBlockFace = BlockFace.Up;
        syntheticWorldRotationYaw = Rotation.None;
        syntheticWorldRotationPitch = Rotation.None;
        syntheticWorldRotationRoll = Rotation.None;
        syntheticWorldOperationIndex = -1;
        syntheticWorldClientElapsedSeconds = 0.0;
        syntheticWorldStarted = false;
    }

    private void spawnFallingBlockFixture() {
        if (
            !options.fidelityFixture().equals(
                EnvironmentOptions.FALLING_BLOCK_FIXTURE
            )
        ) {
            return;
        }
        runOnWorld(() -> {
            BlockType blockType = BlockType.getAssetMap().getAsset(
                FALLING_BLOCK_FIXTURE_ASSET_ID
            );
            if (blockType == null || blockType.getFallingBlockSettings() == null) {
                throw new IllegalStateException(
                    "Falling-block fixture asset is unavailable: "
                        + FALLING_BLOCK_FIXTURE_ASSET_ID
                );
            }
            int runtimeBlockId = BlockType.getAssetMap().getIndex(
                FALLING_BLOCK_FIXTURE_ASSET_ID
            );
            if (runtimeBlockId <= 0) {
                throw new IllegalStateException(
                    "Falling-block fixture has no runtime block index"
                );
            }
            Vector3d position = new Vector3d(
                Math.floor(spawnX) + 4.5,
                Math.floor(spawnY) + 7.0,
                Math.floor(spawnZ) + 0.5
            );
            Holder<EntityStore> holder = FallingBlock.generateFallingBlock(
                blockType,
                position,
                RotationTuple.get(0),
                blockType.getFallingBlockSettings()
            );
            if (holder == null) {
                throw new IllegalStateException(
                    "Native falling-block fixture generation failed"
                );
            }
            holder.addComponent(markerType, new NativeAgentMarker(this));
            Store<EntityStore> store = world.getEntityStore().getStore();
            fallingBlockRef = store.addEntity(holder, AddReason.SPAWN);
            if (fallingBlockRef == null || !fallingBlockRef.isValid()) {
                throw new IllegalStateException(
                    "Native falling-block fixture did not spawn"
                );
            }
            TransformComponent transform = store.getComponent(
                fallingBlockRef,
                TransformComponent.getComponentType()
            );
            Velocity velocity = store.getComponent(
                fallingBlockRef,
                Velocity.getComponentType()
            );
            PhysicsValues physics = store.getComponent(
                fallingBlockRef,
                PhysicsValues.getComponentType()
            );
            BoundingBox boundingBox = store.getComponent(
                fallingBlockRef,
                BoundingBox.getComponentType()
            );
            BlockEntity blockEntity = store.getComponent(
                fallingBlockRef,
                BlockEntity.getComponentType()
            );
            captureFallingBlockTrace(
                "spawn_before_first_tick",
                transform,
                velocity,
                physics,
                boundingBox,
                blockEntity,
                0.0
            );
        }, INITIALIZATION_TIMEOUT);
    }

    void captureFallingBlockBeforeGravity(
        Ref<EntityStore> ref,
        ArchetypeChunk<EntityStore> chunk,
        int index,
        float deltaTime
    ) {
        Ref<EntityStore> expected = fallingBlockRef;
        if (
            expected == null
                || ref == null
                || !expected.equals(ref)
                || !options.fidelityFixture().equals(
                    EnvironmentOptions.FALLING_BLOCK_FIXTURE
                )
        ) {
            return;
        }
        captureFallingBlockTrace(
            "after_block_entity_physics_before_falling_gravity",
            chunk.getComponent(
                index,
                TransformComponent.getComponentType()
            ),
            chunk.getComponent(index, Velocity.getComponentType()),
            chunk.getComponent(index, PhysicsValues.getComponentType()),
            chunk.getComponent(index, BoundingBox.getComponentType()),
            chunk.getComponent(index, BlockEntity.getComponentType()),
            Float.isFinite(deltaTime) && deltaTime >= 0.0f
                ? deltaTime
                : 0.0
        );
    }

    private void captureFallingBlockTrace(
        String stage,
        TransformComponent transform,
        Velocity velocity,
        PhysicsValues physics,
        BoundingBox boundingBox,
        BlockEntity blockEntity,
        double deltaSeconds
    ) {
        if (
            transform == null
                || velocity == null
                || physics == null
                || boundingBox == null
                || blockEntity == null
        ) {
            throw new IllegalStateException(
                "Native falling-block trace lost a required component"
            );
        }
        String blockAssetId = blockEntity.getBlockTypeKey();
        int runtimeBlockId = BlockType.getAssetMap().getIndex(blockAssetId);
        if (runtimeBlockId <= 0) {
            throw new IllegalStateException(
                "Native falling-block trace has no runtime block index"
            );
        }
        Box bounds = boundingBox.getBoundingBox();
        fallingBlockTrace = NativeFallingBlockTrace.sample(
            stage,
            blockAssetId,
            runtimeBlockId,
            FALLING_BLOCK_FIXTURE_IMPACT_TYPE,
            ++fallingBlockSampleIndex,
            world.getTick(),
            deltaSeconds,
            transform.getPosition(),
            velocity.getVelocity(),
            blockEntity.getSimplePhysicsProvider().isOnGround(),
            physics.getMass(),
            physics.getDragCoefficient(),
            physics.isInvertedGravity(),
            bounds.width(),
            bounds.depth()
        );
    }

    private void updateAgentPolicyVerbTelemetry(
        PendingStep step,
        Store<EntityStore> store
    ) {
        Ref<EntityStore> reference = agentRef;
        if (reference == null || !reference.isValid()) return;
        InteractionManager manager = interactionManager(reference, store);
        NativeGuardFork.Request pendingFork = pendingAgentAbilityFork;
        if (pendingFork != null) {
            InteractionChain spawned = NativeGuardFork.findSpawned(
                pendingFork
            );
            if (spawned != null) {
                bindSyntheticCombatClient(
                    reference,
                    store,
                    spawned,
                    pendingFork.requestedChargeSeconds(),
                    targetRef
                );
                trackedAgentAbilityChain = spawned;
                trackedAgentAbilityParentChain = pendingFork.parent();
                trackedAgentAbilitySlot = pendingFork.abilitySlot();
                trackedAgentAbilityId = pendingFork.ability().id();
                trackedAgentAbilityType = pendingFork.forkType().name();
                pendingAgentAbilityFork = null;
            } else if (
                pendingFork.parent().getServerState()
                    != InteractionState.NotFinished
            ) {
                step.abilityFailed = true;
                step.abilityRejectReason =
                    "native_guard_fork_rejected_before_start";
                trackedAgentAbilityFinishTick = world.getTick();
                trackedAgentAbilitySlot = -1;
                pendingAgentAbilityFork = null;
            }
        }
        InteractionChain ability = trackedAgentAbilityChain;
        if (ability != null) {
            boolean registered = manager != null
                && (
                    manager.getChains().values().contains(ability)
                        || NativeGuardFork.containsChild(
                            trackedAgentAbilityParentChain,
                            ability
                        )
                );
            InteractionState state = ability.getServerState();
            /*
             * queueExecuteChain is provisional: executeChain0 applies the
             * authoritative cooldown gate on the next InteractionManager
             * tick.  Successful server chains receive a negative chain id;
             * a chain rejected before start remains id 0 and Failed.
             */
            boolean started = registered || ability.getChainId() < 0;
            if (started && trackedAgentAbilityStartTick < 0L) {
                trackedAgentAbilityStartTick = world.getTick();
                step.abilityAccepted = true;
                step.abilityAcceptedSlot = trackedAgentAbilitySlot;
                step.abilityStarted = true;
            }
            boolean finished = started
                && (state != InteractionState.NotFinished || !registered);
            if (finished && trackedAgentAbilityFinishTick < 0L) {
                trackedAgentAbilityFinishTick = world.getTick();
                // Retain the admitted compact root on the terminal edge. A
                // client may have lost the start acknowledgement and must not
                // infer this profile-specific slot from the authored index.
                step.abilityAcceptedSlot = trackedAgentAbilitySlot;
                step.abilityFinished = true;
                step.abilityFailed = state == InteractionState.Failed;
                trackedAgentAbilityChain = null;
                trackedAgentAbilityParentChain = null;
                trackedAgentAbilitySlot = -1;
            } else if (
                !started
                    && state == InteractionState.Failed
            ) {
                step.abilityFailed = true;
                step.abilityRejectReason =
                    "native_interaction_cooldown_or_rules";
                trackedAgentAbilityFinishTick = world.getTick();
                trackedAgentAbilityChain = null;
                trackedAgentAbilityParentChain = null;
                trackedAgentAbilitySlot = -1;
            }
        }

        InteractionChain dodge = trackedAgentDodgeChain;
        if (dodge != null) {
            boolean registered = manager != null
                && manager.getChains().values().contains(dodge);
            InteractionState state = dodge.getServerState();
            NativeQueuedInteractionLifecycle.Observation observation =
                NativeQueuedInteractionLifecycle.classify(
                    registered,
                    dodge.getChainId(),
                    state
                );
            boolean started = observation.admitted();
            if (started && trackedAgentDodgeStartTick < 0L) {
                trackedAgentDodgeStartTick = world.getTick();
                step.dodgeAccepted = true;
                step.dodgeStarted = true;
            }
            if (observation.terminal()) {
                if (trackedAgentDodgeFinishTick < 0L) {
                    trackedAgentDodgeFinishTick = world.getTick();
                    step.dodgeFinished = true;
                    step.dodgeFailed = state == InteractionState.Failed;
                }
                trackedAgentDodgeChain = null;
            } else if (observation.rejectedBeforeStart()) {
                trackedAgentDodgeFinishTick = world.getTick();
                step.dodgeFailed = true;
                step.dodgeRejectReason =
                    "native_interaction_cooldown_or_rules";
                trackedAgentDodgeChain = null;
            }
        }

        InteractionChain use = trackedAgentUseChain;
        if (use != null) {
            boolean registered = manager != null
                && manager.getChains().values().contains(use);
            InteractionState state = use.getServerState();
            boolean started = registered
                || use.getChainId() < 0
                || state != InteractionState.NotFinished;
            if (started && trackedAgentUseStartTick < 0L) {
                trackedAgentUseStartTick = world.getTick();
            }
            step.useStarted |= started;
            boolean finished = started
                && (state != InteractionState.NotFinished || !registered);
            if (finished && trackedAgentUseFinishTick < 0L) {
                trackedAgentUseFinishTick = world.getTick();
            }
            step.useFinished |= finished;
            step.useFailed |= state == InteractionState.Failed;
        }

        boolean active = hasActiveWielding(reference, store);
        boolean guardStarted = active && !nativeGuardActive;
        boolean guardFinished = !active && nativeGuardActive;
        if (guardStarted && trackedNativeGuardStartTick < 0L) {
            trackedNativeGuardStartTick = world.getTick();
        }
        if (guardFinished && trackedNativeGuardFinishTick < 0L) {
            trackedNativeGuardFinishTick = world.getTick();
        }
        step.guardStarted |= guardStarted;
        step.guardFinished |= guardFinished;
        nativeGuardActive = active;
        step.observeLegacyPrimaryCombat(
            nativeGuardActive,
            trackedAgentAbilityChain != null,
            trackedAgentDodgeChain != null
        );
    }

    private void updateLegacyActorZeroTypedWorldVerbTelemetry(
        PendingStep step,
        Store<EntityStore> store
    ) {
        WorldVerbTelemetry.Execution execution =
            legacyActorZeroTypedWorldVerbExecution;
        CraftingManager crafting = legacyActorZeroTypedWorldVerbCraftingManager;
        if (crafting != null && execution.pending()) {
            if (!execution.started()) {
                execution.observeStarted(world.getTick());
            }
            if (crafting.getRemainingQueueSize() != 0) return;

            CraftingRecipe recipe = legacyActorZeroTypedWorldVerbCraftingRecipe;
            if (recipe == null) {
                execution.observeFinishedWithoutAcknowledgement(
                    true,
                    world.getTick()
                );
            } else {
                String inventoryAfter =
                    NativeHeadlessCrafting.canonicalInventory(
                        agentRef,
                        store
                    );
                int outputAfter = NativeHeadlessCrafting.outputQuantity(
                    agentRef,
                    store,
                    recipe
                );
                String geometrySha256 = sha256Hex(
                    "hytalerl_world_verb_geometry_delta_v1\n"
                        + legacyActorZeroTypedWorldVerbGeometryBefore
                        + "\n"
                        + legacyActorZeroTypedWorldVerbGeometryBefore
                );
                String inventorySha256 = sha256Hex(
                    "hytalerl_world_verb_inventory_delta_v1\n"
                        + execution.request().sourceContainer()
                        + "\n-1\n"
                        + legacyActorZeroTypedWorldVerbInventoryBefore
                        + "\n"
                        + inventoryAfter
                );
                execution.observeFinished(
                    false,
                    world.getTick(),
                    "not_applicable",
                    0,
                    0,
                    outputAfter,
                    false,
                    !legacyActorZeroTypedWorldVerbInventoryBefore.equals(
                        inventoryAfter
                    ),
                    geometrySha256,
                    inventorySha256,
                    ""
                );
            }
            if (crafting.hasBenchSet()) {
                crafting.clearBench(agentRef, store);
            }
            legacyActorZeroTypedWorldVerbCraftingManager = null;
            legacyActorZeroTypedWorldVerbCraftingRecipe = null;
            legacyActorZeroTypedWorldVerbGeometryBefore = "";
            legacyActorZeroTypedWorldVerbInventoryBefore = "";
            if (step.legacyActorZeroWorldVerbExecution == execution) {
                step.legacyActorZeroWorldVerbExecution = execution;
            }
            return;
        }

        InteractionChain chain = legacyActorZeroTypedWorldVerbChain;
        if (chain == null || !execution.pending()) return;

        InteractionManager manager = interactionManager(agentRef, store);
        boolean registered = manager != null
            && manager.getChains().values().contains(chain);
        InteractionState state = chain.getServerState();
        boolean started = registered
            || chain.getChainId() < 0
            || state != InteractionState.NotFinished;
        if (started) execution.observeStarted(world.getTick());
        boolean finished = started
            && (state != InteractionState.NotFinished || !registered);
        if (!finished) return;

        NativeWorldVerbRequest request = execution.request();
        WorldVerbCell after = captureWorldVerbCell(request);
        boolean unarmed = request.sourceContainer().equals("unarmed");
        ItemStack stack = worldVerbSourceStack(request, chain, store);
        if (after == null) {
            execution.observeFinishedWithoutAcknowledgement(
                state == InteractionState.Failed,
                world.getTick()
            );
        } else {
            String inventoryAfter = unarmed
                ? "unarmed"
                : canonicalWorldVerbInventoryState(stack);
            String geometrySha256 = sha256Hex(
                "hytalerl_world_verb_geometry_delta_v1\n"
                    + legacyActorZeroTypedWorldVerbGeometryBefore
                    + "\n"
                    + after.canonical()
            );
            String inventorySha256 = sha256Hex(
                "hytalerl_world_verb_inventory_delta_v1\n"
                    + request.sourceContainer()
                    + "\n"
                    + request.sourceSlot()
                    + "\n"
                    + legacyActorZeroTypedWorldVerbInventoryBefore
                    + "\n"
                    + inventoryAfter
            );
            execution.observeFinished(
                state == InteractionState.Failed,
                world.getTick(),
                after.semanticBlockId(),
                after.runtimeBlockId(),
                unarmed
                    ? 0
                    : ItemStack.isEmpty(stack)
                        ? 0
                        : stack.getQuantity(),
                -1,
                !legacyActorZeroTypedWorldVerbGeometryBefore.equals(
                    after.canonical()
                ),
                !legacyActorZeroTypedWorldVerbInventoryBefore.equals(inventoryAfter),
                geometrySha256,
                inventorySha256,
                ""
            );
        }
        legacyActorZeroTypedWorldVerbChain = null;
        legacyActorZeroTypedWorldVerbGeometryBefore = "";
        legacyActorZeroTypedWorldVerbInventoryBefore = "";
        if (syntheticWorldChain == chain) {
            clearSyntheticWorldInteractionRun();
        }
        if (step.legacyActorZeroWorldVerbExecution == execution) {
            step.legacyActorZeroWorldVerbExecution = execution;
        }
    }

    private void suppressAutonomousActorCombat(
        Ref<EntityStore> ref,
        Role role,
        Store<EntityStore> store,
        NativePolicyActorState actorState,
        boolean guardHeld
    ) {
        CombatSupport combat = role.getCombatSupport();
        InteractionManager manager = interactionManager(ref, store);
        if (combat == null || manager == null) return;

        clearMarkedTargets(role);

        int actorId = actorState.actorId();
        if (
            actorId >= 0
                && actorId < nativePolicyWorldVerbHandlesByActor.length
                && nativePolicyWorldVerbHandlesByActor[actorId] != null
        ) {
            actorState.setManualAttackWindow(false);
            combat.setExecutingAttack(null, false, 0.0);
            return;
        }

        if (
            actorState.manualAttackWindow()
                && (combat.isExecutingAttack()
                    || hasUnfinishedInteractionTree(manager))
        ) {
            return;
        }
        if (actorState.actorId() != 0) {
            actorState.setManualAttackWindow(false);
            clearNativeCombat(manager, combat);
            return;
        }
        /*
         * The guard chain owns both the held level and its release edge.
         * NativeControlSystem runs after InteractionManager, so on the first
         * guard-off tick the chain is still active and the synthetic client
         * has not yet delivered chargeValue=0.  Clearing it here would cancel
         * WieldingInteraction before its authored Next operation can apply
         * the stamina-regeneration delay.  Preserve the chain until
         * updateNativeGuard observes its terminal state; guardHeld alone is
         * not a sufficient lifetime test.
         */
        if (nativeGuardChain != null) return;
        if (
            trackedAgentDodgeChain != null
                && manager.getChains().values().contains(
                    trackedAgentDodgeChain
                )
        ) {
            actorState.setManualAttackWindow(false);
            combat.setExecutingAttack(null, false, 0.0);
            nativeGuardChain = null;
            return;
        }
        if (
            trackedAgentUseChain != null
                && manager.getChains().values().contains(
                    trackedAgentUseChain
                )
        ) {
            actorState.setManualAttackWindow(false);
            combat.setExecutingAttack(null, false, 0.0);
            nativeGuardChain = null;
            return;
        }
        if (
            syntheticWorldChain != null
                && manager.getChains().values().contains(
                    syntheticWorldChain
                )
        ) {
            actorState.setManualAttackWindow(false);
            combat.setExecutingAttack(null, false, 0.0);
            nativeGuardChain = null;
            return;
        }
        actorState.setManualAttackWindow(false);
        clearNativeCombat(manager, combat);
        nativeGuardChain = null;
    }

    private void clearAgentDecisionTargets(Store<EntityStore> store) {
        Ref<EntityStore> reference = agentRef;
        if (reference == null || !reference.isValid()) return;
        NPCEntity agent = store.getComponent(reference, NPCEntity.getComponentType());
        Role role = agent == null ? null : agent.getRole();
        if (role != null) clearMarkedTargets(role);
    }

    private void captureTargetNavigationTrace(Store<EntityStore> store) {
        Ref<EntityStore> reference = targetRef;
        NPCEntity target = reference == null || !reference.isValid()
            ? null
            : store.getComponent(reference, NPCEntity.getComponentType());
        TransformComponent transform = reference == null || !reference.isValid()
            ? null
            : store.getComponent(
                reference,
                TransformComponent.getComponentType()
            );
        targetNavigationTrace = NativePathFollowerTrace.capture(
            target == null ? null : target.getRole(),
            transform == null ? null : transform.getPosition(),
            world == null ? 0L : world.getTick(),
            targetNavigationTrace
        );
    }

    private void suppressPassiveTargetCombat(
        PendingStep step,
        Store<EntityStore> store
    ) {
        boolean duelSetup = nativeDuelFixture()
            && step.actionForEntity(0) != null;
        if (duelSetup) nativeDuelAutonomous = false;
        if (!taskId.equals("kill_trork")
            || (options.combatTargetActive() && !duelSetup)
            || targetMemoryFixture != null
            || step.actionForEntity(1) != null) return;
        Ref<EntityStore> reference = targetRef;
        if (reference == null || !reference.isValid()) return;
        NPCEntity target = store.getComponent(reference, NPCEntity.getComponentType());
        Role role = target == null ? null : target.getRole();
        CombatSupport combat = role == null ? null : role.getCombatSupport();
        InteractionManager manager = interactionManager(reference, store);
        if (manager != null && combat != null) clearNativeCombat(manager, combat);
        if (role != null) {
            StateSupport stateSupport = role.getStateSupport();
            if (stateSupport != null) {
                stateSupport.setState(reference, "Idle", null, store);
            }
            Steering body = role.getBodySteering();
            if (body != null) body.clear();
            Steering head = role.getHeadSteering();
            if (head != null) head.clear();
        }
    }

    /**
     * Samples the target's authored interaction chain once per native engine tick.
     *
     * <p>This deliberately runs before a multi-tick Gym step can complete so the
     * policy receives the same windup/sweep/recovery state regardless of
     * {@code ticks_per_step}.</p>
     */
    private void updateTargetCombatTelemetry(Store<EntityStore> store) {
        Ref<EntityStore> reference = targetRef;
        if (!taskId.equals("kill_trork")
            || reference == null
            || !reference.isValid()) {
            resetTargetAttackTelemetry();
            return;
        }

        NPCEntity target = store.getComponent(reference, NPCEntity.getComponentType());
        Role role = target == null ? null : target.getRole();
        CombatSupport combat = role == null ? null : role.getCombatSupport();
        InteractionManager manager = interactionManager(reference, store);
        if (combat == null || manager == null) {
            resetTargetAttackTelemetry();
            return;
        }

        InteractionChain activeChain = null;
        int activeIndex = -1;
        String activeId = "";
        for (InteractionChain chain : manager.getChains().values()) {
            if (chain == null || chain.getInitialRootInteraction() == null) continue;
            String interactionId = chain.getInitialRootInteraction().getId();
            int attackIndex = HytaleCombatAssets.brawlerAttackIndex(interactionId);
            if (attackIndex >= 0) {
                activeChain = chain;
                activeIndex = attackIndex;
                activeId = interactionId;
                break;
            }
        }

        if (activeChain != null) {
            if (activeChain != trackedTargetAttackChain) {
                trackedTargetAttackChain = activeChain;
                targetAttackElapsedTicks = 1;
            } else {
                targetAttackElapsedTicks++;
            }
            targetAttackIndex = activeIndex;
            targetAttackId = activeId;
            MeleeAttackProfile profile =
                HytaleCombatAssets.TRORK_BRAWLER_ATTACKS.get(activeIndex);
            targetAttackPhase = profile.phaseAt(targetAttackElapsedTicks);
            targetAttackProgress = profile.phaseProgressAt(targetAttackElapsedTicks);
            return;
        }

        trackedTargetAttackChain = null;
        targetAttackElapsedTicks = 0;
        if (combat.isExecutingAttack()) {
            targetAttackPhase = CombatPhase.COOLDOWN;
            targetAttackProgress = 1.0;
        } else {
            targetAttackPhase = CombatPhase.IDLE;
            targetAttackProgress = 0.0;
            targetAttackIndex = -1;
            targetAttackId = "";
        }
    }

    private void resetTargetAttackTelemetry() {
        trackedTargetAttackChain = null;
        targetAttackPhase = CombatPhase.IDLE;
        targetAttackProgress = 0.0;
        targetAttackIndex = -1;
        targetAttackElapsedTicks = 0;
        targetAttackId = "";
    }

    private void resolveNativeCombatInteraction() {
        if (
            options.hasNativeCombatItem()
                && Item.getAssetMap().getAsset(options.nativeCombatItemId()) == null
        ) {
            throw new IllegalArgumentException(
                "Native combat item asset not found: "
                    + options.nativeCombatItemId()
            );
        }
        if (options.hasNativeCombatInteraction()) {
            NativeInteractionBinding binding = resolveNativeInteractionBinding(
                options.nativeCombatInteractionId(),
                options.nativeCombatInteractionType()
            );
            nativeCombatInteractionType = binding.type();
            nativeCombatRootInteraction = binding.root();
        }
        if (options.hasNativeCombatAbilities()) {
            NativeInteractionBinding[] bindings = resolveNativeAbilityBindings(
                options.nativeCombatItemId(),
                options.nativeCombatAbilityInteractionIds().toArray(String[]::new),
                options.nativeCombatAbilityInteractionTypes().toArray(String[]::new)
            );
            nativeCombatAbilities = List.of(bindings);
        }
        if (options.hasNativeGuardInteraction()) {
            nativeGuardInteraction = resolveNativeInteractionBinding(
                options.nativeGuardInteractionId(),
                options.nativeGuardInteractionType()
            );
        }
        nativeDodgeLeftInteraction = NativeDodgeBindingResolver.resolve(
            NativeDodgeProgram.LEFT_DIRECTION
        );
        nativeDodgeRightInteraction = NativeDodgeBindingResolver.resolve(
            NativeDodgeProgram.RIGHT_DIRECTION
        );
        if (options.hasNativePolicyCombatBindings()) {
            for (
                NativePolicyCombatBindingSpec spec
                    : options.nativePolicyCombatBindings()
            ) {
                NativePolicyCombatFacade.Binding binding =
                    NativePolicyCombatFacade.bind(
                        spec.itemId(),
                        spec.abilitySlots().stream()
                            .mapToInt(Integer::intValue)
                            .toArray(),
                        spec.abilityInteractionIds().toArray(String[]::new),
                        spec.abilityInteractionTypes().toArray(String[]::new),
                        spec.guardInteractionId(),
                        spec.guardInteractionType()
                    );
                if (!binding.accepted() || binding.handle() == null) {
                    throw new IllegalArgumentException(
                        "Native policy-combat binding for actor "
                            + spec.entityId() + " is unavailable: "
                            + binding.rejectReason()
                    );
                }
                nativePolicyCombatSpecs[spec.entityId()] = spec;
                nativePolicyCombatHandles[spec.entityId()] = binding.handle();
            }
        }
    }

    private void updatePolicyWorldVerbTelemetry(
        PendingStep step,
        Store<EntityStore> store
    ) {
        for (
            int actorId = 0;
            actorId < nativePolicyWorldVerbHandlesByActor.length;
            actorId++
        ) {
            Object handle = nativePolicyWorldVerbHandlesByActor[actorId];
            if (handle == null) continue;
            Ref<EntityStore> actor = policyActorRef(actorId);
            NativePolicyWorldVerbFacade.Lifecycle lifecycle =
                NativePolicyWorldVerbFacade.poll(handle, actor, store);
            NativeWorldVerbRequest request =
                nativePolicyWorldVerbRequestsByActor[actorId];
            step.observeWorldVerb(actorId, request, lifecycle);
            if (lifecycle.finished() || lifecycle.failed()) {
                nativePolicyWorldVerbHandlesByActor[actorId] = null;
                nativePolicyWorldVerbRequestsByActor[actorId] = null;
            }
        }
    }

    private boolean syntheticWorldInteractionsEnabled() {
        return options.fidelityFixture().equals(
            EnvironmentOptions.SYNTHETIC_WORLD_VERBS_FIXTURE
        );
    }

    private boolean headlessWorldVerbsEnabled() {
        return options.nativeWorldVerbs()
            || syntheticWorldInteractionsEnabled();
    }

    /** Install the Adventure context whenever typed World verbs request it. */
    private boolean installHeadlessWorldVerbContextAtReset() {
        return headlessWorldVerbsEnabled();
    }

    private boolean syntheticBlockUseFixtureEnabled() {
        return options.fidelityFixture().equals(
            EnvironmentOptions.SYNTHETIC_BLOCK_USE_FIXTURE
        );
    }

    private SyntheticWorldInteractionEvidence
    syntheticWorldInteractionEvidence(AgentAction action) {
        boolean available = syntheticWorldInteractionsEnabled();
        available = available
            && syntheticWorldFixturePlaceTarget != null
            && syntheticWorldFixtureBreakTarget != null;
        return new SyntheticWorldInteractionEvidence(
            available,
            action,
            available ? SYNTHETIC_PLACE_BLOCK_ID : "",
            available ? syntheticWorldFixtureBlockId : 0,
            available ? SYNTHETIC_PLACE_ITEM_ID : "",
            available ? SYNTHETIC_BREAK_TOOL_ITEM_ID : "",
            available ? syntheticWorldFixturePlaceSourceQuantity : 0,
            available ? syntheticWorldFixtureBreakSourceQuantity : 0,
            available ? syntheticWorldFixturePlaceTarget : null,
            available ? syntheticWorldFixtureBreakTarget : null
        );
    }

    private void resolveSyntheticWorldInteractions() {
        if (!syntheticWorldInteractionsEnabled()) return;
        if (Item.getAssetMap().getAsset(SYNTHETIC_PLACE_ITEM_ID) == null) {
            throw new IllegalStateException(
                "Synthetic World fixture placement item is missing: "
                    + SYNTHETIC_PLACE_ITEM_ID
            );
        }
        if (
            !SYNTHETIC_PLACE_BLOCK_ID.equals(
                new ItemStack(SYNTHETIC_PLACE_ITEM_ID).getBlockKey()
            )
        ) {
            throw new IllegalStateException(
                "Synthetic World placement item does not resolve fixture block: "
                    + SYNTHETIC_PLACE_ITEM_ID
            );
        }
        if (
            Item.getAssetMap().getAsset(SYNTHETIC_BREAK_TOOL_ITEM_ID) == null
        ) {
            throw new IllegalStateException(
                "Synthetic World fixture break tool is missing: "
                    + SYNTHETIC_BREAK_TOOL_ITEM_ID
            );
        }
        if (
            BlockType.getAssetMap().getAsset(SYNTHETIC_PLACE_BLOCK_ID)
                == null
        ) {
            throw new IllegalStateException(
                "Synthetic World fixture block is missing: "
                    + SYNTHETIC_PLACE_BLOCK_ID
            );
        }
        syntheticWorldFixtureBlockId = BlockType.getAssetMap().getIndex(
            SYNTHETIC_PLACE_BLOCK_ID
        );
        if (syntheticWorldFixtureBlockId <= 0) {
            throw new IllegalStateException(
                "Synthetic World fixture block has no runtime index: "
                    + SYNTHETIC_PLACE_BLOCK_ID
            );
        }
        syntheticPlaceRoot = resolvedWorldProbeRoot(
            SYNTHETIC_PLACE_INTERACTION_ID
        );
        syntheticBreakRoot = resolvedWorldProbeRoot(
            SYNTHETIC_BREAK_INTERACTION_ID
        );
    }

    private void resolveNativeAgentArmorOverride() {
        if (!options.hasNativeAgentArmorOverride()) return;

        List<NativeArmorEntry> resolved = new ArrayList<>(
            options.nativeAgentArmorItemIds().size()
        );
        Set<Short> occupiedSlots = new HashSet<>();
        for (String itemId : options.nativeAgentArmorItemIds()) {
            Item item = Item.getAssetMap().getAsset(itemId);
            if (item == null) {
                throw new IllegalArgumentException(
                    "Native agent armor item asset not found: " + itemId
                );
            }
            ItemArmor armor = item.getArmor();
            if (armor == null) {
                throw new IllegalArgumentException(
                    "Native agent armor override is not armor: " + itemId
                );
            }
            short slot = (short) armor.getArmorSlot().ordinal();
            if (!occupiedSlots.add(slot)) {
                throw new IllegalArgumentException(
                    "Native agent armor override contains two items for "
                        + armor.getArmorSlot()
                );
            }
            resolved.add(new NativeArmorEntry(slot, itemId));
        }
        nativeAgentArmorOverride = List.copyOf(resolved);
    }

    private void replaceNativeAgentArmor(NPCEntity npc) {
        if (nativeAgentArmorOverride == null) return;
        Inventory inventory = npc.getInventory();
        if (inventory == null || inventory.getArmor() == null) {
            throw new IllegalStateException(
                "Native agent armor override requires an NPC armor inventory"
            );
        }
        ItemContainer armor = inventory.getArmor();
        for (NativeArmorEntry entry : nativeAgentArmorOverride) {
            if (entry.slot() < 0 || entry.slot() >= armor.getCapacity()) {
                throw new IllegalStateException(
                    "Native NPC armor inventory has no slot for "
                        + entry.itemId()
                );
            }
        }
        if (!armor.clear().succeeded()) {
            throw new IllegalStateException(
                "Native NPC rejected clearing its role-authored armor"
            );
        }
        for (NativeArmorEntry entry : nativeAgentArmorOverride) {
            if (!armor.setItemStackForSlot(
                entry.slot(),
                new ItemStack(entry.itemId())
            ).succeeded()) {
                throw new IllegalStateException(
                    "Native NPC rejected armor item: " + entry.itemId()
                );
            }
        }
    }

    private void equipNativeCombatItem(
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        if (!options.hasNativeCombatItem()) return;
        equipNativeCombatItem(
            agentRef,
            npc,
            store,
            options.nativeCombatItemId()
        );
    }

    private void equipNativeCombatItem(
        Ref<EntityStore> reference,
        NPCEntity npc,
        Store<EntityStore> store,
        String itemId
    ) {
        Inventory inventory = npc.getInventory();
        if (inventory == null || inventory.getHotbar().getCapacity() < 1) {
            throw new IllegalStateException(
                "Native combat interaction requires an NPC hotbar"
            );
        }
        if (!inventory.getHotbar().setItemStackForSlot(
            (short) 0,
            new ItemStack(itemId)
        ).succeeded()) {
            throw new IllegalStateException(
                "Native NPC rejected combat item: " + itemId
            );
        }
        inventory.setActiveHotbarSlot(reference, (byte) 0, store);
        EntityStatMap stats = store.getComponent(
            reference,
            EntityStatsModule.get().getEntityStatMapComponentType()
        );
        if (stats != null) {
            // Fresh NPC hotbars already select slot zero, so selecting zero
            // again emits no InventorySetActiveSlotEvent. Schedule the same
            // native item-stat recalculation explicitly after replacing the
            // active stack; warmup ticks apply it before evidence capture.
            stats.getStatModifiersManager().scheduleRecalculate();
        }
    }

    private void settleNativeCombatItemStats() {
        runOnWorld(() -> {
            Store<EntityStore> store = world.getEntityStore().getStore();
            if (options.hasNativeCombatItem()) {
                settleNativeCombatItemStats(agentRef, store);
            }
            if (options.hasNativePolicyCombatBindings()) {
                for (
                    NativePolicyCombatBindingSpec spec
                        : options.nativePolicyCombatBindings()
                ) {
                    Ref<EntityStore> reference = policyActorRef(spec.entityId());
                    if (reference == null || !reference.isValid()) {
                        throw new IllegalStateException(
                            "Native policy-combat actor is unavailable while "
                                + "settling item stats: " + spec.entityId()
                        );
                    }
                    settleNativeCombatItemStats(reference, store);
                }
            }
        }, INITIALIZATION_TIMEOUT);
    }

    private void settleNativeCombatItemStats(
        Ref<EntityStore> reference,
        Store<EntityStore> store
    ) {
            ComponentType<EntityStore, EntityStatMap> statMapType =
                EntityStatsModule.get().getEntityStatMapComponentType();
            EntityStatMap settledStats = store.getComponent(
                reference,
                statMapType
            );
            if (settledStats == null) {
                throw new IllegalStateException(
                    "Native combat item requires an entity-stat map"
                );
            }

            /*
             * EntityStatMap.update() does not backfill null holes that were
             * already inside a sparse role map. Hytale's clone() starts from
             * the current stat catalog and then copies the role's exact
             * values/modifiers, so use that native copy primitive only when
             * the settled role omitted a catalog row required by the held
             * item. No weapon or resource name is special-cased here.
             */
            EntityStatMap catalogBackedStats = settledStats.clone();
            EntityStatMap effectiveStats = settledStats;
            for (int index = 0; index < catalogBackedStats.size(); index++) {
                if (
                    catalogBackedStats.get(index) != null
                        && (
                            index >= settledStats.size()
                                || settledStats.get(index) == null
                        )
                ) {
                    store.replaceComponent(
                        reference,
                        statMapType,
                        catalogBackedStats
                    );
                    effectiveStats = catalogBackedStats;
                    break;
                }
            }

            effectiveStats.getStatModifiersManager().scheduleRecalculate();
            effectiveStats
                .getStatModifiersManager()
                .recalculateEntityStatModifiers(
                    reference,
                    effectiveStats,
                    store
                );
    }

    private void applyNativeFidelityResourceOverrides() {
        runOnWorld(() -> {
            Store<EntityStore> store = world.getEntityStore().getStore();
            for (
                NativeResourceOverride override
                    : options.nativeFidelityResourceOverrides()
            ) {
                Ref<EntityStore> reference = policyActorRef(
                    override.entityId()
                );
                if (reference == null || !reference.isValid()) {
                    throw new IllegalStateException(
                        "Native resource override actor is unavailable: "
                            + override.entityId()
                    );
                }
                EntityStatMap stats = store.getComponent(
                    reference,
                    EntityStatsModule.get().getEntityStatMapComponentType()
                );
                EntityStatValue stat = stat(
                    stats,
                    override.resourceId()
                );
                if (stat == null) {
                    throw new IllegalStateException(
                        "Native resource override is unavailable after item "
                            + "stat settlement: "
                            + override.entityId()
                            + ":"
                            + override.resourceId()
                    );
                }
                float requested = override.value();
                if (requested < stat.getMin() || requested > stat.getMax()) {
                    throw new IllegalStateException(
                        "Native resource override is outside the settled "
                            + "range: "
                            + override.entityId()
                            + ":"
                            + override.resourceId()
                            + "="
                            + requested
                            + " not in ["
                            + stat.getMin()
                            + ","
                            + stat.getMax()
                            + "]"
                    );
                }
                float applied = stats.setStatValue(
                    stat.getIndex(),
                    requested
                );
                if (Float.compare(applied, requested) != 0) {
                    throw new IllegalStateException(
                        "Native resource override was not applied exactly: "
                            + override.entityId()
                            + ":"
                            + override.resourceId()
                    );
                }
            }
        }, INITIALIZATION_TIMEOUT);
    }

    private void equipSyntheticWorldItems(
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        Inventory inventory = npc.getInventory();
        if (
            inventory == null
                || inventory.getHotbar() == null
                || inventory.getHotbar().getCapacity() < 2
        ) {
            throw new IllegalStateException(
                "Synthetic World interaction requires an NPC hotbar"
            );
        }
        if (
            !inventory.getHotbar().setItemStackForSlot(
                (short) 0,
                new ItemStack(SYNTHETIC_PLACE_ITEM_ID)
            ).succeeded()
        ) {
            throw new IllegalStateException(
                "Native NPC rejected synthetic World placement item: "
                    + SYNTHETIC_PLACE_ITEM_ID
            );
        }
        if (
            !inventory.getHotbar().setItemStackForSlot(
                (short) 1,
                new ItemStack(SYNTHETIC_BREAK_TOOL_ITEM_ID)
            ).succeeded()
        ) {
            throw new IllegalStateException(
                "Native NPC rejected synthetic World break tool: "
                    + SYNTHETIC_BREAK_TOOL_ITEM_ID
            );
        }
        syntheticWorldFixturePlaceSourceQuantity =
            inventory.getHotbar().getItemStack((short) 0).getQuantity();
        syntheticWorldFixtureBreakSourceQuantity =
            inventory.getHotbar().getItemStack((short) 1).getQuantity();
        if (
            syntheticWorldFixturePlaceSourceQuantity <= 0
                || syntheticWorldFixtureBreakSourceQuantity <= 0
        ) {
            throw new IllegalStateException(
                "Synthetic World source quantities are unavailable"
            );
        }
        inventory.setActiveHotbarSlot(agentRef, (byte) 0, store);
    }

    private void equipNativeWorldHotbar(
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        Inventory inventory = npc.getInventory();
        if (inventory == null || inventory.getHotbar() == null) {
            throw new IllegalStateException(
                "Native World hotbar override requires an NPC hotbar"
            );
        }
        InventoryComponent.Hotbar hotbarComponent = store.getComponent(
            agentRef,
            InventoryComponent.Hotbar.getComponentType()
        );
        if (hotbarComponent == null) {
            throw new IllegalStateException(
                "Native World hotbar override requires a hotbar component"
            );
        }
        int capacity = Short.toUnsignedInt(
            hotbarComponent.getInventory().getCapacity()
        );
        int requiredCapacity = requiredHotbarCapacity(
            options.nativeWorldHotbarSlots(),
            capacity
        );
        if (requiredCapacity > capacity) {
            List<ItemStack> remainder = new ArrayList<>();
            hotbarComponent.ensureCapacity(
                (short) requiredCapacity,
                remainder
            );
            if (!remainder.isEmpty()) {
                throw new IllegalStateException(
                    "Native World hotbar expansion displaced existing items"
                );
            }
            capacity = requiredCapacity;
        }
        ItemContainer hotbar = hotbarComponent.getInventory();
        for (int index = 0;
            index < options.nativeWorldHotbarSlots().size();
            index++) {
            int slot = options.nativeWorldHotbarSlots().get(index);
            String itemId = options.nativeWorldHotbarItemIds().get(index);
            int quantity = options.nativeWorldHotbarQuantities().get(index);
            if (slot >= capacity) {
                throw new IllegalStateException(
                    "Native World hotbar slot " + slot
                        + " exceeds NPC hotbar capacity " + capacity
                );
            }
            if (Item.getAssetMap().getAsset(itemId) == null) {
                throw new IllegalStateException(
                    "Native World hotbar item is unknown: " + itemId
                );
            }
            if (!hotbar.setItemStackForSlot(
                (short) slot,
                new ItemStack(itemId, quantity)
            ).succeeded()) {
                throw new IllegalStateException(
                    "Native NPC rejected hotbar item "
                        + itemId + " in slot " + slot
                );
            }
        }
        int activeSlot = options.nativeWorldHotbarSlots().get(0);
        inventory.setActiveHotbarSlot(agentRef, (byte) activeSlot, store);
    }

    private void installSyntheticPlayer(
        int actorId,
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        PlayerRef installed = NativeHeadlessWorldVerbContext.install(
            actor,
            store,
            tickAnchor
        );
        syntheticPlayerRefsByActor[actorId] = installed;
        if (actorId == 0) syntheticPlayerRef = installed;
        Player player = store.getComponent(
            actor,
            Player.getComponentType()
        );
        String heldItem = player == null
            ? ""
            : activeItemId(player.getInventory());
        if (
            actorId == 0
                && syntheticWorldInteractionsEnabled()
                && !SYNTHETIC_PLACE_ITEM_ID.equals(heldItem)
        ) {
            throw new IllegalStateException(
                "Synthetic Player inventory is not linked to the NPC hotbar"
            );
        }
    }

    private boolean hasNativeAttackAction() {
        return nativeCombatRootInteraction != null
            || policyActorState(0).hasRoleAttack();
    }

    private int nativeAttackActionCount() {
        return nativeCombatRootInteraction == null
            ? policyActorState(0).roleAttackCount()
            : 1;
    }

    private String nativeAttackActionPaths() {
        if (nativeCombatRootInteraction != null) {
            return options.nativeCombatInteractionId();
        }
        return String.join(
            ";",
            policyActorState(0).roleAttacks().stream()
                .map(ActionAttack::getBreadCrumbs)
                .map(path -> path == null ? "" : path)
                .toList()
        );
    }

    private void createFidelityFixture() {
        if (
            options.fidelityFixture().equals(
                EnvironmentOptions.STATIC_REGION_FIXTURE
            )
        ) {
            if (options.hasSpawn() || !usesGeneratedWorld()) {
                throw new IllegalArgumentException(
                    "static_region requires generated Hytale world and default spawn"
                );
            }
            return;
        }
        if (options.hasSpawn() || !options.world().equals("flat")) {
            if (!options.fidelityFixture().equals("default")) {
                throw new IllegalArgumentException(
                    "Named fidelity fixtures require flat world and default spawn"
                );
            }
            return;
        }
        String fixture = options.fidelityFixture();
        if (fixture.equals("default") && !taskId.equals("native_fidelity")) return;
        boolean combatFixture = fixture.equals("los_wall")
            || isDiagonalLosFixture(fixture)
            || fixture.equals(EnvironmentOptions.TARGET_MEMORY_FIXTURE)
            || fixture.equals(EnvironmentOptions.PATH_FOLLOWER_FIXTURE)
            || nativeDuelFixture();
        if (combatFixture && !taskId.equals("kill_trork")) {
            throw new IllegalArgumentException(
                fixture + " requires the kill_trork native task"
            );
        }
        if (!fixture.equals("default")
            && !combatFixture
            && !taskId.equals("native_fidelity")) {
            throw new IllegalArgumentException(
                fixture + " requires the native_fidelity task"
            );
        }
        runOnWorld(() -> {
            if (
                fixture.equals(EnvironmentOptions.DIVE_MOTION_FIXTURE)
            ) {
                createDiveMotionFixture();
                return;
            }
            if (fixture.equals("ledge")) {
                for (int x = -2; x <= 2; x++) {
                    for (int y = 62; y <= 64; y++) {
                        for (int z = -5; z <= -1; z++) {
                            world.setBlock(x, y, z, BlockType.EMPTY_KEY);
                        }
                    }
                }
                return;
            }
            if (fixture.equals("ceiling")) {
                for (int x = -1; x <= 1; x++) {
                    for (int z = -1; z <= 1; z++) {
                        world.setBlock(x, 67, z, "Rock_Stone");
                    }
                }
                return;
            }
            if (fixture.equals("half_block")) {
                world.setBlock(0, 65, -2, "Rock_Stone_Half");
                return;
            }
            if (fixture.equals("los_wall")) {
                for (int x = -1; x <= 1; x++) {
                    for (int y = 65; y <= 68; y++) {
                        world.setBlock(x, y, -1, "Rock_Stone");
                    }
                }
                return;
            }
            if (fixture.equals("los_diagonal_graze")) {
                world.setBlock(1, 66, 0, "Rock_Stone");
                return;
            }
            if (fixture.equals("los_diagonal_block")) {
                world.setBlock(1, 66, -1, "Rock_Stone");
                return;
            }
            if (fixture.equals(EnvironmentOptions.TARGET_MEMORY_FIXTURE)) {
                return;
            }
            if (fixture.equals(EnvironmentOptions.PATH_FOLLOWER_FIXTURE)) {
                for (
                    int x = -PATH_FOLLOWER_WALL_HALF_WIDTH;
                    x <= PATH_FOLLOWER_WALL_HALF_WIDTH;
                    x++
                ) {
                    for (int y = 65; y <= 68; y++) {
                        world.setBlock(
                            x,
                            y,
                            PATH_FOLLOWER_WALL_Z,
                            "Rock_Stone"
                        );
                    }
                }
                return;
            }
            if (
                fixture.equals(
                    EnvironmentOptions.SYNTHETIC_WORLD_VERBS_FIXTURE
                )
                    || fixture.equals(
                        EnvironmentOptions.SYNTHETIC_BLOCK_USE_FIXTURE
                    )
            ) {
                // Bound after native warmup to the actor's settled transform.
                return;
            }
            // Default native-fidelity fixture: a four-block-high stone wall
            // directly ahead, far enough away for acceleration measurements.
            for (int x = -3; x <= 3; x++) {
                for (int y = 65; y <= 68; y++) {
                    world.setBlock(x, y, -4, "Rock_Stone");
                }
            }
        }, INITIALIZATION_TIMEOUT);
    }

    private void createDiveMotionFixture() {
        Fluid water = Fluid.getAssetMap().getAsset("Water_Source");
        if (water == null || water == Fluid.EMPTY) {
            throw new IllegalStateException(
                "Installed assets do not provide Water_Source"
            );
        }
        int fluidId = Fluid.getAssetMap().getIndex(water.getId());
        byte fluidLevel = (byte) water.getMaxFluidLevel();
        Store<ChunkStore> store = world.getChunkStore().getStore();
        for (
            int x = -DIVE_POOL_HALF_WIDTH;
            x <= DIVE_POOL_HALF_WIDTH;
            x++
        ) {
            for (
                int z = -DIVE_POOL_HALF_WIDTH;
                z <= DIVE_POOL_HALF_WIDTH;
                z++
            ) {
                boolean wall = Math.abs(x) == DIVE_POOL_HALF_WIDTH
                    || Math.abs(z) == DIVE_POOL_HALF_WIDTH;
                for (int y = DIVE_POOL_MIN_Y; y <= DIVE_POOL_MAX_Y; y++) {
                    if (wall) {
                        world.setBlock(x, y, z, "Rock_Stone");
                    } else if (y < DIVE_POOL_MAX_Y) {
                        setFixtureFluid(
                            store,
                            x,
                            y,
                            z,
                            fluidId,
                            fluidLevel
                        );
                    }
                }
            }
        }
    }

    private void setFixtureFluid(
        Store<ChunkStore> store,
        int x,
        int y,
        int z,
        int fluidId,
        byte fluidLevel
    ) {
        WorldChunk chunk = world.getChunkIfLoaded(
            ChunkUtil.indexChunkFromBlock(x, z)
        );
        if (chunk == null || chunk.getReference() == null) {
            throw new IllegalStateException(
                "Dive fixture chunk is not loaded"
            );
        }
        ChunkColumn column = store.getComponent(
            chunk.getReference(),
            ChunkColumn.getComponentType()
        );
        if (column == null) {
            throw new IllegalStateException(
                "Dive fixture chunk has no column"
            );
        }
        Ref<ChunkStore> section = column.getSection(
            ChunkUtil.chunkCoordinate(y)
        );
        FluidSection fluidSection = store.ensureAndGetComponent(
            section,
            FluidSection.getComponentType()
        );
        fluidSection.setFluid(x, y, z, fluidId, fluidLevel);
    }

    private void initializeTask(Observation observation) {
        previousHealth = observation.health();
        NativeSnapshot snapshot = lastSnapshot;
        previousTargetHealth = snapshot == null ? 0.0 : snapshot.targetHealth();
        Random random = new Random(seed);
        double angle = random.nextDouble() * Math.PI * 2.0;
        double distance = 10.0 + random.nextDouble() * 6.0;
        targetX = observation.x() + Math.cos(angle) * distance;
        targetY = observation.y();
        targetZ = observation.z() + Math.sin(angle) * distance;
        previousTargetDistance = horizontalDistance(observation.x(), observation.z());
    }

    private StepResult advance(
        AgentAction action,
        int requestedTicks,
        boolean episodeStep
    ) {
        return advance(
            NativeGroupAction.legacy(action),
            requestedTicks,
            episodeStep
        );
    }

    private StepResult advance(
        NativeGroupAction actions,
        int requestedTicks,
        boolean episodeStep
    ) {
        validateNativeGroupAction(actions);
        NativeSnapshot snapshot = lastSnapshot;
        Observation observation = snapshot == null
            ? null
            : snapshot.observation();
        loadAndPinGeometryChunkHalo(
            observation == null ? spawnX : observation.x(),
            observation == null ? spawnZ : observation.z()
        );
        PendingStep step = new PendingStep(actions, requestedTicks, episodeStep);
        step.legacyActorZeroWorldVerbExecution =
            WorldVerbTelemetry.Execution.unrequested();
        step.syntheticWorldInteraction =
            syntheticWorldInteractionEvidence(actions.primaryAction());
        if (!pending.compareAndSet(null, step)) {
            throw new IllegalStateException("A native step is already in progress");
        }
        try {
            world.execute(() -> {
                if (closed) {
                    if (pending.compareAndSet(step, null)) {
                        step.future.completeExceptionally(
                            new IllegalStateException("Native environment is closed")
                        );
                    }
                    return;
                }
                if (actions.actors().isEmpty()) {
                    pinNativeDuelTargets(world.getEntityStore().getStore());
                }
                setWorldPausedWithoutAnchorBroadcast(false);
            });
            long timeoutSeconds = Math.max(10L, requestedTicks / 10L + 10L);
            return step.future.get(timeoutSeconds, TimeUnit.SECONDS);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            pending.compareAndSet(step, null);
            throw new IllegalStateException("Interrupted during native Hytale step", exception);
        } catch (ExecutionException exception) {
            pending.compareAndSet(step, null);
            Throwable cause = exception.getCause();
            if (cause instanceof RuntimeException runtime) throw runtime;
            throw new IllegalStateException("Native Hytale step failed", cause);
        } catch (TimeoutException exception) {
            pending.compareAndSet(step, null);
            world.execute(() -> setWorldPausedWithoutAnchorBroadcast(true));
            throw new IllegalStateException(
                "Timed out after " + step.executedTicks + "/" + requestedTicks
                    + " native engine ticks",
                exception
            );
        }
    }

    private StepResult finishStep(PendingStep step, NativeSnapshot snapshot) {
        lastSnapshot = snapshot;
        AudioFrame audio = step.episodeStep
            ? step.audio.finish(step.simulatedSeconds)
            : AudioFrame.headlessPartial(List.of());
        Observation observation = snapshot.observation().withAudio(audio);
        if (!step.episodeStep) {
            StepResult warmup = new StepResult(
                observation,
                0.0,
                false,
                false,
                episodeEngineTicks,
                episodeSteps,
                buildInfo(
                    snapshot,
                    step.executedTicks,
                    step.simulatedSeconds,
                    step.minimumDeltaSeconds(),
                    step.maximumDeltaSeconds,
                    step.action
                )
            );
            lastResult = warmup;
            return warmup;
        }

        episodeEngineTicks += step.executedTicks;
        episodeSteps++;
        boolean dead = observation.health() <= 0.0 || !snapshot.agentPresent();
        boolean reachedTarget = taskId.equals("navigate")
            && horizontalDistance(observation.x(), observation.z()) <= NAVIGATE_SUCCESS_RADIUS;
        boolean targetDefeated = taskId.equals("kill_trork")
            && (!snapshot.targetPresent() || snapshot.targetHealth() <= 0.0);
        boolean terminated = dead || reachedTarget || targetDefeated;
        boolean truncated = !terminated
            && maxEpisodeSteps > 0
            && episodeSteps >= maxEpisodeSteps;
        double reward = computeReward(
            snapshot,
            step.executedTicks,
            dead,
            reachedTarget,
            targetDefeated
        );
        done = terminated || truncated;

        previousHealth = observation.health();
        previousTargetHealth = snapshot.targetHealth();
        previousTargetDistance = horizontalDistance(observation.x(), observation.z());
        StepResult result = new StepResult(
            observation,
            reward,
            terminated,
            truncated,
            episodeEngineTicks,
            episodeSteps,
            buildInfo(
                snapshot,
                step.executedTicks,
                step.simulatedSeconds,
                step.minimumDeltaSeconds(),
                step.maximumDeltaSeconds,
                step.action
            )
        );
        lastResult = result;
        return result;
    }

    private double computeReward(
        NativeSnapshot snapshot,
        int executedTicks,
        boolean dead,
        boolean reachedTarget,
        boolean targetDefeated
    ) {
        Observation observation = snapshot.observation();
        if (taskId.equals("native_fidelity")) return 0.0;
        if (taskId.equals("survive")) {
            return executedTicks * 0.01 + (observation.health() - previousHealth);
        }
        if (taskId.equals("kill_trork")) {
            double damageDealt = Math.max(0.0, previousTargetHealth - snapshot.targetHealth());
            double damageTaken = Math.max(0.0, previousHealth - observation.health());
            double reward =
                damageDealt * COMBAT_RULES.reward().targetDamageScale()
                + damageTaken * COMBAT_RULES.reward().agentDamageScale();
            if (targetDefeated && previousTargetHealth > 0.0) {
                reward += COMBAT_RULES.reward().completion();
            }
            if (dead && previousHealth > 0.0) {
                reward += COMBAT_RULES.reward().death();
            }
            return reward;
        }
        double distance = horizontalDistance(observation.x(), observation.z());
        double reward = previousTargetDistance - distance;
        if (reachedTarget) reward += 100.0;
        if (dead) reward -= 100.0;
        return reward;
    }

    private NativeSnapshot captureSnapshot(Store<EntityStore> store) {
        TransformComponent transform = store.getComponent(
            agentRef,
            TransformComponent.getComponentType()
        );
        NPCEntity npc = store.getComponent(agentRef, NPCEntity.getComponentType());
        if (transform == null || npc == null) {
            nativeUseAvailable = false;
            nativeUseAvailableInteractionId = "";
            nativeUseAvailableBlockInteractionId = "";
            nativeUseAvailableItemId = "";
            nativeUseAvailableSourceContainer = "";
            nativeUseAvailableSourceSlot = -1;
            nativeUseAvailableSourceQuantity = -1;
            nativeUseAvailableTarget = null;
            nativeUseAvailableMaximumDistance = 0.0;
            nativeUseUnavailableReason = "controlled_entity_unavailable";
            return deadSnapshot();
        }

        Vector3d position = transform.getPosition();
        Velocity velocity = store.getComponent(agentRef, Velocity.getComponentType());
        HeadRotation head = store.getComponent(agentRef, HeadRotation.getComponentType());
        EntityStatMap stats = store.getComponent(
            agentRef,
            EntityStatsModule.get().getEntityStatMapComponentType()
        );
        EntityStatValue healthStat = stat(stats, "Health");
        EntityStatValue staminaStat = stat(stats, "Stamina");
        EntityStatValue manaStat = stat(stats, "Mana");
        Role role = npc.getRole();
        NativePolicyActorState actorState = policyActorState(0);
        NativeBlockUse.Availability useAvailability = NativeBlockUse.inspect(
            agentRef,
            store,
            world,
            interactionManager(agentRef, store),
            actorState.desiredPitch(),
            actorState.desiredYaw(),
            headlessWorldVerbsEnabled()
        );
        nativeUseAvailable = useAvailability.available();
        nativeUseAvailableInteractionId =
            useAvailability.interactionId();
        nativeUseAvailableBlockInteractionId =
            useAvailability.blockInteractionId();
        nativeUseAvailableItemId = useAvailability.itemId();
        nativeUseAvailableSourceContainer =
            useAvailability.sourceContainer();
        nativeUseAvailableSourceSlot = useAvailability.sourceSlot();
        nativeUseAvailableSourceQuantity =
            useAvailability.sourceQuantity();
        nativeUseAvailableTarget = useAvailability.target();
        nativeUseAvailableMaximumDistance =
            useAvailability.maximumDistance();
        nativeUseUnavailableReason = useAvailability.rejectReason();

        double health = healthStat == null
            ? (role == null ? 0.0 : role.getInitialMaxHealth())
            : healthStat.get();
        double maxHealth = healthStat == null
            ? (role == null ? health : role.getInitialMaxHealth())
            : healthStat.getMax();
        double stamina = staminaStat == null ? 0.0 : staminaStat.get();
        double mana = manaStat == null ? 0.0 : manaStat.get();
        float pitchRadians = head == null
            ? transform.getRotation().pitch()
            : head.getRotation().pitch();

        int baseX = (int) Math.floor(position.x);
        int baseY = (int) Math.floor(position.y);
        int baseZ = (int) Math.floor(position.z);
        TargetState target = captureTargetState(store, position);
        GeometryFrame geometry = RegionGeometryCapture.captureGeometry(nativeCellSemanticsCache, world, 
            store,
            agentRef,
            targetRef,
            role,
            baseX,
            baseY,
            baseZ,
            position,
            target.present()
        );
        List<int[]> nearbyBlocks = geometry.cells().stream()
            .filter(cell -> cell.runtimeBlockId() != 0)
            .map(cell -> new int[] {
                cell.dx(),
                cell.dy(),
                cell.dz(),
                cell.runtimeBlockId()
            })
            .toList();
        int blockBelow = world.getBlock(baseX, baseY - 1, baseZ);
        WorldTimeResource time = store.getResource(WorldTimeResource.getResourceType());
        int timeOfDay = time == null
            ? 0
            : Math.max(0, Math.min(24000, Math.round(time.getDayProgress() * 24000.0f)));
        Inventory inventory = npc.getInventory();
        int[] inventoryIds = inventoryIds(inventory);
        int activeHotbar = inventory == null ? 0 : Byte.toUnsignedInt(inventory.getActiveHotbarSlot());
        List<int[]> nearbyEntities = captureNearbyEntities(store, position);
        CombatSupport combat = role == null ? null : role.getCombatSupport();
        CombatTelemetry combatTelemetry = new CombatTelemetry(
            combat != null && combat.isExecutingAttack(),
            geometry.targetLineOfSightValid()
                ? geometry.targetLineOfSight()
                : target.present(),
            targetAttackPhase,
            targetAttackProgress,
            targetAttackIndex,
            targetAttackElapsedTicks,
            targetAttackId,
            target.facingErrorDegrees(),
            target.yawDegrees(),
            target.headYawDegrees(),
            target.headPitchDegrees(),
            target.velocityX(),
            target.velocityZ()
        );
        int nativeNpcCount = store.getEntityCountFor(Query.and(
            NPCEntity.getComponentType(),
            TransformComponent.getComponentType()
        ));
        EntityModelEvidence agentModelEvidence = captureEntityModelEvidence(
            store,
            agentRef
        );
        EntityModelEvidence targetModelEvidence = target.present()
            ? captureEntityModelEvidence(store, targetRef)
            : EntityModelEvidence.absent();
        MovementStateFrame agentMovementStates = captureMovementStates(
            store,
            agentRef
        );
        MovementStateFrame targetMovementStates = target.present()
            ? captureMovementStates(store, targetRef)
            : MovementStateFrame.unavailable();

        Observation observation = new Observation(
            position.x,
            position.y,
            position.z,
            velocity == null ? 0.0 : velocity.getX(),
            velocity == null ? 0.0 : velocity.getY(),
            velocity == null ? 0.0 : velocity.getZ(),
            normalizeDegrees(Math.toDegrees(transform.getRotation().yaw())),
            Math.toDegrees(pitchRadians),
            health,
            maxHealth,
            0,
            stamina,
            mana,
            inventoryIds,
            List.copyOf(nearbyBlocks),
            nearbyEntities,
            timeOfDay,
            0,
            geometry
        );
        observation = observation.withNativeInventory(
            captureNativeInventory(inventory)
        );
        observation = observation.withNativeActorEvidence(
            captureNativeActorEvidence(
                store,
                geometry,
                target,
                agentMovementStates,
                targetMovementStates
            )
        );
        return new NativeSnapshot(
            observation,
            true,
            role != null && role.isOnGround(),
            npc.getRoleName(),
            blockBelow,
            world.getTick(),
            activeHotbar,
            healthStat != null,
            staminaStat != null,
            manaStat != null,
            target.present(),
            target.roleName(),
            target.health(),
            target.maxHealth(),
            target.distance(),
            target.x(),
            target.y(),
            target.z(),
            nativeNpcCount,
            agentModelEvidence,
            targetModelEvidence,
            agentMovementStates,
            targetMovementStates,
            combatTelemetry
        );
    }



    private NativeSnapshot deadSnapshot() {
        NativeSnapshot previous = lastSnapshot;
        Observation old = previous == null ? Observation.empty() : previous.observation();
        Observation dead = new Observation(
            old.x(), old.y(), old.z(),
            0.0, 0.0, 0.0,
            old.yaw(), old.pitch(),
            0.0, old.maxHealth(), old.foodBuffTimer(),
            old.stamina(), old.mana(), old.inventory(),
            old.nearbyBlocks(), old.nearbyEntities(), old.timeOfDay(),
            old.craftableRecipeCount(),
            old.geometry()
        );
        return new NativeSnapshot(
            dead,
            false,
            false,
            options.npcRole(),
            previous == null ? 0 : previous.blockBelowId(),
            world == null ? 0L : world.getTick(),
            previous == null ? 0 : previous.activeHotbarSlot(),
            previous != null && previous.hasHealthStat(),
            previous != null && previous.hasStaminaStat(),
            previous != null && previous.hasManaStat(),
            previous != null && previous.targetPresent(),
            previous == null ? "" : previous.targetRoleName(),
            previous == null ? 0.0 : previous.targetHealth(),
            previous == null ? 0.0 : previous.targetMaxHealth(),
            previous == null ? -1.0 : previous.targetDistance(),
            previous == null ? 0.0 : previous.targetX(),
            previous == null ? 0.0 : previous.targetY(),
            previous == null ? 0.0 : previous.targetZ(),
            previous == null ? 0 : previous.nativeNpcCount(),
            EntityModelEvidence.absent(),
            EntityModelEvidence.absent(),
            MovementStateFrame.unavailable(),
            MovementStateFrame.unavailable(),
            CombatTelemetry.idle(previous != null && previous.targetPresent())
        );
    }

    private List<int[]> captureNearbyEntities(Store<EntityStore> store, Vector3d agentPosition) {
        List<EntityReading> readings = new ArrayList<>();
        Query<EntityStore> query = Query.and(
            NPCEntity.getComponentType(),
            TransformComponent.getComponentType()
        );
        store.forEachChunk(query, (chunk, commandBuffer) -> {
            for (int index = 0; index < chunk.size(); index++) {
                Ref<EntityStore> reference = chunk.getReferenceTo(index);
                if (sameEntity(reference, agentRef)) continue;
                NPCEntity entity = chunk.getComponent(index, NPCEntity.getComponentType());
                TransformComponent entityTransform = chunk.getComponent(
                    index,
                    TransformComponent.getComponentType()
                );
                if (entity == null || entityTransform == null) continue;
                Vector3d entityPosition = entityTransform.getPosition();
                double dx = entityPosition.x - agentPosition.x;
                double dz = entityPosition.z - agentPosition.z;
                double distanceSquared = dx * dx + dz * dz;
                if (distanceSquared >= NEARBY_ENTITY_RADIUS * NEARBY_ENTITY_RADIUS) continue;
                double entityHealth = entityHealth(store, reference, entity);
                if (entityHealth <= 0.0) continue;
                readings.add(new EntityReading(
                    new int[] {
                        canonicalEntityType(entity.getRoleName()),
                        ObservationEncoding.encodeNearbyEntityScalar(dx),
                        ObservationEncoding.encodeNearbyEntityScalar(dz),
                        ObservationEncoding.encodeNearbyEntityScalar(entityHealth)
                    },
                    distanceSquared,
                    reference.getIndex()
                ));
            }
        });
        readings.sort(
            Comparator.comparingDouble(EntityReading::distanceSquared)
                .thenComparingInt(EntityReading::entityIndex)
        );
        List<int[]> encoded = new ArrayList<>(
            Math.min(MAX_NEARBY_ENTITIES, readings.size())
        );
        for (int i = 0; i < readings.size() && i < MAX_NEARBY_ENTITIES; i++) {
            encoded.add(readings.get(i).encoded());
        }
        return List.copyOf(encoded);
    }

    private TargetState captureTargetState(Store<EntityStore> store, Vector3d agentPosition) {
        Ref<EntityStore> reference = targetRef;
        if (reference == null || !reference.isValid()) return TargetState.absent();
        NPCEntity entity = store.getComponent(reference, NPCEntity.getComponentType());
        TransformComponent transform = store.getComponent(
            reference,
            TransformComponent.getComponentType()
        );
        if (entity == null || transform == null) return TargetState.absent();
        double health = entityHealth(store, reference, entity);
        EntityStatMap stats = store.getComponent(
            reference,
            EntityStatsModule.get().getEntityStatMapComponentType()
        );
        EntityStatValue healthStat = stat(stats, "Health");
        Role role = entity.getRole();
        double maxHealth = healthStat == null
            ? (role == null ? health : role.getInitialMaxHealth())
            : healthStat.getMax();
        Vector3d targetPosition = transform.getPosition();
        Velocity velocity = store.getComponent(reference, Velocity.getComponentType());
        double distance = Math.hypot(
            targetPosition.x - agentPosition.x,
            targetPosition.z - agentPosition.z
        );
        double yawDegrees = normalizeDegrees(
            Math.toDegrees(transform.getRotation().yaw())
        );
        HeadRotation head = store.getComponent(
            reference,
            HeadRotation.getComponentType()
        );
        double headYawDegrees = head == null
            ? yawDegrees
            : normalizeDegrees(Math.toDegrees(head.getRotation().yaw()));
        double headPitchDegrees = head == null
            ? normalizeDegrees(Math.toDegrees(transform.getRotation().pitch()))
            : normalizeDegrees(Math.toDegrees(head.getRotation().pitch()));
        double agentBearing = Math.toDegrees(Math.atan2(
            -(agentPosition.x - targetPosition.x),
            -(agentPosition.z - targetPosition.z)
        ));
        double facingError = normalizeDegrees(agentBearing - yawDegrees);
        return new TargetState(
            true,
            entity.getRoleName(),
            health,
            maxHealth,
            distance,
            targetPosition.x,
            targetPosition.y,
            targetPosition.z,
            yawDegrees,
            headYawDegrees,
            headPitchDegrees,
            facingError,
            velocity == null ? 0.0 : velocity.getX(),
            velocity == null ? 0.0 : velocity.getZ()
        );
    }

    private int[] inventoryIds(Inventory inventory) {
        int[] result = new int[INVENTORY_SIZE];
        if (inventory == null) return result;
        int offset = copyContainer(inventory.getHotbar(), result, 0);
        copyContainer(inventory.getStorage(), result, offset);
        return result;
    }

    private NativeInventoryFrame captureNativeInventory(Inventory inventory) {
        if (inventory == null) {
            return NativeInventoryFrame.unavailable("inventory_unavailable");
        }
        ItemContainer storage = inventory.getStorage();
        ItemContainer armor = inventory.getArmor();
        ItemContainer hotbar = inventory.getHotbar();
        ItemContainer utility = inventory.getUtility();
        ItemContainer tools = inventory.getTools();
        ItemContainer backpack = inventory.getBackpack();
        return new NativeInventoryFrame(
            true,
            "",
            hotbar == null ? -1 : inventory.getActiveHotbarSlot(),
            utility == null ? -1 : inventory.getActiveUtilitySlot(),
            tools == null ? -1 : inventory.getActiveToolsSlot(),
            List.of(
                captureNativeInventoryContainer("storage", -2, storage),
                captureNativeInventoryContainer("armor", -3, armor),
                captureNativeInventoryContainer("hotbar", -1, hotbar),
                captureNativeInventoryContainer("utility", -5, utility),
                captureNativeInventoryContainer("tools", -8, tools),
                captureNativeInventoryContainer("backpack", -9, backpack)
            )
        );
    }

    private NativeInventoryFrame.Container captureNativeInventoryContainer(
        String name,
        int sectionId,
        ItemContainer container
    ) {
        if (container == null) {
            return NativeInventoryFrame.Container.unavailable(
                name,
                sectionId,
                "inventory_container_unavailable"
            );
        }
        int capacity = Short.toUnsignedInt(container.getCapacity());
        List<NativeInventoryFrame.Slot> occupied = new ArrayList<>();
        for (int slot = 0; slot < capacity; slot++) {
            ItemStack stack = container.getItemStack((short) slot);
            if (ItemStack.isEmpty(stack)) continue;
            String itemId = stack.getItemId();
            occupied.add(new NativeInventoryFrame.Slot(
                slot,
                itemId,
                itemIds.getOrDefault(itemId, 0),
                stack.getQuantity(),
                stack.getDurability(),
                stack.getMaxDurability(),
                stack.getMetadata() != null
            ));
        }
        return new NativeInventoryFrame.Container(
            name,
            sectionId,
            true,
            "",
            capacity,
            occupied
        );
    }

    private int copyContainer(ItemContainer container, int[] destination, int offset) {
        if (container == null || offset >= destination.length) return offset;
        int capacity = Short.toUnsignedInt(container.getCapacity());
        for (int slot = 0; slot < capacity && offset < destination.length; slot++, offset++) {
            ItemStack stack = container.getItemStack((short) slot);
            if (!ItemStack.isEmpty(stack)) {
                destination[offset] = itemIds.getOrDefault(stack.getItemId(), 0);
            }
        }
        return offset;
    }

    private Map<String, Object> buildInfo(
        NativeSnapshot snapshot,
        int executedTicks,
        double simulatedSeconds,
        double minimumDeltaSeconds,
        double maximumDeltaSeconds,
        AgentAction action
    ) {
        LinkedHashMap<String, Object> info = new LinkedHashMap<>();
        info.put("backend", "native");
        info.put("native_server_version", NATIVE_SERVER_VERSION);
        info.put("worldgen_provider", worldgenProvider);
        info.put("worldgen_version", worldgenVersion);
        info.put("worldgen_seed", seed);
        info.put("worldgen_structure", worldgenWorldStructure);
        info.put("fidelity", "server_authoritative_headless_npc");
        info.put("fidelity_scope",
            "world_ticks,worldgen,npc_steering,physics,collision,stats,blocks,inventory,"
                + "npc_combat,interaction_chains");
        info.put("world", worldName == null ? "" : worldName);
        NativeCellSemanticsCache.Stats cellCache =
            nativeCellSemanticsCache.stats();
        info.put(
            "native_cell_semantics_cache_schema",
            NativeCellSemanticsCache.SCHEMA
        );
        info.put(
            "native_cell_semantics_cache_enabled",
            NATIVE_CELL_SEMANTICS_CACHE
        );
        info.put("native_cell_semantics_cache_capacity", cellCache.capacity());
        info.put("native_cell_semantics_cache_entries", cellCache.entries());
        info.put("native_cell_semantics_cache_hits", cellCache.hits());
        info.put("native_cell_semantics_cache_misses", cellCache.misses());
        info.put(
            "native_cell_semantics_cache_generation_resets",
            cellCache.generationResets()
        );
        info.put("native_cell_semantics_cache_hit_rate", cellCache.hitRate());
        List<String> policyActorIdentities = new ArrayList<>(
            policyActorStates.length
        );
        for (int actorId = 0; actorId < policyActorStates.length; actorId++) {
            String identity = this.policyActorIdentities[actorId];
            policyActorIdentities.add(identity == null ? "" : identity);
        }
        String policyActorIdentity = policyActorIdentities.isEmpty()
            ? ""
            : policyActorIdentities.get(0);
        info.put("native_policy_actor_identity", policyActorIdentity);
        info.put("native_policy_actor_identities", policyActorIdentities);
        info.put(
            "native_policy_world_action_capture_schema",
            NativePolicyWorldActionCapture.SCHEMA
        );
        info.put(
            "native_policy_world_action_capture_version",
            NativePolicyWorldActionCapture.VERSION
        );
        info.put(
            "native_policy_world_action_capture_contract_sha256",
            NativePolicyWorldActionCapture.CONTRACT_SHA256
        );
        info.put("world_template", options.world());
        info.put(
            "fidelity_fixture",
            fidelityFixtureDescription()
        );
        info.put(
            "block_ticking_enabled",
            world != null && world.getWorldConfig().isBlockTicking()
        );
        info.put("resolved_spawn_x", spawnX);
        info.put("resolved_spawn_y", spawnY);
        info.put("resolved_spawn_z", spawnZ);
        info.put("npc_role", snapshot.roleName());
        info.put("combat_target_role", combatTargetRole());
        info.put("native_actor_target_attitude", nativeActorTargetAttitude);
        info.put("native_target_actor_attitude", nativeTargetActorAttitude);
        info.put("native_mutual_hostility", nativeMutualHostility);
        info.put("native_hostility_authority", "WorldSupport.getAttitude");
        info.put("task", taskId);
        info.put("curriculum_phase", curriculumPhase);
        List<String> supportedActions = new ArrayList<>(
            List.of(
                "forward",
                "back",
                "left",
                "right",
                "jump",
                "camera_delta_yaw",
                "camera_delta_pitch",
                "hotbar_slot",
                "world_move_direction",
                "use"
            )
        );
        List<String> unsupportedActions = new ArrayList<>();
        if (options.nativeWorldVerbs()) {
            supportedActions.add("place_block");
            supportedActions.add("break_block");
            supportedActions.add("craft_recipe");
        } else {
            unsupportedActions.add("place_block");
            unsupportedActions.add("break_block");
            unsupportedActions.add("craft_recipe");
        }
        if (hasNativeAttackAction()) supportedActions.add("attack");
        else unsupportedActions.add("attack");
        if (nativeGuardInteraction != null) supportedActions.add("guard_held");
        else unsupportedActions.add("guard_held");
        if (!nativeCombatAbilities.isEmpty()) supportedActions.add("ability_slot");
        else unsupportedActions.add("ability_slot");
        if (
            nativeDodgeLeftInteraction != null
                && nativeDodgeRightInteraction != null
        ) {
            supportedActions.add("dodge_direction");
        } else {
            unsupportedActions.add("dodge_direction");
        }
        info.put("supported_actions", String.join(",", supportedActions));
        info.put("unsupported_actions", String.join(",", unsupportedActions));
        info.put("requested_unsupported_actions", requestedUnsupportedActions(action));
        PendingStep activeStep = pending.get();
        info.put("native_group_action_schema", NativeGroupActionContract.SCHEMA);
        info.put("native_group_action_version", NativeGroupActionContract.VERSION);
        info.put(
            "native_group_action_contract_sha256",
            NativeGroupActionContract.sha256()
        );
        info.put(
            "native_group_action_actor_capacity",
            NativeGroupActionContract.actorCapacity()
        );
        info.put(
            "native_group_action_available",
            taskId.equals("kill_trork")
        );
        info.put(
            "native_group_action_empty_actor_step",
            NativeGroupActionContract.EMPTY_ACTOR_STEP
        );
        info.put(
            "native_group_action_nonzero_actor_surface",
            NativeGroupActionContract.NONZERO_ACTOR_SURFACE
        );
        info.put(
            "native_group_action_binding_schema",
            NativePolicyCombatBindingSpec.SCHEMA
        );
        info.put(
            "native_group_action_binding_version",
            NativePolicyCombatBindingSpec.VERSION
        );
        info.put(
            "native_group_action_policy_combat_bound",
            nativePolicyCombatBoundByActor()
        );
        info.put(
            "native_group_action_ability_slot_masks",
            nativePolicyCombatAbilityMasksByActor()
        );
        info.put(
            "native_group_action_guard_available",
            nativePolicyCombatGuardByActor()
        );
        info.put(
            "native_group_action_dodge_available",
            nativePolicyCombatDodgeByActor()
        );
        info.put(
            "native_group_action_item_ids",
            nativePolicyCombatItemIdsByActor()
        );
        info.put(
            "native_group_action_requested_entity_ids",
            activeStep == null ? "" : activeStep.requestedEntityIds()
        );
        info.put(
            "native_group_action_control_ticks",
            activeStep == null ? "" : activeStep.controlTicksByActor()
        );
        info.put(
            "native_group_action_attack_requested",
            activeStep == null ? "" : activeStep.attackRequestedByActor()
        );
        info.put(
            "native_group_action_attack_accepted",
            activeStep == null ? "" : activeStep.attackAcceptedByActor()
        );
        info.put(
            "native_group_action_attack_reject_reasons",
            activeStep == null ? "" : activeStep.attackRejectReasonsByActor()
        );
        info.put(
            "native_group_action_ability_requested",
            activeStep == null ? "" : activeStep.abilityRequestedByActor()
        );
        info.put(
            "native_group_action_ability_accepted",
            activeStep == null ? "" : activeStep.abilityAcceptedByActor()
        );
        info.put(
            "native_group_action_ability_started",
            activeStep == null ? "" : activeStep.abilityStartedByActor()
        );
        info.put(
            "native_group_action_ability_finished",
            activeStep == null ? "" : activeStep.abilityFinishedByActor()
        );
        info.put(
            "native_group_action_ability_failed",
            activeStep == null ? "" : activeStep.abilityFailedByActor()
        );
        info.put(
            "native_group_action_ability_active",
            activeStep == null ? "" : activeStep.abilityActiveByActor()
        );
        info.put(
            "native_group_action_ability_slots",
            activeStep == null ? "" : activeStep.abilitySlotsByActor()
        );
        info.put(
            "native_group_action_ability_reject_reasons",
            activeStep == null ? "" : activeStep.abilityRejectReasonsByActor()
        );
        info.put(
            "native_group_action_ability_interaction_ids",
            activeStep == null ? "" : activeStep.abilityInteractionIdsByActor()
        );
        info.put(
            "native_group_action_guard_requested",
            activeStep == null ? "" : activeStep.guardRequestedByActor()
        );
        info.put(
            "native_group_action_guard_accepted",
            activeStep == null ? "" : activeStep.guardAcceptedByActor()
        );
        info.put(
            "native_group_action_guard_started",
            activeStep == null ? "" : activeStep.guardStartedByActor()
        );
        info.put(
            "native_group_action_guard_finished",
            activeStep == null ? "" : activeStep.guardFinishedByActor()
        );
        info.put(
            "native_group_action_guard_active",
            activeStep == null ? "" : activeStep.guardActiveByActor()
        );
        info.put(
            "native_group_action_guard_reject_reasons",
            activeStep == null ? "" : activeStep.guardRejectReasonsByActor()
        );
        info.put(
            "native_group_action_dodge_requested",
            activeStep == null ? "" : activeStep.dodgeRequestedByActor()
        );
        info.put(
            "native_group_action_dodge_accepted",
            activeStep == null ? "" : activeStep.dodgeAcceptedByActor()
        );
        info.put(
            "native_group_action_dodge_started",
            activeStep == null ? "" : activeStep.dodgeStartedByActor()
        );
        info.put(
            "native_group_action_dodge_finished",
            activeStep == null ? "" : activeStep.dodgeFinishedByActor()
        );
        info.put(
            "native_group_action_dodge_failed",
            activeStep == null ? "" : activeStep.dodgeFailedByActor()
        );
        info.put(
            "native_group_action_dodge_active",
            activeStep == null ? "" : activeStep.dodgeActiveByActor()
        );
        info.put(
            "native_group_action_dodge_directions",
            activeStep == null ? "" : activeStep.dodgeDirectionsByActor()
        );
        info.put(
            "native_group_action_dodge_reject_reasons",
            activeStep == null ? "" : activeStep.dodgeRejectReasonsByActor()
        );
        info.put(
            "native_group_action_dodge_interaction_ids",
            activeStep == null ? "" : activeStep.dodgeInteractionIdsByActor()
        );
        info.put(
            "native_group_action_world_verb_requested",
            activeStep == null ? "" : activeStep.worldVerbRequestedByActor()
        );
        info.put(
            "native_group_action_world_verb_accepted",
            activeStep == null ? "" : activeStep.worldVerbAcceptedByActor()
        );
        info.put(
            "native_group_action_world_verb_started",
            activeStep == null ? "" : activeStep.worldVerbStartedByActor()
        );
        info.put(
            "native_group_action_world_verb_finished",
            activeStep == null ? "" : activeStep.worldVerbFinishedByActor()
        );
        info.put(
            "native_group_action_world_verb_failed",
            activeStep == null ? "" : activeStep.worldVerbFailedByActor()
        );
        info.put(
            "native_group_action_world_verb_active",
            activeStep == null ? "" : activeStep.worldVerbActiveByActor()
        );
        info.put(
            "native_group_action_world_verb_reject_reasons",
            activeStep == null ? "" : activeStep.worldVerbRejectReasonsByActor()
        );
        info.put(
            "native_group_action_world_verb_verbs",
            activeStep == null ? "" : activeStep.worldVerbsByActor()
        );
        info.put(
            "native_group_action_world_verb_candidate_generations",
            activeStep == null ? "" : activeStep.worldVerbGenerationsByActor()
        );
        info.put(
            "native_group_action_world_verb_selected_semantics",
            activeStep == null ? "" : activeStep.worldVerbSemanticsByActor()
        );
        WorldVerbTelemetry.put(
            info,
            action,
            reservationId.toString(),
            activeStep == null || activeStep.legacyActorZeroWorldVerbExecution == null
                ? WorldVerbTelemetry.Execution.unrequested()
                : activeStep.legacyActorZeroWorldVerbExecution
        );
        boolean nativeUseActive =
            trackedAgentUseChain != null
                && trackedAgentUseStartTick >= 0L
                && trackedAgentUseFinishTick < 0L
                && trackedAgentUseChain.getServerState()
                    == InteractionState.NotFinished;
        info.put("native_use_available", nativeUseAvailable);
        info.put(
            "native_use_available_interaction_id",
            nativeUseAvailableInteractionId
        );
        info.put(
            "native_use_available_block_interaction_id",
            nativeUseAvailableBlockInteractionId
        );
        info.put(
            "native_use_available_item_id",
            nativeUseAvailableItemId
        );
        info.put(
            "native_use_available_source_container",
            nativeUseAvailableSourceContainer
        );
        info.put(
            "native_use_available_source_slot",
            nativeUseAvailableSourceSlot
        );
        info.put(
            "native_use_available_source_quantity",
            nativeUseAvailableSourceQuantity
        );
        info.put(
            "native_use_available_target",
            nativeUseAvailableTarget != null
        );
        info.put(
            "native_use_available_target_x",
            nativeUseAvailableTarget == null
                ? 0
                : nativeUseAvailableTarget.x
        );
        info.put(
            "native_use_available_target_y",
            nativeUseAvailableTarget == null
                ? 0
                : nativeUseAvailableTarget.y
        );
        info.put(
            "native_use_available_target_z",
            nativeUseAvailableTarget == null
                ? 0
                : nativeUseAvailableTarget.z
        );
        info.put(
            "native_use_available_maximum_distance",
            nativeUseAvailableMaximumDistance
        );
        info.put(
            "native_use_unavailable_reason",
            nativeUseUnavailableReason
        );
        info.put("native_use_requested", action.use());
        info.put(
            "native_use_accepted",
            activeStep != null && activeStep.useAccepted
        );
        info.put(
            "native_use_started",
            activeStep != null && activeStep.useStarted
        );
        info.put("native_use_active", nativeUseActive);
        info.put(
            "native_use_finished",
            activeStep != null && activeStep.useFinished
        );
        info.put(
            "native_use_failed",
            activeStep != null && activeStep.useFailed
        );
        info.put(
            "native_use_reject_reason",
            activeStep == null ? "" : activeStep.useRejectReason
        );
        info.put(
            "native_use_interaction_id",
            trackedAgentUseInteractionId
        );
        info.put(
            "native_use_block_interaction_id",
            trackedAgentUseBlockInteractionId
        );
        info.put(
            "native_use_scope",
            "authored_block_target_use;entity_target_use_unavailable"
        );
        info.put("native_use_entity_target_supported", false);
        info.put(
            "native_use_target_available",
            trackedAgentUseTarget != null
        );
        info.put(
            "native_use_target_x",
            trackedAgentUseTarget == null ? 0 : trackedAgentUseTarget.x
        );
        info.put(
            "native_use_target_y",
            trackedAgentUseTarget == null ? 0 : trackedAgentUseTarget.y
        );
        info.put(
            "native_use_target_z",
            trackedAgentUseTarget == null ? 0 : trackedAgentUseTarget.z
        );
        info.put(
            "native_use_maximum_distance",
            trackedAgentUseMaximumDistance
        );
        info.put(
            "native_use_chain_start_world_tick",
            trackedAgentUseStartTick
        );
        info.put(
            "native_use_chain_finish_world_tick",
            trackedAgentUseFinishTick
        );
        SyntheticWorldInteractionEvidence syntheticEvidence =
            activeStep == null
                || activeStep.syntheticWorldInteraction == null
                ? syntheticWorldInteractionEvidence(action)
                : activeStep.syntheticWorldInteraction;
        syntheticEvidence.putInto(info);
        NativeFallingBlockTrace currentFallingBlockTrace =
            fallingBlockTrace;
        NativeFallingBlockTrace effectiveFallingBlockTrace =
            currentFallingBlockTrace == null
                ? NativeFallingBlockTrace.unavailable(
                    options.fidelityFixture().equals(
                        EnvironmentOptions.FALLING_BLOCK_FIXTURE
                    )
                )
                : currentFallingBlockTrace;
        effectiveFallingBlockTrace.putInto(info);
        info.put("requested_charge_time", action.requestedChargeTime());
        info.put("native_policy_combat_protocol_version", 1);
        info.put("native_world_move_direction", action.worldMoveDirection());
        NativeGuardProgram.putInto(info);
        boolean guardAccepted = activeStep != null
            && activeStep.guardAccepted;
        boolean guardStarted = activeStep != null
            && activeStep.guardStarted;
        boolean guardFinished = activeStep != null
            && activeStep.guardFinished;
        boolean retainedGuardTerminal = NativeGuardLifecycle.terminalRetained(
            nativeGuardLifecycle.requestGeneration(),
            trackedNativeGuardInteractionGeneration,
            trackedNativeGuardFinishTick
        );
        guardFinished |= retainedGuardTerminal;
        info.put("native_guard_requested", action.guardHeld());
        info.put("native_guard_accepted", guardAccepted);
        info.put("native_guard_started", guardStarted);
        info.put("native_guard_active", nativeGuardActive);
        info.put("native_guard_finished", guardFinished);
        boolean guardLifecycleVisible = guardLifecycleVisible(
            nativeGuardActive,
            guardAccepted,
            guardStarted,
            guardFinished,
            nativeGuardLifecycle.requestGeneration(),
            trackedNativeGuardInteractionGeneration,
            trackedNativeGuardFinishTick
        );
        info.put(
            "native_guard_interaction_id",
            guardLifecycleVisible ? trackedNativeGuardInteractionId : ""
        );
        info.put(
            "native_guard_request_generation",
            action.guardHeld() || guardLifecycleVisible
                ? nativeGuardLifecycle.requestGeneration()
                : -1L
        );
        info.put(
            "native_guard_interaction_generation",
            guardLifecycleVisible
                ? trackedNativeGuardInteractionGeneration
                : -1L
        );
        info.put(
            "native_guard_chain_start_world_tick",
            guardLifecycleVisible ? trackedNativeGuardStartTick : -1L
        );
        info.put(
            "native_guard_chain_finish_world_tick",
            guardLifecycleVisible ? trackedNativeGuardFinishTick : -1L
        );
        info.put(
            "native_guard_reject_reason",
            activeStep == null ? "" : activeStep.guardRejectReason
        );
        info.put("native_dodge_requested", action.hasDodge());
        info.put("native_dodge_direction", action.dodgeDirection());
        info.put(
            "native_dodge_accepted",
            activeStep != null && activeStep.dodgeAccepted
        );
        info.put(
            "native_dodge_started",
            activeStep != null && activeStep.dodgeStarted
        );
        info.put(
            "native_dodge_finished",
            activeStep != null && activeStep.dodgeFinished
        );
        info.put(
            "native_dodge_failed",
            activeStep != null && activeStep.dodgeFailed
        );
        info.put(
            "native_dodge_reject_reason",
            activeStep == null ? "" : activeStep.dodgeRejectReason
        );
        info.put(
            "native_dodge_interaction_id",
            activeStep == null || activeStep.dodgeInteractionId.isEmpty()
                ? trackedAgentDodgeInteractionId
                : activeStep.dodgeInteractionId
        );
        info.put(
            "native_dodge_active_direction",
            trackedAgentDodgeDirection
        );
        info.put(
            "native_dodge_chain_start_world_tick",
            trackedAgentDodgeStartTick
        );
        info.put(
            "native_dodge_chain_finish_world_tick",
            trackedAgentDodgeFinishTick
        );
        info.put("native_ability_requested", action.hasAbility());
        info.put("native_ability_slot", action.abilitySlot());
        info.put(
            "native_ability_accepted",
            activeStep != null && activeStep.abilityAccepted
        );
        info.put(
            "native_ability_accepted_slot",
            activeStep == null ? -1 : activeStep.abilityAcceptedSlot
        );
        info.put(
            "native_ability_started",
            activeStep != null && activeStep.abilityStarted
        );
        info.put(
            "native_ability_finished",
            activeStep != null && activeStep.abilityFinished
        );
        info.put(
            "native_ability_failed",
            activeStep != null && activeStep.abilityFailed
        );
        info.put(
            "native_ability_reject_reason",
            activeStep == null ? "" : activeStep.abilityRejectReason
        );
        info.put(
            "native_ability_interaction_id",
            activeStep == null || activeStep.abilityInteractionId.isEmpty()
                ? trackedAgentAbilityId
                : activeStep.abilityInteractionId
        );
        info.put(
            "native_ability_interaction_type",
            activeStep == null || activeStep.abilityInteractionType.isEmpty()
                ? trackedAgentAbilityType
                : activeStep.abilityInteractionType
        );
        info.put("native_ability_active_slot", trackedAgentAbilitySlot);
        info.put(
            "native_ability_chain_start_world_tick",
            trackedAgentAbilityStartTick
        );
        info.put(
            "native_ability_chain_finish_world_tick",
            trackedAgentAbilityFinishTick
        );
        info.put("audio_schema", AudioFrame.SCHEMA);
        info.put("audio_available", true);
        info.put("audio_capture_complete", false);
        info.put("audio_captures_2d", true);
        info.put("audio_captures_3d", false);
        info.put("audio_captures_entity", true);
        info.put(
            "audio_capture_limitation",
            "headless_world_broadcasts_only_no_spatial_3d_listener"
        );
        info.put(
            "audio_event_count",
            activeStep == null ? 0 : activeStep.audio.retainedCount()
        );
        info.put(
            "audio_overwritten_count",
            activeStep == null ? 0 : activeStep.audio.overwrittenCount()
        );
        info.put(
            "audio_invalid_count",
            activeStep == null ? 0 : activeStep.audio.invalidCount()
        );
        if (
            options.fidelityFixture().equals(
                EnvironmentOptions.DIVE_MOTION_FIXTURE
            )
        ) {
            NativeDiveTrace trace = activeStep == null
                ? null
                : activeStep.diveTrace;
            if (trace == null) {
                info.put(
                    "native_dive_trace_schema",
                    NativeDiveTrace.SCHEMA
                );
                info.put(
                    "native_dive_trace_version",
                    NativeDiveTrace.VERSION
                );
                info.put("native_dive_trace_available", false);
                info.put("native_dive_trace_reason", "not_captured");
            } else {
                trace.putInto(info);
            }
        }
        info.put("native_attack_requested", action.attack());
        info.put(
            "native_attack_accepted",
            activeStep != null && activeStep.attackAccepted
        );
        info.put(
            "native_attack_executing",
            snapshot.combatTelemetry().agentAttackExecuting()
        );
        info.put("native_attack_action_count", nativeAttackActionCount());
        info.put("native_attack_action_paths", nativeAttackActionPaths());
        info.put(
            "native_attack_source",
            nativeCombatRootInteraction == null
                ? "npc_role_action_attack"
                : "reset_pinned_item_interaction"
        );
        info.put(
            "native_combat_item_id",
            options.hasNativeCombatItem()
                ? options.nativeCombatItemId()
                : ""
        );
        info.put(
            "native_combat_ability_interaction_ids",
            options.hasNativeCombatAbilities()
                ? String.join(",", options.nativeCombatAbilityInteractionIds())
                : ""
        );
        info.put(
            "native_combat_ability_interaction_types",
            options.hasNativeCombatAbilities()
                ? String.join(",", options.nativeCombatAbilityInteractionTypes())
                : ""
        );
        info.put(
            "native_actor_evidence_resource_stat_ids",
            String.join(",", nativeActorEvidenceResourceStatIds())
        );
        info.put(
            "native_combat_ability_execution_root_ids",
            nativeCombatAbilities.stream()
                .map(binding -> binding.root().getId())
                .reduce((left, right) -> left + "," + right)
                .orElse("")
        );
        info.put(
            "native_guard_interaction_id",
            options.hasNativeGuardInteraction()
                ? options.nativeGuardInteractionId()
                : ""
        );
        info.put(
            "native_guard_interaction_type",
            options.hasNativeGuardInteraction()
                ? options.nativeGuardInteractionType()
                : ""
        );
        info.put(
            "native_combat_interaction_id",
            options.hasNativeCombatInteraction()
                ? options.nativeCombatInteractionId()
                : ""
        );
        info.put(
            "native_combat_interaction_type",
            options.hasNativeCombatInteraction()
                ? options.nativeCombatInteractionType()
                : ""
        );
        SyntheticClientInfo.contribute(info, syntheticCombatClient, options);
        EngineTimingInfo.contribute(
            info,
            snapshot,
            executedTicks,
            simulatedSeconds,
            minimumDeltaSeconds,
            maximumDeltaSeconds,
            world,
            options,
            tickAnchor,
            episodeEngineTicks
        );
        if (options.nativeNavigationTrace()) {
            targetNavigationTrace.putInto(info);
        }
        info.put("agent_present", snapshot.agentPresent());
        info.put("on_ground", snapshot.onGround());
        info.put("block_below_id", snapshot.blockBelowId());
        GeometryFrame geometry = snapshot.observation().geometry();
        GeometryInfo.contribute(
            info,
            geometry,
            geometryChunkIndices,
            loadedChunkIndices
        );
        info.put("native_entity_model_evidence_version", 1);
        snapshot.agentModelEvidence().putInto(info, "agent");
        snapshot.targetModelEvidence().putInto(info, "target");
        info.put("movement_states_schema", MovementStateFrame.SCHEMA);
        info.put("movement_states_version", MovementStateFrame.VERSION);
        info.put("movement_states_count", MovementStateFrame.STATE_COUNT);
        info.put("movement_states_order", MovementStateFrame.ORDER);
        snapshot.agentMovementStates().putInto(info, "agent");
        snapshot.targetMovementStates().putInto(info, "target");
        NativeActorEvidenceFrame actorEvidence =
            snapshot.observation().nativeActorEvidence();
        info.put(
            "native_actor_evidence_schema",
            NativeActorEvidenceFrame.SCHEMA
        );
        info.put(
            "native_actor_evidence_version",
            NativeActorEvidenceFrame.VERSION
        );
        info.put(
            "native_actor_evidence_contract_sha256",
            NativeActorEvidenceFrame.CONTRACT_SHA256
        );
        info.put(
            "native_actor_evidence_available",
            actorEvidence.available()
        );
        info.put(
            "native_actor_evidence_unavailable_reason",
            actorEvidence.unavailableReason()
        );
        info.put("active_hotbar_slot", snapshot.activeHotbarSlot());
        info.put("has_native_health_stat", snapshot.hasHealthStat());
        info.put("has_native_stamina_stat", snapshot.hasStaminaStat());
        info.put("has_native_mana_stat", snapshot.hasManaStat());
        info.put("max_health", snapshot.observation().maxHealth());
        info.put(
            "inventory_encoding",
            "sorted_"
                + NATIVE_SERVER_VERSION
                + "_item_asset_index;quantities_omitted"
        );
        info.put(
            "nearby_entity_encoding",
            "0=trork,1=outlander,2=scarak,3=kweebec,4=item_drop,5..999=stable_role_hash"
        );
        info.put("native_npc_count", snapshot.nativeNpcCount());
        snapshot.combatTelemetry().putInto(info);
        if (taskId.equals("navigate")) {
            info.put("target_x", targetX);
            info.put("target_y", targetY);
            info.put("target_z", targetZ);
            info.put("target_distance", horizontalDistance(
                snapshot.observation().x(),
                snapshot.observation().z()
            ));
            info.put("target_is_task_overlay", true);
        }
        if (taskId.equals("kill_trork")) {
            info.put("target_role", snapshot.targetRoleName());
            info.put("target_present", snapshot.targetPresent());
            info.put("target_health", snapshot.targetHealth());
            info.put("target_max_health", snapshot.targetMaxHealth());
            info.put("target_distance", snapshot.targetDistance());
            info.put("target_x", snapshot.targetX());
            info.put("target_y", snapshot.targetY());
            info.put("target_z", snapshot.targetZ());
            info.put("target_is_native_entity", true);
            info.put(
                "target_forced_combat_state",
                options.hasCombatTargetRole()
                    ? "authored_state;reciprocal_locked_target"
                    : targetMemoryFixture != null
                    ? "natural"
                    : options.fidelityFixture().equals(
                        EnvironmentOptions.PATH_FOLLOWER_FIXTURE
                    )
                        ? "Chase.Attack"
                    : options.combatTargetActive()
                        ? "Chase.Attack"
                        : "disabled"
            );
            if (targetMemoryFixture != null) {
                targetMemoryFixture.putInto(info);
            }
        }
        return Map.copyOf(info);
    }

    private Map<String, Object> buildResetInfo(NativeSnapshot snapshot) {
        LinkedHashMap<String, Object> info = new LinkedHashMap<>(
            buildInfo(snapshot, 0, 0.0, 0.0, 0.0, AgentAction.noop())
        );
        NativeWorldActionCapabilities.forSession(
            options.nativeWorldVerbs()
        ).putInto(info);
        return Map.copyOf(info);
    }

    private String requestedUnsupportedActions(AgentAction action) {
        List<String> requested = new ArrayList<>(
            requestedUnsupportedWorldVerbs(
                action,
                headlessWorldVerbsEnabled()
            )
        );
        if (action.attack() && !hasNativeAttackAction()) requested.add("attack");
        if (action.guardHeld() && nativeGuardInteraction == null) {
            requested.add("guard_held");
        }
        if (
            action.hasAbility()
                && action.abilitySlot() >= nativeCombatAbilities.size()
        ) {
            requested.add("ability_slot");
        }
        if (
            action.hasDodge()
                && !NativeDodgeProgram.isAvailable(
                    action.dodgeDirection(),
                    nativeDodgeLeftInteraction != null,
                    nativeDodgeRightInteraction != null
                )
        ) {
            requested.add("dodge_direction");
        }
        return String.join(",", requested);
    }

    private String fidelityFixtureDescription() {
        if (options.world().equals("hytale")) return "native_hytale_worldgen";
        if (options.world().equals("hytale_generator")) {
            return "native_hytale_generator_worldgen";
        }
        return switch (options.fidelityFixture()) {
            case "ledge" -> "flat_floor;three_block_trench_z=-1..-5";
            case "ceiling" -> "flat_floor;stone_ceiling_y=67";
            case "half_block" -> "flat_floor;stone_half_block_z=-2";
            case "los_wall" -> "flat_floor;los_wall_z=-1";
            case "los_diagonal_graze" ->
                "flat_floor;diagonal_los_grazes_block_x=1,z=0";
            case "los_diagonal_block" ->
                "flat_floor;diagonal_los_crosses_block_x=1,z=-1";
            case EnvironmentOptions.TARGET_MEMORY_FIXTURE ->
                "flat_floor;natural_brawler;dynamic_sealed_occluder";
            case EnvironmentOptions.PATH_FOLLOWER_FIXTURE ->
                "flat_floor;chase_attack;stone_wall_z=-3,x=-2..2";
            case EnvironmentOptions.NATIVE_PROJECTILE_DUEL_FIXTURE ->
                "flat_floor;outlander_hunter_vs_kweebec;range=10";
            case EnvironmentOptions.DIVE_MOTION_FIXTURE ->
                "flat_floor;sealed_water_source_pool_xz=-7..7,y=65..74";
            case EnvironmentOptions.SYNTHETIC_WORLD_VERBS_FIXTURE ->
                "flat_floor;fixture_only_synthetic_adventure_player;"
                    + "place_x=1;break_x=-1";
            case EnvironmentOptions.SYNTHETIC_BLOCK_USE_FIXTURE ->
                "flat_floor;fixture_only_headless_npc;"
                    + "authored_use_block=Deco_Lantern;x=1";
            case EnvironmentOptions.FALLING_BLOCK_FIXTURE ->
                "flat_floor;native_falling_block_trace";
            default -> taskId.equals("native_fidelity") && !options.hasSpawn()
                ? "flat_floor;stone_wall_z=-4"
                : "flat_floor";
        };
    }

    private double horizontalDistance(double x, double z) {
        return Math.hypot(x - targetX, z - targetZ);
    }

    private NativeActorEvidenceFrame captureNativeActorEvidence(
        Store<EntityStore> store,
        GeometryFrame geometry,
        TargetState target,
        MovementStateFrame agentMovementStates,
        MovementStateFrame targetMovementStates
    ) {
        if (!options.hasNativeActorEvidenceRequest()) {
            return NativeActorEvidenceFrame.unavailable("not_negotiated");
        }

        CapturedActorEvidence agent = captureActorEvidence(
            store,
            agentRef,
            0,
            true,
            agentMovementStates,
            trackedAgentAbilitySlot,
            itemIds,
            nativeActorEvidenceResourceStatIds(),
            true
        );
        boolean targetPerceptible = target.present()
            && target.health() > 0.0
            && target.distance() <= NEARBY_ENTITY_RADIUS
            && geometry.targetLineOfSightValid()
            && geometry.targetLineOfSight();
        CapturedActorEvidence targetActor = targetPerceptible
            ? captureActorEvidence(
                store,
                targetRef,
                1,
                true,
                targetMovementStates,
                nativePolicyCombatActiveAbilitySlots[1],
                itemIds,
                nativeActorEvidenceResourceStatIds(),
                false
            )
            : new CapturedActorEvidence(absentActor(1), false, false);
        CapturedRoleOpacity roleOpacity = captureRoleOpacity(
            store,
            agentRef,
            geometry
        );

        boolean[] skillMask = new boolean[] {
            true,
            true,
            true,
            true,
            true,
            true,
            hasNativeAttackAction(),
            hasNativeAttackAction(),
            hasNativeAttackAction()
        };
        InteractionManager capabilityManager = interactionManager(agentRef, store);
        NPCEntity capabilityNpc = agentRef == null || !agentRef.isValid()
            ? null
            : store.getComponent(agentRef, NPCEntity.getComponentType());
        Role capabilityRole = capabilityNpc == null
            ? null
            : capabilityNpc.getRole();
        boolean abilityCombatAvailable = capabilityRole != null
            && capabilityRole.getCombatSupport() != null;
        List<NativeActorEvidenceFrame.Ability> abilities =
            new ArrayList<>(NativeActorEvidenceFrame.ABILITY_CAPACITY);
        for (
            int slot = 0;
            slot < NativeActorEvidenceFrame.ABILITY_CAPACITY;
            slot++
        ) {
            boolean authored = slot < nativeCombatAbilities.size();
            NativeInteractionBinding binding = authored
                ? nativeCombatAbilities.get(slot)
                : null;
            NativeInteractionCapabilityView.State capability = authored
                ? NativeInteractionCapabilityView.capture(
                    capabilityManager,
                    binding
                )
                : new NativeInteractionCapabilityView.State(true, false);
            if (!capability.available()) {
                return NativeActorEvidenceFrame.unavailable(
                    "interaction_capability_unavailable"
                );
            }
            boolean tracked = authored && slot == trackedAgentAbilitySlot;
            boolean active = tracked
                && trackedAgentAbilityStartTick >= 0L
                && trackedAgentAbilityFinishTick < 0L;
            abilities.add(new NativeActorEvidenceFrame.Ability(
                slot,
                authored ? binding.id() : "",
                authored ? binding.type().name() : "",
                authored,
                capability.legal() && abilityCombatAvailable,
                active,
                tracked ? trackedAgentAbilityStartTick : -1L,
                tracked ? trackedAgentAbilityFinishTick : -1L
            ));
        }
        int mechanicsFailureBits = 0;
        if (agent.statusOverflow() || targetActor.statusOverflow()) {
            mechanicsFailureBits |= 1;
        }
        if (agent.statusInvalid() || targetActor.statusInvalid()) {
            mechanicsFailureBits |= 1 << 2;
        }
        NativeInteractionCapabilityView.State guardCapability =
            guardCapability(capabilityManager, store);
        NativeInteractionCapabilityView.State dodgeLeftCapability =
            optionalCapability(capabilityManager, nativeDodgeLeftInteraction);
        NativeInteractionCapabilityView.State dodgeRightCapability =
            optionalCapability(capabilityManager, nativeDodgeRightInteraction);
        if (
            !guardCapability.available()
                || !dodgeLeftCapability.available()
                || !dodgeRightCapability.available()
        ) {
            return NativeActorEvidenceFrame.unavailable(
                "interaction_capability_unavailable"
            );
        }
        boolean[] dodgeActionMask = NativeDodgeProgram.actionMask(
            dodgeLeftCapability.legal(),
            dodgeRightCapability.legal()
        );
        return new NativeActorEvidenceFrame(
            true,
            "",
            world.getTick(),
            List.of(agent.actor(), targetActor.actor()),
            abilities,
            geometry.available(),
            roleOpacity.available(),
            roleOpacity.cellMask(),
            skillMask,
            true,
            guardCapability.legal(),
            dodgeActionMask,
            new boolean[NativeActorEvidenceFrame.DOOR_CANDIDATE_CAPACITY][
                NativeActorEvidenceFrame.DOOR_INTENT_COUNT
            ],
            0,
            mechanicsFailureBits,
            0
        );
    }

    private List<String> nativeActorEvidenceResourceStatIds() {
        return options.nativeActorEvidenceRequest() == null
            ? NativeActorEvidenceRequest.DEFAULT_RESOURCE_STAT_IDS
            : options.nativeActorEvidenceRequest().resourceStatIds();
    }

    private NativeInteractionCapabilityView.State guardCapability(
        InteractionManager manager,
        Store<EntityStore> store
    ) {
        if (nativeGuardInteraction == null) {
            return new NativeInteractionCapabilityView.State(true, false);
        }
        if (manager == null) {
            return NativeInteractionCapabilityView.State.unavailable();
        }
        if (nativeGuardChain != null) {
            boolean active = nativeGuardChain.getServerState()
                == InteractionState.NotFinished
                || hasActiveWielding(agentRef, store);
            return new NativeInteractionCapabilityView.State(
                true,
                active && nativeGuardLifecycle.inputHeld()
            );
        }
        if (nativeGuardLifecycle.inputHeld()) {
            return new NativeInteractionCapabilityView.State(true, false);
        }
        return NativeInteractionCapabilityView.capture(
            manager,
            nativeGuardInteraction
        );
    }

    private static NativeInteractionCapabilityView.State optionalCapability(
        InteractionManager manager,
        NativeInteractionBinding binding
    ) {
        return binding == null
            ? new NativeInteractionCapabilityView.State(true, false)
            : NativeInteractionCapabilityView.capture(manager, binding);
    }

    private NpcObservationSnapshot captureTraceObservation(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        List<ActionAttack> attackActions,
        List<Ref<EntityStore>> worldviewCandidates
    ) {
        if (ref == null || !ref.isValid()) {
            throw new IllegalStateException("Traced NPC is unavailable");
        }
        NPCEntity npc = store.getComponent(ref, NPCEntity.getComponentType());
        TransformComponent transform = store.getComponent(
            ref,
            TransformComponent.getComponentType()
        );
        if (npc == null || transform == null) {
            throw new IllegalStateException("Traced NPC lost observation components");
        }
        Role role = npc.getRole();
        MarkedEntitySupport markedEntities = role == null
            ? null
            : role.getMarkedEntitySupport();
        Ref<EntityStore> targetRef = markedEntities == null
            ? null
            : markedEntities.getMarkedEntityRef(
                MarkedEntitySupport.DEFAULT_TARGET_SLOT
            );
        NPCEntity targetNpc = targetRef == null || !targetRef.isValid()
            ? null
            : store.getComponent(targetRef, NPCEntity.getComponentType());
        TransformComponent targetTransform = targetRef == null
            || !targetRef.isValid()
            ? null
            : store.getComponent(targetRef, TransformComponent.getComponentType());
        boolean targetPresent = targetNpc != null && targetTransform != null;
        if (!targetPresent) targetRef = null;

        Vector3d position = transform.getPosition();
        int originX = (int) Math.floor(position.x);
        int originY = (int) Math.floor(position.y);
        int originZ = (int) Math.floor(position.z);
        GeometryFrame geometry = RegionGeometryCapture.captureGeometry(nativeCellSemanticsCache, world, 
            store,
            ref,
            targetRef,
            role,
            originX,
            originY,
            originZ,
            position,
            targetPresent
        );
        CapturedRoleOpacity roleOpacity = captureRoleOpacity(store, ref, geometry);
        NativePerceptionChannels localPerception = buildPerceptionChannels(
            geometryCubePositions(geometry)
        );
        CombatSupport combat = role == null ? null : role.getCombatSupport();
        TraceAttackCandidates candidates = traceAttackCandidates(attackActions);
        List<ActionAttack> targetActions = targetPresent
            ? discoverAttackActions(targetNpc.getRole())
            : List.of();
        TraceAttackCandidates targetCandidates = traceAttackCandidates(
            targetActions
        );
        CapturedActorEvidence actor = captureActorEvidence(
            store,
            ref,
            0,
            true,
            captureMovementStates(store, ref),
            candidates.activeSlot(),
            itemIds,
            nativeActorEvidenceResourceStatIds(),
            true
        );
        CapturedActorEvidence target = targetPresent
            ? captureActorEvidence(
                store,
                targetRef,
                1,
                true,
                captureMovementStates(store, targetRef),
                targetCandidates.activeSlot(),
                itemIds,
                nativeActorEvidenceResourceStatIds(),
                false
            )
            : new CapturedActorEvidence(absentActor(1), false, false);

        double targetDistance = targetPresent
            ? position.distance(targetTransform.getPosition())
            : 0.0;
        boolean targetPerceptible = targetPresent
            && entityHealth(store, targetRef, targetNpc) > 0.0
            && Math.hypot(
                targetTransform.getPosition().x - position.x,
                targetTransform.getPosition().z - position.z
            ) <= NEARBY_ENTITY_RADIUS
            && geometry.targetLineOfSightValid()
            && geometry.targetLineOfSight();
        PerceptibleEntities perception = capturePerceptibleEntities(
            store,
            ref,
            role,
            position,
            worldviewCandidates
        );
        boolean perceptionAvailable = perception.available()
            && (!targetPresent || geometry.targetLineOfSightValid());
        if (!perceptionAvailable) targetPerceptible = false;
        NativeActorEvidenceFrame.Actor targetEvidence = actorPerceptibility(
            target.actor(),
            targetPerceptible
        );
        NativeActorEvidenceFrame actionCapabilityEvidence =
            traceActionCapabilityEvidence(
                ref,
                targetRef,
                store,
                geometry,
                position
            );
        int statusFailureBits = 0;
        if (actor.statusOverflow()) {
            statusFailureBits |= NpcObservationSnapshot.STATUS_ACTOR_OVERFLOW;
        }
        if (target.statusOverflow()) {
            statusFailureBits |= NpcObservationSnapshot.STATUS_TARGET_OVERFLOW;
        }
        if (actor.statusInvalid()) {
            statusFailureBits |= NpcObservationSnapshot.STATUS_ACTOR_INVALID;
        }
        if (target.statusInvalid()) {
            statusFailureBits |= NpcObservationSnapshot.STATUS_TARGET_INVALID;
        }
        return new NpcObservationSnapshot(
            store.getExternalData().getWorld().getTick(),
            actor.actor(),
            targetEvidence,
            actionCapabilityEvidence,
            geometry,
            captureNativeInventory(npc.getInventory()),
            roleOpacity.available(),
            roleOpacity.cellMask(),
            localPerception,
            true,
            combat != null && combat.isExecutingAttack(),
            NativeNpcCombatIntrospection.attackPauseSeconds(combat),
            candidates.emitted(),
            candidates.total(),
            candidates.overflow(),
            targetCandidates.emitted(),
            targetCandidates.total(),
            targetCandidates.overflow(),
            perceptionAvailable,
            targetPresent,
            targetPerceptible,
            targetDistance,
            perceptionAvailable ? perception.emitted() : List.of(),
            perceptionAvailable ? perception.total() : 0,
            perceptionAvailable && perception.overflow(),
            perceptionAvailable ? perception.emittedEntityIndices() : List.of(),
            perceptionAvailable ? perception.entityTotal() : 0,
            perceptionAvailable && perception.entityOverflow(),
            statusFailureBits
        );
    }

    private NativeActorEvidenceFrame traceActionCapabilityEvidence(
        Ref<EntityStore> actor,
        Ref<EntityStore> traceTarget,
        Store<EntityStore> store,
        GeometryFrame geometry,
        Vector3d actorPosition
    ) {
        if (!sameEntity(actor, agentRef)) {
            return NativeActorEvidenceFrame.unavailable(
                "trace_actor_is_not_controlled_actor"
            );
        }
        boolean controlledTargetPresent = targetRef != null && targetRef.isValid();
        boolean traceTargetPresent = traceTarget != null && traceTarget.isValid();
        if (
            controlledTargetPresent != traceTargetPresent
                || (
                    controlledTargetPresent
                        && !sameEntity(traceTarget, targetRef)
                )
        ) {
            return NativeActorEvidenceFrame.unavailable(
                "trace_target_is_not_controlled_target"
            );
        }
        TargetState target = captureTargetState(store, actorPosition);
        return captureNativeActorEvidence(
            store,
            geometry,
            target,
            captureMovementStates(store, actor),
            target.present()
                ? captureMovementStates(store, targetRef)
                : MovementStateFrame.unavailable()
        );
    }

    private static int[] geometryCubePositions(GeometryFrame geometry) {
        return geometryCubePositions(
            geometry.originX(), geometry.originY(), geometry.originZ()
        );
    }

    private void primeNpcTracePerception(TransformComponent transform) {
        if (transform == null) return;
        Vector3d position = transform.getPosition();
        buildPerceptionChannels(geometryCubePositions(
            (int) Math.floor(position.x),
            (int) Math.floor(position.y),
            (int) Math.floor(position.z)
        ));
    }

    private static int[] geometryCubePositions(int originX, int originY, int originZ) {
        int[] result = new int[GeometryContract.CELL_COUNT * 3];
        for (int dx = -GeometryContract.RADIUS; dx <= GeometryContract.RADIUS; dx++) {
            for (int dy = -GeometryContract.RADIUS; dy <= GeometryContract.RADIUS; dy++) {
                for (int dz = -GeometryContract.RADIUS; dz <= GeometryContract.RADIUS; dz++) {
                    int index = GeometryContract.cellIndex(dx, dy, dz) * 3;
                    result[index] = originX + dx;
                    result[index + 1] = originY + dy;
                    result[index + 2] = originZ + dz;
                }
            }
        }
        return result;
    }

    private static TraceAttackCandidates traceAttackCandidates(
        List<ActionAttack> values
    ) {
        List<ActionAttack> source = values == null ? List.of() : values;
        int emittedCount = Math.min(
            source.size(),
            NpcObservationSnapshot.ATTACK_CANDIDATE_CAPACITY
        );
        List<NpcAttackActionSnapshot> emitted = new ArrayList<>(emittedCount);
        int activeSlot = -1;
        for (int index = 0; index < emittedCount; index++) {
            NpcAttackActionSnapshot value = NativeNpcCombatIntrospection.attack(
                source.get(index)
            );
            emitted.add(value);
            if (activeSlot < 0 && value.active()) activeSlot = index;
        }
        return new TraceAttackCandidates(
            List.copyOf(emitted),
            source.size(),
            source.size() > emittedCount,
            activeSlot
        );
    }

    private static NativeActorEvidenceFrame.Actor actorPerceptibility(
        NativeActorEvidenceFrame.Actor actor,
        boolean perceptible
    ) {
        if (!actor.present() || actor.perceptible() == perceptible) return actor;
        return new NativeActorEvidenceFrame.Actor(
            actor.entityId(),
            true,
            perceptible,
            actor.roleId(),
            actor.itemId(),
            actor.itemRuntimeIndex(),
            actor.activeAbilitySlot(),
            actor.position(),
            actor.velocity(),
            actor.motionForce(),
            actor.yawDegrees(),
            actor.pitchDegrees(),
            actor.health(),
            actor.maxHealth(),
            actor.resourceValues(),
            actor.resourceMaximums(),
            actor.resourceAvailable(),
            actor.defenseValues(),
            actor.defenseAvailable(),
            actor.statuses(),
            actor.actorWorldValues(),
            actor.actorWorldAvailable(),
            actor.movementStates()
        );
    }

    private PerceptibleEntities capturePerceptibleEntities(
        Store<EntityStore> store,
        Ref<EntityStore> self,
        Role role,
        Vector3d position,
        List<Ref<EntityStore>> worldviewCandidates
    ) {
        if (role == null || role.getPositionCache() == null) {
            return PerceptibleEntities.unavailable();
        }
        List<UUID> visibleNpcs = new ArrayList<>();
        List<Integer> visibleEntities = new ArrayList<>();
        List<Ref<EntityStore>> candidates = TRACE_PERCEPTION_WORLDVIEW_REUSE
            ? worldviewCandidates
            : tracePerceptionCandidates(store);
        for (Ref<EntityStore> candidate : candidates) {
            if (!candidate.isValid() || sameEntity(self, candidate)) continue;
            TransformComponent candidateTransform = store.getComponent(
                candidate,
                TransformComponent.getComponentType()
            );
            NPCEntity candidateNpc = store.getComponent(
                candidate,
                NPCEntity.getComponentType()
            );
            if (!TRACE_PERCEPTION_PROJECTILES && candidateNpc == null) continue;
            if (candidateTransform == null || (
                candidateNpc != null
                    && entityHealth(store, candidate, candidateNpc) <= 0.0
            )) {
                continue;
            }
            Vector3d candidatePosition = candidateTransform.getPosition();
            double dx = candidatePosition.x - position.x;
            double dz = candidatePosition.z - position.z;
            if (dx * dx + dz * dz
                > NEARBY_ENTITY_RADIUS * NEARBY_ENTITY_RADIUS) {
                continue;
            }
            try {
                if (role.getPositionCache().hasLineOfSight(
                    self,
                    candidate,
                    store
                )) {
                    visibleEntities.add(candidate.getIndex());
                    if (candidateNpc != null) {
                        UUIDComponent identity = store.getComponent(
                            candidate,
                            UUIDComponent.getComponentType()
                        );
                        if (identity == null) {
                            return PerceptibleEntities.unavailable();
                        } else {
                            visibleNpcs.add(identity.getUuid());
                        }
                    }
                }
            } catch (RuntimeException unavailable) {
                return PerceptibleEntities.unavailable();
            }
        }
        visibleNpcs.sort((left, right) -> {
            int most = Long.compareUnsigned(
                left.getMostSignificantBits(),
                right.getMostSignificantBits()
            );
            return most != 0
                ? most
                : Long.compareUnsigned(
                    left.getLeastSignificantBits(),
                    right.getLeastSignificantBits()
                );
        });
        visibleEntities.sort(Integer::compare);
        int npcTotal = visibleNpcs.size();
        int emittedNpcCount = Math.min(
            npcTotal,
            NpcObservationSnapshot.PERCEPTIBLE_NPC_CAPACITY
        );
        int entityTotal = visibleEntities.size();
        int emittedEntityCount = Math.min(
            entityTotal,
            NpcObservationSnapshot.PERCEPTIBLE_ENTITY_CAPACITY
        );
        return new PerceptibleEntities(
            true,
            List.copyOf(visibleNpcs.subList(0, emittedNpcCount)),
            npcTotal,
            npcTotal > emittedNpcCount,
            List.copyOf(visibleEntities.subList(0, emittedEntityCount)),
            entityTotal,
            entityTotal > emittedEntityCount
        );
    }

    private List<Ref<EntityStore>> tracePerceptionCandidates(
        Store<EntityStore> store
    ) {
        long tick = store.getExternalData().getWorld().getTick();
        if (
            TRACE_PERCEPTION_CANDIDATE_CACHE
                && tracePerceptionCandidateTick == tick
        ) {
            return tracePerceptionCandidates;
        }
        List<Ref<EntityStore>> candidates = new ArrayList<>();
        Query<EntityStore> query = TRACE_PERCEPTION_PROJECTILES
            ? TRACE_PERCEPTION_QUERY
            : TRACE_NPC_PERCEPTION_QUERY;
        store.forEachChunk(query, (chunk, commandBuffer) -> {
            for (int index = 0; index < chunk.size(); index++) {
                Ref<EntityStore> candidate = chunk.getReferenceTo(index);
                if (candidate != null) candidates.add(candidate);
            }
        });
        List<Ref<EntityStore>> result = List.copyOf(candidates);
        if (TRACE_PERCEPTION_CANDIDATE_CACHE) {
            tracePerceptionCandidates = result;
            tracePerceptionCandidateTick = tick;
        }
        return result;
    }

    private record TraceAttackCandidates(
        List<NpcAttackActionSnapshot> emitted,
        int total,
        boolean overflow,
        int activeSlot
    ) {}

    private record PerceptibleEntities(
        boolean available,
        List<UUID> emitted,
        int total,
        boolean overflow,
        List<Integer> emittedEntityIndices,
        int entityTotal,
        boolean entityOverflow
    ) {
        private static PerceptibleEntities unavailable() {
            return new PerceptibleEntities(
                false, List.of(), 0, false, List.of(), 0, false
            );
        }
    }

    private void runOnWorld(CheckedRunnable runnable, Duration timeout) {
        World currentWorld = world;
        if (currentWorld == null) throw new IllegalStateException("Native world is unavailable");
        CompletableFuture<Void> future = new CompletableFuture<>();
        currentWorld.execute(() -> {
            try {
                runnable.run();
                future.complete(null);
            } catch (Throwable throwable) {
                future.completeExceptionally(throwable);
            }
        });
        try {
            future.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted waiting for native world thread", exception);
        } catch (ExecutionException exception) {
            Throwable cause = exception.getCause();
            if (cause instanceof RuntimeException runtime) throw runtime;
            throw new IllegalStateException("Native world-thread task failed", cause);
        } catch (TimeoutException exception) {
            throw new IllegalStateException("Timed out waiting for native world thread", exception);
        }
    }

    private void validateNativeGroupAction(NativeGroupAction actions) {
        for (var actor : actions.actors()) {
            int actorId = actor.entityId();
            Ref<EntityStore> reference = policyActorRef(actorId);
            if (reference == null || !reference.isValid()) {
                throw new IllegalStateException(
                    "native group actor is unavailable after reset: " + actorId
                );
            }
            if (actorId == 0) continue;
            if (!taskId.equals("kill_trork")) {
                throw new IllegalStateException(
                    "nonzero native group actors require the kill_trork arena"
                );
            }
            if (options.combatTargetActive()) {
                throw new IllegalStateException(
                    "nonzero native group actors require "
                        + "combat_target_active=false"
                );
            }
            String rejectReason =
                NativeGroupActionContract.nonzeroActorRejectReason(
                    actor.action()
                );
            if (!rejectReason.isEmpty()) {
                throw new IllegalArgumentException(
                    "native group actor " + actorId
                        + " requested an entity-zero-only verb: "
                        + rejectReason
                );
            }
        }
    }

    private void pinNativeDuelTargets(Store<EntityStore> store) {
        if (!nativeBehaviorMatchup()
            || !taskId.equals("kill_trork")
            || agentRef == null
            || !agentRef.isValid()
            || targetRef == null
            || !targetRef.isValid()) return;
        NPCEntity agent = store.getComponent(agentRef, NPCEntity.getComponentType());
        NPCEntity target = store.getComponent(targetRef, NPCEntity.getComponentType());
        Role agentRole = agent == null ? null : agent.getRole();
        Role targetRole = target == null ? null : target.getRole();
        requireNativeDuelHostility(store, agentRole, targetRole);
        if (agentRole != null) agentRole.setMarkedTarget("LockedTarget", targetRef);
        if (nativeProjectileDuelFixture()) {
            if (targetRole != null) {
                clearMarkedTargets(targetRole);
            }
            StateSupport agentState = agentRole == null
                ? null
                : agentRole.getStateSupport();
            StateSupport targetState = targetRole == null
                ? null
                : targetRole.getStateSupport();
            if (agentState != null
                && !agentState.getStateName().equals("Shoot.Ranged")) {
                agentState.setState(agentRef, "Shoot", "Ranged", store);
            }
            if (targetState != null
                && !targetState.getStateName().equals("Idle.Default")) {
                targetState.setState(targetRef, "Idle", null, store);
            }
            nativeDuelAutonomous = true;
            return;
        }
        if (targetRole != null) {
            targetRole.setMarkedTarget("LockedTarget", agentRef);
        }
        if (options.hasCombatTargetRole()) {
            nativeDuelAutonomous = true;
            return;
        }
        if (!nativeDuelAutonomous && targetRole != null) {
            StateSupport state = targetRole.getStateSupport();
            if (state != null) {
                state.setState(targetRef, "Chase", "Attack", store);
            }
            nativeDuelAutonomous = true;
        }
    }

    private void requireNativeDuelHostility(
        Store<EntityStore> store,
        Role agentRole,
        Role targetRole
    ) {
        if (agentRole == null || targetRole == null) {
            throw new IllegalStateException(
                "Native behavior matchup requires two live NPC roles"
            );
        }
        agentRole.getWorldSupport().requireAttitudeCache();
        targetRole.getWorldSupport().requireAttitudeCache();
        Attitude actorToTarget = agentRole.getWorldSupport().getAttitude(
            agentRef,
            targetRef,
            store
        );
        Attitude targetToActor = targetRole.getWorldSupport().getAttitude(
            targetRef,
            agentRef,
            store
        );
        nativeActorTargetAttitude = attitudeName(actorToTarget);
        nativeTargetActorAttitude = attitudeName(targetToActor);
        nativeMutualHostility = actorToTarget == Attitude.HOSTILE
            && targetToActor == Attitude.HOSTILE;
        if (options.hasCombatTargetRole() && !nativeMutualHostility) {
            throw new IllegalArgumentException(
                "Native behavior matchup is not mutually hostile: "
                    + options.npcRole() + "->" + combatTargetRole() + "="
                    + nativeActorTargetAttitude + ", " + combatTargetRole()
                    + "->" + options.npcRole() + "="
                    + nativeTargetActorAttitude
            );
        }
    }

    private static String attitudeName(Attitude attitude) {
        return attitude == null ? "UNAVAILABLE" : attitude.name();
    }

    private boolean nativeDuelFixture() {
        return options.fidelityFixture().equals(
            EnvironmentOptions.NATIVE_DUEL_FIXTURE
        ) || nativeProjectileDuelFixture();
    }

    private boolean nativeBehaviorMatchup() {
        return nativeDuelFixture() || options.hasCombatTargetRole();
    }

    private String combatTargetRole() {
        if (nativeProjectileDuelFixture()) return NATIVE_PROJECTILE_TARGET_ROLE;
        return options.hasCombatTargetRole()
            ? options.combatTargetRole()
            : KILL_TRORK_TARGET_ROLE;
    }

    private boolean nativeProjectileDuelFixture() {
        return options.fidelityFixture().equals(
            EnvironmentOptions.NATIVE_PROJECTILE_DUEL_FIXTURE
        );
    }

    private String nativePolicyCombatBoundByActor() {
        boolean[] values = new boolean[nativePolicyCombatHandles.length];
        values[0] = nativeGuardInteraction != null
            || !nativeCombatAbilities.isEmpty()
            || nativeDodgeLeftInteraction != null
            || nativeDodgeRightInteraction != null;
        for (int actorId = 1; actorId < values.length; actorId++) {
            values[actorId] = nativePolicyCombatHandles[actorId] != null;
        }
        return joinActorBooleans(values);
    }

    private String nativePolicyCombatAbilityMasksByActor() {
        int[] values = new int[nativePolicyCombatHandles.length];
        values[0] = nativeCombatAbilities.isEmpty()
            ? 0
            : (1 << nativeCombatAbilities.size()) - 1;
        for (int actorId = 1; actorId < values.length; actorId++) {
            NativePolicyCombatBindingSpec spec =
                nativePolicyCombatSpecs[actorId];
            values[actorId] = spec == null ? 0 : spec.abilityMaskBits();
        }
        return joinActorInts(values);
    }

    private String nativePolicyCombatGuardByActor() {
        boolean[] values = new boolean[nativePolicyCombatHandles.length];
        values[0] = nativeGuardInteraction != null;
        for (int actorId = 1; actorId < values.length; actorId++) {
            NativePolicyCombatBindingSpec spec =
                nativePolicyCombatSpecs[actorId];
            values[actorId] = spec != null && spec.hasGuard();
        }
        return joinActorBooleans(values);
    }

    private String nativePolicyCombatDodgeByActor() {
        boolean[] values = new boolean[nativePolicyCombatHandles.length];
        values[0] = nativeDodgeLeftInteraction != null
            && nativeDodgeRightInteraction != null;
        for (int actorId = 1; actorId < values.length; actorId++) {
            values[actorId] = nativePolicyCombatHandles[actorId] != null;
        }
        return joinActorBooleans(values);
    }

    private String nativePolicyCombatItemIdsByActor() {
        String[] values = new String[nativePolicyCombatHandles.length];
        values[0] = options.hasNativeCombatItem()
            ? options.nativeCombatItemId()
            : "";
        for (int actorId = 1; actorId < values.length; actorId++) {
            NativePolicyCombatBindingSpec spec =
                nativePolicyCombatSpecs[actorId];
            values[actorId] = spec == null ? "" : spec.itemId();
        }
        return String.join(",", values);
    }

    private static String joinActorBooleans(boolean[] values) {
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < values.length; index++) {
            if (index > 0) result.append(',');
            result.append(values[index] ? '1' : '0');
        }
        return result.toString();
    }

    private static String joinActorInts(int[] values) {
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < values.length; index++) {
            if (index > 0) result.append(',');
            result.append(values[index]);
        }
        return result.toString();
    }

    private static NativePolicyActorState[] createPolicyActorStates() {
        NativePolicyActorState[] result = new NativePolicyActorState[
            NativeGroupActionContract.actorCapacity()
        ];
        for (int actorId = 0; actorId < result.length; actorId++) {
            result[actorId] = new NativePolicyActorState(actorId);
        }
        return result;
    }

    private static int[] createInactivePolicyCombatSlots() {
        int[] slots = new int[NativeActorEvidenceFrame.ENTITY_COUNT];
        for (int actorId = 0; actorId < slots.length; actorId++) {
            slots[actorId] = -1;
        }
        return slots;
    }

    private NativePolicyActorState policyActorState(int actorId) {
        if (actorId < 0 || actorId >= policyActorStates.length) {
            throw new IllegalArgumentException(
                "native policy actor is outside negotiated capacity: "
                    + actorId
            );
        }
        return policyActorStates[actorId];
    }

    private Ref<EntityStore> policyActorRef(int actorId) {
        return switch (actorId) {
            case 0 -> agentRef;
            case 1 -> targetRef;
            default -> null;
        };
    }

    private boolean usesGeneratedWorld() {
        return options.world().equals("hytale")
            || options.world().equals("hytale_generator");
    }


}
