package com.hytalerlbridge.environment;

import com.hytalerlbridge.BridgeConfig;
import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.action.group.NativeGroupAction;
import com.hytalerlbridge.entity.PrivilegedEntityQuery;
import com.hytalerlbridge.entity.PrivilegedEntitySnapshot;
import com.hytalerlbridge.entity.PrivilegedNpcSnapshot;
import com.hytalerlbridge.entity.PrivilegedNpcUuidQuery;
import com.hytalerlbridge.imitation.NpcTraceBatch;
import com.hytalerlbridge.observation.Observation;
import com.hytalerlbridge.task.RLTask;
import com.hytalerlbridge.task.TaskRegistry;
import com.hytalerlbridge.worldgen.NativeRegionManifest;
import com.hytalerlbridge.worldgen.NativeRegionBlockSemanticSection;
import com.hytalerlbridge.worldgen.NativeRegionFluidSemanticSection;
import com.hytalerlbridge.worldgen.NativeRegionLightSection;
import com.hytalerlbridge.worldgen.NativeRegionSection;
import com.hytalerlbridge.worldgen.NativePerceptionChannels;
import com.hytalerlbridge.worldgen.NativeLineOfSightEvidence;
import com.hytalerlbridge.worldgen.NativeMutableBlockEvidence;
import com.hytalerlbridge.worldgen.NativeMutableBlockCells;
import com.hytalerlbridge.worldgen.NativeDoorTransitionEvidence;
import com.hytalerlbridge.worldgen.NativeDropProgramEvidence;
import com.hytalerlbridge.worldgen.NativeCraftingCatalogEvidence;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence;
import com.hytalerlbridge.worldgen.NativeTraversalProbe;
import com.hytalerlbridge.worldgen.NativeTraversalEdgeProbe;
import com.hytalerlbridge.worldgen.NativeNavigationPathProbe;
import com.hytalerlbridge.worldgen.NativeNavigationSuccessorProbe;
import com.hytalerlbridge.worldgen.NativeExplosionCandidateProbe;
import com.hytalerlbridge.worldgen.NativeExplosionMutationProbe;
import com.hytalerlbridge.worldgen.NativeExplosionDynamicsProbe;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerQuery;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerSnapshot;
import com.hytalerlbridge.worldgen.policyactions.capture.PolicyWorldActionCaptureRequest;
import com.hytalerlbridge.worldgen.policyactions.capture.model.NativePolicyWorldActionCapture;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.msgpack.value.Value;

/** Manages isolated RL environments. Each reset creates a fresh task instance. */
public final class EnvironmentManager {

    private static final System.Logger LOGGER =
        System.getLogger(EnvironmentManager.class.getName());

    private final TaskRegistry taskRegistry;
    private final BridgeConfig config;
    private final NativeBackendProvider nativeBackendProvider;
    private final ConcurrentHashMap<String, EnvironmentSession> environments =
        new ConcurrentHashMap<>();

    public EnvironmentManager(TaskRegistry taskRegistry, BridgeConfig config) {
        this(taskRegistry, config, null);
    }

    public EnvironmentManager(
        TaskRegistry taskRegistry,
        BridgeConfig config,
        NativeBackendProvider nativeBackendProvider
    ) {
        this.taskRegistry = taskRegistry;
        this.config = config;
        this.nativeBackendProvider = nativeBackendProvider;
    }

    public String create(
        String requestedTaskId,
        long seed,
        int curriculumPhase,
        int ticksPerStep,
        int maxEpisodeSteps
    ) {
        return create(
            requestedTaskId,
            seed,
            curriculumPhase,
            ticksPerStep,
            maxEpisodeSteps,
            EnvironmentOptions.simulator()
        );
    }

    public String create(
        String requestedTaskId,
        long seed,
        int curriculumPhase,
        int ticksPerStep,
        int maxEpisodeSteps,
        EnvironmentOptions options
    ) {
        String taskId = requestedTaskId == null || requestedTaskId.isBlank()
            ? config.getDefaultTask()
            : requestedTaskId;
        String environmentId = UUID.randomUUID().toString();
        int validatedTicks = validateTicksPerStep(ticksPerStep);
        int validatedMaxSteps = validateMaxEpisodeSteps(maxEpisodeSteps);
        EnvironmentOptions actualOptions = options == null
            ? EnvironmentOptions.simulator()
            : options;

        EnvironmentSession session;
        if (actualOptions.isNative()) {
            if (nativeBackendProvider == null) {
                throw new IllegalStateException(
                    "Native backend is available only inside a Hytale 0.5.7 server"
                );
            }
            session = nativeBackendProvider.create(
                taskId,
                seed,
                curriculumPhase,
                validatedTicks,
                validatedMaxSteps,
                actualOptions
            );
        } else if (actualOptions.backend().equals("simulator")) {
            RLTask task = taskRegistry.create(taskId);
            if (task == null) {
                throw new IllegalArgumentException("Task not found: " + taskId);
            }
            task.configure(actualOptions);
            session = new EnvironmentInstance(
                task,
                seed,
                curriculumPhase,
                validatedTicks,
                validatedMaxSteps
            );
            session.reset();
        } else {
            throw new IllegalArgumentException(
                "Unknown backend: " + actualOptions.backend()
                    + " (expected simulator, native, or headless)"
            );
        }

        environments.put(environmentId, session);
        LOGGER.log(System.Logger.Level.DEBUG,
            "Created " + actualOptions.backend() + " environment " + environmentId
                + " with task " + taskId);
        return environmentId;
    }

    public StepResult observe(String environmentId) {
        EnvironmentSession instance = environments.get(environmentId);
        return instance == null
            ? StepResult.initial(Observation.empty())
            : instance.observe();
    }

    public StepResult step(String environmentId, Map<Value, Value> actionMap) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException("No environment active - call reset first");
        }
        return instance.step(AgentAction.fromMap(actionMap));
    }

    public StepResult step(String environmentId, NativeGroupAction actions) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        return instance.step(actions);
    }

    public NativeRegionManifest regionManifest(String environmentId) {
        return regionManifest(environmentId, null, null);
    }

    public NativeRegionManifest regionManifest(
        String environmentId,
        Integer coreMinChunkX,
        Integer coreMinChunkZ
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeRegionSource source)) {
            throw new IllegalStateException(
                "Region capture requires the native Hytale backend"
            );
        }
        if ((coreMinChunkX == null) != (coreMinChunkZ == null)) {
            throw new IllegalArgumentException(
                "Region core chunk coordinates must be supplied together"
            );
        }
        return coreMinChunkX == null
            ? source.regionManifest()
            : source.regionManifest(coreMinChunkX, coreMinChunkZ);
    }

    public NativeRegionSection captureRegionSection(
        String environmentId,
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeRegionSource source)) {
            throw new IllegalStateException(
                "Region capture requires the native Hytale backend"
            );
        }
        return source.captureRegionSection(chunkX, chunkZ, sectionY);
    }

    public NativeRegionLightSection captureRegionLightSection(
        String environmentId,
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeRegionSource source)) {
            throw new IllegalStateException(
                "Region light capture requires the native Hytale backend"
            );
        }
        return source.captureRegionLightSection(chunkX, chunkZ, sectionY);
    }

    public NativeRegionBlockSemanticSection
        captureRegionBlockSemanticSection(
            String environmentId,
            int chunkX,
            int chunkZ,
            int sectionY
        ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeRegionSource source)) {
            throw new IllegalStateException(
                "Region block semantics require the native Hytale backend"
            );
        }
        return source.captureRegionBlockSemanticSection(
            chunkX,
            chunkZ,
            sectionY
        );
    }

    public NativeRegionFluidSemanticSection
        captureRegionFluidSemanticSection(
            String environmentId,
            int chunkX,
            int chunkZ,
            int sectionY
        ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeRegionSource source)) {
            throw new IllegalStateException(
                "Region fluid semantics require the native Hytale backend"
            );
        }
        return source.captureRegionFluidSemanticSection(
            chunkX,
            chunkZ,
            sectionY
        );
    }

    public NativePerceptionChannels capturePerceptionChannels(
        String environmentId,
        int[] positions
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativePerceptionChannelSource source)) {
            throw new IllegalStateException(
                "Perception channel capture requires the native Hytale backend"
            );
        }
        return source.capturePerceptionChannels(positions);
    }

    public NativeLineOfSightEvidence captureLineOfSightEvidence(
        String environmentId
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeLineOfSightSource source)) {
            throw new IllegalStateException(
                "LOS evidence capture requires the native Hytale backend"
            );
        }
        return source.captureLineOfSightEvidence();
    }

    public NativeMutableBlockEvidence captureMutableBlockEvidence(
        String environmentId
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeMutableBlockSource source)) {
            throw new IllegalStateException(
                "Mutable block evidence requires the native Hytale backend"
            );
        }
        return source.captureMutableBlockEvidence();
    }

    public NativeMutableBlockCells captureMutableBlockCells(
        String environmentId,
        int[] positions
    ) {
        return captureMutableBlockCells(environmentId, positions, null);
    }

    public NativeMutableBlockCells captureMutableBlockCells(
        String environmentId,
        int[] positions,
        String expectedWorldEpoch
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeMutableBlockSource source)) {
            throw new IllegalStateException(
                "Mutable block cell capture requires the native Hytale backend"
            );
        }
        return source.captureMutableBlockCells(
            positions,
            expectedWorldEpoch
        );
    }

    public NativeDropProgramEvidence captureDropProgramEvidence(
        String environmentId,
        String blockAssetId,
        String route,
        int sampleCount
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeDropProgramSource source)) {
            throw new IllegalStateException(
                "Drop-program evidence requires the native Hytale backend"
            );
        }
        return source.captureDropProgramEvidence(
            blockAssetId,
            route,
            sampleCount
        );
    }

    public NativeCraftingCatalogEvidence captureCraftingCatalogEvidence(
        String environmentId
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeCraftingCatalogSource source)) {
            throw new IllegalStateException(
                "Crafting catalog evidence requires the native Hytale backend"
            );
        }
        return source.captureCraftingCatalogEvidence();
    }

    public NativePolicyWorldActionCapture capturePolicyWorldActions(
        String environmentId,
        PolicyWorldActionCaptureRequest request
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativePolicyWorldActionCaptureSource source)) {
            throw new IllegalStateException(
                "Policy World-action capture requires the native Hytale backend"
            );
        }
        return source.capturePolicyWorldActions(request);
    }

    public NativeItemInteractionEvidence captureItemInteractionEvidence(
        String environmentId,
        int equippedSlot
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeItemInteractionSource source)) {
            throw new IllegalStateException(
                "Item interaction evidence requires the native Hytale backend"
            );
        }
        return source.captureItemInteractionEvidence(equippedSlot);
    }

    public NativeDoorTransitionEvidence captureDoorTransitionEvidence(
        String environmentId
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeDoorTransitionSource source)) {
            throw new IllegalStateException(
                "Door evidence capture requires the native Hytale backend"
            );
        }
        return source.captureDoorTransitionEvidence();
    }

    public NativeTraversalProbe captureTraversalProbe(
        String environmentId,
        double[] positions,
        double[] upwardLimits
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeTraversalProbeSource source)) {
            throw new IllegalStateException(
                "Traversal probe requires the native Hytale backend"
            );
        }
        return source.captureTraversalProbe(positions, upwardLimits);
    }

    public NativeTraversalEdgeProbe captureTraversalEdgeProbe(
        String environmentId,
        double[] startPositions,
        double[] targetPositions,
        double[] horizontalArrivalTolerances,
        double[] verticalArrivalTolerances
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeTraversalProbeSource source)) {
            throw new IllegalStateException(
                "Traversal edge probe requires the native Hytale backend"
            );
        }
        return source.captureTraversalEdgeProbe(
            startPositions,
            targetPositions,
            horizontalArrivalTolerances,
            verticalArrivalTolerances
        );
    }

    public NativeNavigationPathProbe captureNavigationPathProbe(
        String environmentId,
        double[] startPositions,
        double[] targetPositions,
        int maximumPathLength,
        int openNodesLimit,
        int totalNodesLimit,
        int nodesPerIteration
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeTraversalProbeSource source)) {
            throw new IllegalStateException(
                "Navigation path probe requires the native Hytale backend"
            );
        }
        return source.captureNavigationPathProbe(
            startPositions,
            targetPositions,
            maximumPathLength,
            openNodesLimit,
            totalNodesLimit,
            nodesPerIteration
        );
    }

    public NativeNavigationSuccessorProbe captureNavigationSuccessorProbe(
        String environmentId,
        double[] startPositions,
        byte[] directionIndices
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeTraversalProbeSource source)) {
            throw new IllegalStateException(
                "Navigation successor probe requires the native Hytale backend"
            );
        }
        return source.captureNavigationSuccessorProbe(
            startPositions,
            directionIndices
        );
    }

    public PrivilegedEntitySnapshot capturePrivilegedEntitySnapshot(
        String environmentId,
        PrivilegedEntityQuery query
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativePrivilegedEntitySource source)) {
            throw new IllegalStateException(
                "Privileged entity capture requires the native Hytale backend"
            );
        }
        return source.capturePrivilegedEntitySnapshot(query);
    }

    public PrivilegedNpcSnapshot capturePrivilegedNpcSnapshot(
        String environmentId,
        PrivilegedEntityQuery query
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativePrivilegedEntitySource source)) {
            throw new IllegalStateException(
                "Privileged NPC capture requires the native Hytale backend"
            );
        }
        return source.capturePrivilegedNpcSnapshot(query);
    }

    public PrivilegedNpcSnapshot capturePrivilegedNpcSnapshotByUuid(
        String environmentId,
        PrivilegedNpcUuidQuery query
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativePrivilegedEntitySource source)) {
            throw new IllegalStateException(
                "Privileged NPC capture requires the native Hytale backend"
            );
        }
        return source.capturePrivilegedNpcSnapshotByUuid(query);
    }

    public NpcTraceBatch startNpcTrace(
        String environmentId,
        UUID npcUuid,
        String expectedRole,
        int capacity
    ) {
        NativeNpcTraceSource source = npcTraceSource(environmentId);
        return source.startNpcTrace(npcUuid, expectedRole, capacity);
    }

    public NpcTraceBatch drainNpcTrace(
        String environmentId,
        UUID traceUuid,
        int maxFrames,
        boolean stop
    ) {
        NativeNpcTraceSource source = npcTraceSource(environmentId);
        return source.drainNpcTrace(traceUuid, maxFrames, stop);
    }

    private NativeNpcTraceSource npcTraceSource(String environmentId) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeNpcTraceSource source)) {
            throw new IllegalStateException(
                "NPC imitation tracing requires the native Hytale backend"
            );
        }
        return source;
    }

    public WorldgenStructureMarkerSnapshot captureWorldgenStructureMarkers(
        String environmentId,
        WorldgenStructureMarkerQuery query
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeWorldgenStructureMarkerSource source)) {
            throw new IllegalStateException(
                "WorldGen structure marker capture requires the native Hytale backend"
            );
        }
        return source.captureWorldgenStructureMarkers(query);
    }

    public NativeExplosionCandidateProbe captureExplosionCandidateProbe(
        String environmentId,
        double[] origin,
        int blockDamageRadius,
        float entityDamageRadius,
        boolean ignoreControlledActor,
        int capacity
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeExplosionProbeSource source)) {
            throw new IllegalStateException(
                "Explosion candidate probe requires the native Hytale backend"
            );
        }
        return source.captureExplosionCandidateProbe(
            origin,
            blockDamageRadius,
            entityDamageRadius,
            ignoreControlledActor,
            capacity
        );
    }

    public NativeExplosionMutationProbe captureExplosionMutation(
        String environmentId,
        String worldEpoch,
        String fixtureKind,
        int cellCapacity,
        int dropCapacity
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeExplosionMutationSource source)) {
            throw new IllegalStateException(
                "Explosion mutation requires the native Hytale backend"
            );
        }
        return source.captureExplosionMutation(
            worldEpoch,
            fixtureKind,
            cellCapacity,
            dropCapacity
        );
    }

    public NativeExplosionDynamicsProbe captureExplosionDynamics(
        String environmentId,
        String worldEpoch,
        String fixtureKind,
        int capacity
    ) {
        EnvironmentSession instance = environments.get(environmentId);
        if (instance == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        if (!(instance instanceof NativeExplosionDynamicsSource source)) {
            throw new IllegalStateException(
                "Explosion dynamics requires the native Hytale backend"
            );
        }
        return source.captureExplosionDynamics(
            worldEpoch,
            fixtureKind,
            capacity
        );
    }

    public void destroy(String environmentId) {
        if (environmentId == null) return;
        EnvironmentSession instance = environments.remove(environmentId);
        if (instance != null) {
            instance.close();
        }
    }

    public int getDefaultTicksPerStep() {
        return config.getTicksPerStep();
    }

    public int getDefaultMaxEpisodeSteps() {
        return config.getMaxEpisodeSteps();
    }

    public void shutdown() {
        environments.values().forEach(EnvironmentSession::close);
        environments.clear();
    }

    private static int validateTicksPerStep(int value) {
        if (value < 1 || value > 1000) {
            throw new IllegalArgumentException("tick_rate must be 1..1000");
        }
        return value;
    }

    private static int validateMaxEpisodeSteps(int value) {
        if (value < 0) {
            throw new IllegalArgumentException("max_episode_steps cannot be negative");
        }
        return value;
    }
}
