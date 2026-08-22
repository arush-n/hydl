package com.hytalerlbridge.policy.commands;

import com.hypixel.hytale.codec.validation.Validators;
import com.hypixel.hytale.common.util.RandomUtil;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.shape.Box;
import com.hypixel.hytale.math.util.MathUtil;
import com.hypixel.hytale.math.vector.Rotation3f;
import com.hypixel.hytale.protocol.PlayerSkin;
import com.hypixel.hytale.server.core.Message;
import com.hypixel.hytale.server.core.asset.type.model.config.Model;
import com.hypixel.hytale.server.core.asset.type.model.config.ModelAsset;
import com.hypixel.hytale.server.core.command.system.AbstractCommand;
import com.hypixel.hytale.server.core.command.system.CommandContext;
import com.hypixel.hytale.server.core.command.system.arguments.system.FlagArg;
import com.hypixel.hytale.server.core.command.system.arguments.system.OptionalArg;
import com.hypixel.hytale.server.core.command.system.arguments.types.ArgTypes;
import com.hypixel.hytale.server.core.command.system.basecommands.AbstractPlayerCommand;
import com.hypixel.hytale.server.core.command.system.exceptions.GeneralCommandException;
import com.hypixel.hytale.server.core.cosmetics.CosmeticsModule;
import com.hypixel.hytale.server.core.entity.Frozen;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.modules.entity.component.HeadRotation;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entity.player.PlayerSkinComponent;
import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.NPCPlugin;
import com.hypixel.hytale.server.npc.asset.builder.Builder;
import com.hypixel.hytale.server.npc.asset.builder.BuilderInfo;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.spawning.ISpawnableWithModel;
import com.hypixel.hytale.server.spawning.SpawningContext;
import com.hytalerlbridge.policy.PolicyAgentPlugin;
import it.unimi.dsi.fastutil.Pair;
import java.util.Random;
import java.util.logging.Level;
import java.util.concurrent.ThreadLocalRandom;
import javax.annotation.Nonnull;
import org.joml.Vector3d;

/**
 * {@code /hydl spawn} -- put policy-driven agents where you are standing.
 *
 * <p><b>Why this is a player command.</b> It extends
 * {@link AbstractPlayerCommand}, so the server console cannot run it, and that
 * is not an oversight: the whole point is "spawn at <em>my</em> location", and
 * the console has no location. Use {@code /hydl status} from the console
 * instead -- that one is deliberately senderless.
 *
 * <p><b>Placement is done twice, on purpose.</b> A position is passed to
 * {@code spawnEntity}, and then the entity's own {@code TransformComponent} is
 * read back and written again. That looks redundant and is not: the engine's
 * own {@code NPCSpawnCommand} does exactly this, because the position handed to
 * {@code spawnEntity} is a request, not a placement -- the spawn resolves its
 * own final position, and the only way to put an NPC somewhere exactly is to
 * mutate the live vector afterwards. Skipping the second write produced agents
 * that were created and claimed by the policy but nowhere to be seen.
 *
 * <p><b>Why the role defaults to the configured one.</b> The policy only drives
 * NPCs whose role name matches what {@code PolicyAttachmentSystem} was armed
 * with. Spawning any other role produces an NPC that stands there running its
 * authored behaviour, looking like a broken agent when nothing is broken.
 */
public final class HydlSpawnCommand extends AbstractPlayerCommand {

    /** Matches {@code /npc spawn}, so a mixed test scatters identically. */
    private static final double DEFAULT_RADIUS = 8.0;

    private final PolicyAgentPlugin plugin;

    @Nonnull
    private final OptionalArg<Integer> countArg = withOptionalArg(
            "count", "How many agents to spawn (default 1).", ArgTypes.INTEGER)
            .addValidator(Validators.greaterThan(0));
    @Nonnull
    private final OptionalArg<Double> radiusArg = withOptionalArg(
            "radius", "Scatter radius when count > 1 (default 8).",
            ArgTypes.DOUBLE)
            .addValidator(Validators.greaterThan(0.0));
    @Nonnull
    private final OptionalArg<String> roleArg = withOptionalArg(
            "role", "Role to spawn; defaults to the role the policy drives.",
            ArgTypes.STRING);
    @Nonnull
    private final OptionalArg<String> modelArg = withOptionalArg(
            "model", "ModelAsset id to use as the body instead of the role's own.",
            ArgTypes.STRING);
    @Nonnull
    private final OptionalArg<Float> scaleArg = withOptionalArg(
            "scale", "Body scale; clamped to the model's own limits.",
            ArgTypes.FLOAT);
    @Nonnull
    private final OptionalArg<Double> distanceArg = withOptionalArg(
            "distance", "Metres in front of you to place a single agent.",
            ArgTypes.DOUBLE);
    @Nonnull
    private final FlagArg randomModelArg = withFlagArg(
            "randomModel", "Give each agent a random player body and skin.");
    @Nonnull
    private final FlagArg frozenArg = withFlagArg(
            "frozen", "Spawn inert, so you can stage a fight before it starts.");
    @Nonnull
    private final FlagArg facingArg = withFlagArg(
            "facing", "Face the agents toward you instead of away.");

    public HydlSpawnCommand(PolicyAgentPlugin plugin) {
        super("spawn", "Spawn policy-driven agents at your position.");
        this.plugin = plugin;
    }

    /**
     * Thin wrapper whose only job is to make a failure visible.
     *
     * <p>{@link AbstractPlayerCommand} runs {@code execute} inside
     * {@code runAsync(...)}, so the body executes on a world thread inside a
     * {@code CompletableFuture}. A {@code RuntimeException} thrown there is
     * captured by the future and, with nothing attached to observe it,
     * <em>silently discarded</em> -- no stack trace in the log, no message to
     * the player. The command shows up in the log as "executed" and simply
     * does nothing, which is exactly the symptom that made this command look
     * like it spawned on the first call and never again.
     *
     * <p>The engine's own {@code NPCSpawnCommand} guards the same way: it
     * catches {@code IllegalState/NullPointer/IllegalArgument}, logs
     * "Spawn failed: ...", and rethrows as a {@link GeneralCommandException}
     * so the sender is told. Copying that is not defensive clutter -- without
     * it, this whole class is undebuggable.
     */
    @Override
    protected void execute(@Nonnull CommandContext context,
                           @Nonnull Store<EntityStore> store,
                           @Nonnull Ref<EntityStore> ref,
                           @Nonnull PlayerRef playerRef,
                           @Nonnull World world) {
        try {
            spawnAgents(context, store, ref, playerRef, world);
        } catch (GeneralCommandException expected) {
            throw expected;                       // already carries its reason
        } catch (RuntimeException failure) {
            AbstractCommand.LOGGER.at(Level.SEVERE).withCause(failure).log(
                "hydl spawn failed: %s", failure);
            throw new GeneralCommandException(Message.raw(
                "spawn failed: " + failure + " -- full trace in the server log."));
        }
    }

    private void spawnAgents(@Nonnull CommandContext context,
                             @Nonnull Store<EntityStore> store,
                             @Nonnull Ref<EntityStore> ref,
                             @Nonnull PlayerRef playerRef,
                             @Nonnull World world) {
        if (!plugin.isArmed()) {
            throw new GeneralCommandException(Message.raw(
                "The policy is not armed, so a spawned agent would just stand "
                    + "there. Run /hydl status to see why."));
        }

        String armedRole = plugin.controlledRole();
        String roleName = roleArg.provided(context) ? roleArg.get(context) : armedRole;
        int count = countArg.provided(context) ? countArg.get(context) : 1;
        double radius = radiusArg.provided(context) ? radiusArg.get(context) : DEFAULT_RADIUS;

        NPCPlugin npc = NPCPlugin.get();
        int roleIndex = npc.getIndex(roleName);
        if (roleIndex == Integer.MIN_VALUE) {
            throw new GeneralCommandException(Message.raw(
                "No such NPC role: " + roleName));
        }
        BuilderInfo info = npc.getRoleBuilderInfo(roleIndex);
        if (info == null) {
            throw new GeneralCommandException(Message.raw(
                "Role has no builder: " + roleName));
        }
        npc.forceValidation(roleIndex);
        if (!npc.testAndValidateRole(info)) {
            throw new GeneralCommandException(Message.raw(
                "Role failed validation: " + roleName));
        }

        TransformComponent transform = store.getComponent(
            ref, TransformComponent.getComponentType());
        HeadRotation head = store.getComponent(ref, HeadRotation.getComponentType());
        BoundingBox bounds = store.getComponent(ref, BoundingBox.getComponentType());
        if (transform == null || head == null || bounds == null) {
            throw new GeneralCommandException(Message.raw(
                "Your entity is missing a transform; try rejoining."));
        }
        // Copied, not aliased: getPosition() hands back the live vector, and
        // the engine moves entities by mutating it. Holding the player's own
        // vector here and writing to it would teleport the player.
        Vector3d origin = new Vector3d(transform.getPosition());
        Box playerBox = bounds.getBoundingBox();

        float yaw = head.getRotation().y();
        Rotation3f rotation = new Rotation3f(
            0.0F, facingArg.get(context) ? yaw - (float) Math.PI : yaw, 0.0F);

        // A single agent lands slightly in front of you rather than inside you:
        // spawned at the exact player position it is hidden by your own body at
        // first-person, which reads as "nothing spawned".
        double forward = distanceArg.provided(context) ? distanceArg.get(context) : 2.0;
        double offsetX = -Math.sin(yaw) * forward;
        double offsetZ = Math.cos(yaw) * forward;

        Random random = ThreadLocalRandom.current();
        Vector3d last = null;
        int spawned = 0;

        for (int i = 0; i < count; i++) {
            Builder<Role> builder = npc.tryGetCachedValidRole(roleIndex);
            if (!(builder instanceof ISpawnableWithModel spawnable)
                    || !builder.isSpawnable()) {
                throw new GeneralCommandException(Message.raw(
                    "Role is not directly spawnable (abstract template?): "
                        + roleName));
            }
            SpawningContext spawning = new SpawningContext();
            if (!spawning.setSpawnable(spawnable)) {
                throw new GeneralCommandException(Message.raw(
                    "Could not prepare a spawn for " + roleName));
            }

            Model model = chooseModel(context, spawning, store);

            Vector3d request = new Vector3d(origin);
            request.y = request.y + playerBox.min.y - model.getBoundingBox().min.y;
            if (count == 1) {
                request.x += offsetX;
                request.z += offsetZ;
            }

            // Logged server-side, not just replied to the player: a
            // GeneralCommandException is delivered to the sender's chat and
            // never reaches the log, so a failure here is invisible to anyone
            // reading the server. That cost a full debugging round.
            AbstractCommand.LOGGER.at(Level.INFO).log(
                "hydl spawn: i=%d role=%s player=%.1f/%.1f/%.1f "
                    + "request=%.1f/%.1f/%.1f model=%s modelMinY=%.2f",
                i, roleName, origin.x, origin.y, origin.z,
                request.x, request.y, request.z,
                model.getModelAssetId(), model.getBoundingBox().min.y);

            Pair<Ref<EntityStore>, NPCEntity> pair =
                npc.spawnEntity(store, roleIndex, request, rotation, model, null);
            if (pair == null || pair.first() == null || pair.second() == null) {
                AbstractCommand.LOGGER.at(Level.WARNING).log(
                    "hydl spawn: spawnEntity returned nothing for %s", roleName);
                throw new GeneralCommandException(Message.raw(
                    "spawnEntity produced no entity for " + roleName
                        + " -- see the server log."));
            }
            Ref<EntityStore> agentRef = pair.first();
            NPCEntity entity = pair.second();

            // The second placement, and the one that counts -- see the class
            // docstring. `getPosition()` is the entity's live vector; mutating
            // it is how the engine's own spawn command positions an NPC.
            TransformComponent agentTransform = store.getComponent(
                agentRef, TransformComponent.getComponentType());
            if (agentTransform == null) {
                throw new GeneralCommandException(Message.raw(
                    "Spawned entity has no transform; aborting."));
            }
            Vector3d placed = agentTransform.getPosition();
            double x = placed.x();
            double y = placed.y();
            double z = placed.z();
            if (count > 1) {
                x += random.nextDouble() * 2.0 * radius - radius;
                z += random.nextDouble() * 2.0 * radius - radius;
            }
            y += 0.1;
            placed.set(x, y, z);
            last = new Vector3d(placed);

            // Without this the NPC walks back to wherever it thinks it spawned,
            // which for a hand-placed agent is the wrong anchor.
            AbstractCommand.LOGGER.at(Level.INFO).log(
                "hydl spawn: placed at %.1f/%.1f/%.1f (requested %.1f/%.1f/%.1f)",
                placed.x(), placed.y(), placed.z(), request.x, request.y, request.z);
            entity.saveLeashInformation(placed, agentTransform.getRotation());
            if (frozenArg.get(context)) {
                store.ensureComponent(agentRef, Frozen.getComponentType());
            }
            spawned++;
        }

        StringBuilder said = new StringBuilder()
            .append("Spawned ").append(spawned).append(" x ").append(roleName);
        if (last != null) {
            // Reporting the position is the difference between "it did not
            // work" and "it is 40 blocks under me": an agent can be created,
            // claimed and ticking while being nowhere you are looking.
            said.append(String.format(" at %.1f/%.1f/%.1f", last.x, last.y, last.z));
        }
        if (roleName.equals(armedRole)) {
            said.append(" -- policy will claim ")
                .append(spawned == 1 ? "it" : "them").append('.');
        } else {
            said.append(" -- NOTE: the policy drives '").append(armedRole)
                .append("', so it will NOT attach to these.");
        }
        if (frozenArg.get(context)) {
            said.append(" Frozen; /npc thaw to start them.");
        }
        context.sendMessage(Message.raw(said.toString()));
    }

    /**
     * The body: the role's own, a named {@link ModelAsset}, or a random player.
     *
     * <p>Scale is clamped to the model's declared limits rather than accepted
     * as given -- an out-of-range scale produces a model the client cannot draw
     * sensibly, which is indistinguishable from the agent not existing.
     */
    private Model chooseModel(@Nonnull CommandContext context,
                              @Nonnull SpawningContext spawning,
                              @Nonnull Store<EntityStore> store) {
        Model model;
        if (randomModelArg.get(context)) {
            PlayerSkin skin = CosmeticsModule.get()
                .generateRandomSkin(RandomUtil.getSecureRandom());
            model = CosmeticsModule.get().createModel(skin);
        } else if (modelArg.provided(context)) {
            String id = modelArg.get(context);
            ModelAsset asset = ModelAsset.getAssetMap().getAsset(id);
            if (asset == null) {
                throw new GeneralCommandException(Message.raw(
                    "No such ModelAsset: " + id));
            }
            model = Model.createScaledModel(asset, asset.getMinScale() > 1.0F
                ? asset.getMinScale() : 1.0F);
        } else {
            model = spawning.getModel();
        }

        if (scaleArg.provided(context)) {
            ModelAsset asset = ModelAsset.getAssetMap()
                .getAsset(model.getModelAssetId());
            if (asset != null) {
                float scale = MathUtil.clamp(
                    scaleArg.get(context), asset.getMinScale(), asset.getMaxScale());
                model = Model.createScaledModel(asset, scale);
            }
        }
        return model;
    }
}
