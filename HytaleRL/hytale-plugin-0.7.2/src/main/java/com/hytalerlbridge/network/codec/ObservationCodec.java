package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.environment.StepResult;
import com.hytalerlbridge.observation.NativeActorEvidenceRequest;
import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import com.hytalerlbridge.observation.NativeInventoryFrame;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;
import org.msgpack.core.MessagePacker;
import org.msgpack.value.Value;
import static com.hytalerlbridge.network.wire.MessageFields.getStringField;
import static com.hytalerlbridge.network.wire.MessageFields.getLongField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalStringListField;
import static com.hytalerlbridge.network.wire.MessageFields.checkedInt;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;
import static com.hytalerlbridge.network.wire.FrameWriter.packInfoValue;
import static com.hytalerlbridge.network.wire.BridgeIdentity.BRIDGE_SHA256;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeFloat64LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeInt32LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.packBinary;
import com.hytalerlbridge.network.BridgeRuntimeIdentity;

/**
 * Extracted verbatim from {@code ClientHandler}.
 */
public final class ObservationCodec {

    private ObservationCodec() {}

    public static void sendObservation(DataOutputStream output, StepResult result)
        throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(6);
            packer.packString("type");
            packer.packString("observation");
            packer.packString("obs");
            packObservation(packer, result);
            packer.packString("reward");
            packer.packDouble(result.reward());
            packer.packString("terminated");
            packer.packBoolean(result.terminated());
            packer.packString("truncated");
            packer.packBoolean(result.truncated());
            packer.packString("info");
            Map<String, Object> taskInfo = result.info();
            packer.packMapHeader(3 + taskInfo.size() + (BRIDGE_SHA256.isPresent() ? 1 : 0));
            packer.packString("tick");
            packer.packInt(result.tick());
            packer.packString("server_step");
            packer.packInt(result.step());
            packer.packString("native_server_process_uptime_seconds");
            packer.packDouble(BridgeRuntimeIdentity.processUptimeSeconds());
            if (BRIDGE_SHA256.isPresent()) {
                packer.packString("bridge_sha256");
                packer.packString(BRIDGE_SHA256.orElseThrow());
            }
            for (Map.Entry<String, Object> entry : taskInfo.entrySet()) {
                packer.packString(entry.getKey());
                packInfoValue(packer, entry.getValue());
            }
        });
    }


    public static void packObservation(MessagePacker packer, StepResult result)
        throws IOException {
        var observation = result.observation();
        packer.packMapHeader(17);

        packer.packString("position");
        packer.packArrayHeader(3);
        packer.packDouble(observation.x());
        packer.packDouble(observation.y());
        packer.packDouble(observation.z());

        packer.packString("velocity");
        packer.packArrayHeader(3);
        packer.packDouble(observation.vx());
        packer.packDouble(observation.vy());
        packer.packDouble(observation.vz());

        packer.packString("yaw");
        packer.packDouble(observation.yaw());
        packer.packString("pitch");
        packer.packDouble(observation.pitch());
        packer.packString("health");
        packer.packDouble(observation.health());
        packer.packString("food_buff_timer");
        packer.packInt(observation.foodBuffTimer());
        packer.packString("stamina");
        packer.packDouble(observation.stamina());
        packer.packString("mana");
        packer.packDouble(observation.mana());

        packer.packString("inventory");
        int[] inventory = observation.inventory();
        packer.packArrayHeader(inventory.length);
        for (int slot : inventory) packer.packInt(slot);

        packer.packString("nearby_blocks");
        var blocks = observation.nearbyBlocks();
        packer.packArrayHeader(blocks.size());
        for (int[] block : blocks) {
            packer.packArrayHeader(4);
            for (int value : block) packer.packInt(value);
        }

        packer.packString("nearby_entities");
        var entities = observation.nearbyEntities();
        packer.packArrayHeader(entities.size());
        for (int[] entity : entities) {
            packer.packArrayHeader(4);
            for (int value : entity) packer.packInt(value);
        }

        packer.packString("time_of_day");
        packer.packInt(observation.timeOfDay());
        packer.packString("craftable_count");
        packer.packInt(observation.craftableRecipeCount());

        packer.packString("geometry");
        packGeometry(packer, observation.geometry());

        packer.packString("audio");
        packAudio(packer, observation.audio());

        packer.packString("native_actor_evidence");
        packNativeActorEvidence(packer, observation.nativeActorEvidence());

        packer.packString("native_inventory");
        packNativeInventory(packer, observation.nativeInventory());
    }


    public static void packNativeInventory(
        MessagePacker packer,
        NativeInventoryFrame inventory
    ) throws IOException {
        packer.packMapHeader(7);
        packer.packString("schema");
        packer.packString(NativeInventoryFrame.SCHEMA);
        packer.packString("version");
        packer.packInt(NativeInventoryFrame.VERSION);
        packer.packString("contract_sha256");
        packer.packString(NativeInventoryFrame.CONTRACT_SHA256);
        packer.packString("available");
        packer.packBoolean(inventory.available());
        packer.packString("unavailable_reason");
        packer.packString(inventory.unavailableReason());
        packer.packString("active_slots");
        packer.packMapHeader(3);
        packer.packString("hotbar");
        packer.packInt(inventory.activeHotbarSlot());
        packer.packString("utility");
        packer.packInt(inventory.activeUtilitySlot());
        packer.packString("tools");
        packer.packInt(inventory.activeToolsSlot());
        packer.packString("containers");
        packer.packArrayHeader(inventory.containers().size());
        for (NativeInventoryFrame.Container container : inventory.containers()) {
            packer.packMapHeader(6);
            packer.packString("name");
            packer.packString(container.name());
            packer.packString("section_id");
            packer.packInt(container.sectionId());
            packer.packString("available");
            packer.packBoolean(container.available());
            packer.packString("unavailable_reason");
            packer.packString(container.unavailableReason());
            packer.packString("capacity");
            packer.packInt(container.capacity());
            packer.packString("occupied_slots");
            packer.packArrayHeader(container.occupiedSlots().size());
            for (NativeInventoryFrame.Slot slot : container.occupiedSlots()) {
                packer.packMapHeader(7);
                packer.packString("slot");
                packer.packInt(slot.slot());
                packer.packString("item_id");
                packer.packString(slot.itemId());
                packer.packString("item_runtime_index");
                packer.packInt(slot.itemRuntimeIndex());
                packer.packString("quantity");
                packer.packInt(slot.quantity());
                packer.packString("durability");
                packer.packDouble(slot.durability());
                packer.packString("max_durability");
                packer.packDouble(slot.maxDurability());
                packer.packString("metadata_present");
                packer.packBoolean(slot.metadataPresent());
            }
        }
    }


    public static void packNativeActorEvidence(
        MessagePacker packer,
        NativeActorEvidenceFrame evidence
    ) throws IOException {
        packer.packMapHeader(20);
        packer.packString("schema");
        packer.packString(NativeActorEvidenceFrame.SCHEMA);
        packer.packString("version");
        packer.packInt(NativeActorEvidenceFrame.VERSION);
        packer.packString("available");
        packer.packBoolean(evidence.available());
        packer.packString("unavailable_reason");
        packer.packString(evidence.unavailableReason());
        packer.packString("world_tick");
        packer.packLong(evidence.worldTick());
        packer.packString("evidence_contract_sha256");
        packer.packString(NativeActorEvidenceFrame.CONTRACT_SHA256);

        packer.packString("capacities");
        packer.packMapHeader(8);
        packer.packString("entities");
        packer.packInt(NativeActorEvidenceFrame.ENTITY_COUNT);
        packer.packString("resources");
        packer.packInt(NativeActorEvidenceFrame.RESOURCE_COUNT);
        packer.packString("statuses_per_entity");
        packer.packInt(NativeActorEvidenceFrame.STATUS_CAPACITY);
        packer.packString("abilities_per_entity");
        packer.packInt(NativeActorEvidenceFrame.ABILITY_CAPACITY);
        packer.packString("skills");
        packer.packInt(NativeActorEvidenceFrame.SKILL_COUNT);
        packer.packString("dodge_directions");
        packer.packInt(NativeActorEvidenceFrame.DODGE_DIRECTION_COUNT);
        packer.packString("door_candidates");
        packer.packInt(NativeActorEvidenceFrame.DOOR_CANDIDATE_CAPACITY);
        packer.packString("door_intents");
        packer.packInt(NativeActorEvidenceFrame.DOOR_INTENT_COUNT);

        packer.packString("actors");
        packer.packArrayHeader(evidence.actors().size());
        for (NativeActorEvidenceFrame.Actor actor : evidence.actors()) {
            packNativeActor(packer, actor);
        }

        packer.packString("abilities");
        packer.packArrayHeader(evidence.abilities().size());
        for (NativeActorEvidenceFrame.Ability ability : evidence.abilities()) {
            packNativeAbility(packer, ability);
        }

        packer.packString("world_geometry_available");
        packer.packBoolean(evidence.worldGeometryAvailable());
        packer.packString("role_opaque_cell_mask_available");
        packer.packBoolean(evidence.roleOpaqueCellMaskAvailable());
        packer.packString("role_opaque_cell_mask");
        packBooleans(packer, evidence.roleOpaqueCellMask());
        packer.packString("skill_action_mask");
        packBooleans(packer, evidence.skillActionMask());
        packer.packString("jump_action_available");
        packer.packBoolean(evidence.jumpActionAvailable());
        packer.packString("guard_action_available");
        packer.packBoolean(evidence.guardActionAvailable());
        packer.packString("dodge_action_mask");
        packBooleans(packer, evidence.dodgeActionMask());
        packer.packString("door_action_mask");
        boolean[][] doorMask = evidence.doorActionMask();
        packer.packArrayHeader(doorMask.length);
        for (boolean[] row : doorMask) packBooleans(packer, row);
        packer.packString("loadout_failure");
        packer.packInt(evidence.loadoutFailure());
        packer.packString("mechanics_failure_bits");
        packer.packInt(evidence.mechanicsFailureBits());
        packer.packString("arsenal_failure_bits");
        packer.packInt(evidence.arsenalFailureBits());
    }


    public static void packNativeActor(
        MessagePacker packer,
        NativeActorEvidenceFrame.Actor actor
    ) throws IOException {
        packer.packMapHeader(23);
        packer.packString("entity_id");
        packer.packInt(actor.entityId());
        packer.packString("present");
        packer.packBoolean(actor.present());
        packer.packString("perceptible");
        packer.packBoolean(actor.perceptible());
        packer.packString("role_id");
        packer.packString(actor.roleId());
        packer.packString("item_id");
        packer.packString(actor.itemId());
        packer.packString("item_runtime_index");
        packer.packInt(actor.itemRuntimeIndex());
        packer.packString("active_ability_slot");
        packer.packInt(actor.activeAbilitySlot());
        packer.packString("position");
        packDoubles(packer, actor.position());
        packer.packString("velocity");
        packDoubles(packer, actor.velocity());
        packer.packString("motion_force");
        packNativeMotionForce(packer, actor.motionForce());
        packer.packString("yaw_degrees");
        packer.packDouble(actor.yawDegrees());
        packer.packString("pitch_degrees");
        packer.packDouble(actor.pitchDegrees());
        packer.packString("health");
        packer.packDouble(actor.health());
        packer.packString("max_health");
        packer.packDouble(actor.maxHealth());
        packer.packString("resource_values");
        packDoubles(packer, actor.resourceValues());
        packer.packString("resource_maximums");
        packDoubles(packer, actor.resourceMaximums());
        packer.packString("resource_available");
        packBooleans(packer, actor.resourceAvailable());
        packer.packString("defense_values");
        packDoubles(packer, actor.defenseValues());
        packer.packString("defense_available");
        packBooleans(packer, actor.defenseAvailable());
        packer.packString("statuses");
        packer.packArrayHeader(actor.statuses().size());
        for (NativeActorEvidenceFrame.Status status : actor.statuses()) {
            packer.packMapHeader(7);
            packer.packString("runtime_effect_index");
            packer.packInt(status.runtimeEffectIndex());
            packer.packString("effect_id");
            packer.packString(status.effectId());
            packer.packString("initial_duration_seconds");
            packer.packDouble(status.initialDurationSeconds());
            packer.packString("remaining_duration_seconds");
            packer.packDouble(status.remainingDurationSeconds());
            packer.packString("infinite");
            packer.packBoolean(status.infinite());
            packer.packString("debuff");
            packer.packBoolean(status.debuff());
            packer.packString("invulnerable");
            packer.packBoolean(status.invulnerable());
        }
        packer.packString("actor_world_values");
        packDoubles(packer, actor.actorWorldValues());
        packer.packString("actor_world_available");
        packBooleans(packer, actor.actorWorldAvailable());
        packer.packString("movement_states");
        packer.packMapHeader(2);
        packer.packString("available");
        packer.packBoolean(actor.movementStates().available());
        packer.packString("bits");
        packer.packInt(actor.movementStates().bits());
    }


    public static void packNativeMotionForce(
        MessagePacker packer,
        NativeActorEvidenceFrame.MotionForce force
    ) throws IOException {
        packer.packMapHeader(10);
        packer.packString("legacy_external_available");
        packer.packBoolean(force.legacyExternalAvailable());
        packer.packString("legacy_external_velocity");
        packDoubles(packer, force.legacyExternalVelocity());
        packer.packString("configured_applied_available");
        packer.packBoolean(force.configuredAppliedAvailable());
        packer.packString("configured_applied_velocity");
        packDoubles(packer, force.configuredAppliedVelocity());
        packer.packString("configured_applied_count");
        packer.packInt(force.configuredAppliedCount());
        packer.packString("pending_knockback_available");
        packer.packBoolean(force.pendingKnockbackAvailable());
        packer.packString("pending_knockback_velocity");
        packDoubles(packer, force.pendingKnockbackVelocity());
        packer.packString("projected_available");
        packer.packBoolean(force.projectedAvailable());
        packer.packString("projected_velocity");
        packDoubles(packer, force.projectedVelocity());
        packer.packString("projection_source");
        packer.packInt(force.projectionSource());
    }


    public static void packNativeAbility(
        MessagePacker packer,
        NativeActorEvidenceFrame.Ability ability
    ) throws IOException {
        packer.packMapHeader(8);
        packer.packString("slot");
        packer.packInt(ability.slot());
        packer.packString("interaction_id");
        packer.packString(ability.interactionId());
        packer.packString("interaction_type");
        packer.packString(ability.interactionType());
        packer.packString("authored");
        packer.packBoolean(ability.authored());
        packer.packString("host_legal");
        packer.packBoolean(ability.hostLegal());
        packer.packString("active");
        packer.packBoolean(ability.active());
        packer.packString("start_world_tick");
        packer.packLong(ability.startWorldTick());
        packer.packString("finish_world_tick");
        packer.packLong(ability.finishWorldTick());
    }


    public static void packDoubles(
        MessagePacker packer,
        double[] values
    ) throws IOException {
        packer.packArrayHeader(values.length);
        for (double value : values) packer.packDouble(value);
    }


    public static void packBooleans(
        MessagePacker packer,
        boolean[] values
    ) throws IOException {
        packer.packArrayHeader(values.length);
        for (boolean value : values) packer.packBoolean(value);
    }


    public static void packAudio(
        MessagePacker packer,
        com.hytalerlbridge.observation.AudioFrame audio
    ) throws IOException {
        packer.packMapHeader(5);
        packer.packString("schema");
        packer.packString(com.hytalerlbridge.observation.AudioFrame.SCHEMA);
        packer.packString("version");
        packer.packInt(com.hytalerlbridge.observation.AudioFrame.VERSION);
        packer.packString("available");
        packer.packBoolean(audio.available());

        packer.packString("capture_mask");
        packer.packArrayHeader(3);
        packer.packBoolean(audio.captures2d());
        packer.packBoolean(audio.captures3d());
        packer.packBoolean(audio.capturesEntity());

        packer.packString("events");
        packer.packArrayHeader(audio.events().size());
        for (var event : audio.events()) {
            // Positional layout is versioned in protocol/messages.md.
            packer.packArrayHeader(10);
            packer.packInt(event.kind());
            packer.packInt(event.soundEventIndex());
            packer.packInt(event.category());
            packer.packBoolean(event.spatial());
            packer.packDouble(event.relativeX());
            packer.packDouble(event.relativeY());
            packer.packDouble(event.relativeZ());
            packer.packDouble(event.volumeModifier());
            packer.packDouble(event.pitchModifier());
            packer.packDouble(event.ageSeconds());
        }
    }


    public static void packGeometry(
        MessagePacker packer,
        com.hytalerlbridge.geometry.GeometryFrame geometry
    ) throws IOException {
        packer.packMapHeader(15);

        packer.packString("schema");
        packer.packString(com.hytalerlbridge.geometry.GeometryContract.SCHEMA);
        packer.packString("version");
        packer.packInt(com.hytalerlbridge.geometry.GeometryContract.VERSION);
        packer.packString("available");
        packer.packBoolean(geometry.available());
        packer.packString("exact_collision_shapes");
        packer.packBoolean(geometry.exactCollisionShapes());

        packer.packString("origin");
        packer.packArrayHeader(3);
        packer.packInt(geometry.originX());
        packer.packInt(geometry.originY());
        packer.packInt(geometry.originZ());

        packer.packString("cells");
        var cells = geometry.cells();
        packer.packArrayHeader(cells.size());
        for (var cell : cells) {
            // Positional layout is versioned in protocol/messages.md.
            packer.packArrayHeader(29);
            packer.packInt(cell.dx());
            packer.packInt(cell.dy());
            packer.packInt(cell.dz());
            packer.packInt(cell.runtimeBlockId());
            packer.packInt(cell.runtimeFluidId());
            packer.packInt(cell.shapeId());
            packer.packInt(cell.rotationIndex());
            packer.packInt(cell.flags());
            packer.packInt(cell.fluidLevel());
            packer.packDouble(cell.fluidFillHeight());
            packer.packInt(cell.supportValue());
            packer.packInt(cell.blockDamage());
            packer.packInt(cell.fluidDamage());
            packer.packDouble(cell.friction());
            packer.packDouble(cell.drag());
            packer.packDouble(cell.horizontalSpeedMultiplier());
            packer.packDouble(cell.jumpForceMultiplier());
            packer.packDouble(cell.climbUpSpeedMultiplier());
            packer.packDouble(cell.climbDownSpeedMultiplier());
            packer.packDouble(cell.climbLateralSpeedMultiplier());
            packer.packDouble(cell.terminalVelocityModifier());
            packer.packDouble(cell.bounceVelocity());
            packer.packDouble(cell.fluidSwimUpSpeed());
            packer.packDouble(cell.fluidSwimDownSpeed());
            packer.packDouble(cell.fluidSinkSpeed());
            packer.packDouble(cell.fluidHorizontalSpeedMultiplier());
            packer.packDouble(cell.fluidFieldOfViewMultiplier());
            packer.packDouble(cell.fluidEntryVelocityMultiplier());
            double[] boxes = cell.collisionBoxes();
            packer.packArrayHeader(boxes.length);
            for (double value : boxes) packer.packDouble(value);
        }

        packer.packString("agent_bounds");
        double[] bounds = geometry.agentBounds();
        packer.packArrayHeader(bounds.length);
        for (double value : bounds) packer.packDouble(value);

        packer.packString("target_bounds");
        double[] targetBounds = geometry.targetBounds();
        packer.packArrayHeader(targetBounds.length);
        for (double value : targetBounds) packer.packDouble(value);

        packer.packString("agent_los_offset");
        double[] agentLosOffset = geometry.agentLosOffset();
        packer.packArrayHeader(agentLosOffset.length);
        for (double value : agentLosOffset) packer.packDouble(value);

        packer.packString("target_los_offset");
        double[] targetLosOffset = geometry.targetLosOffset();
        packer.packArrayHeader(targetLosOffset.length);
        for (double value : targetLosOffset) packer.packDouble(value);

        packer.packString("contacts");
        var contacts = geometry.contacts();
        packer.packArrayHeader(contacts.size());
        for (var contact : contacts) {
            packer.packArrayHeader(13);
            packer.packInt(contact.dx());
            packer.packInt(contact.dy());
            packer.packInt(contact.dz());
            packer.packInt(contact.detailBoxIndex());
            packer.packDouble(contact.normalX());
            packer.packDouble(contact.normalY());
            packer.packDouble(contact.normalZ());
            packer.packDouble(contact.pointX());
            packer.packDouble(contact.pointY());
            packer.packDouble(contact.pointZ());
            packer.packDouble(contact.collisionStart());
            packer.packDouble(contact.collisionEnd());
            int flags = (contact.touching() ? 1 : 0)
                | (contact.overlapping() ? 2 : 0);
            packer.packInt(flags);
        }

        packer.packString("grounded");
        packer.packBoolean(geometry.grounded());
        packer.packString("ceiling_contact");
        packer.packBoolean(geometry.ceilingContact());
        packer.packString("target_los");
        packer.packBoolean(geometry.targetLineOfSight());
        packer.packString("target_los_valid");
        packer.packBoolean(geometry.targetLineOfSightValid());
    }

    /** Packs trace geometry as fixed little-endian columns, preserving v5 values. */
    public static void packTraceGeometry(
        MessagePacker packer,
        com.hytalerlbridge.geometry.GeometryFrame geometry
    ) throws IOException {
        var cells = geometry.cells();
        int[] cellIntegers = new int[cells.size() * 12];
        double[] cellValues = new double[cells.size() * 16];
        byte[] boxCounts = new byte[cells.size()];
        int totalBoxes = cells.stream()
            .mapToInt(com.hytalerlbridge.geometry.GeometryCell::collisionBoxCount)
            .sum();
        double[] boxes = new double[totalBoxes * 6];
        int boxOffset = 0;
        for (int row = 0; row < cells.size(); row++) {
            var cell = cells.get(row);
            int integerOffset = row * 12;
            int[] integers = {
                cell.dx(), cell.dy(), cell.dz(), cell.runtimeBlockId(),
                cell.runtimeFluidId(), cell.shapeId(), cell.rotationIndex(),
                cell.flags(), cell.fluidLevel(), cell.supportValue(),
                cell.blockDamage(), cell.fluidDamage()
            };
            System.arraycopy(integers, 0, cellIntegers, integerOffset, integers.length);
            int valueOffset = row * 16;
            double[] values = {
                cell.fluidFillHeight(), cell.friction(), cell.drag(),
                cell.horizontalSpeedMultiplier(), cell.jumpForceMultiplier(),
                cell.climbUpSpeedMultiplier(), cell.climbDownSpeedMultiplier(),
                cell.climbLateralSpeedMultiplier(), cell.terminalVelocityModifier(),
                cell.bounceVelocity(), cell.fluidSwimUpSpeed(),
                cell.fluidSwimDownSpeed(), cell.fluidSinkSpeed(),
                cell.fluidHorizontalSpeedMultiplier(),
                cell.fluidFieldOfViewMultiplier(),
                cell.fluidEntryVelocityMultiplier()
            };
            System.arraycopy(values, 0, cellValues, valueOffset, values.length);
            double[] cellBoxes = cell.collisionBoxes();
            boxCounts[row] = (byte) cell.collisionBoxCount();
            System.arraycopy(cellBoxes, 0, boxes, boxOffset, cellBoxes.length);
            boxOffset += cellBoxes.length;
        }

        double[] boundsAndOffsets = new double[18];
        int offset = 0;
        for (double[] values : new double[][] {
            geometry.agentBounds(), geometry.targetBounds(),
            geometry.agentLosOffset(), geometry.targetLosOffset()
        }) {
            System.arraycopy(values, 0, boundsAndOffsets, offset, values.length);
            offset += values.length;
        }
        var contacts = geometry.contacts();
        double[] contactValues = new double[contacts.size() * 13];
        for (int row = 0; row < contacts.size(); row++) {
            var contact = contacts.get(row);
            int start = row * 13;
            double[] values = {
                contact.dx(), contact.dy(), contact.dz(), contact.detailBoxIndex(),
                contact.normalX(), contact.normalY(), contact.normalZ(),
                contact.pointX(), contact.pointY(), contact.pointZ(),
                contact.collisionStart(), contact.collisionEnd(),
                (contact.touching() ? 1 : 0) | (contact.overlapping() ? 2 : 0)
            };
            System.arraycopy(values, 0, contactValues, start, values.length);
        }
        int stateFlags = (geometry.grounded() ? 1 : 0)
            | (geometry.ceilingContact() ? 2 : 0)
            | (geometry.targetLineOfSight() ? 4 : 0)
            | (geometry.targetLineOfSightValid() ? 8 : 0);

        packer.packMapHeader(15);
        packer.packString("schema");
        packer.packString(com.hytalerlbridge.geometry.GeometryContract.SCHEMA);
        packer.packString("version");
        packer.packInt(com.hytalerlbridge.geometry.GeometryContract.VERSION);
        packer.packString("transport");
        packer.packString("hytale_geometry_binary_v1");
        packer.packString("available");
        packer.packBoolean(geometry.available());
        packer.packString("exact_collision_shapes");
        packer.packBoolean(geometry.exactCollisionShapes());
        packBinary(
            packer,
            "origin_i32_le",
            encodeInt32LittleEndian(new int[] {
                geometry.originX(), geometry.originY(), geometry.originZ()
            })
        );
        packer.packString("cell_count");
        packer.packInt(cells.size());
        packBinary(packer, "cell_i32_le", encodeInt32LittleEndian(cellIntegers));
        packBinary(packer, "cell_f64_le", encodeFloat64LittleEndian(cellValues));
        packBinary(packer, "collision_box_counts_u8", boxCounts);
        packBinary(packer, "collision_boxes_f64_le", encodeFloat64LittleEndian(boxes));
        packBinary(
            packer,
            "bounds_offsets_f64_le",
            encodeFloat64LittleEndian(boundsAndOffsets)
        );
        packer.packString("contact_count");
        packer.packInt(contacts.size());
        packBinary(
            packer,
            "contacts_f64_le",
            encodeFloat64LittleEndian(contactValues)
        );
        packer.packString("state_flags");
        packer.packInt(stateFlags);
    }


    public static NativeActorEvidenceRequest getNativeActorEvidenceRequest(
        Map<Value, Value> options
    ) {
        String contractSha256 = getStringField(
            options,
            "learner_v3_actor_evidence_contract_sha256"
        );
        long entityCount = getLongField(
            options,
            "learner_v3_actor_evidence_entity_count",
            -1
        );
        long statusCapacity = getLongField(
            options,
            "learner_v3_actor_evidence_status_capacity",
            -1
        );
        long abilityCapacity = getLongField(
            options,
            "learner_v3_actor_evidence_ability_capacity",
            -1
        );
        java.util.List<String> resourceStatIds = getOptionalStringListField(
            options,
            "learner_v3_actor_evidence_resource_stat_ids"
        );
        boolean any = !contractSha256.isBlank()
            || entityCount != -1
            || statusCapacity != -1
            || abilityCapacity != -1
            || resourceStatIds != null;
        boolean all = !contractSha256.isBlank()
            && entityCount != -1
            && statusCapacity != -1
            && abilityCapacity != -1;
        if (!any) return null;
        if (!all) {
            throw new IllegalArgumentException(
                "learner-v3 actor-evidence negotiation requires contract SHA-256, "
                    + "entity count, status capacity, and ability capacity"
            );
        }
        return new NativeActorEvidenceRequest(
            contractSha256,
            checkedInt(
                entityCount,
                "learner_v3_actor_evidence_entity_count"
            ),
            checkedInt(
                statusCapacity,
                "learner_v3_actor_evidence_status_capacity"
            ),
            checkedInt(
                abilityCapacity,
                "learner_v3_actor_evidence_ability_capacity"
            ),
            resourceStatIds
        );
    }
}
