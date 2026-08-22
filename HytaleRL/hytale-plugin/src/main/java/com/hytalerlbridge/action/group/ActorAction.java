package com.hytalerlbridge.action.group;

import com.hytalerlbridge.action.AgentAction;
import java.util.Map;
import org.msgpack.value.Value;

/** One reset-pinned native entity and its action for the shared step. */
public record ActorAction(int entityId, AgentAction action) {

    public ActorAction {
        if (entityId < 0 || entityId >= NativeGroupActionContract.actorCapacity()) {
            throw new IllegalArgumentException(
                "group action entity_id must be in [0, "
                    + (NativeGroupActionContract.actorCapacity() - 1) + "]"
            );
        }
        if (action == null) {
            throw new IllegalArgumentException("group action payload is required");
        }
    }

    public static ActorAction fromValue(Value value) {
        if (value == null || !value.isMapValue()) {
            throw new IllegalArgumentException(
                "each group action must be a MessagePack map"
            );
        }
        Map<Value, Value> map = value.asMapValue().map();
        Value entity = field(map, "entity_id");
        if (entity == null || !entity.isIntegerValue()) {
            throw new IllegalArgumentException(
                "group action entity_id must be an integer"
            );
        }
        long decoded = entity.asIntegerValue().asLong();
        if (decoded < Integer.MIN_VALUE || decoded > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                "group action entity_id is outside int32"
            );
        }
        Value action = field(map, "action");
        if (action == null || !action.isMapValue()) {
            throw new IllegalArgumentException(
                "group action payload must be a MessagePack map"
            );
        }
        return new ActorAction(
            (int) decoded,
            AgentAction.fromMap(action.asMapValue().map())
        );
    }

    private static Value field(Map<Value, Value> map, String name) {
        for (Map.Entry<Value, Value> entry : map.entrySet()) {
            Value key = entry.getKey();
            if (
                key.isStringValue()
                    && key.asStringValue().asString().equals(name)
            ) {
                return entry.getValue();
            }
        }
        return null;
    }
}

