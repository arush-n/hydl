package com.hytalerlbridge.action.group;

import com.hytalerlbridge.action.AgentAction;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import org.msgpack.value.Value;

/** Canonically ordered actions applied during one shared native step. */
public record NativeGroupAction(List<ActorAction> actors) {

    public NativeGroupAction {
        if (actors == null) {
            throw new IllegalArgumentException("group action actors are required");
        }
        if (actors.size() > NativeGroupActionContract.actorCapacity()) {
            throw new IllegalArgumentException(
                "group action exceeds the negotiated actor capacity"
            );
        }
        ArrayList<ActorAction> ordered = new ArrayList<>(actors);
        HashSet<Integer> seen = new HashSet<>();
        for (ActorAction actor : ordered) {
            if (actor == null || !seen.add(actor.entityId())) {
                throw new IllegalArgumentException(
                    "group actions require unique entity_id values"
                );
            }
            String evidenceRejectReason =
                NativeGroupActionContract.worldVerbEvidenceRejectReason(
                    actor.action()
                );
            if (!evidenceRejectReason.isEmpty()) {
                throw new IllegalArgumentException(evidenceRejectReason);
            }
        }
        ordered.sort(Comparator.comparingInt(ActorAction::entityId));
        actors = List.copyOf(ordered);
    }

    public static NativeGroupAction legacy(AgentAction action) {
        return new NativeGroupAction(List.of(new ActorAction(0, action)));
    }

    public static NativeGroupAction autonomous() {
        return new NativeGroupAction(List.of());
    }

    public static NativeGroupAction fromValue(Value value) {
        if (value == null || !value.isArrayValue()) {
            throw new IllegalArgumentException(
                "step actions must be a MessagePack array"
            );
        }
        List<Value> values = value.asArrayValue().list();
        ArrayList<ActorAction> actors = new ArrayList<>(values.size());
        for (Value actor : values) {
            actors.add(ActorAction.fromValue(actor));
        }
        return new NativeGroupAction(actors);
    }

    public AgentAction actionForEntity(int entityId) {
        for (ActorAction actor : actors) {
            if (actor.entityId() == entityId) return actor.action();
        }
        return null;
    }

    public AgentAction primaryAction() {
        AgentAction primary = actionForEntity(0);
        return primary == null ? AgentAction.noop() : primary;
    }

    public NativeGroupAction continuation() {
        return new NativeGroupAction(
            actors.stream()
                .map(actor -> new ActorAction(
                    actor.entityId(),
                    actor.action().continuation()
                ))
                .toList()
        );
    }
}
