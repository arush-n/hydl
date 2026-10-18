package com.hytalerlbridge.worldgen;

import com.hypixel.hytale.server.core.entity.InteractionContext;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.DragPlaceBlockInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.data.Collector;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.data.CollectorTag;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.Edge;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.Node;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Breadth-first interaction-graph collector for
 * {@link NativeItemInteractionEvidence}.
 *
 * <p>Extracted verbatim. Package-private: it has no callers outside this
 * package and is not part of the evidence API.</p>
 */
final class NativeItemInteractionGraphCollector implements Collector {

    private final List<Node> nodes = new ArrayList<>();
    private final List<Edge> edges = new ArrayList<>();
    private final Set<String> expanded = new HashSet<>();
    private final Deque<Interaction> parents = new ArrayDeque<>();

    @Override
    public void start() {
        nodes.clear();
        edges.clear();
        expanded.clear();
        parents.clear();
    }

    @Override
    public void into(
        InteractionContext context,
        Interaction interaction
    ) {
        if (interaction != null) {
            parents.addLast(interaction);
        }
    }

    @Override
    public boolean collect(
        CollectorTag tag,
        InteractionContext context,
        Interaction interaction
    ) {
        Interaction parent = parents.peekLast();
        String relation = NativeItemInteractionEvidenceBuilders.tag(tag);
        edges.add(new Edge(
            parent == null ? "" : parent.getId(),
            interaction.getId(),
            relation
        ));
        boolean repeated = !expanded.add(interaction.getId());
        // 0.7.2 drag placement: a DragPlaceBlockInteraction (or subclass) forks one place node per
        // block. 0.5 had no such wrapper and set AllowDragPlacement on the place node instead.
        if (!repeated) nodes.add(NativeItemInteractionEvidenceBuilders.node(
            interaction,
            parent instanceof DragPlaceBlockInteraction && "Fork".equals(relation)
        ));
        if (
            nodes.size() > NativeItemInteractionEvidence.INTERACTION_CAPACITY
                || edges.size() > NativeItemInteractionEvidence.EDGE_CAPACITY
        ) {
            throw new IllegalStateException(
                "Native interaction graph capacity exceeded"
            );
        }
        return repeated;
    }

    @Override
    public void outof() {
        if (!parents.isEmpty()) parents.removeLast();
    }

    @Override
    public void finished() {
        nodes.sort(Comparator.comparing(Node::interactionId));
        edges.sort(
            Comparator.comparing(Edge::parentInteractionId)
                .thenComparing(Edge::childInteractionId)
                .thenComparing(Edge::relation)
        );
    }

    List<Node> nodes() {
        return List.copyOf(nodes);
    }

    List<Edge> edges() {
        return List.copyOf(edges);
    }
}
