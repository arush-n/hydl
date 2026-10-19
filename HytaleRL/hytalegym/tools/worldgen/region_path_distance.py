#!/usr/bin/env python3
"""Build the path-distance table that the pursuit path progress reads.

The consumer is arena/training/skills/path_distance.py. It loads an NPZ with
node_position [N, 3], landmark_position [L, 3], node_to_landmark [N, L] uint8
(half-blocks, 255 unreachable) and a metadata JSON string, and refuses a table
whose region_semantic_sha256 is not the loaded Region's digest.

Run it with the repo venv from any directory, for example:
  /Users/arush/hydl/.venv/bin/python /Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/region_path_distance.py --seed 1618574389 --server-version 0.5.7 --output <new file>.npz

--server-version picks the pilot library: 0.7.2 (the default, gen 7) reads
artifacts/worldgen/region-library-pilot-v2-072 and 0.5.7 reads
artifacts/worldgen/region-library-pilot-v2. --library names another root that
holds manifest.json and traversal-v2/traversal-manifest.json. --dry-run loads
the Region and counts nodes and landmarks, then stops before the search.

Method. The Region and its traversal graph come from the library's own loaders
(RegionArtifactLibrary.load_seed_selection audits the NPZ and its semantic
digest; RegionTraversalGraphLibrary.load_for_region_entries). The runtime pads
the same arrays into its atlas (jax/world/region/traversal.py:46) and contains
no shortest-path search, so distances are one Dijkstra per landmark on the
reversed directed graph. Active edges are those with edge_mask set, with
edge_destination and edge_cost (blocks) as stored.

Landmarks: one per 4-block X/Z cell and surface layer. A cell's nodes are sorted
by foot height, and a new layer starts where the gap to the next node exceeds
3.0 blocks. Each layer's landmark is its node nearest the cell centre (ties go to
the lower node index).

Table: node_to_landmark[n, j] = round(d / 0.5), where d is the directed shortest
path in blocks from node n to landmark j. 255 means unreachable, and it also
covers any path longer than 127 blocks (254 is the largest stored distance). The
consumer reads 255 as unreachable, so a move from inside that band to outside it
pays -clip when both points are on the graph and the body gap exceeds 4 blocks
(arena/training/contracts/pursuit.py:625-642). Saturating at 254 would avoid the
penalty but flatten progress past the band.

Inferred, not confirmed: the 4-block cell (the consumer comment's "~2.8 blocks"
snap is a 4-block cell's half-diagonal) and the 3.0-block layer gap (the
consumer's _MAXIMUM_SUPPORT_GAP). The consumer fixes neither.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from hytalegym.worldgen.region import RegionArtifactLibrary, RegionTraversalGraphLibrary

ARTIFACT_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "worldgen"
LIBRARY_DIRECTORIES = {
    "0.7.2": "region-library-pilot-v2-072",
    "0.5.7": "region-library-pilot-v2",
}
TRAVERSAL_MANIFEST = Path("traversal-v2") / "traversal-manifest.json"
DISTANCE_UNIT = 0.5
UNREACHABLE = 255
CAPACITY_BLOCKS = 127.0
CELL_BLOCKS = 4.0
LAYER_GAP_BLOCKS = 3.0
LANDMARK_RULE = (
    f"one per {CELL_BLOCKS:g}-block X/Z cell and surface layer; a cell's nodes sorted "
    f"by foot height split into layers where the gap exceeds {LAYER_GAP_BLOCKS:g} blocks; "
    "each layer's landmark is its node nearest the cell centre, ties to the lower node index"
)


def library_root(server_version: str, override: str | None) -> Path:
    if override is not None:
        return Path(override).resolve()
    return ARTIFACT_ROOT / LIBRARY_DIRECTORIES[server_version]


def landmark_nodes(node_position: np.ndarray) -> np.ndarray:
    """Node indices of the landmarks, sorted, by the rule in LANDMARK_RULE."""
    cells: dict[tuple[int, int], list[int]] = {}
    for node, (x, _, z) in enumerate(node_position.tolist()):
        key = (math.floor(x / CELL_BLOCKS), math.floor(z / CELL_BLOCKS))
        cells.setdefault(key, []).append(node)
    chosen: list[int] = []
    for (cell_x, cell_z), members in cells.items():
        members.sort(key=lambda node: (node_position[node, 1], node))
        layers = [[members[0]]]
        for below, node in zip(members, members[1:]):
            if node_position[node, 1] - node_position[below, 1] > LAYER_GAP_BLOCKS:
                layers.append([node])
            else:
                layers[-1].append(node)
        centre_x = (cell_x + 0.5) * CELL_BLOCKS
        centre_z = (cell_z + 0.5) * CELL_BLOCKS
        for layer in layers:
            chosen.append(
                min(
                    layer,
                    key=lambda node: (
                        (node_position[node, 0] - centre_x) ** 2
                        + (node_position[node, 2] - centre_z) ** 2,
                        node,
                    ),
                )
            )
    return np.asarray(sorted(chosen), dtype=np.int64)


def reverse_adjacency(graph) -> list[list[tuple[int, float]]]:
    """For each node v, the (u, cost) pairs of the active edges u -> v."""
    sources, slots = np.nonzero(graph.edge_mask)
    targets = graph.edge_destination[sources, slots]
    costs = graph.edge_cost[sources, slots]
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(graph.node_count)]
    for source, target, cost in zip(sources.tolist(), targets.tolist(), costs.tolist()):
        adjacency[target].append((source, cost))
    return adjacency


def distances_to(adjacency: list[list[tuple[int, float]]], landmark: int) -> list[float]:
    """Directed distance in blocks from every node to ``landmark``; inf beyond the capacity."""
    distance = [math.inf] * len(adjacency)
    distance[landmark] = 0.0
    frontier = [(0.0, landmark)]
    while frontier:
        here, node = heapq.heappop(frontier)
        if here > distance[node]:
            continue
        for source, cost in adjacency[node]:
            candidate = here + cost
            if candidate <= CAPACITY_BLOCKS and candidate < distance[source]:
                distance[source] = candidate
                heapq.heappush(frontier, (candidate, source))
    return distance


def write_table(path: Path, graph, metadata: dict) -> dict:
    """Search every landmark, quantize and write the NPZ atomically.

    Refuses a traversal graph whose source Region digest differs from
    ``metadata["region_semantic_sha256"]``.
    """
    if graph.source_region_semantic_sha256 != metadata["region_semantic_sha256"]:
        raise ValueError(
            f"traversal graph is for Region {graph.source_region_semantic_sha256[:12]}, "
            f"not {metadata['region_semantic_sha256'][:12]}"
        )
    landmarks = landmark_nodes(graph.node_position)
    started = time.perf_counter()
    adjacency = reverse_adjacency(graph)
    table = np.full((graph.node_count, landmarks.size), UNREACHABLE, dtype=np.uint8)
    for column, landmark in enumerate(landmarks.tolist()):
        distance = np.asarray(distances_to(adjacency, landmark), dtype=np.float64)
        reached = np.isfinite(distance)
        table[reached, column] = np.rint(distance[reached] / DISTANCE_UNIT).astype(np.uint8)
    searched = time.perf_counter() - started
    node_position = np.asarray(graph.node_position, dtype=np.float32)
    full_metadata = dict(
        metadata,
        distance_unit=DISTANCE_UNIT,
        unreachable=UNREACHABLE,
        capacity_blocks=CAPACITY_BLOCKS,
        cell_blocks=CELL_BLOCKS,
        layer_gap_blocks=LAYER_GAP_BLOCKS,
        landmark_rule=LANDMARK_RULE,
        node_count=graph.node_count,
        landmark_count=int(landmarks.size),
    )
    started = time.perf_counter()
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            node_position=node_position,
            landmark_position=node_position[landmarks],
            node_to_landmark=table,
            metadata=np.asarray(json.dumps(full_metadata, sort_keys=True)),
        )
    os.replace(temporary, path)
    return {
        "bytes": path.stat().st_size,
        "search_seconds": searched,
        "write_seconds": time.perf_counter() - started,
        "finite_fraction": float(np.mean(table != UNREACHABLE)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the path-distance table for one Region seed.")
    parser.add_argument("--seed", type=int, required=True, help="Region seed in the library")
    parser.add_argument(
        "--server-version",
        choices=sorted(LIBRARY_DIRECTORIES),
        default="0.7.2",
        help="pilot library line (default 0.7.2, gen 7)",
    )
    parser.add_argument(
        "--library", help="Region library root, overriding the --server-version directory"
    )
    parser.add_argument(
        "--output",
        help="NPZ to write (default: path_distance_<version>_seed_<seed>.npz in the working directory)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="load and count only; no search, no write"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = library_root(args.server_version, args.library)
    manifest = root / "manifest.json"
    traversal_manifest = root / TRAVERSAL_MANIFEST
    for required in (manifest, traversal_manifest):
        if not required.is_file():
            raise SystemExit(f"no Region library file at {required}; pass --library for another root")
    started = time.perf_counter()
    library = RegionArtifactLibrary.load(manifest, verify_artifacts=False)
    matches = [entry for entry in library.entries if entry.seed == args.seed]
    if not matches:
        raise SystemExit(f"seed {args.seed} is not in {root.name} ({len(library.entries)} entries)")
    entry = matches[0]
    library.load_seed_selection(entry.split, [entry.seed])
    graph, = RegionTraversalGraphLibrary.load(traversal_manifest, library).load_for_region_entries([entry])
    loaded = time.perf_counter() - started
    nodes = graph.node_count
    landmarks = landmark_nodes(graph.node_position)
    print(
        f"{root.name} seed {entry.seed} ({entry.split}): nodes {nodes}, active edges "
        f"{int(np.count_nonzero(graph.edge_mask))}, landmarks {landmarks.size}, table "
        f"{nodes} x {landmarks.size} uint8 = {nodes * landmarks.size / 2**20:.1f} MiB raw; "
        f"loaded in {loaded:.2f} s"
    )
    if args.dry_run:
        print("dry run: no search, nothing written")
        return 0
    output = (
        Path(args.output)
        if args.output
        else Path.cwd() / f"path_distance_{args.server_version}_seed_{args.seed}.npz"
    )
    stats = write_table(
        output,
        graph,
        {
            "region_semantic_sha256": entry.semantic_sha256.lower(),
            "server_version": args.server_version,
            "library": root.name,
            "library_semantic_sha256": library.semantic_sha256,
            "traversal_graph_semantic_sha256": graph.metadata["graph_semantic_sha256"],
            "seed": entry.seed,
            "split": entry.split,
        },
    )
    print(
        f"wrote {output}: {stats['bytes'] / 2**20:.2f} MiB; search {stats['search_seconds']:.1f} s; "
        f"write {stats['write_seconds']:.1f} s; finite entries {stats['finite_fraction']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
