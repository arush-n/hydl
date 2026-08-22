"""Classify terrain at the places an agent can actually stand.

The first attempt at this classified whole regions and reported all 20 as
"mountainous" — useless, because a region is 160x320x160 blocks and the label
averaged a mountain, a lake and a plain into one word. Terrain variety is
*within* a region, not between regions.

The right granularity is the traversal graph: `node_position` gives 39,836
world-space positions per region, each one somewhere the native
`MotionControllerWalk` proved an actor can stand, with `node_clearance`
(headroom) and 12 directional edges. Classifying those answers the question that
matters — "where can I put an agent so it experiences X" — rather than "what is
this region on average".

**Water is not reachable through this graph.** The traversal metadata states
`fluid_policy: exclude_swept_actor_aabb_with_one_cell_horizontal_halo` and
`fluid_excluded_candidate_count: 37330`. Fluid positions are deliberately
excluded, so no legal node is submerged. Water exists in the captures (all 20
regions, columns 15-32 cells deep) but the navigation graph will not route an
agent into it. Drowning therefore needs a spawn placed outside the graph, not a
different `node_seed`.

    python -m worlds.terrain
    python -m worlds.terrain 1618574389   # one region in detail
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
#: 288 captured Regions, 256 of them train split -- sixteen times the pilot
#: library that preceded it. Both LIBRARY constants must move together;
#: `arena/tests/test_region_library_pin.py` fails if they drift apart.
LIBRARY = _ROOT / "HytaleRL" / "artifacts" / "worldgen" / "region-library-v3-288"
#: v3 keeps its traversal sidecar at the library root; the pilot libraries
#: nested it under `traversal-v2/`. Resolve rather than pin so either layout
#: works and a missing sidecar fails loudly instead of silently degrading to a
#: surrogate graph.
TRAVERSAL = (
    LIBRARY
    if (LIBRARY / "traversal-manifest.json").is_file()
    else LIBRARY / "traversal-v2"
)

#: `node_clearance` is headroom in blocks. The capture uses a large sentinel
#: where nothing blocks above, so "open sky" is a threshold rather than a flag.
OPEN_SKY = 32.0
#: Below an actor's own height plus a margin, the node is under something.
ENCLOSED = 4.0

__all__ = ["TerrainProfile", "classify", "regions", "LIBRARY", "TRAVERSAL"]


@dataclass(frozen=True)
class TerrainProfile:
    """Per-node terrain labels for one region, as index arrays into the graph."""

    seed: int
    position: np.ndarray            # (N, 3) world space
    clearance: np.ndarray           # (N,)
    degree: np.ndarray              # (N,) legal outgoing edges, 0..12
    relief: np.ndarray              # (N,) height spread among reachable neighbours
    label: np.ndarray               # (N,) string labels

    def where(self, label: str) -> np.ndarray:
        return np.flatnonzero(self.label == label)

    def counts(self) -> dict[str, int]:
        names, totals = np.unique(self.label, return_counts=True)
        return {str(n): int(c) for n, c in zip(names, totals)}


def _graph_path(seed: int) -> Path:
    manifest = json.loads((TRAVERSAL / "traversal-manifest.json").read_text())
    regions_manifest = json.loads((LIBRARY / "manifest.json").read_text())
    match = next((e for e in regions_manifest["entries"]
                  if int(e["seed"]) == seed), None)
    if match is None:
        raise ValueError(f"no region with seed {seed}")
    entry = next((e for e in manifest["entries"]
                  if e["source_region_semantic_sha256"]
                  == match["semantic_sha256"]), None)
    if entry is None:
        raise ValueError(f"no traversal graph for seed {seed}")
    return TRAVERSAL / entry["path"]


@lru_cache(maxsize=32)
def classify(seed: int) -> TerrainProfile:
    """Label every traversal node in one region by its local terrain."""

    archive = np.load(_graph_path(seed))
    position = np.asarray(archive["node_position"])
    clearance = np.asarray(archive["node_clearance"])
    mask = np.asarray(archive["edge_mask"])
    destination = np.asarray(archive["edge_destination"])
    degree = mask.sum(axis=1).astype(np.int32)

    # Height spread among reachable neighbours. This is the honest local slope
    # signal: a node whose neighbours are all at its own height is flat ground,
    # one with a 3-block spread is a stair or a cliff edge. Computed only over
    # legal edges, so it describes terrain the actor can actually traverse.
    height = position[:, 1]
    neighbour = np.where(mask, destination, 0)
    neighbour_height = height[neighbour]
    # Isolated nodes have no legal edge, so an all-NaN row is expected rather
    # than exceptional -- fill it with the node's own height so the reduction
    # is well defined and its relief comes out as 0.
    neighbour_height = np.where(mask, neighbour_height, height[:, None])
    high = neighbour_height.max(axis=1)
    low = neighbour_height.min(axis=1)
    relief = np.where(degree > 0, high - low, 0.0)

    label = np.full(len(position), "open_flat", dtype=object)
    label[clearance < ENCLOSED] = "enclosed"
    label[(clearance >= ENCLOSED) & (clearance < OPEN_SKY)] = "sheltered"
    # Relief wins over cover: a cliff edge under an overhang is a cliff edge.
    label[relief >= 2.0] = "steep"
    label[(relief >= 0.75) & (relief < 2.0)] = "broken"
    # A node with few exits is a ledge or a dead end regardless of the rest.
    label[degree <= 2] = "confined"
    return TerrainProfile(
        seed=seed, position=position, clearance=clearance, degree=degree,
        relief=relief, label=np.asarray(label))


def regions() -> list[int]:
    manifest = json.loads((LIBRARY / "manifest.json").read_text())
    return [int(e["seed"]) for e in manifest["entries"]]


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv:
        profile = classify(int(argv[0]))
        print(f"region {profile.seed}: {len(profile.position)} traversal nodes")
        print(f"{'terrain':<12}{'nodes':>8}{'share':>9}   example world position")
        for name, count in sorted(profile.counts().items(),
                                  key=lambda kv: -kv[1]):
            where = profile.where(name)
            spot = profile.position[where[0]]
            print(f"{name:<12}{count:>8}{count/len(profile.position):>9.1%}"
                  f"   ({spot[0]:.1f}, {spot[1]:.1f}, {spot[2]:.1f})")
        print(f"\nclearance  min/median/max : {profile.clearance.min():.1f} / "
              f"{np.median(profile.clearance):.1f} / {profile.clearance.max():.1f}")
        print(f"relief     max            : {profile.relief.max():.2f}")
        print(f"degree     min/median/max : {profile.degree.min()} / "
              f"{int(np.median(profile.degree))} / {profile.degree.max()}")
        return 0

    print(f"{'seed':<12}{'nodes':>8}   terrain mix")
    totals: dict[str, int] = {}
    for seed in regions():
        profile = classify(seed)
        counts = profile.counts()
        for name, count in counts.items():
            totals[name] = totals.get(name, 0) + count
        mix = "  ".join(f"{n}={c}" for n, c in
                        sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"{seed:<12}{len(profile.position):>8}   {mix}")

    grand = sum(totals.values())
    print(f"\n=== library totals ({grand} nodes) ===")
    for name, count in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<12}{count:>9}{count / grand:>9.1%}")
    print("\nEvery one of these is a real standable position from the native "
          "traversal capture. Aim `node_seed` at a terrain type rather than "
          "accepting the default spawn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
