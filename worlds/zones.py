"""Split the captured regions into distinct arenas that are provably navigable.

A region is not one environment. Region 1618574389 holds 39,836 standable
positions spread over 121 blocks of x, 121 of z and 160 of y -- mountainside,
overhang, pit and plain, all under one seed. Spawning at a random node samples
that mixture, which is why "train on Region" has never meant a specific place.

This module cuts a region into **zones**: maximal sets of nodes that can all
reach each other. The cut is the strongly-connected components of the traversal
graph.

## Why strong connectivity is the right cut

The traversal edges are **directed**, and measurably asymmetric. Classifying the
241,159 legal edges of one region by their height change:

    kind 1  level    145,542   dy  +0.00
    kind 2  climb     41,203   dy  +0.05 .. +1.28   (maximum_climb_height 1.3)
    kind 3  drop      54,414   dy  -0.05 .. -6.72   (maximum_drop_height 3.0)

An actor can fall 6.72 blocks in one edge and can only climb 1.3, so a drop is
usually a one-way door. Treating the graph as undirected hides that: the same
region gives 2,725 weakly-connected components but **3,848 strongly-connected**
ones, and the largest falls from 29,288 nodes (73.5%) to 14,236 (35.7%).

That difference is exactly the failure mode to avoid -- a pit an agent drops
into and cannot leave, or an opponent parked on a ledge it can never be reached
on. Inside a strongly-connected component, every node reaches every other node,
so **there is no gap that cannot be navigated** by construction rather than by
inspection.

## What is spawnable

`jax_arsenal_world_benchmark.py:1516-1530` restricts spawn candidates to nodes
inside the region **core** -- `CORE_CHUNKS_PER_AXIS = 3`, so 96x96 blocks of the
160x160 capture, the rest being a one-chunk halo kept for context. A zone body
outside the core is real terrain an agent can walk through but can never be
spawned into, so zones are ranked by their *core* population.

## What the blocks are

Ground material comes from the `block-semantics-v1` sidecar, read at the cell
**below** each node's feet. Air is not sampled: the cell under a standable node
is solid in 100.0% of 6,000 sampled nodes, which is what makes the ground read
meaningful and is also how the index math below was validated. The region
snapshot independently declares `"section_index_order": "y_z_x"`, matching.

## What this does not claim

Nothing here is a Hytale biome. The captures record seed, world id and chunk
API and **no biome or zone identity** (`worldgen_version` is `"0.0.0"`), and no
biome definition asset exists in this tree. The server does carry the generator
vocabulary -- `server/worldgen/zone/Zone.java` is a record of
`(id, name, discoveryConfig, caveGenerator, biomePatternGenerator,
uniquePrefabContainer)` -- and Hytale's announced World Generation V2 is built
on the same shape, but neither supplies a label for a position in these
captures. Every label below is derived from measured geometry and says so.

    python -m worlds.zones                 # catalogue the library
    python -m worlds.zones 1618574389      # one region in detail
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from worlds.terrain import LIBRARY, TRAVERSAL, classify, regions

#: `CORE_CHUNKS_PER_AXIS * CHUNK_SIZE` from
#: `hytalegym/worldgen/region/contract.py:34`. The capture is 5 chunks per axis;
#: the outer ring is halo and is never a spawn candidate.
CORE_BLOCKS_PER_AXIS = 96
CHUNK_SIZE = 32
#: Capture chunks per axis (3 core + 1 halo each side), so the cell arrays are
#: 5x5 chunk columns and their origin is one chunk *before* `core_min_chunk_xz`.
CAPTURE_CHUNKS_PER_AXIS = 5

SEMANTICS = LIBRARY / "block-semantics-v1"

#: Below this a component is too small to fight in -- the opponent spawns one
#: edge away and the episode runs hundreds of ticks.
MINIMUM_NODES = 500

__all__ = [
    "Zone", "zones", "library", "ground_codes", "seed_for", "selection_for",
    "select", "CORE_BLOCKS_PER_AXIS", "SPLIT_METHOD",
]


@dataclass(frozen=True)
class Zone:
    """One mutually-reachable arena inside a captured region."""

    seed: int
    component: int
    #: Node indices into the region's traversal graph. Every one of these
    #: reaches every other; that is what makes the zone an arena.
    nodes: np.ndarray
    position: np.ndarray            # (N, 3) world space
    #: Subset of `nodes` that the benchmark would accept as a spawn.
    core_nodes: np.ndarray
    terrain: dict[str, int]
    #: Palette codes of the solid cell under each node, air excluded.
    ground: np.ndarray              # (N,) uint16

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def spawnable(self) -> int:
        return len(self.core_nodes)

    @property
    def span(self) -> np.ndarray:
        """World-space extent (x, y, z) the zone covers."""
        return self.position.max(axis=0) - self.position.min(axis=0)

    @property
    def materials(self) -> int:
        """Distinct ground block types under the zone."""
        return int(len(np.unique(self.ground[self.ground != 0])))

    def dominant(self) -> str:
        """The terrain label holding the most nodes."""
        return max(self.terrain, key=lambda k: self.terrain[k])

    def character(self) -> str:
        """A label derived from the measured mix, not from any Hytale biome.

        Ordered most-specific first: a zone that is mostly enclosed is a cave
        regardless of how broken its floor is.
        """
        total = max(self.size, 1)
        share = {k: v / total for k, v in self.terrain.items()}
        vertical = float(self.span[1])
        if share.get("enclosed", 0.0) >= 0.30:
            return "cavern"
        if share.get("steep", 0.0) >= 0.50:
            return "cliffs"
        if share.get("confined", 0.0) >= 0.40:
            return "warren"
        if share.get("open_flat", 0.0) >= 0.40:
            return "plain"
        if share.get("sheltered", 0.0) >= 0.30:
            return "canopy"
        if vertical >= 24.0:
            return "slope"
        return "rough"


def _semantic_sha(seed: int) -> str:
    manifest = json.loads((LIBRARY / "manifest.json").read_text())
    match = next((e for e in manifest["entries"] if int(e["seed"]) == seed), None)
    if match is None:
        raise ValueError(f"no region with seed {seed}")
    return str(match["semantic_sha256"])


def _graph(seed: int):
    manifest = json.loads((TRAVERSAL / "traversal-manifest.json").read_text())
    sha = _semantic_sha(seed)
    entry = next((e for e in manifest["entries"]
                  if e["source_region_semantic_sha256"] == sha), None)
    if entry is None:
        raise ValueError(f"no traversal graph for seed {seed}")
    return np.load(TRAVERSAL / entry["path"])


def ground_material_available() -> bool:
    """Whether this library ships the block-semantic sidecars zones need.

    ``region-library-v3-288`` does not: its Regions and traversal graphs are
    complete, but the live server no longer regenerates the terrain they were
    captured from, so the sidecars cannot be produced. Terrain *classification*
    still works -- it reads the traversal graph -- only ground *material* is
    unavailable. See ``worlds.region.block_semantics_available``.
    """

    return SEMANTICS.is_dir()


@lru_cache(maxsize=32)
def _ground_archive(seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = SEMANTICS / "sidecars" / f"{_semantic_sha(seed)}.block-semantics.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"no block-semantic sidecar for seed {seed} in {SEMANTICS}. "
            "Ground material needs sidecars this library does not ship; "
            "terrain classification and selection keys do not. Guard the call "
            "with worlds.zones.ground_material_available()."
        )
    with np.load(path, allow_pickle=True) as archive:
        code = np.asarray(archive["cell_code"])
        origin = np.asarray(archive["core_min_chunk_xz"], dtype=np.int64) - 1
    return code, origin


def ground_codes(seed: int, position: np.ndarray) -> np.ndarray:
    """Palette code of the solid cell beneath each position.

    The layout is `column = cx * 5 + cz`, `cell = y * 1024 + z * 32 + x`, with
    the column origin one chunk before `core_min_chunk_xz` because the capture
    carries a halo. Derived by testing every candidate layout against the
    traversal graph -- the winner puts a solid cell under 100.0% of standable
    nodes and the rest sit near chance -- and confirmed by the snapshot's own
    `"section_index_order": "y_z_x"`.

    Returns 0 where the position falls outside the captured volume.
    """

    code, origin = _ground_archive(seed)

    block = np.floor(position - np.array([0.0, 0.5, 0.0])).astype(np.int64)
    x, y, z = block[:, 0], block[:, 1], block[:, 2]
    cx = np.floor_divide(x, CHUNK_SIZE) - origin[0]
    cz = np.floor_divide(z, CHUNK_SIZE) - origin[1]
    inside = (
        (cx >= 0) & (cx < CAPTURE_CHUNKS_PER_AXIS)
        & (cz >= 0) & (cz < CAPTURE_CHUNKS_PER_AXIS)
        & (y >= 0) & (y < code.shape[1] * CHUNK_SIZE)
    )
    column = np.clip(cx * CAPTURE_CHUNKS_PER_AXIS + cz, 0, code.shape[0] - 1)
    section = np.clip(np.floor_divide(y, CHUNK_SIZE), 0, code.shape[1] - 1)
    cell = (np.mod(y, CHUNK_SIZE) * 1024
            + np.mod(z, CHUNK_SIZE) * CHUNK_SIZE
            + np.mod(x, CHUNK_SIZE))
    return np.where(inside, code[column, section, cell], 0).astype(np.uint16)


@lru_cache(maxsize=32)
def zones(seed: int, minimum_nodes: int = MINIMUM_NODES) -> tuple[Zone, ...]:
    """Cut one region into mutually-reachable arenas, largest first."""

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    archive = _graph(seed)
    position = np.asarray(archive["node_position"])
    mask = np.asarray(archive["edge_mask"])
    destination = np.asarray(archive["edge_destination"])
    count = len(position)

    flat = mask.ravel()
    source = np.repeat(np.arange(count), mask.shape[1])[flat]
    target = destination.ravel()[flat]
    graph = csr_matrix(
        (np.ones(source.size, dtype=np.int8), (source, target)),
        shape=(count, count),
    )
    _, label = connected_components(graph, directed=True, connection="strong")

    terrain = classify(seed).label
    # Ground *material* is the only thing here needing block-semantic sidecars,
    # and `region-library-v3-288` ships none -- they cannot be regenerated,
    # because the live server no longer produces the terrain they were captured
    # from. Everything a zone is actually selected on -- the traversal graph,
    # connected components, core nodes, terrain classification -- reads the
    # graph and works fine without them.
    #
    # Unguarded, this raised FileNotFoundError out of `library()`, so
    # `runner.zone_options()` swallowed it and returned zero zones: the picker
    # silently offered nothing and `zone="rotate"` could not resolve. Zeros mean
    # "material unknown"; `_ground_mix` already renders that as "-" and
    # `Zone.materials` counts non-zero codes, so both degrade rather than lie.
    ground = (
        ground_codes(seed, position)
        if ground_material_available()
        else np.zeros(len(position), dtype=np.int64)
    )

    origin = np.asarray(archive["core_min_chunk_xz"], dtype=np.int64) * CHUNK_SIZE
    block = np.floor(position).astype(np.int64)
    in_core = np.all(
        (block[:, (0, 2)] >= origin)
        & (block[:, (0, 2)] < origin + CORE_BLOCKS_PER_AXIS),
        axis=1,
    ) & mask.any(axis=1)

    sizes = np.bincount(label)
    found = []
    for component in np.flatnonzero(sizes >= minimum_nodes):
        nodes = np.flatnonzero(label == component)
        names, totals = np.unique(terrain[nodes], return_counts=True)
        found.append(Zone(
            seed=seed,
            component=int(component),
            nodes=nodes,
            position=position[nodes],
            core_nodes=nodes[in_core[nodes]],
            terrain={str(n): int(c) for n, c in zip(names, totals)},
            ground=ground[nodes],
        ))
    found.sort(key=lambda z: -z.spawnable)
    return tuple(found)


@lru_cache(maxsize=4)
def _library(minimum_nodes: int) -> tuple[Zone, ...]:
    """Scan every captured region once; the artifacts behind it are immutable.

    Worth caching rather than leaving to `zones()`: this walks every seed in
    `regions()` while that cache holds only 32 entries, so a library-wide scan
    evicts itself and each call pays the full cost again. It used to be cheap
    only because it raised on the first missing block-semantic sidecar.
    """

    found: list[Zone] = []
    for seed in regions():
        found.extend(zones(seed, minimum_nodes))
    return tuple(found)


def library(minimum_nodes: int = MINIMUM_NODES) -> list[Zone]:
    """Every qualifying zone across all captured regions."""

    # A fresh list per call: the cache holds a tuple so a caller that sorts or
    # trims the result cannot corrupt the next caller's view.
    return list(_library(minimum_nodes))


def seed_for(zone: Zone, limit: int = 4096) -> int | None:
    """Find a `node_seed` whose spawn lands inside `zone`, or None.

    This replicates the loader's own draw
    (`jax_arsenal_world_benchmark.py:1556`)::

        candidates = node_mask & inside_core & any(edge_mask)
        first_source = np.random.default_rng(node_seed).choice(candidates)

    **It predicts the first candidate, not the final spawn.** The loader then
    calls `_first_actor_legal_region_source`, which walks a permutation of the
    rest if that node is not actor-legal, so a returned seed is a strong
    starting point that still has to be confirmed by building the scene and
    reading `agent_spawn`. `SceneConfig(world="region", node_seed=...)` does
    that; treat an unverified seed as a candidate, never as a placement.
    """

    archive = _graph(zone.seed)
    mask = np.asarray(archive["edge_mask"])
    position = np.asarray(archive["node_position"])
    origin = np.asarray(archive["core_min_chunk_xz"], dtype=np.int64) * CHUNK_SIZE
    block = np.floor(position).astype(np.int64)
    candidates = np.flatnonzero(
        np.all(
            (block[:, (0, 2)] >= origin)
            & (block[:, (0, 2)] < origin + CORE_BLOCKS_PER_AXIS),
            axis=1,
        )
        & mask.any(axis=1)
    )
    wanted = set(zone.core_nodes.tolist())
    for seed in range(limit):
        if int(np.random.default_rng(seed).choice(candidates)) in wanted:
            return seed
    return None


#: `RegionArtifactLibrary.select` ranks by a keyed digest under this method
#: name (`hytalegym/worldgen/region/library.py:183-193`).
SPLIT_METHOD = "sha256_rank_v1"


@lru_cache(maxsize=4)
def _split_candidates(split: str) -> tuple[tuple[int, str], ...]:
    """`(seed, semantic_sha256)` for one split, read from the manifest once."""

    manifest = json.loads((LIBRARY / "manifest.json").read_text())
    return tuple(
        (int(entry["seed"]), str(entry["semantic_sha256"]))
        for entry in manifest["entries"]
        if entry["split"] == split
    )


@lru_cache(maxsize=4)
def _selection_table(split: str, limit: int) -> dict[int, int]:
    """`seed -> lowest selection_key that selects it`, built in one pass.

    The ranking `min` below does not depend on which seed is being asked about,
    so the old code recomputed the identical `limit x candidates` SHA-256 sweep
    for every zone. `options()` calls this once per zone, which is why the Run
    tab's endpoint took 154s: the work is quadratic in the library for no
    reason. Computing the whole key->winner map once makes each lookup O(1) and
    returns exactly the same answer -- the first key whose winner is `seed`.
    """

    candidates = _split_candidates(split)
    table: dict[int, int] = {}
    for key in range(limit):
        winner = min(
            candidates,
            key=lambda entry: (
                hashlib.sha256(
                    f"{SPLIT_METHOD}\0{key}\0{entry[1]}".encode()
                ).digest(),
                entry[1],
            ),
        )
        table.setdefault(winner[0], key)
    return table


def selection_for(seed: int, *, split: str = "train",
                  limit: int = 4096) -> int | None:
    """Find a `selection_key` that makes `seed` the region the loader picks.

    A zone needs **two** keys, not one. `node_seed` only chooses a node inside
    whichever region is already selected; the region itself comes from
    `selection_key`, which sorts the split by
    ``sha256(f"{SPLIT_METHOD}\\0{key}\\0{semantic_sha256}")`` and takes the
    first `count` entries. With the reset pool at capacity 1 (see ISSUES #1)
    that is exactly one region.

    Discovering this the hard way is why `seed_for` alone is not enough:
    passing only a `node_seed` applies it to the *default* region, which lands
    somewhere unrelated to the zone that was asked for.

    Only `split="train"` entries are candidates, so a heldout region cannot be
    selected this way at all.
    """

    if not any(entry[0] == seed for entry in _split_candidates(split)):
        return None
    return _selection_table(split, limit).get(seed)


def selected_seeds(key: int, *, count: int, split: str = "train") -> list[int]:
    """The seeds a `selection_key` actually resolves to, without a scene build.

    `RegionArtifactLibrary.select` ranks the split by a keyed digest and takes
    the first `count`, so the whole resident set is predictable offline from
    `seed` and `semantic_sha256` alone -- no GPU, no artifact load.
    """

    import hashlib

    manifest = json.loads((LIBRARY / "manifest.json").read_text())
    candidates = [e for e in manifest["entries"] if e["split"] == split]
    ranked = sorted(
        candidates,
        key=lambda e: (
            hashlib.sha256(
                f"{SPLIT_METHOD}\0{key}\0{e['semantic_sha256']}".encode()
            ).digest(),
            e["semantic_sha256"],
        ),
    )
    return [int(e["seed"]) for e in ranked[:count]]


def selection_key_for_terrain(
    *,
    by: str = "open",
    count: int = 16,
    keys: int = 20000,
    minimum_spawnable: int = 100,
    split: str = "train",
) -> list[tuple[float, float, int]]:
    """Search `selection_key` values for a resident set with the terrain you want.

    Returns ``(mean, floor, key)`` triples, best first, where both scores are
    the `by` axis of :func:`select` measured over the `count` worlds that key
    resolves to.

    **`selection_key` is a rank, not a filter.** There is no way to compose "8
    flat and 8 varied" -- the digest order decides the whole set, so the only
    control is to search for a key whose set happens to be what you want. That
    is also why the default was never a decision: measured 2026-08-18, the key
    every pursuit run had been using (365747984) sits at the **4.2nd
    percentile** for flatness out of 20,000, with a mean `open_flat` share of
    0.280 against a library median of 0.378. The agent had been training on
    nearly the most rugged terrain available, by hash accident.

    **Rank on the floor, not just the mean.** One very flat world will carry an
    average while the learner still spends most of its episodes on cliffs: key
    10580 scores the best mean found (0.488) but contains a 0.176 world, where
    key 326 gives up a little mean (0.478) for a much better floor (0.342).
    """

    zones = library()
    axis = {
        "open": lambda z: z.terrain.get("open_flat", 0) / max(z.size, 1),
        "enclosed": lambda z: z.terrain.get("enclosed", 0) / max(z.size, 1),
    }
    if by not in axis:
        raise ValueError(f"unknown axis {by!r}; have {sorted(axis)}")
    score = axis[by]

    # One representative zone per seed -- its largest component, which is the
    # arena an episode will almost always be placed in.
    best: dict[int, Zone] = {}
    for zone in zones:
        if zone.spawnable < minimum_spawnable:
            continue
        if zone.seed not in best or zone.size > best[zone.seed].size:
            best[zone.seed] = zone
    by_seed = {seed: score(zone) for seed, zone in best.items()}

    ranked = []
    for key in range(keys):
        values = [by_seed[s] for s in selected_seeds(key, count=count, split=split)
                  if s in by_seed]
        if len(values) < count:
            continue
        ranked.append((sum(values) / len(values), min(values), key))
    ranked.sort(reverse=True)
    return ranked


def select(found: list[Zone], *, by: str, count: int = 5,
           minimum_spawnable: int = 100) -> list[Zone]:
    """Rank zones on one measured axis, hardest first.

    The axes are the ones the captures can actually support, named after the
    sub-systems Hytale's generator organises a biome by -- terrain shape,
    material, and cave -- so a preset asks for a property rather than a label.

    * `vertical`   -- y extent, the axis that exercises climb and drop
    * `area`       -- horizontal footprint, the axis that exercises pursuit
    * `materials`  -- distinct ground block types, the axis world verbs read
    * `enclosed`   -- share of nodes under cover
    * `open`       -- share of flat nodes, for a clean control arena
    """

    usable = [z for z in found if z.spawnable >= minimum_spawnable]
    keys = {
        "vertical": lambda z: float(z.span[1]),
        "area": lambda z: float(z.span[0]) * float(z.span[2]),
        "materials": lambda z: z.materials,
        "enclosed": lambda z: z.terrain.get("enclosed", 0) / max(z.size, 1),
        "open": lambda z: z.terrain.get("open_flat", 0) / max(z.size, 1),
    }
    if by not in keys:
        raise ValueError(f"unknown axis {by!r}; have {sorted(keys)}")
    return sorted(usable, key=keys[by], reverse=True)[:count]


def spread(found: list[Zone], *, by: str, count: int = 5,
           minimum_spawnable: int = 100) -> list[Zone]:
    """`count` zones spanning one axis, not `count` copies of its extreme.

    :func:`select` returns the top of a ranking, so every zone it offers is an
    extreme: ask it for five `vertical` arenas and you get the five tallest,
    which resemble each other far more than they resemble the corpus. Training
    on those is not terrain variety, and the middle of the space -- where most
    real terrain sits -- is never offered at all.

    This walks evenly spaced ranks from the extreme to the median instead, so a
    request for five spans the axis it names. Rank 0 is still the extreme, so
    everything :func:`select` offered remains reachable.

    The median end is deliberate rather than the far tail: the bottom of the
    `vertical` ranking is a flat zone, which is already what the `open` axis
    offers, so walking the whole range would return the same arenas under two
    different names.
    """

    ranked = select(found, by=by, count=len(found),
                    minimum_spawnable=minimum_spawnable)
    if not ranked or count < 1:
        return []
    if count == 1 or len(ranked) == 1:
        return ranked[:1]
    # Extreme to median, so the set spans one half of the axis rather than
    # meeting the neighbouring axis at the far end.
    last = max(1, (len(ranked) - 1) // 2)
    step = last / (count - 1)
    # Deduped by (seed, component), not by the Zone itself: `Zone` is a
    # dataclass holding numpy arrays, so its generated `__eq__` compares arrays
    # and `in` would raise on the ambiguous truth value.
    picked: list[Zone] = []
    seen: set[tuple[int, int]] = set()
    for index in range(count):
        zone = ranked[min(int(round(index * step)), len(ranked) - 1)]
        key = (zone.seed, zone.component)
        if key not in seen:
            seen.add(key)
            picked.append(zone)
    return picked


def _ground_mix(zone: Zone, top: int = 4) -> str:
    solid = zone.ground[zone.ground != 0]
    if not len(solid):
        return "-"
    code, count = np.unique(solid, return_counts=True)
    share = count / count.sum()
    order = np.argsort(-share)[:top]
    return "  ".join(f"{int(code[i])}:{share[i]:.0%}" for i in order)


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv

    if argv:
        seed = int(argv[0])
        found = zones(seed)
        print(f"region {seed}: {len(found)} navigable zones "
              f"(>= {MINIMUM_NODES} mutually reachable nodes)\n")
        header = (f"{'zone':>5}{'nodes':>8}{'spawnable':>11}{'character':>10}"
                  f"{'materials':>11}   span (x,y,z)          terrain mix")
        print(header)
        for zone in found:
            span = zone.span
            mix = " ".join(f"{n}={c}" for n, c in
                           sorted(zone.terrain.items(), key=lambda kv: -kv[1])[:3])
            print(f"{zone.component:>5}{zone.size:>8}{zone.spawnable:>11}"
                  f"{zone.character():>10}{zone.materials:>11}"
                  f"   {span[0]:>5.0f}{span[1]:>5.0f}{span[2]:>5.0f}"
                  f"          {mix}")

        print(f"\n{'zone':>5}   dominant ground material (palette code: share)")
        for zone in found:
            print(f"{zone.component:>5}   {_ground_mix(zone)}")
        print("\nPalette codes are indices into THIS region's palette. Comparing "
              "them across regions is meaningless -- use `palette_semantic_key` "
              "from adk.environments.blocks for that.")
        return 0

    found = library()
    print(f"{len(found)} navigable zones across {len(regions())} regions "
          f"(>= {MINIMUM_NODES} mutually reachable nodes each)\n")

    by_character: dict[str, list[Zone]] = {}
    for zone in found:
        by_character.setdefault(zone.character(), []).append(zone)

    print(f"{'character':<10}{'zones':>7}{'nodes':>10}{'spawnable':>11}"
          f"{'materials':>11}{'median y span':>15}")
    for name in sorted(by_character, key=lambda n: -len(by_character[n])):
        group = by_character[name]
        print(f"{name:<10}{len(group):>7}"
              f"{sum(z.size for z in group):>10}"
              f"{sum(z.spawnable for z in group):>11}"
              f"{int(np.median([z.materials for z in group])):>11}"
              f"{np.median([z.span[1] for z in group]):>15.0f}")

    spawnable = sum(z.spawnable for z in found)
    print(f"\nnodes inside a navigable zone : {sum(z.size for z in found):,}")
    print(f"of those, spawnable (in core) : {spawnable:,}")
    print(f"zones with >= 100 spawn nodes : "
          f"{sum(1 for z in found if z.spawnable >= 100)}")
    print("\nEvery zone is a strongly-connected component, so any node in it "
          "reaches any other -- no one-way drop can strand an agent inside a "
          "zone. Labels are derived from measured geometry, not Hytale biomes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
