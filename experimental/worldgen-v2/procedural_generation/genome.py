"""Seeded world-genome sampling without named terrain presets.

The Studio's authored-family mode intentionally remains available for exact
replay of old designs.  This module owns the separate procedural mode: user
hyperparameters define a probability distribution, and a seed resolves that
distribution into coherent macro-regions plus a connected traversal spine.

The output is a portable design IR.  It is not a claim that Hytale executed the
graph; native authority still begins after V2 pack compilation and capture.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


WORLD_GENOME_SCHEMA = "hytalerl_worldgen_v2_procedural_genome_v1"
COMPONENTS = (
    "mountains",
    "hills",
    "valleys",
    "plateaus",
    "canyons",
    "pillars",
    "craters",
    "flats",
)
_WEIGHT_FIELDS = {
    "mountains": "terrain_mountain_weight",
    "hills": "terrain_hill_weight",
    "valleys": "terrain_valley_weight",
    "plateaus": "terrain_plateau_weight",
    "canyons": "terrain_canyon_weight",
    "pillars": "terrain_pillar_weight",
    "craters": "terrain_crater_weight",
    "flats": "terrain_flat_weight",
}


def _mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF


def _hash01(seed: int, x: int, z: int, salt: int = 0) -> float:
    value = (
        seed * 0x9E3779B1
        + x * 0x85EBCA77
        + z * 0xC2B2AE3D
        + salt * 0x27D4EB2F
    )
    return _mix32(value) / 4_294_967_295.0


def _normalized(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(values.get(name, 0.0))) for name in COMPONENTS)
    if total <= 1e-12:
        raise ValueError("procedural terrain requires at least one non-zero component weight")
    return {
        name: max(0.0, float(values.get(name, 0.0))) / total
        for name in COMPONENTS
    }


def _weighted_choice(weights: Mapping[str, float], value: float) -> str:
    cumulative = 0.0
    for name in COMPONENTS:
        cumulative += weights[name]
        if value <= cumulative:
            return name
    return COMPONENTS[-1]


def _region_centers(seed: int, count: int, width: float, depth: float,
                    variation: float) -> list[tuple[float, float]]:
    """Return a jittered stratified cover, avoiding random region clumps."""

    aspect = max(0.25, min(4.0, width / max(1.0, depth)))
    columns = max(1, math.ceil(math.sqrt(count * aspect)))
    rows = max(1, math.ceil(count / columns))
    cells = [(x, z) for z in range(rows) for x in range(columns)]
    cells.sort(key=lambda cell: _mix32(seed + cell[0] * 92821 + cell[1] * 68917))
    jitter = 0.08 + variation * 0.28
    result = []
    for index, (column, row) in enumerate(cells[:count]):
        u = (column + 0.5) / columns
        v = (row + 0.5) / rows
        u += (_hash01(seed, index, 0, 701) * 2.0 - 1.0) * jitter / columns
        v += (_hash01(seed, index, 1, 701) * 2.0 - 1.0) * jitter / rows
        x = (max(0.04, min(0.96, u)) - 0.5) * width
        z = (max(0.04, min(0.96, v)) - 0.5) * depth
        result.append((round(x, 6), round(z, 6)))
    return result


def _route_edges(seed: int, centers: list[tuple[float, float]],
                 route_density: float) -> list[tuple[int, int]]:
    """Build a connected MST and deterministic density-controlled shortcuts."""

    if len(centers) < 2:
        return []
    start = min(range(len(centers)), key=lambda index: math.hypot(*centers[index]))
    connected = {start}
    remaining = set(range(len(centers))) - connected
    edges: list[tuple[int, int]] = []
    while remaining:
        candidates = []
        for left in connected:
            for right in remaining:
                distance = math.dist(centers[left], centers[right])
                tie = _hash01(seed, left, right, 751) * 1e-5
                candidates.append((distance + tie, left, right))
        _, left, right = min(candidates)
        edges.append((min(left, right), max(left, right)))
        connected.add(right)
        remaining.remove(right)

    existing = set(edges)
    candidates = []
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            if (left, right) in existing:
                continue
            distance = math.dist(centers[left], centers[right])
            candidates.append((distance, _hash01(seed, left, right, 752), left, right))
    candidates.sort()
    maximum_extra = max(0, len(centers) * 2 - len(edges))
    target_extra = round(maximum_extra * max(0.0, min(1.0, route_density)))
    # Favor useful local loops; the hash only breaks similar-distance ties.
    for _, _, left, right in candidates[:target_extra]:
        edges.append((left, right))
    return edges


def build_world_genome(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one deterministic macro-world from a hyperparameter envelope."""

    seed = int(config["seed"])
    width = float(config["world_width"])
    depth = float(config["world_depth"])
    variation = max(0.0, min(1.0, float(config["world_variation"])))
    base_weights = _normalized({
        component: float(config[field])
        for component, field in _WEIGHT_FIELDS.items()
    })
    requested_count = int(config["macro_region_count"])
    count_multiplier = 1.0 + (
        _hash01(seed, 0, 0, 703) * 2.0 - 1.0
    ) * variation * 0.22
    region_count = max(1, min(16, round(requested_count * count_multiplier)))
    centers = _region_centers(seed, region_count, width, depth, variation)
    regions: list[dict[str, Any]] = []
    dominant_counts = {name: 0 for name in COMPONENTS}
    for index, (x, z) in enumerate(centers):
        sampled = {
            name: base_weights[name] * (
                1.0 + (_hash01(seed, index, component_index, 711) * 2.0 - 1.0)
                * variation * 0.92
            )
            for component_index, name in enumerate(COMPONENTS)
        }
        sampled = _normalized(sampled)
        dominant = _weighted_choice(sampled, _hash01(seed, index, 0, 712))
        focus = 0.18 + variation * 0.58
        sampled = _normalized({
            name: weight * (1.0 - focus) + (focus if name == dominant else 0.0)
            for name, weight in sampled.items()
        })
        dominant_counts[dominant] += 1
        regions.append({
            "id": f"region-{index:02d}",
            "center": [x, z],
            "dominant_component": dominant,
            "component_weights": {
                name: round(sampled[name], 6) for name in COMPONENTS
            },
            "relief_multiplier": round(
                1.0 + (_hash01(seed, index, 1, 713) * 2.0 - 1.0)
                * variation * 0.32,
                6,
            ),
            "scale_multiplier": round(
                1.0 + (_hash01(seed, index, 2, 713) * 2.0 - 1.0)
                * variation * 0.28,
                6,
            ),
            "roughness_multiplier": round(
                1.0 + (_hash01(seed, index, 3, 713) * 2.0 - 1.0)
                * variation * 0.36,
                6,
            ),
        })

    edge_indexes = _route_edges(seed, centers, float(config["route_density"]))
    nodes = [
        {"id": region["id"], "x": region["center"][0], "z": region["center"][1]}
        for region in regions
    ]
    edges = [
        {
            "id": f"route-{index:02d}",
            "from": regions[left]["id"],
            "to": regions[right]["id"],
            "start": list(regions[left]["center"]),
            "end": list(regions[right]["center"]),
            "width": round(float(config["route_width"]), 3),
            "length": round(math.dist(centers[left], centers[right]), 3),
        }
        for index, (left, right) in enumerate(edge_indexes)
    ]
    genome: dict[str, Any] = {
        "schema": WORLD_GENOME_SCHEMA,
        "version": 1,
        "seed": seed,
        "source": "hyperparameter_distribution",
        "named_preset_selected": False,
        "world_variation": variation,
        "region_blend": float(config["region_blend"]),
        "base_component_weights": {
            name: round(base_weights[name], 6) for name in COMPONENTS
        },
        "regions": regions,
        "dominant_component_counts": {
            name: count for name, count in dominant_counts.items() if count
        },
        "route_graph": {
            "connected": len(nodes) <= 1 or len(edges) >= len(nodes) - 1,
            "nodes": nodes,
            "edges": edges,
            "total_length": round(sum(edge["length"] for edge in edges), 3),
        },
        "digest": "",
    }
    stable = dict(genome)
    stable["digest"] = ""
    genome["digest"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return genome


def blend_at(genome: Mapping[str, Any], x: float, z: float) -> tuple[
    tuple[float, ...], float, float, float
]:
    """Blend region component weights and multipliers at one world coordinate."""

    regions = genome["regions"]
    if len(regions) == 1:
        region = regions[0]
        return (
            tuple(float(region["component_weights"][name]) for name in COMPONENTS),
            float(region["relief_multiplier"]),
            float(region["scale_multiplier"]),
            float(region["roughness_multiplier"]),
        )
    blend = max(0.0, min(1.0, float(genome["region_blend"])))
    exponent = 0.72 + (1.0 - blend) * 2.35
    scored = []
    for region in regions:
        cx, cz = region["center"]
        distance_sq = (x - float(cx)) ** 2 + (z - float(cz)) ** 2
        scored.append((1.0 / (1.0 + distance_sq) ** exponent, region))
    total = sum(score for score, _ in scored)
    weights = []
    for name in COMPONENTS:
        weights.append(sum(
            score * float(region["component_weights"][name])
            for score, region in scored
        ) / total)
    relief = sum(
        score * float(region["relief_multiplier"]) for score, region in scored
    ) / total
    scale = sum(
        score * float(region["scale_multiplier"]) for score, region in scored
    ) / total
    roughness = sum(
        score * float(region["roughness_multiplier"]) for score, region in scored
    ) / total
    return tuple(weights), relief, scale, roughness


def _point_segment_distance_sq(x: float, z: float, start: list[float],
                               end: list[float]) -> float:
    dx, dz = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
    length_sq = dx * dx + dz * dz
    if length_sq <= 1e-12:
        return (x - float(start[0])) ** 2 + (z - float(start[1])) ** 2
    amount = max(0.0, min(1.0, (
        (x - float(start[0])) * dx + (z - float(start[1])) * dz
    ) / length_sq))
    px = float(start[0]) + dx * amount
    pz = float(start[1]) + dz * amount
    return (x - px) ** 2 + (z - pz) ** 2


def route_influence(genome: Mapping[str, Any], x: float, z: float) -> float:
    """Return smooth [0,1] proximity to the connected traversal spine."""

    influence = 0.0
    for edge in genome["route_graph"]["edges"]:
        width = max(0.5, float(edge["width"]))
        distance = math.sqrt(_point_segment_distance_sq(x, z, edge["start"], edge["end"]))
        amount = max(0.0, 1.0 - distance / width)
        amount = amount * amount * (3.0 - 2.0 * amount)
        influence = max(influence, amount)
    return influence
