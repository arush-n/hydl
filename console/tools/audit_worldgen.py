"""Audit WorldGen Studio previews across held-out seeds.

This is deliberately stronger than counting unique digests.  It measures the
height field, semantic surface, structure layouts, content presence, spawn
safety and the preview's own plausibility admission gate.

Example::

    python -m console.tools.audit_worldgen --seeds 64 \
      --output experimental/worldgen-v2/console-studio/audits/default-64-seed.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from console.core.worlds import worldgen


def _flat(rows):
    return [value for row in rows for value in row]


def _summary(values: list[float | int]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)

    def at(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "minimum": round(ordered[0], 6),
        "p10": round(at(0.10), 6),
        "median": round(at(0.50), 6),
        "p90": round(at(0.90), 6),
        "maximum": round(ordered[-1], 6),
        "mean": round(mean(ordered), 6),
    }


def audit(*, start_seed: int = 0, seeds: int = 64,
          config: dict[str, Any] | None = None) -> dict[str, Any]:
    if seeds < 2:
        raise ValueError("seeds must be at least 2")
    normalized = worldgen.normalize(config)
    seed_values = list(range(start_seed, start_seed + seeds))
    previews = [worldgen.preview({**normalized, "seed": seed}) for seed in seed_values]
    heights = [_flat(item["grid"]["heights"]) for item in previews]
    biomes = [_flat(item["grid"]["biomes"]) for item in previews]

    height_mae = []
    biome_disagreement = []
    for left, right in itertools.combinations(range(seeds), 2):
        height_mae.append(sum(abs(a - b) for a, b in zip(heights[left], heights[right]))
                          / len(heights[left]))
        biome_disagreement.append(sum(a != b for a, b in zip(biomes[left], biomes[right]))
                                  / len(biomes[left]))

    layouts = {
        tuple((site["kind"], round(site["x"], 3), round(site["z"], 3))
              for site in item["structures"])
        for item in previews
    }
    warning_counts = Counter(warning for item in previews for warning in item["warnings"])
    ready = sum(not item["warnings"] for item in previews)
    source = Path(worldgen.__file__).resolve()
    workspace = Path(__file__).resolve().parents[2]

    measurements = {
        "unique_digests": len({item["digest"] for item in previews}),
        "unique_height_fields": len({tuple(values) for values in heights}),
        "unique_semantic_surfaces": len({tuple(values) for values in biomes}),
        "unique_structure_layouts": len(layouts),
        "minimum_pairwise_height_mae_blocks": round(min(height_mae), 6),
        "mean_pairwise_height_mae_blocks": round(mean(height_mae), 6),
        "minimum_pairwise_biome_disagreement": round(min(biome_disagreement), 6),
        "mean_pairwise_biome_disagreement": round(mean(biome_disagreement), 6),
        "safe_spawns": sum(item["spawn"]["safe"] for item in previews),
        "seeds_with_trees": sum(item["metrics"]["tree_count"] > 0 for item in previews),
        "seeds_with_structures": sum(item["metrics"]["structure_count"] > 0 for item in previews),
        "preview_ready": ready,
        "preview_ready_fraction": round(ready / seeds, 6),
        "warning_counts": dict(sorted(warning_counts.items())),
        "distributions": {
            key: _summary([item["metrics"][key] for item in previews])
            for key in (
                "relief", "water_fraction", "traversable_fraction",
                "maximum_slope", "tree_count", "structure_count",
            )
        },
    }
    gates = {
        "all_content_digests_unique": measurements["unique_digests"] == seeds,
        "all_height_fields_unique": measurements["unique_height_fields"] == seeds,
        "all_semantic_surfaces_unique": measurements["unique_semantic_surfaces"] == seeds,
        "all_structure_layouts_unique": measurements["unique_structure_layouts"] == seeds,
        "minimum_height_mae_above_5_blocks": measurements["minimum_pairwise_height_mae_blocks"] > 5,
        "minimum_biome_disagreement_above_45_percent": measurements["minimum_pairwise_biome_disagreement"] > 0.45,
        "at_least_95_percent_preview_ready": measurements["preview_ready_fraction"] >= 0.95,
        "all_spawns_safe": measurements["safe_spawns"] == seeds,
        "all_seeds_have_trees": measurements["seeds_with_trees"] == seeds,
        "all_seeds_have_structures": measurements["seeds_with_structures"] == seeds,
        "default_water_is_bounded": (
            measurements["distributions"]["water_fraction"]["minimum"] > 0
            and measurements["distributions"]["water_fraction"]["maximum"] < 0.20
        ),
    }
    return {
        "schema": "hytalerl_worldgen_v2_console_seed_audit_v1",
        "version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "source": {
            "path": source.relative_to(workspace).as_posix(),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "preview_schema": worldgen.PREVIEW_SCHEMA,
            "preview_resolution": worldgen.PREVIEW_RESOLUTION,
            "traversable_slope": worldgen.TRAVERSABLE_SLOPE,
        },
        "seed_range": {"start": start_seed, "stop_exclusive": start_seed + seeds, "count": seeds},
        "config_without_seed": {key: value for key, value in normalized.items() if key != "seed"},
        "measurements": measurements,
        "gates": gates,
        "interpretation": (
            "Passing establishes deterministic, spatially distinct and preview-plausible "
            "designs. It does not establish native Hytale fidelity or JAX capture exactness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(start_seed=args.start_seed, seeds=args.seeds)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
        print(args.output)
    print(encoded, end="")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
