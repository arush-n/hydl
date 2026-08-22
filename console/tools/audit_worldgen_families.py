"""Audit authored terrain families using geometry, not seed-bearing digests.

The Studio preview digest intentionally includes the normalized seed.  That is
useful for provenance, but it cannot prove that two seeds produced different
terrain.  This audit fingerprints the generated height, biome, structure and
cave payloads independently, checks their coordinates against rectangular
world bounds, and measures nearest-neighbour height-field separation.

The report remains preview evidence.  It does not replace native WorldGen V2
generation, Region capture, or JAX replay verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from console.core.worlds import worldgen


SCHEMA = "hytalerl_worldgen_v2_family_seed_audit_v1"
VERSION = 1

CAVE_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "none",
        "cave_style": "none",
        "cave_density": 0.0,
        "cave_scale": 32.0,
        "cave_verticality": 0.0,
        "cave_entrances": 0.0,
    },
    {
        "id": "sparse_tunnels",
        "cave_style": "tunnels",
        "cave_density": 0.16,
        "cave_scale": 56.0,
        "cave_verticality": 0.2,
        "cave_entrances": 0.2,
    },
    {
        "id": "dense_tunnels",
        "cave_style": "tunnels",
        "cave_density": 0.88,
        "cave_scale": 16.0,
        "cave_verticality": 0.7,
        "cave_entrances": 0.74,
    },
    {
        "id": "deeproot_caverns",
        "cave_style": "caverns",
        "cave_density": 0.7,
        "cave_scale": 30.0,
        "cave_verticality": 0.46,
        "cave_entrances": 0.28,
    },
    {
        "id": "deeproot_labyrinth",
        "cave_style": "labyrinth",
        "cave_density": 0.76,
        "cave_scale": 22.0,
        "cave_verticality": 0.62,
        "cave_entrances": 0.36,
    },
    {
        "id": "mixed",
        "cave_style": "mixed",
        "cave_density": 0.56,
        "cave_scale": 26.0,
        "cave_verticality": 0.52,
        "cave_entrances": 0.5,
    },
)


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flatten(rows: Iterable[Iterable[Any]]) -> tuple[Any, ...]:
    return tuple(value for row in rows for value in row)


def _quantized_heights(preview: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        round(value * 1000)
        for value in _flatten(preview["grid"]["heights"])
    )


def _layout_fingerprint(rows: list[dict[str, Any]]) -> str:
    return _canonical_digest([
        {
            key: row[key]
            for key in row
            if key not in {"id", "seed"}
        }
        for row in rows
    ])


def _cave_fingerprint(caves: dict[str, Any]) -> str:
    return _canonical_digest({
        "topology": caves["topology"],
        "segments": caves["segments"],
        "chambers": caves["chambers"],
        "entrances": caves["entrances"],
    })


def _bounds_ok(preview: dict[str, Any]) -> bool:
    config = preview["config"]
    half_x = config["world_width"] * 0.5
    half_z = config["world_depth"] * 0.5

    def point_ok(point: dict[str, Any] | list[float]) -> bool:
        if isinstance(point, dict):
            x, y, z = point["x"], point["y"], point["z"]
        else:
            x, y, z = point
        return (
            -half_x <= float(x) <= half_x
            and 0 <= float(y) <= 255
            and -half_z <= float(z) <= half_z
        )

    height_ok = all(
        1 <= float(height) <= 254
        for height in _flatten(preview["grid"]["heights"])
    )
    surface_points = [
        preview["spawn"],
        *preview["trees"],
        *preview["structures"],
        *preview["caves"]["chambers"],
        *preview["caves"]["entrances"],
    ]
    cave_points = [
        point
        for segment in preview["caves"]["segments"]
        for point in (segment["start"], segment["end"])
    ]
    return height_ok and all(point_ok(point) for point in surface_points + cave_points)


def _nearest_height_mae(fields: list[tuple[int, ...]]) -> float:
    if len(fields) < 2:
        return 0.0
    nearest = math.inf
    for left_index, left in enumerate(fields[:-1]):
        for right in fields[left_index + 1:]:
            mae = sum(abs(a - b) for a, b in zip(left, right)) / (
                len(left) * 1000
            )
            nearest = min(nearest, mae)
    return round(nearest, 6)


def _family_base(style: str) -> dict[str, Any]:
    """Choose an authored baseline without coupling this tool to minigames."""

    for preset in worldgen.options()["presets"]:
        if preset["values"]["terrain_style"] == style:
            return dict(preset["values"])
    return worldgen.normalize({
        "name": f"audit-{style.replace('_', '-')}",
        "terrain_style": style,
        "style_strength": 0.9,
        "world_width": 192,
        "world_depth": 96,
        "objective": "exploration",
    })


def audit(*, start_seed: int = 810_000, seeds_per_style: int = 24,
          resolution: int = 25, batch_seeds: int = 256) -> dict[str, Any]:
    if seeds_per_style < 3:
        raise ValueError("seeds_per_style must be at least 3")
    if resolution not in worldgen.FAST_BATCH_RESOLUTIONS:
        raise ValueError(
            f"resolution must be one of {list(worldgen.FAST_BATCH_RESOLUTIONS)}"
        )
    if not 3 <= batch_seeds <= worldgen.MAX_BATCH_SEEDS:
        raise ValueError(
            f"batch_seeds must be between 3 and {worldgen.MAX_BATCH_SEEDS}"
        )

    started = time.perf_counter()
    families: list[dict[str, Any]] = []
    every_composite: set[str] = set()
    all_safe = True
    all_bounded = True
    all_deterministic = True
    all_height_unique = True

    styles = list(worldgen.FIELDS["terrain_style"]["choices"])
    for style_index, style in enumerate(styles):
        base = _family_base(style)
        # Every family is exercised as a non-square environment.  Preset
        # ecology/hydrology remains intact while geometry uses the same extent.
        base.update({"world_width": 192, "world_depth": 96})
        seed_values = [
            start_seed + style_index * 10_000 + index
            for index in range(seeds_per_style)
        ]
        previews = [
            worldgen._preview(
                worldgen.normalize({**base, "seed": seed}),
                resolution=resolution,
            )
            for seed in seed_values
        ]
        repeated = worldgen._preview(
            worldgen.normalize({**base, "seed": seed_values[0]}),
            resolution=resolution,
        )

        height_fields = [_quantized_heights(item) for item in previews]
        height_fingerprints = {_canonical_digest(field) for field in height_fields}
        biome_fingerprints = {
            _canonical_digest(_flatten(item["grid"]["biomes"]))
            for item in previews
        }
        structure_fingerprints = {
            _layout_fingerprint(item["structures"]) for item in previews
        }
        cave_fingerprints = {
            _cave_fingerprint(item["caves"]) for item in previews
        }
        composites = {
            _canonical_digest({
                "height": _canonical_digest(height_fields[index]),
                "biomes": _canonical_digest(_flatten(item["grid"]["biomes"])),
                "structures": _layout_fingerprint(item["structures"]),
                "caves": _cave_fingerprint(item["caves"]),
            })
            for index, item in enumerate(previews)
        }
        every_composite.update(composites)
        deterministic = previews[0] == repeated
        safe_spawns = sum(bool(item["spawn"]["safe"]) for item in previews)
        bounded = sum(_bounds_ok(item) for item in previews)
        unique_heights = len(height_fingerprints)
        all_safe &= safe_spawns == seeds_per_style
        all_bounded &= bounded == seeds_per_style
        all_deterministic &= deterministic
        all_height_unique &= unique_heights == seeds_per_style

        families.append({
            "terrain_style": style,
            "native_profile": previews[0]["design"]["native_handoff"]
                ["template_compilation"]["profile"],
            "extent": {"width": 192, "depth": 96},
            "seed_count": seeds_per_style,
            "deterministic_replay": deterministic,
            "safe_spawns": safe_spawns,
            "bounded_worlds": bounded,
            "ready_for_native_authoring": sum(
                item["quality"] == "ready_for_native_authoring"
                for item in previews
            ),
            "unique_height_fields": unique_heights,
            "unique_biome_layouts": len(biome_fingerprints),
            "unique_structure_layouts": len(structure_fingerprints),
            "unique_cave_graphs": len(cave_fingerprints),
            "unique_composite_worlds": len(composites),
            "nearest_pairwise_height_mae_blocks": _nearest_height_mae(height_fields),
            "relief_blocks": {
                "minimum": min(item["metrics"]["relief"] for item in previews),
                "mean": round(mean(item["metrics"]["relief"] for item in previews), 4),
                "maximum": max(item["metrics"]["relief"] for item in previews),
            },
            "traversable_fraction_mean": round(mean(
                item["metrics"]["traversable_fraction"] for item in previews
            ), 6),
            "warning_worlds": sum(bool(item["warnings"]) for item in previews),
        })

    cave_rows: list[dict[str, Any]] = []
    cave_worlds_per_case = max(8, min(seeds_per_style, 24))
    for case_index, case in enumerate(CAVE_CASES):
        seed_values = [
            start_seed + 200_000 + case_index * 1_000 + index
            for index in range(cave_worlds_per_case)
        ]
        previews = [
            worldgen._preview(
                worldgen.normalize({
                    **worldgen.DEFAULTS,
                    **{key: value for key, value in case.items() if key != "id"},
                    "name": f"audit-caves-{case['id']}",
                    "terrain_style": "natural_mountains",
                    "world_width": 192,
                    "world_depth": 96,
                    "seed": seed,
                }),
                resolution=resolution,
            )
            for seed in seed_values
        ]
        networks = [item["metrics"]["cave_network_count"] for item in previews]
        segments = [item["metrics"]["cave_segment_count"] for item in previews]
        chambers = [item["metrics"]["cave_chamber_count"] for item in previews]
        entrances = [item["metrics"]["cave_entrance_count"] for item in previews]
        fingerprints = {_cave_fingerprint(item["caves"]) for item in previews}
        cave_rows.append({
            "case": case["id"],
            "controls": {
                key: case[key]
                for key in (
                    "cave_style", "cave_density", "cave_scale",
                    "cave_verticality", "cave_entrances",
                )
            },
            "native_profile": previews[0]["design"]["native_handoff"]
                ["template_compilation"]["profile"],
            "seed_count": cave_worlds_per_case,
            "unique_cave_graphs": len(fingerprints),
            "bounded_worlds": sum(_bounds_ok(item) for item in previews),
            "network_count": {
                "minimum": min(networks),
                "mean": round(mean(networks), 4),
                "maximum": max(networks),
            },
            "segment_count_mean": round(mean(segments), 4),
            "segments_per_network_mean": round(mean(
                segment / network if network else 0
                for segment, network in zip(segments, networks)
            ), 4),
            "chamber_count_mean": round(mean(chambers), 4),
            "entrance_count_mean": round(mean(entrances), 4),
        })

    cave_by_id = {row["case"]: row for row in cave_rows}

    extent_rows: list[dict[str, Any]] = []
    for extent_index, (width, depth) in enumerate(
        ((32, 32), (96, 96), (32, 512), (512, 32), (512, 512))
    ):
        preview = worldgen._preview(
            worldgen.normalize({
                **worldgen.DEFAULTS,
                "world_width": width,
                "world_depth": depth,
                "seed": start_seed + 300_000 + extent_index,
            }),
            resolution=resolution,
        )
        capture_plan = preview["design"]["extent"]["native_capture"]
        extent_rows.append({
            "width": width,
            "depth": depth,
            "step_x": preview["grid"]["step_x"],
            "step_z": preview["grid"]["step_z"],
            "estimated_region_columns": preview["metrics"]["estimated_region_columns"],
            "capture_tiles": capture_plan["capture_tiles"],
            "exact_port_supported_now": capture_plan["exact_port_supported_now"],
            "extent_status": capture_plan["status"],
            "bounded": _bounds_ok(preview),
            "safe_spawn": bool(preview["spawn"]["safe"]),
            "content": {
                "trees": preview["metrics"]["tree_count"],
                "structures": preview["metrics"]["structure_count"],
                "cave_segments": preview["metrics"]["cave_segment_count"],
            },
        })

    batch = worldgen.batch({
        "config": {
            **worldgen.DEFAULTS,
            "world_width": 256,
            "world_depth": 128,
        },
        "start_seed": start_seed + 900_000,
        "count": batch_seeds,
        "stride": 17,
        "resolution": 17,
        "strategy": "native_seeds",
    })
    total_worlds = len(styles) * seeds_per_style
    gates = {
        "every_seed_has_unique_height_geometry_within_family": all_height_unique,
        "every_seed_has_unique_composite_content_globally": (
            len(every_composite) == total_worlds
        ),
        "same_seed_replays_bit_exactly_in_every_family": all_deterministic,
        "all_spawns_are_safe": all_safe,
        "all_content_stays_inside_rectangular_bounds": all_bounded,
        "all_native_profiles_resolve": all(row["native_profile"] for row in families),
        "hundreds_seed_batch_is_complete": len(batch["rows"]) == batch_seeds,
        "hundreds_seed_batch_has_unique_previews": (
            batch["aggregate"]["unique_preview_digests"] == batch_seeds
        ),
        "hundreds_seed_batch_uses_one_native_recipe": (
            batch["aggregate"]["unique_design_digests"] == 1
            and batch["native_seed_plan"]["asset_packs_required"] == 1
        ),
        "deeproot_native_profile_is_exercised": all(
            cave_by_id[name]["native_profile"] == "deeproot_caverns"
            for name in ("deeproot_caverns", "deeproot_labyrinth")
        ),
        "no_caves_means_an_empty_graph": (
            cave_by_id["none"]["network_count"]["maximum"] == 0
            and cave_by_id["none"]["unique_cave_graphs"] == 1
        ),
        "every_enabled_cave_case_varies_spatially_by_seed": all(
            row["unique_cave_graphs"] == cave_worlds_per_case
            for row in cave_rows if row["case"] != "none"
        ),
        "cave_density_and_scale_change_network_count": (
            cave_by_id["dense_tunnels"]["network_count"]["mean"]
            > cave_by_id["sparse_tunnels"]["network_count"]["mean"]
        ),
        "labyrinth_style_has_more_branches_per_network": (
            cave_by_id["deeproot_labyrinth"]["segments_per_network_mean"]
            > cave_by_id["sparse_tunnels"]["segments_per_network_mean"]
        ),
        "cavern_style_creates_chambers": (
            cave_by_id["deeproot_caverns"]["chamber_count_mean"] > 0
        ),
        "all_cave_graphs_stay_inside_rectangular_bounds": all(
            row["bounded_worlds"] == cave_worlds_per_case for row in cave_rows
        ),
        "minimum_maximum_and_extreme_aspect_extents_are_bounded": all(
            row["bounded"] and row["safe_spawn"] for row in extent_rows
        ),
        "extent_controls_change_real_axis_scale": math.isclose(
            next(
                row for row in extent_rows
                if row["width"] == 32 and row["depth"] == 512
            )["step_z"],
            16 * next(
                row for row in extent_rows
                if row["width"] == 32 and row["depth"] == 512
            )["step_x"],
            abs_tol=1e-5,
        ),
        "extent_planner_matches_96_block_region_core_contract": (
            next(
                row for row in extent_rows
                if row["width"] == 96 and row["depth"] == 96
            )["exact_port_supported_now"]
            and next(
                row for row in extent_rows
                if row["width"] == 32 and row["depth"] == 512
            )["capture_tiles"] == {"x": 1, "z": 6, "total": 6}
            and next(
                row for row in extent_rows
                if row["width"] == 512 and row["depth"] == 512
            )["capture_tiles"] == {"x": 6, "z": 6, "total": 36}
        ),
    }
    source = Path(worldgen.__file__).resolve()
    workspace = Path(__file__).resolve().parents[2]
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "source": {
            "path": source.relative_to(workspace).as_posix(),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "preview_schema": worldgen.PREVIEW_SCHEMA,
        },
        "audit": {
            "start_seed": start_seed,
            "terrain_families": len(styles),
            "seeds_per_family": seeds_per_style,
            "family_worlds": total_worlds,
            "cave_control_worlds": len(CAVE_CASES) * cave_worlds_per_case,
            "total_semantic_previews": (
                total_worlds + len(CAVE_CASES) * cave_worlds_per_case
                + len(extent_rows) + batch_seeds
            ),
            "resolution": resolution,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        },
        "families": families,
        "cave_topology_controls": {
            "count_contract": (
                "Network count is deterministic for one fixed design; seeds "
                "vary graph placement and shape. Density and scale change the "
                "authored count across designs."
            ),
            "cases": cave_rows,
        },
        "extent_matrix": extent_rows,
        "batch_optimization": {
            "seed_count": batch_seeds,
            "resolution": batch["preview_resolution"],
            **batch["aggregate"],
            "capture_plan": batch["native_seed_plan"],
        },
        "gates": gates,
        "interpretation": (
            "Passing establishes deterministic, bounded, spatially distinct "
            "semantic preview worlds across every authored terrain family, "
            "including non-square extents, tunable cave topology and "
            "hundreds-seed batching. It does "
            "not establish block-exact native Hytale or JAX fidelity."
        ),
        "ownership_boundary": (
            "This audit covers environment content and neutral anchors only; "
            "it does not implement minigame rewards, rules, scoring or logic."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=810_000)
    parser.add_argument("--seeds-per-style", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=25)
    parser.add_argument("--batch-seeds", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        start_seed=args.start_seed,
        seeds_per_style=args.seeds_per_style,
        resolution=args.resolution,
        batch_seeds=args.batch_seeds,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
        print(args.output)
    print(encoded, end="")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
