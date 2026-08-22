"""Compare structures from one or more split-consistent V2 analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Sequence


ANALYSIS_SCHEMA = "hytalerl_worldgen_v2_region_analysis_v2"
COMPARISON_SCHEMA = "hytalerl_worldgen_v2_region_comparison_v1"
COMPARISON_VERSION = 1


def compare_analysis_reports(
    analysis_paths: Sequence[str | Path],
    *,
    output: str | Path,
) -> dict[str, Any]:
    """Validate, merge, and compare disjoint analyzed structures."""

    if isinstance(analysis_paths, (str, Path)):
        raise TypeError("analysis_paths must be a sequence")
    paths = tuple(Path(value).resolve() for value in analysis_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("comparison requires one or more unique analyses")
    destination = Path(output).resolve()
    if destination in paths:
        raise ValueError("comparison output cannot overwrite an input")

    reports = []
    inputs = []
    split_identity: tuple[str, tuple[tuple[str, str], ...]] | None = None
    structures: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        report = _json_object(path)
        if report.get("schema") != ANALYSIS_SCHEMA or report.get("version") != 2:
            raise ValueError("unsupported WorldGen V2 Region analysis")
        if _report_sha256(report) != report.get("report_semantic_sha256"):
            raise ValueError("analysis report semantic SHA-256 mismatch")
        split = _mapping(
            report.get("measurement_split_contract"),
            "measurement split contract",
        )
        effective = _mapping(split.get("effective_mapping"), "effective mapping")
        identity = (
            _sha256(split.get("effective_salt_sha256"), "effective split salt"),
            tuple(sorted((str(seed), _split(value)) for seed, value in effective.items())),
        )
        if split_identity is None:
            split_identity = identity
        elif split_identity != identity:
            raise ValueError("analysis reports use different measurement splits")
        resolution = _mapping(
            report.get("block_asset_resolution"),
            "block asset resolution",
        )
        if resolution.get("status") != "resolved_from_capture_assets":
            raise ValueError("comparison requires resolved block asset IDs")
        for structure in _structure_list(report.get("structures")):
            name = _string(structure.get("structure"), "structure")
            if name in structures:
                raise ValueError(f"duplicate analyzed structure {name!r}")
            structures[name] = structure
        inputs.append(
            {
                "path": Path(
                    os.path.relpath(path, destination.parent)
                ).as_posix(),
                "file_sha256": _file_sha256(path),
                "semantic_sha256": report["report_semantic_sha256"],
                "source_capture_report_file_sha256": report[
                    "source_capture_report_file_sha256"
                ],
            }
        )
        reports.append(report)

    assert split_identity is not None
    if len(structures) < 2:
        raise ValueError("comparison requires at least two structures")

    summaries = {
        name: _structure_summary(structure)
        for name, structure in sorted(structures.items())
    }
    comparison: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "version": COMPARISON_VERSION,
        "inputs": inputs,
        "measurement_split_contract": {
            "effective_salt_sha256": split_identity[0],
            "effective_mapping": dict(split_identity[1]),
            "source_report_overrides_present": any(
                bool(
                    _mapping(
                        report["measurement_split_contract"],
                        "split contract",
                    ).get("overrides_source_report")
                )
                for report in reports
            ),
        },
        "structure_count": len(structures),
        "artifact_count": sum(
            int(summary["artifact_count"]) for summary in summaries.values()
        ),
        "structures": list(summaries.values()),
        "cross_structure_material_union_jaccard": _union_jaccard(summaries),
        "same_seed_material_jaccard": _same_seed_jaccard(structures),
        "observability_boundary": {
            "fluid_asset_identity": (
                "measured_exact_stable_Fluid_getId_sidecars"
                if all(
                    summary["stable_fluid_identity"]["complete"]
                    for summary in summaries.values()
                )
                else "unmeasured_capture_has_no_complete_fluid_sidecars"
            ),
            "biome_identity": "unmeasured_capture_has_no_per_cell_biome_key",
            "road_instances": "unmeasured_capture_has_no_road_markers",
            "placed_structures": (
                "unmeasured_generator_preset_name_is_not_instance_identity"
            ),
            "policy_generalization": "unmeasured_requires_policy_consumer",
        },
        "report_semantic_sha256": "",
    }
    comparison["report_semantic_sha256"] = _report_sha256(comparison)
    _atomic_json(destination, comparison)
    return comparison


def _structure_summary(structure: Mapping[str, Any]) -> dict[str, Any]:
    name = _string(structure.get("structure"), "structure")
    artifacts = _artifact_list(structure.get("artifacts"))
    by_split: dict[str, list[Mapping[str, Any]]] = {
        "calibration": [],
        "heldout": [],
    }
    by_seed: dict[str, set[str]] = {}
    for artifact in artifacts:
        split = _split(artifact.get("split"))
        by_split[split].append(artifact)
        seed = str(_integer(artifact.get("seed"), "seed"))
        if seed in by_seed:
            raise ValueError(f"structure {name!r} repeats seed {seed}")
        by_seed[seed] = _asset_ids(artifact)
    if len(by_split["calibration"]) < 2 or len(by_split["heldout"]) < 1:
        raise ValueError("structure comparison requires calibration and heldout rows")
    calibration_assets = set().union(
        *(_asset_ids(value) for value in by_split["calibration"])
    )
    heldout_assets = set().union(*(_asset_ids(value) for value in by_split["heldout"]))
    union = set().union(*by_seed.values())
    heldout_new = heldout_assets - calibration_assets
    pairwise = structure.get("pairwise_surface_comparison")
    if not isinstance(pairwise, list) or not pairwise:
        raise ValueError("structure lacks pairwise surface comparisons")
    disagreement = [float(row["differing_surface_fraction"]) for row in pairwise]
    relief = [int(_mapping(row.get("terrain"), "terrain")["relief_blocks"]) for row in artifacts]
    cave = [
        float(
            _mapping(row.get("cave_overhang_proxy"), "cave proxy")[
                "subsurface_void_fraction"
            ]
        )
        for row in artifacts
    ]
    fluids = [
        int(_mapping(row.get("fluids_and_hazards"), "hazards")["fluid_cells"])
        for row in artifacts
    ]
    damaging = [
        int(
            _mapping(row.get("fluids_and_hazards"), "hazards")[
                "damaging_flag_cells"
            ]
        )
        for row in artifacts
    ]
    stable_fluid_rows = [
        (
            _mapping(row.get("stable_fluid_identity"), "stable fluid identity")
            if row.get("stable_fluid_identity") is not None
            else {"status": "unmeasured_no_fluid_semantic_sidecar"}
        )
        for row in artifacts
    ]
    stable_complete = all(
        row.get("status") == "measured_exact" for row in stable_fluid_rows
    )
    stable_fluid_counts: dict[str, int] = {}
    if stable_complete:
        for row in stable_fluid_rows:
            assets = row.get("assets")
            if not isinstance(assets, list):
                raise ValueError("stable fluid assets must be a list")
            for asset in assets:
                item = _mapping(asset, "stable fluid asset")
                asset_id = _string(item.get("asset_id"), "fluid asset ID")
                count = _integer(
                    item.get("full_cell_count"),
                    "fluid asset cell count",
                )
                stable_fluid_counts[asset_id] = (
                    stable_fluid_counts.get(asset_id, 0) + count
                )
    split_distributions = {
        split: _split_distribution(rows)
        for split, rows in by_split.items()
    }
    return {
        "structure": name,
        "artifact_count": len(artifacts),
        "seed_asset_ids": {seed: sorted(values) for seed, values in sorted(by_seed.items())},
        "non_air_asset_union": sorted(union),
        "non_air_asset_union_count": len(union),
        "non_air_asset_count_per_seed": _range([len(value) for value in by_seed.values()]),
        "heldout_asset_novelty": {
            "calibration_union_count": len(calibration_assets),
            "heldout_union_count": len(heldout_assets),
            "new_in_heldout_count": len(heldout_new),
            "new_in_heldout": sorted(heldout_new),
            "heldout_fraction_new": (
                float(len(heldout_new) / len(heldout_assets))
                if heldout_assets
                else 0.0
            ),
        },
        "split_distributions": split_distributions,
        "heldout_outside_calibration_range": _heldout_range_coverage(
            split_distributions["calibration"],
            split_distributions["heldout"],
        ),
        "terrain_relief_blocks": _range(relief),
        "pairwise_surface_disagreement_fraction": _range(disagreement),
        "subsurface_void_fraction": _range(cave),
        "fluid_cells_in_measured_core": _range(fluids),
        "stable_fluid_identity": {
            "complete": stable_complete,
            "asset_ids": sorted(stable_fluid_counts),
            "full_asset_cell_counts": dict(sorted(stable_fluid_counts.items())),
        },
        "damaging_flag_cells_in_measured_core": _range(damaging),
    }


def _split_distribution(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stable = [
        _mapping(row.get("stable_fluid_identity"), "stable fluid identity")
        if row.get("stable_fluid_identity") is not None
        else {"status": "unmeasured_no_fluid_semantic_sidecar"}
        for row in artifacts
    ]
    spawn_deltas = [
        _mapping(row.get("spawn"), "spawn").get(
            "feet_y_minus_surface_cell_ceiling"
        )
        for row in artifacts
    ]
    spawn_deltas = [value for value in spawn_deltas if value is not None]
    stable_assets = {
        _string(asset.get("asset_id"), "stable fluid asset ID")
        for row in stable
        if row.get("status") == "measured_exact"
        for asset in row.get("assets", [])
        if isinstance(asset, Mapping)
    }
    return {
        "artifact_count": len(artifacts),
        "terrain_relief_blocks": _distribution(
            [
                _integer(
                    _mapping(row.get("terrain"), "terrain").get(
                        "relief_blocks"
                    ),
                    "terrain relief",
                )
                for row in artifacts
            ]
        ),
        "subsurface_void_fraction": _distribution(
            [
                float(
                    _mapping(
                        row.get("cave_overhang_proxy"),
                        "cave proxy",
                    )["subsurface_void_fraction"]
                )
                for row in artifacts
            ]
        ),
        "unique_non_air_material_ids": _distribution(
            [
                _integer(
                    _mapping(row.get("materials"), "materials").get(
                        "unique_non_air_asset_ids"
                    ),
                    "unique non-air materials",
                )
                for row in artifacts
            ]
        ),
        "physical_fluid_cells_core": _distribution(
            [
                _integer(
                    _mapping(
                        row.get("fluids_and_hazards"),
                        "hazards",
                    ).get("fluid_cells"),
                    "core fluid cells",
                )
                for row in artifacts
            ]
        ),
        "stable_fluid_cells_full": _distribution(
            [
                _integer(row.get("full_fluid_cells"), "full stable fluid cells")
                for row in stable
                if row.get("status") == "measured_exact"
            ]
        ),
        "damaging_flag_cells_core": _distribution(
            [
                _integer(
                    _mapping(
                        row.get("fluids_and_hazards"),
                        "hazards",
                    ).get("damaging_flag_cells"),
                    "damaging flag cells",
                )
                for row in artifacts
            ]
        ),
        "spawn_feet_y_minus_surface": _distribution(
            [float(value) for value in spawn_deltas]
        ),
        "stable_fluid_asset_union": sorted(stable_assets),
        "stable_fluid_identity_complete": len(stable) == len(artifacts)
        and all(row.get("status") == "measured_exact" for row in stable),
    }


def _heldout_range_coverage(
    calibration: Mapping[str, Any],
    heldout: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "terrain_relief_blocks",
        "subsurface_void_fraction",
        "unique_non_air_material_ids",
        "physical_fluid_cells_core",
        "stable_fluid_cells_full",
        "damaging_flag_cells_core",
        "spawn_feet_y_minus_surface",
    ):
        left = _mapping(calibration.get(name), f"calibration {name}")
        right = _mapping(heldout.get(name), f"heldout {name}")
        if not left.get("count") or not right.get("count"):
            result[name] = {"measured": False}
            continue
        result[name] = {
            "measured": True,
            "calibration_minimum": left["minimum"],
            "calibration_maximum": left["maximum"],
            "heldout_minimum": right["minimum"],
            "heldout_maximum": right["maximum"],
            "heldout_range_extends_calibration": bool(
                right["minimum"] < left["minimum"]
                or right["maximum"] > left["maximum"]
            ),
        }
    return result


def _distribution(values: Sequence[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def _union_jaccard(
    summaries: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    names = sorted(summaries)
    result = []
    for index, left_name in enumerate(names):
        left = set(summaries[left_name]["non_air_asset_union"])
        for right_name in names[index + 1 :]:
            right = set(summaries[right_name]["non_air_asset_union"])
            result.append(
                _jaccard_row(left_name, right_name, left, right)
            )
    return result


def _same_seed_jaccard(
    structures: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    assets = {
        name: {
            str(_integer(row.get("seed"), "seed")): _asset_ids(row)
            for row in _artifact_list(structure.get("artifacts"))
        }
        for name, structure in structures.items()
    }
    names = sorted(assets)
    seeds = set.intersection(*(set(value) for value in assets.values()))
    result = []
    for seed in sorted(seeds, key=int):
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                row = _jaccard_row(
                    left_name,
                    right_name,
                    assets[left_name][seed],
                    assets[right_name][seed],
                )
                row["seed"] = int(seed)
                result.append(row)
    return result


def _jaccard_row(
    left_name: str,
    right_name: str,
    left: set[str],
    right: set[str],
) -> dict[str, Any]:
    union = left | right
    intersection = left & right
    return {
        "left_structure": left_name,
        "right_structure": right_name,
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": float(len(intersection) / len(union)) if union else 1.0,
    }


def _asset_ids(artifact: Mapping[str, Any]) -> set[str]:
    materials = _mapping(artifact.get("materials"), "materials")
    rows = materials.get("assets")
    if not isinstance(rows, list):
        raise ValueError("material assets must be a list")
    result = {
        _string(row.get("asset_id"), "asset ID")
        for row in rows
        if isinstance(row, Mapping) and row.get("is_air") is False
    }
    expected = _integer(
        materials.get("unique_non_air_asset_ids"),
        "unique non-air assets",
    )
    if len(result) != expected:
        raise ValueError("resolved material IDs differ from stable asset count")
    return result


def _range(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("range requires values")
    return {"minimum": min(values), "maximum": max(values)}


def _structure_list(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError("analysis structures must be a nonempty list")
    return value


def _artifact_list(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError("structure artifacts must be a nonempty list")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _split(value: object) -> str:
    if value not in ("calibration", "heldout"):
        raise ValueError("split must be calibration or heldout")
    return str(value)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _report_sha256(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_semantic_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = compare_analysis_reports(args.analysis, output=args.output)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "structure_count": result["structure_count"],
                "artifact_count": result["artifact_count"],
                "report_semantic_sha256": result["report_semantic_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
