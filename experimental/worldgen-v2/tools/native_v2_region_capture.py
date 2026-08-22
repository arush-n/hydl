"""Capture exact bounded HytaleGenerator Regions and semantic sidecars.

The small V2 pilot proves that the provider can create worlds.  This tool is
the next boundary: it captures complete 5x5x10 Regions, verifies every physical
palette column after serialization, captures the block-semantic sidecar needed
to count actual Hytale block assets, and can capture a separate stable-fluid
sidecar without changing the physical Region schema.

Nothing here changes the meaning of the frozen ``world="hytale"`` template.
The only live world template accepted by this experiment is the explicit
``world="hytale_generator"`` option.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import numpy as np

from _workspace import use_workspace_hytalerl


use_workspace_hytalerl()

from hytalegym.envs.hytale_env import HytaleEnv  # noqa: E402
from hytalegym.geometry.contract import (  # noqa: E402
    FLAG_DAMAGING,
    FLAG_FLUID,
)
from hytalegym.worldgen.native_lease import (  # noqa: E402
    acquire_native_evidence_lease,
    update_owned_native_evidence_lease,
)
from hytalegym.worldgen.region import (  # noqa: E402
    NativeRegionBlockSemanticSnapshot,
    NativeRegionSnapshot,
    capture_native_region,
    capture_native_region_block_semantics,
    create_region_artifact_library,
    create_region_block_semantic_library,
    save_region_library_diversity_report,
)
from hytalegym.worldgen.region.stability import (  # noqa: E402
    require_current_native_runtime_bridge,
)
from hytalegym.worldgen.region.native import (  # noqa: E402
    capture_native_region_pass,
)

from native_v2_region_fluid_sidecar import (  # noqa: E402
    NativeRegionFluidSemanticSnapshot,
    capture_native_region_fluid_semantics,
    require_fluid_snapshot_identity,
)


WORLD_TEMPLATE = "hytale_generator"
WORLDGEN_PROVIDER = "HytaleGenerator"
PRIMARY_STRUCTURES = (
    "Default",
    "Zone1_Plains1",
    "Zone2_Desert1",
    "Zone3_Taiga1",
    "Zone4_Volcanic1",
)

CAPTURE_PLAN_SCHEMA = "hytalerl_worldgen_v2_region_capture_plan_v1"
CAPTURE_PLAN_VERSION = 1
REGION_REPORT_SCHEMA = "hytalerl_worldgen_v2_region_report_v1"
REGION_REPORT_VERSION = 1

_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_PALETTE_ARCHIVE_FIELDS = {
    "flags": "cell_flags",
    "shape_index": "cell_shape_index",
    "fluid_level": "cell_fluid_level",
    "fluid_fill_height": "cell_fluid_fill_height",
    "support": "cell_support",
    "block_damage": "cell_block_damage",
    "fluid_damage": "cell_fluid_damage",
}
_PLAN_REQUEST_FIELDS = (
    "bridge_jar_sha256",
    "server_jar_sha256",
    "assets_sha256",
    "seeds",
    "structures",
    "heldout_count",
    "role",
    "core_min_chunk_xz",
)
_PLAN_FIELDS = frozenset(
    {
        "schema",
        "version",
        "world_template",
        "worldgen_provider",
        *_PLAN_REQUEST_FIELDS,
        "split_salt_sha256",
        "seed_split",
        "capture_contract",
    }
)

_T = TypeVar("_T")


def _timed_call(
    timings: dict[str, dict[str, float | int]],
    stage: str,
    operation: Callable[[], _T],
) -> _T:
    """Measure one native-pipeline operation, including failed attempts."""

    started = time.perf_counter()
    try:
        return operation()
    finally:
        elapsed = time.perf_counter() - started
        row = timings.setdefault(stage, {"seconds": 0.0, "operations": 0})
        row["seconds"] = float(row["seconds"]) + elapsed
        row["operations"] = int(row["operations"]) + 1


def _timing_report(
    timings: Mapping[str, Mapping[str, float | int]],
    *,
    total_seconds: float,
) -> dict[str, Any]:
    stages = {}
    measured = 0.0
    for name, row in sorted(timings.items()):
        seconds = float(row["seconds"])
        operations = int(row["operations"])
        measured += seconds
        stages[name] = {
            "seconds": round(seconds, 6),
            "operations": operations,
            "mean_seconds": round(seconds / operations, 6),
        }
    return {
        "total_seconds": round(total_seconds, 6),
        "instrumented_stage_seconds": round(measured, 6),
        "unattributed_seconds": round(max(0.0, total_seconds - measured), 6),
        "stages": stages,
        "interpretation": (
            "Measured native capture stages, not Studio preview throughput. "
            "Use the dominant stage before choosing a parallelism or caching change."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture exact Region-scale WorldGen V2 evidence. The running "
            "server must already contain the explicitly named candidate bridge."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="repeat for explicit seeds; at least three unique seeds are required",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        help="first seed in a deterministic range (use with --seed-count)",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        help="number of seeds in a deterministic range (maximum 4096)",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help="positive range stride; defaults to 1",
    )
    parser.add_argument(
        "--seed-plan",
        type=Path,
        help=(
            "WorldGen Studio batch JSON; reads native_seed_plan.seeds and "
            "supports resumable hundreds-seed captures"
        ),
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=list(PRIMARY_STRUCTURES),
    )
    parser.add_argument("--heldout-count", type=int, default=1)
    parser.add_argument("--split-salt-sha256")
    parser.add_argument(
        "--fluid-semantics",
        action="store_true",
        help=(
            "capture stable Fluid.getId() sidecars; requires a bridge with "
            "the experimental region_fluid_semantics request"
        ),
    )
    parser.add_argument("--role", default="Kweebec_Razorleaf")
    parser.add_argument("--core-min-chunk-x", type=int)
    parser.add_argument("--core-min-chunk-z", type=int)
    parser.add_argument("--bridge-jar", type=Path, required=True)
    parser.add_argument("--server-jar", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only artifacts that pass the complete frozen plan",
    )
    args = parser.parse_args()
    args.seed = _resolve_capture_seeds(parser, args)
    _validate_args(parser, args)

    output = args.output.resolve()
    provenance = {
        name: _provenance(path)
        for name, path in (
            ("bridge_jar", args.bridge_jar),
            ("server_jar", args.server_jar),
            ("assets", args.assets),
        )
    }
    plan = _capture_plan(parser, args, output, provenance)
    with acquire_native_evidence_lease(
        host=args.host,
        port=args.port,
        purpose="worldgen_v2_region_capture",
        stage=output,
        evidence_kinds=(
            "worldgen_v2",
            "native_region",
            "block_semantics",
            "palette_hazards",
            *(("stable_fluid_semantics",) if args.fluid_semantics else ()),
        ),
    ) as lease:
        report_path = run(args, plan, provenance, lease=lease)
    print("WORLDGEN_V2_REGION_CAPTURE_PASS", flush=True)
    print(report_path, flush=True)


def run(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    provenance: Mapping[str, Mapping[str, Any]],
    *,
    lease: Any | None = None,
) -> Path:
    """Execute one already-frozen capture plan and publish its report."""

    output = Path(args.output).resolve()
    expected_bridge = str(plan["bridge_jar_sha256"]).upper()
    structure_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for structure_index, structure in enumerate(plan["structures"], start=1):
        if lease is not None:
            update_owned_native_evidence_lease(
                lease,
                phase="region_capture",
                phase_status="running",
                detail={
                    "structure": structure,
                    "structure_index": structure_index,
                    "structure_count": len(plan["structures"]),
                },
            )
        structure_reports.append(
            _capture_structure(
                args,
                plan,
                structure=str(structure),
                expected_bridge=expected_bridge,
                output=output,
            )
        )

    all_rows = [
        row
        for structure in structure_reports
        for row in structure["artifacts"]
    ]
    report: dict[str, Any] = {
        "schema": REGION_REPORT_SCHEMA,
        "version": REGION_REPORT_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "capture_seconds": time.perf_counter() - started,
        "world_template": WORLD_TEMPLATE,
        "worldgen_provider": WORLDGEN_PROVIDER,
        "provenance": dict(provenance),
        "capture_plan_sha256": _canonical_sha256(plan),
        "capture_plan": "capture-plan.json",
        "artifact_count": len(all_rows),
        "structures": structure_reports,
        "block_identity_metric": {
            "field": "unique_local_block_ids",
            "identity": "sha256_hytale_block_asset_key",
            "scope": "all_used_cells_in_complete_5x5x10_region",
            "includes_air": True,
            "warning": (
                "This is stable hashed Hytale asset identity from the Region "
                "block-semantic sidecar, not a process-local integer ID and "
                "not a geometry-palette count."
            ),
        },
        "measurement_split_contract": {
            "method": "sha256_seed_rank_v1",
            "scope": "seed_identity_shared_across_all_structures",
            "mapping": plan["seed_split"],
            "warning": (
                "Each nested exact Region library retains its own artifact-"
                "hash split for compatibility. Measurement claims in this "
                "report use this seed-level split so a held-out seed cannot "
                "leak through another V2 structure."
            ),
        },
        "split_summary": _split_summary(all_rows),
        "validated": {
            "explicit_v2_world_template": True,
            "provider_and_structure_identity": True,
            "runtime_bridge_identity": True,
            "same_world_region_confirmation": True,
            "physical_palette_archive_fields": sorted(
                _PALETTE_ARCHIVE_FIELDS
            ),
            "physical_palette_round_trip": True,
            "block_asset_identity_sidecar": True,
            "block_semantic_round_trip": True,
            "calibration_and_heldout_split": True,
        },
        "report_semantic_sha256": "",
    }
    if plan["capture_contract"].get("stable_fluid_identity"):
        report["stable_fluid_identity_metric"] = {
            "field": "fluid_semantics",
            "identity": "stable_hytale_fluid_getId_string",
            "scope": "all_FLAG_FLUID_cells_in_complete_5x5x10_region",
            "physical_alignment": "exact_FLAG_FLUID_cell_equality",
            "same_world_stability": "forward_reverse_exact",
            "physical_region_schema_changed": False,
        }
        report["validated"]["stable_fluid_identity_sidecar"] = True
        report["validated"]["stable_fluid_forward_reverse_exact"] = True
    report["report_semantic_sha256"] = _report_sha256(report)
    destination = output / "report.json"
    _atomic_json(destination, report)
    if lease is not None:
        update_owned_native_evidence_lease(
            lease,
            phase="complete",
            phase_status="passed",
            detail={
                "artifact_count": len(all_rows),
                "report": str(destination),
                "report_semantic_sha256": report["report_semantic_sha256"],
            },
        )
    return destination


def _capture_structure(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    *,
    structure: str,
    expected_bridge: str,
    output: Path,
) -> dict[str, Any]:
    structure_started = time.perf_counter()
    timings: dict[str, dict[str, float | int]] = {}
    slug = _structure_slug(structure)
    structure_root = output / "structures" / slug
    region_root = structure_root / "region"
    artifact_root = region_root / "artifacts"
    semantic_root = structure_root / "block-semantics"
    sidecar_root = semantic_root / "sidecars"
    capture_fluid_semantics = bool(
        plan["capture_contract"].get("stable_fluid_identity")
    )
    fluid_root = structure_root / "fluid-semantics"
    fluid_sidecar_root = fluid_root / "sidecars"
    artifact_root.mkdir(parents=True, exist_ok=True)
    sidecar_root.mkdir(parents=True, exist_ok=True)
    if capture_fluid_semantics:
        fluid_sidecar_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    metric_asset_keys: dict[str, set[str]] = {}
    artifact_paths: list[Path] = []
    sidecar_paths: list[Path] = []
    fluid_sidecar_paths: list[Path] = []
    env = _timed_call(
        timings,
        "environment_initialization",
        lambda: HytaleEnv(
            task="kill_trork",
            backend="native",
            world=WORLD_TEMPLATE,
            worldgen_structure=structure,
            host=args.host,
            port=args.port,
            ticks_per_step=1,
            max_episode_steps=100,
            npc_role=plan["role"],
            combat_target_active=False,
            fidelity_fixture="static_region",
        ),
    )
    try:
        for seed_index, seed in enumerate(plan["seeds"], start=1):
            artifact_path = artifact_root / f"seed_{seed}.npz"
            sidecar_path = sidecar_root / f"seed_{seed}.block-semantics.npz"
            fluid_sidecar_path = (
                fluid_sidecar_root / f"seed_{seed}.fluid-semantics.npz"
            )
            if artifact_path.exists() and not args.resume:
                raise ValueError(
                    f"artifact already exists; pass --resume: {artifact_path}"
                )

            live_info: Mapping[str, Any] | None = None
            live_physical_confirmed = False
            if artifact_path.exists():
                snapshot = NativeRegionSnapshot.load(artifact_path)
                _require_v2_snapshot_identity(
                    snapshot,
                    seed=seed,
                    structure=structure,
                    expected_bridge=expected_bridge,
                )
                _verify_palette_archive(artifact_path, snapshot)
                print(
                    f"reuse physical structure={structure} "
                    f"seed={seed} ({seed_index}/{len(plan['seeds'])})",
                    flush=True,
                )
            else:
                observation, live_info = _timed_call(
                    timings, "native_world_reset", lambda: env.reset(seed=seed)
                )
                _require_live_v2_identity(
                    live_info,
                    seed=seed,
                    structure=structure,
                    expected_bridge=expected_bridge,
                )
                spawn_position = _spawn_position(observation)
                snapshot = _timed_call(
                    timings,
                    "physical_region_capture",
                    lambda: capture_native_region(
                        env,
                        core_min_chunk_x=args.core_min_chunk_x,
                        core_min_chunk_z=args.core_min_chunk_z,
                        native_evidence_jar_sha256=expected_bridge,
                        progress=_progress(
                            structure,
                            seed,
                            phase="physical",
                        ),
                    ),
                )
                snapshot = _bind_v2_snapshot_identity(
                    snapshot,
                    live_info,
                    seed=seed,
                    structure=structure,
                    expected_bridge=expected_bridge,
                    spawn_position=spawn_position,
                )
                snapshot.save(artifact_path)
                restored = NativeRegionSnapshot.load(artifact_path)
                _require_same_snapshot(snapshot, restored)
                _verify_palette_archive(artifact_path, restored)
                snapshot = restored
                live_physical_confirmed = True
                print(
                    f"captured physical structure={structure} seed={seed} "
                    f"semantic={snapshot.semantic_artifact_digest()}",
                    flush=True,
                )

            if sidecar_path.exists():
                if not args.resume:
                    raise ValueError(
                        f"sidecar already exists; pass --resume: {sidecar_path}"
                    )
                sidecar = NativeRegionBlockSemanticSnapshot.load(sidecar_path)
                _require_sidecar_identity(
                    sidecar,
                    snapshot,
                    expected_bridge=expected_bridge,
                )
                print(
                    f"reuse semantics structure={structure} seed={seed}",
                    flush=True,
                )
            else:
                if live_info is None:
                    observation, live_info = _timed_call(
                        timings, "native_world_reset", lambda: env.reset(seed=seed)
                    )
                    _require_live_v2_identity(
                        live_info,
                        seed=seed,
                        structure=structure,
                        expected_bridge=expected_bridge,
                    )
                    _require_live_spawn(snapshot, observation)
                sidecar = _timed_call(
                    timings,
                    "block_semantic_capture",
                    lambda: capture_native_region_block_semantics(
                        env,
                        snapshot,
                        evidence_bridge_sha256=expected_bridge,
                        progress=_progress(
                            structure,
                            seed,
                            phase="block_semantics",
                        ),
                    ),
                )
                sidecar.save(sidecar_path)
                restored_sidecar = NativeRegionBlockSemanticSnapshot.load(
                    sidecar_path
                )
                _require_same_sidecar(sidecar, restored_sidecar)
                _require_sidecar_identity(
                    restored_sidecar,
                    snapshot,
                    expected_bridge=expected_bridge,
                )
                sidecar = restored_sidecar
                live_physical_confirmed = True
                print(
                    f"captured semantics structure={structure} seed={seed} "
                    f"semantic={sidecar.semantic_sha256()}",
                    flush=True,
                )

            fluid_sidecar: NativeRegionFluidSemanticSnapshot | None = None
            if capture_fluid_semantics:
                if fluid_sidecar_path.exists():
                    if not args.resume:
                        raise ValueError(
                            "fluid sidecar already exists; pass --resume: "
                            f"{fluid_sidecar_path}"
                        )
                    fluid_sidecar = NativeRegionFluidSemanticSnapshot.load(
                        fluid_sidecar_path
                    )
                    require_fluid_snapshot_identity(
                        fluid_sidecar,
                        snapshot,
                        evidence_bridge_sha256=expected_bridge,
                    )
                    print(
                        f"reuse fluid semantics structure={structure} "
                        f"seed={seed}",
                        flush=True,
                    )
                else:
                    if live_info is None:
                        observation, live_info = _timed_call(
                            timings, "native_world_reset", lambda: env.reset(seed=seed)
                        )
                        _require_live_v2_identity(
                            live_info,
                            seed=seed,
                            structure=structure,
                            expected_bridge=expected_bridge,
                        )
                        _require_live_spawn(snapshot, observation)
                    if not live_physical_confirmed:
                        core = snapshot.core_min_chunk_xz
                        live_physical = _timed_call(
                            timings,
                            "physical_confirmation_capture",
                            lambda: capture_native_region_pass(
                                env,
                                requested_core=(int(core[0]), int(core[1])),
                                order="forward",
                            ),
                        )
                        if (
                            live_physical.semantic_digest()
                            != snapshot.semantic_digest()
                        ):
                            raise ValueError(
                                "live physical Region differs from fluid "
                                "semantic capture base"
                            )
                        live_physical_confirmed = True
                    fluid_sidecar = _timed_call(
                        timings,
                        "fluid_semantic_capture",
                        lambda: capture_native_region_fluid_semantics(
                            env,
                            snapshot,
                            evidence_bridge_sha256=expected_bridge,
                            progress=_progress(
                                structure,
                                seed,
                                phase="fluid_semantics",
                            ),
                        ),
                    )
                    fluid_sidecar.save(fluid_sidecar_path)
                    restored_fluid = NativeRegionFluidSemanticSnapshot.load(
                        fluid_sidecar_path
                    )
                    if (
                        restored_fluid.semantic_sha256()
                        != fluid_sidecar.semantic_sha256()
                        or not np.array_equal(
                            restored_fluid.cell_code,
                            fluid_sidecar.cell_code,
                        )
                    ):
                        raise ValueError(
                            "Region fluid-semantic round trip differs"
                        )
                    require_fluid_snapshot_identity(
                        restored_fluid,
                        snapshot,
                        evidence_bridge_sha256=expected_bridge,
                    )
                    fluid_sidecar = restored_fluid
                    print(
                        f"captured fluid semantics structure={structure} "
                        f"seed={seed} "
                        f"semantic={fluid_sidecar.semantic_sha256()}",
                        flush=True,
                    )

            metrics, asset_keys = _timed_call(
                timings,
                "content_metrics",
                lambda: region_content_metrics(snapshot, sidecar),
            )
            semantic = snapshot.semantic_artifact_digest()
            row = {
                "seed": seed,
                "structure": structure,
                "region_file": artifact_path.relative_to(output).as_posix(),
                "region_file_sha256": _timed_call(
                    timings, "artifact_hashing", lambda: _file_sha256(artifact_path)
                ),
                "region_semantic_sha256": semantic,
                "block_semantic_file": sidecar_path.relative_to(
                    output
                ).as_posix(),
                "block_semantic_file_sha256": _timed_call(
                    timings, "artifact_hashing", lambda: _file_sha256(sidecar_path)
                ),
                "block_semantic_sha256": sidecar.semantic_sha256(),
                "spawn_position": list(snapshot.metadata["spawn_position"]),
                **metrics,
            }
            if fluid_sidecar is not None:
                full_fluid_counts = fluid_sidecar.asset_cell_counts()
                core_fluid_counts = fluid_sidecar.asset_cell_counts(
                    core_only=True
                )
                row["fluid_semantics"] = {
                    "file": fluid_sidecar_path.relative_to(output).as_posix(),
                    "file_sha256": _timed_call(
                        timings,
                        "artifact_hashing",
                        lambda: _file_sha256(fluid_sidecar_path),
                    ),
                    "semantic_sha256": fluid_sidecar.semantic_sha256(),
                    "full_asset_cell_counts": full_fluid_counts,
                    "core_asset_cell_counts": core_fluid_counts,
                    "full_fluid_cells": sum(full_fluid_counts.values()),
                    "core_fluid_cells": sum(core_fluid_counts.values()),
                }
                fluid_sidecar_paths.append(fluid_sidecar_path)
            rows.append(row)
            metric_asset_keys[semantic] = asset_keys
            artifact_paths.append(artifact_path)
            sidecar_paths.append(sidecar_path)
    finally:
        _timed_call(timings, "environment_close", env.close)

    library = _timed_call(
        timings,
        "region_library_build",
        lambda: create_region_artifact_library(
            region_root,
            artifact_paths,
            heldout_count=int(plan["heldout_count"]),
            split_salt_sha256=str(plan["split_salt_sha256"]),
        ),
    )
    diversity_path = _timed_call(
        timings,
        "diversity_analysis",
        lambda: save_region_library_diversity_report(
            library,
            region_root / "diversity.json",
        ),
    )
    semantic_library = _timed_call(
        timings,
        "block_semantic_library_build",
        lambda: create_region_block_semantic_library(
            semantic_root,
            sidecar_paths,
            library,
        ),
    )
    for row in rows:
        row["split"] = plan["seed_split"][str(row["seed"])]
    rows.sort(key=lambda row: (row["split"], row["seed"]))
    summary = _split_summary(rows, asset_keys_by_semantic=metric_asset_keys)
    structure_report: dict[str, Any] = {
        "structure": structure,
        "directory": structure_root.relative_to(output).as_posix(),
        "region_library_manifest": library.manifest_path.relative_to(
            output
        ).as_posix(),
        "region_library_semantic_sha256": library.semantic_sha256,
        "block_semantic_library_manifest": (
            semantic_library.manifest_path.relative_to(output).as_posix()
        ),
        "block_semantic_library_sha256": semantic_library.semantic_sha256,
        "geometric_diversity_report": diversity_path.relative_to(
            output
        ).as_posix(),
        "artifact_count": len(rows),
        "measurement_split_contract": {
            "method": "sha256_seed_rank_v1",
            "mapping": plan["seed_split"],
            "nested_region_library_split_is_measurement_authority": False,
        },
        "split_summary": summary,
        "performance": _timing_report(
            timings,
            total_seconds=time.perf_counter() - structure_started,
        ),
        "artifacts": rows,
        "report_semantic_sha256": "",
    }
    if capture_fluid_semantics:
        structure_report["fluid_semantic_sidecars"] = {
            "directory": fluid_root.relative_to(output).as_posix(),
            "artifact_count": len(fluid_sidecar_paths),
            "all_physical_fluid_cells_named": True,
            "physical_region_schema_changed": False,
        }
    structure_report["report_semantic_sha256"] = _report_sha256(
        structure_report
    )
    _atomic_json(structure_root / "report.json", structure_report)
    return structure_report


def region_content_metrics(
    snapshot: NativeRegionSnapshot,
    sidecar: NativeRegionBlockSemanticSnapshot,
) -> tuple[dict[str, Any], set[str]]:
    """Return exact Region-scale block diversity and hazard diagnostics.

    ``unique_local_block_ids`` intentionally preserves the pilot's field name,
    but its identity is stronger here: the count is over used SHA-256 Hytale
    asset keys from the block-semantic sidecar, including canonical air.
    """

    if not isinstance(snapshot, NativeRegionSnapshot):
        raise TypeError("snapshot must be NativeRegionSnapshot")
    if not isinstance(sidecar, NativeRegionBlockSemanticSnapshot):
        raise TypeError("sidecar must be NativeRegionBlockSemanticSnapshot")
    if sidecar.source_region_semantic_sha256 != snapshot.semantic_digest():
        raise ValueError("block-semantic sidecar names another Region")
    if not np.array_equal(sidecar.core_min_chunk_xz, snapshot.core_min_chunk_xz):
        raise ValueError("block-semantic sidecar core differs from Region")

    semantic_codes = np.unique(sidecar.cell_code)
    asset_keys = {
        _asset_key_hex(sidecar.palette[int(code)].asset_key)
        for code in semantic_codes
    }
    non_air_asset_keys = {
        _asset_key_hex(sidecar.palette[int(code)].asset_key)
        for code in semantic_codes
        if any(sidecar.palette[int(code)].asset_key)
    }
    palette = snapshot.cell_palette
    physical_counts = np.bincount(
        snapshot.cell_code.reshape(-1),
        minlength=palette.size,
    ).astype(np.int64)
    used = physical_counts > 0
    flags = palette.flags.astype(np.uint16)
    fluid = (flags & np.uint16(FLAG_FLUID)) != 0
    damaging = (flags & np.uint16(FLAG_DAMAGING)) != 0
    block_damage = palette.block_damage != 0
    fluid_damage = palette.fluid_damage != 0
    return (
        {
            "unique_local_block_ids": len(asset_keys),
            "unique_non_air_block_ids": len(non_air_asset_keys),
            "unique_physical_cell_semantics": int(np.count_nonzero(used)),
            "cell_palette_size": palette.size,
            "shape_palette_size": snapshot.shape_palette.size,
            "total_region_cells": int(snapshot.cell_code.size),
            "hazards": {
                "palette_fields_complete": True,
                "used_palette_entries": int(np.count_nonzero(used)),
                "fluid_cells": _weighted_count(physical_counts, fluid),
                "damaging_flag_cells": _weighted_count(
                    physical_counts,
                    damaging,
                ),
                "nonzero_block_damage_cells": _weighted_count(
                    physical_counts,
                    block_damage,
                ),
                "nonzero_fluid_damage_cells": _weighted_count(
                    physical_counts,
                    fluid_damage,
                ),
                "maximum_block_damage": int(
                    np.max(palette.block_damage[used], initial=0)
                ),
                "maximum_fluid_damage": int(
                    np.max(palette.fluid_damage[used], initial=0)
                ),
                "maximum_fluid_level": int(
                    np.max(palette.fluid_level[used], initial=0)
                ),
                "maximum_fluid_fill_height": float(
                    np.max(palette.fluid_fill_height[used], initial=0.0)
                ),
            },
        },
        asset_keys,
    )


def _verify_palette_archive(
    path: Path,
    snapshot: NativeRegionSnapshot,
) -> None:
    """Hard-gate every physical palette column against the saved NPZ."""

    with np.load(path, allow_pickle=False) as archive:
        for attribute, archive_key in _PALETTE_ARCHIVE_FIELDS.items():
            if archive_key not in archive.files:
                raise ValueError(
                    f"V2 Region archive is missing palette field {archive_key}"
                )
            expected = np.asarray(getattr(snapshot.cell_palette, attribute))
            actual = archive[archive_key]
            if actual.shape != (snapshot.cell_palette.size,):
                raise ValueError(
                    f"V2 Region palette field {archive_key} has shape "
                    f"{actual.shape}, expected ({snapshot.cell_palette.size},)"
                )
            if actual.dtype != expected.dtype or not np.array_equal(
                actual,
                expected,
            ):
                raise ValueError(
                    f"V2 Region palette field {archive_key} failed round-trip"
                )


def _bind_v2_snapshot_identity(
    snapshot: NativeRegionSnapshot,
    info: Mapping[str, Any],
    *,
    seed: int,
    structure: str,
    expected_bridge: str,
    spawn_position: Sequence[float],
) -> NativeRegionSnapshot:
    metadata = dict(snapshot.metadata)
    if metadata.get("seed") != seed:
        raise ValueError("Region manifest seed differs from reset seed")
    if metadata.get("worldgen_provider") != info.get("worldgen_provider"):
        raise ValueError("Region manifest provider differs from reset provider")
    if metadata.get("worldgen_version") != info.get("worldgen_version"):
        raise ValueError("Region manifest version differs from reset provider")
    metadata.update(
        {
            "world_template": WORLD_TEMPLATE,
            "worldgen_structure": structure,
            "worldgen_seed": seed,
            "runtime_bridge_sha256": expected_bridge.upper(),
            "spawn_position": list(_position(spawn_position, "spawn_position")),
        }
    )
    return NativeRegionSnapshot(
        metadata=metadata,
        core_min_chunk_xz=snapshot.core_min_chunk_xz,
        section_known=snapshot.section_known,
        cell_code=snapshot.cell_code,
        cell_palette=snapshot.cell_palette,
        shape_palette=snapshot.shape_palette,
        filler_root_offset_packed=snapshot.filler_root_offset_packed,
    )


def _require_live_v2_identity(
    info: Mapping[str, Any],
    *,
    seed: int,
    structure: str,
    expected_bridge: str,
) -> None:
    require_current_native_runtime_bridge(info, expected=expected_bridge.upper())
    expected = {
        "world_template": WORLD_TEMPLATE,
        "worldgen_provider": WORLDGEN_PROVIDER,
        "worldgen_structure": structure,
        "worldgen_seed": seed,
        "block_ticking_enabled": False,
    }
    mismatches = {
        key: (value, info.get(key))
        for key, value in expected.items()
        if info.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"live V2 Region identity differs: {mismatches}")
    if info.get("worldgen_version") in (None, "", "unknown", "uninitialized"):
        raise RuntimeError("live V2 provider lacks a worldgen version")


def _require_v2_snapshot_identity(
    snapshot: NativeRegionSnapshot,
    *,
    seed: int,
    structure: str,
    expected_bridge: str,
) -> None:
    expected = {
        "seed": seed,
        "worldgen_seed": seed,
        "world_template": WORLD_TEMPLATE,
        "worldgen_provider": WORLDGEN_PROVIDER,
        "worldgen_structure": structure,
        "runtime_bridge_sha256": expected_bridge.upper(),
    }
    mismatches = {
        key: (value, snapshot.metadata.get(key))
        for key, value in expected.items()
        if snapshot.metadata.get(key) != value
    }
    evidence = str(snapshot.metadata.get("native_evidence_jar_sha256", ""))
    if evidence.upper() != expected_bridge.upper():
        mismatches["native_evidence_jar_sha256"] = (
            expected_bridge.upper(),
            evidence,
        )
    try:
        _position(snapshot.metadata.get("spawn_position"), "spawn_position")
    except (TypeError, ValueError):
        mismatches["spawn_position"] = (
            "finite xyz",
            snapshot.metadata.get("spawn_position"),
        )
    if mismatches:
        raise ValueError(f"frozen V2 Region identity differs: {mismatches}")


def _require_sidecar_identity(
    sidecar: NativeRegionBlockSemanticSnapshot,
    snapshot: NativeRegionSnapshot,
    *,
    expected_bridge: str,
) -> None:
    if sidecar.source_region_semantic_sha256 != snapshot.semantic_digest():
        raise ValueError("semantic sidecar names another physical Region")
    if sidecar.evidence_bridge_sha256.upper() != expected_bridge.upper():
        raise ValueError("semantic sidecar names another bridge")
    if not np.array_equal(sidecar.core_min_chunk_xz, snapshot.core_min_chunk_xz):
        raise ValueError("semantic sidecar core differs from physical Region")


def _require_same_snapshot(
    expected: NativeRegionSnapshot,
    actual: NativeRegionSnapshot,
) -> None:
    if expected.metadata != actual.metadata:
        raise ValueError("V2 Region metadata failed round-trip")
    if expected.semantic_artifact_digest() != actual.semantic_artifact_digest():
        raise ValueError("V2 Region geometry failed round-trip")
    for attribute in _PALETTE_ARCHIVE_FIELDS:
        if not np.array_equal(
            getattr(expected.cell_palette, attribute),
            getattr(actual.cell_palette, attribute),
        ):
            raise ValueError(f"V2 Region palette {attribute} failed round-trip")


def _require_same_sidecar(
    expected: NativeRegionBlockSemanticSnapshot,
    actual: NativeRegionBlockSemanticSnapshot,
) -> None:
    if expected.semantic_sha256() != actual.semantic_sha256():
        raise ValueError("V2 block-semantic sidecar failed round-trip")
    if expected.evidence_bridge_sha256 != actual.evidence_bridge_sha256:
        raise ValueError("V2 block-semantic bridge identity failed round-trip")


def _capture_plan(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    output: Path,
    provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "capture-plan.json"
    request: dict[str, Any] = {
        "schema": CAPTURE_PLAN_SCHEMA,
        "version": CAPTURE_PLAN_VERSION,
        "world_template": WORLD_TEMPLATE,
        "worldgen_provider": WORLDGEN_PROVIDER,
        "bridge_jar_sha256": provenance["bridge_jar"]["sha256"],
        "server_jar_sha256": provenance["server_jar"]["sha256"],
        "assets_sha256": provenance["assets"]["sha256"],
        "seeds": list(args.seed),
        "structures": list(args.structures),
        "heldout_count": args.heldout_count,
        "role": args.role,
        "core_min_chunk_xz": [
            args.core_min_chunk_x,
            args.core_min_chunk_z,
        ],
        "split_salt_sha256": None,
        "seed_split": None,
        "capture_contract": {
            "physical_region": "complete_5x5x10_forward_then_reverse",
            "block_identity": "complete_region_block_semantic_sidecar",
            "palette_round_trip_fields": sorted(_PALETTE_ARCHIVE_FIELDS),
            "legacy_hytale_template_unchanged": True,
        },
    }
    if getattr(args, "fluid_semantics", False):
        request["capture_contract"]["stable_fluid_identity"] = (
            "complete_region_fluid_getId_sidecar_forward_then_reverse"
        )
    if path.exists():
        if not args.resume:
            parser.error("capture plan already exists; pass --resume")
        existing = _json_object(path)
        if (
            set(existing) != _PLAN_FIELDS
            or existing.get("schema") != CAPTURE_PLAN_SCHEMA
            or existing.get("version") != CAPTURE_PLAN_VERSION
        ):
            parser.error("existing V2 Region capture plan is unsupported")
        mismatches = {
            field: (request[field], existing.get(field))
            for field in _PLAN_REQUEST_FIELDS
            if request[field] != existing.get(field)
        }
        if existing.get("world_template") != WORLD_TEMPLATE:
            mismatches["world_template"] = (
                WORLD_TEMPLATE,
                existing.get("world_template"),
            )
        if existing.get("worldgen_provider") != WORLDGEN_PROVIDER:
            mismatches["worldgen_provider"] = (
                WORLDGEN_PROVIDER,
                existing.get("worldgen_provider"),
            )
        if existing.get("capture_contract") != request["capture_contract"]:
            mismatches["capture_contract"] = (
                request["capture_contract"],
                existing.get("capture_contract"),
            )
        if args.split_salt_sha256 is not None and (
            args.split_salt_sha256.lower()
            != str(existing.get("split_salt_sha256", "")).lower()
        ):
            mismatches["split_salt_sha256"] = (
                args.split_salt_sha256.lower(),
                existing.get("split_salt_sha256"),
            )
        if mismatches:
            parser.error(f"existing V2 Region capture plan differs: {mismatches}")
        salt = _require_sha256(
            existing.get("split_salt_sha256"),
            "capture-plan salt",
        )
        expected_split = _seed_split(
            args.seed,
            heldout_count=args.heldout_count,
            salt=salt,
        )
        if existing.get("seed_split") != expected_split:
            parser.error("existing V2 Region seed split differs")
        return existing

    request["split_salt_sha256"] = (
        args.split_salt_sha256.lower()
        if args.split_salt_sha256 is not None
        else secrets.token_hex(32)
    )
    request["seed_split"] = _seed_split(
        args.seed,
        heldout_count=args.heldout_count,
        salt=request["split_salt_sha256"],
    )
    _atomic_json(path, request)
    return request


def _resolve_capture_seeds(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[int]:
    """Resolve explicit, ranged, or Studio-planned seed input exactly once."""

    explicit = list(args.seed or [])
    has_range = args.seed_start is not None or args.seed_count is not None
    sources = int(bool(explicit)) + int(has_range) + int(args.seed_plan is not None)
    if sources != 1:
        parser.error(
            "choose exactly one seed source: repeated --seed, "
            "--seed-start/--seed-count, or --seed-plan"
        )
    if explicit:
        if args.seed_stride != 1:
            parser.error("--seed-stride is only valid with --seed-start")
        return explicit
    if has_range:
        if args.seed_start is None or args.seed_count is None:
            parser.error("--seed-start and --seed-count must be provided together")
        if not 3 <= args.seed_count <= 4096:
            parser.error("--seed-count must be between 3 and 4096")
        if args.seed_stride < 1:
            parser.error("--seed-stride must be positive")
        last = args.seed_start + (args.seed_count - 1) * args.seed_stride
        if args.seed_start < 0 or last >= 2**31:
            parser.error("seed range must remain in [0, 2^31)")
        return [
            args.seed_start + index * args.seed_stride
            for index in range(args.seed_count)
        ]

    if args.seed_stride != 1:
        parser.error("--seed-stride is only valid with --seed-start")
    try:
        value = json.loads(args.seed_plan.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        parser.error(f"could not read --seed-plan: {exc}")
    if not isinstance(value, dict):
        parser.error("--seed-plan must be a JSON object")
    plan = value.get("native_seed_plan", value)
    if not isinstance(plan, dict) or not isinstance(plan.get("seeds"), list):
        parser.error("--seed-plan must contain native_seed_plan.seeds")
    seeds = plan["seeds"]
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        parser.error("--seed-plan seeds must be integers")
    return list(seeds)


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if len(args.seed) < 3 or len(args.seed) > 4096 or len(set(args.seed)) != len(args.seed):
        parser.error("seed input requires 3 to 4096 unique values")
    if any(isinstance(seed, bool) or not 0 <= seed < 2**31 for seed in args.seed):
        parser.error("--seed values must be in [0, 2^31)")
    if not 0 < args.heldout_count < len(args.seed) - 1:
        parser.error(
            "--heldout-count must leave at least two calibration seeds"
        )
    if not args.structures or len(set(args.structures)) != len(args.structures):
        parser.error("--structures must be nonempty and unique")
    if any(
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        for value in args.structures
    ):
        parser.error("--structures values must be nonempty and trimmed")
    try:
        structure_slugs = [_structure_slug(value) for value in args.structures]
    except ValueError as error:
        parser.error(str(error))
    if len(set(structure_slugs)) != len(structure_slugs):
        parser.error("--structures values collide after path normalization")
    if (
        not isinstance(args.role, str)
        or not args.role.strip()
        or args.role != args.role.strip()
    ):
        parser.error("--role must be nonempty and trimmed")
    if (args.core_min_chunk_x is None) != (args.core_min_chunk_z is None):
        parser.error("--core-min-chunk-x and --core-min-chunk-z go together")
    if args.split_salt_sha256 is not None:
        try:
            _require_sha256(args.split_salt_sha256, "--split-salt-sha256")
        except ValueError as error:
            parser.error(str(error))


def _seed_split(
    seeds: Sequence[int],
    *,
    heldout_count: int,
    salt: str,
) -> dict[str, str]:
    """Assign held-out identity once by seed, independently of structure."""

    normalized_salt = _require_sha256(salt, "seed split salt")
    ranked = sorted(
        (int(seed) for seed in seeds),
        key=lambda seed: (
            hashlib.sha256(
                (
                    "hytalerl_worldgen_v2_seed_split_v1\0"
                    f"{normalized_salt}\0{seed}"
                ).encode()
            ).digest(),
            seed,
        ),
    )
    heldout = set(ranked[:heldout_count])
    return {
        str(seed): "heldout" if seed in heldout else "calibration"
        for seed in seeds
    }


def _spawn_position(observation: Mapping[str, Any]) -> tuple[float, float, float]:
    if not isinstance(observation, Mapping):
        raise TypeError("native reset observation must be a mapping")
    return _position(observation.get("position"), "reset spawn_position")


def _require_live_spawn(
    snapshot: NativeRegionSnapshot,
    observation: Mapping[str, Any],
) -> None:
    frozen = _position(snapshot.metadata.get("spawn_position"), "frozen spawn")
    live = _spawn_position(observation)
    if live != frozen:
        raise RuntimeError(
            f"live V2 spawn differs from frozen Region: live={live}, frozen={frozen}"
        )


def _position(value: object, label: str) -> tuple[float, float, float]:
    try:
        position = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric xyz") from error
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"{label} must be finite xyz")
    return tuple(float(component) for component in position)


def _split_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    asset_keys_by_semantic: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("calibration", "heldout"):
        selected = [row for row in rows if row.get("split") == split]
        if not selected:
            continue
        unique_counts = np.asarray(
            [row["unique_local_block_ids"] for row in selected],
            dtype=np.int64,
        )
        physical_counts = np.asarray(
            [row["unique_physical_cell_semantics"] for row in selected],
            dtype=np.int64,
        )
        union_count: int | None = None
        if asset_keys_by_semantic is not None:
            union: set[str] = set()
            for row in selected:
                union.update(
                    asset_keys_by_semantic[row["region_semantic_sha256"]]
                )
            union_count = len(union)
        result[split] = {
            "artifact_count": len(selected),
            "unique_local_block_ids": _integer_summary(unique_counts),
            "unique_physical_cell_semantics": _integer_summary(
                physical_counts
            ),
            "unique_block_asset_ids_union": union_count,
            "regions_with_fluid": sum(
                row["hazards"]["fluid_cells"] > 0 for row in selected
            ),
            "regions_with_damaging_flags": sum(
                row["hazards"]["damaging_flag_cells"] > 0
                for row in selected
            ),
            "regions_with_nonzero_block_damage": sum(
                row["hazards"]["nonzero_block_damage_cells"] > 0
                for row in selected
            ),
            "regions_with_nonzero_fluid_damage": sum(
                row["hazards"]["nonzero_fluid_damage_cells"] > 0
                for row in selected
            ),
        }
        fluid_rows = [
            row["fluid_semantics"]
            for row in selected
            if "fluid_semantics" in row
        ]
        if fluid_rows:
            stable_counts: dict[str, int] = {}
            for fluid in fluid_rows:
                for asset_id, count in fluid["full_asset_cell_counts"].items():
                    stable_counts[asset_id] = (
                        stable_counts.get(asset_id, 0) + int(count)
                    )
            result[split]["stable_fluid_identity"] = {
                "sidecar_count": len(fluid_rows),
                "complete_for_split": len(fluid_rows) == len(selected),
                "unique_asset_ids": len(stable_counts),
                "full_asset_cell_counts": dict(sorted(stable_counts.items())),
                "regions_with_named_fluid": sum(
                    fluid["full_fluid_cells"] > 0 for fluid in fluid_rows
                ),
            }
    return result


def _integer_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "minimum": int(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": int(np.max(values)),
    }


def _weighted_count(counts: np.ndarray, mask: np.ndarray) -> int:
    return int(np.sum(counts[mask], dtype=np.int64))


def _asset_key_hex(words: Iterable[int]) -> str:
    return np.asarray(tuple(words), dtype=">u4").tobytes().hex()


def _progress(structure: str, seed: int, *, phase: str):
    def report(done: int, total: int, _task: object) -> None:
        if done in (1, total) or done % 50 == 0:
            print(
                f"structure={structure} seed={seed} phase={phase} "
                f"section={done}/{total}",
                flush=True,
            )

    return report


def _structure_slug(structure: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", structure).strip("._")
    if not slug:
        raise ValueError(f"worldgen structure has no safe path name: {structure!r}")
    return slug


def _provenance(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"provenance file not found: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _report_sha256(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_semantic_sha256", None)
    return _canonical_sha256(payload)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _json_object(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}: {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
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
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
