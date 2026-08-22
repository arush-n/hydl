"""Capture stable native fluid IDs aligned to one frozen V2 Region artifact.

This is an additive experimental sidecar.  It does not change the physical
Region schema and it refuses to name a fluid unless the bridge supplies the
stable Hytale ``Fluid.getId()`` string for every physical fluid cell.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from typing import Any

import numpy as np

from _workspace import use_workspace_hytalerl


use_workspace_hytalerl()

from hytalegym.envs.hytale_env import HytaleEnv  # noqa: E402
from hytalegym.geometry.contract import FLAG_FLUID  # noqa: E402
from hytalegym.worldgen.region import (  # noqa: E402
    CAPTURE_CHUNKS_PER_AXIS,
    SECTION_VOLUME,
    NativeRegionSnapshot,
    RegionCapturePlan,
    capture_chunk_slot,
)


SCHEMA = "hytalerl_worldgen_v2_region_fluid_identity_probe_v1"
VERSION = 1
SECTION_SCHEMA = "hytalerl_native_region_fluid_semantics_v1"
SECTION_VERSION = 1
CODE_ENCODING = "uint16_le_y_z_x"
CODE_BYTES = SECTION_VOLUME * np.dtype("<u2").itemsize
WORLD_TEMPLATE = "hytale_generator"
WORLDGEN_PROVIDER = "HytaleGenerator"


def parse_fluid_semantic_section(
    response: Mapping[str, Any],
    *,
    expected_chunk_x: int,
    expected_chunk_z: int,
    expected_section_y: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Validate one wire response and return codes plus stable asset IDs."""

    raw = dict(response)
    expected = {
        "type": "region_fluid_semantics",
        "schema": SECTION_SCHEMA,
        "version": SECTION_VERSION,
        "chunk_x": expected_chunk_x,
        "chunk_z": expected_chunk_z,
        "section_y": expected_section_y,
        "code_encoding": CODE_ENCODING,
    }
    mismatches = {
        key: (value, raw.get(key))
        for key, value in expected.items()
        if raw.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Region fluid-semantic identity changed: {mismatches}")

    encoded = raw.get("cell_codes")
    if not isinstance(encoded, (bytes, bytearray, memoryview)):
        raise ValueError("Region fluid-semantic codes must be MessagePack binary")
    view = memoryview(encoded)
    if view.nbytes != CODE_BYTES:
        raise ValueError("Region fluid-semantic code payload has the wrong size")
    codes = np.frombuffer(view, dtype="<u2", count=SECTION_VOLUME).copy()

    raw_palette = raw.get("palette")
    if not isinstance(raw_palette, Sequence) or isinstance(
        raw_palette,
        (str, bytes, bytearray),
    ):
        raise ValueError("Region fluid-semantic palette must be an array")
    palette = tuple(raw_palette)
    if not palette or len(palette) > (1 << 16):
        raise ValueError("Region fluid-semantic palette exceeds uint16 capacity")
    if not all(isinstance(asset_id, str) for asset_id in palette):
        raise ValueError("Region fluid-semantic asset IDs must be strings")
    if palette[0] != "":
        raise ValueError("Region fluid-semantic palette zero must be empty")
    if any(not asset_id or asset_id != asset_id.strip() for asset_id in palette[1:]):
        raise ValueError("Region fluid-semantic nonzero asset ID is invalid")
    if len(set(palette)) != len(palette):
        raise ValueError("Region fluid-semantic palette contains duplicate IDs")
    if np.any(codes.astype(np.uint32) >= len(palette)):
        raise ValueError("Region fluid-semantic code exceeds its palette")
    return codes, palette


def aligned_section_metrics(
    codes: np.ndarray,
    palette: Sequence[str],
    physical_fluid: np.ndarray,
) -> dict[str, Any]:
    """Require exact physical alignment and count stable fluid identities."""

    code = np.asarray(codes, dtype=np.uint16)
    physical = np.asarray(physical_fluid, dtype=np.bool_)
    if code.shape != (SECTION_VOLUME,) or physical.shape != (SECTION_VOLUME,):
        raise ValueError("Region fluid-semantic alignment requires one section")
    semantic_fluid = code != 0
    if not np.array_equal(semantic_fluid, physical):
        difference = int(np.count_nonzero(semantic_fluid != physical))
        raise ValueError(
            "Region fluid identity differs from physical FLAG_FLUID at "
            f"{difference} cells"
        )
    used, counts = np.unique(code, return_counts=True)
    identities = {
        str(palette[int(index)]): int(count)
        for index, count in zip(used, counts, strict=True)
        if int(index) != 0
    }
    return {
        "fluid_cells": int(np.count_nonzero(semantic_fluid)),
        "asset_cell_counts": identities,
        "section_semantic_sha256": _section_digest(code, palette),
    }


def run(args: argparse.Namespace) -> Path:
    reference = NativeRegionSnapshot.load(args.reference_region)
    metadata = reference.metadata
    structure = str(metadata.get("worldgen_structure", ""))
    seed = int(metadata.get("worldgen_seed", metadata.get("seed", -1)))
    if structure != args.structure or seed != args.seed:
        raise ValueError(
            "reference Region identity differs: "
            f"expected {args.structure}/{args.seed}, got {structure}/{seed}"
        )
    bridge_sha = _file_sha256(args.bridge_jar)
    core_x = int(reference.core_min_chunk_xz[0])
    core_z = int(reference.core_min_chunk_xz[1])
    plan = RegionCapturePlan.complete(
        core_x,
        core_z,
        maximum_source_reach_blocks=0.0,
    )
    physical_total, physical_core = _physical_fluid_counts(reference)

    env = HytaleEnv(
        task="kill_trork",
        backend="native",
        world=WORLD_TEMPLATE,
        worldgen_structure=structure,
        host=args.host,
        port=args.port,
        ticks_per_step=1,
        max_episode_steps=100,
        npc_role="Kweebec_Razorleaf",
        combat_target_active=False,
        fidelity_fixture="static_region",
    )
    try:
        _, info = env.reset(seed=seed)
        _require_reset_identity(info, structure, seed, bridge_sha)
        manifest = env.capture_region_manifest(core_x, core_z)
        _require_manifest_identity(manifest, reference, structure, seed)
        forward = _capture_pass(env, reference, plan.tasks)
        reverse = _capture_pass(env, reference, tuple(reversed(plan.tasks)))
    finally:
        env.close()

    if forward != reverse:
        raise ValueError("forward/reverse Region fluid semantics differ")
    if forward["full_fluid_cells"] != physical_total:
        raise ValueError("full fluid identity count differs from physical Region")
    if forward["core_fluid_cells"] != physical_core:
        raise ValueError("core fluid identity count differs from physical Region")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "world_template": WORLD_TEMPLATE,
        "worldgen_provider": WORLDGEN_PROVIDER,
        "worldgen_structure": structure,
        "seed": seed,
        "core_min_chunk_x": core_x,
        "core_min_chunk_z": core_z,
        "reference_region": {
            "path": str(args.reference_region.resolve()),
            "file_sha256": _file_sha256(args.reference_region),
            "semantic_sha256": reference.semantic_digest(),
            "physical_full_fluid_cells": physical_total,
            "physical_core_fluid_cells": physical_core,
        },
        "bridge_jar": {
            "path": str(args.bridge_jar.resolve()),
            "sha256": bridge_sha,
        },
        "section_protocol": {
            "schema": SECTION_SCHEMA,
            "version": SECTION_VERSION,
            "code_encoding": CODE_ENCODING,
            "identity": "stable_hytale_fluid_getId_string",
            "physical_alignment": "exact_FLAG_FLUID_cell_equality",
            "same_world_stability": "forward_reverse_exact",
        },
        "measurement": forward,
        "observability": {
            "measured": "stable fluid asset identity and cell occupancy",
            "not_inferred": "fluid identity from structure name, damage, or block layer",
        },
        "report_semantic_sha256": "",
    }
    report["report_semantic_sha256"] = _report_digest(report)
    _atomic_json(args.output, report)
    return args.output


def _capture_pass(
    env: HytaleEnv,
    reference: NativeRegionSnapshot,
    tasks: Sequence[Any],
) -> dict[str, Any]:
    connection = getattr(env, "_connection", None)
    if connection is None:
        raise RuntimeError("native connection is unavailable after reset")
    core_x = int(reference.core_min_chunk_xz[0])
    core_z = int(reference.core_min_chunk_xz[1])
    full_counts: Counter[str] = Counter()
    core_counts: Counter[str] = Counter()
    section_digests: dict[tuple[int, int, int], str] = {}
    for task in tasks:
        response = connection.send_and_recv(
            {
                "type": "region_fluid_semantics",
                "chunk_x": int(task.chunk_x),
                "chunk_z": int(task.chunk_z),
                "section_y": int(task.section_y),
            }
        )
        codes, palette = parse_fluid_semantic_section(
            response,
            expected_chunk_x=int(task.chunk_x),
            expected_chunk_z=int(task.chunk_z),
            expected_section_y=int(task.section_y),
        )
        slot = capture_chunk_slot(
            int(task.chunk_x),
            int(task.chunk_z),
            core_x - 1,
            core_z - 1,
        )
        physical_codes = reference.cell_code[slot, int(task.section_y)]
        physical_fluid = (
            reference.cell_palette.flags[physical_codes]
            & np.uint16(FLAG_FLUID)
        ) != 0
        metrics = aligned_section_metrics(codes, palette, physical_fluid)
        full_counts.update(metrics["asset_cell_counts"])
        if (
            core_x <= int(task.chunk_x) < core_x + 3
            and core_z <= int(task.chunk_z) < core_z + 3
        ):
            core_counts.update(metrics["asset_cell_counts"])
        key = (int(task.chunk_x), int(task.chunk_z), int(task.section_y))
        section_digests[key] = metrics["section_semantic_sha256"]
    ordered_sections = [
        {
            "chunk_x": key[0],
            "chunk_z": key[1],
            "section_y": key[2],
            "semantic_sha256": section_digests[key],
        }
        for key in sorted(section_digests)
    ]
    return {
        "section_count": len(section_digests),
        "full_fluid_cells": sum(full_counts.values()),
        "core_fluid_cells": sum(core_counts.values()),
        "full_asset_cell_counts": dict(sorted(full_counts.items())),
        "core_asset_cell_counts": dict(sorted(core_counts.items())),
        "section_semantic_sha256": ordered_sections,
        "region_fluid_semantic_sha256": _canonical_json_digest(
            {
                "full": dict(sorted(full_counts.items())),
                "core": dict(sorted(core_counts.items())),
                "sections": ordered_sections,
            }
        ),
    }


def _physical_fluid_counts(reference: NativeRegionSnapshot) -> tuple[int, int]:
    flags = reference.cell_palette.flags[reference.cell_code]
    fluid = (flags & np.uint16(FLAG_FLUID)) != 0
    full = int(np.count_nonzero(fluid))
    core_slots = []
    for chunk_z in range(1, CAPTURE_CHUNKS_PER_AXIS - 1):
        for chunk_x in range(1, CAPTURE_CHUNKS_PER_AXIS - 1):
            core_slots.append(chunk_z * CAPTURE_CHUNKS_PER_AXIS + chunk_x)
    core = int(np.count_nonzero(fluid[np.asarray(core_slots, dtype=np.intp)]))
    return full, core


def _require_reset_identity(
    info: Mapping[str, Any],
    structure: str,
    seed: int,
    bridge_sha: str,
) -> None:
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
    observed_bridge = str(info.get("bridge_sha256", "")).upper()
    if observed_bridge != bridge_sha:
        mismatches["bridge_sha256"] = (bridge_sha, observed_bridge)
    if mismatches:
        raise ValueError(f"live V2 reset identity differs: {mismatches}")


def _require_manifest_identity(
    manifest: Mapping[str, Any],
    reference: NativeRegionSnapshot,
    structure: str,
    seed: int,
) -> None:
    expected = {
        "seed": seed,
        "worldgen_provider": WORLDGEN_PROVIDER,
        "worldgen_version": reference.metadata.get("worldgen_version"),
        "core_min_chunk_x": int(reference.core_min_chunk_xz[0]),
        "core_min_chunk_z": int(reference.core_min_chunk_xz[1]),
    }
    mismatches = {
        key: (value, manifest.get(key))
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"live Region manifest differs: {mismatches}")
    if structure != reference.metadata.get("worldgen_structure"):
        raise ValueError("reference Region structure identity changed")


def _section_digest(codes: np.ndarray, palette: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(SECTION_SCHEMA.encode("ascii"))
    digest.update(struct.pack("<I", SECTION_VERSION))
    digest.update(np.asarray(codes, dtype="<u2").tobytes(order="C"))
    digest.update(struct.pack("<I", len(palette)))
    for asset_id in palette:
        encoded = asset_id.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _report_digest(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_semantic_sha256", None)
    payload.pop("captured_at_utc", None)
    return _canonical_json_digest(payload)


def _canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")
        temporary = Path(destination.name)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--reference-region", type=Path, required=True)
    parser.add_argument("--bridge-jar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    for path, label in (
        (args.reference_region, "reference Region"),
        (args.bridge_jar, "bridge JAR"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    output = run(args)
    print(output)


if __name__ == "__main__":
    main()
