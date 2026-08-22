"""Freeze and validate live evidence for a compiled procedural V2 graph.

Compilation and live certification are deliberately separate.  A compiler
receipt proves deterministic authoring against pinned inputs; this receipt
proves that one exact pack was decoded, selected, generated, replayed and
captured by the pinned native runtime.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping


LIVE_VALIDATION_SCHEMA = (
    "hytalerl_worldgen_v2_procedural_native_validation_v1"
)
LIVE_VALIDATION_VERSION = 1
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    copied = dict(value)
    copied["semantic_sha256"] = ""
    return hashlib.sha256(_canonical_bytes(copied)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _safe_relative(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe evidence path: {relative}")
    return relative


def _rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(row, dict) for row in value
    ):
        raise ValueError(f"{label} must be a list of objects")
    return value


def _finite(values: list[float], label: str) -> list[float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} has no finite measurements")
    return values


def _find_structure(value: Mapping[str, Any], asset_id: str) -> dict[str, Any]:
    rows = _rows(value.get("structures"), "structures")
    matches = [row for row in rows if row.get("structure") == asset_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one structure row for {asset_id}, found {len(matches)}"
        )
    return matches[0]


def _live_claims(
    *,
    pack_root: Path,
    pilot: Mapping[str, Any],
    capture: Mapping[str, Any],
    analysis: Mapping[str, Any],
    server_log: Path,
) -> dict[str, Any]:
    # Local import avoids a native_graph_compiler -> design_compiler cycle.
    from design_compiler import validate_pack

    compiled = validate_pack(pack_root)
    if compiled["native_profile"] != "procedural_graph_v1":
        raise ValueError("live evidence pack is not procedural_graph_v1")
    asset_id = str(compiled["asset_id"])
    authoring = _object(
        pack_root / "studio-authoring-receipt.json",
        "Studio authoring receipt",
    )
    manifest = _object(pack_root / "manifest.json", "asset-pack manifest")

    if pilot.get("schema") != "hytalerl_worldgen_v2_native_pilot_v1":
        raise ValueError("unsupported native pilot report")
    pilot_captures = _rows(pilot.get("captures"), "pilot captures")
    custom = [row for row in pilot_captures if row.get("structure") == asset_id]
    controls = [row for row in pilot_captures if row.get("structure") == "Default"]
    replay = pilot.get("replay")
    if len(custom) != 1 or len(controls) != 1 or not isinstance(replay, dict):
        raise ValueError("pilot must contain custom, Default and replay rows")
    custom_row = custom[0]
    control_row = controls[0]
    if replay.get("structure") != asset_id:
        raise ValueError("pilot replay does not select the compiled structure")
    if custom_row.get("local_frame_sha256") != replay.get("local_frame_sha256"):
        raise ValueError("same-seed custom pilot replay differs")
    if custom_row.get("local_frame_sha256") == control_row.get(
        "local_frame_sha256"
    ):
        raise ValueError("compiled graph silently matches the Default frame")

    if capture.get("schema") != "hytalerl_worldgen_v2_region_report_v1":
        raise ValueError("unsupported Region capture report")
    capture_structure = _find_structure(capture, asset_id)
    capture_artifacts = _rows(
        capture_structure.get("artifacts"), "Region capture artifacts"
    )
    if len(capture_artifacts) < 3:
        raise ValueError("live graph certification requires at least three seeds")
    seeds = [int(row["seed"]) for row in capture_artifacts]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Region certification contains duplicate seeds")
    region_hashes = {row.get("region_semantic_sha256") for row in capture_artifacts}
    block_hashes = {row.get("block_semantic_sha256") for row in capture_artifacts}
    fluid_hashes = {
        row.get("fluid_semantics", {}).get("semantic_sha256")
        for row in capture_artifacts
    }
    if not all(len(hashes) == len(seeds) for hashes in (
        region_hashes, block_hashes, fluid_hashes
    )):
        raise ValueError("distinct seeds did not produce distinct Region semantics")
    if not all(
        row.get("hazards", {}).get("palette_fields_complete") is True
        for row in capture_artifacts
    ):
        raise ValueError("captured Region palette fields are incomplete")

    if analysis.get("schema") != "hytalerl_worldgen_v2_region_analysis_v2":
        raise ValueError("unsupported Region analysis report")
    analysis_structure = _find_structure(analysis, asset_id)
    analyzed = _rows(analysis_structure.get("artifacts"), "analysis artifacts")
    if {int(row["seed"]) for row in analyzed} != set(seeds):
        raise ValueError("analysis and capture seed sets differ")
    pairwise = _rows(
        analysis_structure.get("pairwise_surface_comparison"),
        "pairwise surface comparisons",
    )
    expected_pairs = len(seeds) * (len(seeds) - 1) // 2
    if len(pairwise) != expected_pairs:
        raise ValueError("analysis does not cover every seed pair")

    differing = _finite(
        [float(row["differing_surface_fraction"]) for row in pairwise],
        "pairwise surface differences",
    )
    height_delta = _finite(
        [float(row["absolute_height_delta"]["mean"]) for row in pairwise],
        "pairwise height deltas",
    )
    relief = _finite(
        [float(row["terrain"]["relief_blocks"]) for row in analyzed],
        "terrain relief",
    )
    water = _finite(
        [float(row["fluids_and_hazards"]["fluid_fraction"]) for row in analyzed],
        "fluid fractions",
    )
    cave_proxy = _finite(
        [
            float(row["cave_overhang_proxy"]["subsurface_void_fraction"])
            for row in analyzed
        ],
        "cave proxies",
    )
    material_counts = [
        int(row["materials"]["unique_non_air_asset_ids"]) for row in analyzed
    ]
    if not all(
        row.get("spawn", {}).get("inside_measured_core") is True
        and float(
            row["spawn"]["feet_y_minus_surface_cell_ceiling"]
        ) == 0.0
        for row in analyzed
    ):
        raise ValueError("one or more native spawns are not on the measured surface")

    text = server_log.read_text(encoding="utf-8", errors="replace")
    lines = [_ANSI.sub("", line) for line in text.splitlines()]
    pack_marker = f"Loaded pack: HytaleRL:{manifest['Name']}"
    if not any(pack_marker in line for line in lines):
        raise ValueError("server log does not show the compiled asset pack loading")
    if not any(
        f"Loaded World Structure {asset_id}:" in line for line in lines
    ):
        raise ValueError("server log does not show the compiled structure loading")
    if not any("Hytale Server Booted!" in line for line in lines):
        raise ValueError("server log never reached the ready state")
    diagnostic_words = (" WARN]", " ERROR]", " SEVERE]", "Exception")
    generated_diagnostics = [
        line for line in lines
        if asset_id in line and any(word in line for word in diagnostic_words)
    ]
    if generated_diagnostics:
        raise ValueError(
            "server emitted diagnostics for the generated asset: "
            + " | ".join(generated_diagnostics)
        )

    observability = analysis.get("observability", {})
    unmeasured = observability.get("unmeasured", {})
    if not isinstance(unmeasured, dict):
        raise ValueError("analysis omitted its observability boundary")

    return {
        "asset_id": asset_id,
        "environment_id": compiled["environment_id"],
        "design_digest": compiled["design_digest"],
        "pack_semantic_sha256": compiled["pack_semantic_sha256"],
        "authoring_receipt_semantic_sha256": compiled[
            "receipt_semantic_sha256"
        ],
        "density_graph_semantic_sha256": authoring["native_graph"][
            "density_graph_semantic_sha256"
        ],
        "graph_contract_semantic_sha256": authoring["native_graph"][
            "semantic_sha256"
        ],
        "native_profile": compiled["native_profile"],
        "native_codec_acceptance": "passed_hytale_0.5.7_reference_pack",
        "provider_selection": "hytale_generator/HytaleGenerator",
        "legacy_world_identity_unchanged": True,
        "same_seed_replay": True,
        "different_from_default_control": True,
        "seed_count": len(seeds),
        "seeds": sorted(seeds),
        "distinct_region_semantics": len(region_hashes),
        "distinct_block_semantics": len(block_hashes),
        "distinct_fluid_semantics": len(fluid_hashes),
        "pairwise_surface_differing_fraction": {
            "minimum": min(differing),
            "maximum": max(differing),
        },
        "pairwise_mean_absolute_height_delta_blocks": {
            "minimum": min(height_delta),
            "maximum": max(height_delta),
        },
        "relief_blocks": {"minimum": min(relief), "maximum": max(relief)},
        "fluid_fraction": {"minimum": min(water), "maximum": max(water)},
        "subsurface_void_fraction_proxy": {
            "minimum": min(cave_proxy),
            "maximum": max(cave_proxy),
        },
        "unique_non_air_material_ids": {
            "minimum": min(material_counts),
            "maximum": max(material_counts),
        },
        "safe_surface_spawns": len(seeds),
        "palette_fields_complete": True,
        "stable_fluid_identity_complete": True,
        "generated_asset_diagnostics": [],
        "not_certified": unmeasured,
        "pack_specific_validation_required_for_new_compilations": True,
    }


def build_live_validation(
    *,
    evidence_root: str | Path,
    pack: str | Path,
    pilot_report: str | Path,
    capture_report: str | Path,
    analysis_report: str | Path,
    server_log: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Build one immutable validation receipt from already captured evidence."""

    root = Path(evidence_root).resolve()
    output_path = Path(output).resolve()
    if output_path.parent != root:
        raise ValueError("live validation receipt must be written at evidence root")
    sources = {
        "pack": Path(pack).resolve(),
        "pilot_report": Path(pilot_report).resolve(),
        "capture_report": Path(capture_report).resolve(),
        "analysis_report": Path(analysis_report).resolve(),
        "server_log": Path(server_log).resolve(),
    }
    for label, path in sources.items():
        if label == "pack":
            if not path.is_dir():
                raise ValueError(f"{label} directory is missing: {path}")
        elif not path.is_file():
            raise ValueError(f"{label} file is missing: {path}")
        _safe_relative(path, root)

    claims = _live_claims(
        pack_root=sources["pack"],
        pilot=_object(sources["pilot_report"], "native pilot report"),
        capture=_object(sources["capture_report"], "Region capture report"),
        analysis=_object(sources["analysis_report"], "Region analysis report"),
        server_log=sources["server_log"],
    )
    evidence_files = {
        _safe_relative(path, root): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() != output_path
    }
    receipt: dict[str, Any] = {
        "schema": LIVE_VALIDATION_SCHEMA,
        "version": LIVE_VALIDATION_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sources": {
            label: _safe_relative(path, root)
            for label, path in sources.items()
        },
        "claims": claims,
        "evidence_files": evidence_files,
        "semantic_sha256": "",
    }
    receipt["semantic_sha256"] = _semantic_sha256(receipt)
    _atomic_json(output_path, receipt)
    return validate_live_validation(output_path)


def validate_live_validation(path: str | Path) -> dict[str, Any]:
    """Validate a frozen live receipt and every file it binds."""

    receipt_path = Path(path).resolve()
    receipt = _object(receipt_path, "native graph live validation receipt")
    if receipt.get("schema") != LIVE_VALIDATION_SCHEMA:
        raise ValueError("unsupported native graph live validation schema")
    if receipt.get("version") != LIVE_VALIDATION_VERSION:
        raise ValueError("unsupported native graph live validation version")
    if receipt.get("semantic_sha256") != _semantic_sha256(receipt):
        raise ValueError("native graph live validation semantic hash differs")
    expected = receipt.get("evidence_files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("native graph live validation has no evidence files")
    root = receipt_path.parent
    observed = {
        _safe_relative(candidate, root): _file_sha256(candidate)
        for candidate in sorted(root.rglob("*"))
        if candidate.is_file() and candidate.resolve() != receipt_path
    }
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        added = sorted(set(observed) - set(expected))
        changed = sorted(
            key for key in set(expected) & set(observed)
            if expected[key] != observed[key]
        )
        raise ValueError(
            "native graph live evidence files differ: "
            f"missing={missing}, added={added}, changed={changed}"
        )
    sources = receipt.get("sources", {})
    if not isinstance(sources, dict):
        raise ValueError("native graph live validation sources are invalid")
    claims = _live_claims(
        pack_root=root / PurePosixPath(sources["pack"]),
        pilot=_object(
            root / PurePosixPath(sources["pilot_report"]),
            "native pilot report",
        ),
        capture=_object(
            root / PurePosixPath(sources["capture_report"]),
            "Region capture report",
        ),
        analysis=_object(
            root / PurePosixPath(sources["analysis_report"]),
            "Region analysis report",
        ),
        server_log=root / PurePosixPath(sources["server_log"]),
    )
    if receipt.get("claims") != claims:
        raise ValueError("native graph live validation claims differ")
    return receipt


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


__all__ = [
    "LIVE_VALIDATION_SCHEMA",
    "LIVE_VALIDATION_VERSION",
    "build_live_validation",
    "validate_live_validation",
]
