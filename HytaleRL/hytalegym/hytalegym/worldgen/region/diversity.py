"""Held-out geometric coverage diagnostics for exact Region libraries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from hytalegym.geometry.contract import (
    FLAG_BOUNCY,
    FLAG_CLIMBABLE,
    FLAG_DAMAGING,
    FLAG_FLUID,
    FLAG_HAS_FLUID_MOVEMENT_SETTINGS,
    FLAG_HAS_MOVEMENT_SETTINGS,
    FLAG_OPAQUE,
    FLAG_PROTRUDES_CELL,
    FLAG_SOLID,
    FLAG_TRIGGER,
)
from hytalegym.worldgen.region.contract import (
    CAPTURE_CHUNKS_PER_AXIS,
    CAPTURE_HALO_CHUNKS,
    CORE_CHUNKS_PER_AXIS,
    HEIGHT_SECTIONS,
)
from hytalegym.worldgen.region.library import RegionArtifactLibrary
from hytalegym.worldgen.region.snapshot import NativeRegionSnapshot

REGION_LIBRARY_DIVERSITY_SCHEMA = "hytalerl_region_library_diversity_v1"
REGION_LIBRARY_DIVERSITY_VERSION = 1
_REPORT_FIELDS = {
    "schema",
    "version",
    "library_semantic_sha256",
    "metric",
    "feature_contract",
    "split",
    "geometric_coverage_proxy",
    "policy_generalization",
    "report_semantic_sha256",
}
_FLAGS = (
    FLAG_SOLID,
    FLAG_OPAQUE,
    FLAG_FLUID,
    FLAG_DAMAGING,
    FLAG_CLIMBABLE,
    FLAG_BOUNCY,
    FLAG_TRIGGER,
    FLAG_PROTRUDES_CELL,
    FLAG_HAS_MOVEMENT_SETTINGS,
    FLAG_HAS_FLUID_MOVEMENT_SETTINGS,
)


def region_library_diversity_report(
    library: RegionArtifactLibrary,
) -> dict[str, Any]:
    """Measure held-out world coverage without claiming policy generalization."""

    if not isinstance(library, RegionArtifactLibrary):
        raise TypeError("library must be a RegionArtifactLibrary")
    vectors: dict[str, list[np.ndarray]] = {"train": [], "heldout": []}
    semantics: dict[str, list[str]] = {"train": [], "heldout": []}
    for split in ("train", "heldout"):
        entries = library.select(
            split,
            len(library.entries_for_split(split)),
            selection_key=0,
        )
        snapshots = library.load_selection(
            split,
            len(entries),
            selection_key=0,
        )
        vectors[split] = [_feature_vector(snapshot) for snapshot in snapshots]
        semantics[split] = [entry.semantic_sha256 for entry in entries]

    train = np.stack(vectors["train"])
    heldout = np.stack(vectors["heldout"])
    train_nearest = _leave_one_out_nearest(train)
    heldout_nearest = _nearest(heldout, train)
    train_p95 = float(np.quantile(train_nearest, 0.95))
    train_median = float(np.median(train_nearest))
    heldout_median = float(np.median(heldout_nearest))
    result: dict[str, Any] = {
        "schema": REGION_LIBRARY_DIVERSITY_SCHEMA,
        "version": REGION_LIBRARY_DIVERSITY_VERSION,
        "library_semantic_sha256": library.semantic_sha256,
        "metric": "normalized_euclidean_v1",
        "feature_contract": {
            "global_scope": "complete_5x5x10_capture",
            "global_flag_fractions": len(_FLAGS),
            "global_collision_fractions": 2,
            "core_section_shape": [
                CORE_CHUNKS_PER_AXIS,
                HEIGHT_SECTIONS,
                CORE_CHUNKS_PER_AXIS,
            ],
            "core_section_channels": ["solid_fraction", "fluid_fraction"],
            "feature_count": int(train.shape[1]),
            "position_scale": "native_32x32x32_sections",
        },
        "split": {
            "train_count": int(train.shape[0]),
            "heldout_count": int(heldout.shape[0]),
            "train_semantic_sha256": semantics["train"],
            "heldout_semantic_sha256": semantics["heldout"],
        },
        "geometric_coverage_proxy": {
            "train_leave_one_out_nearest": _summary(train_nearest),
            "heldout_to_train_nearest": _summary(heldout_nearest),
            "train_p95_threshold": train_p95,
            "heldout_within_train_p95_fraction": float(
                np.mean(heldout_nearest <= train_p95)
            ),
            "heldout_to_train_median_ratio": (
                heldout_median / train_median
                if train_median > 0.0
                else None
            ),
            "identical_feature_vectors": _duplicate_feature_count(
                np.concatenate((train, heldout), axis=0)
            ),
        },
        "policy_generalization": {
            "status": "unmeasured_requires_policy_consumer",
            "warning": (
                "geometric coverage is a library diagnostic, not a policy "
                "held-out return"
            ),
        },
        "report_semantic_sha256": "",
    }
    result["report_semantic_sha256"] = _report_digest(result)
    return result


def save_region_library_diversity_report(
    library: RegionArtifactLibrary,
    path: str | Path,
) -> Path:
    """Write the canonical diagnostic after all artifact checks pass."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = region_library_diversity_report(library)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(report, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_region_library_diversity_report(
    path: str | Path,
    *,
    library: RegionArtifactLibrary | None = None,
) -> dict[str, Any]:
    """Load and hash-check one labeled, non-policy diversity report."""

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or set(report) != _REPORT_FIELDS:
        raise ValueError("Region library diversity report fields differ")
    if (
        report["schema"] != REGION_LIBRARY_DIVERSITY_SCHEMA
        or isinstance(report["version"], bool)
        or not isinstance(report["version"], int)
        or report["version"] != REGION_LIBRARY_DIVERSITY_VERSION
    ):
        raise ValueError("unsupported Region library diversity report")
    if report["report_semantic_sha256"] != _report_digest(report):
        raise ValueError("Region library diversity report SHA-256 mismatch")
    if (
        library is not None
        and report["library_semantic_sha256"] != library.semantic_sha256
    ):
        raise ValueError("Region library diversity report names another library")
    policy = report["policy_generalization"]
    if not isinstance(policy, dict) or policy.get("status") != (
        "unmeasured_requires_policy_consumer"
    ):
        raise ValueError("Region diversity report makes an unsupported policy claim")
    return report


def _feature_vector(snapshot: NativeRegionSnapshot) -> np.ndarray:
    palette = snapshot.cell_palette
    counts = np.bincount(
        snapshot.cell_code.reshape(-1),
        minlength=palette.size,
    ).astype(np.float64)
    total = float(np.sum(counts))
    flags = palette.flags.astype(np.uint16)
    global_features = [
        float(np.sum(counts[(flags & flag) != 0]) / total)
        for flag in _FLAGS
    ]
    shape_counts = np.count_nonzero(
        snapshot.shape_palette.box_mask,
        axis=1,
    )
    cell_shape_counts = shape_counts[palette.shape_index]
    global_features.extend(
        (
            float(np.sum(counts[cell_shape_counts > 0]) / total),
            float(np.sum(counts[cell_shape_counts > 1]) / total),
        )
    )

    spatial: list[float] = []
    minimum = CAPTURE_HALO_CHUNKS
    for chunk_x in range(minimum, minimum + CORE_CHUNKS_PER_AXIS):
        for section_y in range(HEIGHT_SECTIONS):
            for chunk_z in range(minimum, minimum + CORE_CHUNKS_PER_AXIS):
                slot = chunk_x * CAPTURE_CHUNKS_PER_AXIS + chunk_z
                section_flags = flags[snapshot.cell_code[slot, section_y]]
                spatial.extend(
                    (
                        float(np.mean((section_flags & FLAG_SOLID) != 0)),
                        float(np.mean((section_flags & FLAG_FLUID) != 0)),
                    )
                )
    return np.asarray((*global_features, *spatial), dtype=np.float32)


def _distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.mean(delta * delta, axis=-1))


def _leave_one_out_nearest(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] < 2:
        raise ValueError("Region diversity requires at least two train artifacts")
    distances = _distance(vectors, vectors)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def _nearest(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.min(_distance(source, target), axis=1)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def _duplicate_feature_count(vectors: np.ndarray) -> int:
    contiguous = np.ascontiguousarray(vectors)
    unique = {row.tobytes() for row in contiguous}
    return int(vectors.shape[0] - len(unique))


def _report_digest(report: dict[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_semantic_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "REGION_LIBRARY_DIVERSITY_SCHEMA",
    "REGION_LIBRARY_DIVERSITY_VERSION",
    "load_region_library_diversity_report",
    "region_library_diversity_report",
    "save_region_library_diversity_report",
]
