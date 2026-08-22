"""Select and publish a replayable Basic pursuit checkpoint from one run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from arena.training.runs.pursuit_run import (
    PURSUIT_SELECTION_LAW,
    PURSUIT_SELECTION_SCHEMA,
    pursuit_candidate_is_better,
    pursuit_selection_assessment,
)
from hytalegym.jax.training.checkpoint import (
    load_policy_checkpoint,
    save_policy_checkpoint,
)


PUBLICATION_SCHEMA = "basic-pursuit-candidate-publication-v1"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _rows(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = {int(row["update"]): row for row in report.get("curve", ())}
    if not rows or 0 not in rows or len(rows) != len(report.get("curve", ())):
        raise ValueError("pursuit curve must contain unique updates including zero")
    return rows


def select_available_pursuit_update(
    report: Mapping[str, Any], available_updates: set[int]
) -> tuple[int, dict[str, Any]]:
    """Reapply the current selection law only to checkpoints that exist."""

    rows = _rows(report)
    if 0 not in available_updates or not available_updates.issubset(rows):
        raise ValueError("available pursuit checkpoints must include curve update zero")
    source = pursuit_selection_assessment(rows[0]["recurrent"])
    incumbent = pursuit_selection_assessment(
        rows[0]["recurrent"], source_success_rate=source.success_rate
    )
    selected = 0
    for update in sorted(available_updates - {0}):
        candidate = pursuit_selection_assessment(
            rows[update]["recurrent"], source_success_rate=source.success_rate
        )
        if pursuit_candidate_is_better(candidate, incumbent):
            selected, incumbent = update, candidate
    if not incumbent.eligible:
        raise ValueError(f"no eligible pursuit checkpoint: {incumbent.flags}")
    return selected, asdict(incumbent)


def _available_checkpoints(
    report_path: Path, report: Mapping[str, Any]
) -> dict[int, Path]:
    candidates: dict[int, Mapping[str, Any]] = {}
    for entry in report.get("milestone_checkpoints", ()):
        candidates[int(entry["update"])] = entry
    selection = report["selection"]
    candidates[int(selection["selected_update"])] = report["checkpoint"]
    candidates[int(report["curve"][-1]["update"])] = report["latest_checkpoint"]

    resolved = {}
    for update, entry in candidates.items():
        stamped = Path(str(entry["path"]))
        path = stamped if stamped.is_file() else report_path.parent / stamped.name
        if not path.is_file():
            raise FileNotFoundError(
                f"missing pursuit checkpoint for update {update}: {path}"
            )
        if _sha256(path) != str(entry["sha256"]).upper():
            raise ValueError(f"pursuit checkpoint hash mismatch at update {update}")
        resolved[update] = path
    return resolved


def publish_pursuit_candidate(report_path: Path, output: Path) -> dict[str, Any]:
    """Publish the best available policy without mutating the source receipt."""

    report_path = report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoints = _available_checkpoints(report_path, report)
    selected_update, assessment = select_available_pursuit_update(
        report, set(checkpoints)
    )
    source = checkpoints[selected_update]
    params, config, metadata = load_policy_checkpoint(source)
    output = output.resolve()
    save_policy_checkpoint(
        output,
        params,
        config,
        metadata={
            **metadata,
            "publication_schema": PUBLICATION_SCHEMA,
            "checkpoint_role": "selected_recurrent_pursuit_candidate",
            "selection_schema": PURSUIT_SELECTION_SCHEMA,
            "selection_law": PURSUIT_SELECTION_LAW,
            "selection_assessment": assessment,
            "selected_update": selected_update,
            "updates": selected_update,
            "environment_steps": int(
                _rows(report)[selected_update]["environment_steps"]
            ),
            "source_report_sha256": _sha256(report_path),
            "source_checkpoint_sha256": _sha256(source),
            "policy_only": True,
            "resumable": False,
        },
    )
    receipt = {
        "schema": PUBLICATION_SCHEMA,
        "source_report": str(report_path),
        "source_report_sha256": _sha256(report_path),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": _sha256(source),
        "previous_selected_update": int(report["selection"]["selected_update"]),
        "selected_update": selected_update,
        "assessment": assessment,
        "selection_schema": PURSUIT_SELECTION_SCHEMA,
        "selection_law": PURSUIT_SELECTION_LAW,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
        "live_policy_compatible": False,
        "promotion_status": "not_assessed",
    }
    receipt_path = output.with_name(f"{output.name}.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish_pursuit_candidate(args.report, args.output), indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "PUBLICATION_SCHEMA",
    "publish_pursuit_candidate",
    "select_available_pursuit_update",
]
