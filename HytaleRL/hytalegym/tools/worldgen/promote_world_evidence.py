"""Promote a validated World evidence candidate into its server line's folders.

The recertification (``hytalegym.worldgen.region.recertification``) stages a candidate and never
publishes it (``publication.automatic`` is false by design). This tool is the manual promotion step
made checkable and repeatable: it verifies the staged candidate, runs the pre-promotion checks of
``tools/worldgen/README.md`` ("Checks before promotion"), and copies the publication targets into the
line's own folders without overwriting anything. Pointers and defaults are not touched.

Usage (dry run by default; nothing is written without ``--apply``)::

    python tools/worldgen/promote_world_evidence.py <stage> --server-version 0.7.2
    python tools/worldgen/promote_world_evidence.py <stage> --server-version 0.7.2 --apply

Destinations, by server line (``hytalegym/server_versions``):

- JSON fixtures and the stationary NPZ: ``world_fixture_root(version)`` (0.7.2: ``tests/fidelity/fixtures/world/v0_7/``).
- ``traversal-v2/``: ``artifacts/worldgen/<ServerProfile.region_library>/traversal-v2/``.
- The capacity report (first capture only): ``artifacts/worldgen/<ServerProfile.world_capacity_report>``.
- The receipt ``publication-v<N>.json`` and copies of the manifest, the source-pin report and the cascade:
  ``artifacts/worldgen/world-evidence-history/<bridge8>/`` (N is the recertification schema version).

Every destination must be absent (or byte-identical, in which case it is skipped). Rollback: remove the paths the
receipt lists under ``created_paths``; nothing else is changed. All of these folders are gitignored, so promotion is
local state, not a commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HYTALEGYM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HYTALEGYM))

from hytalegym.server_versions import release_line, server_profile, world_fixture_root  # noqa: E402

RECEIPT_SCHEMA = "hytalerl_world_evidence_publication_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _same(a: object, b: object) -> bool:
    return str(a).lower() == str(b).lower()


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [row for row in self.rows if not row[1]]


def run_checks(stage: Path, manifest: dict, version: str, repository_root: Path) -> tuple[Checks, dict]:
    """Verify the candidate rows and the runbook's pre-promotion checks; return (checks, facts)."""

    checks = Checks()
    profile = server_profile(version)
    facts: dict = {"server_version": version, "certified_version": profile.certified_version}

    checks.add("manifest status", manifest.get("status") == "validated_candidate_not_published", str(manifest.get("status")))
    checks.add("publication is manual", manifest.get("publication", {}).get("automatic") is False)
    installed = str(manifest.get("installed_hytale_version"))
    checks.add(
        "installed line equals the profile line",
        release_line(installed) == release_line(profile.certified_version),
        f"installed {installed}, profile {profile.certified_version}",
    )
    bridge = str(manifest.get("bridge_sha256", ""))
    if profile.bridge_jar_sha256:
        checks.add("bridge equals the line's pin", _same(bridge, profile.bridge_jar_sha256), bridge[:12])
    facts["bridge_sha256"] = bridge.upper()

    for name, row in sorted(manifest.get("candidates", {}).items()):
        path = stage / row["path"]
        ok = path.is_file() and _sha256(path) == row["file_sha256"]
        checks.add(f"candidate {name}", ok, row["path"])

    mode = manifest.get("reference_mode", "compare")
    facts["reference_mode"] = mode
    library_sha = profile.region_library_sha256
    if mode == "first_capture":
        block = manifest.get("first_capture", {})
        if library_sha:
            checks.add("first_capture.region_library_sha256", _same(block.get("region_library_sha256"), library_sha))
        if profile.seed_projection_sha256:
            checks.add(
                "first_capture.projection_reference_sha256",
                _same(block.get("projection_reference_sha256"), profile.seed_projection_sha256),
            )
        repeat = block.get("stationary_repeat_file_sha256", [])
        checks.add("stationary captures byte-identical", len(repeat) == 2 and repeat[0] == repeat[1], str([r[:12] for r in repeat]))
        contract = stage / manifest["publication"]["targets"]["seed_projection_contract"]
        checks.add(
            "staged projection contract hash recorded",
            contract.is_file() and _sha256(contract) == block.get("projection_contract_file_sha256"),
        )

    cascade = _read_json(stage / "contract-cascade.json")
    relationships = cascade.get("relationships", {})
    checks.add("cascade relationships all true", bool(relationships) and all(relationships.values()), json.dumps(relationships))
    traversal_manifest = _read_json(stage / "traversal-v2" / "traversal-manifest.json")
    identities = cascade.get("identities", {})
    if library_sha:
        checks.add("cascade identities.region_library", _same(identities.get("region_library"), library_sha))
    checks.add(
        "cascade identities.traversal_library equals the traversal manifest",
        _same(identities.get("traversal_library"), traversal_manifest.get("library_semantic_sha256")),
    )
    facts["traversal_library_semantic_sha256"] = str(traversal_manifest.get("library_semantic_sha256", "")).upper()

    checks.add("traversal manifest bridge", _same(traversal_manifest.get("native_evidence_jar_sha256"), bridge))
    if library_sha:
        checks.add(
            "traversal manifest source library",
            _same(traversal_manifest.get("source_region_library_semantic_sha256"), library_sha),
        )
    graph_count = int(traversal_manifest.get("graph_count", 0))
    checks.add("traversal graph_count equals its entries", graph_count == len(traversal_manifest.get("entries", [])), str(graph_count))
    recording = _read_json(stage / "traversal-v2" / "native-traversal-recording.json")
    checks.add("traversal recording full_edge_census", recording.get("full_edge_census") is True)
    checks.add("traversal recording artifact_count", int(recording.get("artifact_count", -1)) == graph_count)
    facts["graph_count"] = graph_count

    difference = _read_json(stage / "native_region_difference.json")
    comparison = difference.get("comparison", {})
    cap = difference.get("seed_projection_contract", {}).get("max_non_damaging_fluid_only_cells")
    fluid = comparison.get("non_damaging_fluid_only_cell_count")
    checks.add("seed projection accepted by its contract", difference.get("accepted_by_seed_projection_contract") is True)
    checks.add("cross-reset other differing cells zero", comparison.get("other_differing_cell_count") == 0, str(comparison.get("other_differing_cell_count")))
    checks.add("cross-reset fluid-only cells within the cap", cap is not None and fluid is not None and fluid <= cap, f"{fluid} of {cap}")
    facts["cross_reset_non_damaging_fluid_only_cells"] = fluid
    facts["cross_reset_other_differing_cells"] = comparison.get("other_differing_cell_count")
    facts["seed_projection_semantic_sha256"] = str(comparison.get("second_seed_projection_digest", "")).upper()

    if mode == "first_capture":
        capacity = _read_json(stage / "world-token-capacity.json")
        if library_sha:
            checks.add("capacity report library", _same(capacity.get("library_semantic_sha256"), library_sha))
        checks.add("capacity report seeds", len(capacity.get("artifact_seeds", [])) == graph_count)

    stationary = _read_json(stage / manifest["publication"]["targets"]["stationary_traversal_recording"])
    checks.add("stationary recording exact", stationary.get("exact") is True)
    facts["stationary_projection_semantic_sha256"] = str(stationary.get("report_semantic_sha256", "")).upper()

    pins = _read_json(stage / "source-pin-staleness.json")
    checks.add("source-pin report status", pins.get("status") not in (None, "stale_or_unavailable"), str(pins.get("status")))
    facts["source_pin_report_semantic_sha256"] = str(pins.get("report_semantic_sha256", "")).upper()
    return checks, facts


def copy_plan(stage: Path, manifest: dict, version: str, repository_root: Path) -> list[tuple[Path, Path]]:
    """Return (source, destination) pairs for every publication target."""

    profile = server_profile(version)
    package_root = repository_root / "hytalegym"
    fixtures = world_fixture_root(package_root, version)
    worldgen = repository_root / "artifacts" / "worldgen"
    library = worldgen / (profile.region_library or "region-library-pilot-v2")
    targets = manifest["publication"]["targets"]
    plan: list[tuple[Path, Path]] = []
    for name, relative in sorted(targets.items()):
        source = stage / relative
        if name == "traversal_directory":
            plan.append((source, library / "traversal-v2"))
        elif name == "world_capacity_report":
            report = profile.world_capacity_report or "world-token-capacity-v4.json"
            plan.append((source, worldgen / report))
        else:
            plan.append((source, fixtures / source.name))
    artifact = manifest["candidates"].get("stationary_traversal_artifact")
    if artifact:
        source = stage / artifact["path"]
        plan.append((source, fixtures / source.name))
    return plan


def _destination_state(source: Path, destination: Path) -> str:
    if not destination.exists():
        return "absent"
    if source.is_dir():
        return "exists"
    return "identical" if _sha256(destination) == _sha256(source) else "differs"


def _git_head(repository_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repository_root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", type=Path, help="the recertification stage directory")
    parser.add_argument("--server-version", default=os.environ.get("HYTALERL_SERVER_VERSION", "0.7.2"))
    parser.add_argument("--repository-root", type=Path, default=HYTALEGYM.parent, help="the HytaleRL directory")
    parser.add_argument("--apply", action="store_true", help="copy the targets and write the receipt (default: dry run)")
    args = parser.parse_args(argv)

    stage = args.stage.resolve()
    repository_root = args.repository_root.resolve()
    manifest = _read_json(stage / "recertification-manifest.json")
    checks, facts = run_checks(stage, manifest, args.server_version, repository_root)
    for name, ok, detail in checks.rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    plan = copy_plan(stage, manifest, args.server_version, repository_root)
    states = {destination: _destination_state(source, destination) for source, destination in plan}
    print("\nCopy plan:")
    for source, destination in plan:
        print(f"  {states[destination]:9s} {source.relative_to(stage)} -> {destination}")
    blocked = [d for d, state in states.items() if state in ("exists", "differs")]
    if checks.failed or blocked:
        print(f"\nNOT PROMOTABLE: {len(checks.failed)} failed check(s), {len(blocked)} destination(s) already present.")
        return 1
    if not args.apply:
        print("\nDry run: every check passed and every destination is free. Re-run with --apply to promote.")
        return 0

    created: list[str] = []
    for source, destination in plan:
        if states[destination] == "identical":
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
            created.extend(str(p) for p in sorted(destination.rglob("*")) if p.is_file())
        else:
            shutil.copy2(source, destination)
            created.append(str(destination))
        if source.is_file() and _sha256(destination) != _sha256(source):
            raise RuntimeError(f"copy changed bytes: {destination}")

    history = repository_root / "artifacts" / "worldgen" / "world-evidence-history" / facts["bridge_sha256"][:8]
    history.mkdir(parents=True, exist_ok=True)
    number = int(manifest.get("version", 0))
    copies = {
        "recertification_manifest": ("recertification-manifest.json", f"recertification-manifest-v{number}.json"),
        "source_pin_report": ("source-pin-staleness.json", f"source-pin-staleness-v{number}.json"),
        "publication_contract_cascade": ("contract-cascade.json", f"contract-cascade-at-publication-v{number}.json"),
        "region_difference_report": ("native_region_difference.json", f"native_region_difference-v{number}.json"),
    }
    receipt_rows: dict[str, dict[str, str]] = {}
    for key, (source_name, target_name) in copies.items():
        target = history / target_name
        if target.exists() and _sha256(target) != _sha256(stage / source_name):
            raise RuntimeError(f"history copy already present with other bytes: {target}")
        if not target.exists():
            shutil.copy2(stage / source_name, target)
            created.append(str(target))
        receipt_rows[key] = {"path": target_name, "sha256": _sha256(target).upper()}

    fixture_files = [d for s, d in plan if not s.is_dir() and d.suffix == ".json"]
    traversal_files = [p for p in created if "/traversal-v2/" in p]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "version": 2,
        "status": "published_after_hash_and_semantic_review",
        "published_at_utc": datetime.now(UTC).isoformat(),
        "repository_head": _git_head(repository_root),
        "server_version": args.server_version,
        "reference_mode": facts["reference_mode"],
        "bridge_sha256": facts["bridge_sha256"],
        "random_interposed_control_seed": manifest.get("random_interposed_control_seed"),
        "candidate_count": len(manifest.get("candidates", {})),
        "publication_target_count": len(manifest["publication"]["targets"]),
        "published_fixture_files": len(fixture_files),
        "published_traversal_files": len(traversal_files),
        "cross_reset_non_damaging_fluid_only_cells": facts["cross_reset_non_damaging_fluid_only_cells"],
        "cross_reset_other_differing_cells": facts["cross_reset_other_differing_cells"],
        "traversal_library_semantic_sha256": facts["traversal_library_semantic_sha256"],
        "stationary_projection_semantic_sha256": facts["stationary_projection_semantic_sha256"],
        "seed_projection_semantic_sha256": facts["seed_projection_semantic_sha256"],
        "first_capture": manifest.get("first_capture"),
        "checks": [{"name": name, "passed": ok, "detail": detail} for name, ok, detail in checks.rows],
        "created_paths": created,
        **receipt_rows,
    }
    receipt_path = history / f"publication-v{number}.json"
    if receipt_path.exists():
        raise RuntimeError(f"receipt already present: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created.append(str(receipt_path))
    print(f"\nPROMOTED {len(created)} file(s); receipt {receipt_path}")
    print("Rollback: delete the paths listed under created_paths in the receipt (nothing else changed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
