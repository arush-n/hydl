#!/usr/bin/env python3
"""Recapture the Region terrain on a Hytale 0.7 server and compare it.

Gate sequence, each step on its own fresh bridge launch (one launch per
scenario, BRIDGE-PORTING.md section 7):

1. capture: examples/capture_native_region.py writes one staged NPZ under
   artifacts/worldgen and must report 500/500 sections. An existing output
   is never overwritten; --skip-capture reuses it and runs only the compare.
2. compare: tests/fidelity/native_region_capture_compare.py on a fresh server.
   The default scope is exact: the capture's semantic digest is passed as
   --expected-semantic-sha256, so a fresh process must reproduce the capture.
   --seed-projection-contract PATH or --fluid-only-cap N selects the contract
   scope instead, the 0.5.7 method for non-damaging fluid drift. --fluid-only-cap
   N writes that v2 contract from this capture. Runbook: tools/worldgen/README.md.
3. diff (only with --reference): read-only per-section and per-cell comparison
   of the new NPZ with a reference NPZ. Measurement only, not a gate.

Logs, the launch reports and receipt.json go under --run-root. Nothing is
swapped, promoted or pinned: see tools/worldgen/README.md.

Run it with the repo venv from any directory, for example:
  /Users/arush/hydl/.venv/bin/python /Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/recapture_region.py --version 0.7.2 --seed 570057 --bridge-port 5556 --game-port 25574 --run-root <new empty scratch dir> --reference /Users/arush/hydl/HytaleRL/artifacts/worldgen/native-region-light-v1/seed-570057/region.npz
  /Users/arush/hydl/.venv/bin/python /Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/recapture_region.py --version 0.7.2 --seed 570057 --bridge-port 5556 --game-port 25574 --run-root <new empty scratch dir> --skip-capture --fluid-only-cap 6
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
HYTALE_ROOT = Path(__file__).resolve().parents[3]
CAPTURE = HYTALE_ROOT / "examples" / "capture_native_region.py"
COMPARE = PACKAGE_ROOT / "tests" / "fidelity" / "native_region_capture_compare.py"
ARTIFACTS = HYTALE_ROOT / "artifacts" / "worldgen"
LAUNCHER = HYTALE_ROOT / "hytale-plugin-0.7.2" / "tools" / "launch_bridge_072.py"
SECTIONS = 250
PASSES = 2
PROGRESS_DONE = f"captured {SECTIONS * PASSES}/{SECTIONS * PASSES} sections"
DIGEST_LINE = re.compile(r"^semantic_sha256: ([0-9a-f]{64})$", re.MULTILINE)
FILE_LINE = re.compile(r"^file_sha256: ([0-9a-f]{64})$", re.MULTILINE)
COMPARE_PASS = "NATIVE_REGION_CAPTURE_DIFFERENTIAL_PASS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _port_free(port: int, kind: int) -> bool:
    with socket.socket(socket.AF_INET, kind) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _run_launch(
    *,
    name: str,
    run_root: Path,
    version: str,
    jar: Path,
    bridge_port: int,
    game_port: int,
    then: list[str],
) -> tuple[int, dict[str, Any] | None, str]:
    """Start one bridge server, run `then` against it, and stop it."""

    log_path = run_root / f"{name}.launcher.log"
    command = [
        sys.executable,
        str(LAUNCHER),
        "--run-dir",
        str(run_root / name),
        "--jar",
        str(jar),
        "--bridge-port",
        str(bridge_port),
        "--game-port",
        str(game_port),
        "--ready-timeout",
        "900",
        "--then-timeout",
        "7200",
        "--then",
        shlex.join(then),
    ]
    env = dict(os.environ)
    env["HYTALERL_SERVER_VERSION"] = version
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        code = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        ).returncode
    report_path = run_root / name / "launch-report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else None
    )
    return code, report, log_path.read_text(encoding="utf-8", errors="replace")


def _contract_document(
    *,
    version: str,
    seed: int,
    bridge_sha256: str,
    output: Path,
    semantic_sha256: str,
    max_fluid_only_cells: int,
) -> dict[str, Any]:
    """Version 2 seed-projection contract bound to one passing capture."""

    from hytalegym.worldgen.region import NativeRegionSnapshot
    from hytalegym.worldgen.region.stability import (
        REGION_SEED_PROJECTION_SCHEMA,
        REGION_SEED_PROJECTION_VERSION,
        region_seed_projection_digest,
    )

    return {
        "schema": REGION_SEED_PROJECTION_SCHEMA,
        "version": REGION_SEED_PROJECTION_VERSION,
        "server_version": version,
        "seed": seed,
        "evidence_bridge_sha256": bridge_sha256,
        "reference_artifact_bridge_sha256": bridge_sha256,
        "reference_artifact_file_sha256": _sha256(output).upper(),
        "reference_artifact_semantic_sha256": semantic_sha256.upper(),
        "reference_seed_projection_sha256": region_seed_projection_digest(
            NativeRegionSnapshot.load(output)
        ).upper(),
        "max_non_damaging_fluid_only_cells": max_fluid_only_cells,
    }


def offline_diff(candidate: Path, reference: Path, difference_limit: int) -> dict[str, Any]:
    """Read-only comparison of two NPZ snapshots.

    compare_region_semantics returns early, with zero cell counts, when the
    identity metadata differs (for example server_version). The cell counts
    therefore come from an in-memory copy whose differing identity fields take
    the reference's values. Nothing is saved, and the real identity
    differences are reported separately.
    """

    from hytalegym.worldgen.region import NativeRegionSnapshot, compare_region_semantics

    first = NativeRegionSnapshot.load(candidate)
    second = NativeRegionSnapshot.load(reference)
    identity = compare_region_semantics(first, second, require_same_world=False, difference_limit=1)
    relabelled: list[str] = []
    structural: list[str] = []
    for mismatch in identity.layout_mismatches:
        if mismatch.startswith("metadata.") and mismatch.endswith(" differs"):
            relabelled.append(mismatch[len("metadata.") : -len(" differs")])
        else:
            structural.append(mismatch)
    result: dict[str, Any] = {
        "candidate": str(candidate),
        "reference": str(reference),
        "candidate_semantic_sha256": identity.first_digest,
        "reference_semantic_sha256": identity.second_digest,
        "identity_mismatches": list(identity.layout_mismatches),
        "identity_fields_relabelled_in_memory": relabelled,
    }
    if structural:
        result["terrain"] = f"not comparable: {structural}"
        return result
    metadata = dict(first.metadata)
    for field in relabelled:
        metadata[field] = second.metadata.get(field)
    relabelled_first = NativeRegionSnapshot(
        metadata=metadata,
        core_min_chunk_xz=first.core_min_chunk_xz,
        section_known=first.section_known,
        cell_code=first.cell_code,
        cell_palette=first.cell_palette,
        shape_palette=first.shape_palette,
        filler_root_offset_packed=first.filler_root_offset_packed,
    )
    terrain = compare_region_semantics(
        relabelled_first,
        second,
        require_same_world=False,
        difference_limit=difference_limit,
    )
    flagged = [section for section in terrain.section_digests if not section.equal]
    result.update(
        {
            "sections_total": len(terrain.section_digests),
            "sections_differing": len(flagged),
            "differing_sections_chunk_xz_section_y": [
                [section.chunk_x, section.chunk_z, section.section_y] for section in flagged
            ],
            "differing_cells": terrain.differing_cell_count,
            "non_damaging_fluid_only_cells": terrain.non_damaging_fluid_only_cell_count,
            "other_differing_cells": terrain.other_differing_cell_count,
            "differing_cells_by_field": dict(
                collections.Counter(
                    field for difference in terrain.differences for field in difference.fields
                )
            ),
            "first_cell_differences": [
                {
                    "chunk_xz": [difference.chunk_x, difference.chunk_z],
                    "section_y": difference.section_y,
                    "block_xyz": [difference.block_x, difference.block_y, difference.block_z],
                    "fields": list(difference.fields),
                }
                for difference in terrain.differences[:16]
            ],
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--version", required=True, help="server release line with a bridge pin (0.7.2)")
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument("--bridge-port", type=int, required=True)
    parser.add_argument("--game-port", type=int, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="new or empty scratch directory for launches, logs and receipt.json",
    )
    parser.add_argument("--reference", type=Path, help="NPZ for the offline diff, not a gate")
    parser.add_argument(
        "--difference-limit",
        type=int,
        default=2_000_000,
        help="cells kept by the offline diff for field counts",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="check the pin, ports and paths, then exit without launching",
    )
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="reuse the existing output NPZ (hashed here) and run only the compare",
    )
    parser.add_argument(
        "--seed-projection-contract",
        type=Path,
        help="v2 seed-projection contract for the compare, instead of exact digest equality",
    )
    parser.add_argument(
        "--fluid-only-cap",
        type=int,
        help=(
            "after a passing capture, write a v2 seed-projection contract allowing "
            "up to N non-damaging fluid-only cells, and compare with it"
        ),
    )
    args = parser.parse_args(argv)
    if args.seed_projection_contract is not None and args.fluid_only_cap is not None:
        parser.error("--seed-projection-contract and --fluid-only-cap are alternatives")
    if args.fluid_only_cap is not None and args.fluid_only_cap < 0:
        parser.error("--fluid-only-cap must be non-negative")

    from hytalegym.server_versions import server_profile

    profile = server_profile(args.version)
    if profile.release_line != (0, 7) or profile.bridge_jar is None or profile.bridge_jar_sha256 is None:
        parser.error(f"{args.version} has no 0.7 bridge pin this tool can launch")
    jar = HYTALE_ROOT / profile.bridge_jar
    if not jar.is_file():
        parser.error(f"bridge jar missing: {jar}")
    observed = _sha256(jar).upper()
    if observed != profile.bridge_jar_sha256:
        parser.error(f"bridge jar {jar} hashes to {observed}; pin is {profile.bridge_jar_sha256}")
    if args.bridge_port == args.game_port:
        parser.error("bridge and game ports must differ")
    for port, kind, label in (
        (args.bridge_port, socket.SOCK_STREAM, "bridge"),
        (args.game_port, socket.SOCK_DGRAM, "game"),
    ):
        if not _port_free(port, kind):
            parser.error(f"{label} port {port} is busy")
    output = ARTIFACTS / f"region_{args.version}_seed_{args.seed}.npz"
    if args.skip_capture:
        if not output.is_file():
            parser.error(f"--skip-capture needs an existing NPZ: {output}")
    elif output.exists():
        parser.error(f"refusing to overwrite {output}")
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        parser.error(f"run-root must be new or empty: {run_root}")
    if args.reference is not None and not args.reference.is_file():
        parser.error(f"reference NPZ missing: {args.reference}")
    contract_file = None
    if args.seed_projection_contract is not None:
        from hytalegym.worldgen.region import load_region_seed_projection_contract

        contract_file = args.seed_projection_contract.resolve()
        try:
            contract = load_region_seed_projection_contract(contract_file)
        except (OSError, TypeError, ValueError) as error:
            parser.error(f"seed-projection contract unusable: {error}")
        if contract.seed != args.seed or contract.evidence_bridge_sha256 != observed:
            parser.error("seed-projection contract names another seed or bridge")

    print(f"pin ok: {jar.name} sha256 {observed}")
    print(f"output: {output}")
    print(f"run_root: {run_root}")
    if args.skip_capture:
        print("capture: skipped, reusing the output NPZ")
    if contract_file is not None:
        print(f"scope: seed_projection_contract {contract_file}")
    elif args.fluid_only_cap is not None:
        print(f"scope: fluid-only cap {args.fluid_only_cap}, contract written after the capture")
    else:
        print("scope: exact_semantic_sha256")
    capture_then = [
        sys.executable,
        str(CAPTURE),
        "--port",
        str(args.bridge_port),
        "--seed",
        str(args.seed),
        "--native-evidence-jar",
        str(jar),
        "--output",
        str(output),
    ]
    if args.preflight_only:
        if not args.skip_capture:
            print("capture command: " + shlex.join(capture_then))
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "version": args.version,
        "seed": args.seed,
        "bridge_port": args.bridge_port,
        "game_port": args.game_port,
        "bridge_jar": str(jar),
        "bridge_jar_sha256": observed,
        "output": str(output),
    }
    if args.skip_capture:
        from hytalegym.worldgen.region import NativeRegionSnapshot

        capture_ok = True
        file_sha = _sha256(output)
        semantic_sha = NativeRegionSnapshot.load(output).semantic_digest()
        receipt["capture"] = {
            "skipped": True,
            "passed": None,
            "semantic_sha256": semantic_sha,
            "file_sha256_on_disk": file_sha,
            "log": None,
        }
        print(f"capture: SKIPPED, reusing {output} (file_sha256 {file_sha})")
    else:
        code, report, text = _run_launch(
            name="capture",
            run_root=run_root,
            version=args.version,
            jar=jar,
            bridge_port=args.bridge_port,
            game_port=args.game_port,
            then=capture_then,
        )
        digest_match = DIGEST_LINE.search(text)
        file_match = FILE_LINE.search(text)
        capture_ok = (
            code == 0
            and PROGRESS_DONE in text
            and digest_match is not None
            and output.is_file()
            and report is not None
            and report.get("then_exit_code") == 0
        )
        semantic_sha = digest_match.group(1) if digest_match else None
        receipt["capture"] = {
            "launcher_exit": code,
            "passed": capture_ok,
            "progress_500_of_500": PROGRESS_DONE in text,
            "semantic_sha256": semantic_sha,
            "file_sha256_reported": file_match.group(1) if file_match else None,
            "file_sha256_on_disk": _sha256(output) if output.is_file() else None,
            "launch_report": report,
            "log": str(run_root / "capture.launcher.log"),
        }
        print(f"capture: {'PASS' if capture_ok else 'FAIL'} ({PROGRESS_DONE if PROGRESS_DONE in text else 'no 500/500 line'})")

    compare_ok = False
    if capture_ok and semantic_sha is not None:
        if args.fluid_only_cap is not None:
            contract_file = run_root / "native_region_seed_projection_contract.json"
            contract_file.write_text(
                json.dumps(
                    _contract_document(
                        version=args.version,
                        seed=args.seed,
                        bridge_sha256=observed,
                        output=output,
                        semantic_sha256=semantic_sha,
                        max_fluid_only_cells=args.fluid_only_cap,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        scope = (
            ["--seed-projection-contract", str(contract_file)]
            if contract_file is not None
            else ["--expected-semantic-sha256", semantic_sha]
        )
        scope_name = (
            "seed_projection_contract" if contract_file is not None else "exact_semantic_sha256"
        )
        compare_then = [
            sys.executable,
            str(COMPARE),
            "--port",
            str(args.bridge_port),
            "--seed",
            str(args.seed),
            "--native-evidence-jar",
            str(jar),
            "--server-lifecycle-assumption",
            "fresh_process",
            *scope,
            "--difference-report",
            str(run_root / "compare-difference-report.json"),
        ]
        code2, report2, text2 = _run_launch(
            name="compare",
            run_root=run_root,
            version=args.version,
            jar=jar,
            bridge_port=args.bridge_port,
            game_port=args.game_port,
            then=compare_then,
        )
        compare_ok = (
            code2 == 0
            and COMPARE_PASS in text2
            and report2 is not None
            and report2.get("then_exit_code") == 0
        )
        receipt["compare"] = {
            "scope": scope_name,
            "seed_projection_contract": (
                None
                if contract_file is None
                else {"path": str(contract_file), "file_sha256_on_disk": _sha256(contract_file)}
            ),
            "launcher_exit": code2,
            "passed": compare_ok,
            "pass_marker": COMPARE_PASS in text2,
            "launch_report": report2,
            "log": str(run_root / "compare.launcher.log"),
        }
        print(f"compare: {'PASS' if compare_ok else 'FAIL'} ({scope_name})")

    if args.reference is not None and capture_ok:
        diff = offline_diff(output, args.reference, args.difference_limit)
        (run_root / "offline-diff.json").write_text(json.dumps(diff, indent=2), encoding="utf-8")
        receipt["offline_diff"] = str(run_root / "offline-diff.json")
        print(
            f"offline diff: sections differing {diff.get('sections_differing')}/{diff.get('sections_total')}, "
            f"cells {diff.get('differing_cells')} (fluid-only {diff.get('non_damaging_fluid_only_cells')}, "
            f"other {diff.get('other_differing_cells')}), identity mismatches {diff['identity_mismatches']}"
        )

    receipt["passed"] = capture_ok and compare_ok
    (run_root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"receipt: {run_root / 'receipt.json'}")
    return 0 if capture_ok and compare_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
