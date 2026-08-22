"""Capture a resumable random-seed pilot of exact native Region artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile

from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.worldgen.native_lease import acquire_native_evidence_lease
from hytalegym.worldgen.region import (
    NativeRegionSnapshot,
    RegionArtifactLibrary,
    capture_native_region,
    create_region_artifact_library,
    current_native_evidence_jar_sha256,
    load_region_library_diversity_report,
    save_region_library_diversity_report,
)

_PLAN_SCHEMA = "hytalerl_region_library_capture_plan_v2"
_PLAN_VERSION = 2
_PLAN_FIELDS = {
    "schema",
    "version",
    "native_evidence_jar_sha256",
    "seeds",
    "split_salt_sha256",
    "heldout_count",
    "role",
    "core_min_chunk_xz",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--role", default="Kweebec_Razorleaf")
    parser.add_argument("--core-min-chunk-x", type=int)
    parser.add_argument("--core-min-chunk-z", type=int)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--heldout-count", type=int, default=4)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--split-salt-sha256")
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--native-evidence-jar", type=Path)
    evidence.add_argument("--native-evidence-jar-sha256")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/worldgen/region-library-pilot-v1"),
    )
    args = parser.parse_args()

    with acquire_native_evidence_lease(
        host=args.host,
        port=args.port,
        purpose="capture_native_region_library",
    ):
        _run(parser, args)


def _run(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    evidence_hash = _native_evidence_hash(parser, args)
    root = args.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    plan_path = root / "capture-plan.json"
    if manifest_path.exists() and not plan_path.exists():
        parser.error("existing Region library is missing its capture plan")
    seeds, split_salt = _capture_plan(parser, args, root, evidence_hash)
    if manifest_path.exists():
        library = RegionArtifactLibrary.load(manifest_path)
        if (
            library.manifest["capture_contract"][
                "native_evidence_jar_sha256"
            ]
            != evidence_hash
            or len(library.entries) != args.count
            or len(library.entries_for_split("heldout"))
            != args.heldout_count
            or {entry.seed for entry in library.entries} != set(seeds)
            or library.manifest["split_contract"]["salt_sha256"]
            != split_salt
        ):
            parser.error("existing Region library differs from this request")
        report_path = root / "diversity.json"
        if report_path.exists():
            load_region_library_diversity_report(
                report_path,
                library=library,
            )
        else:
            save_region_library_diversity_report(library, report_path)
        print("verified existing library:", library.semantic_sha256)
        return

    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []

    env = HytaleEnv(
        task="kill_trork",
        backend="native",
        world="hytale",
        host=args.host,
        port=args.port,
        ticks_per_step=1,
        max_episode_steps=100,
        npc_role=args.role,
        combat_target_active=False,
        fidelity_fixture="static_region",
    )
    try:
        for index, seed in enumerate(seeds, start=1):
            destination = artifacts / f"seed_{seed}.npz"
            if destination.exists():
                snapshot = NativeRegionSnapshot.load(destination)
                if (
                    snapshot.metadata.get("seed") != seed
                    or snapshot.metadata.get("native_evidence_jar_sha256")
                    != evidence_hash
                ):
                    raise ValueError(
                        f"existing artifact does not match capture plan: {destination}"
                    )
                print(f"reused {index}/{len(seeds)} seed={seed}")
            else:
                env.reset(seed=seed)
                snapshot = capture_native_region(
                    env,
                    core_min_chunk_x=args.core_min_chunk_x,
                    core_min_chunk_z=args.core_min_chunk_z,
                    native_evidence_jar_sha256=evidence_hash,
                    progress=_progress(index, len(seeds)),
                )
                snapshot.save(destination)
                print(
                    f"captured {index}/{len(seeds)} seed={seed} "
                    f"semantic={snapshot.semantic_artifact_digest()}"
                )
            artifact_paths.append(destination)
    finally:
        env.close()

    library = create_region_artifact_library(
        root,
        artifact_paths,
        heldout_count=args.heldout_count,
        split_salt_sha256=split_salt,
    )
    report_path = save_region_library_diversity_report(
        library,
        root / "diversity.json",
    )
    disk_bytes = sum(path.stat().st_size for path in artifact_paths)
    print("manifest:", library.manifest_path)
    print("library_semantic_sha256:", library.semantic_sha256)
    print("artifacts:", len(library.entries))
    print("train:", len(library.entries_for_split("train")))
    print("heldout:", len(library.entries_for_split("heldout")))
    print("compressed_bytes:", disk_bytes)
    print("diversity_report:", report_path)


def _capture_plan(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    root: Path,
    evidence_hash: str,
) -> tuple[tuple[int, ...], str]:
    path = root / "capture-plan.json"
    if path.exists():
        plan = _load_plan(parser, path)
        if (
            set(plan) != _PLAN_FIELDS
            or plan.get("schema") != _PLAN_SCHEMA
            or not isinstance(plan.get("version"), int)
            or isinstance(plan.get("version"), bool)
            or plan["version"] != _PLAN_VERSION
            or plan.get("native_evidence_jar_sha256") != evidence_hash
            or plan.get("heldout_count") != args.heldout_count
            or plan.get("role") != args.role
            or plan.get("core_min_chunk_xz")
            != [args.core_min_chunk_x, args.core_min_chunk_z]
        ):
            parser.error("existing capture plan is incompatible")
        seeds = tuple(_seed(parser, value) for value in plan["seeds"])
        if len(seeds) != args.count or len(set(seeds)) != len(seeds):
            parser.error("existing capture plan seed count is incompatible")
        salt = _sha256(parser, plan["split_salt_sha256"], "capture-plan salt")
        if args.seed is not None:
            requested_seeds = tuple(_seed(parser, value) for value in args.seed)
            if requested_seeds != seeds:
                parser.error("--seed values differ from the capture plan")
        if (
            args.split_salt_sha256 is not None
            and _sha256(
                parser,
                args.split_salt_sha256,
                "--split-salt-sha256",
            )
            != salt
        ):
            parser.error("--split-salt-sha256 differs from the capture plan")
        return seeds, salt

    count = args.count
    if isinstance(count, bool) or count < 3:
        parser.error("--count must be at least 3")
    seeds = (
        tuple(_seed(parser, value) for value in args.seed)
        if args.seed
        else tuple(secrets.SystemRandom().sample(range(1, 2**31), count))
    )
    if len(seeds) != count or len(set(seeds)) != len(seeds):
        parser.error("--seed values must be unique and match --count")
    if not 0 < args.heldout_count < len(seeds) - 1:
        parser.error("--heldout-count must leave at least two train artifacts")
    salt = (
        _sha256(parser, args.split_salt_sha256, "--split-salt-sha256")
        if args.split_salt_sha256
        else secrets.token_hex(32)
    )
    plan = {
        "schema": _PLAN_SCHEMA,
        "version": _PLAN_VERSION,
        "native_evidence_jar_sha256": evidence_hash,
        "seeds": list(seeds),
        "split_salt_sha256": salt,
        "heldout_count": args.heldout_count,
        "role": args.role,
        "core_min_chunk_xz": [
            args.core_min_chunk_x,
            args.core_min_chunk_z,
        ],
    }
    _atomic_json(path, plan)
    print("capture_plan:", path)
    print("seeds:", ",".join(str(seed) for seed in seeds))
    return seeds, salt


def _progress(library_index: int, library_count: int):
    def report(done: int, total: int, _task: object) -> None:
        if done in (1, total) or done % 50 == 0:
            print(
                f"artifact {library_index}/{library_count}: "
                f"section {done}/{total}"
            )

    return report


def _load_plan(
    parser: argparse.ArgumentParser,
    path: Path,
) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                parser.error(f"capture plan repeats field {key!r}")
            result[key] = value
        return result

    try:
        plan = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read capture plan: {error}")
    if not isinstance(plan, dict):
        parser.error("capture plan must be a JSON object")
    return plan


def _native_evidence_hash(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> str:
    expected = current_native_evidence_jar_sha256()
    if args.native_evidence_jar is not None:
        actual = hashlib.sha256(args.native_evidence_jar.read_bytes()).hexdigest()
    else:
        actual = args.native_evidence_jar_sha256
    actual = _sha256(parser, actual, "native evidence bridge").upper()
    if actual != expected:
        parser.error(
            "native evidence bridge differs from the canonical pin: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _seed(parser: argparse.ArgumentParser, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**31:
        parser.error("capture seeds must be integers in [0, 2^31)")
    return value


def _sha256(
    parser: argparse.ArgumentParser,
    value: object,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        parser.error(f"{label} must be a SHA-256")
    return value.lower()


def _atomic_json(path: Path, value: object) -> None:
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
            json.dump(value, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
