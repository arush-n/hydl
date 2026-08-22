"""Stage bridge-bound Region evidence after a native bridge deployment."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
from typing import Any, Mapping, Sequence

from hytalegym.worldgen.native_lease import (
    NativeEvidenceLease,
    acquire_native_evidence_lease,
    update_owned_native_evidence_lease,
)
from hytalegym.worldgen.source_pins import (
    verify_installed_world_source_pins,
)
from hytalegym.worldgen.surrogate import default_hytale_assets_path
from hytalegym.worldgen.region.library import RegionArtifactLibrary
from hytalegym.worldgen.region.stability import (
    current_native_evidence_jar_sha256,
    load_region_seed_projection_contract,
)
from hytalegym.worldgen.region.traversal_library import (
    RegionTraversalGraphLibrary,
)

RECERTIFICATION_SCHEMA = "hytalerl_world_evidence_recertification_v12"
RECERTIFICATION_VERSION = 12
CONTRACT_CASCADE_SCHEMA = "hytalerl_world_contract_cascade_v2"
_TARGET_SEED = 570057


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_directory", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--control-seed", type=int)
    parser.add_argument("--deployed-bridge", type=Path)
    parser.add_argument(
        "--expected-bridge-sha256",
        required=True,
        help="explicit canonical bridge identity for this deployment",
    )
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--server-jar", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse a prior candidate stage after all identity gates pass",
    )
    args = parser.parse_args()

    report_path = recapture_world_evidence(
        args.stage_directory,
        repository_root=args.repository_root,
        host=args.host,
        port=args.port,
        control_seed=args.control_seed,
        deployed_bridge=args.deployed_bridge,
        expected_bridge_sha256=args.expected_bridge_sha256,
        assets=args.assets,
        server_jar=args.server_jar,
        resume=args.resume,
    )
    print(f"WORLD_EVIDENCE_RECERTIFICATION_CANDIDATE={report_path}")
    print("WORLD_EVIDENCE_RECERTIFICATION_PASS published=false")


def recapture_world_evidence(
    stage_directory: str | Path,
    *,
    repository_root: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 5556,
    control_seed: int | None = None,
    deployed_bridge: str | Path | None = None,
    expected_bridge_sha256: str | None = None,
    assets: str | Path | None = None,
    server_jar: str | Path | None = None,
    resume: bool = False,
) -> Path:
    """Capture a complete candidate without mutating published evidence."""

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else _repository_root()
    )
    stage = Path(stage_directory).resolve()
    with acquire_native_evidence_lease(
        root,
        host=host,
        port=port,
        purpose="world_recertification",
        stage=stage,
        evidence_kinds=("installed_source_pins", "bridge_differentials"),
    ) as lease:
        return _recapture_world_evidence_unlocked(
            stage,
            repository_root=root,
            host=host,
            port=port,
            control_seed=control_seed,
            deployed_bridge=deployed_bridge,
            expected_bridge_sha256=expected_bridge_sha256,
            assets=assets,
            server_jar=server_jar,
            progress_lease=lease,
            resume=resume,
        )


def _recapture_world_evidence_unlocked(
    stage: Path,
    *,
    repository_root: Path,
    host: str,
    port: int,
    control_seed: int | None,
    deployed_bridge: str | Path | None,
    assets: str | Path | None,
    server_jar: str | Path | None,
    progress_lease: NativeEvidenceLease,
    resume: bool,
    expected_bridge_sha256: str | None = None,
) -> Path:
    root = repository_root
    package_root = root / "hytalegym"
    fixture_root = package_root / "tests" / "fidelity" / "fixtures" / "world"
    region_root = root / "artifacts" / "worldgen" / "region-library-pilot-v2"
    graph_root = region_root / "traversal-v2"
    bridge = (
        Path(deployed_bridge).resolve()
        if deployed_bridge is not None
        else _default_deployed_bridge()
    )
    installed_assets = (
        Path(assets).resolve()
        if assets is not None
        else default_hytale_assets_path().resolve()
    )
    installed_server = (
        Path(server_jar).resolve()
        if server_jar is not None
        else (installed_assets.parent / "Server" / "HytaleServer.jar").resolve()
    )
    _prepare_stage(
        stage,
        published_roots=(fixture_root, graph_root),
        allow_existing=resume,
    )
    source_pin_path = stage / "source-pin-staleness.json"
    source_pin_failure = stage / "source-pin-staleness-failure.json"
    _update_recertification_progress(
        progress_lease,
        phase="source_pins",
        phase_status="running",
    )
    try:
        source_pin_report = verify_installed_world_source_pins(
            package_root / "tests" / "fidelity",
            installed_assets,
            installed_server,
        )
    except Exception as error:
        _write_json(
            source_pin_failure,
            {
                "schema": "hytalerl_world_source_pin_staleness_failure_v1",
                "status": "stale_or_unavailable",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            },
        )
        _update_recertification_progress(
            progress_lease,
            phase="source_pins",
            phase_status="failed",
            detail={"failure_report": source_pin_failure.name},
        )
        raise
    source_pin_failure.unlink(missing_ok=True)
    _write_json(source_pin_path, source_pin_report)
    _update_recertification_progress(
        progress_lease,
        phase="bridge_identity",
        phase_status="running",
        detail={
            "source_pin_report_semantic_sha256": source_pin_report[
                "report_semantic_sha256"
            ]
        },
    )
    canonical_bridge = current_native_evidence_jar_sha256()
    expected_bridge = (
        canonical_bridge
        if expected_bridge_sha256 is None
        else _sha256(expected_bridge_sha256)
    )
    if expected_bridge != canonical_bridge:
        raise RuntimeError(
            "requested bridge differs from canonical world evidence: "
            f"requested={expected_bridge} canonical={canonical_bridge}"
        )
    observed_bridge = _file_sha256(bridge).upper()
    if observed_bridge != expected_bridge:
        raise RuntimeError(
            "deployed bridge differs from canonical world evidence: "
            f"observed={observed_bridge} expected={expected_bridge}"
        )

    _update_recertification_progress(
        progress_lease,
        phase="contract_cascade",
        phase_status="running",
    )
    cascade_path = stage / "contract-cascade.json"
    _write_json(
        cascade_path,
        derive_contract_cascade(root, expected_bridge),
    )
    _require_listener(host, port)

    selected_control = (
        _random_control_seed() if control_seed is None else _control_seed(control_seed)
    )
    candidate_fixtures = stage / "fixtures"
    candidate_fixtures.mkdir(exist_ok=resume)
    projection_contract, traversal_recording = build_evidence_candidates(
        fixture_root,
        candidate_fixtures,
        bridge_sha256=expected_bridge,
        captured_at_utc=datetime.now(UTC).date().isoformat(),
    )

    region_manifest = region_root / "manifest.json"
    capacity_report = root / "artifacts" / "worldgen" / "world-token-capacity-v4.json"
    expected_graph_manifest = graph_root / "traversal-manifest.json"
    graph_candidate = stage / "traversal-v2"
    scripts = package_root / "tests" / "fidelity"
    candidate_region = stage / f"native_region_candidate_seed_{_TARGET_SEED}.npz"
    difference_report = stage / "native_region_difference.json"
    commands = (
        (
            "region_seed_projection",
            (
                sys.executable,
                str(scripts / "native_region_capture_compare.py"),
                "--host",
                host,
                "--port",
                str(port),
                "--seed",
                str(_TARGET_SEED),
                "--interposed-control-seed",
                str(selected_control),
                "--server-lifecycle-assumption",
                "existing_process",
                "--native-evidence-jar",
                str(bridge),
                "--seed-projection-contract",
                str(projection_contract),
                "--output",
                str(candidate_region),
                "--difference-report",
                str(difference_report),
            ),
        ),
        (
            "stationary_traversal",
            (
                sys.executable,
                str(scripts / "surrogate_traversal_native_compare.py"),
                str(region_manifest),
                str(capacity_report),
                "--host",
                host,
                "--port",
                str(port),
                "--deployed-bridge",
                str(bridge),
                "--expect-recording",
                str(traversal_recording),
            ),
        ),
        (
            "region_traversal_graphs",
            (
                sys.executable,
                str(scripts / "native_region_traversal_graph_compare.py"),
                str(region_manifest),
                str(graph_candidate),
                "--host",
                host,
                "--port",
                str(port),
                "--deployed-bridge",
                str(bridge),
                "--expect-manifest",
                str(expected_graph_manifest),
            ),
        ),
        *_navigation_probe_commands(
            scripts,
            region_manifest=region_manifest,
            traversal_manifest=graph_candidate / "traversal-manifest.json",
            host=host,
            port=port,
            control_seed=selected_control,
            recording_directory=candidate_fixtures,
        ),
        *_world_action_probe_commands(
            scripts,
            host=host,
            port=port,
            bridge=bridge,
            bridge_sha256=expected_bridge,
            assets=installed_assets,
            server_jar=installed_server,
            control_seed=selected_control,
            recording_directory=candidate_fixtures,
        ),
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)
    for phase, command in commands:
        _update_recertification_progress(
            progress_lease,
            phase=phase,
            phase_status="running",
        )
        subprocess.run(
            command,
            cwd=package_root,
            env=environment,
            check=True,
        )

    library = RegionArtifactLibrary.load(region_manifest)
    graphs = RegionTraversalGraphLibrary.load(
        graph_candidate / "traversal-manifest.json",
        library,
    )
    graph_bridge = graphs.manifest["native_evidence_jar_sha256"]
    if graph_bridge != expected_bridge:
        raise RuntimeError("candidate graph library retained stale bridge evidence")
    report: dict[str, Any] = {
        "schema": RECERTIFICATION_SCHEMA,
        "version": RECERTIFICATION_VERSION,
        "status": "validated_candidate_not_published",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "host": host,
        "port": port,
        "bridge_sha256": expected_bridge,
        "installed_hytale_version": source_pin_report["installed_build"][
            "implementation_version"
        ],
        "installed_hytale_revision": source_pin_report["installed_build"][
            "implementation_revision"
        ],
        "source_pin_report_semantic_sha256": source_pin_report[
            "report_semantic_sha256"
        ],
        "target_seed": _TARGET_SEED,
        "random_interposed_control_seed": selected_control,
        "source_region_library_semantic_sha256": library.semantic_sha256,
        "candidate_traversal_library_semantic_sha256": graphs.semantic_sha256,
        "candidates": {
            "source_pin_staleness_report": _candidate_row(
                stage,
                source_pin_path,
            ),
            "contract_cascade": _candidate_row(stage, cascade_path),
            "seed_projection_contract": _candidate_row(
                stage,
                projection_contract,
            ),
            "stationary_traversal_recording": _candidate_row(
                stage,
                traversal_recording,
            ),
            "stationary_traversal_artifact": _candidate_row(
                stage,
                candidate_fixtures / "native_traversal_projection_v2_recording.npz",
            ),
            "region_snapshot": _candidate_row(stage, candidate_region),
            "traversal_manifest": _candidate_row(
                stage,
                graph_candidate / "traversal-manifest.json",
            ),
            "navigation_path_recording": _candidate_row(
                stage,
                candidate_fixtures / "native_navigation_path_recording.json",
            ),
            "walk_probe_subset_recording": _candidate_row(
                stage,
                candidate_fixtures / "region_walk_probe_subset_recording.json",
            ),
            "mutable_block_recording": _candidate_row(
                stage,
                candidate_fixtures / "native_mutable_block_recording.json",
            ),
            "explosion_mutation_recording": _candidate_row(
                stage,
                candidate_fixtures / "native_explosion_mutation_recording.json",
            ),
            "explosion_dynamics_recording": _candidate_row(
                stage,
                candidate_fixtures / "native_explosion_dynamics_recording.json",
            ),
            "falling_block_recording": _candidate_row(
                stage,
                candidate_fixtures / "native_falling_block_recording.json",
            ),
            "drop_program_recording": _candidate_row(
                stage,
                candidate_fixtures / "native_drop_program_recording.json",
            ),
            "crafting_catalog_recording": _candidate_row(
                stage,
                candidate_fixtures / "native_crafting_catalog_recording.json",
            ),
            "item_interaction_recording": _candidate_row(
                stage,
                candidate_fixtures / "native_item_interaction_recording.json",
            ),
            "block_use_recording": _candidate_row(
                stage,
                candidate_fixtures / "native_block_use_recording.json",
            ),
            "world_action_capabilities_recording": _candidate_row(
                stage,
                candidate_fixtures / "native_world_action_capabilities_recording.json",
            ),
        },
        "publication": {
            "automatic": False,
            "reason": (
                "cross-directory evidence promotion is deliberately separate "
                "from native capture and review"
            ),
            "targets": {
                "seed_projection_contract": projection_contract.relative_to(
                    stage
                ).as_posix(),
                "stationary_traversal_recording": (
                    traversal_recording.relative_to(stage).as_posix()
                ),
                "traversal_directory": graph_candidate.relative_to(stage).as_posix(),
                "navigation_path_recording": (
                    candidate_fixtures / "native_navigation_path_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
                "walk_probe_subset_recording": (
                    candidate_fixtures / "region_walk_probe_subset_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
                "mutable_block_recording": (
                    candidate_fixtures / "native_mutable_block_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
                "explosion_mutation_recording": (
                    candidate_fixtures / "native_explosion_mutation_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
                "explosion_dynamics_recording": (
                    candidate_fixtures / "native_explosion_dynamics_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
                "falling_block_recording": (
                    candidate_fixtures / "native_falling_block_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
                "drop_program_recording": (
                    candidate_fixtures / "native_drop_program_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
                "crafting_catalog_recording": (
                    candidate_fixtures / "native_crafting_catalog_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
                "item_interaction_recording": (
                    candidate_fixtures / "native_item_interaction_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
                "block_use_recording": (
                    candidate_fixtures / "native_block_use_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
                "world_action_capabilities_recording": (
                    candidate_fixtures
                    / "native_world_action_capabilities_recording.json"
                )
                .relative_to(stage)
                .as_posix(),
            },
        },
    }
    destination = stage / "recertification-manifest.json"
    _write_json(destination, report)
    _update_recertification_progress(
        progress_lease,
        phase="complete",
        phase_status="passed",
        detail={"candidate_manifest": destination.name},
    )
    return destination


def _navigation_probe_commands(
    scripts: Path,
    *,
    region_manifest: Path,
    traversal_manifest: Path,
    host: str,
    port: int,
    control_seed: int,
    recording_directory: Path,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    common = (
        str(region_manifest),
        str(traversal_manifest),
    )
    return (
        (
            "native_navigation_paths",
            (
                sys.executable,
                str(scripts / "surrogate_navigation_native_compare.py"),
                *common,
                str(recording_directory / "native_navigation_path_recording.json"),
                "--host",
                host,
                "--port",
                str(port),
                "--control-selection-key",
                str(control_seed),
            ),
        ),
        (
            "native_walk_probe_subset",
            (
                sys.executable,
                str(scripts / "surrogate_navigation_probe_subset_native_compare.py"),
                *common,
                str(recording_directory / "region_walk_probe_subset_recording.json"),
                "--host",
                host,
                "--port",
                str(port),
                "--selection-key",
                str(_TARGET_SEED),
                "--selection-key",
                "570058",
                "--selection-key",
                str(control_seed),
                "--control-selection-key",
                str(control_seed),
            ),
        ),
    )


def _world_action_probe_commands(
    scripts: Path,
    *,
    host: str,
    port: int,
    bridge: Path,
    bridge_sha256: str,
    assets: Path,
    server_jar: Path,
    control_seed: int,
    recording_directory: Path,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    common = (
        "--host",
        host,
        "--port",
        str(port),
        "--assets",
        str(assets),
        "--server-jar",
        str(server_jar),
        "--control-seed",
        str(control_seed),
        "--recording-directory",
        str(recording_directory),
        "--native-evidence-jar",
        str(bridge),
    )
    return (
        (
            "mutable_block_transition",
            (
                sys.executable,
                str(scripts / "surrogate_mutable_block_native_compare.py"),
                *common,
            ),
        ),
        (
            "explosion_mutation",
            (
                sys.executable,
                str(scripts / "surrogate_explosion_mutation_native_compare.py"),
                *common,
            ),
        ),
        (
            "explosion_entity_dynamics",
            (
                sys.executable,
                str(scripts / "surrogate_explosion_dynamics_native_compare.py"),
                *common,
            ),
        ),
        (
            "falling_block_motion",
            (
                sys.executable,
                str(scripts / "surrogate_falling_block_native_compare.py"),
                "--host",
                host,
                "--port",
                str(port),
                "--assets",
                str(assets),
                "--server-jar",
                str(server_jar),
                "--control-seed",
                str(control_seed),
                "--native-evidence-jar-sha256",
                _sha256(bridge_sha256),
                "--record",
                str(recording_directory / "native_falling_block_recording.json"),
            ),
        ),
        (
            "drop_program_distributions",
            (
                sys.executable,
                str(scripts / "surrogate_drop_program_native_compare.py"),
                *common,
            ),
        ),
        (
            "resolved_crafting_catalog",
            (
                sys.executable,
                str(scripts / "surrogate_crafting_native_compare.py"),
                *common,
            ),
        ),
        (
            "resolved_item_interactions",
            (
                sys.executable,
                str(scripts / "surrogate_item_interaction_native_compare.py"),
                *common,
            ),
        ),
        (
            "native_block_use",
            (
                sys.executable,
                str(scripts / "surrogate_block_use_native_compare.py"),
                *common,
            ),
        ),
        (
            "world_action_capabilities",
            (
                sys.executable,
                str(scripts / "surrogate_world_action_capabilities_native_compare.py"),
                *common,
            ),
        ),
    )


def build_evidence_candidates(
    fixture_root: str | Path,
    destination: str | Path,
    *,
    bridge_sha256: str,
    captured_at_utc: str,
) -> tuple[Path, Path]:
    """Build bridge-only candidate manifests while retaining frozen arrays."""

    source = Path(fixture_root)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    bridge = _sha256(bridge_sha256)

    projection_name = "native_region_seed_projection_contract.json"
    projection = _read_json(source / projection_name)
    projection["evidence_bridge_sha256"] = bridge
    projection_path = target / projection_name
    _write_json(projection_path, projection)
    load_region_seed_projection_contract(projection_path).require_evidence_bridge(
        bridge
    )

    recording_name = "native_traversal_projection_v2_recording.json"
    recording = _read_json(source / recording_name)
    artifact_name = recording.get("artifact")
    if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
        raise ValueError("stationary traversal artifact must be a basename")
    source_artifact = source / artifact_name
    target_artifact = target / artifact_name
    shutil.copy2(source_artifact, target_artifact)
    if _file_sha256(source_artifact) != _file_sha256(target_artifact):
        raise RuntimeError("stationary traversal artifact copy changed bytes")

    recording["evidence_bridge_sha256"] = bridge.lower()
    recording["captured_at_utc"] = captured_at_utc
    recording.pop("report_semantic_sha256", None)
    semantic = dict(recording)
    semantic.pop("evidence_bridge_sha256", None)
    semantic.pop("reference_artifact_bridge_sha256", None)
    recording["report_semantic_sha256"] = _semantic_sha256(semantic)
    recording_path = target / recording_name
    _write_json(recording_path, recording)
    return projection_path, recording_path


def derive_contract_cascade(
    repository_root: str | Path,
    bridge_sha256: str,
) -> dict[str, object]:
    """Re-derive the current cross-lane contract chain without publishing."""

    from hytalegym.jax.combat.arsenal.schema.spec import (
        combat_arsenal_contract_manifest,
        combat_arsenal_contract_sha256,
    )
    from hytalegym.jax.combat.observation.v3.policy import (
        arsenal_policy_contract_manifest,
        arsenal_policy_contract_sha256,
    )
    from hytalegym.jax.combat.observation.v3.policy.surface import (
        action_surface_staging_contract_manifest,
        action_surface_staging_contract_sha256,
    )
    from hytalegym.jax.combat.observation.v3.schema.spec import (
        learner_observation_v3_contract_sha256,
    )
    from hytalegym.jax.world.tokens import (
        world_geometry_token_contract_sha256,
    )

    root = Path(repository_root).resolve()
    bridge = _sha256(bridge_sha256)
    region = RegionArtifactLibrary.load(
        root / "artifacts" / "worldgen" / "region-library-pilot-v2" / "manifest.json"
    )
    traversal = RegionTraversalGraphLibrary.load(
        root
        / "artifacts"
        / "worldgen"
        / "region-library-pilot-v2"
        / "traversal-v2"
        / "traversal-manifest.json",
        region,
    )
    world_token = world_geometry_token_contract_sha256().upper()
    learner = learner_observation_v3_contract_sha256().upper()
    arsenal = combat_arsenal_contract_sha256().upper()
    policy = arsenal_policy_contract_sha256().upper()
    action_surface = action_surface_staging_contract_sha256().upper()
    arsenal_manifest = combat_arsenal_contract_manifest()
    policy_manifest = arsenal_policy_contract_manifest()
    action_surface_manifest = action_surface_staging_contract_manifest()
    if arsenal_manifest["evidence"]["native_evidence_jar_sha256"] != bridge:
        raise RuntimeError("Combat Arsenal retained a stale bridge identity")
    relationships = {
        "policy_to_world_token": (
            policy_manifest["world_geometry_token_contract_sha256"].upper()
            == world_token
        ),
        "policy_to_learner": (
            policy_manifest["learner_observation_v3_sha256"].upper() == learner
        ),
        "policy_to_arsenal": (policy_manifest["arsenal_sha256"].upper() == arsenal),
        "action_surface_to_policy": (
            action_surface_manifest["current_public_policy_sha256"].upper() == policy
        ),
    }
    if not all(relationships.values()):
        raise RuntimeError("derived contract cascade is internally inconsistent")
    return {
        "schema": CONTRACT_CASCADE_SCHEMA,
        "bridge_sha256": bridge,
        "dependency_order": [
            "region_library",
            "traversal_library",
            "world_geometry_token",
            "learner_observation_v3",
            "combat_arsenal",
            "arsenal_policy",
            "action_surface_staging",
        ],
        "identities": {
            "region_library": region.semantic_sha256.upper(),
            "traversal_library": traversal.semantic_sha256.upper(),
            "world_geometry_token": world_token,
            "learner_observation_v3": learner,
            "combat_arsenal": arsenal,
            "arsenal_policy": policy,
            "action_surface_staging": action_surface,
        },
        "relationships": relationships,
        "shape": {
            "observation_size": policy_manifest["observation"]["size"],
            "action_logit_size": policy_manifest["action"]["logit_size"],
            "action_head_sizes": policy_manifest["action"]["head_sizes"],
        },
        "publication": {
            "automatic": False,
            "historical_evidence_relabeling": False,
            "consumer_artifact_recapture": "owner_required",
        },
    }


def _prepare_stage(
    stage: Path,
    *,
    published_roots: Sequence[Path],
    allow_existing: bool = False,
) -> None:
    for root in published_roots:
        if stage == root.resolve() or stage.is_relative_to(root.resolve()):
            raise ValueError("stage directory must be outside published evidence")
    if stage.exists() and any(stage.iterdir()) and not allow_existing:
        raise FileExistsError("stage directory must be absent or empty")
    stage.mkdir(parents=True, exist_ok=True)


def _update_recertification_progress(
    lease: NativeEvidenceLease,
    *,
    phase: str,
    phase_status: str,
    detail: Mapping[str, object] | None = None,
) -> None:
    update_owned_native_evidence_lease(
        lease,
        phase=phase,
        phase_status=phase_status,
        detail=detail,
    )


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "hytalegym" / "pyproject.toml").is_file() and (
            candidate / "artifacts" / "worldgen"
        ).is_dir():
            return candidate
    raise RuntimeError("HytaleRL repository root is unavailable")


def _default_deployed_bridge() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is unavailable; pass --deployed-bridge")
    return (
        Path(appdata)
        / "Hytale"
        / "install"
        / "release"
        / "package"
        / "game"
        / "latest"
        / "Server"
        / "mods"
        / "HytaleRLBridge-0.1.0.jar"
    ).resolve()


def _require_listener(host: str, port: int) -> None:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as error:
        raise RuntimeError(
            f"native Hytale listener is unavailable at {host}:{port}"
        ) from error


def _random_control_seed() -> int:
    while True:
        value = secrets.randbelow(2**31 - 1) + 1
        if value != _TARGET_SEED:
            return value


def _control_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("control seed must be an integer")
    if not 1 <= value < 2**31 or value == _TARGET_SEED:
        raise ValueError("control seed must be positive int32 and differ from target")
    return value


def _candidate_row(stage: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(stage).as_posix(),
        "file_sha256": _file_sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _semantic_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(value: str) -> str:
    result = value.upper()
    if len(result) != 64 or any(
        character not in "0123456789ABCDEF" for character in result
    ):
        raise ValueError("bridge SHA-256 must be 64 hexadecimal characters")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
