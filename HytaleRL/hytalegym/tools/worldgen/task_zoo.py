#!/usr/bin/env python3
"""Reach and pursue task families over a Region library, for one server line.

Tracked rebuild of the zoo that the private tool `agent_skills/training_runs/
tools/task_zoo.py` produced. That file is not read, listed or copied here. The
shape is the one `arena/tests/test_private_task_suite.py` pins (read only, not
imported): `families()` returns {family: suite}; each suite has `terrains`
({name: {"seed", "split", ...}}) and `tasks` (a list of {"name", "settings",
"tags"}).

Run from any directory with the repo venv, for example:
  /Users/arush/hydl/.venv/bin/python /Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/task_zoo.py --server-version 0.5.7 --json <new file in scratch>
  /Users/arush/hydl/.venv/bin/python /Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/task_zoo.py --server-version 0.7.2 --horizons 512,768 --json <new file>

Inputs (confirmed by reading):
  Library: `manifest.json` (schema hytalerl_region_artifact_library_v1) under
    0.7.2 -> HytaleRL/artifacts/worldgen/region-library-pilot-v2-072 (staged by another lane, not promoted; its STATUS.md
             says the manifest FAILS RegionArtifactLibrary.load: a chunk API fingerprint check, and six split labels that differ
             from the package's split rule)
    0.5.7 -> HytaleRL/artifacts/worldgen/region-library-pilot-v2 (20 entries: 16 train, 4 heldout)
    --library DIR overrides either one.
  Tick base (seconds per environment step), imported from the profiles:
    0.7.2 -> 1/30 s   (hytalegym/sim/gen7/profile.py:71)
    0.5.7 -> 0.045 s  (hytalegym/sim/common/profile.py:163, the gen-5 loaded tick; the test pins it)

Rules. "Pinned" means the test asserts it. Everything else is inferred, a choice made here:
  1. Terrains. Pinned: split "train" and a seed outside the reserved set. Inferred: the reserved
     set is the two seeds the test pins, because no tracked doc lists the audit seeds; GEN7-PLAN.md:431
     says audit worlds stay unseen. A seed is trainable only if its manifest label is "train" AND the
     package's split rule does not select it as held-out (`sha256_rank_v1` re-implemented here because
     importing the package loads jax; under `inherited_seed_labels_v1` the loader certifies the labels
     themselves, so the rule is the labels, 2026-10-19). The union is the fail-closed choice: a manifest
     whose labels and rule disagree lists the seeds per suite under `split_disagreements` (the 0.7.2
     library did on six seeds until it was restaged with the inherited method). Terrains are keyed
     "seed_<seed>" and sorted by seed. Excluded entries are
     counted per suite under `excluded` (reserved, heldout, package_heldout), never listed.
  2. Families. Pinned: `pursuit_perception` (objective pursue, perception field_of_view). Inferred:
     `reach` (objective reach_point, no perception key) and `pursuit` (objective pursue, perception
     omniscient, the PursuitStageConfig default at arena/training/contracts/pursuit.py:319).
  3. Tasks. Inferred: one task per family, terrain and horizon, enumerated in that order. Names are
     "<family>/<terrain>/h<ticks>", so no name repeats across suites (pinned: a shared name must be
     the same task).
  4. Horizons. Inferred default (512,) = PURSUIT_DEFAULT_TICKS (arena/training/contracts/pursuit.py:59).
     Pinned band 512..1000 ticks (1000 = PURSUIT_MAXIMUM_TICKS, arena/training/contracts/pursuit.py:60).
     The catch-time horizons tool (catch_time_horizons.py) is the intended source of other values.
  5. Settings. Inferred, named after PursuitRunConfig fields (arena/training/runs/pursuit_run.py:399,
     429-431): objective, world_artifact_seed, world_split "train", node_seed = SUITE_NODE_SEED = 570071
     (the runner default at pursuit_run.py:429; the test pins only equality), evaluation_steps and
     episode_ticks both equal to the horizon, and stage_options.
  6. Tags. Pinned: horizon_seconds == round(ticks * tick_seconds, 1), checked at 0.045 s. Inferred: family,
     terrain and horizon_ticks.

Not verified: no tracked runner accepts objective reach_point (the name occurs only as an intent family
in experimental/neural_agent, not as a runner objective), so reach tasks are declarations only. The 0.7.2
output is provisional: it reads a manifest that the package loader rejects (see Inputs). The test calls
the private `families()` with no arguments and pins the 0.045 s base; this tool's default is 0.7.2, so
that call does not apply here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
HYTALE_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = HYTALE_ROOT / "artifacts" / "worldgen"
sys.path.insert(0, str(PACKAGE_ROOT))

from hytalegym.sim.common.profile import StepDuration  # noqa: E402
from hytalegym.sim.gen7.profile import PROFILE as GEN7_PROFILE  # noqa: E402

MANIFEST_SCHEMA = "hytalerl_region_artifact_library_v1"
LIBRARIES = {
    "0.7.2": ARTIFACTS / "region-library-pilot-v2-072",
    "0.5.7": ARTIFACTS / "region-library-pilot-v2",
}
TICK_SECONDS = {
    "0.7.2": GEN7_PROFILE.step_duration.nominal_seconds,
    "0.5.7": StepDuration().loaded_seconds,
}
SUITE_NODE_SEED = 570_071
AUDIT_WORLD_SEEDS = frozenset({2114253645, 653448846})
HORIZON_MIN, HORIZON_MAX = 512, 1000
DEFAULT_HORIZONS = (512,)
#: (family, objective, stage_options). This order is the enumeration order.
FAMILIES = (
    ("reach", "reach_point", {}),
    ("pursuit", "pursue", {"perception": "omniscient"}),
    ("pursuit_perception", "pursue", {"perception": "field_of_view"}),
)


SPLIT_METHOD = "sha256_rank_v1"  # hytalegym/worldgen/region/library.py:26
INHERITED_SPLIT_METHOD = "inherited_seed_labels_v1"  # library.py: labels certified against the source library


def package_heldout_seeds(manifest: dict[str, Any]) -> set[int]:
    """Held-out seeds that the package's split rule selects for this manifest.

    `sha256_rank_v1`: re-implements `_split_rank` (library.py) and the sort in
    `create_region_artifact_library`: the first `heldout_count` entries by (rank, semantic_sha256)
    are held out. The test checks it against the 0.5.7 manifest labels. `inherited_seed_labels_v1`
    (2026-10-19): the loader certifies the manifest's own labels against the source library, so the
    rule is those labels. Any other method fails closed.
    """
    method = manifest["split_contract"]["method"]
    if method == INHERITED_SPLIT_METHOD:
        return {int(entry["seed"]) for entry in manifest["entries"] if entry["split"] == "heldout"}
    if method != SPLIT_METHOD:
        raise ValueError(f"unsupported Region artifact split method {method!r}")
    salt = manifest["split_contract"]["salt_sha256"]
    ranked = sorted(
        manifest["entries"],
        key=lambda entry: (
            hashlib.sha256(
                f"{SPLIT_METHOD}\0{salt}\0{entry['semantic_sha256']}".encode()
            ).digest(),
            entry["semantic_sha256"],
        ),
    )
    return {int(entry["seed"]) for entry in ranked[: int(manifest["heldout_count"])]}


def load_manifest(library: Path) -> dict[str, Any]:
    path = library / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no Region library manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"{path}: schema {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r}"
        )
    return manifest


def terrains_from(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[int]]:
    package_heldout = package_heldout_seeds(manifest)
    terrains: dict[str, dict[str, Any]] = {}
    excluded: dict[str, int] = {"reserved": 0, "heldout": 0, "package_heldout": 0}
    disagreements: list[int] = []
    for entry in sorted(manifest["entries"], key=lambda item: int(item["seed"])):
        seed = int(entry["seed"])
        label_heldout = entry["split"] != "train"
        rule_heldout = seed in package_heldout
        if label_heldout != rule_heldout:
            disagreements.append(seed)
        if seed in AUDIT_WORLD_SEEDS:
            excluded["reserved"] += 1
        elif label_heldout:
            excluded["heldout"] += 1
        elif rule_heldout:
            excluded["package_heldout"] += 1
        else:
            name = f"seed_{seed}"
            if name in terrains:
                raise ValueError(f"the library lists seed {seed} twice")
            terrains[name] = {
                "seed": seed,
                "split": "train",
                "semantic_sha256": entry["semantic_sha256"],
                "path": entry["path"],
            }
    return terrains, excluded, disagreements


def families(
    server_version: str = "0.7.2",
    *,
    library: Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, dict[str, Any]]:
    """Return {family: suite} for one server line. The rules are in the module docstring."""
    if server_version not in LIBRARIES:
        raise ValueError(f"server_version must be one of {sorted(LIBRARIES)}")
    ticks_each = tuple(sorted({int(ticks) for ticks in horizons}))
    if not ticks_each or any(not HORIZON_MIN <= ticks <= HORIZON_MAX for ticks in ticks_each):
        raise ValueError(
            f"horizons must lie in {HORIZON_MIN}..{HORIZON_MAX} ticks, got {list(horizons)}"
        )
    root = Path(library) if library is not None else LIBRARIES[server_version]
    manifest = load_manifest(root)
    terrains, excluded, disagreements = terrains_from(manifest)
    if not terrains:
        raise ValueError(f"{root}: no train terrains after exclusions")
    tick = TICK_SECONDS[server_version]
    suites: dict[str, dict[str, Any]] = {}
    for family, objective, stage_options in FAMILIES:
        tasks = []
        for terrain_name, terrain in terrains.items():
            for ticks in ticks_each:
                tasks.append({
                    "name": f"{family}/{terrain_name}/h{ticks}",
                    "settings": {
                        "objective": objective,
                        "world_artifact_seed": terrain["seed"],
                        "world_split": "train",
                        "node_seed": SUITE_NODE_SEED,
                        "evaluation_steps": ticks,
                        "episode_ticks": ticks,
                        "stage_options": dict(stage_options),
                    },
                    "tags": {
                        "family": family,
                        "terrain": terrain_name,
                        "horizon_ticks": ticks,
                        "horizon_seconds": round(ticks * tick, 1),
                    },
                })
        suites[family] = {
            "server_version": server_version,
            "tick_seconds": tick,
            "library": str(root),
            "library_semantic_sha256": manifest["library_semantic_sha256"],
            "excluded": dict(excluded),
            "split_disagreements": list(disagreements),
            "terrains": {name: dict(terrain) for name, terrain in terrains.items()},
            "tasks": tasks,
        }
    return suites


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the reach and pursue task zoo for one server line.")
    parser.add_argument("--server-version", choices=sorted(LIBRARIES), default="0.7.2")
    parser.add_argument("--library", type=Path, help="Region library directory (overrides the version default)")
    parser.add_argument(
        "--horizons",
        default="512",
        help="comma-separated horizons in ticks, each 512..1000 (default: 512)",
    )
    parser.add_argument("--json", type=Path, metavar="OUT", help="write the families as JSON to OUT")
    args = parser.parse_args(argv)
    horizons = tuple(int(part) for part in args.horizons.split(",") if part.strip())
    suites = families(args.server_version, library=args.library, horizons=horizons)
    first = next(iter(suites.values()))
    print(f"server {args.server_version}: library {first['library']}, tick {first['tick_seconds']} s")
    for family, suite in suites.items():
        print(
            f"{family}: {len(suite['terrains'])} terrains, {len(suite['tasks'])} tasks, "
            f"excluded {suite['excluded']}"
        )
    if args.json is not None:
        args.json.write_text(json.dumps(suites, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
