"""Assert the repository is safe to publish.

Two questions, both answered by ``git check-ignore`` exit codes rather than by
reading ``.gitignore`` and believing it:

1. Does anything that must never be published still resolve as trackable?
2. Did an ignore rule get broad enough to swallow real source or documentation?

Question 2 is not hypothetical. Three rules in this repository's ``.gitignore``
each failed it at once:

  ``**.json``  excluded every JSON in the tree, including ``console/package.json``
  ``docs/``    excluded ~258 documentation files across seven directories
  ``.gitignore`` listed itself, so a fresh clone inherited no rules at all

Run: ``python tools/check_publication_hygiene.py``
Exit code 0 means clean; 1 means at least one assertion failed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths that must never be publishable. Representative members are enough --
# git resolves the rule, and we care about the rule, not the file count.
MUST_BE_IGNORED = (
    "decompiled/com/hypixel/hytale/assetstore/AssetConstants.java",
    "decompiled/anything.java",
    "HytaleRL/.codex-local/decompiled-actions/x.java",
    "artifacts/console-hytale/mods/HytalePolicyAgent-0.1.0.jar",
    "artifacts/console-hytale/server-candidate-X/universe/worlds/default/resources/Time.json",
    "HytaleRL/artifacts/native-npc-contract-v8-07f8/report.json",
    ".codex-local/agent-state/adk/HANDOFF-ADK.md",
    "output/playwright/x.png",
    ".playwright-cli/console.log",
    "some/built/plugin.jar",
    "AGENT-INSTRUCTIONS-PUBLICATION.md",
    # Per-package developer documentation is contributor material. User-facing
    # docs are README.md, which is deliberately NOT ignored.
    "DEV.md",
    "docs/DEV.md",
    "HytaleRL/hytale-plugin/src/main/java/com/hytalerlbridge/nativebackend/DEV.md",
    # The root design/knowledge/register tree is development history. The rule
    # is `/docs/`, anchored: the pairing with `adk/docs/...` in
    # MUST_BE_TRACKABLE catches a future unanchored edit.
    "docs/TRAINING-AN-AGENT.md",
    "docs/designs/PORT-CRITICAL-PATH.md",
    "docs/registers/DECOMPILED-SURVEY.md",
    # The gym's reverse-engineering evidence tree is contributor material.
    "HytaleRL/docs/combat/arsenal-v1/hytale-0.5.7-evidence.md",
    "HytaleRL/docs/worldgen/evidence/runtime-layout.md",
    # Contributor-facing design notes, same category as DEV.md.
    "experimental/3d_audio/hearing/DESIGN.md",
    "experimental/cnn_training/MIGRATION.md",
    "experimental/worldgen-v2/AUDIT-0.5.7.md",
    "adk/docs/CAPABILITY-AUDIT.md",
    "adk/docs/IMITATION-WORKFLOW.md",
    "adk/docs/CATALOG.md",
    "adk/docs/PERFORMANCE.md",
    "adk/docs/WIRING-NOTES.md",
    "adk/CHANGELOG.md",
    "adk/ISSUES.md",
    "agents/ppo/TRAINING-PLAN.md",
    "agents/design/REVISIONS.md",
    "agents/dawn/artifacts/README.md",
    "adk/tools/replay_data.json",
    "adk/tools/region_map3d.json",
    "adk/tools/REPLAY_VIEWER_SPEC.md",
    "HytaleRL/hytale-plugin/whitelist.json",
    "HytaleRL/hytale-plugin/telemetry/2026-08-21_20-43-37_fc0e881e.jsonl",
    "HytaleRL/hytale-plugin/mods/Hytale_HytaleGenerator/biome_editor.json",
    "experimental/jvm-agent/harness/Main.java",
    "experimental/jvm-agent/run_tests.sh",
    "experimental/jvm-agent/deployment/native-equipment-fixture/manifest.json",
    "experimental/jvm-agent/src/com/hytalerlbridge/policy/testing/HeadlessCombatFixture.java",
    "experimental/worldgen-v2/verification/conftest.py",
    "experimental/worldgen-v2/jax_port/evidence/worldgen-v2-r1-plains-570059-load.receipt",
    "adk/probes/HEAD-LEGALITY.md",
    "adk/probes/head_legality_v1.json",
    "console/tools/backfill_pursuit_replay_world.py",
    "HytaleRL/scripts/docs/migrate_readme_status.py",
    "HytaleRL/hytalegym/hytalegym/jax/combat/PACKAGE-MAP.md",
    "HytaleRL/hytalegym/hytalegym/jax/combat/observation/v3/NATIVE-PORT.md",
    "HytaleRL/examples/jax_arsenal_benchmark.py",
    "HytaleRL/examples/jax_force_clearance_packing_prototype.py",
    "HytaleRL/examples/audit_hytale_combat_assets.py",
    "HytaleRL/examples/jax_arsenal_outcome_controls.py",
    "experimental/worldgen-v2/jax_port/verification/minigame_runtime_check.py",
    "experimental/worldgen-v2/jax_port/bundles/benchmark_bundle.py",
    "experimental/worldgen-v2/custom_environments/audit_corpus.py",
    "experimental/worldgen-v2/procedural_generation/audit.py",
    "experimental/worldgen-v2/tools/surrogate_biome_asset_audit.py",
    "experimental/worldgen-v2/tools/surrogate_biome_prop_asset_audit.py",
    "experimental/worldgen-v2/tools/surrogate_world_offline_smoke.py",
    "agents/basic/TRAINING.md",
    "console/docs/TRAINING-CONTROLS.md",
    "console/docs/LIVE-SERVER.md",
    "console/docs/EXTENDING.md",
    "agents/profiles/dawn-pursuit-v7-incumbent.json",
    "agents/profiles/dawn-vision-locomotion-g2.json",
    "agents/profiles/combat-32x32-fixture.json",
    # Generated session telemetry, an external mod config, and experimental
    # trees with no published consumer.
    "telemetry/2026-08-17_18-16-38_f662bfa2.jsonl.gz",
    "mods/Hytale_HytaleGenerator/biome_editor.json",
    "experimental/cnn_training/pipeline.py",
    "experimental/hierarchical_planner/DESIGN.md",
    "adk/docs/surface-survey-v1.json",
    "adk/docs/surface-survey-geometry-v1.json",
)

# Source and documentation that must survive whatever the ignore rules say.
MUST_BE_TRACKABLE = (
    ".gitignore",
    "LICENSE",
    "arena/imitation/native.py",
    "arena/imitation/contract.py",
    "adk/arena.py",
    "adk/docs/README.md",
    "console/package.json",
    "npc/loop.py",
    "tools/check_publication_hygiene.py",
    "README.md",
    "telemetry/README.md",
    # The two experimental trees that DO publish. Pinned 2026-08-22 so a future
    # rule broadened to `experimental/` cannot quietly take the live
    # native-transfer work or the drift-guarded worldgen source with it.
    "HytaleRL/hytale-plugin/src/main/java/com/hytalerlbridge/HytaleRLBridgePlugin.java",
    "HytaleRL/examples/jax_arsenal_world_benchmark.py",
    "experimental/jvm-agent/src/com/hytalerlbridge/policy/PolicyAgentPlugin.java",
    "experimental/worldgen-v2/jax_port/worlds/minigame_runtime.py",
    "experimental/worldgen-v2/__init__.py",
)


def is_ignored(path: str) -> bool:
    """True when git would refuse to track ``path``.

    ``git check-ignore`` exits 0 for ignored, 1 for not ignored. Key on the exit
    code: with ``-v`` it also prints negation rules, so a non-empty stdout does
    not mean the path is excluded.
    """

    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


#: Top-level packages that ship from this repository. An import of anything
#: else is a third-party or stdlib dependency and is not this check's business.
FIRST_PARTY = frozenset({
    "adk", "agents", "arena", "console", "hytalegym", "npc", "tools",
    "worldgen", "worlds",
})

#: `hytalegym` is published from a nested directory rather than the repo root.
IMPORT_ROOTS = ("", "HytaleRL/hytalegym/")


def _published_paths() -> set[str]:
    """Every path git would add right now, tracked or newly trackable."""

    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return {
        line[3:].strip().strip('"')
        for line in result.stdout.splitlines()
        if line[3:].strip()
    }


def _module_state(module: str, published: set[str]) -> bool | None:
    """True if published, False if excluded, None if not ours."""

    relative = "/".join(module.split("."))
    for root in IMPORT_ROOTS:
        candidate = f"{root}{relative}"
        if f"{candidate}.py" in published or f"{candidate}/__init__.py" in published:
            return True
    for root in IMPORT_ROOTS:
        candidate = REPO_ROOT / f"{root}{relative}"
        if candidate.with_suffix(".py").exists() or (candidate / "__init__.py").exists():
            return False
    return None


def unimportable() -> list[tuple[str, str]]:
    """Published modules importing first-party modules that are excluded.

    This is the failure that survives every other check: the hygiene lists
    below pass, the tree looks complete, and a fresh clone dies on `import`
    because a broad rule swallowed real source. `**/skills/` was written for
    agent skill folders and also excluded `arena/training/skills/`, which
    `arena/training/__init__.py` imports -- so `import arena.training` failed
    on a clone while every path assertion here still passed.
    """

    published = _published_paths()
    broken: list[tuple[str, str]] = []
    for path in sorted(p for p in published if p.endswith(".py")):
        try:
            tree = ast.parse((REPO_ROOT / path).read_text(
                encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if module.split(".")[0] not in FIRST_PARTY:
                    continue
                if _module_state(module, published) is False:
                    broken.append((path, module))
    return broken


def main() -> int:
    leaks = [p for p in MUST_BE_IGNORED if not is_ignored(p)]
    lost = [p for p in MUST_BE_TRACKABLE if is_ignored(p)]
    dangling = unimportable()

    for path, module in dangling:
        print(f"UNIMPORTABLE  {path}\n              imports {module!r}, "
              f"which no ignore rule lets a clone have")
    if dangling:
        print(f"\nFAILED: {len(dangling)} published import(s) of excluded source")
        print("Find the responsible rule with: git check-ignore -v <path>")
        return 1

    for path in leaks:
        print(f"LEAK  {path}\n      must never be published, but git would track it")
    for path in lost:
        print(f"LOST  {path}\n      is real source/docs, but an ignore rule excludes it")

    if leaks or lost:
        print(f"\nFAILED: {len(leaks)} leaking, {len(lost)} lost")
        print("Find the responsible rule with: git check-ignore -v <path>")
        return 1

    print(
        f"ok: {len(MUST_BE_IGNORED)} excluded paths stay excluded, "
        f"{len(MUST_BE_TRACKABLE)} source paths stay trackable"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
