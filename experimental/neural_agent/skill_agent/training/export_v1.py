"""B2: V1's trained policy as skill bodies (``distill_v3.py``, TRAINING_PLAN.md 8.8).

A V1 run's generalist head serves two verbs, walk_to and look_at, so the run becomes one trunk and two modules. Each
module is a :class:`~model.skill_module.SkillModule` on the run's trunk, with its skill's verb and scope and the verbs
it reads (both), and the run's generalist parameters unchanged. The combiner runs a module through
``runtime/state_tick.ModuleBody``.

CLI::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu nice -n 10 \\
        .venv/bin/python -m experimental.neural_agent.skill_agent.training.export_v1 \\
        --params RUN.npz --arm memory|control --out DIR

``DIR`` gets ``trunk.npz``, ``walk_to.npz`` and ``look_at.npz`` (each with its ``.json`` sidecar) and ``manifest.json``:
the run's report (``RUN.json``, beside its parameters), its bars (the report's ``evaluation``), the arm, today's
date and the git commit. A folder that already holds files is refused, never overwritten.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
from pathlib import Path
from typing import Sequence

import jax

from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.model.encoder import token_layout
from experimental.neural_agent.skill_agent.model.skill_module import SkillModule, make_trunk, save_module, save_trunk
from experimental.neural_agent.skill_agent.neural_skills.look_at import skill as look_at_skill
from experimental.neural_agent.skill_agent.neural_skills.walk_to import skill as walk_to_skill
from experimental.neural_agent.skill_agent.training import blocks, distill, distill_v3


def as_trained(config: distill_v3.DistillV3Config) -> distill_v3.DistillV3Config:
    """V1's runs (2026-10-24) predate the ``sound`` token type (S12-a, ``../docs/VIRTUAL_HUMAN.md`` 28.8), so their
    trunks were trained without it. The encoder ignores a type its config lacks, so they run on the new tokens."""

    types = tuple(entry for entry in config.encoder.token_types if entry[0] != "sound")
    return dataclasses.replace(config, encoder=dataclasses.replace(config.encoder, token_types=types))


ARMS = {"memory": as_trained(distill_v3.DistillV3Config()),
        "control": as_trained(distill_v3.memoryless(distill_v3.DistillV3Config()))}
"""The two arms, as the run's own configuration: ``memory`` is the default run, ``control`` the memory-less one."""


def load_run(params_path: Path, arm: str) -> dict:
    """The run's parameters (``{"trunk", "generalist"}``), loaded into the arm's configuration."""

    config = ARMS[arm]
    template = distill.init_params(jax.random.key(0), distill.DistillConfig(encoder=config.encoder,
                                                                           generalist=config.generalist))
    return blocks.load_params(params_path, template)


def export_modules(params: dict, arm: str, out: Path) -> None:
    """Write the trunk and one module per skill (``walk_to``, ``look_at``) into ``out``."""

    config = ARMS[arm]
    out.mkdir(parents=True, exist_ok=True)
    trunk = make_trunk(params["trunk"], config.encoder)
    layout = token_layout(v3.TOKEN_SCHEMA_V3)
    trained = {name: layout[name] for name, _ in config.encoder.token_types}  # the trunk's own types
    save_trunk(out / "trunk.npz", trunk, trained)  # by feature name: migratable later
    skills = (walk_to_skill.spec(), look_at_skill.spec())
    known = tuple(sorted(spec.verb for spec in skills))
    for spec in skills:
        module = SkillModule(skill=spec.name, scope=spec.scope, verb=spec.verb, known=known,
                             config=distill.DEFAULT_GENERALIST, params=params["generalist"], trunk_sha256=trunk.sha256)
        save_module(out / f"{spec.name}.npz", module)


def write_manifest(params_path: Path, arm: str, out: Path) -> None:
    """``manifest.json``: the run's report, its bars, the arm, today's date, and the git commit of this export (and
    whether that tree had uncommitted changes) beside the commit the run was trained at."""

    report_path = params_path.with_suffix(".json")
    report = json.loads(report_path.read_text())
    provenance = blocks.provenance()
    manifest = {"run": str(report_path), "arm": arm, "bars": report["evaluation"],
                "date": datetime.date.today().isoformat(), "git_commit": provenance["base_commit"],
                "dirty": provenance["dirty"], "run_commit": report["provenance"]["base_commit"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export V1's trained run as a trunk and one module per skill.")
    parser.add_argument("--params", type=Path, required=True, help="the run's parameters (distill_v3 --params-out)")
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--out", type=Path, required=True, help="the folder to write; it must be empty or absent")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty; choose a new folder")
    export_modules(load_run(args.params, args.arm), args.arm, args.out)
    write_manifest(args.params, args.arm, args.out)
    print(f"exported {args.params} ({args.arm}) to {args.out}", flush=True)
    return 0


__all__ = ["ARMS", "export_modules", "load_run", "main", "write_manifest"]


if __name__ == "__main__":
    raise SystemExit(main())
