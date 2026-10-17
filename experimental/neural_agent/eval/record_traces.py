"""Record J0 decision traces on an arena and merge them into a J1 training set.

One ``.npz`` trace per seed is written by :func:`traces.record_j0_traces` with
the ``j0`` arm of ``eval/arms.py`` on the named arena, then the decision rows of
every trace are concatenated into ``dataset.npz``, which
``jev/train_j1.py --data`` accepts. A decision row is a row with
``manager_due`` set: on the other rows the recorder copies ``current_skill``
into ``target_skill`` (``eval/traces.py``), so training on them would teach
"keep the goal" with no decision behind it. The scalar ``num_skills`` that the
trainer requires is the catalogue size of the skill registry.

Run (CPU, from the repository root)::

    python -m experimental.neural_agent.eval.record_traces \
        --arena combat_open_flat --batch 8 --ticks 256 --seeds 0 1 2 3 \
        --out experimental/neural_agent/eval/out/traces/train

``summary.json`` next to the dataset records the row counts and the label
balance, so the ledger can quote them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experimental.neural_agent.jev.train_j1 import FIELDS, validate_dataset

#: The arm whose decisions are recorded. J0 is deterministic in (belief, executive).
RECORDED_ARM: str = "j0"


def record_seeds(
    out_dir: str | Path,
    *,
    arena: str,
    batch: int,
    ticks: int,
    seeds: Sequence[int],
) -> tuple[list[Path], tuple[str, ...]]:
    """Record one J0 trace per seed into ``out_dir`` and return the paths and the catalogue.

    Each seed builds its own arena (``load_arena``) and composes the ``j0`` arm
    on it (``run_composition._compose``), so the recorded run is the same run the
    composition CLI scores for that seed.

    Raises:
        ValueError: From the composition helpers for a bad batch, tick count or seed.
    """

    from experimental.neural_agent.eval.arms import get_arm
    from experimental.neural_agent.eval.run_composition import _compose, load_arena, load_context
    from experimental.neural_agent.eval.traces import record_j0_traces

    target = Path(out_dir)
    ctx, _task = load_context(batch, ticks)
    arm = get_arm(RECORDED_ARM)
    paths: list[Path] = []
    for seed in seeds:
        built = load_arena(arena, batch, seed, ticks).built  # BuiltArena.built is the environment
        composed = _compose(arm, ctx, built=built, ticks=ticks, episodes=1, seed=seed)
        paths.append(
            record_j0_traces(
                target / f"seed_{seed}.npz",
                built=built,
                core=composed.core,
                initial_state=composed.initial,
                executive=ctx.executive,
                skill_ids=ctx.skill_ids,
                ticks=ticks,
                seed=seed,
            )
        )
    return paths, tuple(ctx.skill_ids)


def merge_decisions(paths: Sequence[str | Path], *, num_skills: int) -> dict[str, np.ndarray]:
    """Concatenate the ``manager_due`` rows of the traces into one validated dataset.

    Returns:
        The fields of ``train_j1.FIELDS`` plus the scalar ``num_skills``.

    Raises:
        ValueError: If no row is due, or the merged arrays fail ``validate_dataset``.
    """

    parts: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path) as archive:
            due = np.asarray(archive["manager_due"], dtype=np.bool_)
            parts.append({name: np.asarray(archive[name])[due] for name in FIELDS})
    data = {name: np.concatenate([part[name] for part in parts], axis=0) for name in FIELDS}
    if data["current_skill"].shape[0] == 0:
        raise ValueError("no decision row is due in the recorded traces")
    data["num_skills"] = np.asarray(num_skills, dtype=np.int32)
    validate_dataset(data)
    return data


def summarize(paths: Sequence[Path], dataset: Mapping[str, Any], skill_ids: Sequence[str]) -> dict[str, Any]:
    """Return the row counts and label balance of a recording, JSON-ready."""

    rows = 0
    due = 0
    for path in paths:
        with np.load(path) as archive:
            flags = np.asarray(archive["manager_due"], dtype=np.bool_)
            rows += int(flags.shape[0])
            due += int(flags.sum())
    target = np.asarray(dataset["target_skill"])
    counts = np.bincount(target, minlength=len(skill_ids))
    return {
        "traces": [str(path) for path in paths],
        "skill_ids": list(skill_ids),
        "rows_total": rows,
        "rows_due": due,
        "target_skill_counts": {skill_ids[i]: int(counts[i]) for i in range(len(skill_ids))},
        "replace_rate": float(np.mean(np.asarray(dataset["target_replace"], dtype=np.float64))),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Record the traces, write ``dataset.npz`` and ``summary.json``, print the summary."""

    from experimental.neural_agent.eval.run_composition import ARENA_CHOICES, DEFAULT_ARENA, DEFAULT_TICKS

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.eval.record_traces",
        description="Record J0 decision traces per seed and merge their decision rows into a J1 dataset.",
    )
    parser.add_argument("--arena", choices=ARENA_CHOICES, default=DEFAULT_ARENA)
    parser.add_argument("--batch", type=int, default=8, help="lanes per run (default 8)")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help=f"ticks per run (default {DEFAULT_TICKS})")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="one run per seed")
    parser.add_argument("--out", type=Path, required=True, help="directory for seed_<s>.npz, dataset.npz, summary.json")
    args = parser.parse_args(argv)

    paths, skill_ids = record_seeds(args.out, arena=args.arena, batch=args.batch, ticks=args.ticks, seeds=args.seeds)
    dataset = merge_decisions(paths, num_skills=len(skill_ids))
    np.savez(args.out / "dataset.npz", **dataset)
    summary = summarize(paths, dataset, skill_ids)
    summary.update({"arena": args.arena, "batch": args.batch, "ticks": args.ticks, "seeds": list(args.seeds)})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
