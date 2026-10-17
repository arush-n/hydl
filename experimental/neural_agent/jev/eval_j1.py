"""Offline evaluation of J1 parameters on a decision dataset (ADR-001 D2, H16).

The dataset is the ``.npz`` that ``eval/record_traces.py`` writes, or any mapping
that ``train_j1.validate_dataset`` accepts. Every row is scored at once, so the
accuracies are over the whole file and not over a training batch::

    skill_accuracy   = mean_n [ argmax_k z_nk == target_skill_n ]   (z masked to the legal skills)
    replace_accuracy = mean_n [ (r_n > 0) == target_replace_n ]
    confusion[i, j]  = #{ n : target_skill_n = i and argmax_k z_nk = j }

Run::

    python -m experimental.neural_agent.jev.eval_j1 --params j1.npz --data heldout/dataset.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from experimental.neural_agent.jev.network import J1Params, apply_j1, j1_allowed_skills
from experimental.neural_agent.jev.train_j1 import (
    _MASKED_LOGIT,
    _records,
    build_inputs,
    load_params,
    validate_dataset,
)


def evaluate(params: J1Params, dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Return the whole-file metrics of ``params`` on ``dataset``, JSON-ready.

    Raises:
        ValueError: If the dataset is invalid or its ``num_skills`` differs from the parameters.
    """

    validate_dataset(dataset)
    num_skills = int(np.asarray(dataset["num_skills"]))
    batch = {name: np.asarray(dataset[name]) for name in dataset if name != "num_skills"}
    _belief, executive, _goal = _records(batch)
    x = build_inputs(batch, num_skills=num_skills)
    logits, replace_logit, _params_head = apply_j1(params, x)
    if logits.shape[-1] != num_skills:
        raise ValueError(f"the parameters score {logits.shape[-1]} skills; the dataset has {num_skills}")
    allowed = j1_allowed_skills(executive, num_skills=num_skills)
    masked = np.where(np.asarray(allowed), np.asarray(logits), _MASKED_LOGIT)
    predicted = masked.argmax(axis=-1)
    target = np.asarray(dataset["target_skill"])
    replace_target = np.asarray(dataset["target_replace"], dtype=np.bool_)
    replace_predicted = np.asarray(replace_logit) > 0.0
    confusion = np.zeros((num_skills, num_skills), dtype=np.int64)
    np.add.at(confusion, (target, predicted), 1)
    support = confusion.sum(axis=1)
    recall = [float(confusion[i, i] / support[i]) if support[i] else None for i in range(num_skills)]
    return {
        "rows": int(target.shape[0]),
        "skill_accuracy": float(np.mean(predicted == target)),
        "replace_accuracy": float(np.mean(replace_predicted == replace_target)),
        "confusion_rows_target_cols_predicted": confusion.tolist(),
        "per_skill_recall": recall,
        "target_counts": support.tolist(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate ``--params`` on ``--data`` and print the metrics as JSON."""

    parser = argparse.ArgumentParser(prog="python -m experimental.neural_agent.jev.eval_j1")
    parser.add_argument("--params", type=Path, required=True, help="J1 parameters .npz (train_j1 --out)")
    parser.add_argument("--data", type=Path, required=True, help="decision dataset .npz with num_skills")
    parser.add_argument("--out", type=Path, default=None, help="optional path for the metrics JSON")
    args = parser.parse_args(argv)
    with np.load(args.data) as archive:
        dataset = {name: archive[name] for name in archive.files}
    with jax.default_device(jax.devices("cpu")[0]):
        metrics = evaluate(load_params(args.params), dataset)
    text = json.dumps(metrics, indent=2)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
