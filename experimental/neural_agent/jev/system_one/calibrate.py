"""Stage B of J-S1: fit the confidence head with the other heads frozen (SYSTEM_ONE.md 8.4).

Only (w_c, b_c) train; every other head and the trunk are fixed, so the skill choice
cannot move and confidence cannot improve by changing answers (the non-hackable
form, [RLCR-HACK]). Loss and report, over N rows with k_i = argmax s~_i::

    L_B  = mean_i (c_i - 1[k_i = y_i])^2                          Brier score, a proper scoring rule [RLCR]
    ECE  = sum_b (n_b / N) |acc_b - conf_b|,  10 equal-width bins of c on [0, 1]   (bin b holds c in [b/10, (b+1)/10), c = 1 in bin 9)

Full-batch Adam (``train_j1.py`` optimizer settings) for ``--updates`` steps. The
report gives accuracy (mean 1[k = y]), mean confidence, Brier and ECE on the
``--eval-data`` rows (default: the ``--data`` rows), before and after the fit.

Command (CPU)::

    python -m experimental.neural_agent.jev.system_one.calibrate --heads /tmp/js1/heads.npz \\
        --data experimental/neural_agent/eval/out/traces/train/dataset.npz \\
        --eval-data experimental/neural_agent/eval/out/traces/heldout/dataset.npz --out /tmp/js1/heads_b.npz
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experimental.neural_agent.jev.system_one.data import catalogue_embeddings, rows_from_dataset
from experimental.neural_agent.jev.system_one.heads import HeadsParams, apply_heads, heads_dims
from experimental.neural_agent.jev.system_one.train_supervised import (
    DEFAULT_SKILL_IDS,
    EncodedRows,
    encode_rows,
    load_dataset,
    load_executive,
    load_heads,
    make_trunk,
    parse_skills,
    save_heads,
)
from experimental.neural_agent.jev.train_j1 import ADAM_EPSILON, MAX_GRADIENT_NORM

ECE_BINS: int = 10
DEFAULT_UPDATES: int = 500
DEFAULT_LR: float = 1.0e-2


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    """Return ECE = sum_b (n_b / N) |acc_b - conf_b| over ``bins`` equal-width bins of [0, 1]."""

    conf = np.asarray(confidence, dtype=np.float64)
    hits = np.asarray(correct, dtype=np.float64)
    index = np.minimum((conf * bins).astype(np.int64), bins - 1)
    total = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            total += members.mean() * abs(hits[members].mean() - conf[members].mean())
    return float(total)


def _outputs(params: HeadsParams, encoded: EncodedRows, skill_embeddings: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return (confidence float[N], correct bool[N]) of ``params`` on ``encoded``."""

    out = jax.jit(apply_heads)(
        params, encoded.h, encoded.u, encoded.current_onehot, encoded.allowed_mask, skill_embeddings
    )
    correct = np.asarray(jnp.argmax(out.skill_logits, axis=-1)) == encoded.y_skill
    return np.asarray(out.confidence), correct


def calibration_report(params: HeadsParams, encoded: EncodedRows, skill_embeddings: Any = None) -> dict[str, float]:
    """Return rows, accuracy, mean_confidence, brier and ece of ``params`` on ``encoded``.

    ``skill_embeddings`` (float32[K, D]) is required by a matching head.
    """

    confidence, correct = _outputs(params, encoded, skill_embeddings)
    return {
        "rows": int(correct.shape[0]),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "brier": float(np.mean((confidence - correct) ** 2)),
        "ece": expected_calibration_error(confidence, correct),
    }


def fit_confidence(
    params: HeadsParams,
    encoded: EncodedRows,
    skill_embeddings: Any = None,
    *,
    updates: int = DEFAULT_UPDATES,
    lr: float = DEFAULT_LR,
) -> HeadsParams:
    """Return ``params`` with (w_c, b_c) fitted by full-batch Adam on L_B; every other leaf unchanged.

    ``skill_embeddings`` (float32[K, D]) is required by a matching head.
    """

    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError(f"updates must be a positive integer, got {updates!r}")
    _, correct = _outputs(params, encoded, skill_embeddings)
    target = jnp.asarray(correct, dtype=jnp.float32)
    inputs = (encoded.h, encoded.u, encoded.current_onehot, encoded.allowed_mask, skill_embeddings)

    def brier(head: tuple[jax.Array, jax.Array], frozen: HeadsParams, data: tuple, hits: jax.Array) -> jax.Array:
        out = apply_heads(frozen._replace(w_c=head[0], b_c=head[1]), *data)
        return jnp.mean((out.confidence - hits) ** 2)

    optimizer = optax.chain(optax.clip_by_global_norm(MAX_GRADIENT_NORM), optax.adam(float(lr), eps=ADAM_EPSILON))
    head = (params.w_c, params.b_c)
    state = optimizer.init(head)

    @jax.jit
    def step(current, opt_state, frozen, data, hits):
        gradients = jax.grad(brier)(current, frozen, data, hits)
        step_updates, opt_state = optimizer.update(gradients, opt_state, current)
        return optax.apply_updates(current, step_updates), opt_state

    for _ in range(updates):
        head, state = step(head, state, params, inputs, target)
    return params._replace(w_c=head[0], b_c=head[1])


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the confidence head on --data, write the heads, print the JSON report on --eval-data."""

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.system_one.calibrate",
        description="Stage B of J-S1 (CPU): Brier fit of the confidence head with every other head frozen.",
    )
    parser.add_argument("--heads", type=Path, required=True, help="stage-A heads .npz")
    parser.add_argument("--data", type=Path, required=True, help="dataset .npz the confidence head is fitted on")
    parser.add_argument("--eval-data", type=Path, default=None, help="dataset .npz reported on (default --data)")
    parser.add_argument("--executive", type=Path, default=None, help="executive.json; default the CompositionTask E_t")
    parser.add_argument("--skills", type=parse_skills, default=DEFAULT_SKILL_IDS, help="catalogue, default idle,pursuit,evader")
    parser.add_argument("--trunk", default="hashed", help="the trunk the heads were trained with (default hashed)")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES, help=f"Adam updates (default {DEFAULT_UPDATES})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Adam learning rate (default {DEFAULT_LR})")
    parser.add_argument("--out", type=Path, required=True, help="output heads .npz with the fitted confidence head")
    args = parser.parse_args(argv)
    try:
        trunk = make_trunk(args.trunk)
    except (ImportError, ValueError) as error:
        parser.error(str(error))

    executive = load_executive(args.executive, args.skills)
    params = load_heads(args.heads)
    embeddings = None if heads_dims(params).skill_head == "fixed" else catalogue_embeddings(trunk, args.skills)
    fit = encode_rows(rows_from_dataset(load_dataset(args.data), executive, args.skills), trunk)
    scored = fit if args.eval_data is None else encode_rows(
        rows_from_dataset(load_dataset(args.eval_data), executive, args.skills), trunk
    )
    before = calibration_report(params, scored, embeddings)
    params = fit_confidence(params, fit, embeddings, updates=args.updates, lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_heads(args.out, params)
    report = {
        "fit_rows": int(fit.h.shape[0]),
        "eval": str(args.eval_data or args.data),
        "before": before,
        "after": calibration_report(params, scored, embeddings),
        "heads": str(args.out),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
