"""Benchmark of the interpret stage (SYSTEM_ONE.md 8.13): text encoders against the stage's gate.

The sets are G3-NL's (``data.goal_language_sets`` with the key ``train_supervised`` uses for
``--goal-language``): the reader and the matcher are fitted on the training set's texts (training wordings,
training intent families), then measured on the three held-out splits. Per encoder:

    kind accuracy          held-out wordings, non-empty ``when`` texts: the reader (with the unit rule) and the
                           classifier alone (its argmax, the design as written); the training texts as a fit check
    argument exact-match   held-out wordings, numeric kinds: argument_for(true kind, read_number(when)) == the
                           generator's argument (encoder-free: the number reader alone)
    intent -> skill        zero-shot and fitted, every goal of held-out wordings, the goals of a held-out family
                           in held-out families, every goal of held-out sizes; against the state's catalogue
    end to end (M13)       decide() on the interpreted goals (reader + zero-shot or fitted matcher) against
                           goal_language.teacher on the true meanings, skill and replace agreement, over ``--rows-per-state`` belief rows
                           per state drawn as ``data.goal_language_rows`` draws them (same keys, same dataset,
                           the two tick columns times TICKS_SATURATION)
    encode time            seconds to encode 1,000 distinct goal texts (ENCODE_BATCH per call, fresh cache)

Gates (section 8.13): kind >= 0.95, argument >= 0.98, intent -> skill >= 0.90 on held-out wordings and
families; the end-to-end agreement is reported, not gated. ``hashed`` is the floor.

    PYTHONPATH=. JAX_PLATFORMS=cpu python -m experimental.neural_agent.jev.system_one.interpret_benchmark \\
        --encoder hashed,torch:sentence-transformers/all-MiniLM-L6-v2 --train 4096 --eval 1024 --seed 0

When every ``torch:`` model is in the local Hugging Face cache and ``HF_HUB_OFFLINE`` is unset, the run sets
``HF_HUB_OFFLINE=1`` (the cached revision, no network); a model not cached is downloaded.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import numpy as np

from experimental.neural_agent.contracts import TICKS_SATURATION, feature_index
from experimental.neural_agent.executive.goal_language import teacher
from experimental.neural_agent.jev.system_one.data import GoalLanguageSample, _numpy_rng, goal_language_sets
from experimental.neural_agent.jev.system_one.interpret import (
    KINDS,
    NUMERIC_UNITS,
    ConditionReader,
    IntentMatcher,
    _embed,
    argument_for,
    decide,
    interpret,
    read_number,
)
from experimental.neural_agent.jev.system_one.train_supervised import load_dataset, make_trunk
from experimental.neural_agent.jev.train_j1 import validate_dataset

DEFAULT_BELIEFS: Path = Path(__file__).resolve().parents[2] / "eval" / "out" / "traces" / "heldout" / "dataset.npz"
"""The recorded held-out belief rows G3-NL drew its held-out rows from (``--heldout`` of the failed gate)."""
KIND_BAR, ARGUMENT_BAR, INTENT_BAR = 0.95, 0.98, 0.90
TIMED_TEXTS: int = 1000
SPLITS: tuple[str, ...] = ("heldout_wordings", "heldout_families", "heldout_sizes")
_TICK_FEATURES: tuple[str, ...] = ("ticks_since_damage", "ticks_since_target_seen")


def gate_sets(seed: int, n_train: int, n_eval: int) -> tuple[Any, tuple[jax.Array, ...]]:
    """The G3-NL sets and the row keys of the three held-out splits, keyed as ``train_supervised --goal-language``."""

    sets_key, rows_key = jax.random.split(jax.random.fold_in(jax.random.key(seed), 2))
    _, *split_keys = jax.random.split(rows_key, 4)
    return goal_language_sets(sets_key, n_train, n_eval), tuple(split_keys)


def belief_rows(
    samples: Sequence[GoalLanguageSample], dataset: Mapping[str, np.ndarray], key: jax.Array, per_state: int
) -> list[list[tuple[list[float], list[bool], str]]]:
    """Per sample, ``per_state`` rows (features in physical units, known flags, current skill id), drawn as
    ``data.goal_language_rows`` draws them for the same key."""

    validate_dataset(dataset)
    features, known, current = (np.asarray(dataset[name]) for name in ("features", "feature_known", "current_skill"))
    ticks = [feature_index(name) for name in _TICK_FEATURES]
    rng = _numpy_rng(key)
    rows = []
    for sample in samples:
        drawn = rng.choice(len(current), size=per_state, replace=per_state > len(current))
        physical = features[drawn].astype(np.float32)
        physical[:, ticks] *= np.float32(TICKS_SATURATION)
        ids = tuple(sample.catalogue)
        rows.append([
            (values, flags, ids[skill])
            for values, flags, skill in zip(physical.tolist(), known[drawn].tolist(), current[drawn].tolist())
        ])
    return rows


def _goals(samples: Sequence[GoalLanguageSample]):
    return [(goal, meaning, sample) for sample in samples for goal, meaning in zip(sample.section.goals, sample.meanings)]


def _rate(hits: Sequence[bool]) -> float:
    return float(np.mean(hits)) if len(hits) else float("nan")


def _mark(value: float, bar: float) -> str:
    return "PASS" if value >= bar else "FAIL"


def number_reader_accuracy(samples: Sequence[GoalLanguageSample]) -> tuple[float, int]:
    """Argument exact-match of the number reader over the numeric goals of ``samples`` and their count."""

    hits = [
        argument_for(meaning.predicate.kind, read_number(goal.when)) == meaning.predicate.argument
        for goal, meaning, _ in _goals(samples)
        if meaning.predicate.kind in NUMERIC_UNITS
    ]
    return _rate(hits), len(hits)


def run_encoder(spec: str, sets: Any, rows: Mapping[str, list], timed: Sequence[str]) -> dict[str, Any]:
    """Fit and measure one encoder; print its block and return its metrics."""

    started = time.perf_counter()
    encoder = make_trunk(spec)
    clock = time.perf_counter()
    _embed(encoder, list(timed), {})
    seconds = time.perf_counter() - clock
    train = _goals(sets.train)
    reader = ConditionReader(encoder).fit([goal.when for goal, _, _ in train], [m.predicate.kind for _, m, _ in train])
    matcher = IntentMatcher(encoder).fit(
        [goal.intent for goal, _, _ in train], [sample.catalogue[m.skill_id] for _, m, sample in train]
    )
    metrics: dict[str, Any] = {"encoder": spec, "dim": int(encoder.dim), "seconds_per_1000": seconds * 1000 / len(timed)}

    def kinds(samples):
        worded = [(goal.when, meaning.predicate.kind) for goal, meaning, _ in _goals(samples) if goal.when.strip()]
        texts = [text for text, _ in worded]
        read = [item.kind for item in reader.read(texts)]
        alone = [KINDS[int(index)] for index in np.argmax(reader.probabilities(texts), axis=1)]
        truth = [kind for _, kind in worded]
        return _rate([a == b for a, b in zip(read, truth)]), _rate([a == b for a, b in zip(alone, truth)]), len(truth)

    metrics["kind"], metrics["kind_alone"], kind_count = kinds(sets.heldout_wordings)
    metrics["kind_train"], _, train_count = kinds(sets.train)
    held_ids = set(sets.heldout_family_ids)
    for split in SPLITS:
        goals = [
            item for item in _goals(getattr(sets, split))
            if split != "heldout_families" or item[1].intent_family in held_ids
        ]
        for zero_shot in (True, False):
            hits = [
                matcher.match([goal.intent], sample.catalogue, zero_shot=zero_shot)[0].skill_id == meaning.skill_id
                for goal, meaning, sample in goals
            ]
            metrics[f"intent_{split}_{'zero_shot' if zero_shot else 'fitted'}"] = _rate(hits)
        metrics[f"intent_{split}_count"] = len(goals)
        for zero_shot in (True, False):
            name = f"m13_{split}_{'zero_shot' if zero_shot else 'fitted'}"
            skill_hits, replace_hits = [], []
            for sample, sample_rows in zip(getattr(sets, split), rows[split]):
                readings = [
                    interpret(goal, sample.catalogue, reader, matcher, zero_shot=zero_shot) for goal in sample.section.goals
                ]
                for values, flags, current in sample_rows:
                    expected = teacher(sample.section, sample.meanings, values, flags, current)
                    got = decide(sample.section, readings, values, flags, current)
                    skill_hits.append(got[0] == expected[0])
                    replace_hits.append(got[1] == expected[1])
            metrics[name], metrics[f"{name}_replace"], metrics[f"m13_{split}_rows"] = (
                _rate(skill_hits), _rate(replace_hits), len(skill_hits)
            )

    floor = " (the floor)" if spec == "hashed" else ""
    print(f"encoder {spec}{floor}: dim {encoder.dim}, encode {metrics['seconds_per_1000']:.3f} s per 1,000 texts")
    print(
        f"  kind accuracy, held-out wordings ({kind_count} non-empty when texts): reader {metrics['kind']:.4f} "
        f"[>= {KIND_BAR}: {_mark(metrics['kind'], KIND_BAR)}], classifier alone {metrics['kind_alone']:.4f}; "
        f"training texts ({train_count}) {metrics['kind_train']:.4f}"
    )
    labels = {
        "heldout_wordings": "held-out wordings (all goals)",
        "heldout_families": f"held-out families ({', '.join(sets.heldout_family_ids)} goals)",
        "heldout_sizes": "held-out sizes (all goals)",
    }
    for split in SPLITS:
        zero, fitted = metrics[f"intent_{split}_zero_shot"], metrics[f"intent_{split}_fitted"]
        bar = f" [>= {INTENT_BAR}: zero-shot {_mark(zero, INTENT_BAR)}, fitted {_mark(fitted, INTENT_BAR)}]" if split != "heldout_sizes" else ""
        print(
            f"  intent -> skill, {labels[split]} ({metrics[f'intent_{split}_count']}): zero-shot {zero:.4f}, "
            f"fitted {fitted:.4f}{bar}"
        )
    for split in SPLITS:
        zero, fitted = f"m13_{split}_zero_shot", f"m13_{split}_fitted"
        print(
            f"  end-to-end M13 agreement, {split.replace('heldout_', 'held-out ')} ({metrics[f'm13_{split}_rows']} rows), "
            f"skill (replace): zero-shot matcher {metrics[zero]:.4f} ({metrics[zero + '_replace']:.4f}), "
            f"fitted matcher {metrics[fitted]:.4f} ({metrics[fitted + '_replace']:.4f})"
        )
    print(f"  [{time.perf_counter() - started:.1f} s]", flush=True)
    return metrics


def _offline_if_cached(specs: Sequence[str]) -> None:
    """Set HF_HUB_OFFLINE=1 when it is unset and every ``torch:`` model has a snapshot in the local cache."""

    if "HF_HUB_OFFLINE" in os.environ:
        return
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", home / "hub"))
    names = [spec[len("torch:") :] for spec in specs if spec.startswith("torch:")]
    if names and all((hub / f"models--{name.replace('/', '--')}" / "snapshots").is_dir() for name in names):
        os.environ["HF_HUB_OFFLINE"] = "1"


def main(argv: Sequence[str] | None = None) -> int:
    """Print one block per encoder and a summary table; return 0."""

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.system_one.interpret_benchmark",
        description="Interpret-stage benchmark (SYSTEM_ONE.md 8.13) on the G3-NL sets.",
    )
    parser.add_argument("--encoder", default="hashed", help="comma-separated: hashed, torch:<hf-model-id>")
    parser.add_argument("--train", type=int, default=4096, help="training states (default 4096)")
    parser.add_argument("--eval", type=int, default=1024, help="states per held-out split (default 1024)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beliefs", type=Path, default=DEFAULT_BELIEFS, help="recorded dataset .npz of the belief rows")
    parser.add_argument("--rows-per-state", type=int, default=32, help="belief rows per held-out state (default 32)")
    args = parser.parse_args(argv)
    specs = [spec.strip() for spec in args.encoder.split(",") if spec.strip()]
    if not specs:
        parser.error("--encoder needs at least one encoder")
    for spec in specs:
        if spec != "hashed" and not (spec.startswith("torch:") and len(spec) > len("torch:")):
            parser.error(f"--encoder entries must be hashed or torch:<model>, got {spec!r}")
    if args.train < 1 or args.eval < 1 or args.rows_per_state < 1:
        parser.error("--train, --eval and --rows-per-state must be positive")
    _offline_if_cached(specs)
    sets, split_keys = gate_sets(args.seed, args.train, args.eval)
    beliefs = load_dataset(args.beliefs)
    rows = {split: belief_rows(getattr(sets, split), beliefs, key, args.rows_per_state) for split, key in zip(SPLITS, split_keys)}
    every = {text for split in sets[:4] for goal, _, _ in _goals(split) for text in (goal.when, goal.intent) if text.strip()}
    timed = sorted(every)[:TIMED_TEXTS]
    argument, count = number_reader_accuracy(sets.heldout_wordings)
    print(
        f"sets: seed {args.seed}, {len(sets.train)} train states, {args.eval} states per held-out split, held-out "
        f"families {', '.join(sets.heldout_family_ids)}; {args.rows_per_state} belief rows per state from {args.beliefs}"
    )
    print(
        f"argument exact-match, held-out wordings ({count} numeric when texts), number reader: {argument:.4f} "
        f"[>= {ARGUMENT_BAR}: {_mark(argument, ARGUMENT_BAR)}]",
        flush=True,
    )
    results = [run_encoder(spec, sets, rows, timed) for spec in specs]
    print(
        "\n| encoder | kind (wordings) | argument | intent wordings zs / fit | intent families zs / fit | "
        "intent sizes zs / fit | M13 wordings zs / fit | M13 families zs / fit | M13 sizes zs / fit | s per 1,000 texts |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for item in results:
        intents = " | ".join(
            f"{item[f'intent_{split}_zero_shot']:.4f} / {item[f'intent_{split}_fitted']:.4f}" for split in SPLITS
        )
        agreement = " | ".join(
            f"{item[f'm13_{split}_zero_shot']:.4f} / {item[f'm13_{split}_fitted']:.4f}" for split in SPLITS
        )
        print(
            f"| {item['encoder']} | {item['kind']:.4f} | {argument:.4f} | {intents} | {agreement} | "
            f"{item['seconds_per_1000']:.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
