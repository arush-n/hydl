"""Fine-tuning pilot of the interpret stage (docs/SYSTEM_ONE.md 8.13 option (b); section 9, J-S2).

Frozen encoders failed the interpret gate (8.13; ledger "interpret stage with frozen encoders"): predicate kind
0.71 at best against 0.95, with polarity errors, and intent -> skill zero-shot 0.85 on unseen families against
0.90. This pilot fine-tunes one small encoder end to end on the goal-language generator and measures it on the
benchmark's held-out sets, so its rows sit beside the frozen ones; it decides whether J-S2's interpreter is worth
building. Nothing here is wired into the decision path.

Data. ``interpret_benchmark.gate_sets(seed, n_train, n_eval)``, G3-NL's sets and row keys. Training uses the
training set only (training wordings and training intent families; real and synthetic catalogues of 3..5
skills), one example per training goal::

    condition   (when, kind)              when non-empty; an empty when is always by rule and never trained
    intent      (intent, d+, catalogue)   d+ = catalogue[skill_id], the canonical description of its family

Model. f = a pretrained ``transformers`` encoder, masked mean pool over its last hidden state (as
``trunk.TorchTransformerTrunk``), fine-tuned end to end with two heads trained jointly; per step a batch of
BATCH_GOALS shuffled training goals::

    kind head   p(. | when) = softmax(W f(when) + b) over KINDS (nine)              W: 9 x D, b: 9
    L_kind      = mean over the batch's non-empty whens of -log p(kind | when)
    matching    u(t) = f(t) / ||f(t)||;   s_ij = u(intent_i) . u(desc_j) / TAU
                desc_j: the distinct descriptions of the batch's catalogues;   j+(i): the d+ of goal i
    L_match     = mean_i [ logsumexp_j s_ij - s_i,j+(i) ]                            (InfoNCE)
    L           = L_kind + L_match

AdamW (lr ENCODER_LR on f, HEAD_LR on W and b, weight decay WEIGHT_DECAY), the rate warmed up linearly over
WARMUP_FRACTION of the steps then decayed linearly to zero, gradient norm clipped at CLIP_NORM, goals reshuffled
every epoch. No weight depends on the catalogue: a skill is matched by similarity, so a description never seen in
training is read the same way. The constants were set before the first run; none was tuned on a held-out set.

Reading, the interpret stage of ``interpret.py`` with the fine-tuned pieces::

    kind        FineTunedConditionReader: ConditionReader.read over the kind head's p (the unit rule, the
                argument from the exact number reader, "" -> always)
    skill       IntentMatcher(model) zero-shot: argmax_j u(intent) . u(desc_j), the nearest description
    decision    interpret.interpret + interpret.decide (M13) against goal_language.teacher, on the benchmark's
                belief rows (interpret_benchmark.belief_rows, same keys)

Every metric is the benchmark's (same goals, same filters, zero-shot matching), measured for the frozen encoder
with the benchmark's fitted ConditionReader ("frozen", the before) and for the fine-tuned model ("fine-tuned"),
plus the per-kind accuracy and the polarity confusions on held-out wordings. Bars (section 9.6): kind >= 0.95
(I1), intent -> skill >= 0.90 on each split (I1), end-to-end M13 >= 0.90 on each split (I2).

Family draws. ``gate_sets(seed)`` holds out two intent families. ``--family-draws 2`` (the default) also trains
and measures a fresh model on the sets of the first later seed whose held-out pair is disjoint from the first
draw's (seed 0 holds out flank and follow_ally, seed 3 reach_point and patrol); every metric is per draw.

Weights go to ``--out`` (DEFAULT_OUT, outside the repository; a path inside it is refused): per draw
``<out>/<model>-seed<s>-<device>/`` holds the encoder and tokenizer (``save_pretrained``), ``kind_head.pt`` and
``metrics.json``; :meth:`PilotModel.load` reads them back. When the model is in the local Hugging Face cache and
``HF_HUB_OFFLINE`` is unset, the run sets ``HF_HUB_OFFLINE=1``::

    PYTHONPATH=. JAX_PLATFORMS=cpu python -m experimental.neural_agent.jev.system_one.finetune_pilot \\
        --encoder torch:sentence-transformers/all-MiniLM-L6-v2 --epochs 3 --seed 0 --device auto
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experimental.neural_agent.executive.goal_language import ALWAYS, teacher
from experimental.neural_agent.jev.system_one.data import GoalLanguageSample
from experimental.neural_agent.jev.system_one.interpret import (
    KINDS,
    ConditionReader,
    IntentMatcher,
    _embed,
    _softmax,
    decide,
    interpret,
)
from experimental.neural_agent.jev.system_one.interpret_benchmark import (
    DEFAULT_BELIEFS,
    INTENT_BAR,
    KIND_BAR,
    SPLITS,
    TIMED_TEXTS,
    _goals,
    _mark,
    _offline_if_cached,
    _rate,
    belief_rows,
    gate_sets,
)
from experimental.neural_agent.jev.system_one.train_supervised import load_dataset, make_trunk

DEFAULT_ENCODER: str = "torch:sentence-transformers/all-MiniLM-L6-v2"
BATCH_GOALS: int = 32
"""Training goals per step; each gives its non-empty ``when`` to L_kind and its ``intent`` to L_match."""
ENCODER_LR: float = 2e-5
"""AdamW learning rate of the pretrained encoder."""
HEAD_LR: float = 1e-3
"""AdamW learning rate of the kind head, which starts at random."""
WEIGHT_DECAY: float = 0.01
WARMUP_FRACTION: float = 0.1
CLIP_NORM: float = 1.0
TAU: float = 0.05
"""InfoNCE temperature on cosine similarities."""
MAX_TOKENS: int = 128
"""Truncation length; every goal text and description is far shorter."""
M13_BAR: float = 0.90
"""End-to-end agreement with M13 on each split, gate I2 of section 9.6."""
POLARITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("target_visible", "target_not_visible"),
    ("distance_below", "distance_above"),
)
DEVICES: tuple[str, ...] = ("auto", "mps", "cpu")
DEFAULT_OUT: Path = Path.home() / ".claude" / "jobs" / "31947bd7" / "tmp" / "finetune_pilot"
"""Where the weights go: outside the repository."""
REPO_ROOT: Path = Path(__file__).resolve().parents[4]
SEED_SEARCH: int = 100
"""How many later seeds the second family draw may try."""


@dataclass(frozen=True)
class GoalExample:
    """One training goal: its condition text and kind index, its intent text, its skill's description and the
    descriptions of its catalogue."""

    when: str
    kind: int
    intent: str
    description: str
    catalogue: tuple[str, ...]


def training_examples(samples: Sequence[GoalLanguageSample]) -> tuple[GoalExample, ...]:
    """One GoalExample per goal of ``samples`` (module docstring, Data), in goal order."""

    return tuple(
        GoalExample(
            goal.when,
            KINDS.index(meaning.predicate.kind),
            goal.intent,
            sample.catalogue[meaning.skill_id],
            tuple(sample.catalogue.values()),
        )
        for goal, meaning, sample in _goals(samples)
    )


def batch_inputs(
    examples: Sequence[GoalExample],
) -> tuple[list[str], list[int], list[str], list[str], list[int]]:
    """(whens, kinds, intents, descriptions, positives) of one batch: the non-empty whens with their kinds, every
    intent, the distinct descriptions of the batch's catalogues (sorted), and each intent's own description's index."""

    worded = [example for example in examples if example.when.strip()]
    descriptions = sorted({text for example in examples for text in example.catalogue})
    return (
        [example.when for example in worded],
        [example.kind for example in worded],
        [example.intent for example in examples],
        descriptions,
        [descriptions.index(example.description) for example in examples],
    )


def pilot_losses(
    conditions: torch.Tensor,
    kinds: torch.Tensor,
    intents: torch.Tensor,
    descriptions: torch.Tensor,
    positives: torch.Tensor,
    head: torch.nn.Module,
    tau: float = TAU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(L_kind, L_match) of the module docstring on pooled embeddings; L_kind is 0 when there is no condition."""

    functional = torch.nn.functional
    kind_loss = functional.cross_entropy(head(conditions), kinds) if len(kinds) else conditions.new_zeros(())
    similarity = functional.normalize(intents, dim=-1) @ functional.normalize(descriptions, dim=-1).T / tau
    return kind_loss, functional.cross_entropy(similarity, positives)


def resolve_device(name: str) -> str:
    """Return "mps" or "cpu": ``auto`` takes the Mac GPU when ``torch.backends.mps`` is available, else the CPU.

    Raises:
        ValueError: If ``name`` is not one of DEVICES, or ``mps`` is asked for and is not available.
    """

    if name not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {name!r}")
    available = torch.backends.mps.is_available()
    if name == "mps" and not available:
        raise ValueError("device mps: torch.backends.mps is not available")
    return "mps" if name == "mps" or (name == "auto" and available) else "cpu"


def family_draw_seeds(seed: int, draws: int) -> tuple[int, ...]:
    """``seed``, then for each further draw the first later seed whose held-out intent families (``gate_sets``)
    are disjoint from those of every earlier draw.

    Raises:
        ValueError: If no such seed lies within SEED_SEARCH of ``seed``.
    """

    seeds, held, candidate = [seed], set(gate_sets(seed, 0, 0)[0].heldout_family_ids), seed
    while len(seeds) < draws:
        candidate += 1
        if candidate > seed + SEED_SEARCH:
            raise ValueError(f"no seed in {seed + 1}..{seed + SEED_SEARCH} holds out families disjoint from {sorted(held)}")
        families = set(gate_sets(candidate, 0, 0)[0].heldout_family_ids)
        if not families & held:
            seeds.append(candidate)
            held |= families
    return tuple(seeds)


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


class PilotModel:
    """The encoder f (masked mean pool) and the kind head on one device. Through ``dim`` and :meth:`encode`
    (eval mode, no gradient) it is a ``trunk.TrunkEncoder``, so ``interpret.IntentMatcher`` reads with it."""

    def __init__(self, encoder: Any, tokenizer: Any, head: torch.nn.Linear, device: str) -> None:
        self.encoder, self.tokenizer, self.head, self.device = encoder, tokenizer, head, device
        self.dim = int(encoder.config.hidden_size)
        self.to(device)

    @classmethod
    def pretrained(cls, model_name: str, device: str, seed: int) -> PilotModel:
        """The pretrained encoder and a fresh kind head (initialised under ``torch.manual_seed(seed)``)."""

        import transformers

        torch.manual_seed(seed)
        encoder = transformers.AutoModel.from_pretrained(model_name)
        head = torch.nn.Linear(int(encoder.config.hidden_size), len(KINDS))
        return cls(encoder, transformers.AutoTokenizer.from_pretrained(model_name), head, device)

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> PilotModel:
        """A model written by :meth:`save`."""

        import transformers

        encoder = transformers.AutoModel.from_pretrained(directory)
        head = torch.nn.Linear(int(encoder.config.hidden_size), len(KINDS))
        head.load_state_dict(torch.load(Path(directory) / "kind_head.pt", map_location="cpu"))
        return cls(encoder.eval(), transformers.AutoTokenizer.from_pretrained(directory), head, device)

    def save(self, directory: Path) -> None:
        """Write the encoder and tokenizer (``save_pretrained``) and ``kind_head.pt`` into ``directory``."""

        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        torch.save(self.head.state_dict(), directory / "kind_head.pt")

    def to(self, device: str) -> PilotModel:
        self.encoder.to(device)
        self.head.to(device)
        self.device = device
        return self

    def parameters(self) -> list[torch.nn.Parameter]:
        return [*self.encoder.parameters(), *self.head.parameters()]

    def parameter_counts(self) -> tuple[int, int]:
        """(encoder parameters, kind head parameters)."""

        return sum(item.numel() for item in self.encoder.parameters()), sum(item.numel() for item in self.head.parameters())

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """f(texts), float32[B, D] on the device, in the encoder's current mode and gradient context."""

        batch = self.tokenizer(list(texts), padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
        batch = {name: value.to(self.device) for name, value in batch.items()}
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """f(texts) in eval mode without gradient, float32[B, D] on the host."""

        self.encoder.eval()
        with torch.no_grad():
            return self.embed(texts).float().cpu().numpy()


class FineTunedConditionReader(ConditionReader):
    """``ConditionReader.read`` (the unit rule, the exact argument, "" -> always) over the fine-tuned kind head's p.

    Build it after training: it caches one embedding per distinct text.

    Args:
        model: The fine-tuned model (anything with ``dim``, ``encode`` and a ``head`` mapping D -> 9 logits).
    """

    def __init__(self, model: PilotModel) -> None:
        super().__init__(model)

    def fit(self, texts: Sequence[str], kinds: Sequence[str]) -> FineTunedConditionReader:
        """Refused: this reader is trained by :func:`train`, not fitted on frozen embeddings.

        Raises:
            TypeError: Always.
        """

        raise TypeError("FineTunedConditionReader is trained by finetune_pilot.train, not fitted")

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """p over KINDS from the kind head, float64[len(texts), 9]; an empty text is one-hot on always."""

        out = np.zeros((len(texts), len(KINDS)))
        out[:, KINDS.index(ALWAYS.kind)] = 1.0
        rows = [row for row, text in enumerate(texts) if text.strip()]
        if rows:
            x = _embed(self.encoder, [texts[row] for row in rows], self._cache)
            weight = self.encoder.head.weight.detach().cpu().numpy().astype(np.float64)
            bias = self.encoder.head.bias.detach().cpu().numpy().astype(np.float64)
            out[rows] = _softmax(x @ weight.T + bias)
        return out


@dataclass(frozen=True)
class TrainLog:
    """Steps taken, steps per epoch, wall seconds (device synchronised) and the mean (L_kind, L_match) per epoch."""

    steps: int
    steps_per_epoch: int
    seconds: float
    epoch_losses: tuple[tuple[float, float], ...]


def train(
    model: PilotModel, examples: Sequence[GoalExample], *, epochs: int, seed: int, max_steps: int | None = None
) -> TrainLog:
    """Fine-tune ``model`` in place (module docstring, Model), printing one line per epoch.

    Raises:
        ValueError: If there is no example, ``epochs`` is below 1 or ``max_steps`` is below 1.
    """

    if not examples:
        raise ValueError("train needs at least one example")
    if epochs < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError(f"epochs and max_steps must be at least 1, got {epochs} and {max_steps}")
    per_epoch = math.ceil(len(examples) / BATCH_GOALS)
    total = epochs * per_epoch if max_steps is None else min(max_steps, epochs * per_epoch)
    warmup = max(1, round(WARMUP_FRACTION * total))
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": ENCODER_LR},
            {"params": list(model.head.parameters()), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, (total - step) / max(1, total - warmup))
    )
    rng = np.random.default_rng(seed)
    model.encoder.train()
    epoch_losses: list[tuple[float, float]] = []
    step = 0
    _synchronize(model.device)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(examples))
        sums, count = torch.zeros(2, device=model.device), 0
        for start in range(0, len(order), BATCH_GOALS):
            if step == total:
                break
            whens, kinds, intents, descriptions, positives = batch_inputs(
                [examples[index] for index in order[start : start + BATCH_GOALS]]
            )
            embedded = model.embed([*whens, *intents, *descriptions])
            first, second = len(whens), len(whens) + len(intents)
            kind_loss, match_loss = pilot_losses(
                embedded[:first],
                torch.tensor(kinds, dtype=torch.long, device=model.device),
                embedded[first:second],
                embedded[second:],
                torch.tensor(positives, dtype=torch.long, device=model.device),
                model.head,
            )
            optimizer.zero_grad()
            (kind_loss + match_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            schedule.step()
            sums += torch.stack([kind_loss.detach(), match_loss.detach()])
            count, step = count + 1, step + 1
        if count == 0:
            break
        kind_mean, match_mean = (sums / count).tolist()
        epoch_losses.append((kind_mean, match_mean))
        print(
            f"  epoch {epoch}/{epochs}: {count} steps, mean loss kind {kind_mean:.4f}, match {match_mean:.4f}, "
            f"{time.perf_counter() - started:.1f} s",
            flush=True,
        )
    _synchronize(model.device)
    seconds = time.perf_counter() - started
    model.encoder.eval()
    return TrainLog(step, per_epoch, seconds, tuple(epoch_losses))


def polarity_confusion(truth: Sequence[str], read: Sequence[str], pair: tuple[str, str]) -> dict[str, Any]:
    """Per true kind of ``pair``: how often it was read as each kind of the pair, as any other kind, and its count."""

    table = {}
    for kind in pair:
        got = Counter(answer for true, answer in zip(truth, read) if true == kind)
        table[kind] = {
            "as": {name: got[name] for name in pair},
            "other": {name: count for name, count in got.most_common() if name not in pair},
            "count": sum(got.values()),
        }
    return table


def evaluate(reader: ConditionReader, matcher: IntentMatcher, sets: Any, rows: Mapping[str, list]) -> dict[str, Any]:
    """The benchmark's metrics for one reader and the zero-shot matcher (module docstring), plus the per-kind
    accuracy and the polarity confusions on held-out wordings and the fit on the training set."""

    def read_kinds(samples):
        worded = [(goal.when, meaning.predicate.kind) for goal, meaning, _ in _goals(samples) if goal.when.strip()]
        texts = [text for text, _ in worded]
        alone = [KINDS[int(index)] for index in np.argmax(reader.probabilities(texts), axis=1)] if texts else []
        return [kind for _, kind in worded], [item.kind for item in reader.read(texts)], alone

    truth, read, alone = read_kinds(sets.heldout_wordings)
    metrics: dict[str, Any] = {
        "kind": _rate([a == b for a, b in zip(read, truth)]),
        "kind_alone": _rate([a == b for a, b in zip(alone, truth)]),
        "kind_count": len(truth),
        "per_kind": {
            kind: [_rate([answer == kind for true, answer in zip(truth, read) if true == kind]), truth.count(kind)]
            for kind in KINDS
        },
        "polarity": {f"{a}/{b}": polarity_confusion(truth, read, (a, b)) for a, b in POLARITY_PAIRS},
    }
    train_truth, train_read, _ = read_kinds(sets.train)
    metrics["kind_train"] = _rate([a == b for a, b in zip(train_read, train_truth)])
    metrics["kind_train_count"] = len(train_truth)
    held_ids = set(sets.heldout_family_ids)
    for split in ("train", *SPLITS):
        samples = getattr(sets, split)
        intents = sorted({goal.intent for goal, _, _ in _goals(samples)})
        descriptions = sorted({text for sample in samples for text in sample.catalogue.values()})
        matcher.scores(intents, descriptions, zero_shot=True)  # fills the matcher's cache in batches
        goals = [item for item in _goals(samples) if split != "heldout_families" or item[1].intent_family in held_ids]
        hits = [
            matcher.match([goal.intent], sample.catalogue, zero_shot=True)[0].skill_id == meaning.skill_id
            for goal, meaning, sample in goals
        ]
        metrics[f"intent_{split}"], metrics[f"intent_{split}_count"] = _rate(hits), len(hits)
    for split in SPLITS:
        samples = getattr(sets, split)
        reader.probabilities(sorted({goal.when for goal, _, _ in _goals(samples)}))  # fills the reader's cache
        skill_hits, replace_hits = [], []
        for sample, sample_rows in zip(samples, rows[split]):
            readings = [interpret(goal, sample.catalogue, reader, matcher, zero_shot=True) for goal in sample.section.goals]
            for values, flags, current in sample_rows:
                expected = teacher(sample.section, sample.meanings, values, flags, current)
                got = decide(sample.section, readings, values, flags, current)
                skill_hits.append(got[0] == expected[0])
                replace_hits.append(got[1] == expected[1])
        metrics[f"m13_{split}"], metrics[f"m13_{split}_replace"] = _rate(skill_hits), _rate(replace_hits)
        metrics[f"m13_{split}_rows"] = len(skill_hits)
    return metrics


def _confusion_text(table: Mapping[str, Any], pair: tuple[str, str]) -> str:
    parts = []
    for kind in pair:
        row = table[kind]
        other = ", ".join(f"{name} {count}" for name, count in row["other"].items())
        parts.append(
            f"{kind} read as {pair[0]} {row['as'][pair[0]]}, {pair[1]} {row['as'][pair[1]]}, "
            f"other {sum(row['other'].values())}{f' ({other})' if other else ''} of {row['count']}"
        )
    return "; ".join(parts)


def report(stage: str, metrics: Mapping[str, Any], sets: Any) -> None:
    """Print one stage's metric lines (``frozen`` or ``fine-tuned``)."""

    kind = metrics["kind"]
    print(
        f"  {stage}: kind accuracy, held-out wordings ({metrics['kind_count']} non-empty when texts): reader "
        f"{kind:.4f} [>= {KIND_BAR}: {_mark(kind, KIND_BAR)}], classifier alone {metrics['kind_alone']:.4f}; "
        f"training texts ({metrics['kind_train_count']}) {metrics['kind_train']:.4f}"
    )
    per_kind = ", ".join(f"{name} {rate:.4f} ({count})" for name, (rate, count) in metrics["per_kind"].items())
    print(f"  {stage}: per-kind accuracy, held-out wordings: {per_kind}")
    for a, b in POLARITY_PAIRS:
        print(f"  {stage}: polarity {a} / {b}, held-out wordings: {_confusion_text(metrics['polarity'][f'{a}/{b}'], (a, b))}")
    labels = {
        "heldout_wordings": "held-out wordings (all goals)",
        "heldout_families": f"held-out families ({', '.join(sets.heldout_family_ids)} goals)",
        "heldout_sizes": "held-out sizes (all goals)",
    }
    for split in SPLITS:
        value = metrics[f"intent_{split}"]
        print(
            f"  {stage}: intent -> skill (nearest description), {labels[split]} ({metrics[f'intent_{split}_count']}): "
            f"{value:.4f} [>= {INTENT_BAR}: {_mark(value, INTENT_BAR)}]"
        )
    print(f"  {stage}: intent -> skill, training goals ({metrics['intent_train_count']}): {metrics['intent_train']:.4f}")
    for split in SPLITS:
        value = metrics[f"m13_{split}"]
        print(
            f"  {stage}: end-to-end M13 agreement, {split.replace('heldout_', 'held-out ')} "
            f"({metrics[f'm13_{split}_rows']} rows): skill {value:.4f} [>= {M13_BAR}: {_mark(value, M13_BAR)}], "
            f"replace {metrics[f'm13_{split}_replace']:.4f}",
            flush=True,
        )


def seconds_per_thousand(encoder: Any, texts: Sequence[str]) -> float:
    """Seconds to encode 1,000 of ``texts`` through ``interpret._embed`` (ENCODE_BATCH per call, fresh cache), after
    one warm-up call."""

    encoder.encode(list(texts[:8]))
    clock = time.perf_counter()
    _embed(encoder, list(texts), {})
    return (time.perf_counter() - clock) * 1000 / len(texts)


def run_draw(args: argparse.Namespace, draw: int, seed: int, frozen: Any, beliefs: Mapping[str, np.ndarray], device: str) -> dict[str, Any]:
    """Build the draw's sets, measure the frozen reader, fine-tune, measure, time, save; return the record."""

    sets, split_keys = gate_sets(seed, args.train, args.eval)
    rows = {split: belief_rows(getattr(sets, split), beliefs, key, args.rows_per_state) for split, key in zip(SPLITS, split_keys)}
    examples = training_examples(sets.train)
    print(
        f"draw {draw}: seed {seed}, held-out families {', '.join(sets.heldout_family_ids)}; {len(sets.train)} training "
        f"states -> {len(examples)} goals ({sum(1 for item in examples if item.when.strip())} non-empty conditions, "
        f"{len({item.intent for item in examples})} distinct intents, {len({item.description for item in examples})} "
        f"descriptions); {args.eval} states per held-out split, {args.rows_per_state} belief rows per state",
        flush=True,
    )
    train_goals = _goals(sets.train)
    frozen_reader = ConditionReader(frozen).fit(
        [goal.when for goal, _, _ in train_goals], [meaning.predicate.kind for _, meaning, _ in train_goals]
    )
    before = evaluate(frozen_reader, IntentMatcher(frozen), sets, rows)
    report("frozen", before, sets)
    model = PilotModel.pretrained(args.encoder[len("torch:") :], device, seed)
    log = train(model, examples, epochs=args.epochs, seed=seed, max_steps=args.max_steps)
    print(
        f"  training on {device}: {log.steps} steps ({args.epochs} epochs x {log.steps_per_epoch}, batch {BATCH_GOALS} "
        f"goals) in {log.seconds:.1f} s; mean loss kind {log.epoch_losses[0][0]:.4f} -> {log.epoch_losses[-1][0]:.4f}, "
        f"match {log.epoch_losses[0][1]:.4f} -> {log.epoch_losses[-1][1]:.4f} (first -> last epoch)",
        flush=True,
    )
    after = evaluate(FineTunedConditionReader(model), IntentMatcher(model), sets, rows)
    report("fine-tuned", after, sets)
    every = {text for split in sets[:4] for goal, _, _ in _goals(split) for text in (goal.when, goal.intent) if text.strip()}
    timed = sorted(every)[:TIMED_TEXTS]
    on_device = seconds_per_thousand(model, timed)
    on_cpu = on_device if device == "cpu" else seconds_per_thousand(model.to("cpu"), timed)
    model.to("cpu")
    encoder_count, head_count = model.parameter_counts()
    steps = "" if args.max_steps is None else f"-steps{args.max_steps}"
    directory = args.out / f"{args.encoder[len('torch:') :].replace('/', '--')}-seed{seed}-{device}{steps}"
    model.save(directory)
    record = {
        "draw": draw,
        "seed": seed,
        "families": list(sets.heldout_family_ids),
        "encoder": args.encoder,
        "device": device,
        "epochs": args.epochs,
        "train_states": len(sets.train),
        "eval_states": args.eval,
        "rows_per_state": args.rows_per_state,
        "constants": {
            "batch_goals": BATCH_GOALS, "encoder_lr": ENCODER_LR, "head_lr": HEAD_LR, "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_FRACTION, "clip_norm": CLIP_NORM, "tau": TAU, "max_tokens": MAX_TOKENS,
        },
        "training": asdict(log),
        "seconds_per_1000": {device: on_device, "cpu": on_cpu},
        "timed_texts": len(timed),
        "parameters": {"encoder": encoder_count, "kind_head": head_count},
        "load_average": list(os.getloadavg()),
        "frozen": before,
        "fine-tuned": after,
    }
    (directory / "metrics.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    inference = f"{on_device:.3f} s per 1,000 texts on {device}" + (f", {on_cpu:.3f} s on cpu" if device != "cpu" else "")
    print(
        f"  timing (load average {os.getloadavg()[0]:.2f}): training {log.seconds:.1f} s on {device}; inference "
        f"{inference} ({len(timed)} distinct goal texts); parameters {encoder_count + head_count:,} (encoder "
        f"{encoder_count:,} + kind head {head_count:,})",
        flush=True,
    )
    print(f"  weights: {directory}", flush=True)
    return record


def _inside(path: Path, root: Path) -> bool:
    resolved = path.expanduser().resolve()
    return resolved == root or root in resolved.parents


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pilot: per family draw, the frozen and the fine-tuned reader on the three held-out splits; return 0."""

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.system_one.finetune_pilot",
        description="Fine-tuning pilot of the interpret stage (SYSTEM_ONE.md 8.13 (b), 9) on the G3-NL sets.",
    )
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="torch:<hf-model-id>")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="the first draw's seed (gate_sets)")
    parser.add_argument("--device", choices=DEVICES, default="auto", help="auto: mps when available, else cpu")
    parser.add_argument("--train", type=int, default=4096, help="training states (default 4096)")
    parser.add_argument("--eval", type=int, default=1024, help="states per held-out split (default 1024)")
    parser.add_argument("--rows-per-state", type=int, default=32, help="belief rows per held-out state (default 32)")
    parser.add_argument("--beliefs", type=Path, default=DEFAULT_BELIEFS, help="recorded dataset .npz of the belief rows")
    parser.add_argument("--family-draws", type=int, choices=(1, 2), default=2, help="held-out family draws (default 2)")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (timing runs)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="weights directory, outside the repository")
    args = parser.parse_args(argv)
    if not (args.encoder.startswith("torch:") and len(args.encoder) > len("torch:")):
        parser.error(f"--encoder must be torch:<model>, got {args.encoder!r}")
    if min(args.epochs, args.train, args.eval, args.rows_per_state) < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("--epochs, --train, --eval, --rows-per-state and --max-steps must be positive")
    if _inside(args.out, REPO_ROOT):
        parser.error(f"--out must lie outside the repository {REPO_ROOT}, got {args.out}")
    _offline_if_cached([args.encoder])
    device = resolve_device(args.device)
    import transformers

    transformers.utils.logging.disable_progress_bar()
    transformers.utils.logging.set_verbosity_error()  # e.g. the MLM head weights AutoModel leaves unused
    print(
        f"finetune pilot: encoder {args.encoder}, device {device} (requested {args.device}; "
        f"torch.backends.mps.is_available() {torch.backends.mps.is_available()}), torch {torch.__version__}, "
        f"transformers {transformers.__version__}, {torch.get_num_threads()} CPU threads; epochs {args.epochs}, batch "
        f"{BATCH_GOALS} goals, encoder lr {ENCODER_LR:g}, head lr {HEAD_LR:g}, tau {TAU:g}; load average "
        f"{' '.join(f'{value:.2f}' for value in os.getloadavg())}",
        flush=True,
    )
    seeds = family_draw_seeds(args.seed, args.family_draws)
    frozen = make_trunk(args.encoder)
    beliefs = load_dataset(args.beliefs)
    results = [run_draw(args, draw, seed, frozen, beliefs, device) for draw, seed in enumerate(seeds, start=1)]
    print(
        "\n| draw (held-out families) | stage | kind, wordings | intent wordings | intent families | intent sizes | "
        "M13 wordings | M13 families | M13 sizes |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for result in results:
        for stage in ("frozen", "fine-tuned"):
            metrics = result[stage]
            cells = [
                (metrics["kind"], KIND_BAR),
                *((metrics[f"intent_{split}"], INTENT_BAR) for split in SPLITS),
                *((metrics[f"m13_{split}"], M13_BAR) for split in SPLITS),
            ]
            print(
                f"| {result['draw']}: seed {result['seed']} ({', '.join(result['families'])}) | {stage} | "
                + " | ".join(f"{value:.4f} {_mark(value, bar)}" for value, bar in cells)
                + " |"
            )
    print(f"load average at the end: {' '.join(f'{value:.2f}' for value in os.getloadavg())}")
    return 0


__all__ = [
    "BATCH_GOALS",
    "CLIP_NORM",
    "DEFAULT_ENCODER",
    "DEFAULT_OUT",
    "DEVICES",
    "ENCODER_LR",
    "FineTunedConditionReader",
    "GoalExample",
    "HEAD_LR",
    "M13_BAR",
    "MAX_TOKENS",
    "POLARITY_PAIRS",
    "PilotModel",
    "REPO_ROOT",
    "TAU",
    "TrainLog",
    "WARMUP_FRACTION",
    "WEIGHT_DECAY",
    "batch_inputs",
    "evaluate",
    "family_draw_seeds",
    "main",
    "pilot_losses",
    "polarity_confusion",
    "report",
    "resolve_device",
    "run_draw",
    "seconds_per_thousand",
    "train",
    "training_examples",
]

if __name__ == "__main__":
    raise SystemExit(main())
