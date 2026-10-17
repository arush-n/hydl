"""Intent -> skill matching on the owner's real catalogue: does description matching scale (SYSTEM_ONE.md 8.13)?

``interpret.IntentMatcher`` reads an LLM ``intent`` text into the catalogue skill whose description it is nearest
to; ``interpret_benchmark`` measured it on ten synthetic families in catalogues of 3 to 8 skills. This module
measures the same matcher on the owner's catalogue of Hytale 0.7.2 skills, whole, with intents written for it.
Both inputs are private and untracked; each reader returns None when its file is absent, and the CLI then prints
a skip line and returns 0::

    catalogue    agent_skills/training_runs/SKILL_CATALOGUE.md             --catalogue
    paraphrases  agent_skills/training_runs/intent_paraphrases_072.json    --paraphrases

Catalogue reader (:func:`parse_catalogue`). Under a heading ``## N. Title``, a markdown table whose first header
cell is ``skill`` gives one row per line: ids = the skill cell split on commas (``walk_to, run_to, sprint_to`` is
one row: one description, the pace a parameter), group = Title, description = the ``what a person does`` cell,
kind = the first token of the ``kind`` cell (L, C, A or ?). A table whose first header cell is ``activity`` gives
kind A rows described by their ``sequence`` cell. Other tables (``Ranges at a glance``) are skipped.

Candidates. The scored choice is over the learned and command rows (kind L or C): the skills the matcher picks
among, since activities are chained by the middle layer. The "all rows" column ranks every row instead, the A and
? rows acting as distractors.

Paraphrases (:func:`load_paraphrases`): ``{skill_id: {"split_group": group, "paraphrases": [...]}}``, one key per
L or C row (any of its ids), ``split_group`` equal to the row's group. Rules (:func:`paraphrase_violations`: the
generator's rules, as ``test_na_goal_language`` checks them), with tokens(t) = the lower-cased runs of letters and
digits of t once apostrophes are removed, and content(t) = {first five letters of each token not in STOP_WORDS}::

    count     at least MIN_PARAPHRASES per L or C row
    unique    tokens(t) distinct over every paraphrase of every skill
    distance  token Levenshtein >= 2 between two paraphrases of one skill
    disjoint  3 * #{t : content(t) & content(description) = {}} >= #paraphrases, per skill
    no ids    no paraphrase holds tokens(id) as a contiguous run, for every id of every row (any kind)

Matchers and splits (u, s_j and Delta as in ``interpret``)::

    zero-shot              s_j = u(intent) . u(desc_j) on every paraphrase; nothing is fitted
    (a) held-out wordings  fold(t) = sigma_k(position of t among skill k's paraphrases), sigma_k a seeded permutation:
                           fold f holds out one paraphrase of every skill that has more than f; the map is fitted
                           on the other folds (every skill seen) and scores fold f; each paraphrase is scored once
    (b) held-out groups    per group g with paraphrases: the map is fitted on the paraphrases of the other groups and
                           scores g's paraphrases against the whole catalogue; every group is held out once

Metrics over the scored paraphrases, y the true row, rank = the candidates by s, best first, ties to the lower row::

    top-1          mean [rank_1 = y]                                                 bar TOP1_BAR = 0.90
    top-3          mean [y in rank_1..3]
    within group   #{errors with group(rank_1) = group(y)} / #errors; the most frequent (y -> rank_1) pairs within
                   the true group, across groups, and within the worst group
    per group      top-1 over the group's paraphrases: its mean over groups and the worst group
    by size K      top-1 when the candidates are y and K - 1 other L or C rows drawn uniformly per paraphrase
                   (seeded by --seed, the same draws for every run and encoder), K in SIZES, then every candidate
    cost           seconds to encode every row's description once (fresh cache, after one warm-up call), and
                   seconds per 1,000 distinct intents matched zero-shot (encode, score, rank; descriptions cached)

``--weights DIR[,DIR]`` adds fine-tuned encoders saved with ``save_pretrained`` (the fine-tuning pilot's
directories), each read as ``torch:DIR`` (masked mean pool, the pilot's own pooling) and labelled ``weights:<name>``::

    PYTHONPATH=. JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 python -m experimental.neural_agent.jev.system_one.catalogue_matching \\
        --encoder hashed,torch:sentence-transformers/all-MiniLM-L6-v2,torch:answerdotai/ModernBERT-base --seed 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experimental.neural_agent.executive.goal_language import STOP_WORDS
from experimental.neural_agent.jev.system_one.interpret import IntentMatcher, _embed
from experimental.neural_agent.jev.system_one.trunk import TrunkEncoder

_PRIVATE = Path(__file__).resolve().parents[4] / "agent_skills" / "training_runs"
DEFAULT_CATALOGUE: Path = _PRIVATE / "SKILL_CATALOGUE.md"
"""The owner's catalogue (private, untracked)."""
DEFAULT_PARAPHRASES: Path = _PRIVATE / "intent_paraphrases_072.json"
"""The intent paraphrases written for it (private, untracked)."""
ROW_KINDS: tuple[str, ...] = ("L", "C", "A", "?")
MATCHED_KINDS: tuple[str, ...] = ("L", "C")
"""The kinds the matcher chooses among."""
MIN_PARAPHRASES: int = 6
TOP1_BAR: float = 0.90
TOP_CONFUSIONS: int = 5
"""Confused (true -> chosen) pairs printed per list."""
SIZES: tuple[int, ...] = (4, 8, 16, 32, 64)
"""Sub-catalogue sizes of the size curve."""
RUNS: tuple[tuple[str, str], ...] = (
    ("zero_shot", "zero-shot, every paraphrase"),
    ("wordings", "fitted, (a) held-out wordings"),
    ("groups", "fitted, (b) held-out groups"),
)

_HEADING = re.compile(r"^##\s+\d+\.\s+(.+?)\s*$")
_ID = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class CatalogueRow:
    """One catalogue row: its ids, group (section title), description and kind (L, C, A or ?)."""

    ids: tuple[str, ...]
    group: str
    description: str
    kind: str


def parse_catalogue(text: str) -> tuple[CatalogueRow, ...]:
    """Return the rows of the skill and activity tables under numbered sections (module docstring).

    Raises:
        ValueError: If a row's cell count differs from its header's, an id is malformed or a kind is unknown.
    """

    rows: list[CatalogueRow] = []
    group: str | None = None
    header: list[str] | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("#"):
            heading = _HEADING.match(line)
            group, header = (heading.group(1) if heading else None), None
            continue
        if not line.lstrip().startswith("|"):
            header = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = cells if group is not None and cells[0] in ("skill", "activity") else []
            continue
        if not header or set(line.strip()) <= set("|-: "):
            continue
        if len(cells) != len(header):
            raise ValueError(f"line {number}: {len(cells)} cells under a header of {len(header)}")
        named = dict(zip(header, cells))
        ids = tuple(item.strip() for item in cells[0].split(","))
        if not all(_ID.match(item) for item in ids):
            raise ValueError(f"line {number}: malformed skill ids {cells[0]!r}")
        if header[0] == "activity":
            kind, description = "A", named["sequence"]
        else:
            kind, description = (named["kind"].split() or [""])[0], named["what a person does"]
        if kind not in ROW_KINDS:
            raise ValueError(f"line {number}: unknown kind {named.get('kind')!r}; expected one of {ROW_KINDS}")
        rows.append(CatalogueRow(ids, group, description, kind))
    return tuple(rows)


def read_catalogue(path: Path | str = DEFAULT_CATALOGUE) -> tuple[CatalogueRow, ...] | None:
    """Return the rows of the catalogue at ``path``, or None when the file is absent."""

    path = Path(path)
    return parse_catalogue(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_paraphrases(path: Path | str, rows: Sequence[CatalogueRow]) -> dict[int, tuple[str, ...]] | None:
    """Return {row index: paraphrases} from the JSON at ``path`` (module docstring), or None when it is absent.

    Raises:
        ValueError: If a key is not an id of an L or C row, two keys name one row, a ``split_group`` is not the
            row's group, or the paraphrases are not a list of non-empty strings.
    """

    path = Path(path)
    if not path.is_file():
        return None
    row_of = {skill_id: index for index, row in enumerate(rows) for skill_id in row.ids}
    found: dict[int, tuple[str, ...]] = {}
    for key, entry in json.loads(path.read_text(encoding="utf-8")).items():
        index = row_of.get(key)
        if index is None or rows[index].kind not in MATCHED_KINDS:
            raise ValueError(f"{key!r} is not an id of a learned or command row")
        if index in found:
            raise ValueError(f"{key!r} names the row {rows[index].ids} a second time")
        if entry.get("split_group") != rows[index].group:
            raise ValueError(f"{key!r}: split_group {entry.get('split_group')!r} is not its section {rows[index].group!r}")
        texts = entry.get("paraphrases")
        if not isinstance(texts, list) or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError(f"{key!r}: paraphrases must be a list of non-empty strings")
        found[index] = tuple(texts)
    return dict(sorted(found.items()))


def _tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower().replace("'", "").replace("’", "")).split()


def _content(text: str) -> set[str]:
    return {token[:5] for token in _tokens(text) if token not in STOP_WORDS}


def _token_distance(first: Sequence[str], second: Sequence[str]) -> int:
    """Levenshtein distance over tokens."""

    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (left != right)))
        previous = current
    return previous[-1]


def paraphrase_violations(rows: Sequence[CatalogueRow], paraphrases: Mapping[int, Sequence[str]]) -> list[str]:
    """Return one line per broken paraphrase rule (module docstring); empty when every rule holds."""

    runs = [_tokens(skill_id) for row in rows for skill_id in row.ids]
    owner: dict[str, str] = {}
    problems: list[str] = []
    for index, row in enumerate(rows):
        if row.kind not in MATCHED_KINDS:
            continue
        name, texts = row.ids[0], tuple(paraphrases.get(index, ()))
        if len(texts) < MIN_PARAPHRASES:
            problems.append(f"{name}: {len(texts)} paraphrases, fewer than {MIN_PARAPHRASES}")
        reference = _content(row.description)
        disjoint = sum(not _content(text) & reference for text in texts)
        if 3 * disjoint < len(texts):
            problems.append(f"{name}: only {disjoint} of {len(texts)} share no content word with the description")
        for first, second in itertools.combinations(texts, 2):
            if _token_distance(_tokens(first), _tokens(second)) < 2:
                problems.append(f"{name}: {first!r} and {second!r} differ by fewer than two tokens")
        for text in texts:
            tokens = _tokens(text)
            key = " ".join(tokens)
            if key in owner:
                problems.append(f"{name}: {text!r} repeats a paraphrase of {owner[key]}")
            owner.setdefault(key, name)
            held = [
                "_".join(run) for run in runs
                if any(tokens[start : start + len(run)] == run for start in range(len(tokens) - len(run) + 1))
            ]
            if held:
                problems.append(f"{name}: {text!r} holds the skill id tokens of {held}")
    return problems


def queries(paraphrases: Mapping[int, Sequence[str]]) -> tuple[list[str], np.ndarray]:
    """Every paraphrase, skill-major in row order, and its true row index."""

    order = sorted(paraphrases)
    texts = [text for index in order for text in paraphrases[index]]
    return texts, np.asarray([index for index in order for _ in paraphrases[index]], dtype=np.int64)


def wording_folds(paraphrases: Mapping[int, Sequence[str]], seed: int) -> np.ndarray:
    """The fold of every query of :func:`queries`: a seeded permutation of 0..m-1 per skill of m paraphrases."""

    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(len(paraphrases[index])) for index in sorted(paraphrases)])


def ranked(scores: np.ndarray, candidates: np.ndarray | None = None) -> np.ndarray:
    """Row indices per query, best first, ties to the lower row; rows outside ``candidates`` come last."""

    if candidates is not None:
        scores = np.where(candidates, scores, -np.inf)
    return np.argsort(-scores, axis=1, kind="stable")


def summary(rank: np.ndarray, truth: np.ndarray, groups: Sequence[str]) -> dict[str, Any]:
    """top-1, top-3, top-1 per group, and the errors: how many stay in the true row's group, the commonest
    confusions within it, across groups and in the worst group."""

    chosen = rank[:, 0]
    hits = chosen == truth
    confusions = Counter((int(true), int(pick)) for true, pick in zip(truth[~hits], chosen[~hits]))
    within = Counter({pair: count for pair, count in confusions.items() if groups[pair[0]] == groups[pair[1]]})
    true_groups = np.asarray([groups[index] for index in truth])
    per_group = {group: float(hits[true_groups == group].mean()) for group in dict.fromkeys(true_groups.tolist())}
    worst = min(per_group, key=per_group.get)
    in_worst = Counter({pair: count for pair, count in confusions.items() if groups[pair[0]] == worst})
    return {
        "n": int(len(truth)),
        "top1": float(hits.mean()),
        "top3": float((rank[:, :3] == truth[:, None]).any(axis=1).mean()),
        "errors": int((~hits).sum()),
        "within": sum(within.values()),
        "within_confusions": within.most_common(TOP_CONFUSIONS),
        "cross_confusions": (confusions - within).most_common(TOP_CONFUSIONS),
        "per_group": per_group,
        "group_mean": float(np.mean(list(per_group.values()))),
        "worst": (worst, per_group[worst]),
        "worst_confusions": in_worst.most_common(TOP_CONFUSIONS),
    }


def size_curve(scores: np.ndarray, truth: np.ndarray, candidates: np.ndarray, seed: int) -> dict[int, float]:
    """top-1 by catalogue size K in SIZES below the candidate count: per paraphrase, the true row and K - 1 other
    candidates drawn uniformly (seeded), ties to the lower row."""

    pool = np.flatnonzero(candidates)
    rng = np.random.default_rng(seed)
    curve = {}
    for size in (size for size in SIZES if size < len(pool)):
        hits = 0
        for row, true in zip(scores, truth):
            subset = np.sort(np.append(rng.choice(pool[pool != true], size=size - 1, replace=False), true))
            hits += int(subset[np.argmax(row[subset])] == true)
        curve[size] = hits / len(truth)
    return curve


def evaluate(
    encoder: TrunkEncoder, rows: Sequence[CatalogueRow], paraphrases: Mapping[int, Sequence[str]], seed: int
) -> dict[str, dict[str, Any]]:
    """Zero-shot, (a) and (b) on one encoder (module docstring): a :func:`summary` per run over the L and C
    candidates, plus ``top1_all`` (every row a candidate) and ``sizes`` (:func:`size_curve`, then all candidates)."""

    texts, truth = queries(paraphrases)
    descriptions = [row.description for row in rows]
    groups = [row.group for row in rows]
    candidates = np.asarray([row.kind in MATCHED_KINDS for row in rows])
    matcher = IntentMatcher(encoder)
    folds = {
        "wordings": wording_folds(paraphrases, seed),
        "groups": np.asarray([groups[index] for index in truth]),
    }
    results = {}
    for name, _ in RUNS:
        if name == "zero_shot":
            scores = matcher.scores(texts, descriptions, zero_shot=True)
        else:
            scores = np.zeros((len(texts), len(rows)))
            for fold in dict.fromkeys(folds[name].tolist()):
                test = folds[name] == fold
                train = np.flatnonzero(~test)
                matcher.fit([texts[i] for i in train], [descriptions[truth[i]] for i in train])
                scores[test] = matcher.scores([texts[i] for i in np.flatnonzero(test)], descriptions)
        results[name] = summary(ranked(scores, candidates), truth, groups)
        results[name]["top1_all"] = float((ranked(scores)[:, 0] == truth).mean())
        results[name]["sizes"] = {**size_curve(scores, truth, candidates, seed), int(candidates.sum()): results[name]["top1"]}
    return results


def costs(encoder: TrunkEncoder, rows: Sequence[CatalogueRow], texts: Sequence[str]) -> tuple[float, float]:
    """(seconds to encode every row's description once, seconds per 1,000 distinct intents matched zero-shot)."""

    descriptions = [row.description for row in rows]
    encoder.encode(["warm up"])
    clock = time.perf_counter()
    _embed(encoder, descriptions, {})
    encode = time.perf_counter() - clock
    matcher = IntentMatcher(encoder)
    matcher.scores([], descriptions, zero_shot=True)  # caches the descriptions
    distinct = sorted(set(texts))
    clock = time.perf_counter()
    ranked(matcher.scores(distinct, descriptions, zero_shot=True))
    return encode, (time.perf_counter() - clock) * 1000 / len(distinct)


def _mark(value: float) -> str:
    return "PASS" if value >= TOP1_BAR else "FAIL"


def _pairs(rows: Sequence[CatalogueRow], pairs: Sequence[tuple[tuple[int, int], int]]) -> str:
    return ", ".join(f"{rows[true].ids[0]} -> {rows[pick].ids[0]} x{count}" for (true, pick), count in pairs) or "none"


def report_encoder(label: str, encoder: TrunkEncoder, rows: Sequence[CatalogueRow], paraphrases, seed: int) -> dict:
    """Measure one encoder, print its block and return its metrics."""

    started = time.perf_counter()
    texts, _ = queries(paraphrases)
    encode, match = costs(encoder, rows, texts)
    results = evaluate(encoder, rows, paraphrases, seed)
    floor = " (the floor)" if label == "hashed" else ""
    print(
        f"\nencoder {label}{floor}: dim {encoder.dim}; encode the {len(rows)} descriptions once {encode:.4f} s; "
        f"match {match:.4f} s per 1,000 intents (load {os.getloadavg()[0]:.1f})"
    )
    folds = int(wording_folds(paraphrases, seed).max()) + 1
    for name, title in RUNS:
        item = results[name]
        split = {"zero_shot": "", "wordings": f", {folds} folds", "groups": f", {len(item['per_group'])} groups"}[name]
        print(
            f"  {title} ({item['n']}{split}): top-1 {item['top1']:.4f} [>= {TOP1_BAR}: {_mark(item['top1'])}], "
            f"top-3 {item['top3']:.4f}, top-1 over all {len(rows)} rows {item['top1_all']:.4f}; group mean "
            f"{item['group_mean']:.4f}, worst {item['worst'][0]} {item['worst'][1]:.4f}; errors {item['errors']}, "
            f"within the true group {item['within']}"
        )
        print(f"    confusions within the true group ({item['within']} of {item['errors']} errors): "
              f"{_pairs(rows, item['within_confusions'])}")
        print(f"    confusions across groups: {_pairs(rows, item['cross_confusions'])}")
        print(f"    confusions in the worst group, {item['worst'][0]}: {_pairs(rows, item['worst_confusions'])}")
        print("    top-1 by catalogue size K: " + ", ".join(f"{size} {value:.4f}" for size, value in item["sizes"].items()))
    print("  | group | skills | paraphrases | zero-shot | fitted (a) | fitted (b), group held out |")
    print("  |---|---|---|---|---|---|")
    for group in results["zero_shot"]["per_group"]:
        skills = [index for index in paraphrases if rows[index].group == group]
        cells = " | ".join(f"{results[name]['per_group'][group]:.4f}" for name, _ in RUNS)
        print(f"  | {group} | {len(skills)} | {sum(len(paraphrases[i]) for i in skills)} | {cells} |")
    print(f"  [{time.perf_counter() - started:.1f} s]", flush=True)
    return {"encoder": label, "dim": int(encoder.dim), "encode": encode, "match": match, **results}


def main(argv: Sequence[str] | None = None) -> int:
    """Print the catalogue counts, one block per encoder and the summary tables; return 0 (2 on broken rules)."""

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.system_one.catalogue_matching",
        description="Intent -> skill matching on the owner's catalogue (SYSTEM_ONE.md 8.13), every encoder.",
    )
    parser.add_argument("--encoder", default="hashed", help="comma-separated: hashed, torch:<hf-model-id or path>")
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--paraphrases", type=Path, default=DEFAULT_PARAPHRASES)
    parser.add_argument("--seed", type=int, default=0, help="seed of the held-out wording folds and the size curve")
    parser.add_argument(
        "--weights", default="", help="comma-separated save_pretrained directories of fine-tuned encoders, read as torch:DIR"
    )
    args = parser.parse_args(argv)
    specs = [spec.strip() for spec in args.encoder.split(",") if spec.strip()]
    if not specs:
        parser.error("--encoder needs at least one encoder")
    for spec in specs:
        if spec != "hashed" and not (spec.startswith("torch:") and len(spec) > len("torch:")):
            parser.error(f"--encoder entries must be hashed or torch:<model>, got {spec!r}")
    weights = [Path(item.strip()) for item in args.weights.split(",") if item.strip()]
    for directory in weights:
        if not (directory / "config.json").is_file():
            parser.error(f"--weights entries must be save_pretrained directories; {directory} has no config.json")
    encoders = [(spec, spec) for spec in specs] + [(f"weights:{path.name}", f"torch:{path}") for path in weights]
    rows = read_catalogue(args.catalogue)
    if rows is None:
        print(f"skipped: no catalogue at {args.catalogue}")
        return 0
    paraphrases = load_paraphrases(args.paraphrases, rows)
    if paraphrases is None:
        print(f"skipped: no paraphrases at {args.paraphrases}")
        return 0
    kinds = Counter(row.kind for row in rows)
    matched = [row for row in rows if row.kind in MATCHED_KINDS]
    print(
        f"catalogue {args.catalogue}: {len(rows)} rows, {sum(len(row.ids) for row in rows)} ids; "
        + ", ".join(f"{kind} {kinds[kind]}" for kind in ROW_KINDS)
        + f"; candidates (L or C) {len(matched)} rows in {len({row.group for row in matched})} groups"
    )
    print("| group | L | C | A | ? |\n|---|---|---|---|---|")
    for group in dict.fromkeys(row.group for row in rows):
        counts = Counter(row.kind for row in rows if row.group == group)
        print(f"| {group} | " + " | ".join(str(counts[kind]) for kind in ROW_KINDS) + " |")
    same: dict[str, list[str]] = {}
    for row in matched:
        same.setdefault(row.description, []).append(row.ids[0])
    shared = [f"{text!r} ({', '.join(ids)})" for text, ids in same.items() if len(ids) > 1]
    print(f"candidate descriptions shared by two or more rows: {'; '.join(shared) or 'none'}")
    sizes = [len(texts) for texts in paraphrases.values()]
    problems = paraphrase_violations(rows, paraphrases)
    print(
        f"paraphrases {args.paraphrases}: {len(paraphrases)} of {len(matched)} candidate rows, {sum(sizes)} texts "
        f"({min(sizes)} to {max(sizes)} per row); rules: {'OK' if not problems else f'{len(problems)} broken'}"
    )
    if problems:
        print("\n".join(f"  {line}" for line in problems))
        return 2
    print(f"load average at start {' '.join(f'{value:.1f}' for value in os.getloadavg())} ({os.cpu_count()} cores)")
    from experimental.neural_agent.jev.system_one.train_supervised import make_trunk

    results = [report_encoder(label, make_trunk(spec), rows, paraphrases, args.seed) for label, spec in encoders]
    print(f"\nload average at end {' '.join(f'{value:.1f}' for value in os.getloadavg())}")
    print(
        f"\n| encoder | matcher, split | n | top-1 [>= {TOP1_BAR}] | top-3 | top-1, all {len(rows)} rows | errors in the "
        "true group | group mean | worst group |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for item in results:
        for name, label in RUNS:
            run = item[name]
            print(
                f"| {item['encoder']} | {label} | {run['n']} | {run['top1']:.4f} {_mark(run['top1'])} | {run['top3']:.4f} | "
                f"{run['top1_all']:.4f} | {run['within']}/{run['errors']} | {run['group_mean']:.4f} | "
                f"{run['worst'][0]} {run['worst'][1]:.4f} |"
            )
    ks = list(results[0]["zero_shot"]["sizes"])
    print("\n| encoder | matcher, split | " + " | ".join(f"top-1, K = {k}" for k in ks) + " |")
    print("|---|---|" + "---|" * len(ks))
    for item in results:
        for name, label in RUNS:
            print(f"| {item['encoder']} | {label} | " + " | ".join(f"{value:.4f}" for value in item[name]["sizes"].values()) + " |")
    base = results[0]
    print(f"\n| encoder | dim | encode {len(rows)} descriptions, s | match, s per 1,000 intents | relative to {base['encoder']} |")
    print("|---|---|---|---|---|")
    for item in results:
        print(
            f"| {item['encoder']} | {item['dim']} | {item['encode']:.4f} | {item['match']:.4f} | "
            f"x{item['encode'] / base['encode']:.1f} encode, x{item['match'] / base['match']:.1f} match |"
        )
    return 0


__all__ = [
    "CatalogueRow",
    "DEFAULT_CATALOGUE",
    "DEFAULT_PARAPHRASES",
    "MATCHED_KINDS",
    "MIN_PARAPHRASES",
    "ROW_KINDS",
    "SIZES",
    "TOP1_BAR",
    "costs",
    "evaluate",
    "load_paraphrases",
    "main",
    "paraphrase_violations",
    "parse_catalogue",
    "queries",
    "ranked",
    "read_catalogue",
    "size_curve",
    "summary",
    "wording_folds",
]

if __name__ == "__main__":
    raise SystemExit(main())
