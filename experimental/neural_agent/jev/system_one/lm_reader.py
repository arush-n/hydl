"""Decoder-LM reader of the interpret stage: a small instruction-tuned LM reads conditions and picks skills (ledger "P6").

docs/SYSTEM_ONE.md 8.13 splits reading into *interpret* (once per executive version) and *evaluate* (every decision).
Three encoder readers failed it (G3-NL; the frozen interpret stage; the fine-tune pilot P2, which reads the feature at
0.99 but loses the side of a comparison and negations, 9.12), and description matching failed over the real
catalogue (P3: 0.42 top-1 over 119 rows, 9.11). The owner then allowed a larger reader at the version timescale
(ledger, "owner rulings after the reading failures": Qwen-class decoders, accuracy the bar, hundreds of milliseconds
per version acceptable). This module measures such a reader on exactly the earlier experiments' held-out sets, so its
numbers sit beside theirs. Nothing here is wired into the decision path.

Prompts (fixed bytes for fixed inputs; :func:`condition_system_prompt`, :func:`skill_system_prompt`)::

    condition   the nine kinds of goal_language with their meaning (its Predicates table) and argument unit, then one
                example per kind: the kind's first TRAINING paraphrase under the run's wording split (never a held-out
                one; :func:`prompt_examples`), {n} written as digits and unit (5 blocks, 60 ticks, 30%)
    skill       the catalogue, one line "ids: description" per row, and the rule: always one id, no "none"
                (SYSTEM_ONE.md 8.12 item 4: "choose the best possible choice")
    user        Condition: "<when>"   |   Intent: "<intent>"          (json-quoted)
    schema      {"kind": <one of KINDS>, "argument": <number | null>}   |   {"skill_id": <one of the ids>}

Qwen3 thinking is off: ``enable_thinking=False`` in the chat template (transformers), ``think: false`` (Ollama).

Schema-constrained decoding.
  transformers (:class:`TransformersBackend`). The assistant turn is forced to the schema's literal text up to the
  value, '{"kind": "' or '{"skill_id": "'. Every allowed value v, with its closing text ('", "argument":' or '"}'),
  is scored by its sequence log-probability::

      score(v) = sum_t log p(v_t | prompt, v_<t)          value = argmax_v score(v)          top-3: the next best

  in one forward pass per pack of queries: the system prompt is a cached prefix (computed once per prompt), and the
  queries with all their candidates are packed into one sequence whose 4-D mask lets a token see the prefix, its own
  query and its own candidate's earlier tokens only, with positions restarting per candidate. The greedy
  grammar-constrained value (the best allowed next token, step by step: what Ollama's grammar does at temperature 0)
  is read from the same log-probs (:func:`greedy_choice`) and reported beside it. The condition's number is then
  generated greedily from the chosen kind's context under the grammar NUMBER_DONE = ' ?(null|-?\\d+(\\.\\d+)?)}',
  each step taking the best token whose text keeps the output a prefix of it (NUMBER_PREFIX), at most
  NUMBER_TOKENS tokens. Weights in float32 on MPS and CPU (the bfloat16 checkpoint upcast; no quantisation): in
  bfloat16 on MPS the log-probs moved by up to 0.4 from float32 on both the packed and the naive path, in float32
  the packed scores equal one full forward per candidate to 1e-4 (checked on Qwen3-0.6B, 2026-10-17).
  Ollama (:class:`OllamaBackend`). ``format`` = the JSON schema (llama.cpp grammar), temperature 0, seed 0.
  Every output goes through the strict parser (:func:`parse_condition`, :func:`parse_skill`): a JSON object with
  exactly the schema's keys and a value in the vocabulary. Repairs, tried only when the strict parse fails: "format"
  (the first {...} in the text parses strictly; the answer is unchanged) and "content" (the value is snapped to the
  vocabulary by case, spaces and hyphens; the answer changed, so it counts as an error); otherwise "invalid".

Argument. The predicate's argument is ``interpret.argument_for(kind, read_number(when))``, the exact reader (1.000 in
the interpret benchmark); the model's own number is measured beside it and never used.

Two-pass variant (a model name + RESTATE_SUFFIX, "+restate"; coordinator relaying the owner, 2026-10-17 night: polarity
was the encoders' failure mode). Pass one asks, under the same system prompt, for the condition restated in one plain
sentence in the predicates' terms, as {"restatement": "<sentence>"} (RESTATE_SCHEMA): transformers force
'{"restatement": "' and decode greedily until the string closes (at most RESTATE_TOKENS tokens); Ollama uses ``format``,
because its qwen3:4b is the think-always Qwen3-4B-Thinking-2507 build, whose free text is a reasoning monologue whatever
the think flag. Pass two continues the conversation (that JSON as the assistant turn, then EMIT_USER) and reads
{kind, argument} exactly as one pass does. Thinking stays off in both passes.

Ollama placement. GPU mode sends ``num_gpu`` = OLLAMA_GPU_LAYERS and CPU mode ``num_gpu`` 0, and each backend checks
/api/ps after its first request (VRAM held for the GPU, none for the CPU), failing closed otherwise: Ollama keeps
serving a model from a CPU-only runner to requests that do not name ``num_gpu`` (measured, 2026-10-17).

Caps and outputs: ``--split-states K`` reads the split metrics (intents per split, M13) on the first K states per
split; ``--conditions-only`` reads no intent (for a variant that changes only how conditions are read);
``--latency-devices`` picks where latency is timed; ``--out`` (default DEFAULT_OUT, outside the repository) receives
one JSON per model with the metrics, the timing and the readings (catalogue readings as row indices only: the
paraphrases are private).

Metrics (``interpret_benchmark.gate_sets(seed, 4096, 1024)``, the sets of every earlier reading experiment)::

    kind          held-out wordings, non-empty when texts: value == true kind (bar 0.95); per kind; polarity pairs
                  (target_visible / target_not_visible, distance_below / distance_above) as finetune_pilot reports them
    argument      held-out wordings, numeric kinds: argument_for(true kind, read_number) (encoder-free),
                  argument_for(model kind, read_number) (the reading's argument; bar 0.98), the model's own number
    catalogue     the 714 private paraphrases over the 119 learned and command rows (catalogue_matching): top-1 row
                  (bar 0.90), top-3 rows when the backend scores, errors within the true row's group
    splits        intent -> skill on held-out wordings / families (the held-out families' goals) / sizes, over each
                  state's own catalogue (the benchmark's intent columns)
    M13           interpret.decide on the read goals vs goal_language.teacher, 32 belief rows per state
                  (interpret_benchmark.belief_rows, same keys): skill (replace) agreement with LM conditions + LM
                  intents (the benchmark's end to end) and with LM conditions + the typed ids (the 9.12 contract)
    latency       batch 1, system prompts warm: per goal (one condition, then one intent over the real catalogue)
                  and per version of four goals (the four conditions, then the four intents, one text per call);
                  cold = the first read of each system prompt; MPS and CPU (Ollama: GPU and CPU with num_gpu 0)

An empty ``when`` is ``always`` by rule (as every earlier reader); an output that did not parse reads as a numeric
kind without an argument, which never fires (``interpret.evaluate``), and an unparsed skill as idle. ``--n`` caps
every metric at n items (condition texts, catalogue paraphrases in a seeded order, states per split).

    PYTHONPATH=. JAX_PLATFORMS=cpu python -m experimental.neural_agent.jev.system_one.lm_reader \\
        --models qwen3-0.6b,qwen3-1.7b,qwen3-4b-ollama --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import time
import urllib.request
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import jax
import numpy as np

from experimental.neural_agent.contracts import SKILL_IDLE
from experimental.neural_agent.executive.goal_language import CONDITION_FAMILIES, split_wordings, teacher
from experimental.neural_agent.jev.system_one.catalogue_matching import (
    DEFAULT_CATALOGUE,
    DEFAULT_PARAPHRASES,
    MATCHED_KINDS,
    load_paraphrases,
    read_catalogue,
)
from experimental.neural_agent.jev.system_one.data import _numpy_rng
from experimental.neural_agent.jev.system_one.finetune_pilot import REPO_ROOT, POLARITY_PAIRS, _inside, polarity_confusion
from experimental.neural_agent.jev.system_one.interpret import (
    KINDS,
    NUMERIC_UNITS,
    GoalInterpretation,
    argument_for,
    decide,
    read_number,
)
from experimental.neural_agent.jev.system_one.interpret_benchmark import (
    ARGUMENT_BAR,
    DEFAULT_BELIEFS,
    INTENT_BAR,
    KIND_BAR,
    SPLITS,
    _goals,
    _offline_if_cached,
    belief_rows,
    gate_sets,
)
from experimental.neural_agent.jev.system_one.train_supervised import load_dataset

MODELS: dict[str, tuple[str, str]] = {
    "qwen3-0.6b": ("transformers", "Qwen/Qwen3-0.6B"),
    "qwen3-1.7b": ("transformers", "Qwen/Qwen3-1.7B"),
    "qwen3-4b-ollama": ("ollama", "qwen3:4b"),
    "qwen3-8b-ollama": ("ollama", "qwen3:8b"),
    "qwen3-14b-ollama": ("ollama", "qwen3:14b"),
}
"""CLI name -> (backend, model id or Ollama tag); a name + RESTATE_SUFFIX reads conditions in two passes."""
RESTATE_SUFFIX: str = "+restate"
RESTATE_TOKENS: int = 48
"""Most tokens of a pass-one restatement."""
RESTATE_FORCED: str = '{"restatement": "'
RESTATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"restatement": {"type": "string", "maxLength": 240}},
    "required": ["restatement"],
    "additionalProperties": False,
}
"""Pass one's output: the restatement as a JSON string (Ollama's qwen3:4b is a think-always build: free text from it
is a reasoning monologue whatever the think flag, so the grammar must start the answer)."""
EMIT_USER: str = 'Now reply with the predicate as one JSON object: {"kind": <one kind above>, "argument": <the number, or null>}.'
"""Pass two's request, after the model's restatement."""
KIND_MEANINGS: dict[str, str] = {
    "always": "no condition: the goal applies all the time; argument null",
    "on_damage": "the agent took damage within the last 40 ticks; argument null",
    "target_visible": "the agent can see its target right now; argument null",
    "target_not_visible": "the agent cannot see its target right now; argument null",
    "target_lost_for": "the target has been out of sight for more than N ticks; argument N in ticks",
    "distance_below": "the target is less than N blocks away; argument N in blocks",
    "distance_above": "the target is more than N blocks away; argument N in blocks",
    "self_health_below": "the agent's own health is below the fraction x of its maximum; argument x from 0 to 1",
    "target_health_below": "the target's health is below the fraction x of its maximum; argument x from 0 to 1",
}
"""The meaning of each kind told to the reader: goal_language's Predicates table in words (no sampling grids)."""
EXAMPLE_ARGUMENTS: dict[str, tuple[str, float]] = {"blocks": ("5 blocks", 5), "ticks": ("60 ticks", 60), "fraction": ("30%", 0.3)}
"""How {n} is written in a prompt example, per unit, and the argument it stands for."""
CONDITION_FORCED, CONDITION_CLOSE = '{"kind": "', '", "argument":'
SKILL_FORCED, SKILL_CLOSE = '{"skill_id": "', '"}'
NUMBER_PREFIX = re.compile(r" ?(?:n|nu|nul|null|null\}|-|-?\d+|-?\d+\.|-?\d+\.\d+|-?\d+\}|-?\d+\.\d+\})?")
"""Every prefix of NUMBER_DONE."""
NUMBER_DONE = re.compile(r" ?(?:null|-?\d+(?:\.\d+)?)\}")
"""The argument and the closing brace, after '"argument":'."""
NUMBER_CHARS = frozenset(" 0123456789.-}nul")
NUMBER_TOKENS: int = 12
PACK_TOKENS: int = 2048
"""Most packed tokens per forward pass (a pack holds at least one query)."""
PACK_QUERIES: int = 16
CACHED_PREFIXES: int = 4
"""System-prompt caches kept per transformers backend (least recently used dropped)."""
LOGIT_ROWS: int = 256
"""Hidden rows per vocabulary projection."""
LATENCY_GOALS: int = 20
"""Goals timed per device; consecutive fours make the versions."""
DEFAULT_OUT: Path = Path.home() / ".claude" / "jobs" / "31947bd7" / "tmp" / "lm_reader"
"""Where --out writes one JSON per model (metrics, timing, readings): outside the repository."""
OLLAMA_URL: str = "http://127.0.0.1:11434"
OLLAMA_OPTIONS: dict[str, Any] = {"temperature": 0, "seed": 0, "num_predict": 64, "num_ctx": 4096}
OLLAMA_GPU_LAYERS: int = 999
"""``num_gpu`` of GPU mode (every layer): it must be explicit, because Ollama keeps serving a model from a CPU-only
runner (``num_gpu`` 0, left by a CPU timing) to requests that do not name ``num_gpu`` (measured, 2026-10-17)."""
CONDITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": list(KINDS)}, "argument": {"type": ["number", "null"]}},
    "required": ["kind", "argument"],
    "additionalProperties": False,
}
_SENTINEL = "⁣USER⁣"


# ---------------------------------------------------------------------------------------------------- prompts


def training_wordings(seed: int) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """(training, held-out) paraphrases per family of ``gate_sets(seed, ...)``: the same keys and draw as
    ``data.goal_language_sets`` (checked against the held-out set by :func:`check_wordings`)."""

    sets_key, _ = jax.random.split(jax.random.fold_in(jax.random.key(seed), 2))
    return split_wordings(_numpy_rng(jax.random.split(sets_key, 5)[0]), 0.25)


def check_wordings(sets: Any, held: Mapping[str, Sequence[str]]) -> None:
    """Raise RuntimeError unless every held-out-wording goal's intent is a held-out paraphrase of its family."""

    for goal, meaning, _ in _goals(sets.heldout_wordings):
        if goal.intent not in held[meaning.intent_family]:
            raise RuntimeError(f"wording split mismatch: {goal.intent!r} is not held out for {meaning.intent_family}")


def prompt_examples(train: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, float | None]]:
    """kind -> (example text, argument): the kind's first training paraphrase, {n} written per EXAMPLE_ARGUMENTS."""

    examples = {}
    for family in CONDITION_FAMILIES:
        text, argument = train[family.family_id][0], None
        if family.numeric:
            words, argument = EXAMPLE_ARGUMENTS[NUMERIC_UNITS[family.family_id]]
            text = text.replace("{n}", words)
        examples[family.family_id] = (text, argument)
    return examples


def _condition_json(kind: str, argument: float | None) -> str:
    return json.dumps({"kind": kind, "argument": argument})


def condition_system_prompt(examples: Mapping[str, tuple[str, float | None]]) -> str:
    """The condition reader's system prompt (module docstring); the same bytes for the same examples."""

    kinds = "\n".join(f"- {kind}: {KIND_MEANINGS[kind]}" for kind in KINDS)
    shots = "\n".join(f"Condition: {json.dumps(examples[kind][0])} -> {_condition_json(kind, examples[kind][1])}" for kind in KINDS)
    return (
        "A planner writes goals for a game agent; each goal may carry a condition in the planner's own words. "
        "Translate one condition into the single predicate below that the agent's controller can evaluate.\n\n"
        f"Predicates (kind: meaning; argument):\n{kinds}\n\n"
        'Reply with one JSON object and nothing else: {"kind": <one kind above>, "argument": <the number, or null>}. '
        "Write a percentage as a fraction (30% is 0.3).\n\n"
        f"Examples:\n{shots}"
    )


def skill_system_prompt(rows: Sequence[tuple[Sequence[str], str]]) -> str:
    """The skill selector's system prompt for catalogue ``rows`` of (ids, description); the same bytes for the same rows."""

    lines = "\n".join(f"- {', '.join(ids)}: {description}" for ids, description in rows)
    return (
        "A planner writes goals for a game agent; each goal names what the agent should do as an intent in the "
        "planner's own words. Choose the agent's skill that carries out one intent.\n\n"
        f"Skills (ids: what the skill does; a line may give several ids of one skill):\n{lines}\n\n"
        'Reply with one JSON object and nothing else: {"skill_id": <one id above>}. Always choose one of the ids '
        "above; when no skill fits exactly, choose the closest."
    )


def condition_user(when: str) -> str:
    return f"Condition: {json.dumps(when)}"


def skill_user(intent: str) -> str:
    return f"Intent: {json.dumps(intent)}"


def restate_user(when: str) -> str:
    """Pass one of the two-pass variant: the condition and the request to restate it in the predicates' terms."""

    return (
        f"{condition_user(when)}\nBefore answering, restate this condition in one plain sentence in the terms of the "
        "predicate meanings above: what must be true of the agent or of its target, and, where there is a number, on "
        'which side of it the value lies or how long it lasts. Reply with one JSON object: {"restatement": <that sentence>}.'
    )


def _condition_turns(when: str, restated: str, restate: bool) -> list[tuple[str, str]]:
    """The conversation a condition is read in: one user turn, or (two-pass) pass one's request, the model's
    restatement and EMIT_USER."""

    if not restate:
        return [("user", condition_user(when))]
    return [("user", restate_user(when)), ("assistant", json.dumps({"restatement": restated})), ("user", EMIT_USER)]


def _restatement_of(text: str) -> str:
    """The sentence of a pass-one output {"restatement": ...}; a truncated output keeps what it holds."""

    try:
        value = json.loads(text).get("restatement", "")
    except (json.JSONDecodeError, AttributeError):
        match = re.search(r'"restatement"\s*:\s*"([^"\n]*)', text)
        value = match.group(1) if match else ""
    return " ".join(str(value).split())


def _batches(lengths: Sequence[int], extra: int) -> list[list[int]]:
    """Consecutive query indices per pack: at most PACK_QUERIES queries and PACK_TOKENS tokens (lengths + ``extra``
    each), at least one query."""

    packs: list[list[int]] = []
    current, total = [], 0
    for index, length in enumerate(lengths):
        if current and (len(current) == PACK_QUERIES or total + length + extra > PACK_TOKENS):
            packs.append(current)
            current, total = [], 0
        current.append(index)
        total += length + extra
    if current:
        packs.append(current)
    return packs


def skill_schema(ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"skill_id": {"type": "string", "enum": list(ids)}},
        "required": ["skill_id"],
        "additionalProperties": False,
    }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------- parser


@dataclass(frozen=True)
class Answer:
    """One reading. ``value`` is the kind or skill id (None when the output did not parse); ``argument`` the model's
    own number (conditions); ``ranked`` every allowed value best first and ``greedy`` the greedy grammar choice, when
    the backend scores values; ``repair`` "" (strict), "format", "content" (counted wrong) or "invalid"; ``restated``
    pass one's sentence in the two-pass variant."""

    value: str | None
    argument: float | None = None
    ranked: tuple[str, ...] = ()
    greedy: str | None = None
    repair: str = ""
    text: str = ""
    restated: str = ""


def _snap(value: Any, vocabulary: Sequence[str]) -> str | None:
    """``value`` matched to the vocabulary ignoring case, surrounding space, inner spaces and hyphens, else None."""

    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s\-]+", "_", value.strip().lower())
    return next((item for item in vocabulary if item == key), None)


def _objects(text: str) -> list[tuple[Any, bool]]:
    """(JSON value, whether it is the whole text): the whole text first, then each innermost {...} span in it."""

    found = []
    try:
        found.append((json.loads(text), True))
    except (json.JSONDecodeError, TypeError):
        pass
    for match in re.finditer(r"\{[^{}]*\}", text if isinstance(text, str) else ""):
        try:
            found.append((json.loads(match.group()), False))
        except json.JSONDecodeError:
            continue
    return found


def _finite_or_none(value: Any) -> tuple[bool, float | None]:
    if value is None:
        return True, None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        return False, None
    return True, float(value)


def parse_condition(text: str) -> Answer:
    """Strict parse of {"kind", "argument"}, then the format and content repairs (module docstring)."""

    candidates = _objects(text)
    for item, whole in candidates:
        if isinstance(item, dict) and set(item) == {"kind", "argument"} and item["kind"] in KINDS:
            ok, argument = _finite_or_none(item["argument"])
            if ok:
                return Answer(item["kind"], argument, repair="" if whole else "format", text=text)
    for item, _ in candidates:
        if isinstance(item, dict) and _snap(item.get("kind"), KINDS):
            ok, argument = _finite_or_none(item.get("argument"))
            return Answer(_snap(item["kind"], KINDS), argument if ok else None, repair="content", text=text)
    return Answer(None, repair="invalid", text=text)


def parse_skill(text: str, ids: Sequence[str]) -> Answer:
    """Strict parse of {"skill_id"} over ``ids``, then the format and content repairs (module docstring)."""

    candidates = _objects(text)
    for item, whole in candidates:
        if isinstance(item, dict) and set(item) == {"skill_id"} and item["skill_id"] in ids:
            return Answer(item["skill_id"], repair="" if whole else "format", text=text)
    for item, _ in candidates:
        if isinstance(item, dict) and _snap(item.get("skill_id"), ids):
            return Answer(_snap(item["skill_id"], ids), repair="content", text=text)
    return Answer(None, repair="invalid", text=text)


def usable(answer: Answer) -> bool:
    """Whether the reading has a value no content repair produced (strict or format-repaired)."""

    return answer.value is not None and answer.repair in ("", "format")


def correct(answer: Answer, truth: str) -> bool:
    """A reading counts when it is usable and its value is the truth."""

    return usable(answer) and answer.value == truth


def greedy_choice(candidates: Sequence[Sequence[int]], logprobs: Sequence[Sequence[float]]) -> int:
    """Index of the value greedy grammar-constrained decoding reaches: at each depth the best next token among the
    candidates still alive (equal prefixes share a context, so any alive candidate's log-prob of a token serves)."""

    alive, depth = list(range(len(candidates))), 0
    while len(alive) > 1:
        best = max(alive, key=lambda c: logprobs[c][depth] if depth < len(candidates[c]) else -np.inf)
        token = candidates[best][depth]
        alive = [c for c in alive if depth < len(candidates[c]) and candidates[c][depth] == token]
        depth += 1
    return alive[0]


# ---------------------------------------------------------------------------------------------------- backends


class Backend(Protocol):
    """A reader: ``conditions`` reads when texts, ``skills`` picks one of ``ids`` per intent; one Answer per text."""

    label: str
    where: str
    parameters: str
    quantisation: str

    def conditions(self, system: str, texts: Sequence[str]) -> list[Answer]: ...

    def skills(self, system: str, ids: Sequence[str], intents: Sequence[str]) -> list[Answer]: ...

    def forget(self) -> None:
        """Drop client-side prompt caches, so the next read of a system prompt is cold."""

    def close(self) -> None: ...


def _tril(size: int) -> Any:
    import torch

    return torch.ones(size, size, dtype=torch.bool).tril()


class TransformersBackend:
    """A Hugging Face causal LM on one device, read by exact constrained scoring (module docstring).

    Args:
        model_id: A model id of the local Hugging Face cache, for example ``"Qwen/Qwen3-0.6B"``.
        device: ``"mps"`` or ``"cpu"``; float32 on both.
        restate: Read conditions in two passes (module docstring, Two-pass variant).
    """

    def __init__(self, model_id: str, device: str, restate: bool = False) -> None:
        import torch
        import transformers

        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        self._torch, self.device, self.restate = torch, device, restate
        dtype = torch.float32
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
        self.model.to(device).eval().requires_grad_(False)
        name = str(dtype).removeprefix("torch.")
        self.label = f"{model_id} (transformers {transformers.__version__}, torch {torch.__version__}){' two-pass' if restate else ''}"
        self.where = f"{device}, {name}"
        self.parameters = f"{sum(item.numel() for item in self.model.parameters()):,}"
        self.quantisation = f"none ({name} weights)"
        pieces = self.tokenizer.batch_decode([[index] for index in range(len(self.tokenizer))])
        self._number_ids = [index for index, piece in enumerate(pieces) if piece and set(piece) <= NUMBER_CHARS]
        self._number_pieces = [pieces[index] for index in self._number_ids]
        self._number_tensor = torch.tensor(self._number_ids)
        self._stops = {self.tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "<|endoftext|>")}
        self._caches: OrderedDict[str, tuple[Any, list[int]]] = OrderedDict()

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _render(self, system: str, turns: Sequence[tuple[str, str]]) -> str:
        messages = [{"role": "system", "content": system}, *({"role": role, "content": text} for role, text in turns)]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _prefix(self, system: str) -> tuple[Any, list[int]]:
        """(KV cache, token ids) of the chat text before the first user message, computed once per system prompt."""

        if system in self._caches:
            self._caches.move_to_end(system)
            return self._caches[system]
        rendered = self._render(system, [("user", _SENTINEL)])
        ids = self._encode(rendered[: rendered.index(_SENTINEL)])
        with self._torch.no_grad():
            cache = self.model.model(input_ids=self._torch.tensor([ids], device=self.device), use_cache=True).past_key_values
        self._caches[system] = (cache, ids)
        while len(self._caches) > CACHED_PREFIXES:
            self._caches.popitem(last=False)
        return cache, ids

    def _queries(self, system: str, conversations: Sequence[Sequence[tuple[str, str]]], forced: str, check: tuple | None = None):
        """(cache, prefix ids, per conversation the token ids after the prefix of its rendering + ``forced``), checking
        that the prefix and (``check`` = (text, ids)) the first candidate tokenise the same inside the whole text."""

        cache, prefix = self._prefix(system)
        queries = []
        for turns in conversations:
            rendered = self._render(system, turns) + forced
            ids = self._encode(rendered)
            if ids[: len(prefix)] != prefix or (check is not None and self._encode(rendered + check[0]) != ids + check[1]):
                raise RuntimeError(f"tokenisation of the prompt is not prefix-stable for {turns[-1][1]!r}")
            queries.append(ids[len(prefix) :])
        return cache, prefix, queries

    def _choose(self, system: str, conversations: Sequence, forced: str, values: Sequence[str], close: str, number: bool) -> list[tuple]:
        """Per conversation: (scores of ``values``, per-token log-probs, greedy index, generated number text or None)."""

        start = len(self._encode(forced))
        candidates = [self._encode(forced + value + close)[start:] for value in values]
        cache, prefix, queries = self._queries(system, conversations, forced, (values[0] + close, candidates[0]))
        results = []
        for pack in _batches([len(query) for query in queries], sum(len(item) for item in candidates)):
            results += self._pack(cache, len(prefix), [queries[index] for index in pack], candidates, number)
        return results

    def _logprobs(self, hidden: Any, rows: Sequence[int], targets: Sequence[int]) -> np.ndarray:
        """log p(targets[i] | hidden[rows[i]]), float64[len(rows)], LOGIT_ROWS rows per vocabulary projection."""

        torch, out = self._torch, []
        for start in range(0, len(rows), LOGIT_ROWS):
            index = torch.tensor(rows[start : start + LOGIT_ROWS], device=self.device)
            logits = self.model.lm_head(hidden[index]).float()
            wanted = torch.tensor(targets[start : start + LOGIT_ROWS], device=self.device)[:, None]
            out.append((logits.gather(1, wanted)[:, 0] - torch.logsumexp(logits, dim=1)).cpu())
        return torch.cat(out).double().numpy()

    def _forward(self, cache: Any, tokens: Sequence[int], positions: Sequence[int], allow: Any) -> Any:
        """Hidden states of ``tokens`` appended to ``cache``; ``allow`` (bool[len(tokens), cache + len(tokens)]) becomes an
        additive mask (0 or the dtype's minimum), which eager and SDPA attention both read."""

        torch = self._torch
        dtype = self.model.dtype
        mask = torch.zeros(allow.shape, dtype=dtype).masked_fill(~allow, torch.finfo(dtype).min)
        with torch.no_grad():
            return self.model.model(
                input_ids=torch.tensor([list(tokens)], device=self.device),
                attention_mask=mask[None, None].to(self.device),
                position_ids=torch.tensor([list(positions)], device=self.device),
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state[0]

    def _pack(self, cache: Any, size: int, queries: Sequence[list[int]], candidates: Sequence[list[int]], number: bool) -> list[tuple]:
        """Score every candidate of every query in one packed forward over the cached prefix of ``size`` tokens
        (module docstring); then, with ``number``, generate each query's number after its best candidate."""

        torch = self._torch
        tokens, positions, spans = [], [], []
        for query in queries:
            first = len(tokens)
            tokens += query
            positions += range(size, size + len(query))
            starts = []
            for candidate in candidates:
                starts.append(len(tokens))
                tokens += candidate
                positions += range(size + len(query), size + len(query) + len(candidate))
            spans.append((first, len(query), starts))
        allow = torch.zeros(len(tokens), size + len(tokens), dtype=torch.bool)
        allow[:, :size] = True
        for first, length, starts in spans:
            allow[first : first + length, size + first : size + first + length] = _tril(length)
            for begin, candidate in zip(starts, candidates):
                allow[begin : begin + len(candidate), size + first : size + first + length] = True
                allow[begin : begin + len(candidate), size + begin : size + begin + len(candidate)] = _tril(len(candidate))
        try:
            hidden = self._forward(cache, tokens, positions, allow)
            rows, targets = [], []
            for first, length, starts in spans:
                for begin, candidate in zip(starts, candidates):
                    rows += [first + length - 1, *range(begin, begin + len(candidate) - 1)]
                    targets += candidate
            flat, offset, results = self._logprobs(hidden, rows, targets), 0, []
            for _ in spans:
                per = []
                for candidate in candidates:
                    per.append(flat[offset : offset + len(candidate)])
                    offset += len(candidate)
                results.append([np.asarray([item.sum() for item in per]), per, greedy_choice(candidates, per), None])
            if number:
                contexts, places, rows = [], [], []
                for (first, length, starts), result in zip(spans, results):
                    chosen = int(np.argmax(result[0]))
                    begin, width = starts[chosen], len(candidates[chosen])
                    contexts.append([*range(size + first, size + first + length), *range(size + begin, size + begin + width)])
                    places.append(size + length + width)
                    rows.append(begin + width - 1)
                states = hidden[torch.tensor(rows, device=self.device)]
                pieces = dict(zip(self._number_ids, self._number_pieces))
                generated = self._generate(cache, size, contexts, places, states, self._number_choice, NUMBER_TOKENS)
                for result, output in zip(results, generated):
                    result[3] = "".join(pieces[token] for token in output)
        finally:
            cache.crop(size - cache.get_seq_length())
        return [tuple(item) for item in results]

    def _number_choice(self, row: Any, output: Sequence[int]) -> tuple[int, bool]:
        """The best number-vocabulary token that keeps the output a prefix of NUMBER_DONE, and whether it completes it."""

        pieces = dict(zip(self._number_ids, self._number_pieces))
        text = "".join(pieces[token] for token in output)
        for index in self._torch.argsort(row[self._number_tensor], descending=True).tolist():
            if NUMBER_PREFIX.fullmatch(text + self._number_pieces[index]):
                return self._number_ids[index], NUMBER_DONE.fullmatch(text + self._number_pieces[index]) is not None
        raise RuntimeError(f"no token continues the number grammar after {text!r}")

    def _string_choice(self, row: Any, output: Sequence[int]) -> tuple[int, bool]:
        """The model's best token; done when it closes the JSON string (a quote), breaks the line or ends the turn."""

        token = int(self._torch.argmax(row))
        piece = self.tokenizer.decode([token])
        return token, token in self._stops or '"' in piece or "\n" in piece

    def _generate(self, cache: Any, size: int, contexts: list[list[int]], places: Sequence[int], states: Any, choose: Any, limit: int) -> list[list[int]]:
        """Greedy decoding of every query at once, one packed step per token: query q sees the prefix, the cache columns
        ``contexts[q]`` and its own new tokens; ``states`` predict each query's first token; ``choose(logits, output)``
        returns (token, done). At most ``limit`` tokens per query; the cache keeps the fed tokens (the caller crops)."""

        torch = self._torch
        outputs: list[list[int]] = [[] for _ in contexts]
        active = list(range(len(contexts)))
        for step in range(limit):
            logits = self.model.lm_head(states).float().cpu()
            feed = []
            for row, query in enumerate(active):
                token, done = choose(logits[row], outputs[query])
                outputs[query].append(token)
                if not done:
                    feed.append(query)
            if not feed or step == limit - 1:
                break
            length = cache.get_seq_length()
            allow = torch.zeros(len(feed), length + len(feed), dtype=torch.bool)
            allow[:, :size] = True
            for row, query in enumerate(feed):
                contexts[query].append(length + row)
                allow[row, contexts[query]] = True
            states = self._forward(cache, [outputs[query][-1] for query in feed], [places[query] + step for query in feed], allow)
            active = feed
        return outputs

    def _extend(self, cache: Any, size: int, queries: Sequence[list[int]], choose: Any, limit: int) -> list[list[int]]:
        """Greedy continuation of each query (token ids after the cached prefix of ``size`` tokens), packed."""

        torch = self._torch
        tokens, positions, spans = [], [], []
        for query in queries:
            spans.append((len(tokens), len(query)))
            tokens += query
            positions += range(size, size + len(query))
        allow = torch.zeros(len(tokens), size + len(tokens), dtype=torch.bool)
        allow[:, :size] = True
        for first, length in spans:
            allow[first : first + length, size + first : size + first + length] = _tril(length)
        try:
            hidden = self._forward(cache, tokens, positions, allow)
            states = hidden[torch.tensor([first + length - 1 for first, length in spans], device=self.device)]
            contexts = [list(range(size + first, size + first + length)) for first, length in spans]
            places = [size + length for _, length in spans]
            return self._generate(cache, size, contexts, places, states, choose, limit)
        finally:
            cache.crop(size - cache.get_seq_length())

    def _restatements(self, system: str, texts: Sequence[str]) -> list[str]:
        """Pass one of the two-pass variant: after the forced '{"restatement": "', greedy text until the string closes."""

        conversations = [[("user", restate_user(text))] for text in texts]
        cache, prefix, queries = self._queries(system, conversations, RESTATE_FORCED)
        out = []
        for pack in _batches([len(query) for query in queries], RESTATE_TOKENS):
            for output in self._extend(cache, len(prefix), [queries[index] for index in pack], self._string_choice, RESTATE_TOKENS):
                text = self.tokenizer.decode([token for token in output if token not in self._stops])
                out.append(_restatement_of(RESTATE_FORCED + text.split('"')[0].split("\n")[0] + '"}'))
        return out

    def conditions(self, system: str, texts: Sequence[str]) -> list[Answer]:
        restated = self._restatements(system, texts) if self.restate else [""] * len(texts)
        turns = [_condition_turns(text, line, self.restate) for text, line in zip(texts, restated)]
        answers = []
        chosen = self._choose(system, turns, CONDITION_FORCED, KINDS, CONDITION_CLOSE, True)
        for (scores, _, greedy, number), line in zip(chosen, restated):
            order = np.argsort(-scores, kind="stable")
            parsed = parse_condition(CONDITION_FORCED + KINDS[order[0]] + CONDITION_CLOSE + number)
            answers.append(
                Answer(parsed.value, parsed.argument, tuple(KINDS[i] for i in order), KINDS[greedy], parsed.repair, parsed.text, line)
            )
        return answers

    def skills(self, system: str, ids: Sequence[str], intents: Sequence[str]) -> list[Answer]:
        conversations = [[("user", skill_user(text))] for text in intents]
        answers = []
        for scores, _, greedy, _ in self._choose(system, conversations, SKILL_FORCED, ids, SKILL_CLOSE, False):
            order = np.argsort(-scores, kind="stable")
            parsed = parse_skill(SKILL_FORCED + ids[order[0]] + SKILL_CLOSE, ids)
            answers.append(Answer(parsed.value, None, tuple(ids[i] for i in order), ids[greedy], parsed.repair, parsed.text))
        return answers

    def forget(self) -> None:
        self._caches.clear()

    def close(self) -> None:
        self._caches.clear()
        del self.model
        if self.device == "mps":
            self._torch.mps.empty_cache()


class OllamaBackend:
    """A local Ollama model over ``/api/chat`` with ``format`` = the JSON schema (module docstring).

    Args:
        model: The Ollama tag, for example ``"qwen3:4b"``.
        cpu: Run on the CPU (``num_gpu`` 0) instead of the GPU.
        restate: Read conditions in two passes (module docstring, Two-pass variant).
    """

    def __init__(self, model: str, cpu: bool = False, restate: bool = False, base_url: str = OLLAMA_URL) -> None:
        self.model, self.base_url, self.restate, self.cpu, self._resident = model, base_url, restate, cpu, False
        self.options = {**OLLAMA_OPTIONS, "num_gpu": 0 if cpu else OLLAMA_GPU_LAYERS}
        details = self._post("/api/show", {"model": model}).get("details", {})
        with urllib.request.urlopen(f"{base_url}/api/version", timeout=10) as response:
            version = json.loads(response.read())["version"]
        self.label = f"{model} (Ollama {version}, llama.cpp grammar from the JSON schema){' two-pass' if restate else ''}"
        self.where = "cpu, num_gpu 0" if cpu else "gpu (Metal)"
        self.parameters = str(details.get("parameter_size", "?"))
        self.quantisation = str(details.get("quantization_level", "?"))

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())

    def _chat(self, system: str, turns: Sequence[tuple[str, str]], schema: Mapping[str, Any] | None, **options: Any) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *({"role": role, "content": text} for role, text in turns)],
            "think": False,
            "stream": False,
            "keep_alive": "30m",
            "options": {**self.options, **options},
        }
        if schema is not None:
            body["format"] = schema
        content = self._post("/api/chat", body)["message"]["content"]
        if not self._resident:
            self._check_residency()
        return content

    def _check_residency(self) -> None:
        """Fail closed unless /api/ps shows the model where ``where`` says (VRAM for the GPU, none for the CPU); the
        bytes in VRAM are appended to ``where``."""

        with urllib.request.urlopen(f"{self.base_url}/api/ps", timeout=10) as response:
            loaded = {item["name"]: item for item in json.loads(response.read())["models"]}
        item = loaded.get(self.model)
        if item is None:
            raise RuntimeError(f"{self.model} is not loaded after a request")
        vram, size = int(item.get("size_vram", 0)), int(item.get("size", 0))
        if self.cpu != (vram == 0):
            raise RuntimeError(f"{self.model} holds {vram} bytes in VRAM; expected {'none' if self.cpu else 'its layers'}")
        self.where += f", {vram / 1e9:.2f} of {size / 1e9:.2f} GB in VRAM"
        self._resident = True

    def conditions(self, system: str, texts: Sequence[str]) -> list[Answer]:
        answers = []
        for text in texts:
            line = ""
            if self.restate:
                line = _restatement_of(self._chat(system, [("user", restate_user(text))], RESTATE_SCHEMA, num_predict=2 * RESTATE_TOKENS))
            answer = parse_condition(self._chat(system, _condition_turns(text, line, self.restate), CONDITION_SCHEMA))
            answers.append(replace(answer, restated=line))
        return answers

    def skills(self, system: str, ids: Sequence[str], intents: Sequence[str]) -> list[Answer]:
        schema = skill_schema(ids)
        return [parse_skill(self._chat(system, [("user", skill_user(text))], schema), ids) for text in intents]

    def forget(self) -> None:
        return None

    def close(self) -> None:
        return None


def make_backend(name: str, device: str) -> Backend:
    """The backend of model ``name`` (a MODELS key, optionally with RESTATE_SUFFIX) on ``device`` ("gpu" or "cpu")."""

    base, restate = name.removesuffix(RESTATE_SUFFIX), name.endswith(RESTATE_SUFFIX)
    kind, ident = MODELS[base]
    if kind == "transformers":
        import torch

        return TransformersBackend(ident, "mps" if device != "cpu" and torch.backends.mps.is_available() else "cpu", restate)
    return OllamaBackend(ident, cpu=device == "cpu", restate=restate)


# ---------------------------------------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class EvalData:
    """What a run reads: the held-out sets and their belief rows, the condition prompt, the real catalogue."""

    sets: Any
    rows: Mapping[str, list]
    examples: Mapping[str, tuple[str, float | None]]
    condition_prompt: str
    catalogue: Sequence[tuple[tuple[str, ...], str, str]] | None
    """(ids, description, group) per learned or command row, or None when the private files are absent."""
    paraphrases: Sequence[tuple[str, int]] | None
    """(text, true row index into ``catalogue``) in a seeded order."""


def _rate(hits: Sequence[bool]) -> float:
    return float(np.mean(hits)) if len(hits) else float("nan")


def _states(sets: Any, split: str, n: int | None) -> tuple:
    return getattr(sets, split)[: n if n is not None else None]


def _catalogue_prompt(rows: Sequence[tuple[Sequence[str], str]]) -> tuple[str, list[str]]:
    return skill_system_prompt(rows), [skill_id for ids, _ in rows for skill_id in ids]


def _sample_catalogue(sample: Any) -> tuple[str, list[str]]:
    return _catalogue_prompt([((skill_id,), text) for skill_id, text in sample.catalogue.items()])


def evaluate(
    backend: Backend, data: EvalData, n: int | None = None, split_states: int | None = None, conditions_only: bool = False
) -> dict[str, Any]:
    """Every metric of the module docstring for one backend. ``split_states`` caps the states per split of the split
    metrics (intents and M13) beside ``n``; ``conditions_only`` skips every intent reading (the catalogue, the split
    intents, M13 with LM intents), for a variant that changes only how conditions are read."""

    sets, clock = data.sets, time.perf_counter()
    cap = min([item for item in (n, split_states) if item is not None], default=None)
    metrics: dict[str, Any] = {"conditions_only": conditions_only, "split_cap": cap}
    worded = [(goal.when, meaning.predicate) for goal, meaning, _ in _goals(sets.heldout_wordings) if goal.when.strip()]
    worded = worded[: n if n is not None else None]
    texts = sorted(
        {text for text, _ in worded}
        | {goal.when for split in SPLITS for sample in _states(sets, split, cap) for goal in sample.section.goals if goal.when.strip()}
    )
    read = dict(zip(texts, backend.conditions(data.condition_prompt, texts)))
    metrics["seconds_conditions"], metrics["count_conditions"] = time.perf_counter() - clock, len(texts)
    answers = [read[text] for text, _ in worded]
    truth = [predicate.kind for _, predicate in worded]
    metrics["kind"] = _rate([correct(answer, kind) for answer, kind in zip(answers, truth)])
    metrics["kind_greedy"] = _rate([answer.greedy == kind for answer, kind in zip(answers, truth) if answer.greedy is not None])
    metrics["kind_count"] = len(truth)
    labels = [answer.value if usable(answer) else "unread" for answer in answers]
    metrics["per_kind"] = {kind: (_rate([a == kind for a, t in zip(labels, truth) if t == kind]), truth.count(kind)) for kind in KINDS}
    metrics["polarity"] = {f"{a}/{b}": polarity_confusion(truth, labels, (a, b)) for a, b in POLARITY_PAIRS}
    metrics["confusion"] = {kind: dict(Counter(label for label, true in zip(labels, truth) if true == kind)) for kind in KINDS}
    numeric = [(text, predicate, read[text]) for text, predicate in worded if predicate.kind in NUMERIC_UNITS]
    metrics["argument_true_kind"] = _rate([argument_for(p.kind, read_number(t)) == p.argument for t, p, _ in numeric])
    metrics["argument_model_kind"] = _rate(
        [usable(a) and argument_for(a.value, read_number(t)) == p.argument for t, p, a in numeric]
    )
    metrics["argument_model_number"] = _rate([a.argument is not None and abs(a.argument - p.argument) < 1e-9 for _, p, a in numeric])
    metrics["argument_count"] = len(numeric)
    metrics["repairs"] = dict(Counter(answer.repair or "strict" for answer in read.values()))
    dumped: dict[str, Any] = {
        "conditions": {text: [a.value, a.argument, a.repair, a.greedy, a.restated] for text, a in read.items()},
        "splits": {},
    }
    if data.catalogue is not None and not conditions_only:
        clock = time.perf_counter()
        system, ids = _catalogue_prompt([(row_ids, description) for row_ids, description, _ in data.catalogue])
        row_of = {skill_id: index for index, (row_ids, _, _) in enumerate(data.catalogue) for skill_id in row_ids}
        queries = list(data.paraphrases[: n if n is not None else None])
        picks = backend.skills(system, ids, [text for text, _ in queries])
        metrics["seconds_catalogue"], metrics["count_catalogue"] = time.perf_counter() - clock, len(queries)
        hits, top3, greedy, within = [], [], [], 0
        dumped["catalogue"] = [
            [row, row_of.get(answer.value, -1) if usable(answer) else -1, row_of.get(answer.greedy, -1), answer.repair]
            for answer, (_, row) in zip(picks, queries)
        ]
        for answer, (_, row) in zip(picks, queries):
            chosen = row_of.get(answer.value) if usable(answer) else None
            hits.append(chosen == row)
            if chosen is not None and chosen != row and data.catalogue[chosen][2] == data.catalogue[row][2]:
                within += 1
            if answer.ranked:
                top3.append(row in list(dict.fromkeys(row_of[item] for item in answer.ranked))[:3])
            if answer.greedy is not None:
                greedy.append(row_of[answer.greedy] == row)
        metrics["catalogue_top1"], metrics["catalogue_top3"], metrics["catalogue_greedy"] = _rate(hits), _rate(top3), _rate(greedy)
        metrics["catalogue_errors"], metrics["catalogue_within"] = int(len(hits) - sum(hits)), within
        metrics["catalogue_repairs"] = dict(Counter(answer.repair or "strict" for answer in picks))
        metrics["catalogue_rows"], metrics["catalogue_ids"] = len(data.catalogue), len(ids)
    clock, count, split_repairs = time.perf_counter(), 0, Counter()
    for split in SPLITS:
        states = _states(sets, split, cap)
        skills: dict[int, list[Answer]] = {index: [] for index in range(len(states))}
        groups: dict[str, list[int]] = {}
        for index, sample in enumerate(states):
            groups.setdefault(json.dumps(list(sample.catalogue.items())), []).append(index)
        for members in () if conditions_only else groups.values():
            system, ids = _sample_catalogue(states[members[0]])
            intents = [goal.intent for index in members for goal in states[index].section.goals]
            picked = iter(backend.skills(system, ids, intents))
            count += len(intents)
            for index in members:
                skills[index] = [next(picked) for _ in states[index].section.goals]
                split_repairs.update(answer.repair or "strict" for answer in skills[index])
        dumped["splits"][split] = [
            [index, slot, meaning.skill_id, answer.value, answer.repair]
            for index, sample in enumerate(states)
            for slot, (answer, meaning) in enumerate(zip(skills[index], sample.meanings))
        ]
        held = set(sets.heldout_family_ids)
        if not conditions_only:
            hits = [
                correct(answer, meaning.skill_id)
                for index, sample in enumerate(states)
                for answer, meaning in zip(skills[index], sample.meanings)
                if split != "heldout_families" or meaning.intent_family in held
            ]
            metrics[f"intent_{split}"], metrics[f"intent_{split}_count"] = _rate(hits), len(hits)
        for mode in ("typed",) if conditions_only else ("lm", "typed"):
            skill_hits, replace_hits = [], []
            for index, (sample, sample_rows) in enumerate(zip(states, data.rows[split])):
                readings = [
                    _interpretation(goal, read, None if mode == "typed" else skills[index][slot])
                    for slot, goal in enumerate(sample.section.goals)
                ]
                for values, flags, current in sample_rows:
                    expected = teacher(sample.section, sample.meanings, values, flags, current)
                    got = decide(sample.section, readings, values, flags, current)
                    skill_hits.append(got[0] == expected[0])
                    replace_hits.append(got[1] == expected[1])
            metrics[f"m13_{split}_{mode}"], metrics[f"m13_{split}_{mode}_replace"] = _rate(skill_hits), _rate(replace_hits)
            metrics[f"m13_{split}_rows"] = len(skill_hits)
    metrics["seconds_splits"], metrics["count_splits"] = time.perf_counter() - clock, count
    metrics["split_repairs"] = dict(split_repairs)
    metrics["readings"] = dumped
    return metrics


def _interpretation(goal: Any, read: Mapping[str, Answer], skill: Answer | None) -> GoalInterpretation:
    """One goal read: the kind from the LM ("" -> always; unread -> a numeric kind without argument, which never fires),
    the argument from read_number, the skill from the LM (unread -> idle) or, when ``skill`` is None, the typed id."""

    kind, argument = "always", None
    if goal.when.strip():
        answer = read[goal.when]
        kind = answer.value if usable(answer) else "target_lost_for"
        argument = argument_for(kind, read_number(goal.when)) if usable(answer) else None
    if skill is None:
        skill_id = goal.skill_id
    else:
        skill_id = skill.value if usable(skill) else SKILL_IDLE
    return GoalInterpretation(kind, argument, skill_id, (), 1.0)


def latency(backend: Backend, data: EvalData, goals: int = LATENCY_GOALS) -> dict[str, Any]:
    """Batch-1 seconds, system prompts warm: per goal (its condition, then its intent) and per version of four goals
    (the four conditions, then the four intents, one text per call); cold = the first read of each system prompt."""

    pairs = [(goal.when, goal.intent, sample) for goal, _, sample in _goals(data.sets.heldout_wordings) if goal.when.strip()][:goals]
    if data.catalogue is not None:
        system, ids = _catalogue_prompt([(row_ids, description) for row_ids, description, _ in data.catalogue])
        catalogue = [(system, ids)] * len(pairs)
        pairs = [(when, text, None) for (when, _, _), (text, _) in zip(pairs, data.paraphrases)]
    else:
        catalogue = [_sample_catalogue(sample) for _, _, sample in pairs]
    backend.forget()
    backend.skills("Choose a skill.", ["idle"], ["wait"])  # warm-up: model loaded, kernels compiled
    started = os.getloadavg()
    clock = time.perf_counter()
    backend.conditions(data.condition_prompt, [pairs[0][0]])
    cold_condition = time.perf_counter() - clock
    clock = time.perf_counter()
    backend.skills(*catalogue[0], [pairs[0][1]])
    cold_skill = time.perf_counter() - clock
    per_goal, versions = [], []
    for (when, intent, _), (system, ids) in zip(pairs, catalogue):
        clock = time.perf_counter()
        backend.conditions(data.condition_prompt, [when])
        backend.skills(system, ids, [intent])
        per_goal.append(time.perf_counter() - clock)
    for start in range(0, len(pairs) - 3, 4):
        clock = time.perf_counter()
        for when, _, _ in pairs[start : start + 4]:
            backend.conditions(data.condition_prompt, [when])
        for (_, intent, _), (system, ids) in zip(pairs[start : start + 4], catalogue[start : start + 4]):
            backend.skills(system, ids, [intent])
        versions.append(time.perf_counter() - clock)
    return {
        "where": backend.where,
        "goal_median": statistics.median(per_goal),
        "goal_p90": float(np.percentile(per_goal, 90)),
        "version_median": statistics.median(versions) if versions else float("nan"),
        "version_max": max(versions) if versions else float("nan"),
        "cold_condition": cold_condition,
        "cold_skill": cold_skill,
        "goals": len(per_goal),
        "versions": len(versions),
        "load": (started, os.getloadavg()),
    }


# ---------------------------------------------------------------------------------------------------- report


def _mark(value: float, bar: float) -> str:
    return "PASS" if value >= bar else "FAIL"


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}"


def _load(values: Sequence[float]) -> str:
    return " ".join(f"{value:.1f}" for value in values)


def _pair(table: Mapping[str, Any], pair: tuple[str, str]) -> str:
    parts = []
    for kind, flipped in (pair, pair[::-1]):
        row = table[kind]
        other = ", ".join(f"{name} {count}" for name, count in row["other"].items()) or "none"
        parts.append(f"{kind} {row['as'][kind]}/{row['count']} right, read as {flipped} {row['as'][flipped]}, other: {other}")
    return "; ".join(parts)


def report(name: str, backend: Backend, metrics: Mapping[str, Any], timing: Mapping[str, Mapping[str, Any]], sets: Any) -> None:
    """Print one model's block."""

    print(f"\nmodel {name}: {backend.label}, evaluated on {backend.where}; parameters {backend.parameters}; quantisation {backend.quantisation}")
    kind = metrics["kind"]
    greedy = f", greedy grammar decoding {metrics['kind_greedy']:.4f}" if metrics["kind_greedy"] == metrics["kind_greedy"] else ""
    print(f"  kind, held-out wordings ({metrics['kind_count']} non-empty when texts): {kind:.4f} [>= {KIND_BAR}: {_mark(kind, KIND_BAR)}]{greedy}")
    print("  per kind: " + ", ".join(f"{k} {rate:.4f} ({count})" for k, (rate, count) in metrics["per_kind"].items()))
    for a, b in POLARITY_PAIRS:
        print(f"  polarity: {_pair(metrics['polarity'][f'{a}/{b}'], (a, b))}")
    off = [(true, read, count) for true, row in metrics["confusion"].items() for read, count in row.items() if read != true]
    print(
        "  confusion off the diagonal (true -> read: count): "
        + (", ".join(f"{true} -> {read}: {count}" for true, read, count in sorted(off, key=lambda item: -item[2])) or "none")
    )
    value = metrics["argument_model_kind"]
    print(
        f"  argument, held-out wordings ({metrics['argument_count']} numeric): read_number on the model's kind {value:.4f} "
        f"[>= {ARGUMENT_BAR}: {_mark(value, ARGUMENT_BAR)}], on the true kind {metrics['argument_true_kind']:.4f}; "
        f"the model's own number {metrics['argument_model_number']:.4f}"
    )
    print(f"  condition outputs ({metrics['count_conditions']} distinct texts): {metrics['repairs']}")
    if "catalogue_top1" in metrics:
        top1, top3 = metrics["catalogue_top1"], metrics["catalogue_top3"]
        extra = f", top-3 {top3:.4f}, greedy grammar decoding {metrics['catalogue_greedy']:.4f}" if top3 == top3 else ", top-3 n/a (no scores)"
        print(
            f"  intent -> skill, real catalogue ({metrics['count_catalogue']} paraphrases, {metrics['catalogue_rows']} rows, "
            f"{metrics['catalogue_ids']} ids, no \"none\" option): top-1 {top1:.4f} [>= {INTENT_BAR}: {_mark(top1, INTENT_BAR)}]"
            f"{extra}; errors {metrics['catalogue_errors']}, within the true group {metrics['catalogue_within']}; "
            f"outputs {metrics['catalogue_repairs']}"
        )
    held = ", ".join(sets.heldout_family_ids)
    states = f"the first {metrics['split_cap']} states per split" if metrics["split_cap"] is not None else "every state"
    if metrics["conditions_only"]:
        print("  intents: not read (--conditions-only: this variant changes only how conditions are read)")
    else:
        print(
            "  intent -> skill, generator splits: "
            + ", ".join(f"{split.replace('heldout_', '')} {metrics[f'intent_{split}']:.4f} ({metrics[f'intent_{split}_count']})" for split in SPLITS)
            + f" [families: the {held} goals; {states}]; outputs {metrics['split_repairs']}"
        )
    for mode, title in (("lm", "LM conditions + LM intents"), ("typed", "LM conditions + typed ids")):
        if f"m13_heldout_wordings_{mode}" not in metrics:
            continue
        print(
            f"  M13 agreement, {title}, skill (replace): "
            + ", ".join(
                f"{split.replace('heldout_', '')} {metrics[f'm13_{split}_{mode}']:.4f} ({metrics[f'm13_{split}_{mode}_replace']:.4f})"
                for split in SPLITS
            )
            + f" [{metrics['m13_heldout_wordings_rows']} rows per split, {states}]"
        )
    rates = [f"conditions {metrics['seconds_conditions'] * 1000 / max(metrics['count_conditions'], 1):.3f} s per 1,000"]
    if metrics["count_splits"]:
        rates.append(f"split intents {metrics['seconds_splits'] * 1000 / max(metrics['count_splits'], 1):.3f} s per 1,000")
    if "seconds_catalogue" in metrics:
        rates.insert(1, f"catalogue intents {metrics['seconds_catalogue'] * 1000 / max(metrics['count_catalogue'], 1):.3f} s per 1,000")
    print("  throughput (evaluation, batched where the backend allows): " + ", ".join(rates))
    for device, item in timing.items():
        if "error" in item:
            print(f"  latency {device}: not measured ({item['error']})")
            continue
        print(
            f"  latency {item['where']}, batch 1 ({item['goals']} goals, {item['versions']} versions; load {_load(item['load'][0])} -> "
            f"{_load(item['load'][1])}): per goal median {_ms(item['goal_median'])} ms (p90 {_ms(item['goal_p90'])}), per "
            f"4-goal version median {_ms(item['version_median'])} ms (max {_ms(item['version_max'])}); cold first read: "
            f"condition {_ms(item['cold_condition'])} ms, catalogue intent {_ms(item['cold_skill'])} ms"
        )


def load_data(seed: int, catalogue: Path, paraphrases: Path, beliefs: Path, rows_per_state: int) -> EvalData:
    """The held-out sets, belief rows, condition prompt and (when present) the real catalogue of a run."""

    sets, split_keys = gate_sets(seed, 0, 1024)
    train, held = training_wordings(seed)
    check_wordings(sets, held)
    dataset = load_dataset(beliefs)
    rows = {split: belief_rows(getattr(sets, split), dataset, key, rows_per_state) for split, key in zip(SPLITS, split_keys)}
    examples = prompt_examples(train)
    prompt = condition_system_prompt(examples)
    parsed = read_catalogue(catalogue)
    texts = load_paraphrases(paraphrases, parsed) if parsed is not None else None
    if texts is None:
        return EvalData(sets, rows, examples, prompt, None, None)
    candidates = [index for index, row in enumerate(parsed) if row.kind in MATCHED_KINDS]
    position = {index: number for number, index in enumerate(candidates)}
    rows_out = [(parsed[index].ids, parsed[index].description, parsed[index].group) for index in candidates]
    queries = [(text, position[index]) for index in sorted(texts) for text in texts[index]]
    order = np.random.default_rng(seed).permutation(len(queries))
    return EvalData(sets, rows, examples, prompt, rows_out, [queries[i] for i in order])


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate each model, print its block, the summary table and the LoRA step's status; return 0."""

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.system_one.lm_reader",
        description="Decoder-LM reader of the interpret stage (ledger P6) on the earlier reading experiments' held-out sets.",
    )
    parser.add_argument(
        "--models", default="qwen3-0.6b",
        help=f"comma-separated, of {', '.join(MODELS)}; append {RESTATE_SUFFIX} to read conditions in two passes",
    )
    parser.add_argument("--seed", type=int, default=0, help="gate_sets seed (wording and family splits, belief rows)")
    parser.add_argument("--n", type=int, default=None, help="cap every metric at n items (default: the full sets)")
    parser.add_argument("--lora", action="store_true", help="run the LoRA step (needs peft) when zero-shot misses a bar")
    parser.add_argument("--latency-goals", type=int, default=LATENCY_GOALS, help="goals timed per device")
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--paraphrases", type=Path, default=DEFAULT_PARAPHRASES)
    parser.add_argument("--beliefs", type=Path, default=DEFAULT_BELIEFS)
    parser.add_argument("--rows-per-state", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for one JSON per model, outside the repository")
    parser.add_argument("--split-states", type=int, default=None, help="cap the states per split of the intent and M13 metrics")
    parser.add_argument("--conditions-only", action="store_true", help="read conditions only (no catalogue, split intents)")
    parser.add_argument("--latency-devices", default="gpu,cpu", help="where latency is timed: gpu (MPS or Metal), cpu, or both")
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in names if name.removesuffix(RESTATE_SUFFIX) not in MODELS]
    if not names or unknown:
        parser.error(f"--models must name some of {', '.join(MODELS)} (optionally +restate); unknown {unknown}")
    if any(item is not None and item < 1 for item in (args.n, args.split_states)) or args.latency_goals < 4 or args.rows_per_state < 1:
        parser.error("--n, --split-states and --rows-per-state must be positive and --latency-goals at least 4")
    devices = [item.strip() for item in args.latency_devices.split(",") if item.strip()]
    if not devices or set(devices) - {"gpu", "cpu"}:
        parser.error(f"--latency-devices must name gpu, cpu or both, got {args.latency_devices!r}")
    if _inside(args.out, REPO_ROOT):
        parser.error(f"--out must lie outside the repository {REPO_ROOT}, got {args.out}")
    bases = [MODELS[name.removesuffix(RESTATE_SUFFIX)] for name in names]
    _offline_if_cached([f"torch:{ident}" for kind, ident in bases if kind == "transformers"])
    data = load_data(args.seed, args.catalogue, args.paraphrases, args.beliefs, args.rows_per_state)
    print(
        f"lm_reader: seed {args.seed}, n {args.n or 'full'}; held-out sets of interpret_benchmark.gate_sets({args.seed}, 4096, 1024) "
        f"(held-out families {', '.join(data.sets.heldout_family_ids)}), {args.rows_per_state} belief rows per state from "
        f"{args.beliefs}; load average {_load(os.getloadavg())} ({os.cpu_count()} cores)"
    )
    print(
        f"condition prompt: {len(data.condition_prompt)} characters, sha256 {_digest(data.condition_prompt)}; examples (first "
        "training paraphrase per kind): " + "; ".join(f"{kind} {text!r}" for kind, (text, _) in data.examples.items())
    )
    if data.catalogue is None:
        print(f"catalogue: skipped (no private catalogue or paraphrases at {args.catalogue}, {args.paraphrases})")
    else:
        system, ids = _catalogue_prompt([(row_ids, description) for row_ids, description, _ in data.catalogue])
        print(
            f"catalogue prompt: {len(data.catalogue)} rows, {len(ids)} ids, {len(data.paraphrases)} paraphrases; {len(system)} "
            f"characters, sha256 {_digest(system)}"
        )
    results = []
    for name in names:
        started = time.perf_counter()
        backend = make_backend(name, "gpu")
        metrics = evaluate(backend, data, args.n, args.split_states, args.conditions_only)
        timing = {}
        try:
            if "gpu" in devices:
                timing["accelerator"] = latency(backend, data, args.latency_goals)
        finally:
            backend.close()
        if "cpu" in devices:
            try:
                cpu = make_backend(name, "cpu")
                try:
                    timing["cpu"] = latency(cpu, data, args.latency_goals)
                finally:
                    cpu.close()
            except Exception as error:  # noqa: BLE001 - a CPU timing failure is reported, not fatal
                timing["cpu"] = {"error": f"{type(error).__name__}: {error}"}
        report(name, backend, metrics, timing, data.sets)
        args.out.mkdir(parents=True, exist_ok=True)
        tags = [f"-n{args.n}" if args.n else "", f"-s{args.split_states}" if args.split_states else "", "-conditions" if args.conditions_only else ""]
        dump = args.out / f"{name}-seed{args.seed}{''.join(tags)}.json"
        record = {"model": name, "label": backend.label, "where": backend.where, "parameters": backend.parameters,
                  "quantisation": backend.quantisation, "metrics": metrics, "timing": timing}
        dump.write_text(json.dumps(record, default=str), encoding="utf-8")
        print(f"  [{time.perf_counter() - started:.0f} s; readings {dump}]", flush=True)
        results.append((name, backend, metrics, timing))
    print(
        "\n| model | parameters | quantisation | kind (0.95) | visible / not visible / below / above | argument (0.98): reading / "
        "model's number | intent top-1, 119 rows (0.90) | top-3 | M13 wordings / families / sizes, LM intents | M13, typed ids | "
        f"goal ms, {' / '.join(devices)} | version ms, {' / '.join(devices)} |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, backend, metrics, timing in results:
        polarity = " / ".join(
            f"{metrics['per_kind'][kind][0]:.2f}" for kind in ("target_visible", "target_not_visible", "distance_below", "distance_above")
        )
        m13 = {
            mode: " / ".join(f"{metrics[f'm13_{split}_{mode}']:.4f}" for split in SPLITS) if f"m13_heldout_wordings_{mode}" in metrics else "n/a"
            for mode in ("lm", "typed")
        }
        clocks = [item for item in timing.values()]
        goal = " / ".join(_ms(item["goal_median"]) if "goal_median" in item else "n/a" for item in clocks)
        version = " / ".join(_ms(item["version_median"]) if "version_median" in item else "n/a" for item in clocks)
        top1 = f"{metrics['catalogue_top1']:.4f}" if "catalogue_top1" in metrics else "n/a"
        top3 = f"{metrics['catalogue_top3']:.4f}" if "catalogue_top1" in metrics else "n/a"
        print(
            f"| {name} | {backend.parameters} | {backend.quantisation} | {metrics['kind']:.4f} | {polarity} | "
            f"{metrics['argument_model_kind']:.4f} / {metrics['argument_model_number']:.4f} | {top1} | {top3} | {m13['lm']} | "
            f"{m13['typed']} | {goal} | {version} |"
        )
    missed = [
        name for name, _, metrics, _ in results
        if metrics["kind"] < KIND_BAR or metrics.get("catalogue_top1", 1.0) < INTENT_BAR or metrics["argument_model_kind"] < ARGUMENT_BAR
    ]
    peft = importlib.util.find_spec("peft") is not None
    status = "not needed" if not missed else "not run: peft is not installed" if not peft else "not run: not built here"
    print(
        f"\nLoRA step ({'requested' if args.lora else 'not requested'}; zero-shot misses a bar for "
        f"{', '.join(missed) or 'no model'}; peft {'installed' if peft else 'not installed'}): {status}"
    )
    print(f"load average at the end {_load(os.getloadavg())}")
    return 0


__all__ = [
    "Answer",
    "Backend",
    "CONDITION_SCHEMA",
    "DEFAULT_OUT",
    "EvalData",
    "KIND_MEANINGS",
    "MODELS",
    "NUMBER_DONE",
    "NUMBER_PREFIX",
    "OllamaBackend",
    "RESTATE_SUFFIX",
    "TransformersBackend",
    "check_wordings",
    "condition_system_prompt",
    "condition_user",
    "correct",
    "usable",
    "evaluate",
    "greedy_choice",
    "latency",
    "load_data",
    "main",
    "make_backend",
    "parse_condition",
    "parse_skill",
    "prompt_examples",
    "restate_user",
    "skill_schema",
    "skill_system_prompt",
    "skill_user",
    "training_wordings",
]

if __name__ == "__main__":
    raise SystemExit(main())
