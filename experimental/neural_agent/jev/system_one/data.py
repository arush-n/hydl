"""Rendering and supervised rows for J-S1 (``docs/SYSTEM_ONE.md`` section 8, FORMULAS (M2), (M5)).

Two timescales (FORMULAS (M2)). The trunk reads text only on the slow path:

    x_E = render_executive(skill_ids, descriptions, E_t)     once per executive version
    h_E = trunk.encode([x_E])[0]                             cached until E_t changes
    e_j = trunk.encode([desc_j])[0]                          once per catalogue (catalogue_embeddings)

``render_executive`` writes the stable part only: the catalogue with descriptions (the
skill id when no description is given), the goals and the facts, in the line formats of
``jev/lm/prompting.py``. It leaves out the belief lines and the current skill (they reach
the heads as numbers on the fast path), and the version and issue tick (bookkeeping, not
content: equal goals and facts give equal text, so a re-issued E_t keeps its encoding).

A supervised row is ``(h_E of the row's E_t, u, m, current) -> (y_skill, y_replace, y_params)``:

    executive_texts[executive_index]   x_E of the row's E_t; each distinct E_t is rendered and encoded once
    u         = belief_inputs(features, feature_known, latent)       float32[84]
    m         = j1_allowed_skills(E_t arrays)                          bool[K]
    current   = onehot(current_skill)                                  float32[K]
    y_skill   = J0's choice: the recorded target_skill (source 1), or
                scripted_target_skill on a variant's arrays (source 2)
    y_replace = the recorded target_replace, or y_skill != current_skill for a variant
    y_params  = params of the E_t slot that J0 takes for y_skill (highest priority valid
                slot naming it, lowest slot on a tie, network._matching_slot), 0 for none

Source 2, :func:`executive_variants`, templated from the base E_t and the catalogue
(this module's choices): each base goal is dropped with probability 0.25; 0 to 2
goals naming a non-idle catalogue skill are added (at most MAX_GOALS in all); the
goal order is shuffled; the priorities are a permutation of the goals' priorities
(an added goal draws one of the base's priority values, 1.0 when the base has no
goal); each goal carries a random subset of the base's constraint labels; each
horizon is a base horizon times 0.5, 1 or 2. Version, tick, budget and facts are
the base's. Held-out split by content: a variant is held out when
``blake2b(state.to_json()) mod 5 == 0`` and it differs from the base, so a
held-out E_t never appears among the training variants.

Goal reading (SYSTEM_ONE.md 8.9, FORMULAS (M9) to (M11)), independent of the real skill library:

* :func:`goal_reading_states` generates E_t over placeholder catalogues (this module's choices):
  K uniform in ``catalogue_sizes`` (3..8), ids ``idle, skill_a, skill_b, ...``, idle described by
  the library and every other skill by a distinct line drawn from a fixed pool of
  ``DESCRIPTION_POOL_SIZE`` = 24 generated one-liners "<Action> <place> <manner>." (distinct
  action-place pairs, drawn once with seed 0), so ids carry no identity and the model must read
  descriptions, while held-out states and larger catalogues use descriptions seen in training (an
  unseen description is gate G3b's question, not G3's); 1..min(MAX_GOALS, K - 1) goals naming distinct non-idle
  skills; priority U(0, 2] (2 - U[0, 2)); 0..P params U(-1, 1) rounded to 0.01; horizon U[2, 30] s
  rounded to 0.1; a deadline tick U{0..2047} with probability 0.5; each vocabulary label joins a
  goal with probability 0.17, so about a third of the states carry an ``on_damage`` goal; version
  = position + 1, tick 0, the default budget, no facts. Deterministic in the key.
* :func:`slot_inputs` writes the slot inputs of (M9), float32[MAX_GOALS, 2D + 4 + P]:
  ``[e_skill_k ; priority_k ; 1 ; params_k ; horizon_k / 30 ; has_deadline_k ; c_k]`` for goal k,
  c_k the mean of the label embeddings (0 for none), and zeros in an empty slot.
* :func:`generic_rule_labels` is the teacher (M10), J0's rule with the evader replaced by the
  ``on_damage`` label: with the damage window open (ticks_since_damage known and
  < 40 ticks, the feature times TICKS_SATURATION, as J0) and a valid goal labelled ``on_damage``,
  the highest-priority such goal; else the highest-priority valid goal (lowest slot on a tie, in
  float32 as J0 reads the lowered priorities); else idle. y_replace = y_skill != current_skill.
* :func:`goal_reading_rows` pairs each state with ``rows_per_state`` belief rows (features, known
  flags, latent and current skill) drawn uniformly from a recorded dataset, labelled by (M10).
* :func:`goal_reading_split` (M11): states with K > 5 form the held-out catalogue-size split;
  of the others, a state is held out when ``blake2b(x_E) mod 5 == 0`` (its rendered text, so equal
  content lands on one side), and the rest train.

Goal language (SYSTEM_ONE.md 8.10 and 8.11, FORMULAS (M12), (M13)); states, wordings, catalogues and the
teacher come from ``executive/goal_language.py``:

* :func:`slot_inputs` with ``text_embeddings`` writes the slot inputs of (M12), float32[MAX_GOALS, 4D + 4 + P]:
  the (M9) columns, then the embeddings of goal k's ``intent`` and ``when`` (0 for an empty text or a goal
  without words, so a plain ExecutiveState reads with zero text columns). ``mask_skill`` zeroes e_skill_k:
  the text-only reading mode.
* :func:`encode_goal_texts` encodes the distinct non-empty texts not yet cached in one batched trunk call
  and caches them per string (the run's dict in training, the instance's dict on the slow path).
* :func:`goal_language_sets` (this module's choices where 8.11 is silent): one wording split
  (``split_wordings``, 0.25 held out per family) and one intent-family split (``split_families``, 2 held
  out) per key; each set draws from its own key, so the held-out sets do not depend on the training count.
  Training: the real catalogue with probability REAL_CATALOGUE_RATE = 1/8, else ``synthetic_catalogue``
  with K uniform in {3, 4, 5} over the training families; goals from the training families and wordings.
  Held-out wordings: the same catalogues, texts only from the held-out wordings. Held-out families: K
  uniform in {3, 4, 5} over all families, redrawn until the catalogue holds a held-out family's
  description, goals redrawn until a slot uses a held-out family, training wordings. Held-out sizes: K
  uniform in {6, 7, 8} over the training families, training families and wordings. Version = position + 1,
  tick 0, ``sample_state``'s defaults otherwise (note rate 0.25, hint rate 0: no typed label hints).
* :func:`goal_language_rows` pairs each sample with ``rows_per_state`` belief rows drawn uniformly from a
  recorded dataset (as :func:`goal_reading_rows`), labelled by ``goal_language.teacher`` (M13). Units: the
  teacher reads physical units and the belief publishes ``ticks_since_damage`` and
  ``ticks_since_target_seen`` as counter / TICKS_SATURATION, so those two columns are multiplied back first,
  as J0 does (in float32, exact); the current skill is the catalogue id at the recorded index. y_params = 0
  (the generator's goals carry no params). ``allowed_mask``: typed mode, idle and the goals' skills
  (``j1_allowed_skills``); text-only mode, every catalogue skill. The rows carry the embeddings of their
  texts in ``text_embeddings``.

Rows of several catalogues concatenate with ``allowed_mask`` and ``current_onehot`` padded to the
widest K (False and 0); each row keeps its E_t and catalogue in ``executives`` for the slot inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.contracts import (
    MAX_GOALS,
    PARAM_WIDTH,
    SKILL_IDLE,
    TICKS_SATURATION,
    ComputeBudget,
    ExecutiveArrays,
    ExecutiveState,
    GoalSpec,
    executive_arrays,
    feature_index,
)
from experimental.neural_agent.executive.goal_language import (
    INTENT_FAMILIES,
    REAL_CATALOGUE,
    SlotMeaning,
    sample_state,
    split_families,
    split_wordings,
    synthetic_catalogue,
    teacher,
)
from experimental.neural_agent.jev.lm.prompting import _fact_lines, _goal_lines, validate_skill_ids
from experimental.neural_agent.jev.network import IDLE_SKILL_INDEX, _matching_slot, j1_allowed_skills
from experimental.neural_agent.jev.scripted import EVADE_DAMAGE_WINDOW_TICKS, _evader_index
from experimental.neural_agent.jev.system_one.heads import HORIZON_SCALE_SECONDS, SLOT_SCALARS, belief_inputs
from experimental.neural_agent.jev.system_one.trunk import TrunkEncoder
from experimental.neural_agent.jev.train_j1 import scripted_target_skill, validate_dataset
from experimental.neural_agent.shared_state import ExecutiveSection

DROP_PROBABILITY: float = 0.25
MAX_ADDED_GOALS: int = 2
HORIZON_SCALES: tuple[float, ...] = (0.5, 1.0, 2.0)
HELDOUT_MODULUS: int = 5
DAMAGE_CONSTRAINT: str = "on_damage"
CONSTRAINT_VOCABULARY: tuple[str, ...] = ("on_damage", "hold_position", "conserve", "quiet")
CATALOGUE_SIZES: tuple[int, ...] = tuple(range(3, 9))
TRAIN_MAX_CATALOGUE_SIZE: int = 5
# Each vocabulary label joins a goal with this probability: with K uniform in 3..8 and
# 1..min(4, K - 1) goals, 0.33 of the states carry an on_damage goal in expectation.
LABEL_PROBABILITY: float = 0.17
DEADLINE_PROBABILITY: float = 0.5
DEADLINE_TICKS: int = 2048
PRIORITY_MAX: float = 2.0
HORIZON_RANGE_SECONDS: tuple[float, float] = (2.0, 30.0)
_LETTERS: str = "abcdefghijklmnopqrstuvwxyz"
_ACTIONS: tuple[str, ...] = (
    "approach", "guard", "circle", "scout", "block", "follow", "mark", "patrol", "climb", "search", "watch", "clear",
)
_PLACES: tuple[str, ...] = (
    "the beacon", "the north gate", "the supply crate", "the ridge", "the river crossing", "the watchtower",
    "the camp", "the bridge", "the cave mouth", "the old well", "the market square", "the hilltop",
)
_MANNERS: tuple[str, ...] = (
    "quickly", "carefully", "in a wide arc", "while keeping cover", "at walking pace", "from a distance",
)
_ROW_ARRAYS: tuple[str, ...] = ("u", "allowed_mask", "current_onehot", "y_skill", "y_replace", "y_params")
# Placeholder descriptions (at most len(_ACTIONS) * len(_PLACES) = 144, one per action-place pair).
DESCRIPTION_POOL_SIZE: int = 24
LANGUAGE_MODES: tuple[str, ...] = ("typed", "text-only")
LANGUAGE_TRAIN_SIZES: tuple[int, ...] = (3, 4, 5)
LANGUAGE_EVAL_SIZES: tuple[int, ...] = (6, 7, 8)
REAL_CATALOGUE_RATE: float = 1.0 / 8.0
# The belief columns published as counter / TICKS_SATURATION; the teacher (M13) reads them in ticks.
_TICK_FEATURES: tuple[str, ...] = ("ticks_since_damage", "ticks_since_target_seen")


def _description_pool(size: int) -> tuple[str, ...]:
    """Return ``size`` one-line descriptions "<Action> <place> <manner>." with distinct action-place pairs (seed 0)."""

    rng = np.random.default_rng(0)
    pairs = rng.choice(len(_ACTIONS) * len(_PLACES), size=size, replace=False)
    return tuple(
        f"{_ACTIONS[pair // len(_PLACES)].capitalize()} {_PLACES[pair % len(_PLACES)]} "
        f"{_MANNERS[int(rng.integers(len(_MANNERS)))]}."
        for pair in pairs
    )


_DESCRIPTION_POOL: tuple[str, ...] = _description_pool(DESCRIPTION_POOL_SIZE)
_EXECUTIVE_FIELDS: dict[str, str] = {
    "skill_index": "exec_skill_index",
    "priority": "exec_priority",
    "params": "exec_params",
    "goal_valid": "exec_goal_valid",
    "budget": "exec_budget",
    "version": "exec_version",
}


class GoalReadingState(NamedTuple):
    """One E_t over its own catalogue: the state, the skill ids (idle first) and one description per skill.

    ``state`` is an ExecutiveState, or for goal language the ExecutiveSection whose goals carry their words.
    ``mask_skill`` marks the text-only reading mode (SYSTEM_ONE.md 8.11): its slot inputs leave e_skill out.
    """

    state: ExecutiveState | ExecutiveSection
    skill_ids: tuple[str, ...]
    descriptions: tuple[str, ...]
    mask_skill: bool = False


class SupervisedRows(NamedTuple):
    """Stage-A rows, N of them over V distinct executives: executive_texts tuple[str] (V),
    executive_index int32[N] (row -> text), u float32[N, 84], allowed_mask bool[N, K],
    current_onehot float32[N, K], y_skill int32[N], y_replace bool[N], y_params float32[N, P],
    executives tuple[GoalReadingState] (V, the E_t and catalogue of each text; K is the widest
    catalogue, narrower ones padded), and text_embeddings (text -> float32[D] for every goal text of
    the executives, :func:`goal_language_rows`; None for rows without goal text)."""

    executive_texts: tuple[str, ...]
    executive_index: np.ndarray
    u: np.ndarray
    allowed_mask: np.ndarray
    current_onehot: np.ndarray
    y_skill: np.ndarray
    y_replace: np.ndarray
    y_params: np.ndarray
    executives: tuple[GoalReadingState, ...]
    text_embeddings: Mapping[str, np.ndarray] | None = None


class ExecutiveVariant(NamedTuple):
    """One templated E_t: the state, its arrays lowered for one lane, and its split side."""

    state: ExecutiveState
    arrays: ExecutiveArrays
    heldout: bool


def skill_descriptions_of(skill_ids: Sequence[str], skill_descriptions: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return one description per skill: ``skill_descriptions``, or the library's descriptions (ids for unknown skills) when None.

    Raises:
        ValueError: If the catalogue is invalid, or the descriptions are not one non-empty string per skill.
    """

    catalogue = validate_skill_ids(skill_ids)
    if skill_descriptions is None:
        # The library's own descriptions (skill_library/descriptions.py, 2026-10-17); a skill
        # without one is described by its id, as before.
        from experimental.neural_agent.skill_library.descriptions import describe

        return describe(catalogue)
    if isinstance(skill_descriptions, (str, bytes)):
        raise ValueError("skill_descriptions must be a sequence of strings, not one string")
    texts = tuple(skill_descriptions)
    if len(texts) != len(catalogue) or not all(isinstance(text, str) and text for text in texts):
        raise ValueError(f"skill_descriptions must be {len(catalogue)} non-empty strings, one per skill")
    return texts


def catalogue_embeddings(
    trunk: TrunkEncoder, skill_ids: Sequence[str], skill_descriptions: Sequence[str] | None = None
) -> np.ndarray:
    """Return e_j = trunk.encode([desc_j])[0] for every skill, float32[K, D] (slow path, once per catalogue)."""

    return np.asarray(trunk.encode(list(skill_descriptions_of(skill_ids, skill_descriptions))), dtype=np.float32)


def render_executive(
    skill_ids: Sequence[str], skill_descriptions: Sequence[str] | None, executive: ExecutiveState
) -> str:
    """Return x_E, the stable text of one E_t (module docstring): skills with descriptions, goals, facts.

    Raises:
        TypeError: If ``executive`` is not an ExecutiveState.
        ValueError: If the catalogue or the descriptions are invalid, or a goal names a skill
            outside the catalogue.
    """

    if not isinstance(executive, ExecutiveState):
        raise TypeError("executive must be an ExecutiveState")
    catalogue = validate_skill_ids(skill_ids)
    texts = skill_descriptions_of(catalogue, skill_descriptions)
    for slot, goal in enumerate(executive.goals):
        if goal.skill_id not in catalogue:
            raise ValueError(f"executive goal {slot} names unknown skill {goal.skill_id!r}")
    lines = [
        "Skills:",
        *(f"  {skill}: {text}" for skill, text in zip(catalogue, texts)),
        "Goals (priority is the executive's relative weight):",
        *_goal_lines(executive),
        "Facts:",
        *_fact_lines(executive),
    ]
    return "\n".join(lines)


def _rows(
    dataset: Mapping[str, Any],
    text: str,
    executive: GoalReadingState,
    lowered: ExecutiveArrays,
    y_skill: np.ndarray,
    y_replace: np.ndarray,
) -> SupervisedRows:
    """Assemble the rows of ``dataset`` under one E_t (``lowered`` has N lanes, ``text`` is its x_E)."""

    catalogue = executive.skill_ids

    current = np.asarray(dataset["current_skill"], dtype=np.int32)
    rows = current.shape[0]
    skill = jnp.asarray(y_skill, dtype=jnp.int32)
    slot = np.asarray(_matching_slot(lowered, skill))
    params = np.asarray(lowered.params)[np.arange(rows), np.maximum(slot, 0)]
    return SupervisedRows(
        executive_texts=(text,),
        executive_index=np.zeros((rows,), dtype=np.int32),
        u=np.asarray(belief_inputs(dataset["features"], dataset["feature_known"], dataset["latent"])),
        allowed_mask=np.asarray(j1_allowed_skills(lowered, num_skills=len(catalogue))),
        current_onehot=np.eye(len(catalogue), dtype=np.float32)[current],
        y_skill=np.asarray(y_skill, dtype=np.int32),
        y_replace=np.asarray(y_replace, dtype=np.bool_),
        y_params=np.where((slot >= 0)[:, None], params, 0.0).astype(np.float32),
        executives=(executive,),
    )


def _checked(dataset: Mapping[str, Any], skill_ids: Sequence[str]) -> tuple[tuple[str, ...], int]:
    """Validate the dataset and the catalogue together; return (catalogue, N)."""

    validate_dataset(dataset)
    catalogue = validate_skill_ids(skill_ids)
    if int(np.asarray(dataset["num_skills"])) != len(catalogue):
        raise ValueError(
            f"the dataset scores {int(np.asarray(dataset['num_skills']))} skills; the catalogue has {len(catalogue)}"
        )
    return catalogue, int(np.shape(dataset["features"])[0])


def rows_from_dataset(
    dataset: Mapping[str, Any],
    executive: ExecutiveState,
    skill_ids: Sequence[str],
    skill_descriptions: Sequence[str] | None = None,
) -> SupervisedRows:
    """Return the recorded decision rows (source 1) under ``executive``: one x_E for all rows.

    Args:
        dataset: A ``train_j1`` dataset (``validate_dataset``) recorded under ``executive``.
        executive: The E_t of the recording (``executive.json``).
        skill_ids: The catalogue the recording used.
        skill_descriptions: One description per skill, or None for the skill ids.

    Raises:
        ValueError: If the dataset is invalid, its skill count differs from the catalogue,
            or its executive arrays are not ``executive`` lowered onto this catalogue.
    """

    catalogue, rows = _checked(dataset, skill_ids)
    lowered = executive_arrays(executive, catalogue, rows)
    for field, column in _EXECUTIVE_FIELDS.items():
        if not np.array_equal(np.asarray(getattr(lowered, field)), np.asarray(dataset[column])):
            raise ValueError(f"dataset {column} differs from the executive state lowered onto this catalogue")
    text = render_executive(catalogue, skill_descriptions, executive)
    item = GoalReadingState(executive, catalogue, skill_descriptions_of(catalogue, skill_descriptions))
    return _rows(dataset, text, item, lowered, dataset["target_skill"], dataset["target_replace"])


def _heldout(state: ExecutiveState, base: ExecutiveState) -> bool:
    """Return the split side of a variant, from its canonical JSON (module docstring)."""

    text = state.to_json()
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % HELDOUT_MODULUS == 0 and text != base.to_json()


def executive_variants(
    base: ExecutiveState, skill_ids: Sequence[str], key: jax.Array, count: int
) -> tuple[ExecutiveVariant, ...]:
    """Return ``count`` templated variants of ``base`` (module docstring), deterministic in ``key``.

    Raises:
        KeyError: If a goal of ``base`` names a skill outside the catalogue.
        ValueError: If ``count`` is negative or the catalogue is invalid.
    """

    catalogue = validate_skill_ids(skill_ids)
    executive_arrays(base, catalogue, 1)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError(f"count must be a non-negative integer, got {count!r}")
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max)))
    named = [skill for skill in catalogue if skill != SKILL_IDLE]
    priorities = sorted({goal.priority for goal in base.goals}) or [GoalSpec.priority]
    labels = sorted({label for goal in base.goals for label in goal.constraints})
    horizons = [goal.horizon_seconds for goal in base.goals] or [GoalSpec.horizon_seconds]
    variants: list[ExecutiveVariant] = []
    for _ in range(int(count)):
        goals = [goal for goal in base.goals if rng.random() >= DROP_PROBABILITY]
        for _ in range(int(rng.integers(0, MAX_ADDED_GOALS + 1))):
            if len(goals) < MAX_GOALS and named:
                goals.append(
                    GoalSpec(
                        skill_id=str(rng.choice(named)),
                        priority=float(rng.choice(priorities)),
                        horizon_seconds=float(rng.choice(horizons)),
                    )
                )
        goals = [goals[index] for index in rng.permutation(len(goals))]
        shuffled = rng.permutation([goal.priority for goal in goals])
        state = dataclasses.replace(
            base,
            goals=tuple(
                dataclasses.replace(
                    goal,
                    priority=float(shuffled[slot]),
                    constraints=tuple(label for label in labels if rng.random() < 0.5),
                    horizon_seconds=goal.horizon_seconds * float(rng.choice(HORIZON_SCALES)),
                )
                for slot, goal in enumerate(goals)
            ),
        )
        variants.append(ExecutiveVariant(state, executive_arrays(state, catalogue, 1), _heldout(state, base)))
    return tuple(variants)


def variant_rows(
    dataset: Mapping[str, Any],
    variants: Sequence[ExecutiveVariant],
    skill_ids: Sequence[str],
    skill_descriptions: Sequence[str] | None = None,
) -> SupervisedRows:
    """Return every belief row of ``dataset`` under every variant, labelled by J0's rule.

    Each variant contributes one x_E. ``y_skill = scripted_target_skill`` on the variant's
    arrays for the row's belief (the evader is J0's evader of this catalogue),
    ``y_replace = y_skill != current``.

    Raises:
        ValueError: If ``variants`` is empty, the dataset is invalid or its skill count
            differs from the catalogue.
    """

    if not variants:
        raise ValueError("variant_rows needs at least one variant")
    catalogue, rows = _checked(dataset, skill_ids)
    evader = _evader_index(catalogue)
    descriptions = skill_descriptions_of(catalogue, skill_descriptions)
    parts = []
    for variant in variants:
        lowered = jax.tree.map(lambda leaf: np.repeat(np.asarray(leaf), rows, axis=0), variant.arrays)
        y_skill = scripted_target_skill(
            dataset["features"],
            dataset["feature_known"],
            lowered.skill_index,
            lowered.priority,
            lowered.goal_valid,
            evader_index=evader,
        )
        y_replace = y_skill != np.asarray(dataset["current_skill"])
        text = render_executive(catalogue, descriptions, variant.state)
        item = GoalReadingState(variant.state, catalogue, descriptions)
        parts.append(_rows(dataset, text, item, lowered, y_skill, y_replace))
    return concat_rows(parts)


def concat_rows(parts: Sequence[SupervisedRows]) -> SupervisedRows:
    """Concatenate one or more row sets; each part's executive indices are offset past the earlier texts,
    ``allowed_mask`` and ``current_onehot`` are padded to the widest catalogue (False and 0), and the parts'
    ``text_embeddings`` merge (None when no part has any).

    Raises:
        ValueError: If ``parts`` is empty.
    """

    if not parts:
        raise ValueError("concat_rows needs at least one row set")
    offsets = np.cumsum([0] + [len(part.executive_texts) for part in parts[:-1]])
    texts = tuple(text for part in parts for text in part.executive_texts)
    index = np.concatenate([part.executive_index + offset for part, offset in zip(parts, offsets)]).astype(np.int32)
    width = max(part.allowed_mask.shape[1] for part in parts)

    def padded(part: SupervisedRows, name: str) -> np.ndarray:
        array = getattr(part, name)
        if name in ("allowed_mask", "current_onehot"):
            array = np.pad(array, ((0, 0), (0, width - array.shape[1])))
        return array

    arrays = {name: np.concatenate([padded(part, name) for part in parts], axis=0) for name in _ROW_ARRAYS}
    executives = tuple(item for part in parts for item in part.executives)
    tables = [part.text_embeddings for part in parts if part.text_embeddings is not None]
    text_embeddings = {text: vector for table in tables for text, vector in table.items()} if tables else None
    return SupervisedRows(
        executive_texts=texts, executive_index=index, **arrays, executives=executives, text_embeddings=text_embeddings
    )


def _numpy_rng(key: jax.Array) -> np.random.Generator:
    """Return a numpy generator seeded from a JAX key (as :func:`executive_variants` seeds its own)."""

    return np.random.default_rng(int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max)))


def goal_texts(goals: Iterable[Any]) -> set[str]:
    """Return the distinct non-empty ``intent`` and ``when`` texts of ``goals`` (a GoalSpec carries none)."""

    return {text for goal in goals for text in (getattr(goal, "intent", ""), getattr(goal, "when", "")) if text}


def encode_goal_texts(trunk: TrunkEncoder, texts: Iterable[str], cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add e = trunk.encode([text])[0] for every non-empty text of ``texts`` not in ``cache``, all in one
    batched trunk call (none when every text is cached), and return ``cache`` (FORMULAS (M12): cached per
    distinct text). An empty text is never encoded: :func:`slot_inputs` writes zeros for it."""

    new = sorted({text for text in texts if text} - set(cache))
    if new:
        cache.update(zip(new, np.asarray(trunk.encode(new), dtype=np.float32)))
    return cache


def slot_inputs(
    executive: ExecutiveState | ExecutiveSection,
    skill_ids: Sequence[str],
    skill_embeddings: Any,
    label_embeddings: Mapping[str, Any],
    *,
    text_embeddings: Mapping[str, Any] | None = None,
    mask_skill: bool = False,
) -> np.ndarray:
    """Return the slot inputs x_k of (M9), float32[MAX_GOALS, 2D + 4 + P], or with ``text_embeddings``
    those of (M12), float32[MAX_GOALS, 4D + 4 + P] (module docstring and ``heads.py``): goal k of
    ``executive`` fills row k, an empty slot is all zeros.

    Args:
        executive: The E_t: an ExecutiveState, or an ExecutiveSection whose goals carry ``intent`` and ``when``.
        skill_ids: The catalogue that ``skill_embeddings`` describes.
        skill_embeddings: float32[K, D], e_j of each catalogue skill (``catalogue_embeddings``).
        label_embeddings: label -> float32[D], one trunk embedding per constraint label; every
            label of ``executive`` must be present.
        text_embeddings: text -> float32[D], one trunk embedding per distinct intent or condition text
            (:func:`encode_goal_texts`); every non-empty text of ``executive`` must be present. None
            (default) leaves the two text columns out; given, goal k's ``intent`` and ``when`` fill them,
            and an empty text or a goal without words (a GoalSpec) leaves zeros.
        mask_skill: Zero the e_skill columns of every slot (the text-only reading mode).

    Raises:
        KeyError: If a goal names a skill outside the catalogue, or a label or a text has no embedding.
        ValueError: If the catalogue is invalid or ``skill_embeddings`` is not float32[K, D].
    """

    catalogue = validate_skill_ids(skill_ids)
    table = np.asarray(skill_embeddings, dtype=np.float32)
    if table.ndim != 2 or table.shape[0] != len(catalogue):
        raise ValueError(f"skill_embeddings must have shape ({len(catalogue)}, D), got {table.shape}")
    index_of = {skill: position for position, skill in enumerate(catalogue)}
    dim = table.shape[1]
    labels = dim + SLOT_SCALARS
    out = np.zeros((MAX_GOALS, (2 if text_embeddings is None else 4) * dim + SLOT_SCALARS), dtype=np.float32)
    for slot, goal in enumerate(executive.goals):
        if goal.skill_id not in index_of:
            raise KeyError(f"goal {slot} names unknown skill id {goal.skill_id!r}; it is not in skill_ids")
        out[slot, :dim] = table[index_of[goal.skill_id]]
        out[slot, dim] = goal.priority
        out[slot, dim + 1] = 1.0
        out[slot, dim + 2 : dim + 2 + len(goal.params)] = goal.params
        out[slot, dim + 2 + PARAM_WIDTH] = goal.horizon_seconds / HORIZON_SCALE_SECONDS
        out[slot, dim + 3 + PARAM_WIDTH] = float(goal.deadline_tick is not None)
        if goal.constraints:
            out[slot, labels : labels + dim] = np.mean(
                [np.asarray(label_embeddings[label], dtype=np.float32) for label in goal.constraints], axis=0
            )
        if text_embeddings is not None:
            for start, text in ((labels + dim, getattr(goal, "intent", "")), (labels + 2 * dim, getattr(goal, "when", ""))):
                if text:
                    out[slot, start : start + dim] = np.asarray(text_embeddings[text], dtype=np.float32)
    if mask_skill:
        out[:, :dim] = 0.0
    return out


def generic_rule_labels(
    features: Any, feature_known: Any, executive: ExecutiveState, catalogue: Sequence[str], current_skill: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_skill int32[N], y_replace bool[N])`` of the teacher (M10) for N belief rows.

    Args:
        features: float32[N, 10], the damage counter published as counter / TICKS_SATURATION.
        feature_known: bool[N, 10].
        executive: The E_t whose goals and constraint labels the rule reads.
        catalogue: The skill ids the indices refer to, idle first.
        current_skill: int32[N], the skill in force (y_replace compares with it).

    Raises:
        KeyError: If a goal names a skill outside the catalogue.
        ValueError: If the catalogue is invalid or a shape is wrong.
    """

    skills = validate_skill_ids(catalogue)
    lowered = executive_arrays(executive, skills, 1)
    column = feature_index("ticks_since_damage")
    values = np.asarray(features, dtype=np.float32)
    known = np.asarray(feature_known, dtype=np.bool_)
    current = np.asarray(current_skill, dtype=np.int32)
    rows = values.shape[0] if values.ndim == 2 else -1
    if values.shape != known.shape or current.shape != (rows,):
        raise ValueError(f"features, feature_known and current_skill must be (N, 10), (N, 10) and (N,); got "
                         f"{values.shape}, {known.shape} and {current.shape}")
    window = known[:, column] & (values[:, column] * np.float32(TICKS_SATURATION) < EVADE_DAMAGE_WINDOW_TICKS)
    skill = np.asarray(lowered.skill_index)[0]
    valid = np.asarray(lowered.goal_valid)[0]
    priority = np.asarray(lowered.priority)[0]
    labelled = np.zeros_like(valid)
    labelled[: len(executive.goals)] = [DAMAGE_CONSTRAINT in goal.constraints for goal in executive.goals]
    damage = valid & labelled
    best = skill[np.argmax(np.where(valid, priority, -np.inf))] if valid.any() else IDLE_SKILL_INDEX
    damage_best = skill[np.argmax(np.where(damage, priority, -np.inf))] if damage.any() else best
    y_skill = np.where(window & damage.any(), damage_best, best).astype(np.int32)
    return y_skill, y_skill != current


def goal_reading_states(
    key: jax.Array,
    count: int,
    *,
    catalogue_sizes: Sequence[int] = CATALOGUE_SIZES,
    max_goals: int = MAX_GOALS,
    constraint_vocabulary: Sequence[str] = CONSTRAINT_VOCABULARY,
) -> tuple[GoalReadingState, ...]:
    """Return ``count`` generated E_t over placeholder catalogues (module docstring), deterministic in ``key``.

    Raises:
        ValueError: If ``count`` is negative, a catalogue size is outside [2, 25] or none is given,
            ``max_goals`` is outside [1, MAX_GOALS], or the vocabulary holds an empty or repeated label.
    """

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError(f"count must be a non-negative integer, got {count!r}")
    sizes = tuple(int(size) for size in catalogue_sizes)
    largest = min(len(_DESCRIPTION_POOL), len(_LETTERS)) + 1
    if not sizes or min(sizes) < 2 or max(sizes) > largest:
        raise ValueError(f"catalogue_sizes must lie in [2, {largest}], got {sizes}")
    if isinstance(max_goals, bool) or not isinstance(max_goals, (int, np.integer)) or not 1 <= max_goals <= MAX_GOALS:
        raise ValueError(f"max_goals must be an integer in [1, {MAX_GOALS}], got {max_goals!r}")
    labels = tuple(constraint_vocabulary)
    if len(set(labels)) != len(labels) or not all(isinstance(label, str) and label for label in labels):
        raise ValueError(f"constraint_vocabulary must hold distinct non-empty strings, got {labels!r}")
    rng = _numpy_rng(key)
    idle = skill_descriptions_of((SKILL_IDLE,))[0]
    states: list[GoalReadingState] = []
    for position in range(int(count)):
        size = int(rng.choice(sizes))
        named = tuple(f"skill_{letter}" for letter in _LETTERS[: size - 1])
        drawn = rng.choice(len(_DESCRIPTION_POOL), size=size - 1, replace=False)
        descriptions = (idle, *(_DESCRIPTION_POOL[int(index)] for index in drawn))
        goals = []
        for skill in rng.choice(named, size=int(rng.integers(1, min(max_goals, size - 1) + 1)), replace=False):
            params = tuple(round(float(value), 2) for value in rng.uniform(-1.0, 1.0, int(rng.integers(0, PARAM_WIDTH + 1))))
            deadline = int(rng.integers(0, DEADLINE_TICKS)) if rng.random() < DEADLINE_PROBABILITY else None
            goals.append(
                GoalSpec(
                    skill_id=str(skill),
                    priority=float(PRIORITY_MAX - rng.uniform(0.0, PRIORITY_MAX)),
                    params=params,
                    constraints=tuple(label for label in labels if rng.random() < LABEL_PROBABILITY),
                    horizon_seconds=round(float(rng.uniform(*HORIZON_RANGE_SECONDS)), 1),
                    deadline_tick=deadline,
                )
            )
        state = ExecutiveState(
            version=position + 1, issued_tick=0, goals=tuple(goals), budget=ComputeBudget(), facts=()
        )
        states.append(GoalReadingState(state, (SKILL_IDLE, *named), descriptions))
    return tuple(states)


def goal_reading_rows(
    states: Sequence[GoalReadingState], belief_dataset: Mapping[str, Any], key: jax.Array, rows_per_state: int
) -> SupervisedRows:
    """Return ``rows_per_state`` rows per state: belief rows (features, known flags, latent, current
    skill) drawn uniformly from ``belief_dataset`` (without replacement when it has enough rows),
    labelled by (M10), one x_E per state. Deterministic in ``key``.

    Raises:
        ValueError: If ``states`` is empty, ``rows_per_state`` is not positive, the dataset is
            invalid, or a recorded current skill lies outside the smallest catalogue.
    """

    if not states:
        raise ValueError("goal_reading_rows needs at least one state")
    if isinstance(rows_per_state, bool) or not isinstance(rows_per_state, (int, np.integer)) or rows_per_state < 1:
        raise ValueError(f"rows_per_state must be a positive integer, got {rows_per_state!r}")
    validate_dataset(belief_dataset)
    columns = {name: np.asarray(belief_dataset[name]) for name in ("features", "feature_known", "latent", "current_skill")}
    total = columns["current_skill"].shape[0]
    smallest = min(len(item.skill_ids) for item in states)
    if int(columns["current_skill"].max()) >= smallest:
        raise ValueError(f"a recorded current skill lies outside the smallest catalogue ({smallest} skills)")
    rng = _numpy_rng(key)
    parts = []
    for item in states:
        index = rng.choice(total, size=int(rows_per_state), replace=rows_per_state > total)
        sample = {name: column[index] for name, column in columns.items()}
        y_skill, y_replace = generic_rule_labels(
            sample["features"], sample["feature_known"], item.state, item.skill_ids, sample["current_skill"]
        )
        lowered = executive_arrays(item.state, item.skill_ids, int(rows_per_state))
        text = render_executive(item.skill_ids, item.descriptions, item.state)
        parts.append(_rows(sample, text, item, lowered, y_skill, y_replace))
    return concat_rows(parts)


def goal_reading_split(
    states: Sequence[GoalReadingState],
) -> tuple[tuple[GoalReadingState, ...], tuple[GoalReadingState, ...], tuple[GoalReadingState, ...]]:
    """Return ``(train, held-out states, held-out catalogue sizes)`` of (M11) (module docstring)."""

    train: list[GoalReadingState] = []
    held: list[GoalReadingState] = []
    sizes: list[GoalReadingState] = []
    for item in states:
        if len(item.skill_ids) > TRAIN_MAX_CATALOGUE_SIZE:
            sizes.append(item)
            continue
        text = render_executive(item.skill_ids, item.descriptions, item.state)
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        (held if int.from_bytes(digest, "little") % HELDOUT_MODULUS == 0 else train).append(item)
    return tuple(train), tuple(held), tuple(sizes)


class GoalLanguageSample(NamedTuple):
    """One generated goal-language state (``goal_language.sample_state``): the section in the LLM's words,
    one SlotMeaning per goal, and its catalogue (skill id -> description, idle first)."""

    section: ExecutiveSection
    meanings: tuple[SlotMeaning, ...]
    catalogue: Mapping[str, str]


class GoalLanguageSets(NamedTuple):
    """The G3-NL sets of SYSTEM_ONE.md 8.11 (:func:`goal_language_sets`) and the held-out intent family ids."""

    train: tuple[GoalLanguageSample, ...]
    heldout_wordings: tuple[GoalLanguageSample, ...]
    heldout_families: tuple[GoalLanguageSample, ...]
    heldout_sizes: tuple[GoalLanguageSample, ...]
    heldout_family_ids: tuple[str, ...]


def goal_language_sets(
    key: jax.Array,
    n_train: int,
    n_eval: int,
    *,
    catalogue_sizes_train: Sequence[int] = LANGUAGE_TRAIN_SIZES,
    catalogue_sizes_eval: Sequence[int] = LANGUAGE_EVAL_SIZES,
    heldout_wordings: float = 0.25,
    heldout_families: int = 2,
) -> GoalLanguageSets:
    """Return ``n_train`` training samples and ``n_eval`` samples in each held-out split of G3-NL (module
    docstring: training, held-out wordings, held-out families, held-out sizes), deterministic in ``key``.

    Raises:
        ValueError: If a count is negative, or as ``split_wordings``, ``split_families`` and
            ``synthetic_catalogue`` (a catalogue size larger than its family pool allows).
    """

    for name, count in (("n_train", n_train), ("n_eval", n_eval)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {count!r}")
    split_key, train_key, wordings_key, families_key, sizes_key = jax.random.split(key, 5)
    rng = _numpy_rng(split_key)
    train_words, held_words = split_wordings(rng, heldout_wordings)
    train_ids, held_ids = split_families(rng, heldout_families)
    train_pool = tuple(family for family in INTENT_FAMILIES if family.family_id in train_ids)
    held_descriptions = {family.description for family in INTENT_FAMILIES if family.family_id in held_ids}

    def draw(key, count, sizes, wordings, pool, real_rate, families, novel):
        """``count`` samples; ``novel`` redraws until the catalogue and a goal hold a held-out family."""

        rng = _numpy_rng(key)
        samples: list[GoalLanguageSample] = []
        while len(samples) < int(count):
            if real_rate and rng.random() < real_rate:
                catalogue = dict(REAL_CATALOGUE)
            else:
                catalogue = synthetic_catalogue(rng, int(rng.choice(sizes)), pool)
            if novel and not held_descriptions & set(catalogue.values()):
                continue
            while True:
                section, meanings = sample_state(
                    rng, catalogue, wordings, version=len(samples) + 1, intent_families=families
                )
                if not novel or any(meaning.intent_family in held_ids for meaning in meanings):
                    break
            samples.append(GoalLanguageSample(section, meanings, catalogue))
        return tuple(samples)

    return GoalLanguageSets(
        train=draw(train_key, n_train, catalogue_sizes_train, train_words, train_pool, REAL_CATALOGUE_RATE, train_ids, False),
        heldout_wordings=draw(
            wordings_key, n_eval, catalogue_sizes_train, held_words, train_pool, REAL_CATALOGUE_RATE, train_ids, False
        ),
        heldout_families=draw(families_key, n_eval, catalogue_sizes_train, train_words, INTENT_FAMILIES, 0.0, None, True),
        heldout_sizes=draw(sizes_key, n_eval, catalogue_sizes_eval, train_words, train_pool, 0.0, train_ids, False),
        heldout_family_ids=tuple(held_ids),
    )


def goal_language_rows(
    samples: Sequence[GoalLanguageSample],
    belief_dataset: Mapping[str, Any],
    key: jax.Array,
    rows_per_state: int,
    *,
    trunk_or_text_embeddings: TrunkEncoder | Mapping[str, Any],
    mode: str,
) -> SupervisedRows:
    """Return ``rows_per_state`` rows per sample labelled by the teacher (M13) (module docstring), one x_E
    per sample, padded to the widest catalogue. Deterministic in ``key``.

    Args:
        samples: ``(section, meanings, catalogue)`` triples (:class:`GoalLanguageSample`).
        belief_dataset: A recorded ``train_j1`` dataset; its belief rows (features, known flags, latent,
            current skill) are drawn uniformly, without replacement when it has enough rows.
        key: PRNG key of the draws.
        rows_per_state: Belief rows per sample.
        trunk_or_text_embeddings: A trunk, which encodes the samples' distinct texts in one batched call,
            or a mapping text -> float32[D] that holds every text (the run's cache).
        mode: ``"typed"`` or ``"text-only"`` (``allowed_mask`` and ``GoalReadingState.mask_skill``).

    Raises:
        KeyError: If a mapping lacks a text of the samples.
        ValueError: If ``samples`` is empty, ``mode`` is unknown, ``rows_per_state`` is not positive, the
            dataset is invalid, a recorded current skill lies outside the smallest catalogue, or a goal
            names a skill outside its catalogue.
    """

    if not samples:
        raise ValueError("goal_language_rows needs at least one sample")
    if mode not in LANGUAGE_MODES:
        raise ValueError(f"mode must be one of {LANGUAGE_MODES}, got {mode!r}")
    if isinstance(rows_per_state, bool) or not isinstance(rows_per_state, (int, np.integer)) or rows_per_state < 1:
        raise ValueError(f"rows_per_state must be a positive integer, got {rows_per_state!r}")
    validate_dataset(belief_dataset)
    columns = {name: np.asarray(belief_dataset[name]) for name in ("features", "feature_known", "latent", "current_skill")}
    total = columns["current_skill"].shape[0]
    smallest = min(len(sample.catalogue) for sample in samples)
    if int(columns["current_skill"].max()) >= smallest:
        raise ValueError(f"a recorded current skill lies outside the smallest catalogue ({smallest} skills)")
    texts = goal_texts(goal for sample in samples for goal in sample.section.goals)
    source = trunk_or_text_embeddings
    table = encode_goal_texts(source, texts, {}) if isinstance(source, TrunkEncoder) else source
    text_embeddings = {text: np.asarray(table[text], dtype=np.float32) for text in sorted(texts)}
    per_state, width = int(rows_per_state), max(len(sample.catalogue) for sample in samples)
    count = len(samples) * per_state
    ticks = [feature_index(name) for name in _TICK_FEATURES]
    rng = _numpy_rng(key)
    index = np.empty((count,), dtype=np.int64)
    allowed = np.zeros((count, width), dtype=np.bool_)
    y_skill = np.empty((count,), dtype=np.int32)
    y_replace = np.empty((count,), dtype=np.bool_)
    rendered: list[str] = []
    executives: list[GoalReadingState] = []
    for position, sample in enumerate(samples):
        start = position * per_state
        drawn = rng.choice(total, size=per_state, replace=per_state > total)
        index[start : start + per_state] = drawn
        skill_ids, descriptions = tuple(sample.catalogue), tuple(sample.catalogue.values())
        rendered.append(render_executive(skill_ids, descriptions, sample.section.to_executive_state()))
        column_of = {skill: column for column, skill in enumerate(skill_ids)}
        physical = columns["features"][drawn].astype(np.float32)
        physical[:, ticks] *= np.float32(TICKS_SATURATION)
        rows = zip(physical.tolist(), columns["feature_known"][drawn].tolist(), columns["current_skill"][drawn].tolist())
        for offset, (values, known, current) in enumerate(rows):
            chosen, replace = teacher(sample.section, sample.meanings, values, known, skill_ids[current])
            y_skill[start + offset], y_replace[start + offset] = column_of[chosen], replace
        if mode == "text-only":
            allowed[start : start + per_state, : len(skill_ids)] = True
        else:
            for column in {IDLE_SKILL_INDEX, *(column_of[goal.skill_id] for goal in sample.section.goals)}:
                allowed[start : start + per_state, column] = True
        executives.append(GoalReadingState(sample.section, skill_ids, descriptions, mode == "text-only"))
    current_onehot = np.zeros((count, width), dtype=np.float32)
    current_onehot[np.arange(count), columns["current_skill"][index]] = 1.0
    return SupervisedRows(
        executive_texts=tuple(rendered),
        executive_index=np.repeat(np.arange(len(samples), dtype=np.int32), per_state),
        u=np.asarray(belief_inputs(columns["features"][index], columns["feature_known"][index], columns["latent"][index])),
        allowed_mask=allowed,
        current_onehot=current_onehot,
        y_skill=y_skill,
        y_replace=y_replace,
        y_params=np.zeros((count, PARAM_WIDTH), dtype=np.float32),
        executives=tuple(executives),
        text_embeddings=text_embeddings,
    )


__all__ = [
    "CATALOGUE_SIZES",
    "CONSTRAINT_VOCABULARY",
    "DAMAGE_CONSTRAINT",
    "DESCRIPTION_POOL_SIZE",
    "DROP_PROBABILITY",
    "ExecutiveVariant",
    "GoalLanguageSample",
    "GoalLanguageSets",
    "GoalReadingState",
    "HELDOUT_MODULUS",
    "HORIZON_SCALES",
    "LABEL_PROBABILITY",
    "LANGUAGE_EVAL_SIZES",
    "LANGUAGE_MODES",
    "LANGUAGE_TRAIN_SIZES",
    "MAX_ADDED_GOALS",
    "REAL_CATALOGUE_RATE",
    "SupervisedRows",
    "TRAIN_MAX_CATALOGUE_SIZE",
    "catalogue_embeddings",
    "concat_rows",
    "encode_goal_texts",
    "executive_variants",
    "generic_rule_labels",
    "goal_language_rows",
    "goal_language_sets",
    "goal_reading_rows",
    "goal_reading_split",
    "goal_reading_states",
    "goal_texts",
    "render_executive",
    "rows_from_dataset",
    "skill_descriptions_of",
    "slot_inputs",
    "variant_rows",
]
