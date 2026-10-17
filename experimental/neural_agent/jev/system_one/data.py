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
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.contracts import (
    MAX_GOALS,
    SKILL_IDLE,
    ExecutiveArrays,
    ExecutiveState,
    GoalSpec,
    executive_arrays,
)
from experimental.neural_agent.jev.lm.prompting import _fact_lines, _goal_lines, validate_skill_ids
from experimental.neural_agent.jev.network import _matching_slot, j1_allowed_skills
from experimental.neural_agent.jev.scripted import _evader_index
from experimental.neural_agent.jev.system_one.heads import belief_inputs
from experimental.neural_agent.jev.system_one.trunk import TrunkEncoder
from experimental.neural_agent.jev.train_j1 import scripted_target_skill, validate_dataset

DROP_PROBABILITY: float = 0.25
MAX_ADDED_GOALS: int = 2
HORIZON_SCALES: tuple[float, ...] = (0.5, 1.0, 2.0)
HELDOUT_MODULUS: int = 5
_EXECUTIVE_FIELDS: dict[str, str] = {
    "skill_index": "exec_skill_index",
    "priority": "exec_priority",
    "params": "exec_params",
    "goal_valid": "exec_goal_valid",
    "budget": "exec_budget",
    "version": "exec_version",
}


class SupervisedRows(NamedTuple):
    """Stage-A rows, N of them over V distinct executives: executive_texts tuple[str] (V),
    executive_index int32[N] (row -> text), u float32[N, 84], allowed_mask bool[N, K],
    current_onehot float32[N, K], y_skill int32[N], y_replace bool[N], y_params float32[N, P]."""

    executive_texts: tuple[str, ...]
    executive_index: np.ndarray
    u: np.ndarray
    allowed_mask: np.ndarray
    current_onehot: np.ndarray
    y_skill: np.ndarray
    y_replace: np.ndarray
    y_params: np.ndarray


class ExecutiveVariant(NamedTuple):
    """One templated E_t: the state, its arrays lowered for one lane, and its split side."""

    state: ExecutiveState
    arrays: ExecutiveArrays
    heldout: bool


def skill_descriptions_of(skill_ids: Sequence[str], skill_descriptions: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return one description per skill: ``skill_descriptions``, or the skill ids when it is None.

    Raises:
        ValueError: If the catalogue is invalid, or the descriptions are not one non-empty string per skill.
    """

    catalogue = validate_skill_ids(skill_ids)
    if skill_descriptions is None:
        return catalogue
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
    catalogue: tuple[str, ...],
    lowered: ExecutiveArrays,
    y_skill: np.ndarray,
    y_replace: np.ndarray,
) -> SupervisedRows:
    """Assemble the rows of ``dataset`` under one E_t (``lowered`` has N lanes, ``text`` is its x_E)."""

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
    return _rows(dataset, text, catalogue, lowered, dataset["target_skill"], dataset["target_replace"])


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
        text = render_executive(catalogue, skill_descriptions, variant.state)
        parts.append(_rows(dataset, text, catalogue, lowered, y_skill, y_replace))
    return concat_rows(parts)


def concat_rows(parts: Sequence[SupervisedRows]) -> SupervisedRows:
    """Concatenate one or more row sets; each part's executive indices are offset past the earlier texts.

    Raises:
        ValueError: If ``parts`` is empty.
    """

    if not parts:
        raise ValueError("concat_rows needs at least one row set")
    offsets = np.cumsum([0] + [len(part.executive_texts) for part in parts[:-1]])
    texts = tuple(text for part in parts for text in part.executive_texts)
    index = np.concatenate([part.executive_index + offset for part, offset in zip(parts, offsets)]).astype(np.int32)
    arrays = [np.concatenate([getattr(part, name) for part in parts], axis=0) for name in SupervisedRows._fields[2:]]
    return SupervisedRows(texts, index, *arrays)


__all__ = [
    "DROP_PROBABILITY",
    "ExecutiveVariant",
    "HELDOUT_MODULUS",
    "HORIZON_SCALES",
    "MAX_ADDED_GOALS",
    "SupervisedRows",
    "catalogue_embeddings",
    "concat_rows",
    "executive_variants",
    "render_executive",
    "rows_from_dataset",
    "skill_descriptions_of",
    "variant_rows",
]
