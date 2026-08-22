"""Composable success criteria, so a custom game needs no array code.

A game's success condition is the part people get wrong, because it is written
once and then trusted forever. Two failure modes this module exists to prevent:

* **Indexing the observation by position.** The width moved twice on
  2026-08-03 (238 -> 241 named columns, 8259 -> 8271 flat). :func:`col` reads
  by name and raises with the current schema size when a name is gone, so a
  moved contract fails loudly at authoring time instead of silently scoring
  the wrong column.
* **Reducing over the wrong axis.** Records are ``(ticks, batch, features)``.
  Hand-written predicates routinely collapse the batch and report one number
  for the whole rollout. Every reducer here collapses **ticks only** and
  returns one value per environment.

Compose selectors -> reducers -> comparisons -> combinators::

    from arena.tasks.framework.criteria import col, at_end, minimum, above, below, all_of

    survived   = above(at_end(col("self_f32", "alive")), 0.5)
    killed     = below(minimum(col("target_f32", "health_fraction")), 1e-6)
    flawless   = all_of(survived, killed)

Each stage is an ordinary callable, so anything missing here can be written
inline without adopting a framework::

    custom = lambda record: my_own_numpy(record) > 3
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import jax.numpy as np

from arena.jax_contract import GROUP_FEATURES

#: ``(ticks, batch)`` -- one scalar per environment per step.
Series = Callable[[Mapping[str, Any]], np.ndarray]

#: ``(batch,)`` -- one value per environment, ticks already collapsed.
PerEnv = Callable[[Mapping[str, Any]], np.ndarray]

#: ``(batch,)`` of bool -- did this environment satisfy the condition.
Criterion = Callable[[Mapping[str, Any]], np.ndarray]
_GROUPS_ATTRIBUTE = "_arena_required_groups"


def mark_required_groups(fn: Callable, groups: frozenset[str] | None):
    """Attach static recording requirements; ``None`` means unknown."""

    setattr(fn, _GROUPS_ATTRIBUTE, groups)
    return fn


def criterion_groups(fn: Callable) -> frozenset[str] | None:
    """Observation groups needed by a criterion, when statically known."""

    return getattr(fn, _GROUPS_ATTRIBUTE, None)


def _combined_groups(functions) -> frozenset[str] | None:
    groups = [criterion_groups(fn) for fn in functions]
    if any(group is None for group in groups):
        return None
    return frozenset().union(*groups)


# -- selectors -----------------------------------------------------------------


def col(group: str, name: str) -> Series:
    """One named observation column, ``(ticks, batch)``."""

    if group not in GROUP_FEATURES:
        raise KeyError(
            f"unknown observation group {group!r}; have: "
            + ", ".join(sorted(GROUP_FEATURES))
        )
    features = GROUP_FEATURES[group]
    if name not in features:
        raise KeyError(
            f"{group}.{name} is not in the current schema ({len(features)} "
            "columns). Read by name and update the game -- never by position."
        )
    index = features.index(name)

    def select(record: Mapping[str, Any]) -> np.ndarray:
        if group not in record:
            raise KeyError(
                f"group {group!r} was not recorded; the rollout's record "
                f"function must retain it for this criterion"
            )
        return np.asarray(record[group])[..., index]

    select.__name__ = f"col[{group}.{name}]"
    return mark_required_groups(select, frozenset({group}))


def recorded(key: str) -> Series:
    """A value the task's own record function added, e.g. an info channel."""

    def select(record: Mapping[str, Any]) -> np.ndarray:
        if key not in record:
            raise KeyError(f"{key!r} was not recorded by this task")
        return np.asarray(record[key])

    select.__name__ = f"recorded[{key}]"
    return select


def derived(
    fn: Callable[[Mapping[str, Any]], np.ndarray], label: str = "derived"
) -> Series:
    """Escape hatch: any callable producing ``(ticks, batch)``."""

    fn.__name__ = label
    return fn


# -- reducers: collapse ticks, keep batch --------------------------------------


def _reducer(
    name: str, op: Callable[[np.ndarray], np.ndarray]
) -> Callable[[Series], PerEnv]:
    def make(series: Series) -> PerEnv:
        def reduce(record: Mapping[str, Any]) -> np.ndarray:
            values = np.asarray(series(record))
            if values.ndim < 1:
                raise ValueError(f"{name}: expected (ticks, batch), got scalar")
            return op(values)

        reduce.__name__ = f"{name}({getattr(series, '__name__', 'series')})"
        return mark_required_groups(reduce, criterion_groups(series))

    return make


at_end = _reducer("at_end", lambda v: v[-1])
at_start = _reducer("at_start", lambda v: v[0])
minimum = _reducer("minimum", lambda v: v.min(axis=0))
maximum = _reducer("maximum", lambda v: v.max(axis=0))
mean = _reducer("mean", lambda v: v.mean(axis=0))
total = _reducer("total", lambda v: v.sum(axis=0))
span = _reducer("span", lambda v: v.max(axis=0) - v.min(axis=0))


def fraction_of_time(
    series: Series, predicate: Callable[[np.ndarray], np.ndarray]
) -> PerEnv:
    """Fraction of ticks where ``predicate`` holds, per environment."""

    def reduce(record: Mapping[str, Any]) -> np.ndarray:
        return predicate(np.asarray(series(record))).mean(axis=0)

    reduce.__name__ = f"fraction_of_time({getattr(series, '__name__', 'series')})"
    return mark_required_groups(reduce, criterion_groups(series))


# -- comparisons ---------------------------------------------------------------


def ever_below(
    series: Series,
    threshold: float,
    *,
    available: Series,
    ticks: int | None = None,
    inclusive: bool = False,
) -> Criterion:
    """Whether an available sample crosses a lower threshold.

    Masked observation fields are zero-filled, so reducing ``series`` before
    applying ``available`` can turn missing evidence into a success.  This
    comparator masks first and fails closed when every sample is unavailable.
    """

    if ticks is not None and ticks < 1:
        raise ValueError("ever_below(ticks=) must be positive")

    def check(record: Mapping[str, Any]) -> np.ndarray:
        values = np.asarray(series(record))
        evidence = np.asarray(available(record)) > 0.5
        if values.shape != evidence.shape:
            raise ValueError(
                "ever_below value and availability shapes differ: "
                f"{values.shape} != {evidence.shape}"
            )
        if values.ndim < 1:
            raise ValueError("ever_below expected (ticks, batch), got scalar")
        if ticks is not None:
            if values.shape[0] < ticks:
                raise ValueError(
                    f"ever_below(ticks={ticks}) needs {ticks} samples; "
                    f"the record has {values.shape[0]}"
                )
            values, evidence = values[:ticks], evidence[:ticks]
        crossed = values <= threshold if inclusive else values < threshold
        return (evidence & crossed).any(axis=0)

    comparator = "<=" if inclusive else "<"
    window = "" if ticks is None else f", first {ticks} ticks"
    check.__name__ = (
        f"ever({getattr(series, '__name__', '?')} {comparator} {threshold}"
        f" when {getattr(available, '__name__', '?')}{window})"
    )
    return mark_required_groups(check, _combined_groups((series, available)))


def above(value: PerEnv, threshold: float) -> Criterion:
    def check(record: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(value(record)) > threshold

    check.__name__ = f"({getattr(value, '__name__', '?')} > {threshold})"
    return mark_required_groups(check, criterion_groups(value))


def below(value: PerEnv, threshold: float) -> Criterion:
    def check(record: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(value(record)) < threshold

    check.__name__ = f"({getattr(value, '__name__', '?')} < {threshold})"
    return mark_required_groups(check, criterion_groups(value))


def within(value: PerEnv, low: float, high: float) -> Criterion:
    if low > high:
        raise ValueError("within(low, high) requires low <= high")

    def check(record: Mapping[str, Any]) -> np.ndarray:
        found = np.asarray(value(record))
        return (found >= low) & (found <= high)

    check.__name__ = f"({getattr(value, '__name__', '?')} in [{low}, {high}])"
    return mark_required_groups(check, criterion_groups(value))


# -- combinators ---------------------------------------------------------------


def all_of(*criteria: Criterion) -> Criterion:
    if not criteria:
        raise ValueError("all_of needs at least one criterion")

    def check(record: Mapping[str, Any]) -> np.ndarray:
        result = np.asarray(criteria[0](record))
        for other in criteria[1:]:
            result = result & np.asarray(other(record))
        return result

    check.__name__ = " and ".join(getattr(c, "__name__", "?") for c in criteria)
    return mark_required_groups(check, _combined_groups(criteria))


def any_of(*criteria: Criterion) -> Criterion:
    if not criteria:
        raise ValueError("any_of needs at least one criterion")

    def check(record: Mapping[str, Any]) -> np.ndarray:
        result = np.asarray(criteria[0](record))
        for other in criteria[1:]:
            result = result | np.asarray(other(record))
        return result

    check.__name__ = " or ".join(getattr(c, "__name__", "?") for c in criteria)
    return mark_required_groups(check, _combined_groups(criteria))


def negate(criterion: Criterion) -> Criterion:
    def check(record: Mapping[str, Any]) -> np.ndarray:
        return ~np.asarray(criterion(record))

    check.__name__ = f"not {getattr(criterion, '__name__', '?')}"
    return mark_required_groups(check, criterion_groups(criterion))


def explain(criterion: Criterion) -> str:
    """The criterion as a readable expression, for run metadata."""

    return getattr(criterion, "__name__", repr(criterion))


__all__ = [
    "Criterion",
    "PerEnv",
    "Series",
    "all_of",
    "any_of",
    "above",
    "at_end",
    "at_start",
    "below",
    "col",
    "criterion_groups",
    "derived",
    "ever_below",
    "explain",
    "fraction_of_time",
    "maximum",
    "mean",
    "mark_required_groups",
    "minimum",
    "negate",
    "recorded",
    "span",
    "total",
    "within",
]
