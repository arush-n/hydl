"""Normalize result declarations without learning a lane's metric semantics.

Artifacts decide which metrics are diagnostic.  The console only preserves
that declaration, orders declared non-diagnostic metrics first, and echoes any
declared coverage.  In particular, it does not turn an ``unresolved`` count
into a ``censored`` count or infer a preferred direction from a metric name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

UNDECLARED = "undeclared"


def declared_result_blocks(document: Any) -> list[dict[str, Any]]:
    """Return every mapping that declares ``diagnostic_only_metrics``.

    The declaration may live anywhere in an artifact.  Paths are retained so
    a reader can distinguish two result blocks without the console assigning
    domain-specific names to either one.
    """

    found: list[dict[str, Any]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            if "diagnostic_only_metrics" in value:
                found.append(normalize_result_block(value, path=".".join(path)))
            for key, child in value.items():
                visit(child, (*path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))

    visit(document, ())
    return found


def normalize_result_block(
    block: Mapping[str, Any], *, path: str = "",
) -> dict[str, Any]:
    """Apply the declared diagnostic/coverage display contract to one block."""

    raw_diagnostic = block.get("diagnostic_only_metrics")
    declaration_valid = isinstance(raw_diagnostic, list) and all(
        isinstance(item, str) for item in raw_diagnostic
    )
    diagnostic = set(raw_diagnostic if declaration_valid else [])
    primary = block.get("primary_metric")
    primary = primary if isinstance(primary, str) else None
    metric_map = _metric_map(block, primary=primary, diagnostic=diagnostic)

    metrics: list[dict[str, Any]] = []
    for raw_name, raw_value in metric_map.items():
        name = _canonical_name(str(raw_name), diagnostic)
        is_diagnostic = name in diagnostic
        value, direction = _metric_value(raw_value)
        coverage = _coverage(block, name, raw_value)
        metrics.append({
            "name": name,
            "source_name": str(raw_name),
            "value": value,
            "primary": name == primary,
            "diagnostic_only": is_diagnostic,
            "direction": direction,
            "coverage": coverage if _is_rate(name) else None,
        })

    # A diagnostic declaration always wins over a primary declaration.  If an
    # artifact contradicts itself, the metric remains greyed rather than being
    # promoted by the console.
    metrics.sort(key=lambda metric: (
        metric["diagnostic_only"],
        not metric["primary"],
        metric["name"],
    ))
    return {
        "path": path or "[root]",
        "primary_metric": primary or UNDECLARED,
        "diagnostic_only_metrics": sorted(diagnostic),
        "declaration_valid": declaration_valid,
        "metrics": metrics,
        "metric_source": _metric_source(block, metric_map),
    }


def _metric_map(
    block: Mapping[str, Any], *, primary: str | None, diagnostic: set[str],
) -> Mapping[str, Any]:
    """Choose the nearest mapping that contains the declared metric names.

    This is structural discovery only.  A mapping scores when its keys equal a
    declared primary/diagnostic name (allowing the explicit
    ``_diagnostic_only`` suffix) and when its values are scalar result records.
    No metric value or domain meaning participates in selection.
    """

    explicit = block.get("metrics")
    if isinstance(explicit, Mapping):
        return {
            str(key): value for key, value in explicit.items()
            if _looks_like_metric_value(value)
        }

    wanted = diagnostic | ({primary} if primary else set())
    if not wanted:
        # Without an explicit ``metrics`` mapping or a named declaration there
        # is no principled way to distinguish a result from scalar metadata.
        return {}
    candidates: list[tuple[int, int, str, Mapping[str, Any]]] = []

    def visit(value: Any, path: tuple[str, ...], depth: int) -> None:
        if not isinstance(value, Mapping) or depth > 4:
            return
        keys = {_canonical_name(str(key), diagnostic) for key in value}
        overlap = len(keys & wanted)
        record_values = sum(
            _looks_like_metric_value(child) for child in value.values()
        )
        if overlap or (not wanted and record_values):
            # Declared-name overlap is decisive; shallower and smaller maps
            # break ties so a per-seed expansion cannot eclipse its summary.
            score = overlap * 100 + record_values
            candidates.append((score, -depth, ".".join(path), value))
        for key, child in value.items():
            if key in {"diagnostic_only_metrics", "coverage", "coverage_by_metric"}:
                continue
            visit(child, (*path, str(key)), depth + 1)

    # A conventional explicit metrics mapping is still just structure, and it
    # should win a tie over a deeper summary with the same declared names.
    visit(block, (), 0)
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (item[0], item[1], -len(item[3])), reverse=True)
    chosen = candidates[0][3]
    excluded = {
        "diagnostic_only_metrics", "primary_metric", "coverage",
        "coverage_by_metric", "episodes", "mode", "evaluation_seed",
    }
    return {
        str(key): value for key, value in chosen.items()
        if key not in excluded and _looks_like_metric_value(value)
    }


def _metric_source(block: Mapping[str, Any], chosen: Mapping[str, Any]) -> str:
    if not chosen:
        return UNDECLARED

    def visit(value: Any, path: tuple[str, ...]) -> str | None:
        if value is chosen:
            return ".".join(path) or "[block]"
        if isinstance(value, Mapping):
            for key, child in value.items():
                located = visit(child, (*path, str(key)))
                if located is not None:
                    return located
        return None

    # ``_metric_map`` returns a filtered copy, so identity usually cannot be
    # recovered.  Keep the provenance honest rather than inventing a label.
    return visit(block, ()) or "declared result mapping"


def _looks_like_metric_value(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float, str)):
        return True
    if not isinstance(value, Mapping):
        return False
    return any(key in value for key in ("value", "rate", "mean")) or any(
        str(key).startswith("direction_favoring") for key in value
    )


def _canonical_name(name: str, diagnostic: Iterable[str]) -> str:
    suffix = "_diagnostic_only"
    candidate = name[:-len(suffix)] if name.endswith(suffix) else name
    # The suffix is display metadata only when the declaration names the
    # unsuffixed metric.  A suggestive key alone never becomes a declaration.
    return candidate if candidate in diagnostic else name


def _metric_value(value: Any) -> tuple[Any, str]:
    if not isinstance(value, Mapping):
        return value, UNDECLARED
    display = next(
        (value[key] for key in ("value", "rate", "mean") if key in value),
        UNDECLARED,
    )
    direction = next(
        (str(item) for key, item in value.items()
         if str(key).startswith("direction_favoring")),
        UNDECLARED,
    )
    return display, direction


def _coverage(
    block: Mapping[str, Any], metric: str, value: Any,
) -> dict[str, Any]:
    candidates: list[Any] = []
    if isinstance(value, Mapping):
        candidates.extend((value.get("coverage"), value))
    by_metric = block.get("coverage_by_metric")
    if isinstance(by_metric, Mapping):
        candidates.append(by_metric.get(metric))
    common = block.get("coverage")
    if isinstance(common, Mapping):
        candidates.extend((common.get(metric), common))
    for candidate in candidates:
        if isinstance(candidate, Mapping) and (
            "resolved" in candidate or "censored" in candidate
        ):
            return {
                "resolved": candidate.get("resolved", UNDECLARED),
                "censored": candidate.get("censored", UNDECLARED),
            }
    return {"resolved": UNDECLARED, "censored": UNDECLARED}


def _is_rate(name: str) -> bool:
    lowered = name.lower()
    return lowered == "rate" or lowered.endswith("_rate") or "_rate_" in lowered
