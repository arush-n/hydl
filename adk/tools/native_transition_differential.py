"""Fail-closed comparison of paired one-step native transition evidence.

The comparator consumes JSON-compatible data only.  It never starts a server,
steps JAX, or interprets an unavailable value as agreement.

The input schema is ``hytalerl_native_matched_transition_v1``::

    {
      "schema": "hytalerl_native_matched_transition_v1",
      "schema_version": 1,
      "identity": {
        "pair_id": "episode-000/step-000",
        "sequence": 0,
        "reference": {"id": "server", "provenance": {"run": "..."}},
        "candidate": {"id": "jax", "provenance": {"run": "..."}}
      },
      "required_fields": ["initial.observation", "next.observation"],
      "tolerances": {"initial.observation": {"atol": 1e-6, "rtol": 0.0}},
      "initial": {
        "observation": {"reference": [...], "candidate": [...]},
        "mask": {"reference": [...], "candidate": [...]}
      },
      "actions": {
        "requested": {"status": "exact", "reference": ..., "candidate": ...},
        "issued": {"status": "unsupported", "gap": "no receipt"},
        "applied": {"status": "exact", "reference": ..., "candidate": ...}
      },
      "next": {"observation": {...}, "mask": {...}},
      "state": {"position": {...}, "health": {...}},
      "reward": {...},
      "terminated": {...},
      "truncated": {...},
      "unsupported": {"state.velocity": "not captured"}
    }

Each pair record may declare one of ``exact``, ``approximate``,
``unsupported``, or ``invalid``.  A declared unsupported/invalid record is
never upgraded by the comparator.  For ordinary records, the comparator
derives the status from the two values and the field tolerance.  Action
records must declare a status when they are present; an omitted action record
is represented in the result as unsupported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any


SCHEMA = "hytalerl_native_matched_transition_v1"
RESULT_SCHEMA = "hytalerl_native_matched_transition_result_v1"
SCHEMA_VERSION = 1
STATUS_LABELS = frozenset({"exact", "approximate", "unsupported", "invalid"})

COMPARISON_FIELDS = (
    "initial.observation",
    "initial.mask",
    "actions.requested",
    "actions.issued",
    "actions.applied",
    "next.observation",
    "next.mask",
    "reward",
    "terminated",
    "truncated",
)
_SECTION_FIELDS = {
    "initial": frozenset({"observation", "mask"}),
    "next": frozenset({"observation", "mask"}),
    "actions": frozenset({"requested", "issued", "applied"}),
}


class TransitionArtifactError(ValueError):
    """Structural or evidence validation failure with a logical field path."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _error(path: str, reason: str) -> None:
    raise TransitionArtifactError(path, reason)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if _is_number(value):
        if isinstance(value, float) and not math.isfinite(value):
            _error(path, "nonfinite numeric value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _error(path, "object keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    _error(path, f"value is not JSON-compatible ({type(value).__name__})")


def _known_field(path: str) -> bool:
    return path in COMPARISON_FIELDS or (
        path.startswith("state.") and len(path) > len("state.")
    )


def _validate_identity(identity: Any) -> None:
    path = "identity"
    if not isinstance(identity, Mapping):
        _error(path, "must be an object")
    if not isinstance(identity.get("pair_id"), str) or not identity["pair_id"]:
        _error(f"{path}.pair_id", "must be a non-empty string")
    sequence = identity.get("sequence")
    if not _is_integer(sequence) or sequence < 0:
        _error(f"{path}.sequence", "must be a non-negative integer")
    for side in ("reference", "candidate"):
        side_path = f"{path}.{side}"
        value = identity.get(side)
        if not isinstance(value, Mapping):
            _error(side_path, "must be an object")
        if not isinstance(value.get("id"), str) or not value["id"]:
            _error(f"{side_path}.id", "must be a non-empty string")
        if not isinstance(value.get("provenance"), Mapping) or not value[
            "provenance"
        ]:
            _error(f"{side_path}.provenance", "must be a non-empty object")
        if "sequence" in value:
            if not _is_integer(value["sequence"]) or value["sequence"] != sequence:
                _error(
                    f"{side_path}.sequence",
                    "must equal identity.sequence",
                )


def _validate_status(value: Any, path: str) -> None:
    if not isinstance(value, str) or value not in STATUS_LABELS:
        _error(path, f"must be one of {sorted(STATUS_LABELS)}")


def _validate_pair(value: Any, path: str, *, action: bool = False) -> None:
    if not isinstance(value, Mapping):
        _error(path, "must be an object")
    if action and "status" not in value:
        _error(f"{path}.status", "action evidence must declare a status")
    if "status" in value:
        _validate_status(value["status"], f"{path}.status")
    if "gap" in value and (
        not isinstance(value["gap"], str) or not value["gap"]
    ):
        _error(f"{path}.gap", "must be a non-empty string")
    declared = value.get("status")
    if declared not in {"unsupported", "invalid"}:
        missing = [
            side for side in ("reference", "candidate") if side not in value
        ]
        if missing:
            _error(path, "missing " + ", ".join(missing) + " value")


def _validate_unsupported(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, list):
        result: dict[str, str] = {}
        for index, path in enumerate(value):
            if not isinstance(path, str) or not path:
                _error(f"unsupported[{index}]", "must be a non-empty field path")
            if not _known_field(path):
                _error(f"unsupported[{index}]", "unknown field path")
            result[path] = "declared unsupported"
        return result
    if not isinstance(value, Mapping):
        _error("unsupported", "must be an object or list of field paths")
    result = {}
    for path, reason in value.items():
        if not isinstance(path, str) or not path or not _known_field(path):
            _error("unsupported", f"unknown field path {path!r}")
        if not isinstance(reason, str) or not reason:
            _error(f"unsupported.{path}", "must be a non-empty reason")
        result[path] = reason
    return result


def _validate_tolerances(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        _error("tolerances", "must be an object")
    for path, tolerance in value.items():
        if not isinstance(path, str) or not _known_field(path):
            _error("tolerances", f"unknown field path {path!r}")
        if _is_number(tolerance):
            if float(tolerance) < 0.0 or not math.isfinite(float(tolerance)):
                _error(f"tolerances.{path}", "must be finite and non-negative")
            continue
        if not isinstance(tolerance, Mapping):
            _error(f"tolerances.{path}", "must be a number or object")
        unknown = set(tolerance) - {"atol", "rtol", "absolute", "relative"}
        if unknown:
            _error(f"tolerances.{path}", f"unknown keys {sorted(unknown)}")
        for name in ("atol", "rtol", "absolute", "relative"):
            if name in tolerance:
                amount = tolerance[name]
                if (
                    not _is_number(amount)
                    or not math.isfinite(float(amount))
                    or float(amount) < 0.0
                ):
                    _error(
                        f"tolerances.{path}.{name}",
                        "must be finite and non-negative",
                    )


def validate_transition_artifact(artifact: Any) -> None:
    """Raise ``TransitionArtifactError`` unless *artifact* is valid JSON evidence."""

    _validate_json_value(artifact, "artifact")
    if not isinstance(artifact, Mapping):
        _error("artifact", "must be an object")
    if artifact.get("schema") != SCHEMA:
        _error("schema", f"must equal {SCHEMA!r}")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        _error("schema_version", f"must equal {SCHEMA_VERSION}")
    _validate_identity(artifact.get("identity"))

    required = artifact.get("required_fields")
    if not isinstance(required, list) or not required:
        _error("required_fields", "must be a non-empty list")
    if any(not isinstance(path, str) or not path for path in required):
        _error("required_fields", "must contain non-empty strings")
    if len(set(required)) != len(required):
        _error("required_fields", "must not contain duplicates")
    for path in required:
        if not _known_field(path):
            _error("required_fields", f"unknown field path {path!r}")

    _validate_tolerances(artifact.get("tolerances"))
    _validate_unsupported(artifact.get("unsupported"))

    for section, allowed in _SECTION_FIELDS.items():
        if section not in artifact:
            continue
        value = artifact[section]
        if not isinstance(value, Mapping):
            _error(section, "must be an object")
        unknown = set(value) - allowed
        if unknown and section != "actions":
            _error(section, f"unknown keys {sorted(unknown)}")
        if unknown and section == "actions":
            _error(section, f"unknown keys {sorted(unknown)}")
        for name in allowed:
            if name in value:
                _validate_pair(
                    value[name],
                    f"{section}.{name}",
                    action=section == "actions",
                )

    if "state" in artifact:
        state = artifact["state"]
        if not isinstance(state, Mapping):
            _error("state", "must be an object")
        for name, value in state.items():
            if not isinstance(name, str) or not name:
                _error("state", "field names must be non-empty strings")
            _validate_pair(value, f"state.{name}")

    for name in ("reward", "terminated", "truncated"):
        if name in artifact:
            _validate_pair(artifact[name], name)


def _tolerance(artifact: Mapping[str, Any], path: str) -> tuple[float, float]:
    raw = artifact.get("tolerances", {}).get(path, 0.0)
    if _is_number(raw):
        return float(raw), 0.0
    if not isinstance(raw, Mapping):
        return 0.0, 0.0
    atol = raw.get("atol", raw.get("absolute", 0.0))
    rtol = raw.get("rtol", raw.get("relative", 0.0))
    return float(atol), float(rtol)


def _value_comparison(
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float,
) -> tuple[str, float | None, str | None]:
    if reference is None or candidate is None:
        return "unsupported", None, "unavailable value"
    if _is_number(reference) and _is_number(candidate):
        difference = abs(float(candidate) - float(reference))
        limit = atol + rtol * abs(float(reference))
        if reference == candidate:
            return "exact", 0.0, None
        if difference <= limit:
            return "approximate", difference, None
        return "invalid", difference, "outside supplied tolerance"
    if isinstance(reference, bool) or isinstance(candidate, bool):
        if isinstance(reference, bool) and isinstance(candidate, bool):
            return ("exact", 0.0, None) if reference == candidate else (
                "invalid",
                1.0,
                "boolean mismatch",
            )
        return "invalid", None, "value types differ"
    if isinstance(reference, list) or isinstance(candidate, list):
        if not isinstance(reference, list) or not isinstance(candidate, list):
            return "invalid", None, "value types differ"
        if len(reference) != len(candidate):
            return "invalid", None, "sequence lengths differ"
        statuses = [
            _value_comparison(left, right, atol=atol, rtol=rtol)
            for left, right in zip(reference, candidate, strict=True)
        ]
        return _merge_comparisons(statuses)
    if isinstance(reference, Mapping) or isinstance(candidate, Mapping):
        if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
            return "invalid", None, "value types differ"
        if set(reference) != set(candidate):
            return "invalid", None, "object keys differ"
        statuses = [
            _value_comparison(reference[key], candidate[key], atol=atol, rtol=rtol)
            for key in reference
        ]
        return _merge_comparisons(statuses)
    if type(reference) is not type(candidate):
        return "invalid", None, "value types differ"
    if reference == candidate:
        return "exact", 0.0, None
    return "invalid", None, "values differ"


def _merge_comparisons(
    comparisons: Sequence[tuple[str, float | None, str | None]],
) -> tuple[str, float | None, str | None]:
    if any(status == "unsupported" for status, _, _ in comparisons):
        return "unsupported", None, "nested value unavailable"
    invalid = next(
        (comparison for comparison in comparisons if comparison[0] == "invalid"),
        None,
    )
    if invalid is not None:
        return invalid
    if any(status == "approximate" for status, _, _ in comparisons):
        errors = [error for _, error, _ in comparisons if error is not None]
        return "approximate", max(errors, default=0.0), None
    return "exact", 0.0, None


def _field_record(artifact: Mapping[str, Any], path: str) -> Any:
    if path.startswith("state."):
        return artifact.get("state", {}).get(path[6:])
    if path in {"reward", "terminated", "truncated"}:
        return artifact.get(path)
    section, name = path.split(".", 1)
    return artifact.get(section, {}).get(name)


def _comparison_paths(artifact: Mapping[str, Any], unsupported: Mapping[str, str]) -> list[str]:
    paths = list(COMPARISON_FIELDS)
    state = artifact.get("state", {})
    if isinstance(state, Mapping):
        paths.extend(f"state.{name}" for name in state)
    for path in unsupported:
        if path not in paths:
            paths.append(path)
    return paths


def _field_result(
    artifact: Mapping[str, Any],
    path: str,
    unsupported: Mapping[str, str],
) -> dict[str, Any]:
    required = path in artifact["required_fields"]
    atol, rtol = _tolerance(artifact, path)
    record = _field_record(artifact, path)
    declared = record.get("status") if isinstance(record, Mapping) else None
    if path in unsupported:
        status, error, gap = "unsupported", None, unsupported[path]
    elif record is None:
        status, error, gap = "unsupported", None, "evidence record is absent"
    elif declared == "unsupported":
        status, error, gap = "unsupported", None, record.get(
            "gap", "evidence declared unsupported"
        )
    elif declared == "invalid":
        status, error, gap = "invalid", None, record.get(
            "gap", "evidence declared invalid"
        )
    elif "reference" not in record or "candidate" not in record:
        status, error, gap = "unsupported", None, "paired value is absent"
    else:
        status, error, gap = _value_comparison(
            record["reference"],
            record["candidate"],
            atol=atol,
            rtol=rtol,
        )
    result: dict[str, Any] = {
        "required": required,
        "status": status,
        "match": status in {"exact", "approximate"},
        "atol": atol,
        "rtol": rtol,
    }
    if declared is not None:
        result["declared_status"] = declared
    if error is not None:
        result["maximum_absolute_error"] = error
    if gap is not None:
        result["gap"] = gap
    return result


def _logical_error_field(path: str) -> str:
    """Reduce a structural error to the comparable field it invalidates."""

    if path.startswith("artifact."):
        path = path[len("artifact.") :]
    parts = path.split(".")
    if len(parts) >= 2 and parts[0] in {"initial", "next", "actions"}:
        return ".".join(parts[:2])
    if len(parts) >= 2 and parts[0] == "state":
        return ".".join(parts[:2])
    if parts[0] in {"reward", "terminated", "truncated"}:
        return parts[0]
    return "artifact"


def compare_transition_artifact(artifact: Any) -> dict[str, Any]:
    """Return a JSON-compatible fail-closed comparison report.

    ``passed`` is true only when the artifact validates and every path in
    ``required_fields`` has derived status ``exact`` or ``approximate``.
    Optional unsupported fields remain visible in ``fields`` and ``gaps`` but
    do not block the result.
    """

    try:
        validate_transition_artifact(artifact)
    except TransitionArtifactError as failure:
        required = []
        if isinstance(artifact, Mapping) and isinstance(
            artifact.get("required_fields"), list
        ):
            required = [
                path
                for path in artifact["required_fields"]
                if isinstance(path, str)
            ]
        path = _logical_error_field(failure.path)
        field = {
            "required": path in required,
            "status": "invalid",
            "match": False,
            "gap": failure.reason,
        }
        return {
            "schema": RESULT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "schema_valid": False,
            "passed": False,
            "required_fields": required,
            "fields": {path: field},
            "gaps": [
                {
                    "field": path,
                    "required": path in required,
                    "status": "invalid",
                    "reason": failure.reason,
                }
            ],
        }

    assert isinstance(artifact, Mapping)
    unsupported = _validate_unsupported(artifact.get("unsupported"))
    fields = {
        path: _field_result(artifact, path, unsupported)
        for path in _comparison_paths(artifact, unsupported)
    }
    gaps = [
        {
            "field": path,
            "required": result["required"],
            "status": result["status"],
            "reason": result["gap"],
        }
        for path, result in fields.items()
        if result["status"] not in {"exact", "approximate"}
    ]
    passed = all(
        not result["required"] or result["status"] in {"exact", "approximate"}
        for result in fields.values()
    )
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "schema_valid": True,
        "passed": passed,
        "identity": artifact["identity"],
        "required_fields": list(artifact["required_fields"]),
        "fields": fields,
        "gaps": gaps,
    }


validate = validate_transition_artifact
compare = compare_transition_artifact


__all__ = [
    "COMPARISON_FIELDS",
    "RESULT_SCHEMA",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STATUS_LABELS",
    "TransitionArtifactError",
    "compare",
    "compare_transition_artifact",
    "validate",
    "validate_transition_artifact",
]
