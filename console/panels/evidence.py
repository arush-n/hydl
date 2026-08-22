"""Declared evidence contracts from any artifact-producing lane.

The console deliberately does not know what a duel, block, planner, or model
means.  It reads six top-level fields that independently appeared in artifacts
from very different lanes::

    schema, status, claim, gates, limitations, next_required_evidence

Strings are shown verbatim.  ``gates`` is the sole machine-readable verdict:
named booleans become pass/fail chips and any explicit ``false`` makes the
artifact amber.  Missing values remain ``undeclared``.  Limitations are
disclosure, not failure, and therefore never change a card's tone.

This module is also the small public helper for agent-owned Evidence frames.
An agent only supplies paths; the normalization and six-row rendering stay
generic and shared.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from console.core.evidence.evidence_contract import EVIDENCE_CONTRACT_FIELDS
from console.core.panels import panel

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_FIELDS = EVIDENCE_CONTRACT_FIELDS
UNDECLARED = "Not reported"
_IGNORED_PARTS = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "node_modules",
}
_CACHE: dict[Path, tuple[int, int, dict[str, Any]]] = {}


def read_contract(
    path: str | Path,
    *,
    lane: str | None = None,
    workspace: str | Path = ROOT,
    selection: str = "declared artifact",
) -> dict[str, Any]:
    """Read one JSON artifact without interpreting lane-specific semantics.

    The returned object keeps the raw six fields, records which were actually
    present, extracts declared identity hashes, and carries enough file
    provenance to render a useful card even when no contract has been adopted.
    Results are cached by file identity; live frames still see an overwritten
    artifact because both mtime and size participate in the cache key.
    """

    artifact = Path(path)
    try:
        stat = artifact.stat()
    except OSError as exc:
        return _missing_contract(
            artifact, lane=lane, workspace=Path(workspace), selection=selection,
            error=str(exc),
        )

    cache_key = artifact.resolve()
    cached = _CACHE.get(cache_key)
    if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
        result = copy.deepcopy(cached[2])
    else:
        result = _read_uncached(artifact, stat.st_mtime, Path(workspace))
        _CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, copy.deepcopy(result))

    result["lane"] = lane or result.get("lane") or _producer_name(artifact)
    result["selection"] = selection
    return result


def contract_payload(
    paths: Iterable[str | Path],
    *,
    lane: str,
    summary: str = "",
    selection: str = "lane-selected artifact",
    workspace: str | Path = ROOT,
) -> dict[str, Any]:
    """Build the standard Evidence-frame payload for lane-owned paths."""

    cards = [
        read_contract(
            path, lane=lane, workspace=workspace, selection=selection,
        )
        for path in paths
    ]
    return evidence_payload(cards, summary=summary)


def evidence_payload(
    cards: Iterable[dict[str, Any]], *, summary: str = "",
    group_by_lane: bool = False,
) -> dict[str, Any]:
    """Render normalized contracts as identical six-row detail sections."""

    materialized = list(cards)
    failed_gates = sum(card.get("gate_state") == "has_failure" for card in materialized)
    malformed = sum(bool(card.get("error") or card.get("contract_errors"))
                    for card in materialized)
    full = sum(card.get("adoption") == len(CONTRACT_FIELDS) for card in materialized)
    updated = max((float(card.get("mtime") or 0) for card in materialized), default=0)

    if malformed:
        frame_tone = "crit"
    elif failed_gates:
        frame_tone = "warn"
    else:
        # Incomplete adoption is expected and must not masquerade as success.
        frame_tone = "info"

    if not materialized:
        return {
            "view": "note",
            "text": "No JSON artifacts were found in the declared artifact roots.",
            "status": {"label": "no artifacts", "tone": "absent"},
            "count": 0,
        }

    sections = (
        _lane_sections(materialized)
        if group_by_lane else [contract_section(card) for card in materialized]
    )
    # A lane-owned card with one or two selected artifacts should read at a
    # glance.  The cross-lane frame stays compact, opening only declarations
    # that are complete or need attention; every other artifact remains one
    # click away and is still searchable by the generic frame toolbar.
    if not group_by_lane and len(sections) <= 2:
        for section in sections:
            section["open"] = True

    source_count = len({card.get("lane") or "artifact" for card in materialized})
    needs_review = failed_gates + malformed
    return {
        "view": "detail",
        "status": {
            "label": (
                f"{full} complete · {needs_review} need review"
                if group_by_lane else f"{full}/{len(materialized)} full contracts"
            ),
            "tone": frame_tone,
            "detail": (
                f"{failed_gates} file(s) report a failed check; "
                f"{malformed} file(s) are unreadable or invalid"
            ),
        },
        "summary": summary or (
            "Every section uses the same declared contract. Values are never "
            "re-derived from lane-specific metrics."
        ),
        "sections": sections,
        "updated_at": updated or None,
        "count": (
            f"{len(materialized)} files · {source_count} sources"
            if group_by_lane else f"{len(materialized)} artifact(s)"
        ),
        "tags": (
            ["newest file per source", "most complete report per source"]
            if group_by_lane else [
                {"label": f"{failed_gates} false-gate artifact(s)", "tone": "warn"}
                if failed_gates else {"label": "no declared false gates", "tone": "info"},
                f"{full} full",
            ]
        ),
        "note": (
            "Open a source, then a file. Missing information is shown as Not "
            "reported; it is not counted as a failed check."
            if group_by_lane else
            "Limitations are shown as disclosures and never treated as failed "
            "gates. An absent field is greyed as undeclared; the console does "
            "not infer it from filenames, metrics, or lane knowledge."
        ),
    }


def contract_section(card: dict[str, Any]) -> dict[str, Any]:
    """The canonical Claim · Status · Gates · Limits · Next · Identity card."""

    declared = card.get("declared") or {}
    schema = card.get("schema")

    if declared.get("claim"):
        claim = _display(card.get("claim"))
        claim_detail = ""
        claim_tone = ""
    elif declared.get("schema"):
        claim = _display(schema)
        claim_detail = "No claim reported; showing the report type instead"
        claim_tone = "absent"
    else:
        claim = UNDECLARED
        claim_detail = ""
        claim_tone = "absent"

    rows = [
        _contract_row("What it claims", claim, tone=claim_tone, detail=claim_detail),
        _contract_row(
            "Reported status",
            _display(card.get("status")) if declared.get("status") else UNDECLARED,
            tone="info" if declared.get("status") else "absent",
        ),
        _gates_row(card),
        _contract_row(
            "Limitations",
            _display(card.get("limitations"))
            if declared.get("limitations") else UNDECLARED,
            tone="" if declared.get("limitations") else "absent",
        ),
        _contract_row(
            "Next evidence",
            _display(card.get("next_required_evidence"))
            if declared.get("next_required_evidence") else UNDECLARED,
            tone="" if declared.get("next_required_evidence") else "absent",
        ),
        _identity_row(card, label="Technical identity"),
    ]

    status = (_display(card.get("status")) if declared.get("status")
              else "status undeclared")
    description = (
        f"{card.get('path', card.get('name', 'artifact'))} · "
        f"{card.get('adoption', 0)}/{len(CONTRACT_FIELDS)} fields declared · "
        f"{card.get('selection', 'declared artifact')} · "
        f"modified {_iso_time(card.get('mtime'))}"
    )
    if card.get("error"):
        description += f" · unreadable: {card['error']}"
    elif card.get("contract_errors"):
        description += " · " + "; ".join(card["contract_errors"])

    return {
        "title": f"{card.get('lane') or 'artifact'} · {card.get('name') or 'unknown'}",
        "description": description,
        "badge": status,
        "tone": card.get("tone") or "absent",
        "view": "rows",
        "size": "full",
        "collapsible": True,
        "open": (
            card.get("adoption") == len(CONTRACT_FIELDS)
            or card.get("gate_state") in {"malformed", "has_failure"}
        ),
        "columns": ["Report field", "What the file says"],
        "rows": rows,
    }


def _lane_sections(cards: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group cross-workspace results without changing artifact discovery."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        grouped[str(card.get("lane") or "artifact")].append(card)

    sections = []
    for lane, lane_cards in grouped.items():
        failed = sum(card.get("gate_state") == "has_failure" for card in lane_cards)
        malformed = sum(
            bool(card.get("error") or card.get("contract_errors"))
            for card in lane_cards
        )
        full = sum(
            card.get("adoption") == len(CONTRACT_FIELDS) for card in lane_cards
        )
        if malformed:
            tone, badge = "crit", "invalid report"
        elif failed:
            tone, badge = "warn", "failed check"
        elif full:
            tone, badge = "good", f"{full} complete"
        else:
            tone, badge = "info", "incomplete reports"
        count = len(lane_cards)
        sections.append({
            "title": lane.replace("/", " / "),
            "description": (
                f"{count} selected file{'s' if count != 1 else ''} · "
                "newest and most complete are separate when needed"
            ),
            "badge": badge,
            "tone": tone,
            "view": "rows",
            "size": "full",
            "collapsible": True,
            "open": False,
            "columns": ["File", "Claim / status", "Checks", "Next evidence"],
            "rows": [_artifact_row(card) for card in lane_cards],
        })
    return sections


def _artifact_row(card: dict[str, Any]) -> dict[str, Any]:
    """Compact cross-workspace summary; the six-field reader remains shared."""

    declared = card.get("declared") or {}
    claim = (
        _display(card.get("claim")) if declared.get("claim")
        else _display(card.get("schema")) if declared.get("schema")
        else UNDECLARED
    )
    status = (
        _display(card.get("status")) if declared.get("status") else UNDECLARED
    )
    limitations = (
        _display(card.get("limitations"))
        if declared.get("limitations") else UNDECLARED
    )
    next_evidence = (
        _display(card.get("next_required_evidence"))
        if declared.get("next_required_evidence") else UNDECLARED
    )
    check_cell = copy.deepcopy(_gates_row(card)["cells"][1])
    check_cell["grow"] = False
    check_cell["wrap"] = True
    return {
        "tone": card.get("tone") or "absent",
        "badge": f"{int(card.get('adoption') or 0)}/{len(CONTRACT_FIELDS)} fields",
        "cells": [
            {
                "value": f"{_selection_label(card.get('selection'))} · "
                         f"{card.get('name') or 'unknown'}",
                "detail": (
                    f"{card.get('path', card.get('name', 'artifact'))} · "
                    f"modified {_iso_time(card.get('mtime'))}"
                ),
                "mono": False,
                "wrap": True,
            },
            {
                "value": claim,
                "detail": f"Status: {status} · Limitations: {limitations}",
                "mono": False,
                "grow": True,
                "wrap": True,
            },
            check_cell,
            {
                "value": next_evidence, "mono": False,
                "grow": True, "wrap": True,
            },
        ],
    }


def _selection_label(value: Any) -> str:
    labels = {
        "newest JSON by file mtime": "Newest file",
        "fullest six-field declaration": "Most complete report",
    }
    return labels.get(str(value), "Selected file")


def discover_evidence(
    workspace: str | Path = ROOT, *, max_per_lane: int = 2,
) -> list[dict[str, Any]]:
    """Find evidence without a registry of architectures or lane semantics.

    Artifact roots are structural conventions already present in the repo:
    ``agents/*/{artifacts,benchmarks}``,
    ``experimental/*/{artifacts,results}``, and the two shared ``artifacts``
    trees.  For each filesystem producer we show the newest JSON and, when it
    differs, the JSON declaring the most of the six-field contract.  Both
    selections are named on the card so neither is silently called canonical.
    """

    root = Path(workspace)
    groups = _artifact_groups(root)
    selected: list[dict[str, Any]] = []
    for lane, paths in sorted(groups.items()):
        if not paths:
            continue
        cards = [read_contract(path, lane=lane, workspace=root) for path in paths]
        newest = max(cards, key=lambda card: float(card.get("mtime") or 0))
        newest["selection"] = "newest JSON by file mtime"
        lane_cards = [newest]

        declared_cards = [card for card in cards if card.get("adoption", 0) > 0]
        if declared_cards and max_per_lane > 1:
            fullest = max(
                declared_cards,
                key=lambda card: (
                    int(card.get("adoption") or 0),
                    float(card.get("mtime") or 0),
                ),
            )
            if fullest.get("resolved_path") != newest.get("resolved_path"):
                fullest["selection"] = "fullest six-field declaration"
                lane_cards.append(fullest)

        selected.extend(lane_cards[:max(1, max_per_lane)])

    return selected


@panel(
    "cross-lane-evidence",
    title="Artifact reports",
    tab="evidence",
    order=10,
    agent="built-in",
    refresh=30.0,
    size="wide",
    tags=("artifacts", "checks"),
    description="Files produced by runs and experiments, grouped by their source.",
)
def evidence() -> dict[str, Any]:
    cards = discover_evidence()
    return evidence_payload(
        cards,
        group_by_lane=True,
        summary=(
            "Open a source to see its newest file and its most complete report. "
            "These can be different files; neither is treated as canonical."
        ),
    )


def _read_uncached(path: Path, mtime: float, workspace: Path) -> dict[str, Any]:
    error = ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raw = {}
        error = str(exc)
    if not error and not isinstance(raw, dict):
        raw = {}
        error = "top-level JSON value is not an object"

    declared = {
        field: field in raw and raw[field] is not None
        for field in CONTRACT_FIELDS
    }
    gates = raw.get("gates") if declared["gates"] else None
    contract_errors: list[str] = []
    if declared["gates"] and not isinstance(gates, dict):
        contract_errors.append("gates is declared but is not an object")
    elif isinstance(gates, dict):
        non_boolean = [name for name, value in gates.items()
                       if not isinstance(value, bool)]
        if non_boolean:
            contract_errors.append(
                "non-boolean gates: " + ", ".join(map(str, non_boolean))
            )

    if error or contract_errors:
        gate_state = "malformed"
        tone = "crit"
    elif not declared["gates"]:
        gate_state = "undeclared"
        tone = "absent"
    elif any(value is False for value in gates.values()):
        gate_state = "has_failure"
        tone = "warn"
    elif gates:
        gate_state = "all_pass"
        tone = "good"
    else:
        gate_state = "declared_empty"
        tone = "info"

    try:
        relative = path.resolve().relative_to(workspace.resolve())
        display_path = relative.as_posix()
    except (OSError, ValueError):
        display_path = str(path)

    return {
        "name": path.name,
        "path": display_path,
        "resolved_path": str(path.resolve()),
        "mtime": mtime,
        "size": path.stat().st_size,
        "error": error,
        "contract_errors": contract_errors,
        "declared": declared,
        "adoption": sum(declared.values()),
        "schema": raw.get("schema") if declared["schema"] else None,
        "status": raw.get("status") if declared["status"] else None,
        "claim": raw.get("claim") if declared["claim"] else None,
        "gates": gates,
        "limitations": (
            raw.get("limitations") if declared["limitations"] else None
        ),
        "next_required_evidence": (
            raw.get("next_required_evidence")
            if declared["next_required_evidence"] else None
        ),
        "identities": _identity_entries(raw),
        "gate_state": gate_state,
        "tone": tone,
    }


def _missing_contract(
    path: Path, *, lane: str | None, workspace: Path, selection: str, error: str,
) -> dict[str, Any]:
    try:
        display_path = path.resolve().relative_to(workspace.resolve()).as_posix()
    except (OSError, ValueError):
        display_path = str(path)
    return {
        "lane": lane or _producer_name(path),
        "selection": selection,
        "name": path.name,
        "path": display_path,
        "resolved_path": str(path.resolve()),
        "mtime": 0.0,
        "size": 0,
        "error": error,
        "contract_errors": [],
        "declared": {field: False for field in CONTRACT_FIELDS},
        "adoption": 0,
        "schema": None,
        "status": None,
        "claim": None,
        "gates": None,
        "limitations": None,
        "next_required_evidence": None,
        "identities": [],
        "gate_state": "malformed",
        "tone": "crit",
    }


def _identity_entries(raw: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[tuple[str, Any]] = []
    if raw.get("schema") is not None:
        entries.append(("schema", raw["schema"]))
    if raw.get("version") is not None:
        entries.append(("version", raw["version"]))

    contracts = raw.get("contracts")
    if contracts is not None:
        _flatten_identity(contracts, "contracts", entries, depth=0)
    # Evidence artifacts can contain tens of thousands of rollout rows.  Hash
    # discovery is provenance garnish, not permission to walk an unbounded
    # result tree on every dashboard refresh, so the generic recursive search
    # has an explicit node budget.  The declared ``contracts`` block above is
    # always read in full first.
    _find_hashes(raw, "", entries, depth=0, budget=[2_000])

    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for label, value in entries:
        rendered = _display(value)
        key = (label, rendered)
        if key in seen:
            continue
        seen.add(key)
        found.append({
            "label": label,
            "value": rendered,
            "short": _short_identity(rendered),
        })
        if len(found) >= 24:
            break
    return found


def _flatten_identity(
    value: Any, prefix: str, entries: list[tuple[str, Any]], *, depth: int,
) -> None:
    if depth >= 4 or len(entries) >= 24:
        entries.append((prefix, value))
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten_identity(nested, f"{prefix}.{key}", entries, depth=depth + 1)
    else:
        entries.append((prefix, value))


def _find_hashes(
    value: Any,
    prefix: str,
    entries: list[tuple[str, Any]],
    *,
    depth: int,
    budget: list[int],
) -> None:
    budget[0] -= 1
    if budget[0] < 0 or depth >= 6 or len(entries) >= 48:
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            if budget[0] < 0:
                break
            label = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower().endswith("_sha256"):
                entries.append((label, nested))
            else:
                _find_hashes(
                    nested, label, entries, depth=depth + 1, budget=budget,
                )
    elif isinstance(value, list):
        for index, nested in enumerate(value[:8]):
            if budget[0] < 0:
                break
            _find_hashes(
                nested, f"{prefix}[{index}]", entries,
                depth=depth + 1, budget=budget,
            )


def _artifact_groups(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, set[Path]] = defaultdict(set)

    agents = root / "agents"
    if agents.is_dir():
        for owner in agents.iterdir():
            if not owner.is_dir() or owner.name.startswith(".") or owner.name == "profiles":
                continue
            for folder in (owner / "artifacts", owner / "benchmarks"):
                for artifact in _json_files(folder):
                    groups[owner.name].add(artifact)

    experimental = root / "experimental"
    if experimental.is_dir():
        for owner in experimental.iterdir():
            if not owner.is_dir() or owner.name.startswith("."):
                continue
            for folder in (owner / "artifacts", owner / "results"):
                for artifact in _json_files(folder):
                    groups[owner.name].add(artifact)

    for label, folder in (
        ("workspace", root / "artifacts"),
        ("hytalerl", root / "HytaleRL" / "artifacts"),
    ):
        for artifact in _json_files(folder):
            try:
                first = artifact.relative_to(folder).parts[0]
            except ValueError:
                first = artifact.parent.name
            # This is a filesystem grouping, not a semantic lane registry.
            family = first.split("-", 1)[0].split("_", 1)[0]
            groups[f"{label}/{family}"].add(artifact)

    return {lane: sorted(paths) for lane, paths in groups.items() if paths}


def _json_files(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        return ()
    return (
        path for path in folder.rglob("*.json")
        if not any(part in _IGNORED_PARTS for part in path.parts)
    )


def _producer_name(path: Path) -> str:
    parts = path.parts
    for marker in ("agents", "experimental"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return path.parent.name or "artifact"


def _gates_row(card: dict[str, Any]) -> dict[str, Any]:
    declared = card.get("declared") or {}
    gates = card.get("gates")
    if not declared.get("gates"):
        return _contract_row("Checks", UNDECLARED, tone="absent")
    if not isinstance(gates, dict):
        return _contract_row("Checks", _display(gates), tone="crit",
                             detail="Reported checks must be an object")
    if not gates:
        return _contract_row("Checks", "Reported with no named checks", tone="info")

    chips = []
    for name, value in gates.items():
        if value is True:
            chips.append({"label": f"{name}: pass", "tone": "good"})
        elif value is False:
            chips.append({"label": f"{name}: fail", "tone": "warn"})
        else:
            chips.append({"label": f"{name}: {_display(value)}", "tone": "crit"})
    return {
        "tone": card.get("tone"),
        "cells": [
            {"value": "Checks", "mono": False, "grow": False},
            {"value": "", "mono": False, "chips": chips,
             "grow": True, "wrap": True},
        ],
    }


def _identity_row(
    card: dict[str, Any], *, label: str = "Identity",
) -> dict[str, Any]:
    identities = card.get("identities") or []
    if not identities:
        return _contract_row(label, UNDECLARED, tone="absent")
    text = " · ".join(
        f"{entry['label']}: {entry['short']}" for entry in identities
    )
    detail = "\n".join(
        f"{entry['label']}: {entry['value']}" for entry in identities
        if entry["short"] != entry["value"]
    )
    return _contract_row(label, text, detail=detail)


def _contract_row(
    label: str, value: Any, *, tone: str = "", detail: str = "",
) -> dict[str, Any]:
    return {
        "tone": tone,
        "cells": [
            {"value": label, "mono": False, "grow": False},
            {
                "value": _display(value),
                "mono": False,
                "tone": tone,
                "detail": detail,
                "grow": True,
                "wrap": True,
            },
        ],
    }


def _display(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return UNDECLARED
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return " • ".join(_display(item) for item in value) if value else "[]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _short_identity(value: str) -> str:
    compact = value.strip()
    if len(compact) >= 32 and all(character in "0123456789abcdefABCDEF"
                                  for character in compact):
        return compact[:12] + "…"
    return compact


def _iso_time(value: Any) -> str:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return UNDECLARED
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(
        timespec="seconds",
    )
