"""Field-level diff between two versions' table directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value and prefix:
            return {prefix: {}}
        out: dict[str, Any] = {}
        for key in sorted(value):
            out.update(flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        if not value and prefix:
            return {prefix: []}
        out = {}
        for index, child in enumerate(value):
            out.update(flatten(child, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def same(a: Any, b: Any) -> bool:
    """Value equality where 6 == 6.0 but True != 1."""

    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def diff_rows(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed: dict[str, dict[str, list[Any]]] = {}
    moved: dict[str, list[Any]] = {}
    for key in sorted(set(old) & set(new)):
        a, b = flatten(old[key]), flatten(new[key])
        fields = {
            path: [a.get(path), b.get(path)]
            for path in sorted(set(a) | set(b))
            if path not in a or path not in b or not same(a[path], b[path])
        }
        if set(fields) == {"source"}:
            moved[key] = fields["source"]
        elif fields:
            changed[key] = fields
    return {"added": added, "removed": removed, "changed": changed, "moved": moved}


def load_tables(directory: str | Path) -> dict[str, dict[str, Any]]:
    tables = {}
    for path in sorted(Path(directory).glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "family" in document and "rows" in document:
            tables[document["family"]] = document
    return tables


def diff_directories(old_dir: str | Path, new_dir: str | Path) -> dict[str, Any]:
    old, new = load_tables(old_dir), load_tables(new_dir)
    report: dict[str, Any] = {"families": {}}
    for family in sorted(set(old) | set(new)):
        if family not in old or family not in new:
            report["families"][family] = {"missing_in": "old" if family not in old else "new"}
            continue
        report["families"][family] = {
            "old_game_version": old[family]["game_version"],
            "new_game_version": new[family]["game_version"],
            **diff_rows(old[family]["rows"], new[family]["rows"]),
        }
    return report


def markdown(report: Mapping[str, Any], *, per_family: int = 40) -> str:
    lines = ["# Content diff", ""]
    lines.append("| Family | Added | Removed | Changed | Moved only |")
    lines.append("|---|---:|---:|---:|---:|")
    for family, entry in report["families"].items():
        if "missing_in" in entry:
            lines.append(f"| {family} | (missing in {entry['missing_in']}) | | | |")
            continue
        lines.append(
            f"| {family} | {len(entry['added'])} | {len(entry['removed'])} | {len(entry['changed'])} "
            f"| {len(entry['moved'])} |"
        )
    for family, entry in report["families"].items():
        if "missing_in" in entry or not entry["changed"]:
            continue
        lines += ["", f"## {family}", ""]
        for row in list(entry["changed"])[:per_family]:
            for path, (a, b) in list(entry["changed"][row].items())[:12]:
                lines.append(f"- `{row}` `{path}`: {json.dumps(a)} -> {json.dumps(b)}")
        hidden = len(entry["changed"]) - per_family
        if hidden > 0:
            lines.append(f"- ... {hidden} more changed rows (see the JSON report)")
    return "\n".join(lines) + "\n"
