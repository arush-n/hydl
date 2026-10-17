"""Every place the committed gen-5 source reads the engine step duration.

Owner ruling (2026-10-17): the agent operates at 30 Hz and the step duration is
content: gen 5 draws a per-lane timing profile (nominal 1/30 s, loaded 0.045 s
on alternating ticks), gen 7 sets one value, 1/30 s. A render must redirect each
site below. Sources are read with ``git show HEAD:<path>`` and tokenized, so
comments and docstrings never count; string tokens count only when they are a
timing key (a dict key or subscript such as ``"loaded_delta_seconds"``).

Kinds: ``loaded_tick_literal`` (0.045), ``nominal_tick_literal`` (1/30,
1.0/30, 1.0 / 30.0), ``per_tick_division`` (x / 30), ``tick_rate_multiplication``
(x * 30), ``delta_field`` (a per-lane or per-call delta: ``*delta_seconds``,
``motion_delta``, ``nominal_dt``, ``loaded_dt``, ...), ``tick_rate_constant``
(``*ticks_per_second``), ``timing_key`` (a ruleset key string),
``ruleset_json`` (the packaged ruleset's engine block). Scope ``engine`` is the
package a render copies; ``consumer`` is tracked code that drives it
(``arena/``, ``worlds/``, ``HytaleRL/examples/``).
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import tokenize
from pathlib import Path

PACKAGE = "HytaleRL/hytalegym/hytalegym/"
EXCLUDE = ("HytaleRL/hytalegym/hytalegym/sim/",)
# Engine = the package a render copies; consumers = tracked code that drives it.
SCOPES = {"engine": (PACKAGE,), "consumer": ("arena/", "worlds/", "HytaleRL/examples/")}
_TICK_RATE_NAME = re.compile(r"(?i)ticks_per_second")
_DELTA_NAME = re.compile(
    r"(delta_seconds|motion_delta|nominal_dt|loaded_dt|delta_s\b|tick_seconds|dt_seconds|seconds_per_tick)",
)
_TIMING_KEYS = {
    "nominal_delta_seconds",
    "loaded_delta_seconds",
    "motion_timing_profile_count",
    "ticks_per_second",
    "delta_seconds",
    "dt_seconds",
}
RULESET_JSON = "HytaleRL/hytalegym/hytalegym/rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _number(text: str) -> float | None:
    try:
        return float(text.rstrip("jJ"))
    except ValueError:
        return None


def _symbols(source: str) -> list[tuple[int, int, str]]:
    """(start, end, qualified name) for every def/class and module assignment."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    spans: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                spans.append((child.lineno, child.end_lineno or child.lineno, name))
                visit(child, f"{name}.")
            elif isinstance(child, (ast.Assign, ast.AnnAssign)) and not prefix:
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if names:
                    spans.append((child.lineno, child.end_lineno or child.lineno, names[0]))

    visit(tree, "")
    return spans


def _enclosing(spans: list[tuple[int, int, str]], line: int) -> str:
    best = None
    for start, end, name in spans:
        if start <= line <= end and (best is None or end - start < best[1] - best[0]):
            best = (start, end, name)
    return best[2] if best else "<module>"


def scan_source(path: str, source: str) -> list[dict]:
    sites: list[dict] = []
    spans = _symbols(source)
    try:
        tokens = [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT)]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return sites
    lines = source.splitlines()

    def add(index: int, kind: str, token: str) -> None:
        line = tokens[index].start[0]
        sites.append(
            {
                "module": path,
                "head_line": line,
                "symbol": _enclosing(spans, line),
                "kind": kind,
                "token": token,
                "text": lines[line - 1].strip()[:160],
            }
        )

    for i, token in enumerate(tokens):
        if token.type == tokenize.NUMBER:
            value = _number(token.string)
            if value == 0.045:
                add(i, "loaded_tick_literal", token.string)
            elif value in (30.0,) and i >= 1:
                previous = tokens[i - 1]
                before = tokens[i - 2] if i >= 2 else None
                if previous.string == "/" and before is not None and before.type == tokenize.NUMBER and _number(before.string) == 1.0:
                    add(i - 2, "nominal_tick_literal", f"{before.string}/{token.string}")
                elif previous.string == "/":
                    add(i, "per_tick_division", f"/ {token.string}")
                elif previous.string == "*" or (i + 1 < len(tokens) and tokens[i + 1].string == "*"):
                    add(i, "tick_rate_multiplication", f"* {token.string}")
        elif token.type == tokenize.NAME and _DELTA_NAME.search(token.string):
            add(i, "delta_field", token.string)
        elif token.type == tokenize.NAME and _TICK_RATE_NAME.search(token.string):
            add(i, "tick_rate_constant", token.string)
        elif token.type == tokenize.STRING:
            try:
                literal = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                continue
            if isinstance(literal, str) and literal in _TIMING_KEYS:
                neighbour = tokens[i - 1].string if i else ""
                following = tokens[i + 1].string if i + 1 < len(tokens) else ""
                if neighbour in ("[", "(", ",", "{") or following in ("]", ":", ")", ","):
                    add(i, "timing_key", literal)
    return sites


def scan(repo: str | Path) -> dict:
    root = Path(repo)
    head = _git(root, "rev-parse", "HEAD").strip()
    sites: list[dict] = []
    for scope, prefixes in SCOPES.items():
        files = [
            name
            for prefix in prefixes
            for name in _git(root, "ls-tree", "-r", "--name-only", "HEAD", prefix).splitlines()
            if name.endswith(".py") and not name.startswith(EXCLUDE)
        ]
        for name in sorted(files):
            for site in scan_source(name, _git(root, "show", f"HEAD:{name}")):
                site["scope"] = scope
                sites.append(site)
    ruleset = _git(root, "show", f"HEAD:{RULESET_JSON}").splitlines()
    for number, line in enumerate(ruleset, start=1):
        for key in ("nominal_delta_seconds", "loaded_delta_seconds", "motion_timing_profile_count", "ticks_per_second"):
            if f'"{key}"' in line and ":" in line and not line.strip().startswith('"engine.'):
                sites.append({"module": RULESET_JSON, "head_line": number, "symbol": f"engine.{key}", "kind": "ruleset_json", "token": key, "text": line.strip(), "scope": "engine"})
    counts: dict[str, int] = {}
    for site in sites:
        label = f"{site['scope']}:{site['kind']}"
        counts[label] = counts.get(label, 0) + 1
    return {"head": head, "site_count": len(sites), "kinds": dict(sorted(counts.items())), "sites": sites}
