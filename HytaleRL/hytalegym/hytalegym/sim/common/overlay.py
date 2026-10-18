"""Overlay spec: one generation's declared changes to the frozen gen-5 source.

The gen-5 engine under ``hytalegym/jax/`` is never edited. A generation's
simulator package is *rendered* from that source as committed (``HEAD``) plus
the generation's :class:`Overlay` (``sim/common/render.py``). The overlay is the
whole port in one reviewable record:

``modules``   source roots under ``hytalegym/jax/`` (``"combat/"`` = every module
              below it); the renderer adds every ``hytalegym.jax`` module they
              import, transitively, so the rendered package imports nothing from
              ``hytalegym.jax``.
``redirects`` a value the engine reads, replaced at render time by a literal:
              the right-hand side of a module-level assignment, or (with
              ``expression``) one exact expression inside a module-level
              function or assignment, or (with ``line``) on one HEAD line. The
              value comes from the content layer
              (``hytalegym.sim.common.content.load(profile)``): a table field,
              or with ``family="adapter"`` an adapter's output
              (``adapters.<row>(tables)[field]``, the swap plan's rows); or,
              with ``family="profile"``, an attribute of the generation's
              ``SimProfile`` (``step_duration.loaded_seconds``).
              ``status="todo"`` rows are validated for location and listed.
              ``swap_plan_redirects`` and ``step_clock_redirects`` build rows
              from ``sim/common/content/redirects.json``.
``hooks``     a module-level function replaced, right after its definition, by
              ``construct(original, profile)`` from ``sim/genN/mechanics/``; the
              construct reads its own profile switch. ``status="stub"`` hooks are
              validated and listed, not applied.
``patches``   exact text replacements (``old`` must occur exactly once in HEAD),
              each carrying the fidelity row it answers and its differential.
``profile``   ``"module:attr"`` of the generation's ``SimProfile``; it binds the
              bridge-side ``ServerProfile`` through ``SimProfile.server_profile()``.

This module is stdlib-only at import time so ``render.py`` can be loaded before
``hytalegym`` is (see ``render.redirect_imports``).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath

#: Repository-relative directory holding the frozen gen-5 engine.
SOURCE_PREFIX = "HytaleRL/hytalegym/hytalegym/jax"
#: Its import name.
SOURCE_PACKAGE = "hytalegym.jax"
#: Repository-relative directory of the ``hytalegym`` package.
PACKAGE_ROOT = "HytaleRL/hytalegym/hytalegym"
REPO_ROOT = Path(__file__).resolve().parents[5]

REDIRECT_STATUSES = ("active", "todo")
HOOK_STATUSES = ("active", "stub")
CASTS = (None, "float", "int", "bool", "str", "tuple_float", "tuple_int",
         "reciprocal_int", "reciprocal_float")


@dataclass(frozen=True)
class Patch:
    """Replace ``old`` (exactly one occurrence in HEAD) by ``new`` in ``path``."""

    id: str
    path: str  # relative to hytalegym/jax/
    old: str
    new: str
    fidelity_row: str  # FIDELITY-0.7.2.md table (a) row, e.g. "5b"
    differential: str  # the measurement that must pass before the row is MATCH
    note: str = ""


@dataclass(frozen=True)
class Redirect:
    """Replace a value the engine reads by a content-table or profile value.

    Without ``expression``: the right-hand side of the single module-level
    assignment of ``symbol``. With it: the one occurrence of ``expression``
    inside the source of module-level ``symbol`` (function or assignment) or,
    when ``line`` is set, on that line of the committed file, replaced by
    ``template`` with ``{value}`` substituted.
    """

    id: str
    path: str
    symbol: str
    family: str  # content family ("movement"), "adapter" (row = adapter name) or "profile"
    row: str  # row key inside the family table, the adapter's name, or "" for profile attributes
    field: str  # dotted path inside the effective row, the adapter output key, or the SimProfile attribute
    cast: str | None = None
    status: str = "active"
    source: str = ""  # swap-plan / fidelity reference (file:line, adapter)
    fidelity_row: str = ""
    expression: str = ""
    template: str = "{value}"
    line: int = 0  # 1-based line of the committed file holding ``expression``


@dataclass(frozen=True)
class Hook:
    """Replace module-level ``function`` by ``construct(function, profile)``."""

    id: str
    path: str
    function: str
    construct: str  # "module.path:attr" under hytalegym.sim.genN.mechanics
    switch: str  # the Constructs field the construct reads
    status: str = "stub"
    fidelity_row: str = ""
    differential: str = ""
    note: str = ""


@dataclass(frozen=True)
class Overlay:
    generation: str
    profile: str  # "hytalegym.sim.genN.profile:PROFILE"
    modules: tuple[str, ...]
    redirects: tuple[Redirect, ...] = ()
    hooks: tuple[Hook, ...] = ()
    patches: tuple[Patch, ...] = ()
    package: str | None = None  # default hytalegym.sim.<generation>.engine
    description: str = ""
    # Free-form (key, value) facts recorded in the manifest (e.g. site coverage).
    audit: tuple[tuple[str, object], ...] = ()

    @property
    def package_name(self) -> str:
        return self.package or f"hytalegym.sim.{self.generation}.engine"

    def to_json(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        """Identity of the overlay content (canonical JSON of every field)."""

        text = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def sim_profile(self):
        """The bound ``SimProfile`` (imports ``hytalegym``)."""

        module_name, _, attr = self.profile.partition(":")
        from importlib import import_module

        return getattr(import_module(module_name), attr)

    def server_profile(self):
        """The bridge-side ``ServerProfile`` this overlay's profile binds."""

        return self.sim_profile().server_profile()


def profile_attribute(profile: object, path: str) -> object:
    """``step_duration.loaded_seconds`` -> ``profile.step_duration.loaded_seconds``."""

    value = profile
    for part in path.split("."):
        value = getattr(value, part)
    return value


class OverlayError(ValueError):
    """An overlay failed validation; ``problems`` lists every failure."""

    def __init__(self, problems: list[str]):
        super().__init__("overlay invalid:\n  " + "\n  ".join(problems))
        self.problems = problems


# --------------------------------------------------------------------------
# The committed baseline


class HeadSource:
    """Read-only view of ``hytalegym/jax`` (and its package) at one commit.

    Equivalent to ``git show <commit>:<path>`` per file, batched through one
    ``git cat-file --batch``. The commit is resolved once, so a commit made by
    another session mid-render cannot mix two baselines.
    """

    def __init__(self, repo: Path = REPO_ROOT, commit: str = "HEAD"):
        self.repo = Path(repo)
        self.commit = self._git("rev-parse", "--verify", f"{commit}^{{commit}}").strip()
        listing = self._git("ls-tree", "-r", "--full-tree", self.commit, "--", PACKAGE_ROOT)
        self._blobs: dict[str, str] = {}
        for line in listing.splitlines():
            meta, _, path = line.partition("\t")
            mode, kind, blob = meta.split()
            if kind == "blob":
                self._blobs[path] = blob
        self._cache: dict[str, bytes] = {}

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._blobs))

    def exists(self, repo_path: str) -> bool:
        return repo_path in self._blobs

    def blob_id(self, repo_path: str) -> str:
        return self._blobs[repo_path]

    def read_bytes(self, repo_path: str) -> bytes:
        if repo_path not in self._cache:
            self.prefetch([repo_path])
        return self._cache[repo_path]

    def read(self, repo_path: str) -> str:
        return self.read_bytes(repo_path).decode("utf-8")

    def prefetch(self, repo_paths) -> None:
        wanted = [p for p in repo_paths if p not in self._cache]
        if not wanted:
            return
        missing = [p for p in wanted if p not in self._blobs]
        if missing:
            raise FileNotFoundError(f"not in {self.commit[:12]}: {missing[:5]}")
        out = subprocess.run(
            ["git", "-C", str(self.repo), "cat-file", "--batch"],
            input="".join(f"{self._blobs[p]}\n" for p in wanted).encode(),
            check=True,
            capture_output=True,
        ).stdout
        pos = 0
        for path in wanted:
            newline = out.index(b"\n", pos)
            blob, kind, size = out[pos:newline].split()
            if kind != b"blob" or blob.decode() != self._blobs[path]:
                raise RuntimeError(f"git cat-file returned {kind!r} for {path}")
            size = int(size)
            self._cache[path] = out[newline + 1 : newline + 1 + size]
            pos = newline + 1 + size + 1

    @cached_property
    def modules(self) -> dict[str, str]:
        """Import name -> repo path for every ``.py`` under the package root."""

        found = {}
        for path in self._blobs:
            if path.endswith(".py"):
                found[module_name(path)] = path
        return found


def module_name(repo_path: str) -> str:
    rel = PurePosixPath(repo_path).relative_to(PurePosixPath(PACKAGE_ROOT).parent)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def source_path(rel: str) -> str:
    """``combat/types.py`` -> ``HytaleRL/hytalegym/hytalegym/jax/combat/types.py``."""

    return f"{SOURCE_PREFIX}/{rel}".rstrip("/")


def package_relative(repo_path: str) -> str:
    return str(PurePosixPath(repo_path).relative_to(SOURCE_PREFIX))


# --------------------------------------------------------------------------
# Import closure

_MODULE_STRING = re.compile(r"^hytalegym\.jax(\.[A-Za-z_]\w*)*$")


def _is_package(repo_path: str) -> bool:
    return repo_path.endswith("/__init__.py")


def imported_modules(source: HeadSource, repo_path: str) -> set[str]:
    """Every ``hytalegym`` module a file imports (statements anywhere, plus
    ``hytalegym.jax...`` string constants, which ``import_module`` consumers use)."""

    tree = ast.parse(source.read(repo_path), filename=repo_path)
    own = module_name(repo_path).split(".")
    base = own if _is_package(repo_path) else own[:-1]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = base[: len(base) - (node.level - 1)]
                target = ".".join(anchor + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            names.add(target)
            names.update(f"{target}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _MODULE_STRING.match(node.value):
                names.add(node.value)
    found: set[str] = set()
    modules = source.modules
    for name in names:
        if not name.startswith("hytalegym"):
            continue
        parts = name.split(".")
        for k in range(len(parts), 0, -1):
            candidate = ".".join(parts[:k])
            if candidate in modules:
                found.add(candidate)
                break
    return found


def resolve_modules(overlay: Overlay, source: HeadSource) -> tuple[str, ...]:
    """Repo paths of every module the rendered package needs, sorted.

    Roots are the overlay's ``modules``; the closure follows imports into
    ``hytalegym.jax`` only (other ``hytalegym`` packages stay shared) and adds
    every parent ``__init__.py``.
    """

    modules = source.modules
    pending: list[str] = []
    for root in overlay.modules:
        path = source_path(root)
        if root.endswith("/") or not path.endswith(".py"):
            prefix = path.rstrip("/") + "/"
            pending.extend(p for p in source.paths() if p.startswith(prefix) and p.endswith(".py"))
        else:
            pending.append(path)
    pending.append(source_path("__init__.py"))
    seen: set[str] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        name = module_name(path)
        parts = name.split(".")
        for k in range(2, len(parts)):
            parent = ".".join(parts[:k])
            if parent.startswith(SOURCE_PACKAGE) and parent in modules:
                pending.append(modules[parent])
        for target in imported_modules(source, path):
            if target == SOURCE_PACKAGE or target.startswith(SOURCE_PACKAGE + "."):
                pending.append(modules[target])
    return tuple(sorted(seen))


def resolve_data_files(source: HeadSource, rendered: tuple[str, ...]) -> tuple[str, ...]:
    """Tracked non-Python files whose nearest package directory is rendered."""

    packages = {str(PurePosixPath(p).parent) for p in rendered if _is_package(p)}
    data = []
    for path in source.paths():
        if not path.startswith(SOURCE_PREFIX + "/") or path.endswith(".py"):
            continue
        directory = PurePosixPath(path).parent
        while str(directory) != SOURCE_PREFIX and not source.exists(f"{directory}/__init__.py"):
            directory = directory.parent
        if str(directory) in packages:
            data.append(path)
    return tuple(sorted(data))


def external_back_edges(source: HeadSource, rendered: tuple[str, ...]) -> dict[str, list[str]]:
    """Shared (non-rendered) modules the rendered package imports that
    themselves import ``hytalegym.jax`` directly: there, gen-5 code is still
    reached from a rendered generation."""

    modules = source.modules
    outside: set[str] = set()
    for path in rendered:
        for target in imported_modules(source, path):
            if not target.startswith(SOURCE_PACKAGE):
                outside.add(target)
    edges = {}
    for name in sorted(outside):
        hits = sorted(
            t for t in imported_modules(source, modules[name])
            if t == SOURCE_PACKAGE or t.startswith(SOURCE_PACKAGE + ".")
        )
        if hits:
            edges[name] = hits
    return edges


# --------------------------------------------------------------------------
# Validation


def module_level_assignments(text: str, symbol: str) -> list[ast.stmt]:
    tree = ast.parse(text)
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets):
                found.append(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == symbol and node.value is not None:
                found.append(node)
    return found


def module_level_symbols(text: str, symbol: str) -> list[ast.stmt]:
    """Module-level functions, classes or assignments that define ``symbol``."""

    tree = ast.parse(text)
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            found.append(node)
    return found + module_level_assignments(text, symbol)


def node_source(text: str, node: ast.AST) -> tuple[int, int]:
    """Character span of a top-level statement (decorators included)."""

    lines = text.splitlines(keepends=True)
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    start = sum(len(line) for line in lines[: first - 1])
    end = sum(len(line) for line in lines[: node.end_lineno])
    return start, end


def module_level_functions(text: str, function: str) -> list[ast.stmt]:
    tree = ast.parse(text)
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
    ]


def _constructs_fields(source: HeadSource | None) -> set[str] | None:
    """Field names of ``sim.common.profile.Constructs`` (parsed, not imported)."""

    path = Path(__file__).resolve().parent / "profile.py"
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Constructs":
            return {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    return None


def validate(overlay: Overlay, source: HeadSource) -> list[str]:
    """Every problem with ``overlay`` against ``source``; empty when valid."""

    problems: list[str] = []
    ids = [x.id for x in (*overlay.redirects, *overlay.hooks, *overlay.patches)]
    for duplicate in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"duplicate change id {duplicate!r}")
    if not overlay.modules:
        problems.append("modules is empty")
    for root in overlay.modules:
        path = source_path(root)
        prefix = path.rstrip("/") + "/"
        if not source.exists(path) and not any(p.startswith(prefix) for p in source.paths()):
            problems.append(f"module root {root!r} not in {source.commit[:12]}")
    if ":" not in overlay.profile:
        problems.append(f"profile {overlay.profile!r} is not 'module:attr'")
    try:
        rendered = set(resolve_modules(overlay, source)) if not problems else set()
    except (KeyError, SyntaxError, FileNotFoundError) as exc:
        problems.append(f"module closure failed: {exc!r}")
        rendered = set()

    def file_text(kind: str, change_id: str, rel: str) -> str | None:
        path = source_path(rel)
        if not source.exists(path):
            problems.append(f"{kind} {change_id}: {rel} not in {source.commit[:12]}")
            return None
        if rendered and path not in rendered:
            problems.append(f"{kind} {change_id}: {rel} is not a rendered module")
        return source.read(path)

    for redirect in overlay.redirects:
        if redirect.status not in REDIRECT_STATUSES:
            problems.append(f"redirect {redirect.id}: status {redirect.status!r}")
        if redirect.cast not in CASTS:
            problems.append(f"redirect {redirect.id}: cast {redirect.cast!r}")
        text = file_text("redirect", redirect.id, redirect.path)
        if text is not None and redirect.line:
            lines = text.splitlines()
            if not 1 <= redirect.line <= len(lines):
                problems.append(f"redirect {redirect.id}: line {redirect.line} not in {redirect.path}")
            elif not redirect.expression or lines[redirect.line - 1].count(redirect.expression) != 1:
                problems.append(
                    f"redirect {redirect.id}: expression must occur once on {redirect.path}:{redirect.line}"
                )
        elif text is not None and not redirect.expression:
            count = len(module_level_assignments(text, redirect.symbol))
            if count != 1:
                problems.append(
                    f"redirect {redirect.id}: {redirect.symbol} has {count} module-level "
                    f"assignments in {redirect.path} (need exactly 1)"
                )
        elif text is not None:
            nodes = module_level_symbols(text, redirect.symbol)
            if len(nodes) != 1:
                problems.append(
                    f"redirect {redirect.id}: {redirect.symbol} has {len(nodes)} module-level "
                    f"definitions in {redirect.path} (need exactly 1)"
                )
            else:
                start, end = node_source(text, nodes[0])
                count = text[start:end].count(redirect.expression)
                if count != 1:
                    problems.append(
                        f"redirect {redirect.id}: expression occurs {count} times in {redirect.symbol}"
                    )
        if redirect.expression and redirect.template.count("{value}") != 1:
            problems.append(f"redirect {redirect.id}: template needs exactly one {{value}}")
        if redirect.family == "profile":
            try:
                profile_attribute(overlay.sim_profile(), redirect.field)
            except (AttributeError, ImportError) as exc:
                problems.append(f"redirect {redirect.id}: profile has no {redirect.field!r} ({exc!r})")
        elif redirect.family == "adapter":
            if redirect.status == "active":
                try:
                    from importlib import import_module

                    if not callable(getattr(import_module("hytalegym.sim.common.content.adapters"), redirect.row)):
                        raise AttributeError(redirect.row)
                except (AttributeError, ImportError) as exc:
                    problems.append(f"redirect {redirect.id}: no content adapter {redirect.row!r} ({exc!r})")
        elif redirect.status == "active" and not (redirect.family and redirect.row and redirect.field):
            problems.append(f"redirect {redirect.id}: active rows need family, row and field")

    switches = _constructs_fields(source)
    for hook in overlay.hooks:
        if hook.status not in HOOK_STATUSES:
            problems.append(f"hook {hook.id}: status {hook.status!r}")
        text = file_text("hook", hook.id, hook.path)
        if text is not None:
            count = len(module_level_functions(text, hook.function))
            if count != 1:
                problems.append(
                    f"hook {hook.id}: {hook.function} has {count} module-level "
                    f"definitions in {hook.path} (need exactly 1)"
                )
        if ":" not in hook.construct:
            problems.append(f"hook {hook.id}: construct {hook.construct!r} is not 'module:attr'")
        if switches is not None and hook.switch not in switches:
            problems.append(f"hook {hook.id}: switch {hook.switch!r} is not a Constructs field")

    patched: dict[str, str] = {}
    for patch in overlay.patches:
        if patch.old == patch.new:
            problems.append(f"patch {patch.id}: old == new")
        if not patch.fidelity_row or not patch.differential:
            problems.append(f"patch {patch.id}: needs fidelity_row and differential")
        if patch.path not in patched:
            text = file_text("patch", patch.id, patch.path)
            if text is None:
                continue
            patched[patch.path] = text
        head_count = source.read(source_path(patch.path)).count(patch.old)
        if head_count != 1:
            problems.append(f"patch {patch.id}: old occurs {head_count} times in HEAD {patch.path}")
            continue
        current = patched[patch.path]
        count = current.count(patch.old)
        if count != 1:
            problems.append(
                f"patch {patch.id}: old occurs {count} times after earlier patches to {patch.path}"
            )
            continue
        patched[patch.path] = current.replace(patch.old, patch.new, 1)
    return problems


# --------------------------------------------------------------------------
# Rows from the swap plan (sim/common/content/redirects.json)

SWAP_PLAN = Path(__file__).resolve().parent / "content" / "redirects.json"
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def load_swap_plan(path: Path = SWAP_PLAN) -> dict:
    return json.loads(Path(path).read_text())


def _in_source(module: str) -> str | None:
    """``HytaleRL/.../jax/combat/x.py`` -> ``combat/x.py``; None outside it."""

    prefix = SOURCE_PREFIX + "/"
    return module[len(prefix):] if module.startswith(prefix) and module.endswith(".py") else None


def swap_plan_redirects(plan: dict, *, status: str = "todo") -> tuple[tuple[Redirect, ...], tuple[dict, ...]]:
    """The swap plan's records as redirect rows valued by their adapter
    (``adapters.<adapter>(tables)[target]``), plus the records that are not a
    Python module of the rendered source (JSON tables, shared modules, symbols
    absent at HEAD), which a render cannot reach.

    Module constants are located by symbol; defaults and inline literals by
    their HEAD line and the record's text, so the rows validate their location
    against the commit. Activating a non-constant row also needs a template.
    """

    rows: list[Redirect] = []
    unreachable: list[dict] = []
    for record in plan["redirects"]:
        rel = _in_source(record["module"])
        if rel is None or record["status"] == "absent_in_head":
            unreachable.append(record)
            continue
        identity = f"{record['adapter']}:{record['target']}"
        note = (f"redirects.json {record['kind']} {rel}:{record['head_line']} "
                f"[{record['status']}] content={'; '.join(record['content'])}")
        symbol = record["symbol"].strip().rstrip("=").strip()
        if record["kind"] == "module_constant" and _IDENTIFIER.fullmatch(symbol):
            rows.append(Redirect(identity, rel, symbol, "adapter", record["adapter"], record["target"],
                                 status=status, source=note))
        else:
            rows.append(Redirect(identity, rel, "", "adapter", record["adapter"], record["target"],
                                 status=status, source=note, expression=record["symbol"],
                                 line=record["head_line"]))
    return tuple(rows), tuple(unreachable)


# ruleset ``engine`` key -> (SimProfile attribute, literal cast mirroring the JSON type)
STEP_CLOCK_KEYS = {
    "nominal_delta_seconds": ("step_duration.nominal_seconds", "float"),
    "loaded_delta_seconds": ("step_duration.loaded_seconds", "float"),
    "ticks_per_second": ("step_duration.nominal_seconds", "reciprocal_int"),
    "motion_timing_profile_count": ("step_duration.timing_profile_count", "int"),
}
_ENGINE_KEY = re.compile(r'(?:load_combat_ruleset\(\)\["engine"\]|engine)\["(\w+)"\]')


def step_clock_redirects(plan: dict, *, fidelity_row: str = "1c") -> tuple[tuple[Redirect, ...], dict]:
    """``step_duration`` sites that take the step clock from a constant ->
    active redirects to ``SimProfile.step_duration``; every other site in the
    rendered source reads one of them (a consumer) and follows.

    Sources: a read of the ruleset ``engine`` block (``timing_key``), a module
    constant holding a literal tick rate (``tick_rate_constant``), a one-tick
    literal ``1.0 / 30.0`` (``nominal_tick_literal``), a literal ``* 30.0``
    (``tick_rate_multiplication``).
    """

    rows: list[Redirect] = []
    counts = {"sites": 0, "in_source": 0, "redirected": 0, "consumers": 0, "outside_source": 0}
    for site in plan["step_duration"]["sites"]:
        counts["sites"] += 1
        rel = _in_source(site["module"])
        if rel is None:
            counts["outside_source"] += 1
            continue
        counts["in_source"] += 1
        text, kind = site["text"], site["kind"]
        expression = field = cast = None
        key = _ENGINE_KEY.search(text)
        if kind == "timing_key" and key and key.group(1) in STEP_CLOCK_KEYS:
            expression = key.group(0)
            field, cast = STEP_CLOCK_KEYS[key.group(1)]
        elif kind == "tick_rate_constant" and re.fullmatch(r"[A-Z_][A-Z0-9_]* = \d+", text.strip()):
            expression = text.strip().rsplit(" ", 1)[1]
            field, cast = "step_duration.nominal_seconds", "reciprocal_int"
        elif kind == "nominal_tick_literal" and "1.0 / 30.0" in text:
            expression, field, cast = "1.0 / 30.0", "step_duration.nominal_seconds", "float"
        elif kind == "tick_rate_multiplication" and re.search(r"\* 30\.0\b", text):
            expression, field, cast = "30.0", "step_duration.nominal_seconds", "reciprocal_float"
        if expression is None:
            counts["consumers"] += 1
            continue
        counts["redirected"] += 1
        rows.append(Redirect(
            f"step_clock:{rel}:{site['head_line']}", rel, site["symbol"], "profile", "", field,
            cast=cast, status="active", fidelity_row=fidelity_row, expression=expression,
            line=site["head_line"],
            source=f"redirects.json step_duration {kind} {rel}:{site['head_line']}: {text.strip()}",
        ))
    return tuple(rows), counts


def check(overlay: Overlay, source: HeadSource) -> None:
    problems = validate(overlay, source)
    if problems:
        raise OverlayError(problems)


__all__ = [
    "Hook",
    "HeadSource",
    "Overlay",
    "OverlayError",
    "Patch",
    "Redirect",
    "SOURCE_PACKAGE",
    "SOURCE_PREFIX",
    "check",
    "external_back_edges",
    "module_level_assignments",
    "module_level_functions",
    "module_level_symbols",
    "module_name",
    "profile_attribute",
    "load_swap_plan",
    "step_clock_redirects",
    "swap_plan_redirects",
    "node_source",
    "resolve_data_files",
    "resolve_modules",
    "source_path",
    "validate",
]
