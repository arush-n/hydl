"""Render a generation's simulator package: frozen gen-5 source + overlay.

``render(overlay, out_dir)`` reads every module the overlay needs from the
committed baseline (``git show <HEAD>:<path>``, batched; never the working
tree), and per file, in order:

1. relocates it: ``hytalegym.jax`` references become the rendered package,
   ``Path(__file__).resolve().parents[N]`` that climb out of the package are
   re-anchored on ``hytalegym.__file__`` so they still name the same directory,
   and tracked data files next to rendered packages are copied;
2. applies the overlay's active redirects (content-table values written as
   literals), active hooks and patches, each of which must apply exactly once;
3. checks the result compiles and imports nothing from ``hytalegym.jax``.

Only when every step applied does it write ``out_dir`` (atomically, through a
sibling staging directory) with ``MANIFEST.json``: source commit, overlay
sha256, per-file sha256 before and after, the applied and pending changes, and
the shared modules that still import gen-5 code. A previous render at
``out_dir`` is replaced; any other non-empty directory is refused.

CLI: ``python -m hytalegym.sim.common.render --generation gen5|gen7 [--out DIR]``
(default ``sim/<generation>/engine/``).

Importing a rendered package that lives outside ``hytalegym/sim/genN/engine``:
``mount(out_dir)``. Running code written against ``hytalegym.jax`` on a render:
``redirect_imports(out_dir)`` before ``hytalegym`` is first imported. This
module imports only the standard library at import time for that reason.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from hytalegym.sim.common.overlay import HeadSource, Overlay

MANIFEST_NAME = "MANIFEST.json"
MANIFEST_SCHEMA = "hytalerl_sim_render_v1"
SOURCE_PACKAGE = "hytalegym.jax"
GENERATION_OVERLAYS = {
    "gen5": "hytalegym.sim.gen5.overlay:OVERLAY",
    "gen7": "hytalegym.sim.gen7.overlay:OVERLAY",
}

_JAX_REF = re.compile(r"(?<![\w.])hytalegym\.jax(?!\w)")
_FILE_TOKEN = re.compile(r"\b__file__\b")
_FILE_ANCHOR = re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]")
_FILE_LOCAL = re.compile(r"Path\(__file__\)\.(?:with_name\(|parent\b)")
_JAX_STRING = re.compile(r"^hytalegym\.jax(\.|$)")


class RenderError(RuntimeError):
    """A render step did not apply; nothing was written."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Relocation


def relocate_refs(text: str, package: str) -> tuple[str, int]:
    """Rewrite dotted ``hytalegym.jax`` references (not ``hytalegym.jax_*``)."""

    return _JAX_REF.subn(package, text)


def relocate(text: str, rel_path: str, package: str) -> tuple[str, list[str]]:
    """Relocate one module from ``hytalegym/jax/<rel_path>`` into ``package``."""

    tree = ast.parse(text, filename=rel_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "hytalegym" and not node.level:
            if any(alias.name == "jax" for alias in node.names):
                raise RenderError(f"{rel_path}:{node.lineno}: 'from hytalegym import jax' is not relocatable")
    changes: list[str] = []
    depth = 1 + len(PurePosixPath(rel_path).parent.parts)

    anchored = {m.start() for m in _FILE_ANCHOR.finditer(text)}
    local = {m.start() for m in _FILE_LOCAL.finditer(text)}
    for match in _FILE_TOKEN.finditer(text):
        start = match.start() - len("Path(")
        if start not in anchored and start not in local:
            line = text.count("\n", 0, match.start()) + 1
            raise RenderError(f"{rel_path}:{line}: unhandled __file__ use; add a relocation rule")

    def anchor(match: re.Match) -> str:
        level = int(match.group(1))
        if level < depth:
            return match.group(0)
        changes.append(f"relocate:anchor parents[{level}] -> hytalegym.__file__ parents[{level - depth}]")
        return f'Path(__import__("hytalegym").__file__).resolve().parents[{level - depth}]'

    text = _FILE_ANCHOR.sub(anchor, text)
    text, count = relocate_refs(text, package)
    if count:
        changes.insert(0, f"relocate:refs {count}")
    return text, changes


def _check_rendered(text: str, rel_path: str, package: str, rendered_names: set[str]) -> list[str]:
    problems = []
    try:
        tree = ast.parse(text, filename=rel_path)
        compile(tree, rel_path, "exec")
    except SyntaxError as exc:
        return [f"{rel_path}: does not compile: {exc}"]
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            targets = [node.module]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _JAX_STRING.match(node.value):
                problems.append(f"{rel_path}:{node.lineno}: string still names {node.value!r}")
        for target in targets:
            if _JAX_STRING.match(target):
                problems.append(f"{rel_path}:{node.lineno}: still imports {target}")
            elif (target == package or target.startswith(package + ".")) and target not in rendered_names:
                problems.append(f"{rel_path}:{node.lineno}: imports {target}, which is not rendered")
    return problems


# --------------------------------------------------------------------------
# Overlay steps


def _splice(text: str, node: ast.AST, replacement: str) -> str:
    """Replace ``node``'s exact source span (ast offsets are UTF-8 bytes)."""

    lines = text.splitlines(keepends=True)
    start = sum(len(line.encode()) for line in lines[: node.lineno - 1]) + node.col_offset
    end = sum(len(line.encode()) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
    data = text.encode()
    return (data[:start] + replacement.encode() + data[end:]).decode()


def _insert_line(text: str, after_line: int, block: str) -> str:
    lines = text.splitlines(keepends=True)
    if lines and not lines[after_line - 1].endswith("\n"):
        lines[after_line - 1] += "\n"
    return "".join(lines[:after_line] + [block] + lines[after_line:])


def literal(value: Any, cast: str | None) -> str:
    """Python source for a content value; round-trips through literal_eval."""

    if cast == "float":
        value = float(value)
    elif cast == "int":
        if float(value) != int(value):
            raise RenderError(f"cast int would truncate {value!r}")
        value = int(value)
    elif cast == "bool":
        value = bool(value)
    elif cast == "str":
        value = str(value)
    elif cast == "tuple_float":
        value = tuple(float(v) for v in value)
    elif cast == "tuple_int":
        value = tuple(int(v) for v in value)
    elif cast in ("reciprocal_int", "reciprocal_float"):
        rate = 1.0 / float(value)
        if abs(rate - round(rate)) > 1e-9:
            raise RenderError(f"{cast}: 1/{value!r} is not a whole rate")
        value = int(round(rate)) if cast == "reciprocal_int" else float(round(rate))
    source = repr(value)
    if ast.literal_eval(source) != value:
        raise RenderError(f"value {value!r} does not round-trip as a literal")
    return source


def _content_value(tables: Any, redirect, profile: Any) -> Any:
    if redirect.family == "profile":
        from hytalegym.sim.common.overlay import profile_attribute

        return profile_attribute(profile, redirect.field)
    if redirect.family == "adapter":
        adapter = getattr(importlib.import_module("hytalegym.sim.common.content.adapters"), redirect.row)
        output = adapter(tables)
        if not isinstance(output, dict):
            # a scalar adapter (walk_steering_sprint_multiplier) returns its value itself
            return output
        if redirect.field not in output:
            raise RenderError(f"redirect {redirect.id}: adapter {redirect.row} has no {redirect.field!r}")
        return output[redirect.field]
    value: Any = tables.effective_row(redirect.family, redirect.row)
    for part in redirect.field.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, (list, tuple)) and part.isdigit():
            value = value[int(part)]
        else:
            raise RenderError(
                f"redirect {redirect.id}: {redirect.family}/{redirect.row} has no field {redirect.field!r}"
            )
    return value


def apply_redirect(text: str, redirect, value: Any) -> str:
    from hytalegym.sim.common.overlay import module_level_assignments, module_level_symbols, node_source

    source = f"{redirect.family}/{redirect.row}#{redirect.field}" if redirect.row else f"{redirect.family}#{redirect.field}"
    comment = f"# render redirect {redirect.id}: {source} (fidelity {redirect.fidelity_row or '-'})"
    rendered = literal(value, redirect.cast)
    if redirect.line:
        lines = text.splitlines(keepends=True)
        target = lines[redirect.line - 1]
        if target.count(redirect.expression) != 1:
            raise RenderError(f"redirect {redirect.id}: expression not once on line {redirect.line}")
        lines[redirect.line - 1] = target.replace(redirect.expression, redirect.template.replace("{value}", rendered))
        text = "".join(lines)
        line = redirect.line
    elif not redirect.expression:
        nodes = module_level_assignments(text, redirect.symbol)
        if len(nodes) != 1:
            raise RenderError(f"redirect {redirect.id}: {len(nodes)} assignments of {redirect.symbol}")
        node = nodes[0]
        text = _splice(text, node.value, rendered)
        line = node.lineno
    else:
        nodes = module_level_symbols(text, redirect.symbol)
        if len(nodes) != 1:
            raise RenderError(f"redirect {redirect.id}: {len(nodes)} definitions of {redirect.symbol}")
        start, end = node_source(text, nodes[0])
        count = text[start:end].count(redirect.expression)
        if count != 1:
            raise RenderError(f"redirect {redirect.id}: expression occurs {count} times in {redirect.symbol}")
        at = text.index(redirect.expression, start)
        text = text[:at] + redirect.template.replace("{value}", rendered) + text[at + len(redirect.expression):]
        line = text.count("\n", 0, at) + 1
    lines = text.splitlines(keepends=True)
    indent = re.match(r"[ \t]*", lines[line - 1]).group(0)
    return "".join(lines[: line - 1] + [f"{indent}{comment}\n"] + lines[line - 1 :])


def _head_line(text: str, redirect) -> int:
    if redirect.line:
        return redirect.line
    from hytalegym.sim.common.overlay import module_level_assignments, module_level_symbols, node_source

    if not redirect.expression:
        nodes = module_level_assignments(text, redirect.symbol)
        return nodes[0].lineno if nodes else 0
    nodes = module_level_symbols(text, redirect.symbol)
    if not nodes:
        return 0
    start, end = node_source(text, nodes[0])
    at = text.find(redirect.expression, start, end)
    return text.count("\n", 0, at) + 1 if at >= 0 else nodes[0].lineno


def apply_hook(text: str, hook, profile_ref: str) -> str:
    from hytalegym.sim.common.overlay import module_level_functions

    nodes = module_level_functions(text, hook.function)
    if len(nodes) != 1:
        raise RenderError(f"hook {hook.id}: {len(nodes)} definitions of {hook.function}")
    construct_module, _, construct_attr = hook.construct.partition(":")
    profile_module, _, profile_attr = profile_ref.partition(":")
    name = re.sub(r"\W", "_", hook.id)
    block = (
        f"\n\n# render hook {hook.id}: {hook.function} -> {hook.construct}"
        f" (switch {hook.switch}; fidelity {hook.fidelity_row or '-'})\n"
        f"from {construct_module} import {construct_attr} as _render_hook_{name}  # noqa: E402\n"
        f"from {profile_module} import {profile_attr} as _render_profile_{name}  # noqa: E402\n"
        f"{hook.function} = _render_hook_{name}({hook.function}, _render_profile_{name})\n"
    )
    return _insert_line(text, nodes[0].end_lineno, block)


def apply_patch(text: str, patch, package: str) -> str:
    old, _ = relocate_refs(patch.old, package)
    new, _ = relocate_refs(patch.new, package)
    count = text.count(old)
    if count != 1:
        raise RenderError(f"patch {patch.id}: old occurs {count} times in rendered {patch.path}")
    return text.replace(old, new, 1)


# --------------------------------------------------------------------------
# Render


def load_overlay(generation: str) -> "Overlay":
    reference = GENERATION_OVERLAYS.get(generation)
    if reference is None:
        raise RenderError(f"no overlay for generation {generation!r}; known: {sorted(GENERATION_OVERLAYS)}")
    module_name, _, attr = reference.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def default_out_dir(overlay: "Overlay") -> Path:
    from hytalegym.sim.common.overlay import REPO_ROOT, PACKAGE_ROOT

    return REPO_ROOT / PACKAGE_ROOT / "sim" / overlay.generation / "engine"


def _default_content_loader(profile):
    from hytalegym.sim.common.content import load

    return load(profile)


def _profile_record(overlay: "Overlay") -> dict:
    from dataclasses import asdict

    profile = overlay.sim_profile()
    record = {
        "ref": overlay.profile,
        "generation": profile.generation,
        "certified_version": profile.certified_version,
        "release_line": list(profile.release_line),
        "constructs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(profile.constructs).items()},
    }
    if hasattr(profile, "step_duration"):
        record["step_duration"] = asdict(profile.step_duration)
    try:
        server = asdict(profile.server_profile())
        server["release_line"] = list(server["release_line"])
        record["server_profile"] = server
    except Exception as exc:  # recorded, not fatal: the bridge profile is optional context
        record["server_profile_error"] = repr(exc)
    return record


def render(
    overlay: "Overlay",
    out_dir: str | Path | None = None,
    *,
    source: "HeadSource | None" = None,
    content_loader: Callable[[Any], Any] | None = None,
    write: bool = True,
) -> dict:
    """Render ``overlay`` into ``out_dir``; return the manifest.

    Raises :class:`RenderError` (or ``OverlayError``) and writes nothing when
    any step does not apply. ``content_loader(profile)`` must return an object
    with ``effective_row(family, key)`` (default: the content layer's ``load``).
    """

    from hytalegym.sim.common import overlay as ov

    source = source or ov.HeadSource()
    ov.check(overlay, source)
    package = overlay.package_name
    modules = ov.resolve_modules(overlay, source)
    data_files = ov.resolve_data_files(source, modules)
    source.prefetch(modules + data_files)
    rendered_names = {package + ov.module_name(p)[len(SOURCE_PACKAGE):] for p in modules}

    active_redirects = [r for r in overlay.redirects if r.status == "active"]
    profile = overlay.sim_profile()
    tables = None
    content_sha = None
    if any(r.family != "profile" for r in active_redirects):
        loader = content_loader or _default_content_loader
        tables = loader(profile)
        content_sha = getattr(tables, "content_sha256", None)

    by_path: dict[str, dict[str, list]] = {}
    for kind, items in (
        ("redirect", active_redirects),
        ("hook", [h for h in overlay.hooks if h.status == "active"]),
        ("patch", list(overlay.patches)),
    ):
        for item in items:
            by_path.setdefault(ov.source_path(item.path), {}).setdefault(kind, []).append(item)

    outputs: dict[str, bytes] = {}
    files: list[dict] = []
    applied: list[dict] = []
    problems: list[str] = []
    for path in modules:
        rel = ov.package_relative(path)
        before = source.read_bytes(path)
        text, changes = relocate(before.decode("utf-8"), rel, package)
        steps = by_path.get(path, {})
        head_text = before.decode("utf-8")
        # Bottom-up, so a comment line inserted for one redirect never shifts
        # the committed line another one is anchored to.
        for redirect in sorted(steps.get("redirect", []), key=lambda r: -_head_line(head_text, r)):
            value = _content_value(tables, redirect, profile)
            text = apply_redirect(text, redirect, value)
            changes.append(f"redirect:{redirect.id}")
            applied.append({"kind": "redirect", "id": redirect.id, "path": rel, "symbol": redirect.symbol,
                            "value": json.loads(json.dumps(value)), "fidelity_row": redirect.fidelity_row})
        for hook in steps.get("hook", []):
            text = apply_hook(text, hook, overlay.profile)
            changes.append(f"hook:{hook.id}")
            applied.append({"kind": "hook", "id": hook.id, "path": rel, "function": hook.function,
                            "construct": hook.construct, "switch": hook.switch,
                            "fidelity_row": hook.fidelity_row, "differential": hook.differential})
        for patch in steps.get("patch", []):
            text = apply_patch(text, patch, package)
            changes.append(f"patch:{patch.id}")
            applied.append({"kind": "patch", "id": patch.id, "path": rel, "fidelity_row": patch.fidelity_row,
                            "differential": patch.differential, "note": patch.note})
        problems.extend(_check_rendered(text, rel, package, rendered_names))
        after = text.encode("utf-8")
        outputs[rel] = after
        files.append({"path": rel, "source": path, "source_blob": source.blob_id(path),
                      "sha256_before": _sha256(before), "sha256_after": _sha256(after),
                      "changes": changes})
    for path in data_files:
        rel = ov.package_relative(path)
        data = source.read_bytes(path)
        outputs[rel] = data
        files.append({"path": rel, "source": path, "source_blob": source.blob_id(path),
                      "sha256_before": _sha256(data), "sha256_after": _sha256(data), "changes": ["data"]})

    expected = len(active_redirects) + sum(h.status == "active" for h in overlay.hooks) + len(overlay.patches)
    if len(applied) != expected:
        problems.append(f"{expected} overlay steps declared, {len(applied)} applied")
    if problems:
        raise RenderError("render refused:\n  " + "\n  ".join(problems))

    here = Path(__file__).resolve()
    pending = [
        {"kind": "redirect", "id": r.id, "status": r.status, "path": r.path, "symbol": r.symbol,
         "family": r.family, "row": r.row, "field": r.field, "source": r.source, "fidelity_row": r.fidelity_row}
        for r in overlay.redirects if r.status != "active"
    ] + [
        {"kind": "hook", "id": h.id, "status": h.status, "path": h.path, "function": h.function,
         "construct": h.construct, "switch": h.switch, "fidelity_row": h.fidelity_row,
         "differential": h.differential, "note": h.note}
        for h in overlay.hooks if h.status != "active"
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generation": overlay.generation,
        "package": package,
        "source": {"commit": source.commit, "repository_prefix": ov.SOURCE_PREFIX, "package": SOURCE_PACKAGE,
                   "read_from": "git cat-file --batch <commit>:<path> (equivalent to git show)"},
        "overlay": {"sha256": overlay.sha256(), "modules": list(overlay.modules),
                    "audit": {key: value for key, value in overlay.audit},
                    "redirects": len(overlay.redirects), "hooks": len(overlay.hooks),
                    "patches": len(overlay.patches), "description": overlay.description},
        "profile": _profile_record(overlay),
        "renderer": {"render_py_sha256": _sha256(here.read_bytes()),
                     "overlay_py_sha256": _sha256((here.parent / "overlay.py").read_bytes())},
        "content_sha256": content_sha,
        "totals": {"modules": len(modules), "data_files": len(data_files),
                   "files_changed_beyond_relocation": sum(
                       any(not c.startswith("relocate:") for c in f["changes"]) and f["changes"] != ["data"]
                       for f in files),
                   "applied": len(applied), "pending": len(pending)},
        "applied": applied,
        "pending": pending,
        "external_back_edges": ov.external_back_edges(source, modules),
        "files": files,
    }
    if write:
        _write_atomically(Path(out_dir) if out_dir is not None else default_out_dir(overlay), outputs, manifest)
    return manifest


def _write_atomically(out: Path, outputs: dict[str, bytes], manifest: dict) -> None:
    out = out.resolve()
    if out.exists():
        if not out.is_dir():
            raise RenderError(f"{out} exists and is not a directory")
        if any(out.iterdir()) and not (out / MANIFEST_NAME).is_file():
            raise RenderError(f"refusing to replace {out}: not empty and not a previous render")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.rendering-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        for rel, data in sorted(outputs.items()):
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        if out.exists():
            old = out.parent / f".{out.name}.old-{os.getpid()}"
            out.rename(old)
            staging.rename(out)
            shutil.rmtree(old)
        else:
            staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


# --------------------------------------------------------------------------
# Importing a render


def read_manifest(out_dir: str | Path) -> dict:
    manifest = json.loads((Path(out_dir) / MANIFEST_NAME).read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RenderError(f"{out_dir} is not a render ({manifest.get('schema')!r})")
    return manifest


def _ensure_package(name: str) -> types.ModuleType | None:
    """Import ``name``; if absent, register an empty namespace package for it."""

    if not name:
        return None
    if name in sys.modules:
        return sys.modules[name]
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
    parent_name, _, leaf = name.rpartition(".")
    parent = _ensure_package(parent_name)
    module = types.ModuleType(name)
    module.__path__ = []
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    sys.modules[name] = module
    if parent is not None:
        setattr(parent, leaf, module)
    return module


def mount(out_dir: str | Path) -> types.ModuleType:
    """Import the rendered package at ``out_dir`` under its manifest name."""

    out = Path(out_dir).resolve()
    package = read_manifest(out)["package"]
    existing = sys.modules.get(package)
    if existing is not None:
        paths = [str(Path(p).resolve()) for p in getattr(existing, "__path__", [])]
        if paths == [str(out)]:
            return existing
        raise RenderError(f"{package} is already imported from {paths}, not {out}")
    parent_name, _, leaf = package.rpartition(".")
    parent = _ensure_package(parent_name)
    spec = importlib.util.spec_from_file_location(
        package, out / "__init__.py", submodule_search_locations=[str(out)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(package, None)
        raise
    if parent is not None:
        setattr(parent, leaf, module)
    return module


class _AliasLoader:
    """Serve ``hytalegym.jax.X`` as the already-rendered ``<package>.X``."""

    def __init__(self, target: str):
        self.target = target
        self._spec = None

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self._spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self._spec  # module_from_spec overwrote it with the alias spec


class _RenderFinder:
    def __init__(self, package: str, out_dir: Path, alias: bool):
        self.package = package
        self.out_dir = out_dir
        self.alias = alias
        # rendered module name -> is a package
        self.rendered: dict[str, bool] = {}
        for path in out_dir.rglob("*.py"):
            parts = list(path.relative_to(out_dir).with_suffix("").parts)
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            self.rendered[".".join([package, *parts])] = is_package

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.package:
            return importlib.util.spec_from_file_location(
                fullname, self.out_dir / "__init__.py", submodule_search_locations=[str(self.out_dir)]
            )
        if self.package.startswith(fullname + "."):
            # A parent of the render: whatever finder normally serves it (the
            # editable install, sys.path), else an empty namespace package.
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                found = finder.find_spec(fullname, path, target)
                if found is not None:
                    return found
            return importlib.machinery.ModuleSpec(fullname, None, is_package=True)
        if self.alias and (fullname == SOURCE_PACKAGE or fullname.startswith(SOURCE_PACKAGE + ".")):
            target_name = self.package + fullname[len(SOURCE_PACKAGE):]
            if target_name in self.rendered:
                return importlib.machinery.ModuleSpec(
                    fullname, _AliasLoader(target_name), is_package=self.rendered[target_name]
                )
            # Not rendered (e.g. hytalegym.jax.training): the original module,
            # which then imports the rendered engine through these aliases.
            parent_name = fullname.rpartition(".")[0]
            parent = sys.modules.get(parent_name)
            if parent is None or not getattr(parent, "__name__", "").startswith(self.package):
                return None  # the parent is an original package: default import
            hytalegym = sys.modules.get("hytalegym")
            if hytalegym is None:
                return None
            relative = parent_name[len(SOURCE_PACKAGE):].lstrip(".")
            search = [str(Path(root, "jax", *relative.split(".")) if relative else Path(root, "jax"))
                      for root in hytalegym.__path__]
            return importlib.machinery.PathFinder.find_spec(fullname, search)
        return None


def redirect_imports(out_dir: str | Path, *, alias: bool = True) -> str:
    """Make ``import hytalegym.jax...`` load the render at ``out_dir``.

    Call before ``hytalegym`` is first imported (load this file with
    ``importlib.util.spec_from_file_location`` to avoid importing it as part of
    ``hytalegym``). Every ``hytalegym.jax.X`` is then the very module object
    ``<package>.X`` for every module the render contains; modules it does not
    contain (``hytalegym.jax.training``...) load from the original tree and
    import the rendered engine through the aliases. Returns the package.
    """

    if any(name == "hytalegym" or name.startswith("hytalegym.") for name in sys.modules):
        raise RenderError("redirect_imports must run before hytalegym is imported")
    out = Path(out_dir).resolve()
    package = read_manifest(out)["package"]
    sys.meta_path.insert(0, _RenderFinder(package, out, alias))
    return package


# --------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hytalegym.sim.common.render", description=__doc__.split("\n\n")[0])
    parser.add_argument("--generation", required=True, choices=sorted(GENERATION_OVERLAYS))
    parser.add_argument("--out", type=Path, default=None, help="default: hytalegym/sim/<generation>/engine")
    parser.add_argument("--check", action="store_true", help="validate and resolve only; write nothing")
    args = parser.parse_args(argv)
    overlay = load_overlay(args.generation)
    out = args.out or default_out_dir(overlay)
    try:
        manifest = render(overlay, out, write=not args.check)
    except Exception as exc:  # report and fail the CLI with the reason
        print(f"render {args.generation}: REFUSED\n{exc}", file=sys.stderr)
        return 1
    totals = manifest["totals"]
    print(f"render {args.generation}: {'checked' if args.check else 'wrote ' + str(out)}")
    print(f"  source {manifest['source']['commit'][:12]}  overlay {manifest['overlay']['sha256'][:12]}  "
          f"package {manifest['package']}")
    print(f"  modules {totals['modules']}  data {totals['data_files']}  applied {totals['applied']}  "
          f"pending {totals['pending']}  changed beyond relocation {totals['files_changed_beyond_relocation']}")
    for item in manifest["applied"]:
        print(f"  applied {item['kind']:8s} {item['id']:40s} {item['path']}  (fidelity {item.get('fidelity_row') or '-'})")
    for item in manifest["pending"]:
        print(f"  pending {item['kind']:8s} {item['id']:40s} {item['path']}  [{item['status']}]")
    print(f"  shared modules importing gen-5 code: {len(manifest['external_back_edges'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
