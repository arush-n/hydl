"""A registry so an agent can add a dashboard panel while the console is running.

The console was a fixed set of screens: adding "runs from last Tuesday" or
"everything on iron_daggers" meant editing `routes.py`, `index.html`, a JS
module and the CSS. Four files to answer one question is why nobody adds a
panel, and why questions get answered in a scratch script that is thrown away.

A panel here is **one Python file, dropped into `console/panels/`**::

    from console.core.panels import panel

    @panel("slow-runs", title="Slow runs", tab="archive", agent="world-lane")
    def slow_runs():
        return {
            "view": "rows",
            "rows": [{"cells": [r["run_id"], r["summary"]["wall_seconds"]]}
                     for r in store.index() if ...],
        }

**Nothing is restarted and nothing is pre-registered.** :func:`registry`
rescans the directory on every call, so a file written now is a panel on the
next page refresh; edit it and the change is live; delete it and the panel
goes. The route and the rendering come for free -- the frontend asks
`/api/panels` what exists and draws each by its `view`, so a panel needs **no
JavaScript at all**.

Each panel records **which agent created it** and when, so a dashboard that
several lanes are adding to stays attributable rather than becoming a pile of
anonymous cards.

## The frame contract

The original four primitives remain valid: ``facts``, ``rows``, ``series`` and
``note``. Agent-owned frames may now also use ``progress`` and ``code``, or a
``detail`` view that composes those primitives as titled ``sections``. This is
one generic composition contract rather than one renderer per agent.

Every payload may carry ``summary``, ``status``, ``tags``, ``links``,
``updated_at``, ``note`` and ``count``. A ``rows`` payload may request a bounded
scroll region with ``scroll: true``. A detail section may declare
``collapsible`` and its initial ``open`` state. The decorator's ``size``
controls the card footprint. A panel that raises is reported as failed in its own card
rather than breaking the page -- one bad panel must not take the console down,
and an agent iterating on one should see its traceback in place.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from console.core.catalog import agents

#: Where a panel may be placed. These names match ``data-panel`` in the page.
#: Old workspace spellings remain accepted so agent-owned panels do not break
#: when their presentation label changes.
TABS = (
    "status", "evidence", "run", "custom", "train", "archive", "compute", "java",
)
TAB_ALIASES = {"worldgen": "custom", "runs": "archive"}

VIEWS = ("facts", "rows", "series", "note", "progress", "code", "detail")
SIZES = ("compact", "standard", "wide")

#: Drop a `.py` file here and it becomes a panel. Scanned per request.
PANEL_DIR = Path(__file__).resolve().parents[1] / "panels"


@dataclass(frozen=True)
class Panel:
    id: str
    title: str
    tab: str
    load: Callable[[], dict[str, Any]]
    #: Lower sorts first within a tab.
    order: int = 100
    description: str = ""
    #: Seconds between automatic refreshes; 0 means load once.
    refresh: float = 0.0
    #: Who added this. Shown on the card so a shared dashboard stays
    #: attributable rather than becoming a pile of anonymous panels.
    agent: str = "unknown"
    #: Card footprint in the responsive panel grid.
    size: str = "standard"
    #: Static, searchable labels. Dynamic labels may also be returned as
    #: ``tags`` by the loader.
    tags: tuple[str, ...] = ()
    #: Set by the loader from the file's mtime, not by the author.
    source: str = ""
    added_at: float = 0.0

    def spec(self, owner: dict[str, Any] | None = None) -> dict[str, Any]:
        owner_spec = {
            "id": self.agent,
            "title": self.agent,
            "architecture": "built-in" if self.agent == "built-in" else "unspecified",
            "summary": "",
            "declares_console": self.agent == "built-in",
        }
        if owner is not None:
            owner_spec.update({
                "title": owner.get("title") or self.agent,
                "architecture": owner.get("architecture") or "unspecified",
                "summary": owner.get("summary") or "",
                "declares_console": bool(owner.get("declares_console")),
            })
        return {
            "id": self.id, "title": self.title, "tab": self.tab,
            "order": self.order, "description": self.description,
            "refresh": self.refresh, "agent": self.agent,
            "owner": owner_spec, "size": self.size, "tags": list(self.tags),
            "source": self.source, "added_at": self.added_at,
        }


_REGISTRY: dict[str, Panel] = {}
#: module name -> (mtime, panel ids it registered), so a rescan can tell a
#: changed file from an unchanged one and can drop a deleted file's panels.
_LOADED: dict[str, tuple[float, list[str]]] = {}
_IMPORTING: str | None = None
_ERRORS: dict[str, str] = {}


def panel(panel_id: str, *, title: str, tab: str = "status", order: int = 100,
          description: str = "", refresh: float = 0.0, agent: str = "unknown",
          size: str = "standard", tags: tuple[str, ...] | list[str] = ()):
    """Register an agent frame. Use as a decorator on a zero-argument loader.

    ``panel`` remains the API name for compatibility; the UI calls the result
    a frame because it can now contain several related detail sections.
    """

    canonical_tab = TAB_ALIASES.get(tab, tab)
    if canonical_tab not in TABS:
        accepted = [*TABS, *TAB_ALIASES]
        raise ValueError(f"unknown tab {tab!r}; have {accepted}")
    if size not in SIZES:
        raise ValueError(f"unknown panel size {size!r}; have {list(SIZES)}")
    refresh_value = float(refresh)
    if not math.isfinite(refresh_value) or refresh_value < 0:
        raise ValueError("panel refresh must be finite and zero or positive")
    if isinstance(tags, str):
        tags = (tags,)

    def register(load: Callable[[], dict[str, Any]]) -> Callable[[], dict[str, Any]]:
        # Ids are global, so two files claiming the same one used to overwrite
        # silently -- the second author's panel simply replaced the first
        # author's with no signal anywhere. Refuse it the same way an unknown
        # tab is refused: the loader catches this and shows it on Status with
        # the offending module named, rather than one lane's panel vanishing.
        owner = next(
            (mod for mod, (_, ids) in _LOADED.items()
             if panel_id in ids and mod != _IMPORTING),
            None,
        )
        if owner is not None:
            raise ValueError(
                f"panel id {panel_id!r} is already registered by {owner!r}; "
                "prefix agent panel ids with the agent name to keep them unique"
            )
        _REGISTRY[panel_id] = Panel(
            id=panel_id, title=title, tab=canonical_tab, load=load, order=order,
            description=description, refresh=refresh_value, agent=agent,
            size=size, tags=tuple(str(tag) for tag in tags),
            source=_IMPORTING or "",
            added_at=_LOADED.get(_IMPORTING, (0.0, []))[0]
            if _IMPORTING else 0.0)
        if _IMPORTING:
            _LOADED.setdefault(_IMPORTING, (0.0, []))[1].append(panel_id)
        return load

    return register


def scan() -> None:
    """Import new panel files, reload changed ones, forget deleted ones.

    Called before every read, so the panel set is whatever is on disk *now*.
    An agent writing a file mid-session gets a panel without a restart, which
    is the whole point -- a dashboard you have to redeploy to extend is one
    nobody extends.
    """

    global _IMPORTING

    present: dict[str, float] = {}
    sources: dict[str, Path] = {}
    if PANEL_DIR.is_dir():
        for path in sorted(PANEL_DIR.glob("*.py")):
            if path.name.startswith("_"):
                continue
            name = f"console.panels.{path.stem}"
            present[name] = path.stat().st_mtime
            sources[name] = path

    # Panels an agent shipped in `agents/<name>/console/panels.py`, so an agent
    # can extend this dashboard without editing the console. That folder is not
    # an importable package -- and requiring an `__init__.py` there would make
    # the agent-side contract more ceremony than "drop a file in" -- so these
    # load by path under a synthetic module name.
    for path in agents.panel_modules():
        try:
            mtime = path.stat().st_mtime
        except OSError:  # removed between the scan and the stat
            continue
        present[f"agents.{path.parent.parent.name}.console.panels"] = mtime
        sources[f"agents.{path.parent.parent.name}.console.panels"] = path

    # Forget panels whose file is gone.
    for name in [n for n in _LOADED if n not in present]:
        for panel_id in _LOADED[name][1]:
            _REGISTRY.pop(panel_id, None)
        _LOADED.pop(name, None)
        _ERRORS.pop(name, None)
        sys.modules.pop(name, None)

    for name, mtime in present.items():
        known = _LOADED.get(name)
        if known and known[0] == mtime:
            continue
        # A changed file re-registers its panels; drop the old ids first so a
        # renamed panel does not leave its predecessor behind.
        if known:
            for panel_id in known[1]:
                _REGISTRY.pop(panel_id, None)
        _LOADED[name] = (mtime, [])
        _ERRORS.pop(name, None)
        _IMPORTING = name
        try:
            if name.startswith("console.panels."):
                if name in sys.modules:
                    importlib.reload(sys.modules[name])
                else:
                    importlib.import_module(name)
            else:
                # Agent panel modules are loaded under a synthetic dotted name;
                # their parent is intentionally not a Python package, so
                # importlib.reload cannot resolve it. Re-exec the owned file.
                sys.modules.pop(name, None)
                spec = importlib.util.spec_from_file_location(
                    name, sources[name])
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot load panel file {sources[name]}")
                module = importlib.util.module_from_spec(spec)
                # Registered before exec so `reload` on the next changed-mtime
                # pass finds it, and popped again below if exec raises.
                sys.modules[name] = module
                spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001 - a bad panel file is not fatal
            _ERRORS[name] = traceback.format_exc(limit=6)
            if not name.startswith("console.panels."):
                sys.modules.pop(name, None)
        finally:
            _IMPORTING = None
        # `panel()` appended to a fresh list under the same key; keep the mtime.
        _LOADED[name] = (mtime, _LOADED[name][1])


def registry() -> list[Panel]:
    """Every panel currently on disk, in tab then order."""

    scan()
    return sorted(_REGISTRY.values(),
                  key=lambda p: (TABS.index(p.tab), p.order, p.id))


def specs() -> dict[str, Any]:
    """What the frontend needs to lay the dashboard out, plus any load errors."""

    panels = registry()
    owners = {agent["id"]: agent for agent in agents.index()}
    return {
        "panels": [p.spec(owners.get(p.agent)) for p in panels],
        "tabs": list(TABS),
        "tab_aliases": dict(TAB_ALIASES),
        "views": list(VIEWS),
        "sizes": list(SIZES),
        "directory": str(PANEL_DIR),
        # A file that failed to import is surfaced rather than silently
        # missing: "my panel did not appear" is otherwise unanswerable.
        "errors": [{"module": name, "traceback": tb}
                   for name, tb in sorted(_ERRORS.items())],
    }


def render(panel_id: str) -> dict[str, Any] | None:
    """Run one panel's loader, or ``None`` if no such panel is registered.

    A panel that raises returns a failed card carrying its traceback rather
    than propagating: the console is a viewer, one broken panel must not take
    down the page, and an agent iterating on a panel needs to see why it broke
    where the panel would have been.
    """

    scan()
    found = _REGISTRY.get(panel_id)
    if found is None:
        return None
    try:
        payload = dict(found.load() or {})
    except Exception:  # noqa: BLE001 - surfaced in the card, never fatal
        return {**found.spec(agents.get(found.agent)), "view": "note", "failed": True,
                "text": traceback.format_exc(limit=6)}
    view = payload.get("view", "note")
    if view not in VIEWS:
        return {**found.spec(agents.get(found.agent)), "view": "note", "failed": True,
                "text": f"panel returned unknown view {view!r}; have {list(VIEWS)}"}
    if view == "detail":
        sections = payload.get("sections", [])
        if not isinstance(sections, list):
            return {**found.spec(agents.get(found.agent)), "view": "note", "failed": True,
                    "text": "detail panel 'sections' must be a list"}
        for index, section in enumerate(sections):
            section_view = section.get("view", "note") if isinstance(section, dict) else None
            if section_view not in VIEWS or section_view == "detail":
                return {**found.spec(agents.get(found.agent)), "view": "note", "failed": True,
                        "text": f"detail section {index} has invalid view {section_view!r}"}
    # Registry metadata is authoritative. A loader can change content and
    # state, but cannot impersonate another agent or move its frame to a tab.
    return {**payload, **found.spec(agents.get(found.agent)), "view": view}
