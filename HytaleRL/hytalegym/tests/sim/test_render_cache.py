"""Cached renders (``sim/common/render.py`` ``render_cached``): one render per key, shared, byte-identical to a fresh one.

Asked by the skill layer (2026-10-25): each gen-7 render takes 30 to 85 s under load and every slow test and probe rendered
its own copy. The key covers the committed sources, the overlay and profile, the working ``hytalegym/sim`` package and the
assets archive's identity, so a cached render is only reused when a fresh one would be identical.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_cached_render_is_reused_and_equals_a_fresh_render(tmp_path):
    source = HeadSource()
    cache = tmp_path / "cache"
    first = R.render_cached(GEN7, source=source, cache_root=cache)
    stamp = (first / R.MANIFEST_NAME).stat().st_mtime_ns
    second = R.render_cached(GEN7, source=source, cache_root=cache)
    assert second == first
    assert (second / R.MANIFEST_NAME).stat().st_mtime_ns == stamp  # not rendered again
    assert [p.name for p in cache.iterdir()] == [first.name]  # no private copies left behind

    fresh = tmp_path / "fresh"
    R.render(GEN7, fresh, source=source)
    assert _files(first) == _files(fresh)


def test_the_key_follows_the_overlay():
    source = HeadSource()
    changed = dataclasses.replace(GEN7, description=GEN7.description + " (changed)")
    assert R.render_key(changed, source) != R.render_key(GEN7, source)
    assert R.render_key(GEN7, source) == R.render_key(GEN7, source)
