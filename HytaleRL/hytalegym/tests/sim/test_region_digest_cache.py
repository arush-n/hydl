"""Opt-in cache of Region snapshot semantic digests (``HYTALEGYM_REGION_DIGEST_CACHE``).

A digest is keyed by the artifact file's SHA-256 and the SHA-256 of the modules the digest reads (``snapshot.py`` and
the region and geometry contracts), and is only used when it equals the manifest's ``semantic_sha256``. With the variable
unset nothing is read or written.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from hytalegym.worldgen.region import snapshot as snapshot_module
from hytalegym.worldgen.region import library as library_module
from hytalegym.worldgen.region.library import (
    REGION_DIGEST_CACHE_ENV,
    RegionArtifactLibrary,
)
from hytalegym.worldgen.region.snapshot import NativeRegionSnapshot

_MANIFEST = (
    Path(__file__).parents[3]
    / "artifacts"
    / "worldgen"
    / "region-library-v3-288"
    / "manifest.json"
)
_COUNT = 2
_KEY = 91


@pytest.fixture(autouse=True)
def _fresh_memory():
    library_module._DIGEST_CACHE_MEMORY.clear()
    yield
    library_module._DIGEST_CACHE_MEMORY.clear()


@pytest.fixture
def library():
    if not _MANIFEST.is_file():
        pytest.skip(f"Region library artifacts absent: {_MANIFEST}")
    return RegionArtifactLibrary.load(_MANIFEST, verify_artifacts=False)


def _load(library):
    return library.load_selection("train", _COUNT, selection_key=_KEY)


def _assert_same(left, right, path="snapshot"):
    if dataclasses.is_dataclass(left):
        for field in dataclasses.fields(left):
            _assert_same(getattr(left, field.name), getattr(right, field.name), f"{path}.{field.name}")
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert left.tobytes() == right.tobytes(), path
    else:
        assert left == right, path


def _cache_files(directory: Path):
    return sorted(p.name for p in directory.iterdir() if not p.name.startswith("."))


def test_unset_variable_reads_and_writes_nothing(library, monkeypatch):
    monkeypatch.delenv(REGION_DIGEST_CACHE_ENV, raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("digest cache touched while the variable is unset")

    monkeypatch.setattr(library_module, "_read_digest_cache", forbidden)
    monkeypatch.setattr(library_module, "_write_digest_cache", forbidden)
    assert len(_load(library)) == _COUNT


def test_first_load_writes_one_entry_per_artifact_and_warm_load_reads_them(library, tmp_path, monkeypatch):
    cache = tmp_path / "digests"
    cache.mkdir()
    monkeypatch.delenv(REGION_DIGEST_CACHE_ENV, raising=False)
    uncached = _load(library)
    monkeypatch.setenv(REGION_DIGEST_CACHE_ENV, str(cache))

    cold = _load(library)
    assert len(_cache_files(cache)) == _COUNT
    for file in cache.iterdir():
        assert len(file.read_text(encoding="ascii").strip()) == 64

    library_module._DIGEST_CACHE_MEMORY.clear()

    def no_hashing(self, *args, **kwargs):
        raise AssertionError("warm load recomputed a semantic digest")

    monkeypatch.setattr(NativeRegionSnapshot, "semantic_digest", no_hashing)
    monkeypatch.setattr(NativeRegionSnapshot, "semantic_artifact_digest", no_hashing)
    warm = _load(library)
    assert len(_cache_files(cache)) == _COUNT

    for expected, got_cold, got_warm in zip(uncached, cold, warm):
        _assert_same(expected, got_cold)
        _assert_same(expected, got_warm)


def test_wrong_well_formed_digest_is_ignored_and_overwritten(library, tmp_path, monkeypatch):
    cache = tmp_path / "digests"
    cache.mkdir()
    monkeypatch.setenv(REGION_DIGEST_CACHE_ENV, str(cache))
    _load(library)
    files = {p.name: p for p in cache.iterdir()}
    for path in files.values():
        path.write_text("0" * 64 + "\n", encoding="ascii")
    library_module._DIGEST_CACHE_MEMORY.clear()

    loaded = _load(library)
    assert len(loaded) == _COUNT
    for path in files.values():
        assert path.read_text(encoding="ascii").strip() != "0" * 64
    expected = {entry.semantic_sha256 for entry in library.select("train", _COUNT, selection_key=_KEY)}
    assert {p.read_text(encoding="ascii").strip() for p in files.values()} == expected


def test_changed_code_digest_misses_and_writes_a_new_key(library, tmp_path, monkeypatch):
    cache = tmp_path / "digests"
    cache.mkdir()
    monkeypatch.setenv(REGION_DIGEST_CACHE_ENV, str(cache))
    _load(library)
    before = set(_cache_files(cache))
    monkeypatch.setattr(library_module, "_DIGEST_CODE_SHA256", "f" * 64)
    library_module._DIGEST_CACHE_MEMORY.clear()

    assert len(_load(library)) == _COUNT
    after = set(_cache_files(cache))
    assert before < after
    assert len(after - before) == _COUNT
    assert all(name.endswith("-" + "f" * 16) for name in after - before)


def test_code_key_covers_every_module_the_digest_reads():
    from hytalegym.geometry import contract as geometry_contract
    from hytalegym.worldgen.region import contract as region_contract

    modules = (snapshot_module, region_contract, geometry_contract)
    sources = [Path(m.__file__).read_bytes() for m in modules]
    base = library_module._hash_sources(sources)
    assert library_module._digest_code_sha256() == base
    for index in range(len(sources)):
        changed = list(sources)
        changed[index] = changed[index] + b" "
        assert library_module._hash_sources(changed) != base, index
    # Length prefixes keep the split unambiguous.
    assert library_module._hash_sources([b"ab", b""]) != library_module._hash_sources([b"a", b"b"])
