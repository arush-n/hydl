"""The patch of ground under a run, for the map to draw.

The console used to draw an abstract grid because a flat scene has no geometry
to draw. A Region run does, and the difference matters: "the agent walked into a
wall" and "the agent stopped for no reason" look identical on a grid.

Only a **local patch** is sent — the standable nodes inside the trajectory's own
bounding box plus a margin. A region holds ~40,000 nodes and a fight visits a
few dozen, so shipping the whole graph would be most of a megabyte of JSON that
is off-screen. One patch goes out with the run payload rather than a request per
frame.

## Colour

Ground colour is derived from `palette_semantic_key`, the 256-bit stable block
identity. That has one property worth having: **the same block type is the same
colour in every region**, because the key is what makes "the same block across
regions" meaningful at all.

The hues themselves are **arbitrary**. Block *names* are not recoverable from
these captures — the snapshots carry seed, world id and chunk API and no
material names — so a colour here means "this is a different material from that
one", never "this is stone". Naming them would be invented semantics.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
import hashlib
import sys
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worlds import zones as zone_index  # noqa: E402

#: Blocks of slack around the trajectory, so the ground does not end exactly at
#: the actors and the arena reads as a place rather than a platform.
MARGIN = 14.0

#: Vertical band kept around the fight, and it is **asymmetric on purpose**.
#:
#: A horizontal box alone selects the whole column — measured 9.0 to 137.0 on
#: one arena, so the map drew every floor and cave stacked under the actors.
#: A symmetric band fixed that and introduced a worse one: on a hillside, the
#: standable ground *above* the fight is a ceiling between the camera and the
#: scene, and the map rendered the underside of it.
#:
#: So: a little headroom, because a ledge just above the fight is context, and
#: much more below, because that is the slope you are standing on.
ABOVE = 1.5
BELOW = 18.0

#: Hard cap on nodes in one payload. Above this the patch is thinned by stride,
#: which keeps the shape readable while bounding the JSON.
MAX_NODES = 4000

#: Stable codes for the terrain classes `worlds.zones.classify` emits, used to
#: colour ground nodes on a library that ships no block-semantic sidecars.
#: Ordered flattest-first so neighbouring codes are visually adjacent, and
#: 1-based so 0 keeps its meaning of "unclassified".
_TERRAIN_CLASS_CODES = {
    "open_flat": 1,
    "sheltered": 2,
    "confined": 3,
    "enclosed": 4,
    "broken": 5,
    "steep": 6,
}

#: Exposed native cells, not the solid interior. This bounds a long rollout's
#: JSON without flattening cliffs or filling caves with invented blocks.
MAX_BLOCKS = 8000
PATCH_CACHE_SIZE = 64
SEMANTIC_CAPTURE_CACHE_SIZE = 4
_PATCH_CACHE: OrderedDict[str, dict[str, Any] | None] = OrderedDict()
_PATCH_CACHE_LOCK = Lock()

FACE_TOP = 1
FACE_X_NEG = 2
FACE_X_POS = 4
FACE_Z_NEG = 8
FACE_Z_POS = 16


def _fold(key: np.ndarray) -> int:
    """A stable 32-bit fold of one block identity.

    The semantic key is 8 uint32 words; fold all of them rather than trusting
    any single word to vary between block types. `hash()` would not do: Python
    salts string and bytes hashing per process, so every server restart would
    repaint the world.
    """

    folded = 0
    for word in np.asarray(key, dtype=np.uint32).tolist():
        folded = (folded * 2654435761 + int(word)) & 0xFFFFFFFF
    return folded


def patch(seed: int, xs: list[float], ys: list[float],
          zs: list[float]) -> dict[str, Any] | None:
    """Standable ground around a trajectory, with a colour per material.

    Returns ``None`` when the region has no captured geometry to offer, which
    is the honest answer for a flat scene rather than an empty patch that would
    render as a hole.
    """

    try:
        profile = zone_index.classify(seed)
    except (ValueError, FileNotFoundError, OSError):
        return None

    position = profile.position
    x0, x1 = min(xs) - MARGIN, max(xs) + MARGIN
    z0, z1 = min(zs) - MARGIN, max(zs) + MARGIN
    y0, y1 = min(ys) - BELOW, max(ys) + ABOVE
    inside = np.flatnonzero(
        (position[:, 0] >= x0) & (position[:, 0] <= x1)
        & (position[:, 2] >= z0) & (position[:, 2] <= z1)
        & (position[:, 1] >= y0) & (position[:, 1] <= y1)
    )
    if not inside.size:
        return None

    thinned_by = 1
    if inside.size > MAX_NODES:
        thinned_by = int(np.ceil(inside.size / MAX_NODES))
        inside = inside[::thinned_by]

    local = position[inside]
    # Ground *material* is the only thing needing block-semantic sidecars, and
    # `region-library-v3-288` ships none -- they cannot be regenerated, because
    # the live server no longer produces the terrain they were captured from.
    # Unguarded, `ground_codes` raised FileNotFoundError straight out of the
    # replay route as a bare 500, taking the whole replay with it: geometry,
    # trajectory and block shell included, none of which need the sidecars.
    # `_block_shell` below already degrades this way; this call did not.
    #
    # Falling back to a constant produced one flat colour over every node, which
    # is unreadable -- the map became grey confetti with no shape. Terrain
    # *classification* needs no sidecars (it reads the traversal graph), so the
    # fallback colours by class instead: `open_flat` reads differently from
    # `steep` and `broken`, which is what makes the ground legible at all.
    # `ground_material_available` in the payload says which of the two is on, so
    # the colour is never mistaken for a block identity.
    material_available = zone_index.ground_material_available()
    if material_available:
        codes = zone_index.ground_codes(seed, local)
    else:
        label = np.asarray(profile.label, dtype=object)[inside]
        codes = np.array(
            [_TERRAIN_CLASS_CODES.get(str(value), 0) for value in label],
            dtype=np.int64,
        )
    blocks = _block_shell(seed, xs, ys, zs)

    # One colour per distinct material present, not per node.
    palette_codes = codes if blocks is None else np.concatenate(
        (codes, blocks["codes"]), axis=0)
    palette = _palette(seed, palette_codes)
    index = {code: i for i, code in enumerate(palette["codes"])}

    rendered_blocks = None if blocks is None else {
        "x": blocks["x"].tolist(),
        "y": blocks["y"].tolist(),
        "z": blocks["z"].tolist(),
        "material": [index[int(code)] for code in blocks["codes"]],
        "faces": blocks["faces"].tolist(),
        "visible": int(len(blocks["x"])),
        "available": int(blocks["available"]),
    }

    return {
        "seed": int(seed),
        "x": [round(float(v), 2) for v in local[:, 0]],
        "y": [round(float(v), 2) for v in local[:, 1]],
        "z": [round(float(v), 2) for v in local[:, 2]],
        "material": [index.get(int(c), 0) for c in codes],
        "colors": palette["colors"],
        "nodes": int(len(local)),
        "thinned_by": thinned_by,
        "blocks": rendered_blocks,
        "ground_material_available": bool(material_available),
        "note": (
            "Occupied cells and standable positions from the native Region "
            "capture. The block shell preserves cliffs, walls and overhangs; "
            "colour is a stable material identity, not an invented block name."
            if material_available
            else "Standable positions from the native Region capture. This "
            "library ships no block-semantic sidecars, so ground material and "
            "the block shell are both unavailable: colour is the terrain class "
            "(open_flat, sheltered, confined, enclosed, broken, steep), not a "
            "block identity, and cliffs and overhangs are not drawn."
        ),
    }


def cached_patch(
    cache_key: str,
    seed: int,
    xs: list[float],
    ys: list[float],
    zs: list[float],
) -> dict[str, Any] | None:
    """Hydrate one replay crop once; callers supply a content-stable identity."""

    with _PATCH_CACHE_LOCK:
        if cache_key in _PATCH_CACHE:
            value = _PATCH_CACHE.pop(cache_key)
            _PATCH_CACHE[cache_key] = value
            return deepcopy(value)
    value = patch(seed, xs, ys, zs)
    with _PATCH_CACHE_LOCK:
        _PATCH_CACHE[cache_key] = value
        while len(_PATCH_CACHE) > PATCH_CACHE_SIZE:
            _PATCH_CACHE.popitem(last=False)
    return deepcopy(value)


def cache_state() -> dict[str, Any]:
    with _PATCH_CACHE_LOCK:
        return {
            "schema": "console-terrain-replay-cache-v1",
            "entries": len(_PATCH_CACHE),
            "capacity": PATCH_CACHE_SIZE,
        }


def clear_cache() -> int:
    with _PATCH_CACHE_LOCK:
        count = len(_PATCH_CACHE)
        _PATCH_CACHE.clear()
    return count


def artifact_semantic_sha256(seed: int) -> str | None:
    """Return the immutable native Region identity for ``seed`` when present."""

    try:
        return str(zone_index._semantic_sha(seed)).lower()
    except (ValueError, FileNotFoundError, OSError, KeyError):
        return None


@lru_cache(maxsize=SEMANTIC_CAPTURE_CACHE_SIZE)
def _semantic_block_capture(
    path_text: str,
    file_mtime_ns: int,
    file_size: int,
    expected_file_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one authenticated WorldGen block sidecar for replay rendering."""

    del file_mtime_ns, file_size
    path = Path(path_text)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected_file_sha256.lower():
        raise ValueError("WorldGen block sidecar SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["cell_code"]),
            np.asarray(archive["section_known"]),
            np.asarray(archive["core_min_chunk_xz"], dtype=np.int64) - 1,
            np.asarray(archive["palette_semantic_key"], dtype=np.uint32),
        )


def semantic_patch(
    *,
    seed: int,
    sidecar: Path,
    sidecar_file_sha256: str,
    region_semantic_sha256: str,
    xs: list[float],
    ys: list[float],
    zs: list[float],
) -> dict[str, Any] | None:
    """Render a local patch from an exact, hash-pinned WorldGen sidecar."""

    path = sidecar.resolve()
    stat = path.stat()
    code, known, origin, keys = _semantic_block_capture(
        str(path), stat.st_mtime_ns, stat.st_size, sidecar_file_sha256
    )
    blocks = _surface_blocks(code, known, origin, xs, ys, zs)
    if blocks is None:
        return None
    palette = _palette_from_keys(blocks["codes"], keys)
    index = {code: i for i, code in enumerate(palette["codes"])}
    rendered = {
        "x": blocks["x"].tolist(),
        "y": blocks["y"].tolist(),
        "z": blocks["z"].tolist(),
        "material": [index[int(code)] for code in blocks["codes"]],
        "faces": blocks["faces"].tolist(),
        "visible": int(len(blocks["x"])),
        "available": int(blocks["available"]),
    }
    return {
        "seed": int(seed),
        "semantic_sha256": region_semantic_sha256.lower(),
        "x": rendered["x"],
        "y": rendered["y"],
        "z": rendered["z"],
        "material": rendered["material"],
        "colors": palette["colors"],
        "nodes": rendered["visible"],
        "thinned_by": 1,
        "blocks": rendered,
        "note": (
            "Exposed cells from the exact hash-pinned WorldGen Region sidecar. "
            "The seed selects the capture; the semantic hash authenticates it."
        ),
    }


@lru_cache(maxsize=16)
@lru_cache(maxsize=SEMANTIC_CAPTURE_CACHE_SIZE)
def _region_artifact_capture(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the block shell straight out of the Region capture.

    The block-semantic sidecar was only ever a redundant copy: every Region
    artifact already stores `__cell_code__` (25 chunks x 10 height sections x
    32^3 cells), `__section_known__` and `__core_min_chunk_xz__` -- the exact
    three arrays the shell needs, in the same layout and with the same
    `core_min - 1` capture origin.

    `region-library-v3-288` ships no sidecars and they cannot be regenerated, so
    without this the shell was simply absent: the map drew standable nodes with
    no cliffs, walls or overhangs, which reads as confetti rather than terrain.
    """

    library = zone_index.SEMANTICS.parent
    path = library / "artifacts" / f"seed_{int(seed)}.npz"
    with np.load(path, allow_pickle=True) as archive:
        return (
            np.asarray(archive["__cell_code__"]),
            np.asarray(archive["__section_known__"]),
            np.asarray(archive["__core_min_chunk_xz__"], dtype=np.int64) - 1,
        )


def _block_capture(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = (
        zone_index.SEMANTICS
        / "sidecars"
        / f"{zone_index._semantic_sha(seed)}.block-semantics.npz"
    )
    if not path.is_file():
        return _region_artifact_capture(seed)
    with np.load(path, allow_pickle=True) as archive:
        return (
            np.asarray(archive["cell_code"]),
            np.asarray(archive["section_known"]),
            np.asarray(archive["core_min_chunk_xz"], dtype=np.int64) - 1,
        )


def _block_shell(seed: int, xs: list[float], ys: list[float],
                 zs: list[float]) -> dict[str, Any] | None:
    """Exposed occupied cells around one trajectory."""

    try:
        code, known, origin = _block_capture(seed)
        return _surface_blocks(
            code,
            known,
            origin,
            xs, ys, zs,
        )
    except (ValueError, FileNotFoundError, OSError, KeyError):
        return None


def _surface_blocks(code: np.ndarray, known: np.ndarray,
                    origin_chunk: np.ndarray, xs: list[float],
                    ys: list[float], zs: list[float]) -> dict[str, Any] | None:
    """Decode a local dense crop and retain only cells with exposed faces."""

    gx = np.arange(int(np.floor(min(xs) - MARGIN)),
                   int(np.ceil(max(xs) + MARGIN)), dtype=np.int64)
    gy = np.arange(max(0, int(np.floor(min(ys) - BELOW))),
                   min(code.shape[1] * zone_index.CHUNK_SIZE,
                       int(np.ceil(max(ys) + ABOVE))), dtype=np.int64)
    gz = np.arange(int(np.floor(min(zs) - MARGIN)),
                   int(np.ceil(max(zs) + MARGIN)), dtype=np.int64)
    if not gx.size or not gy.size or not gz.size:
        return None

    x, y, z = np.meshgrid(gx, gy, gz, indexing="ij")
    cx = np.floor_divide(x, zone_index.CHUNK_SIZE) - origin_chunk[0]
    cz = np.floor_divide(z, zone_index.CHUNK_SIZE) - origin_chunk[1]
    columns = zone_index.CAPTURE_CHUNKS_PER_AXIS
    inside = ((cx >= 0) & (cx < columns) & (cz >= 0) & (cz < columns))
    column = np.clip(cx * columns + cz, 0, code.shape[0] - 1)
    section = np.clip(np.floor_divide(y, zone_index.CHUNK_SIZE),
                      0, code.shape[1] - 1)
    cell = (np.mod(y, zone_index.CHUNK_SIZE) * zone_index.CHUNK_SIZE ** 2
            + np.mod(z, zone_index.CHUNK_SIZE) * zone_index.CHUNK_SIZE
            + np.mod(x, zone_index.CHUNK_SIZE))
    material = code[column, section, cell]
    solid = inside & known[column, section] & (material != 0)

    padded = np.pad(solid, 1)
    faces = np.zeros(solid.shape, dtype=np.uint8)
    faces[~padded[1:-1, 2:, 1:-1]] |= FACE_TOP
    faces[~padded[:-2, 1:-1, 1:-1]] |= FACE_X_NEG
    faces[~padded[2:, 1:-1, 1:-1]] |= FACE_X_POS
    faces[~padded[1:-1, 1:-1, :-2]] |= FACE_Z_NEG
    faces[~padded[1:-1, 1:-1, 2:]] |= FACE_Z_POS
    bx, by, bz = np.nonzero(solid & (faces != 0))
    if not bx.size:
        return None

    available = len(bx)
    if available > MAX_BLOCKS:
        distance = np.full(available, np.inf)
        path_step = max(1, len(xs) // 256)
        for px, pz in zip(xs[::path_step], zs[::path_step]):
            distance = np.minimum(
                distance,
                (gx[bx] + 0.5 - px) ** 2 + (gz[bz] + 0.5 - pz) ** 2,
            )
        keep = np.argpartition(distance, MAX_BLOCKS - 1)[:MAX_BLOCKS]
        bx, by, bz = bx[keep], by[keep], bz[keep]

    return {
        "x": gx[bx].astype(float) + 0.5,
        # `y` is the top face, matching traversal positions and actor feet.
        "y": gy[by].astype(float) + 1.0,
        "z": gz[bz].astype(float) + 0.5,
        "codes": material[bx, by, bz].astype(np.uint16),
        "faces": faces[bx, by, bz],
        "available": available,
    }


def _palette(seed: int, codes: np.ndarray) -> dict[str, Any]:
    try:
        from adk.environments.blocks import load

        keys = load(seed).semantic_key
    except (ValueError, FileNotFoundError, OSError, KeyError):
        keys = None

    return _palette_from_keys(codes, keys)


def _palette_from_keys(
    codes: np.ndarray,
    keys: np.ndarray | None,
) -> dict[str, Any]:
    present = [int(c) for c in np.unique(codes)]
    colors: list[str] = []

    for code in present:
        if code == 0:
            # Air under a standable node means the position fell outside the
            # captured volume; draw it as a neutral rather than inventing rock.
            colors.append("#5A6472")
            continue
        if keys is not None and code < len(keys):
            folded = _fold(keys[code])
        else:
            folded = code * 2654435761
        # Hue alone collapses: with 40+ materials in one patch, two blocks 6
        # degrees apart are the same colour on screen. Saturation and lightness
        # take independent slices of the same fold so near-hues still separate,
        # while the range stays narrow enough to read as ground rather than
        # confetti.
        hue = folded % 360
        sat = 26 + (folded >> 9) % 26          # 26-52%
        light = 34 + (folded >> 17) % 26       # 34-60%
        colors.append(f"hsl({hue} {sat}% {light}%)")
    return {"codes": present, "colors": colors}
