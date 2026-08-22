"""Find water in the captured regions, so swimming can be tested faithfully.

Swimming and drowning are fully modelled and demonstrably live -- oxygen max
100, suffocating -3.0 per 0.5s (~16.7s to drown), breathable +25.0 per 0.5s,
breathing damage every 1.0s, and the regen timers tick every step. But oxygen
never leaves 100.0 on any scene built so far, because none of them contain
fluid. The mechanic is reachable in principle and unreached in practice.

Rather than synthesise water -- which would be a control fixture like
`open_flat`, never a fidelity claim -- this locates the water that the 20 native
captures actually contain (0.2-0.6% of cells) and reports where, so a spawn can
be aimed at it.

Reads the region `.npz` files directly: no JAX, no scene build, no native
session. Cheap enough to sweep the whole library.

    from adk.environments import water
    water.survey()                      # every region, ranked by water
    water.columns(seed, minimum_depth=2)  # submersible spots in one region
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
LIBRARY = _ROOT / "HytaleRL" / "artifacts" / "worldgen" / "region-library-pilot-v2"

#: A capture is 5x5 sections horizontally by 10 vertically, each 32^3 cells.
SECTION = 32

__all__ = ["RegionWater", "survey", "columns", "LIBRARY"]


@dataclass(frozen=True)
class RegionWater:
    seed: int
    split: str
    water_cells: int
    total_cells: int
    #: Deepest continuous water column found, in cells.
    deepest: int
    #: Vertical section indices that contain any water.
    layers: tuple[int, ...]

    @property
    def fraction(self) -> float:
        return self.water_cells / max(self.total_cells, 1)


def _water_mask(archive) -> tuple[np.ndarray, np.ndarray]:
    """(water mask, section-known mask) for one region archive."""

    code = archive["__cell_code__"]                 # (25, 10, 32768) uint16
    known = archive["__section_known__"]            # (25, 10)
    fluid_level = archive["cell_fluid_level"]       # (palette,)
    safe = np.clip(code, 0, len(fluid_level) - 1)
    return (fluid_level[safe] > 0) & (code != 0), known


def survey(library: Path = LIBRARY) -> list[RegionWater]:
    """Every region, ranked by how much water it holds."""

    manifest = json.loads((library / "manifest.json").read_text())
    results = []
    for entry in manifest["entries"]:
        archive = np.load(library / entry["path"])
        water, known = _water_mask(archive)
        live = known[..., None]
        counted = water & live

        # Which vertical section layers hold water, and the deepest run within
        # a single cell column. Depth matters: a one-cell puddle cannot submerge
        # anything, and submersion is what drives oxygen.
        layers = tuple(int(y) for y in range(water.shape[1])
                       if bool(counted[:, y].any()))
        deepest = 0
        if layers:
            # Cells are flattened per section; reshape to (32,32,32) as z,y,x
            # so a column is a fixed (z,x) across y. Only the richest section is
            # examined -- this is a locator, not a survey of every column.
            richest = int(np.argmax(counted.sum(axis=(1, 2))))
            block = counted[richest].reshape(
                water.shape[1], SECTION, SECTION, SECTION)
            # (y_section, z, y, x) -> collapse the two vertical axes
            column = block.transpose(1, 3, 0, 2).reshape(
                SECTION, SECTION, -1)
            for run in column.reshape(-1, column.shape[-1]):
                if not run.any():
                    continue
                best = current = 0
                for cell in run:
                    current = current + 1 if cell else 0
                    best = max(best, current)
                deepest = max(deepest, best)

        results.append(RegionWater(
            seed=int(entry["seed"]), split=entry["split"],
            water_cells=int(counted.sum()),
            total_cells=int(np.broadcast_to(live, water.shape).sum()),
            deepest=deepest, layers=layers))
    return sorted(results, key=lambda r: r.water_cells, reverse=True)


def columns(seed: int, *, minimum_depth: int = 2,
            library: Path = LIBRARY) -> list[tuple[int, int, int]]:
    """Section-local (z, x, depth) of water columns at least `minimum_depth` deep.

    Returned coordinates are section-local, not world-space. Converting to a
    spawn point needs the capture's `__core_min_chunk_xz__` origin; treat this
    as "does a submersible column exist and how deep", not as a spawn API.
    """

    manifest = json.loads((library / "manifest.json").read_text())
    entry = next((e for e in manifest["entries"] if int(e["seed"]) == seed), None)
    if entry is None:
        raise ValueError(f"no region with seed {seed}")
    archive = np.load(library / entry["path"])
    water, known = _water_mask(archive)
    counted = water & known[..., None]
    richest = int(np.argmax(counted.sum(axis=(1, 2))))
    block = counted[richest].reshape(water.shape[1], SECTION, SECTION, SECTION)
    column = block.transpose(1, 3, 0, 2).reshape(SECTION, SECTION, -1)

    found = []
    for z in range(SECTION):
        for x in range(SECTION):
            run = column[z, x]
            best = current = 0
            for cell in run:
                current = current + 1 if cell else 0
                best = max(best, current)
            if best >= minimum_depth:
                found.append((z, x, best))
    return sorted(found, key=lambda item: item[2], reverse=True)


def main() -> int:
    rows = survey()
    print(f"{'seed':<12}{'split':<9}{'water_cells':>12}{'fraction':>10}"
          f"{'deepest':>9}  layers")
    for row in rows:
        print(f"{row.seed:<12}{row.split:<9}{row.water_cells:>12}"
              f"{row.fraction:>10.4%}{row.deepest:>9}  {list(row.layers)}")

    wet = [r for r in rows if r.deepest >= 2]
    print(f"\nregions with a column at least 2 cells deep: {len(wet)} / {len(rows)}")
    if not wet:
        print("NO submersible water in the captured library -- drowning cannot "
              "be tested faithfully without new captures.")
        return 0

    # `rows` is sorted by water VOLUME; the deepest column is a different
    # ranking, and conflating them reported a 23-cell region as the deepest
    # while a 32-cell one was in the same table.
    best = max(wet, key=lambda r: r.deepest)
    wettest = rows[0]
    print(f"deepest column : seed {best.seed} ({best.split}) at "
          f"{best.deepest} cells")
    print(f"most water     : seed {wettest.seed} ({wettest.split}) at "
          f"{wettest.fraction:.4%}")
    spots = columns(best.seed, minimum_depth=2)
    print(f"  submersible columns in its richest section: {len(spots)}")
    print(f"  top 5 (z, x, depth): {spots[:5]}")
    print("\ndrowning IS reachable in the captured library. The default spawn "
          "simply never lands in it -- aim a spawn at one of these columns "
          "rather than synthesising water.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
