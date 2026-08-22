"""Authored world designs as a training library, keyed by content not by seed.

The Region library identifies a world by its capture seed, so "which world did
this run train on" is answered by a number the generator chose. A seed is not a
description: two seeds from the same recipe are the same arena, and one seed
re-rolled against a different recipe is a different arena under the same name.
Selecting by seed therefore lets a run reach terrain nobody authored.

A design is identified here by ``native_recipe_digest`` instead -- the console's
own content hash over the asset graph, with ``seed`` and ``extent`` removed. Two
designs collide exactly when they describe the same world, whatever seed either
was previewed at, so a selection cannot wander off the authored set.

    python -m worlds.designs            # what is available, and what is usable
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Mapping

__all__ = [
    "Design",
    "DESIGN_RANK_METHOD",
    "discover",
    "by_id",
    "select",
]

#: Mirrors the Region library's ``sha256_rank_v1``: rank the pool by a keyed
#: digest and take a prefix. Same shape, so a reader who knows one knows both.
DESIGN_RANK_METHOD = "design_sha256_rank_v1"

#: A pursuit arena has to be walkable ground, not a canyon the evader falls
#: into. These are the thresholds `unsuitable()` reports against; every one is
#: measured by the console's own preview receipt rather than re-derived here.
MAXIMUM_TRAINING_RELIEF = 4.0
MAXIMUM_TRAINING_SLOPE = 0.5
MINIMUM_TRAVERSABLE_FRACTION = 0.95


@dataclass(frozen=True, slots=True)
class Design:
    """One authored world, addressed by the content of its recipe."""

    design_id: str
    path: Path
    #: Seed-independent identity. Equal digests mean the same authored world.
    recipe_digest: str
    name: str
    seed: int
    base_height: float
    metrics: Mapping[str, float]

    @property
    def relief(self) -> float:
        return float(self.metrics.get("relief", 0.0))

    @property
    def maximum_slope(self) -> float:
        return float(self.metrics.get("maximum_slope", 0.0))

    @property
    def cave_volume(self) -> float:
        return float(self.metrics.get("estimated_cave_volume", 0.0))

    @property
    def structure_count(self) -> int:
        return int(self.metrics.get("structure_count", 0))

    @property
    def traversable_fraction(self) -> float:
        return float(self.metrics.get("traversable_fraction", 0.0))

    def unsuitable(self) -> tuple[str, ...]:
        """Why this design cannot back a pursuit arena, or an empty tuple.

        Reported rather than raised so a caller can list a whole library and
        show the reasons, and so a selection that filters these out can say
        what it dropped instead of silently returning fewer worlds.
        """

        reasons: list[str] = []
        if self.cave_volume > 0.0:
            reasons.append(
                f"caves: estimated volume {self.cave_volume:g} "
                f"({int(self.metrics.get('cave_network_count', 0))} networks)"
            )
        if self.relief > MAXIMUM_TRAINING_RELIEF:
            reasons.append(
                f"relief {self.relief:g} > {MAXIMUM_TRAINING_RELIEF:g}"
            )
        if self.maximum_slope > MAXIMUM_TRAINING_SLOPE:
            reasons.append(
                f"slope {self.maximum_slope:g} > {MAXIMUM_TRAINING_SLOPE:g}"
            )
        if self.traversable_fraction < MINIMUM_TRAVERSABLE_FRACTION:
            reasons.append(
                f"traversable {self.traversable_fraction:g} < "
                f"{MINIMUM_TRAVERSABLE_FRACTION:g}"
            )
        return tuple(reasons)

    @property
    def usable(self) -> bool:
        return not self.unsuitable()


def _read(path: Path) -> Design | None:
    """Parse one saved-design record, or None if it is not one.

    The store is scratch and holds partial writes, so an unreadable file is
    skipped rather than failing a whole listing.
    """

    from console.core.worlds.worldgen import native_recipe_digest

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        config = record["config"]
        receipt = record["preview_receipt"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return Design(
        design_id=str(record.get("design_id") or path.stem),
        path=path,
        recipe_digest=native_recipe_digest(config),
        name=str(config.get("name", path.stem)),
        seed=int(config.get("seed", 0)),
        base_height=float(config.get("base_height", 0.0)),
        metrics=dict(metrics),
    )


@lru_cache(maxsize=8)
def _discover(root: str | None) -> tuple[Design, ...]:
    from console.core.worlds.worldgen import (
        DESIGN_DIR,
        LEGACY_DESIGN_DIR,
        SEED_DESIGN_DIR,
    )

    directories = (
        (Path(root),)
        if root
        else (DESIGN_DIR, LEGACY_DESIGN_DIR, SEED_DESIGN_DIR)
    )
    found: dict[str, Design] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            design = _read(path)
            # A design present in both the live store and the seed copies is
            # one design; the live store wins because it may have been edited.
            if design is not None:
                found.setdefault(design.design_id, design)
    return tuple(found.values())


def discover(root: Path | str | None = None) -> tuple[Design, ...]:
    """Every readable design, live store first then the seeded copies."""

    return _discover(str(root) if root is not None else None)


def by_id(design_id: str, root: Path | str | None = None) -> Design:
    """One design by its id, or a ValueError naming what is available."""

    for design in discover(root):
        if design.design_id == design_id:
            return design
    available = ", ".join(sorted(d.design_id for d in discover(root))) or "none"
    raise ValueError(f"no design {design_id!r}; available: {available}")


def _rank(key: int, digest: str) -> bytes:
    return hashlib.sha256(
        f"{DESIGN_RANK_METHOD}\0{key}\0{digest}".encode()
    ).digest()


def select(
    count: int = 1,
    *,
    selection_key: int = 0,
    design_ids: tuple[str, ...] | None = None,
    root: Path | str | None = None,
    require_usable: bool = True,
) -> tuple[Design, ...]:
    """Pick `count` designs, by explicit id or by keyed rank over the pool.

    `design_ids` is the deterministic route and is returned in the order given.
    Otherwise the pool is ranked by ``sha256(method, key, recipe_digest)``, so
    the choice varies with `selection_key` and is stable for a fixed library --
    and because the digest excludes the seed, re-previewing a design at a new
    seed does not move it in the ranking or admit a world nobody authored.
    """

    if count < 1:
        raise ValueError("design selection needs a positive count")
    if design_ids:
        chosen = tuple(by_id(name, root) for name in design_ids)
        if require_usable:
            for design in chosen:
                reasons = design.unsuitable()
                if reasons:
                    raise ValueError(
                        f"design {design.design_id!r} cannot back a pursuit "
                        f"arena: {'; '.join(reasons)}"
                    )
        return chosen
    pool = [d for d in discover(root) if d.usable or not require_usable]
    if not pool:
        raise ValueError(
            "no usable design in the library; run `python -m worlds.designs` "
            "to see why each was rejected"
        )
    ranked = sorted(pool, key=lambda d: (_rank(selection_key, d.recipe_digest),
                                         d.design_id))
    # Fewer designs than asked for is a real answer, not an error: the library
    # is what the author saved. Cycling would train several lanes on one world
    # while the manifest claimed `count` distinct ones.
    return tuple(ranked[:count])


def main() -> None:
    designs = discover()
    if not designs:
        print("no designs found")
        return
    width = max(len(d.design_id) for d in designs)
    for design in sorted(designs, key=lambda d: (not d.usable, d.design_id)):
        reasons = design.unsuitable()
        mark = "ok    " if not reasons else "reject"
        print(
            f"{mark} {design.design_id:<{width}} "
            f"digest {design.recipe_digest[:10]} "
            f"base_height {design.base_height:7.2f} "
            f"relief {design.relief:6.2f} "
            f"structures {design.structure_count:3d}"
        )
        for reason in reasons:
            print(f"       {'':<{width}} - {reason}")


if __name__ == "__main__":
    main()
