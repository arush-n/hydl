"""What block types the captured regions contain, and what they afford.

The region snapshots carry a coarse 98-entry shape palette. The
`block-semantics-v1` sidecars carry a much finer **443-entry** semantic palette
with the fields that decide whether a block can be interacted with:

    palette_semantic_key          (N, 8) uint32  256-bit stable identity
    palette_asset_key             (N, 8) uint32
    palette_affordance_tags       (N,)   uint16  16 opaque flag bits
    palette_gather_type_index     (N,)   uint8   how it is harvested
    palette_required_tool_quality (N,)   int16   tool tier needed
    palette_rotation_index        (N,)   int32

**Affordance bits are deliberately not named here.** The Gym treats them as
opaque — `BLOCK_AFFORDANCE_TAG_BIT_COUNT = 16` and the bits go into the policy
observation individually for the network to learn from
(`observation/v3/policy/candidates/blocks.py`). Their meanings live in the
server's assets, not in anything readable from this tree. Naming them from
guesswork would be exactly the kind of invented semantics this project keeps
getting burned by, so this reports bit *patterns* and prevalence and stops
there.

Likewise `palette_semantic_key` is a 256-bit hash, not a name. It is a stable
identity for "the same block type across regions", which is what it is used for
below, and nothing more.

    python -m adk.environments.blocks
    python -m adk.environments.blocks 1618574389
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
LIBRARY = _ROOT / "HytaleRL" / "artifacts" / "worldgen" / "region-library-pilot-v2"
SEMANTICS = LIBRARY / "block-semantics-v1"

__all__ = ["BlockPalette", "load", "survey", "SEMANTICS"]


@dataclass(frozen=True)
class BlockPalette:
    seed: int
    semantic_key: np.ndarray        # (N, 8) uint32
    affordance: np.ndarray          # (N,) uint16
    gather_type: np.ndarray         # (N,) uint8
    tool_quality: np.ndarray        # (N,) int16
    valid: np.ndarray               # (N,) bool
    #: Cells per palette entry across the captured sections.
    usage: np.ndarray               # (N,) int64

    @property
    def live(self) -> np.ndarray:
        """Entries that are both valid and actually placed somewhere."""
        return self.valid & (self.usage > 0)

    def interactive(self) -> np.ndarray:
        """Entries with any affordance bit set — something can be done to them."""
        return self.live & (self.affordance != 0)


def _entry(seed: int) -> dict:
    regions = json.loads((LIBRARY / "manifest.json").read_text())
    match = next((e for e in regions["entries"] if int(e["seed"]) == seed), None)
    if match is None:
        raise ValueError(f"no region with seed {seed}")
    return match


@lru_cache(maxsize=32)
def load(seed: int) -> BlockPalette:
    """Read one region's block-semantic sidecar and count palette usage."""

    match = _entry(seed)
    path = (SEMANTICS / "sidecars"
            / f"{match['semantic_sha256']}.block-semantics.npz")
    if not path.is_file():
        raise FileNotFoundError(f"no block-semantics sidecar at {path}")
    archive = np.load(path, allow_pickle=True)

    code = archive["cell_code"]
    known = archive["section_known"]
    valid = archive["palette_valid"]
    counted = np.where(known[..., None], code, 0)
    usage = np.bincount(counted.ravel(), minlength=len(valid)).astype(np.int64)
    # Index 0 is the air/empty slot; counting it would swamp everything.
    usage[0] = 0

    return BlockPalette(
        seed=seed,
        semantic_key=np.asarray(archive["palette_semantic_key"]),
        affordance=np.asarray(archive["palette_affordance_tags"]),
        gather_type=np.asarray(archive["palette_gather_type_index"]),
        tool_quality=np.asarray(archive["palette_required_tool_quality"]),
        valid=np.asarray(valid),
        usage=usage[: len(valid)],
    )


def survey() -> list[BlockPalette]:
    regions = json.loads((LIBRARY / "manifest.json").read_text())
    return [load(int(e["seed"])) for e in regions["entries"]]


def _bits(value: int) -> str:
    return " ".join(str(b) for b in range(16) if value & (1 << b)) or "-"


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv

    if argv:
        palette = load(int(argv[0]))
        live = palette.live
        print(f"region {palette.seed}: {int(live.sum())} block types placed "
              f"({int(palette.valid.sum())} valid in palette)")
        order = np.argsort(-palette.usage)
        print(f"\n{'usage':>10}{'afford':>8}{'bits set':>14}{'gather':>8}"
              f"{'tool':>6}")
        for index in order[:15]:
            if not live[index]:
                continue
            print(f"{palette.usage[index]:>10}"
                  f"{palette.affordance[index]:>8}"
                  f"{_bits(int(palette.affordance[index])):>14}"
                  f"{palette.gather_type[index]:>8}"
                  f"{palette.tool_quality[index]:>6}")
        interactive = palette.interactive()
        print(f"\ninteractive types placed: {int(interactive.sum())} / "
              f"{int(live.sum())}")
        print(f"cells of interactive block: "
              f"{int(palette.usage[interactive].sum()):,} / "
              f"{int(palette.usage[live].sum()):,}")
        return 0

    palettes = survey()
    keys: dict[bytes, int] = {}
    affordances: dict[int, int] = {}
    gathers: dict[int, int] = {}
    tools: dict[int, int] = {}
    for palette in palettes:
        live = palette.live
        for key in palette.semantic_key[live]:
            keys[key.tobytes()] = keys.get(key.tobytes(), 0) + 1
        for value, count in zip(*np.unique(palette.affordance[live],
                                           return_counts=True)):
            affordances[int(value)] = affordances.get(int(value), 0) + int(count)
        for value, count in zip(*np.unique(palette.gather_type[live],
                                           return_counts=True)):
            gathers[int(value)] = gathers.get(int(value), 0) + int(count)
        for value, count in zip(*np.unique(palette.tool_quality[live],
                                           return_counts=True)):
            tools[int(value)] = tools.get(int(value), 0) + int(count)

    print(f"regions surveyed              : {len(palettes)}")
    print(f"distinct block types (by key) : {len(keys)}")
    shared = sum(1 for count in keys.values() if count == len(palettes))
    print(f"present in every region       : {shared}")
    print(f"present in exactly one        : "
          f"{sum(1 for c in keys.values() if c == 1)}")

    print(f"\n{'affordance':>11}{'bits set':>16}{'type-instances':>16}")
    for value in sorted(affordances, key=lambda v: -affordances[v]):
        print(f"{value:>11}{_bits(value):>16}{affordances[value]:>16}")

    print(f"\n{'gather_type':>12}{'type-instances':>16}")
    for value in sorted(gathers, key=lambda v: -gathers[v]):
        print(f"{value:>12}{gathers[value]:>16}")

    print(f"\n{'tool_quality':>13}{'type-instances':>16}")
    for value in sorted(tools):
        print(f"{value:>13}{tools[value]:>16}")

    print("\nAffordance bit meanings are server-side and deliberately not "
          "guessed here; the Gym passes all 16 bits to the policy as opaque "
          "features. Patterns above are identities, not interpretations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
