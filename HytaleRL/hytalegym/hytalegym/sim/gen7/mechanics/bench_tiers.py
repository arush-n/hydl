"""Bench tiers (gen 7, row 11b, stage 2). Switch ``Constructs.crafting_bench_tier``.

Measured on the live 0.7.2 bridge (FIDELITY-0.7.2.md section (s), jar A6C310DD; receipts
``~/.claude/jobs/31947bd7/tmp/phase4i/receipts/repo_farm_*.json``): a bench's tier is ``BenchBlock.tierLevel``, a fresh
bench is tier 1, and a recipe whose ``requiredTierLevel`` exceeds the bench's tier is refused. Farmingbench Lettuce
(requires 2) is refused at tier 1 and finished at tier 2; Cauliflower finished at tier 4; Onion is refused at tier 7 and
finished at tier 8.

Gen 5 (``zero_fill``) returns the original: every Region bench row carries tier 0 and only tier-0 requirements are
admitted (``_lossless_region_crafting``). Gen 7 (``placed_bench``) hooks ``compile_region_fieldcraft_assets``: each bench
row's tier is the scenario tier, a fresh bench (1) by default, an integer for every bench or a ``{bench id: tier}``
mapping. The overlay patch ``bench_tier_admission`` admits requirements up to tier 8; the tier gate itself is the
engine's ``_recipe_precondition`` (``requirement_tier_level <= context.bench_tier_level``), which the provider applies
to each bench context.

Limit (inferred from the engine, not measured): the simulator has one tier per bench row (per bench type), not one per
placed bench instance. The bridge's per-instance readout is ``native_crafting_bench_tier``.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from inspect import unwrap
from typing import Any

SWITCH = "crafting_bench_tier"
IMPLEMENTED = True
ZERO_FILL = "zero_fill"
PLACED_BENCH = "placed_bench"
#: A fresh bench's tier (BenchBlock.tierLevel, oracle section (s)).
FRESH_BENCH_TIER = 1
#: The highest tier a native bench reaches (Farmingbench 1 to 8; section (s)).
MAX_BENCH_TIER = 8


def _enabled(profile: Any) -> bool:
    value = profile.constructs.crafting_bench_tier
    if value == ZERO_FILL:
        return False
    if value != PLACED_BENCH:
        raise ValueError(f"unknown crafting_bench_tier switch {value!r}")
    return True


def checked_tier(tier: Any) -> int:
    if isinstance(tier, bool) or not isinstance(tier, int) or not 1 <= tier <= MAX_BENCH_TIER:
        raise ValueError(f"bench tier must be an integer in [1, {MAX_BENCH_TIER}], got {tier!r}")
    return tier


def row_tiers(bench_id_hash: Any, bench_tiers: Any, identity_words: Any) -> list[int]:
    """The tier of each bench row. ``bench_id_hash``: rows of eight uint32 words (``identity_sha256_words``).

    ``bench_tiers``: None (fresh benches), an int (every bench) or a mapping of bench id to tier.
    """

    if bench_tiers is None:
        default, named = FRESH_BENCH_TIER, {}
    elif isinstance(bench_tiers, Mapping):
        default = FRESH_BENCH_TIER
        named = {tuple(int(word) for word in identity_words(bench_id)): checked_tier(tier)
                 for bench_id, tier in bench_tiers.items()}
    else:
        default, named = checked_tier(bench_tiers), {}
    tiers = []
    for row in bench_id_hash:
        words = tuple(int(word) for word in row)
        tiers.append(named.get(words, default))
    return tiers


def construct(original: Any, profile: Any) -> Any:
    """Hook for ``compile_region_fieldcraft_assets``: the bench rows' tier from the scenario (``bench_tiers=``)."""

    if not _enabled(profile):
        return original
    env = unwrap(original).__globals__
    jnp, np, identity_words = env["jnp"], env["np"], env["identity_sha256_words"]

    @wraps(original)
    def compile_region_fieldcraft_assets(recording_path, inventory_layout, *args, bench_tiers=None, **kwargs):
        assets = original(recording_path, inventory_layout, *args, **kwargs)
        tiers = row_tiers(np.asarray(assets.bench_id_hash), bench_tiers, identity_words)
        return assets._replace(bench_tier_level=jnp.asarray(tiers, dtype=jnp.int32))

    return compile_region_fieldcraft_assets
