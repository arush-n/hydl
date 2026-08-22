"""Deterministic custom-world design previews for the console.

This module is intentionally not a second Hytale generator.  It gives an
author a fast, coherent preview of the *design* they are about to compile into
WorldGen V2 assets, and records the parameters needed for that native step.
Only a native generation + Region capture can claim exact Hytale/JAX content.

The preview still has useful contracts of its own:

* one normalized design and seed is bit-for-bit reproducible;
* changing the seed changes a spatially coherent terrain field, not a bag of
  independently sampled decorations;
* water, biomes, trees, structures and spawn placement are derived from that
  same field; and
* saved designs are portable JSON rather than browser-only state.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


SCHEMA = "hytalerl_worldgen_v2_console_design_v1"
VERSION = 2
PREVIEW_SCHEMA = "hytalerl_worldgen_v2_console_preview_v1"
BATCH_SCHEMA = "hytalerl_worldgen_v2_console_seed_batch_v1"
VOXEL_VOLUME_SCHEMA = "hytalerl_worldgen_v2_semantic_voxel_volume_v1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Saved designs live with the other world artifacts, not in the console's
#: scratch tree. A design is an input a training run is identified by, so it has
#: to outlive a scratch wipe.
DESIGN_DIR = PROJECT_ROOT / "artifacts" / "worlds" / "designs"
#: Where designs used to be written. Still read so records saved before the move
#: stay discoverable; nothing is written here any more.
LEGACY_DESIGN_DIR = (
    PROJECT_ROOT / ".codex-local" / "console" / "worldgen-designs"
)
#: Derived from the legacy root on purpose: builds and minigames did not move,
#: and deriving them from `DESIGN_DIR` would have relocated both silently.
BUILD_DIR = LEGACY_DESIGN_DIR.parent / "worldgen-builds"

#: THE flat world. Reference this id rather than writing a design id literal,
#: so there is one answer to "which flat world" and it survives a re-import.
#: `save()` names records `<slug>-<digest10>`, which is fine for one-off designs
#: and useless for a design other code has to find by name.
CANONICAL_FLAT_DESIGN_ID = "flat_world_main"
#: Seed copies of designs that must outlive the local design store, which is
#: scratch and gets wiped. `ensure_canonical_designs()` restores any missing.
SEED_DESIGN_DIR = Path(__file__).resolve().parent / "designs"


def ensure_canonical_designs() -> list[str]:
    """Restore seeded designs into the local store. Returns what it wrote.

    Non-destructive: a record already in the store is left alone, so a design
    edited locally is never clobbered by the seed copy.
    """

    if not SEED_DESIGN_DIR.is_dir():
        return []
    restored: list[str] = []
    for seed in sorted(SEED_DESIGN_DIR.glob("*.json")):
        target = DESIGN_DIR / seed.name
        if target.is_file():
            continue
        DESIGN_DIR.mkdir(parents=True, exist_ok=True)
        target.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
        restored.append(seed.stem)
    return restored
_CAPTURE_ROOT_OVERRIDE = os.environ.get("HYTALERL_WORLDGEN_CAPTURE_ROOT", "").strip()
CAPTURE_ROOTS = (
    PROJECT_ROOT / ".codex-local" / "console" / "worldgen-captures",
    PROJECT_ROOT / "experimental" / "worldgen-v2" / "evidence",
) + ((Path(_CAPTURE_ROOT_OVERRIDE),) if _CAPTURE_ROOT_OVERRIDE else ())
_CAPTURE_ID = re.compile(r"[0-9a-f]{64}")
CAPTURE_RECORD_CACHE_SIZE = 128
CAPTURE_VOLUME_CACHE_SIZE = 16

# The renderer uses one vertex per sample.  49 gives 2,304 terrain quads: high
# enough to read ridges and channels, small enough to redraw while a slider is
# moving on an ordinary laptop.
PREVIEW_RESOLUTION = 49
FAST_BATCH_RESOLUTIONS = (17, 25, 33)
MAX_BATCH_SEEDS = 512
VOXEL_TARGET_COLUMNS = 64
VOXEL_WORLD_HEIGHT = 320
# One exact Region artifact exposes a 3x3-chunk (96x96-block) usable core;
# its surrounding chunks are capture halo, not additional environment area.
# The existing environment-recipe contract supports any rectangular bounds
# window inside that core. Larger requested extents need multiple native
# captures and a stitched JAX lookup; keep that gap visible in every receipt.
NATIVE_CAPTURE_CORE_BLOCKS = 96
# At the default 128-block extent, adjacent samples are 2.67 blocks apart.  A
# 0.85 rise/run cutoff is a conservative broad-route screen, not a promise
# that every voxel step is walkable; native traversal remains the authority.
TRAVERSABLE_SLOPE = 0.85

BIOMES = (
    {"id": 0, "name": "deep_water", "color": "#255e89"},
    {"id": 1, "name": "shallow_water", "color": "#3e8ca8"},
    {"id": 2, "name": "shore", "color": "#c8aa67"},
    {"id": 3, "name": "grassland", "color": "#6f9d53"},
    {"id": 4, "name": "forest", "color": "#356c42"},
    {"id": 5, "name": "rock", "color": "#777b78"},
    {"id": 6, "name": "alpine", "color": "#a7aea9"},
    {"id": 7, "name": "snow", "color": "#e8edf0"},
    {"id": 8, "name": "dryland", "color": "#a98a52"},
)

_VOLUME_PALETTE = (
    {"id": 0, "name": "air", "color": "#000000", "opacity": 0.0,
     "layer": "air"},
    {"id": 1, "name": "rock_mass", "color": "#596166", "opacity": 1.0,
     "layer": "terrain"},
    {"id": 2, "name": "subsoil", "color": "#705c43", "opacity": 1.0,
     "layer": "terrain"},
    *tuple({
        "id": 10 + entry["id"],
        "name": f"surface_{entry['name']}",
        "color": entry["color"],
        "opacity": 1.0,
        "layer": "terrain",
    } for entry in BIOMES),
    {"id": 20, "name": "cave_wall", "color": "#66566f", "opacity": 1.0,
     "layer": "terrain"},
    {"id": 30, "name": "water", "color": "#3d91b4", "opacity": 0.58,
     "layer": "water"},
    {"id": 40, "name": "tree_trunk", "color": "#5b4027", "opacity": 1.0,
     "layer": "trees"},
    {"id": 41, "name": "tree_leaves", "color": "#356c42", "opacity": 1.0,
     "layer": "trees"},
    {"id": 50, "name": "structure_stone", "color": "#8d8b83", "opacity": 1.0,
     "layer": "structures"},
    {"id": 51, "name": "structure_wood", "color": "#9a6d43", "opacity": 1.0,
     "layer": "structures"},
    {"id": 60, "name": "traversal_route", "color": "#d1b36a", "opacity": 1.0,
     "layer": "routes"},
)


DEFAULTS: dict[str, Any] = {
    "name": "mountain-forest-lakes",
    "seed": 20260810,
    "world_width": 128,
    "world_depth": 128,
    "generation_mode": "authored_family",
    "world_variation": 0.72,
    "macro_region_count": 6,
    "region_blend": 0.58,
    "terrain_mountain_weight": 0.82,
    "terrain_hill_weight": 0.58,
    "terrain_valley_weight": 0.42,
    "terrain_plateau_weight": 0.24,
    "terrain_canyon_weight": 0.22,
    "terrain_pillar_weight": 0.12,
    "terrain_crater_weight": 0.14,
    "terrain_flat_weight": 0.08,
    "route_density": 0.52,
    "route_width": 5.0,
    "traversal_bias": 0.46,
    "structure_variety": 0.72,
    "terrain_style": "natural_mountains",
    "style_strength": 0.78,
    "base_height": 112.0,
    "relief": 58.0,
    "mountain_scale": 72.0,
    "roughness": 0.58,
    "ridge_strength": 0.72,
    "erosion": 0.34,
    "terrace_steps": 7,
    "pillar_density": 0.34,
    "pillar_width": 9.0,
    "pillar_height": 48.0,
    "canyon_strength": 0.5,
    "crater_density": 0.28,
    "crater_depth": 26.0,
    "water_level": 108.0,
    "lake_coverage": 0.34,
    "river_strength": 0.38,
    "moisture": 0.72,
    "temperature": 0.48,
    "snow_line": 148.0,
    "forest_density": 0.72,
    "tree_spacing": 7.0,
    "cave_style": "tunnels",
    "cave_density": 0.38,
    "cave_scale": 28.0,
    "cave_verticality": 0.32,
    "cave_entrances": 0.42,
    "structure_density": 0.68,
    "structure_spacing": 42.0,
    "structure_jitter": 0.34,
    "structure_set": "mixed_ruins",
    "structure_layout": "scattered",
    "structure_material": "shale",
    "objective": "exploration",
    "spawn_safety_radius": 8.0,
}


PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "mountain_forest_lakes",
        "name": "Mountain forest + lakes",
        "description": "Tall connected ranges, dense woodland, lakes, a river and scattered ruins.",
        "values": {},
    },
    {
        "id": "alpine_expedition",
        "name": "Alpine expedition",
        "description": "Sharper ridges, a lower tree line, snow fields and sparse watchtowers.",
        "values": {
            "name": "alpine-expedition",
            "terrain_style": "alpine_ridges",
            "relief": 68.0, "mountain_scale": 64.0, "roughness": 0.35,
            "ridge_strength": 0.55, "erosion": 0.6, "water_level": 102.0,
            "lake_coverage": 0.2, "moisture": 0.48, "temperature": 0.26,
            "snow_line": 132.0, "forest_density": 0.42,
            "structure_density": 0.42, "structure_set": "watchtowers",
            "objective": "traversal",
        },
    },
    {
        "id": "forested_archipelago",
        "name": "Forested archipelago",
        "description": "Low wooded islands, broad water and landmark ruins for navigation tasks.",
        "values": {
            "name": "forested-archipelago",
            "terrain_style": "archipelago", "style_strength": 0.86,
            "base_height": 104.0, "relief": 32.0, "mountain_scale": 54.0,
            "roughness": 0.42, "ridge_strength": 0.3, "erosion": 0.62,
            "water_level": 107.0, "lake_coverage": 0.82,
            "river_strength": 0.15, "moisture": 0.9, "temperature": 0.68,
            "snow_line": 190.0, "forest_density": 0.86,
            "tree_spacing": 6.0, "structure_density": 0.55,
            "structure_spacing": 36.0, "structure_set": "mixed_ruins",
            "cave_style": "none", "cave_density": 0.0,
            "objective": "ruin_expedition",
        },
    },
    {
        "id": "dry_highlands",
        "name": "Dry highlands",
        "description": "Long eroded mesas, scarce water, scrub trees and fortified camps.",
        "values": {
            "name": "dry-highlands",
            "terrain_style": "desert_mesas", "style_strength": 0.82,
            "base_height": 116.0, "relief": 48.0, "mountain_scale": 92.0,
            "roughness": 0.38, "ridge_strength": 0.48, "erosion": 0.72,
            "water_level": 91.0, "lake_coverage": 0.08,
            "river_strength": 0.2, "moisture": 0.2, "temperature": 0.84,
            "snow_line": 220.0, "forest_density": 0.16,
            "tree_spacing": 11.0, "structure_density": 0.58,
            "structure_spacing": 46.0, "structure_set": "fortified_camps",
            "objective": "survival",
        },
    },
    {
        "id": "woodland_valley",
        "name": "Woodland valley",
        "description": "A gentler, highly traversable valley with dense trees and village sites.",
        "values": {
            "name": "woodland-valley",
            "terrain_style": "woodland_valley", "style_strength": 0.72,
            "base_height": 104.0, "relief": 28.0, "mountain_scale": 105.0,
            "roughness": 0.28, "ridge_strength": 0.24, "erosion": 0.8,
            "water_level": 103.0, "lake_coverage": 0.16,
            "river_strength": 0.56, "moisture": 0.84, "temperature": 0.58,
            "snow_line": 190.0, "forest_density": 0.9,
            "tree_spacing": 6.0, "structure_density": 0.8,
            "structure_spacing": 34.0, "structure_set": "villages",
            "objective": "exploration",
        },
    },
    {
        "id": "flat_pillar_arena",
        "name": "Flat pillar field",
        "description": "A deliberate flat playfield with generated stone pillars and no incidental ecology.",
        "values": {
            "name": "flat-pillar-field", "terrain_style": "flat",
            "style_strength": 1.0, "base_height": 96.0, "relief": 4.0,
            "roughness": 0.0, "ridge_strength": 0.0, "erosion": 1.0,
            "water_level": 64.0, "lake_coverage": 0.0,
            "river_strength": 0.0, "forest_density": 0.0,
            "cave_style": "none", "cave_density": 0.0,
            "structure_density": 0.9, "structure_spacing": 20.0,
            "structure_jitter": 0.12, "structure_set": "arena_pillars",
            "structure_layout": "grid", "objective": "traversal",
        },
    },
    {
        "id": "marble_pillar_wilds",
        "name": "Marble pillar wilds",
        "description": "Native-template-backed pillar terrain with broad columns, sparse water and ruins between them.",
        "values": {
            "name": "marble-pillar-wilds", "terrain_style": "pillar_fields",
            "style_strength": 0.94, "base_height": 92.0, "relief": 70.0,
            "pillar_density": 0.56, "pillar_width": 10.0,
            "pillar_height": 66.0, "water_level": 88.0,
            "lake_coverage": 0.1, "river_strength": 0.0,
            "forest_density": 0.14, "cave_style": "none",
            "cave_density": 0.0, "structure_density": 0.5,
            "structure_set": "mixed_ruins",
        },
    },
    {
        "id": "terraced_mesa_routes",
        "name": "Terraced mesa routes",
        "description": "Stepped highlands and canyon cuts for route and elevation curriculum design.",
        "values": {
            "name": "terraced-mesa-routes", "terrain_style": "terraced_highlands",
            "style_strength": 0.92, "base_height": 108.0, "relief": 52.0,
            "terrace_steps": 8, "canyon_strength": 0.6,
            "water_level": 82.0, "lake_coverage": 0.0,
            "river_strength": 0.18, "moisture": 0.18,
            "temperature": 0.82, "forest_density": 0.08,
            "cave_style": "tunnels", "cave_density": 0.24,
            "structure_set": "fortified_camps", "structure_layout": "route",
            "objective": "traversal",
        },
    },
    {
        "id": "crater_lake",
        "name": "Crater lake",
        "description": "Overlapping impact basins with a central lake, rim landmarks and optional cavern pockets.",
        "values": {
            "name": "crater-lake", "terrain_style": "cratered",
            "style_strength": 0.95, "base_height": 112.0, "relief": 46.0,
            "crater_density": 0.62, "crater_depth": 42.0,
            "water_level": 106.0, "lake_coverage": 0.28,
            "river_strength": 0.0, "forest_density": 0.42,
            "cave_style": "caverns", "cave_density": 0.36,
            "structure_layout": "ring", "structure_set": "watchtowers",
            "objective": "exploration",
        },
    },
    {
        "id": "canyon_labyrinth",
        "name": "Canyon + cave labyrinth",
        "description": "Branching surface canyons above a dense, vertically connected cave network.",
        "values": {
            "name": "canyon-cave-labyrinth", "terrain_style": "canyon_badlands",
            "style_strength": 0.9, "base_height": 120.0, "relief": 58.0,
            "canyon_strength": 0.9, "water_level": 82.0,
            "lake_coverage": 0.0, "river_strength": 0.22,
            "moisture": 0.28, "temperature": 0.74,
            "forest_density": 0.1, "cave_style": "labyrinth",
            "cave_density": 0.82, "cave_scale": 20.0,
            "cave_verticality": 0.62, "cave_entrances": 0.68,
            "structure_set": "mixed_ruins", "structure_layout": "clusters",
            "objective": "traversal",
        },
    },
    {
        "id": "rooted_sky_islands",
        "name": "Rooted sky islands",
        "description": "Surreal separated highlands mapped to Hytale's experimental rooted-island template.",
        "values": {
            "name": "rooted-sky-islands", "terrain_style": "floating_islands",
            "style_strength": 0.96, "base_height": 128.0, "relief": 62.0,
            "water_level": 72.0, "lake_coverage": 0.0,
            "river_strength": 0.0, "forest_density": 0.58,
            "cave_style": "mixed", "cave_density": 0.46,
            "structure_density": 0.42, "structure_layout": "landmarks",
            "structure_set": "shrines", "objective": "exploration",
        },
    },
)


FIELDS: dict[str, dict[str, Any]] = {
    "name": {"type": "text", "label": "design name", "group": "design"},
    "seed": {"type": "integer", "label": "seed", "min": 0, "max": 2_147_483_647, "step": 1, "group": "design"},
    "world_width": {"type": "choice", "label": "world width", "choices": [32, 48, 64, 96, 128, 192, 256, 384, 512], "min": 32, "max": 512, "unit": "blocks", "group": "design"},
    "world_depth": {"type": "choice", "label": "world length (Z)", "choices": [32, 48, 64, 96, 128, 192, 256, 384, 512], "min": 32, "max": 512, "unit": "blocks", "group": "design"},
    "generation_mode": {"type": "choice", "label": "generation source", "choices": ["authored_family", "procedural_genome"], "group": "design"},
    "world_variation": {"type": "number", "label": "seed variation", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "macro_region_count": {"type": "integer", "label": "macro regions", "min": 1, "max": 16, "step": 1, "group": "randomization"},
    "region_blend": {"type": "number", "label": "region blending", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "terrain_mountain_weight": {"type": "number", "label": "mountain probability", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "terrain_hill_weight": {"type": "number", "label": "hill probability", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "terrain_valley_weight": {"type": "number", "label": "valley probability", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "terrain_plateau_weight": {"type": "number", "label": "plateau probability", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "terrain_canyon_weight": {"type": "number", "label": "canyon probability", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "terrain_pillar_weight": {"type": "number", "label": "pillar probability", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "terrain_crater_weight": {"type": "number", "label": "crater probability", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "terrain_flat_weight": {"type": "number", "label": "flat probability", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "route_density": {"type": "number", "label": "route connectivity", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "route_width": {"type": "number", "label": "route width", "min": 1, "max": 16, "step": 0.5, "unit": "blocks", "group": "randomization"},
    "traversal_bias": {"type": "number", "label": "traversable corridor bias", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "structure_variety": {"type": "number", "label": "structure variety", "min": 0, "max": 1, "step": 0.01, "group": "randomization"},
    "terrain_style": {"type": "choice", "label": "authored terrain family", "choices": ["natural_mountains", "rolling_hills", "alpine_ridges", "woodland_valley", "archipelago", "desert_mesas", "canyon_badlands", "flat", "pillar_fields", "terraced_highlands", "cratered", "floating_islands", "volcanic_caldera"], "group": "terrain"},
    "style_strength": {"type": "number", "label": "family strength", "min": 0, "max": 1, "step": 0.01, "group": "terrain"},
    "base_height": {"type": "number", "label": "base height", "min": 32, "max": 224, "step": 1, "unit": "blocks", "group": "terrain"},
    "relief": {"type": "number", "label": "mountain relief", "min": 4, "max": 112, "step": 1, "unit": "blocks", "group": "terrain"},
    "mountain_scale": {"type": "number", "label": "feature scale", "min": 20, "max": 180, "step": 1, "unit": "blocks", "group": "terrain"},
    "roughness": {"type": "number", "label": "roughness", "min": 0, "max": 1, "step": 0.01, "group": "terrain"},
    "ridge_strength": {"type": "number", "label": "ridge strength", "min": 0, "max": 1, "step": 0.01, "group": "terrain"},
    "erosion": {"type": "number", "label": "erosion / smoothing", "min": 0, "max": 1, "step": 0.01, "group": "terrain"},
    "terrace_steps": {"type": "integer", "label": "terrace levels", "min": 2, "max": 20, "step": 1, "group": "terrain"},
    "pillar_density": {"type": "number", "label": "pillar density", "min": 0.05, "max": 1, "step": 0.01, "group": "terrain"},
    "pillar_width": {"type": "number", "label": "pillar width", "min": 3, "max": 24, "step": 0.5, "unit": "blocks", "group": "terrain"},
    "pillar_height": {"type": "number", "label": "pillar height", "min": 4, "max": 112, "step": 1, "unit": "blocks", "group": "terrain"},
    "canyon_strength": {"type": "number", "label": "canyon carving", "min": 0, "max": 1, "step": 0.01, "group": "terrain"},
    "crater_density": {"type": "number", "label": "crater density", "min": 0, "max": 1, "step": 0.01, "group": "terrain"},
    "crater_depth": {"type": "number", "label": "crater depth", "min": 2, "max": 96, "step": 1, "unit": "blocks", "group": "terrain"},
    "water_level": {"type": "number", "label": "water level", "min": 0, "max": 255, "step": 1, "unit": "Y", "group": "water"},
    "lake_coverage": {"type": "number", "label": "lake basins", "min": 0, "max": 1, "step": 0.01, "group": "water"},
    "river_strength": {"type": "number", "label": "river carving", "min": 0, "max": 1, "step": 0.01, "group": "water"},
    "moisture": {"type": "number", "label": "moisture", "min": 0, "max": 1, "step": 0.01, "group": "climate"},
    "temperature": {"type": "number", "label": "temperature", "min": 0, "max": 1, "step": 0.01, "group": "climate"},
    "snow_line": {"type": "number", "label": "snow line", "min": 48, "max": 255, "step": 1, "unit": "Y", "group": "climate"},
    "forest_density": {"type": "number", "label": "forest density", "min": 0, "max": 1, "step": 0.01, "group": "ecology"},
    "tree_spacing": {"type": "number", "label": "tree spacing", "min": 3, "max": 24, "step": 0.5, "unit": "blocks", "group": "ecology"},
    "cave_style": {"type": "choice", "label": "cave topology", "choices": ["none", "tunnels", "caverns", "labyrinth", "mixed", "procedural"], "group": "caves"},
    "cave_density": {"type": "number", "label": "cave density", "min": 0, "max": 1, "step": 0.01, "group": "caves"},
    "cave_scale": {"type": "number", "label": "network scale", "min": 8, "max": 72, "step": 1, "unit": "blocks", "group": "caves"},
    "cave_verticality": {"type": "number", "label": "verticality", "min": 0, "max": 1, "step": 0.01, "group": "caves"},
    "cave_entrances": {"type": "number", "label": "surface entrances", "min": 0, "max": 1, "step": 0.01, "group": "caves"},
    "structure_density": {"type": "number", "label": "structure density", "min": 0, "max": 1, "step": 0.01, "group": "structures"},
    "structure_spacing": {"type": "number", "label": "structure spacing", "min": 16, "max": 96, "step": 1, "unit": "blocks", "group": "structures"},
    "structure_jitter": {"type": "number", "label": "placement jitter", "min": 0, "max": 0.48, "step": 0.01, "group": "structures"},
    "structure_set": {"type": "choice", "label": "structure set", "choices": ["none", "mixed_ruins", "procedural_mix", "watchtowers", "fortified_camps", "villages", "arena_pillars", "shrines", "bridges"], "group": "structures"},
    "structure_layout": {"type": "choice", "label": "placement layout", "choices": ["scattered", "grid", "clusters", "ring", "route", "terrain_network", "landmarks"], "group": "structures"},
    "structure_material": {"type": "choice", "label": "build material", "choices": ["shale", "stone", "marble", "sandstone"], "group": "structures"},
    "objective": {"type": "choice", "label": "minigame intent", "choices": ["exploration", "traversal", "survival", "king_of_the_hill", "ruin_expedition"], "group": "gameplay"},
    "spawn_safety_radius": {"type": "number", "label": "spawn clearing", "min": 3, "max": 24, "step": 1, "unit": "blocks", "group": "gameplay"},
}


# These names are deliberate capabilities, not claims that the semantic
# preview executes the native graph.  The compiler pins each profile to one
# installed 0.5.7 biome entry and records any intent it cannot translate.
NATIVE_TEMPLATE_PROFILES: tuple[dict[str, Any], ...] = (
    {"id": "plains_mountains", "label": "Plains mountains + tunnels", "styles": ["natural_mountains", "rolling_hills", "woodland_valley"], "caves": ["tunnels", "mixed", "none"]},
    {"id": "taiga_mountains", "label": "Taiga alpine mountains", "styles": ["alpine_ridges"], "caves": ["tunnels", "mixed", "none"]},
    {"id": "plains_gorges", "label": "Plains gorges + labyrinth", "styles": ["canyon_badlands"], "caves": ["tunnels", "labyrinth", "mixed", "none"]},
    {"id": "deeproot_caverns", "label": "Deeproot caverns", "styles": ["natural_mountains", "woodland_valley"], "caves": ["caverns", "labyrinth", "mixed"]},
    {"id": "desert_stacks", "label": "Desert stacks / mesas", "styles": ["desert_mesas", "terraced_highlands"], "caves": ["none", "tunnels", "mixed"]},
    {"id": "default_flat", "label": "Default flat", "styles": ["flat"], "caves": ["none"]},
    {"id": "marble_pillars", "label": "Generative marble pillars", "styles": ["pillar_fields"], "caves": ["none", "mixed"]},
    {"id": "twist_crater", "label": "Twist crater", "styles": ["cratered"], "caves": ["none", "caverns", "mixed"]},
    {"id": "rooted_islands", "label": "Experimental rooted islands", "styles": ["archipelago", "floating_islands"], "caves": ["none", "mixed"]},
    {"id": "volcanic_caldera", "label": "Volcanic caldera", "styles": ["volcanic_caldera"], "caves": ["tunnels", "caverns", "mixed", "none"]},
)

PROCEDURAL_GRAPH_REFERENCE_VALIDATION: dict[str, Any] = {
    "status": "passed_hytale_0.5.7_reference_pack",
    "schema": "hytalerl_worldgen_v2_procedural_native_validation_v1",
    "path": (
        "experimental/worldgen-v2/evidence/"
        "procedural-native-graph-0.5.7-5b80344f-three-seeds-r1/"
        "validation.json"
    ),
    "semantic_sha256": (
        "3f0554e77cc226f803e4f709a1a7ee7c1f6de65d2e5aeb21460ef5f3167295d0"
    ),
    "pack_semantic_sha256": (
        "687e036fc91559abda798aa7f3d6e746e7678700c49ea794f07d487365d5b4ca"
    ),
    "seed_count": 3,
    "minimum_pairwise_surface_difference": 0.9519961977186312,
    "pack_specific_validation_required": True,
}


def options() -> dict[str, Any]:
    """Metadata used to build the panel without duplicating field contracts."""

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "defaults": dict(DEFAULTS),
        "fields": FIELDS,
        "presets": [
            {**preset, "values": {**DEFAULTS, **preset["values"]}}
            for preset in PRESETS
        ],
        "preview_resolution": PREVIEW_RESOLUTION,
        "batch": {
            "maximum_seeds": MAX_BATCH_SEEDS,
            "resolutions": list(FAST_BATCH_RESOLUTIONS),
            "strategies": [
                {"id": "native_seeds", "label": "same recipe, many seeds", "packs_per_seed": False},
                {"id": "parametric_worlds", "label": "hyperparameter worlds (no presets)", "packs_per_seed": False},
                {"id": "plausible_variants", "label": "bounded recipe variations", "packs_per_seed": True},
            ],
        },
        "native_template_profiles": list(NATIVE_TEMPLATE_PROFILES),
        "preview_traversable_slope": TRAVERSABLE_SLOPE,
        "biomes": list(BIOMES),
        "pipeline": [
            {"state": "now", "label": "Design preview", "exact": False},
            {"state": "next", "label": "WorldGen V2 asset authoring", "exact": False},
            {"state": "native", "label": "Native seeded generation", "exact": True},
            {"state": "jax", "label": "Region capture + JAX import", "exact": True},
        ],
        "boundary": (
            "The 3D panel is a deterministic design preview. It does not claim "
            "block-exact Hytale output; exactness begins after the design is "
            "compiled into WorldGen V2 assets, generated natively, and captured."
        ),
        "evidence": {
            "custom_pack": "mountain-forest-lakes-v1",
            "audited_native_seeds": 20,
            "unique_region_semantics": 20,
            "minimum_pairwise_surface_disagreement": 0.934353,
            "playable_seeds": 16,
            "published_jax_envs": 64,
            "procedural_graph_reference": dict(
                PROCEDURAL_GRAPH_REFERENCE_VALIDATION
            ),
        },
    }


def _number(name: str, value: Any, *, integer: bool = False) -> float | int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if integer and not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    parsed = int(numeric) if integer else numeric
    field = FIELDS[name]
    if parsed < field["min"] or parsed > field["max"]:
        raise ValueError(
            f"{name} must be between {field['min']} and {field['max']}"
        )
    return parsed


def normalize(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and canonicalize a browser or API design payload."""

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("worldgen design must be a JSON object")
    payload = dict(payload)
    if "world_length" in payload:
        if "world_depth" in payload:
            raise ValueError("world_length cannot be combined with world_depth")
        payload["world_depth"] = payload.pop("world_length")
    if "world_size" in payload:
        if "world_width" in payload or "world_depth" in payload:
            raise ValueError(
                "world_size is a legacy square alias and cannot be combined "
                "with world_width or world_depth"
            )
        payload["world_width"] = payload["world_size"]
        payload["world_depth"] = payload.pop("world_size")
    unknown = sorted(set(payload) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown worldgen field(s): {', '.join(unknown)}")

    config = {**DEFAULTS, **payload}
    raw_name = str(config["name"]).strip().lower().replace(" ", "-")
    name = re.sub(r"[^a-z0-9_-]+", "-", raw_name).strip("-_")
    if not name:
        raise ValueError("name must contain at least one letter or digit")
    if len(name) > 64:
        raise ValueError("name must be at most 64 characters")
    config["name"] = name

    for name_, field in FIELDS.items():
        kind = field["type"]
        if kind == "number":
            config[name_] = _number(name_, config[name_])
        elif kind == "integer":
            config[name_] = _number(name_, config[name_], integer=True)
        elif kind == "choice":
            choices = field["choices"]
            value = config[name_]
            if choices and isinstance(choices[0], int):
                value = _number(name_, value, integer=True)
            else:
                value = str(value)
            if value not in choices:
                raise ValueError(f"{name_} must be one of {choices}")
            config[name_] = value

    if config["cave_style"] == "none":
        # A disabled topology owns its density. Canonicalizing it here keeps
        # visually identical no-cave designs content-identical.
        config["cave_density"] = 0.0
    if config["structure_set"] == "none":
        config["structure_density"] = 0.0
    if config["generation_mode"] == "procedural_genome" and not any(
        config[name] > 0
        for name in (
            "terrain_mountain_weight", "terrain_hill_weight",
            "terrain_valley_weight", "terrain_plateau_weight",
            "terrain_canyon_weight", "terrain_pillar_weight",
            "terrain_crater_weight", "terrain_flat_weight",
        )
    ):
        raise ValueError(
            "procedural terrain requires at least one non-zero component weight"
        )
    return config


def _mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF


def _hash01(seed: int, x: int, z: int, salt: int = 0) -> float:
    value = (
        seed * 0x9E3779B1
        + x * 0x85EBCA77
        + z * 0xC2B2AE3D
        + salt * 0x27D4EB2F
    )
    return _mix32(value) / 4_294_967_295.0


@lru_cache(maxsize=1)
def _procedural_module() -> Any:
    """Load the experimental generator without making its hyphenated parent a package."""

    import sys

    package_root = PROJECT_ROOT / "experimental" / "worldgen-v2"
    if not package_root.is_dir():
        raise FileNotFoundError(f"procedural WorldGen source is absent: {package_root}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    import procedural_generation

    return procedural_generation


@lru_cache(maxsize=512)
def _cached_world_genome(canonical_config: str) -> dict[str, Any]:
    return _procedural_module().build_world_genome(json.loads(canonical_config))


def _world_genome(config: dict[str, Any]) -> dict[str, Any] | None:
    if config["generation_mode"] != "procedural_genome":
        return None
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    # Callers treat this cached mapping as immutable. Public responses copy it
    # through the preview serialization boundary before leaving this module.
    return _cached_world_genome(canonical)


def _smooth(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _lerp(a: float, b: float, amount: float) -> float:
    return a + (b - a) * amount


def _noise(seed: int, x: float, z: float, scale: float, salt: int) -> float:
    """Bilinear value noise in [-1, 1], stable across Python versions."""

    px, pz = x / scale, z / scale
    x0, z0 = math.floor(px), math.floor(pz)
    tx, tz = _smooth(px - x0), _smooth(pz - z0)
    a = _hash01(seed, x0, z0, salt) * 2.0 - 1.0
    b = _hash01(seed, x0 + 1, z0, salt) * 2.0 - 1.0
    c = _hash01(seed, x0, z0 + 1, salt) * 2.0 - 1.0
    d = _hash01(seed, x0 + 1, z0 + 1, salt) * 2.0 - 1.0
    return _lerp(_lerp(a, b, tx), _lerp(c, d, tx), tz)


def _fbm(seed: int, x: float, z: float, scale: float, roughness: float,
         salt: int) -> float:
    persistence = 0.30 + 0.34 * roughness
    amplitude = total = value = 0.0
    amplitude = 1.0
    for octave in range(5):
        value += amplitude * _noise(seed, x, z, max(2.5, scale), salt + octave)
        total += amplitude
        amplitude *= persistence
        scale *= 0.5
    return value / total


def _nearest_feature(seed: int, x: float, z: float, spacing: float,
                     jitter: float, salt: int) -> tuple[float, float, float]:
    """Distance and center of the nearest coordinate-keyed feature cell."""

    gx, gz = math.floor(x / spacing), math.floor(z / spacing)
    best = (float("inf"), 0.0, 0.0)
    for cz in range(gz - 1, gz + 2):
        for cx in range(gx - 1, gx + 2):
            px = (cx + 0.5) * spacing + (
                _hash01(seed, cx, cz, salt) * 2.0 - 1.0
            ) * spacing * jitter
            pz = (cz + 0.5) * spacing + (
                _hash01(seed, cx, cz, salt + 1) * 2.0 - 1.0
            ) * spacing * jitter
            candidate = (math.hypot(x - px, z - pz), px, pz)
            if candidate < best:
                best = candidate
    return best


def _procedural_terrain(
    config: dict[str, Any],
    x: float,
    z: float,
    land: float,
    broad: float,
    ridge: float,
    weights: tuple[float, ...],
) -> float:
    """Blend terrain primitives from genome weights instead of choosing a preset."""

    seed = config["seed"]
    width, depth = config["world_width"], config["world_depth"]
    size = min(width, depth)
    half_x, half_z = max(1.0, width * 0.5), max(1.0, depth * 0.5)
    scale = config["mountain_scale"]
    phase = _hash01(seed, 0, 0, 801) * math.tau

    mountains = land * 0.62 + ridge * (0.48 + config["ridge_strength"] * 0.38)
    hills = broad * 0.68 + _noise(seed, x, z, scale * 0.62, 803) * 0.14
    valley_center = (
        math.sin(x / max(8.0, scale) * 1.37 + phase) * half_z * 0.2
        + _noise(seed, x, 0.0, scale * 1.7, 804) * half_z * 0.08
    )
    valley_distance = min(1.0, abs(z - valley_center) / (half_z * 0.72))
    valleys = -0.2 + valley_distance * valley_distance * 0.78 + broad * 0.15
    steps = config["terrace_steps"]
    plateaus = round((land * 0.72 + broad * 0.28) * steps) / steps

    canyon_width = 2.5 + (1.0 - config["canyon_strength"]) * 12.0
    canyon_a = math.sin(x / max(10.0, scale) * 2.05 + phase) * half_z * 0.3
    canyon_b = math.cos(z / max(10.0, scale) * 1.72 - phase) * half_x * 0.27
    channel_a = _smooth(max(0.0, 1.0 - abs(z - canyon_a) / canyon_width))
    channel_b = _smooth(max(
        0.0, 1.0 - abs(x - canyon_b) / max(1.0, canyon_width * 0.82)
    ))
    canyons = 0.27 + broad * 0.34 - max(channel_a, channel_b * 0.78) * (
        0.34 + config["canyon_strength"] * 0.82
    )

    pillar_spacing = max(
        config["pillar_width"] * 2.15,
        34.0 - 18.0 * config["pillar_density"],
    )
    pillar_distance, _, _ = _nearest_feature(
        seed, x, z, pillar_spacing, 0.24, 807
    )
    pillar_radius = config["pillar_width"] * (
        0.72 + config["pillar_density"] * 0.42
    )
    pillar_core = _smooth(max(
        0.0, 1.0 - pillar_distance / max(1.0, pillar_radius)
    ))
    pillars = -0.16 + (
        pillar_core ** 0.28 if pillar_core > 0 else 0.0
    ) * (config["pillar_height"] / max(4.0, config["relief"]))
    pillars += broad * 0.035

    craters = land * 0.38 + broad * 0.24
    for index in range(1 + round(config["crater_density"] * 5)):
        cx = (_hash01(seed, index, 0, 809) * 2.0 - 1.0) * half_x * 0.74
        cz = (_hash01(seed, index, 1, 809) * 2.0 - 1.0) * half_z * 0.74
        radius = size * (0.08 + 0.1 * _hash01(seed, index, 2, 809))
        distance = math.hypot(x - cx, z - cz) / max(1.0, radius)
        bowl = _smooth(max(0.0, 1.0 - distance))
        rim = _smooth(max(0.0, 1.0 - abs(distance - 1.0) / 0.24))
        depth_amount = config["crater_depth"] / max(4.0, config["relief"])
        craters += rim * depth_amount * 0.25 - bowl * depth_amount
    flats = _noise(seed, x, z, max(18.0, scale), 811) * 0.018

    targets = (
        mountains, hills, valleys, plateaus, canyons, pillars, craters, flats,
    )
    mixed = sum(weight * target for weight, target in zip(weights, targets))
    return _lerp(land, mixed, config["style_strength"])


def _terrain_family(config: dict[str, Any], x: float, z: float,
                    land: float, broad: float, ridge: float,
                    *, genome: dict[str, Any] | None = None,
                    procedural_blend: tuple[Any, ...] | None = None) -> float:
    """Apply one deliberate high-level terrain family to the shared field.

    These families make the preview useful for authoring.  Native compilation
    maps each family to a pinned Hytale template; it does not claim this Python
    expression is the template's density graph.
    """

    if genome is not None:
        if procedural_blend is None:
            procedural_blend = _procedural_module().blend_at(genome, x, z)
        return _procedural_terrain(
            config, x, z, land, broad, ridge, procedural_blend[0]
        )

    style = config["terrain_style"]
    amount = config["style_strength"]
    if amount <= 0 or style == "natural_mountains":
        return land
    seed = config["seed"]
    width, depth = config["world_width"], config["world_depth"]
    size = min(width, depth)
    half_x, half_z = max(1.0, width * 0.5), max(1.0, depth * 0.5)
    scale = config["mountain_scale"]

    if style == "rolling_hills":
        target = broad * 0.66 + _noise(seed, x, z, scale * 0.62, 401) * 0.12
    elif style == "alpine_ridges":
        alpine = 1.0 - abs(_fbm(seed, x, z, scale * 0.72, 0.38, 405))
        target = land * 0.48 + (alpine * alpine * 1.65 - 0.62) * 0.72
    elif style == "woodland_valley":
        phase = _hash01(seed, 0, 0, 409) * math.tau
        center = math.sin(x / max(8.0, scale) * 1.4 + phase) * half_z * 0.18
        valley = min(1.0, abs(z - center) / (half_z * 0.75))
        target = -0.18 + valley * valley * 0.74 + broad * 0.16
    elif style == "archipelago":
        islands = _fbm(seed, x, z, scale * 0.82, 0.46, 413)
        mask = _smooth(max(0.0, min(1.0, (islands + 0.38) / 0.72)))
        target = -0.55 + mask * (0.72 + max(-0.12, land) * 0.56)
    elif style == "desert_mesas":
        mesa = max(-0.34, min(0.82, land * 1.08 + broad * 0.22))
        target = round(mesa * 5.0) / 5.0
    elif style in ("canyon_badlands", "terraced_highlands"):
        phase = _hash01(seed, 0, 0, 417) * math.tau
        center_a = math.sin(x / max(10.0, scale) * 2.1 + phase) * half_z * 0.28
        center_b = math.cos(z / max(10.0, scale) * 1.7 - phase) * half_x * 0.25
        width = 2.5 + (1.0 - config["canyon_strength"]) * 12.0
        channel_a = _smooth(max(0.0, 1.0 - abs(z - center_a) / width))
        channel_b = _smooth(max(0.0, 1.0 - abs(x - center_b) / (width * 0.8)))
        carve = max(channel_a, channel_b * 0.78) * config["canyon_strength"]
        target = 0.28 + broad * 0.32 - carve * 1.12
        if style == "terraced_highlands":
            steps = config["terrace_steps"]
            target = round(target * steps) / steps
    elif style == "flat":
        target = _noise(seed, x, z, max(18.0, scale), 421) * 0.018
    elif style == "pillar_fields":
        spacing = max(config["pillar_width"] * 2.15, 34.0 - 18.0 * config["pillar_density"])
        distance, _, _ = _nearest_feature(seed, x, z, spacing, 0.24, 425)
        radius = config["pillar_width"] * (0.72 + config["pillar_density"] * 0.42)
        core = _smooth(max(0.0, 1.0 - distance / max(1.0, radius)))
        # A low exponent gives broad, readable tops and sharp shoulders.
        column = core ** 0.28 if core > 0 else 0.0
        target = -0.16 + column * (config["pillar_height"] / max(4.0, config["relief"]))
        target += broad * 0.035
    elif style == "cratered":
        target = land * 0.36 + broad * 0.26
        count = 1 + round(config["crater_density"] * 5)
        for index in range(count):
            cx = (_hash01(seed, index, 0, 429) * 2.0 - 1.0) * half_x * 0.72
            cz = (_hash01(seed, index, 1, 429) * 2.0 - 1.0) * half_z * 0.72
            radius = size * (0.08 + 0.1 * _hash01(seed, index, 2, 429))
            distance = math.hypot(x - cx, z - cz) / max(1.0, radius)
            bowl = _smooth(max(0.0, 1.0 - distance))
            rim = _smooth(max(0.0, 1.0 - abs(distance - 1.0) / 0.24))
            depth = config["crater_depth"] / max(4.0, config["relief"])
            target += rim * depth * 0.25 - bowl * depth
    elif style == "floating_islands":
        spacing = max(22.0, scale * 0.7)
        distance, _, _ = _nearest_feature(seed, x, z, spacing, 0.32, 433)
        radius = spacing * (0.3 + 0.14 * _noise(seed, x, z, scale, 435))
        island = _smooth(max(0.0, 1.0 - distance / max(3.0, radius)))
        target = -0.92 + island * (1.25 + max(0.0, ridge) * 0.42)
    elif style == "volcanic_caldera":
        ox = (_hash01(seed, 0, 0, 437) * 2.0 - 1.0) * half_x * 0.18
        oz = (_hash01(seed, 0, 1, 437) * 2.0 - 1.0) * half_z * 0.18
        radial = math.hypot((x - ox) / half_x, (z - oz) / half_z)
        cone = max(0.0, 1.0 - radial) ** 1.45
        bowl = _smooth(max(0.0, 1.0 - radial / 0.23))
        target = broad * 0.2 + cone * 1.28 - bowl * 0.92
    else:  # normalize() makes this unreachable; retain a fail-closed guard.
        raise ValueError(f"unsupported terrain_style {style!r}")
    return _lerp(land, target, amount)


def _terrain_height(config: dict[str, Any], x: float, z: float,
                    genome: dict[str, Any] | None = None) -> float:
    seed = config["seed"]
    scale = config["mountain_scale"]
    procedural_blend = None
    relief_multiplier = 1.0
    if genome is None:
        genome = _world_genome(config)
    if genome is not None:
        procedural_blend = _procedural_module().blend_at(genome, x, z)
        relief_multiplier = float(procedural_blend[1])
        scale *= float(procedural_blend[2])
        roughness = max(0.0, min(
            1.0, config["roughness"] * float(procedural_blend[3])
        ))
    else:
        roughness = config["roughness"]
    # A low-frequency domain warp makes ranges bend instead of lining up with
    # the preview grid.  Every subsequent feature samples the warped terrain.
    warp = _noise(seed, x, z, scale * 1.65, 90) * scale * 0.16
    base = _fbm(seed, x + warp, z - warp, scale, roughness, 10)
    broad = _noise(seed, x - warp, z + warp, scale * 2.35, 31)
    ridge_noise = _fbm(seed, x - warp, z - warp, scale * 0.78, 0.48, 51)
    ridge = 1.0 - abs(ridge_noise)
    ridge = ridge * ridge * 1.45 - 0.48

    detailed = 0.62 * base + 0.38 * broad
    eroded = 0.35 * base + 0.65 * broad
    land = _lerp(detailed, eroded, config["erosion"])
    land += config["ridge_strength"] * ridge * 0.48
    land = _terrain_family(
        config, x, z, land, broad, ridge,
        genome=genome, procedural_blend=procedural_blend,
    )
    if genome is not None and config["traversal_bias"] > 0:
        route_amount = _procedural_module().route_influence(genome, x, z)
        route_target = broad * 0.18 + _noise(
            seed, x, z, max(24.0, scale * 1.55), 813
        ) * 0.035
        land = _lerp(
            land,
            route_target,
            route_amount * config["traversal_bias"] * 0.76,
        )

    # Hydrology starts by shaping broad valleys in the land field.  The narrow
    # wet channel is applied after the bounded lowland datum is anchored, but
    # doing the large-scale work here prevents a river or lake from slicing a
    # near-vertical trench through an unrelated mountain face.
    coverage = config["lake_coverage"]
    width, depth = config["world_width"], config["world_depth"]
    half_x, half_z = width * 0.5, depth * 0.5
    size = min(width, depth)
    if coverage > 0:
        for index in range(1 + round(coverage * 4)):
            cx = (_hash01(seed, index, 0, 72) * 2.0 - 1.0) * half_x * 0.72
            cz = (_hash01(seed, index, 1, 72) * 2.0 - 1.0) * half_z * 0.72
            radius = size * (0.09 + 0.12 * coverage) * (
                0.78 + 0.44 * _hash01(seed, index, 2, 72)
            )
            aspect = 0.65 + 0.7 * _hash01(seed, index, 3, 72)
            inside = max(0.0, 1.0 - math.hypot(
                (x - cx) / max(1.0, radius),
                (z - cz) / max(1.0, radius * aspect),
            ))
            land -= _smooth(inside) * (0.08 + 0.14 * coverage)

    river = config["river_strength"]
    if river > 0:
        phase = _hash01(seed, 0, 0, 113) * math.tau
        center = (
            math.sin((x / max(half_x, 1.0)) * 2.15 + phase) * half_z * 0.18
            + _noise(seed, x, 0.0, scale * 1.2, 114) * half_z * 0.09
        )
        valley_width = 5.0 + river * (7.0 + size * 0.035)
        valley = _smooth(max(0.0, 1.0 - abs(z - center) / valley_width))
        land -= valley * (0.06 + 0.09 * river)
    return config["base_height"] + config["relief"] * relief_multiplier * land


def _hydrology_height(config: dict[str, Any], x: float, z: float,
                      land_height: float) -> float:
    """Carve bounded, contiguous lakes and one continuous river.

    Broad value-noise thresholds made the *amount* of water depend on whether
    one finite preview happened to land in the low half of the field.  A lake
    control of 0.34 could therefore mean 0% water on one seed and 90% on the
    next.  Seeded elliptical basins retain organic placement while bounding
    their area, which is the behavior an environment-authoring knob promises.
    """

    seed = config["seed"]
    width, depth = config["world_width"], config["world_depth"]
    size = min(width, depth)
    half_x, half_z = width * 0.5, depth * 0.5
    height = land_height
    coverage = config["lake_coverage"]
    if coverage > 0:
        lake_count = 1 + round(coverage * 4)
        for index in range(lake_count):
            cx = (_hash01(seed, index, 0, 72) * 2.0 - 1.0) * half_x * 0.72
            cz = (_hash01(seed, index, 1, 72) * 2.0 - 1.0) * half_z * 0.72
            radius = size * (0.045 + 0.085 * coverage) * (
                0.78 + 0.44 * _hash01(seed, index, 2, 72)
            )
            aspect = 0.65 + 0.7 * _hash01(seed, index, 3, 72)
            dx = (x - cx) / max(1.0, radius)
            dz = (z - cz) / max(1.0, radius * aspect)
            inside = max(0.0, 1.0 - math.hypot(dx, dz))
            if inside <= 0:
                continue
            bowl = _smooth(inside)
            target = config["water_level"] - (1.4 + 4.2 * coverage * bowl)
            target = max(target, land_height - (4.0 + 8.0 * coverage))
            height = min(height, _lerp(height, target, bowl))

    # One seed-dependent river crosses the complete bounded Region.  Strength
    # changes channel width/depth; it does not turn the river into independent
    # wet cells.
    river = config["river_strength"]
    if river > 0:
        scale = config["mountain_scale"]
        phase = _hash01(seed, 0, 0, 113) * math.tau
        center = (
            math.sin((x / max(half_x, 1.0)) * 2.15 + phase) * half_z * 0.18
            + _noise(seed, x, 0.0, scale * 1.2, 114) * half_z * 0.09
        )
        width = 1.1 + river * (2.2 + size * 0.024)
        channel = _smooth(max(0.0, 1.0 - abs(z - center) / width))
        target = config["water_level"] - (1.0 + river * 3.4)
        target = max(target, land_height - (3.0 + river * 8.0))
        height = min(height, _lerp(height, target, channel))
    return max(1.0, min(254.0, height))


def _grid(config: dict[str, Any], resolution: int = PREVIEW_RESOLUTION,
          genome: dict[str, Any] | None = None) -> tuple[
    list[list[float]], float, float
]:
    step_x = config["world_width"] / (resolution - 1)
    step_z = config["world_depth"] / (resolution - 1)
    half_x = config["world_width"] * 0.5
    half_z = config["world_depth"] * 0.5
    land = []
    for iz in range(resolution):
        z = -half_z + iz * step_z
        land.append([
            _terrain_height(config, -half_x + ix * step_x, z, genome)
            for ix in range(resolution)
        ])
    # ``base_height`` is the lowland datum, matching the native authoring spec,
    # not the mean of mountain tops and valleys.  Pin the bounded field's 15th
    # percentile to it.  Low-frequency seed offsets then change range shape
    # without randomly sinking an otherwise identical design below sea level.
    ordered = sorted(value for row in land for value in row)
    lowland = ordered[round((len(ordered) - 1) * 0.15)]
    offset = config["base_height"] - lowland
    heights = []
    for iz, row in enumerate(land):
        z = -half_z + iz * step_z
        heights.append([
            round(_hydrology_height(
                config,
                -half_x + ix * step_x,
                z,
                max(1.0, min(254.0, value + offset)),
            ), 3)
            for ix, value in enumerate(row)
        ])
    return heights, step_x, step_z


def _slopes(heights: list[list[float]], step_x: float,
            step_z: float) -> list[list[float]]:
    resolution = len(heights)
    result: list[list[float]] = []
    for z in range(resolution):
        row = []
        for x in range(resolution):
            left, right = heights[z][max(0, x - 1)], heights[z][min(resolution - 1, x + 1)]
            down, up = heights[max(0, z - 1)][x], heights[min(resolution - 1, z + 1)][x]
            dx = (right - left) / (step_x * (1 if x in (0, resolution - 1) else 2))
            dz = (up - down) / (step_z * (1 if z in (0, resolution - 1) else 2))
            row.append(round(math.hypot(dx, dz), 4))
        result.append(row)
    return result


def _biome(config: dict[str, Any], x: float, z: float, height: float,
           slope: float) -> int:
    water = config["water_level"]
    if height < water - 3.5:
        return 0
    if height < water - 0.15:
        return 1
    if height < water + 1.8:
        return 2
    if height >= config["snow_line"]:
        return 7
    if height >= config["snow_line"] - 9.0 or slope > 1.18:
        return 6
    if slope > 0.72:
        return 5
    local_moisture = config["moisture"] + 0.25 * _noise(
        config["seed"], x, z, max(18.0, config["mountain_scale"] * 0.65), 131
    )
    if config["temperature"] > 0.67 and local_moisture < 0.36:
        return 8
    forest_threshold = 0.76 - config["forest_density"] * 0.58
    if local_moisture > forest_threshold:
        return 4
    return 3


def _sample(values: list[list[float]], config: dict[str, Any], x: float,
            z: float) -> float:
    width, depth = config["world_width"], config["world_depth"]
    last = len(values) - 1
    gx = max(0.0, min(float(last), (x / width + 0.5) * last))
    gz = max(0.0, min(float(last), (z / depth + 0.5) * last))
    x0, z0 = int(gx), int(gz)
    x1, z1 = min(last, x0 + 1), min(last, z0 + 1)
    tx, tz = gx - x0, gz - z0
    return _lerp(
        _lerp(values[z0][x0], values[z0][x1], tx),
        _lerp(values[z1][x0], values[z1][x1], tx),
        tz,
    )


def _caves(config: dict[str, Any], heights: list[list[float]]) -> dict[str, Any]:
    """Create a bounded semantic cave graph for cutaway design inspection.

    It supplies topology, entrances and stable anchors to downstream tooling;
    it is not presented as a voxel-exact execution of Hytale's density AST.
    """

    style = config["cave_style"]
    density = config["cave_density"]
    if style == "none" or density <= 0:
        return {"topology": "none", "segments": [], "chambers": [], "entrances": []}

    seed = config["seed"]
    width, depth = config["world_width"], config["world_depth"]
    size = min(width, depth)
    half_x, half_z = width * 0.5, depth * 0.5
    scale = config["cave_scale"]
    network_count = max(1, min(24, round((size / scale) * (0.8 + density * 2.6))))
    base_segments_by_style = {
        "tunnels": 4,
        "caverns": 3,
        "labyrinth": 7,
        "mixed": 6,
    }
    segments: list[dict[str, Any]] = []
    chambers: list[dict[str, Any]] = []
    entrances: list[dict[str, Any]] = []
    margin = min(max(4.0, scale * 0.25), half_x - 1.0, half_z - 1.0)

    for network in range(network_count):
        network_style = style
        if style == "procedural":
            choice = _hash01(seed, network, 0, 499)
            network_style = (
                "tunnels" if choice < 0.34 else
                "caverns" if choice < 0.58 else
                "labyrinth" if choice < 0.79 else
                "mixed"
            )
        x = (_hash01(seed, network, 0, 501) * 2.0 - 1.0) * (half_x - margin)
        z = (_hash01(seed, network, 1, 501) * 2.0 - 1.0) * (half_z - margin)
        surface = _sample(heights, config, x, z)
        depth = 7.0 + scale * (0.22 + 0.36 * _hash01(seed, network, 2, 501))
        y = max(4.0, surface - depth)
        angle = _hash01(seed, network, 3, 501) * math.tau
        radius_base = 1.4 + density * 1.8
        if network_style in ("caverns", "mixed"):
            radius_base += 0.9

        if _hash01(seed, network, 4, 501) < config["cave_entrances"]:
            entrance = {
                "id": f"cave-{network:02d}-entrance",
                "network": network,
                "x": round(x, 3), "y": round(surface + 0.15, 3),
                "z": round(z, 3), "radius": round(radius_base * 0.8, 2),
            }
            entrances.append(entrance)
            segments.append({
                "id": f"cave-{network:02d}-entry",
                "network": network, "kind": "entrance",
                "start": [entrance["x"], entrance["y"], entrance["z"]],
                "end": [round(x, 3), round(y, 3), round(z, 3)],
                "radius": round(radius_base * 0.72, 2),
            })

        segment_count = base_segments_by_style[network_style] + round(
            density * (4 if network_style == "labyrinth" else 2)
        )
        for index in range(segment_count):
            turn = (_hash01(seed, network, index, 511) * 2.0 - 1.0)
            turn_scale = 1.35 if network_style == "labyrinth" else 0.72
            angle += turn * turn_scale
            length = scale * (0.35 + 0.38 * _hash01(seed, network, index, 512))
            nx = max(-half_x + margin, min(half_x - margin, x + math.cos(angle) * length))
            nz = max(-half_z + margin, min(half_z - margin, z + math.sin(angle) * length))
            surface_end = _sample(heights, config, nx, nz)
            vertical = (
                _hash01(seed, network, index, 513) * 2.0 - 1.0
            ) * scale * 0.24 * config["cave_verticality"]
            ny = max(4.0, min(surface_end - 3.0, y + vertical))
            radius = radius_base * (
                0.72 + 0.62 * _hash01(seed, network, index, 514)
            )
            segments.append({
                "id": f"cave-{network:02d}-segment-{index:02d}",
                "network": network,
                "kind": "tunnel" if network_style != "labyrinth" else "passage",
                "topology": network_style,
                "start": [round(x, 3), round(y, 3), round(z, 3)],
                "end": [round(nx, 3), round(ny, 3), round(nz, 3)],
                "radius": round(radius, 2),
            })
            x, y, z = nx, ny, nz

            chamber_chance = 0.58 if network_style == "caverns" else (
                0.3 if network_style == "mixed" else 0.08
            )
            if _hash01(seed, network, index, 515) < chamber_chance:
                chambers.append({
                    "id": f"cave-{network:02d}-chamber-{index:02d}",
                    "network": network,
                    "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
                    "radius": round(radius * (1.8 + _hash01(seed, network, index, 516)), 2),
                })
    return {
        "topology": style,
        "segments": segments,
        "chambers": chambers,
        "entrances": entrances,
    }


def _structure_kind(structure_set: str, choice: float,
                    variety: float = 1.0) -> str:
    if structure_set == "mixed_ruins":
        return ("ruin", "arch", "shrine")[min(2, int(choice * 3))]
    if structure_set == "procedural_mix":
        kinds = (
            "ruin", "arch", "shrine", "watchtower", "fortified_camp",
            "village_house", "stone_pillar", "bridge",
        )
        available = max(1, min(len(kinds), 1 + round(variety * (len(kinds) - 1))))
        return kinds[min(available - 1, int(choice * available))]
    return {
        "watchtowers": "watchtower",
        "fortified_camps": "fortified_camp",
        "villages": "village_house",
        "arena_pillars": "stone_pillar",
        "shrines": "shrine",
        "bridges": "bridge",
    }[structure_set]


def _structures(config: dict[str, Any], heights: list[list[float]],
                slopes: list[list[float]]) -> list[dict[str, Any]]:
    if config["structure_set"] == "none" or config["structure_density"] <= 0:
        return []
    spacing = config["structure_spacing"]
    half_x = config["world_width"] * 0.5
    half_z = config["world_depth"] * 0.5
    count_x = max(1, math.ceil(config["world_width"] / spacing))
    count_z = max(1, math.ceil(config["world_depth"] / spacing))
    layout = config["structure_layout"]
    candidates: list[tuple[int, int, float, float]] = []
    candidate_routes: dict[tuple[int, int], str] = {}

    if layout == "terrain_network":
        genome = _world_genome(config)
        if genome is None:
            # Authored-family designs have no macro graph; retain the older
            # single-route behavior rather than inventing hidden regions.
            layout = "route"
        else:
            edges = genome["route_graph"]["edges"]
            if not edges:
                for index, node in enumerate(genome["route_graph"]["nodes"]):
                    candidates.append((index, 0, float(node["x"]), float(node["z"])))
            for edge_index, edge in enumerate(edges):
                length = max(1.0, float(edge["length"]))
                samples = max(2, math.ceil(length / spacing))
                start, end = edge["start"], edge["end"]
                dx, dz = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
                inverse = 1.0 / max(1e-9, math.hypot(dx, dz))
                normal_x, normal_z = -dz * inverse, dx * inverse
                for sample_index in range(samples + 1):
                    if edge_index and sample_index == 0:
                        continue
                    amount = sample_index / samples
                    key = (edge_index * 1000 + sample_index, edge_index)
                    lateral = (
                        _hash01(config["seed"], key[0], key[1], 211) * 2.0 - 1.0
                    ) * spacing * config["structure_jitter"] * 0.48
                    x = _lerp(float(start[0]), float(end[0]), amount) + normal_x * lateral
                    z = _lerp(float(start[1]), float(end[1]), amount) + normal_z * lateral
                    candidates.append((key[0], key[1], x, z))
                    candidate_routes[key] = str(edge["id"])

    if layout == "ring":
        count = max(6, round(math.tau * min(half_x, half_z) * 0.62 / spacing))
        for index in range(count):
            angle = math.tau * index / count
            radial = 1.0 + (
                _hash01(config["seed"], index, 0, 207) * 2.0 - 1.0
            ) * config["structure_jitter"]
            candidates.append((index, 0, math.cos(angle) * half_x * 0.62 * radial,
                               math.sin(angle) * half_z * 0.62 * radial))
    elif layout == "route":
        count = max(3, round(config["world_width"] / spacing) + 1)
        phase = _hash01(config["seed"], 0, 0, 208) * math.tau
        for index in range(count):
            x = -half_x * 0.78 + (index / max(1, count - 1)) * half_x * 1.56
            z = math.sin(index * 0.92 + phase) * half_z * 0.34
            candidates.append((index, 0, x, z))
    elif layout != "terrain_network":
        for iz in range(-count_z, count_z + 1):
            for ix in range(-count_x, count_x + 1):
                jitter_scale = 0.12 if layout == "grid" else 1.0
                jitter = spacing * config["structure_jitter"] * jitter_scale
                x = ix * spacing + (
                    _hash01(config["seed"], ix, iz, 202) * 2 - 1
                ) * jitter
                z = iz * spacing + (
                    _hash01(config["seed"], ix, iz, 203) * 2 - 1
                ) * jitter
                if layout == "clusters":
                    cluster = _mix32(ix * 92821 + iz * 68917 + config["seed"]) % 4
                    cx = (_hash01(config["seed"], cluster, 0, 209) * 2 - 1) * half_x * .62
                    cz = (_hash01(config["seed"], cluster, 1, 209) * 2 - 1) * half_z * .62
                    x, z = _lerp(x, cx, .56), _lerp(z, cz, .56)
                candidates.append((ix, iz, x, z))

    structures = []
    for ix, iz, x, z in candidates:
        chance = _hash01(config["seed"], ix, iz, 201)
        if chance > config["structure_density"]:
            continue
        if not (-half_x + 5 <= x <= half_x - 5 and -half_z + 5 <= z <= half_z - 5):
            continue
        y = _sample(heights, config, x, z)
        slope = _sample(slopes, config, x, z)
        if y <= config["water_level"] + 1.5 or slope > 1.05:
            continue
        kind = _structure_kind(
            config["structure_set"],
            _hash01(config["seed"], ix, iz, 204),
            config["structure_variety"],
        )
        base_footprint = 3.0 if kind == "stone_pillar" else (
            7.5 if kind in ("fortified_camp", "bridge") else 5.0
        )
        structures.append({
            "id": f"structure-{len(structures):03d}",
            "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
            "kind": kind,
            "rotation": round(_hash01(config["seed"], ix, iz, 205) * 360, 1),
            "footprint": round(base_footprint + _hash01(config["seed"], ix, iz, 206) * 4.0, 2),
            "height": round(
                (12.0 + _hash01(config["seed"], ix, iz, 210) * 26.0)
                if kind == "stone_pillar" else
                (6.0 + _hash01(config["seed"], ix, iz, 210) * 7.0),
                2,
            ),
            "layout": layout,
            "route_id": candidate_routes.get((ix, iz)),
        })
    return structures


def _spawn(config: dict[str, Any], heights: list[list[float]],
    slopes: list[list[float]], structures: list[dict[str, Any]]) -> dict[str, Any]:
    resolution = len(heights)
    half_x = config["world_width"] * 0.5
    half_z = config["world_depth"] * 0.5
    step_x = config["world_width"] / (resolution - 1)
    step_z = config["world_depth"] / (resolution - 1)
    candidates: list[tuple[float, float, int, int]] = []
    genome = _world_genome(config)
    for iz in range(2, resolution - 2):
        z = -half_z + iz * step_z
        for ix in range(2, resolution - 2):
            x = -half_x + ix * step_x
            height, slope = heights[iz][ix], slopes[iz][ix]
            if height <= config["water_level"] + 1.8 or slope > 0.62:
                continue
            center = math.hypot(x / max(half_x, 1.0), z / max(half_z, 1.0))
            if config["objective"] == "king_of_the_hill":
                score = -height * 0.055 + slope * 4.0 + center * 2.0
            elif config["objective"] == "traversal":
                score = abs(slope - 0.3) * 5.0 + center
            elif config["objective"] == "ruin_expedition" and structures:
                nearest = min(math.hypot(x - s["x"], z - s["z"]) for s in structures)
                score = abs(nearest - 15.0) * 0.13 + slope * 3.5 + center * 0.5
            else:
                score = slope * 5.0 + center * 1.7
            if genome is not None:
                # Prefer, but do not require, a safe point on the generated
                # traversal spine. Safety and slope remain hard filters above.
                score += (
                    1.0 - _procedural_module().route_influence(genome, x, z)
                ) * (0.8 + config["traversal_bias"] * 2.4)
            # Stable microscopic tie-breaker avoids a repeated preference for
            # the north-west cell without compromising the geometric score.
            score += _hash01(config["seed"], ix, iz, 250) * 0.015
            candidates.append((score, x, ix, iz))
    if not candidates:
        # This remains visible as a failed safety state; it is better than
        # omitting a spawn and crashing the renderer on an extreme design.
        ix = iz = resolution // 2
        x = z = 0.0
        safe = False
    else:
        _, x, ix, iz = min(candidates)
        z = -half_z + iz * step_z
        safe = True
    return {
        "x": round(x, 3), "y": round(heights[iz][ix] + 1.0, 3),
        "z": round(z, 3), "slope": slopes[iz][ix], "safe": safe,
        "clearance": config["spawn_safety_radius"],
    }


def _trees(config: dict[str, Any], heights: list[list[float]],
           slopes: list[list[float]], spawn: dict[str, Any],
           structures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spacing = config["tree_spacing"]
    half_x = config["world_width"] * 0.5
    half_z = config["world_depth"] * 0.5
    count_x = math.ceil(config["world_width"] / spacing)
    count_z = math.ceil(config["world_depth"] / spacing)
    trees = []
    genome = _world_genome(config)
    for iz in range(-count_z, count_z + 1):
        for ix in range(-count_x, count_x + 1):
            if _hash01(config["seed"], ix, iz, 301) > config["forest_density"]:
                continue
            jitter = spacing * 0.43
            x = ix * spacing + (_hash01(config["seed"], ix, iz, 302) * 2 - 1) * jitter
            z = iz * spacing + (_hash01(config["seed"], ix, iz, 303) * 2 - 1) * jitter
            if not (-half_x + 1 <= x <= half_x - 1 and -half_z + 1 <= z <= half_z - 1):
                continue
            y = _sample(heights, config, x, z)
            slope = _sample(slopes, config, x, z)
            if y <= config["water_level"] + 1.0 or slope > 0.74 or y >= config["snow_line"]:
                continue
            moisture = config["moisture"] + 0.25 * _noise(
                config["seed"], x, z,
                max(18.0, config["mountain_scale"] * 0.65), 131,
            )
            if moisture < 0.24 + (1.0 - config["forest_density"]) * 0.28:
                continue
            if math.hypot(x - spawn["x"], z - spawn["z"]) < config["spawn_safety_radius"]:
                continue
            if (
                genome is not None
                and config["traversal_bias"] > 0
                and _procedural_module().route_influence(genome, x, z)
                > 0.42 + (1.0 - config["traversal_bias"]) * 0.42
            ):
                continue
            if any(math.hypot(x - s["x"], z - s["z"]) < s["footprint"] * 0.8
                   for s in structures):
                continue
            kind_choice = _hash01(config["seed"], ix, iz, 304)
            kind = "pine" if y > config["snow_line"] - 22 or config["temperature"] < 0.38 else (
                "oak" if kind_choice < 0.66 else "birch"
            )
            trees.append({
                "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
                "height": round(4.2 + _hash01(config["seed"], ix, iz, 305) * 5.4, 2),
                "kind": kind,
            })
    return trees


def _surface_route_graph(
    config: dict[str, Any],
    genome: dict[str, Any] | None,
    heights: list[list[float]],
    slopes: list[list[float]],
    spawn: dict[str, Any],
    structures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Route the genome spine over dry, moderate-slope sampled terrain."""

    if genome is None:
        return {
            "source": "disabled_for_authored_family",
            "connected": False,
            "nodes": [],
            "edges": [],
            "total_length": 0.0,
        }
    resolution = len(heights)
    step_x = config["world_width"] / max(1, resolution - 1)
    step_z = config["world_depth"] / max(1, resolution - 1)
    origin_x = -config["world_width"] * 0.5
    origin_z = -config["world_depth"] * 0.5

    def world_cell(x: float, z: float) -> tuple[int, int]:
        return (
            max(0, min(resolution - 1, round((x - origin_x) / step_x))),
            max(0, min(resolution - 1, round((z - origin_z) / step_z))),
        )

    def safe_cell(x: float, z: float) -> tuple[int, int]:
        candidates = []
        for iz in range(resolution):
            world_z = origin_z + iz * step_z
            for ix in range(resolution):
                height, slope = heights[iz][ix], slopes[iz][ix]
                dry = height > config["water_level"] + 1.5
                moderate = slope <= TRAVERSABLE_SLOPE
                penalty = (0 if dry else 1_000_000) + (0 if moderate else 100_000)
                distance = (x - (origin_x + ix * step_x)) ** 2 + (z - world_z) ** 2
                candidates.append((penalty + distance, ix, iz))
        _, ix, iz = min(candidates)
        return ix, iz

    def cell_point(cell: tuple[int, int]) -> list[float]:
        ix, iz = cell
        return [
            round(origin_x + ix * step_x, 3),
            round(heights[iz][ix] + 1.0, 3),
            round(origin_z + iz * step_z, 3),
        ]

    neighbors = (
        (-1, -1), (0, -1), (1, -1), (-1, 0),
        (1, 0), (-1, 1), (0, 1), (1, 1),
    )

    route_trees: dict[
        tuple[int, int],
        tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, int]]],
    ] = {}

    def routed_cells(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        if start == goal:
            return [start]
        if start not in route_trees:
            queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
            costs = {start: 0.0}
            previous: dict[tuple[int, int], tuple[int, int]] = {}
            while queue:
                cost, ix, iz = heapq.heappop(queue)
                cell = (ix, iz)
                if cost != costs.get(cell):
                    continue
                for dx, dz in neighbors:
                    nx, nz = ix + dx, iz + dz
                    if not (0 <= nx < resolution and 0 <= nz < resolution):
                        continue
                    distance = math.hypot(dx * step_x, dz * step_z)
                    height = heights[nz][nx]
                    slope = slopes[nz][nx]
                    wet = height <= config["water_level"] + 1.5
                    rise = abs(height - heights[iz][ix]) / max(distance, 1e-9)
                    step_cost = distance * (
                        1.0 + min(4.0, slope) * 2.4 + min(4.0, rise) * 3.2
                        + (38.0 if wet else 0.0)
                    )
                    candidate = cost + step_cost
                    neighbor = (nx, nz)
                    if candidate >= costs.get(neighbor, float("inf")):
                        continue
                    costs[neighbor] = candidate
                    previous[neighbor] = cell
                    heapq.heappush(queue, (candidate, nx, nz))
            route_trees[start] = costs, previous
        costs, previous = route_trees[start]
        if goal not in costs:
            return [start, goal]
        path = [goal]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return path

    def path_edge(identifier: str, kind: str, source: dict[str, Any],
                  target: dict[str, Any], width: float) -> dict[str, Any]:
        cells = routed_cells(
            world_cell(float(source["x"]), float(source["z"])),
            world_cell(float(target["x"]), float(target["z"])),
        )
        points = [cell_point(cell) for cell in cells]
        start_exact = [source["x"], source["y"], source["z"]]
        end_exact = [target["x"], target["y"], target["z"]]
        if points[0] != start_exact:
            points.insert(0, start_exact)
        if points[-1] != end_exact:
            points.append(end_exact)
        length = sum(
            math.hypot(
                float(right[0]) - float(left[0]),
                float(right[2]) - float(left[2]),
            )
            for left, right in zip(points, points[1:])
        )
        return {
            "id": identifier, "kind": kind,
            "from": source["id"], "to": target["id"],
            "start": start_exact, "end": end_exact,
            "points": points,
            "width": round(float(width), 3), "length": round(length, 3),
        }

    nodes = []
    by_id: dict[str, dict[str, Any]] = {}
    for row in genome["route_graph"]["nodes"]:
        ix, iz = safe_cell(float(row["x"]), float(row["z"]))
        x, z = origin_x + ix * step_x, origin_z + iz * step_z
        node = {
            "id": str(row["id"]), "kind": "macro_region",
            "x": round(x, 3), "y": round(heights[iz][ix] + 1.0, 3),
            "z": round(z, 3), "slope": round(slopes[iz][ix], 4),
        }
        nodes.append(node)
        by_id[node["id"]] = node

    edges = []
    for row in genome["route_graph"]["edges"]:
        start, end = by_id[str(row["from"])], by_id[str(row["to"])]
        edges.append(path_edge(
            str(row["id"]), "terrain_spine", start, end, float(row["width"])
        ))
    macro_nodes = list(nodes)

    def attach(identifier: str, kind: str, x: float, y: float, z: float) -> None:
        node = {
            "id": identifier, "kind": kind,
            "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
            "slope": round(_sample(slopes, config, x, z), 4),
        }
        if macro_nodes:
            target = min(
                macro_nodes,
                key=lambda candidate: math.hypot(
                    x - float(candidate["x"]), z - float(candidate["z"])
                ),
            )
            edges.append(path_edge(
                f"attachment-{identifier}", f"{kind}_link",
                target, node, float(config["route_width"]),
            ))
        nodes.append(node)

    attach("spawn", "spawn", spawn["x"], spawn["y"], spawn["z"])
    for structure in structures:
        attach(
            str(structure["id"]), "structure",
            float(structure["x"]), float(structure["y"]) + 1.0,
            float(structure["z"]),
        )
    return {
        "source": "grid_cost_routed_hyperparameter_world_genome",
        "connected": bool(nodes) and len(edges) >= len(nodes) - 1,
        "nodes": nodes,
        "edges": edges,
        "total_length": round(sum(float(edge["length"]) for edge in edges), 3),
        "pathfinding": {
            "resolution": resolution,
            "dry_penalty": 38.0,
            "slope_weight": 2.4,
            "rise_weight": 3.2,
        },
    }


def _route_quality(
    config: dict[str, Any],
    heights: list[list[float]],
    slopes: list[list[float]],
    route_graph: dict[str, Any],
) -> dict[str, Any]:
    """Measure the support spine against the same dry/slope traversal proxy."""

    if not route_graph["edges"]:
        return {
            "sample_count": 0, "traversable_samples": 0,
            "traversable_fraction": 0.0, "maximum_slope": 0.0,
        }
    resolution = len(heights)
    step = max(1.0, min(
        config["world_width"] / max(1, resolution - 1),
        config["world_depth"] / max(1, resolution - 1),
    ))
    samples: list[tuple[float, bool]] = []
    for edge in route_graph["edges"]:
        points = edge.get("points") or [edge["start"], edge["end"]]
        for start, end in zip(points, points[1:]):
            length = math.hypot(
                float(end[0]) - float(start[0]),
                float(end[2]) - float(start[2]),
            )
            count = max(1, math.ceil(length / step))
            for index in range(count + 1):
                amount = index / count
                x = _lerp(float(start[0]), float(end[0]), amount)
                z = _lerp(float(start[2]), float(end[2]), amount)
                height = _sample(heights, config, x, z)
                slope = _sample(slopes, config, x, z)
                samples.append((
                    slope,
                    height > config["water_level"] + 1.5
                    and slope <= TRAVERSABLE_SLOPE,
                ))
    accepted = sum(ok for _, ok in samples)
    return {
        "sample_count": len(samples),
        "traversable_samples": accepted,
        "traversable_fraction": round(accepted / len(samples), 4),
        "maximum_slope": round(max(slope for slope, _ in samples), 4),
    }


def _voxel_unit(width: int, depth: int) -> int:
    requested = max(1, math.ceil(max(width, depth) / VOXEL_TARGET_COLUMNS))
    return 1 << math.ceil(math.log2(requested))


def _nearest_grid_value(
    values: list[list[int]],
    config: dict[str, Any],
    x: float,
    z: float,
) -> int:
    last = len(values) - 1
    gx = max(0, min(last, round((x / config["world_width"] + 0.5) * last)))
    gz = max(0, min(last, round((z / config["world_depth"] + 0.5) * last)))
    return int(values[gz][gx])


def _point_segment_distance_sq(
    point: tuple[float, float, float],
    start: list[float],
    end: list[float],
) -> float:
    direction = tuple(end[index] - start[index] for index in range(3))
    length_sq = sum(value * value for value in direction)
    if length_sq <= 1e-12:
        return sum((point[index] - start[index]) ** 2 for index in range(3))
    amount = max(0.0, min(1.0, sum(
        (point[index] - start[index]) * direction[index]
        for index in range(3)
    ) / length_sq))
    return sum(
        (point[index] - (start[index] + direction[index] * amount)) ** 2
        for index in range(3)
    )


def _route_graph_influence(route_graph: dict[str, Any], x: float, z: float) -> float:
    influence = 0.0
    for edge in route_graph.get("edges", []):
        width = max(0.5, float(edge["width"]))
        points = edge.get("points") or [edge["start"], edge["end"]]
        for start, end in zip(points, points[1:]):
            dx = float(end[0]) - float(start[0])
            dz = float(end[2]) - float(start[2])
            length_sq = dx * dx + dz * dz
            if length_sq <= 1e-12:
                distance_sq = (
                    (x - float(start[0])) ** 2 + (z - float(start[2])) ** 2
                )
            else:
                amount = max(0.0, min(1.0, (
                    (x - float(start[0])) * dx + (z - float(start[2])) * dz
                ) / length_sq))
                px = float(start[0]) + dx * amount
                pz = float(start[2]) + dz * amount
                distance_sq = (x - px) ** 2 + (z - pz) ** 2
            distance = math.sqrt(distance_sq)
            amount = max(0.0, 1.0 - distance / width)
            influence = max(influence, amount * amount * (3.0 - 2.0 * amount))
    return influence


def _semantic_voxel_volume(
    config: dict[str, Any],
    heights: list[list[float]],
    biomes: list[list[int]],
    trees: list[dict[str, Any]],
    structures: list[dict[str, Any]],
    caves: dict[str, Any],
    route_graph: dict[str, Any],
    *,
    preview_digest: str,
    region_tiles: int,
) -> dict[str, Any]:
    """Voxelize the complete semantic preview into compact vertical runs.

    Unlike the former renderer-only height extrusion, this is an occupied 3D
    field. Cave capsules and chambers remove cells, hollow structures add
    shells and interiors, and water/trees retain separate material identities.
    It remains preview authority rather than a native Region capture.
    """

    width = int(config["world_width"])
    depth = int(config["world_depth"])
    unit = _voxel_unit(width, depth)
    nx = math.ceil(width / unit)
    nz = math.ceil(depth / unit)
    origin_x = -width * 0.5
    origin_z = -depth * 0.5
    highest = max(
        max(value for row in heights for value in row),
        float(config["water_level"]),
        *(float(row["y"]) + float(row["height"]) for row in trees),
        *(float(row["y"]) + float(row["height"]) for row in structures),
    )
    ny = min(
        math.ceil(VOXEL_WORLD_HEIGHT / unit),
        max(1, math.ceil(highest / unit) + 2),
    )
    voxels = bytearray(nx * ny * nz)

    def offset(x: int, y: int, z: int) -> int:
        return ((z * nx + x) * ny) + y

    def set_cell(x: int, y: int, z: int, material: int) -> None:
        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            voxels[offset(x, y, z)] = material

    terrain_materials = {1, 2, 20, 60, *(10 + row["id"] for row in BIOMES)}
    for z_index in range(nz):
        world_z = min(depth * 0.5, origin_z + (z_index + 0.5) * unit)
        for x_index in range(nx):
            world_x = min(width * 0.5, origin_x + (x_index + 0.5) * unit)
            surface = _sample(heights, config, world_x, world_z)
            biome = _nearest_grid_value(biomes, config, world_x, world_z)
            surface_cells = max(1, min(ny, math.ceil(surface / unit)))
            rock_end = max(0, surface_cells - 3)
            soil_end = max(rock_end, surface_cells - 1)
            base = offset(x_index, 0, z_index)
            voxels[base:base + rock_end] = bytes([1]) * rock_end
            voxels[base + rock_end:base + soil_end] = (
                bytes([2]) * (soil_end - rock_end)
            )
            voxels[base + soil_end:base + surface_cells] = (
                bytes([10 + biome]) * (surface_cells - soil_end)
            )
            if (
                route_graph["edges"]
                and biome not in (0, 1)
                and _route_graph_influence(route_graph, world_x, world_z) >= 0.34
            ):
                voxels[base + surface_cells - 1] = 60
            if biome in (0, 1):
                water_cells = max(
                    surface_cells,
                    min(ny, math.ceil(float(config["water_level"]) / unit)),
                )
                voxels[base + surface_cells:base + water_cells] = (
                    bytes([30]) * (water_cells - surface_cells)
                )

    carved: set[int] = set()

    def carve_bounds(
        minimum: tuple[float, float, float],
        maximum: tuple[float, float, float],
        predicate: Any,
    ) -> None:
        x0 = max(0, math.floor((minimum[0] - origin_x) / unit))
        x1 = min(nx - 1, math.floor((maximum[0] - origin_x) / unit))
        y0 = max(0, math.floor(minimum[1] / unit))
        y1 = min(ny - 1, math.floor(maximum[1] / unit))
        z0 = max(0, math.floor((minimum[2] - origin_z) / unit))
        z1 = min(nz - 1, math.floor((maximum[2] - origin_z) / unit))
        for z_index in range(z0, z1 + 1):
            world_z = origin_z + (z_index + 0.5) * unit
            for x_index in range(x0, x1 + 1):
                world_x = origin_x + (x_index + 0.5) * unit
                for y_index in range(y0, y1 + 1):
                    world_y = (y_index + 0.5) * unit
                    cell = offset(x_index, y_index, z_index)
                    if voxels[cell] in terrain_materials and predicate(
                        (world_x, world_y, world_z)
                    ):
                        voxels[cell] = 0
                        carved.add(cell)

    for segment in caves["segments"]:
        start = segment["start"]
        end = segment["end"]
        radius = float(segment["radius"]) + unit * 0.24
        minimum = tuple(min(start[index], end[index]) - radius for index in range(3))
        maximum = tuple(max(start[index], end[index]) + radius for index in range(3))
        radius_sq = radius * radius
        carve_bounds(
            minimum,
            maximum,
            lambda point, a=start, b=end, limit=radius_sq: (
                _point_segment_distance_sq(point, a, b) <= limit
            ),
        )
    for chamber in caves["chambers"]:
        center = (float(chamber["x"]), float(chamber["y"]), float(chamber["z"]))
        radius = float(chamber["radius"]) + unit * 0.24
        minimum = tuple(value - radius for value in center)
        maximum = tuple(value + radius for value in center)
        radius_sq = radius * radius
        carve_bounds(
            minimum,
            maximum,
            lambda point, c=center, limit=radius_sq: sum(
                (point[index] - c[index]) ** 2 for index in range(3)
            ) <= limit,
        )

    cave_walls: set[int] = set()
    for cell in carved:
        column, y_index = divmod(cell, ny)
        z_index, x_index = divmod(column, nx)
        for dx, dy, dz in (
            (-1, 0, 0), (1, 0, 0), (0, -1, 0),
            (0, 1, 0), (0, 0, -1), (0, 0, 1),
        ):
            nx_index, ny_index, nz_index = x_index + dx, y_index + dy, z_index + dz
            if not (0 <= nx_index < nx and 0 <= ny_index < ny and 0 <= nz_index < nz):
                continue
            neighbor = offset(nx_index, ny_index, nz_index)
            if voxels[neighbor] in terrain_materials:
                cave_walls.add(neighbor)
    for cell in cave_walls:
        voxels[cell] = 20

    def horizontal_index(world_x: float, world_z: float) -> tuple[int, int]:
        return (
            max(0, min(nx - 1, math.floor((world_x - origin_x) / unit))),
            max(0, min(nz - 1, math.floor((world_z - origin_z) / unit))),
        )

    for structure in structures:
        half = max(unit * 0.5, float(structure["footprint"]) * 0.5)
        radius = half * math.sqrt(2.0) + unit
        x0 = max(0, math.floor((float(structure["x"]) - radius - origin_x) / unit))
        x1 = min(nx - 1, math.floor((float(structure["x"]) + radius - origin_x) / unit))
        z0 = max(0, math.floor((float(structure["z"]) - radius - origin_z) / unit))
        z1 = min(nz - 1, math.floor((float(structure["z"]) + radius - origin_z) / unit))
        y0 = max(0, min(ny - 1, math.ceil(float(structure["y"]) / unit)))
        y1 = max(y0 + 1, min(ny, math.ceil(
            (float(structure["y"]) + float(structure["height"])) / unit
        )))
        angle = math.radians(float(structure["rotation"]))
        cosine, sine = math.cos(angle), math.sin(angle)
        kind = str(structure["kind"])
        material = 51 if kind in {"village_house", "fortified_camp", "bridge"} else 50
        for z_index in range(z0, z1 + 1):
            world_z = origin_z + (z_index + 0.5) * unit
            for x_index in range(x0, x1 + 1):
                world_x = origin_x + (x_index + 0.5) * unit
                dx = world_x - float(structure["x"])
                dz = world_z - float(structure["z"])
                local_x = dx * cosine + dz * sine
                local_z = -dx * sine + dz * cosine
                if abs(local_x) > half or abs(local_z) > half:
                    continue
                for y_index in range(y0, y1):
                    if kind == "stone_pillar":
                        occupied = local_x * local_x + local_z * local_z <= half * half
                    elif kind == "bridge":
                        deck = y_index in {y0, min(y1 - 1, y0 + 1)}
                        support = (
                            abs(local_x) > half - unit * 1.2
                            and abs(local_z) < max(unit, half * 0.32)
                        )
                        occupied = deck or support
                    elif kind == "arch":
                        occupied = (
                            abs(local_x) > half - unit * 1.15
                            or y_index == y1 - 1
                        )
                    else:
                        edge = (
                            abs(local_x) > half - unit * 1.05
                            or abs(local_z) > half - unit * 1.05
                        )
                        floor = y_index == y0
                        roof = y_index == y1 - 1 and kind != "ruin"
                        doorway = (
                            local_z < -half + unit * 1.25
                            and abs(local_x) < unit * 0.85
                            and y_index < min(y1, y0 + max(1, math.ceil(3 / unit)))
                        )
                        occupied = (floor or roof or edge) and not doorway
                    if occupied:
                        set_cell(x_index, y_index, z_index, material)

    for tree in trees:
        x_index, z_index = horizontal_index(float(tree["x"]), float(tree["z"]))
        y0 = max(0, min(ny - 1, math.ceil(float(tree["y"]) / unit)))
        trunk_top = max(y0 + 1, min(ny, math.ceil(
            (float(tree["y"]) + float(tree["height"]) * 0.62) / unit
        )))
        for y_index in range(y0, trunk_top):
            set_cell(x_index, y_index, z_index, 40)
        crown_center_y = float(tree["y"]) + float(tree["height"]) * 0.76
        crown_radius = max(unit * 0.72, float(tree["height"]) * 0.34)
        horizontal_radius = max(1, math.ceil(crown_radius / unit))
        vertical_radius = max(1, math.ceil(crown_radius * 1.15 / unit))
        center_y = max(0, min(ny - 1, math.floor(crown_center_y / unit)))
        for dz in range(-horizontal_radius, horizontal_radius + 1):
            for dx in range(-horizontal_radius, horizontal_radius + 1):
                for dy in range(-vertical_radius, vertical_radius + 1):
                    distance = (
                        (dx / horizontal_radius) ** 2
                        + (dz / horizontal_radius) ** 2
                        + (dy / vertical_radius) ** 2
                    )
                    target_x, target_y, target_z = x_index + dx, center_y + dy, z_index + dz
                    if distance <= 1.0 and (
                        0 <= target_x < nx and 0 <= target_y < ny and 0 <= target_z < nz
                    ):
                        target = offset(target_x, target_y, target_z)
                        if voxels[target] == 0:
                            voxels[target] = 41

    columns: list[list[int]] = []
    run_count = 0
    material_counts: dict[int, int] = {}
    for z_index in range(nz):
        for x_index in range(nx):
            base = offset(x_index, 0, z_index)
            column: list[int] = []
            y_index = 0
            while y_index < ny:
                material = voxels[base + y_index]
                if material == 0:
                    y_index += 1
                    continue
                start = y_index
                while y_index < ny and voxels[base + y_index] == material:
                    y_index += 1
                length = y_index - start
                column.extend((start, length, int(material)))
                material_counts[int(material)] = material_counts.get(int(material), 0) + length
                run_count += 1
            columns.append(column)

    palette_names = {row["id"]: row["name"] for row in _VOLUME_PALETTE}
    surface_floor = max(
        0,
        math.floor((min(value for row in heights for value in row) - unit * 6) / unit)
        * unit,
    )
    cave_lows = [
        min(segment["start"][1], segment["end"][1]) - float(segment["radius"])
        for segment in caves["segments"]
    ] + [
        float(chamber["y"]) - float(chamber["radius"])
        for chamber in caves["chambers"]
    ]
    cutaway_floor = max(
        0,
        math.floor((min(cave_lows, default=surface_floor) - unit * 2) / unit) * unit,
    )
    cave_wall_y_counts: dict[int, int] = {}
    for cell in cave_walls:
        y_index = cell % ny
        cave_wall_y_counts[y_index] = cave_wall_y_counts.get(y_index, 0) + 1
    if cave_wall_y_counts:
        cave_slice_y = max(
            cave_wall_y_counts,
            key=lambda index: (cave_wall_y_counts[index], -index),
        )
        recommended_cut_axis = "y"
        recommended_cut_percent = max(
            5,
            min(90, round(100 * (1 - (cave_slice_y + 1) / ny))),
        )
    else:
        recommended_cut_axis = "z"
        recommended_cut_percent = 48
    value: dict[str, Any] = {
        "schema": VOXEL_VOLUME_SCHEMA,
        "version": 1,
        "authority": "deterministic_semantic_preview_not_native_capture",
        "source_preview_digest": preview_digest,
        "semantic_sha256": "",
        "axis_order": "x_y_z",
        "origin": [origin_x, 0.0, origin_z],
        "bounds": {
            "minimum": [origin_x, 0.0, origin_z],
            "maximum": [width * 0.5, ny * unit, depth * 0.5],
        },
        "voxel_size": unit,
        "dimensions": [nx, ny, nz],
        "column_encoding": "flat_y_start_length_material_runs",
        "palette": list(_VOLUME_PALETTE),
        "columns": columns,
        "recommended_view": {
            "surface_floor_y": surface_floor,
            "cutaway_floor_y": min(surface_floor, cutaway_floor),
            "cut_axis": recommended_cut_axis,
            "cut_percent": recommended_cut_percent,
        },
        "jax_world": {
            "world_count": 1,
            "world_id": 0,
            "region_tiles": region_tiles,
            "shared_world_id": True,
        },
        "metrics": {
            "occupied_voxels": sum(material_counts.values()),
            "carved_cave_voxels": len(carved),
            "cave_wall_voxels": len(cave_walls),
            "run_count": run_count,
            "material_voxels": {
                palette_names[material]: count
                for material, count in sorted(material_counts.items())
            },
        },
    }
    stable = dict(value)
    stable["semantic_sha256"] = ""
    value["semantic_sha256"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return value


def _native_profile(config: dict[str, Any]) -> dict[str, Any]:
    style = config["terrain_style"]
    cave = config["cave_style"]
    if config["generation_mode"] == "procedural_genome":
        return {
            "status": "procedural_graph_compiler_live_reference_available",
            "profile": "procedural_graph_v1",
            "surface_family": "hyperparameter_distribution",
            "requested_cave_style": cave,
            "warnings": [
                "The procedural compiler family has a pinned live 0.5.7 "
                "reference, but this exact pack and seed still require native "
                "generation, Region capture, and traversal validation."
            ],
            "reference_validation": dict(
                PROCEDURAL_GRAPH_REFERENCE_VALIDATION
            ),
            "mapped_controls": [
                "world_variation", "macro_region_count", "region_blend",
                "terrain_component_weights", "base_height", "relief",
                "mountain_scale", "roughness", "ridge_strength", "erosion",
                "pillar_density", "pillar_width", "pillar_height",
                "canyon_strength", "crater_density", "crater_depth",
                "water_level", "lake_coverage", "river_strength",
                "moisture", "temperature", "snow_line", "forest_density",
                "tree_spacing", "cave_style", "cave_density", "cave_scale",
                "cave_verticality", "cave_entrances",
            ],
            "capture_only_controls": [
                "world_width", "world_depth", "route_density", "route_width",
                "traversal_bias", "structure_variety", "spawn_safety_radius",
            ],
            "preview_only_until_density_graph_compiler": [
                "style_strength", "terrace_steps", "route_density",
                "route_width", "traversal_bias", "structure_variety",
                "spawn_safety_radius",
            ],
        }
    if style in ("natural_mountains", "woodland_valley") and cave in (
        "caverns", "labyrinth"
    ):
        profile = "deeproot_caverns"
    else:
        profile = {
            "natural_mountains": "plains_mountains",
            "rolling_hills": "plains_mountains",
            "alpine_ridges": "taiga_mountains",
            "woodland_valley": "plains_mountains",
            "archipelago": "rooted_islands",
            "desert_mesas": "desert_stacks",
            "canyon_badlands": "plains_gorges",
            "flat": "default_flat",
            "pillar_fields": "marble_pillars",
            "terraced_highlands": "desert_stacks",
            "cratered": "twist_crater",
            "floating_islands": "rooted_islands",
            "volcanic_caldera": "volcanic_caldera",
        }[style]
    warnings = []
    native_cave_profiles = {
        "plains_mountains", "taiga_mountains", "plains_gorges",
        "deeproot_caverns", "desert_stacks", "volcanic_caldera",
    }
    if cave == "none" and profile in native_cave_profiles:
        warnings.append(
            "The selected pinned native template may retain authored cave density; "
            "generic cave removal is not silently synthesized."
        )
    if cave != "none" and profile in {"default_flat", "marble_pillars", "rooted_islands"}:
        warnings.append(
            "The selected pinned native template does not certify the requested cave topology."
        )
    return {
        "status": "template_backed",
        "profile": profile,
        "surface_family": style,
        "requested_cave_style": cave,
        "warnings": warnings,
        "mapped_controls": [
            "base_height", "water_level", "tree_spacing",
            "structure_spacing", "structure_jitter", "structure_set",
        ],
        "capture_only_controls": ["world_width", "world_depth"],
        "preview_only_until_density_graph_compiler": [
            "relief", "mountain_scale", "roughness", "ridge_strength",
            "erosion", "style_strength", "terrace_steps", "pillar_density",
            "pillar_width", "pillar_height", "canyon_strength",
            "crater_density", "crater_depth", "cave_density", "cave_scale",
            "cave_verticality", "cave_entrances",
        ],
    }


def extent_capture_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Describe how a bounded Studio extent maps onto exact Region cores.

    Native WorldGen terrain is not made finite by these controls. Width/depth
    define the desired captured environment. Every 96x96 core remains a
    separate capture tile, but adjacent tiles are published as one composite
    artifact and share one JAX world ID. A large environment is therefore one
    world with several Region lookup tiles, never several independent worlds.
    """

    width = int(config["world_width"])
    depth = int(config["world_depth"])
    tiles_x = math.ceil(width / NATIVE_CAPTURE_CORE_BLOCKS)
    tiles_z = math.ceil(depth / NATIVE_CAPTURE_CORE_BLOCKS)
    coverage_width = tiles_x * NATIVE_CAPTURE_CORE_BLOCKS
    coverage_depth = tiles_z * NATIVE_CAPTURE_CORE_BLOCKS
    tile_count = tiles_x * tiles_z
    bounds_window_required = coverage_width != width or coverage_depth != depth
    if tile_count > 1:
        status = "composite_region_one_jax_world_ready_after_capture"
    else:
        status = "single_region_bounds_ready"
    region_core_bounds = None
    if tile_count == 1:
        offset_x = (NATIVE_CAPTURE_CORE_BLOCKS - width) / 2
        offset_z = (NATIVE_CAPTURE_CORE_BLOCKS - depth) / 2
        region_core_bounds = {
            "coordinate_mode": "region_core",
            "minimum": [offset_x, 0.0, offset_z],
            "maximum": [
                offset_x + width, 320.0, offset_z + depth,
            ],
        }
    return {
        "requested_bounded_extent": {"width": width, "depth": depth},
        "native_generation_extent": "unbounded",
        "capture_core_blocks_per_axis": NATIVE_CAPTURE_CORE_BLOCKS,
        "capture_tiles": {"x": tiles_x, "z": tiles_z, "total": tile_count},
        "captured_core_coverage": {
            "width": coverage_width, "depth": coverage_depth,
        },
        "runtime_bounds_window_required": bounds_window_required,
        "region_core_bounds": region_core_bounds,
        "multi_region_capture_required": tile_count > 1,
        "status": status,
        "exact_port_supported_now": True,
        "exact_capture_artifacts_required": True,
        "jax_world": {
            "world_count": 1,
            "region_tiles": tile_count,
            "shared_world_id": True,
            "world_id": 0,
            "composite_manifest_schema": (
                "hytalerl_worldgen_v2_composite_jax_world_v1"
            ),
            "overlap_validation": [
                "physical_geometry",
                "stable_block_identity",
                "stable_fluid_identity",
            ],
        },
        "placement_resolution": (
            "Resolve chunk-aligned tile origins around the native spawn after "
            "generation, capture every tile without another reset, then bind "
            "them to one composite JAX world; do not infer native origins from "
            "the semantic preview."
        ),
    }


def _design(config: dict[str, Any],
            genome: dict[str, Any] | None = None) -> dict[str, Any]:
    """Portable handoff grouped by native authoring concern."""

    if genome is None:
        genome = _world_genome(config)
    value = {
        "schema": SCHEMA,
        "version": VERSION,
        "environment_id": config["name"],
        "seed": config["seed"],
        "extent": {
            "width": config["world_width"],
            "depth": config["world_depth"],
            "native_capture": extent_capture_plan(config),
        },
        "terrain": {
            key: config[key] for key in (
                "terrain_style", "style_strength", "base_height", "relief",
                "mountain_scale", "roughness", "ridge_strength", "erosion",
                "terrace_steps", "pillar_density", "pillar_width",
                "pillar_height", "canyon_strength", "crater_density",
                "crater_depth",
            )
        },
        "hydrology": {
            key: config[key] for key in (
                "water_level", "lake_coverage", "river_strength",
            )
        },
        "climate": {
            key: config[key] for key in ("moisture", "temperature", "snow_line")
        },
        "ecology": {
            key: config[key] for key in ("forest_density", "tree_spacing")
        },
        "caves": {
            key: config[key] for key in (
                "cave_style", "cave_density", "cave_scale",
                "cave_verticality", "cave_entrances",
            )
        },
        "structures": {
            key: config[key] for key in (
                "structure_density", "structure_spacing", "structure_jitter",
                "structure_set", "structure_layout", "structure_material",
            )
        },
        "gameplay": {
            key: config[key] for key in ("objective", "spawn_safety_radius")
        },
        "native_handoff": {
            "generator": "hytale_generator",
            "legacy_world_identity_unchanged": True,
            "template_compilation": _native_profile(config),
            "required_steps": [
                "compile parameters to project-authored WorldGen V2 assets",
                "validate authored assets against the pinned 0.5.7 archive",
                "generate requested seeds with the native server",
                "capture bounded Regions including palette hazard columns",
                "validate semantics, traversal, spawn and held-out diversity",
                "publish an immutable captured-world corpus for JAX",
            ],
        },
    }
    if genome is not None:
        value["generation"] = {
            "mode": "procedural_genome",
            "source": "hyperparameter_distribution_not_named_preset",
            "distribution": {
                key: config[key] for key in (
                    "world_variation", "macro_region_count", "region_blend",
                    "terrain_mountain_weight", "terrain_hill_weight",
                    "terrain_valley_weight", "terrain_plateau_weight",
                    "terrain_canyon_weight", "terrain_pillar_weight",
                    "terrain_crater_weight", "terrain_flat_weight",
                    "route_density", "route_width", "traversal_bias",
                    "structure_variety",
                )
            },
            "resolved_world_genome": genome,
            "native_graph_contract": {
                "seeded_runtime_graph": True,
                "packs_for_seed_corpus": 1,
                "current_graph_compiler_support": True,
                "compiler_family_native_codec_reference_certified": True,
                "requested_pack_native_codec_acceptance_certified": False,
                "reference_validation": dict(
                    PROCEDURAL_GRAPH_REFERENCE_VALIDATION
                ),
                "fail_closed_on_graph_or_receipt_drift": True,
            },
        }
    return value


def native_recipe_digest(config: dict[str, Any]) -> str:
    """Content identity for one asset graph, excluding seed/capture extent."""

    recipe = _design(config)
    recipe.pop("seed", None)
    # Native WorldGen remains unbounded. Width/depth select later environment
    # bounds and must not force duplicate asset packs for the same graph.
    recipe.pop("extent", None)
    generation = recipe.get("generation")
    if isinstance(generation, dict):
        # The native graph identity is the hyperparameter distribution. The
        # resolved macro layout is a seed output, not a separate asset pack.
        generation.pop("resolved_world_genome", None)
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _preview(
    config: dict[str, Any],
    *,
    resolution: int,
    include_volume: bool = False,
) -> dict[str, Any]:
    """Build one normalized deterministic semantic preview."""

    genome = _world_genome(config)
    heights, step_x, step_z = _grid(config, resolution, genome)
    slopes = _slopes(heights, step_x, step_z)
    half_x = config["world_width"] * 0.5
    half_z = config["world_depth"] * 0.5
    biomes: list[list[int]] = []
    for iz, row in enumerate(heights):
        z = -half_z + iz * step_z
        biomes.append([
            _biome(config, -half_x + ix * step_x, z, value, slopes[iz][ix])
            for ix, value in enumerate(row)
        ])

    structures = _structures(config, heights, slopes)
    spawn = _spawn(config, heights, slopes, structures)
    trees = _trees(config, heights, slopes, spawn, structures)
    caves = _caves(config, heights)
    route_graph = _surface_route_graph(
        config, genome, heights, slopes, spawn, structures
    )
    route_quality = _route_quality(config, heights, slopes, route_graph)
    route_graph["quality"] = route_quality
    flat_heights = [value for row in heights for value in row]
    flat_slopes = [value for row in slopes for value in row]
    flat_biomes = [value for row in biomes for value in row]
    total = len(flat_heights)
    water_points = sum(value in (0, 1) for value in flat_biomes)
    traversable = sum(
        biome not in (0, 1) and slope <= TRAVERSABLE_SLOPE
        for biome, slope in zip(flat_biomes, flat_slopes)
    )
    biome_counts = {
        entry["name"]: flat_biomes.count(entry["id"])
        for entry in BIOMES if entry["id"] in flat_biomes
    }
    cave_volume = 0.0
    for segment in caves["segments"]:
        start, end = segment["start"], segment["end"]
        length = math.dist(start, end)
        cave_volume += math.pi * segment["radius"] ** 2 * length
    for chamber in caves["chambers"]:
        cave_volume += 4.0 / 3.0 * math.pi * chamber["radius"] ** 3

    metrics = {
        "minimum_height": round(min(flat_heights), 2),
        "maximum_height": round(max(flat_heights), 2),
        "mean_height": round(sum(flat_heights) / total, 2),
        "relief": round(max(flat_heights) - min(flat_heights), 2),
        "water_fraction": round(water_points / total, 4),
        "forest_fraction": round(flat_biomes.count(4) / total, 4),
        "traversable_fraction": round(traversable / total, 4),
        "maximum_slope": round(max(flat_slopes), 3),
        "biome_count": len(biome_counts),
        "biome_counts": biome_counts,
        "tree_count": len(trees),
        "structure_count": len(structures),
        "structure_kind_count": len({row["kind"] for row in structures}),
        "cave_network_count": len({row["network"] for row in caves["segments"]}),
        "cave_segment_count": len(caves["segments"]),
        "cave_chamber_count": len(caves["chambers"]),
        "cave_entrance_count": len(caves["entrances"]),
        "estimated_cave_volume": round(cave_volume, 1),
        "terrain_region_count": len(genome["regions"]) if genome else 1,
        "terrain_component_count": (
            len(genome["dominant_component_counts"]) if genome else 1
        ),
        "route_node_count": len(route_graph["nodes"]),
        "route_edge_count": len(route_graph["edges"]),
        "route_length": route_graph["total_length"],
        "route_connected": route_graph["connected"],
        "route_traversable_fraction": route_quality["traversable_fraction"],
        "route_maximum_slope": route_quality["maximum_slope"],
        "estimated_region_columns": config["world_width"] * config["world_depth"],
    }

    warnings = []
    if not spawn["safe"]:
        warnings.append("No dry, low-slope spawn candidate exists in the preview.")
    if metrics["traversable_fraction"] < 0.38:
        warnings.append("Less than 38% of sampled terrain is dry and moderate-slope; traversal may fragment.")
    if metrics["water_fraction"] > 0.62:
        warnings.append("Water covers more than 62% of the preview; verify connected dry land natively.")
    if metrics["water_fraction"] < 0.01 and (config["lake_coverage"] > 0.2 or config["river_strength"] > 0.2):
        warnings.append("The requested hydrology does not reach the selected water level in this seed.")
    if metrics["tree_count"] == 0 and config["forest_density"] > 0.25:
        warnings.append("Forest was requested but no tree sites survived water, slope and snow constraints.")
    if metrics["structure_count"] == 0 and config["structure_set"] != "none":
        warnings.append("No structure site survived water and slope constraints for this seed.")
    if config["cave_style"] != "none" and metrics["cave_segment_count"] == 0:
        warnings.append("Caves were requested but the bounded cave graph is empty.")
    if genome is not None and not route_graph["connected"]:
        warnings.append("The procedural traversal support graph is not connected.")
    if genome is not None and route_quality["traversable_fraction"] < 0.58:
        warnings.append(
            "Less than 58% of the procedural route spine is dry and "
            "moderate-slope; reject or retune this seed before native capture."
        )
    if (
        config["terrain_style"] != "flat"
        and metrics["relief"] < config["relief"] * 0.25
    ):
        warnings.append("Realized relief is low for the requested mountain amplitude; try another seed or smaller scale.")

    if genome is not None:
        warnings.append(
            "Procedural graph compilation is available and has one pinned live "
            "0.5.7 reference; this preview is still not native evidence for the "
            "requested pack or seed."
        )

    plausibility_checks = {
        "safe_spawn": bool(spawn["safe"]),
        "global_traversal": metrics["traversable_fraction"] >= 0.38,
        "bounded_water": metrics["water_fraction"] <= 0.62,
        "structures_realized": (
            config["structure_set"] == "none" or metrics["structure_count"] > 0
        ),
        "caves_realized": (
            config["cave_style"] == "none" or metrics["cave_segment_count"] > 0
        ),
        "route_connected": genome is None or bool(route_graph["connected"]),
        "route_traversable": (
            genome is None or metrics["route_traversable_fraction"] >= 0.58
        ),
    }
    plausibility = {
        "passed": all(plausibility_checks.values()),
        "checks": plausibility_checks,
        "authority": "semantic_preview_screen_not_native_traversal",
    }

    design = _design(config, genome)
    high_index = max(range(total), key=flat_heights.__getitem__)
    low_index = min(range(total), key=flat_heights.__getitem__)

    def surface_anchor(index: int, label: str) -> dict[str, Any]:
        iz, ix = divmod(index, resolution)
        return {
            "id": label,
            "x": round(-half_x + ix * step_x, 3),
            "y": round(heights[iz][ix] + 1.0, 3),
            "z": round(-half_z + iz * step_z, 3),
        }

    support_features = {
        "spawn": {"id": "spawn", "x": spawn["x"], "y": spawn["y"], "z": spawn["z"]},
        "high_point": surface_anchor(high_index, "terrain-high-point"),
        "low_point": surface_anchor(low_index, "terrain-low-point"),
        "structure_anchors": [
            {key: row[key] for key in ("id", "kind", "x", "y", "z")}
            for row in structures
        ],
        "cave_entrances": list(caves["entrances"]),
        "route_graph": route_graph,
        "contract": (
            "Environment anchors only. Reward, rules, scoring and minigame "
            "semantics belong to the minigame lane."
        ),
    }
    identity = {
        "config": config,
        "height_milliblocks": [round(v * 1000) for v in flat_heights],
        "biomes": flat_biomes,
        "trees": trees,
        "structures": structures,
        "caves": caves,
        "spawn": spawn,
        "world_genome_digest": genome["digest"] if genome else None,
        "route_graph": route_graph,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    content_digests = {
        "terrain": hashlib.sha256(json.dumps(
            [round(v * 1000) for v in flat_heights], separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "structures": hashlib.sha256(json.dumps(
            structures, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "caves": hashlib.sha256(json.dumps(
            caves, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "routes": hashlib.sha256(json.dumps(
            route_graph, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "world_genome": genome["digest"] if genome else None,
    }
    result = {
        "schema": PREVIEW_SCHEMA,
        "version": VERSION,
        "digest": digest,
        "config": config,
        "world_genome": genome,
        "content_digests": content_digests,
        "design": design,
        "grid": {
            "resolution": resolution,
            "world_width": config["world_width"],
            "world_depth": config["world_depth"],
            "step_x": round(step_x, 6),
            "step_z": round(step_z, 6),
            "heights": heights,
            "slopes": slopes,
            "biomes": biomes,
        },
        "biome_legend": list(BIOMES),
        "trees": trees,
        "structures": structures,
        "caves": caves,
        "routes": route_graph,
        "spawn": spawn,
        "support_features": support_features,
        "metrics": metrics,
        "plausibility": plausibility,
        "warnings": warnings,
        "quality": "review" if warnings else "ready_for_native_authoring",
        "boundary": options()["boundary"],
    }
    if include_volume:
        result["volume"] = _semantic_voxel_volume(
            config,
            heights,
            biomes,
            trees,
            structures,
            caves,
            route_graph,
            preview_digest=digest,
            region_tiles=design["extent"]["native_capture"]["jax_world"][
                "region_tiles"
            ],
        )
    return result


@lru_cache(maxsize=192)
def _cached_preview(canonical_config: str, resolution: int) -> str:
    config = json.loads(canonical_config)
    return json.dumps(
        _preview(config, resolution=resolution, include_volume=True),
        sort_keys=True,
        separators=(",", ":"),
    )


def preview(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a cached full-detail semantic mesh and authoring diagnostics."""

    config = normalize(payload)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    # Decode a private cached serialization so callers can never mutate cache
    # state shared with a later request.
    return json.loads(_cached_preview(canonical, PREVIEW_RESOLUTION))


def _captured_modules() -> tuple[Any, Any]:
    """Import the experimental exact adapter only when it is requested."""

    import sys

    package_root = PROJECT_ROOT / "experimental" / "worldgen-v2"
    hytalegym_source = PROJECT_ROOT / "HytaleRL" / "hytalegym"
    for source in (package_root, hytalegym_source):
        if not source.is_dir():
            raise FileNotFoundError(f"required WorldGen source is absent: {source}")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    from jax_port.worlds.composite_world import V2CompositeWorld
    from jax_port.worlds.voxel_volume import captured_preview

    return V2CompositeWorld, captured_preview


def _contained_artifact(root: Path, relative: str) -> bool:
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return False
    return candidate.is_file()


def _capture_candidates(manifest: Path, root: Path) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            candidate.resolve()
            for candidate in (
                manifest.parent / "artifacts",
                manifest.parent,
                manifest.parent.parent / "artifacts",
                root,
            )
            if candidate.is_dir()
        )
    )


@lru_cache(maxsize=CAPTURE_RECORD_CACHE_SIZE)
def _cached_capture_record(
    manifest_text: str,
    manifest_mtime_ns: int,
    manifest_size: int,
    candidate_signatures: tuple[tuple[str, int], ...],
) -> tuple[dict[str, Any] | None, str]:
    """Authenticate immutable capture metadata once per filesystem identity."""

    del manifest_mtime_ns, manifest_size
    V2CompositeWorld, _ = _captured_modules()
    manifest = Path(manifest_text)
    failure = "no artifact root contains every referenced tile"
    for artifact_text, _artifact_mtime_ns in candidate_signatures:
        artifact_root = Path(artifact_text)
        try:
            world = V2CompositeWorld.load(
                manifest,
                artifact_root,
                verify_artifacts=False,
            )
            references = (
                relative
                for tile in world.tiles
                for relative in (
                    tile.region_path,
                    tile.block_semantic_path,
                    tile.fluid_semantic_path,
                )
            )
            if not all(
                _contained_artifact(artifact_root, relative)
                for relative in references
            ):
                continue
            return {
                "capture_id": world.semantic_sha256,
                "environment_id": world.environment_id,
                "structure": world.structure,
                "seed": world.seed,
                "split": world.split,
                "tile_grid": list(world.tile_grid),
                "region_tiles": len(world.tiles),
                "bounds_min": list(world.bounds_min),
                "bounds_max": list(world.bounds_max),
                "one_jax_world": True,
                "manifest_path": str(manifest.resolve()),
                "artifact_root": str(artifact_root),
            }, ""
        except (OSError, TypeError, ValueError) as exc:
            failure = str(exc)
    return None, failure


def _capture_records() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Discover published composites without accepting browser-supplied paths."""

    rows: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for configured_root in CAPTURE_ROOTS:
        root = Path(configured_root).resolve()
        if not root.is_dir():
            continue
        manifests = sorted({
            *root.rglob("*.composite-world.json"),
            *root.rglob("composite-world.json"),
        })
        for manifest in manifests:
            stat = manifest.stat()
            candidates = _capture_candidates(manifest, root)
            signatures = tuple(
                (str(candidate), candidate.stat().st_mtime_ns)
                for candidate in candidates
            )
            record, failure = _cached_capture_record(
                str(manifest.resolve()),
                stat.st_mtime_ns,
                stat.st_size,
                signatures,
            )
            if record is None:
                errors.append({"path": str(manifest.resolve()), "error": failure})
            else:
                rows.setdefault(record["capture_id"], dict(record))
    ordered = sorted(
        rows.values(),
        key=lambda row: (row["environment_id"], row["seed"], row["capture_id"]),
    )
    return ordered, errors


def captured_worlds() -> dict[str, Any]:
    """List exact composite worlds available to the local console."""

    from console.core.worlds import replay_terrain

    rows, errors = _capture_records()
    return {
        "schema": "hytalerl_worldgen_v2_console_capture_index_v1",
        "captures": [
            {**row, "replay_reference": replay_terrain.captured_reference(row)}
            for row in rows
        ],
        "errors": errors,
        "roots": [str(Path(root)) for root in CAPTURE_ROOTS],
        "cache": capture_cache_state(),
    }


@lru_cache(maxsize=CAPTURE_VOLUME_CACHE_SIZE)
def _cached_captured_volume(
    capture_id: str,
    manifest_path: str,
    manifest_mtime_ns: int,
    manifest_size: int,
    artifact_root: str,
    target_columns: int,
) -> str:
    del capture_id, manifest_mtime_ns, manifest_size
    _, render = _captured_modules()
    return json.dumps(
        render(manifest_path, artifact_root, target_columns=target_columns),
        separators=(",", ":"),
    )


def captured_volume(capture_id: str, *, target_columns: int = 96) -> dict[str, Any]:
    """Authenticate and stream one discovered exact world to the renderer."""

    if not isinstance(capture_id, str) or not _CAPTURE_ID.fullmatch(capture_id):
        raise ValueError("invalid captured world id")
    rows, _ = _capture_records()
    record = next((row for row in rows if row["capture_id"] == capture_id), None)
    if record is None:
        raise FileNotFoundError(f"unknown captured world {capture_id!r}")
    manifest = Path(record["manifest_path"])
    stat = manifest.stat()
    value = json.loads(_cached_captured_volume(
        capture_id,
        record["manifest_path"],
        stat.st_mtime_ns,
        stat.st_size,
        record["artifact_root"],
        target_columns,
    ))
    from console.core.worlds import replay_terrain

    value["replay_reference"] = replay_terrain.captured_reference(record)
    return value


def capture_cache_state() -> dict[str, Any]:
    """Bounded generated-world discovery and replay-cache telemetry."""

    return {
        "schema": "console-worldgen-capture-cache-v1",
        "records": _cached_capture_record.cache_info()._asdict(),
        "volumes": _cached_captured_volume.cache_info()._asdict(),
        "record_limit": CAPTURE_RECORD_CACHE_SIZE,
        "volume_limit": CAPTURE_VOLUME_CACHE_SIZE,
    }


def clear_capture_cache() -> int:
    """Drop generated-world host metadata and rendered voxel payloads."""

    dropped = (
        _cached_capture_record.cache_info().currsize
        + _cached_captured_volume.cache_info().currsize
    )
    _cached_capture_record.cache_clear()
    _cached_captured_volume.cache_clear()
    return dropped


def _batch_integer(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    parsed = int(numeric)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _clamp_field(name: str, value: float) -> float:
    field = FIELDS[name]
    return round(max(float(field["min"]), min(float(field["max"]), value)), 6)


def _variant_config(base: dict[str, Any], *, index: int, seed: int,
                    strategy: str, strength: float) -> tuple[dict[str, Any], dict[str, Any]]:
    if strategy == "native_seeds":
        return {**base, "seed": seed}, {}

    source = dict(base)
    changes: dict[str, Any] = {}

    if strategy == "parametric_worlds":
        def replace(name: str, value: Any) -> None:
            old = source[name]
            source[name] = value
            if old != value:
                changes[name] = {"from": old, "to": value}

        replace("generation_mode", "procedural_genome")
        replace("world_variation", round(strength, 6))
        if source["cave_style"] != "none":
            replace("cave_style", "procedural")
        if source["structure_set"] != "none":
            replace("structure_set", "procedural_mix")
            replace("structure_layout", "terrain_network")
        source["seed"] = seed
        return normalize(source), changes

    def vary(name: str, span: float, salt: int) -> None:
        old = float(source[name])
        delta = (_hash01(seed, index, salt, 601) * 2.0 - 1.0) * span * strength
        new = _clamp_field(name, old + delta)
        source[name] = new
        if new != old:
            changes[name] = {"from": old, "to": new}

    vary("relief", 22.0, 1)
    vary("mountain_scale", 28.0, 2)
    vary("roughness", 0.22, 3)
    vary("ridge_strength", 0.24, 4)
    vary("erosion", 0.2, 5)
    vary("water_level", 6.0, 6)
    vary("lake_coverage", 0.18, 7)
    vary("river_strength", 0.2, 8)
    vary("moisture", 0.18, 9)
    vary("temperature", 0.16, 10)
    vary("forest_density", 0.2, 11)
    vary("cave_density", 0.2, 12)
    vary("structure_density", 0.18, 13)
    source["name"] = f"{base['name']}-v{index + 1:03d}"[:64]
    source["seed"] = seed
    return normalize(source), changes


def _metric_distribution(rows: list[dict[str, Any]], name: str) -> dict[str, float]:
    values = sorted(float(row["metrics"][name]) for row in rows)
    return {
        "minimum": round(values[0], 6),
        "mean": round(sum(values) / len(values), 6),
        "maximum": round(values[-1], 6),
    }


def batch(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate a compact deterministic corpus preview for up to 512 seeds."""

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("worldgen batch must be a JSON object")
    allowed = {
        "config", "start_seed", "count", "stride", "resolution",
        "strategy", "variation_strength",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown worldgen batch field(s): {', '.join(unknown)}")
    base = normalize(payload.get("config"))
    start_seed = _batch_integer(
        "start_seed", payload.get("start_seed", base["seed"]), 0, 2_147_483_647
    )
    count = _batch_integer("count", payload.get("count", 100), 1, MAX_BATCH_SEEDS)
    stride = _batch_integer("stride", payload.get("stride", 1), 1, 1_000_000)
    resolution = _batch_integer(
        "resolution", payload.get("resolution", FAST_BATCH_RESOLUTIONS[0]),
        min(FAST_BATCH_RESOLUTIONS), max(FAST_BATCH_RESOLUTIONS),
    )
    if resolution not in FAST_BATCH_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {list(FAST_BATCH_RESOLUTIONS)}")
    strategy = str(payload.get("strategy", "native_seeds"))
    # Read old downloaded plans without retaining their preset-selection
    # semantics. New output always names the hyperparameter strategy directly.
    if strategy == "wild_variants":
        strategy = "parametric_worlds"
    if strategy not in {"native_seeds", "parametric_worlds", "plausible_variants"}:
        raise ValueError(
            "strategy must be native_seeds, parametric_worlds or plausible_variants"
        )
    try:
        strength = float(payload.get("variation_strength", 0.55))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("variation_strength must be a number") from exc
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("variation_strength must be between 0 and 1")
    last_seed = start_seed + (count - 1) * stride
    if last_seed > 2_147_483_647:
        raise ValueError("seed range exceeds 2^31 - 1")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index in range(count):
        seed = start_seed + index * stride
        config, changes = _variant_config(
            base, index=index, seed=seed, strategy=strategy, strength=strength
        )
        result = _preview(config, resolution=resolution)
        rows.append({
            "index": index,
            "seed": seed,
            "digest": result["digest"],
            "design_digest": native_recipe_digest(config),
            "quality": result["quality"],
            "plausible": result["plausibility"]["passed"],
            "plausibility_checks": result["plausibility"]["checks"],
            "warning_count": len(result["warnings"]),
            "content_digests": result["content_digests"],
            "metrics": {
                key: result["metrics"][key] for key in (
                    "relief", "water_fraction", "traversable_fraction",
                    "tree_count", "structure_count", "cave_network_count",
                    "cave_entrance_count", "structure_kind_count",
                    "terrain_region_count", "terrain_component_count",
                    "route_node_count", "route_edge_count", "route_length",
                    "route_traversable_fraction",
                )
            },
            "native_profile": result["design"]["native_handoff"]["template_compilation"]["profile"],
            "support_features": {
                "spawn": result["support_features"]["spawn"],
                "high_point": result["support_features"]["high_point"],
                "low_point": result["support_features"]["low_point"],
                "structure_anchor_count": len(result["support_features"]["structure_anchors"]),
                "cave_entrance_count": len(result["support_features"]["cave_entrances"]),
                "route_node_count": len(result["support_features"]["route_graph"]["nodes"]),
                "route_edge_count": len(result["support_features"]["route_graph"]["edges"]),
            },
            "world_genome": ({
                "digest": result["world_genome"]["digest"],
                "region_count": len(result["world_genome"]["regions"]),
                "dominant_component_counts": result["world_genome"][
                    "dominant_component_counts"
                ],
                "route_edge_count": len(result["world_genome"]["route_graph"]["edges"]),
            } if result["world_genome"] else None),
            "resolved_config": config,
            "parameter_changes": changes,
        })
    elapsed = time.perf_counter() - started
    ready = sum(row["quality"] == "ready_for_native_authoring" for row in rows)
    plausible = sum(row["plausible"] for row in rows)
    heldout = max(1, round(count * 0.2)) if count >= 3 else 0
    return {
        "schema": BATCH_SCHEMA,
        "version": VERSION,
        "strategy": strategy,
        "base_config": base,
        "seed_range": {
            "start": start_seed, "count": count, "stride": stride,
            "last": last_seed,
        },
        "preview_resolution": resolution,
        "rows": rows,
        "aggregate": {
            "unique_preview_digests": len({row["digest"] for row in rows}),
            "unique_design_digests": len({row["design_digest"] for row in rows}),
            "unique_terrain_fields": len({
                row["content_digests"]["terrain"] for row in rows
            }),
            "unique_structure_layouts": len({
                row["content_digests"]["structures"] for row in rows
            }),
            "unique_cave_layouts": len({
                row["content_digests"]["caves"] for row in rows
            }),
            "unique_route_graphs": len({
                row["content_digests"]["routes"] for row in rows
            }),
            "unique_world_genomes": len({
                row["content_digests"]["world_genome"] for row in rows
                if row["content_digests"]["world_genome"] is not None
            }),
            "ready": ready,
            "review": count - ready,
            "plausible": plausible,
            "plausibility_rejected": count - plausible,
            "generation_seconds": round(elapsed, 4),
            "seeds_per_second": round(count / max(elapsed, 1e-9), 2),
            "distributions": {
                name: _metric_distribution(rows, name)
                for name in (
                    "relief", "water_fraction", "traversable_fraction",
                    "tree_count", "structure_count", "cave_network_count",
                    "structure_kind_count", "terrain_region_count",
                    "route_edge_count", "route_length",
                    "route_traversable_fraction",
                )
            },
        },
        "native_seed_plan": {
            "world": "hytale_generator",
            "legacy_world_identity_unchanged": True,
            "seeds": [row["seed"] for row in rows],
            "heldout_count": heldout,
            "asset_packs_required": (
                1 if strategy in {"native_seeds", "parametric_worlds"} else count
            ),
            "native_compilation_status": (
                "procedural_graph_compiler_live_reference_available"
                if strategy == "parametric_worlds"
                or base["generation_mode"] == "procedural_genome"
                else "template_compiler_available"
            ),
            "capture_resume_supported": True,
            "extent_plan": extent_capture_plan(base),
            "not_native_capture": True,
        },
        "boundary": (
            "This batch is generated preview evidence and a native seed plan. "
            "Exact worlds begin only after pack compilation, native generation "
            "and bounded Region capture."
        ),
    }


def compile_native(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compile a design into a local, pinned V2 asset pack without deploying."""

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("worldgen compile request must be a JSON object")
    if "config" in payload:
        unknown = sorted(set(payload) - {"config"})
        if unknown:
            raise ValueError(
                f"unknown worldgen compile field(s): {', '.join(unknown)}"
            )
        config_value = payload["config"]
    else:
        config_value = payload
    if not isinstance(config_value, dict):
        raise ValueError("worldgen compile config must be a JSON object")
    config = normalize(config_value)
    design_digest = native_recipe_digest(config)
    build_id = _design_id(config["name"], design_digest)
    compiler_root = Path(__file__).resolve().parents[3] / "experimental" / "worldgen-v2"
    import sys

    if str(compiler_root) not in sys.path:
        sys.path.insert(0, str(compiler_root))
    from design_compiler import compile_design

    result = compile_design(
        config,
        output=BUILD_DIR / build_id,
        verify_archive=True,
    )
    return {**result, "build_id": build_id, "deployed": False}


def builds() -> dict[str, Any]:
    """List validated compiler receipts without loading or deploying packs."""

    rows = []
    errors = []
    compiler_root = Path(__file__).resolve().parents[3] / "experimental" / "worldgen-v2"
    import sys

    if str(compiler_root) not in sys.path:
        sys.path.insert(0, str(compiler_root))
    from design_compiler import validate_pack

    if BUILD_DIR.is_dir():
        for path in BUILD_DIR.iterdir():
            if not path.is_dir():
                continue
            receipt = path / "studio-authoring-receipt.json"
            try:
                validated = validate_pack(path)
                value = json.loads(receipt.read_text(encoding="utf-8"))
                rows.append({
                    "build_id": path.name,
                    "environment_id": validated["environment_id"],
                    "native_profile": validated["native_profile"],
                    "pack_semantic_sha256": validated["pack_semantic_sha256"],
                    "generated_at_utc": value["generated_at_utc"],
                    "path": str(path),
                    "validation": "passed",
                    "deployed": False,
                })
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
    rows.sort(key=lambda row: row["generated_at_utc"], reverse=True)
    return {"builds": rows, "errors": errors, "directory": str(BUILD_DIR)}


def _design_id(name: str, digest: str) -> str:
    return f"{name}-{digest[:10]}"


def save(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist a normalized design and its preview receipt atomically."""

    result = preview(payload)
    design_id = _design_id(result["config"]["name"], result["digest"])
    record = {
        "schema": "hytalerl_worldgen_v2_console_saved_design_v1",
        "version": VERSION,
        "design_id": design_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "digest": result["digest"],
        "design": result["design"],
        "config": result["config"],
        "preview_receipt": {
            "quality": result["quality"],
            "metrics": result["metrics"],
            "warnings": result["warnings"],
            "preview_schema": result["schema"],
            "preview_resolution": result["grid"]["resolution"],
            "not_native_capture": True,
        },
    }
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    target = DESIGN_DIR / f"{design_id}.json"
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{design_id}-", suffix=".tmp", dir=DESIGN_DIR
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            pass
    return {
        "design_id": design_id,
        "path": str(target),
        "digest": result["digest"],
        "saved_at": record["saved_at"],
        "quality": result["quality"],
    }


def saved() -> dict[str, Any]:
    """List valid records newest first; a corrupt scratch file is reported."""

    rows = []
    errors = []
    # Seeding deliberately does NOT happen here. It used to, and it made this
    # read-only listing write into whatever DESIGN_DIR points at -- including a
    # test's tmp_path, which broke
    # test_design_save_load_and_listing_are_reproducible by materialising a
    # second design the test never saved. Restoring seeds is a startup concern;
    # see `ensure_canonical_designs`, called from the server.
    seen: set[str] = set()
    # The shared root first, so a record that exists in both wins over the
    # copy left behind in the console's scratch tree.
    for directory in (DESIGN_DIR, LEGACY_DESIGN_DIR):
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            if path.name in seen:
                continue
            seen.add(path.name)
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                rows.append({
                    "design_id": record["design_id"],
                    "name": record["config"]["name"],
                    "seed": record["config"]["seed"],
                    "saved_at": record["saved_at"],
                    "digest": record["digest"],
                    "quality": record["preview_receipt"]["quality"],
                    "path": str(path),
                })
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
    rows.sort(key=lambda row: row["saved_at"], reverse=True)
    return {"designs": rows, "errors": errors, "directory": str(DESIGN_DIR)}


def load(design_id: str) -> dict[str, Any] | None:
    """Load a saved record without allowing path traversal."""

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", design_id or ""):
        raise ValueError("invalid worldgen design id")
    path = next(
        (
            candidate
            for candidate in (
                DESIGN_DIR / f"{design_id}.json",
                LEGACY_DESIGN_DIR / f"{design_id}.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if path is None:
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read saved design {design_id!r}: {exc}") from exc
    if record.get("design_id") != design_id:
        raise ValueError("saved design identity does not match its filename")
    # Re-normalize before handing browser state back; old or manually edited
    # records cannot bypass the current field contract.
    record["config"] = normalize(record.get("config"))
    return record
