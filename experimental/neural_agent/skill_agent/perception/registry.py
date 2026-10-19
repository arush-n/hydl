"""The content registry: every item, block and gather type of a server version's assets as fixed-width properties,
with a description vector per item (design: ``docs/PERCEPTION.md``; owner rulings, 2026-10-19).

Built offline from the game's own asset archive, so the base game and mods, whose content is asset JSON in the same
codecs, are read by one builder. The adapter looks rows up by asset id at run time; nothing here runs per tick.

- **Items** (``Server/Item/Items/**.json``) are resolved through their ``Parent`` chains (a child's fields override
  its parent's, objects merged key by key, lists replaced) and described by ``ITEM_FEATURES``.
- **Blocks** are the items whose resolved ``BlockType`` is not empty, described by ``BLOCK_FEATURES``, with their
  breaking rule: a gather type and a quality.
- **Gather types** form an open table, the union of what blocks and tools name, so a mod's new gather type is a new
  column, never a rejection. A tool's ``Tool.Specs`` give its power and quality per gather type (``tool_power``,
  ``tool_quality``); a spec marked ``IsIncorrect`` counts as no power.
- **Descriptions**: each item's name, description, categories, subcategory and tags, read through the language file,
  embedded by a pinned sentence encoder and projected to ``description_width`` numbers. The projection (mean and basis)
  is fitted once per registry and stored, so content added later (a mod) is projected into the same space.

Saved as ``<prefix>.npz`` (the tables) with ``<prefix>.json`` (the manifest: versions, sizes, names, SHA-256s).
Usage: python -m experimental.neural_agent.skill_agent.perception.registry --assets ASSETS.zip --out PREFIX
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

REGISTRY_SCHEMA = "skill_content_registry_v1"
DEFAULT_VERSION = "0.7.2"
DESCRIPTION_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DESCRIPTION_WIDTH = 32
ITEM_PREFIX = "Server/Item/Items/"
LANGUAGE_FILE = "Server/Languages/en-US/server.lang"

QUALITY_RANK = {"Junk": 0, "Common": 1, "Uncommon": 2, "Rare": 3, "Epic": 4, "Legendary": 5}
"""Gameplay quality as a rank; the top rank is 5."""
NON_GAMEPLAY_QUALITY = frozenset({"Developer", "Technical", "Debug", "Tool"})
"""Qualities that mark editor, debug and builder items, which no agent should hold."""
ARMOR_SLOTS = ("Head", "Chest", "Legs", "Hands")

ITEM_FEATURES = (
    "block", "tool", "weapon", "armor", "consumable", "utility", "furniture", "non_gameplay",
    "quality", "item_level", "max_stack_log2", "max_durability_log10",
    *(f"armor_{slot.lower()}" for slot in ARMOR_SLOTS), "armor_health", "armor_physical", "armor_projectile",
    "tool_best_power", "tool_gather_types",
)
BLOCK_FEATURES = (
    "breakable", "unbreakable", "breaking_quality", "drops", "solid", "opaque", "partly_transparent", "cube", "door",
    "stateful", "usable", "collision_effect", "light", "needs_support", "stackable",
)
BLOCK_DEFAULTS = {"Material": "Empty", "Opacity": "Solid", "IsDoor": False}
"""The engine's defaults for an absent block field, as the line-of-sight semantics checked against the live server
use them (``hytalegym/worldgen/surrogate/semantics.py``, ``_BLOCK_DEFAULTS``)."""
UNIQUE_ITEM_SECTIONS = ("Tool", "Weapon", "Armor", "BuilderTool", "BlockSelectorTool")
"""An item with one of these sections stacks to 1 when ``MaxStack`` is absent, any other to 100 (the crafting
mechanics' rule, ``hytalegym/jax/combat/arsenal/world_actions/crafting.py``, ``_item_properties``)."""


class Registry(NamedTuple):
    """The tables of one server version. Rows of ``item_*`` follow ``item_ids``; rows of ``block_*`` follow
    ``block_ids``; columns of ``tool_*`` follow ``gather_types``."""

    manifest: dict
    item_ids: tuple[str, ...]
    item_features: np.ndarray  # float32[I, len(ITEM_FEATURES)]
    item_descriptions: np.ndarray  # float32[I, description_width]
    tool_power: np.ndarray  # float32[I, G]: the power of each item against each gather type, 0 when it has none
    tool_quality: np.ndarray  # float32[I, G]: the quality it reaches for that gather type
    block_ids: tuple[str, ...]
    block_features: np.ndarray  # float32[K, len(BLOCK_FEATURES)]
    block_gather: np.ndarray  # int32[K]: the block's gather type (an index into gather_types), or -1
    block_quality: np.ndarray  # float32[K]: the quality its breaking rule asks for
    gather_types: tuple[str, ...]
    projection_mean: np.ndarray  # float32[E]: the description projection, for content added later
    projection_basis: np.ndarray  # float32[E, description_width]


@dataclass(frozen=True)
class RegistryConfig:
    version: str = DEFAULT_VERSION
    description_width: int = DESCRIPTION_WIDTH
    encoder: str = DESCRIPTION_ENCODER


Encoder = Callable[[Sequence[str]], np.ndarray]
"""Texts to unit vectors float32[N, E]."""


def _merge(base: dict, child: dict) -> dict:
    out = dict(base)
    for key, value in child.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def read_items(archive: zipfile.ZipFile) -> dict[str, dict]:
    """Every item under ``Server/Item/Items/`` by id (its file name), each resolved through its ``Parent`` chain."""

    raw: dict[str, dict] = {}
    for name in archive.namelist():
        if name.startswith(ITEM_PREFIX) and name.endswith(".json"):
            try:
                value = json.loads(archive.read(name))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(value, dict):
                raw[name.rsplit("/", 1)[1][:-len(".json")]] = value
    resolved: dict[str, dict] = {}

    def resolve(item: str, chain: tuple[str, ...]) -> dict:
        if item in resolved:
            return resolved[item]
        value = raw[item]
        parent = value.get("Parent")
        if isinstance(parent, str) and parent in raw and parent not in chain:
            value = _merge(resolve(parent, chain + (item,)), value)
        resolved[item] = value
        return value

    for item in sorted(raw):
        resolve(item, ())
    return {item: resolved[item] for item in sorted(raw)}


def read_language(archive: zipfile.ZipFile, path: str = LANGUAGE_FILE) -> dict[str, str]:
    """The ``key = value`` lines of a language file (comments skipped, ``\\`` continues a line), keys as written."""

    if path not in archive.namelist():
        return {}
    text = archive.read(path).decode("utf-8-sig")
    entries: dict[str, str] = {}
    pending = ""
    for line in text.splitlines():
        line = pending + line.strip()
        if line.endswith("\\"):
            pending = line[:-1].rstrip() + " "
            continue
        pending = ""
        if not line or line.startswith("#") or " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        entries[key.strip()] = value.strip()
    return entries


def _translated(item: dict, field: str, language: dict[str, str]) -> str:
    key = (item.get("TranslationProperties") or {}).get(field)
    if not isinstance(key, str):
        return ""
    return language.get(key.removeprefix("server."), "")


def _stat(modifiers: Any, stat: str) -> float:
    total = 0.0
    for entry in (modifiers or {}).get(stat, []) if isinstance(modifiers, dict) else []:
        if isinstance(entry, dict) and isinstance(entry.get("Amount"), (int, float)):
            total += float(entry["Amount"])
    return total


def _color_brightness(color: Any) -> float:
    """A light's brightness in [0, 1] from its hex colour (#rgb or #rrggbb), 0 when it has none."""

    if not isinstance(color, str) or not color.startswith("#"):
        return 0.0
    digits = color[1:]
    if len(digits) == 3:
        digits = "".join(2 * digit for digit in digits)
    try:
        channels = [int(digits[i:i + 2], 16) for i in range(0, 6, 2)]
    except ValueError:
        return 0.0
    return max(channels) / 255.0


def _tool_specs(item: dict) -> list[dict]:
    tool = item.get("Tool")
    specs = tool.get("Specs") if isinstance(tool, dict) else None
    return [spec for spec in specs or [] if isinstance(spec, dict) and isinstance(spec.get("GatherType"), str)]


def _breaking(item: dict) -> dict:
    block = item.get("BlockType")
    gathering = block.get("Gathering") if isinstance(block, dict) else None
    breaking = gathering.get("Breaking") if isinstance(gathering, dict) else None
    return breaking if isinstance(breaking, dict) else {}


def gather_types_of(items: dict[str, dict]) -> tuple[str, ...]:
    """The open gather-type table: every gather type a block's breaking rule or a tool's spec names, sorted."""

    names = set()
    for item in items.values():
        gather = _breaking(item).get("GatherType")
        if isinstance(gather, str):
            names.add(gather)
        names.update(spec["GatherType"] for spec in _tool_specs(item))
    return tuple(sorted(names))


def item_row(item: dict) -> np.ndarray:
    """One item's ``ITEM_FEATURES``."""

    block = item.get("BlockType")
    armor = item.get("Armor") if isinstance(item.get("Armor"), dict) else {}
    quality = item.get("Quality")
    categories = [c for c in item.get("Categories") or [] if isinstance(c, str)]
    stack = item.get("MaxStack")
    if stack is None:
        stack = 1 if any(item.get(section) is not None for section in UNIQUE_ITEM_SECTIONS) else 100
    durability = item.get("MaxDurability")
    powers = [float(spec.get("Power", 0.0)) for spec in _tool_specs(item) if not spec.get("IsIncorrect")]
    resistance = armor.get("DamageResistance") if isinstance(armor.get("DamageResistance"), dict) else {}
    values = {
        "block": float(isinstance(block, dict) and bool(block)),
        "tool": float(isinstance(item.get("Tool"), dict)),
        "weapon": float(isinstance(item.get("Weapon"), dict)),
        "armor": float(bool(armor)),
        "consumable": float(bool(item.get("Consumable"))),
        "utility": float(bool(item.get("Utility"))),
        "furniture": float(any(c.split(".")[0] == "Furniture" for c in categories)),
        "non_gameplay": float(quality in NON_GAMEPLAY_QUALITY),
        "quality": QUALITY_RANK.get(quality, 0) / 5.0,
        "item_level": float(item.get("ItemLevel") or 0) / 100.0,
        "max_stack_log2": math.log2(stack) / 7.0 if isinstance(stack, int) and stack > 0 else 0.0,
        "max_durability_log10": (math.log10(1.0 + float(durability)) / 4.0
                                 if isinstance(durability, (int, float)) and durability > 0 else 0.0),
        **{f"armor_{slot.lower()}": float(armor.get("ArmorSlot") == slot) for slot in ARMOR_SLOTS},
        "armor_health": _stat(armor.get("StatModifiers"), "Health") / 100.0,
        "armor_physical": _stat(resistance, "Physical"),
        "armor_projectile": _stat(resistance, "Projectile"),
        "tool_best_power": max(powers, default=0.0),
        "tool_gather_types": len(powers) / 20.0,
    }
    return np.asarray([values[name] for name in ITEM_FEATURES], dtype=np.float32)


def block_row(item: dict) -> np.ndarray:
    """One block's ``BLOCK_FEATURES``, from the item's resolved ``BlockType``."""

    block = {**BLOCK_DEFAULTS, **item["BlockType"]}
    breaking = _breaking(item)
    gather = breaking.get("GatherType")
    interactions = block.get("Interactions") if isinstance(block.get("Interactions"), dict) else {}
    light = block.get("Light") if isinstance(block.get("Light"), dict) else {}
    flags = block.get("Flags") if isinstance(block.get("Flags"), dict) else {}
    values = {
        "breakable": float(isinstance(gather, str) and gather != "Unbreakable"),
        "unbreakable": float(gather == "Unbreakable"),
        "breaking_quality": float(breaking.get("Quality") or 0) / 5.0,
        "drops": float("ItemId" in breaking or "DropList" in breaking),
        "solid": float(block.get("Material") == "Solid"),
        "opaque": float(block.get("Opacity") != "Transparent"),
        "partly_transparent": float(block.get("Opacity") in ("Cutout", "Semitransparent")),
        "cube": float(block.get("DrawType") in ("Cube", "CubeWithModel")),
        "door": float(block.get("IsDoor") is True),
        "stateful": float(bool(block.get("State"))),
        "usable": float("Use" in interactions),
        "collision_effect": float(bool({"Collision", "CollisionEnter"} & set(interactions))),
        "light": _color_brightness(light.get("Color")),
        "needs_support": float(bool(block.get("Support"))),
        "stackable": float(flags.get("IsStackable") is True),
    }
    return np.asarray([values[name] for name in BLOCK_FEATURES], dtype=np.float32)


def description_text(item_id: str, item: dict, language: dict[str, str]) -> str:
    """The words an item is described by: its name, description, categories, subcategory and tags, and its id split
    into words, in a fixed order."""

    parts = [_translated(item, "Name", language), _translated(item, "Description", language)]
    parts += [c.replace(".", " ") for c in item.get("Categories") or [] if isinstance(c, str)]
    if isinstance(item.get("SubCategory"), str):
        parts.append(item["SubCategory"])
    tags = item.get("Tags")
    if isinstance(tags, dict):
        for key in sorted(tags):
            values = tags[key] if isinstance(tags[key], list) else [tags[key]]
            parts.append(f"{key} {' '.join(str(value) for value in values)}")
    parts.append(item_id.replace("_", " "))
    return ". ".join(part for part in parts if part)


def fit_projection(vectors: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """The mean and the top ``width`` principal directions of ``vectors`` float[N, E], each direction's sign fixed so
    its largest component is positive (a deterministic basis)."""

    vectors = vectors.astype(np.float64)
    mean = vectors.mean(axis=0)
    _, _, components = np.linalg.svd(vectors - mean, full_matrices=False)
    basis = components[:width].T
    if basis.shape[1] < width:
        basis = np.concatenate([basis, np.zeros((basis.shape[0], width - basis.shape[1]))], axis=1)
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(width)])
    basis = basis * np.where(signs == 0, 1.0, signs)
    return mean.astype(np.float32), basis.astype(np.float32)


def project(vectors: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return ((vectors - mean) @ basis).astype(np.float32)


def sentence_encoder(model: str = DESCRIPTION_ENCODER) -> Encoder:
    """The pinned local sentence encoder: mean-pooled token states, L2-normalised (the model's own recipe). It loads
    from the local cache only, so a registry build never downloads."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    network = AutoModel.from_pretrained(model, local_files_only=True).eval()

    def encode(texts: Sequence[str]) -> np.ndarray:
        rows = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                batch = tokenizer(list(texts[start:start + 64]), padding=True, truncation=True, max_length=128,
                                  return_tensors="pt")
                states = network(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(states.dtype)
                pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                rows.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
        return np.concatenate(rows).astype(np.float32) if rows else np.zeros((0, 384), dtype=np.float32)

    return encode


def _sha(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.dtype).encode() + str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def build_registry(archive_path: Path, config: RegistryConfig = RegistryConfig(),
                   encoder: Encoder | None = None) -> Registry:
    """Read the archive and build every table. ``encoder`` defaults to the pinned sentence encoder."""

    archive_bytes = Path(archive_path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        items = read_items(archive)
        language = read_language(archive)
    item_ids = tuple(sorted(items))
    gather_types = gather_types_of(items)
    column = {name: index for index, name in enumerate(gather_types)}
    item_features = np.stack([item_row(items[item]) for item in item_ids]) if item_ids else np.zeros(
        (0, len(ITEM_FEATURES)), dtype=np.float32)
    tool_power = np.zeros((len(item_ids), len(gather_types)), dtype=np.float32)
    tool_quality = np.zeros_like(tool_power)
    for row, item in enumerate(item_ids):
        for spec in _tool_specs(items[item]):
            if not spec.get("IsIncorrect"):
                col = column[spec["GatherType"]]
                tool_power[row, col] = max(tool_power[row, col], float(spec.get("Power", 0.0)))
                tool_quality[row, col] = max(tool_quality[row, col], float(spec.get("Quality") or 0))
    block_ids = tuple(item for item in item_ids if isinstance(items[item].get("BlockType"), dict)
                      and items[item]["BlockType"])
    block_features = (np.stack([block_row(items[item]) for item in block_ids]) if block_ids
                      else np.zeros((0, len(BLOCK_FEATURES)), dtype=np.float32))
    block_gather = np.asarray([column.get(_breaking(items[item]).get("GatherType"), -1) for item in block_ids],
                              dtype=np.int32)
    block_quality = np.asarray([float(_breaking(items[item]).get("Quality") or 0) for item in block_ids],
                               dtype=np.float32)

    texts = [description_text(item, items[item], language) for item in item_ids]
    encode = encoder or sentence_encoder(config.encoder)
    embedded = encode(texts)
    mean, basis = fit_projection(embedded, config.description_width)
    item_descriptions = project(embedded, mean, basis)

    manifest = {
        "schema": REGISTRY_SCHEMA,
        "version": config.version,
        "assets_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "encoder": config.encoder if encoder is None else getattr(encoder, "__name__", "injected"),
        "description_width": config.description_width,
        "item_features": list(ITEM_FEATURES),
        "block_features": list(BLOCK_FEATURES),
        "counts": {"items": len(item_ids), "blocks": len(block_ids), "gather_types": len(gather_types),
                   "tools": int((tool_power > 0).any(axis=1).sum())},
        "tables_sha256": _sha(item_features, item_descriptions, tool_power, tool_quality, block_features,
                              block_gather, block_quality, mean, basis),
    }
    return Registry(manifest, item_ids, item_features, item_descriptions, tool_power, tool_quality, block_ids,
                    block_features, block_gather, block_quality, gather_types, mean, basis)


def save_registry(prefix: Path, registry: Registry) -> None:
    arrays = {name: getattr(registry, name) for name in Registry._fields
              if isinstance(getattr(registry, name), np.ndarray)}
    np.savez(Path(prefix).with_suffix(".npz"), **arrays)
    meta = {**registry.manifest, "item_ids": list(registry.item_ids), "block_ids": list(registry.block_ids),
            "gather_types": list(registry.gather_types)}
    Path(prefix).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def load_registry(prefix: Path) -> Registry:
    meta = json.loads(Path(prefix).with_suffix(".json").read_text())
    arrays = np.load(Path(prefix).with_suffix(".npz"))
    names = ("item_ids", "block_ids", "gather_types")
    manifest = {key: value for key, value in meta.items() if key not in names}
    registry = Registry(manifest=manifest, item_ids=tuple(meta["item_ids"]), block_ids=tuple(meta["block_ids"]),
                        gather_types=tuple(meta["gather_types"]),
                        **{name: arrays[name] for name in arrays.files})
    tables = _sha(registry.item_features, registry.item_descriptions, registry.tool_power, registry.tool_quality,
                  registry.block_features, registry.block_gather, registry.block_quality, registry.projection_mean,
                  registry.projection_basis)
    if tables != manifest["tables_sha256"]:
        raise ValueError(f"{prefix}: the tables' SHA-256 differs from the manifest's")
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets", type=Path, required=True, help="the server version's Assets.zip")
    parser.add_argument("--out", type=Path, required=True, help="the output prefix (.npz and .json)")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--description-width", type=int, default=DESCRIPTION_WIDTH)
    args = parser.parse_args(argv)
    registry = build_registry(args.assets, RegistryConfig(version=args.version,
                                                          description_width=args.description_width))
    save_registry(args.out, registry)
    print(json.dumps(registry.manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCK_DEFAULTS",
    "BLOCK_FEATURES",
    "DESCRIPTION_ENCODER",
    "DESCRIPTION_WIDTH",
    "ITEM_FEATURES",
    "REGISTRY_SCHEMA",
    "Registry",
    "RegistryConfig",
    "block_row",
    "build_registry",
    "description_text",
    "fit_projection",
    "gather_types_of",
    "item_row",
    "load_registry",
    "main",
    "project",
    "read_items",
    "read_language",
    "save_registry",
    "sentence_encoder",
]
