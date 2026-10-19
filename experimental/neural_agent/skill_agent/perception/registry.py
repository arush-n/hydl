"""The content registry: every item, block, gather type and creature of a server version's assets as fixed-width
properties, with a description vector per item and per creature (design: ``docs/PERCEPTION.md``; owner rulings,
2026-10-19).

Built offline from the game's own asset archive, so the base game and mods, whose content is asset JSON in the same
codecs, are read by one builder. The adapter looks rows up by asset id at run time; nothing here runs per tick.

- **Items** (``Server/Item/Items/**.json``) are resolved through their ``Parent`` chains (a child's fields override
  its parent's, objects merged key by key, lists replaced) and described by ``ITEM_FEATURES``.
- **Blocks** are the items whose resolved ``BlockType`` is not empty, described by ``BLOCK_FEATURES``, with their
  breaking rule: a gather type and a quality.
- **Gather types** form an open table, the union of what blocks and tools name, so a mod's new gather type is a new
  column, never a rejection. A tool's ``Tool.Specs`` give its power and quality per gather type (``tool_power``,
  ``tool_quality``); a spec marked ``IsIncorrect`` counts as no power.
- **Creatures** are the NPC roles a world can spawn (``Server/NPC/Roles/**.json``, types ``Generic`` and ``Variant``),
  resolved through their references and parameters (``read_roles``) and described by ``CREATURE_FEATURES``: health,
  motion, the size of their model, attack distance, taming and trade.
- **Attitudes**: each creature's attitude toward a player and toward every creature, resolved from the NPC groups and
  the attitude tables by the engine's rules (``npc_groups``, ``attitude_tables``, ``attitude_row``), so a mod's new
  faction needs no code. The rules were read from the 0.7.2 server's classes; each constant below names its class.
- **Descriptions**: each item's name, description, categories, subcategory and tags, and each creature's name, folders
  and category, read through the language file, embedded by a pinned sentence encoder and projected to
  ``description_width`` numbers. Each projection (mean and basis; one for items, one for creatures) is fitted once per
  registry and stored, so content added later (a mod) is projected into the same space.

Saved as ``<prefix>.npz`` (the tables) with ``<prefix>.json`` (the manifest: versions, sizes, names, SHA-256s).
Usage: python -m experimental.neural_agent.skill_agent.perception.registry --assets ASSETS.zip --out PREFIX
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

REGISTRY_SCHEMA = "skill_content_registry_v1"
DEFAULT_VERSION = "0.7.2"
DESCRIPTION_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DESCRIPTION_WIDTH = 64
ITEM_PREFIX = "Server/Item/Items/"
ROLE_PREFIX = "Server/NPC/Roles/"
GROUP_PREFIX = "Server/NPC/Groups/"
ATTITUDE_PREFIX = "Server/NPC/Attitude/Roles/"
MODEL_PREFIX = "Server/Models/"
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

ATTITUDES = ("Ignore", "Hostile", "Neutral", "Friendly", "Revered")
"""The engine's attitudes in its own order (``server.core.asset.type.attitude.Attitude``). An attitude table sets them
in this order and a later one overwrites an earlier one for the same role (``AttitudeMap$Builder.putGroups``), so where
a role is in two of a table's groups the later attitude wins."""
DEFAULT_PLAYER_ATTITUDE = "Hostile"
DEFAULT_NPC_ATTITUDE = "Neutral"
"""A role's attitudes toward players and toward other NPCs when it names none (``SupportConfigBuilder``)."""
PLAYER_ROLE = "$player"
SELF_ROLE = "$self"
"""How the engine's groups hold players and an NPC's own role (``BuilderManager``; the ``Player`` and ``Self``
groups)."""
SHOP_ACTIONS = frozenset({"OpenBarterShop", "OpenShop"})
"""The engine's actions that open a shop (``builtin.adventure.npcshop``)."""
MOTION_SPEEDS = {"Walk": ("MaxWalkSpeed", 3.0), "Fly": ("MaxHorizontalSpeed", 8.0), "Dive": ("MaxSwimSpeed", 3.0)}
"""Each motion controller's top-speed field and the engine's default for it (``BuilderMotionControllerWalk``, ``Fly``,
``Dive``)."""
CREATURE_FEATURES = (
    "non_gameplay", "max_health", "walks", "flies", "swims", "walk_speed", "fly_speed", "swim_speed", "height",
    "width", "eye_height", "attack_distance", "tameable", "trades",
)


class Registry(NamedTuple):
    """The tables of one server version. Rows of ``item_*`` follow ``item_ids``; rows of ``block_*`` follow
    ``block_ids``; columns of ``tool_*`` follow ``gather_types``; rows of ``creature_*`` and ``attitude_*``, and the
    columns of ``attitude_to_creature``, follow ``creature_ids``."""

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
    projection_mean: np.ndarray  # float32[E]: the items' description projection, for content added later
    projection_basis: np.ndarray  # float32[E, description_width]
    creature_ids: tuple[str, ...]
    creature_features: np.ndarray  # float32[C, len(CREATURE_FEATURES)]
    creature_descriptions: np.ndarray  # float32[C, description_width]
    attitude_to_player: np.ndarray  # int8[C]: each creature's attitude toward a player, an index into ATTITUDES
    attitude_to_creature: np.ndarray  # int8[C, C]: row r's attitude toward a creature of column s's role
    creature_projection_mean: np.ndarray  # float32[E]: the creatures' description projection
    creature_projection_basis: np.ndarray  # float32[E, description_width]


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


def _documents(archive: zipfile.ZipFile, prefix: str) -> dict[str, tuple[str, dict]]:
    """Every JSON object under ``prefix`` by id (its file name), with its path. An unreadable file is skipped; an id
    found twice keeps its first path in sorted order."""

    documents: dict[str, tuple[str, dict]] = {}
    for name in sorted(archive.namelist()):
        if not (name.startswith(prefix) and name.endswith(".json")):
            continue
        try:
            value = json.loads(archive.read(name))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        stem = name.rsplit("/", 1)[1][:-len(".json")]
        if isinstance(value, dict) and stem not in documents:
            documents[stem] = (name, value)
    return documents


def _resolve_parents(raw: dict[str, dict]) -> dict[str, dict]:
    """Each document merged over its ``Parent`` chain, by id in sorted order. A parent that is missing or already in
    the chain ends the chain."""

    resolved: dict[str, dict] = {}

    def resolve(name: str, chain: tuple[str, ...]) -> dict:
        if name in resolved:
            return resolved[name]
        value = raw[name]
        parent = value.get("Parent")
        if isinstance(parent, str) and parent in raw and parent not in chain:
            value = _merge(resolve(parent, chain + (name,)), value)
        resolved[name] = value
        return value

    return {name: resolve(name, ()) for name in sorted(raw)}


def read_items(archive: zipfile.ZipFile) -> dict[str, dict]:
    """Every item under ``Server/Item/Items/`` by id (its file name), each resolved through its ``Parent`` chain."""

    return _resolve_parents({item: value for item, (_, value) in _documents(archive, ITEM_PREFIX).items()})


def read_models(archive: zipfile.ZipFile) -> dict[str, dict]:
    """Every model under ``Server/Models/`` by id, each resolved through its ``Parent`` chain."""

    return _resolve_parents({model: value for model, (_, value) in _documents(archive, MODEL_PREFIX).items()})


def _parameter_values(document: dict) -> dict:
    parameters = document.get("Parameters")
    if not isinstance(parameters, dict):
        return {}
    return {name: spec.get("Value") if isinstance(spec, dict) else spec for name, spec in parameters.items()}


def _bare(value: Any, scope: dict) -> Any:
    """``value`` with each ``{"Compute": name}`` that names a parameter of ``scope`` replaced by its value. Any other
    expression stays as it is, and reading it as a property fails."""

    if isinstance(value, dict):
        if set(value) == {"Compute"} and isinstance(value["Compute"], str) and value["Compute"].strip() in scope:
            return scope[value["Compute"].strip()]
        return {key: _bare(child, scope) for key, child in value.items()}
    if isinstance(value, list):
        return [_bare(child, scope) for child in value]
    return value


def read_roles(archive: zipfile.ZipFile) -> tuple[dict[str, dict], dict[str, dict]]:
    """The NPC roles a world can spawn (``Generic`` and ``Variant``) by id, each as its file ``path``, its resolved
    top-level ``fields`` and its ``parameters``; and the ``Component`` roles (instruction fragments) by id.

    A ``Variant`` starts from its ``Reference``, resolved first: the reference's parameters, then the variant's own,
    then each ``Modify`` entry, which sets the parameter it names (computed against the parameters before any
    modification) or, with a leading ``_``, replaces that top-level field. The fields' ``{"Compute": name}`` then take
    the final parameter values. These are the rules of the repo's extractor (``hytalegym/tools/content_extract/
    npc_roles.py``), which wrote the gen-7 content table. Computing each level as it is read gives what scoped
    evaluation gives while every modification passes on a parameter of the same name, which every gameplay role of
    0.7.2 does; a field holding any other expression is refused when a property reads it.
    """

    documents = _documents(archive, ROLE_PREFIX)
    cache: dict[str, tuple[dict, dict]] = {}

    def resolve(stem: str, chain: tuple[str, ...]) -> tuple[dict, dict]:
        if stem in cache:
            return cache[stem]
        if stem in chain:
            raise ValueError(f"role reference cycle: {' -> '.join(chain + (stem,))}")
        document = documents[stem][1]
        if document.get("Type") == "Variant" and isinstance(document.get("Reference"), str):
            fields, parameters = resolve(document["Reference"], chain + (stem,))
            fields, parameters = dict(fields), {**parameters, **_parameter_values(document)}
            scope = dict(parameters)
            for key, value in (document.get("Modify") or {}).items():
                if key.startswith("_"):
                    fields[key[1:]] = value
                else:
                    parameters[key] = _bare(value, scope)
        else:
            fields = {key: value for key, value in document.items() if key != "Parameters"}
            parameters = _parameter_values(document)
        cache[stem] = fields, parameters
        return fields, parameters

    roles: dict[str, dict] = {}
    for stem, (path, document) in documents.items():
        if document.get("Type") in ("Generic", "Variant"):
            try:
                fields, parameters = resolve(stem, ())
            except (KeyError, ValueError) as error:
                raise ValueError(f"role {stem}: {error!r}") from error
            roles[stem] = {"path": path, "parameters": parameters,
                           "fields": {key: _bare(value, parameters) for key, value in fields.items()
                                      if key not in ("Type", "Reference", "Modify")}}
    components = {stem: document for stem, (_, document) in documents.items() if document.get("Type") == "Component"}
    return roles, components


def _glob(pattern: str) -> re.Pattern:
    """The engine's glob (``StringUtil.isGlobMatching``): ``*`` any run of characters, ``?`` one, case-sensitive."""

    return re.compile("".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern), re.DOTALL)


def npc_groups(archive: zipfile.ZipFile, names: Iterable[str]) -> dict[str, frozenset[str]]:
    """Each NPC group (``Server/NPC/Groups``) as the names in it, out of ``names`` (the role ids, with ``$player`` and
    ``$self``). A group adds the members of its ``IncludeGroups``, removes those of its ``ExcludeGroups``, adds the
    names its ``IncludeRoles`` match and removes the names its ``ExcludeRoles`` match, in that order
    (``TagSetLookupTable.createTagSet``); a pattern without ``*`` or ``?`` names one id."""

    documents = {group: value for group, (_, value) in _documents(archive, GROUP_PREFIX).items()}
    names = frozenset(names)
    members: dict[str, frozenset[str]] = {}

    def matching(patterns: Any) -> set[str]:
        found: set[str] = set()
        for pattern in patterns or []:
            if "*" in pattern or "?" in pattern:
                rule = _glob(pattern)
                found.update(name for name in names if rule.fullmatch(name))
            elif pattern in names:
                found.add(pattern)
        return found

    def resolve(group: str, chain: tuple[str, ...]) -> frozenset[str]:
        if group in members:
            return members[group]
        if group in chain:
            raise ValueError(f"NPC group cycle: {' -> '.join(chain + (group,))}")
        document = documents[group]
        found: set[str] = set()
        for other in document.get("IncludeGroups") or []:
            found |= resolve(other, chain + (group,))
        for other in document.get("ExcludeGroups") or []:
            found -= resolve(other, chain + (group,))
        found |= matching(document.get("IncludeRoles"))
        found -= matching(document.get("ExcludeRoles"))
        members[group] = frozenset(found)
        return members[group]

    return {group: resolve(group, ()) for group in sorted(documents)}


def attitude_tables(archive: zipfile.ZipFile, groups: dict[str, frozenset[str]]) -> dict[str, dict[str, int]]:
    """Each attitude table (``Server/NPC/Attitude/Roles``) as name to attitude (an index into ``ATTITUDES``): the
    table's attitudes in the engine's order, each over the members of the groups it lists, a later one overwriting.
    A group that does not exist is skipped, as the engine skips it."""

    tables: dict[str, dict[str, int]] = {}
    for table_id, (_, document) in _documents(archive, ATTITUDE_PREFIX).items():
        listed = document.get("Groups") if isinstance(document.get("Groups"), dict) else {}
        if set(listed) - set(ATTITUDES):
            raise ValueError(f"attitude table {table_id}: unknown attitudes {sorted(set(listed) - set(ATTITUDES))}")
        table: dict[str, int] = {}
        for code, attitude in enumerate(ATTITUDES):
            for group in listed.get(attitude) or []:
                table.update(dict.fromkeys(groups.get(group, ()), code))
        tables[table_id] = table
    return tables


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


def _reaches(value: Any, components: dict[str, dict], types: frozenset[str]) -> bool:
    """Whether ``value``, or a component it names (followed recursively), holds an object whose ``Type`` is one of
    ``types``."""

    seen: set[str] = set()
    pending = [value]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if isinstance(node.get("Type"), str) and node["Type"] in types:
                return True
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, str) and node in components and node not in seen:
            seen.add(node)
            pending.append(components[node])
    return False


def creature_row(role: dict, models: dict[str, dict], components: dict[str, dict]) -> np.ndarray:
    """One creature's ``CREATURE_FEATURES``. Its size is its model's hitbox as declared, before the spawn scale (most
    models 1.0; some 0.75 to 1.45); its eye height is 0 when the model declares none; its attack distance is 0 when
    its role declares none."""

    fields, parameters = role["fields"], role["parameters"]
    model = models.get(fields.get("Appearance"), {})
    box = model.get("HitBox") if isinstance(model.get("HitBox"), dict) else {}
    low, high = box.get("Min") or {}, box.get("Max") or {}
    extent = {axis: float(high.get(axis, 0.0)) - float(low.get(axis, 0.0)) for axis in "XYZ"}
    speed: dict[str, float] = {}
    for controller in fields.get("MotionControllerList") or []:
        if controller.get("Type") in MOTION_SPEEDS:
            key, default = MOTION_SPEEDS[controller["Type"]]
            speed[controller["Type"]] = max(speed.get(controller["Type"], 0.0), float(controller.get(key, default)))
    attack = parameters.get("AttackDistance")
    values = {
        "non_gameplay": float(role["path"].startswith(ROLE_PREFIX + "_Core/")),
        "max_health": math.log10(float(fields["MaxHealth"])) / 4.0,
        "walks": float("Walk" in speed),
        "flies": float("Fly" in speed),
        "swims": float("Dive" in speed),
        "walk_speed": speed.get("Walk", 0.0) / 10.0,
        "fly_speed": speed.get("Fly", 0.0) / 10.0,
        "swim_speed": speed.get("Dive", 0.0) / 10.0,
        "height": extent["Y"] / 4.0,
        "width": max(extent["X"], extent["Z"]) / 4.0,
        "eye_height": float(model.get("EyeHeight") or 0.0) / 4.0,
        "attack_distance": float(attack) / 16.0 if type(attack) in (int, float) else 0.0,
        "tameable": float(parameters.get("IsTameable") is True),
        "trades": float(_reaches(fields, components, SHOP_ACTIONS)),
    }
    return np.asarray([values[name] for name in CREATURE_FEATURES], dtype=np.float32)


def attitude_row(role_id: str, role: dict, tables: dict[str, dict[str, int]],
                 ids: Sequence[str]) -> tuple[int, list[int]]:
    """One creature's attitude toward a player and toward a creature of each role of ``ids``, as the engine resolves
    it from content (``AttitudeView``, ``AttitudeMap.getAttitude``): the table of the creature's ``AttitudeGroup``,
    looked up by the target's role, by ``$player`` for a player, or by ``$self`` for a target of its own role; else
    its default attitude toward players or toward NPCs. The engine's two earlier providers, an attitude overridden at
    run time and friendship between NPCs spawned together, are not content and are left to the adapter."""

    fields = role["fields"]
    toward_player = ATTITUDES.index(fields.get("DefaultPlayerAttitude", DEFAULT_PLAYER_ATTITUDE))
    toward_npc = ATTITUDES.index(fields.get("DefaultNPCAttitude", DEFAULT_NPC_ATTITUDE))
    group = fields.get("AttitudeGroup")
    table = tables[group] if group is not None else {}
    return (table.get(PLAYER_ROLE, toward_player),
            [table.get(SELF_ROLE if other == role_id else other, toward_npc) for other in ids])


def creature_text(role_id: str, role: dict, language: dict[str, str]) -> str:
    """The words a creature is described by: its name, the folders its role file sits in, its memories category, and
    its id split into words, in a fixed order."""

    fields = role["fields"]
    key = fields.get("NameTranslationKey")
    name = language.get(key.removeprefix("server."), "") if isinstance(key, str) else ""
    relative = role["path"][len(ROLE_PREFIX):]
    folders = relative.rsplit("/", 1)[0].replace("/", " ").replace("_", " ") if "/" in relative else ""
    category = fields.get("MemoriesCategory") if isinstance(fields.get("MemoriesCategory"), str) else ""
    parts = [name, folders, category, role_id.replace("_", " ")]
    return ". ".join(part for part in parts if part)


def fit_projection(vectors: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """The mean and the top ``width`` principal directions of ``vectors`` float[N, E], each direction's sign fixed so
    its largest component is positive (a deterministic basis). With no vectors, both are zero."""

    if len(vectors) == 0:
        return np.zeros(vectors.shape[1], np.float32), np.zeros((vectors.shape[1], width), np.float32)
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


def _tables_sha(registry: Registry) -> str:
    return _sha(*(value for value in registry if isinstance(value, np.ndarray)))


def build_registry(archive_path: Path, config: RegistryConfig = RegistryConfig(),
                   encoder: Encoder | None = None) -> Registry:
    """Read the archive and build every table. ``encoder`` defaults to the pinned sentence encoder."""

    archive_bytes = Path(archive_path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        items = read_items(archive)
        language = read_language(archive)
        roles, components = read_roles(archive)
        models = read_models(archive)
        groups = npc_groups(archive, (*roles, PLAYER_ROLE, SELF_ROLE))
        tables = attitude_tables(archive, groups)
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

    creature_ids = tuple(sorted(roles))
    rows, to_player, to_creature = [], [], []
    for role_id in creature_ids:
        try:
            rows.append(creature_row(roles[role_id], models, components))
            player, creatures = attitude_row(role_id, roles[role_id], tables, creature_ids)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"role {role_id}: {error!r}") from error
        to_player.append(player)
        to_creature.append(creatures)
    creature_features = np.stack(rows) if rows else np.zeros((0, len(CREATURE_FEATURES)), dtype=np.float32)
    attitude_to_player = np.asarray(to_player, dtype=np.int8)
    attitude_to_creature = np.asarray(to_creature, dtype=np.int8).reshape(len(creature_ids), len(creature_ids))
    creature_embedded = encode([creature_text(role_id, roles[role_id], language) for role_id in creature_ids])
    creature_mean, creature_basis = fit_projection(creature_embedded, config.description_width)
    creature_descriptions = project(creature_embedded, creature_mean, creature_basis)

    registry = Registry({}, item_ids, item_features, item_descriptions, tool_power, tool_quality, block_ids,
                        block_features, block_gather, block_quality, gather_types, mean, basis, creature_ids,
                        creature_features, creature_descriptions, attitude_to_player, attitude_to_creature,
                        creature_mean, creature_basis)
    manifest = {
        "schema": REGISTRY_SCHEMA,
        "version": config.version,
        "assets_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "encoder": config.encoder if encoder is None else getattr(encoder, "__name__", "injected"),
        "description_width": config.description_width,
        "item_features": list(ITEM_FEATURES),
        "block_features": list(BLOCK_FEATURES),
        "creature_features": list(CREATURE_FEATURES),
        "attitudes": list(ATTITUDES),
        "counts": {"items": len(item_ids), "blocks": len(block_ids), "gather_types": len(gather_types),
                   "tools": int((tool_power > 0).any(axis=1).sum()), "creatures": len(creature_ids),
                   "gameplay_creatures": int((creature_features[:, 0] == 0).sum()), "npc_groups": len(groups),
                   "attitude_tables": len(tables)},
        "tables_sha256": _tables_sha(registry),
    }
    return registry._replace(manifest=manifest)


def save_registry(prefix: Path, registry: Registry) -> None:
    arrays = {name: getattr(registry, name) for name in Registry._fields
              if isinstance(getattr(registry, name), np.ndarray)}
    np.savez(Path(prefix).with_suffix(".npz"), **arrays)
    meta = {**registry.manifest, "item_ids": list(registry.item_ids), "block_ids": list(registry.block_ids),
            "gather_types": list(registry.gather_types), "creature_ids": list(registry.creature_ids)}
    Path(prefix).with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def load_registry(prefix: Path) -> Registry:
    meta = json.loads(Path(prefix).with_suffix(".json").read_text())
    arrays = np.load(Path(prefix).with_suffix(".npz"))
    names = ("item_ids", "block_ids", "gather_types", "creature_ids")
    manifest = {key: value for key, value in meta.items() if key not in names}
    registry = Registry(manifest=manifest, **{name: tuple(meta[name]) for name in names},
                        **{name: arrays[name] for name in arrays.files})
    if _tables_sha(registry) != manifest["tables_sha256"]:
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
    "ATTITUDES",
    "BLOCK_DEFAULTS",
    "BLOCK_FEATURES",
    "CREATURE_FEATURES",
    "DESCRIPTION_ENCODER",
    "DESCRIPTION_WIDTH",
    "ITEM_FEATURES",
    "REGISTRY_SCHEMA",
    "Registry",
    "RegistryConfig",
    "attitude_row",
    "attitude_tables",
    "block_row",
    "build_registry",
    "creature_row",
    "creature_text",
    "description_text",
    "fit_projection",
    "gather_types_of",
    "item_row",
    "load_registry",
    "main",
    "npc_groups",
    "project",
    "read_items",
    "read_language",
    "read_models",
    "read_roles",
    "save_registry",
    "sentence_encoder",
]
