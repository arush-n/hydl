"""Weapon attack tables: items, damage per attack, knockback, hold, chains.

Items are the combat catalog's selection (``combat/assets/catalog.py``): every
non-template item under ``Server/Item/Items/Weapon/`` plus any item whose
resolved ``Categories`` or ``Tags.Type`` marks it a weapon. For each item the
extractor walks the interaction graph the way the catalog's ``_GraphVisitor``
does (roots, ``Parent``, ``Next``/``Failed``/``Interactions``/``DefaultValue``,
``Replace`` variables) and additionally follows ``Config``/``Projectile`` into
``Server/ProjectileConfigs`` (arrow, bolt and rune projectile damage), the
directional branches of ``MovementCondition`` (dodge) and a rune's
``Ability.Cast`` root.

Rows:

* ``item:<id>``: durability, guard stamina, the rune ``Ability`` block (cost,
  cooldown, ``Cast`` root, which the walk follows), and one damage entry per
  ``Replace`` variable whose value deals damage: the item's own
  ``InteractionVars`` override (``origin: item``) or the graph's
  ``DefaultValue`` (``origin: default``). ``base_first``/``angled_first`` are
  the first authored amounts, the rule of
  ``tests/combat/assets/weapon_variants/test_asset_scalar_variants.py``.
* ``interaction:<id>`` / ``root:<id>`` / ``projectile_config:<id>``: every
  graph file reached from any weapon, Parent-resolved and cosmetics-stripped,
  with derived fields: ``hit_interrupt`` (parent chain contains
  ``DamageEntityParent_Hit_Interrupt``, new in 0.7.2), ``knockback``,
  ``damage_class``, ``base_damage``, ``selector``, ``charge_thresholds``,
  ``chain``.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from .archive import GameArchive
from .common import gameplay, is_number, merge_inherited, resolve_parented

FAMILY = "weapons"
VERSION = 1
ITEM_PREFIX = "Server/Item/Items/"
WEAPON_PREFIX = "Server/Item/Items/Weapon/"
ROOT_PREFIX = "Server/Item/RootInteractions/"
INTERACTION_PREFIX = "Server/Item/Interactions/"
PROJECTILE_CONFIG_PREFIX = "Server/ProjectileConfigs/"
EFFECT_PREFIX = "Server/Entity/Effects/"
GRAPH_REFERENCE_KEYS = frozenset({"DefaultValue", "Failed", "Interactions", "Next", "Parent"})
# ``MovementCondition`` (the dodge root) names its branches by travel direction.
MOVEMENT_CONDITION_KEYS = frozenset(
    {"Forward", "Back", "Left", "Right", "ForwardLeft", "ForwardRight", "BackLeft", "BackRight"}
)
HIT_INTERRUPT_PARENT = "DamageEntityParent_Hit_Interrupt"
# Non-weapon items whose graphs are combat or traversal abilities (0.7.2 adds
# runes, the hookshot and a scrap glider); selected by folder or id.
ABILITY_ITEM_FOLDERS = ("Server/Item/Items/Rune/", "Server/Item/Items/Glider/")
ABILITY_ITEM_TOKENS = ("Hookshot",)
# Player movement abilities that are root interactions, not items.
ABILITY_ROOT_PREFIXES = ("Dodge",)
_EFFECT_KEYS = ("EffectId", "EntityEffectId")


def is_semantic_weapon(item: dict[str, Any]) -> bool:
    """Same predicate as ``combat/assets/catalog.py::_is_semantic_weapon``."""

    categories = item.get("Categories", ())
    tags = item.get("Tags", {})
    tag_types = tags.get("Type", ()) if isinstance(tags, dict) else ()
    return (isinstance(categories, list) and "Items.Weapons" in categories) or (
        isinstance(tag_types, list) and "Weapon" in tag_types
    )


class _Graph:
    def __init__(self, archive: GameArchive):
        self.archive = archive
        self.items = archive.index(ITEM_PREFIX)
        self.roots = archive.index(ROOT_PREFIX)
        self.interactions = archive.index(INTERACTION_PREFIX)
        self.projectile_configs = archive.index(PROJECTILE_CONFIG_PREFIX)
        self._resolved: dict[tuple[str, str], tuple[dict[str, Any], tuple[str, ...]]] = {}

    def resolve(self, namespace: str, stem: str) -> tuple[dict[str, Any], tuple[str, ...]]:
        key = (namespace, stem)
        if key not in self._resolved:
            index = {
                "item": self.items,
                "root": self.roots,
                "interaction": self.interactions,
                "projectile_config": self.projectile_configs,
            }[namespace]
            if namespace == "root":
                document = self.archive.read_json(index[stem])
                self._resolved[key] = (document, ())
            else:
                self._resolved[key] = resolve_parented(stem, index, self.archive.read_json)
        return self._resolved[key]

    def path(self, namespace: str, stem: str) -> str:
        return {
            "item": self.items,
            "root": self.roots,
            "interaction": self.interactions,
            "projectile_config": self.projectile_configs,
        }[namespace][stem]


class _Visitor:
    """Reach every named graph document and every ``Replace`` default."""

    def __init__(self, graph: _Graph, variables: dict[str, Any]):
        self.graph = graph
        self.variables = variables
        self.named: dict[str, tuple[str, str]] = {}
        self.replace_defaults: dict[str, Any] = {}
        self.projectile_ids: set[str] = set()

    def visit(self, value: Any, key: str = "", *, prefer_root: bool = False, allow_graph: bool = False) -> None:
        if isinstance(value, str):
            if not allow_graph:
                return
            order = ("root", "interaction") if prefer_root else ("interaction", "root")
            for namespace in order:
                index = self.graph.roots if namespace == "root" else self.graph.interactions
                if value in index:
                    self._visit_named(namespace, value)
                    return
            return
        if isinstance(value, list):
            for child in value:
                self.visit(child, key, prefer_root=prefer_root, allow_graph=allow_graph)
            return
        if not isinstance(value, dict):
            return
        kind = value.get("Type")
        parent = value.get("Parent")
        if isinstance(parent, str):
            self.visit(parent, "Parent", allow_graph=True)
        replace = kind == "Replace"
        variable = value.get("Var")
        if replace and isinstance(variable, str):
            self.replace_defaults.setdefault(variable, value.get("DefaultValue"))
            self.visit(
                self.variables.get(variable, value.get("DefaultValue")),
                "DefaultValue",
                allow_graph=True,
            )
        # ``Config`` (Projectile interactions) and ``Projectile`` (0.7.2
        # LaunchAbilityProjectile) name a Server/ProjectileConfigs asset.
        for key in ("Config", "Projectile"):
            config = value.get(key)
            if isinstance(config, str) and config in self.graph.projectile_configs:
                self._visit_named("projectile_config", config)
        projectile = value.get("ProjectileId")
        if isinstance(projectile, str):
            self.projectile_ids.add(projectile)
        reference_container = allow_graph and kind is None
        movement = kind == "MovementCondition"
        for child_key, child in value.items():
            if child_key == "Parent":
                continue
            if replace and child_key in {"DefaultValue", "Var"}:
                continue
            self.visit(
                child,
                child_key,
                prefer_root=prefer_root,
                allow_graph=(
                    child_key in GRAPH_REFERENCE_KEYS
                    or reference_container
                    or (movement and child_key in MOVEMENT_CONDITION_KEYS)
                ),
            )

    def _visit_named(self, namespace: str, stem: str) -> None:
        identity = f"{namespace}:{stem}"
        if identity in self.named:
            return
        self.named[identity] = (namespace, stem)
        if namespace == "projectile_config":
            document, _ = self.graph.resolve(namespace, stem)
        else:
            document = self.graph.archive.read_json(self.graph.path(namespace, stem))
        self.visit(document)


def effect_ids(value: Any, found: set[str]) -> None:
    """Effect ids named by ``EffectId``/``EntityEffectId``/``EntityEffectIds``."""

    if isinstance(value, list):
        for child in value:
            effect_ids(child, found)
        return
    if not isinstance(value, dict):
        return
    for key in _EFFECT_KEYS:
        if isinstance(value.get(key), str):
            found.add(value[key])
    listed = value.get("EntityEffectIds")
    if isinstance(listed, list):
        found.update(entry for entry in listed if isinstance(entry, str))
    for child in value.values():
        effect_ids(child, found)


def _resolve_embedded(graph: _Graph, value: Any) -> Any:
    """Resolve interaction ``Parent`` references inside a variable value.

    String references are not followed (graphs are cyclic); this is the rule
    of the scalar-variant asset test's ``_resolve_embedded_interactions``.
    """

    if isinstance(value, list):
        return [_resolve_embedded(graph, child) for child in value]
    if not isinstance(value, dict):
        return value
    parent = value.get("Parent")
    result = (
        merge_inherited(graph.resolve("interaction", parent)[0], value)
        if isinstance(parent, str) and parent in graph.interactions
        else dict(value)
    )
    return {key: _resolve_embedded(graph, child) for key, child in result.items()}


def _resolve_default(graph: _Graph, value: Any) -> Any:
    """A ``Replace`` default: its top-level interaction ids resolved once."""

    if isinstance(value, dict) and isinstance(value.get("Interactions"), list):
        value = dict(value)
        value["Interactions"] = [
            graph.resolve("interaction", entry)[0]
            if isinstance(entry, str) and entry in graph.interactions
            else entry
            for entry in value["Interactions"]
        ]
    return _resolve_embedded(graph, value)


def _first_base_damage(value: Any, *, angled: bool) -> tuple[dict[str, Any] | None, float | None]:
    """First ``BaseDamage`` map (and its first amount) in authored order."""

    found: list[tuple[dict[str, Any], float]] = []

    def visit(node: Any, in_angled: bool = False) -> None:
        if found:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                if key == "AngledDamage":
                    visit(child, True)
                elif key == "BaseDamage" and isinstance(child, dict):
                    if in_angled == angled:
                        amounts = [float(a) for a in child.values() if is_number(a)]
                        if amounts:
                            found.append((child, amounts[0]))
                            return
                else:
                    visit(child, in_angled)
        elif isinstance(node, list):
            for child in node:
                visit(child, in_angled)

    visit(value)
    if not found:
        return None, None
    return dict(found[0][0]), found[0][1]


def _first_stamina_cost(value: Any) -> float | None:
    found: list[float] = []

    def visit(node: Any) -> None:
        if found:
            return
        if isinstance(node, dict):
            stamina = node.get("StaminaCost")
            if isinstance(stamina, dict) and is_number(stamina.get("Value")):
                found.append(float(stamina["Value"]))
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return found[0] if found else None


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    return [float(value.get(axis, 0.0)) for axis in ("X", "Y", "Z")]


def _derived(document: dict[str, Any], chain: tuple[str, ...]) -> dict[str, Any]:
    kind = document.get("Type")
    derived: dict[str, Any] = {"type": kind, "parent_chain": list(chain)}
    if kind == "DamageEntity" or "DamageCalculator" in document:
        calculator = document.get("DamageCalculator", {})
        effects = document.get("DamageEffects", {}) or {}
        knockback = effects.get("Knockback") if isinstance(effects, dict) else None
        derived.update(
            {
                "damage_class": calculator.get("Class") if isinstance(calculator, dict) else None,
                "base_damage": calculator.get("BaseDamage") if isinstance(calculator, dict) else None,
                "hit_interrupt": HIT_INTERRUPT_PARENT in chain,
                "knockback": None
                if not isinstance(knockback, dict)
                else {
                    "type": knockback.get("Type"),
                    "force": knockback.get("Force"),
                    "direction": _vector(knockback.get("Direction")),
                    "velocity_type": knockback.get("VelocityType"),
                    "velocity_y": knockback.get("VelocityY"),
                    "relative_x": knockback.get("RelativeX"),
                    "relative_z": knockback.get("RelativeZ"),
                    "duration": knockback.get("Duration"),
                    "velocity_config": knockback.get("VelocityConfig"),
                },
                "entity_stats_on_hit": document.get("EntityStatsOnHit"),
            }
        )
    if kind == "Selector" and isinstance(document.get("Selector"), dict):
        derived["selector"] = document["Selector"]
    if kind == "Charging" and isinstance(document.get("Next"), dict):
        thresholds = []
        for key in document["Next"]:
            try:
                thresholds.append(float(key))
            except ValueError:
                pass
        derived["charge_thresholds"] = sorted(thresholds)
    if kind == "Chaining":
        following = document.get("Next")
        derived["chain"] = {
            "chaining_allowance": document.get("ChainingAllowance"),
            "chain_id": document.get("ChainId"),
            "length": len(following) if isinstance(following, list) else None,
        }
    return derived


def extract(archive: GameArchive) -> dict[str, Any]:
    graph = _Graph(archive)
    selected: set[str] = set()
    for asset_id, path in graph.items.items():
        if asset_id.startswith("Template_"):
            continue
        if path.startswith(WEAPON_PREFIX):
            selected.add(asset_id)
            continue
        item, _ = graph.resolve("item", asset_id)
        if is_semantic_weapon(item):
            selected.add(asset_id)

    for asset_id, path in graph.items.items():
        if asset_id.startswith("Template_"):
            continue
        if path.startswith(ABILITY_ITEM_FOLDERS) or any(token in asset_id for token in ABILITY_ITEM_TOKENS):
            selected.add(asset_id)

    rows: dict[str, Any] = {}
    reached: dict[str, tuple[str, str]] = {}
    effects_named: set[str] = set()
    for stem in sorted(graph.roots):
        if stem.startswith(ABILITY_ROOT_PREFIXES):
            visitor = _Visitor(graph, {})
            visitor.visit(stem, "Interactions", prefer_root=True, allow_graph=True)
            reached.update(visitor.named)
    for asset_id in sorted(selected):
        item, chain = graph.resolve("item", asset_id)
        variables = item.get("InteractionVars", {})
        if not isinstance(variables, dict):
            variables = {}
        visitor = _Visitor(graph, variables)
        equipped = item.get("Interactions", {})
        visitor.visit(equipped, "Interactions", prefer_root=True, allow_graph=True)
        # 0.7.2 rune items: ``Ability.Cast`` names the cast's root interaction.
        ability = item.get("Ability") if isinstance(item.get("Ability"), dict) else None
        if ability is not None and isinstance(ability.get("Cast"), str):
            visitor.visit(ability["Cast"], "Interactions", prefer_root=True, allow_graph=True)
        reached.update(visitor.named)
        effect_ids(variables, effects_named)
        damage: dict[str, Any] = {}
        for name in sorted(set(variables) | set(visitor.replace_defaults)):
            origin = "item" if name in variables else "default"
            if origin == "item":
                resolved = _resolve_embedded(graph, variables[name])
            else:
                resolved = _resolve_default(graph, visitor.replace_defaults[name])
            base, base_first = _first_base_damage(resolved, angled=False)
            angled, angled_first = _first_base_damage(resolved, angled=True)
            if base is None and angled is None:
                continue
            damage[name] = {
                "origin": origin,
                "base": base,
                "base_first": base_first,
                "angled": angled,
                "angled_first": angled_first,
            }
        var_projectiles: dict[str, str] = {}
        for name, value in sorted(variables.items()):
            found: list[str] = []

            def collect(node: Any) -> None:
                if isinstance(node, dict):
                    if isinstance(node.get("ProjectileId"), str):
                        found.append(node["ProjectileId"])
                    for child in node.values():
                        collect(child)
                elif isinstance(node, list):
                    for child in node:
                        collect(child)

            collect(_resolve_embedded(graph, value))
            if found:
                var_projectiles[name] = found[0]
        path = graph.items[asset_id]
        folder = (
            PurePosixPath(path).parts[4] if path.startswith(WEAPON_PREFIX) and len(PurePosixPath(path).parts) > 5 else None
        )
        rows[f"item:{asset_id}"] = {
            "kind": "item",
            "source": path,
            "parent_chain": list(chain),
            "folder": folder,
            "max_stack": item.get("MaxStack"),
            "max_durability": item.get("MaxDurability"),
            "durability_loss_on_hit": item.get("DurabilityLossOnHit"),
            "guard_stamina_value": _first_stamina_cost(variables.get("Guard_Wield", {})),
            "interactions": {
                key: value for key, value in sorted(equipped.items()) if isinstance(value, str)
            }
            if isinstance(equipped, dict)
            else {},
            "interaction_vars": sorted(variables),
            "ability": gameplay(ability) if ability is not None else None,
            "damage": damage,
            "projectile_ids": sorted(visitor.projectile_ids),
            # The ProjectileId each item variable resolves to (its own override
            # wins over the parent interaction's default).
            "var_projectile_ids": var_projectiles,
        }

    for identity, (namespace, stem) in sorted(reached.items()):
        document, chain = graph.resolve(namespace, stem)
        effect_ids(document, effects_named)
        row = {
            "kind": namespace,
            "source": graph.path(namespace, stem),
            **_derived(document, chain),
            "document": gameplay(document),
        }
        rows[identity] = row
    effects = archive.index(EFFECT_PREFIX)
    for effect_id in sorted(effects_named):
        if effect_id not in effects:
            continue
        document, chain = resolve_parented(effect_id, effects, archive.read_json)
        rows[f"effect:{effect_id}"] = {
            "kind": "effect",
            "source": effects[effect_id],
            "parent_chain": list(chain),
            "duration": document.get("Duration"),
            "document": gameplay(
                {key: value for key, value in document.items() if key != "StatusEffectIcon"}
            ),
        }
    return rows
