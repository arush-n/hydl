"""NPC roles and motion parameters (``Server/NPC/Roles/**``).

Role files are ``Abstract`` templates, ``Generic`` roles, ``Component``
instruction fragments, and ``Variant`` roles (``Reference`` + ``Modify`` +
``Parameters``). Resolution, top level only:

1. A ``Variant`` resolves its ``Reference`` first (recursively).
2. Parameters: the reference's ``Parameters`` values, then the variant's own
   ``Parameters``, then ``Modify`` entries naming a parameter. A ``Modify``
   key with a leading ``_`` overrides that top-level field instead
   (``_InteractionVars`` -> ``InteractionVars``).
3. ``{"Compute": expr}`` is evaluated against the parameters with a whitelisted
   evaluator (names, numbers, strings, + - * /, comparisons, ``!``/and/or,
   ``isEmpty``/``isEmptyStringArray``); anything else stays unevaluated.

``Instructions`` trees are not expanded (components carry their own scopes);
their referenced ids are listed. Rows also cover the NPC root interactions and
interactions those roles and components reference, and appearance models
(eye height, hitbox, scale).
"""

from __future__ import annotations

import ast
import copy
import operator
import re
from typing import Any

from .archive import GameArchive
from .common import gameplay, resolve_parented
from .weapons import _derived

FAMILY = "npc_roles"
VERSION = 1
ROLE_PREFIX = "Server/NPC/Roles/"
NPC_ROOT_PREFIX = "Server/Item/RootInteractions/"
INTERACTION_PREFIX = "Server/Item/Interactions/"
MODEL_PREFIX = "Server/Models/"
# Top-level role fields kept in the table (behaviour trees are not content).
ROLE_FIELDS = (
    "AttitudeGroup",
    "DefaultNPCAttitude",
    "DefaultPlayerAttitude",
    "DisableDamageGroups",
    "HotbarItems",
    "HotbarSize",
    "InteractionVars",
    "Invulnerable",
    "OffHandItems",
    "StartState",
)
_NOT = re.compile(r"!(?!=)")
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_COMPARE = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class _Unresolved(Exception):
    pass


def _function(name: str, args: list[Any]) -> Any:
    if name == "isEmpty" and len(args) == 1:
        return args[0] in (None, "", [], {})
    if name == "isEmptyStringArray" and len(args) == 1:
        return not args[0]
    raise _Unresolved(name)


def evaluate(expression: str, scope: dict[str, Any]) -> Any:
    tree = ast.parse(_NOT.sub(" not ", expression).strip(), mode="eval")

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in ("true", "false"):
                return node.id == "true"
            if node.id not in scope:
                raise _Unresolved(node.id)
            return scope[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not walk(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BoolOp):
            values = [walk(value) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            return _COMPARE[type(node.ops[0])](walk(node.left), walk(node.comparators[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            return _function(node.func.id, [walk(arg) for arg in node.args])
        raise _Unresolved(ast.dump(node))

    return walk(tree)


def _substitute(value: Any, scope: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"Compute"} and isinstance(value["Compute"], str):
            try:
                return evaluate(value["Compute"], scope)
            except (_Unresolved, SyntaxError, TypeError, ZeroDivisionError):
                return {"Compute": value["Compute"]}
        return {key: _substitute(child, scope) for key, child in value.items()}
    if isinstance(value, list):
        return [_substitute(child, scope) for child in value]
    return value


def _parameter_values(document: dict[str, Any]) -> dict[str, Any]:
    parameters = document.get("Parameters")
    if not isinstance(parameters, dict):
        return {}
    return {
        name: spec.get("Value") if isinstance(spec, dict) else spec
        for name, spec in parameters.items()
    }


def _strings(value: Any, found: set[str]) -> None:
    if isinstance(value, str):
        found.add(value)
    elif isinstance(value, list):
        for child in value:
            _strings(child, found)
    elif isinstance(value, dict):
        for child in value.values():
            _strings(child, found)


class _Roles:
    def __init__(self, archive: GameArchive):
        self.archive = archive
        self.index = archive.index(ROLE_PREFIX)
        self._cache: dict[str, tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]] = {}

    def resolve(self, stem: str, active: tuple[str, ...] = ()) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
        """Return ``(fields, parameters, reference_chain)`` before Compute."""

        if stem in self._cache:
            return self._cache[stem]
        if stem in active:
            raise ValueError(f"role reference cycle: {active + (stem,)}")
        document = self.archive.read_json(self.index[stem])
        if document.get("Type") == "Variant" and isinstance(document.get("Reference"), str):
            reference = document["Reference"]
            fields, parameters, chain = self.resolve(reference, active + (stem,))
            fields = copy.deepcopy(fields)
            parameters = dict(parameters)
            parameters.update(_parameter_values(document))
            modify = document.get("Modify") if isinstance(document.get("Modify"), dict) else {}
            scope = dict(parameters)
            for key, value in modify.items():
                if key.startswith("_"):
                    fields[key[1:]] = copy.deepcopy(value)
                else:
                    parameters[key] = _substitute(value, scope)
            chain = chain + (reference,)
        else:
            fields = {key: value for key, value in document.items() if key not in ("Parameters",)}
            parameters = _parameter_values(document)
            chain = ()
        result = (fields, parameters, chain)
        self._cache[stem] = result
        return result


def extract(archive: GameArchive) -> dict[str, Any]:
    roles = _Roles(archive)
    roots = archive.index(NPC_ROOT_PREFIX)
    interactions = archive.index(INTERACTION_PREFIX)
    models = archive.index(MODEL_PREFIX)
    rows: dict[str, Any] = {}
    referenced_roots: set[str] = set()
    referenced_interactions: set[str] = set()
    appearances: set[str] = set()
    resolved_parameters: dict[str, dict[str, Any]] = {}

    def parameters_of(stem: str) -> dict[str, Any]:
        if stem not in resolved_parameters:
            _, parameters, _ = roles.resolve(stem)
            scope = dict(parameters)
            values = {name: _substitute(value, scope) for name, value in parameters.items()}
            resolved_parameters[stem] = gameplay(values)
        return resolved_parameters[stem]

    role_types = {stem: archive.read_json(path).get("Type") for stem, path in roles.index.items()}
    for stem in sorted(roles.index):
        raw = archive.read_json(roles.index[stem])
        own_names: set[str] = set()
        _strings({key: value for key, value in raw.items() if key != "Parameters"}, own_names)
        components = sorted(
            name for name in own_names if role_types.get(name) == "Component" and name != stem
        )
        fields, parameters, chain = roles.resolve(stem)
        values = parameters_of(stem)
        scope = dict(parameters)
        scope.update(values)
        top = {key: _substitute(value, scope) for key, value in fields.items() if key not in ("Type", "Reference", "Modify")}
        names: set[str] = set()
        _strings(fields.get("Instructions"), names)
        _strings({key: value for key, value in top.items() if key != "Instructions"}, names)
        _strings(values, names)
        root_refs = sorted(name for name in names if name in roots)
        interaction_refs = sorted(name for name in names if name in interactions)
        referenced_roots.update(root_refs)
        referenced_interactions.update(interaction_refs)
        appearance = top.get("Appearance")
        if isinstance(appearance, str):
            appearances.add(appearance)
        motion = top.get("MotionControllerList")
        if chain:
            # Variant: store only the parameters whose resolved value differs
            # from the referenced role's; the content layer rebuilds the rest
            # by following ``reference_chain`` (exact, not approximate).
            base = parameters_of(chain[-1])
            stored = {key: value for key, value in values.items() if key not in base or base[key] != value}
        else:
            stored = values
        rows[f"role:{stem}"] = {
            "kind": "role",
            "source": roles.index[stem],
            "type": raw.get("Type"),
            "reference_chain": list(chain),
            "max_health": top.get("MaxHealth"),
            "knockback_scale": top.get("KnockbackScale"),
            "appearance": appearance,
            "motion_controllers": gameplay(motion) if isinstance(motion, list) else None,
            "parameters": stored,
            "parameters_complete": not chain,
            "fields": gameplay({key: top[key] for key in ROLE_FIELDS if key in top}),
            "root_references": root_refs,
            "interaction_references": interaction_refs,
            # Components named by this file itself (not inherited): with the
            # reference chain and parameter values this answers questions such
            # as which roles reach Component_Instruction_Hit_Interrupt.
            "component_references": components,
        }
    for stem in sorted(referenced_roots):
        document = archive.read_json(roots[stem])
        names: set[str] = set()
        _strings(document, names)
        referenced_interactions.update(name for name in names if name in interactions)
        rows[f"npc_root:{stem}"] = {
            "kind": "npc_root",
            "source": roots[stem],
            "document": gameplay(document),
        }
    pending = sorted(referenced_interactions)
    seen: set[str] = set()
    while pending:
        stem = pending.pop()
        if stem in seen:
            continue
        seen.add(stem)
        document, chain = resolve_parented(stem, interactions, archive.read_json)
        names = set()
        _strings({key: value for key, value in document.items() if key != "Parent"}, names)
        pending.extend(sorted(name for name in names if name in interactions and name not in seen))
        derived = _derived(document, chain)
        rows[f"npc_interaction:{stem}"] = {
            "kind": "npc_interaction",
            "source": interactions[stem],
            "run_time": document.get("RunTime"),
            **derived,
            "document": gameplay(document),
        }
    for stem in sorted(appearances):
        if stem not in models:
            continue
        model, model_chain = resolve_parented(stem, models, archive.read_json)
        box = model.get("HitBox") if isinstance(model.get("HitBox"), dict) else {}
        rows[f"model:{stem}"] = {
            "kind": "model",
            "source": models[stem],
            "parent_chain": list(model_chain),
            "eye_height": model.get("EyeHeight"),
            "min_scale": model.get("MinScale"),
            "max_scale": model.get("MaxScale"),
            "hitbox": {
                "min": [float(box.get("Min", {}).get(axis, 0.0)) for axis in ("X", "Y", "Z")],
                "max": [float(box.get("Max", {}).get(axis, 0.0)) for axis in ("X", "Y", "Z")],
            }
            if box
            else None,
        }
    return rows
