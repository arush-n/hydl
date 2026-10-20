"""The catalogue parser: skill catalogue rows -> typed records, verb contracts, descriptions, programs.

``docs/AGENT_BRIEF.md`` deliverable A2 and ``docs/BUILD.md`` section 1. The skill catalogue
(``agent_skills/training_runs/SKILL_CATALOGUE.md``, a private file) is the source of truth for the
skill list. This module holds only the reading rules and the vocabulary that maps an instruction's
words to typed arguments; it never stores the catalogue's content, and it reads the file at run time.

What it reads
-------------
Sections ``## N. Title`` with a markdown table each. Sections 1 to 13 have the columns
``skill | what a person does | Hytale mechanic | instruction -> done | kind | gen 5``; sections 9 to
12 omit ``instruction -> done``. Section 14 has ``activity | sequence | context it needs``, then
``program of | sequence | context it needs``: the programs of activity rows that live in other
sections (not rows of their own, so a row count stays one per activity). A skill cell may hold
several ids (``"walk_to, run_to, sprint_to"``): one row, several verbs sharing it.

What it yields
--------------
* ``CatalogueSkill``: one per row, with its ids, section, description, mechanic, instruction, kind
  (``L`` learned, ``C`` command, ``A`` activity, ``?`` unknown) and the gen-5 note (information only).
* ``ArgumentSchema``: the instruction parsed into typed arguments. ``required`` kinds must all be
  present and ``any_of`` kinds at least one; ``position``, ``direction``, ``entity`` and ``cell`` are
  carried by a request, while ``region``, ``item``, ``recipe`` and ``route`` are context the verb reads
  from the scratchpad (a request cannot carry them, SYSTEM_ONE.md section 9.10). ``scalars`` are the
  verb's named numbers, at most ``PARAM_WIDTH``. ``done`` is the done text. Sections 9 to 12 have no
  instruction, so their schemas are group defaults (``source == "group default"``).
* ``VerbContract``: a ``contracts.SkillContract`` (same validation as every skill; control-field
  defaults as today) beside the schema, the scope and the kind. ``SkillContract`` is unchanged:
  the typed schema and the scope travel here rather than in ``contracts.py``.
* ``ActivityProgram``: section 14's activities as ordered steps; each step names the catalogue ids
  in its text, and its exit is the step's ``done`` (the column gives no other exit).
* A description per verb for the middle layer's matching: ``"<id>: <what a person does>. ..."``.

Scopes. A row's scope is its section's (``SECTION_SCOPES``) unless ``VERB_SCOPES`` names the verb;
each departure carries its reason there. A few learned verbs drive a second scope as part of the same
competency (``SECOND_SCOPES``, for example ``build_up`` places blocks with the hands while it jumps):
the one request carries both scopes, and the combiner grants them together or not at all
(``requests.ArgumentSchema.scopes`` is a tuple; agreed with the middle layer on 2026-10-17). Parallel
requests are for independent verbs (walk while looking at something), not for halves of one verb. A verb that strikes
with a weapon (``WEAPON_SECTIONS``, ``STRIKE_VERBS``) also holds ``abilities_magic``: in the gym a weapon's attacks
are its ability slots, so the strike drives the ability head (2026-10-19).

The three proof-of-concept skills (``idle``, ``pursuit``, ``evader``) are not catalogue rows.
``proof_of_concept_contracts`` maps them onto the rows that describe them (``PROOF_OF_CONCEPT``)
with their own ids and declared control (``contracts.default_control_table``) and the whole surface
as their scope, because today one skill drives every head of its lane.

CLI::

    PYTHONPATH=/Users/arush/hydl JAX_PLATFORMS=cpu .venv/bin/python -m \\
        experimental.neural_agent.skill_agent.contract.catalogue [--path FILE] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experimental.neural_agent.contracts import (
    EVENT_NAMES,
    PARAM_WIDTH,
    SkillContract,
    default_control_table,
)
from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES, scope_heads

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CATALOGUE = ROOT / "agent_skills" / "training_runs" / "SKILL_CATALOGUE.md"
CATALOGUE_VERSION = "catalogue-v1"

KINDS: Mapping[str, str] = {"L": "learned", "C": "command", "A": "activity", "?": "unknown"}
CARRIED_KINDS: tuple[str, ...] = ("position", "direction", "entity", "cell")
CONTEXT_KINDS: tuple[str, ...] = ("region", "item", "recipe", "route")
ARGUMENT_KINDS: tuple[str, ...] = CARRIED_KINDS + CONTEXT_KINDS

SECTION_SCOPES: Mapping[int, str] = {
    1: "locomotion",
    2: "gaze",
    3: "hands",
    4: "hands",
    5: "hands",
    6: "hands",
    7: "interaction",
    8: "hands",
    9: "hands",
    10: "hands",
    11: "abilities_magic",
    12: "posture_defence",
    13: "locomotion",
}
"""Default scope by catalogue section; section 14 holds activities, which have no scope."""

VERB_SCOPES: Mapping[str, str] = {
    "dodge": "posture_defence",  # the gen-7 dodge head
    "hookshot_to": "hands",  # a held tool fired with the trigger
    "use_teleporter": "interaction",  # a menu choice of destination
    "use_spyglass": "hands",  # a held tool, secondary use
    "read_map": "interaction",  # a menu
    "pick_up": "locomotion",  # walk over the item
    "loot_chest": "interaction",  # a container menu
    "lure_animal": "locomotion",  # walk while holding food
    "set_rune": "abilities_magic",  # the rune_set head
    "repair_gear": "interaction",  # the repair menu
    "sleep": "interaction",  # use a bed
    "sit": "interaction",  # use a seat
    "avoid_hazards": "locomotion",  # a constraint on movement
    "recover_stamina": "locomotion",  # stop sprinting
    "hide": "locomotion",  # move out of sight
    "shield_guard": "posture_defence",  # the guard head
    "staff_wand_spellbook": "hands",  # a weapon's charged primary
    "punish": "hands",  # attack timing
    "interrupt": "hands",  # attack timing
    "keep_range": "locomotion",  # distance keeping
    "kite": "locomotion",  # back away; the shot is its second scope (hands)
    "flank": "locomotion",  # move to the side
    "finish": "locomotion",  # chase
    "give": "hands",  # item_drop_give
    "trade_with": "interaction",  # the barter menu
    "signal": "gaze",  # look at the thing
    "share_chest": "interaction",  # a container menu
}
"""Departures from ``SECTION_SCOPES``, one reason each."""

SECOND_SCOPES: Mapping[str, tuple[str, ...]] = {
    "build_up": ("locomotion",),
    "bridge": ("locomotion",),
    "mine_in_direction": ("locomotion",),
    "mine_down_stairs": ("locomotion",),
    "chop_tree": ("locomotion",),
    "carry_block": ("locomotion",),
    "hookshot_to": ("gaze",),
    "kite": ("hands",),
    "search_for": ("locomotion",),
    "lure_animal": ("hands",),
}
"""Learned verbs that also drive a second scope; their one request carries both scopes."""

WEAPON_SECTIONS: tuple[int, ...] = (9, 10)
STRIKE_VERBS: frozenset[str] = frozenset({"punish", "interrupt", "kite"})
"""The melee and ranged sections, and the tactics that strike. Their learned verbs that use the hands also hold
``abilities_magic``: in the gym a weapon's attacks are its ability slots (the v3 encoder masks the base action's
attacks for the learner, ``observation/v3/encoding/encoder.py:470-476``), so a strike drives the ability head.
Measured 2026-10-19 in the open-flat combat arena: pressing ability slot 1 took the opponent from 61 health to 1."""

PROGRESS_UNIVERSAL: tuple[str, ...] = ("progress", "elapsed_s")
"""Progress every verb reports: the fraction toward its done condition (0 to 1) and seconds since its grant."""

PROGRESS_FAMILIES: tuple[tuple[str, str], ...] = (
    ("distance_m", r"\b(arriv\w*|reach\w*|there|inside|home|away|across)\b"),
    ("hits", r"\b(hits?|lands?|down|dead|kill\w*|broken|stunned|interrupted|blocked)\b"),
    ("count", r"\b(gain\w*|placed|collect\w*|craft\w*|mined|picked|stored|built|cooked|harvest\w*|planted|tilled"
              r"|filled|made|equipped|eaten|drunk|repaired|fed|tamed|sold|bought|traded|given|loaded|lit|opened|closed)\b"),
    ("angle_deg", r"\b(cent(red|ered)|facing|seen|found|aim\w*|spotted|located)\b"),
)
"""The one measured progress signal a verb adds, by its done text's family (whole words, first match wins;
no match: the universal pair only). Generic English stems, not catalogue text."""

PROOF_OF_CONCEPT: Mapping[str, str] = {"idle": "wait_here", "pursuit": "finish", "evader": "avoid"}
"""Proof-of-concept skill id -> the catalogue row that describes it."""

# ---- the instruction vocabulary ----

_PHRASE_KINDS: Mapping[str, str] = {
    "place": "position", "point": "position", "goal": "position", "goal past obstacle": "position",
    "landing spot": "position", "target under water": "position", "target": "position",
    "destination": "position", "site": "position", "start": "position", "end": "position",
    "gap": "position", "spot": "position",
    "direction": "direction",
    "entity": "entity", "person": "entity", "animal": "entity", "animals": "entity", "creature": "entity",
    "threat": "entity", "observer": "entity", "ally": "entity", "villager": "entity", "merchant": "entity",
    "thing": "entity",
    "cell": "cell", "cells": "cell", "climbable cell": "cell", "hook point": "cell", "pad": "cell",
    "tree": "cell", "plant": "cell", "chest": "cell", "bed": "cell", "seat": "cell", "water": "cell",
    "block": "cell", "bench": "cell",
    "area": "region", "hazard cells": "region", "shape": "region",
    "item": "item", "items": "item", "slot or item": "item", "seed": "item", "food": "item", "potion": "item",
    "rune": "item", "ore": "item", "fuel": "item", "input": "item", "material": "item", "class": "item",
    "recipe": "recipe", "trade": "recipe",
    "waypoints": "route",
}
_PHRASE_SCALARS: Mapping[str, str] = {
    "radius": "radius", "pace": "pace", "tolerance": "tolerance", "time": "time", "length": "length",
    "depth": "depth", "height": "height", "count": "count", "band": "band", "vertical": "vertical",
    "variant": "variant", "from": "from_slot", "to": "to_slot", "slot": "slot",
}
_NORMALISE: tuple[tuple[str, str], ...] = (("up or down", "vertical"),)
_NOISE_WORDS: frozenset[str] = frozenset({"a", "an", "the", "then", "of"})
_PHRASE_OVERRIDES: Mapping[tuple[str, str], str] = {
    ("place_block", "block"): "item",  # the block to place, not a cell
    ("best_tool_for", "block"): "cell",
}

# Sections 9 to 12 have no instruction column: (any_of, required, scalars, done) per section.
_GROUP_DEFAULTS: Mapping[int, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]] = {
    9: ((), ("entity",), ("hold_seconds",), "target hit"),
    10: (("entity", "position"), (), ("hold_seconds",), "projectile hits"),
    11: (("entity", "position", "direction"), (), ("hold_seconds",), "ability lands"),
    12: ((), ("entity",), (), "tactic achieved"),
}
_BAND_TACTICS: frozenset[str] = frozenset({"keep_range", "kite"})
"""Section-12 verbs whose group default also names a distance band."""
_COMMAND_GROUP_DEFAULTS: Mapping[int, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]] = {
    10: (("position", "cell"), (), (), "deployed"),
    11: ((), (), (), "buff applied"),
}

_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_SECTION = re.compile(r"^## (\d+)\. (.+?)\s*$")
_ARROW = re.compile(r"\s*(?:→|->)\s*")


@dataclass(frozen=True)
class CatalogueSkill:
    """One catalogue row as written."""

    ids: tuple[str, ...]
    section: int
    group: str
    description: str
    mechanic: str
    instruction: str
    kind: str
    gen5: str
    row: int


@dataclass(frozen=True)
class ArgumentSchema:
    """A verb's typed arguments, parsed from its instruction (or the group default)."""

    required: tuple[str, ...]
    any_of: tuple[str, ...]
    scalars: tuple[str, ...]
    done: str
    source: str
    unparsed: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """True when every instruction phrase was read, the done text is known and the scalars fit."""

        return not self.unparsed and bool(self.done) and len(self.scalars) <= PARAM_WIDTH

    @property
    def kinds(self) -> tuple[str, ...]:
        """Every argument kind the verb names, in ``ARGUMENT_KINDS`` order."""

        named = set(self.required) | set(self.any_of)
        return tuple(kind for kind in ARGUMENT_KINDS if kind in named)

    def to_dict(self) -> dict[str, Any]:
        return {"required": list(self.required), "any_of": list(self.any_of), "scalars": list(self.scalars),
                "done": self.done, "source": self.source, "unparsed": list(self.unparsed)}


@dataclass(frozen=True)
class VerbContract:
    """A verb's SkillContract beside its typed schema, scope, kind and description."""

    verb: str
    contract: SkillContract
    arguments: ArgumentSchema
    scopes: tuple[str, ...]
    kind: str
    section: int
    description: str
    progress: tuple[str, ...] = PROGRESS_UNIVERSAL

    def to_dict(self) -> dict[str, Any]:
        contract = self.contract
        return {
            "verb": self.verb, "kind": self.kind, "section": self.section, "scopes": list(self.scopes),
            "arguments": self.arguments.to_dict(),
            "description": self.description, "progress": list(self.progress),
            "contract": {
                "skill_id": contract.skill_id, "version": contract.version,
                "observation_groups": list(contract.observation_groups),
                "action_scope_sha256": contract.action_scope_sha256, "param_names": list(contract.param_names),
                "termination": list(contract.termination), "success_signals": list(contract.success_signals),
                "failure_signals": list(contract.failure_signals),
                "minimum_commitment_seconds": contract.minimum_commitment_seconds,
                "maximum_duration_seconds": (None if math.isinf(contract.maximum_duration_seconds)
                                             else contract.maximum_duration_seconds),
                "emergency_events": list(contract.emergency_events),
            },
        }


@dataclass(frozen=True)
class ActivityStep:
    """One step of an activity: its text, the catalogue ids it names, its argument, stage and exit.

    Steps with the same ``stage`` run at the same time (joined by ``+`` in the catalogue); stages run
    in order (separated by ``,``). ``repeat`` marks ``(for each ...)``: the step runs once per target
    the caller supplies. The exit is the step's ``done`` (the catalogue gives no other exit).
    """

    text: str
    verbs: tuple[str, ...]
    argument: str
    exit: str = "done"
    stage: int = 0
    repeat: bool = False


@dataclass(frozen=True)
class ActivityProgram:
    """A section-14 program: its steps (by stage) and the context it needs.

    A program whose id is an activity row of another section is that row's program.
    """

    activity: str
    steps: tuple[ActivityStep, ...]
    context: str

    @property
    def description(self) -> str:
        """The text the middle layer matches against: the steps in order and the context needed."""

        stages: dict[int, list[str]] = {}
        for step in self.steps:
            stages.setdefault(step.stage, []).append(step.text)
        steps = ", then ".join(" together with ".join(texts) for _, texts in sorted(stages.items()))
        return f"{self.activity}: {steps}. Needs {self.context or 'nothing more'}."

    @property
    def stages(self) -> tuple[tuple[ActivityStep, ...], ...]:
        """The steps grouped by stage, in order."""

        grouped: dict[int, list[ActivityStep]] = {}
        for step in self.steps:
            grouped.setdefault(step.stage, []).append(step)
        return tuple(tuple(steps) for _, steps in sorted(grouped.items()))

    def to_dict(self) -> dict[str, Any]:
        return {"activity": self.activity, "context": self.context, "description": self.description,
                "steps": [{"text": step.text, "verbs": list(step.verbs), "argument": step.argument,
                           "exit": step.exit, "stage": step.stage, "repeat": step.repeat} for step in self.steps]}


@dataclass(frozen=True)
class Catalogue:
    """The parsed catalogue: rows, activities, and every table row that could not be read, verbatim."""

    skills: tuple[CatalogueSkill, ...]
    activities: tuple[ActivityProgram, ...]
    unparsed_rows: tuple[str, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        """Every verb id of the skill rows, in catalogue order."""

        return tuple(verb for skill in self.skills for verb in skill.ids)


# ---- parsing ----


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_catalogue(text: str) -> Catalogue:
    """Parse catalogue markdown into rows and activities; rows that cannot be read are kept verbatim."""

    skills: list[CatalogueSkill] = []
    activities: list[ActivityProgram] = []
    unparsed: list[str] = []
    section, group, header = 0, "", None
    row_in_section = 0
    for line in text.splitlines():
        heading = _SECTION.match(line)
        if heading:
            section, group, header, row_in_section = int(heading.group(1)), heading.group(2), None, 0
            continue
        if line.startswith("## "):
            section, header = 0, None
            continue
        if not section or not line.startswith("|"):
            header = None  # a table ends at its first non-table line; the next table brings its own header
            continue
        cells = _cells(line)
        if header is None:
            header = [cell.lower() for cell in cells]
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        row_in_section += 1
        if len(cells) != len(header):
            unparsed.append(line)
            continue
        record = dict(zip(header, cells))
        if "activity" in record or "program of" in record:
            program = _activity(record)
            if program is None:
                unparsed.append(line)
            else:
                activities.append(program)
            continue
        skill = _skill(record, section, group, row_in_section)
        if skill is None:
            unparsed.append(line)
        else:
            skills.append(skill)
    return Catalogue(tuple(skills), tuple(activities), tuple(unparsed))


def _skill(record: Mapping[str, str], section: int, group: str, row: int) -> CatalogueSkill | None:
    ids = tuple(part.strip() for part in record.get("skill", "").split(","))
    kind = record.get("kind", "").strip()[:1]
    if not ids or not all(_ID.match(verb) for verb in ids) or kind not in KINDS:
        return None
    instruction = next((value for key, value in record.items() if key.startswith("instruction")), "")
    return CatalogueSkill(
        ids=ids,
        section=section,
        group=group,
        description=record.get("what a person does", ""),
        mechanic=record.get("hytale mechanic", ""),
        instruction=instruction,
        kind=kind,
        gen5=record.get("gen 5", ""),
        row=row,
    )


def _top_level_split(text: str, separator: str = ",") -> list[str]:
    parts, depth, current = [], 0, []
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        if character == separator and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _activity(record: Mapping[str, str]) -> ActivityProgram | None:
    name = (record.get("activity") or record.get("program of", "")).strip()
    if not _ID.match(name):
        return None
    steps = []
    for stage, stage_text in enumerate(_top_level_split(record.get("sequence", ""))):
        for text in _top_level_split(stage_text, "+"):
            match = re.match(r"^([a-z][a-z0-9_]*)\s*\((.*)\)\s*$", text)
            argument = match.group(2).strip() if match else ""
            steps.append(ActivityStep(text=text, verbs=(), argument=argument, stage=stage,
                                      repeat=argument.lower().startswith("for each")))
    return ActivityProgram(activity=name, steps=tuple(steps), context=record.get("context it needs", ""))


def resolve_activities(catalogue: Catalogue) -> tuple[ActivityProgram, ...]:
    """Return the activities with each step's ``verbs``: the catalogue ids its text names as words.

    Activity ids count too, so a step may name another activity (``find_food`` in ``cook_meal``).
    """

    known = set(catalogue.ids) | {program.activity for program in catalogue.activities}
    resolved = []
    for program in catalogue.activities:
        steps = []
        for step in program.steps:
            words = re.findall(r"[a-z][a-z0-9_]*", step.text.split("(")[0])
            verbs = tuple(dict.fromkeys(word for word in words if word in known))
            steps.append(ActivityStep(step.text, verbs, step.argument, step.exit, step.stage, step.repeat))
        resolved.append(ActivityProgram(program.activity, tuple(steps), program.context))
    return tuple(resolved)


def _normal(phrase: str) -> str:
    return " ".join(word for word in phrase.split() if word not in _NOISE_WORDS)


def parse_instruction(verb: str, instruction: str, section: int, kind: str) -> ArgumentSchema:
    """Read one instruction (``"place, radius, pace -> arrived"``) into an ArgumentSchema.

    Sections without an instruction get their group default. A phrase whose alternatives are joined by
    ``or`` (``"point or cell"``) or a list that ends in one (``"entity, point or cell"``) becomes
    ``any_of``; every other kind phrase is ``required``; scalar phrases name scalars.
    """

    if not instruction:
        defaults = (_COMMAND_GROUP_DEFAULTS if kind == "C" else _GROUP_DEFAULTS).get(section)
        if defaults is None:
            return ArgumentSchema((), (), (), "", "group default", ("no instruction and no group default",))
        any_of, required, scalars, done = defaults
        if section == 12 and verb in _BAND_TACTICS:
            scalars = ("band_min", "band_max")
        return ArgumentSchema(_in_kind_order(required), _in_kind_order(any_of), scalars, done, "group default")
    if _ARROW.search(instruction):
        left, done = _ARROW.split(instruction, maxsplit=1)
    else:
        left, done = "", instruction
    required: list[str] = []
    any_of: list[str] = []
    scalars: list[str] = []
    unparsed: list[str] = []
    clauses = [clause.strip() for clause in left.lower().split(";")] if left.strip() not in ("—", "-", "") else []
    for clause in clauses:
        for old, new in _NORMALISE:
            clause = clause.replace(old, new)
        elements = [element.strip() for part in clause.split(",") for element in re.split(r"\s+and\s+", part)
                    if element.strip()]
        alternatives = [_normal(part) for part in re.split(r"\s+or\s+", elements[-1])] if elements else []
        listed = [_normal(element) for element in elements[:-1]] + alternatives
        if len(alternatives) > 1 and all(_kind(verb, phrase) for phrase in listed):
            any_of.extend(_kind(verb, phrase) for phrase in listed)
            continue
        for element in elements:
            for phrase in (_normal(part) for part in re.split(r"\s+or\s+", element)) if " or " in element else [_normal(element)]:
                if phrase in _PHRASE_SCALARS:
                    scalars.append(_PHRASE_SCALARS[phrase])
                elif _kind(verb, phrase):
                    target = any_of if " or " in element else required
                    target.append(_kind(verb, phrase))
                else:
                    unparsed.append(phrase)
    return ArgumentSchema(
        required=_in_kind_order(required),
        any_of=_in_kind_order(any_of),
        scalars=tuple(dict.fromkeys(scalars)),
        done=done.strip(),
        source="instruction",
        unparsed=tuple(unparsed),
    )


def _in_kind_order(kinds: Sequence[str]) -> tuple[str, ...]:
    order = {kind: index for index, kind in enumerate(ARGUMENT_KINDS)}
    return tuple(sorted(dict.fromkeys(kinds), key=order.__getitem__))


def _kind(verb: str, phrase: str) -> str | None:
    return _PHRASE_OVERRIDES.get((verb, phrase)) or _PHRASE_KINDS.get(phrase)


# ---- contracts and descriptions ----


def scope_for(verb: str, section: int) -> str | None:
    """Return a verb's scope: its ``VERB_SCOPES`` entry, else its section's, else None (activities)."""

    return VERB_SCOPES.get(verb) or SECTION_SCOPES.get(section)


def _label(text: str) -> str:
    label = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return label[:64] or "done"


def _observation_groups(arguments: ArgumentSchema, scopes: Sequence[str]) -> tuple[str, ...]:
    groups = ["self"]
    if "locomotion" in scopes:
        groups += ["geometry", "route"]
    kinds = set(arguments.kinds)
    if "entity" in kinds:
        groups.append("entity")
    if "cell" in kinds or "hands" in scopes:
        groups.append("block_candidate")
    if "item" in kinds or "hands" in scopes:
        groups.append("inventory_slot")
    if {"hands", "abilities_magic", "posture_defence"} & set(scopes):
        groups.append("kit")
    return tuple(dict.fromkeys(groups))


def scope_sha256(scopes: Sequence[str]) -> str:
    """Digest of a scope set: sha256 of the sorted JSON of the scopes and their gen-7 heads."""

    payload = {scope: list(scope_heads(scope, "gen7")) for scope in scopes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def describe(skill: CatalogueSkill, verb: str, arguments: ArgumentSchema) -> str:
    """The text the middle layer matches intents against: what it does, what to give, when it is done."""

    give = ", ".join([*arguments.required, *(("one of " + " / ".join(arguments.any_of),) if arguments.any_of else ()),
                      *arguments.scalars]) or "nothing"
    return f"{verb}: {skill.description}. Give {give}; done when {arguments.done or 'it completes'}."


def verb_contract(skill: CatalogueSkill, verb: str) -> VerbContract | None:
    """Build the contract of one verb of a learned, command or unknown row; activities get None."""

    if skill.kind == "A":
        return None
    arguments = parse_instruction(verb, skill.instruction, skill.section, skill.kind)
    scope = scope_for(verb, skill.section)
    scopes = (scope, *SECOND_SCOPES.get(verb, ())) if scope else ()
    if skill.kind == "L" and "hands" in scopes and (skill.section in WEAPON_SECTIONS or verb in STRIKE_VERBS):
        scopes = (*scopes, "abilities_magic")
    return build_verb_contract(verb, arguments, scopes, skill.kind, skill.section, describe(skill, verb, arguments))


def progress_signals(done: str) -> tuple[str, ...]:
    """The progress signals a verb with this done text reports: the universal pair, plus its family's measure."""

    text = done.lower()
    family = next((name for name, pattern in PROGRESS_FAMILIES if re.search(pattern, text)), None)
    return PROGRESS_UNIVERSAL + ((family,) if family else ())


def build_verb_contract(
    verb: str,
    arguments: ArgumentSchema,
    scopes: Sequence[str],
    kind: str,
    section: int,
    description: str,
    *,
    version: str = CATALOGUE_VERSION,
    checkpoint_sha256: str | None = None,
    **control: Any,
) -> VerbContract:
    """Build a verb's contract from its parts: the one constructor for catalogue and created skills.

    The SkillContract gets the verb id, the observation groups the arguments and scopes imply, the scope
    digest, the scalars as ``param_names``, ``(success, failures...)`` as termination (success is the done
    text as a label; failures ``stuck`` and ``timeout`` for learned verbs, ``precondition_failed`` for
    commands) and the control fields given in ``control`` (``minimum_commitment_seconds``,
    ``maximum_duration_seconds``, ``emergency_events``; today's defaults otherwise).

    Raises:
        ValueError: If ``kind`` is not in ``KINDS``, a scope is unknown, the schema is incomplete, or the
            SkillContract refuses a field.
    """

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {tuple(KINDS)}, got {kind!r}")
    unknown = [scope for scope in scopes if scope not in SCOPE_NAMES]
    if unknown or len(set(scopes)) != len(tuple(scopes)):
        raise ValueError(f"scopes must be distinct names from {SCOPE_NAMES}, got {tuple(scopes)}")
    if not arguments.complete:
        raise ValueError(f"the argument schema of {verb!r} is incomplete: {arguments.unparsed or 'no done text'}")
    success = _label(arguments.done)
    failures = ("precondition_failed",) if kind == "C" else ("stuck", "timeout")
    contract = SkillContract(
        skill_id=verb,
        version=version,
        observation_groups=_observation_groups(arguments, scopes),
        action_scope_sha256=scope_sha256(scopes),
        param_names=arguments.scalars[:PARAM_WIDTH],
        termination=(success, *failures),
        success_signals=(success,),
        failure_signals=failures,
        checkpoint_sha256=checkpoint_sha256,
        **control,
    )
    return VerbContract(verb, contract, arguments, tuple(scopes), kind, section, description,
                        progress_signals(arguments.done))


def verb_contracts(catalogue: Catalogue) -> tuple[VerbContract, ...]:
    """Every verb's contract, in catalogue order (activities excluded)."""

    built = (verb_contract(skill, verb) for skill in catalogue.skills for verb in skill.ids)
    return tuple(contract for contract in built if contract is not None)


def proof_of_concept_contracts(catalogue: Catalogue) -> tuple[VerbContract, ...]:
    """The contracts of idle, pursuit and evader, read from the rows that describe them.

    Each keeps its own id and its declared control (``contracts.default_control_table``) and owns the
    whole surface, because today one skill drives every head of its lane.

    Raises:
        KeyError: If a mapped row is missing from the catalogue.
    """

    rows = {verb: skill for skill in catalogue.skills for verb in skill.ids}
    ids = tuple(PROOF_OF_CONCEPT)
    table = default_control_table(ids)
    built = []
    for index, skill_id in enumerate(ids):
        row = rows[PROOF_OF_CONCEPT[skill_id]]
        base = verb_contract(row, PROOF_OF_CONCEPT[skill_id])
        assert base is not None
        maximum = float(table.maximum_duration_seconds[index])
        events = tuple(name for name, on in zip(EVENT_NAMES, np.asarray(table.emergency_interrupt_mask[index])) if on)
        contract = SkillContract(
            skill_id=skill_id,
            version=CATALOGUE_VERSION,
            observation_groups=base.contract.observation_groups,
            action_scope_sha256=scope_sha256(SCOPE_NAMES),
            param_names=(),
            termination=base.contract.termination,
            success_signals=base.contract.success_signals,
            failure_signals=base.contract.failure_signals,
            checkpoint_sha256=None,
            minimum_commitment_seconds=float(table.minimum_commitment_seconds[index]),
            maximum_duration_seconds=math.inf if math.isinf(maximum) else maximum,
            emergency_events=events,
        )
        built.append(VerbContract(skill_id, contract, base.arguments, SCOPE_NAMES, row.kind, row.section,
                                  f"{skill_id} (proof of concept, catalogue row {PROOF_OF_CONCEPT[skill_id]}): "
                                  f"{row.description}.", progress_signals(base.arguments.done)))
    return tuple(built)


def load_catalogue(path: str | Path = DEFAULT_CATALOGUE) -> Catalogue:
    """Read and parse the catalogue file.

    Raises:
        FileNotFoundError: If the file does not exist (it is private and may be absent).
    """

    return parse_catalogue(Path(path).read_text(encoding="utf-8"))


def summary(catalogue: Catalogue) -> dict[str, Any]:
    """Counts and the lists a gate reports: kinds, scopes, schema completeness, unparsed rows and steps."""

    contracts = verb_contracts(catalogue)
    by_verb = {contract.verb: contract for contract in contracts}
    learned = [skill for skill in catalogue.skills if skill.kind == "L"]
    learned_scoped = [skill for skill in learned if all(1 <= len(by_verb[verb].scopes) <= 3 for verb in skill.ids)]
    several_scopes = [skill for skill in learned if any(len(by_verb[verb].scopes) >= 2 for verb in skill.ids)]
    learned_complete = [skill for skill in learned if all(by_verb[verb].arguments.complete for verb in skill.ids)]
    activities = resolve_activities(catalogue)
    unresolved = [f"{program.activity}: {step.text}" for program in activities for step in program.steps
                  if not step.verbs]
    activity_rows = {verb for skill in catalogue.skills if skill.kind == "A" for verb in skill.ids}
    programs = {program.activity for program in catalogue.activities}
    attached = programs & activity_rows
    kinds = {name: sum(skill.kind == code for skill in catalogue.skills) for code, name in KINDS.items()}
    kinds["activity"] += len(programs - attached)
    return {
        "rows": len(catalogue.skills) + len(programs - attached),
        "skill_rows": len(catalogue.skills),
        "activity_rows": len(programs - attached),
        "activity_programs": len(programs),
        "activities_without_program": sorted(activity_rows - programs),
        "kinds": kinds,
        "verbs": len(catalogue.ids),
        "contracts": len(contracts),
        "learned_rows": len(learned),
        "learned_rows_with_scopes": len(learned_scoped),
        "learned_rows_with_several_scopes": len(several_scopes),
        "learned_rows_with_complete_schema": len(learned_complete),
        "rows_on_group_default": sum(1 for skill in catalogue.skills if skill.kind != "A" and not skill.instruction),
        "incomplete_schemas": [f"{contract.verb}: {list(contract.arguments.unparsed)}" for contract in contracts
                               if not contract.arguments.complete],
        "unparsed_rows": list(catalogue.unparsed_rows),
        "activity_steps": sum(len(program.steps) for program in activities),
        "unresolved_activity_steps": unresolved,
        "scopes": {scope: sum(scope in contract.scopes for contract in contracts) for scope in SCOPE_NAMES},
    }


def catalogue_json(catalogue: Catalogue) -> dict[str, Any]:
    """The JSON dump: the summary, every contract, every activity, and the proof-of-concept mapping.

    ``proof_of_concept`` is empty and ``proof_of_concept_missing`` names the rows when a mapped row is
    absent (a partial catalogue).
    """

    missing = sorted(row for row in PROOF_OF_CONCEPT.values() if row not in set(catalogue.ids))
    proof = [] if missing else [contract.to_dict() for contract in proof_of_concept_contracts(catalogue)]
    return {
        "summary": summary(catalogue),
        "contracts": [contract.to_dict() for contract in verb_contracts(catalogue)],
        "activities": [program.to_dict() for program in resolve_activities(catalogue)],
        "proof_of_concept": proof,
        "proof_of_concept_missing": missing,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse the skill catalogue into contracts and programs.")
    parser.add_argument("--path", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--json", action="store_true", help="print the full JSON dump")
    args = parser.parse_args(argv)
    catalogue = load_catalogue(args.path)
    if args.json:
        sys.stdout.write(json.dumps(catalogue_json(catalogue), indent=1, allow_nan=False) + "\n")
    else:
        sys.stdout.write(json.dumps(summary(catalogue), indent=1) + "\n")
    return 0


__all__ = [
    "ARGUMENT_KINDS",
    "CARRIED_KINDS",
    "CATALOGUE_VERSION",
    "SECOND_SCOPES",
    "STRIKE_VERBS",
    "CONTEXT_KINDS",
    "DEFAULT_CATALOGUE",
    "KINDS",
    "PROGRESS_FAMILIES",
    "PROGRESS_UNIVERSAL",
    "PROOF_OF_CONCEPT",
    "SECTION_SCOPES",
    "VERB_SCOPES",
    "WEAPON_SECTIONS",
    "ActivityProgram",
    "ActivityStep",
    "ArgumentSchema",
    "Catalogue",
    "CatalogueSkill",
    "VerbContract",
    "build_verb_contract",
    "catalogue_json",
    "describe",
    "load_catalogue",
    "parse_catalogue",
    "parse_instruction",
    "progress_signals",
    "proof_of_concept_contracts",
    "resolve_activities",
    "scope_for",
    "scope_sha256",
    "summary",
    "verb_contract",
    "verb_contracts",
]


if __name__ == "__main__":
    raise SystemExit(main())
