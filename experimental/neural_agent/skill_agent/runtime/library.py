"""The skill library: one place to create, register and use skills.

A skill is an *option* in the sense of Sutton, Precup and Singh (1999): an initiation condition, a
policy and a termination condition. Here the initiation condition is the request check
(``protocol.check``: the verb exists, its arguments are present, finite, in the world and known, its
scopes are legal), the policy is the skill's executor, and the termination condition is the done signal
in its contract. The library binds each verb's contract to its executor, so the combiner, the program
runner (``runtime/programs.py``) and the middle layer all read one registry, and a new skill is one call.

Kinds of skill and their executors::

    learned   a Body: (request, ticks since start, observation tokens, belief tokens) -> BodyStep
              (head choices, done, outcome, progress); the tokens come from the agent's context.
              Until a verb's network is trained, a StubBody stands in and ``placeholder`` is True.
    command   a Body as well: an exact script in the same envelope (``script``), so the combiner,
              the status stream and the middle layer treat it like any other verb.
    activity  an ActivityProgram, run by ``programs.ProgramRunner``: it issues requests for its steps.

Create, register, use::

    library = SkillLibrary.from_catalogue()                       # 122 verbs and 19 programs
    library.create("wave", description="wave at someone", scopes=("gaze",),
                   required=("entity",), done="waved", body=my_body)
    request = library.request("walk_to", agent=0, request_id=1, tick=0,
                              position=(4.0, 64.0, -2.0), scalars={"radius": 1.0, "pace": 1.0})
    combiner = library.combiner(layout="gen7")
    step = combiner.step(0, 0, [request], context)

Rules the registry enforces (``SKILL_LIBRARY.md`` section 7, requirement 7, and ADR-001 D1): a verb id is
registered once; replacing it needs ``replace=True`` and a new contract version, so a frozen skill never
changes under the same version; a program may name only registered verbs and programs, and no cycle.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from experimental.neural_agent import requests as middle
from experimental.neural_agent.contracts import PARAM_WIDTH
from experimental.neural_agent.skill_agent.contract.catalogue import (
    DEFAULT_CATALOGUE,
    ActivityProgram,
    ArgumentSchema,
    VerbContract,
    build_verb_contract,
    load_catalogue,
    proof_of_concept_contracts,
    resolve_activities,
    verb_contracts,
)
from experimental.neural_agent.skill_agent.contract.protocol import VerbSpec, check, verb_spec
from experimental.neural_agent.skill_agent.contract.surface import scope_heads, tick_seconds
from experimental.neural_agent.skill_agent.runtime.combiner import (
    AgentContext,
    Body,
    BodyStep,
    Combiner,
    StubBody,
)

DEFAULT_HORIZON_SECONDS: float = 60.0
"""Request lifetime when the verb's contract sets no maximum duration (idle-like verbs)."""
_ID = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class RegisteredSkill:
    """One registered verb: its contract, the middle layer's spec and its executor (None: a placeholder stands in)."""

    contract: VerbContract
    spec: VerbSpec
    executor: Body | None
    done_after: int | None = 1

    @property
    def verb(self) -> str:
        return self.contract.verb

    @property
    def kind(self) -> str:
        return self.contract.kind

    @property
    def placeholder(self) -> bool:
        return self.executor is None

    def body(self, layout: str = "gen7") -> Body:
        """The executor, or a placeholder that fills the verb's heads on ``layout`` (built per layout, so one
        registry serves the gen-5 and gen-7 combiners) and reports done after ``done_after`` ticks (never
        when None)."""

        if self.executor is not None:
            return self.executor
        return placeholder_body(self.contract, layout, self.done_after)


def placeholder_body(contract: VerbContract, layout: str = "gen7", done_after: int | None = 1) -> StubBody:
    """A deterministic stand-in: it fills the verb's scope heads, reports the contract's progress signals, and on
    done (after ``done_after`` ticks) the contract's success label as the status signal."""

    heads = tuple(head for scope in contract.scopes for head in scope_heads(scope, layout))
    return StubBody(layout, done_after, heads, contract.progress, contract.contract.success_signals[0])


def script(frames: Sequence[Mapping[str, int] | Callable[[middle.ActionRequest], Mapping[str, int]]]) -> Body:
    """A command executor: frame ``i`` sets the heads on the ``i``-th tick after the grant; done after the last.

    A frame is a mapping head -> choice, or a function of the request that returns one (to read its
    arguments). The command's outcome is COMPLETED.
    """

    steps = tuple(frames)
    if not steps:
        raise ValueError("a script needs at least one frame")

    def body(request: middle.ActionRequest, ticks_since_start: int, observation_tokens: Any,
             belief_tokens: Any) -> BodyStep:
        frame = steps[min(ticks_since_start, len(steps) - 1)]
        factors = frame(request) if callable(frame) else frame
        return BodyStep(dict(factors), done=ticks_since_start + 1 >= len(steps),
                        progress={"frame": float(ticks_since_start)})

    return body


def hotbar_switch(request: middle.ActionRequest) -> Mapping[str, int]:
    """The ``switch_item`` frame: hotbar choice ``k`` selects slot ``k - 1``; scalar 0 is the 1-based slot."""

    slot = request.args.scalars[0]
    if slot is None or not 1 <= int(slot) <= 9:
        return {"hotbar_none_plus_slots": 0}
    return {"hotbar_none_plus_slots": int(slot)}


#: Commands with a real script today; every other command is a placeholder until its mechanics exist.
COMMAND_SCRIPTS: Mapping[str, Body] = {"switch_item": script([hotbar_switch])}


class SkillLibrary:
    """The registry of verbs (contract + executor) and programs."""

    def __init__(self) -> None:
        self._skills: dict[str, RegisteredSkill] = {}
        self._programs: dict[str, ActivityProgram] = {}

    # ---- building ----

    @classmethod
    def from_catalogue(cls, path: Any = DEFAULT_CATALOGUE, *, executors: Mapping[str, Body] | None = None,
                       proof_of_concept: bool = False) -> SkillLibrary:
        """Register every catalogue verb and program.

        Each verb gets ``executors[verb]`` when given, else a real command script from
        ``COMMAND_SCRIPTS``, else a placeholder; programs are registered after the verbs. With
        ``proof_of_concept`` the three proof-of-concept skills are registered too
        (``proof_of_concept_verbs``); their placeholders never report done, because the skills
        themselves run in-graph and end there.
        """

        catalogue = load_catalogue(path)
        library = cls()
        given = dict(executors or {})
        for contract in verb_contracts(catalogue):
            executor = given.get(contract.verb) or COMMAND_SCRIPTS.get(contract.verb)
            library.register(contract, executor)
        if proof_of_concept:
            for contract in proof_of_concept_verbs(catalogue):
                library.register(contract, given.get(contract.verb), done_after=None)
        programs = {program.activity: program for program in resolve_activities(catalogue)}
        for name in _program_order(programs):
            library.register_program(programs[name])
        return library

    def register(self, contract: VerbContract, executor: Body | None = None, *, replace: bool = False,
                 done_after: int | None = 1) -> RegisteredSkill:
        """Register one verb. Without an executor a placeholder stands in (done after ``done_after`` ticks).

        Raises:
            TypeError: If ``contract`` is not a VerbContract or ``executor`` is not callable.
            ValueError: If the id is malformed or taken (without ``replace``), a replacement keeps the
                version, or the scalars exceed ``PARAM_WIDTH``.
        """

        if not isinstance(contract, VerbContract):
            raise TypeError(f"contract must be a VerbContract, got {type(contract).__name__}")
        verb = contract.verb
        if not _ID.match(verb) or verb in self._programs:
            raise ValueError(f"verb id {verb!r} is malformed or names a program")
        if executor is not None and not callable(executor):
            raise TypeError(f"the executor of {verb!r} must be callable")
        if len(contract.arguments.scalars) > PARAM_WIDTH:
            raise ValueError(f"{verb!r} names {len(contract.arguments.scalars)} scalars; the limit is {PARAM_WIDTH}")
        old = self._skills.get(verb)
        if old is not None:
            if not replace:
                raise ValueError(f"verb {verb!r} is registered; pass replace=True with a new contract version")
            if old.contract.contract.version == contract.contract.version:
                raise ValueError(f"replacing {verb!r} needs a new version (frozen skills): it is still "
                                 f"{contract.contract.version!r}")
        skill = RegisteredSkill(contract, verb_spec(contract), executor, done_after)
        self._skills[verb] = skill
        return skill

    def create(self, verb: str, *, description: str, scopes: Sequence[str], required: Sequence[str] = (),
               any_of: Sequence[str] = (), scalars: Sequence[str] = (), done: str = "done", kind: str = "L",
               version: str = "v1", body: Body | None = None, replace: bool = False, **control: Any) -> VerbContract:
        """Create and register a skill that the catalogue does not list (the library grows; owner).

        Args:
            verb: The id, lowercase with underscores.
            description: What it does, for the middle layer's matching.
            scopes: The scopes its one request owns.
            required, any_of: Argument kinds (position, direction, entity, cell; or context kinds).
            scalars: Its named numbers, at most PARAM_WIDTH.
            done: The done text; its label is the success signal.
            kind: "L" learned or "C" command.
            version: The contract version; a replacement needs a new one.
            body: The executor; None registers a placeholder.
            control: minimum_commitment_seconds, maximum_duration_seconds, emergency_events.
        """

        arguments = ArgumentSchema(tuple(required), tuple(any_of), tuple(scalars), done, "created")
        contract = build_verb_contract(verb, arguments, tuple(scopes), kind, 0, f"{verb}: {description}.",
                                       version=version, **control)
        self.register(contract, body, replace=replace)
        return contract

    def register_program(self, program: ActivityProgram, *, replace: bool = False) -> None:
        """Register an activity program; every step must name a registered verb or program.

        Raises:
            ValueError: If the id is taken (without ``replace``), a step names nothing registered, a
                nested program shares its stage with another step, or the program would form a cycle.
        """

        name = program.activity
        if name in self._skills or (name in self._programs and not replace):
            raise ValueError(f"program id {name!r} is taken")
        for stage in program.stages:
            for step in stage:
                known = [verb for verb in step.verbs if verb in self._skills or verb in self._programs]
                if not known:
                    raise ValueError(f"program {name!r} step {step.text!r} names no registered verb or program")
                if any(verb in self._programs for verb in known[:1]) and len(stage) > 1:
                    raise ValueError(f"program {name!r}: nested program {step.text!r} must be a stage of its own")
        self._programs[name] = program
        try:
            self.expand(name)
        except ValueError:
            del self._programs[name]
            raise

    # ---- reading ----

    def __contains__(self, verb: str) -> bool:
        return verb in self._skills

    def __getitem__(self, verb: str) -> RegisteredSkill:
        try:
            return self._skills[verb]
        except KeyError:
            raise KeyError(f"no skill {verb!r}") from None

    @property
    def verbs(self) -> tuple[str, ...]:
        return tuple(self._skills)

    @property
    def programs(self) -> Mapping[str, ActivityProgram]:
        return dict(self._programs)

    def specs(self) -> dict[str, VerbSpec]:
        return {verb: skill.spec for verb, skill in self._skills.items()}

    def executors(self, layout: str = "gen7") -> dict[str, Body]:
        return {verb: skill.body(layout) for verb, skill in self._skills.items()}

    def placeholders(self) -> tuple[str, ...]:
        """Verbs whose executor is a stand-in (no trained network or written script yet)."""

        return tuple(verb for verb, skill in self._skills.items() if skill.placeholder)

    def descriptions(self) -> dict[str, str]:
        """Every verb's and program's description: the text the middle layer matches intents against."""

        texts = {verb: skill.contract.description for verb, skill in self._skills.items()}
        texts.update({name: program.description for name, program in self._programs.items()})
        return texts

    def expand(self, name: str, _path: tuple[str, ...] = ()) -> tuple[tuple[Any, ...], ...]:
        """A program's stages with nested programs inlined: a tuple of stages of (step, verb) pairs.

        Raises:
            ValueError: If the program names itself, directly or through another program.
        """

        if name in _path:
            raise ValueError(f"program cycle: {' -> '.join((*_path, name))}")
        stages: list[tuple[Any, ...]] = []
        for stage in self._programs[name].stages:
            pairs = []
            for step in stage:
                verb = next(verb for verb in step.verbs if verb in self._skills or verb in self._programs)
                if verb in self._programs:
                    stages.extend(self.expand(verb, (*_path, name)))
                    break
                pairs.append((step, verb))
            else:
                stages.append(tuple(pairs))
        return tuple(stages)

    # ---- using ----

    def combiner(self, layout: str = "gen7") -> Combiner:
        """A combiner over every registered verb."""

        return Combiner(self.specs(), self.executors(layout), layout=layout)

    def request(self, verb: str, *, agent: int, request_id: int, tick: int, horizon_seconds: float | None = None,
                position: Sequence[float] | None = None, direction: Sequence[float] | None = None,
                entity: int | None = None, cell: Sequence[int] | None = None, face: str | None = None,
                scalars: Mapping[str, float] | Sequence[float | None] | None = None,
                args: middle.RequestArgs | None = None,
                mode: str = middle.MODE_PARALLEL, urgency: float = 0.5, confidence: float = 0.5,
                priority: float = 1.0, goal_slot: int = -1, executive_version: int = 0,
                program_step: int = 0, layout: str = "gen7") -> middle.ActionRequest:
        """Build a request for ``verb``: scalars by name, expiry from the contract's maximum duration.

        ``args`` (a whole RequestArgs) replaces the argument keywords when given. The expiry counts ticks
        of ``layout`` (1/30 s on gen 7, 0.045 s on gen 5).

        Raises:
            KeyError: If the verb is not registered or a scalar name is not the verb's.
        """

        contract = self[verb].contract.contract
        names = contract.param_names
        if isinstance(scalars, Mapping):
            unknown = sorted(set(scalars) - set(names))
            if unknown:
                raise KeyError(f"{verb!r} has no scalar(s) {unknown}; its scalars are {names}")
            values: tuple[float | None, ...] = tuple(scalars.get(name) for name in names)
        else:
            values = tuple(scalars or ())
        horizon = horizon_seconds
        if horizon is None:
            horizon = contract.maximum_duration_seconds
            if math.isinf(horizon):
                horizon = DEFAULT_HORIZON_SECONDS
        expires = tick + max(1, math.ceil(horizon / tick_seconds(layout) - 1e-9))
        if args is None:
            args = middle.RequestArgs(position=position, direction=direction, entity=entity,
                                      cell=cell, face=face, scalars=values)
        return middle.ActionRequest(request_id=request_id, agent=agent, issued_tick=tick, expires_tick=expires,
                                    verb=verb, args=args, mode=mode, urgency=urgency, confidence=confidence,
                                    priority=priority, goal_slot=goal_slot, executive_version=executive_version,
                                    program_step=program_step)

    def check(self, request: middle.ActionRequest, context: AgentContext, *, tick: int,
              done_ids: Iterable[int] = ()) -> str | None:
        """The combiner's check for one request, outside the combiner (for the middle layer's own use)."""

        skill = self._skills.get(request.verb)
        return check(request, None if skill is None else skill.spec, tick=tick, world_bounds=context.world_bounds,
                     known_entities=context.known_entities, legal_scopes=context.legal_scopes,
                     done_ids=frozenset(done_ids))

    def example_args(self, verb: str, *, origin: Sequence[float] = (0.0, 64.0, 0.0), entity: int = 7,
                     scalar: float = 1.0) -> middle.RequestArgs:
        """An argument set that satisfies the verb's schema: for smoke tests and demonstrations.

        Every required kind and the first kind of every any-of group is given: a position near
        ``origin``, the unit direction (1, 0, 0), ``entity``, and the cell under ``origin`` with its up
        face; every named scalar is ``scalar``.
        """

        schema = self[verb].spec.schema
        wanted = set(schema.needs()) | {group[0] for group in schema.any_of}
        x, y, z = (float(value) for value in origin)
        names = self[verb].contract.contract.param_names
        return middle.RequestArgs(
            position=(x + 1.0, y, z + 1.0) if "position" in wanted else None,
            direction=(1.0, 0.0, 0.0) if "direction" in wanted else None,
            entity=entity if "entity" in wanted else None,
            cell=(int(x), int(y) - 1, int(z)) if "cell" in wanted else None,
            face="up" if "cell" in wanted else None,
            scalars=tuple(scalar for _ in names),
        )

    def scope_heads(self, verb: str, layout: str = "gen7") -> tuple[str, ...]:
        """The heads a verb's request owns on ``layout``."""

        return tuple(head for scope in self[verb].contract.scopes for head in scope_heads(scope, layout))


def proof_of_concept_verbs(catalogue: Any) -> tuple[VerbContract, ...]:
    """The three proof-of-concept skills (idle, pursuit, evader) as the middle layer's executor requests them.

    Their ids, control (commitment, duration, emergency events) and done text come from the catalogue
    (``catalogue.proof_of_concept_contracts``, tested against the skills themselves). Their arguments and
    scopes come from the middle layer's request registry (``requests.ARGUMENT_SCHEMAS``: idle nothing,
    pursuit and evader the target entity or its remembered position), which is also what the in-graph parity path
    grants, so the host statuses and the compiled action name the same heads. Both the required arguments and the
    ``any_of`` groups are copied: a dropped group would leave a verb requiring nothing.
    """

    built = []
    for base in proof_of_concept_contracts(catalogue):
        schema = middle.ARGUMENT_SCHEMAS[base.verb]
        required = tuple(kind for kind in middle.ARGUMENT_NAMES if getattr(schema, kind))
        any_of = tuple(name for group in schema.any_of for name in group)
        arguments = ArgumentSchema(required, any_of, (), base.arguments.done, "request registry")
        control = base.contract
        built.append(build_verb_contract(
            base.verb, arguments, schema.scopes, base.kind, base.section, base.description,
            minimum_commitment_seconds=control.minimum_commitment_seconds,
            maximum_duration_seconds=control.maximum_duration_seconds,
            emergency_events=control.emergency_events))
    return tuple(built)


def _program_order(programs: Mapping[str, ActivityProgram]) -> list[str]:
    """Programs ordered so that every nested program is registered before the programs that use it."""

    order: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in order:
            return
        if name in visiting:
            raise ValueError(f"program cycle through {name!r}")
        visiting.add(name)
        for step in programs[name].steps:
            for verb in step.verbs:
                if verb in programs and verb != name:
                    visit(verb)
        visiting.discard(name)
        order.append(name)

    for name in programs:
        visit(name)
    return order


__all__ = [
    "COMMAND_SCRIPTS",
    "DEFAULT_HORIZON_SECONDS",
    "RegisteredSkill",
    "SkillLibrary",
    "hotbar_switch",
    "placeholder_body",
    "proof_of_concept_verbs",
    "script",
]
