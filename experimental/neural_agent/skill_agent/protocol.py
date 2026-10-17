"""The lower layer's side of the request protocol: catalogue contracts as the middle layer's schemas.

``docs/SYSTEM_ONE.md`` section 9.10 and ``AGENT_BRIEF.md`` A3. The records (``ActionRequest``,
``RequestArgs``, ``RequestStatus``) and the check (``validate``) belong to the middle layer, in
``experimental/neural_agent/requests.py``; the split between the two sessions was agreed on
2026-10-17. This module supplies ``validate(..., schemas=...)`` from the catalogue and decides which
scopes the action mask leaves legal. A catalogue verb's carried ``required`` kinds become the schema's
flags, its carried ``any_of`` kinds one ``any_of`` group (``requests.ArgumentSchema.any_of``, added for
this on 2026-10-17, so the contracts adapt without loss), and its scopes the schema's scopes (the
catalogue is the source of truth for a verb's scopes).

Faces. ``surface.FACES`` is ``("none", *requests.FACE_NAMES)``: surface index ``k + 1`` is requests
index ``k``. A request's face is None when it gives no cell; a token's face is ``"none"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Iterable, Mapping, Sequence

from experimental.neural_agent import requests as protocol
from experimental.neural_agent.contracts import SkillContract
from experimental.neural_agent.skill_agent.catalogue import CARRIED_KINDS, VerbContract
from experimental.neural_agent.skill_agent.surface import FACES, SCOPE_NAMES, scope_of

if FACES != ("none", *protocol.FACE_NAMES):
    raise RuntimeError(f"surface.FACES {FACES} must be ('none', *requests.FACE_NAMES {protocol.FACE_NAMES})")
if tuple(protocol.SCOPES) != SCOPE_NAMES:
    raise RuntimeError(f"requests.SCOPES {protocol.SCOPES} must equal surface.SCOPE_NAMES {SCOPE_NAMES}")

def token_face(face: str | None) -> str:
    """Return the token face (``surface.FACES``) of a request face (``requests.FACE_NAMES`` or None)."""

    if face is None:
        return "none"
    if face not in protocol.FACE_NAMES:
        raise ValueError(f"unknown request face {face!r}; expected one of {protocol.FACE_NAMES}")
    return face


def request_face(face: str) -> str | None:
    """Return the request face of a token face: None for ``"none"``."""

    if face not in FACES:
        raise ValueError(f"unknown token face {face!r}; expected one of {FACES}")
    return None if face == "none" else face


@dataclass(frozen=True)
class VerbSpec:
    """What the combiner needs about one verb: its contract and the middle layer's schema."""

    contract: SkillContract
    schema: protocol.ArgumentSchema

    @property
    def verb(self) -> str:
        return self.contract.skill_id

    @property
    def scopes(self) -> tuple[str, ...]:
        return tuple(self.schema.scopes)


def verb_spec(verb: VerbContract) -> VerbSpec:
    """Lower one catalogue verb: carried required kinds become flags, carried any-of kinds one group."""

    required = set(verb.arguments.required) & set(CARRIED_KINDS)
    group = tuple(kind for kind in verb.arguments.any_of if kind in CARRIED_KINDS)
    schema = protocol.ArgumentSchema(
        position="position" in required,
        direction="direction" in required,
        entity="entity" in required,
        cell="cell" in required,
        scopes=tuple(verb.scopes),
        any_of=(group,) if group else (),
    )
    return VerbSpec(verb.contract, schema)


def verb_specs(verbs: Iterable[VerbContract]) -> dict[str, VerbSpec]:
    """Lower every verb, keyed by verb id.

    Raises:
        ValueError: If two contracts name the same verb.
    """

    specs: dict[str, VerbSpec] = {}
    for verb in verbs:
        if verb.verb in specs:
            raise ValueError(f"verb {verb.verb!r} appears twice")
        specs[verb.verb] = verb_spec(verb)
    return specs


def request_schemas(specs: Mapping[str, VerbSpec]) -> dict[str, protocol.ArgumentSchema]:
    """The ``schemas=`` mapping for ``requests.validate``, keyed by verb."""

    return {verb: spec.schema for verb, spec in specs.items()}


def check(
    request: protocol.ActionRequest,
    spec: VerbSpec | None,
    *,
    tick: int,
    world_bounds: Sequence[Sequence[float]],
    known_entities: Collection[int],
    legal_scopes: Collection[str],
    done_ids: Collection[int],
) -> str | None:
    """Return why the combiner must reject ``request`` now, or None: ``requests.validate`` with its schema.

    Args:
        request: The request.
        spec: The verb's spec, or None when the library has no such verb.
        tick, world_bounds, known_entities, legal_scopes, done_ids: As in ``requests.validate``.
    """

    contract = None if spec is None else spec.contract
    schemas: Mapping[str, protocol.ArgumentSchema] = {} if spec is None else {request.verb: spec.schema}
    return protocol.validate(request, contract, tick=tick, world_bounds=world_bounds,
                             known_entities=known_entities, legal_scopes=legal_scopes, done_ids=done_ids,
                             schemas=schemas)


def legal_scopes(head_masks: Mapping[str, Sequence[bool]]) -> frozenset[str]:
    """Return the scopes legal under the action mask: every present head of the scope has a legal choice.

    Args:
        head_masks: The present heads (names from ``surface``) -> their per-choice legality.

    A scope with no present head (``interaction`` on the gen-5 surface) is not legal.
    """

    by_scope: dict[str, list[bool]] = {scope: [] for scope in SCOPE_NAMES}
    for head, choices in head_masks.items():
        by_scope[scope_of(head)].append(any(bool(choice) for choice in choices))
    return frozenset(scope for scope, heads in by_scope.items() if heads and all(heads))


__all__ = [
    "VerbSpec",
    "check",
    "legal_scopes",
    "request_face",
    "request_schemas",
    "token_face",
    "verb_spec",
    "verb_specs",
]
