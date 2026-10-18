"""The requests section of S_t: the live typed requests to the skills, written by the decision layer.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping


from experimental.neural_agent.executive.store import validate_against_schema
from experimental.neural_agent.requests import ActionRequest, RequestArgs, RequestStatus

from .._common import (
    REQUEST_LIMIT,
    _instances,
    _outcome_named,
    _positive_int,
)
from ..schema import (
    _REQUESTS_SECTION_SCHEMA,
)


# ---- requests section: written by the JEV (the decision layer) ----


def _request_to_dict(request: ActionRequest) -> dict[str, Any]:
    """Return the JSON form of a request: absent arguments are null, the mode is its string."""

    args = request.args
    mapping = {item.name: getattr(request, item.name) for item in dataclasses.fields(ActionRequest)}
    mapping["args"] = {
        "position": None if args.position is None else list(args.position),
        "direction": None if args.direction is None else list(args.direction),
        "entity": args.entity,
        "cell": None if args.cell is None else list(args.cell),
        "face": args.face,
        "scalars": list(args.scalars),
    }
    return mapping


def _request_from_dict(item: Mapping[str, Any]) -> ActionRequest:
    """Build a request from its JSON form (already checked against the schema)."""

    args = item["args"]
    return ActionRequest(
        **{key: value for key, value in item.items() if key != "args"},
        args=RequestArgs(
            position=args["position"],
            direction=args["direction"],
            entity=args["entity"],
            cell=args["cell"],
            face=args["face"],
            scalars=tuple(args["scalars"]),
        ),
    )


def _status_to_dict(status: RequestStatus) -> dict[str, Any]:
    """Return the JSON form of a status: the outcome by name, the progress as name/value pairs."""

    return {
        "agent": status.agent,
        "request_id": status.request_id,
        "kind": status.kind,
        "tick": status.tick,
        "reason": status.reason,
        "conflict_with": status.conflict_with,
        "outcome": None if status.outcome is None else status.outcome.name,
        "progress": None if status.progress is None else [{"name": n, "value": v} for n, v in status.progress],
        "heads": list(status.heads),
        "signal": status.signal,
    }


def _status_from_dict(item: Mapping[str, Any]) -> RequestStatus:
    """Build a status from its JSON form (already checked against the schema)."""

    progress = item["progress"]
    return RequestStatus(
        agent=item["agent"],
        request_id=item["request_id"],
        kind=item["kind"],
        tick=item["tick"],
        reason=item["reason"],
        conflict_with=item["conflict_with"],
        outcome=None if item["outcome"] is None else _outcome_named(item["outcome"]),
        progress=None if progress is None else tuple((pair["name"], pair["value"]) for pair in progress),
        heads=tuple(item["heads"]),
        signal=item.get("signal", ""),
    )


@dataclass(frozen=True)
class RequestsSection:
    """The decision layer's live requests to the lower layer (SYSTEM_ONE.md section 9.10).

    Attributes:
        live: The live requests of every agent, stored sorted by (agent, request_id). A request id
            appears once per agent. Building the section keeps the ``limit`` highest ids per agent.
            Every argument number must be finite: S_t is JSON, which holds no NaN. A non-finite
            request is the combiner's to reject (``requests.validate``), and it never reaches S_t.
        limit: The bound per agent, at least 1. Default ``REQUEST_LIMIT`` (16).
    """

    live: tuple[ActionRequest, ...] = ()
    limit: int = REQUEST_LIMIT

    def __post_init__(self) -> None:
        """Validate the requests, sort them and keep the newest ``limit`` per agent."""

        limit = _positive_int(self.limit, "RequestsSection.limit")
        requests = _instances(self.live, ActionRequest, "RequestsSection.live")
        for index, request in enumerate(requests):
            if not all(math.isfinite(value) for value in request.args.numbers()):
                raise ValueError(
                    f"RequestsSection.live[{index}] (agent {request.agent}, request {request.request_id}) holds a "
                    "non-finite argument; S_t holds finite numbers only"
                )
        ordered = sorted(requests, key=lambda request: (request.agent, request.request_id))
        keys = [(request.agent, request.request_id) for request in ordered]
        if len(set(keys)) != len(keys):
            raise ValueError("RequestsSection.live repeats a request id for one agent")
        kept: list[ActionRequest] = []
        for agent in sorted({request.agent for request in ordered}):
            kept.extend([request for request in ordered if request.agent == agent][-limit:])
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "live", tuple(kept))

    def for_agent(self, agent: int) -> tuple[ActionRequest, ...]:
        """Return the live requests of ``agent``, oldest id first."""

        return tuple(request for request in self.live if request.agent == agent)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping."""

        return {"live": [_request_to_dict(request) for request in self.live], "limit": self.limit}

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> RequestsSection:
        """Parse the JSON-shaped mapping, schema first."""

        validate_against_schema(mapping, _REQUESTS_SECTION_SCHEMA, "$.requests")
        return cls(live=tuple(_request_from_dict(item) for item in mapping["live"]), limit=mapping["limit"])

