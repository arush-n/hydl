"""Decision language-model protocol and a scripted fake, for the J-LM instance.

ADR-001 D2 defines the quick language-model instance (J-LM) as a JEV that reads
B_t and E_t and emits the same ``ManagedGoal`` under the same latency budget.
This module holds the seam between that instance and a concrete model. The
instance depends only on :class:`DecisionLM`, so the Ollama backend, the
Anthropic backend and the test fake are interchangeable.

One call is one decision: a prompt, the JSON Schema the reply must satisfy, and
a timeout in seconds. The reply is a plain dict. Its content is validated later
by ``prompting.parse_decision``, fail closed, not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol, runtime_checkable


@runtime_checkable
class DecisionLM(Protocol):
    """A language model that maps one prompt to one JSON decision object.

    Implementations must send ``prompt`` as the user message, constrain the reply
    to ``schema``, honour ``timeout_s``, and return the reply as a dict parsed
    from JSON. A reply that exceeds the timeout raises ``TimeoutError``, so the
    caller can count the miss (ADR-001 D2).
    """

    def decide(
        self,
        prompt: str,
        schema: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        """Return one decision object for ``prompt``.

        Args:
            prompt: The user message, built by ``prompting.build_decision_prompt``.
            schema: A JSON Schema object that the reply must satisfy.
            timeout_s: Wall-clock limit in seconds. Finite and positive.

        Returns:
            The decoded JSON object, as a dict.

        Raises:
            TimeoutError: If no reply arrives within ``timeout_s``.
        """


def validate_timeout(timeout_s: Any) -> float:
    """Return ``timeout_s`` as a float after checking that it is finite and positive.

    Args:
        timeout_s: A timeout in seconds. Booleans are rejected.

    Returns:
        The timeout as a Python float, greater than zero.

    Raises:
        TypeError: If ``timeout_s`` is not a real number.
        ValueError: If it is not finite or not greater than zero.
    """

    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError(f"timeout_s must be a real number, got {type(timeout_s).__name__}")
    value = float(timeout_s)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"timeout_s must be finite and positive, got {value}")
    return value


class FakeCall(NamedTuple):
    """One recorded call to :meth:`FakeDecisionLM.decide`.

    Attributes:
        prompt: The prompt text that was sent.
        schema: The JSON Schema that was requested.
        timeout_s: The timeout that was passed.
    """

    prompt: str
    schema: dict[str, Any]
    timeout_s: float


class FakeDecisionLM:
    """A scripted :class:`DecisionLM` for tests. No network and no model.

    Each call consumes the next script entry. A mapping is returned as a fresh
    dict. An exception instance is raised, which scripts a timeout or a failure.
    A call after the script is exhausted raises ``RuntimeError``, so a test that
    makes more calls than it scripted fails loudly.

    Attributes:
        calls: One :class:`FakeCall` per call made, in order.
    """

    def __init__(self, script: Sequence[Mapping[str, Any] | BaseException]) -> None:
        """Store the script.

        Args:
            script: Entries consumed one per call. Each entry is a mapping to
                return, or an exception instance to raise.
        """

        self._script = list(script)
        self.calls: list[FakeCall] = []

    def decide(
        self,
        prompt: str,
        schema: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        """Record the call, then return or raise the next scripted entry.

        Args:
            prompt: The prompt text. It is recorded, not read.
            schema: The schema. It is recorded, not checked.
            timeout_s: The timeout. It is recorded, not enforced.

        Returns:
            A copy of the next mapping in the script.

        Raises:
            RuntimeError: If the script is exhausted.
            BaseException: The next script entry, if it is an exception.
        """

        self.calls.append(FakeCall(prompt, schema, timeout_s))
        if not self._script:
            raise RuntimeError("FakeDecisionLM script is exhausted")
        entry = self._script.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return dict(entry)
