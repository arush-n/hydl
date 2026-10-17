"""DEPRECATED (2026-10-17): superseded by J-S1/J-S2 (only the top LLM may be external); kept with J-LM, though no arm offers it.

Anthropic backend for the quick language-model JEV: one Messages API call per decision.

The call follows the installed anthropic SDK (1.15.0). The structured reply is
requested with ``output_config.format`` ``{"type": "json_schema", "schema": ...}``,
the form the bundled Claude API doc shows for ``messages.create`` (python/claude-api/
tool-use.md, "Structured Outputs"). ``effort`` goes in ``output_config.effort``
(the SDK's OutputConfigParam). Thinking is left at the default, which is adaptive.

Two choices are made here, not by the SDK:

* The default client is built with ``max_retries=0``. One decision is one request,
  so the per-decision timeout is the latency budget. An injected client is used
  as given.
* The bundled doc lists numeric limits, string limits and complex array limits as
  not supported in structured outputs (shared/tool-use-concepts.md, "JSON Schema
  Limitations"). Its automatic removal is documented for the SDK's ``parse``
  helpers, not for ``create``. So those keywords are removed from a copy of the
  schema before sending. :func:`parse_decision` still enforces them.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from experimental.neural_agent.jev.lm.prompting import SYSTEM_PROMPT
from experimental.neural_agent.jev.lm.protocol import validate_timeout

EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS: int = 1024
UNSUPPORTED_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }
)


def anthropic_output_schema(schema: Any) -> Any:
    """Return a copy of a JSON Schema without the keywords structured outputs reject.

    Keys are matched by name at every depth. A property with one of these names
    would also be removed, so keep such names out of the schema. The decision
    schema has none.

    Args:
        schema: A JSON Schema value (dict, list or scalar). It is not modified.

    Returns:
        The same structure with the unsupported keywords removed.
    """

    if isinstance(schema, dict):
        return {
            key: anthropic_output_schema(value)
            for key, value in schema.items()
            if key not in UNSUPPORTED_SCHEMA_KEYWORDS
        }
    if isinstance(schema, list):
        return [anthropic_output_schema(item) for item in schema]
    return schema


class AnthropicQuickBackend:
    """A :class:`DecisionLM` that calls the Anthropic Messages API.

    Attributes:
        model: Model id, default ``'claude-opus-5-5'``. It is a constructor
            parameter so that an experiment can pin or change it.
        effort: Effort level in ``EFFORT_LEVELS``, default ``'low'``.
        last_usage: Token usage of the most recent successful call, or None before
            the first call. Keys are ``input_tokens``, ``output_tokens`` and
            ``cache_read_input_tokens``. Values are ints, or None when the reply
            does not report them.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "low",
        *,
        client: Any | None = None,
    ) -> None:
        """Store the model and effort, and bind a client.

        Args:
            model: Non-empty model id.
            effort: One of ``EFFORT_LEVELS``.
            client: An object with ``messages.create``, for example a test fake.
                When None, a default ``anthropic.Anthropic(max_retries=0)`` is
                built. Building it does not contact the network.

        Raises:
            ValueError: If ``model`` is empty or ``effort`` is not in ``EFFORT_LEVELS``.
        """

        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}, got {effort!r}")
        self.model = model
        self.effort = effort
        self.last_usage: dict[str, int | None] | None = None
        self._client: Any = client if client is not None else anthropic.Anthropic(max_retries=0)

    def decide(self, prompt: str, schema: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """Send one decision request and return the decoded reply.

        Args:
            prompt: The user message.
            schema: The JSON Schema the reply must satisfy. A copy with the
                unsupported keywords removed is sent.
            timeout_s: Per-request timeout in seconds. Finite and positive.

        Returns:
            The decision object parsed from the reply's single text block.

        Raises:
            TimeoutError: If the request times out.
            ValueError: If the reply was cut off at ``max_tokens`` or refused, has
                no single text block, or the text is not a JSON object.
        """

        timeout = validate_timeout(timeout_s)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": self.effort,
                    "format": {
                        "type": "json_schema",
                        "schema": anthropic_output_schema(schema),
                    },
                },
                timeout=timeout,
            )
        except anthropic.APITimeoutError as error:
            raise TimeoutError(f"anthropic did not reply within {timeout} s") from error
        self.last_usage = _usage_of(response)
        return _decision_from_response(response)


def _usage_of(response: Any) -> dict[str, int | None]:
    """Return the usage fields this module records."""

    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
    }


def _decision_from_response(response: Any) -> dict[str, Any]:
    """Return the decision object in the reply's single text block."""

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "max_tokens":
        raise ValueError("anthropic reply was cut off at max_tokens; the decision is incomplete")
    if stop_reason == "refusal":
        raise ValueError("anthropic refused to answer this decision")
    texts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    if len(texts) != 1:
        raise ValueError(f"expected one text block in the reply, got {len(texts)}")
    try:
        decision = json.loads(texts[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"anthropic text block is not JSON: {error}") from error
    if not isinstance(decision, dict):
        raise ValueError("anthropic text block is not a JSON object")
    return decision
