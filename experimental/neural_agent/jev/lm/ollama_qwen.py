"""DEPRECATED (2026-10-17): superseded by J-S1/J-S2 (only the top LLM may be external); kept because the j_lm experiment arm runs it.

Ollama backend for the quick language-model JEV: qwen3:4b over POST /api/chat.

Owner measurements, quoted and not checked here: a warm generation of a very short
output takes about 89 ms, and a cold load of the model takes about 39 s.

Measured here on 2026-10-16 (Ollama 0.33.3, qwen3:4b Q4_K_M): a full decision of
about 75 output tokens took 3.7 to 9.9 s wall, so the 2 s budget is not met in this
configuration. ``think: false`` did not stop a thinking monologue in free-form
output. The ``format`` grammar is what kept every decision reply to JSON. SKILL.md
has the probes and the options.

The backend still sends ``think: false``, as specified.

The request fields (``model``, ``messages``, ``think``, ``format``, ``stream``,
``keep_alive``, ``options``) follow the Ollama REST API (cited in SKILL.md). The
backend uses only ``urllib``, so it adds no dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from experimental.neural_agent.jev.lm.prompting import SYSTEM_PROMPT
from experimental.neural_agent.jev.lm.protocol import validate_timeout


class OllamaQwenBackend:
    """A :class:`DecisionLM` that calls a local Ollama server.

    Attributes:
        model: Ollama model tag, default ``'qwen3:4b'``.
        base_url: Server root, default ``'http://127.0.0.1:11434'``. No trailing slash.
        keep_alive: How long the model stays resident after a request, default ``'30m'``.
        num_predict: Maximum generated tokens, default 40 (measured 2026-10-17: a 96-token reply cost 1.5 s on qwen3:4b, almost all in the free-text reason field). A decision cut off at
            this limit is not valid JSON, so it fails closed.
    """

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
        keep_alive: str = "30m",
        num_predict: int = 40,
    ) -> None:
        """Store the server and model settings.

        Args:
            model: Non-empty Ollama model tag.
            base_url: Non-empty server root. A trailing slash is removed.
            keep_alive: Non-empty keep-alive duration, for example ``'30m'``.
            num_predict: Positive int, the generated-token cap.

        Raises:
            ValueError: If a string is empty or ``num_predict`` is not a positive int.
        """

        self.model = _non_empty(model, "model")
        self.base_url = _non_empty(base_url, "base_url").rstrip("/")
        self.keep_alive = _non_empty(keep_alive, "keep_alive")
        if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict < 1:
            raise ValueError(f"num_predict must be a positive int, got {num_predict!r}")
        self.num_predict = num_predict

    def request_body(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return the JSON body of one ``/api/chat`` request.

        The body has ``model``; ``messages`` (a system message and the user prompt);
        ``think`` false; ``format`` set to the schema, which constrains decoding;
        ``stream`` false; ``keep_alive``; and ``options`` with ``temperature`` 0
        and ``num_predict``.

        Args:
            prompt: The user message.
            schema: The JSON Schema the reply must satisfy.

        Returns:
            A JSON-serialisable dict.
        """

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "think": False,
            "format": schema,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_predict": self.num_predict},
        }

    def decide(self, prompt: str, schema: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """Send one decision request and return the decoded reply.

        Args:
            prompt: The user message.
            schema: The JSON Schema the reply must satisfy.
            timeout_s: Seconds to wait for the HTTP reply. Finite and positive.

        Returns:
            The decision object parsed from ``message.content``.

        Raises:
            TimeoutError: If the server does not reply within ``timeout_s``.
            RuntimeError: If the server is unreachable, returns an HTTP error, or
                returns an ``error`` field.
            ValueError: If the reply, or the content inside it, is not a JSON object.
        """

        timeout = validate_timeout(timeout_s)
        data = json.dumps(self.request_body(prompt, schema), allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _decision_from_reply(self._send(request, timeout))

    def _send(self, request: urllib.request.Request, timeout: float) -> bytes:
        """Return the raw reply bytes, mapping transport failures to the documented errors."""

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"ollama returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TimeoutError(f"ollama did not reply within {timeout} s") from error
            raise RuntimeError(f"ollama is unreachable at {self.base_url}: {error.reason}") from error
        except TimeoutError as error:
            raise TimeoutError(f"ollama did not reply within {timeout} s") from error


def _decision_from_reply(raw: bytes) -> dict[str, Any]:
    """Decode an ``/api/chat`` reply and return the decision object in ``message.content``."""

    try:
        reply = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"ollama reply is not JSON: {error}") from error
    if not isinstance(reply, dict):
        raise ValueError("ollama reply is not a JSON object")
    if "error" in reply:
        raise RuntimeError(f"ollama error: {reply['error']}")
    message = reply.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("ollama reply has no message.content string")
    try:
        decision = json.loads(message["content"])
    except json.JSONDecodeError as error:
        raise ValueError(f"message.content is not JSON: {error}") from error
    if not isinstance(decision, dict):
        raise ValueError("message.content is not a JSON object")
    return decision


def _non_empty(value: Any, name: str) -> str:
    """Return ``value`` when it is a non-empty string."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
