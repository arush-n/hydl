"""KEV: an owner-named example of a JEV instance, not a required component.

Owner's definition (2026-10-17): "KEV is a JEV clone which is just a quantized
Qwen model fine-tuned to focus on quick decision making." Owner's scope note the
same day: "you do not need to use kev, kev was just an example just like jev."

So KEV names one point in the JEV instance ladder (ADR-001 D2): a quick
language-model decision instance whose model is a quantized Qwen, fine-tuned on
decision data. The repository already runs that instance's baseline without the
fine-tune: ``jev/lm/instance.py`` ``LanguageModelJev`` over
``jev/lm/ollama_qwen.py`` ``OllamaQwenBackend`` (``qwen3:4b``, Ollama's default
quantization), the ``j_lm`` arm with ``--lm-backend ollama``. Its measured cost
is 471 ms per decision at p50 against a 20 ms host budget (``docs/LEDGER.md``),
which is why no language-model instance sits on the decision layer's critical
path: the compiled J1 network decides in about 0.1 ms (``probes/probe_jev_latency.py``).

Nothing here is built on purpose. A fine-tuned KEV would be a new Ollama model
tag passed to ``OllamaQwenBackend(model=...)``; the supervised data for it is the
decision dataset ``eval/record_traces.py`` writes, rendered through
``jev/lm/prompting.py``.
"""

from __future__ import annotations

#: The owner's definition, verbatim.
KEV_DEFINITION: str = (
    "KEV is a JEV clone which is just a quantized Qwen model fine-tuned to focus on "
    "quick decision making."
)

#: The un-finetuned baseline that exists today, as an Ollama model tag.
KEV0_MODEL: str = "qwen3:4b"


class KevJev:
    """Placeholder that refuses construction and names the existing baseline instead."""

    name: str = "kev"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "KEV is an example instance, not a required component (owner, 2026-10-17). "
            "Use LanguageModelJev over OllamaQwenBackend(model='qwen3:4b') for the "
            "quantized-Qwen baseline, or a fine-tuned model tag in its place."
        )


__all__ = ["KEV0_MODEL", "KEV_DEFINITION", "KevJev"]
