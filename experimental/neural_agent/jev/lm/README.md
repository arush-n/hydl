# jev/lm/ — J-LM, a quick language model as the decision instance (DEPRECATED 2026-10-17)

Superseded by J-S1 / J-S2 (`../system_one/`): the owner's rule is that nothing external is on the path
except the top LLM, and a per-decision language model is neither fast enough nor needed once the typed
fields carry the decision. Kept because the `j_lm` comparison arm (`../../eval/arms.py`) still runs it and
its measurements are in the ledger (3.7 to 9.9 s per decision through Ollama).

| File | Lines | What it is |
|---|---:|---|
| `instance.py` | 500 | `LanguageModelJev`, the host-mode instance over a backend. |
| `prompting.py` | 294 | Prompt, JSON Schema and fail-closed parsing of the reply. |
| `protocol.py` | 140 | The backend protocol. |
| `ollama_qwen.py` | 171 | Local Ollama backend (allowed: local open weights). |

Gate: `test_na_jev_lm.py` (the live Ollama check runs only with `NEURAL_AGENT_LIVE_OLLAMA=1`).

Removed 2026-10-18 (owner-approved cleanup): `anthropic_quick.py`, the Anthropic backend no arm offered.
