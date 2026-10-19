# skill_agent/model/ — the networks

The trunk every body reads its tokens through, the per-verb adapters that keep frozen skills bit-identical, the
request-conditioned generalist (the working design, `../docs/ARCHITECTURE_RESEARCH.md`), and the adapter from the
simulator's legal observation to the trunk's token sets. Pure JAX: parameters are NamedTuples of arrays.

| Module | Lines | What it holds |
|---|---:|---|
| `encoder.py` | 437 | The token-set trunk: a per-type MLP embedding of every token (`TRUNK_TYPES`), masked self-attention blocks over all of an agent's tokens, and readouts (mean, max, marked, self) into one context vector, plus the per-token embeddings a pointer head indexes (B1). |
| `freezing.py` | 450 | One shared trunk and one adapter per verb: adding a verb never changes a frozen one (B4, E13). `encode_agents` and `encode_contexts` run the trunk once per tick; `VerbBody` is a trained verb as a body. |
| `generalist.py` | 414 | The request-conditioned generalist: request tokens (verb, typed arguments with presence masks, scope, entity and cell pointers), one query per scope, factored head logits, status and progress. |
| `observation_adapter.py` | 206 | `legal_tokens`: the legal observation as the trunk's token sets (the self token and entity slot 0, in the body frame). |

Gates: `test_sa_encoder.py`, `test_sa_freezing.py`, `test_sa_generalist.py`, `test_sa_observation_adapter.py`. The
latency gates are in `../bench/`; the trainer is `../training/`.
