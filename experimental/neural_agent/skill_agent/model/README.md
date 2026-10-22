# skill_agent/model/ — the networks

The trunk every body reads its tokens through, the per-verb adapters that keep frozen skills bit-identical, the
request-conditioned generalist (`../docs/ARCHITECTURE_RESEARCH.md`), each skill's own module on the shared trunk
(owner, 2026-10-19: every skill trains its own), and the adapter from the
simulator's legal observation to the trunk's token sets. Pure JAX: parameters are NamedTuples of arrays.

| Module | Lines | What it holds |
|---|---:|---|
| `encoder.py` | 565 | The token-set trunk: a per-type MLP embedding of every token (`TRUNK_TYPES`), masked self-attention blocks over all of an agent's tokens, and readouts (mean, max, marked, self) into one context vector, plus the per-token embeddings a pointer head indexes (B1). `trunk_types(schema)` and `token_widths(schema)` give the same six types under either token schema; `EncoderConfig(token_types=token_widths(TOKEN_SCHEMA_V2))` reads v2 tokens (2026-10-19). Schema v3 has its own thirteen types and order (`../contract/tokens_v3.py`); `token_layout(schema)` gives each type's feature names. An optional latent memory (`memory_slots`, 2026-10-20 night, off by default so every earlier trunk is unchanged): S slots join the token sequence, pass through the same blocks and are blended into the next tick's memory by a learned gate (`EncoderOutput.memory`, `initial_memory`). Optional conditioning (`condition_types`, 2026-10-21, off by default): the persona and affect tokens modulate every token's embedding (FiLM, zero at initialisation). |
| `migrate.py` | 84 | A trained trunk carried to another token layout by feature name (2026-10-21, `../docs/VIRTUAL_HUMAN.md` section 14): a new feature's input weights start at zero, so a trunk that only grew computes exactly what it did; a merged feature takes its sources' weights (exact where they were equal); a dropped one loses its weights (exact where it was 0); a feature gone without a map or a drop fails. |
| `freezing.py` | 450 | One shared trunk and one adapter per verb: adding a verb never changes a frozen one (B4, E13). `encode_agents` and `encode_contexts` run the trunk once per tick; `VerbBody` is a trained verb as a body. |
| `generalist.py` | 414 | The request-conditioned generalist: request tokens (verb, typed arguments with presence masks, scope, entity and cell pointers), one query per scope, factored head logits, status and progress. |
| `skill_module.py` | 215 | A skill's own module (owner, 2026-10-19): the generalist's head with the skill's own weights on the shared frozen trunk (`Trunk`); it drives its scope's heads, reads its own done, and sees only the requests of skills it knows (`visible_requests`); save and load with the SHA-256 of its weights and of its trunk; `save_trunk` records the token layout and its fingerprint, and `load_trunk` given today's layout migrates a trunk saved under another (`migrate.py`); `trunk_from_run` takes the trunk of an A2 reference run. |
| `observation_adapter.py` | 206 | `legal_tokens`: the legal observation as the trunk's token sets (the self token and entity slot 0, in the body frame). |

Gates: `test_sa_encoder.py`, `test_sa_freezing.py`, `test_sa_generalist.py`, `test_sa_skill_modules.py`,
`test_sa_observation_adapter.py`, `test_sa_tokens_v2.py`, `test_sa_tokens_v3.py`, `test_sa_state_registry.py`. The
latency gates are in `../bench/`; the trainer is `../training/`.
