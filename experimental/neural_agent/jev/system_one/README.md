# jev/system_one/ — J-S1, our own System One model, and the reading experiments

The middle layer's learned instance on two timescales (`../../docs/SYSTEM_ONE.md` section 8): a slow path
once per executive version (reading E_t) and a fast path per decision (0.3 ms: exact predicate evaluation
plus light heads for replace, params and confidence). This folder also holds every experiment on *reading
the LLM's words*, whose verdicts (sections 9.11 to 9.13) fixed the contract: typed fields are what the layer
acts on, words are context, and a goal that arrives with words only is read once per version by a reader
large enough to be accurate.

## The model

| File | Lines | What it is |
|---|---:|---|
| `heads.py` | 552 | The heads in JAX: belief adapter, structured executive encoder, LayerNorm fusion, one GELU layer, parallel heads (skill, replace, params, confidence). |
| `trunk.py` | 157 | Frozen text encoders (text in, one pooled vector out) for the slow path. |
| `data.py` | 939 | Rendering and supervised rows (FORMULAS M2, M5); the goal-language sets and the held-out splits. |
| `train_supervised.py` | 582 | Stage A: supervised fit of the heads on decision rows. |
| `calibrate.py` | 177 | Stage B: fit the confidence head with the other heads frozen. |
| `instance.py` | 347 | `SystemOneJev`, the host-mode instance; `fast_path` with the skill contracts' control table. |
| `probe_latency.py` | 216 | Gate G2: the two timescales' latency, CPU, in process. |

## Reading the words (experiments, in order)

| File | Lines | Experiment | Verdict (ledger) |
|---|---:|---|---|
| `interpret.py` | 556 | The interpret stage: the exact number reader (`read_number`, `argument_for`), a frozen-encoder condition classifier, an intent matcher. | number 1.000 PASS; frozen condition kind 0.71 FAIL (polarity) |
| `interpret_benchmark.py` | 275 | Text encoders against the stage's gate on the three held-out splits. | the floor the later readers are compared with |
| `finetune_pilot.py` | 716 | P2: fine-tuned small encoders as condition readers. | 0.75 / 0.82 kind FAIL, polarity and negation lost |
| `catalogue_matching.py` | 572 | P3: intent to skill by description over the owner's 139-row catalogue (skill rows through `../../skill_agent/contract/catalogue.py` since B5). | 0.42 FAIL: the typed `skill_id` is the selector |
| `lm_reader.py` | 1322 | P6: instruction-tuned decoders (Qwen3 0.6B to 14B) with schema-constrained output, once per version; the Ollama GPU-runner check; the two-pass variant. | kind 0.92 / 0.90 / 0.97 / 1.00 / 1.00 from 0.6B to 14B (bar 0.95); number PASS everywhere; skill by description FAIL at every size (0.36 to 0.88); two-pass lowers kind; LoRA plan recorded |
| `pattern_reader.py` | 230 | The pattern reader of SYSTEM_ONE 9.14: the skill from the catalogue's ids and aliases (anchored id first), the clause through `condition_rules`; gate P7 (`python -m ... pattern_reader --seed 0`). | 1.000 at no breakage, 0.9925 with every perturbation, PASS |
| `nearest_skill.py` | 193 | SYSTEM_ONE 9.16: the nearest catalogue id for a skill id the LLM wrote that the library cannot serve (exact, alias, then id words with stem-tolerant weighted overlap); descriptions only suggest, in the question to the LLM; `real_catalogue_tables()` reads the private catalogue at run time. | gate: 4 of 7 absent families served from the expected group on all 15 phrasings each, 3 near no id and asked about; descriptions were noise (loot_chest for "take cover") |
| `condition_rules.py` | 377 | The deterministic fallback reader: feature, comparator and negation cues to `(kind, argument, confidence)`; below CLEAR is an information request. CLI: `python -m experimental.neural_agent.jev.system_one.condition_rules --seed 0`. | 1.000 kind and argument on held-out wordings, seeds 0 to 2, PASS (cue tables know the generator's vocabulary) |

Gates: `test_na_system_one.py`, `test_na_interpret.py`, `test_na_finetune_pilot.py`,
`test_na_catalogue_matching.py`, `test_na_lm_reader.py`, `test_na_condition_rules.py`, `test_na_nearest_skill.py`.

Measured online (G4, `../../docs/MIDDLE_LAYER.md`): the learned instance passed the composition rate on
seeds 0, 1, 2 once it read the contracts' commitments; its latency misses under machine load are the open
part of the bar. Private inputs: the catalogue and the paraphrases under `agent_skills/training_runs/` are
read, never copied into tracked files.
