# jev/ — Level 2, the decision instances

"JEV" is an interface and a latency budget, not an architecture (ADR-001 D2): anything that reads B_t and
E_t and returns a `ManagedGoal` within budget. This folder holds every instance built so far, grouped by
family, and the runner that binds an in-graph instance to the ADK core. The composition driver's arm
names are in brackets (`../eval/arms.py`).

## The current path (2026-10-18)

| File | Lines | What it is |
|---|---:|---|
| `scripted.py` | 312 | J0, the scripted rule: pursue; evade on damage within 40 ticks with a 2 s commitment. The baseline every arm is measured against. [`j0`] |
| `programs.py` | 1194 | Behaviour programs: a short series of skill steps with exits (predicate, done, timer), interrupts by priority and resumption (and the combat program `P_COMBAT`, `PROGRAMS['COMBAT']`, 2026-10-19), executed exactly on the host fast path; it writes `ActionRequest`s into the scratchpad and applies the correspondence rules to the combiner's statuses. [`jprog`, program by `NA_PROGRAM`] |
| `programs_guerrilla.py` | 81 | The guerrilla program `P_GUERRILLA`, `PROGRAMS['GUERRILLA']` (2026-10-20, task #99): avoid while the target faces the agent or the agent is hurt, shoot from 3.5 to 4.5 blocks when the shot is clean, hide while seen, pursue the remembered position while unseen. Reads the body's reports through `Report` in `programs.py`. [`jprog`, program by `NA_PROGRAM=GUERRILLA`] |
| `correspondence.py` | 279 | How the middle layer reacts to the lower layer's status stream: advance, resample a step, raise urgency, drop, ask the LLM (`../docs/SYSTEM_ONE.md` 9.10). |
| `typed_rule.py` | 285 | The exact typed path: typed conditions (or the legacy constraint labels, or the words through the rule reader) evaluated on the belief, the priority rule as `goal_language.teacher`. Equals the teacher on every decision; J0's rate online. [`jtyped`] |
| `system_one/` | | J-S1, our own System One decision model, and the reading experiments. Its own `README.md`. [`js1`] |
| `runner.py` | 479 | Binds an in-graph instance to the ADK hierarchy core and applies the miss rule to host calls. |

## The learned reactive line, J1 (measured 2026-10-17; superseded as the primary instance by J-S1)

| File | Lines | What it is |
|---|---:|---|
| `network.py` | 633 | J1, a small network over B_t and the lowered E_t. [`j1`, needs `--j1-params`] |
| `train_j1.py` | 612 | Behaviour cloning of J1 from recorded J0 decisions (`../eval/record_traces.py`). |
| `train_j1_ppo.py`, `j1_actor_critic.py`, `sim_goal_env.py` | 800, 188, 435 | PPO over goal decisions as a semi-Markov decision process on the arena (the synthetic toy environment and the MLP stand-in were removed on 2026-10-18). |
| `eval_j1.py` | 91 | Offline evaluation of J1 parameters on a decision dataset. |

## Deprecated

`lm/`: J-LM, a quick language model as the instance (2026-10-17: superseded; nothing external but the top
LLM). Kept because the `j_lm` arm runs it. Its own `README.md`.

Gates: `test_na_jev_core.py` (J0 and the runner), `test_na_programs.py`, `test_na_typed_rule.py`,
`test_na_jev_network.py`, `test_na_jev_bc.py`, `test_na_jev_ppo.py`, `test_na_sim_goal_env.py`,
`test_na_jev_runner_fallback.py`, `test_na_jev_lm.py`. Measurements and verdicts: `../docs/LEDGER.md`;
the consolidated status: `../docs/MIDDLE_LAYER.md`.

Rules every instance follows: it never blocks the skills; a decision slower than the budget is a miss and
the previous goal stands (`../session.py`); the skill's contract carries commitment and duration
(`contracts.control_for`), the instance does not invent them.
