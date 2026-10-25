# experimental/neural_agent/eval

## Files in this folder (map, 2026-10-18)

| File | Lines | What it is |
|---|---:|---|
| `run_composition.py` | 1424 | The CLI. Runs the chosen arms on the composition task (`../tasks/`) and writes `metrics.csv`, `REPORT.md` (every metric with its formula and source) and, for a host-mode arm, `scratchpad_<arm>.json` and `scratchpad_<arm>_llm_view.md`. Flags: `--arms`, `--runner eager|compiled` (host-mode arms need `eager`), `--arena`, `--ticks`, `--batch`, `--episodes`, `--seed`, `--host-respects-commitment` (default on since 2026-10-17; `--no-...` restores the earlier seam), `--host-budget-ms` (default 20; a slower host decision is a miss). |
| `arms.py` | 977 | The arm registry `ARMS`: `flat`, `j0`, `shuffled`, `j1`, `j_lm`, `js1`, `jprog`, `jtyped`; the host decider factories; `J0_LATENCY_BUDGET_MS`; `NA_PROGRAM` / `NA_NO_REISSUE` for `jprog`. |
| `metrics.py` | 472 | Pure metric functions over the recorded per-tick arrays (the definitions below). |
| `eager_loop.py` | 226 | The Python-driven hierarchical step loop that host-mode deciders run in. |
| `traces.py` | 302 | The decision-trace dataset of the learned J1 line. |
| `record_traces.py` | 160 | Record J0 decision traces on an arena and merge them into a J1 training set. |
| `record_dagger.py` | 736 | DAgger-style relabelling for the learned decider `js1`: the rule labels the learner's own states. |
| `run_architecture.py` | 493 | The architecture test (2026-10-24): a host loop in which the decision layer's goals become `walk_to`/`look_at` requests in S_t, the combiner grants them to the skill layer's trained bodies (`state_tick.ModuleBody`), and the state tick (`StateTickBinding`) runs the memory trunk; arm `reference` is the trained student as the skill layer evaluates it, on the same episodes. Section "The architecture test" below. |

Gates: `test_na_eval.py`, `test_na_eager_loop.py`, `test_na_record_traces.py`, `test_na_record_dagger.py`.
The standard comparison every ledger entry since 2026-10-17 uses:

```sh
PYTHONPATH=/Users/arush/hydl JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.eval.run_composition \
  --arms j0,jtyped --runner eager --arena combat_open_flat --ticks 512 --batch 8 --episodes 8 --seed 0 --out <dir>
```

An arm passes when its success rate is not below `j0`'s Wilson lower bound on the same seed, interrupts are
answered, commitment violations are 0 and misses are 0 at the measured latency. Metric differences between two
runs of the same arm are first checked against the misses column: a miss keeps the previous goal and the
trajectory diverges from there (joint check, 2026-10-18). `out/` holds historical runs (gitignored); the ledger
entries that cite them are the evidence. The rest of this document is the 2026-10-17 record of what is measured.

Evaluation of the composition arms of ADR-001's three-level agent (sections 8 and 9).
Last updated **2026-10-17** (UTC; the local clock read 2026-10-16 evening, CDT). Source: `metrics.py`, `arms.py`, `traces.py`, `run_composition.py`, and `tasks/composition.py` for the verdicts.

## Thesis

ADR-001 section 1 (status COMMITTED):

> The research question is how an intelligent agent uses time and compute: when to think, when to act, and when to reuse an existing decision. Task completion is the instrument, not the question.

This package measures the instrument. On the composition task (pursue, evade when damaged, resume pursuit) it reports goal switching, interrupt response, JEV latency and task completion for five arms: a flat pursuit actor, a scripted JEV (J0), a shuffled-goal control, a learned JEV (J1) and a quick language-model JEV (J-LM). `j1` and `j_lm` are wired via `eval/arms.py`. `j1` runs untrained until a checkpoint is supplied, and until then its report labels it `j1-untrained`. `j_lm` is a host-mode arm and requires `--runner eager`, because it runs through `session.run_session`. A structural run is not evidence that the hierarchy works. ADR-001 section 0 and section 6.5 say what the existing evidence does not establish.

ADR-001 is a local record. It is not in this repository, so the section numbers above refer to a copy that the reader may not have.

## What is measured

| Quantity | Definition | Source |
|---|---|---|
| Lane-run | One lane over one collector run of `ticks` steps. `lane_runs = episodes x batch` | `run_composition.py` |
| Episode | The ticks from a reset through the first tick with `done`. The horizon and a catch both end it | `metrics.completed_episodes` |
| Success | An episode that `CompositionTask.classify` returns as `composed` | `tasks/composition.py` |
| Pursue-only | An episode that `classify` returns as `pursue_only` (caught, no evade before the catch, no damage) | `tasks/composition.py` |
| Skill changes | Changes of the active skill index between consecutive ticks, on ticks where a goal was in force (the forced first goal of an episode is excluded) | `metrics.goal_switches` |
| Core replacements | Core switches on ticks where a goal was already in force | `run_composition._score_run` |
| Interrupt latency | Ticks from an interrupt-event onset to answered by the next core goal replacement, or, since 2026-10-17, by a skill change on a tick with a goal in force (host-mode arms write goals between steps and never produce a core replacement) (a switch on a tick where a goal was in force) | `metrics.interrupt_latency_ticks`, fed by `run_composition._score_run` |
| Commitment violations | Ordinary switches made before the goal's minimum commitment | `metrics.commitment_violations` |
| JEV misses and latency | Decisions over the latency budget, and their p50 and p95 | `JevRunner.stats()`, only when measured |

## Formulas

Each formula names the rule that the code applies. `[confirmed]` means read in the cited source. `[inferred]` means derived here, with the reason given.

**1. Trials and success rate** `[confirmed]`

```text
lane_runs            = episodes x batch
episodes_completed   = number of episodes that reached done inside the run
success_rate         = successes / episodes_completed
catches_per_episode  = catches / episodes_completed
```

An episode still running when the run ends is not counted. The arena's horizon is set to `ticks`, so an episode normally ends by a catch or at the end of the run.

**2. Wilson lower bound** `[confirmed: formula, as recipe.py computes it; source Wilson 1927, DOI record checked, paper not read]`

With p = successes / n and z = 1.96:

```text
lower = (p + z^2/(2n) - z * sqrt((p(1-p) + z^2/(4n)) / n)) / (1 + z^2/n)
```

This is the formula of `agent_skills/training_runs/recipe.py` (`wilson_lower`). The test checks equality on three cases. The advance rule of the recipe compares this bound with `advance_lower_bound`. The bound is exactly 0.0 when there are no successes.

**3. Episode verdicts** `[confirmed: tasks/composition.py classify, strict rule of 2026-10-17]`

```text
caught         = terminated[t] for some t in the episode
composed       = caught, and some evade entry e before the catch with
                   a damage tick d where e - 40 <= d <= e   (EVADE_TRIGGER_WINDOW_TICKS = 40)
                   and a pursuit entry p where e < p < catch
composed_loose = caught, and some evade entry before the catch, but not composed
pursue_only    = caught, no evade entry before the catch, and zero damage events
failed         = every other completed episode
```

Only `composed` is a success. `composed_loose` is reported and never counted as a success. Without the optional keys `damage_ticks` and `pursuit_entry_ticks`, every caught episode with an evade entry before the catch is `composed_loose`. An evade entry is a tick where the active skill is the evader and the previous tick is not. A damage event is a rising edge of DAMAGE_RECEIVED.

Why the rule changed: the earlier rule made `composed` mean any evade entry before the catch. The 256-tick gate run of 2026-10-17 then scored the shuffled-goal control 3/5 with no damage in the arena, so the rule counted random evader selections as composition. `run_composition._score_run` records both keys: `damage_ticks` are the episode ticks with DAMAGE_RECEIVED active and known, and `pursuit_entry_ticks` are the ticks where the active skill becomes pursuit, the first tick included (the formula block of `run_composition.py`). Both are recorded step indices, so the 40-tick window is 40 steps (`docs/FORMULAS.md` B5).

**4. Skill changes and core replacements** `[confirmed]`

```text
skill_changes_per_lane_run   = sum over lane-runs of sum_{t=1}^{T-1} [skill[t] != skill[t-1]] * [goal_valid[t]] / lane_runs
core_replacements_per_lane_run = sum over lane-runs of sum_{t>=1} [goal_switched[t] and goal_valid[t]] / lane_runs
```

A replacement with the same skill index counts only in the second quantity. The forced first goal of an episode (`goal_valid` False) is excluded from both, so a catch followed by an autoreset counts as neither a skill change nor a replacement. The first tick of a record is never counted as a change.

**5. Interrupt onset and latency** `[confirmed]`

The interrupt events are PROJECTILE_THREAT_ENTERED and DAMAGE_RECEIVED. ADR-001 section 8 P2 names them as the evade triggers.

```text
onset(t, b)       = event_active[t, b] and not event_active[t-1, b]
replacement(t, b) = goal_switched[t, b] and goal_valid[t, b]
latency           = t' - t,   t' = min{ t' >= t : replacement[t', b] }
unanswered        = an onset with no such t'
```

A latency of 0 means the goal changed on the tick the event became active. This is the core's own rule: an active known event makes the manager due (`adk/architecture/decision_context.py:415`). The forced first goal of an episode is not a reply to an event, so it is not a replacement here.

**6. Commitment violations** `[inferred]`

```text
violations = #{ ordinary switches : goal_age < minimum_commitment }
ordinary   = valid goal, not timed out, not an emergency interrupt
```

The core's switch rule allows an ordinary switch only when `goal_age >= minimum_commitment` (`adk/architecture/hierarchy.py:273` and `:347-356`). So the count should be 0 by construction. A nonzero value points at the core or at a bypass that was recorded wrongly. This is inferred from reading the rule. It is not tested against a live arena.

**7. Terminated means catch** `[inferred]`

The flat arena publishes `info.terminated` as "catch or death", and `info.truncated` as the horizon (`tasks/flat_arena.py`, `step_detailed`). The arena closes combat, so no actor can die there, and terminated is read as a catch. This is inferred from `tasks/events.py` and `step_detailed`, not tested.

**8. JEV miss and latency** `[confirmed: jev/runner.py]`

```text
jev_miss_rate = misses / decisions;  a miss is latency_ms > budget_ms   (strict)
```

`JevRunner` counts eager calls only, and its latency is NaN for in-graph instances. The J0 instance is in-graph, and the compiled collector traces its manager. So the J0 and shuffled arms report JEV columns as not measured.

## Citations

- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association* 22(158): 209-212. https://doi.org/10.1080/01621459.1927.10502953. The Crossref record was checked on 2026-10-16 (metadata only), and the DOI resolves to the publisher, Taylor & Francis (checked 2026-10-17). The paper was not read. The formula above is the one `recipe.py` implements. `docs/REFERENCES.md` [WILSON] holds the same entry.
- Sutton, R. S., Precup, D., Singh, S. (1999). Between MDPs and semi-MDPs: a framework for temporal abstraction in reinforcement learning. *Artificial Intelligence* 112(1-2): 181-211. https://doi.org/10.1016/S0004-3702(99)00052-1. The DOI resolves to Elsevier PII S0004370299000521, and its Crossref record matches the title, authors, volume and pages (checked 2026-10-17). Public copy: https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf (returned a PDF on 2026-10-17). ADR-001 section 11 cites it as [O1], and `docs/REFERENCES.md` [O1] holds the same entry.
- ADR-001 three-level agent, a local record that is not in the repository (no URL): section 8 P2 exit (goal switches, interrupt latency, commitment violations); section 9 (metric families and the ablation matrix); section 1 (thesis).
- `experimental/hierarchical_planner/DESIGN.md` section 6: the shuffled-goal control, and the reading that if the transformer and shuffled arms agree, the manager adds temporal structure rather than goal content.
- `agent_skills/training_runs/recipe.py`: `wilson_lower` and the advance rule.
- `experimental/neural_agent/tasks/composition.py`: the verdicts and the executive. `tasks/events.py`: the DAMAGE_RECEIVED read helper, and the ablation that injects it (ADR-001 D5).
- `adk/architecture/hierarchy.py` (lines 273-282, 347-356 and 372, the goal-valid update), `adk/architecture/decision_context.py` (line 415), `adk/runtime/hierarchical.py` (the collector). These are repository paths, the only locator for them.

## Running

```bash
export PYTHONPATH="/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym" JAX_PLATFORMS=cpu
/Users/arush/hydl/.venv/bin/python -m experimental.neural_agent.eval.run_composition \
    --arms flat,j0,shuffled --episodes 2 --batch 8 --ticks 64 --seed 0 --arena flat_pursuit --out DIR
```

The command writes `metrics.csv` and `REPORT.md` into `DIR`. The `j1` and `j_lm` arms are wired via `eval/arms.py`: `j1` runs untrained until a checkpoint is supplied (`--j1-params PATH`), and `j_lm` requires `--runner eager`. `--arena` picks `combat_baseline`, the default, where DAMAGE_RECEIVED fires and nothing can be caught, or `flat_pursuit`, where a catch ends an episode and no actor takes damage. The 256-tick gate run of 2026-10-17 used the flat pursuit arena. `--runner` is `compiled` (the default) or `eager`. The default `DIR` is `eval/out/`, which `experimental/neural_agent/.gitignore` (line 2) excludes. `REPORT.md` repeats this file's thesis, formulas and citations, with the run's own table and caveats.

## Limits

- The table describes the instrument. It does not show that the hierarchy makes better decisions, and no result here should be read that way.
- Success and pursue-only come from the composition task, as the task defines them. Since 2026-10-17 `combat_open_flat` composes (J0 33/64 at 512 ticks, `docs/LEDGER.md`); of the two earlier arenas, neither composes on its own: `flat_pursuit` has catches and no damage, and `combat_baseline` has damage and no catches (`run_composition.py` docstring). So no episode can be composed unless a damage event is injected on `flat_pursuit`. That injection is an ablation (ADR-001 D5), and the CLI does not apply it.
- The belief parameters are a random initialization with a fixed seed. The arena publishes no speed scales, so two belief features are unknown.
- The trace dataset `traces.py` writes is read by `record_traces.py` (2026-10-17), which keeps the `manager_due` rows and adds `num_skills`; `jev/train_j1.py` trained on it and `jev/eval_j1.py` scored it (`docs/LEDGER.md`). Before that date the layout had no reader and was unverified.

## The architecture test (2026-10-24; `run_architecture.py`)

Asked in the skill layer's ledger entry (dc9037a); the owner said go. The question: do the skill layer's trained bodies still
work when the decision layer drives them through the architecture (goals become typed requests in S_t, the combiner grants
scopes, each body acts on its granted scope, the state tick runs the memory trunk), rather than in their trainer's loop?

**Method.** One host tick for B lanes: the state tick (`StateTickBinding.on_tick` running `state_tick.step_lanes`: the state
layer with the agent's own pose, then the memory trunk); the decision layer (`GoalDecider`: each walk goal is a TaskSpec
`HoldArea` over `blocks.HOLD_TICKS`, scored with `task_spec.hold_step`; each look goal is `LookAtTasks.success` held as long;
a request is withdrawn when its goal is done or out of time); the combiner on S_t (`combine_from_scratchpad`) with one
`state_tick.ModuleBody` per verb, lowered by the skill's own `lowering`; the simulator steps on the combined 12 heads. The
world is the bodies' training world (`training.blocks.environment`: the committed engine, motion timing pinned to 1/30 s).
The `reference` arm is the trained student as the skill layer evaluates it (`distill_v3.evaluate`: one generalist call a
tick with both requests, masked argmax) on the same goals, spawns and keys. `--order session` (removed 2026-10-25) ran the
state tick after the combiner, as `session.run_session` orders its hook, so bodies read the previous tick's view; its rows
below are a record. Since 2026-10-25 the stack arm runs on the gen-7 host session runner
(`session_gen7.run_gen7_session`: state tick, executive, timed decision, batched bodies and the combiner, goal reports, world
step; owner ruling "Host loop for gen 7"), with the same report as before on held-out and seen episodes. `--joint` is a diagnostic:
`JointModuleBody` gives each module every live request of a verb it was trained beside, as in training.

**Results** (bodies `v1_bodies/memory_s0`, 96 lanes, 32 per mode, seed 77001; `out/architecture_2026-10-24/*.json`; one lane
is 0.031):

| Episodes | Arm | walk held | look held | both held | look held while walking | pace within 5 % | illegal |
|---|---|---:|---:|---:|---:|---:|---:|
| seen | reference | 1.000 | 0.938 | 0.812 | 0.812 | 0.983 | 0 |
| seen | stack, state tick first | 1.000 | 0.938 | 0.781 | 0.812 | 0.983 | 0 |
| seen | stack, session order | 1.000 | 0.969 | 0.812 | 0.844 | 0.983 | 0 |
| seen | stack, joint requests | 1.000 | 0.938 | 0.812 | 0.812 | 0.983 | 0 |
| held out | reference | 0.938 | 0.969 | 0.906 | 0.906 | 1.000 | 0 |
| held out | stack, state tick first | 0.938 | 0.969 | 0.750 | 0.781 | 1.000 | 0 |
| held out | stack, session order | 0.969 | 0.969 | 0.750 | 0.781 | 1.000 | 0 |
| held out | stack, joint requests | 0.938 | 0.969 | 0.875 | 0.906 | 1.000 | 0 |
| held out | stack, joint, session order | 0.969 | 0.969 | 0.844 | 0.875 | 1.000 | 0 |

**Reading.**
- The architecture carries the trained skills: with every live request visible to each body, the stack equals the
  reference on the seen episodes and is within one lane of it held out; walk-only and look-only match in every arm; no
  illegal action in any arm. The decision layer ended 61 to 63 of 64 walk goals and 56 to 58 of 64 look goals as done.
- The one loss is the seam: `ModuleBody` puts only its own request in the module's slots, so a body never sees the other
  live request it was trained beside. Looking while walking falls from 0.906 to 0.781 held out (5 lanes of 32); joint
  requests restore it. Fix on the skill side (the module reads the other active requests of skills it was trained beside,
  `neural_skills/spec.py`).
- The session's order (bodies read the previous tick's view) costs at most one lane here.
- Host cost: 300 to 500 ms a tick for 96 agents (the state tick, batched, about 120 ms; the bodies, batch 1 per agent per
  verb, the rest), so about 3 to 5 ms per agent. 30 Hz with many agents needs the bodies batched across agents by verb.
- `own_pose_channel` builds a new NamedTuple class each call: called on the host every tick, it made the compiled state tick
  retrace every tick (6.5 s a tick). The harness keeps one class.
- Successful goals end in the status stream as `aborted` ("withdrawn"): a `ModuleBody` never reports done, and the protocol
  has no "withdrawn as complete".
- Not tested: `session.run_session` itself. Its in-graph actor runs only array bodies (`runtime/granted_actor.py`), so a
  host `ModuleBody` cannot act through it; this harness is the host loop for now.

## The decision layer's programs on gen 7 (2026-10-25; `run_gen7_programs.py`)

The real decision layer (`jev/programs.py` `ProgramExecutor`, unchanged) drives the trained walk_to and look_at bodies through the
gen-7 host session runner (`session_gen7.run_gen7_session`) via `jev/gen7_decider.py` `Gen7ProgramDecider`: every tick the shared
belief (`belief/core.py`, legal observation only) advances on the gen-7 learner observation with the real elapsed seconds; on the
executor's cadence (0.5 s = 15 ticks) the adapter hands the executor stand-ins for the ADK fields it reads (listed in the module
docstring), applies the returned goal where the ADK seam would (commitment), and gives walk_to and look_at their point from the agent's
known-entity record in S_t (walk_to at the entity's feet, look_at at its eye, +1.0 block). Idle and point-less requests are withheld.
Emergency is all False (the gen-7 belief has no events yet).

Run (16 lanes, 300 ticks, seed 77001, a fixed executive with one goal, the program walk_to the other entity then look_at it; bodies
`v1_bodies/memory_s0`, the architecture test's world): 16/16 programs finished, 0 illegal actions, 16/16 within the walk radius,
look facing 13/16 at the other entity's eye (pitch 0 deg), decisions p50 0.52 ms / p95 2.07 ms against a 20 ms budget, host 63 ms a
tick (state tick 47). The 16 lanes are replicas of one scene. Requests are re-issued each cadence with new ids (the executor's request
lifetime is one cadence), so a body's request restarts every 0.5 s; the walk completed regardless.
