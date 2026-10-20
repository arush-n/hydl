# skill_agent/runtime/ — running skills: the registry, the combiner, programs, the scratchpad

Everything that turns the middle layer's requests into one factored action per agent per tick, and sends the
statuses back. Fixed and exact: the thinking happens inside the bodies (`../model/`), not here.

| Module | Lines | What it holds |
|---|---:|---|
| `combiner.py` | 504 | The fixed greedy combiner: per tick and per agent, check each request, grant it by scope (atomically), run the granted bodies, merge their heads, stream the statuses (R1 to R6); `BodyStep`, `AgentContext`, `StubBody`. |
| `library.py` | 456 | `SkillLibrary`: create, register and use verbs and programs; `from_catalogue()` registers the catalogue's verbs and programs; `combiner(layout)` builds one combiner per episode. |
| `programs.py` | 140 | `ProgramRunner`: an activity program (stages in order, the steps of a stage together, for-each) as a stream of requests, with the statuses read back. |
| `scratchpad.py` | 216 | The combiner on S_t: requests in from the shared state's `requests` section, statuses out to `skills`; `CombiningDecider` puts it behind the program executor (B3). With `actor_verbs` its `on_tick` returns the tick's grants as a `GrantedActor`'s state, for the session's per-tick seam (2026-10-19), and writes the actor's reports onto each granted request's status (`progress`), where the decision layer and the LLM read them (`docs/IDEAS.md` idea 8). |
| `sim_adapter.py` | 131 | `HeadAdapter`: the combiner's 21 choices onto the heads a simulator implements (neutral elsewhere), with the drops counted. |
| `parity.py` | 184 | Today's one goal per lane as one request through the combiner path (A6). |
| `joint_check.py` | 104 | B3's joint check: the composition arms with and without the combiner on S_t behind the middle layer's program executor. |
| `neural_pursuit.py` | 226 | The spine's skill (2026-10-19): the A2 C' network composing walk_to and look_at as the decision layer's `pursuit`, in the dispatcher's `Skill` protocol, from the legal observation only. Its body yaw comes from published fields: the head yaw minus the target's head-frame bearing less its body-frame bearing (exact while the target is visible), or the perception channel's body yaw on gen 7. With the target out of sight it walks to the request's remembered position, and with nothing known it turns to search (18 degrees a tick in the flat world). |
| `granted_actor.py` | 145 | The referee controls the body (2026-10-19, owner): `GrantedActor`, the ADK actor whose state is the combiner's active-request table; each scope's heads take the choices of the body granted there, a free scope stays neutral, no stand-in skill acts and the goal index is not read. Every verb's body also reports `scripted_bodies.REPORTS` each tick (`diagnostics["reports"]`). `grants_table` lowers a tick's grants; `referee_library` gives the host combiner `continuous` bodies so grants hold until withdrawn. |
| `scripted_bodies.py` | 291 | The skill layer's scripted bodies (owner: scripted first), and what each reports every tick (`REPORTS`: seen, distance, facing_me, in_reach, acting). A target out of sight is remembered from the request's world position through the perception channel's own position and body yaw (gen 7); `avoid` (retreat beyond a band) and `hide` (to a cover point, else retreat) join the combat bodies: `evader`, `sword` and `punish` (the weapon's first ability: in the gym a weapon attacks through its ability slots), `guard` (on the target's wind-up), `keep_range` (its `band_min` to `band_max`), `idle`; legal observation and action mask only. |
| `gen7.py` | 97 | Runs any skill-layer module or script on a private gen-7 render (owner, 2026-10-19: gen 7 now): renders the 0.7.2 overlay into a directory and mounts it before `hytalegym` is imported; the render's import finder serves the v3 legacy module aliases (`render.py`, 780b5de). `--construct NAME` switches a gen-7 construct on in this process only (`perception_v2`, the perception channel). |
| `spine.py` | 135 | The working spine: `run_composition` up to three ways (`--runs`): the stand-in pursuit (gen 5 only), the neural one in the registry's pursuit slot, and the referee, where the combiner's grants drive the skill layer's bodies through `GrantedActor` (no stand-in acts); every host decider behind the combiner on S_t; the arms' metrics side by side and the statuses streamed. |

Run (from the repo root). Both take `--out` and pass every other flag to the composition run they wrap, so `--help`
lists only `--out`:

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.skill_agent.runtime.parity \
    --arms flat,j0,shuffled,j1 --ticks 64 --batch 8 --episodes 1 --seed 0 --out <dir>
NA_PROGRAM=J0 JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.skill_agent.runtime.joint_check --out <dir> \
    --arms flat,j0,shuffled,jprog --runner eager --ticks 64 --batch 8 --episodes 1 --seed 0
NA_PROGRAM=J0 JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.skill_agent.runtime.spine --out <dir> \
    --arms jprog --runner eager --ticks 512 --batch 8 --episodes 2 --seed 0
JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.skill_agent.runtime.gen7 --engine <dir> [--render] \
    -- <module or script.py> [args]
```

Gates: `test_sa_combiner.py`, `test_sa_library.py`, `test_sa_execute_all.py` (every verb and every program runs on
gen 7), `test_sa_scratchpad.py`, `test_sa_joint.py`, `test_sa_sim_adapter.py`, `test_sa_spine.py`, `test_sa_gen7.py`,
`test_sa_granted_actor.py`.
`parity.py` has no unit test: its gate is the recorded run (A6, in `experimental/neural_agent/docs/LEDGER.md`); the
spine's numbers are recorded there too (2026-10-19).
