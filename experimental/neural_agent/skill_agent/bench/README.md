# skill_agent/bench/ — per-tick latency (the 30 Hz gate) and the encoding audit

The game ticks at 30 Hz (33.3 ms). The lower layer's share of a tick must stay at a p95 of at most 15 ms at batch 8.
Run a bench on a quiet machine and record the load average with its numbers.

The encoding audit (2026-10-21, `../docs/PERCEPTION.md` section 10.5) asks a different question: does the state
layer's encoding carry the information the agent needs, in the simulator and on the live server?

| Module | Lines | What it times |
|---|---:|---|
| `bench_tick.py` | 334 | The resolver, the combiner and the encoder at batch 1, 8 and 64. |
| `bench_bodies.py` | 254 | Trained-shape bodies: the trunk inside every call against the trunk once per tick, and two batched layouts. |
| `bench_generalist.py` | 194 | The generalist: the trunk once for all agents, then one generalist forward (proof A1). |
| `bench_perception.py` | 195 | The v2 tokens (`../docs/PERCEPTION.md` step 5): the trunk at 16, 32 and 64 entities and three description widths beside v1, and one tick's registry gathers. |
| `agent_models.py` | 321 | One model per agent (VIRTUAL_HUMAN.md 16.3; the owner, 2026-10-22): the whole tick (`runtime/state_tick.tick`) on recorded Region ticks under the gen-7 render with `perception_v2`, at N = 1, 2, 4, 8, 16 in five layouts: `shared` (one trunk, one batch of N), `stacked` (N trunks with their own weights, one `vmap` program), `sequential` (N trunks, one batch-1 program called N times), `processes` (N OS processes, N <= 8, XLA's default threads) and `processes_1thread` (each limited to one thread). p50 and p95 per tick and per agent, the first call's compile time, the parameters and bytes. Run under `runtime/gen7` with `--construct perception_v2`. The tests run without the render (the body yaw from the world). |
| `v3_rollout.py` | 274 | Rollout speed on v3 against v1 (TRAINING_PLAN.md 8.3, stage V0; measurements only): one decision tick of a training rollout on the flat world (`training/blocks.environment`, reset and first neutral tick as `skill_dagger.rollout`) at batch 64, 128 and 256. v1 is `blocks.token_kernel`, the trunk with no memory and the generalist head; v3 is `state_tick.tick` with `agent_models.CONFIG` (the state layer timed alone too) and the same head. Env step, tokens, trunk, head, path and total: p50 and p95, decisions per second, the first call, the load at each run's start and end; the paths interleaved round by round. Once per run: one v3 lane's `AgentState` bytes by field and by part, and the parameter bytes of the v1 and v3 trunks and heads. Plain, or under `runtime/gen7` with `--construct perception_v2` (the record says whether the channel was present). |
| `v3_data_size.py` | 228 | What one v3 decision costs to store (TRAINING_PLAN.md 8.3, stage V0; measurements only): the state layer's tick on the trainers' flat world, the Region scenario and the duel, plain and under the gen-7 render with the channel. Per decision, the bytes if each token type keeps its valid rows in float16 plus a mask byte per slot, with mean, p50, p95, max, the share by type, the ratio to dense float32, and the share of float16 values that move by more than 1e-3 relative. Also the walk_to and look_at horizons, in ticks and seconds. |
| `encoding_audit.py` | 295 | The encoding audit on the simulator: the state layer's real tick on rollouts where the agent acts at random (legal, held) against an attacker; per token type and column, the valid rows, how often it is non-zero, its range and its distinct values. Scenarios `duel` (the open-flat duel, sword, shortbow and battleaxe) and `region` (a Region world: terrain, geometry and light; the channel when `perception_v2` is on); `compare` lines up reports. `--library NAME` loads a chosen Region library for the `region` scenario (`region-library-pilot-v2-072` carries the 0.7.2 block identity, 6f180b6); both scenarios read the effects registry (`agent_skills/training_runs/perception/effects_072`, read in place). |
| `live_encoding.py` | 158 | The same on the live 0.7.2 server: a native episode, the tick run on each live observation; it saves every raw observation leaf, the light rows, the body yaw and the actions for replay. |

Run:

```sh
.venv/bin/python -m experimental.neural_agent.skill_agent.bench.bench_tick \
    --batch 1,8,64 --ticks 300 --warmup 30 --seed 0 [--json PATH]
.venv/bin/python -m experimental.neural_agent.skill_agent.bench.bench_bodies --batch 1,8 --skills 1,2,3,6 \
    --verbs shared,distinct --ticks 200 --warmup 20 --seed 0 [--json PATH]
.venv/bin/python -m experimental.neural_agent.skill_agent.bench.bench_generalist \
    --batch 1,8,64 --ticks 300 --warmup 30 --seed 0 [--json PATH]
.venv/bin/python -m experimental.neural_agent.skill_agent.bench.bench_perception --batch 1,8,64 \
    --entities 16,32,64 --description-widths 16,32,64 --ticks 200 --warmup 20 --seed 0 [--json PATH]
```

Encoding audit (from the repo root, with `PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym
JAX_PLATFORMS=cpu`). Gen 7 runs through a private render (the stock engine is gen 5); the channel reaches only the
`region` scenario's single-actor path. The content registry is read from the private folder in place:

```sh
.venv/bin/python -m experimental.neural_agent.skill_agent.runtime.gen7 --engine DIR --render -- \
    experimental.neural_agent.skill_agent.bench.encoding_audit run duel --out duel.json
.venv/bin/python -m experimental.neural_agent.skill_agent.runtime.gen7 --engine DIR --construct perception_v2 -- \
    experimental.neural_agent.skill_agent.bench.encoding_audit run region --out region.json
.venv/bin/python -m experimental.neural_agent.skill_agent.bench.encoding_audit compare duel=duel.json region=region.json
```

The live run is the bridge launcher's follow-on command (`live_encoding.py`'s docstring): check the pin
(`HytaleRL/hytale-plugin-0.7.2/tools/pin_bridge.py --check`), pick a free port pair (`lsof`; 5562/25580 on
2026-10-21; never another lane's), launch `tools/launch_bridge_072.py --then` with a new run directory; the run takes
the shared evidence lease and the server stops when it ends. About 1.5 s a tick under load.

Gates: `test_sa_bench_bodies.py`, `test_sa_bench_perception.py`, `test_sa_bench_encoding_audit.py`;
`test_sa_encoder.py` checks `bench_tick.py`'s imports. The numbers themselves are gates recorded in `experimental/neural_agent/docs/LEDGER.md` (README sections 2.7
and 2.13; the v2 tokens' cost, 2026-10-19).
