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
