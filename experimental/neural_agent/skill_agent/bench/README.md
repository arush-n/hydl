# skill_agent/bench/ — per-tick latency (the 30 Hz gate)

The game ticks at 30 Hz (33.3 ms). The lower layer's share of a tick must stay at a p95 of at most 15 ms at batch 8.
Run a bench on a quiet machine and record the load average with its numbers.

| Module | Lines | What it times |
|---|---:|---|
| `bench_tick.py` | 334 | The resolver, the combiner and the encoder at batch 1, 8 and 64. |
| `bench_bodies.py` | 254 | Trained-shape bodies: the trunk inside every call against the trunk once per tick, and two batched layouts. |
| `bench_generalist.py` | 194 | The generalist: the trunk once for all agents, then one generalist forward (proof A1). |
| `bench_perception.py` | 195 | The v2 tokens (`../docs/PERCEPTION.md` step 5): the trunk at 16, 32 and 64 entities and three description widths beside v1, and one tick's registry gathers. |

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

Gates: `test_sa_bench_bodies.py`, `test_sa_bench_perception.py`; `test_sa_encoder.py` checks `bench_tick.py`'s
imports. The numbers themselves are gates recorded in `experimental/neural_agent/docs/LEDGER.md` (README sections 2.7
and 2.13; the v2 tokens' cost, 2026-10-19).
