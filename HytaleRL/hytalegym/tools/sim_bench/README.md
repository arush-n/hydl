# sim_bench

Baselines every simulator render reports its delta against (`../hytalegym/sim/BASELINES.md`, which holds the
numbers and the exact commands):

- `rollout.py`: a fixed combat rollout: jit compile time, steady-state throughput and the output fingerprint, for a
  list of batch sizes (`python -m sim_bench.rollout --batches 8,64 --steps 64 --repeats 5 --out <json>`).
- `realtime.py`: the real-time factor and per-step wall time of the production training step
  (`python -m sim_bench.realtime --batches 1,8 --steps 300`).

Measure the committed engine and the candidate back to back under the same load; a delta is meaningful only then.
