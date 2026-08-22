# Training runtime

`training/runtime` orchestrates a training experiment around an existing trainer. It owns reproducibility, resource sizing, admission, monitoring, and artifact history—not the environment or the learning algorithm itself.

## File guide

| File | Responsibility |
| --- | --- |
| [`contracts.py`](contracts.py) | Namespaced metrics, parameter domains, stop losses, environment profiles, and immutable training contracts. |
| [`features.py`](features.py) | Static composition of pure-JAX feature state/metric kernels into one traced program. |
| [`execution.py`](execution.py) | Replica stacking, `vmap` execution, common-random-number keys, and VRAM-based replica sizing. |
| [`evaluation.py`](evaluation.py) | Background evaluation over immutable host snapshots with isolated environment/RNG/recurrent state. |
| [`audit.py`](audit.py) | Non-finite, guardrail, positive-reward-without-progress, opposition, and regression checks. |
| [`tuning.py`](tuning.py) | Deterministic mixture of surrogate, local-history, and exploration parameter proposals. |
| [`monitor.py`](monitor.py) | Atomic `status.json` plus sequence-numbered append-only `events.jsonl`. |
| [`starter.py`](starter.py) | One-call recursive train/evaluate/audit/tune/promote loop with rollback/stop policies. |
| [`__init__.py`](__init__.py) | Public runtime surface. |

## Runtime lifecycle

An agent supplies a `TrainingAdapter` with a final `TrainingContract`, initial parameters, an output directory, a `train_and_evaluate()` method, and a `promote()` method. `starter.train()` validates the contract, chooses replicas from the previous full-run VRAM peak, evaluates candidates, audits them, proposes the next candidates, and promotes the best admitted result.

Feature kernels are composed before tracing. Evaluation snapshots are copied to read-only host memory, so an evaluation cannot mutate optimizer state, live environment state, recurrent carry, or RNG. The monitor API is intentionally independent of the console and is suitable for polling by a CLI or adapter.

## Admission versus optimization

The runtime can reject an artifact even when its training loss improved. Reward exploitation, missing metrics, support failures, non-finite values, stop losses, VRAM overages, utility drawdown, and lack of improvement all belong to admission policy. This is why a “best reward” is not automatically a promotable checkpoint.

