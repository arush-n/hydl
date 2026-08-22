# Training reward streams

Reward streams are ingredients for a lesson. They do not define a complete task, reset policy, or action scope; those belong in [`../contracts/`](../contracts/README.md).

| File | Reward stream |
| --- | --- |
| [`combat_fundamentals.py`](combat_fundamentals.py) | Engine-grounded approach, alignment, tracking, attack acceptance, damage, victory, and failure terms derived from combat geometry/lifecycle. |
| [`behavior_prior.py`](behavior_prior.py) | Online penalty/regularizer from a fixed behavior prior evaluated on the actual policy state. |
| [`exact_imitation.py`](exact_imitation.py) | Negative reward for aligned, known action mismatches, excess spin, and illegal/expert-support violations. |
| [`learned_rewards.py`](learned_rewards.py) | Prediction/RND, impact/RIDE, and adversarial GAIL/AIRL kernels with explicit mixing and clipping. |
| [`__init__.py`](__init__.py) | Clarifies and re-exports reward-stream APIs. |

## Design principle

Engine evidence is preferred over teacher predictions. Combat rewards use authored attack envelopes and lifecycle evidence; accepted attacks, attributed damage, and terminal outcomes are not inferred from a geometric guess. Learned reward code stops gradients where appropriate and leaves model updates and running statistics to the agent.

## Mixing with PPO

Reward streams can wrap a collector or be mixed after a rollout, but their discount/termination semantics must agree with the PPO configuration. The current runtime does not provide a second value head for independently discounted intrinsic and extrinsic streams, so callers must not assume those streams are automatically separated.

