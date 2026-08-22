# Training lesson contracts

`training/contracts` defines complete lessons. Each lesson states the reward law, action-head scope, evidence fields, horizon, and content hash needed to say what a policy was trained to do.

## Lesson modules

| File | Lesson |
| --- | --- |
| [`pursuit.py`](pursuit.py) | Chasing a moving target with learner-owned closing and tracking credit; the only lesson currently bound to a collector. |
| [`guard_duel.py`](guard_duel.py) | Two-sided attack/guard timing and guard-streak evidence. |
| [`punish_window.py`](punish_window.py) | Counter-attacking inside an opponent recovery window. |
| [`ability_landing.py`](ability_landing.py) | Landing a selected ability against a deterministic mover. |
| [`evasion.py`](evasion.py) | Avoiding damage while maintaining the required engagement band. |
| [`evader_duel.py`](evader_duel.py) | Pursuit-run evasion reward and contact-band semantics. |
| [`checkpoint_route.py`](checkpoint_route.py) | Reaching a generated checkpoint using distance/progress evidence. |
| [`common.py`](common.py) | Shared horizons, action scopes, manifest hashing, band functions, and the “idle must cost” guard. |
| [`observation.py`](observation.py) | Omniscient training-time context transform and its manifest. |
| [`catalog.py`](catalog.py) | Reader-facing lesson catalog with collector readiness and blockers. |
| [`probes.py`](probes.py) | Deterministic toy-policy comparisons that check reward laws are not farmable. |

## Common rules

The shared contract helpers enforce clipped progress, deadbands, activation costs, per-tick costs, and learner-owned credit where a counterfactual is available. Every action scope pins unscored heads to their physical neutral choice. This keeps a lesson from accidentally training combat, movement, or camera mechanics that its reward never measures.

## Readiness

The catalog deliberately separates “reward/action contract exists” from “collector is launchable.” Pursuit is bound. Guard duel, punish window, ability landing, evasion, and checkpoint route currently expose their contracts but remain unbound until their engine evidence, reset, transition, and episode-boundary seams are connected.

Run-level orchestration belongs in [`../runtime/`](../runtime/README.md); this folder does not own PPO or checkpoint publication.

