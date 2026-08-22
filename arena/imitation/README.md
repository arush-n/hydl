# Native imitation boundary

`arena.imitation` is the algorithm-neutral boundary between native Java/NPC evidence and Python/JAX learning. It preserves what was observed, where it was valid, and which parts are still unavailable instead of manufacturing a dense learner dataset.

## Core trace model

| File | Purpose |
| --- | --- |
| [`contract.py`](contract.py) | Versioned `TraceSpec`, `TraceComponent`, `TraceContract`, `TraceSequence`, and `TraceView` objects. |
| [`projectors.py`](projectors.py) | Fail-closed transformations that add episode structure or derived components. |
| [`corpus.py`](corpus.py) | Groups traces with matching contracts and provenance without silently concatenating incompatible arrays. |
| [`native.py`](native.py) | Converts Hytale native captures into behavior, world-model, duel, and concurrent trace views. |
| [`native_corpus_capture.py`](native_corpus_capture.py) | Live native-NPC capture workflow and its v4 corpus receipt. |

The trace contract separates current and next values, validity, stateful components, episode structure, and provenance. A component may be present in the schema while unavailable on a particular row; consumers must honor its mask.

## Native evidence

Native capture can preserve pose, velocity, rotation, health, grounded state, steering, attack activation/execution, interaction chains, lifecycle, damage, perception, geometry, light, inventory transactions, target worldview, and world clock. It does not automatically provide the dense Arena learner observation or task-specific reward labels.

The native adapter therefore has three useful views:

- behavior: what the NPC selected and what engine state followed;
- contextual behavior: behavior plus spatial/world context for analysis or teacher construction;
- world model: current evidence, next evidence, and event ledgers for prediction tasks.

## Safety rules

- Unknown episode boundaries remain unknown until an authoritative projector installs them.
- Invalid action labels abstain rather than becoming negative examples.
- Legal-action masks and expert-supervision masks are separate.
- Reward components are explicitly namespaced as event signals.
- Native traces from different bridge/schema/world identities cannot be combined casually.

For the learner-facing replay and transfer implementation, continue to [`training/imitation/`](../training/imitation/README.md).
