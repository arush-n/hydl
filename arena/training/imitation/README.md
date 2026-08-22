# Training-time imitation

This folder turns demonstrations and native traces into learner-consumable batches. The lower-level trace contract is in [`arena/imitation/`](../../imitation/README.md); this folder adds action labels, recurrent replay tensors, behavior cloning, and native-to-Arsenal transfer.

## File guide

| File | Responsibility |
| --- | --- |
| [`behavior_cloning.py`](behavior_cloning.py) | Per-head masked categorical loss, supervision/legal masks, class weights, entropy, accuracy, and JIT training. |
| [`native_imitation.py`](native_imitation.py) | Conservative projection of native steering/yaw/pitch/basic-attack evidence into Arsenal action heads. |
| [`native_replay.py`](native_replay.py) | Native replay corpus schema, balanced sampling, recurrent sequence packing, replay PPO environment, and policy loading. |
| [`native_replay_cache.py`](native_replay_cache.py) | Content-addressed compiled corpus/projection cache and held-out split materialization. |
| [`native_transfer.py`](native_transfer.py) | Shared/conditioned/visual observation projectors and contract-checked policy transplantation. |
| [`__init__.py`](__init__.py) | Public exports used by training callers. |

## Demonstration semantics

Legal action support and expert supervision are separate. An action can be legal but absent from the demonstration, and an observed label can be invalid for the current mask. Invalid labels abstain rather than becoming arbitrary class targets. Unknown action heads are left neutral unless an exact binding supplies their role-local meaning.

## Native replay

`build_native_replay_corpus()` retains outcome-neutral features from native episodes, pads them into fixed arrays, assigns length-based episode weights, and records action-head coverage. `split_native_replay_corpus()` holds out whole actor/world groups rather than random rows. Recurrent packing resets at episode starts and never lets hidden state cross an unknown boundary.

## Transfer boundary

The portable shared transfer currently targets the transferable movement/look heads and explicitly bound basic-attack roots. Ability slots require exact source/target role bindings and contract hashes. A successful transplant is a provenance-checked initialization, not evidence that the transferred policy is competent in the target environment.

