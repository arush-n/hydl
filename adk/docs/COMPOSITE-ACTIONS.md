# Composite action SDK

`adk.runtime.actions` is the algorithm-neutral action layer. It derives the
ordered head names and sizes from a live contract, keeps actions as explicit
`int32[B, H]` factors, and keeps the flat policy width only for independent
categorical logits and masks. It has no scalar-packing API.

```python
from adk.contracts.stamp import current_stamp
from adk.runtime.actions import CompositeActionSpec

spec = CompositeActionSpec.from_live_contract(current_stamp())
sample = spec.sample_masked(key, logits, actor_input.action_mask)

# int32[B, 12] on the current surface; the count comes from the live stamp.
factors = sample.factors
behavior_log_probability = sample.log_probability
```

`sample_masked` and `masked_log_probability` call the neutral
`hytalegym.jax.policy` distribution primitives and are safe to place inside
`jax.jit`, `lax.scan`, or a general collector. PPO is one possible consumer,
not an owner of these operations. If a head has no legal choice, or its legal
logits are non-finite, the sample contains `-1` for that head and the row log
probability is `-inf`; it never substitutes a hidden fallback action.

## Structured actions are retained

`StructuredActionCodec` wraps the environment's encoder and decoder directly.
Its `decode()` result is the exact upstream structured object, not an ADK
subset. `raw_codec`, `CompositeActionSpec.raw_contract`, and
`EnvironmentActionReceipt.raw_receipt` are deliberate escape hatches for
backend information that the common SDK does not interpret.

```python
from adk.runtime.actions import StructuredActionCodec
from hytalegym.jax.combat.observation.v3 import policy as action_policy

codec = StructuredActionCodec(
    spec,
    action_policy.encode_arsenal_policy_actions,
    action_policy.decode_arsenal_policy_actions,
    raw_codec=action_policy,
)
structured = codec.require_lossless_round_trip(factors)
```

The host-side `validate_factors()` and `validate_mask()` methods are contract
checks. Their layout-only and distribution methods remain JIT-compatible.

## Per-head legality is not joint legality

A flat mask contains one marginal slice per categorical head.
`selected_per_head_legal()` can prove that each chosen value was admitted by
its own slice. It cannot prove that the whole combination is coherent,
atomic, accepted, or executed. In particular, this SDK does not invent
cross-head Use, Block, Craft, or any other exclusivity rule.

The environment must own that decision and return an
`EnvironmentActionReceipt`:

- `UNAVAILABLE`: no joint validator exists; `require_joint_validated()` raises.
- `JOINT_VALIDATION`: the environment reports a `bool[B]` joint result.
- `JOINT_EXECUTION`: the environment reports both validation and execution.

`EnvironmentActionBoundary` is the protocol for adapters that can submit the
factors and return this receipt. A native or JAX adapter may retain its exact
backend response in `raw_receipt`. Agent code that needs joint validity or
confirmed effects should call `require_joint_validated()` or
`require_executed()` and therefore fail closed when that evidence is absent.

This boundary is intentionally separate from sampling: a policy distribution
does not authorize an environment mutation.
