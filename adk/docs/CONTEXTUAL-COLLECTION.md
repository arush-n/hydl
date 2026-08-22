# Contextual collection

`collect_contextual()` is the algorithm-neutral compiled loop for policies
that need PLAN-style timing, lifecycle, and interrupt evidence. It is additive:
the existing `Policy`, `collect()`, and compact rollout APIs keep their original
`ActorPolicyInput` contract.

A contextual policy receives one JAX PyTree:

```python
ContextualPolicyInput(
    actor_input=exact_actor_policy_input,
    decision_context=normalized_decision_context,
)
```

`actor_input` is the same exact structured legal observation, actor-safe action
surface, dense bridge-compatible row, and action mask used by the rest of the
ADK. `decision_context` contains `DecisionTiming`, `ActionLifecycle`, and
`DecisionEvents` values with their availability masks. It does not contain raw
or privileged diagnostics.

The runtime computes the next context inside the `jax.lax.scan` body as:

```python
jax_decision_context(
    info,
    actor_input=candidate_actor_input,
    previous_actor_input=current_actor_input,
).context
```

The first policy call after reset receives `unknown_decision_context(batch)`.
After a terminal transition, only the next loop carry resets that row to the
fully unknown context; `ContextualLoopTransition.next_policy_input` retains the
real terminal candidate context. The transition's `info` field is the exact,
unfiltered environment-info PyTree, so a recorder can preserve any diagnostic
or learner target without an ADK projection. `raw_info` is an identity-preserving
alias for that same object.

The same transition carries `done` plus optional `terminated` and `truncated`
causes. Both optional values are `None` for the current JAX Arsenal producer,
which publishes only their collapsed disjunction. Context and carry reset on
`done`; code that requires the cause must call `boundary.require_split()` and
will fail closed until the producer publishes it.

```python
def policy(carry, policy_input, key):
    actor = policy_input.actor_input
    context = policy_input.decision_context
    factors = choose_factors(carry, actor, context, key)
    return update_carry(carry, actor, context), factors

def record(transition):
    return {
        "actor": transition.actor_input,
        "context": transition.decision_context,
        "next_context": transition.next_decision_context,
        "raw_info": transition.info,
    }

runner = handle.compile_contextual_collector(
    policy,
    record,
    steps=128,
    initial_carry=initial_carry,
)
final_state, time_major_records = runner(root_key)
```

Use `handle.collect_contextual(...)` for the same semantics without wrapping
the complete collector in `jax.jit`. Both paths use `lax.scan`; policy and
record functions must return fixed-shape PyTrees, and neither may perform host
synchronization. Recurrent algorithm carry is reset per terminal row exactly as
in the original collector.
