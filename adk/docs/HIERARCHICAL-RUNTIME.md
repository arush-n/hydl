# Hierarchical runtime

`runtime.hierarchical` is the executable bridge between
`HierarchicalAgentCore` and the structured SDK environment. It does not choose
a neural architecture, optimizer, replay format, or learning algorithm.

```python
from adk.runtime.hierarchical import compile_hierarchical_collector

runner = compile_hierarchical_collector(
    built,
    core,
    record_transition,
    steps=128,
    initial_state=initial_hierarchy_state,
)
final, records = runner(root_key)
```

The runtime owns the common fixed-shape sequence:

1. reset the structured environment;
2. expose its exact `ActorPolicyInput` and normalized `DecisionContext`;
3. run belief, manager cadence, goal control, and actor through
   `HierarchicalAgentCore`;
4. require explicit `int32[batch, heads]` factor rows;
5. step the canonical detailed-or-legacy `BuiltAgent` interface;
6. derive the candidate decision context from the exact transition;
7. record candidate hierarchy state and diagnostics before reset;
8. reset environment, context, and hierarchy state only on rows selected by
   `EpisodeBoundary.done`.

`HierarchicalLoopTransition` retains both
`candidate_hierarchy_state` and `carried_hierarchy_state`. This distinction is
intentional: terminal component diagnostics and recurrent outputs remain
available to a learner, while the next policy call receives the supplied
initial hierarchy state on that lane. The transition also retains the exact
environment `info`, known or unknown `terminated` / `truncated` split, current
and candidate actor inputs, and current and candidate decision contexts.

The default context mapper is the canonical JAX mapper. Another structured
backend can inject a function with the following lossless normalized seam:

```text
context_fn(info, candidate_actor_input, previous_actor_input)
    -> DecisionContext
```

The mapper, rather than the hierarchy, owns interpretation of backend-specific
diagnostics. The actor and manager continue to receive only legal observation
products and normalized availability-aware context.

The runtime module is directly importable now. Public convenience exports from
`adk.runtime`, `adk.api`, and the package root are a separate packaging
change; no algorithm or PPO adapter is required for execution.
