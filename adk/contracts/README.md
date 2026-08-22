# Cross-layer contracts

`adk.contracts` holds small versioned contracts that cross package boundaries.
It includes the shaping interface used by minigames and the identity stamp
that binds a run to its environment and policy surfaces.

The contracts depend on stable HytaleGym/ADK schemas and are consumed by scene
builders, training adapters, and Console evidence. They do not define task
goals; Arena owns those semantics.

Start with [`shaping.py`](shaping.py) for additive reward terms and
[`stamp.py`](stamp.py) for cross-seam identity.
