# Agent architecture

`adk.architecture` defines structured and hierarchical agent composition:
observation compilation, decision context, goal control, tactical management,
world-model interfaces, and inference validation.

It depends on HytaleGym framework contracts and feeds the ADK policy/runtime
layers. It is deliberately independent of a particular learner or Arena task;
an Arena lesson may consume the resulting policy inputs, but the architecture
package does not choose the lesson.

## Entry points

- [`inputs.py`](inputs.py) defines policy and training input envelopes.
- [`components.py`](components.py) defines composable agent roles.
- [`hierarchy.py`](hierarchy.py) defines hierarchical carry and transitions.
- [`decision_context.py`](decision_context.py) and [`inference.py`](inference.py)
  validate decisions across runtime boundaries.

Continue to [`runtime/`](../runtime/README.md) for executable collection.
