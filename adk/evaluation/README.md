# ADK evaluation

`adk.evaluation` evaluates policies on exact scenes with controls, episode
boundaries, and evidence that can be compared across policy arms. It is an
evaluation consumer, not the Arena promotion gate or the Console archive.

It depends on the ADK runtime, policy surface, and HytaleGym environment. The
main entry point is [`__init__.py`](__init__.py), which exposes exact-scene
evaluation and control-suite helpers.

For replicated task promotion, continue to [`arena/evaluation/`](../../arena/evaluation/README.md).
