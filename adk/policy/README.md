# Policy surface

`adk.policy` is the stable interface an agent author implements. It names
observation groups and action factors, supplies legal masks, and provides
policy/network helpers without prescribing a strategy.

It depends on the ADK architecture and runtime input contracts and is consumed
by JAX collection and native deployment. A policy must honor the supplied
action mask; the environment does not silently turn illegal factors into a
penalty.

## Entry points

- [`__init__.py`](__init__.py) defines the `Policy` protocol.
- [`fields.py`](fields.py) and [`observations.py`](observations.py) expose
  named features.
- [`actions.py`](actions.py) and [`surfaces.py`](surfaces.py) expose action
  and capability surfaces.
- [`networks.py`](networks.py) and [`random_policy.py`](random_policy.py)
  provide reusable policy helpers.
