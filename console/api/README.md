# Console API

`console.api` is the HTTP boundary for the Console. It defines request/response
schemas and routes for training, preflight, runs, worlds, custom games,
telemetry, and health/status views.

It depends on the Console core services and delegates task semantics to Arena
and scene semantics to ADK. It should not contain PPO or environment mechanics.

## Entry points

- [`schemas.py`](schemas.py) defines validated request models and option
  descriptions.
- [`routes.py`](routes.py) binds those models to FastAPI endpoints.
- [`__init__.py`](__init__.py) is the package boundary.

The public endpoint reference is [`../docs/API.md`](../docs/API.md).
