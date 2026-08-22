# Console catalog

`console.core.catalog` resolves the agent profiles, run profiles, and declared
identity records that the Console exposes to its API and browser. It describes
what may be launched and how a configured counterpart is identified; it does
not own task mechanics or policy parameters.

It depends on the concrete `agents/` and Arena contract metadata. Execution,
preflight, and panel code consume it, while the catalog does not depend on the
job runner.

## Entry points

- [`agents.py`](agents.py) discovers agent declarations and their Console
  needs.
- [`profiles.py`](profiles.py) loads named agent configurations.
- [`identities.py`](identities.py) compares declared, built, and live
  identities.

Continue to [`../README.md`](../README.md) for the surrounding Console core.
