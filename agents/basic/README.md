# Basic Console agent

`agents.basic` is the concrete recurrent agent profile driven by the Console.
It translates a resolved HTTP training specification into
`PursuitRunConfig`, runs the Arena pursuit collector in the selected worker
environment, and writes a complete report for the Console archive.

It depends on Arena's pursuit contract and HytaleGym PPO, while the Console
owns launch authorization and job lifecycle. The worker is not a public direct
trainer entry point; use the Console API described in
[`../../console/docs/API.md`](../../console/docs/API.md).

## Entry points

- [`agent.py`](agent.py) resolves the profile and invokes the pursuit runner.
- [`worker.py`](worker.py) consumes the Console-produced worker specification.
- [`publication.py`](publication.py) selects and publishes eligible pursuit
  candidates.

The current runnable lesson is `pursuit`; other Arena lesson contracts are not
automatically enabled by this profile.
