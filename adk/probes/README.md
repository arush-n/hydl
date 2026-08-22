# Probes

`adk.probes` supplies deliberately simple policies and audits that exercise a
scene's legality, liveness, determinism, reset, and action responsiveness. A
probe is a diagnostic driver, not a trained agent.

It depends on the policy and runtime surfaces and feeds results to
[`diagnostics/`](../diagnostics/README.md) and capability measurement. Start
with [`policies.py`](policies.py) for drivers and [`audit.py`](audit.py) for
whole-run checks.

Use probes before interpreting a training score: a flat metric can mean the
scene or action mask cannot express the behavior.
