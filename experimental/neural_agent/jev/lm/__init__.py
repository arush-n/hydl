"""Quick language-model JEV instance (ADR-001 D2, J-LM), experimental.

This package holds the host-mode JEV that reads B_t and E_t and emits one
ManagedGoal under a latency budget. The concrete backends (Ollama, Anthropic) and
the scripted fake sit behind ``protocol.DecisionLM``. Read ``SKILL.md`` for the
rules, formulas, citations and open gaps.

Nothing is imported here, so importing the package does not load a model client
or touch the network.
"""
