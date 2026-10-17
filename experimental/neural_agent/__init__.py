"""ADR-001 three-level neural agent, as code (experimental).

ADR-001 is a local-only record: this clone excludes it through .git/info/exclude and
it is not in the repository.

ADR-001 (``agents/design/ADR-001-three-level-agent.md``) decides three levels:

* Level 3, the LLM executive, is asynchronous and event-driven. It writes the
  executive state E_t (goals, priorities, constraints, horizons, budgets).
* Level 2, the JEV decision model, runs about 2 Hz plus interrupts under a hard
  latency budget. It reads the shared belief B_t and E_t, and emits one managed
  goal (skill, parameters, commitment). Its instances include a scripted J0, a
  learned reactive J1, and a quick language model, J-LM.
* Level 1, the skill policies, emits factored legal actions on the 12-head
  surface at every decision step. It never waits for a JEV search or an LLM call.

This package imports no submodules. ``contracts.py`` is the only shared
interface. Read ``README.md`` for orientation and ``SKILL.md`` for the rules.
"""

__version__ = "0.1.0"
