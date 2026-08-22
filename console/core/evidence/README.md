# Console evidence

`console.core.evidence` is the durable report and replay boundary. It indexes
completed runs, summarizes selected agent artifacts, reconstructs replay/world
data, and keeps evidence identity separate from the training worker.

It depends on report schemas from Arena/agents and on world/replay decoders
from the Console. It does not train policies or decide which candidate wins.

Start with [`store.py`](store.py) for indexing and report loading,
[`results.py`](results.py) for normalized outcomes, and
[`agent_archive.py`](agent_archive.py) for agent-level archive summaries.
