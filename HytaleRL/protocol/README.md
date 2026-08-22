# Bridge protocol

[`messages.md`](messages.md) documents the wire-level messages exchanged by the
Python environment and the Java bridge. It is the narrow contract between
HytaleRL's native session and the higher-level policy/runtime layers.

The protocol depends on the Java plugin and Python gym agreeing on message
schemas, action factors, lifecycle boundaries, and evidence identities. It does
not define PPO, task rewards, or agent strategy.

Start with [`messages.md`](messages.md), then follow the Python consumer in
[`../hytalegym/README.md`](../hytalegym/README.md) or the Java producer in
[`../hytale-plugin/README.md`](../hytale-plugin/README.md).
