# Policy deployment

`adk.deploy` turns a validated checkpoint into a native policy bundle and
checks that the bundle matches the observation, action, role, and bridge
contracts expected by the server.

It depends on ADK policy identity and the HytaleRL native bridge. Deployment is
separate from training: Arena or an ADK algorithm produces a checkpoint, while
this package verifies and exports the artifact that a native session can load.

## Entry points

- [`bundle.py`](bundle.py) exports and verifies bundles.
- [`compatibility.py`](compatibility.py) checks producer/consumer identities.
- [`tensor_identity.py`](tensor_identity.py) binds parameter structure and
  hashes.
- [`isolated.py`](isolated.py) runs a bundle in an isolated native process.

The Java-side consumer is documented in
[`HytaleRL/hytale-plugin/README.md`](../../HytaleRL/hytale-plugin/README.md).
