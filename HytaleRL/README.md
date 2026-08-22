# HytaleRL gym and native bridge

`HytaleRL` is the lower-level environment implementation. It contains the
installable `hytalegym` package, the Java bridge plugin, the wire protocol, and
small examples for stepping, training, and native transfer. Arena and ADK sit
above this layer; the Console and NPC runner use it through their public
interfaces.

The Python gym owns combat, locomotion, observation/action encoding, geometry,
world providers, and JAX training primitives. The Java project owns the native
server adapter and authoritative session transport. Neither project bundles a
Hytale server or its proprietary assets; those are external runtime inputs.

## Entry points

- [`hytalegym/README.md`](hytalegym/README.md) explains the Python package and
  its source-checkout layout.
- [`hytale-plugin/README.md`](hytale-plugin/README.md) explains the Java bridge
  and its Gradle project.
- [`protocol/README.md`](protocol/README.md) introduces the transport contract.
- [`examples/README.md`](examples/README.md) points to the supported example
  categories.

The package's public Python import is `hytalegym`; the repository-root shim
keeps source checkouts and editable installs using the same nested package.
See the root [`README.md`](../README.md) for the Arena/ADK/Console workflow.
