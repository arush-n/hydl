# Native policy agent

`experimental/jvm-agent` contains the Java policy-agent integration used by the
native transfer and deployment tooling. It translates a deployed policy's
structured actions into server-side control and exposes native perception,
combat, movement, world, and lifecycle seams.

It depends on the HytaleRL bridge protocol and the ADK deployment contract. It
is not the JAX simulator and it is not a second PPO implementation; its role is
to execute or certify a policy at the native boundary.

## Entry points

- [`mod/manifest.json`](mod/manifest.json) identifies the native plugin bundle.
- [`src/com/hytalerlbridge/policy/PolicyAgentPlugin.java`](src/com/hytalerlbridge/policy/PolicyAgentPlugin.java)
  is the plugin entry point.
- `src/com/hytalerlbridge/policy/perception/` projects native state into policy
  inputs.
- `src/com/hytalerlbridge/policy/action/` and the combat/world packages apply
  selected actions.
- `tools/export/` contains export and contract utilities used by the native
  bundle path.

For the stable Python-side deployment surface, continue to
[`adk/deploy/README.md`](../../adk/deploy/README.md).
