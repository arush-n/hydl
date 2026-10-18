# HytaleRL bridge plugin (Server 0.7.2 port)

This Gradle project is the Server 0.7.2 port of [`../hytale-plugin`](../hytale-plugin),
which stays on 0.5.x. It keeps the same layout, packages and wire protocol; what
changed for 0.7.2, and why, is in [PORT-0.7.2.md](PORT-0.7.2.md).

It is the Java-side bridge between a running Hytale server and
the Python/JAX environment. It owns native session setup, authoritative state
capture, action application, world/geometry evidence, and the transport used by
native policy evaluation.

It depends on the Hytale server API and its external assets at runtime. The
repository supplies source and the Gradle wrapper, but not the proprietary
server installation, asset archive, or built plugin output. The Python side
consumes the protocol and evidence contracts through `HytaleRL/hytalegym`.

## Entry points

- [`build.gradle.kts`](build.gradle.kts) defines the Gradle project.
- [`src/main/java/com/hytalerlbridge/HytaleRLBridgePlugin.java`](src/main/java/com/hytalerlbridge/HytaleRLBridgePlugin.java)
  is the plugin entry point.
- The `environment`, `combat`, `geometry`, `imitation`, and `worldgen` source
  packages implement the major bridge surfaces.
- The native action frame, including the body-relative movement ring and the
  currently unsupported gait field, is documented in
  [`../hytalegym/hytalegym/jax/combat/observation/v3/NATIVE-PORT.md`](../hytalegym/hytalegym/jax/combat/observation/v3/NATIVE-PORT.md).
- [`gradlew`](gradlew) and [`gradlew.bat`](gradlew.bat) provide the build
  wrapper.

The build is pinned to Maven `com.hypixel.hytale:Server:0.7.2` (byte-identical
to the launcher's 0.7.2 `HytaleServer.jar`) and produces
`build/libs/HytaleRLBridge072-0.1.0.jar`. To probe another installed server JAR,
pass `-PhytaleServerJar=<absolute path>` to Gradle; this changes the compile
classpath only. [`tools/launch_bridge_072.py`](tools/launch_bridge_072.py)
starts a 0.7.2 server with the jar in a scratch directory.

Read [`../protocol/README.md`](../protocol/README.md) for the transport
contract and [`../hytalegym/README.md`](../hytalegym/README.md) for the Python
consumer.
