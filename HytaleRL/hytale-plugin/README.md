# HytaleRL bridge plugin

This Gradle project is the Java-side bridge between a running Hytale server and
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
- [`gradlew`](gradlew) and [`gradlew.bat`](gradlew.bat) provide the build
  wrapper.

Read [`../protocol/README.md`](../protocol/README.md) for the transport
contract and [`../hytalegym/README.md`](../hytalegym/README.md) for the Python
consumer.
