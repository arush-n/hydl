plugins {
    java
}

group = "com.hytalerlbridge"
version = "0.1.0"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

repositories {
    mavenCentral()
    maven {
        name = "hytale"
        url = uri("https://maven.hytale.com/release")
    }
}

// The 0.7.2 port of ../hytale-plugin (which stays pinned to 0.5.7). An
// installed server can still be probed without changing the canonical
// dependency: ./gradlew compileJava -PhytaleServerJar=<path>
val hytaleServerOverride = providers.gradleProperty("hytaleServerJar").orNull
val hytaleServer: Any = hytaleServerOverride?.let { path -> files(path) }
    ?: "com.hypixel.hytale:Server:0.7.2"

dependencies {
    // Hytale Server API
    compileOnly(hytaleServer)

    // MessagePack for fast binary serialization
    implementation("org.msgpack:msgpack-core:0.9.8")

    // Parse the bundled bridge configuration.
    implementation("org.yaml:snakeyaml:2.4")

    // Testing
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.1")
    testCompileOnly(hytaleServer)
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testRuntimeOnly(hytaleServer)
}

tasks.test {
    useJUnitPlatform()
    // 0.7.2's HytaleLogger (loaded by BlockSection) requires java.util.logging.manager
    // to name HytaleLogManager before JUL initializes, so set it at test-JVM start.
    systemProperty("java.util.logging.manager", "com.hypixel.hytale.logger.backend.HytaleLogManager")
}

tasks.jar {
    archiveBaseName.set("HytaleRLBridge072")
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true

    manifest {
        attributes(
            "Main-Class" to "com.hytalerlbridge.SimulatorBridgeMain"
        )
    }

    // Fat jar: include MessagePack dependency
    from(configurations.runtimeClasspath.get().map { if (it.isDirectory) it else zipTree(it) })
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    exclude("META-INF/*.SF", "META-INF/*.RSA", "META-INF/*.DSA")
}

tasks.processResources {
    from("../hytalegym/hytalegym/rulesets") {
        include("**/*.json")
        includeEmptyDirs = false
    }
}
