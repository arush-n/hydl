package com.hytalerlbridge.network;

import com.hytalerlbridge.environment.EnvironmentManager;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/** TCP server with one isolated simulator environment per client connection. */
public final class BridgeServer {

    private static final System.Logger LOGGER = System.getLogger(BridgeServer.class.getName());

    private final String host;
    private final int port;
    private final EnvironmentManager environmentManager;
    private final int bufferSize;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final ExecutorService threadPool = Executors.newVirtualThreadPerTaskExecutor();
    private final CountDownLatch startLatch = new CountDownLatch(1);
    private volatile ServerSocket serverSocket;
    private volatile Thread acceptThread;
    private volatile Throwable startFailure;
    private volatile int boundPort = -1;

    public BridgeServer(
        String host,
        int port,
        EnvironmentManager environmentManager,
        int bufferSize
    ) {
        this.host = host;
        this.port = port;
        this.environmentManager = environmentManager;
        this.bufferSize = bufferSize;
    }

    public void start() {
        if (!running.compareAndSet(false, true)) {
            throw new IllegalStateException("Bridge server is already running");
        }

        acceptThread = Thread.ofVirtual().name("hytalerl-accept").start(() -> {
            try {
                ServerSocket socket = new ServerSocket();
                serverSocket = socket;
                socket.bind(new InetSocketAddress(host, port));
                boundPort = socket.getLocalPort();
                startLatch.countDown();
                LOGGER.log(System.Logger.Level.INFO,
                    "Bridge server listening on " + host + ":" + boundPort);

                while (running.get()) {
                    try {
                        Socket clientSocket = socket.accept();
                        clientSocket.setTcpNoDelay(true);
                        threadPool.submit(
                            new ClientHandler(clientSocket, environmentManager, bufferSize)
                        );
                    } catch (IOException exception) {
                        if (running.get()) {
                            LOGGER.log(System.Logger.Level.ERROR,
                                "Error accepting bridge connection", exception);
                        }
                    }
                }
            } catch (IOException exception) {
                startFailure = exception;
                running.set(false);
                startLatch.countDown();
                LOGGER.log(System.Logger.Level.ERROR,
                    "Failed to start bridge server on " + host + ":" + port, exception);
            }
        });
    }

    /** Wait for binding and return the actual port (useful when configured with port 0). */
    public int awaitBoundPort(Duration timeout) throws IOException, InterruptedException {
        if (!startLatch.await(timeout.toMillis(), TimeUnit.MILLISECONDS)) {
            throw new IOException("Timed out waiting for bridge server to bind");
        }
        if (startFailure != null) {
            throw new IOException("Bridge server failed to bind", startFailure);
        }
        return boundPort;
    }

    public void stop() {
        running.set(false);
        ServerSocket socket = serverSocket;
        if (socket != null && !socket.isClosed()) {
            try {
                socket.close();
            } catch (IOException exception) {
                LOGGER.log(System.Logger.Level.WARNING,
                    "Error closing bridge server socket", exception);
            }
        }
        threadPool.shutdownNow();

        Thread thread = acceptThread;
        if (thread != null && thread != Thread.currentThread()) {
            try {
                thread.join(2000);
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
            }
        }
    }

    public boolean isRunning() {
        return running.get();
    }

    public int getBoundPort() {
        return boundPort;
    }
}
