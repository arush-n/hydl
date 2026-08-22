package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.protocol.ToClientPacket;
import com.hypixel.hytale.protocol.ToServerPacket;
import com.hypixel.hytale.protocol.io.ChannelConnection;
import com.hypixel.hytale.protocol.io.PacketStatsRecorder;
import com.hypixel.hytale.server.core.io.PacketHandler;
import com.hypixel.hytale.server.core.io.ProtocolVersion;
import java.lang.reflect.Proxy;
import java.net.InetSocketAddress;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

/**
 * Observing packet sink for the non-entity PlayerRef that keeps a headless
 * world at live TPS.
 */
final class HeadlessPacketHandler extends PacketHandler {

    private static final System.Logger LOGGER =
        System.getLogger(HeadlessPacketHandler.class.getName());
    private static final ChannelConnection CHANNEL = headlessChannel();

    private final Consumer<ToClientPacket> observer;

    HeadlessPacketHandler(Consumer<ToClientPacket> observer) {
        super(CHANNEL, new ProtocolVersion(0));
        this.observer = Objects.requireNonNull(observer, "observer");
    }

    @Override
    public String getIdentifier() {
        return "HytaleRLHeadlessTickAnchor";
    }

    @Override
    public void accept(ToServerPacket packet) {
        // There is no client connection for a headless tick anchor.
    }

    @Override
    public void write(ToClientPacket... packets) {
        for (ToClientPacket packet : packets) observe(packet);
    }

    @Override
    public void write(ToClientPacket[] packets, ToClientPacket finalPacket) {
        for (ToClientPacket packet : packets) observe(packet);
        observe(finalPacket);
    }

    @Override
    public void write(ToClientPacket packet) {
        observe(packet);
    }

    @Override
    public void writeNoCache(ToClientPacket packet) {
        observe(packet);
    }

    @Override
    public void writePacket(ToClientPacket packet, boolean cache) {
        observe(packet);
    }

    @Override
    public void tryFlush() {
        // Nothing is queued because all writes are discarded.
    }

    @Override
    public boolean stillActive() {
        return true;
    }

    @Override
    public boolean isLocalConnection() {
        return true;
    }

    @Override
    public boolean isLANConnection() {
        return true;
    }

    private void observe(ToClientPacket packet) {
        if (packet == null) {
            LOGGER.log(
                System.Logger.Level.WARNING,
                "Headless packet observer ignored a null packet"
            );
            return;
        }
        try {
            observer.accept(packet);
        } catch (RuntimeException exception) {
            LOGGER.log(
                System.Logger.Level.ERROR,
                "Headless packet observer rejected packet "
                    + packet.getClass().getSimpleName(),
                exception
            );
        }
        // The headless anchor has no network connection; delivery ends here.
    }

    private static ChannelConnection headlessChannel() {
        return (ChannelConnection) Proxy.newProxyInstance(
            ChannelConnection.class.getClassLoader(),
            new Class<?>[]{ChannelConnection.class},
            (proxy, method, args) -> switch (method.getName()) {
                case "isActive", "isWritable", "isFromSameOrigin" -> false;
                case "remoteAddress" -> new InetSocketAddress("127.0.0.1", 0);
                case "formatRemoteAddress" -> "headless";
                case "getPacketStatsRecorder" -> PacketStatsRecorder.NOOP;
                case "getSniHostname" -> "";
                case "setupAuxiliaryChannels" ->
                    CompletableFuture.completedFuture(null);
                case "toString" -> "HeadlessChannelConnection";
                case "hashCode" -> System.identityHashCode(proxy);
                case "equals" -> proxy == args[0];
                default -> {
                    if (method.getReturnType() != Void.TYPE) {
                        throw new UnsupportedOperationException(
                            "Unsupported headless channel call: "
                                + method.getName()
                        );
                    }
                    yield null;
                }
            }
        );
    }
}
