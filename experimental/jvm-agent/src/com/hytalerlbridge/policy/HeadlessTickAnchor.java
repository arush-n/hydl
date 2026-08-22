package com.hytalerlbridge.policy;

import com.hypixel.hytale.protocol.ToClientPacket;
import com.hypixel.hytale.protocol.ToServerPacket;
import com.hypixel.hytale.protocol.io.ChannelConnection;
import com.hypixel.hytale.protocol.io.PacketStatsRecorder;
import com.hypixel.hytale.server.core.io.PacketHandler;
import com.hypixel.hytale.server.core.io.ProtocolVersion;
import com.hypixel.hytale.server.core.modules.entity.player.ChunkTracker;
import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.world.World;
import java.lang.reflect.Proxy;
import java.net.InetSocketAddress;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

/**
 * Keeps a playerless world running at live TPS.
 *
 * <p>Without this, an otherwise-correct hook ticks <b>once</b> and then goes
 * silent, which looks exactly like a broken system. The cause is not the hook:
 * {@code World.isIdle()} is true whenever no {@code PlayerRef} is tracked, and
 * in 0.5.7 an idle {@code TickingThread} skips its 30 TPS limiter and feeds ECS
 * systems tiny wall-clock deltas instead of running them at rate.
 *
 * <p>The fix is a {@code PlayerRef} that is <b>not a player and not an
 * entity</b> -- it has a null holder and a packet sink that discards
 * everything. Its only job is to make the world consider itself non-idle.
 * Adapted from the bridge's {@code installHeadlessTickAnchor} /
 * {@code HeadlessPacketHandler}, which established this behaviour; keeping the
 * two implementations aligned matters if 0.5.7's idle semantics ever change.
 */
public final class HeadlessTickAnchor {

    private HeadlessTickAnchor() {
    }

    /** Track a non-entity PlayerRef so {@code world} leaves its idle path. */
    public static PlayerRef install(World world) {
        PlayerRef anchor = new PlayerRef(
            null,
            UUID.randomUUID(),
            "HytalePolicyAgentTickAnchor",
            "en-US",
            new DiscardingPacketHandler(),
            new ChunkTracker()
        );
        world.trackPlayerRef(anchor);
        return anchor;
    }

    public static void uninstall(World world, PlayerRef anchor) {
        if (world != null && anchor != null) {
            world.untrackPlayerRef(anchor);
        }
    }

    /** Packet sink for an anchor with no client: every write is dropped. */
    private static final class DiscardingPacketHandler extends PacketHandler {

        private static final ChannelConnection CHANNEL = headlessChannel();

        DiscardingPacketHandler() {
            super(CHANNEL, new ProtocolVersion(0));
        }

        @Override
        public String getIdentifier() {
            return "HytalePolicyAgentTickAnchor";
        }

        @Override
        public void accept(ToServerPacket packet) {
            // No client exists to send anything.
        }

        @Override
        public void write(ToClientPacket... packets) {
        }

        @Override
        public void write(ToClientPacket[] packets, ToClientPacket finalPacket) {
        }

        @Override
        public void write(ToClientPacket packet) {
        }

        @Override
        public void writeNoCache(ToClientPacket packet) {
        }

        @Override
        public void writePacket(ToClientPacket packet, boolean cache) {
        }

        @Override
        public void tryFlush() {
            // Nothing is ever queued, because every write is discarded.
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

        /**
         * There is no socket, so the channel is a proxy. Any method not listed
         * that returns a value throws rather than silently returning null --
         * a null here would surface much later as an unrelated NPE.
         */
        private static ChannelConnection headlessChannel() {
            return (ChannelConnection) Proxy.newProxyInstance(
                ChannelConnection.class.getClassLoader(),
                new Class<?>[] {ChannelConnection.class},
                (proxy, method, args) -> switch (method.getName()) {
                    case "isActive", "isWritable", "isFromSameOrigin" -> false;
                    case "remoteAddress" -> new InetSocketAddress("127.0.0.1", 0);
                    case "formatRemoteAddress" -> "headless";
                    case "getPacketStatsRecorder" -> PacketStatsRecorder.NOOP;
                    case "getSniHostname" -> "";
                    case "setupAuxiliaryChannels" ->
                        CompletableFuture.completedFuture(null);
                    case "toString" -> "PolicyAgentHeadlessChannel";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    default -> {
                        if (method.getReturnType() != Void.TYPE) {
                            throw new UnsupportedOperationException(
                                "Unsupported headless channel call: "
                                    + method.getName());
                        }
                        yield null;
                    }
                }
            );
        }
    }
}
