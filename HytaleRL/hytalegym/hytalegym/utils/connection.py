"""TCP connection to the HytaleRLBridge server plugin."""

from __future__ import annotations

import socket
import struct
from typing import Any

import msgpack


class BridgeConnection:
    """
    Manages the TCP connection to the Hytale server's RL bridge.

    Protocol: length-prefixed MessagePack messages.
    Each message is preceded by a 4-byte big-endian uint32 length header.
    """

    MAX_MESSAGE_BYTES = 16 * 1024 * 1024

    def __init__(self, host: str = "127.0.0.1", port: int = 5556, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None

    def connect(self):
        """Establish TCP connection to the bridge server."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.settimeout(self.timeout)
        try:
            self._socket.connect((self.host, self.port))
        except Exception:
            self.close()
            raise

    def send(self, msg: dict[str, Any]):
        """Send a MessagePack message with length prefix."""
        if self._socket is None:
            raise ConnectionError("Bridge connection is not open")
        data = msgpack.packb(msg, use_bin_type=True)
        if len(data) > self.MAX_MESSAGE_BYTES:
            raise ValueError(f"Message is too large: {len(data)} bytes")
        header = struct.pack(">I", len(data))
        self._socket.sendall(header + data)

    def recv(self) -> dict[str, Any]:
        """Receive a length-prefixed MessagePack message."""
        header = self._recv_exact(4)
        length = struct.unpack(">I", header)[0]
        if length <= 0 or length > self.MAX_MESSAGE_BYTES:
            raise ConnectionError(f"Invalid bridge frame length: {length}")
        data = self._recv_exact(length)
        response = msgpack.unpackb(data, raw=False)
        if not isinstance(response, dict):
            raise ConnectionError("Bridge response is not a MessagePack map")
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "Unknown bridge error"))
        return response

    def send_and_recv(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Send a message and wait for the response."""
        self.send(msg)
        return self.recv()

    def close(self):
        """Close the connection."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _recv_exact(self, n: int) -> bytes:
        """Read exactly n bytes from the socket."""
        if self._socket is None:
            raise ConnectionError("Bridge connection is not open")
        buf = bytearray()
        while len(buf) < n:
            chunk = self._socket.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed by server")
            buf.extend(chunk)
        return bytes(buf)

    @property
    def connected(self) -> bool:
        return self._socket is not None
