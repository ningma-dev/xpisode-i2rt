"""A SocketCAN socket, for what happens before the controller takes the bus."""

from __future__ import annotations

import socket
import struct
import time
from pathlib import Path

_FRAME = struct.Struct("=IB3x8s")


class Bus:
    def __init__(self, channel: str) -> None:
        self.channel = channel
        self._socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self._socket.bind((channel,))

    def send(self, can_id: int, data: bytes) -> None:
        self._socket.send(_FRAME.pack(can_id, len(data), data.ljust(8, b"\0")))

    def receive(self, timeout: float) -> tuple[int, bytes] | None:
        self._socket.settimeout(max(timeout, 1e-4))
        try:
            frame = self._socket.recv(_FRAME.size)
        except TimeoutError:
            return None
        can_id, length, data = _FRAME.unpack(frame)
        return can_id, data[:length]

    def drain(self, seconds: float = 0.02) -> int:
        count = 0
        end = time.monotonic() + seconds
        while time.monotonic() < end and self.receive(end - time.monotonic()) is not None:
            count += 1
        return count

    def close(self) -> None:
        self._socket.close()


def require_idle(channel: str, seconds: float = 0.2) -> None:
    """Nothing else may be driving this bus: a running station's arm would lose its commands."""
    counter = Path(f"/sys/class/net/{channel}/statistics/rx_packets")
    if not counter.exists():
        raise RuntimeError(f"no CAN interface {channel}; is it up?")
    before = int(counter.read_text())
    time.sleep(seconds)
    if int(counter.read_text()) != before:
        raise RuntimeError(f"something is using {channel}; stop it first")
