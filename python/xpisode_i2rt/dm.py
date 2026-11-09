"""Damiao MIT frames, for bringing an arm up before the controller takes the bus."""

from __future__ import annotations

from dataclasses import dataclass

from xpisode_i2rt.arm import MotorKind

KP_MAX = 500.0
KD_MAX = 5.0
REPLY_OFFSET = 16
STATE_DISABLED = 0x0
STATE_ENABLED = 0x1
STATES = {
    0x0: "disabled",
    0x1: "enabled",
    0x8: "over voltage",
    0x9: "under voltage",
    0xA: "over current",
    0xB: "MOS over temperature",
    0xC: "rotor over temperature",
    0xD: "lost communication",
    0xE: "overload",
}
ENABLE = bytes([0xFF] * 7 + [0xFC])
CLEAR_ERROR = bytes([0xFF] * 7 + [0xFB])


def _uint(x: float, low: float, high: float, bits: int) -> int:
    return int((min(max(x, low), high) - low) * ((1 << bits) - 1) / (high - low))


def _float(x: int, low: float, high: float, bits: int) -> float:
    return x * (high - low) / ((1 << bits) - 1) + low


def encode(
    kind: MotorKind, position: float, velocity: float, kp: float, kd: float, torque: float
) -> bytes:
    p = _uint(position, -kind.position, kind.position, 16)
    v = _uint(velocity, -kind.velocity, kind.velocity, 12)
    gain_p = _uint(kp, 0.0, KP_MAX, 12)
    gain_d = _uint(kd, 0.0, KD_MAX, 12)
    t = _uint(torque, -kind.torque, kind.torque, 12)
    return bytes(
        [
            p >> 8,
            p & 0xFF,
            v >> 4,
            ((v & 0xF) << 4) | (gain_p >> 8),
            gain_p & 0xFF,
            gain_d >> 4,
            ((gain_d & 0xF) << 4) | (t >> 8),
            t & 0xFF,
        ]
    )


@dataclass(frozen=True)
class Reply:
    state: int
    position: float
    velocity: float
    torque: float


def decode(kind: MotorKind, data: bytes) -> Reply:
    return Reply(
        data[0] >> 4,
        _float((data[1] << 8) | data[2], -kind.position, kind.position, 16),
        _float((data[3] << 4) | (data[4] >> 4), -kind.velocity, kind.velocity, 12),
        _float(((data[4] & 0xF) << 8) | data[5], -kind.torque, kind.torque, 12),
    )
