"""From a bus nobody is using to an arm whose motors are enabled, before the controller starts.

It writes nothing to a motor's Flash. A motor set up differently from what this expects stops the
bring-up with what to fix, since it silently rescales every command and reading otherwise.
"""

from __future__ import annotations

import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from xpisode_i2rt.arm import KINDS, MotorKind
from xpisode_i2rt.dm import (
    CLEAR_ERROR,
    ENABLE,
    REPLY_OFFSET,
    STATE_ENABLED,
    STATES,
    Reply,
    decode,
    encode,
)

REGISTER_ID = 0x7FF
READ_REGISTER = 0x33
REG_CTRL_MODE, REG_GR, REG_PMAX, REG_VMAX, REG_TMAX = 10, 20, 21, 22, 23
MIT_MODE = 1
TOLERANCE = 1e-3
ENABLE_ATTEMPTS = 5

ALL_DEVICES = 0xFF
ENCODER_SET_REPORT_FREQUENCY = 0x01
ENCODER_VERSION = 0x03
ENCODER_REPLIED = 0x80

CALIBRATION_TORQUE = 0.5
CALIBRATION_SECONDS = 2.0
CALIBRATION_STILL = 0.01
CALIBRATION_CHECK_S = 0.05
CALIBRATION_FRAME_S = 0.005


class Bus(Protocol):
    channel: str

    def send(self, can_id: int, data: bytes) -> None: ...
    def receive(self, timeout: float) -> tuple[int, bytes] | None: ...
    def drain(self, seconds: float = 0.02) -> int: ...


@dataclass(frozen=True)
class Motor:
    id: int
    kind: str

    @property
    def spec(self) -> MotorKind:
        return KINDS[self.kind]


def read_register(bus: Bus, motor: int, register: int, tries: int = 5, wait: float = 0.02) -> bytes:
    for _ in range(tries):
        bus.send(REGISTER_ID, bytes([motor, 0, READ_REGISTER, register, 0, 0, 0, 0]))
        end = time.monotonic() + wait
        while (left := end - time.monotonic()) > 0:
            frame = bus.receive(left)
            if frame is None:
                break
            data = frame[1]
            if len(data) == 8 and data[:1] == bytes([motor]) and data[2:4] == bytes([READ_REGISTER, register]):
                return data[4:8]
    raise RuntimeError(f"motor {motor} on {bus.channel} does not answer; is the arm powered?")


def _as_float(raw: bytes) -> float:
    return struct.unpack("<f", raw)[0]


def check_motors(bus: Bus, motors: Sequence[Motor]) -> None:
    """Each motor is the type expected, in MIT mode, with the scale the frames assume."""
    for motor in motors:
        wanted = motor.spec
        gear = _as_float(read_register(bus, motor.id, REG_GR))
        if abs(gear - wanted.gear_ratio) > TOLERANCE:
            raise RuntimeError(
                f"motor {motor.id} on {bus.channel} has gear ratio {gear:g}: not a {motor.kind}"
            )
        mode = int.from_bytes(read_register(bus, motor.id, REG_CTRL_MODE), "little")
        if mode != MIT_MODE:
            raise RuntimeError(
                f"motor {motor.id} on {bus.channel} is in control mode {mode}, not MIT ({MIT_MODE}); "
                "set it with the motor's debugging tool"
            )
        for register, name, value in (
            (REG_PMAX, "PMAX", wanted.position),
            (REG_VMAX, "VMAX", wanted.velocity),
            (REG_TMAX, "TMAX", wanted.torque),
        ):
            actual = _as_float(read_register(bus, motor.id, register))
            if abs(actual - value) > TOLERANCE:
                raise RuntimeError(
                    f"motor {motor.id} on {bus.channel} has {name} {actual:g}, expected {value:g}: "
                    "every command and reading would be off by that ratio"
                )


def check_handle(
    bus: Bus, encoder_id: int, firmware: Sequence[int], wait: float = 0.5
) -> tuple[int, ...]:
    """The handle answers, with firmware that can, and only reports when polled."""
    bus.drain(0.05)
    bus.send(encoder_id, bytes([ALL_DEVICES, ENCODER_VERSION]))
    version = None
    reports = 0
    end = time.monotonic() + wait
    while version is None and (left := end - time.monotonic()) > 0:
        frame = bus.receive(left)
        if frame is None:
            break
        can_id, data = frame
        if can_id == encoder_id + 1:
            reports += 1
        elif can_id == encoder_id and len(data) == 5 and data[1] == ENCODER_VERSION | ENCODER_REPLIED:
            version = tuple(data[2:5])
    if version is None:
        raise RuntimeError(f"the teaching handle on {bus.channel} does not answer")
    if version < tuple(firmware):
        raise RuntimeError(
            f"the teaching handle on {bus.channel} has firmware {version}, older than {tuple(firmware)}"
        )
    if reports or bus.drain(0.05):
        # It streams on its own: polling would race those frames. This persists in the handle.
        bus.send(encoder_id, bytes([ALL_DEVICES, ENCODER_SET_REPORT_FREQUENCY, 0]))
        time.sleep(0.2)
        if bus.drain(0.1):
            raise RuntimeError(f"the teaching handle on {bus.channel} will not stop reporting")
    return version


def enable(bus: Bus, motors: Sequence[Motor], sleep: Callable[[float], None] = time.sleep) -> list[Reply]:
    """Enable each motor, clearing what it reports; a motor that stays in error stops the bring-up."""
    bus.drain(0.05)
    replies = []
    for motor in motors:
        sleep(0.003)
        state = None
        for _ in range(ENABLE_ATTEMPTS):
            reply = _exchange(bus, motor, ENABLE)
            state = None if reply is None else reply.state
            if state == STATE_ENABLED:
                replies.append(reply)
                break
            for _ in range(3):
                bus.send(motor.id, CLEAR_ERROR)
            sleep(0.005)
            bus.drain(0.005)
        else:
            what = "no reply" if state is None else STATES.get(state, hex(state))
            raise RuntimeError(f"motor {motor.id} on {bus.channel} will not enable: {what}")
    return replies


def _exchange(bus: Bus, motor: Motor, data: bytes, wait: float = 0.02) -> Reply | None:
    bus.send(motor.id, data)
    end = time.monotonic() + wait
    while (left := end - time.monotonic()) > 0:
        frame = bus.receive(left)
        if frame is None:
            return None
        if frame[0] == motor.id + REPLY_OFFSET and len(frame[1]) == 8:
            return decode(motor.spec, frame[1])
    return None


def hold(
    bus: Bus,
    motors: Sequence[Motor],
    positions: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    torques: Sequence[float] | None = None,
) -> list[Reply | None]:
    """One round of commands. Motors keep the last one until the controller's first."""
    out = []
    for i, motor in enumerate(motors):
        torque = 0.0 if torques is None else torques[i]
        out.append(_exchange(bus, motor, encode(motor.spec, positions[i], 0.0, kp[i], kd[i], torque), 0.003))
    return out


def calibrate_gripper(
    bus: Bus,
    motors: Sequence[Motor],
    positions: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    direction: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[float, float]:
    """Drive the gripper, the last motor, to both stops on a small torque, the arm holding. The
    limits in motor radians: closed then open, as far as ``direction`` says which is which."""
    hold_kp, hold_kd = list(kp), list(kd)
    hold_kp[-1] = hold_kd[-1] = 0.0
    seen = [positions[-1]]
    for sign in (1.0, -1.0):
        torques = [0.0] * len(motors)
        torques[-1] = sign * CALIBRATION_TORQUE
        begin = clock()
        last_check = begin
        last, still = None, 0
        while clock() - begin < CALIBRATION_SECONDS:
            replies = hold(bus, motors, positions, hold_kp, hold_kd, torques)
            reply = replies[-1]
            if reply is not None and clock() - last_check >= CALIBRATION_CHECK_S:
                last_check = clock()
                seen.append(reply.position)
                still = still + 1 if last is not None and abs(reply.position - last) < CALIBRATION_STILL else 0
                last = reply.position
                if still >= 3:
                    break
            sleep(CALIBRATION_FRAME_S)
        pause_end = clock() + 0.3
        while clock() < pause_end:
            hold(bus, motors, positions, hold_kp, hold_kd, torques)
            sleep(CALIBRATION_FRAME_S)
    hold(bus, motors, positions, hold_kp, hold_kd)
    low, high = min(seen), max(seen)
    return (high, low) if direction > 0 else (low, high)
