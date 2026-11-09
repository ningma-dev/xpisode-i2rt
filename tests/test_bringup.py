"""The bring-up's rules against a bus that answers as told."""

import struct
from collections import deque

import numpy as np
import pytest

from xpisode_i2rt import bringup
from xpisode_i2rt.arm import KINDS
from xpisode_i2rt.bringup import Motor
from xpisode_i2rt.dm import encode


def _reply(motor: Motor, state: int, position: float = 0.0) -> bytes:
    kind = KINDS[motor.kind]
    p = int((position + kind.position) * 65535 / (2 * kind.position))
    return bytes([state << 4 | motor.id, p >> 8, p & 0xFF, 0x80, 0x08, 0x00, 30, 30])


class Bus:
    channel = "can_x"

    def __init__(self) -> None:
        self.sent: list[tuple[int, bytes]] = []
        self.inbox: deque[tuple[int, bytes]] = deque()
        self.registers: dict[tuple[int, int], bytes] = {}
        self.states: dict[int, list[int]] = {}
        self.position = 0.0

    def send(self, can_id: int, data: bytes) -> None:
        self.sent.append((can_id, data))
        if can_id == bringup.REGISTER_ID:
            value = self.registers.get((data[0], data[3]))
            if value is not None:
                self.inbox.append((0x1F, bytes([data[0], 0, data[2], data[3]]) + value))
        elif can_id < 16:
            motor = Motor(can_id, "DM4310")
            if data[-1] == 0xFC:
                states = self.states.get(can_id, [1])
                self.inbox.append((can_id + 16, _reply(motor, states.pop(0) if len(states) > 1 else states[0])))
            elif data[-1] != 0xFB:
                self.inbox.append((can_id + 16, _reply(motor, 1, self.position)))

    def receive(self, timeout: float):  # noqa: ANN201
        return self.inbox.popleft() if self.inbox else None

    def drain(self, seconds: float = 0.02) -> int:
        count = len(self.inbox)
        self.inbox.clear()
        return count


def _good_registers(bus: Bus, motor: Motor) -> None:
    kind = KINDS[motor.kind]
    bus.registers[motor.id, bringup.REG_GR] = struct.pack("<f", kind.gear_ratio)
    bus.registers[motor.id, bringup.REG_CTRL_MODE] = (1).to_bytes(4, "little")
    bus.registers[motor.id, bringup.REG_PMAX] = struct.pack("<f", kind.position)
    bus.registers[motor.id, bringup.REG_VMAX] = struct.pack("<f", kind.velocity)
    bus.registers[motor.id, bringup.REG_TMAX] = struct.pack("<f", kind.torque)


def test_a_motor_set_up_as_expected_passes() -> None:
    bus = Bus()
    motors = [Motor(1, "DM4340"), Motor(4, "DM4310")]
    for motor in motors:
        _good_registers(bus, motor)
    bringup.check_motors(bus, motors)
    assert all(can_id == bringup.REGISTER_ID for can_id, _ in bus.sent), "it only reads"


@pytest.mark.parametrize(
    ("register", "value", "words"),
    [
        (bringup.REG_GR, struct.pack("<f", 10.0), "gear ratio"),
        (bringup.REG_CTRL_MODE, (2).to_bytes(4, "little"), "control mode"),
        (bringup.REG_TMAX, struct.pack("<f", 54.0), "TMAX"),
    ],
)
def test_a_motor_set_up_otherwise_stops_the_bring_up(register: int, value: bytes, words: str) -> None:
    bus = Bus()
    motor = Motor(1, "DM4340")
    _good_registers(bus, motor)
    bus.registers[1, register] = value
    with pytest.raises(RuntimeError, match=words):
        bringup.check_motors(bus, [motor])


def test_a_silent_motor_stops_the_bring_up() -> None:
    with pytest.raises(RuntimeError, match="does not answer"):
        bringup.read_register(Bus(), 1, bringup.REG_GR, tries=1, wait=0.001)


def test_enabling_clears_errors_and_gives_up() -> None:
    motor = Motor(1, "DM4310")
    bus = Bus()
    bus.states[1] = [0xD, 1]
    assert bringup.enable(bus, [motor], sleep=lambda _: None)[0].state == 1
    assert (1, bytes([0xFF] * 7 + [0xFB])) in bus.sent

    stuck = Bus()
    stuck.states[1] = [0xE]
    with pytest.raises(RuntimeError, match="overload"):
        bringup.enable(stuck, [motor], sleep=lambda _: None)


def test_a_handle_with_old_firmware_is_refused() -> None:
    bus = Bus()
    bus.inbox.append((0x50E, bytes([1, 0x83, 2, 1, 0])))
    bus.drain = lambda seconds=0.02: 0  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="older"):
        bringup.check_handle(bus, 0x50E, (2, 2, 12))


def test_calibration_finds_both_stops_with_the_arm_holding() -> None:
    motors = [Motor(1, "DM4340"), Motor(7, "DM4310")]
    bus = Bus()
    t = [0.0]

    def sleep(seconds: float) -> None:
        t[0] += seconds
        # It moves while the torque pushes, up to a stop each way.
        torque_frame = bus.sent[-1][1]
        torque = (((torque_frame[6] & 0xF) << 8) | torque_frame[7]) / 4095 * 20 - 10
        bus.position = float(np.clip(bus.position + np.sign(torque) * seconds * 5 * (abs(torque) > 0.1), 1.2, 6.4))

    bus.position = 3.0
    limits = bringup.calibrate_gripper(
        bus, motors, [0.1, 3.0], [80.0, 20.0], [5.0, 0.5], 1, clock=lambda: t[0], sleep=sleep
    )
    assert limits == pytest.approx((6.4, 1.2), abs=0.01)
    arm_frames = {data for can_id, data in bus.sent if can_id == 1}
    assert arm_frames == {encode(KINDS["DM4340"], 0.1, 0.0, 80.0, 5.0, 0.0)}, "the arm only holds"
