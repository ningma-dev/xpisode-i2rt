"""The devices over a controller that answers as told."""

import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import JsonValue, TypeAdapter

from xpisode_device import ManualClock
from xpisode_device.host import DeviceHost, State
from xpisode_device.transport import InProcessTransport

from xpisode_i2rt import arm
from xpisode_i2rt.core import RealtimeConfig, description
from xpisode_i2rt.limiter import ForceLimiter
from xpisode_i2rt.xpisode import YamFollower, YamLeader, check_joint_limits, wrap_offsets

LIMITS = (6.0, 1.0)


class FakeCore:
    def __init__(self, count: int = 7) -> None:
        self.pushed: list[np.ndarray] = []
        self.position = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 3.5])[:count]
        self.held = 0

    def check(self) -> None:
        pass

    def push(self, q: np.ndarray) -> None:
        self.pushed.append(q)

    def hold(self) -> None:
        self.held += 1

    def alive(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True

    def schedule(self) -> dict[str, Any]:
        return {"scheduler": "fifo", "priority": 80, "memory_locked": True}

    def state(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "target_position": self.pushed[-1] if self.pushed else self.position,
            "velocity": np.array([0.0] * 6 + [1.0])[: len(self.position)],
            "torque": np.arange(float(len(self.position))),
            "encoder_position": -0.35,
            "encoder_inputs": 2,
            "encoder_age_s": 0.2,
            "cycles": self.next_cycle(),
        }

    def next_cycle(self) -> int:
        self.cycle = getattr(self, "cycle", 0) + 1
        return self.cycle


def _follower() -> tuple[YamFollower, FakeCore]:
    device = YamFollower("can_x", gripper_limits=LIMITS)
    core = FakeCore()
    device._core = core  # type: ignore[assignment]
    device._clock = SimpleNamespace(now_ns=lambda: 7)  # type: ignore[assignment]
    device._after_start(np.r_[np.zeros(6), 3.5])
    return device, core


def _leader() -> tuple[YamLeader, FakeCore]:
    device = YamLeader("can_x")
    core = FakeCore(6)
    device._core = core  # type: ignore[assignment]
    device._clock = SimpleNamespace(now_ns=lambda: 7)  # type: ignore[assignment]
    return device, core


def test_follower_targets_are_clipped_and_in_motor_units() -> None:
    device, core = _follower()
    device.submit("joint_position", (np.array([9.0, 0, 0, 0, 0, 0]), np.array([0.5])))
    pushed = core.pushed[-1]
    assert pushed[0] == arm.joint_limits()[0, 1]
    assert pushed[6] == pytest.approx(3.5), "0 closed at 6, 1 open at 1"


def test_the_force_limiter_decides_the_gripper() -> None:
    device, core = _follower()
    device._force_limiter = SimpleNamespace(update=lambda state, now: 1.25)  # type: ignore[assignment]
    device.submit("joint_position", (np.zeros(6), np.array([1.0])))
    assert core.pushed[-1][6] == 1.25


def test_follower_reads_in_endpoint_units() -> None:
    device, _ = _follower()
    reading = device.read("state")
    joint_pos, _, joint_eff, gripper_pos, gripper_vel, gripper_eff = reading.values
    assert joint_pos.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert joint_eff.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert gripper_pos.tolist() == [0.5]
    assert gripper_vel.tolist() == [pytest.approx(-0.2)]
    assert gripper_eff.tolist() == [6.0]


def test_a_leader_reads_its_handle_and_clips_its_targets() -> None:
    device, core = _leader()
    joints, trigger, buttons, age = device.read("input").values
    assert joints.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert trigger.tolist() == [0.5]
    assert buttons.tolist() == [False, True]
    assert age.tolist() == [0.2]
    device.submit("joint_position", (np.array([9.0, 0, 0, 0, 0, 0]),))
    assert core.pushed[-1][0] == arm.joint_limits()[0, 1]


def test_a_leader_keeps_reading_through_a_handle_restart() -> None:
    """A restarting handle is silent for seconds; the joints still come, the handle's age says so."""
    device, core = _leader()
    state = core.state()
    core.state = lambda: {**state, "encoder_age_s": 3.5}  # type: ignore[method-assign]
    joints, trigger, buttons, age = device.read("input").values
    assert joints.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert age.tolist() == [3.5]


def test_a_leader_reports_a_handle_silent_for_too_long() -> None:
    device, core = _leader()
    state = core.state()
    core.state = lambda: {**state, "encoder_age_s": 10.5}  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="has not answered for 10.5 s"):
        device.read("input")


def test_halting_a_leader_lets_it_float() -> None:
    device, core = _leader()
    device.halt("arm")
    assert core.held == 1


def test_what_they_describe_passes_as_device_facts(tmp_path) -> None:  # noqa: ANN001
    """The station refuses a device whose facts are not JSON, and the device dies of it."""
    tuned = tmp_path / "arm.json"
    tuned.write_text('{"params": {"kp": [1, 2, 3, 4, 5, 6]}, "provenance": {"by": "x"}, "gripper_limits": [6.0, 1.0]}')
    follower = YamFollower("can_x", tuned=str(tuned))
    assert follower.realtime.kp == (1, 2, 3, 4, 5, 6)
    follower._core = FakeCore()  # type: ignore[assignment]
    leader, _ = _leader()
    for device in (follower, leader):
        info = device.describe()
        TypeAdapter(dict[str, JsonValue]).validate_python(info.device)
    assert follower.describe().device["realtime"]["tuned"]["by"] == "x"
    assert follower.describe().device["robot_info"]["gripper_limits"] == [6.0, 1.0]


def test_settings_the_controller_does_not_have_are_refused() -> None:
    with pytest.raises(ValueError, match="no such settings"):
        YamFollower("can_x", gripper_limits=LIMITS, kpp=(1,))


def test_a_joint_powered_a_turn_away_is_brought_back() -> None:
    assert wrap_offsets([4.0, -4.0, 0.5, 0, 0, 0, 6.4]) == [2 * np.pi, -2 * np.pi, 0.0, 0.0, 0.0, 0.0]


def test_a_lost_zero_stops_the_bring_up() -> None:
    check_joint_limits("can_x", np.zeros(6))
    with pytest.raises(RuntimeError, match="joints \\[2\\]"):
        check_joint_limits("can_x", np.array([0.0, -0.5, 0, 0, 0, 0]))


def test_what_the_controller_is_told() -> None:
    told = description(
        RealtimeConfig(coulomb=(0.5,) * 6, inertia_scale=0.25),
        "can_x",
        arm.model_path(arm.GRIPPER),
        [0.0] * 6,
        LIMITS,
    )
    motors = told["motors"]
    assert [m["id"] for m in motors] == [1, 2, 3, 4, 5, 6, 7]
    assert motors[0]["kp"] == 80.0
    assert motors[0]["coulomb"] == 0.5
    assert motors[1]["gravity_scale"] == 1.1
    assert motors[3]["inertia_scale"] == 0.25
    assert [m["torque_limit"] for m in (motors[0], motors[3])] == [28.0, 10.0]
    assert (motors[5]["max_velocity"], motors[5]["velocity_scale"]) == (4.0, 1.0)
    assert (motors[6]["max_velocity"], motors[6]["velocity_scale"]) == (30.0, 0.0)
    assert (motors[6]["low"], motors[6]["high"]) == (1.0, 6.0)
    assert told["floats"] is False


def test_a_leader_tells_the_controller_it_floats() -> None:
    told = description(RealtimeConfig(), "can_x", arm.model_path(arm.HANDLE), [0.0] * 6, None, True, 0x50E)
    assert len(told["motors"]) == 6
    assert told["floats"] is True
    assert told["encoder_id"] == 0x50E
    assert told["motors"][3]["float_kd"] == 0.3


def test_the_limiter_is_built_from_the_arm(tmp_path) -> None:  # noqa: ANN001
    device, _ = _follower()
    assert isinstance(device._force_limiter, ForceLimiter)
    assert device._force_limiter.limit_torque == pytest.approx(50.0 * 0.096 / 6.57)


@pytest.mark.parametrize(("device_class", "count"), [(YamFollower, 7), (YamLeader, 6)])
def test_the_real_host_takes_it_from_open_to_close(device_class: type, count: int) -> None:
    """What the host checks, a device's facts among them, only a host can tell."""
    core = FakeCore(count)
    settings = {"gripper_limits": LIMITS} if device_class is YamFollower else {}

    class Opened(device_class):  # type: ignore[misc, valid-type]
        def open(self, clock: Any) -> None:
            self._clock = clock
            self._core = core
            self._after_start(np.r_[np.zeros(6), [3.5] * (count - 6)])
            self._declare_model("state" if count == 7 else "input")
            self._park_joints = np.zeros(6)

    device = Opened("can_x", park_seconds=0.05, kp=[80, 80, 80, 20, 20, 20], **settings)
    clock = ManualClock()
    host = DeviceHost("arm", device, InProcessTransport(), clock)
    host.start()
    deadline = time.monotonic() + 5.0
    while host.state is not State.READY and time.monotonic() < deadline:
        host.step()
        clock.advance(1_000_000)
        time.sleep(0.001)
    assert host.state is State.READY, host._detail
    host.request_close("done")
    while not host.finished and time.monotonic() < deadline:
        host.step()
        clock.advance(1_000_000)
    assert core.closed
    assert core.pushed[-1][:6].tolist() == [0.0] * 6, "it parked before closing"


def test_a_leader_opens_once_its_handle_answers_through_the_controller(
    capsys: pytest.CaptureFixture[str],
) -> None:
    device, core = _leader()
    state = core.state()
    started = time.monotonic()
    # Silent for the first 0.3 s after the controller started, as a handle has been seen to be.
    core.state = lambda: {  # type: ignore[method-assign]
        **state,
        "encoder_age_s": time.monotonic() - started if time.monotonic() - started < 0.3 else 0.01,
    }
    device._wait_for_handle()
    assert time.monotonic() - started >= 0.3
    assert "first answered" in capsys.readouterr().err


def test_a_leader_whose_handle_never_answers_does_not_open() -> None:
    device, core = _leader()
    state = core.state()
    started = time.monotonic()
    core.state = lambda: {**state, "encoder_age_s": time.monotonic() - started}  # type: ignore[method-assign]
    device.realtime = replace(device.realtime, encoder_start_timeout_s=0.2)
    with pytest.raises(RuntimeError, match="did not answer within 0.2 s"):
        device._wait_for_handle()


def test_a_leader_reads_each_controller_cycle_once() -> None:
    device, core = _leader()
    state = core.state()
    core.state = lambda: state  # type: ignore[method-assign]
    assert device.read("input") is not None
    assert device.read("input") is None, "the same cycle again is nothing new"
    state = {**state, "cycles": state["cycles"] + 1}
    assert device.read("input") is not None
