"""The YAM device peers against xpisode's protocol: the follower in MuJoCo sim, the leader
(real hardware only in i2rt) with a stand-in for its CAN bus."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from xpisode_device import DeviceHost, InProcessTransport, ManualClock, MonotonicClock, check
from xpisode_format.models import content_hash

from xpisode_i2rt import YamFollower, YamLeader, yam


@contextmanager
def hosted(device: Any) -> Iterator[InProcessTransport]:
    """The device behind xpisode's generic host, as ``xpisode-device run`` runs it."""
    transport = InProcessTransport()
    host = DeviceHost("arm", device, transport, MonotonicClock())
    stop = threading.Event()
    thread = threading.Thread(target=host.run, args=(stop,), daemon=True)
    thread.start()
    try:
        yield transport
    finally:
        stop.set()
        thread.join(10)


def test_the_sim_follower_conforms() -> None:
    follower = YamFollower(sim=True, park_seconds=0.1)
    with hosted(follower) as transport:
        report = check(
            transport,
            "arm",
            commands={"joint_position": [np.full(6, 0.1), np.array([0.5])]},
            operations=True,
            shutdown=True,
            timeout_s=5.0,
        )
    assert report.ok, report.problems


def test_the_follower_declares_the_model_i2rt_makes() -> None:
    """A self-contained MJCF: MuJoCo loads it from its own directory, its
    joints are the ones joint_pos drives, and it goes when the device closes."""
    mujoco = pytest.importorskip("mujoco")
    follower = YamFollower(sim=True, park_seconds=0.05)
    follower.open(ManualClock())
    model = follower.describe().model
    assert model is not None
    files = Path(model.files)
    assert model.sha256 == content_hash(files)
    loaded = mujoco.MjModel.from_xml_path(str(files / model.entry))
    joints = {loaded.joint(i).name for i in range(loaded.njnt)}
    ((mapped),) = model.joints
    assert (mapped.stream, mapped.field) == ("state", "joint_pos")
    assert set(mapped.names) == {f"joint{i}" for i in range(1, 7)} <= joints
    assert "linear_4310" in model.source
    follower.close()
    assert not files.exists()


def test_the_follower_tracks_targets_and_parks_where_it_started() -> None:
    follower = YamFollower(sim=True, park_seconds=0.05)
    follower.open(ManualClock())
    robot = follower.robot
    follower.submit("joint_position", (np.full(6, 0.3), np.array([0.2])))
    np.testing.assert_allclose(np.asarray(follower.read("state").values[0]), 0.3)
    follower.run("park")
    np.testing.assert_allclose(robot.get_joint_pos()[:6], 0.0, atol=1e-9)
    np.testing.assert_allclose(robot.get_joint_pos()[6], 0.2)  # the gripper stays
    held = robot.get_joint_pos()
    follower.halt("arm")
    np.testing.assert_allclose(robot.get_joint_pos(), held)
    info = follower.describe()
    assert info.device["sim"] is True
    assert info.device["park_joints"] == [0.0] * 6
    follower.close()


class _Handle:
    """i2rt's teaching handle reports, as ``get_same_bus_device_states`` returns them."""

    def __init__(self) -> None:
        self.states: list[Any] | None = None

    def get_same_bus_device_states(self) -> Any:
        return self.states


class _LeaderArm:
    """What get_yam_robot returns for the teaching handle, recording what was done to it."""

    def __init__(self) -> None:
        self.motor_chain = _Handle()
        self.motor_chain.states = [SimpleNamespace(position=0.25, io_inputs=[True, False])]
        self.pos = np.linspace(-0.5, 0.5, 6)
        self.calls: list[str] = []

    def get_observations(self) -> dict:
        return {"joint_pos": self.pos.copy(), "joint_vel": np.zeros(6), "joint_eff": np.zeros(6)}

    def get_joint_pos(self) -> np.ndarray:
        return self.pos.copy()

    def get_robot_info(self) -> dict:
        return {"kp": np.ones(6), "joint_limits": np.tile([-3.0, 3.0], (6, 1))}

    def move_joints(self, target: np.ndarray, time_interval_s: float) -> None:
        self.calls.append("move_joints")
        self.pos = np.asarray(target, dtype=float)

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.calls.append("command_joint_pos")
        self.target = np.asarray(target, dtype=float)

    def enter_gravity_comp_idle(self) -> None:
        self.calls.append("float")

    def close(self) -> None:
        self.calls.append("close")


def test_the_leader_reports_joints_trigger_and_buttons(monkeypatch: pytest.MonkeyPatch) -> None:
    arm = _LeaderArm()
    monkeypatch.setattr(yam, "get_yam_robot", lambda **kwargs: arm)
    leader = YamLeader(channel="can0", park_seconds=0.0)
    leader.open(ManualClock())
    info = leader.describe()
    assert info.model is not None
    assert (info.model.joints[0].stream, info.model.joints[0].field) == ("input", "joint_pos")
    assert "yam_teaching_handle" in info.model.source
    assert [f.name for f in info.streams["input"].spec.fields] == [
        "joint_pos",
        "trigger",
        "buttons",
    ]
    assert info.streams["input"].spec.fields[2].shape == (2,)
    joint_pos, trigger, buttons = (np.asarray(v) for v in leader.read("input").values)
    np.testing.assert_allclose(joint_pos, arm.pos)
    assert trigger.tolist() == [0.25]
    assert buttons.tolist() == [True, False]
    endpoint = info.endpoints["joint_position"]
    assert [f.name for f in endpoint.fields] == ["joint_pos"]  # the trigger cannot be driven
    leader.submit("joint_position", (np.full(6, 0.1),))
    np.testing.assert_allclose(arm.target, 0.1)
    leader.halt("arm")  # it floats again
    assert arm.calls == ["command_joint_pos", "float"]
    arm.calls.clear()
    start = arm.pos.copy()
    arm.pos = np.zeros(6)  # the operator moved it
    leader.close()
    assert arm.calls == ["move_joints", "close"]  # parked where it was switched on, then let go
    np.testing.assert_allclose(arm.pos, start)


def test_a_silent_handle_fails_the_open(monkeypatch: pytest.MonkeyPatch) -> None:
    arm = _LeaderArm()
    arm.motor_chain.states = None
    monkeypatch.setattr(yam, "get_yam_robot", lambda **kwargs: arm)
    monkeypatch.setattr(yam, "ENCODER_WAIT_S", 0.05)
    with pytest.raises(RuntimeError, match="did not report"):
        YamLeader(channel="can0").open(ManualClock())
    assert arm.calls == ["close"]


def test_a_real_arm_has_no_default_bus() -> None:
    with pytest.raises(ValueError, match="CAN channel"):
        YamFollower()
    with pytest.raises(TypeError):
        YamLeader()  # pyright: ignore[reportCallIssue]
