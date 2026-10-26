"""YAM arms as xpisode device peers.

Everything robot-specific stays in i2rt: ``get_yam_robot`` connects, calibrates the gripper,
and runs the control thread (gravity compensation, PD, joint and gripper clipping, the gripper
force limiter); ``command_joint_pos`` sets the target that thread tracks, on a leader too;
``enter_gravity_comp_idle`` lets a leader float again; ``move_joints`` parks.
These classes only state what the robot offers in the xpisode device protocol, so the station
can record and drive it without knowing it is a YAM.

Run each arm in its own process with xpisode's generic host::

    xpisode-device run --station lab --name follower --factory xpisode_i2rt:YamFollower \\
        --config '{"channel": "can1"}'
    xpisode-device run --station lab --name leader --factory xpisode_i2rt:YamLeader \\
        --config '{"channel": "can0"}'
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperType
from xpisode_device import (
    Clock,
    Device,
    DeviceInfo,
    Endpoint,
    Limits,
    Operation,
    Reading,
    Stream,
    Values,
)
from xpisode_format.models import DeclaredModel
from xpisode_format.spec import Axis, Field, RecordStream, Role, TimeMeaning

from xpisode_i2rt.model import yam_model

ENCODER_WAIT_S = 5.0
"""How long the leader waits for the teaching handle's first report after connecting."""


def _json(value: Any) -> Any:
    """i2rt's robot info, as JSON for the Episode header."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if hasattr(value, "value"):  # enums
        return value.value
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    return value


def _joints(name: str, dof: int, unit: str, description: str = "") -> Field:
    labels = tuple(f"joint{i + 1}" for i in range(dof))
    return Field(
        name=name,
        dtype="float64",
        shape=(dof,),
        unit=unit,
        axes=(Axis(name="joint", labels=labels),),
        description=description,
    )


def _scalar(name: str, unit: str, description: str) -> Field:
    return Field(name=name, dtype="float64", shape=(1,), unit=unit, description=description)


def _observation(fields: Sequence[Field], hz: float, role: Role = Role.OBSERVATION) -> Stream:
    # i2rt keeps the newest state its control thread received; read() samples it, so the time
    # is when it was read (i2rt exposes no public state timestamp).
    spec = RecordStream(
        role=role, time_meaning=TimeMeaning.READ, fields=tuple(fields), nominal_hz=hz
    )
    return Stream(spec, poll_hz=hz)


class _Yam(Device):
    """Shared by both arms: connection facts and parking before the motors are let go."""

    def __init__(self, channel: str | None, arm: str, poll_hz: float, park_seconds: float) -> None:
        self.channel = channel
        self.arm_type = ArmType.from_string_name(arm)
        self.poll_hz = poll_hz
        self.park_seconds = park_seconds
        self._clock: Clock | None = None
        self._robot: Robot | None = None
        self._park_joints: np.ndarray | None = None
        self._dof = 0
        self._model_files: tempfile.TemporaryDirectory[str] | None = None
        self._model: DeclaredModel | None = None

    @property
    def robot(self) -> Robot:
        if self._robot is None:
            raise RuntimeError("the arm is not open")
        return self._robot

    def _connected(
        self,
        clock: Clock,
        robot: Robot,
        park_pose: Sequence[float] | None,
        gripper_type: GripperType,
        stream: str,
    ) -> None:
        self._clock, self._robot = clock, robot
        self._dof = len(robot.get_observations()["joint_pos"])
        # The model's directory is the device's while it is open: the station copies it when
        # a recording starts.
        self._model_files = tempfile.TemporaryDirectory(prefix="xpisode-i2rt-model-")
        self._model = yam_model(
            self.arm_type, gripper_type, self._dof, stream, Path(self._model_files.name)
        )
        # The arm rests where it was switched on; that is where it goes back to.
        rest = robot.get_joint_pos()[: self._dof]
        self._park_joints = np.array(park_pose if park_pose is not None else rest, dtype=np.float64)
        if self._park_joints.shape != (self._dof,):
            raise ValueError(f"park_pose needs {self._dof} joint positions")

    def _facts(self, driver: str, gripper_type: GripperType) -> dict[str, Any]:
        info = _json(self.robot.get_robot_info())
        return {
            "driver": driver,
            "i2rt": metadata.version("i2rt"),
            "channel": self.channel,
            "arm": self.arm_type.value,
            "gripper": gripper_type.value,
            "park_joints": self._park_joints.tolist() if self._park_joints is not None else None,
            "robot_info": info,
        }

    def _park(self) -> None:
        """i2rt's own interpolated move, joints only: the gripper stays as it is."""
        assert self._park_joints is not None
        current = self.robot.get_joint_pos()
        target = np.concatenate([self._park_joints, current[self._dof :]])
        self.robot.move_joints(target, time_interval_s=self.park_seconds)  # type: ignore[attr-defined]

    def run(self, operation: str) -> None:
        if operation != "park":
            raise ValueError(f"no operation {operation!r}")
        self._park()

    def close(self) -> None:
        """Park, then close: without power the arm has no brakes and falls."""
        if self._robot is None:
            return
        try:
            self._park()
        finally:
            robot, self._robot = self._robot, None
            robot.close()
            if self._model_files is not None:
                self._model_files.cleanup()


class YamFollower(_Yam):
    """A YAM arm with a gripper, driven by joint position targets.

    Opening moves the gripper when it needs calibration (i2rt's ``linear_4310`` does): i2rt
    drives it to both stops and keeps the measured limits, which go into the Episode header.
    Pass ``gripper_limits`` to skip calibration. The arm then holds the pose it is in.
    """

    driver = "i2rt.yam_follower"

    def __init__(
        self,
        channel: str | None = None,
        arm: str = "yam",
        gripper: str = "linear_4310",
        *,
        sim: bool = False,
        poll_hz: float = 200.0,
        park_pose: Sequence[float] | None = None,
        park_seconds: float = 3.0,
        gripper_limits: Sequence[float] | None = None,
    ) -> None:
        # No default bus: with several arms, a forgotten channel would drive another arm.
        if channel is None and not sim:
            raise ValueError("a real follower needs its CAN channel, such as can_follower_l")
        super().__init__(channel, arm, poll_hz, park_seconds)
        self.gripper_type = GripperType.from_string_name(gripper)
        self.sim = sim
        self.park_pose = park_pose
        self.gripper_limits = gripper_limits

    def open(self, clock: Clock) -> None:
        robot = get_yam_robot(
            channel=self.channel or "",  # i2rt ignores the channel in simulation, where it is None
            arm_type=self.arm_type,
            gripper_type=self.gripper_type,
            zero_gravity_mode=False,  # hold the current pose instead of floating
            gripper_limits_override=None
            if self.gripper_limits is None
            else np.array(self.gripper_limits),
            sim=self.sim,
        )
        if "gripper_pos" not in robot.get_observations():
            robot.close()
            raise ValueError(
                f"gripper {self.gripper_type.value!r} has no motor; the follower needs one"
            )
        self._connected(clock, robot, self.park_pose, self.gripper_type, "state")

    def describe(self) -> DeviceInfo:
        dof = self._dof
        joint_pos = _joints("joint_pos", dof, "rad")
        gripper_pos = _scalar("gripper_pos", "1", "i2rt normalized opening: 0 closed, 1 open")
        state = (
            joint_pos,
            _joints("joint_vel", dof, "rad/s"),
            _joints("joint_eff", dof, "N*m", "measured motor torque"),
            gripper_pos,
            _scalar("gripper_vel", "1/s", "rate of the normalized opening"),
            _scalar("gripper_eff", "N*m", "measured gripper motor torque"),
        )
        limits = np.asarray(self.robot.get_robot_info()["joint_limits"], dtype=np.float64)
        endpoint = Endpoint(
            resource="arm",
            kind="joint_position",
            fields=(joint_pos, gripper_pos),
            limits={
                "joint_pos": Limits(min=tuple(limits[:, 0]), max=tuple(limits[:, 1])),
                "gripper_pos": Limits(min=(0.0,), max=(1.0,)),
            },
            description="Targets for i2rt's position control; i2rt clips them to these limits.",
        )
        return DeviceInfo(
            device={**self._facts(self.driver, self.gripper_type), "sim": self.sim},
            model=self._model,
            streams={"state": _observation(state, self.poll_hz)},
            endpoints={"joint_position": endpoint},
            operations={
                "park": Operation(
                    moves=("arm",),
                    description="Move the joints to the park pose; gripper unchanged.",
                )
            },
        )

    def read(self, stream: str) -> Reading:
        assert self._clock is not None
        obs = self.robot.get_observations()
        keys = ("joint_pos", "joint_vel", "joint_eff", "gripper_pos", "gripper_vel", "gripper_eff")
        return Reading(
            self._clock.now_ns(), tuple(np.asarray(obs[k], dtype=np.float64) for k in keys)
        )

    def submit(self, endpoint: str, values: Values) -> None:
        self.robot.command_joint_pos(np.concatenate([values[0], values[1]]).astype(np.float64))

    def halt(self, resource: str) -> None:
        """Hold still where the arm is now."""
        # A copy: MotorChainRobot returns its live state array, and command_joint_pos clips
        # in place.
        self.robot.command_joint_pos(np.array(self.robot.get_joint_pos(), dtype=np.float64))


class YamLeader(_Yam):
    """A YAM arm with the teaching handle, read as an operator input, and driven to follow
    the follower while somebody else drives that.

    i2rt keeps it in gravity compensation, so it floats under the operator's hand. Joint
    targets make i2rt's PD control track them, with its gains for the arm; halting lets it
    float again. The handle reports its trigger and buttons through the same CAN bus; the
    trigger cannot be driven.
    """

    driver = "i2rt.yam_leader"

    def __init__(
        self,
        channel: str,
        arm: str = "yam",
        *,
        poll_hz: float = 200.0,
        park_pose: Sequence[float] | None = None,
        park_seconds: float = 3.0,
        ee_mass: float | None = None,
    ) -> None:
        super().__init__(channel, arm, poll_hz, park_seconds)
        self.bus = channel
        self.park_pose = park_pose
        self.ee_mass = ee_mass
        self._buttons = 0

    def open(self, clock: Clock) -> None:
        robot = get_yam_robot(
            channel=self.bus,
            arm_type=self.arm_type,
            gripper_type=GripperType.YAM_TEACHING_HANDLE,
            zero_gravity_mode=True,
            ee_mass=self.ee_mass,
        )
        deadline = time.monotonic() + ENCODER_WAIT_S
        while not self._handle(robot):
            if time.monotonic() > deadline:
                robot.close()
                raise RuntimeError(
                    f"the teaching handle on {self.channel} did not report in {ENCODER_WAIT_S} s"
                )
            time.sleep(0.01)
        self._buttons = len(self._handle(robot)[0].io_inputs)
        self._connected(clock, robot, self.park_pose, GripperType.YAM_TEACHING_HANDLE, "input")

    @staticmethod
    def _handle(robot: Robot) -> Any:
        """The handle's encoder reports; None until the CAN thread received the first one."""
        return robot.motor_chain.get_same_bus_device_states()  # type: ignore[attr-defined]

    def describe(self) -> DeviceInfo:
        fields = (
            _joints("joint_pos", self._dof, "rad"),
            _scalar("trigger", "1", "teaching handle encoder: 0 released, 1 squeezed"),
            Field(
                name="buttons",
                dtype="bool",
                shape=(self._buttons,),
                unit="1",
                axes=(Axis(name="button"),),
                description="teaching handle inputs, pressed = true",
            ),
        )
        limits = np.asarray(self.robot.get_robot_info()["joint_limits"], dtype=np.float64)
        endpoint = Endpoint(
            resource="arm",
            kind="joint_position",
            fields=(fields[0],),
            limits={
                "joint_pos": Limits(
                    min=tuple(limits[: self._dof, 0]), max=tuple(limits[: self._dof, 1])
                )
            },
            description="Targets i2rt's PD control tracks; without them the leader floats.",
        )
        return DeviceInfo(
            device=self._facts(self.driver, GripperType.YAM_TEACHING_HANDLE),
            model=self._model,
            streams={"input": _observation(fields, self.poll_hz, Role.INPUT)},
            endpoints={"joint_position": endpoint},
            operations={
                "park": Operation(moves=("arm",), description="Move the leader to its park pose.")
            },
        )

    def read(self, stream: str) -> Reading:
        assert self._clock is not None
        joint_pos = np.asarray(self.robot.get_observations()["joint_pos"], dtype=np.float64)
        handle = self._handle(self.robot)[0]
        now = self._clock.now_ns()
        return Reading(
            now, (joint_pos, np.array([handle.position]), np.array(handle.io_inputs, dtype=bool))
        )

    def submit(self, endpoint: str, values: Values) -> None:
        self.robot.command_joint_pos(np.asarray(values[0], dtype=np.float64))

    def halt(self, resource: str) -> None:
        """A leader stops by floating again under the operator's hand."""
        self.robot.enter_gravity_comp_idle()  # type: ignore[attr-defined]
