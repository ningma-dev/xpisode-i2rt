"""YAM arms as xpisode device peers, under ``i2rt-rtcore``.

Run each arm in its own process with xpisode's generic host::

    xpisode-device run --station lab --name follower --factory xpisode_i2rt.xpisode:YamFollower \\
        --config '{"channel": "can_follower_r", "tuned": "tuned/can_follower_r.json"}'
    xpisode-device run --station lab --name leader --factory xpisode_i2rt.xpisode:YamLeader \\
        --config '{"channel": "can_leader_r"}'

The controller is a process of its own, so Python's pauses reach neither the bus nor the torques.
These classes bring the arm up, start it, and say what it offers in the device protocol, so the
station can record and drive it without knowing it is a YAM.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
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
from xpisode_format.models import DeclaredModel, JointMap, declare
from xpisode_format.spec import Axis, Field, RecordStream, Role, TimeMeaning

from xpisode_i2rt import arm, bringup
from xpisode_i2rt.can import Bus, require_idle
from xpisode_i2rt.core import Core, RealtimeConfig, description, move, stop_orphans
from xpisode_i2rt.limiter import ForceLimiter, GripperState

JOINT_LIMIT_SLACK_RAD = 0.1
"""How far past its limits a joint may be found at startup: further means a lost zero."""
ENTRY = "model.xml"


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


def _observation(fields_: Sequence[Field], hz: float, role: Role = Role.OBSERVATION) -> Stream:
    # The real-time core publishes a state at the start of each cycle from the replies that came
    # in during the cycle before; the time is that start, on the host's CLOCK_MONOTONIC.
    spec = RecordStream(
        role=role,
        time_meaning=TimeMeaning.RECEIVE,
        fields=tuple(fields_),
        nominal_hz=hz,
        description=(
            "Stamped when the real-time core published the state: the start of its cycle; "
            "each motor's reply arrived within the cycle before."
        ),
    )
    return Stream(spec, poll_hz=hz)


def wrap_offsets(positions: Sequence[float]) -> list[float]:
    """A motor's multi-turn reading of a joint that may have been powered a turn away from zero."""
    return [
        -2 * np.pi if q < -np.pi else 2 * np.pi if q > np.pi else 0.0 for q in positions[: arm.DOF]
    ]


def check_joint_limits(channel: str, joints: np.ndarray) -> None:
    limits = arm.joint_limits()
    outside = (joints < limits[:, 0] - JOINT_LIMIT_SLACK_RAD) | (
        joints > limits[:, 1] + JOINT_LIMIT_SLACK_RAD
    )
    if np.any(outside):
        raise RuntimeError(
            f"{channel}: joints {(np.where(outside)[0] + 1).tolist()} are at "
            f"{np.round(joints[outside], 3).tolist()}, past their limits: a lost zero, or the wrong arm"
        )


def _tuned(path: str | None) -> dict[str, Any]:
    return {} if path is None else json.loads(Path(path).read_text())


class _Yam(Device):
    """Shared by both arms: bringing it up, the controller, and parking before letting go."""

    hand = ""
    floats = False

    def __init__(
        self,
        channel: str,
        poll_hz: float,
        park_pose: Sequence[float] | None,
        park_seconds: float,
        tuned: str | None,
        settings: Mapping[str, Any],
    ) -> None:
        names = {f.name for f in fields(RealtimeConfig)}
        unknown = set(settings) - names
        if unknown:
            raise ValueError(f"no such settings: {sorted(unknown)}")
        self.channel = channel
        self.poll_hz = poll_hz
        self.park_pose = park_pose
        self.park_seconds = park_seconds
        self.tuned_path = tuned
        self._tuned = _tuned(tuned)
        merged = {**self._tuned.get("params", {}), **settings}
        self.realtime = RealtimeConfig(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in merged.items()}
        )
        self._clock: Clock | None = None
        self._last_cycle: int | None = None
        self._core: Core | None = None
        self._offsets: list[float] = [0.0] * arm.DOF
        self._park_joints: np.ndarray | None = None
        self._model_files: tempfile.TemporaryDirectory[str] | None = None
        self._model: DeclaredModel | None = None

    @property
    def core(self) -> Core:
        if self._core is None:
            raise RuntimeError("the arm is not open")
        return self._core

    def _motors(self) -> list[bringup.Motor]:
        spec = arm.spec()
        return [bringup.Motor(i + 1, kind) for i, kind in enumerate(spec["motors"])]

    def _encoder_id(self) -> int | None:
        return None

    def _gripper_limits(self) -> Sequence[float] | None:
        return None

    def _bring_up_hand(self, bus: Bus, motors: list[bringup.Motor], raw: list[float]) -> None:
        """What the hand needs with the motors enabled and holding."""

    def _description(self, config: RealtimeConfig) -> dict[str, Any]:
        return description(
            config,
            self.channel,
            arm.model_path(self.hand),
            self._offsets,
            self._gripper_limits(),
            self.floats,
            self._encoder_id(),
        )

    def _open_arm(self, clock: Clock, stream: str) -> None:
        self._clock = clock
        stop_orphans(self.channel)
        require_idle(self.channel)
        spec = arm.spec()
        motors = self._motors()
        bus = Bus(self.channel)
        try:
            bringup.check_motors(bus, motors)
            if self._encoder_id() is not None:
                handle = spec["handle"]
                bringup.check_handle(bus, handle["encoder_id"], handle["firmware"])
            replies = bringup.enable(bus, motors)
            raw = [reply.position for reply in replies]
            self._offsets = wrap_offsets(raw)
            joints = np.array(raw[: arm.DOF]) - np.array(self._offsets)
            check_joint_limits(self.channel, joints)
            gains = [*(self.realtime.kp or spec["kp"])]
            damping = [*(self.realtime.kd or spec["kd"])]
            if self.floats:
                gains, damping = [0.0] * arm.DOF, list(spec["float_kd"])
            else:
                gripper = spec["gripper"]
                gains, damping = [*gains, gripper["kp"]], [*damping, gripper["kd"]]
            # Held from now on, on what the motors last heard, until the controller's first frame.
            bringup.hold(bus, motors, raw, gains, damping)
            self._bring_up_hand(bus, motors, raw)
            core = Core(self._description(self.realtime), self.realtime)
            pose = np.r_[joints, raw[arm.DOF :]]
        finally:
            bus.close()
        self._core = core
        core.start(pose)
        self._after_start(pose)
        self._declare_model(stream)
        # The arm rests where it was switched on; that is where it goes back to.
        self._park_joints = np.array(
            self.park_pose if self.park_pose is not None else joints, dtype=np.float64
        )
        if self._park_joints.shape != (arm.DOF,):
            raise ValueError(f"park_pose needs {arm.DOF} joint positions")

    def _after_start(self, pose: np.ndarray) -> None:
        pass

    def _declare_model(self, stream: str) -> None:
        """The model's directory is the device's while it is open: the station copies it when a
        recording starts."""
        self._model_files = tempfile.TemporaryDirectory(prefix="xpisode-i2rt-model-")
        directory = Path(self._model_files.name)
        shutil.copytree(arm.model_directory(self.hand), directory, dirs_exist_ok=True)
        names = tuple(f"joint{i + 1}" for i in range(arm.DOF))
        self._model = declare(
            directory,
            format="mjcf",
            entry=ENTRY,
            joints=(JointMap(stream=stream, field="joint_pos", names=names),),
            source=f"yam with {self.hand}: {arm.spec()['source']}",
        )

    def replace_controller(self, config: RealtimeConfig) -> None:
        """Change settings on a held arm: the next controller waits on the bus before this one
        lets go, so that the motors never go without commands."""
        pose = self.core.state()["target_position"]
        successor = Core(self._description(config), config)
        self.core.close()
        self._core = successor
        self.realtime = config
        successor.start(pose)

    def _facts(self, driver: str) -> dict[str, Any]:
        spec = arm.spec()
        limits = self._gripper_limits()
        return {
            "driver": driver,
            "channel": self.channel,
            "arm": "yam",
            "gripper": self.hand,
            "park_joints": self._park_joints.tolist() if self._park_joints is not None else None,
            "robot_info": {
                "arm_type": "yam",
                "gripper_type": self.hand,
                "joint_limits": arm.joint_limits().tolist(),
                "gripper_limits": None if limits is None else [float(x) for x in limits],
                "source": spec["source"],
            },
            "realtime": {
                **self.realtime.facts(),
                **self.core.schedule(),
                "tuned": None
                if self.tuned_path is None
                else {**self._tuned.get("provenance", {}), "file": self.tuned_path},
            },
        }

    def _park(self) -> None:
        """Joints only: the gripper stays as it is."""
        assert self._park_joints is not None
        core = self.core
        core.hold()
        time.sleep(3.0 / self.realtime.rate_hz)  # it holds at the start of a cycle
        start = core.state()["target_position"]
        target = start.copy()
        target[: arm.DOF] = self._park_joints
        move(core, start, target, self.park_seconds, self.realtime.period_s)

    def run(self, operation: str) -> None:
        if operation != "park":
            raise ValueError(f"no operation {operation!r}")
        self._park()

    def close(self) -> None:
        """Park, then close: without power the arm has no brakes and falls."""
        core = self._core
        if core is None:
            return
        try:
            # A controller that is gone cannot park; the motors' timeout has let go already.
            if core.alive():
                self._park()
        finally:
            core.close()
            self._core = None
            if self._model_files is not None:
                self._model_files.cleanup()


class YamFollower(_Yam):
    """A YAM arm with a gripper, driven by joint position targets.

    Opening moves the gripper when its limits are not known: it goes to both stops. They come from
    ``gripper_limits`` or the ``tuned`` file, which also holds what ``python -m xpisode_i2rt.tune``
    measured on this arm; settings passed here win. The arm then holds the pose it is in.
    """

    driver = "xpisode_i2rt.yam_follower"
    hand = arm.GRIPPER

    def __init__(
        self,
        channel: str,
        *,
        poll_hz: float = 200.0,
        park_pose: Sequence[float] | None = None,
        park_seconds: float = 3.0,
        gripper_limits: Sequence[float] | None = None,
        tuned: str | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(channel, poll_hz, park_pose, park_seconds, tuned, settings)
        limits = gripper_limits if gripper_limits is not None else self._tuned.get("gripper_limits")
        self._limits: tuple[float, float] | None = None if limits is None else (limits[0], limits[1])
        self._gripper_target = 0.0
        self._force_limiter: ForceLimiter | None = None

    def _motors(self) -> list[bringup.Motor]:
        gripper = arm.spec()["gripper"]
        return [*super()._motors(), bringup.Motor(gripper["id"], gripper["motor"])]

    def _gripper_limits(self) -> Sequence[float] | None:
        return self._limits

    def _bring_up_hand(self, bus: Bus, motors: list[bringup.Motor], raw: list[float]) -> None:
        if self._limits is not None:
            return
        spec = arm.spec()
        gripper = spec["gripper"]
        kp = [*(self.realtime.kp or spec["kp"]), gripper["kp"]]
        kd = [*(self.realtime.kd or spec["kd"]), gripper["kd"]]
        self._limits = bringup.calibrate_gripper(bus, motors, raw, kp, kd, gripper["direction"])

    def _after_start(self, pose: np.ndarray) -> None:
        self._gripper_target = float(pose[arm.DOF])
        gripper = arm.spec()["gripper"]
        self._force_limiter = ForceLimiter(
            gripper["max_force"],
            gripper["kp"],
            gripper["clog_force"],
            gripper["clog_speed"],
            gripper["sign"],
            gripper["motor_stroke"],
            gripper["gripper_stroke"],
        )

    def open(self, clock: Clock) -> None:
        self._open_arm(clock, "state")

    def _to_raw(self, opening: float) -> float:
        assert self._limits is not None
        closed, opened = self._limits
        return closed + opening * (opened - closed)

    def _to_opening(self, raw: float) -> float:
        assert self._limits is not None
        closed, opened = self._limits
        return (raw - closed) / (opened - closed)

    def describe(self) -> DeviceInfo:
        dof = arm.DOF
        joint_pos = _joints("joint_pos", dof, "rad")
        gripper_pos = _scalar("gripper_pos", "1", "normalized opening: 0 closed, 1 open")
        state = (
            joint_pos,
            _joints("joint_vel", dof, "rad/s"),
            _joints("joint_eff", dof, "N*m", "measured motor torque"),
            gripper_pos,
            _scalar("gripper_vel", "1/s", "rate of the normalized opening"),
            _scalar("gripper_eff", "N*m", "measured gripper motor torque"),
        )
        limits = arm.joint_limits()
        endpoint = Endpoint(
            resource="arm",
            kind="joint_position",
            fields=(joint_pos, gripper_pos),
            limits={
                "joint_pos": Limits(min=tuple(limits[:, 0]), max=tuple(limits[:, 1])),
                "gripper_pos": Limits(min=(0.0,), max=(1.0,)),
            },
            description="Targets the controller plays; it clips them to these limits.",
        )
        return DeviceInfo(
            device=self._facts(self.driver),
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

    def read(self, stream: str) -> Reading | None:
        """The state the real-time core published last, stamped when it was published; a cycle
        already read is nothing new."""
        assert self._clock is not None and self._limits is not None
        self.core.check()
        state = self.core.state()
        if state["cycles"] == self._last_cycle:
            return None
        self._last_cycle = state["cycles"]
        dof = arm.DOF
        closed, opened = self._limits
        return Reading(
            int(state["time_ns"]),
            (
                state["position"][:dof],
                state["velocity"][:dof],
                state["torque"][:dof],
                np.array([self._to_opening(state["position"][dof])]),
                np.array([state["velocity"][dof] / (opened - closed)]),
                state["torque"][dof:],
            ),
        )

    def submit(self, endpoint: str, values: Values) -> None:
        """The force limiter decides the gripper's target."""
        assert self._limits is not None and self._force_limiter is not None
        core = self.core
        core.check()
        state = core.state()
        dof = arm.DOF
        limits = arm.joint_limits()
        joints = np.clip(np.asarray(values[0], dtype=np.float64), limits[:, 0], limits[:, 1])
        opening = float(np.clip(np.asarray(values[1], dtype=np.float64)[0], 0.0, 1.0))
        raw = self._force_limiter.update(
            GripperState(
                target=self._to_raw(opening),
                position=float(state["position"][dof]),
                velocity=float(state["velocity"][dof]),
                effort=float(state["torque"][dof]),
                normalized_position=self._to_opening(float(state["position"][dof])),
                normalized_target=opening,
                last_command=self._gripper_target,
            ),
            time.monotonic(),
        )
        self._gripper_target = float(np.clip(raw, min(self._limits), max(self._limits)))
        core.push(np.r_[joints, self._gripper_target])

    def halt(self, resource: str) -> None:
        """Hold still where the target is now."""
        self.core.hold()


HANDLE_SETTLE_S = 0.1
"""Opening looks at the handle's age only after this: a handle that has not answered is then
at least this old, one that answers is younger than ``HANDLE_FRESH_S``."""
HANDLE_FRESH_S = 0.05
"""Five polls at 100 Hz."""


class YamLeader(_Yam):
    """A YAM arm with the teaching handle, read as an operator input, and driven to follow the
    follower while somebody else drives that.

    It floats on gravity compensation under the operator's hand. Targets make it track them; halting
    lets it float again. The handle reports its trigger and buttons on the same bus; the trigger
    cannot be driven.
    """

    driver = "xpisode_i2rt.yam_leader"
    hand = arm.HANDLE
    floats = True

    def __init__(
        self,
        channel: str,
        *,
        poll_hz: float = 200.0,
        park_pose: Sequence[float] | None = None,
        park_seconds: float = 3.0,
        tuned: str | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(channel, poll_hz, park_pose, park_seconds, tuned, settings)
        self._trigger_range = float(arm.spec()["handle"]["range_rad"])

    def _encoder_id(self) -> int | None:
        return int(arm.spec()["handle"]["encoder_id"])

    def open(self, clock: Clock) -> None:
        self._open_arm(clock, "input")
        self._wait_for_handle()

    def _wait_for_handle(self) -> None:
        """Open only once the handle has answered through the controller: until then its age
        counts from the controller's start, and the first reads would report it silent."""
        started = time.monotonic()
        while True:
            self.core.check()
            waited = time.monotonic() - started
            if waited > HANDLE_SETTLE_S and self.core.state()["encoder_age_s"] < HANDLE_FRESH_S:
                break
            if waited > self.realtime.encoder_start_timeout_s:
                raise RuntimeError(
                    f"the teaching handle on {self.channel} did not answer within "
                    f"{self.realtime.encoder_start_timeout_s:g} s of the controller's start"
                )
            time.sleep(0.01)
        if waited > HANDLE_SETTLE_S + 0.1:
            print(
                f"[{self.channel}] the teaching handle first answered {waited:.2f} s "
                "after the controller started",
                file=sys.stderr,
                flush=True,
            )

    def describe(self) -> DeviceInfo:
        dof = arm.DOF
        fields_ = (
            _joints("joint_pos", dof, "rad"),
            _scalar("trigger", "1", "teaching handle encoder: 0 released, 1 squeezed"),
            Field(
                name="buttons",
                dtype="bool",
                shape=(2,),
                unit="1",
                axes=(Axis(name="button"),),
                description="teaching handle inputs, pressed = true",
            ),
            _scalar(
                "handle_age",
                "s",
                "time since the teaching handle last answered; trigger and buttons are that answer",
            ),
        )
        limits = arm.joint_limits()
        endpoint = Endpoint(
            resource="arm",
            kind="joint_position",
            fields=(fields_[0],),
            limits={"joint_pos": Limits(min=tuple(limits[:, 0]), max=tuple(limits[:, 1]))},
            description="Targets the controller tracks; without them the leader floats.",
        )
        return DeviceInfo(
            device=self._facts(self.driver),
            model=self._model,
            streams={"input": _observation(fields_, self.poll_hz, Role.INPUT)},
            endpoints={"joint_position": endpoint},
            operations={
                "park": Operation(moves=("arm",), description="Move the leader to its park pose.")
            },
        )

    def read(self, stream: str) -> Reading | None:
        """The trigger is the distance from center over the range; the inputs are the low bits.

        The joints come from the motors. A handle that restarts goes silent for seconds; its last
        answer is kept meanwhile, with its age, and only a longer silence faults the device.
        A controller cycle already read is nothing new: read as often as the controller runs, a
        read can land twice in one cycle.
        """
        assert self._clock is not None
        self.core.check()
        state = self.core.state()
        if state["cycles"] == self._last_cycle:
            return None
        self._last_cycle = state["cycles"]
        age = state["encoder_age_s"]
        if age > self.realtime.encoder_timeout_s:
            raise RuntimeError(
                f"the teaching handle on {self.channel} has not answered for {age:.1f} s"
            )
        trigger = min(abs(state["encoder_position"]) / self._trigger_range, 1.0)
        inputs = state["encoder_inputs"]
        return Reading(
            int(state["time_ns"]),
            (
                state["position"][: arm.DOF],
                np.array([trigger]),
                np.array([inputs % 2, inputs // 2], dtype=bool),
                np.array([age]),
            ),
        )

    def submit(self, endpoint: str, values: Values) -> None:
        self.core.check()
        limits = arm.joint_limits()
        self.core.push(np.clip(np.asarray(values[0], dtype=np.float64), limits[:, 0], limits[:, 1]))

    def halt(self, resource: str) -> None:
        """A leader stops by floating again under the operator's hand."""
        self.core.hold()
