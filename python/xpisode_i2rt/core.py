"""``i2rt-rtcore``: its process, and the memory shared with it. The arm's own numbers are in
``arm.py``; ``xpisode.py`` makes devices of it.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from xpisode_i2rt.arm import DOF, KINDS, joint_limits
from xpisode_i2rt.arm import spec as arm_spec

MAX_MOTORS = 8
WORDS = 1024
RING = 64
ENTRY = 2 + MAX_MOTORS
MAGIC = 0x3145_524F_4354_5249

_GO, _HOLD, _STOP, _HEAD, _READY = 1, 2, 3, 4, 8
_SEQUENCE, _TIME_NS, _CYCLES, _FAULT, _FAULT_MOTOR, _FAULT_DETAIL = 16, 17, 18, 19, 20, 21
_STARVED, _MISSED, _DROPPED, _LATE_MAX_NS, _POLICY, _PRIORITY, _LOCKED = 22, 23, 24, 25, 26, 27, 28
_ENCODER_POSITION, _ENCODER_VELOCITY, _ENCODER_INPUTS, _ENCODER_AGE_NS = 29, 30, 31, 15
_POSITION = 32
_VELOCITY = _POSITION + MAX_MOTORS
_TORQUE = _VELOCITY + MAX_MOTORS
_TEMPERATURE = _TORQUE + MAX_MOTORS
_TARGET_POSITION = _TEMPERATURE + MAX_MOTORS
_TARGET_VELOCITY = _TARGET_POSITION + MAX_MOTORS
_FEEDFORWARD = _TARGET_VELOCITY + MAX_MOTORS
_RING_START = 128

TEACHING_HANDLE_ENCODER = 0x50E
"""i2rt's ``get_encoder_chain``."""
FAULTS = {1: "a motor stopped answering", 2: "a motor is not enabled", 3: "the bus refused a frame"}
SCHEDULERS = {0: "other", 1: "fifo", 2: "rr"}


@dataclass(frozen=True)
class RealtimeConfig:
    binary: str = ""
    """A path, or a name on PATH; empty means this repository's own build."""
    rate_hz: float = 400.0
    period_s: float = 0.01
    """The station's tick: the spacing of the targets."""
    delay_s: float = 0.01
    """How far behind the clock the targets are played."""
    max_velocity: float = 4.0
    """The arm's joints, rad/s. Recorded teleoperation stays under it, peaking at 3.6."""
    gripper_max_velocity: float = 30.0
    """Rad/s of the gripper's motor, whose whole stroke is about 5 rad: teleoperation closes it in
    a quarter of a second."""
    gripper_velocity_scale: float = 0.0
    kp: tuple[float, ...] = ()
    """The arm's joints; empty means the arm's defaults, as for the next three."""
    kd: tuple[float, ...] = ()
    gravity_factor: tuple[float, ...] = ()
    coulomb: tuple[float, ...] = ()
    """N*m a joint; none by default."""
    viscous: tuple[float, ...] = ()
    """N*m*s/rad a joint; none by default."""
    friction_width: float = 0.1
    velocity_window: int = 4
    """Ticks the target velocity is the slope over: leader samples make the neighbouring difference
    noisy, and kd and friction feedforward turn that into noise the arm plays."""
    inertia_scale: float = 0.0
    acceleration_tau_s: float = 0.02
    reply_timeout_s: float = 0.05
    encoder_every: int = 4
    """Cycles between polls of a teaching handle, about 100 Hz at 400 Hz."""
    encoder_timeout_s: float = 10.0
    """How long a handle may stay silent before the device faults. Until then its last trigger and
    buttons are kept and ``handle_age`` says how old they are: a handle that restarts is silent
    0.9 to 3.5 s (measured on can_leader_r), then answers as before."""
    encoder_start_timeout_s: float = 5.0
    """How long opening waits for the handle's first answer through the controller."""
    orphan_hold_s: float = 60.0
    fifo_priority: int = 80
    cpu: int = -1

    def facts(self) -> dict[str, Any]:
        """As JSON for the Episode header, which has lists and no tuples."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def _symmetric(m: np.ndarray) -> list[float]:
    return [m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]]


def chain_links(model: mujoco.MjModel, dof: int, qpos: np.ndarray | None = None) -> dict[str, Any]:
    """The arm's links for ``dof`` hinge joints from the model. What hangs off the last joint, the
    gripper with its fingers at ``qpos``, is that link's: it moves with it."""
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[: len(qpos)] = qpos
    mujoco.mj_kinematics(model, data)
    links: list[dict[str, Any]] = []
    for j in range(dof):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or np.any(model.jnt_pos[j]):
            raise ValueError(f"joint {j} is not a hinge at its body's origin")
        body = model.jnt_bodyid[j]
        origin, rotation = data.xpos[body], data.xmat[body].reshape(3, 3)
        members = [body]
        if j == dof - 1:
            members = [b for b in range(model.nbody) if _within(model, b, body)]
        mass, moment, inertia = 0.0, np.zeros(3), np.zeros((3, 3))
        for b in members:
            m = float(model.body_mass[b])
            com = rotation.T @ (data.xipos[b] - origin)
            frame = rotation.T @ data.ximat[b].reshape(3, 3)
            own = frame @ np.diag(model.body_inertia[b]) @ frame.T
            mass += m
            moment += m * com
            inertia += own - m * _skew(com) @ _skew(com)
        center = moment / mass
        inertia += mass * _skew(center) @ _skew(center)
        links.append(
            {
                "position": model.body_pos[body].tolist(),
                "quaternion": model.body_quat[body].tolist(),
                "axis": model.jnt_axis[j].tolist(),
                "mass": mass,
                "center_of_mass": center.tolist(),
                "inertia": [float(x) for x in _symmetric(inertia)],
            }
        )
    first = model.jnt_bodyid[0]
    parent = model.body_parentid[first]
    gravity = data.xmat[parent].reshape(3, 3).T @ np.asarray(model.opt.gravity)
    return {"links": links, "gravity": gravity.tolist()}


def _within(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body != 0:
        if body == root:
            return True
        body = model.body_parentid[body]
    return False


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def description(
    config: RealtimeConfig,
    channel: str,
    model_path: str,
    offsets: Sequence[float],
    gripper_limits: Sequence[float] | None = None,
    floats: bool = False,
    encoder_id: int | None = None,
) -> dict[str, Any]:
    """What the controller is told about this arm. ``gripper_limits`` means a seventh motor."""
    arm = arm_spec()
    limits = joint_limits()
    motors = []
    for i in range(DOF):
        motors.append(
            {
                "id": i + 1,
                "kind": arm["motors"][i],
                "direction": 1.0,
                "offset": float(offsets[i]),
                "kp": float((config.kp or arm["kp"])[i]),
                "kd": float((config.kd or arm["kd"])[i]),
                "low": float(limits[i, 0]),
                "high": float(limits[i, 1]),
                "torque_limit": KINDS[arm["motors"][i]].torque,
                "gravity_scale": float((config.gravity_factor or arm["gravity_factor"])[i]),
                "coulomb": float(config.coulomb[i]) if config.coulomb else 0.0,
                "viscous": float(config.viscous[i]) if config.viscous else 0.0,
                "inertia_scale": config.inertia_scale,
                "max_velocity": config.max_velocity,
                "velocity_scale": 1.0,
                "float_kd": float(arm["float_kd"][i]),
            }
        )
    if gripper_limits is not None:
        gripper = arm["gripper"]
        motors.append(
            {
                "id": gripper["id"],
                "kind": gripper["motor"],
                "direction": float(gripper["direction"]),
                "offset": 0.0,
                "kp": float(gripper["kp"]),
                "kd": float(gripper["kd"]),
                "low": float(min(gripper_limits)),
                "high": float(max(gripper_limits)),
                "torque_limit": KINDS[gripper["motor"]].torque,
                "gravity_scale": 0.0,
                "coulomb": 0.0,
                "viscous": 0.0,
                "inertia_scale": 0.0,
                "max_velocity": config.gripper_max_velocity,
                "velocity_scale": config.gripper_velocity_scale,
                "float_kd": 0.0,
            }
        )
    return {
        **chain_links(mujoco.MjModel.from_xml_path(model_path), DOF),
        "channel": channel,
        "motors": motors,
        "floats": floats,
        "encoder_id": encoder_id,
        "dry": False,
    }


def _shared_file(channel: str) -> Path:
    """A file of this controller's own: its successor on the bus starts while it still runs."""
    descriptor, name = tempfile.mkstemp(prefix=f"i2rt-rtcore-{channel}-", dir="/dev/shm")
    with os.fdopen(descriptor, "wb") as file:
        file.truncate(WORDS * 8)
    return Path(name)


def stop_orphans(channel: str) -> list[Path]:
    """Stop a controller left on this bus by a device that died: it would fight the bring-up. The
    arm it was holding has until the motors' timeout, which the bring-up beats or the arm rests."""
    found = sorted(Path("/dev/shm").glob(f"i2rt-rtcore-{channel}-*"))
    for path in found:
        with open(path, "r+b") as file:
            with mmap.mmap(file.fileno(), WORDS * 8) as shared:
                view = memoryview(shared).cast("Q")
                view[_STOP] = 1
                view.release()
    if found:
        time.sleep(0.05)
    for path in found:
        path.unlink(missing_ok=True)
    return found


REPOSITORY = Path(__file__).resolve().parents[2]


def default_binary() -> Path:
    return REPOSITORY / "target" / "release" / "i2rt-rtcore"


def _resolve(binary: str) -> str:
    found = shutil.which(binary) if binary else str(default_binary())
    if found is None or not Path(found).exists():
        raise FileNotFoundError(f"{binary or found!r} not found: run `cargo build --release` here")
    return found


class Core:
    """The controller process and its shared memory. One writer of targets."""

    def __init__(self, told: dict[str, Any], config: RealtimeConfig) -> None:
        self.n = len(told["motors"])
        self.config = config
        self._directory = tempfile.TemporaryDirectory(prefix="i2rt-rtcore-")
        directory = Path(self._directory.name)
        shm = _shared_file(told["channel"])
        self._shm_path = shm
        with open(shm, "r+b") as file:
            self._map = mmap.mmap(file.fileno(), WORDS * 8)
        self._words = memoryview(self._map).cast("Q")
        self._floats = memoryview(self._map).cast("d")
        self._head = 0
        full = {
            **told,
            "shm_path": str(shm),
            "rate_hz": config.rate_hz,
            "period_s": config.period_s,
            "delay_s": config.delay_s,
            "friction_width": config.friction_width,
            "velocity_window": config.velocity_window,
            "acceleration_tau_s": config.acceleration_tau_s,
            "reply_timeout_s": config.reply_timeout_s,
            "encoder_every": config.encoder_every,
            "orphan_hold_s": config.orphan_hold_s,
            "fifo_priority": config.fifo_priority,
            "cpu": config.cpu,
        }
        path = directory / "core.json"
        path.write_text(json.dumps(full))
        self._process = subprocess.Popen([_resolve(config.binary), str(path)])
        deadline = time.monotonic() + 2.0
        while self._words[_READY] != MAGIC:
            if self._process.poll() is not None or time.monotonic() > deadline:
                self.close()
                raise RuntimeError("the controller did not start")
            time.sleep(0.001)

    def _write(self, q: np.ndarray) -> None:
        at = _RING_START + (self._head % RING) * ENTRY
        self._words[at] = 0
        self._words[at + 1] = time.monotonic_ns()
        row = np.zeros(MAX_MOTORS)
        row[: len(q)] = q
        for j in range(MAX_MOTORS):
            self._floats[at + 2 + j] = float(row[j])
        self._words[at] = self._head + 1
        self._head += 1
        self._words[_HEAD] = self._head

    def start(self, q: np.ndarray) -> None:
        """Hold ``q`` and take the bus."""
        self._write(q)
        self._words[_GO] = 1
        # Its state means nothing before its first cycles.
        deadline = time.monotonic() + 0.5
        while self._words[_CYCLES] < 2:
            if not self.alive() or time.monotonic() > deadline:
                raise RuntimeError("the controller did not take the bus")
            time.sleep(0.0005)

    def push(self, q: np.ndarray) -> None:
        self._write(q)

    def hold(self) -> None:
        """Stop where the target is, forgetting the targets not played."""
        self._words[_HOLD] = self._words[_HOLD] + 1

    def alive(self) -> bool:
        return self._process.poll() is None

    def state(self) -> dict[str, Any]:
        for _ in range(100):
            before = self._words[_SEQUENCE]
            if before % 2 == 0:
                state = self._read_state()
                if self._words[_SEQUENCE] == before:
                    return state
        raise RuntimeError("the controller's state stayed busy")

    def _array(self, at: int) -> np.ndarray:
        return np.array(self._floats[at : at + self.n], dtype=np.float64)

    def _read_state(self) -> dict[str, Any]:
        fault = int(self._words[_FAULT])
        return {
            "time_ns": int(self._words[_TIME_NS]),
            "cycles": int(self._words[_CYCLES]),
            "fault": fault,
            "fault_motor": int(np.int64(np.uint64(self._words[_FAULT_MOTOR]))),
            "fault_detail": int(self._words[_FAULT_DETAIL]),
            "starved": int(self._words[_STARVED]),
            "missed": int(self._words[_MISSED]),
            "dropped": int(self._words[_DROPPED]),
            "late_max_ns": int(self._words[_LATE_MAX_NS]),
            "encoder_position": float(self._floats[_ENCODER_POSITION]),
            "encoder_velocity": float(self._floats[_ENCODER_VELOCITY]),
            "encoder_inputs": int(self._words[_ENCODER_INPUTS]),
            "encoder_age_s": int(self._words[_ENCODER_AGE_NS]) * 1e-9,
            "position": self._array(_POSITION),
            "velocity": self._array(_VELOCITY),
            "torque": self._array(_TORQUE),
            "temperature": self._array(_TEMPERATURE),
            "target_position": self._array(_TARGET_POSITION),
            "target_velocity": self._array(_TARGET_VELOCITY),
            "feedforward": self._array(_FEEDFORWARD),
        }

    def schedule(self) -> dict[str, Any]:
        """What the controller actually got, which is what the Episode header says."""
        return {
            "scheduler": SCHEDULERS.get(int(self._words[_POLICY]), str(int(self._words[_POLICY]))),
            "priority": int(self._words[_PRIORITY]),
            "memory_locked": bool(self._words[_LOCKED]),
        }

    def check(self) -> None:
        state_fault = int(self._words[_FAULT])
        if state_fault:
            motor = int(np.int64(np.uint64(self._words[_FAULT_MOTOR])))
            detail = int(self._words[_FAULT_DETAIL])
            raise RuntimeError(
                f"{FAULTS.get(state_fault, state_fault)}: motor {motor}, code {detail}; "
                "the arm holds where it is"
            )
        if not self.alive():
            raise RuntimeError("the controller exited")

    def close(self) -> None:
        self._words[_STOP] = 1
        try:
            self._process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=1.0)
        self._floats.release()
        self._words.release()
        self._map.close()
        self._shm_path.unlink(missing_ok=True)
        self._directory.cleanup()


def _minimum_jerk(u: float) -> float:
    return u**3 * (10 - 15 * u + 6 * u**2)


def move(core: Core, start: np.ndarray, target: np.ndarray, seconds: float, period: float) -> None:
    """Move in the controller's targets at its period; it plays them smoothly."""
    steps = max(1, math.ceil(seconds / period))
    begin = time.monotonic()
    for i in range(1, steps + 1):
        core.check()
        core.push(start + _minimum_jerk(i / steps) * (target - start))
        time.sleep(max(0.0, begin + i * period - time.monotonic()))
