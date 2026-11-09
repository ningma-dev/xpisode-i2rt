"""The bench's one fixed motion, the same on every arm: no recording decides where it goes.

It starts and ends at the rest pose and stays in front of the base, clear of a table the base stands
on and of another arm 0.61 m to the side, whichever side that is: tests/test_bench_motion.py checks
all of that on the arm's own model, every 20 ms of the way.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

FREQUENCIES = np.array([0.31, 0.47, 0.59, 0.73, 0.83, 0.97])
"""Hz per joint: not in step with each other, so the arm does not retrace one path."""
SECONDS = 15.0
REST = np.zeros(6)
REST_TOLERANCE = 0.35
"""rad. How far from the rest pose an arm may start: it only knows the way from there."""
GO_SECONDS = 4.0
TICK_S = 0.01

ARM_SPACING_M = 0.61
"""follower_l/base to follower_r/base in the station."""
WRIST_CAMERA = np.array([0.0, -0.0704, 0.0770])
"""In the gripper's frame, from the station's transforms."""
WRIST_CAMERA_RADIUS_M = 0.05


@dataclass(frozen=True)
class Pose:
    center: tuple[float, ...]
    sway: tuple[float, ...]


POSES = (
    Pose((0.0, 1.20, 1.40, -0.90, 0.0, 0.0), (0.15, 0.15, 0.15, 0.15, 0.15, 0.15)),
    Pose((0.0, 1.45, 1.55, -0.85, 0.0, 0.0), (0.25, 0.25, 0.25, 0.25, 0.25, 0.25)),
    Pose((0.0, 1.70, 1.80, -0.90, 0.0, 0.0), (0.25, 0.25, 0.25, 0.25, 0.25, 0.25)),
    Pose((0.0, 1.30, 1.25, -0.70, 0.0, 0.0), (0.30, 0.25, 0.25, 0.30, 0.30, 0.30)),
)
FIT_POSES = POSES[:2]
"""What friction and gravity are measured on."""
TEST_POSES = POSES[2:]
"""What settings are judged on: not what they were fitted on, and the fastest."""


def angle(pose: Pose, t: float, seconds: float = SECONDS) -> np.ndarray:
    """The joints at ``t``, fading in and out over two seconds."""
    envelope = min(1.0, t / 2.0, max(0.0, (seconds - t) / 2.0))
    return np.array(pose.center) + np.array(pose.sway) * envelope * np.sin(
        2 * np.pi * FREQUENCIES * t
    )


def path(step: float = 0.02) -> np.ndarray:
    """Every pose the fixed motion goes through, in order, as joints: (samples, 6)."""
    points = []
    at = REST
    for pose in POSES:
        center = np.array(pose.center)
        for u in np.linspace(0.0, 1.0, 50):
            points.append(at + u * (center - at))
        points += [angle(pose, t) for t in np.arange(0.0, SECONDS, step)]
        at = center
    for u in np.linspace(0.0, 1.0, 50):
        points.append(at + u * (REST - at))
    return np.array(points)


@dataclass(frozen=True)
class Clearance:
    table_m: float
    """The lowest moving part, or the wrist camera, above the table the base stands on."""
    behind_m: float
    """How far any of it goes behind the base."""
    sideways_m: float
    """How far any of it goes to either side of the base."""
    up_m: float
    self_contacts: int
    stop_margin_rad: float
    tool_speed_mm_s: float


def clearance(model: mujoco.MjModel, joints: np.ndarray, step: float = 0.02) -> Clearance:
    """Where the arm's own meshes go along ``joints``."""
    data = mujoco.MjData(model)
    tcp, gripper = model.site("tcp_site").id, model.body("gripper").id
    base = model.body("base").id
    moving = [
        g
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_bodyid[g] != base
    ]
    verts = {}
    for g in moving:
        mesh = model.geom_dataid[g]
        start = model.mesh_vertadr[mesh]
        verts[g] = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    lowest, highest, back, side = np.inf, -np.inf, 0.0, 0.0
    contacts = 0
    tools = np.zeros((len(joints), 3))
    # Link 1 turns about the base and stays where it stands, on the table.
    far = {g for g in moving if model.geom_bodyid[g] not in (model.body("link1").id,)}
    for i, q in enumerate(joints):
        data.qpos[:] = 0
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        tools[i] = data.site_xpos[tcp]
        for g in moving:
            points = data.geom_xpos[g] + verts[g] @ data.geom_xmat[g].reshape(3, 3).T
            if g in far:
                lowest = min(lowest, float(points[:, 2].min()))
            highest = max(highest, float(points[:, 2].max()))
            back = max(back, float(-points[:, 0].min()))
            side = max(side, float(np.abs(points[:, 1]).max()))
        camera = data.xpos[gripper] + data.xmat[gripper].reshape(3, 3) @ WRIST_CAMERA
        lowest = min(lowest, float(camera[2]) - WRIST_CAMERA_RADIUS_M)
        side = max(side, float(abs(camera[1])) + WRIST_CAMERA_RADIUS_M)
        highest = max(highest, float(camera[2]) + WRIST_CAMERA_RADIUS_M)
        for c in range(data.ncon):
            pair = {model.geom_bodyid[data.contact[c].geom1], model.geom_bodyid[data.contact[c].geom2]}
            names = {model.body(b).name for b in pair}
            # The base and link 1 meshes overlap in every pose: they are the bearing.
            if names != {"base", "link1"} and not names <= {"tip_left", "tip_right", "gripper"}:
                contacts += 1
    stops = model.jnt_range[:6]
    margin = float(min((joints - stops[:, 0]).min(), (stops[:, 1] - joints).min()))
    speed = np.linalg.norm(np.gradient(tools, step, axis=0), axis=1) * 1e3
    return Clearance(lowest, back, side, highest, contacts, margin, float(speed.max()))
