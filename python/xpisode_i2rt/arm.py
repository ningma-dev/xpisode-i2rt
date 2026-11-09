"""What a YAM with its hand is: its motors, gains and limits, and its model. A snapshot of what
i2rt's configs said, in ``data/``; ``data/LICENSE.i2rt`` covers the models and those numbers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DATA = Path(__file__).parent / "data"
GRIPPER = "linear_4310"
HANDLE = "yam_teaching_handle"
DOF = 6


@dataclass(frozen=True)
class MotorKind:
    position: float
    velocity: float
    torque: float
    gear_ratio: float


KINDS = {
    "DM4340": MotorKind(12.5, 10.0, 28.0, 40.0),
    "DM4310": MotorKind(12.5, 30.0, 10.0, 10.0),
}


def spec() -> dict[str, Any]:
    return json.loads((DATA / "yam_v1.json").read_text())


def model_directory(hand: str) -> Path:
    return DATA / f"yam_{hand}"


def model_path(hand: str) -> str:
    return str(model_directory(hand) / "model.xml")


def joint_limits() -> np.ndarray:
    """(6, 2), rad: the model's, with the margin i2rt allows past them."""
    arm = spec()
    limits = np.array(arm["joint_limits"], dtype=np.float64)
    limits[:, 0] -= arm["joint_limit_margin"]
    limits[:, 1] += arm["joint_limit_margin"]
    return limits
