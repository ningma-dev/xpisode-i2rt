"""What a tuning run on the bench decides, without the arm: where it may move, what the logs say
about the joints, and which of the settings it tried tracked best. ``tune.py`` runs it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

SPEED_BINS = ((0, 100), (100, 200), (200, 300), (300, 500), (500, 800))
"""mm/s."""
MIN_BIN_SAMPLES = 30
FRICTION_WIDTHS = (0.05, 0.1, 0.2, 0.4)
GRAVITY_FACTOR_RANGE = (0.8, 2.0)
MIN_GRAVITY_RMS = 0.5
"""N*m. A joint that carries less than this says too little about its gravity factor."""

ENCODER_STEP = 25.0 / 65535.0
"""A motor's position step, rad."""
LEADER_HZ = 107.0
"""Measured under i2rt. The worst case for the noise a leader gives its follower."""


def teleop_like(
    target: Callable[[float], np.ndarray], t: float, hz: float = LEADER_HZ
) -> np.ndarray:
    """What a leader would report at ``t`` of a smooth ``target``: its latest sample, in encoder
    steps. Smooth targets hide what velocity feedforward does to teleoperation."""
    return np.round(target(np.floor(t * hz) / hz) / ENCODER_STEP) * ENCODER_STEP


def high_frequency_rms(samples: np.ndarray, window: int = 9) -> np.ndarray:
    """The RMS, per joint, of ``samples`` (time, joints) minus their centered moving average."""
    if len(samples) <= window:
        return np.zeros(samples.shape[1])
    kernel = np.ones(window) / window
    smooth = np.stack(
        [np.convolve(samples[:, j], kernel, "valid") for j in range(samples.shape[1])], axis=1
    )
    middle = samples[window // 2 : window // 2 + len(smooth)]
    return np.sqrt(np.mean((middle - smooth) ** 2, axis=0))


@dataclass(frozen=True)
class JointFit:
    gravity_factor: float
    coulomb: float
    viscous: float
    residual_rms: float
    gravity_known: bool


def _fit(columns: list[np.ndarray], torque: np.ndarray) -> np.ndarray:
    """Least squares with the columns' coefficients at zero or above: a negative one is a drop out,
    since friction opposes motion and gravity is not negative."""
    active = list(range(len(columns)))
    while True:
        solved = np.linalg.lstsq(np.stack([columns[i] for i in active], 1), torque, rcond=None)[0]
        if np.all(solved >= 0) or len(active) == 1:
            out = np.zeros(len(columns))
            out[active] = np.maximum(solved, 0.0)
            return out
        active.pop(int(np.argmin(solved)))


def fit_joint(
    gravity: np.ndarray, velocity: np.ndarray, torque: np.ndarray, default_factor: float, width: float
) -> JointFit:
    """Measured torque as gravity times a factor, Coulomb friction at the target velocity, viscous."""
    known = np.sqrt(np.mean(gravity**2)) >= MIN_GRAVITY_RMS
    friction = [np.tanh(velocity / width), velocity]
    if known:
        c = _fit([gravity, *friction], torque)
        factor = float(np.clip(c[0], *GRAVITY_FACTOR_RANGE))
        coulomb, viscous = _fit(friction, torque - factor * gravity)
    else:
        factor = default_factor
        coulomb, viscous = _fit(friction, torque - factor * gravity)
    model = factor * gravity + coulomb * np.tanh(velocity / width) + viscous * velocity
    return JointFit(factor, float(coulomb), float(viscous), float(np.std(torque - model)), bool(known))


def fit_arm(
    gravity: np.ndarray, velocity: np.ndarray, torque: np.ndarray, defaults: Sequence[float]
) -> tuple[float, list[JointFit]]:
    """The arm's joints, at the one friction width that leaves the least torque unexplained."""
    joints = range(gravity.shape[1])
    fits = {
        width: [
            fit_joint(gravity[:, j], velocity[:, j], torque[:, j], defaults[j], width)
            for j in joints
        ]
        for width in FRICTION_WIDTHS
    }
    # Each joint as a share of its own torque, so that the shoulder's newton-metres do not decide.
    scale = np.maximum(np.std(torque, axis=0), 1e-6)
    best = min(fits, key=lambda w: sum(f.residual_rms / scale[i] for i, f in enumerate(fits[w])))
    return best, fits[best]


def binned_errors(speed: np.ndarray, error: np.ndarray) -> dict[int, float]:
    """The median tool error by the lower edge of each speed bin with enough samples."""
    out: dict[int, float] = {}
    for low, high in SPEED_BINS:
        mask = (speed >= low) & (speed < high)
        if mask.sum() >= MIN_BIN_SAMPLES:
            out[low] = float(np.median(error[mask]))
    return out


def score(errors: dict[int, float], reference: dict[int, float]) -> float:
    """The mean of the bins both have, each counted alike; lower is better."""
    shared = sorted(set(errors) & set(reference))
    if not shared:
        return float("inf")
    return float(np.mean([errors[b] for b in shared]))


@dataclass(frozen=True)
class Result:
    errors: dict[int, float]
    """Median tool error by speed bin, mm."""
    noise: np.ndarray
    """High-frequency measured torque per joint, N*m."""
    stable: bool


def better(
    candidate: Result,
    current: Result,
    ceiling: np.ndarray | None = None,
    min_gain: float = 0.03,
) -> bool:
    """Whether ``candidate`` tracks measurably better and stays under the ``ceiling`` on noise."""
    if not candidate.stable:
        return False
    if ceiling is not None and np.any(candidate.noise > ceiling):
        return False
    return score(candidate.errors, current.errors) < (1 - min_gain) * score(
        current.errors, candidate.errors
    )


def climb(
    values: Sequence[float],
    start: Result,
    measure: Callable[[float], Result],
    ceiling: np.ndarray | None = None,
) -> tuple[float | None, Result]:
    """Try ``values`` in order and stop at the first that is no better: higher gains get louder and
    then unstable, and there is no reason to try the next one up after that. The best value tried,
    None when none beat ``start``, with its result."""
    best: float | None = None
    result = start
    for value in values:
        tried = measure(value)
        if not better(tried, result, ceiling):
            break
        best, result = value, tried
    return best, result
