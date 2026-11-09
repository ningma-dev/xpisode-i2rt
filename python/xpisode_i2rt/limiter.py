"""Keep a gripper from squeezing past a force: once it presses on something and has stopped, its
position target is what gives that torque. What i2rt's GripperForceLimiter does."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

EFFORT_WINDOW_S = 0.1
RELEASE_EFFORT = 0.2
FRICTION_TORQUE = 0.3
SMOOTHING = 0.1


@dataclass(frozen=True)
class GripperState:
    target: float
    position: float
    velocity: float
    effort: float
    normalized_position: float
    normalized_target: float
    last_command: float


class ForceLimiter:
    def __init__(
        self,
        max_force: float,
        kp: float,
        clog_force: float,
        clog_speed: float,
        sign: float,
        motor_stroke: float,
        gripper_stroke: float,
    ) -> None:
        self.kp, self.clog_force, self.clog_speed, self.sign = kp, clog_force, clog_speed, sign
        self.limit_torque = max_force * gripper_stroke / motor_stroke
        self.clogged = False
        self._efforts: deque[tuple[float, float]] = deque()
        self._adjusted: float | None = None

    def update(self, state: GripperState, now: float) -> float:
        self._efforts.append((now, state.effort))
        while self._efforts and self._efforts[0][0] < now - EFFORT_WINDOW_S:
            self._efforts.popleft()
        average = abs(np.mean([e for _, e in self._efforts]))
        if self.clogged:
            if state.normalized_position < state.normalized_target or average < RELEASE_EFFORT:
                self.clogged = False  # it wants to open, or nothing presses any more
        elif average > self.clog_force and abs(state.velocity) < self.clog_speed:
            self.clogged = True
        if not self.clogged:
            self._adjusted = state.position
            return state.target
        direction = np.sign(state.target - state.position) * self.sign
        zero_effort = state.last_command - direction * abs(state.effort) / self.kp
        wanted = zero_effort + direction * (self.limit_torque + FRICTION_TORQUE) / self.kp
        if self._adjusted is None:
            self._adjusted = wanted
        self._adjusted = (1 - SMOOTHING) * self._adjusted + SMOOTHING * wanted
        return float(self._adjusted)
