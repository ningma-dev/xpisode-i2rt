import pytest

from xpisode_i2rt.limiter import ForceLimiter, GripperState


def _limiter() -> ForceLimiter:
    return ForceLimiter(50.0, 20.0, 0.5, 0.3, 1.0, 6.57, 0.096)


def _state(effort: float, velocity: float = 0.0, target: float = 6.0, position: float = 4.0) -> GripperState:
    return GripperState(target, position, velocity, effort, 0.5, 0.0, 4.5)


def test_free_motion_passes_the_target_through() -> None:
    limiter = _limiter()
    assert limiter.update(_state(0.1, velocity=2.0), 0.0) == 6.0
    assert not limiter.clogged


def test_pressing_and_stopped_holds_the_force() -> None:
    limiter = _limiter()
    out = 0.0
    for i in range(200):
        out = limiter.update(_state(1.0), i * 0.01)
    assert limiter.clogged
    torque = 50.0 * 0.096 / 6.57 + 0.3
    assert out == pytest.approx(4.5 - 1.0 / 20.0 + torque / 20.0, abs=1e-3)


def test_it_lets_go_when_asked_to_open() -> None:
    limiter = _limiter()
    for i in range(50):
        limiter.update(_state(1.0), i * 0.01)
    assert limiter.clogged
    opening = GripperState(1.5, 4.0, 0.0, 1.0, 0.5, 0.9, 4.5)
    assert limiter.update(opening, 0.6) == 1.5
    assert not limiter.clogged


def test_a_stopped_arm_with_no_effort_does_not_clog() -> None:
    limiter = _limiter()
    for i in range(50):
        limiter.update(_state(0.1), i * 0.01)
    assert not limiter.clogged
