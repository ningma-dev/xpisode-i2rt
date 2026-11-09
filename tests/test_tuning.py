import numpy as np

from xpisode_i2rt.tuning import (
    ENCODER_STEP,
    Result,
    better,
    binned_errors,
    climb,
    fit_arm,
    fit_joint,
    high_frequency_rms,
    score,
    teleop_like,
)


def _signals() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 20.0, 0.0025)
    gravity = 4.0 * np.sin(0.7 * t) + 1.0
    velocity = 0.8 * np.sin(2.3 * t)
    return gravity, velocity, rng.normal(0.0, 0.05, len(t))


def test_a_joint_fit_finds_what_is_in_the_torque() -> None:
    gravity, velocity, noise = _signals()
    torque = 1.3 * gravity + 1.5 * np.tanh(velocity / 0.2) + 0.4 * velocity + noise
    fit = fit_joint(gravity, velocity, torque, 1.0, 0.2)
    assert abs(fit.gravity_factor - 1.3) < 0.01
    assert abs(fit.coulomb - 1.5) < 0.02
    assert abs(fit.viscous - 0.4) < 0.03
    assert fit.gravity_known


def test_friction_never_comes_out_negative() -> None:
    gravity, velocity, noise = _signals()
    torque = 1.1 * gravity + 0.5 * np.tanh(velocity / 0.05) - 0.3 * velocity + noise
    fit = fit_joint(gravity, velocity, torque, 1.0, 0.05)
    assert fit.coulomb >= 0
    assert fit.viscous >= 0


def test_a_joint_that_carries_no_load_keeps_its_gravity_factor() -> None:
    gravity, velocity, noise = _signals()
    torque = 0.05 * np.tanh(velocity / 0.1) + noise * 0.1
    fit = fit_joint(gravity * 0.01, velocity, torque, 1.2, 0.1)
    assert fit.gravity_factor == 1.2
    assert not fit.gravity_known


def test_the_gravity_factor_stays_in_a_believable_range() -> None:
    gravity, velocity, noise = _signals()
    fit = fit_joint(gravity, velocity, 9.0 * gravity + noise, 1.0, 0.2)
    assert fit.gravity_factor == 2.0


def test_the_friction_width_is_the_one_in_the_torque() -> None:
    gravity, velocity, noise = _signals()
    torque = 1.2 * gravity + 1.0 * np.tanh(velocity / 0.05) + noise * 0.1
    width, fits = fit_arm(gravity[:, None], velocity[:, None], torque[:, None], [1.0])
    assert width == 0.05
    assert abs(fits[0].coulomb - 1.0) < 0.05


def test_teleop_like_targets_step_and_are_quantized() -> None:
    target = lambda t: np.array([0.5 * t])  # noqa: E731
    a = teleop_like(target, 1.0000)
    b = teleop_like(target, 1.0005)
    assert a == b, "the same leader sample"
    assert abs(a[0] / ENCODER_STEP - round(a[0] / ENCODER_STEP)) < 1e-9


def test_high_frequency_rms_sees_chatter_and_not_motion() -> None:
    t = np.arange(0.0, 2.0, 0.0025)
    slow = np.sin(2 * np.pi * t)[:, None]
    chatter = slow + 0.2 * np.where(np.arange(len(t)) % 2 == 0, 1.0, -1.0)[:, None]
    assert high_frequency_rms(slow)[0] < 0.02
    assert high_frequency_rms(chatter)[0] > 0.15


def test_errors_are_binned_by_speed() -> None:
    speed = np.r_[np.full(40, 50.0), np.full(40, 150.0), np.full(5, 600.0)]
    error = np.r_[np.full(40, 2.0), np.full(40, 4.0), np.full(5, 9.0)]
    assert binned_errors(speed, error) == {0: 2.0, 100: 4.0}


def test_only_the_bins_both_have_count() -> None:
    assert score({0: 2.0, 100: 4.0}, {100: 9.0}) == 4.0


def _result(error: float, noise: float = 0.1, stable: bool = True) -> Result:
    return Result({100: error}, np.array([noise]), stable)


def test_better_tracks_better_and_is_not_louder() -> None:
    now = _result(10.0)
    assert better(_result(8.0), now)
    assert not better(_result(9.9), now), "too small a gain to believe"
    assert not better(_result(8.0, stable=False), now)


def test_small_steps_cannot_add_up_to_a_loud_arm() -> None:
    assert better(_result(8.0, noise=0.14), _result(10.0, noise=0.1))
    assert not better(_result(8.0, noise=0.14), _result(10.0, noise=0.1), np.array([0.12]))


def test_climbing_stops_at_the_first_that_is_no_better() -> None:
    measured = {100.0: _result(8.0), 120.0: _result(6.0), 140.0: _result(5.0, stable=False)}
    tried: list[float] = []

    def measure(value: float) -> Result:
        tried.append(value)
        return measured[value]

    best, result = climb([100.0, 120.0, 140.0, 160.0], _result(10.0), measure)
    assert best == 120.0
    assert result.errors == {100: 6.0}
    assert tried == [100.0, 120.0, 140.0], "160 is not tried on an arm that went unstable at 140"


def test_climbing_keeps_the_start_when_nothing_beats_it() -> None:
    best, result = climb([100.0], _result(5.0), lambda _: _result(7.0))
    assert best is None
    assert result.errors == {100: 5.0}
