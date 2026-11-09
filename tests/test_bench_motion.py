"""The fixed motion, on the arm's own model: the promise that a clear table is all it takes."""

import mujoco
import numpy as np
import pytest

from xpisode_i2rt import arm
from xpisode_i2rt.bench_motion import (
    ARM_SPACING_M,
    FIT_POSES,
    POSES,
    REST,
    SECONDS,
    TEST_POSES,
    angle,
    clearance,
    path,
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(arm.model_path(arm.GRIPPER))


@pytest.fixture(scope="module")
def whole(model: mujoco.MjModel):
    return clearance(model, path())


def test_it_stays_above_the_table(whole) -> None:
    assert whole.table_m >= 0.05


def test_it_does_not_touch_itself(whole) -> None:
    assert whole.self_contacts == 0


def test_it_keeps_clear_of_another_arm_on_either_side(whole) -> None:
    assert whole.sideways_m * 2 < ARM_SPACING_M - 0.1, "even if both ran at once"


def test_it_stays_under_the_head_camera(whole) -> None:
    assert whole.up_m < 0.8


def test_it_goes_no_further_behind_the_base_than_rest_does(model: mujoco.MjModel, whole) -> None:
    assert whole.behind_m <= clearance(model, np.stack([REST, REST])).behind_m + 1e-6


def test_the_sway_keeps_off_the_stops(model: mujoco.MjModel) -> None:
    for pose in POSES:
        joints = np.array([angle(pose, t) for t in np.arange(0.0, SECONDS, 0.02)])
        assert clearance(model, joints).stop_margin_rad >= 0.1


def test_it_is_fast_enough_to_measure_up_to_where_verne_reports(whole) -> None:
    assert 500.0 < whole.tool_speed_mm_s < 800.0


def test_settings_are_judged_on_poses_they_were_not_fitted_on() -> None:
    assert not set(FIT_POSES) & set(TEST_POSES)
