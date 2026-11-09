"""The controller's inverse dynamics against MuJoCo's, on the arm's own model."""

import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import pytest

from xpisode_i2rt import arm
from xpisode_i2rt.core import chain_links

CHECK = Path(__file__).parents[1] / "target/release/dynamics-check"


@pytest.mark.skipif(not CHECK.exists(), reason="the controller is not built")
@pytest.mark.parametrize("hand", [arm.GRIPPER, arm.HANDLE])
def test_matches_mujoco(tmp_path: Path, hand: str) -> None:
    model = mujoco.MjModel.from_xml_path(arm.model_path(hand))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    model.jnt_limited[:] = 0
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    cases, expected = [], []
    for _ in range(200):
        q, v, a = rng.uniform(-2, 2, 6), rng.uniform(-4, 4, 6), rng.uniform(-30, 30, 6)
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qacc[:] = 0
        data.qpos[:6], data.qvel[:6], data.qacc[:6] = q, v, a
        mujoco.mj_inverse(model, data)
        cases.append([*q, *v, *a])
        expected.append(data.qfrc_inverse[:6].copy())
    description = chain_links(model, 6)
    file = tmp_path / "chain.json"
    file.write_text(json.dumps({**description, "cases": cases}))
    out = subprocess.run([str(CHECK), str(file)], capture_output=True, text=True, check=True)
    got = np.array(json.loads(out.stdout))
    assert np.abs(got - np.array(expected)).max() < 1e-6
