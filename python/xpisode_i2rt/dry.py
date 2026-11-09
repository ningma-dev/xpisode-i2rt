"""The controller on a real bus with no torque: zero gains and torque, and motors never enabled; it
stops at the first enabled one. Reports the loop's timing and its gravity torque against MuJoCo's.

    python -m xpisode_i2rt.dry can_follower_r
    python -m xpisode_i2rt.dry can_leader_r --leader
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from xpisode_i2rt import arm
from xpisode_i2rt.core import Core, RealtimeConfig, description


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("channel")
    parser.add_argument("--leader", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=400.0)
    args = parser.parse_args()

    hand = arm.HANDLE if args.leader else arm.GRIPPER
    model = mujoco.MjModel.from_xml_path(arm.model_path(hand))
    model.jnt_limited[:] = 0
    config = RealtimeConfig(rate_hz=args.rate, reply_timeout_s=0.5)
    told = description(
        config,
        args.channel,
        arm.model_path(hand),
        [0.0] * arm.DOF,
        None if args.leader else (-100.0, 100.0),
        args.leader,
        arm.spec()["handle"]["encoder_id"] if args.leader else None,
    )
    told["dry"] = True
    count = len(told["motors"])
    core = Core(told, config)
    core.start(np.zeros(count))
    begin = time.monotonic()
    time.sleep(0.2)
    first, started = core.state()["cycles"], time.monotonic()
    data = mujoco.MjData(model)
    worst = 0.0
    while time.monotonic() - begin < args.seconds:
        state = core.state()
        data.qpos[:] = 0
        data.qpos[: arm.DOF] = state["position"][: arm.DOF]
        data.qvel[:] = 0
        data.qacc[:] = 0
        mujoco.mj_inverse(model, data)
        factors = np.array(arm.spec()["gravity_factor"])
        worst = max(worst, float(np.abs(state["feedforward"][: arm.DOF] - factors * data.qfrc_inverse[: arm.DOF]).max()))
        time.sleep(0.05)
    state = core.state()
    print(
        f"{args.channel}: {(state['cycles'] - first) / (time.monotonic() - started):.1f} Hz, missed "
        f"{state['missed']}, dropped {state['dropped']}, late max {state['late_max_ns'] / 1e3:.0f} us, "
        f"fault {state['fault']}, {core.schedule()}"
    )
    print(f"  position {np.round(state['position'], 3).tolist()}")
    if args.leader:
        print(f"  handle {state['encoder_position']:.3f} rad, inputs {state['encoder_inputs']}")
    print(f"  gravity torque against MuJoCo: worst {worst:.4f} N*m")
    core.close()


if __name__ == "__main__":
    main()
