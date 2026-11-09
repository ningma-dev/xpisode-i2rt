"""Find one follower's controller settings on the bench, with nothing but a clear table.

THE ARM MOVES, faster than teleoperation usually does, for about ten minutes, and the gripper opens
and closes. Clear the table and stay at the arm. Ctrl-C stops and goes back to where it started.
The station must not be running, and the arm must start at rest. It goes through the one fixed motion
of xpisode_i2rt.bench_motion, the same on every arm.

    python -m xpisode_i2rt.tune can_follower_r --plan            # what it would do; nothing moves
    python -m xpisode_i2rt.tune can_follower_r --i-am-at-the-arm

It measures friction and gravity from torque, then tries higher gains and inertia feedforward, one
at a time, with leader-like targets, on poses it did not fit on. What it keeps must track better and
not be louder, and a joint that starts to oscillate ends that climb at once. It writes
stations/tuned/<channel>.json, which the station reads; every trial's numbers go in there too.
"""

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from xpisode_device import MonotonicClock

from xpisode_i2rt import arm
from xpisode_i2rt.bench_motion import (
    FIT_POSES,
    GO_SECONDS,
    POSES,
    REST,
    REST_TOLERANCE,
    SECONDS,
    TEST_POSES,
    TICK_S,
    Pose,
    angle,
    clearance,
    path,
)
from xpisode_i2rt.can import require_idle
from xpisode_i2rt.core import Core, RealtimeConfig, move
from xpisode_i2rt.doctor import can_be_realtime
from xpisode_i2rt.tuning import (
    Result,
    binned_errors,
    climb,
    fit_arm,
    high_frequency_rms,
    score,
    teleop_like,
)
from xpisode_i2rt.xpisode import YamFollower

ARM_KP = (100.0, 120.0, 140.0, 160.0)
WRIST_KP = (15.0, 20.0, 30.0, 40.0)
INERTIA = (0.5, 1.0)
OSCILLATION_NM = np.array([1.5, 1.5, 1.5, 0.6, 0.6, 0.6])
"""High-frequency torque a joint must stay under; beyond it the trial ends where it is."""
RUNAWAY_RAD = 0.25
ROTOR_LIMIT_C = 70.0


class Kinematics:
    def __init__(self, path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("tcp_site").id
        self.inverse_model = mujoco.MjModel.from_xml_path(path)
        self.inverse_model.jnt_limited[:] = 0
        self.inverse_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
        self.inverse_data = mujoco.MjData(self.inverse_model)

    def tool(self, joints: np.ndarray) -> np.ndarray:
        out = np.zeros((len(joints), 3))
        for i, q in enumerate(joints):
            self.data.qpos[:] = 0
            self.data.qpos[:6] = q
            mujoco.mj_kinematics(self.model, self.data)
            out[i] = self.data.site_xpos[self.site]
        return out

    def gravity(self, joints: np.ndarray) -> np.ndarray:
        out = np.zeros((len(joints), 6))
        for i, q in enumerate(joints):
            self.inverse_data.qpos[:] = 0
            self.inverse_data.qvel[:] = 0
            self.inverse_data.qacc[:] = 0
            self.inverse_data.qpos[:6] = q
            mujoco.mj_inverse(self.inverse_model, self.inverse_data)
            out[i] = self.inverse_data.qfrc_inverse[:6]
        return out


class Bench:
    """One arm under a controller that can be replaced while it holds."""

    def __init__(self, follower: YamFollower, kinematics: Kinematics) -> None:
        self.follower, self.kinematics = follower, kinematics
        self.gripper = float(self.core.state()["target_position"][6])
        self.config = follower.realtime
        assert self.config is not None

    @property
    def core(self) -> Core:
        return self.follower.core

    def use(self, config: RealtimeConfig) -> None:
        if config != self.config:
            self.follower.replace_controller(config)
            self.config = config

    def go(self, center: np.ndarray, seconds: float = GO_SECONDS) -> None:
        now = self.core.state()["target_position"]
        target = now.copy()
        target[:6] = center
        target[6] = self.gripper
        move(self.core, now, target, seconds, TICK_S)

    def sway(self, pose: Pose, leader_like: bool) -> dict:
        def target(t: float) -> np.ndarray:
            return angle(pose, t)

        self.go(np.array(pose.center))
        begin = time.monotonic()
        next_tick, tick = begin, 0
        log, asked = [], []
        stable = True
        last_check = begin
        while tick * TICK_S < SECONDS:
            now = time.monotonic()
            if now >= next_tick:
                t = tick * TICK_S
                q = teleop_like(target, t) if leader_like else target(t)
                asked.append([now, *target(t)])
                self.core.check()
                self.core.push(np.r_[q, self.gripper])
                tick += 1
                next_tick = begin + tick * TICK_S
            state = self.core.state()
            log.append([state["time_ns"] * 1e-9, *state["position"][:6], *state["torque"][:6]])
            if now - last_check > 0.05 and len(log) > 300:
                last_check = now
                _, keep = np.unique(np.array(log[-300:])[:, 0], return_index=True)
                recent = np.array(log[-300:])[np.sort(keep)]
                noise = high_frequency_rms(recent[:, 7:13])
                error = np.abs(state["position"][:6] - state["target_position"][:6])
                if np.any(noise > OSCILLATION_NM) or np.any(error > RUNAWAY_RAD):
                    print(f"    stopped: oscillation {np.round(noise, 2).tolist()}, error {np.round(error, 3).tolist()}")
                    self.core.hold()
                    stable = False
                    break
                if state["temperature"].max() > ROTOR_LIMIT_C:
                    raise SystemExit(f"a rotor reached {state['temperature'].max():.0f} C; let it cool and run again")
            time.sleep(0.001)
        log = np.array(log)
        _, keep = np.unique(log[:, 0], return_index=True)
        return {"log": log[np.sort(keep)], "asked": np.array(asked), "stable": stable}

    def measure(self, config: RealtimeConfig, poses: tuple[Pose, ...], label: str) -> Result:
        self.use(config)
        speeds, errors, noises, stable = [], [], [], True
        for pose in poses:
            run = self.sway(pose, leader_like=True)
            stable = stable and run["stable"]
            if not run["stable"]:
                noises.append(OSCILLATION_NM)
                break
            log, asked = run["log"], run["asked"]
            t = log[:, 0]
            grid = np.arange(t[0] + 2.0, t[-1] - 2.0, 0.01)
            measured = np.stack([np.interp(grid, t, log[:, 1 + j]) for j in range(6)], 1)
            wanted = np.stack([np.interp(grid, asked[:, 0], asked[:, 1 + j]) for j in range(6)], 1)
            tool, ideal = self.kinematics.tool(measured), self.kinematics.tool(wanted)
            speeds.append(np.linalg.norm(np.gradient(ideal, 0.01, axis=0), axis=1) * 1e3)
            errors.append(np.linalg.norm(tool - ideal, axis=1) * 1e3)
            noises.append(high_frequency_rms(log[:, 7:13]))
        binned = binned_errors(np.concatenate(speeds), np.concatenate(errors)) if speeds else {}
        result = Result(binned, np.max(noises, axis=0), stable)
        state = self.core.state()
        print(f"  {label:32s} " + " ".join(f"{k}:{v:5.2f}" for k, v in binned.items())
              + f"  noise max {result.noise.max():.3f}"
              + f"  rotor {state['temperature'].max():.0f} C" + ("" if stable else "  UNSTABLE"))
        return result

    def identify(self, poses: tuple[Pose, ...], defaults: list[float]) -> dict:
        gravity, velocity, torque = [], [], []
        for pose in poses:
            run = self.sway(pose, leader_like=False)
            log, asked = run["log"], run["asked"]
            t = log[:, 0]
            grid = np.arange(t[0] + 2.0, t[-1] - 2.0, 0.01)
            wanted = np.stack([np.interp(grid, asked[:, 0], asked[:, 1 + j]) for j in range(6)], 1)
            measured = np.stack([np.interp(grid, t, log[:, 1 + j]) for j in range(6)], 1)
            gravity.append(self.kinematics.gravity(measured))
            velocity.append(np.gradient(wanted, 0.01, axis=0))
            torque.append(np.stack([np.interp(grid, t, log[:, 7 + j]) for j in range(6)], 1))
        width, fits = fit_arm(np.concatenate(gravity), np.concatenate(velocity), np.concatenate(torque), defaults)
        for j, fit in enumerate(fits):
            print(f"  joint {j + 1}: gravity {fit.gravity_factor:.2f}{'' if fit.gravity_known else ' (kept)'}, "
                  f"coulomb {fit.coulomb:.3f}, viscous {fit.viscous:.3f}, unexplained {fit.residual_rms:.3f} N*m")
        return {
            "friction_width": width,
            "coulomb": tuple(round(f.coulomb, 4) for f in fits),
            "viscous": tuple(round(f.viscous, 4) for f in fits),
            "gravity_factor": tuple(round(f.gravity_factor, 4) for f in fits),
        }

    def gripper_report(self) -> dict:
        """Not tuned: how fast it goes, for whoever feels it isn't following."""
        state = self.core.state()
        low, high = sorted(self.follower._gripper_limits())
        span = high - low
        hold = state["target_position"].copy()
        rows = []
        for target in (low + 0.9 * span, low + 0.1 * span, low + 0.9 * span, low + 0.1 * span):
            start = self.core.state()["position"][6]
            begin = time.monotonic()
            reached = None
            while time.monotonic() - begin < 1.2:
                hold[6] = target
                self.core.push(hold)
                position = self.core.state()["position"][6]
                if reached is None and abs(position - target) < 0.1 * abs(target - start):
                    reached = time.monotonic() - begin
                time.sleep(TICK_S)
            rows.append(None if reached is None else round(reached * 1e3))
        hold[6] = self.gripper
        move(self.core, self.core.state()["target_position"], hold, 1.0, TICK_S)
        print(f"  gripper, 80% of its stroke, ms to 90% there: {rows}")
        return {"ms_to_90_percent_of_an_80_percent_stroke": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("channel", help="can_follower_l or can_follower_r")
    parser.add_argument("--plan", action="store_true", help="say where it would go; nothing moves")
    parser.add_argument("--i-am-at-the-arm", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rate", type=float, default=400.0)
    parser.add_argument("--allow-no-fifo", action="store_true", help="settings measured so do not hold")
    args = parser.parse_args()
    if args.plan == args.i_am_at_the_arm:
        raise SystemExit("say either --plan, or --i-am-at-the-arm with the table clear")
    if args.out is None and args.plan is False:
        raise SystemExit("say where the arm's file goes: --out, such as stations/tuned/can_follower_r.json")

    kinematics = Kinematics(arm.model_path(arm.GRIPPER))
    check = mujoco.MjModel.from_xml_path(arm.model_path(arm.GRIPPER))
    whole = clearance(check, path())
    print(f"the fixed motion: {len(POSES)} poses, tool up to {whole.tool_speed_mm_s:.0f} mm/s, "
          f"lowest part {whole.table_m * 100:.1f} cm above the table, "
          f"{whole.sideways_m * 100:.1f} cm to either side of the base, "
          f"highest {whole.up_m * 100:.0f} cm, {whole.self_contacts} self contacts")
    trials = 2 + len(ARM_KP) + len(WRIST_KP) + len(INERTIA)
    print(f"at most {trials} trials, about "
          f"{(SECONDS + GO_SECONDS) * (len(FIT_POSES) + trials * len(TEST_POSES)) / 60:.0f} minutes")
    if args.plan:
        return
    if not can_be_realtime() and not args.allow_no_fifo:
        raise SystemExit("the controller would get no real-time priority: run scripts/setup_host.sh first")
    require_idle(args.channel)

    out = args.out
    previous = json.loads(out.read_text()) if out.exists() else {}
    limits = previous.get("gripper_limits")
    start = RealtimeConfig(rate_hz=args.rate)
    follower = YamFollower(args.channel, gripper_limits=limits, rate_hz=args.rate, park_seconds=4.0)
    follower.open(MonotonicClock())
    table: dict[str, dict] = {}
    try:
        bench = Bench(follower, kinematics)
        at = bench.core.state()["position"][:6]
        if np.abs(at - REST).max() > REST_TOLERANCE:
            raise SystemExit(f"the arm is at {np.round(at, 2).tolist()}, not at rest: put it down and run again")
        spec = arm.spec()
        print(f"schedule {bench.core.schedule()}")

        def record(label: str, result: Result) -> Result:
            table[label] = {"mm_by_speed": result.errors, "noise": result.noise.tolist(), "stable": result.stable}
            return result

        print("as i2rt has it:")
        config = start
        reference = record("start", bench.measure(config, TEST_POSES, "start"))
        ceiling = np.maximum(2.0 * reference.noise, 0.05)
        current = reference

        print("friction and gravity, from torque:")
        fitted = bench.identify(FIT_POSES, [float(x) for x in spec["gravity_factor"]])
        candidate = replace(config, **fitted)
        tried = record("fitted", bench.measure(candidate, TEST_POSES, "fitted"))
        if tried.stable and score(tried.errors, current.errors) < score(current.errors, tried.errors):
            config, current = candidate, tried

        base_kp = tuple(float(x) for x in spec["kp"])
        print("arm gains:")
        def arm_gain(value: float) -> Result:
            kp = (value, value, value, *(config.kp or base_kp)[3:])
            return record(f"arm kp {value:g}", bench.measure(replace(config, kp=kp), TEST_POSES, f"arm kp {value:g}"))
        best, current = climb(ARM_KP, current, arm_gain, ceiling)
        if best is not None:
            config = replace(config, kp=(best, best, best, *(config.kp or base_kp)[3:]))

        print("wrist gains:")
        def wrist_gain(value: float) -> Result:
            kp = (*(config.kp or base_kp)[:3], value, value, value)
            return record(f"wrist kp {value:g}", bench.measure(replace(config, kp=kp), TEST_POSES, f"wrist kp {value:g}"))
        best, current = climb(WRIST_KP, current, wrist_gain, ceiling)
        if best is not None:
            config = replace(config, kp=(*(config.kp or base_kp)[:3], best, best, best))

        print("inertia feedforward:")
        def inertia(value: float) -> Result:
            return record(f"inertia {value:g}", bench.measure(replace(config, inertia_scale=value), TEST_POSES, f"inertia {value:g}"))
        best, current = climb(INERTIA, current, inertia, ceiling)
        if best is not None:
            config = replace(config, inertia_scale=best)

        bench.use(config)
        print("gripper:")
        gripper = bench.gripper_report()

        keys = ("friction_width", "coulomb", "viscous", "gravity_factor", "kp", "inertia_scale")
        params = {k: list(v) if isinstance(v, tuple) else v for k, v in ((k, getattr(config, k)) for k in keys)}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "params": params,
            "gripper_limits": [float(x) for x in follower._gripper_limits()],
            "provenance": {
                "measured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "by": "xpisode_i2rt.tune",
                "channel": args.channel,
                "motion": [{"center": p.center, "sway": p.sway} for p in POSES],
                "schedule": bench.core.schedule(),
            },
            "start_mm_by_speed": reference.errors,
            "final_mm_by_speed": current.errors,
            "gripper": gripper,
            "trials": table,
        }, indent=1))
        print(f"\nstart {reference.errors}\nfinal {current.errors}\nkept  {params}\nwrote {out}")
    except KeyboardInterrupt:
        print("stopped; nothing written")
        bench.core.hold()
    finally:
        follower.close()


if __name__ == "__main__":
    main()
