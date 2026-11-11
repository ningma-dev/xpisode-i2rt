# xpisode-i2rt

A real-time controller for I2RT YAM arms, and those arms as [xpisode](https://github.com/ningma-dev/xpisode)
devices. One process per CAN bus, in Rust, with no Python on the path to the motors; it knows MIT
frames, a serial chain and targets, and nothing about xpisode. This repository does not need i2rt
installed: the models and numbers it took from i2rt are in `python/xpisode_i2rt/data`, under
i2rt's license (`data/LICENSE.i2rt`).

## Set up a machine

```bash
cargo build --release
sudo scripts/setup_host.sh          # real-time priority for your user; once per machine
```

Reboot, then check in the terminal you run the station from: the limits of a desktop's terminals
come from the user's systemd manager, which logging in again may not restart. The doctor changes
nothing, tries for real, and says what is missing:

```bash
python -m xpisode_i2rt.doctor
```

`setup_host.sh` sets a limit on the user, for logins and for the desktop, so rebuilding does not lose it. Without it the controller
warns and runs at normal priority, and the Episode header says so; the tuner refuses, since settings
measured that way do not hold. CAN interfaces are expected up at 1 Mbit/s, named by udev after their
adapters, such as `can_follower_r`.

## Tune an arm

Each arm's friction and gravity differ. Clear the table, put the arm at rest, stop the station:

```bash
python -m xpisode_i2rt.tune can_follower_r --plan   # what it would do; nothing moves
python -m xpisode_i2rt.tune can_follower_r --i-am-at-the-arm --out stations/tuned/can_follower_r.json
```

About ten minutes, through the one fixed motion of `bench_motion.py`, the same on every arm, which
`tests/test_bench_motion.py` checks on the model against the table, itself and another arm. It writes
the settings it kept and every trial's numbers; the device reads that file through `tuned`. It sees
neither contact nor sound: a high `kp` makes the arm stiff against what it touches, so choosing one
is still yours.

## In a station

```toml
"--factory", "xpisode_i2rt.xpisode:YamFollower",
"--config", '{"channel": "can_follower_r", "tuned": "tuned/can_follower_r.json"}'
"--factory", "xpisode_i2rt.xpisode:YamLeader",
"--config", '{"channel": "can_leader_r"}'
```

The `tuned` path is relative to the station file, whose commands run in its directory: a station in
`stations/` reads `stations/tuned/can_follower_r.json`. Without a `gripper_limits` or a tuned file
that has them, opening drives the gripper to both stops.
What the controller was given and got goes into the Episode header.

## How it works

- **Bring-up.** `bringup.py`, in Python before the controller starts: it reads each motor's gear
  ratio, mode and scales and refuses one set up differently, writing nothing to Flash; enables the
  motors, clearing errors a bounded number of times; refuses a joint past its limits; finds the
  gripper's stops; checks the handle's firmware, and sets it to reply only when polled if it streams.
- **The bus.** Each frame goes in a slot of its own instead of a whole round at once. On a 1 Mbit/s
  classic bus through a full-speed gs_usb adapter that is what reaches high rates without dropping
  frames: `rate_hz` 400 holds on four buses at once, 450 nearly, 500 on some adapters. The handle is
  polled every fourth cycle.
- **The targets.** `player.rs` puts arrivals on an even timeline, so that their jitter does not become
  velocity and through `kd` torque, and plays them as a C1 spline, `delay_s` behind the clock. Velocity
  is the slope over `velocity_window` ticks, and acceleration comes from that.
- **The torque.** `dynamics.rs` is recursive Newton-Euler: gravity at the measured pose, inertia and
  Coriolis at the target's motion, plus Coulomb and viscous friction at the target velocity.
- **The gripper.** Its force limit runs in the device, once a tick, and decides its target.
- **When something fails.** A silent motor, a motor no longer enabled or a bus error freezes the
  target at the measured pose (a leader floats) and publishes the fault; nothing is re-enabled. A
  silent handle is no fault of the arm's: it keeps its last reading, says so on stderr past 50 ms, and
  the device reports it past `encoder_timeout_s`. If the device dies the controller holds for
  `orphan_hold_s`, then exits, and the motors' own timeout lets go: the arm has no brakes and falls.
  The next start stops a controller left behind.

## No motion

`dry: true` sends zero gains and torque and stops at the first enabled motor, so the whole loop runs
on a real bus with nothing moving:

```bash
python -m xpisode_i2rt.dry can_leader_r --leader
target/release/probe can_follower_r 7 100,250,400,500 3
target/release/handle-probe can_leader_r 400 4 25
```
