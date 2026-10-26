# xpisode-i2rt

Connect I2RT YAM arms and teaching handles to [xpisode](https://github.com/ningma-dev/xpisode) for teleoperation, policy control and recording.

The adapter publishes joint states, gripper state and handle inputs through the device protocol. I2RT's SDK owns the CAN bus, motor control, gravity compensation and gripper calibration. Each arm runs as its own device process.

## Supported hardware

- YAM follower with a `linear_4310` gripper.
- YAM leader with a teaching handle.
- I2RT's MuJoCo simulator for trying the follower without hardware.

The other arm and gripper names accepted by the SDK are not validated by this adapter. Use Linux with one CAN interface per real arm, configured at 1 Mbit/s. Give interfaces stable names before configuring a multi-arm station; see the [SDK CAN setup guide](https://github.com/ningma-dev/i2rt/blob/xpisode/docs/guides/set-persistent-can-ids.md).

## Install and try it

Running a station requires access to the private xpisode core repository or core wheel files supplied by the maintainer. The adapter is public; the core and its SDKs are not on PyPI.

Follow [Install a station](docs/installation.md). It creates a complete station project, installs the core and adapter, and copies both simulator and hardware configurations.

From that station project, start the supplied simulator:

```bash
uv run xpisode run stations/yam/sim_station.toml
```

In another terminal, in the same project:

```bash
uv run xpisode status --station yam_sim
uv run xpisode record start --station yam_sim --task demo
# Perform or observe the demonstration, then:
uv run xpisode record stop --station yam_sim
uv run xpisode shutdown --station yam_sim
uv run xpisode verify data/yam_sim/episodes
```

The example uses I2RT's MuJoCo follower and a moving demonstration handle. It opens no CAN interfaces. For development of the adapter alone, clone this repository and run `uv sync --python 3.12`; the pinned SDK sources in `pyproject.toml` and `uv.lock` are fetched using your Git credentials, which must have access to the private core repository.

## Connect real arms

Edit `stations/yam/yam_station.toml`: the example assigns the teaching handle to `can0` and the follower to `can1`. Set these to your actual interfaces. Keep `teleop.toml` beside the station file.

```bash
uv run xpisode run stations/yam/yam_station.toml
```

Opening the follower calibrates its gripper unless `gripper_limits` is supplied. The leader then moves to the follower's pose before teleoperation starts. Keep the workspace clear and release the handle while it aligns.

- Button 0 switches between teleoperation and the policy holding the arm; without a policy, the follower holds.
- Button 1 starts or stops a recording.
- When a policy drives the follower, the leader follows it so the operator can take over from the current pose.
- Ctrl+C shuts down the station. Both arms move to their park poses before releasing the motors; release the leader during this motion.

The arms have no brakes: removing power can make them fall. If the station loses control, the follower holds and the leader returns to gravity compensation. The SDK's logs report CAN or motor failures.

## Configure an arm

Factories are `xpisode_i2rt:YamFollower` and `xpisode_i2rt:YamLeader`. Pass their options as the JSON argument to `xpisode-device run --config`, as shown in the supplied station files.

| Option | Default | Meaning |
| --- | --- | --- |
| `channel` | required for real hardware | CAN interface; the simulator does not use it |
| `arm` | `yam` | SDK arm type |
| `poll_hz` | 200 | Published state or input sampling rate |
| `park_pose` | pose measured on connection | Joint positions in radians used for parking |
| `park_seconds` | 3.0 | Duration of the parking move |

Follower options:

| Option | Default | Meaning |
| --- | --- | --- |
| `gripper` | `linear_4310` | SDK gripper type; a powered gripper is required |
| `sim` | `false` | Use the SDK simulator |
| `gripper_limits` | calibrated on connection | SDK closed/open limits; supplying them skips calibration |

Leader option: `ee_mass` overrides the handle mass used by the SDK for gravity compensation.

The follower exposes `state`, with joint position, velocity and measured torque plus gripper state. The leader exposes `input`, with joint positions, trigger and buttons. Both accept the `joint_position` endpoint and a `park` operation. Joint positions use radians; gripper position is normalized from 0 (closed) to 1 (open). State timestamps identify the host read time because the SDK does not expose acquisition timestamps.

The adapter declares a self-contained robot model; xpisode saves it with the station's data. Calibration facts and SDK version are recorded in the device description.

## Develop and test

See [Contributing](CONTRIBUTING.md) for lint, type checks, simulator tests and building wheel/source archives. The tests do not move physical arms.

MIT licensed. The I2RT SDK remains a separate dependency with its own license.
