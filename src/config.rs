//! What one bus's controller is told when it starts, as JSON.

use serde::Deserialize;

fn every_cycle() -> u64 {
    1
}

#[derive(Clone, Debug, Deserialize)]
pub struct Motor {
    pub id: u32,
    /// "DM4340" or "DM4310".
    pub kind: String,
    pub direction: f64,
    pub offset: f64,
    pub kp: f64,
    pub kd: f64,
    /// Joint position limits, rad.
    pub low: f64,
    pub high: f64,
    pub torque_limit: f64,
    pub gravity_scale: f64,
    /// Coulomb friction, N*m, fed forward at the target velocity.
    pub coulomb: f64,
    /// Viscous friction, N*m*s/rad.
    pub viscous: f64,
    pub inertia_scale: f64,
    pub max_velocity: f64,
    /// The share of the target's velocity to command. A gripper that closes on something has no
    /// use for it: kd times the velocity is force the force limit does not see.
    pub velocity_scale: f64,
    /// The damping while it floats.
    #[serde(default)]
    pub float_kd: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Link {
    /// The joint's origin in the parent link, and the rest rotation from the parent, w x y z.
    pub position: [f64; 3],
    pub quaternion: [f64; 4],
    pub axis: [f64; 3],
    pub mass: f64,
    pub center_of_mass: [f64; 3],
    /// About the center of mass, in the link's frame: xx yy zz xy xz yz.
    pub inertia: [f64; 6],
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub channel: String,
    pub shm_path: String,
    pub motors: Vec<Motor>,
    /// The chain the first motors drive; any further motors get no dynamics feedforward.
    pub links: Vec<Link>,
    /// In the frame of the first link's parent.
    pub gravity: [f64; 3],
    pub rate_hz: f64,
    /// The spacing of the targets, s.
    pub period_s: f64,
    /// How far behind the clock targets are played, s. At one period the player knows both ends of
    /// the segment it is on.
    pub delay_s: f64,
    /// The periods the target velocity is the slope over. See `player.rs`.
    pub velocity_window: usize,
    /// The target velocity, rad/s, around which the Coulomb torque changes sign.
    pub friction_width: f64,
    /// Filter time constant of the feedforward acceleration, s.
    pub acceleration_tau_s: f64,
    /// A motor silent this long is a fault, s.
    pub reply_timeout_s: f64,
    /// How long to hold after the parent died, before letting go, s.
    pub orphan_hold_s: f64,
    /// SCHED_FIFO priority; 0 leaves the scheduler alone.
    pub fifo_priority: i32,
    /// The CPU to pin to; negative leaves affinity alone.
    pub cpu: i32,
    /// Whether the arm rests by floating under an operator's hand, on gravity compensation, rather
    /// than by holding its pose. Targets make it track; `hold` and faults let it float again.
    #[serde(default)]
    pub floats: bool,
    /// A passive encoder on the same bus, polled once a cycle in a slot of its own.
    #[serde(default)]
    pub encoder_id: Option<u32>,
    /// The encoder is polled every this many cycles. Beside enabled motors a teaching handle has
    /// been seen to go silent while it was polled every cycle at 400 Hz; i2rt polls it at about 100.
    #[serde(default = "every_cycle")]
    pub encoder_every: u64,
    /// No torque: gains and torque are zero, and an enabled motor stops the process.
    pub dry: bool,
}
