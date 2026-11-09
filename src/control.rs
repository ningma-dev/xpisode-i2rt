//! From the newest replies and the played target to each motor's next command. No I/O.

use crate::config::Config;
use crate::dm::{Command, EncoderReply, Reply, STATE_ENABLED};
use crate::dynamics::Chain;
use crate::player::{Player, Sample};
use crate::MAX_MOTORS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None = 0,
    /// A motor stopped answering.
    Silent = 1,
    /// A motor reports something other than enabled.
    MotorState = 2,
    /// The bus refused a frame.
    Bus = 3,
}

pub struct Controller {
    n: usize,
    config: Config,
    chain: Chain,
    pub player: Player,
    /// Joint space, not motor space: the first `n`.
    pub position: [f64; MAX_MOTORS],
    pub velocity: [f64; MAX_MOTORS],
    pub torque: [f64; MAX_MOTORS],
    pub temperature: [f64; MAX_MOTORS],
    pub target_position: [f64; MAX_MOTORS],
    pub target_velocity: [f64; MAX_MOTORS],
    pub feedforward: [f64; MAX_MOTORS],
    acceleration: [f64; MAX_MOTORS],
    last_reply: [f64; MAX_MOTORS],
    pub encoder: EncoderReply,
    /// When the encoder last answered. Its silence is no fault of the arm's: the motors are
    /// answering and it can still park. Whoever reads the encoder decides what its age means.
    pub encoder_time: f64,
    floating: bool,
    pub fault: Fault,
    pub fault_motor: i32,
    pub fault_detail: i32,
    pub starved_cycles: u64,
    holding: bool,
    last_cycle: f64,
}

impl Controller {
    pub fn new(config: Config, now: f64) -> Controller {
        let n = config.motors.len();
        assert!(n <= MAX_MOTORS && config.links.len() <= n);
        let chain = Chain::new(&config.links, config.gravity);
        let floating = config.floats;
        let player = Player::new(n, config.period_s, config.delay_s, config.velocity_window);
        Controller {
            n,
            config,
            chain,
            player,
            position: [0.0; MAX_MOTORS],
            velocity: [0.0; MAX_MOTORS],
            torque: [0.0; MAX_MOTORS],
            temperature: [0.0; MAX_MOTORS],
            target_position: [0.0; MAX_MOTORS],
            target_velocity: [0.0; MAX_MOTORS],
            feedforward: [0.0; MAX_MOTORS],
            acceleration: [0.0; MAX_MOTORS],
            last_reply: [now; MAX_MOTORS],
            encoder: EncoderReply::default(),
            encoder_time: now,
            floating,
            fault: Fault::None,
            fault_motor: -1,
            fault_detail: 0,
            starved_cycles: 0,
            holding: true,
            last_cycle: now,
        }
    }

    /// The pose to hold before the first target, joint space.
    pub fn hold_at(&mut self, q: [f64; MAX_MOTORS]) {
        self.position = q;
        self.target_position = q;
        self.target_velocity = [0.0; MAX_MOTORS];
        self.acceleration = [0.0; MAX_MOTORS];
        self.holding = true;
        self.floating = self.config.floats;
        self.player.clear();
    }

    /// Stop where the target is now; an arm that floats starts floating.
    pub fn hold(&mut self) {
        let q = self.target_position;
        self.target_velocity = [0.0; MAX_MOTORS];
        self.acceleration = [0.0; MAX_MOTORS];
        self.target_position = q;
        self.holding = true;
        self.floating = self.config.floats;
        self.player.clear();
    }

    pub fn push(&mut self, now: f64, q: [f64; MAX_MOTORS]) {
        if self.fault != Fault::None {
            return;
        }
        if self.floating {
            // It tracks from where the operator left it, not from where a target last was.
            self.floating = false;
            self.target_position = self.position;
        }
        self.holding = false;
        self.player.push(now, q);
    }

    pub fn raise(&mut self, fault: Fault, motor: i32, detail: i32) {
        if self.fault == Fault::None {
            self.fault = fault;
            self.fault_motor = motor;
            self.fault_detail = detail;
            // Hold where the arm is, not where it was asked to go.
            self.target_position = self.position;
            self.hold();
        }
    }

    pub fn on_reply(&mut self, i: usize, reply: &Reply, now: f64) {
        let m = &self.config.motors[i];
        self.position[i] = (reply.position - m.offset) * m.direction;
        self.velocity[i] = reply.velocity * m.direction;
        self.torque[i] = reply.torque * m.direction;
        self.temperature[i] = reply.temperature_rotor as f64;
        self.last_reply[i] = now;
        if !self.config.dry && reply.state != STATE_ENABLED {
            self.raise(Fault::MotorState, i as i32, reply.state as i32);
        }
    }

    pub fn on_encoder(&mut self, reply: EncoderReply, now: f64) {
        self.encoder = reply;
        self.encoder_time = now;
    }

    /// Once a cycle: where the targets are now, and the torque to go with them.
    pub fn cycle(&mut self, now: f64) {
        let dt = (now - self.last_cycle).clamp(1e-4, 0.05);
        self.last_cycle = now;
        for i in 0..self.n {
            if now - self.last_reply[i] > self.config.reply_timeout_s {
                self.raise(Fault::Silent, i as i32, 0);
            }
        }
        let sample: Option<Sample> = if self.holding { None } else { self.player.sample(now) };
        if let Some(s) = sample {
            if s.starved {
                self.starved_cycles += 1;
            }
            let alpha = 1.0 - (-dt / self.config.acceleration_tau_s.max(1e-6)).exp();
            for j in 0..self.n {
                let m = &self.config.motors[j];
                let limit = m.max_velocity;
                // Never faster than the limit, whatever the targets ask: a jump is a slew.
                let step = (s.position[j] - self.target_position[j]).clamp(-limit * dt, limit * dt);
                let q = (self.target_position[j] + step).clamp(m.low, m.high);
                let mut v = m.velocity_scale * s.velocity[j].clamp(-limit, limit);
                // At a limit the position is clipped; a velocity would keep pushing into it.
                if q <= m.low || q >= m.high {
                    v = 0.0;
                }
                self.target_position[j] = q;
                self.target_velocity[j] = v;
                self.acceleration[j] += alpha * (s.acceleration[j] - self.acceleration[j]);
            }
        }
        if self.floating {
            self.target_position = self.position;
        }
        self.compute_feedforward();
    }

    fn compute_feedforward(&mut self) {
        let links = self.chain.len();
        let mut gravity = [0.0; MAX_MOTORS];
        let mut motion = [0.0; MAX_MOTORS];
        let zeros = [0.0; MAX_MOTORS];
        self.chain.inverse_dynamics(&self.position, &zeros, &zeros, true, &mut gravity);
        self.chain.inverse_dynamics(
            &self.target_position,
            &self.target_velocity,
            &self.acceleration,
            false,
            &mut motion,
        );
        for j in 0..self.n {
            let m = &self.config.motors[j];
            let v = self.target_velocity[j];
            let friction =
                m.coulomb * (v / self.config.friction_width).tanh() + m.viscous * v;
            let dynamic = if j < links { m.gravity_scale * gravity[j] + m.inertia_scale * motion[j] } else { 0.0 };
            self.feedforward[j] = (dynamic + friction).clamp(-m.torque_limit, m.torque_limit);
        }
    }

    /// The command for motor `i`, in motor space.
    pub fn command(&self, i: usize) -> Command {
        let m = &self.config.motors[i];
        if self.config.dry {
            return Command::default();
        }
        if self.floating {
            return Command {
                position: self.target_position[i] * m.direction + m.offset,
                velocity: 0.0,
                kp: 0.0,
                kd: m.float_kd,
                torque: self.feedforward[i] * m.direction,
            };
        }
        Command {
            position: self.target_position[i] * m.direction + m.offset,
            velocity: self.target_velocity[i] * m.direction,
            kp: m.kp,
            kd: m.kd,
            torque: self.feedforward[i] * m.direction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Link, Motor};

    fn config(floats: bool) -> Config {
        let motor = Motor {
            id: 1,
            kind: "DM4340".into(),
            direction: 1.0,
            offset: 0.0,
            kp: 80.0,
            kd: 5.0,
            low: -3.0,
            high: 3.0,
            torque_limit: 28.0,
            gravity_scale: 1.0,
            coulomb: 0.0,
            viscous: 0.0,
            inertia_scale: 0.0,
            max_velocity: 4.0,
            velocity_scale: 1.0,
            float_kd: 0.1,
        };
        let link = Link {
            position: [0.0; 3],
            quaternion: [1.0, 0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
            mass: 1.0,
            center_of_mass: [0.5, 0.0, 0.0],
            inertia: [0.0; 6],
        };
        Config {
            channel: "x".into(),
            shm_path: "x".into(),
            motors: vec![motor],
            links: vec![link],
            gravity: [0.0, 0.0, -9.81],
            rate_hz: 400.0,
            period_s: 0.01,
            delay_s: 0.01,
            velocity_window: 4,
            friction_width: 0.2,
            acceleration_tau_s: 0.02,
            reply_timeout_s: 0.05,
            orphan_hold_s: 60.0,
            fifo_priority: 0,
            cpu: -1,
            floats,
            encoder_id: None,
            encoder_every: 1,
            dry: false,
        }
    }

    fn reply(position: f64) -> Reply {
        Reply { state: STATE_ENABLED, position, ..Default::default() }
    }

    #[test]
    fn a_leader_floats_tracks_and_floats_again() {
        let mut c = Controller::new(config(true), 0.0);
        c.hold_at([0.0; MAX_MOTORS]);
        c.on_reply(0, &reply(0.4), 0.001);
        c.cycle(0.002);
        let floating = c.command(0);
        assert_eq!((floating.kp, floating.kd, floating.velocity), (0.0, 0.1, 0.0));
        assert!(floating.torque.abs() > 1.0, "gravity compensation goes on while it floats");

        let mut q = [0.0; MAX_MOTORS];
        q[0] = 0.4;
        c.push(0.003, q);
        c.cycle(0.004);
        let tracking = c.command(0);
        assert_eq!((tracking.kp, tracking.kd), (80.0, 5.0));
        assert!((tracking.position - 0.4).abs() < 1e-9, "from where the operator left it");

        c.hold();
        c.cycle(0.005);
        assert_eq!(c.command(0).kp, 0.0);
    }

    #[test]
    fn a_fault_lets_a_leader_float_and_a_follower_hold() {
        for (floats, kp) in [(true, 0.0), (false, 80.0)] {
            let mut c = Controller::new(config(floats), 0.0);
            c.hold_at([0.0; MAX_MOTORS]);
            c.cycle(0.1);
            assert_eq!(c.fault, Fault::Silent);
            assert_eq!(c.command(0).kp, kp);
        }
    }
}
