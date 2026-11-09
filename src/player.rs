//! Plays targets that arrive one a tick as a smooth motion, on a timeline of its own.
//!
//! The arrival times jitter, and a target's velocity is its difference over the time between
//! targets: jitter there is torque through kd. So a target that arrives about when it should is
//! put exactly one period after the one before, with a slow correction for clock drift. One that
//! arrives much later starts anew from where the arm stopped.
//!
//! Velocity needs more than that. Leader encoders step and their samples beat with the tick, so
//! the difference of neighbouring targets jumps by a large part of a rad/s even in a smooth motion,
//! and kd and the friction feedforward make noise of it that the arm can be heard to play. The
//! velocity is therefore the slope over the last `velocity_window` periods. It is late by half of
//! that, which costs far less position than the noise costs torque.

use crate::MAX_MOTORS;

const KNOTS: usize = 32;
const DRIFT_CORRECTION: f64 = 0.01;
const STALE_PERIODS: f64 = 1.5;
/// Velocity falls to nothing with this time constant when the targets run out.
const RUN_OUT_TAU_S: f64 = 0.01;

#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    pub position: [f64; MAX_MOTORS],
    pub velocity: [f64; MAX_MOTORS],
    pub acceleration: [f64; MAX_MOTORS],
    /// There was no target for this time: the arm holds the newest.
    pub starved: bool,
}

#[derive(Clone, Copy)]
struct Knot {
    t: f64,
    q: [f64; MAX_MOTORS],
    /// The motion stops here: a tangent through it would overshoot.
    rest: bool,
}

pub struct Player {
    n: usize,
    period: f64,
    delay: f64,
    velocity_window: usize,
    knots: [Knot; KNOTS],
    count: usize,
    run_out_velocity: [f64; MAX_MOTORS],
    last_sample: f64,
}

impl Player {
    pub fn new(n: usize, period: f64, delay: f64, velocity_window: usize) -> Player {
        Player {
            n,
            period,
            delay,
            velocity_window: velocity_window.max(1),
            knots: [Knot { t: 0.0, q: [0.0; MAX_MOTORS], rest: false }; KNOTS],
            count: 0,
            run_out_velocity: [0.0; MAX_MOTORS],
            last_sample: 0.0,
        }
    }

    pub fn clear(&mut self) {
        self.count = 0;
        self.run_out_velocity = [0.0; MAX_MOTORS];
    }

    fn append(&mut self, t: f64, q: [f64; MAX_MOTORS], rest: bool) {
        if self.count == KNOTS {
            self.knots.copy_within(1.., 0);
            self.count -= 1;
        }
        self.knots[self.count] = Knot { t, q, rest };
        self.count += 1;
    }

    pub fn push(&mut self, now: f64, q: [f64; MAX_MOTORS]) {
        if self.count == 0 {
            self.append(now, q, false);
            return;
        }
        let newest = self.knots[self.count - 1];
        let expected = newest.t + self.period;
        let early = now - expected;
        if early.abs() < STALE_PERIODS * self.period {
            self.append(expected + DRIFT_CORRECTION * early, q, false);
        } else {
            // A pause: hold where the last target was until a period before this one.
            self.knots[self.count - 1].rest = true;
            let hold_until = now - self.period;
            if hold_until > newest.t {
                self.append(hold_until, newest.q, true);
            }
            self.append(now.max(newest.t + self.period), q, false);
        }
    }

    /// The slope over up to `velocity_window` periods ending at knot `i`, back to the last rest.
    fn slope(&self, i: usize, j: usize) -> f64 {
        let mut first = i;
        while first > 0 && i - first < self.velocity_window && !self.knots[first].rest {
            first -= 1;
        }
        if first == i {
            return 0.0;
        }
        (self.knots[i].q[j] - self.knots[first].q[j]) / (self.knots[i].t - self.knots[first].t)
    }

    pub fn sample(&mut self, now: f64) -> Option<Sample> {
        if self.count == 0 {
            return None;
        }
        let dt = (now - self.last_sample).clamp(0.0, 0.1);
        self.last_sample = now;
        let at = now - self.delay;
        let n = self.n;
        let mut out = Sample::default();
        let newest = &self.knots[self.count - 1];
        if at >= newest.t {
            let decay = (-dt / RUN_OUT_TAU_S).exp();
            for j in 0..n {
                self.run_out_velocity[j] *= decay;
                out.velocity[j] = self.run_out_velocity[j];
            }
            out.position = newest.q;
            // Past the end of a motion that has stopped, there is nothing to be starved of.
            out.starved = self.count >= 2 && at - newest.t < 1.0;
            return Some(out);
        }
        if at <= self.knots[0].t {
            out.position = self.knots[0].q;
            return Some(out);
        }
        let mut k = self.count - 2;
        while self.knots[k].t > at {
            k -= 1;
        }
        let (a, b) = (&self.knots[k], &self.knots[k + 1]);
        let h = b.t - a.t;
        let u = (at - a.t) / h;
        let before = if k > 0 { Some(&self.knots[k - 1]) } else { None };
        let after = if k + 2 < self.count { Some(&self.knots[k + 2]) } else { None };
        for j in 0..n {
            let chord = (b.q[j] - a.q[j]) / h;
            let m0 = if a.rest { 0.0 } else { before.map_or(chord, |p| (b.q[j] - p.q[j]) / (b.t - p.t)) };
            let m1 = if b.rest { 0.0 } else { after.map_or(chord, |p| (p.q[j] - a.q[j]) / (p.t - a.t)) };
            let (u2, u3) = (u * u, u * u * u);
            out.position[j] = (2.0 * u3 - 3.0 * u2 + 1.0) * a.q[j]
                + (u3 - 2.0 * u2 + u) * h * m0
                + (-2.0 * u3 + 3.0 * u2) * b.q[j]
                + (u3 - u2) * h * m1;
            out.velocity[j] = (1.0 - u) * self.slope(k, j) + u * self.slope(k + 1, j);
            // From the smoothed velocity: the spline's own second derivative is all target noise.
            out.acceleration[j] = (self.slope(k + 1, j) - self.slope(k, j)) / h;
        }
        self.run_out_velocity = out.velocity;
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(x: f64) -> [f64; MAX_MOTORS] {
        let mut q = [0.0; MAX_MOTORS];
        q[0] = x;
        q
    }

    #[test]
    fn a_ramp_plays_at_its_velocity_with_arrival_jitter() {
        let mut player = Player::new(1, 0.01, 0.01, 5);
        let jitter = [0.0, 0.003, -0.002, 0.004, -0.004, 0.001];
        for i in 0..60 {
            player.push(i as f64 * 0.01 + jitter[i % 6], q(i as f64 * 0.01));
        }
        let s = player.sample(0.45).unwrap();
        assert!((s.velocity[0] - 1.0).abs() < 0.02, "{}", s.velocity[0]);
        assert!(s.acceleration[0].abs() < 3.0, "{}", s.acceleration[0]);
        assert!(!s.starved);
    }

    #[test]
    fn a_stale_sample_barely_moves_the_velocity() {
        // A ramp where every fourth target repeats the one before, as a leader's samples do when
        // they beat with the tick. The neighbouring difference would swing by 1.3 rad/s.
        let mut player = Player::new(1, 0.01, 0.01, 4);
        let mut x = 0.0;
        for i in 0..60 {
            if i % 4 != 0 {
                x += 0.01;
            }
            player.push(i as f64 * 0.01, q(x));
        }
        let (mut low, mut high) = (f64::MAX, f64::MIN);
        for i in 0..50 {
            let v = player.sample(0.42 + i as f64 * 0.0025).unwrap().velocity[0];
            low = low.min(v);
            high = high.max(v);
        }
        assert!(high - low < 0.25, "{low}..{high}");
        assert!((0.5 * (high + low) - 0.75).abs() < 0.1, "{low}..{high}");
    }

    #[test]
    fn the_motion_is_continuous_across_knots() {
        let mut player = Player::new(1, 0.01, 0.01, 5);
        for i in 0..50 {
            let t = i as f64 * 0.01;
            player.push(t, q((3.0 * t).sin()));
        }
        let mut last = player.sample(0.2).unwrap();
        for i in 1..100 {
            let s = player.sample(0.2 + i as f64 * 0.0025 * 0.5).unwrap();
            assert!((s.position[0] - last.position[0]).abs() < 0.005);
            assert!((s.velocity[0] - last.velocity[0]).abs() < 0.05);
            last = s;
        }
    }

    #[test]
    fn running_out_holds_the_newest_and_stops_moving() {
        let mut player = Player::new(1, 0.01, 0.01, 5);
        for i in 0..10 {
            player.push(i as f64 * 0.01, q(i as f64 * 0.01));
        }
        player.sample(0.09);
        let s = player.sample(0.2).unwrap();
        assert_eq!(s.position[0], 0.09);
        assert!(s.velocity[0].abs() < 0.01);
        assert!(s.starved);
    }

    #[test]
    fn a_pause_starts_from_where_it_stopped() {
        let mut player = Player::new(1, 0.01, 0.0, 5);
        player.push(0.0, q(0.0));
        player.push(0.01, q(1.0));
        player.push(1.0, q(1.5));
        let s = player.sample(0.5).unwrap();
        assert_eq!(s.position[0], 1.0, "holds through the pause");
        assert!(s.velocity[0].abs() < 0.2, "{}", s.velocity[0]);
    }
}
