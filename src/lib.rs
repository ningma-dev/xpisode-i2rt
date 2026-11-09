//! A real-time joint controller for I2RT YAM arms over SocketCAN.
//!
//! One process owns one bus. It sends each motor's MIT frame in a slot of its own, so that the
//! adapter's queue never fills, and plays the targets it is given with velocity, acceleration and
//! torque feedforward. See README.md.

pub mod can;
pub mod config;
pub mod control;
pub mod dm;
pub mod dynamics;
pub mod player;
pub mod shm;

/// The most motors one bus carries.
pub const MAX_MOTORS: usize = 8;
