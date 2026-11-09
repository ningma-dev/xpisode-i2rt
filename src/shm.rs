//! The shared memory between the controller and whoever gives it targets: 64-bit words, no locks.
//! The writer of each part is single. `xpisode_i2rt.realtime` holds the same layout.

use crate::MAX_MOTORS;
use std::ffi::CString;
use std::io::{Error, Result};
use std::sync::atomic::{fence, AtomicU64, Ordering};

pub const MAGIC: u64 = 0x3145_524f_4354_5249; // "IRTCORE1", little endian
pub const WORDS: usize = 1024;
pub const RING: usize = 64;
const ENTRY: usize = 2 + MAX_MOTORS;

// Words the other side writes.
const GO: usize = 1;
const HOLD: usize = 2;
const STOP: usize = 3;
const HEAD: usize = 4;
// Words the controller writes.
const READY: usize = 8;
const ENCODER_AGE_NS: usize = 15;
const SEQUENCE: usize = 16;
const TIME_NS: usize = 17;
const CYCLES: usize = 18;
const FAULT: usize = 19;
const FAULT_MOTOR: usize = 20;
const FAULT_DETAIL: usize = 21;
const STARVED: usize = 22;
const MISSED: usize = 23;
const DROPPED: usize = 24;
const LATE_MAX_NS: usize = 25;
const SCHED_POLICY: usize = 26;
const SCHED_PRIORITY: usize = 27;
const LOCKED: usize = 28;
const ENCODER_POSITION: usize = 29;
const ENCODER_VELOCITY: usize = 30;
const ENCODER_INPUTS: usize = 31;
const POSITION: usize = 32;
const VELOCITY: usize = POSITION + MAX_MOTORS;
const TORQUE: usize = VELOCITY + MAX_MOTORS;
const TEMPERATURE: usize = TORQUE + MAX_MOTORS;
const TARGET_POSITION: usize = TEMPERATURE + MAX_MOTORS;
const TARGET_VELOCITY: usize = TARGET_POSITION + MAX_MOTORS;
const FEEDFORWARD: usize = TARGET_VELOCITY + MAX_MOTORS;
const RING_START: usize = 128;

#[derive(Clone, Copy, Default)]
pub struct Published {
    pub time_ns: u64,
    pub cycles: u64,
    pub fault: u64,
    pub fault_motor: i64,
    pub fault_detail: u64,
    pub starved: u64,
    pub missed: u64,
    pub dropped: u64,
    pub late_max_ns: u64,
    pub encoder_position: f64,
    pub encoder_velocity: f64,
    pub encoder_inputs: u64,
    pub encoder_age_ns: u64,
    pub position: [f64; MAX_MOTORS],
    pub velocity: [f64; MAX_MOTORS],
    pub torque: [f64; MAX_MOTORS],
    pub temperature: [f64; MAX_MOTORS],
    pub target_position: [f64; MAX_MOTORS],
    pub target_velocity: [f64; MAX_MOTORS],
    pub feedforward: [f64; MAX_MOTORS],
}

pub struct Shared {
    words: *const AtomicU64,
    read: u64,
}

impl Shared {
    pub fn open(path: &str) -> Result<Shared> {
        let name = CString::new(path)?;
        unsafe {
            let fd = libc::open(name.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
            if fd < 0 {
                return Err(Error::last_os_error());
            }
            let map = libc::mmap(
                std::ptr::null_mut(),
                WORDS * 8,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);
            if map == libc::MAP_FAILED {
                return Err(Error::last_os_error());
            }
            Ok(Shared { words: map as *const AtomicU64, read: 0 })
        }
    }

    fn word(&self, i: usize) -> &AtomicU64 {
        debug_assert!(i < WORDS);
        unsafe { &*self.words.add(i) }
    }

    fn store(&self, i: usize, v: u64) {
        self.word(i).store(v, Ordering::Relaxed);
    }

    fn store_all(&self, at: usize, values: &[f64; MAX_MOTORS]) {
        for (j, v) in values.iter().enumerate() {
            self.store(at + j, v.to_bits());
        }
    }

    pub fn set_ready(&self) {
        self.word(READY).store(MAGIC, Ordering::Release);
    }

    pub fn set_schedule(&self, policy: i32, priority: i32, locked: bool) {
        self.store(SCHED_POLICY, policy as u64);
        self.store(SCHED_PRIORITY, priority as u64);
        self.store(LOCKED, locked as u64);
    }

    pub fn go(&self) -> u64 {
        self.word(GO).load(Ordering::Acquire)
    }
    pub fn hold(&self) -> u64 {
        self.word(HOLD).load(Ordering::Acquire)
    }
    pub fn stop(&self) -> u64 {
        self.word(STOP).load(Ordering::Acquire)
    }

    /// The next target not read yet, oldest first; a torn or overrun one is skipped.
    pub fn next_target(&mut self) -> Option<[f64; MAX_MOTORS]> {
        let head = self.word(HEAD).load(Ordering::Acquire);
        if head.wrapping_sub(self.read) > RING as u64 {
            self.read = head - RING as u64;
        }
        while self.read != head {
            let at = RING_START + (self.read as usize % RING) * ENTRY;
            let sequence = self.word(at).load(Ordering::Acquire);
            let mut q = [0.0; MAX_MOTORS];
            for (j, v) in q.iter_mut().enumerate() {
                *v = f64::from_bits(self.word(at + 2 + j).load(Ordering::Relaxed));
            }
            fence(Ordering::Acquire);
            let wanted = self.read + 1;
            let intact = sequence == wanted && self.word(at).load(Ordering::Relaxed) == sequence;
            self.read += 1;
            if intact {
                return Some(q);
            }
        }
        None
    }

    pub fn publish(&self, p: &Published) {
        let sequence = self.word(SEQUENCE).load(Ordering::Relaxed).wrapping_add(1);
        self.word(SEQUENCE).store(sequence, Ordering::Relaxed);
        fence(Ordering::Release);
        self.store(TIME_NS, p.time_ns);
        self.store(CYCLES, p.cycles);
        self.store(FAULT, p.fault);
        self.store(FAULT_MOTOR, p.fault_motor as u64);
        self.store(FAULT_DETAIL, p.fault_detail);
        self.store(STARVED, p.starved);
        self.store(MISSED, p.missed);
        self.store(DROPPED, p.dropped);
        self.store(LATE_MAX_NS, p.late_max_ns);
        self.store(ENCODER_POSITION, p.encoder_position.to_bits());
        self.store(ENCODER_VELOCITY, p.encoder_velocity.to_bits());
        self.store(ENCODER_INPUTS, p.encoder_inputs);
        self.store(ENCODER_AGE_NS, p.encoder_age_ns);
        self.store_all(POSITION, &p.position);
        self.store_all(VELOCITY, &p.velocity);
        self.store_all(TORQUE, &p.torque);
        self.store_all(TEMPERATURE, &p.temperature);
        self.store_all(TARGET_POSITION, &p.target_position);
        self.store_all(TARGET_VELOCITY, &p.target_velocity);
        self.store_all(FEEDFORWARD, &p.feedforward);
        self.word(SEQUENCE).store(sequence.wrapping_add(1), Ordering::Release);
    }
}
