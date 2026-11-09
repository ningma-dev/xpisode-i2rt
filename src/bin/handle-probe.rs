//! How a teaching handle's encoder answers beside six motors. No torque, as `probe`.
//!
//!     handle-probe can_leader_r 400 1 30    (rate, a poll every n cycles, seconds)

use xpisode_i2rt::can::{add_ns, diff_ns, monotonic, Bus, Frame};
use xpisode_i2rt::dm::{encode, Command, DM4310, ENCODER_TRIGGER, REPLY_OFFSET, STATE_ENABLED};

const ENCODER: u32 = 0x50E;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let hz: f64 = args[2].parse().unwrap();
    let every: usize = args[3].parse().unwrap();
    let seconds: f64 = args[4].parse().unwrap();
    let bus = Bus::open(&args[1]).unwrap();
    let data = encode(&Command::default(), &DM4310);
    let slot = (1e9 / hz / 7.0) as i64;
    let total = (seconds * hz) as usize * 7;
    let (mut polls, mut answers, mut motors) = (0usize, 0usize, 0usize);
    let start = monotonic();
    let (mut last, mut gaps) = (start, Vec::new());
    let mut next = start;
    for n in 0..total {
        let s = n % 7;
        if s < 6 {
            bus.send(&Frame { id: s as u32 + 1, len: 8, data }).unwrap();
        } else if (n / 7) % every == 0 {
            let mut d = [0u8; 8];
            d[..2].copy_from_slice(&ENCODER_TRIGGER);
            polls += bus.send(&Frame { id: ENCODER, len: 2, data: d }).unwrap() as usize;
        }
        next = add_ns(&next, slot);
        while let Some(frame) = bus.receive_until(&next).unwrap() {
            if frame.id == ENCODER + 1 {
                let now = monotonic();
                gaps.push(diff_ns(&now, &last));
                last = now;
                answers += 1;
            } else if frame.id > REPLY_OFFSET && frame.id <= REPLY_OFFSET + 6 {
                if frame.data[0] >> 4 == STATE_ENABLED {
                    eprintln!("motor {} is enabled: stopping", frame.id - REPLY_OFFSET);
                    std::process::exit(2);
                }
                motors += 1;
            }
        }
    }
    gaps.push(diff_ns(&monotonic(), &last));
    gaps.sort_unstable();
    let at = |p: f64| gaps[((gaps.len() - 1) as f64 * p) as usize] as f64 / 1e6;
    println!(
        "{} {hz:.0}Hz every {every}: polls {polls} answers {answers} ({:.1}%) motor replies {motors}; gap ms p50 {:.1} p99 {:.1} p99.9 {:.1} max {:.1}; over 20ms {} over 50ms {}",
        args[1], 100.0 * answers as f64 / polls.max(1) as f64, at(0.5), at(0.99), at(0.999), at(1.0),
        gaps.iter().filter(|g| **g > 20_000_000).count(), gaps.iter().filter(|g| **g > 50_000_000).count()
    );
}
