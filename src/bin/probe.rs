//! How fast a bus exchanges real MIT frames. No torque: every gain and torque is zero and the
//! motors are never enabled. It stops at the first reply from an enabled motor.
//!
//!     probe can_follower_r 7 100,250,333,400,500 3

use xpisode_i2rt::can::{add_ns, diff_ns, monotonic, Bus, Frame};
use xpisode_i2rt::dm::{encode, Command, DM4310, REPLY_OFFSET, STATE_ENABLED};

fn percentile(values: &mut [i64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_unstable();
    values[((values.len() - 1) as f64 * p) as usize] as f64 / 1e3
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let channel = &args[1];
    let motors: u32 = args[2].parse().unwrap();
    let rates: Vec<f64> = args[3].split(',').map(|r| r.parse().unwrap()).collect();
    let seconds: f64 = args[4].parse().unwrap();
    let bus = Bus::open(channel).unwrap();
    let data = encode(&Command::default(), &DM4310);
    println!("{channel} hz sent dropped replies complete% motor_hz rtt_p50_us rtt_p99_us late_p99_us late_max_us");
    for &hz in &rates {
        let period = (1e9 / hz) as i64;
        let cycles = (seconds * hz) as usize;
        let (mut sent, mut dropped, mut replies, mut complete) = (0usize, 0usize, 0usize, 0usize);
        let (mut rtts, mut lates) = (Vec::with_capacity(cycles), Vec::with_capacity(cycles));
        let mut next = add_ns(&monotonic(), period);
        // Replies to a cycle count until the next cycle starts; a late one counts for the next.
        for _ in 0..cycles {
            let start = monotonic();
            lates.push(diff_ns(&start, &add_ns(&next, -period)).max(0));
            for id in 1..=motors {
                if bus.send(&Frame { id, len: 8, data }).unwrap() {
                    sent += 1;
                } else {
                    dropped += 1;
                }
            }
            let mut got = 0u32;
            while let Some(frame) = bus.receive_until(&next).unwrap() {
                if frame.id <= REPLY_OFFSET || frame.id > REPLY_OFFSET + motors {
                    continue;
                }
                if frame.data[0] >> 4 == STATE_ENABLED {
                    eprintln!("motor {} is enabled: stopping", frame.id - REPLY_OFFSET);
                    std::process::exit(2);
                }
                replies += 1;
                got += 1;
                if got == motors {
                    rtts.push(diff_ns(&monotonic(), &start));
                    complete += 1;
                }
            }
            unsafe {
                libc::clock_nanosleep(libc::CLOCK_MONOTONIC, libc::TIMER_ABSTIME, &next, std::ptr::null_mut());
            }
            next = add_ns(&next, period);
        }
        println!(
            "{channel} {hz:.0} {sent} {dropped} {replies} {:.1} {:.0} {:.0} {:.0} {:.0} {:.0}",
            100.0 * complete as f64 / cycles as f64,
            replies as f64 / motors as f64 / seconds,
            percentile(&mut rtts, 0.5),
            percentile(&mut rtts, 0.99),
            percentile(&mut lates, 0.99),
            percentile(&mut lates, 1.0),
        );
    }
}
