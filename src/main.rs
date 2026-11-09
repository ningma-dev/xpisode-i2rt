//! `i2rt-rtcore config.json`: one bus's controller. It holds what it is told from the first frame on.

use xpisode_i2rt::can::{add_ns, diff_ns, monotonic, Bus, Frame};
use xpisode_i2rt::config::Config;
use xpisode_i2rt::control::{Controller, Fault};
use xpisode_i2rt::dm::{
    decode, decode_encoder, encode, Ranges, DM4310, DM4340, ENCODER_REPLY_OFFSET, ENCODER_TRIGGER,
    REPLY_OFFSET, STATE_ENABLED,
};
use xpisode_i2rt::shm::{Published, Shared};
use xpisode_i2rt::MAX_MOTORS;

/// A longer silence of the encoder goes to stderr, where the operator can see it.
const ENCODER_GAP_S: f64 = 0.05;

fn seconds(t: &libc::timespec) -> f64 {
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
}

/// What it could get is what it records: without the permission it warns and goes on.
fn schedule(config: &Config) -> (i32, i32, bool) {
    unsafe {
        let locked = libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) == 0;
        if !locked {
            eprintln!("i2rt-rtcore: memory not locked: {}", std::io::Error::last_os_error());
        }
        if config.cpu >= 0 {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(config.cpu as usize, &mut set);
            if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
                eprintln!("i2rt-rtcore: not pinned: {}", std::io::Error::last_os_error());
            }
        }
        if config.fifo_priority > 0 {
            let param = libc::sched_param { sched_priority: config.fifo_priority };
            if libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) != 0 {
                eprintln!("i2rt-rtcore: no SCHED_FIFO: {}", std::io::Error::last_os_error());
            }
        }
        let mut param: libc::sched_param = std::mem::zeroed();
        libc::sched_getparam(0, &mut param);
        (libc::sched_getscheduler(0), param.sched_priority, locked)
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: i2rt-rtcore config.json");
    let text = std::fs::read_to_string(&path).expect("the config");
    let config: Config = serde_json::from_str(&text).expect("a valid config");
    let n = config.motors.len();
    let ranges: Vec<Ranges> = config
        .motors
        .iter()
        .map(|m| match m.kind.as_str() {
            "DM4340" => DM4340,
            "DM4310" => DM4310,
            other => panic!("no ranges for motor {other}"),
        })
        .collect();
    let ids: Vec<u32> = config.motors.iter().map(|m| m.id).collect();
    let channel = config.channel.clone();
    let bus = Bus::open(&channel).expect("the bus");
    let mut shared = Shared::open(&config.shm_path).expect("the shared memory");
    let (policy, priority, locked) = schedule(&config);
    shared.set_schedule(policy, priority, locked);
    let parent = unsafe { libc::getppid() };
    let shm_path = config.shm_path.clone();
    let encoder_id = config.encoder_id;
    let encoder_every = config.encoder_every.max(1);
    let slots = n + encoder_id.is_some() as usize;
    let (dry, orphan_hold_s, slot_ns) = (
        config.dry,
        config.orphan_hold_s,
        (1e9 / config.rate_hz / slots as f64) as i64,
    );
    let cycle_ns = slot_ns * slots as i64;
    shared.set_ready();

    // The first target is the pose to hold; until then nothing goes on the bus.
    let idle = add_ns(&monotonic(), 0);
    let mut waited = idle;
    while shared.go() == 0 {
        if shared.stop() != 0 {
            return;
        }
        waited = add_ns(&waited, 1_000_000);
        unsafe {
            libc::clock_nanosleep(libc::CLOCK_MONOTONIC, libc::TIMER_ABSTIME, &waited, std::ptr::null_mut());
        }
        if unsafe { libc::getppid() } != parent {
            return;
        }
    }
    let start = monotonic();
    let mut controller = Controller::new(config, seconds(&start));
    controller.hold_at(shared.next_target().unwrap_or([0.0; MAX_MOTORS]));

    let mut published = Published::default();
    let mut slot = 0usize;
    let mut next = monotonic();
    let mut hold_seen = shared.hold();
    let mut orphaned_at: Option<f64> = None;
    let (mut sent, mut replies) = (0u64, 0u64);
    loop {
        let begin = monotonic();
        let now = seconds(&begin);
        published.late_max_ns = published.late_max_ns.max(diff_ns(&begin, &next).max(0) as u64);
        if slot == 0 {
            if shared.stop() != 0 {
                break;
            }
            if shared.hold() != hold_seen {
                hold_seen = shared.hold();
                while shared.next_target().is_some() {}
                controller.hold();
            }
            while let Some(q) = shared.next_target() {
                controller.push(now, q);
            }
            if unsafe { libc::getppid() } != parent {
                let since = *orphaned_at.get_or_insert_with(|| {
                    eprintln!("i2rt-rtcore: the parent died; holding");
                    controller.hold();
                    now
                });
                if now - since > orphan_hold_s {
                    eprintln!("i2rt-rtcore: letting go");
                    let _ = std::fs::remove_file(&shm_path);
                    break;
                }
            }
            // Frames on their way are not lost; a few short of the total is normal.
            published.missed = sent.saturating_sub(replies);
            controller.cycle(now);
            published.time_ns = (begin.tv_sec as u64) * 1_000_000_000 + begin.tv_nsec as u64;
            published.cycles += 1;
            published.fault = controller.fault as u64;
            published.fault_motor = controller.fault_motor as i64;
            published.fault_detail = controller.fault_detail as u64;
            published.starved = controller.starved_cycles;
            published.encoder_position = controller.encoder.position;
            published.encoder_velocity = controller.encoder.velocity;
            published.encoder_inputs = controller.encoder.inputs as u64;
            published.encoder_age_ns = ((now - controller.encoder_time).max(0.0) * 1e9) as u64;
            published.position = controller.position;
            published.velocity = controller.velocity;
            published.torque = controller.torque;
            published.temperature = controller.temperature;
            published.target_position = controller.target_position;
            published.target_velocity = controller.target_velocity;
            published.feedforward = controller.feedforward;
            shared.publish(&published);
        }

        let frame = if slot < n {
            Some(Frame { id: ids[slot], len: 8, data: encode(&controller.command(slot), &ranges[slot]) })
        } else if published.cycles % encoder_every == 0 {
            let mut data = [0u8; 8];
            data[..2].copy_from_slice(&ENCODER_TRIGGER);
            Some(Frame { id: encoder_id.unwrap_or(0), len: 2, data })
        } else {
            None
        };
        if let Some(frame) = frame {
            match bus.send(&frame) {
                Ok(true) => sent += 1,
                Ok(false) => published.dropped += 1,
                Err(_) => controller.raise(Fault::Bus, slot as i32, 0),
            }
        }

        next = add_ns(&next, slot_ns);
        // A cycle lost to a stall is not caught up: frames would only pile up in the adapter.
        if diff_ns(&begin, &next) > cycle_ns {
            next = add_ns(&begin, slot_ns);
        }
        loop {
            match bus.receive_until(&next) {
                Ok(Some(frame)) => {
                    if encoder_id.map(|e| e + ENCODER_REPLY_OFFSET) == Some(frame.id) {
                        replies += 1;
                        let at = seconds(&monotonic());
                        let silent = at - controller.encoder_time;
                        if silent > ENCODER_GAP_S {
                            eprintln!("i2rt-rtcore: {channel}: the encoder was silent for {:.0} ms", silent * 1e3);
                        }
                        controller.on_encoder(decode_encoder(&frame.data), at);
                        continue;
                    }
                    let id = frame.id.wrapping_sub(REPLY_OFFSET);
                    if let Some(i) = ids.iter().position(|m| *m == id) {
                        let reply = decode(&frame.data, &ranges[i]);
                        if dry && reply.state == STATE_ENABLED {
                            eprintln!("i2rt-rtcore: motor {id} is enabled on a dry run: stopping");
                            std::process::exit(2);
                        }
                        controller.on_reply(i, &reply, seconds(&monotonic()));
                        replies += 1;
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    controller.raise(Fault::Bus, slot as i32, 1);
                    break;
                }
            }
        }
        slot = (slot + 1) % slots;
    }
}
