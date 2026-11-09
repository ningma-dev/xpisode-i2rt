//! Damiao motors' MIT frames: the command a motor takes and the reply it gives.

/// The ranges a motor type maps its integers to; they must match its PMAX, VMAX and TMAX registers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ranges {
    pub position: f64,
    pub velocity: f64,
    pub torque: f64,
}

pub const DM4340: Ranges = Ranges { position: 12.5, velocity: 10.0, torque: 28.0 };
pub const DM4310: Ranges = Ranges { position: 12.5, velocity: 30.0, torque: 10.0 };
const KP_MAX: f64 = 500.0;
const KD_MAX: f64 = 5.0;

/// Replies come from the motor's id plus this.
pub const REPLY_OFFSET: u32 = 16;

/// The state in the high nibble of a reply's first byte.
pub const STATE_DISABLED: u8 = 0;
pub const STATE_ENABLED: u8 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Command {
    pub position: f64,
    pub velocity: f64,
    pub kp: f64,
    pub kd: f64,
    pub torque: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reply {
    pub state: u8,
    pub position: f64,
    pub velocity: f64,
    pub torque: f64,
    pub temperature_mos: u8,
    pub temperature_rotor: u8,
}

fn to_uint(x: f64, min: f64, max: f64, bits: u32) -> u32 {
    let scale = ((1u32 << bits) - 1) as f64;
    ((x.clamp(min, max) - min) * scale / (max - min)) as u32
}

fn to_float(x: u32, min: f64, max: f64, bits: u32) -> f64 {
    let scale = ((1u32 << bits) - 1) as f64;
    x as f64 * (max - min) / scale + min
}

pub fn encode(command: &Command, r: &Ranges) -> [u8; 8] {
    let p = to_uint(command.position, -r.position, r.position, 16);
    let v = to_uint(command.velocity, -r.velocity, r.velocity, 12);
    let kp = to_uint(command.kp, 0.0, KP_MAX, 12);
    let kd = to_uint(command.kd, 0.0, KD_MAX, 12);
    let t = to_uint(command.torque, -r.torque, r.torque, 12);
    [
        (p >> 8) as u8,
        p as u8,
        (v >> 4) as u8,
        (((v & 0xF) << 4) | (kp >> 8)) as u8,
        kp as u8,
        (kd >> 4) as u8,
        (((kd & 0xF) << 4) | (t >> 8)) as u8,
        t as u8,
    ]
}

pub fn decode(data: &[u8; 8], r: &Ranges) -> Reply {
    let p = ((data[1] as u32) << 8) | data[2] as u32;
    let v = ((data[3] as u32) << 4) | (data[4] as u32 >> 4);
    let t = ((data[4] as u32 & 0xF) << 8) | data[5] as u32;
    Reply {
        state: data[0] >> 4,
        position: to_float(p, -r.position, r.position, 16),
        velocity: to_float(v, -r.velocity, r.velocity, 12),
        torque: to_float(t, -r.torque, r.torque, 12),
        temperature_mos: data[6],
        temperature_rotor: data[7],
    }
}

/// What makes a passive encoder report, and the id it reports from.
pub const ENCODER_TRIGGER: [u8; 2] = [0xFF, 0x02];
pub const ENCODER_REPLY_OFFSET: u32 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EncoderReply {
    /// Radians from the encoder's center.
    pub position: f64,
    pub velocity: f64,
    /// One bit an input.
    pub inputs: u8,
}

/// Device number, position and velocity in 1/4096 of a turn, inputs; big endian.
pub fn decode_encoder(data: &[u8; 8]) -> EncoderReply {
    let turn = 2.0 * std::f64::consts::PI / 4096.0;
    EncoderReply {
        position: i16::from_be_bytes([data[1], data[2]]) as f64 * turn,
        velocity: i16::from_be_bytes([data[3], data[4]]) as f64 * turn,
        inputs: data[5],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_what_i2rt_encodes() {
        // From i2rt's DMSingleMotorCanInterface.set_control, DM4340.
        let command = Command { position: 1.0, velocity: -2.0, kp: 80.0, kd: 5.0, torque: 3.0 };
        let c = |x: f64, min: f64, max: f64, bits: u32| to_uint(x, min, max, bits);
        let p = c(1.0, -12.5, 12.5, 16);
        let v = c(-2.0, -10.0, 10.0, 12);
        let kp = c(80.0, 0.0, 500.0, 12);
        let kd = c(5.0, 0.0, 5.0, 12);
        let t = c(3.0, -28.0, 28.0, 12);
        assert_eq!((p, v, kp, kd, t), (35388, 1638, 655, 4095, 2266));
        assert_eq!(
            encode(&command, &DM4340),
            [0x8A, 0x3C, 0x66, 0x62, 0x8F, 0xFF, 0xF8, 0xDA]
        );
    }

    #[test]
    fn decodes_a_reply() {
        let reply = decode(&[0x11, 0x80, 0x00, 0x80, 0x08, 0x00, 30, 31], &DM4310);
        assert_eq!(reply.state, STATE_ENABLED);
        assert!(reply.position.abs() < 0.001);
        assert!(reply.velocity.abs() < 0.02);
        assert!(reply.torque.abs() < 0.01);
        assert_eq!((reply.temperature_mos, reply.temperature_rotor), (30, 31));
    }

    #[test]
    fn decodes_an_encoder_report() {
        // 512 and -256 in 1/4096 of a turn, both inputs on.
        let reply = decode_encoder(&[0x0E, 0x02, 0x00, 0xFF, 0x00, 0x03, 0, 0]);
        assert!((reply.position - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((reply.velocity + std::f64::consts::FRAC_PI_8).abs() < 1e-12);
        assert_eq!(reply.inputs, 3);
    }

    #[test]
    fn clamps_to_the_range() {
        let high = encode(&Command { torque: 1e9, ..Default::default() }, &DM4310);
        let low = encode(&Command { torque: -1e9, ..Default::default() }, &DM4310);
        assert_eq!(high[7], 0xFF);
        assert_eq!(high[6] & 0xF, 0xF);
        assert_eq!((low[7], low[6] & 0xF), (0, 0));
    }
}
