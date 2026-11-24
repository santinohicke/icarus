//! Deterministic replays: seed + vehicle config + per-tick quantized inputs +
//! injected events.
//!
//! A replay is enough to re-simulate a run exactly, so a shared replay proves
//! its own score. Replays are untrusted input (they arrive in URLs), so the
//! decoder validates everything, caps every size before allocating, and never
//! panics.
//!
//! Binary layout (little-endian), then base64url without padding:
//!
//! ```text
//! magic "ICRP" | format u8 | physics_version u16 | config u8 | seed u64
//! | ticks u32 | run_count u32 | runs: (length LEB128-u32, q_throttle, q_gimbal, q_rcs)*
//! | event_count u16 | events: (tick u32, kind u8, a f64, b f64)*
//! ```

use core::fmt;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;

use crate::PHYSICS_VERSION;
use crate::physics::params::VehicleConfig;
use crate::physics::scenario::Outcome;
use crate::physics::state::{Controls, QUANT_MAX};
use crate::sim::{Injection, Sim};

/// Version of the binary replay format.
pub const REPLAY_FORMAT_VERSION: u8 = 1;

/// Largest number of ticks a replay may contain.
pub const MAX_REPLAY_TICKS: u32 = 1 << 16;

/// Largest number of events a replay may contain.
pub const MAX_REPLAY_EVENTS: usize = 256;

/// Longest encoded replay accepted (characters).
pub const MAX_ENCODED_LEN: usize = 256 * 1024;

const MAGIC: &[u8; 4] = b"ICRP";

/// An event recorded at a tick (applied before that tick's step).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayEvent {
    /// Tick before which the event is applied.
    pub tick: u32,
    /// The event.
    pub injection: Injection,
}

/// A recorded run.
#[derive(Clone, Debug, PartialEq)]
pub struct Replay {
    /// Vehicle configuration.
    pub config: VehicleConfig,
    /// Episode seed.
    pub seed: u64,
    /// Quantized `[throttle, gimbal, rcs]` per tick.
    pub inputs: Vec<[u8; 3]>,
    /// Injected events, in tick order.
    pub events: Vec<ReplayEvent>,
}

/// Why a replay could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    /// The encoded string is longer than [`MAX_ENCODED_LEN`].
    TooLong,
    /// Not valid base64url.
    Base64,
    /// Not a replay (wrong magic bytes).
    BadMagic,
    /// Unknown replay format version.
    UnsupportedFormat(u8),
    /// Recorded with a different physics version.
    PhysicsVersion {
        /// Version in the replay.
        found: u16,
        /// Version of this build.
        expected: u16,
    },
    /// Unknown vehicle configuration identifier.
    UnknownConfig(u8),
    /// The data ended early.
    Truncated,
    /// Extra bytes after the end of the replay.
    TrailingBytes,
    /// More ticks or events than allowed, or a malformed length.
    Limits,
    /// Run lengths do not add up to the tick count.
    InconsistentLength,
    /// A quantized input level above 254.
    BadInput,
    /// An event with an unknown kind, a non-finite value or out of order.
    BadEvent,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplayError::TooLong => write!(f, "replay is too long"),
            ReplayError::Base64 => write!(f, "replay is not valid base64url"),
            ReplayError::BadMagic => write!(f, "not an Icarus replay"),
            ReplayError::UnsupportedFormat(v) => write!(f, "unsupported replay format version {v}"),
            ReplayError::PhysicsVersion { found, expected } => {
                write!(
                    f,
                    "replay recorded with physics version {found}, this build uses {expected}"
                )
            }
            ReplayError::UnknownConfig(c) => write!(f, "unknown vehicle configuration {c}"),
            ReplayError::Truncated => write!(f, "replay data ends early"),
            ReplayError::TrailingBytes => write!(f, "unexpected data after the replay"),
            ReplayError::Limits => write!(f, "replay exceeds size limits"),
            ReplayError::InconsistentLength => {
                write!(f, "replay input runs do not match the tick count")
            }
            ReplayError::BadInput => write!(f, "replay contains an invalid input level"),
            ReplayError::BadEvent => write!(f, "replay contains an invalid event"),
        }
    }
}

impl std::error::Error for ReplayError {}

fn injection_parts(i: &Injection) -> (u8, f64, f64) {
    match *i {
        Injection::GimbalStuck { angle } => (1, angle, 0.0),
        Injection::ThrustLoss { factor } => (2, factor, 0.0),
        Injection::Gust {
            amplitude,
            duration,
        } => (3, amplitude, duration),
    }
}

fn injection_from_parts(kind: u8, a: f64, b: f64) -> Option<Injection> {
    if !(a.is_finite() && b.is_finite()) {
        return None;
    }
    // Kinds with a single value must carry an exact +0.0 in the unused slot
    // (canonical encoding).
    match kind {
        1 if b.to_bits() == 0 => Some(Injection::GimbalStuck { angle: a }),
        2 if b.to_bits() == 0 => Some(Injection::ThrustLoss { factor: a }),
        3 => Some(Injection::Gust {
            amplitude: a,
            duration: b,
        }),
        _ => None,
    }
}

/// Bounds-checked little-endian reader.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ReplayError> {
        let end = self.pos.checked_add(n).ok_or(ReplayError::Truncated)?;
        let slice = self.data.get(self.pos..end).ok_or(ReplayError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], ReplayError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, ReplayError> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, ReplayError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, ReplayError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, ReplayError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn f64(&mut self) -> Result<f64, ReplayError> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    /// Unsigned LEB128, at most 5 bytes, must fit in a `u32`, and must be the
    /// minimal encoding (no trailing zero groups), so every value has exactly
    /// one byte representation.
    fn varint(&mut self) -> Result<u32, ReplayError> {
        let mut value: u64 = 0;
        for i in 0..5 {
            let byte = self.u8()?;
            value |= u64::from(byte & 0x7F) << (7 * i);
            if byte & 0x80 == 0 {
                if byte == 0 && i > 0 {
                    return Err(ReplayError::Limits);
                }
                return u32::try_from(value).map_err(|_| ReplayError::Limits);
            }
        }
        Err(ReplayError::Limits)
    }

    fn is_done(&self) -> bool {
        self.pos == self.data.len()
    }
}

fn write_varint(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

impl Replay {
    /// An empty replay.
    pub fn new(config: VehicleConfig, seed: u64) -> Self {
        Self {
            config,
            seed,
            inputs: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Largest tick count accepted for this replay's vehicle configuration.
    fn max_ticks(&self) -> u32 {
        let episode = self.config.params().max_ticks();
        u32::try_from(episode)
            .unwrap_or(u32::MAX)
            .min(MAX_REPLAY_TICKS)
    }

    /// Checks the limits the decoder enforces, so that `to_bytes` never
    /// produces a replay that `from_bytes` would reject.
    pub fn validate(&self) -> Result<(), ReplayError> {
        if self.inputs.len() > self.max_ticks() as usize || self.events.len() > MAX_REPLAY_EVENTS {
            return Err(ReplayError::Limits);
        }
        if self.inputs.iter().flatten().any(|&v| v > QUANT_MAX) {
            return Err(ReplayError::BadInput);
        }
        let mut last_tick = 0;
        for e in &self.events {
            let (kind, a, b) = injection_parts(&e.injection);
            if e.tick < last_tick
                || e.tick as usize > self.inputs.len()
                || injection_from_parts(kind, a, b).is_none()
            {
                return Err(ReplayError::BadEvent);
            }
            last_tick = e.tick;
        }
        Ok(())
    }

    /// Serializes to the binary format. Fails if the replay exceeds the
    /// limits the decoder accepts (see [`Replay::validate`]).
    pub fn to_bytes(&self) -> Result<Vec<u8>, ReplayError> {
        self.validate()?;
        let mut out = Vec::with_capacity(64 + self.inputs.len() / 4);
        out.extend_from_slice(MAGIC);
        out.push(REPLAY_FORMAT_VERSION);
        out.extend_from_slice(&PHYSICS_VERSION.to_le_bytes());
        out.push(self.config.id());
        out.extend_from_slice(&self.seed.to_le_bytes());
        // Lengths below are bounded by `validate`, so the conversions succeed.
        let ticks = u32::try_from(self.inputs.len()).map_err(|_| ReplayError::Limits)?;
        out.extend_from_slice(&ticks.to_le_bytes());

        // Run-length encode the inputs.
        let mut runs: Vec<(u32, [u8; 3])> = Vec::new();
        for &q in &self.inputs {
            match runs.last_mut() {
                Some((n, last)) if *last == q => *n += 1,
                _ => runs.push((1, q)),
            }
        }
        let run_count = u32::try_from(runs.len()).map_err(|_| ReplayError::Limits)?;
        out.extend_from_slice(&run_count.to_le_bytes());
        for (n, q) in runs {
            write_varint(&mut out, n);
            out.extend_from_slice(&q);
        }

        let event_count = u16::try_from(self.events.len()).map_err(|_| ReplayError::Limits)?;
        out.extend_from_slice(&event_count.to_le_bytes());
        for e in &self.events {
            let (kind, a, b) = injection_parts(&e.injection);
            out.extend_from_slice(&e.tick.to_le_bytes());
            out.push(kind);
            out.extend_from_slice(&a.to_le_bytes());
            out.extend_from_slice(&b.to_le_bytes());
        }
        Ok(out)
    }

    /// Parses and validates the binary format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ReplayError> {
        let mut r = Reader { data, pos: 0 };
        if r.take(4)? != MAGIC {
            return Err(ReplayError::BadMagic);
        }
        let format = r.u8()?;
        if format != REPLAY_FORMAT_VERSION {
            return Err(ReplayError::UnsupportedFormat(format));
        }
        let physics = r.u16()?;
        if physics != PHYSICS_VERSION {
            return Err(ReplayError::PhysicsVersion {
                found: physics,
                expected: PHYSICS_VERSION,
            });
        }
        let config_id = r.u8()?;
        let config =
            VehicleConfig::from_id(config_id).ok_or(ReplayError::UnknownConfig(config_id))?;
        let seed = r.u64()?;
        let ticks = r.u32()?;
        let mut replay = Self::new(config, seed);
        if ticks > replay.max_ticks() {
            return Err(ReplayError::Limits);
        }
        let run_count = r.u32()?;
        if run_count > ticks {
            return Err(ReplayError::InconsistentLength);
        }
        // Safe to allocate: bounded by MAX_REPLAY_TICKS.
        let mut inputs = Vec::with_capacity(ticks as usize);
        let mut previous: Option<[u8; 3]> = None;
        for _ in 0..run_count {
            let n = r.varint()?;
            let q: [u8; 3] = r.array()?;
            if q.iter().any(|&v| v > QUANT_MAX) {
                return Err(ReplayError::BadInput);
            }
            if previous == Some(q) {
                // Adjacent equal runs are not canonical: reject them so every
                // run has exactly one encoding.
                return Err(ReplayError::InconsistentLength);
            }
            previous = Some(q);
            let total = u32::try_from(inputs.len()).map_err(|_| ReplayError::Limits)?;
            if n == 0 || total.checked_add(n).is_none_or(|t| t > ticks) {
                return Err(ReplayError::InconsistentLength);
            }
            inputs.extend(core::iter::repeat_n(q, n as usize));
        }
        if inputs.len() != ticks as usize {
            return Err(ReplayError::InconsistentLength);
        }

        let event_count = usize::from(r.u16()?);
        if event_count > MAX_REPLAY_EVENTS {
            return Err(ReplayError::Limits);
        }
        let mut events = Vec::with_capacity(event_count);
        let mut last_tick = 0;
        for _ in 0..event_count {
            let tick = r.u32()?;
            let kind = r.u8()?;
            let (a, b) = (r.f64()?, r.f64()?);
            let injection = injection_from_parts(kind, a, b).ok_or(ReplayError::BadEvent)?;
            if tick < last_tick || tick > ticks {
                return Err(ReplayError::BadEvent);
            }
            last_tick = tick;
            events.push(ReplayEvent { tick, injection });
        }
        if !r.is_done() {
            return Err(ReplayError::TrailingBytes);
        }
        replay.inputs = inputs;
        replay.events = events;
        Ok(replay)
    }

    /// Encodes as URL-safe base64 without padding.
    pub fn encode(&self) -> Result<String, ReplayError> {
        Ok(URL_SAFE_NO_PAD.encode(self.to_bytes()?))
    }

    /// Decodes a string produced by [`Replay::encode`]. Leading and trailing
    /// whitespace is ignored.
    pub fn decode(text: &str) -> Result<Self, ReplayError> {
        let text = text.trim();
        if text.len() > MAX_ENCODED_LEN {
            return Err(ReplayError::TooLong);
        }
        let bytes = URL_SAFE_NO_PAD
            .decode(text)
            .map_err(|_| ReplayError::Base64)?;
        Self::from_bytes(&bytes)
    }

    /// Re-simulates the replay and returns the finished (or interrupted)
    /// simulation.
    pub fn run(&self) -> Sim {
        let mut sim = Sim::from_config(self.config, self.seed);
        let mut events = self.events.iter().peekable();
        for (tick, &q) in self.inputs.iter().enumerate() {
            while let Some(e) = events.next_if(|e| e.tick as usize == tick) {
                sim.inject(e.injection);
            }
            if sim.step(Controls::from_quantized(q)).is_some() {
                break;
            }
        }
        sim
    }
}

/// A simulation that records its quantized inputs and injections as a
/// [`Replay`]. Used by interactive front-ends.
#[derive(Clone, Debug)]
pub struct RecordingSim {
    sim: Sim,
    replay: Replay,
}

impl RecordingSim {
    /// Starts a recorded episode of a built-in configuration.
    pub fn new(config: VehicleConfig, seed: u64) -> Self {
        Self {
            sim: Sim::from_config(config, seed),
            replay: Replay::new(config, seed),
        }
    }

    /// Steps with quantized inputs and records them.
    pub fn step_quantized(&mut self, q: [u8; 3]) -> Option<Outcome> {
        if self.sim.outcome().is_none() {
            // Store the level actually applied (levels above 254 act as 254).
            let applied = q.map(|v| v.min(QUANT_MAX));
            self.replay.inputs.push(applied);
        }
        self.sim.step(Controls::from_quantized(q))
    }

    /// Injects an event and records it if it was accepted.
    pub fn inject(&mut self, injection: Injection) -> bool {
        if self.replay.events.len() >= MAX_REPLAY_EVENTS {
            return false;
        }
        let accepted = self.sim.inject(injection);
        if accepted {
            let tick = u32::try_from(self.sim.tick()).unwrap_or(u32::MAX);
            self.replay.events.push(ReplayEvent { tick, injection });
        }
        accepted
    }

    /// The simulation.
    pub fn sim(&self) -> &Sim {
        &self.sim
    }

    /// The recording so far.
    pub fn replay(&self) -> &Replay {
        &self.replay
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn scripted_run(config: VehicleConfig, seed: u64) -> RecordingSim {
        let mut rec = RecordingSim::new(config, seed);
        let mut rng = Rng::from_seed(seed ^ 0xABCD);
        let mut q = [140u8, 127, 127];
        for tick in 0..2_000u32 {
            if tick % 20 == 0 {
                q = [
                    100 + (rng.next_u64() % 155) as u8,
                    (rng.next_u64() % 255) as u8,
                    127,
                ];
            }
            if tick == 300 {
                rec.inject(Injection::Gust {
                    amplitude: 6.0,
                    duration: 3.0,
                });
            }
            if tick == 500 {
                rec.inject(Injection::ThrustLoss { factor: 0.85 });
            }
            if rec.step_quantized(q).is_some() {
                break;
            }
        }
        rec
    }

    #[test]
    fn encode_decode_round_trip() {
        let rec = scripted_run(VehicleConfig::Easy, 5);
        let text = rec.replay().encode().unwrap();
        assert!(
            text.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        );
        let decoded = Replay::decode(&text).unwrap();
        assert_eq!(&decoded, rec.replay());
    }

    #[test]
    fn replay_reproduces_the_run_bit_for_bit() {
        for config in VehicleConfig::ALL {
            let rec = scripted_run(config, 17);
            let replayed = Replay::decode(&rec.replay().encode().unwrap())
                .unwrap()
                .run();
            assert_eq!(replayed.tick(), rec.sim().tick());
            assert_eq!(replayed.outcome(), rec.sim().outcome());
            assert_eq!(replayed.score(), rec.sim().score());
            let a = replayed.telemetry().map(f64::to_bits);
            let b = rec.sim().telemetry().map(f64::to_bits);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn keyboard_runs_compress_well() {
        let mut rec = RecordingSim::new(VehicleConfig::Easy, 1);
        for tick in 0..3_600u32 {
            let throttle = if (tick / 90) % 2 == 0 { 150 } else { 120 };
            rec.step_quantized([throttle, 127, 127]);
        }
        let len = rec.replay().encode().unwrap().len();
        assert!(
            len < 600,
            "one minute of keyboard input encodes to {len} chars"
        );
    }

    #[test]
    fn rejects_malformed_input_without_panicking() {
        let good = scripted_run(VehicleConfig::Easy, 3)
            .replay()
            .to_bytes()
            .unwrap();
        // Every truncation is rejected.
        for n in 0..good.len() {
            assert!(Replay::from_bytes(&good[..n]).is_err(), "truncated at {n}");
        }
        // Trailing data is rejected.
        let mut longer = good.clone();
        longer.push(0);
        assert_eq!(Replay::from_bytes(&longer), Err(ReplayError::TrailingBytes));
        // Random byte flips never panic.
        let mut rng = Rng::from_seed(99);
        for _ in 0..5_000 {
            let mut bytes = good.clone();
            let i = (rng.next_u64() % bytes.len() as u64) as usize;
            bytes[i] ^= 1 << (rng.next_u64() % 8);
            let _ = Replay::from_bytes(&bytes);
        }
        // Random garbage never panics.
        for len in 0..200 {
            let bytes: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
            let _ = Replay::from_bytes(&bytes);
        }
        assert_eq!(Replay::decode("not base64 !!"), Err(ReplayError::Base64));
        assert_eq!(
            Replay::decode(&"A".repeat(MAX_ENCODED_LEN + 1)),
            Err(ReplayError::TooLong)
        );
    }

    #[test]
    fn rejects_other_physics_versions() {
        let mut bytes = Replay::new(VehicleConfig::Easy, 1).to_bytes().unwrap();
        bytes[5] = bytes[5].wrapping_add(1);
        assert!(matches!(
            Replay::from_bytes(&bytes),
            Err(ReplayError::PhysicsVersion { .. })
        ));
    }

    #[test]
    fn rejects_huge_declared_sizes() {
        let mut bytes = Replay::new(VehicleConfig::Easy, 1).to_bytes().unwrap();
        // ticks field starts after magic(4) + format(1) + physics(2) + config(1) + seed(8).
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(Replay::from_bytes(&bytes), Err(ReplayError::Limits));
    }

    #[test]
    fn varint_round_trip() {
        for v in [0u32, 1, 127, 128, 300, 16_383, 16_384, u32::MAX] {
            let mut out = Vec::new();
            write_varint(&mut out, v);
            let mut r = Reader { data: &out, pos: 0 };
            assert_eq!(r.varint().unwrap(), v);
            assert!(r.is_done());
        }
        let mut r = Reader {
            data: &[0xFF, 0xFF, 0xFF, 0xFF, 0x7F],
            pos: 0,
        };
        assert_eq!(r.varint(), Err(ReplayError::Limits), "overflows u32");
        let mut r = Reader {
            data: &[0x81, 0x00],
            pos: 0,
        };
        assert_eq!(r.varint(), Err(ReplayError::Limits), "non-minimal encoding");
    }

    /// Header (magic … ticks) followed by the given run section and no events.
    fn with_runs(ticks: u32, runs: &[(u8, [u8; 3])]) -> Vec<u8> {
        let mut bytes = Replay::new(VehicleConfig::Easy, 1).to_bytes().unwrap();
        bytes.truncate(16);
        bytes.extend_from_slice(&ticks.to_le_bytes());
        bytes.extend_from_slice(&(runs.len() as u32).to_le_bytes());
        for (n, q) in runs {
            bytes.push(*n);
            bytes.extend_from_slice(q);
        }
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes
    }

    #[test]
    fn rejects_non_canonical_encodings() {
        let a = [130, 127, 127];
        let b = [140, 127, 127];
        assert!(Replay::from_bytes(&with_runs(5, &[(2, a), (3, b)])).is_ok());
        assert_eq!(
            Replay::from_bytes(&with_runs(5, &[(2, a), (3, a)])),
            Err(ReplayError::InconsistentLength),
            "adjacent equal runs"
        );

        // A thrust-loss event with a non-zero unused value.
        let mut replay = Replay::new(VehicleConfig::Easy, 1);
        replay.inputs = vec![a; 3];
        replay.events.push(ReplayEvent {
            tick: 1,
            injection: Injection::ThrustLoss { factor: 0.5 },
        });
        let mut bytes = replay.to_bytes().unwrap();
        let len = bytes.len();
        bytes[len - 8..].copy_from_slice(&1.0f64.to_le_bytes());
        assert_eq!(Replay::from_bytes(&bytes), Err(ReplayError::BadEvent));
    }

    #[test]
    fn rejects_more_ticks_than_an_episode_can_last() {
        let max = VehicleConfig::Easy.params().max_ticks() as u32;
        assert!(Replay::from_bytes(&with_runs(max, &[(0x7F, [130, 127, 127])])).is_err());
        let mut replay = Replay::new(VehicleConfig::Easy, 1);
        replay.inputs = vec![[130, 127, 127]; max as usize + 1];
        assert_eq!(replay.to_bytes(), Err(ReplayError::Limits));
        replay.inputs.pop();
        assert!(Replay::decode(&replay.encode().unwrap()).is_ok());
    }

    #[test]
    fn encoder_refuses_what_the_decoder_would_reject() {
        let mut replay = Replay::new(VehicleConfig::Easy, 1);
        replay.inputs = vec![[255, 127, 127]];
        assert_eq!(replay.to_bytes(), Err(ReplayError::BadInput));
        replay.inputs = vec![[130, 127, 127]; 2];
        replay.events.push(ReplayEvent {
            tick: 5,
            injection: Injection::ThrustLoss { factor: 0.5 },
        });
        assert_eq!(
            replay.to_bytes(),
            Err(ReplayError::BadEvent),
            "event after the last tick"
        );
    }
}
