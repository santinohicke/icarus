//! WebAssembly bindings for the Icarus simulator.
//!
//! The JS side creates one [`WasmSim`] per run, calls [`WasmSim::step`] once
//! per fixed tick with quantized inputs, and reads the whole state back with
//! [`WasmSim::telemetry`] (one `Float64Array`, indexed by
//! [`telemetry_fields`]). Every run is recorded and can be exported as a
//! replay string; [`WasmReplay`] plays one back tick by tick.

use icarus_core::physics::scenario::Outcome;
use icarus_core::{Injection, RecordingSim, Replay, Sim, TELEMETRY_FIELDS, VehicleConfig};
use wasm_bindgen::prelude::*;

/// Indices into the array returned by [`WasmSim::vehicle_info`].
const VEHICLE_INFO_FIELDS: [&str; 18] = [
    "length",
    "diameter",
    "leg_half_span",
    "pad_half_width",
    "gimbal_max",
    "max_thrust",
    "min_throttle",
    "throttle_cutoff",
    "rcs_torque",
    "dry_mass",
    "initial_propellant",
    "dt",
    "max_vertical_speed",
    "max_horizontal_speed",
    "max_tilt",
    "max_angular_rate",
    "max_ticks",
    "initial_throttle_level",
];

fn parse_config(name: &str) -> Result<VehicleConfig, JsError> {
    VehicleConfig::from_name(name).ok_or_else(|| {
        JsError::new(&format!(
            "unknown vehicle configuration {name:?} (expected \"easy\" or \"realistic\")"
        ))
    })
}

fn vehicle_info(sim: &Sim) -> Vec<f64> {
    let p = sim.params();
    let (v, l) = (&p.vehicle, &p.landing);
    vec![
        v.length,
        v.diameter,
        v.leg_half_span,
        l.pad_half_width,
        v.gimbal_max,
        v.max_thrust,
        v.min_throttle,
        v.throttle_cutoff,
        v.rcs_torque,
        v.dry_mass,
        sim.initial_propellant(),
        p.dt,
        l.max_vertical_speed,
        l.max_horizontal_speed,
        l.max_tilt,
        l.max_angular_rate,
        sim.max_ticks() as f64,
        f64::from(sim.initial_throttle_level()),
    ]
}

fn outcome_code(o: Option<Outcome>) -> u8 {
    o.map_or(0, Outcome::code)
}

/// A recorded, interactive simulation.
#[wasm_bindgen]
pub struct WasmSim {
    inner: RecordingSim,
}

#[wasm_bindgen]
impl WasmSim {
    /// Starts a run. `config` is `"easy"` or `"realistic"`; `seed` is a
    /// `BigInt` on the JS side.
    #[wasm_bindgen(constructor)]
    pub fn new(config: &str, seed: u64) -> Result<WasmSim, JsError> {
        Ok(Self {
            inner: RecordingSim::new(parse_config(config)?, seed),
        })
    }

    /// Advances one tick with quantized inputs (`0..=254`, 127 = neutral).
    /// Returns the outcome code (0 while running).
    pub fn step(&mut self, throttle: u8, gimbal: u8, rcs: u8) -> u8 {
        outcome_code(self.inner.step_quantized([throttle, gimbal, rcs]))
    }

    /// Every observable quantity; see [`telemetry_fields`].
    pub fn telemetry(&self) -> Vec<f64> {
        self.inner.sim().telemetry().to_vec()
    }

    /// Static vehicle and scenario values; see [`vehicle_info_fields`].
    #[wasm_bindgen(js_name = vehicleInfo)]
    pub fn vehicle_info(&self) -> Vec<f64> {
        vehicle_info(self.inner.sim())
    }

    /// Jams the gimbal at `angle` radians (clamped to the gimbal range).
    #[wasm_bindgen(js_name = injectGimbalStuck)]
    pub fn inject_gimbal_stuck(&mut self, angle: f64) -> bool {
        self.inner.inject(Injection::GimbalStuck { angle })
    }

    /// Reduces max thrust to `factor` (clamped to `[0, 1]`).
    #[wasm_bindgen(js_name = injectThrustLoss)]
    pub fn inject_thrust_loss(&mut self, factor: f64) -> bool {
        self.inner.inject(Injection::ThrustLoss { factor })
    }

    /// Adds a discrete gust.
    #[wasm_bindgen(js_name = injectGust)]
    pub fn inject_gust(&mut self, amplitude: f64, duration: f64) -> bool {
        self.inner.inject(Injection::Gust {
            amplitude,
            duration,
        })
    }

    /// Outcome code (0 while running).
    pub fn outcome(&self) -> u8 {
        outcome_code(self.inner.sim().outcome())
    }

    /// Outcome name (empty while running).
    #[wasm_bindgen(js_name = outcomeName)]
    pub fn outcome_name(&self) -> String {
        self.inner
            .sim()
            .outcome()
            .map_or("", Outcome::name)
            .to_owned()
    }

    /// Score (0 unless landed).
    pub fn score(&self) -> u32 {
        self.inner.sim().score()
    }

    /// Ticks simulated so far.
    pub fn tick(&self) -> u32 {
        u32::try_from(self.inner.sim().tick()).unwrap_or(u32::MAX)
    }

    /// Seed of the run (`BigInt`).
    pub fn seed(&self) -> u64 {
        self.inner.sim().seed()
    }

    /// The recorded run as a URL-safe replay string.
    pub fn replay(&self) -> Result<String, JsError> {
        self.inner
            .replay()
            .encode()
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

/// Plays a replay back tick by tick.
#[wasm_bindgen]
pub struct WasmReplay {
    replay: Replay,
    sim: Sim,
    cursor: usize,
    next_event: usize,
}

#[wasm_bindgen]
impl WasmReplay {
    /// Decodes and validates a replay string.
    #[wasm_bindgen(constructor)]
    pub fn new(encoded: &str) -> Result<WasmReplay, JsError> {
        let replay = Replay::decode(encoded).map_err(|e| JsError::new(&e.to_string()))?;
        let sim = Sim::from_config(replay.config, replay.seed);
        Ok(Self {
            replay,
            sim,
            cursor: 0,
            next_event: 0,
        })
    }

    /// Advances one recorded tick. Returns `true` when the playback is over.
    pub fn step(&mut self) -> bool {
        if self.is_done() {
            return true;
        }
        while let Some(e) = self.replay.events.get(self.next_event) {
            if e.tick as usize != self.cursor {
                break;
            }
            self.sim.inject(e.injection);
            self.next_event += 1;
        }
        let q = self.replay.inputs[self.cursor];
        self.cursor += 1;
        self.sim.step(icarus_core::Controls::from_quantized(q));
        self.is_done()
    }

    /// `true` when every input has been played or the episode ended.
    #[wasm_bindgen(js_name = isDone)]
    pub fn is_done(&self) -> bool {
        self.cursor >= self.replay.inputs.len() || self.sim.outcome().is_some()
    }

    /// Vehicle configuration name.
    pub fn config(&self) -> String {
        self.replay.config.name().to_owned()
    }

    /// Seed (`BigInt`).
    pub fn seed(&self) -> u64 {
        self.replay.seed
    }

    /// Number of recorded ticks.
    #[wasm_bindgen(js_name = totalTicks)]
    pub fn total_ticks(&self) -> u32 {
        u32::try_from(self.replay.inputs.len()).unwrap_or(u32::MAX)
    }

    /// Every observable quantity; see [`telemetry_fields`].
    pub fn telemetry(&self) -> Vec<f64> {
        self.sim.telemetry().to_vec()
    }

    /// Static vehicle and scenario values; see [`vehicle_info_fields`].
    #[wasm_bindgen(js_name = vehicleInfo)]
    pub fn vehicle_info(&self) -> Vec<f64> {
        vehicle_info(&self.sim)
    }

    /// Outcome code (0 while running).
    pub fn outcome(&self) -> u8 {
        outcome_code(self.sim.outcome())
    }

    /// Score (0 unless landed).
    pub fn score(&self) -> u32 {
        self.sim.score()
    }

    /// Quantized inputs `[throttle, gimbal, rcs]` of the last played tick
    /// (neutral before the first one), for displaying the pilot's commands.
    #[wasm_bindgen(js_name = lastInput)]
    pub fn last_input(&self) -> Vec<u8> {
        self.cursor
            .checked_sub(1)
            .and_then(|i| self.replay.inputs.get(i))
            .map_or_else(|| vec![127, 127, 127], |q| q.to_vec())
    }

    /// Ticks played so far.
    pub fn tick(&self) -> u32 {
        u32::try_from(self.cursor).unwrap_or(u32::MAX)
    }
}

/// Names of the telemetry values, in array order.
#[wasm_bindgen(js_name = telemetryFields)]
pub fn telemetry_fields() -> Vec<String> {
    TELEMETRY_FIELDS.iter().map(|s| (*s).to_owned()).collect()
}

/// Names of the vehicle-info values, in array order.
#[wasm_bindgen(js_name = vehicleInfoFields)]
pub fn vehicle_info_fields() -> Vec<String> {
    VEHICLE_INFO_FIELDS
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
}

/// Name of an outcome code (empty for 0 or unknown codes).
#[wasm_bindgen(js_name = outcomeName)]
pub fn outcome_name(code: u8) -> String {
    Outcome::from_code(code)
        .map_or("", Outcome::name)
        .to_owned()
}

/// Physics version of this build.
#[wasm_bindgen(js_name = physicsVersion)]
pub fn physics_version() -> u16 {
    icarus_core::PHYSICS_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vehicle_info_matches_its_field_list() {
        let sim = Sim::from_config(VehicleConfig::Easy, 1);
        assert_eq!(vehicle_info(&sim).len(), VEHICLE_INFO_FIELDS.len());
    }
}
