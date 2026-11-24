//! The simulation loop: one call to [`Sim::step`] advances one fixed tick.
//!
//! Order of operations inside a tick (fixed, part of the physics version):
//!
//! 1. scheduled faults whose time has come are applied;
//! 2. engine cut-off / ignition logic (discrete, never inside RK4);
//! 3. commands are computed and the wind for this tick is sampled;
//! 4. one RK4 step with all inputs held constant;
//! 5. touchdown detection (bisection on the contact time, re-integrating from
//!    the start of the step), then flameout, divergence, bounds and the time
//!    limit;
//! 6. the sensors take a new measurement.

use crate::physics::dynamics::{HeldInputs, derivatives, mass_properties};
use crate::physics::faults::{Fault, FaultState};
use crate::physics::integrator::rk4_step;
use crate::physics::params::{ParamError, Params, VehicleConfig};
use crate::physics::scenario::{
    ContactGeometry, Outcome, contact_geometry, evaluate_touchdown, sample_initial, score,
};
use crate::physics::sensors::{MEAS_DIM, Sensors};
use crate::physics::state::{Controls, QUANT_LEVELS, STATE_DIM, State, dequantize, quantize};
use crate::physics::wind::Wind;
use crate::rng::{Rng, Stream};

/// Bisection iterations used to locate the touchdown time within a step.
/// 48 halvings of 1/60 s locate it to well below a nanosecond.
pub const TOUCHDOWN_ITERATIONS: u32 = 48;

/// Largest gust that can be injected manually (m/s).
pub const MAX_INJECTED_GUST: f64 = 50.0;

/// Longest gust that can be injected manually (s).
pub const MAX_INJECTED_GUST_DURATION: f64 = 30.0;

/// An event injected from outside the simulation (web buttons, scripted
/// scenarios). Injections are recorded in replays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Injection {
    /// Jam the gimbal at `angle` (rad), clamped to the gimbal range.
    GimbalStuck {
        /// Jam angle (rad).
        angle: f64,
    },
    /// Reduce max thrust to `factor` (clamped to `[0, 1]`).
    ThrustLoss {
        /// Remaining fraction of max thrust.
        factor: f64,
    },
    /// Add a discrete gust (amplitude clamped to ±50 m/s, duration to
    /// `(0, 30]` s).
    Gust {
        /// Peak speed (m/s, signed).
        amplitude: f64,
        /// Duration (s).
        duration: f64,
    },
}

/// Indices into the array returned by [`Sim::telemetry`].
pub mod tm {
    /// Simulated time (s).
    pub const TIME: usize = 0;
    /// Tick counter.
    pub const TICK: usize = 1;
    /// Center-of-mass x (m).
    pub const X: usize = 2;
    /// Center-of-mass y (m).
    pub const Y: usize = 3;
    /// Horizontal velocity (m/s).
    pub const VX: usize = 4;
    /// Vertical velocity (m/s).
    pub const VY: usize = 5;
    /// Tilt, unwrapped (rad).
    pub const THETA: usize = 6;
    /// Angular velocity (rad/s).
    pub const OMEGA: usize = 7;
    /// Total mass (kg).
    pub const MASS: usize = 8;
    /// Actual gimbal angle (rad).
    pub const GIMBAL: usize = 9;
    /// Actual thrust (N).
    pub const THRUST: usize = 10;
    /// Propellant left (kg).
    pub const PROPELLANT: usize = 11;
    /// Center-of-mass height above the tail (m).
    pub const CM_HEIGHT: usize = 12;
    /// Height of the lowest point above the ground (m).
    pub const CLEARANCE: usize = 13;
    /// 1 if the engine is running.
    pub const ENGINE_ON: usize = 14;
    /// Ignitions left.
    pub const IGNITIONS_LEFT: usize = 15;
    /// Thrust command of the last tick (N).
    pub const THRUST_CMD: usize = 16;
    /// Gimbal command of the last tick (rad).
    pub const GIMBAL_CMD: usize = 17;
    /// RCS torque of the last tick (N·m).
    pub const RCS_TORQUE: usize = 18;
    /// Wind of the last tick (m/s).
    pub const WIND: usize = 19;
    /// Observed x (m), delayed and noisy.
    pub const OBS_X: usize = 20;
    /// Observed y (m).
    pub const OBS_Y: usize = 21;
    /// Observed vx (m/s).
    pub const OBS_VX: usize = 22;
    /// Observed vy (m/s).
    pub const OBS_VY: usize = 23;
    /// Observed θ (rad).
    pub const OBS_THETA: usize = 24;
    /// Observed ω (rad/s).
    pub const OBS_OMEGA: usize = 25;
    /// Remaining fraction of max thrust (1 without a thrust-loss fault).
    pub const THRUST_FACTOR: usize = 26;
    /// 1 if the gimbal is stuck.
    pub const GIMBAL_STUCK: usize = 27;
    /// Jam angle of a stuck gimbal (rad), 0 otherwise.
    pub const GIMBAL_STUCK_ANGLE: usize = 28;
    /// Outcome code (0 while running, see `Outcome::code`).
    pub const OUTCOME: usize = 29;
    /// Score (0 unless landed).
    pub const SCORE: usize = 30;
    /// Actual thrust as a fraction of max thrust.
    pub const THRUST_FRACTION: usize = 31;
    /// Horizontal velocity of the lowest point (the one judged at touchdown).
    pub const CONTACT_VX: usize = 32;
    /// Vertical velocity of the lowest point (the one judged at touchdown).
    pub const CONTACT_VY: usize = 33;
    /// Horizontal position of the tail (engine), between the legs (m).
    pub const TAIL_X: usize = 34;
    /// Pad half-width minus the farthest leg tip's distance from the pad
    /// center (m): negative means a leg tip is off the pad.
    pub const PAD_MARGIN: usize = 35;
}

/// Number of telemetry values.
pub const TELEMETRY_LEN: usize = 36;

/// Names of the telemetry values, indexed by the constants in [`tm`].
pub const TELEMETRY_FIELDS: [&str; TELEMETRY_LEN] = [
    "time",
    "tick",
    "x",
    "y",
    "vx",
    "vy",
    "theta",
    "omega",
    "mass",
    "gimbal",
    "thrust",
    "propellant",
    "cm_height",
    "clearance",
    "engine_on",
    "ignitions_left",
    "thrust_cmd",
    "gimbal_cmd",
    "rcs_torque",
    "wind",
    "obs_x",
    "obs_y",
    "obs_vx",
    "obs_vy",
    "obs_theta",
    "obs_omega",
    "thrust_factor",
    "gimbal_stuck",
    "gimbal_stuck_angle",
    "outcome",
    "score",
    "thrust_fraction",
    "contact_vx",
    "contact_vy",
    "tail_x",
    "pad_margin",
];

/// One landing episode.
#[derive(Clone, Debug)]
pub struct Sim {
    params: Params,
    seed: u64,
    state: State,
    engine_on: bool,
    ignitions_left: u32,
    tick: u64,
    max_ticks: u64,
    wind: Wind,
    sensors: Sensors,
    faults: FaultState,
    outcome: Option<Outcome>,
    contact: Option<ContactGeometry>,
    touchdown_time: Option<f64>,
    initial_propellant: f64,
    controls: Controls,
    inputs: HeldInputs,
}

/// Below this clearance (m) at the end of a step, the middle of the step is
/// also checked for ground contact, so a contact that happens and "un-happens"
/// within one step (near-hover just above the ground) is not missed.
const NEAR_GROUND: f64 = 2.0;

impl Sim {
    /// Creates an episode with custom parameters.
    pub fn new(params: Params, seed: u64) -> Result<Self, ParamError> {
        params.validate()?;
        Ok(Self::build(params, seed))
    }

    /// Creates an episode of a built-in configuration (always valid; checked
    /// by the test suite).
    pub fn from_config(config: VehicleConfig, seed: u64) -> Self {
        Self::build(config.params(), seed)
    }

    fn build(params: Params, seed: u64) -> Self {
        let mut init_rng = Rng::for_stream(seed, Stream::InitialConditions);
        let ic = sample_initial(&params, &mut init_rng);
        let wind = Wind::new(&params.wind, params.dt, Rng::for_stream(seed, Stream::Wind));
        let sensors = Sensors::new(
            &params.sensors,
            Rng::for_stream(seed, Stream::Sensors),
            &ic.state,
        );
        let faults = FaultState::new(&params.faults, Rng::for_stream(seed, Stream::Faults));
        let initial_propellant = ic.state.mass - params.vehicle.dry_mass;
        let inputs = HeldInputs {
            thrust_cmd: ic.state.thrust,
            gimbal_cmd: 0.0,
            rcs_torque: 0.0,
            wind: wind.current(),
        };
        let max_ticks = params.max_ticks();
        Self {
            params,
            seed,
            state: ic.state,
            engine_on: ic.engine_on,
            ignitions_left: ic.ignitions,
            tick: 0,
            max_ticks,
            wind,
            sensors,
            faults,
            outcome: None,
            contact: None,
            touchdown_time: None,
            initial_propellant,
            controls: Controls::default(),
            inputs,
        }
    }

    /// Thrust fraction (of max thrust) for a normalized throttle command at or
    /// above the cut-off.
    fn throttle_fraction(&self, throttle: f64) -> f64 {
        let v = &self.params.vehicle;
        let span = 1.0 - v.throttle_cutoff;
        let t = ((throttle - v.throttle_cutoff) / span).clamp(0.0, 1.0);
        v.min_throttle + (1.0 - v.min_throttle) * t
    }

    fn integrate(&self, s: &[f64; STATE_DIM], inputs: &HeldInputs, h: f64) -> [f64; STATE_DIM] {
        let (v, env) = (&self.params.vehicle, &self.params.environment);
        rk4_step(|x: &[f64; STATE_DIM]| derivatives(v, env, x, inputs), s, h)
    }

    /// Returns a time within the step (from `s0`) at which the vehicle is on
    /// or below the ground, if any.
    fn contact_within_step(
        &self,
        s0: &[f64; STATE_DIM],
        inputs: &HeldInputs,
        end: &State,
    ) -> Option<f64> {
        let v = &self.params.vehicle;
        let dt = self.params.dt;
        let end_clearance = contact_geometry(v, end).clearance;
        if end_clearance <= 0.0 {
            return Some(dt);
        }
        if end_clearance < NEAR_GROUND {
            let mid = State::from_array(&self.integrate(s0, inputs, 0.5 * dt));
            if contact_geometry(v, &mid).clearance <= 0.0 {
                return Some(0.5 * dt);
            }
        }
        None
    }

    /// Finds the first ground contact in `(0, hi]` after `s0` by bisection.
    /// Returns the state at contact (clearance ≤ 0, within a nanosecond of the
    /// first contact) and the time since the start of the step.
    fn find_touchdown(&self, s0: &[f64; STATE_DIM], inputs: &HeldInputs, hi: f64) -> (State, f64) {
        let v = &self.params.vehicle;
        let (mut lo, mut hi) = (0.0, hi);
        for _ in 0..TOUCHDOWN_ITERATIONS {
            let mid = 0.5 * (lo + hi);
            let s = State::from_array(&self.integrate(s0, inputs, mid));
            if contact_geometry(v, &s).clearance > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (State::from_array(&self.integrate(s0, inputs, hi)), hi)
    }

    /// Advances one tick. Returns the outcome once the episode has ended;
    /// further calls do nothing.
    pub fn step(&mut self, controls: Controls) -> Option<Outcome> {
        if self.outcome.is_some() {
            return self.outcome;
        }
        let controls = Controls::new(controls.throttle, controls.gimbal, controls.rcs);
        self.faults.update(self.time());

        // Engine cut-off and ignition.
        let (cutoff, dry_mass) = (
            self.params.vehicle.throttle_cutoff,
            self.params.vehicle.dry_mass,
        );
        let commanded_on = controls.throttle >= cutoff;
        if self.engine_on && !commanded_on {
            self.engine_on = false;
        } else if !self.engine_on
            && commanded_on
            && self.ignitions_left > 0
            && self.state.mass > dry_mass
        {
            self.engine_on = true;
            self.ignitions_left -= 1;
        }

        let v = &self.params.vehicle;
        let thrust_cmd = if self.engine_on {
            self.throttle_fraction(controls.throttle) * v.max_thrust * self.faults.thrust_factor()
        } else {
            0.0
        };
        let gimbal_cmd = self
            .faults
            .gimbal_stuck()
            .unwrap_or(controls.gimbal * v.gimbal_max);
        let rcs_torque = controls.rcs * v.rcs_torque;
        let wind = self.wind.advance();
        let inputs = HeldInputs {
            thrust_cmd,
            gimbal_cmd,
            rcs_torque,
            wind,
        };
        self.controls = controls;
        self.inputs = inputs;

        let s0 = self.state.to_array();
        let dt = self.params.dt;
        let next = State::from_array(&self.integrate(&s0, &inputs, dt));
        let step_start = self.time();
        self.tick += 1;

        if !next.is_finite() {
            // Keep the last finite state: telemetry must never contain NaN.
            self.outcome = Some(Outcome::Diverged);
        } else if let Some(hi) = self.contact_within_step(&s0, &inputs, &next) {
            let (touchdown, elapsed) = self.find_touchdown(&s0, &inputs, hi);
            let geometry = contact_geometry(&self.params.vehicle, &touchdown);
            self.state = touchdown;
            self.contact = Some(geometry);
            self.touchdown_time = Some(step_start + elapsed);
            self.outcome = Some(evaluate_touchdown(
                &self.params.landing,
                &touchdown,
                &geometry,
            ));
        } else {
            self.state = next;
            let lim = &self.params.limits;
            if self.state.mass <= dry_mass {
                self.state.mass = dry_mass;
                self.state.thrust = 0.0;
                self.engine_on = false;
                self.outcome = Some(Outcome::OutOfPropellant);
            } else if self.state.x.abs() > lim.max_abs_x || self.state.y > lim.max_altitude {
                self.outcome = Some(Outcome::OutOfBounds);
            } else if self.tick >= self.max_ticks {
                self.outcome = Some(Outcome::TimeLimit);
            }
        }

        self.sensors.measure(&self.state);
        self.outcome
    }

    /// Applies an external event now (before the next step). Returns `false`
    /// if it was rejected (non-finite values, no free gust slot, or the
    /// episode is over).
    pub fn inject(&mut self, injection: Injection) -> bool {
        if self.outcome.is_some() {
            return false;
        }
        let gimbal_max = self.params.vehicle.gimbal_max;
        match injection {
            Injection::GimbalStuck { angle } => self
                .faults
                .apply_clamped(Fault::GimbalStuck { angle }, gimbal_max),
            Injection::ThrustLoss { factor } => self
                .faults
                .apply_clamped(Fault::ThrustLoss { factor }, gimbal_max),
            Injection::Gust {
                amplitude,
                duration,
            } => {
                if !(amplitude.is_finite() && duration.is_finite() && duration > 0.0) {
                    return false;
                }
                self.wind.add_gust(
                    amplitude.clamp(-MAX_INJECTED_GUST, MAX_INJECTED_GUST),
                    duration.min(MAX_INJECTED_GUST_DURATION),
                )
            }
        }
    }

    /// Parameters of the episode.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// Seed of the episode.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Continuous state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Ticks simulated so far.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Simulated time (s). After a touchdown, the exact contact time.
    pub fn time(&self) -> f64 {
        self.touchdown_time
            .unwrap_or(self.tick as f64 * self.params.dt)
    }

    /// Tick at which the episode is truncated.
    pub fn max_ticks(&self) -> u64 {
        self.max_ticks
    }

    /// Outcome, once the episode has ended.
    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    /// Contact geometry at touchdown, if the episode ended on the ground.
    pub fn contact(&self) -> Option<&ContactGeometry> {
        self.contact.as_ref()
    }

    /// Current contact geometry (clearance, leg positions, …).
    pub fn geometry(&self) -> ContactGeometry {
        contact_geometry(&self.params.vehicle, &self.state)
    }

    /// Whether the engine is running.
    pub fn engine_on(&self) -> bool {
        self.engine_on
    }

    /// Ignitions left.
    pub fn ignitions_left(&self) -> u32 {
        self.ignitions_left
    }

    /// Propellant left (kg).
    pub fn propellant(&self) -> f64 {
        (self.state.mass - self.params.vehicle.dry_mass).max(0.0)
    }

    /// Propellant at the start of the episode (kg).
    pub fn initial_propellant(&self) -> f64 {
        self.initial_propellant
    }

    /// Wind applied during the last tick (m/s).
    pub fn wind(&self) -> f64 {
        self.inputs.wind
    }

    /// Controls applied during the last tick (sanitized).
    pub fn controls(&self) -> Controls {
        self.controls
    }

    /// Inputs held during the last tick.
    pub fn inputs(&self) -> HeldInputs {
        self.inputs
    }

    /// Delayed, noisy observation `[x, y, vx, vy, θ, ω]`.
    pub fn observation(&self) -> [f64; MEAS_DIM] {
        self.sensors.observation()
    }

    /// Noise added to the latest measurement.
    pub fn last_sensor_noise(&self) -> [f64; MEAS_DIM] {
        self.sensors.last_noise()
    }

    /// Remaining fraction of max thrust.
    pub fn thrust_factor(&self) -> f64 {
        self.faults.thrust_factor()
    }

    /// Jam angle of the gimbal, if stuck.
    pub fn gimbal_stuck(&self) -> Option<f64> {
        self.faults.gimbal_stuck()
    }

    /// Quantized throttle level that keeps the initial engine state: the
    /// level whose thrust is closest to the starting throttle when the engine
    /// starts running (never below the cut-off), and 0 (off) otherwise.
    /// Interactive front-ends start their throttle control at this level.
    pub fn initial_throttle_level(&self) -> u8 {
        let (start, v) = (&self.params.start, &self.params.vehicle);
        if !start.engine_on {
            return 0;
        }
        let u = if v.min_throttle >= 1.0 {
            1.0
        } else {
            let t = (start.initial_throttle - v.min_throttle) / (1.0 - v.min_throttle);
            v.throttle_cutoff + t * (1.0 - v.throttle_cutoff)
        };
        let mut q = quantize(u);
        while q < QUANT_LEVELS - 1 && dequantize(q) < v.throttle_cutoff {
            q += 1;
        }
        q
    }

    /// Score of the episode (0 unless landed).
    pub fn score(&self) -> u32 {
        score(
            &self.params.landing,
            self.outcome,
            self.contact.as_ref(),
            self.propellant(),
            self.initial_propellant,
        )
    }

    /// Every observable quantity as a flat array, indexed by [`tm`].
    pub fn telemetry(&self) -> [f64; TELEMETRY_LEN] {
        let s = &self.state;
        let obs = self.observation();
        let g = self.geometry();
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        let mut t = [0.0; TELEMETRY_LEN];
        t[tm::TIME] = self.time();
        t[tm::TICK] = self.tick as f64;
        t[tm::X] = s.x;
        t[tm::Y] = s.y;
        t[tm::VX] = s.vx;
        t[tm::VY] = s.vy;
        t[tm::THETA] = s.theta;
        t[tm::OMEGA] = s.omega;
        t[tm::MASS] = s.mass;
        t[tm::GIMBAL] = s.gimbal;
        t[tm::THRUST] = s.thrust;
        t[tm::PROPELLANT] = self.propellant();
        t[tm::CM_HEIGHT] = mass_properties(&self.params.vehicle, s.mass).cm_height;
        t[tm::CLEARANCE] = g.clearance;
        t[tm::ENGINE_ON] = flag(self.engine_on);
        t[tm::IGNITIONS_LEFT] = f64::from(self.ignitions_left);
        t[tm::THRUST_CMD] = self.inputs.thrust_cmd;
        t[tm::GIMBAL_CMD] = self.inputs.gimbal_cmd;
        t[tm::RCS_TORQUE] = self.inputs.rcs_torque;
        t[tm::WIND] = self.inputs.wind;
        t[tm::OBS_X..=tm::OBS_OMEGA].copy_from_slice(&obs);
        t[tm::THRUST_FACTOR] = self.faults.thrust_factor();
        t[tm::GIMBAL_STUCK] = flag(self.faults.gimbal_stuck().is_some());
        t[tm::GIMBAL_STUCK_ANGLE] = self.faults.gimbal_stuck().unwrap_or(0.0);
        t[tm::OUTCOME] = f64::from(self.outcome.map_or(0, Outcome::code));
        t[tm::SCORE] = f64::from(self.score());
        t[tm::THRUST_FRACTION] = s.thrust / self.params.vehicle.max_thrust;
        t[tm::CONTACT_VX] = g.point_velocity.0;
        t[tm::CONTACT_VY] = g.point_velocity.1;
        t[tm::TAIL_X] = g.tail.0;
        t[tm::PAD_MARGIN] =
            self.params.landing.pad_half_width - g.left_leg.0.abs().max(g.right_leg.0.abs());
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::params::{DEG, Range};
    use crate::physics::scenario::CrashReason;

    #[test]
    fn telemetry_names_match_indices() {
        let pairs = [
            (tm::TIME, "time"),
            (tm::TICK, "tick"),
            (tm::X, "x"),
            (tm::Y, "y"),
            (tm::THETA, "theta"),
            (tm::CLEARANCE, "clearance"),
            (tm::WIND, "wind"),
            (tm::OBS_X, "obs_x"),
            (tm::OBS_OMEGA, "obs_omega"),
            (tm::OUTCOME, "outcome"),
            (tm::SCORE, "score"),
            (tm::THRUST_FRACTION, "thrust_fraction"),
            (tm::CONTACT_VX, "contact_vx"),
            (tm::CONTACT_VY, "contact_vy"),
            (tm::TAIL_X, "tail_x"),
            (tm::PAD_MARGIN, "pad_margin"),
        ];
        for (i, name) in pairs {
            assert_eq!(TELEMETRY_FIELDS[i], name);
        }
        let mut names = TELEMETRY_FIELDS.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), TELEMETRY_LEN, "names are unique");
    }

    #[test]
    fn engine_off_in_vacuum_falls_until_it_crashes() {
        let mut p = Params::easy();
        p.environment.rho0 = 0.0;
        p.wind.mean = Range::fixed(0.0);
        p.wind.turbulence_sigma = 0.0;
        let mut sim = Sim::new(p, 1).unwrap();
        let mut ticks = 0;
        while sim.step(Controls::new(-1.0, 0.0, 0.0)).is_none() {
            ticks += 1;
            assert!(ticks < 10_000);
        }
        assert!(matches!(sim.outcome(), Some(Outcome::Crashed(_))));
        assert!(
            sim.contact().unwrap().clearance.abs() < 1e-6,
            "touchdown located precisely"
        );
        assert_eq!(sim.score(), 0);
    }

    #[test]
    fn ignitions_are_limited() {
        let mut sim = Sim::from_config(VehicleConfig::Easy, 3);
        assert!(sim.engine_on());
        assert_eq!(sim.ignitions_left(), 3);
        for _ in 0..3 {
            sim.step(Controls::new(-1.0, 0.0, 0.0)); // cut-off
            assert!(!sim.engine_on());
            sim.step(Controls::new(0.0, 0.0, 0.0)); // relight
            assert!(sim.engine_on());
        }
        assert_eq!(sim.ignitions_left(), 0);
        sim.step(Controls::new(-1.0, 0.0, 0.0));
        sim.step(Controls::new(1.0, 0.0, 0.0));
        assert!(!sim.engine_on(), "no ignitions left");
    }

    #[test]
    fn realistic_starts_with_the_engine_off() {
        let mut sim = Sim::from_config(VehicleConfig::Realistic, 3);
        assert!(!sim.engine_on());
        assert_eq!(sim.state().thrust, 0.0);
        sim.step(Controls::new(-1.0, 0.0, 0.0));
        assert!(!sim.engine_on());
        assert_eq!(sim.ignitions_left(), 1);
        sim.step(Controls::new(0.5, 0.0, 0.0));
        assert!(sim.engine_on());
        assert_eq!(sim.ignitions_left(), 0);
    }

    #[test]
    fn initial_throttle_level_keeps_the_engine_state() {
        let mut sim = Sim::from_config(VehicleConfig::Easy, 1);
        let q = sim.initial_throttle_level();
        let thrust0 = sim.state().thrust;
        sim.step(Controls::from_quantized([q, 127, 127]));
        assert!(sim.engine_on());
        let rel = (sim.inputs().thrust_cmd - thrust0).abs() / thrust0;
        assert!(rel < 0.01, "thrust command changed by {rel}");

        let sim = Sim::from_config(VehicleConfig::Realistic, 1);
        assert_eq!(sim.initial_throttle_level(), 0);
    }

    #[test]
    fn throttle_maps_to_min_and_max_thrust() {
        let sim = Sim::from_config(VehicleConfig::Easy, 0);
        let v = &sim.params().vehicle;
        assert_eq!(sim.throttle_fraction(v.throttle_cutoff), v.min_throttle);
        assert_eq!(sim.throttle_fraction(1.0), 1.0);
    }

    #[test]
    fn running_out_of_propellant_ends_the_episode() {
        let mut p = Params::easy();
        p.start.propellant = Range::fixed(50.0);
        let mut sim = Sim::new(p, 1).unwrap();
        let mut outcome = None;
        for _ in 0..600 {
            outcome = sim.step(Controls::new(1.0, 0.0, 0.0));
            if outcome.is_some() {
                break;
            }
        }
        assert_eq!(outcome, Some(Outcome::OutOfPropellant));
        assert_eq!(sim.propellant(), 0.0);
        assert!(!sim.engine_on());
    }

    #[test]
    fn time_limit_truncates() {
        let mut p = Params::easy();
        p.limits.max_time = 1.0;
        let mut sim = Sim::new(p, 1).unwrap();
        let mut steps = 0;
        while sim.step(Controls::new(0.0, 0.0, 0.0)).is_none() {
            steps += 1;
        }
        assert_eq!(steps + 1, 60);
        assert_eq!(sim.outcome(), Some(Outcome::TimeLimit));
        assert!(sim.outcome().unwrap().is_truncation());
    }

    #[test]
    fn climbing_away_goes_out_of_bounds() {
        let mut p = Params::easy();
        p.limits.max_altitude = 1_000.0;
        let mut sim = Sim::new(p, 1).unwrap();
        let mut outcome = None;
        for _ in 0..7_200 {
            outcome = sim.step(Controls::new(1.0, 0.0, 0.0));
            if outcome.is_some() {
                break;
            }
        }
        assert_eq!(outcome, Some(Outcome::OutOfBounds));
    }

    #[test]
    fn stuck_gimbal_overrides_the_command() {
        let mut sim = Sim::from_config(VehicleConfig::Easy, 1);
        assert!(sim.inject(Injection::GimbalStuck { angle: 1.0 * DEG }));
        for _ in 0..120 {
            sim.step(Controls::new(0.0, -1.0, 0.0));
        }
        assert!((sim.state().gimbal - 1.0 * DEG).abs() < 1e-9);
        assert_eq!(sim.telemetry()[tm::GIMBAL_STUCK], 1.0);
    }

    #[test]
    fn thrust_loss_scales_the_command() {
        let mut sim = Sim::from_config(VehicleConfig::Easy, 1);
        assert!(sim.inject(Injection::ThrustLoss { factor: 0.8 }));
        sim.step(Controls::new(1.0, 0.0, 0.0));
        assert_eq!(
            sim.inputs().thrust_cmd,
            0.8 * sim.params().vehicle.max_thrust
        );
    }

    #[test]
    fn injection_rejects_invalid_values_and_finished_episodes() {
        let mut sim = Sim::from_config(VehicleConfig::Easy, 1);
        assert!(!sim.inject(Injection::Gust {
            amplitude: f64::NAN,
            duration: 1.0
        }));
        assert!(!sim.inject(Injection::ThrustLoss {
            factor: f64::INFINITY
        }));
        while sim.step(Controls::new(-1.0, 0.0, 0.0)).is_none() {}
        assert!(!sim.inject(Injection::Gust {
            amplitude: 5.0,
            duration: 1.0
        }));
    }

    #[test]
    fn soft_vertical_touchdown_on_the_pad_lands() {
        // Start just above the pad, upright, slow, engine near hover.
        let mut p = Params::easy();
        p.environment.rho0 = 0.0;
        p.wind.mean = Range::fixed(0.0);
        p.wind.turbulence_sigma = 0.0;
        p.start.altitude = Range::fixed(p.vehicle.length + p.vehicle.leg_half_span + 0.5);
        p.start.vertical_speed = Range::fixed(-1.0);
        p.start.horizontal_speed = Range::fixed(0.0);
        p.start.horizontal_offset = Range::fixed(0.0);
        p.start.tilt = Range::fixed(0.0);
        p.start.angular_rate = Range::fixed(0.0);
        let mut sim = Sim::new(p.clone(), 1).unwrap();
        let v = &p.vehicle;
        let mut outcome = None;
        for _ in 0..7_200 {
            // Simple descent-rate controller holding about -1 m/s.
            let s = *sim.state();
            let accel = 2.0 * (-1.0 - s.vy);
            let frac = (s.mass * (p.environment.gravity + accel) / v.max_thrust)
                .clamp(v.min_throttle, 1.0);
            let u = v.throttle_cutoff
                + (frac - v.min_throttle) / (1.0 - v.min_throttle) * (1.0 - v.throttle_cutoff);
            outcome = sim.step(Controls::new(u, 0.0, 0.0));
            if outcome.is_some() {
                break;
            }
        }
        assert_eq!(outcome, Some(Outcome::Landed), "{:?}", sim.contact());
        assert!(sim.score() > 1_000);
    }

    #[test]
    fn crash_reason_is_reported() {
        let mut sim = Sim::from_config(VehicleConfig::Realistic, 9);
        while sim.step(Controls::new(-1.0, 0.0, 0.0)).is_none() {}
        assert!(matches!(
            sim.outcome(),
            Some(Outcome::Crashed(
                CrashReason::VerticalSpeed | CrashReason::OffPad | CrashReason::TippedOver
            ))
        ));
    }
}
