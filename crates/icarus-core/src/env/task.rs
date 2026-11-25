//! Training tasks: the parameters an episode is drawn from.
//!
//! A [`Task`] is a validated parameter set (a vehicle configuration, an
//! evaluation scenario and named overrides on top) plus a [`Randomization`]
//! that perturbs it at every reset (domain randomization). The curriculum, in
//! Python, only ever builds tasks and hands them to the environments; the
//! per-step logic never leaves Rust.
//!
//! Randomization draws from its own random stream
//! ([`Stream::Randomization`]) and always makes the same number of draws, so
//! the initial conditions, wind, sensor noise and fault schedule of a seed are
//! the same with or without it (only the parameters they are drawn from
//! change).

use core::fmt;

use crate::eval::Scenario;
use crate::physics::params::{ParamError, Params, Range, VehicleConfig};
use crate::physics::sensors::{MAX_DELAY_TICKS, MEAS_DIM};
use crate::rng::{Rng, Stream};

/// Most ignitions an override may give.
const MAX_IGNITIONS: f64 = 100.0;

/// Largest multiplicative factor a randomization range may use.
pub const MAX_SCALE: f64 = 10.0;

/// An invalid task or environment configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskError(pub String);

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid task: {}", self.0)
    }
}

impl std::error::Error for TaskError {}

impl From<ParamError> for TaskError {
    fn from(e: ParamError) -> Self {
        TaskError(e.to_string())
    }
}

fn err<T>(msg: impl Into<String>) -> Result<T, TaskError> {
    Err(TaskError(msg.into()))
}

/// Per-episode multiplicative perturbations of the parameters.
///
/// Every field is a range of factors (`1.0` keeps the value); the default
/// changes nothing. Factors are drawn uniformly, once per episode.
#[derive(Clone, Debug, PartialEq)]
pub struct Randomization {
    /// Dry mass and dry inertia (the airframe gets heavier as a whole).
    pub dry_mass: Range,
    /// Both bounds of the initial propellant range.
    pub propellant: Range,
    /// Wind: the mean range, the turbulence and the gust amplitudes.
    pub wind: Range,
    /// Engine thrust time constant `τ`.
    pub thrust_time_constant: Range,
    /// Gimbal actuator time constant.
    pub gimbal_time_constant: Range,
    /// Sensor noise standard deviations.
    pub sensor_noise: Range,
    /// Sensor delay in ticks, drawn uniformly from `min..=max`; `None`
    /// keeps the task's delay.
    pub sensor_delay: Option<(usize, usize)>,
}

impl Default for Randomization {
    fn default() -> Self {
        Self {
            dry_mass: Range::fixed(1.0),
            propellant: Range::fixed(1.0),
            wind: Range::fixed(1.0),
            thrust_time_constant: Range::fixed(1.0),
            gimbal_time_constant: Range::fixed(1.0),
            sensor_noise: Range::fixed(1.0),
            sensor_delay: None,
        }
    }
}

/// How a randomization range is turned into one value.
#[derive(Clone, Copy)]
enum Pick {
    Min,
    Max,
}

impl Randomization {
    /// No randomization.
    pub fn none() -> Self {
        Self::default()
    }

    fn scale_ranges(&self) -> [(&'static str, Range); 6] {
        [
            ("dry_mass", self.dry_mass),
            ("propellant", self.propellant),
            ("wind", self.wind),
            ("thrust_time_constant", self.thrust_time_constant),
            ("gimbal_time_constant", self.gimbal_time_constant),
            ("sensor_noise", self.sensor_noise),
        ]
    }

    /// Checks that every factor range is finite, ordered and within
    /// `(0, MAX_SCALE]` (the wind and noise factors may reach 0), and that
    /// the delay range is ordered and supported.
    pub fn validate(&self) -> Result<(), TaskError> {
        for (name, r) in self.scale_ranges() {
            let may_be_zero = matches!(name, "wind" | "sensor_noise");
            let low_ok = if may_be_zero {
                r.min >= 0.0
            } else {
                r.min > 0.0
            };
            if !(r.min.is_finite() && r.max.is_finite() && low_ok && r.min <= r.max)
                || r.max > MAX_SCALE
            {
                let low = if may_be_zero { "[0" } else { "(0" };
                return err(format!(
                    "randomization {name} must be an ordered range of factors in \
                     {low}, {MAX_SCALE}], got [{}, {}]",
                    r.min, r.max
                ));
            }
        }
        if let Some((lo, hi)) = self.sensor_delay {
            if lo > hi || hi > MAX_DELAY_TICKS {
                return err(format!(
                    "randomization sensor_delay must satisfy min <= max <= {MAX_DELAY_TICKS}, \
                     got ({lo}, {hi})"
                ));
            }
        }
        Ok(())
    }

    /// Applies one draw per field (always seven draws, in field order).
    pub fn sample(&self, base: &Params, rng: &mut Rng) -> Params {
        let mut draw = |r: Range| r.sample(rng);
        let factors = [
            draw(self.dry_mass),
            draw(self.propellant),
            draw(self.wind),
            draw(self.thrust_time_constant),
            draw(self.gimbal_time_constant),
            draw(self.sensor_noise),
        ];
        let u = rng.uniform();
        let delay = self.sensor_delay.map(|(lo, hi)| {
            // `u < 1`, so the index is in `0..=hi-lo`; the cast truncates a
            // small non-negative integer-valued float.
            let span = (hi - lo + 1) as f64;
            (lo + libm::floor(u * span) as usize).min(hi)
        });
        Self::apply(base, factors, delay)
    }

    /// The parameters at one extreme of every range (for validation).
    fn extreme(&self, base: &Params, pick: Pick) -> Params {
        let choose = |r: Range| match pick {
            Pick::Min => r.min,
            Pick::Max => r.max,
        };
        let factors = self.scale_ranges().map(|(_, r)| choose(r));
        let delay = self.sensor_delay.map(|(lo, hi)| match pick {
            Pick::Min => lo,
            Pick::Max => hi,
        });
        Self::apply(base, factors, delay)
    }

    fn apply(base: &Params, f: [f64; 6], delay: Option<usize>) -> Params {
        let [dry, propellant, wind, thrust_tau, gimbal_tau, noise] = f;
        let mut p = base.clone();
        p.vehicle.dry_mass *= dry;
        p.vehicle.dry_inertia *= dry;
        p.start.propellant = scale_range(p.start.propellant, propellant);
        p.wind.mean = scale_range(p.wind.mean, wind);
        p.wind.turbulence_sigma *= wind;
        p.wind.gust_amplitude = scale_range(p.wind.gust_amplitude, wind);
        p.vehicle.thrust_time_constant *= thrust_tau;
        p.vehicle.gimbal_time_constant *= gimbal_tau;
        for s in &mut p.sensors.noise_sigma {
            *s *= noise;
        }
        if let Some(d) = delay {
            p.sensors.delay_ticks = d;
        }
        p
    }
}

fn scale_range(r: Range, k: f64) -> Range {
    Range::new(r.min * k, r.max * k)
}

/// A validated task: base parameters plus their randomization.
#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    params: Params,
    randomization: Randomization,
}

impl Task {
    /// A task from parameters and a randomization. Both are validated,
    /// including the parameters at the two extremes of every randomization
    /// range (each parameter is monotonic in its factor, so this covers every
    /// draw).
    pub fn new(params: Params, randomization: Randomization) -> Result<Self, TaskError> {
        params.validate()?;
        randomization.validate()?;
        for (pick, name) in [(Pick::Min, "lower"), (Pick::Max, "upper")] {
            randomization
                .extreme(&params, pick)
                .validate()
                .map_err(|e| TaskError(format!("at the {name} end of the randomization: {e}")))?;
        }
        Ok(Self {
            params,
            randomization,
        })
    }

    /// A built-in configuration with no randomization.
    pub fn from_config(config: VehicleConfig) -> Self {
        Self::new(config.params(), Randomization::none())
            .expect("built-in configurations are valid (checked by the tests)")
    }

    /// An evaluation scenario on a configuration, with no randomization.
    pub fn from_scenario(config: VehicleConfig, scenario: Scenario) -> Self {
        Self::new(scenario.params(config), Randomization::none())
            .expect("built-in scenarios are valid (checked by the tests)")
    }

    /// The base (unrandomized) parameters.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The randomization applied at every reset.
    pub fn randomization(&self) -> &Randomization {
        &self.randomization
    }

    /// The longest sensor delay an episode of this task can have (ticks).
    pub fn max_delay_ticks(&self) -> usize {
        self.randomization
            .sensor_delay
            .map_or(self.params.sensors.delay_ticks, |(_, hi)| hi)
    }

    /// The parameters of the episode with seed `seed`. Always valid.
    pub fn episode_params(&self, seed: u64) -> Params {
        let mut rng = Rng::for_stream(seed, Stream::Randomization);
        self.randomization.sample(&self.params, &mut rng)
    }
}

/// The names accepted by [`apply_override`], with the number of values each
/// takes. Ranges take `[min, max]`; angles are in radians.
pub const OVERRIDE_KEYS: [(&str, usize); 28] = [
    ("start.altitude", 2),
    ("start.vertical_speed", 2),
    ("start.horizontal_speed", 2),
    ("start.horizontal_offset", 2),
    ("start.tilt", 2),
    ("start.angular_rate", 2),
    ("start.propellant", 2),
    ("start.engine_on", 1),
    ("start.initial_throttle", 1),
    ("start.ignitions", 1),
    ("wind.mean", 2),
    ("wind.turbulence_sigma", 1),
    ("wind.gust_rate", 1),
    ("wind.gust_amplitude", 2),
    ("wind.gust_duration", 2),
    ("sensors.delay_ticks", 1),
    ("sensors.noise_sigma", MEAS_DIM),
    ("faults.probability", 1),
    ("faults.gimbal_stuck_share", 1),
    ("faults.trigger_time", 2),
    ("faults.stuck_angle", 2),
    ("faults.thrust_factor", 2),
    ("landing.pad_half_width", 1),
    ("landing.max_vertical_speed", 1),
    ("landing.max_horizontal_speed", 1),
    ("landing.max_tilt", 1),
    ("landing.max_angular_rate", 1),
    ("limits.max_time", 1),
];

/// Sets one named parameter. The caller validates the final parameter set
/// (with [`Task::new`]); this only checks the key and the number of values,
/// and that the values are finite.
pub fn apply_override(params: &mut Params, key: &str, values: &[f64]) -> Result<(), TaskError> {
    let Some(&(_, expected)) = OVERRIDE_KEYS.iter().find(|(k, _)| *k == key) else {
        let valid: Vec<&str> = OVERRIDE_KEYS.iter().map(|(k, _)| *k).collect();
        return err(format!(
            "unknown override {key:?}; expected one of {valid:?}"
        ));
    };
    if values.len() != expected {
        return err(format!(
            "override {key:?} takes {expected} value(s), got {}",
            values.len()
        ));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return err(format!("override {key:?} must be finite, got {values:?}"));
    }
    let range = || Range::new(values[0], values[1]);
    let one = values[0];
    match key {
        "start.altitude" => params.start.altitude = range(),
        "start.vertical_speed" => params.start.vertical_speed = range(),
        "start.horizontal_speed" => params.start.horizontal_speed = range(),
        "start.horizontal_offset" => params.start.horizontal_offset = range(),
        "start.tilt" => params.start.tilt = range(),
        "start.angular_rate" => params.start.angular_rate = range(),
        "start.propellant" => params.start.propellant = range(),
        "start.engine_on" => {
            if one != 0.0 && one != 1.0 {
                return err(format!("start.engine_on must be 0 or 1, got {one}"));
            }
            params.start.engine_on = one == 1.0;
        }
        "start.initial_throttle" => params.start.initial_throttle = one,
        "start.ignitions" => {
            if !(0.0..=MAX_IGNITIONS).contains(&one) || libm::floor(one) != one {
                return err(format!(
                    "start.ignitions must be an integer in 0..={MAX_IGNITIONS}, got {one}"
                ));
            }
            // A small non-negative integer: the cast is exact.
            params.start.ignitions = one as u32;
        }
        "wind.mean" => params.wind.mean = range(),
        "wind.turbulence_sigma" => params.wind.turbulence_sigma = one,
        "wind.gust_rate" => params.wind.gust_rate = one,
        "wind.gust_amplitude" => params.wind.gust_amplitude = range(),
        "wind.gust_duration" => params.wind.gust_duration = range(),
        "sensors.delay_ticks" => {
            if one < 0.0 || one > MAX_DELAY_TICKS as f64 || libm::floor(one) != one {
                return err(format!(
                    "sensors.delay_ticks must be an integer in 0..={MAX_DELAY_TICKS}, got {one}"
                ));
            }
            // An exact small non-negative integer: the cast is exact.
            params.sensors.delay_ticks = one as usize;
        }
        "sensors.noise_sigma" => params.sensors.noise_sigma.copy_from_slice(values),
        "faults.probability" => params.faults.probability = one,
        "faults.gimbal_stuck_share" => params.faults.gimbal_stuck_share = one,
        "faults.trigger_time" => params.faults.trigger_time = range(),
        "faults.stuck_angle" => params.faults.stuck_angle = range(),
        "faults.thrust_factor" => params.faults.thrust_factor = range(),
        "landing.pad_half_width" => params.landing.pad_half_width = one,
        "landing.max_vertical_speed" => params.landing.max_vertical_speed = one,
        "landing.max_horizontal_speed" => params.landing.max_horizontal_speed = one,
        "landing.max_tilt" => params.landing.max_tilt = one,
        "landing.max_angular_rate" => params.landing.max_angular_rate = one,
        "limits.max_time" => params.limits.max_time = one,
        _ => unreachable!("every key of OVERRIDE_KEYS is handled"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::params::DEG;

    fn wide() -> Randomization {
        Randomization {
            dry_mass: Range::new(0.9, 1.1),
            propellant: Range::new(0.8, 1.2),
            wind: Range::new(0.0, 1.5),
            thrust_time_constant: Range::new(0.8, 1.25),
            gimbal_time_constant: Range::new(0.8, 1.25),
            sensor_noise: Range::new(0.5, 2.0),
            sensor_delay: Some((0, 6)),
        }
    }

    #[test]
    fn every_override_key_is_handled() {
        for (key, n) in OVERRIDE_KEYS {
            let mut p = Params::easy();
            // Values that are valid for every key (as a range or a scalar).
            let values = vec![1.0; n];
            apply_override(&mut p, key, &values).unwrap_or_else(|e| panic!("{key}: {e}"));
        }
    }

    #[test]
    fn overrides_reject_bad_input() {
        let mut p = Params::easy();
        assert!(apply_override(&mut p, "start.nope", &[1.0, 2.0]).is_err());
        assert!(apply_override(&mut p, "start.altitude", &[1.0]).is_err());
        assert!(apply_override(&mut p, "wind.gust_rate", &[f64::NAN]).is_err());
        assert!(apply_override(&mut p, "sensors.delay_ticks", &[2.5]).is_err());
        assert!(apply_override(&mut p, "sensors.delay_ticks", &[13.0]).is_err());
        assert!(apply_override(&mut p, "sensors.delay_ticks", &[-1.0]).is_err());
        assert!(apply_override(&mut p, "start.engine_on", &[0.5]).is_err());
        assert!(apply_override(&mut p, "start.ignitions", &[1.5]).is_err());
        assert!(apply_override(&mut p, "start.ignitions", &[1000.0]).is_err());
        assert_eq!(p, Params::easy(), "rejected overrides change nothing");
    }

    #[test]
    fn overrides_set_the_named_field() {
        let mut p = Params::easy();
        apply_override(&mut p, "start.tilt", &[-0.1, 0.2]).unwrap();
        apply_override(&mut p, "sensors.delay_ticks", &[4.0]).unwrap();
        apply_override(&mut p, "faults.probability", &[0.25]).unwrap();
        assert_eq!(p.start.tilt, Range::new(-0.1, 0.2));
        assert_eq!(p.sensors.delay_ticks, 4);
        assert_eq!(p.faults.probability, 0.25);

        // The realistic vehicle with its engine already running and relights.
        let mut p = Params::realistic();
        apply_override(&mut p, "start.engine_on", &[1.0]).unwrap();
        apply_override(&mut p, "start.initial_throttle", &[0.4]).unwrap();
        apply_override(&mut p, "start.ignitions", &[3.0]).unwrap();
        assert!(p.start.engine_on);
        assert_eq!(p.start.ignitions, 3);
        assert!(Task::new(p, Randomization::none()).is_ok());
    }

    #[test]
    fn task_validation_checks_both_extremes() {
        // A thrust time constant small enough to break RK4 stability at the
        // lower end of its range.
        let mut r = Randomization::none();
        r.thrust_time_constant = Range::new(0.01, 1.0);
        let e = Task::new(Params::easy(), r).unwrap_err();
        assert!(e.0.contains("lower end"), "{e}");

        let mut r = Randomization::none();
        r.dry_mass = Range::new(0.0, 1.0);
        assert!(Task::new(Params::easy(), r).is_err(), "zero dry mass");

        let mut r = Randomization::none();
        r.sensor_delay = Some((3, 2));
        assert!(Task::new(Params::easy(), r).is_err(), "unordered delay");

        let mut r = Randomization::none();
        r.wind = Range::new(0.5, f64::INFINITY);
        assert!(Task::new(Params::easy(), r).is_err(), "infinite factor");

        let mut p = Params::easy();
        p.start.altitude = Range::new(10.0, 20.0);
        assert!(Task::new(p, Randomization::none()).is_err(), "base checked");

        assert!(Task::new(Params::realistic(), wide()).is_ok());
    }

    #[test]
    fn no_randomization_keeps_the_parameters() {
        let task = Task::from_config(VehicleConfig::Realistic);
        for seed in 0..20 {
            assert_eq!(task.episode_params(seed), Params::realistic());
        }
    }

    #[test]
    fn randomized_parameters_stay_in_range_and_are_valid() {
        let task = Task::new(Params::easy(), wide()).unwrap();
        let base = Params::easy();
        let mut delays = [false; 7];
        for seed in 0..2_000 {
            let p = task.episode_params(seed);
            p.validate().unwrap();
            let k = p.vehicle.dry_mass / base.vehicle.dry_mass;
            assert!((0.9..=1.1).contains(&k));
            let ki = p.vehicle.dry_inertia / base.vehicle.dry_inertia;
            assert!((ki - k).abs() < 1e-12, "inertia scales with the mass");
            let w = p.wind.turbulence_sigma / base.wind.turbulence_sigma;
            assert!((0.0..=1.5).contains(&w));
            assert!(p.sensors.delay_ticks <= 6);
            delays[p.sensors.delay_ticks] = true;
        }
        assert!(
            delays.iter().all(|&d| d),
            "every delay is drawn: {delays:?}"
        );
        assert_eq!(task.episode_params(7), task.episode_params(7));
        assert_ne!(task.episode_params(7), task.episode_params(8));
    }

    #[test]
    fn scenarios_and_configs_build_tasks() {
        for config in VehicleConfig::ALL {
            let _ = Task::from_config(config);
            for scenario in Scenario::ALL {
                let t = Task::from_scenario(config, scenario);
                assert_eq!(t.params(), &scenario.params(config));
            }
        }
        let t = Task::from_scenario(VehicleConfig::Easy, Scenario::InitialTilt);
        assert!(t.params().start.tilt.min >= 30.0 * DEG - 1e-12);
    }
}
