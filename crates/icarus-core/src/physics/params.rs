//! Vehicle, environment and scenario parameters.
//!
//! Two built-in configurations share the same airframe:
//!
//! - [`VehicleConfig::Easy`]: an engine weak enough to hover (minimum
//!   thrust-to-weight below 1), engine running at the start, several relights.
//! - [`VehicleConfig::Realistic`]: a first-stage class engine whose minimum
//!   thrust exceeds the vehicle weight at every reachable mass, so the stage
//!   cannot hover and must perform a single well-timed landing burn.

use core::fmt;

use crate::physics::sensors::{MAX_DELAY_TICKS, MEAS_DIM};
use crate::rng::Rng;

/// Degrees to radians.
pub const DEG: f64 = core::f64::consts::PI / 180.0;

/// Standard gravity used in the definition of specific impulse (m/s²).
pub const G0: f64 = 9.806_65;

/// Largest magnitude accepted for any range bound (keeps `max − min` finite).
pub const MAX_MAGNITUDE: f64 = 1.0e7;

/// Largest number of ticks an episode may last.
pub const MAX_TICKS: u64 = 10_000_000;

/// Built-in vehicle configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VehicleConfig {
    /// Can hover; forgiving.
    Easy,
    /// Cannot hover; hoverslam required.
    Realistic,
    /// Easy vehicle, low and slow, almost no wind: the first lesson.
    Training,
    /// Easy vehicle in strong crosswind and gusts, on a narrower pad.
    Crosswind,
    /// Realistic vehicle in a storm, with laggier sensors.
    Storm,
    /// Realistic vehicle with a guaranteed failure: stuck gimbal or thrust loss.
    EngineOut,
}

impl VehicleConfig {
    /// All configurations, in identifier order.
    pub const ALL: [VehicleConfig; 6] = [
        VehicleConfig::Easy,
        VehicleConfig::Realistic,
        VehicleConfig::Training,
        VehicleConfig::Crosswind,
        VehicleConfig::Storm,
        VehicleConfig::EngineOut,
    ];

    /// The campaign, in order of difficulty.
    pub const CAMPAIGN: [VehicleConfig; 6] = [
        VehicleConfig::Training,
        VehicleConfig::Easy,
        VehicleConfig::Crosswind,
        VehicleConfig::Realistic,
        VehicleConfig::Storm,
        VehicleConfig::EngineOut,
    ];

    /// Stable numeric identifier (used in replays).
    pub const fn id(self) -> u8 {
        match self {
            VehicleConfig::Easy => 0,
            VehicleConfig::Realistic => 1,
            VehicleConfig::Training => 2,
            VehicleConfig::Crosswind => 3,
            VehicleConfig::Storm => 4,
            VehicleConfig::EngineOut => 5,
        }
    }

    /// Inverse of [`VehicleConfig::id`].
    pub const fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(VehicleConfig::Easy),
            1 => Some(VehicleConfig::Realistic),
            2 => Some(VehicleConfig::Training),
            3 => Some(VehicleConfig::Crosswind),
            4 => Some(VehicleConfig::Storm),
            5 => Some(VehicleConfig::EngineOut),
            _ => None,
        }
    }

    /// Lower-case name (`"easy"`, `"realistic"`, `"training"`, ...).
    pub const fn name(self) -> &'static str {
        match self {
            VehicleConfig::Easy => "easy",
            VehicleConfig::Realistic => "realistic",
            VehicleConfig::Training => "training",
            VehicleConfig::Crosswind => "crosswind",
            VehicleConfig::Storm => "storm",
            VehicleConfig::EngineOut => "engine_out",
        }
    }

    /// Inverse of [`VehicleConfig::name`] (case-sensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == name)
    }

    /// The parameters of this configuration.
    pub fn params(self) -> Params {
        match self {
            VehicleConfig::Easy => Params::easy(),
            VehicleConfig::Realistic => Params::realistic(),
            VehicleConfig::Training => Params::training(),
            VehicleConfig::Crosswind => Params::crosswind(),
            VehicleConfig::Storm => Params::storm(),
            VehicleConfig::EngineOut => Params::engine_out(),
        }
    }
}

/// Closed interval sampled uniformly. Sampling always draws exactly one
/// random number, even when `min == max`, so the number of draws never
/// depends on parameter values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    /// Lower bound.
    pub min: f64,
    /// Upper bound.
    pub max: f64,
}

impl Range {
    /// Creates a range.
    pub const fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    /// A range containing a single value.
    pub const fn fixed(value: f64) -> Self {
        Self {
            min: value,
            max: value,
        }
    }

    /// A range symmetric around zero.
    pub const fn symmetric(half_width: f64) -> Self {
        Self {
            min: -half_width,
            max: half_width,
        }
    }

    /// Draws one uniform sample.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        let u = rng.uniform();
        self.min + (self.max - self.min) * u
    }

    /// Finite, ordered, and small enough that `max - min` cannot overflow.
    fn is_valid(&self) -> bool {
        self.min.is_finite()
            && self.max.is_finite()
            && self.min <= self.max
            && self.min.abs() <= MAX_MAGNITUDE
            && self.max.abs() <= MAX_MAGNITUDE
    }
}

/// Airframe, engine and actuator parameters.
///
/// Heights are measured along the body axis from the tail (engine gimbal).
#[derive(Clone, Debug, PartialEq)]
pub struct Vehicle {
    /// Overall length (m).
    pub length: f64,
    /// Body diameter (m).
    pub diameter: f64,
    /// Dry mass, including legs and fins (kg).
    pub dry_mass: f64,
    /// Height of the dry center of mass above the tail (m).
    pub dry_cm_height: f64,
    /// Pitch moment of inertia of the dry structure about its own center of
    /// mass (kg·m²).
    pub dry_inertia: f64,
    /// Height of the propellant center of mass above the tail (m).
    pub propellant_cm_height: f64,
    /// Radius of gyration of the propellant about its own center of mass (m).
    pub propellant_gyration_radius: f64,
    /// Maximum thrust (N).
    pub max_thrust: f64,
    /// Minimum throttle while running, as a fraction of `max_thrust`.
    pub min_throttle: f64,
    /// Specific impulse (s).
    pub isp: f64,
    /// Time constant of the thrust response (s).
    pub thrust_time_constant: f64,
    /// Gimbal deflection limit (rad).
    pub gimbal_max: f64,
    /// Time constant of the gimbal actuator (s).
    pub gimbal_time_constant: f64,
    /// Gimbal slew-rate limit (rad/s).
    pub gimbal_rate_max: f64,
    /// Maximum pure torque of the cold-gas attitude thrusters (N·m).
    pub rcs_torque: f64,
    /// Height of the center of pressure above the tail (m).
    pub cp_height: f64,
    /// Drag coefficient times reference area, flow along the body axis (m²).
    pub cda_axial: f64,
    /// Drag coefficient times reference area, flow across the body (m²).
    pub cda_side: f64,
    /// Aerodynamic pitch damping coefficient: torque `-c·ρ·|v_rel|·ω` (m⁴).
    pub pitch_damping: f64,
    /// Lateral distance from the body axis to each landing-leg tip (m).
    pub leg_half_span: f64,
    /// Throttle commands below this normalized value shut the engine down.
    pub throttle_cutoff: f64,
}

/// Planetary environment.
#[derive(Clone, Debug, PartialEq)]
pub struct Environment {
    /// Gravitational acceleration (m/s²).
    pub gravity: f64,
    /// Air density at sea level (kg/m³).
    pub rho0: f64,
    /// Density scale height (m).
    pub scale_height: f64,
}

/// Horizontal wind model: mean + Ornstein–Uhlenbeck turbulence + gusts.
#[derive(Clone, Debug, PartialEq)]
pub struct WindParams {
    /// Mean wind speed, drawn once per episode (m/s, positive = to the right).
    pub mean: Range,
    /// Stationary standard deviation of the turbulence (m/s).
    pub turbulence_sigma: f64,
    /// Correlation time of the turbulence (s).
    pub turbulence_time_constant: f64,
    /// Average number of discrete gusts per second.
    pub gust_rate: f64,
    /// Peak speed of a discrete gust (m/s, sign drawn at random).
    pub gust_amplitude: Range,
    /// Duration of a discrete gust (s).
    pub gust_duration: Range,
}

/// Navigation sensor model.
#[derive(Clone, Debug, PartialEq)]
pub struct SensorParams {
    /// Observation delay in ticks (at most [`MAX_DELAY_TICKS`]).
    pub delay_ticks: usize,
    /// Noise standard deviation for `[x, y, vx, vy, θ, ω]`.
    pub noise_sigma: [f64; MEAS_DIM],
}

/// Distribution of the initial conditions.
#[derive(Clone, Debug, PartialEq)]
pub struct StartParams {
    /// Center-of-mass altitude (m).
    pub altitude: Range,
    /// Vertical speed (m/s).
    pub vertical_speed: Range,
    /// Horizontal speed (m/s).
    pub horizontal_speed: Range,
    /// Horizontal offset from the pad (m).
    pub horizontal_offset: Range,
    /// Tilt (rad).
    pub tilt: Range,
    /// Angular rate (rad/s).
    pub angular_rate: Range,
    /// Propellant mass (kg).
    pub propellant: Range,
    /// Whether the engine is running at the start.
    pub engine_on: bool,
    /// Initial throttle (fraction of max thrust) when the engine starts on.
    pub initial_throttle: f64,
    /// Ignitions available (each start of a stopped engine spends one).
    pub ignitions: u32,
}

/// Random fault schedule (used for training and evaluation scenarios).
#[derive(Clone, Debug, PartialEq)]
pub struct FaultParams {
    /// Probability that an episode contains a fault.
    pub probability: f64,
    /// Probability that a fault is a stuck gimbal (otherwise thrust loss).
    pub gimbal_stuck_share: f64,
    /// Time at which the fault happens (s).
    pub trigger_time: Range,
    /// Magnitude of the stuck gimbal angle (rad, sign drawn at random).
    pub stuck_angle: Range,
    /// Remaining fraction of max thrust after a thrust loss.
    pub thrust_factor: Range,
}

/// Touchdown limits for a successful landing.
#[derive(Clone, Debug, PartialEq)]
pub struct LandingCriteria {
    /// Half-width of the landing pad (m). Both leg tips must be on it.
    pub pad_half_width: f64,
    /// Vertical speed limit of the contact point (m/s).
    pub max_vertical_speed: f64,
    /// Horizontal speed limit of the contact point (m/s).
    pub max_horizontal_speed: f64,
    /// Tilt limit (rad).
    pub max_tilt: f64,
    /// Angular rate limit (rad/s).
    pub max_angular_rate: f64,
}

/// Episode limits.
#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    /// Simulated time after which the episode is truncated (s).
    pub max_time: f64,
    /// Horizontal distance from the pad beyond which the episode ends (m).
    pub max_abs_x: f64,
    /// Altitude above which the episode ends (m).
    pub max_altitude: f64,
}

/// Complete simulation parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Integration time step (s).
    pub dt: f64,
    /// Airframe and engine.
    pub vehicle: Vehicle,
    /// Planet.
    pub environment: Environment,
    /// Wind.
    pub wind: WindParams,
    /// Sensors.
    pub sensors: SensorParams,
    /// Initial conditions.
    pub start: StartParams,
    /// Random faults.
    pub faults: FaultParams,
    /// Success criteria.
    pub landing: LandingCriteria,
    /// Episode limits.
    pub limits: Limits,
}

/// Invalid parameter set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamError(pub String);

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid parameters: {}", self.0)
    }
}

impl std::error::Error for ParamError {}

fn airframe(max_thrust: f64) -> Vehicle {
    let length = 42.0;
    let diameter = 3.7;
    let dry_mass = 25_000.0;
    Vehicle {
        length,
        diameter,
        dry_mass,
        dry_cm_height: 15.0,
        // Engines make the structure bottom-heavy: a bit less than the
        // uniform-rod value m·L²/12.
        dry_inertia: dry_mass * length * length / 14.0,
        propellant_cm_height: 10.0,
        propellant_gyration_radius: 3.0,
        max_thrust,
        min_throttle: 0.4,
        isp: 282.0,
        thrust_time_constant: 0.25,
        gimbal_max: 7.0 * DEG,
        gimbal_time_constant: 0.1,
        gimbal_rate_max: 20.0 * DEG,
        rcs_torque: 100_000.0,
        // Grid fins near the top put the center of pressure above the center
        // of mass: aerodynamically stable when falling tail-first.
        cp_height: 30.0,
        cda_axial: 10.0,
        cda_side: 1.2 * length * diameter,
        pitch_damping: 3_000.0,
        leg_half_span: 9.0,
        throttle_cutoff: -0.9,
    }
}

fn earth() -> Environment {
    Environment {
        gravity: 9.81,
        rho0: 1.225,
        scale_height: 8_500.0,
    }
}

fn no_faults() -> FaultParams {
    FaultParams {
        probability: 0.0,
        gimbal_stuck_share: 0.5,
        trigger_time: Range::new(5.0, 25.0),
        stuck_angle: Range::new(0.5 * DEG, 1.0 * DEG),
        thrust_factor: Range::fixed(0.8),
    }
}

fn landing(pad_half_width: f64) -> LandingCriteria {
    LandingCriteria {
        pad_half_width,
        max_vertical_speed: 2.0,
        max_horizontal_speed: 1.0,
        max_tilt: 5.0 * DEG,
        max_angular_rate: 0.1,
    }
}

const SENSOR_SIGMA: [f64; MEAS_DIM] = [0.5, 0.5, 0.1, 0.1, 0.2 * DEG, 0.005];

const LIMITS: Limits = Limits {
    max_time: 120.0,
    max_abs_x: 3_000.0,
    max_altitude: 4_000.0,
};

impl Params {
    /// Fixed physics time step: 60 ticks per second.
    pub const DT: f64 = 1.0 / 60.0;

    /// The `easy` configuration: can hover.
    pub fn easy() -> Self {
        Params {
            dt: Self::DT,
            vehicle: airframe(500_000.0),
            environment: earth(),
            wind: WindParams {
                mean: Range::symmetric(3.0),
                turbulence_sigma: 1.0,
                turbulence_time_constant: 3.0,
                gust_rate: 0.0,
                gust_amplitude: Range::new(2.0, 4.0),
                gust_duration: Range::new(2.0, 4.0),
            },
            sensors: SensorParams {
                delay_ticks: 0,
                noise_sigma: SENSOR_SIGMA,
            },
            start: StartParams {
                altitude: Range::new(700.0, 900.0),
                vertical_speed: Range::new(-45.0, -35.0),
                horizontal_speed: Range::symmetric(5.0),
                horizontal_offset: Range::symmetric(100.0),
                tilt: Range::symmetric(3.0 * DEG),
                angular_rate: Range::symmetric(0.01),
                propellant: Range::new(5_800.0, 6_200.0),
                engine_on: true,
                initial_throttle: 0.6,
                ignitions: 3,
            },
            faults: no_faults(),
            landing: landing(40.0),
            limits: LIMITS,
        }
    }

    /// The `realistic` configuration: cannot hover, engine off at the start,
    /// a single ignition.
    pub fn realistic() -> Self {
        Params {
            dt: Self::DT,
            vehicle: airframe(845_000.0),
            environment: earth(),
            wind: WindParams {
                mean: Range::symmetric(8.0),
                turbulence_sigma: 2.0,
                turbulence_time_constant: 3.0,
                gust_rate: 1.0 / 15.0,
                gust_amplitude: Range::new(4.0, 10.0),
                gust_duration: Range::new(2.0, 6.0),
            },
            sensors: SensorParams {
                delay_ticks: 3,
                noise_sigma: SENSOR_SIGMA,
            },
            start: StartParams {
                altitude: Range::new(1_400.0, 1_600.0),
                vertical_speed: Range::new(-100.0, -80.0),
                // A good boostback and entry leave the stage roughly aimed at
                // the pad: the landing burn only trims the remaining error.
                horizontal_speed: Range::symmetric(4.0),
                horizontal_offset: Range::symmetric(80.0),
                tilt: Range::symmetric(4.0 * DEG),
                angular_rate: Range::symmetric(0.02),
                propellant: Range::new(4_800.0, 5_200.0),
                engine_on: false,
                initial_throttle: 0.4,
                ignitions: 1,
            },
            faults: no_faults(),
            landing: landing(25.0),
            limits: LIMITS,
        }
    }

    /// The `training` configuration: the easy vehicle, lower, slower and
    /// almost windless, with a wide pad.
    pub fn training() -> Self {
        let mut p = Self::easy();
        p.wind.mean = Range::symmetric(1.0);
        p.wind.turbulence_sigma = 0.3;
        p.start.altitude = Range::new(400.0, 500.0);
        p.start.vertical_speed = Range::new(-25.0, -20.0);
        p.start.horizontal_speed = Range::symmetric(2.0);
        p.start.horizontal_offset = Range::symmetric(30.0);
        p.start.tilt = Range::symmetric(1.0 * DEG);
        p.start.angular_rate = Range::symmetric(0.005);
        p.landing.pad_half_width = 50.0;
        p
    }

    /// The `crosswind` configuration: the easy vehicle in strong wind and
    /// frequent gusts, further from a narrower pad.
    pub fn crosswind() -> Self {
        let mut p = Self::easy();
        p.wind.mean = Range::symmetric(10.0);
        p.wind.turbulence_sigma = 2.5;
        p.wind.gust_rate = 1.0 / 10.0;
        p.wind.gust_amplitude = Range::new(4.0, 8.0);
        p.wind.gust_duration = Range::new(2.0, 5.0);
        p.start.horizontal_offset = Range::symmetric(150.0);
        p.landing.pad_half_width = 30.0;
        p
    }

    /// The `storm` configuration: the realistic vehicle in a storm, with
    /// twice the sensor delay.
    pub fn storm() -> Self {
        let mut p = Self::realistic();
        p.wind.mean = Range::symmetric(14.0);
        p.wind.turbulence_sigma = 3.0;
        p.wind.gust_rate = 1.0 / 8.0;
        p.wind.gust_amplitude = Range::new(6.0, 12.0);
        p.wind.gust_duration = Range::new(2.0, 5.0);
        p.sensors.delay_ticks = 6;
        p
    }

    /// The `engine_out` configuration: the realistic vehicle with a failure
    /// in every episode, during the descent: a slightly stuck gimbal (to be
    /// cancelled with RCS) or a loss of thrust.
    pub fn engine_out() -> Self {
        let mut p = Self::realistic();
        p.faults = FaultParams {
            probability: 1.0,
            gimbal_stuck_share: 0.5,
            trigger_time: Range::new(4.0, 10.0),
            stuck_angle: Range::new(0.3 * DEG, 0.8 * DEG),
            thrust_factor: Range::new(0.82, 0.9),
        };
        p
    }

    /// Minimum thrust-to-weight ratio at a given total mass.
    pub fn twr_min(&self, mass: f64) -> f64 {
        self.vehicle.min_throttle * self.vehicle.max_thrust / (mass * self.environment.gravity)
    }

    /// Maximum thrust-to-weight ratio at a given total mass.
    pub fn twr_max(&self, mass: f64) -> f64 {
        self.vehicle.max_thrust / (mass * self.environment.gravity)
    }

    /// Number of ticks after which the episode is truncated.
    pub fn max_ticks(&self) -> u64 {
        // Both operands are positive and validated; the cast saturates.
        libm::ceil(self.limits.max_time / self.dt - 1e-9) as u64
    }

    /// Checks that every parameter is finite and physically meaningful.
    pub fn validate(&self) -> Result<(), ParamError> {
        fn check(ok: bool, what: &str) -> Result<(), ParamError> {
            if ok {
                Ok(())
            } else {
                Err(ParamError(what.to_owned()))
            }
        }
        fn pos(x: f64) -> bool {
            x.is_finite() && x > 0.0
        }
        fn nonneg(x: f64) -> bool {
            x.is_finite() && x >= 0.0
        }
        fn unit(x: f64) -> bool {
            x.is_finite() && (0.0..=1.0).contains(&x)
        }

        check(pos(self.dt) && self.dt <= 0.1, "dt must be in (0, 0.1]")?;
        check(
            self.limits.max_time.is_finite() && self.limits.max_time / self.dt <= MAX_TICKS as f64,
            "max_time / dt exceeds the tick limit",
        )?;

        let v = &self.vehicle;
        check(
            pos(v.length) && pos(v.diameter),
            "length and diameter must be positive",
        )?;
        check(
            pos(v.dry_mass) && pos(v.dry_inertia),
            "dry mass and inertia must be positive",
        )?;
        for (h, name) in [
            (v.dry_cm_height, "dry_cm_height"),
            (v.propellant_cm_height, "propellant_cm_height"),
            (v.cp_height, "cp_height"),
        ] {
            check(
                nonneg(h) && h <= v.length,
                &format!("{name} must be within the vehicle length"),
            )?;
        }
        check(
            nonneg(v.propellant_gyration_radius),
            "propellant gyration radius must be >= 0",
        )?;
        check(
            pos(v.max_thrust) && pos(v.isp),
            "max thrust and Isp must be positive",
        )?;
        check(
            v.min_throttle.is_finite() && v.min_throttle > 0.0 && v.min_throttle <= 1.0,
            "min_throttle must be in (0, 1]",
        )?;
        check(
            pos(v.thrust_time_constant) && pos(v.gimbal_time_constant),
            "actuator time constants must be positive",
        )?;
        // Explicit RK4 on a first-order lag is stable for dt/τ < ~2.78.
        check(
            self.dt / v.thrust_time_constant < 2.5 && self.dt / v.gimbal_time_constant < 2.5,
            "actuator time constants too small for dt (RK4 stability)",
        )?;
        check(
            pos(v.gimbal_max) && v.gimbal_max < 30.0 * DEG,
            "gimbal_max must be in (0, 30°)",
        )?;
        check(pos(v.gimbal_rate_max), "gimbal rate limit must be positive")?;
        check(nonneg(v.rcs_torque), "rcs torque must be >= 0")?;
        check(
            nonneg(v.cda_axial) && nonneg(v.cda_side),
            "drag areas must be >= 0",
        )?;
        check(nonneg(v.pitch_damping), "pitch damping must be >= 0")?;
        check(nonneg(v.leg_half_span), "leg half span must be >= 0")?;
        check(
            v.throttle_cutoff.is_finite() && v.throttle_cutoff > -1.0 && v.throttle_cutoff < 1.0,
            "throttle cutoff must be in (-1, 1)",
        )?;

        let e = &self.environment;
        check(
            pos(e.gravity) && nonneg(e.rho0) && pos(e.scale_height),
            "invalid environment",
        )?;

        let w = &self.wind;
        check(w.mean.is_valid(), "invalid wind mean range")?;
        check(
            nonneg(w.turbulence_sigma) && pos(w.turbulence_time_constant),
            "invalid turbulence",
        )?;
        check(
            nonneg(w.gust_rate) && w.gust_rate * self.dt <= 1.0,
            "invalid gust rate",
        )?;
        check(
            w.gust_amplitude.is_valid() && w.gust_amplitude.min >= 0.0,
            "gust amplitude must be a valid range of magnitudes",
        )?;
        check(
            w.gust_duration.is_valid() && w.gust_duration.min > 0.0,
            "gust duration must be a valid positive range",
        )?;

        let s = &self.sensors;
        check(
            s.delay_ticks <= MAX_DELAY_TICKS,
            "sensor delay exceeds MAX_DELAY_TICKS",
        )?;
        check(
            s.noise_sigma.iter().all(|&x| nonneg(x)),
            "sensor noise must be >= 0",
        )?;

        let st = &self.start;
        for (r, name) in [
            (st.altitude, "altitude"),
            (st.vertical_speed, "vertical_speed"),
            (st.horizontal_speed, "horizontal_speed"),
            (st.horizontal_offset, "horizontal_offset"),
            (st.tilt, "tilt"),
            (st.angular_rate, "angular_rate"),
            (st.propellant, "propellant"),
        ] {
            check(r.is_valid(), &format!("invalid start range {name}"))?;
        }
        check(
            st.altitude.min > v.length + v.leg_half_span,
            "start altitude must clear the ground",
        )?;
        check(
            st.altitude.max < self.limits.max_altitude,
            "start altitude above the altitude limit",
        )?;
        check(
            st.horizontal_offset.min.abs() < self.limits.max_abs_x
                && st.horizontal_offset.max.abs() < self.limits.max_abs_x,
            "start offset outside the horizontal limit",
        )?;
        check(
            st.tilt.min > -45.0 * DEG && st.tilt.max < 45.0 * DEG,
            "start tilt must be within ±45°",
        )?;
        check(st.propellant.min >= 0.0, "start propellant must be >= 0")?;
        check(
            !st.engine_on || (st.initial_throttle >= v.min_throttle && st.initial_throttle <= 1.0),
            "initial throttle must be in [min_throttle, 1]",
        )?;

        let f = &self.faults;
        check(
            unit(f.probability) && unit(f.gimbal_stuck_share),
            "fault probabilities must be in [0, 1]",
        )?;
        check(
            f.trigger_time.is_valid() && f.trigger_time.min >= 0.0,
            "invalid fault trigger time",
        )?;
        check(
            f.stuck_angle.is_valid()
                && f.stuck_angle.min >= 0.0
                && f.stuck_angle.max <= v.gimbal_max,
            "stuck angle must be within [0, gimbal_max]",
        )?;
        check(
            f.thrust_factor.is_valid() && f.thrust_factor.min >= 0.0 && f.thrust_factor.max <= 1.0,
            "thrust factor must be within [0, 1]",
        )?;

        let l = &self.landing;
        check(
            pos(l.pad_half_width)
                && pos(l.max_vertical_speed)
                && pos(l.max_horizontal_speed)
                && pos(l.max_tilt)
                && pos(l.max_angular_rate),
            "landing criteria must be positive",
        )?;

        let lim = &self.limits;
        check(
            pos(lim.max_time) && pos(lim.max_abs_x) && pos(lim.max_altitude),
            "limits must be positive",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_configs_are_valid() {
        for c in VehicleConfig::ALL {
            c.params()
                .validate()
                .unwrap_or_else(|e| panic!("{}: {e}", c.name()));
        }
    }

    #[test]
    fn config_ids_and_names_round_trip() {
        for c in VehicleConfig::ALL {
            assert_eq!(VehicleConfig::from_id(c.id()), Some(c));
            assert_eq!(VehicleConfig::from_name(c.name()), Some(c));
        }
        assert_eq!(VehicleConfig::from_id(9), None);
        assert_eq!(VehicleConfig::from_name("hard"), None);
    }

    #[test]
    fn easy_can_hover_at_every_reachable_mass() {
        let p = Params::easy();
        let lightest = p.vehicle.dry_mass;
        let heaviest = p.vehicle.dry_mass + p.start.propellant.max;
        assert!(p.twr_min(lightest) < 1.0 && p.twr_min(heaviest) < 1.0);
        assert!(
            p.twr_max(heaviest) > 1.3,
            "needs margin to decelerate when full"
        );
    }

    #[test]
    fn campaign_levels_keep_their_vehicle_class() {
        for c in [VehicleConfig::Training, VehicleConfig::Crosswind] {
            let p = c.params();
            let heaviest = p.vehicle.dry_mass + p.start.propellant.max;
            assert!(p.twr_min(heaviest) < 1.0, "{} must hover", c.name());
        }
        for c in [VehicleConfig::Storm, VehicleConfig::EngineOut] {
            let p = c.params();
            assert!(
                p.twr_min(p.vehicle.dry_mass) > 1.0,
                "{} must not hover",
                c.name()
            );
        }
        assert_eq!(VehicleConfig::EngineOut.params().faults.probability, 1.0);
        // Engine-out must stay survivable: thrust still beats weight when full.
        let p = Params::engine_out();
        let heaviest = p.vehicle.dry_mass + p.start.propellant.max;
        assert!(p.twr_max(heaviest) * p.faults.thrust_factor.min > 1.5);
        let mut ids: Vec<u8> = VehicleConfig::CAMPAIGN.iter().map(|c| c.id()).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            (0..6).collect::<Vec<u8>>(),
            "campaign covers every config once"
        );
    }

    #[test]
    fn realistic_cannot_hover_at_any_reachable_mass() {
        let p = Params::realistic();
        let lightest = p.vehicle.dry_mass;
        let heaviest = p.vehicle.dry_mass + p.start.propellant.max;
        assert!(p.twr_min(heaviest) > 1.0 && p.twr_min(lightest) > 1.0);
    }

    #[test]
    fn validation_rejects_bad_values() {
        let mut p = Params::easy();
        p.dt = f64::NAN;
        assert!(p.validate().is_err());

        let mut p = Params::easy();
        p.vehicle.thrust_time_constant = 0.001;
        assert!(p.validate().is_err(), "RK4 stability check");

        let mut p = Params::easy();
        p.sensors.delay_ticks = MAX_DELAY_TICKS + 1;
        assert!(p.validate().is_err());

        let mut p = Params::easy();
        p.start.altitude = Range::new(10.0, 20.0);
        assert!(p.validate().is_err(), "must start above the ground");

        let mut p = Params::easy();
        p.wind.mean = Range::new(-1.7e308, 1.7e308);
        assert!(p.validate().is_err(), "range width would overflow");

        let mut p = Params::easy();
        p.limits.max_time = 1.0e12;
        assert!(p.validate().is_err(), "too many ticks");

        let mut p = Params::easy();
        p.faults.stuck_angle = Range::new(0.0, 10.0 * DEG);
        assert!(p.validate().is_err(), "stuck angle beyond the gimbal range");
    }

    #[test]
    fn max_ticks_matches_time_limit() {
        assert_eq!(Params::easy().max_ticks(), 7_200);
    }
}
