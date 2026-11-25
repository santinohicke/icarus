//! The reinforcement-learning environment: observation, reward, termination.
//!
//! It lives in the core so that training (Python) and the web (WebAssembly)
//! share it: Python never computes per-step logic.
//!
//! - **Decisions.** The agent acts every [`EnvConfig::action_repeat`] physics
//!   ticks (3 by default: 20 Hz over the 60 Hz simulation); each action is
//!   held for those ticks. Actions are the normalized controls of the
//!   simulator (`[throttle, gimbal, rcs]` in `[-1, 1]`) and, by default, are
//!   quantized to the same 255 levels as human input, so an agent's flight is
//!   an ordinary replay.
//! - **Observation** ([`obs`]): what a flight computer knows. The delayed,
//!   noisy navigation sensors (position and velocity relative to the pad,
//!   `sin θ`, `cos θ`, `ω`), the propellant gauge, the actual gimbal angle and
//!   thrust, engine on/off, ignitions left, the sensor delay `k`, the
//!   nominal thrust-to-weight ratio, and the last actions. The width is
//!   fixed while the delay changes per episode; only the actions taken within
//!   the delay window are shown (the rest are zero), which is what makes a
//!   delayed observation Markov again.
//! - **Reward** ([`RewardParams`]): potential-based shaping
//!   `F = γ·Φ(s') − Φ(s)` with a bounded `Φ ≤ 0` (distance, altitude,
//!   excess speed over a braking profile, tilt and rate), plus a landing
//!   bonus, a crash penalty larger than `max |Φ|`, a small cost per kilogram
//!   of propellant and a cost per second of flight (see
//!   [`RewardParams::time_weight`] for why hovering must never pay). A
//!   terminal state keeps its potential by default (see
//!   [`RewardParams::zero_terminal_potential`] for why), and so does the
//!   state at a time-limit truncation (the learner bootstraps from it).
//! - **Termination** follows the simulator's outcomes: the time limit is a
//!   truncation; every other outcome is terminal.
//!
//! Everything uses `libm`, so the environment is bit-identical natively and
//! in WebAssembly.

pub mod task;

pub use task::{OVERRIDE_KEYS, Randomization, Task, TaskError, apply_override};

use core::fmt;

use crate::physics::dynamics::mass_properties;
use crate::physics::params::Params;
use crate::physics::scenario::{Outcome, wrap_angle};
use crate::physics::sensors::MAX_DELAY_TICKS;
use crate::physics::state::{Controls, State, dequantize, quantize};
use crate::rng::SplitMix64;
use crate::sim::Sim;

/// Number of action components: throttle, gimbal, RCS.
pub const ACTION_DIM: usize = 3;

/// Longest supported action history (decisions): enough for the longest
/// sensor delay even at one tick per decision.
pub const MAX_HISTORY: usize = MAX_DELAY_TICKS;

/// Smallest accepted potential scale (keeps `|x| / scale` finite).
pub const MIN_SCALE: f64 = 1e-6;

/// Default action history: enough decisions to cover the longest sensor
/// delay at the default action repeat (`⌈12 / 3⌉ = 4`).
pub const DEFAULT_HISTORY: usize = MAX_DELAY_TICKS.div_ceil(3);

/// Indices of the observation vector. The action history follows
/// [`obs::HISTORY`], most recent decision first, `ACTION_DIM` values each.
pub mod obs {
    /// Measured horizontal offset from the pad center (/100 m).
    pub const X: usize = 0;
    /// Measured height of the tail above the ground, from the measured
    /// altitude, tilt and the gauge's center of mass (/100 m).
    pub const HEIGHT: usize = 1;
    /// Measured horizontal velocity (/10 m/s).
    pub const VX: usize = 2;
    /// Measured vertical velocity (/10 m/s).
    pub const VY: usize = 3;
    /// `sin θ` of the measured tilt.
    pub const SIN_THETA: usize = 4;
    /// `cos θ` of the measured tilt.
    pub const COS_THETA: usize = 5;
    /// Measured angular velocity (/0.1 rad/s).
    pub const OMEGA: usize = 6;
    /// Propellant left, as a fraction of the base task's largest initial
    /// load (domain randomization can push it above 1).
    pub const PROPELLANT: usize = 7;
    /// Actual gimbal angle as a fraction of the gimbal range.
    pub const GIMBAL: usize = 8;
    /// Actual thrust as a fraction of max thrust.
    pub const THRUST: usize = 9;
    /// 1 if the engine is running.
    pub const ENGINE_ON: usize = 10;
    /// Ignitions left.
    pub const IGNITIONS: usize = 11;
    /// Sensor delay as a fraction of the longest supported delay.
    pub const DELAY: usize = 12;
    /// Nominal maximum thrust-to-weight ratio at the gauge's mass (max
    /// thrust over the flight computer's own mass estimate; a thrust loss
    /// is not known to it). It tells the vehicles apart: without it one
    /// memoryless policy had to fly an engine that can hover and one that
    /// cannot from identical observations, and learning the second erased
    /// the first.
    pub const TWR: usize = 13;
    /// First value of the action history.
    pub const HISTORY: usize = 14;
}

/// Observation width without the action history.
pub const BASE_OBS_DIM: usize = obs::HISTORY;

const POSITION_UNIT: f64 = 100.0;
const VELOCITY_UNIT: f64 = 10.0;
const RATE_UNIT: f64 = 0.1;

/// Reward shaping and terminal rewards.
#[derive(Clone, Debug, PartialEq)]
pub struct RewardParams {
    /// Discount factor of the learner, per decision. The shaping
    /// `γ·Φ(s') − Φ(s)` is policy-invariant only with the learner's `γ`. The
    /// default, 0.999 at 20 Hz, is a horizon of about 50 s: it must cover a whole
    /// landing (20–60 s), or the landing bonus arrives discounted to nothing
    /// and hovering, crashing and landing look alike to the learner.
    pub gamma: f64,
    /// Reward for landing.
    pub landing_bonus: f64,
    /// Penalty for any other terminal outcome. Must exceed the sum of the
    /// potential weights (`max |Φ|`), so crashing early never pays.
    pub crash_penalty: f64,
    /// Cost of burning the task's largest initial propellant load.
    pub fuel_weight: f64,
    /// Cost per second of flight. It sets which way an agent that cannot land
    /// yet is pushed. Comparing "crash now" with "one more decision, then
    /// crash", the difference is `(1 − γ)·C − c`, plus at most
    /// `(1 − γ)·max|Φ|` from the shaping (`C` the crash penalty, `c` the cost
    /// per decision):
    ///
    /// - `c < (1 − γ)·C`: postponing pays; training runs learned to hover or
    ///   climb until the propellant ran out;
    /// - `c > (1 − γ)·C`: crashing early pays; a run learned to cut the
    ///   engine at once;
    /// - `c = (1 − γ)·C` (the default): the timing of a crash is neutral up to
    ///   the shaping's own `(1 − γ)·|Φ|` (0.065 at most against a crash penalty
    ///   of 100), and
    ///   only the terminal state's potential (contact speed, tilt, distance)
    ///   ranks failures, which leads towards softer, upright touchdowns on the
    ///   pad, i.e. towards landing. Hovering still costs `(1 − γ)·(C − |Φ|)`
    ///   per decision, since `|Φ| < C`.
    ///
    /// Like the fuel cost, it is part of the objective, not of the shaping.
    pub time_weight: f64,
    /// Weight of the distance to the pad.
    pub position_weight: f64,
    /// Distance at which the position term is half its weight (m).
    pub position_scale: f64,
    /// Weight of the height above the ground.
    pub altitude_weight: f64,
    /// Height at which the altitude term is half its weight (m).
    pub altitude_scale: f64,
    /// Weight of the speed error: horizontal speed away from the approach
    /// reference, climbing, and descent faster than the braking profile.
    pub velocity_weight: f64,
    /// Speed error at which the velocity term is half its weight (m/s).
    pub velocity_scale: f64,
    /// Lateral approach reference `vx_ref = −clamp(gain·x, ±max)`: moving
    /// towards the pad at that speed costs nothing (1/s). Without it any
    /// sideways speed was penalized, even towards the pad, and a trained
    /// agent touched down softly but wherever it happened to be.
    pub lateral_gain: f64,
    /// Largest lateral approach speed of the reference (m/s).
    pub lateral_speed_max: f64,
    /// Weight of the tilt.
    pub tilt_weight: f64,
    /// Tilt at which the tilt term is half its weight (rad).
    pub tilt_scale: f64,
    /// Weight of the angular rate.
    pub rate_weight: f64,
    /// Rate at which the rate term is half its weight (rad/s).
    pub rate_scale: f64,
    /// Braking profile speed at the ground, as a fraction of the landing
    /// limit: `v_ref(h) = √(v_td² + 2·a·h)`.
    pub touchdown_speed_fraction: f64,
    /// Braking profile deceleration `a`, as a fraction of the net
    /// deceleration at full thrust of the fully loaded vehicle.
    pub decel_fraction: f64,
    /// Use `Φ(terminal) = 0` (the textbook convention, which keeps the
    /// optimal policy exactly invariant). Off by default: the terminal
    /// decision then refunds `|Φ(s)|`, so crashing far from the pad pays more
    /// at that step than crashing next to it (−56 against −98 here), and only
    /// exact long-horizon credit assignment undoes it. A PPO run with it
    /// learned to climb at full throttle to 2.8 km and run out of propellant.
    /// Keeping `Φ(s_T)` makes the shaping sum `γ^T·Φ(s_T) − Φ(s_0)`: ending
    /// close to the target is worth more, which only ranks failures (a
    /// landing still dominates through its bonus).
    pub zero_terminal_potential: bool,
}

impl Default for RewardParams {
    fn default() -> Self {
        Self {
            gamma: 0.999,
            landing_bonus: 100.0,
            crash_penalty: 100.0,
            fuel_weight: 10.0,
            // (1 − 0.999) · 100 per decision, at 20 Hz = 2 per second.
            time_weight: 2.0,
            position_weight: 25.0,
            position_scale: 50.0,
            altitude_weight: 10.0,
            altitude_scale: 500.0,
            velocity_weight: 20.0,
            velocity_scale: 10.0,
            lateral_gain: 0.1,
            lateral_speed_max: 5.0,
            tilt_weight: 5.0,
            tilt_scale: 10.0 * crate::physics::params::DEG,
            rate_weight: 5.0,
            rate_scale: 0.2,
            touchdown_speed_fraction: 0.5,
            decel_fraction: 0.5,
            zero_terminal_potential: false,
        }
    }
}

impl RewardParams {
    /// Sum of the potential weights: `|Φ|` is always strictly below it.
    pub fn max_abs_potential(&self) -> f64 {
        self.position_weight
            + self.altitude_weight
            + self.velocity_weight
            + self.tilt_weight
            + self.rate_weight
    }

    fn validate(&self) -> Result<(), TaskError> {
        let nonneg = |x: f64| x.is_finite() && x >= 0.0;
        let pos = |x: f64| x.is_finite() && x > 0.0;
        let check = |ok: bool, what: &str| {
            if ok {
                Ok(())
            } else {
                Err(TaskError(format!("reward: {what}")))
            }
        };
        check(
            self.gamma.is_finite() && self.gamma > 0.0 && self.gamma <= 1.0,
            "gamma must be in (0, 1]",
        )?;
        check(
            nonneg(self.landing_bonus) && nonneg(self.fuel_weight) && nonneg(self.time_weight),
            "landing bonus, fuel weight and time weight must be >= 0",
        )?;
        for (w, s, name) in [
            (self.position_weight, self.position_scale, "position"),
            (self.altitude_weight, self.altitude_scale, "altitude"),
            (self.velocity_weight, self.velocity_scale, "velocity"),
            (self.tilt_weight, self.tilt_scale, "tilt"),
            (self.rate_weight, self.rate_scale, "rate"),
        ] {
            check(
                nonneg(w) && s.is_finite() && s >= MIN_SCALE,
                &format!("{name} weight must be >= 0 and its scale >= {MIN_SCALE}"),
            )?;
        }
        check(
            self.crash_penalty.is_finite() && self.crash_penalty > self.max_abs_potential(),
            "crash penalty must exceed the sum of the potential weights",
        )?;
        check(
            nonneg(self.lateral_gain) && nonneg(self.lateral_speed_max),
            "lateral gain and speed limit must be >= 0",
        )?;
        check(
            pos(self.touchdown_speed_fraction) && self.touchdown_speed_fraction <= 1.0,
            "touchdown speed fraction must be in (0, 1]",
        )?;
        check(
            pos(self.decel_fraction) && self.decel_fraction <= 1.0,
            "decel fraction must be in (0, 1]",
        )
    }

    /// Braking-profile deceleration for a parameter set (m/s²), never below
    /// 0.5 m/s².
    pub fn profile_decel(&self, p: &Params) -> f64 {
        let heaviest = p.vehicle.dry_mass + p.start.propellant.max;
        let net = p.vehicle.max_thrust / heaviest - p.environment.gravity;
        (self.decel_fraction * net).max(0.5)
    }

    /// The potential `Φ(s) ∈ (−max_abs_potential, 0]` of a (true) state.
    pub fn potential(&self, p: &Params, decel: f64, s: &State) -> f64 {
        fn sat(u: f64) -> f64 {
            // `u ≥ 0`; capped so an extreme state never gives ∞/∞ = NaN.
            let u = u.min(1e12);
            u / (1.0 + u)
        }
        let cm = mass_properties(&p.vehicle, s.mass).cm_height;
        let height = (s.y - cm * libm::cos(s.theta)).max(0.0);
        let v_td = self.touchdown_speed_fraction * p.landing.max_vertical_speed;
        let v_ref = libm::sqrt(v_td * v_td + 2.0 * decel * height);
        let excess = (-s.vy - v_ref).max(0.0) + s.vy.max(0.0);
        let vx_ref =
            -(self.lateral_gain * s.x).clamp(-self.lateral_speed_max, self.lateral_speed_max);
        let lateral = s.vx - vx_ref;
        let speed_error = libm::sqrt(lateral * lateral + excess * excess);
        let tilt = wrap_angle(s.theta).abs();
        -(self.position_weight * sat(s.x.abs() / self.position_scale)
            + self.altitude_weight * sat(height / self.altitude_scale)
            + self.velocity_weight * sat(speed_error / self.velocity_scale)
            + self.tilt_weight * sat(tilt / self.tilt_scale)
            + self.rate_weight * sat(s.omega.abs() / self.rate_scale))
    }
}

/// Environment configuration (shared by every episode).
#[derive(Clone, Debug, PartialEq)]
pub struct EnvConfig {
    /// Physics ticks per decision (each action is held this long).
    pub action_repeat: u32,
    /// Decisions of action history in the observation (fixed width).
    pub history_len: usize,
    /// Show the action history (`false` zeroes it: an ablation).
    pub action_history: bool,
    /// Quantize actions to the 255 levels of human input (replayable).
    pub quantize_actions: bool,
    /// Reward parameters.
    pub reward: RewardParams,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            action_repeat: 3,
            history_len: DEFAULT_HISTORY,
            action_history: true,
            quantize_actions: true,
            reward: RewardParams::default(),
        }
    }
}

impl EnvConfig {
    /// Checks the configuration.
    pub fn validate(&self) -> Result<(), TaskError> {
        if !(1..=60).contains(&self.action_repeat) {
            return Err(TaskError(format!(
                "action_repeat must be in 1..=60, got {}",
                self.action_repeat
            )));
        }
        if self.history_len > MAX_HISTORY {
            return Err(TaskError(format!(
                "history_len must be at most {MAX_HISTORY}, got {}",
                self.history_len
            )));
        }
        self.reward.validate()
    }

    /// Decisions of history needed to cover every delay `task` can draw
    /// (0 without the history).
    pub fn required_history(&self, task: &Task) -> usize {
        if self.action_history {
            task.max_delay_ticks().div_ceil(self.action_repeat as usize)
        } else {
            0
        }
    }

    /// Checks that the history covers every delay of `task`: otherwise the
    /// delayed observation would silently stop being Markov.
    pub fn check_task(&self, task: &Task) -> Result<(), TaskError> {
        let needed = self.required_history(task);
        if self.history_len < needed {
            return Err(TaskError(format!(
                "history_len {} does not cover a {}-tick sensor delay at {} ticks per decision \
                 (needs {needed})",
                self.history_len,
                task.max_delay_ticks(),
                self.action_repeat
            )));
        }
        Ok(())
    }

    /// Width of the observation vector.
    pub const fn obs_dim(&self) -> usize {
        BASE_OBS_DIM + ACTION_DIM * self.history_len
    }
}

/// Misuse of an environment, or an invalid configuration.
#[derive(Clone, Debug, PartialEq)]
pub enum EnvError {
    /// Invalid task or configuration.
    Invalid(TaskError),
    /// `step` before the first `reset`.
    NotReset,
    /// `step` after the episode ended (call `reset`).
    EpisodeOver,
    /// An action component was NaN or infinite.
    NonFiniteAction([f64; ACTION_DIM]),
    /// An observation buffer of the wrong width.
    BadBuffer {
        /// Required width.
        expected: usize,
        /// Width given.
        got: usize,
    },
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnvError::Invalid(e) => write!(f, "{e}"),
            EnvError::NotReset => write!(f, "step() called before reset()"),
            EnvError::EpisodeOver => write!(f, "step() called after the episode ended"),
            EnvError::NonFiniteAction(a) => write!(f, "action must be finite, got {a:?}"),
            EnvError::BadBuffer { expected, got } => {
                write!(
                    f,
                    "observation buffer must hold {expected} values, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for EnvError {}

impl From<TaskError> for EnvError {
    fn from(e: TaskError) -> Self {
        EnvError::Invalid(e)
    }
}

/// What one decision produced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step {
    /// Reward of the decision.
    pub reward: f64,
    /// The episode reached a terminal state.
    pub terminated: bool,
    /// The episode hit the time limit.
    pub truncated: bool,
    /// The outcome, once the episode has ended.
    pub outcome: Option<Outcome>,
}

impl Step {
    /// The episode ended (terminated or truncated).
    pub const fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

#[derive(Clone, Debug)]
struct Episode {
    sim: Sim,
    seed: u64,
    potential: f64,
    decel: f64,
    history: [[f64; ACTION_DIM]; MAX_HISTORY],
    episode_return: f64,
    decisions: u32,
}

/// One environment: a task, a configuration and the current episode.
#[derive(Clone, Debug)]
pub struct Env {
    config: EnvConfig,
    task: Task,
    next_task: Option<Task>,
    reference_propellant: f64,
    episode: Option<Episode>,
}

impl Env {
    /// Creates an environment. Call [`Env::reset`] before stepping.
    pub fn new(config: EnvConfig, task: Task) -> Result<Self, EnvError> {
        config.validate()?;
        config.check_task(&task)?;
        let reference_propellant = reference_propellant(task.params());
        Ok(Self {
            config,
            task,
            next_task: None,
            reference_propellant,
            episode: None,
        })
    }

    /// The configuration.
    pub fn config(&self) -> &EnvConfig {
        &self.config
    }

    /// The task episodes are drawn from (a task set with [`Env::set_task`]
    /// takes over at the next reset).
    pub fn task(&self) -> &Task {
        self.next_task.as_ref().unwrap_or(&self.task)
    }

    /// Replaces the task from the next reset on (the curriculum's hook).
    /// Rejects a task whose sensor delay the action history cannot cover.
    pub fn set_task(&mut self, task: Task) -> Result<(), EnvError> {
        self.config.check_task(&task)?;
        self.next_task = Some(task);
        Ok(())
    }

    /// Width of the observation.
    pub fn obs_dim(&self) -> usize {
        self.config.obs_dim()
    }

    /// Starts a new episode with `seed` (domain randomization included).
    pub fn reset(&mut self, seed: u64) {
        if let Some(task) = self.next_task.take() {
            self.reference_propellant = reference_propellant(task.params());
            self.task = task;
        }
        let params = self.task.episode_params(seed);
        let decel = self.config.reward.profile_decel(&params);
        let sim = Sim::new(params, seed).expect("Task guarantees valid episode parameters");
        let initial_throttle = dequantize(sim.initial_throttle_level());
        let potential = self
            .config
            .reward
            .potential(sim.params(), decel, sim.state());
        self.episode = Some(Episode {
            sim,
            seed,
            potential,
            decel,
            // Before the first decision, the command that keeps the initial
            // engine state (not zero, which would mean mid-throttle).
            history: [[initial_throttle, 0.0, 0.0]; MAX_HISTORY],
            episode_return: 0.0,
            decisions: 0,
        });
    }

    /// The controls an action maps to.
    fn controls(&self, action: [f64; ACTION_DIM]) -> Controls {
        if self.config.quantize_actions {
            Controls::from_quantized(action.map(quantize))
        } else {
            let [t, g, r] = action;
            Controls::new(t, g, r)
        }
    }

    /// Applies one action for `action_repeat` ticks (fewer if the episode
    /// ends first).
    pub fn step(&mut self, action: [f64; ACTION_DIM]) -> Result<Step, EnvError> {
        if action.iter().any(|a| !a.is_finite()) {
            return Err(EnvError::NonFiniteAction(action));
        }
        let controls = self.controls(action);
        let repeat = self.config.action_repeat;
        let reward_params = &self.config.reward;
        let reference = self.reference_propellant;
        let ep = self.episode.as_mut().ok_or(EnvError::NotReset)?;
        if ep.sim.outcome().is_some() {
            return Err(EnvError::EpisodeOver);
        }

        let propellant_before = ep.sim.propellant();
        let time_before = ep.sim.time();
        let mut outcome = None;
        for _ in 0..repeat {
            outcome = ep.sim.step(controls);
            if outcome.is_some() {
                break;
            }
        }
        ep.history.copy_within(0..MAX_HISTORY - 1, 1);
        ep.history[0] = [controls.throttle, controls.gimbal, controls.rcs];
        ep.decisions += 1;

        let terminated = outcome.is_some_and(|o| !o.is_truncation());
        let truncated = outcome.is_some_and(Outcome::is_truncation);
        let potential = if terminated && reward_params.zero_terminal_potential {
            0.0
        } else {
            reward_params.potential(ep.sim.params(), ep.decel, ep.sim.state())
        };
        let burnt = (propellant_before - ep.sim.propellant()) / reference;
        let flown = ep.sim.time() - time_before;
        let mut reward = reward_params.gamma * potential
            - ep.potential
            - reward_params.fuel_weight * burnt
            - reward_params.time_weight * flown;
        match outcome {
            Some(Outcome::Landed) => reward += reward_params.landing_bonus,
            Some(_) if terminated => reward -= reward_params.crash_penalty,
            _ => {}
        }
        ep.potential = potential;
        ep.episode_return += reward;
        Ok(Step {
            reward,
            terminated,
            truncated,
            outcome,
        })
    }

    /// Writes the observation of the current tick into `out`
    /// (`out.len() == obs_dim`). Zeros before the first reset.
    pub fn observe(&self, out: &mut [f32]) -> Result<(), EnvError> {
        self.check_buffer(out)?;
        out.fill(0.0);
        let Some(ep) = self.episode.as_ref() else {
            return Ok(());
        };
        let sim = &ep.sim;
        let p = sim.params();
        let v = &p.vehicle;
        let [x, y, vx, vy, theta, omega] = sim.observation();
        let propellant = sim.propellant();
        let cm = mass_properties(v, v.dry_mass + propellant).cm_height;
        let height = y - cm * libm::cos(theta);
        let delay = p.sensors.delay_ticks;
        let s = sim.state();
        let values: [f64; BASE_OBS_DIM] = [
            x / POSITION_UNIT,
            height / POSITION_UNIT,
            vx / VELOCITY_UNIT,
            vy / VELOCITY_UNIT,
            libm::sin(theta),
            libm::cos(theta),
            omega / RATE_UNIT,
            propellant / self.reference_propellant,
            s.gimbal / v.gimbal_max,
            s.thrust / v.max_thrust,
            if sim.engine_on() { 1.0 } else { 0.0 },
            f64::from(sim.ignitions_left()),
            delay as f64 / MAX_DELAY_TICKS as f64,
            v.max_thrust / ((v.dry_mass + propellant) * p.environment.gravity),
        ];
        for (o, value) in out.iter_mut().zip(values) {
            // Narrowing to f32 is the documented observation precision.
            *o = value as f32;
        }
        let shown = if self.config.action_history {
            delay
                .div_ceil(self.config.action_repeat as usize)
                .min(self.config.history_len)
        } else {
            0
        };
        for (k, action) in ep.history.iter().take(shown).enumerate() {
            let at = obs::HISTORY + k * ACTION_DIM;
            for (o, a) in out[at..at + ACTION_DIM].iter_mut().zip(action) {
                *o = *a as f32;
            }
        }
        Ok(())
    }

    fn check_buffer(&self, out: &[f32]) -> Result<(), EnvError> {
        let expected = self.obs_dim();
        if out.len() == expected {
            Ok(())
        } else {
            Err(EnvError::BadBuffer {
                expected,
                got: out.len(),
            })
        }
    }

    /// The observation as a new vector.
    pub fn observation(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.obs_dim()];
        self.observe(&mut out)
            .expect("the buffer has the observation width");
        out
    }

    /// The running simulation, after the first reset.
    pub fn sim(&self) -> Option<&Sim> {
        self.episode.as_ref().map(|e| &e.sim)
    }

    /// Seed of the current episode.
    pub fn seed(&self) -> Option<u64> {
        self.episode.as_ref().map(|e| e.seed)
    }

    /// Undiscounted return of the current episode so far.
    pub fn episode_return(&self) -> f64 {
        self.episode.as_ref().map_or(0.0, |e| e.episode_return)
    }

    /// Decisions taken in the current episode.
    pub fn episode_length(&self) -> u32 {
        self.episode.as_ref().map_or(0, |e| e.decisions)
    }

    /// Current potential `Φ` (0 once terminated with
    /// [`RewardParams::zero_terminal_potential`]).
    pub fn potential(&self) -> f64 {
        self.episode.as_ref().map_or(0.0, |e| e.potential)
    }
}

/// The propellant fraction's reference: the task's largest initial load
/// (never zero).
fn reference_propellant(p: &Params) -> f64 {
    p.start.propellant.max.max(1.0)
}

/// Seed of episode `episode` of environment `env_index` in a run seeded with
/// `base`. Training seeds always have the top bit set, so they never collide
/// with evaluation seeds (from 0) or controller-tuning seeds (from 10⁶).
pub fn episode_seed(base: u64, env_index: u64, episode: u64) -> u64 {
    let a = SplitMix64::new(base).next_u64();
    let b = SplitMix64::new(a ^ env_index.wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64();
    let c = SplitMix64::new(b ^ episode.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)).next_u64();
    c | (1 << 63)
}

/// One environment of a vectorized batch: auto-reset and seeding.
///
/// With auto-reset, a finished episode's last observation goes to the
/// terminal-observation buffer and the next episode starts at once (its
/// first observation is returned), as Stable-Baselines3 expects. Without it
/// (evaluation), a finished slot stays finished and ignores actions.
#[derive(Clone, Debug)]
pub struct EnvSlot {
    env: Env,
    base_seed: u64,
    index: u64,
    episodes: u64,
    autoreset: bool,
}

/// What one slot produced in one vectorized step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SlotStep {
    /// Reward of the decision (0 for an idle slot).
    pub reward: f64,
    /// The episode reached a terminal state.
    pub terminated: bool,
    /// The episode hit the time limit.
    pub truncated: bool,
    /// Outcome code of the finished episode (0 while running).
    pub outcome: u8,
    /// Score of the finished episode.
    pub score: u32,
    /// Undiscounted return of the finished episode.
    pub episode_return: f64,
    /// Decisions in the finished episode.
    pub episode_length: u32,
    /// Seed of the finished episode.
    pub seed: u64,
    /// The slot was already finished and did nothing (no auto-reset).
    pub idle: bool,
}

impl EnvSlot {
    /// Slot `index` of a batch seeded with `base_seed`.
    pub fn new(env: Env, base_seed: u64, index: u64, autoreset: bool) -> Self {
        Self {
            env,
            base_seed,
            index,
            episodes: 0,
            autoreset,
        }
    }

    /// The environment.
    pub fn env(&self) -> &Env {
        &self.env
    }

    /// The environment, mutably (for `set_task`).
    pub fn env_mut(&mut self) -> &mut Env {
        &mut self.env
    }

    /// Starts an episode with `seed`, or with the slot's next seed in its
    /// [`episode_seed`] sequence, and writes its observation.
    pub fn reset(&mut self, seed: Option<u64>, obs: &mut [f32]) -> Result<(), EnvError> {
        self.env.check_buffer(obs)?;
        let seed = seed.unwrap_or_else(|| self.next_seed());
        self.env.reset(seed);
        self.env.observe(obs)
    }

    /// Restarts the slot's seed sequence from a new base seed (takes effect
    /// at the next reset without an explicit seed).
    pub fn set_base_seed(&mut self, base_seed: u64) {
        self.base_seed = base_seed;
        self.episodes = 0;
    }

    fn next_seed(&mut self) -> u64 {
        let s = episode_seed(self.base_seed, self.index, self.episodes);
        self.episodes += 1;
        s
    }

    /// Whether the current episode has ended.
    pub fn finished(&self) -> bool {
        self.env.sim().is_none_or(|s| s.outcome().is_some())
    }

    /// One decision. `obs` receives the next observation; when the episode
    /// ends, `terminal_obs` receives its last observation (with auto-reset,
    /// `obs` then holds the first observation of the next episode).
    pub fn step(
        &mut self,
        action: [f64; ACTION_DIM],
        obs: &mut [f32],
        terminal_obs: &mut [f32],
    ) -> Result<SlotStep, EnvError> {
        // Checked first, so a bad buffer never leaves a half-done step.
        self.env.check_buffer(obs)?;
        self.env.check_buffer(terminal_obs)?;
        if !self.autoreset && self.env.sim().is_some() && self.finished() {
            self.env.observe(obs)?;
            return Ok(SlotStep {
                idle: true,
                ..SlotStep::default()
            });
        }
        let step = self.env.step(action)?;
        let mut out = SlotStep {
            reward: step.reward,
            terminated: step.terminated,
            truncated: step.truncated,
            ..SlotStep::default()
        };
        if let Some(outcome) = step.outcome {
            let sim = self.env.sim().expect("stepped, so reset");
            out.outcome = outcome.code();
            out.score = sim.score();
            out.episode_return = self.env.episode_return();
            out.episode_length = self.env.episode_length();
            out.seed = self.env.seed().expect("stepped, so reset");
            self.env.observe(terminal_obs)?;
            if self.autoreset {
                self.reset(None, obs)?;
            } else {
                obs.copy_from_slice(terminal_obs);
            }
        } else {
            self.env.observe(obs)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod lateral_tests {
    use super::*;
    use crate::physics::params::VehicleConfig;

    #[test]
    fn moving_towards_the_pad_at_the_reference_speed_costs_nothing() {
        let r = RewardParams::default();
        let p = VehicleConfig::Easy.params();
        let decel = r.profile_decel(&p);
        let mass = p.vehicle.dry_mass;
        let cm = mass_properties(&p.vehicle, mass).cm_height;
        let at = |x: f64, vx: f64| {
            let s = State {
                x,
                vx,
                y: cm,
                mass,
                ..State::default()
            };
            r.potential(&p, decel, &s)
        };
        let position_only = |x: f64| {
            let u = x.abs() / r.position_scale;
            -r.position_weight * u / (1.0 + u)
        };
        // 30 m right of the pad: the reference is 3 m/s to the left.
        assert!((at(30.0, -3.0) - position_only(30.0)).abs() < 1e-12);
        assert!(
            at(30.0, 3.0) < at(30.0, 0.0),
            "moving away is worse than hovering"
        );
        assert!(
            at(30.0, 0.0) < at(30.0, -3.0),
            "hovering is worse than approaching"
        );
        // Far away the reference saturates at the speed limit.
        assert!((at(-500.0, 5.0) - position_only(-500.0)).abs() < 1e-12);
        // Over the pad, any sideways speed costs.
        assert!(at(0.0, 1.0) < at(0.0, 0.0));
    }
}
