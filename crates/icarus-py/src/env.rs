//! The reinforcement-learning environment for Python: one environment
//! (`icarus.Env`, wrapped by the Gymnasium adapter) and a vectorized batch
//! (`icarus.VecEnv`, wrapped by the Stable-Baselines3 adapter).
//!
//! Every per-step computation (physics, observation, reward, termination,
//! auto-reset) runs in `icarus_core::env`. The batch steps its environments
//! in parallel with rayon while the GIL is released; each episode stays
//! deterministic on its own, so results never depend on the thread count.

use std::collections::HashMap;

use icarus_core::env::{
    ACTION_DIM, BASE_OBS_DIM, Env, EnvConfig, EnvError, EnvSlot, OVERRIDE_KEYS, Randomization,
    RewardParams, SlotStep, Task, apply_override,
};
use icarus_core::eval::summarize;
use icarus_core::physics::params::{Range, VehicleConfig};
use icarus_core::physics::scenario::Outcome;
use icarus_core::{Scenario, TELEMETRY_FIELDS};
use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::{parse_config, parse_scenario};

/// Largest batch accepted by `VecEnv` (a guard against accidental huge
/// allocations).
const MAX_ENVS: usize = 65_536;

/// Batches at least this large are stepped in parallel by default.
const PARALLEL_THRESHOLD: usize = 16;

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn env_error(e: EnvError) -> PyErr {
    match e {
        EnvError::Invalid(_) | EnvError::NonFiniteAction(_) | EnvError::BadBuffer { .. } => {
            value_error(e)
        }
        EnvError::NotReset | EnvError::EpisodeOver => PyRuntimeError::new_err(e.to_string()),
    }
}

/// A Python mapping as `(key, value)` pairs, or no pairs for `None`.
fn mapping<'py>(
    obj: Option<&Bound<'py, PyAny>>,
    what: &str,
) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(Vec::new());
    };
    let dict = obj
        .cast::<PyDict>()
        .map_err(|_| value_error(format!("{what} must be a dict")))?;
    dict.iter()
        .map(|(k, v)| {
            let key: String = k
                .extract()
                .map_err(|_| value_error(format!("{what} keys must be strings")))?;
            Ok((key, v))
        })
        .collect()
}

fn unknown_key(what: &str, key: &str, valid: &[&str]) -> PyErr {
    value_error(format!(
        "unknown {what} key {key:?}; expected one of {valid:?}"
    ))
}

/// One number or a list of numbers.
fn numbers(value: &Bound<'_, PyAny>, key: &str) -> PyResult<Vec<f64>> {
    if let Ok(x) = value.extract::<f64>() {
        return Ok(vec![x]);
    }
    value
        .extract::<Vec<f64>>()
        .map_err(|_| value_error(format!("{key} must be a number or a list of numbers")))
}

fn range(value: &Bound<'_, PyAny>, key: &str) -> PyResult<Range> {
    match numbers(value, key)?.as_slice() {
        [a, b] => Ok(Range::new(*a, *b)),
        [a] => Ok(Range::fixed(*a)),
        other => Err(value_error(format!(
            "{key} must be [min, max] or a single number, got {other:?}"
        ))),
    }
}

fn parse_randomization(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Randomization> {
    const KEYS: [&str; 7] = [
        "dry_mass",
        "propellant",
        "wind",
        "thrust_time_constant",
        "gimbal_time_constant",
        "sensor_noise",
        "sensor_delay",
    ];
    let mut r = Randomization::none();
    for (key, value) in mapping(obj, "randomization")? {
        let name = format!("randomization.{key}");
        match key.as_str() {
            "dry_mass" => r.dry_mass = range(&value, &name)?,
            "propellant" => r.propellant = range(&value, &name)?,
            "wind" => r.wind = range(&value, &name)?,
            "thrust_time_constant" => r.thrust_time_constant = range(&value, &name)?,
            "gimbal_time_constant" => r.gimbal_time_constant = range(&value, &name)?,
            "sensor_noise" => r.sensor_noise = range(&value, &name)?,
            "sensor_delay" => {
                r.sensor_delay = if value.is_none() {
                    None
                } else {
                    let bounds: Vec<usize> = value.extract().map_err(|_| {
                        value_error(format!("{name} must be [min, max] ticks or None"))
                    })?;
                    let [lo, hi] = bounds[..] else {
                        return Err(value_error(format!(
                            "{name} must be [min, max] ticks or None, got {bounds:?}"
                        )));
                    };
                    Some((lo, hi))
                };
            }
            other => return Err(unknown_key("randomization", other, &KEYS)),
        }
    }
    Ok(r)
}

/// Builds a task from a dict: `{"config": "easy", "scenario": "nominal",
/// "overrides": {"start.altitude": [min, max], ...}, "randomization": {...}}`.
pub fn parse_task(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Task> {
    const KEYS: [&str; 4] = ["config", "scenario", "overrides", "randomization"];
    let mut config = VehicleConfig::Easy;
    let mut scenario = Scenario::Nominal;
    let mut overrides: Vec<(String, Bound<'_, PyAny>)> = Vec::new();
    let mut randomization = Randomization::none();
    for (key, value) in mapping(obj, "task")? {
        match key.as_str() {
            "config" => config = parse_config(&value.extract::<String>()?)?,
            "scenario" => scenario = parse_scenario(&value.extract::<String>()?)?,
            "overrides" => overrides = mapping(Some(&value), "task overrides")?,
            "randomization" => randomization = parse_randomization(Some(&value))?,
            other => return Err(unknown_key("task", other, &KEYS)),
        }
    }
    let mut params = scenario.params(config);
    for (key, value) in &overrides {
        apply_override(&mut params, key, &numbers(value, key)?).map_err(value_error)?;
    }
    Task::new(params, randomization).map_err(value_error)
}

fn parse_reward(obj: &Bound<'_, PyAny>) -> PyResult<RewardParams> {
    let mut r = RewardParams::default();
    let fields: [(&str, &mut f64); 19] = [
        ("gamma", &mut r.gamma),
        ("landing_bonus", &mut r.landing_bonus),
        ("crash_penalty", &mut r.crash_penalty),
        ("fuel_weight", &mut r.fuel_weight),
        ("time_weight", &mut r.time_weight),
        ("position_weight", &mut r.position_weight),
        ("position_scale", &mut r.position_scale),
        ("altitude_weight", &mut r.altitude_weight),
        ("altitude_scale", &mut r.altitude_scale),
        ("velocity_weight", &mut r.velocity_weight),
        ("velocity_scale", &mut r.velocity_scale),
        ("lateral_gain", &mut r.lateral_gain),
        ("lateral_speed_max", &mut r.lateral_speed_max),
        ("tilt_weight", &mut r.tilt_weight),
        ("tilt_scale", &mut r.tilt_scale),
        ("rate_weight", &mut r.rate_weight),
        ("rate_scale", &mut r.rate_scale),
        ("touchdown_speed_fraction", &mut r.touchdown_speed_fraction),
        ("decel_fraction", &mut r.decel_fraction),
    ];
    let mut names: Vec<&str> = fields.iter().map(|(n, _)| *n).collect();
    names.push("zero_terminal_potential");
    let mut by_name: HashMap<&str, &mut f64> = fields.into_iter().collect();
    let mut zero_terminal = None;
    for (key, value) in mapping(Some(obj), "reward")? {
        if key == "zero_terminal_potential" {
            zero_terminal = Some(
                value
                    .extract::<bool>()
                    .map_err(|_| value_error("reward.zero_terminal_potential must be a bool"))?,
            );
            continue;
        }
        let slot = by_name
            .get_mut(key.as_str())
            .ok_or_else(|| unknown_key("reward", &key, &names))?;
        **slot = value
            .extract()
            .map_err(|_| value_error(format!("reward.{key} must be a number")))?;
    }
    if let Some(z) = zero_terminal {
        r.zero_terminal_potential = z;
    }
    Ok(r)
}

/// Builds an environment configuration from a dict: `{"action_repeat": 3,
/// "history_len": 4, "action_history": True, "quantize_actions": True,
/// "reward": {...}}`. Validated.
pub fn parse_env_config(obj: Option<&Bound<'_, PyAny>>) -> PyResult<EnvConfig> {
    const KEYS: [&str; 5] = [
        "action_repeat",
        "history_len",
        "action_history",
        "quantize_actions",
        "reward",
    ];
    let mut c = EnvConfig::default();
    for (key, value) in mapping(obj, "env_config")? {
        match key.as_str() {
            "action_repeat" => c.action_repeat = value.extract()?,
            "history_len" => c.history_len = value.extract()?,
            "action_history" => c.action_history = value.extract()?,
            "quantize_actions" => c.quantize_actions = value.extract()?,
            "reward" => c.reward = parse_reward(&value)?,
            other => return Err(unknown_key("env_config", other, &KEYS)),
        }
    }
    c.validate().map_err(value_error)?;
    Ok(c)
}

fn reward_dict<'py>(py: Python<'py>, r: &RewardParams) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (name, value) in [
        ("gamma", r.gamma),
        ("landing_bonus", r.landing_bonus),
        ("crash_penalty", r.crash_penalty),
        ("fuel_weight", r.fuel_weight),
        ("time_weight", r.time_weight),
        ("position_weight", r.position_weight),
        ("position_scale", r.position_scale),
        ("altitude_weight", r.altitude_weight),
        ("altitude_scale", r.altitude_scale),
        ("velocity_weight", r.velocity_weight),
        ("velocity_scale", r.velocity_scale),
        ("lateral_gain", r.lateral_gain),
        ("lateral_speed_max", r.lateral_speed_max),
        ("tilt_weight", r.tilt_weight),
        ("tilt_scale", r.tilt_scale),
        ("rate_weight", r.rate_weight),
        ("rate_scale", r.rate_scale),
        ("touchdown_speed_fraction", r.touchdown_speed_fraction),
        ("decel_fraction", r.decel_fraction),
    ] {
        d.set_item(name, value)?;
    }
    d.set_item("zero_terminal_potential", r.zero_terminal_potential)?;
    Ok(d)
}

fn env_config_dict<'py>(py: Python<'py>, c: &EnvConfig) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("action_repeat", c.action_repeat)?;
    d.set_item("history_len", c.history_len)?;
    d.set_item("action_history", c.action_history)?;
    d.set_item("quantize_actions", c.quantize_actions)?;
    d.set_item("reward", reward_dict(py, &c.reward)?)?;
    Ok(d)
}

/// The default environment configuration, as a dict.
#[pyfunction]
pub fn default_env_config(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    env_config_dict(py, &EnvConfig::default())
}

/// Names of the observation components for a history length.
#[pyfunction]
#[pyo3(signature = (history_len = icarus_core::env::DEFAULT_HISTORY))]
pub fn obs_fields(history_len: usize) -> PyResult<Vec<String>> {
    if history_len > icarus_core::env::MAX_HISTORY {
        return Err(value_error(format!(
            "history_len must be at most {}, got {history_len}",
            icarus_core::env::MAX_HISTORY
        )));
    }
    const BASE: [&str; BASE_OBS_DIM] = [
        "x",
        "height",
        "vx",
        "vy",
        "sin_theta",
        "cos_theta",
        "omega",
        "propellant",
        "gimbal",
        "thrust",
        "engine_on",
        "ignitions",
        "delay",
        "twr",
    ];
    let mut names: Vec<String> = BASE.iter().map(|s| (*s).to_owned()).collect();
    for k in 0..history_len {
        for axis in ["throttle", "gimbal", "rcs"] {
            names.push(format!("action{k}_{axis}"));
        }
    }
    Ok(names)
}

/// Names accepted in a task's ``overrides`` and the number of values each takes.
#[pyfunction]
pub fn override_keys() -> Vec<(&'static str, usize)> {
    OVERRIDE_KEYS.to_vec()
}

/// Checks and copies a `(3,)` action.
fn action(a: &PyArrayLike1<'_, f64, AllowTypeChange>) -> PyResult<[f64; ACTION_DIM]> {
    let view = a.as_array();
    if view.len() != ACTION_DIM {
        return Err(value_error(format!(
            "action must have {ACTION_DIM} components, got {}",
            view.len()
        )));
    }
    let out = [view[0], view[1], view[2]];
    if out.iter().any(|v| !v.is_finite()) {
        return Err(value_error(format!("action must be finite, got {out:?}")));
    }
    Ok(out)
}

/// One environment (single-episode API, used by the Gymnasium wrapper).
///
/// Args:
///     task: dict with ``config``, ``scenario``, ``overrides`` and
///         ``randomization`` (all optional; default: the ``easy`` vehicle).
///     env_config: dict with ``action_repeat``, ``history_len``,
///         ``action_history``, ``quantize_actions`` and ``reward``.
#[pyclass(module = "icarus", name = "Env")]
pub struct PyEnv {
    inner: Env,
}

#[pymethods]
impl PyEnv {
    #[new]
    #[pyo3(signature = (task = None, env_config = None))]
    fn new(
        task: Option<&Bound<'_, PyAny>>,
        env_config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let inner =
            Env::new(parse_env_config(env_config)?, parse_task(task)?).map_err(env_error)?;
        Ok(Self { inner })
    }

    /// Starts an episode; returns its first observation (float32).
    fn reset<'py>(&mut self, py: Python<'py>, seed: u64) -> Bound<'py, PyArray1<f32>> {
        self.inner.reset(seed);
        self.inner.observation().into_pyarray(py)
    }

    /// One decision: ``(obs, reward, terminated, truncated, info)``.
    ///
    /// ``info`` has ``outcome`` (name or ``None``), ``landed``, ``score`` and
    /// ``seed``. Raises ``ValueError`` for a malformed or non-finite action
    /// and ``RuntimeError`` before ``reset`` or after the episode ended.
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action_in: PyArrayLike1<'py, f64, AllowTypeChange>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        f64,
        bool,
        bool,
        Bound<'py, PyDict>,
    )> {
        let a = action(&action_in)?;
        let step = self.inner.step(a).map_err(env_error)?;
        let info = PyDict::new(py);
        info.set_item("outcome", step.outcome.map(Outcome::name))?;
        info.set_item("landed", step.outcome == Some(Outcome::Landed))?;
        let sim = self.inner.sim().expect("stepped, so reset");
        info.set_item("score", sim.score())?;
        info.set_item("seed", self.inner.seed())?;
        Ok((
            self.inner.observation().into_pyarray(py),
            step.reward,
            step.terminated,
            step.truncated,
            info,
        ))
    }

    /// Replaces the task from the next reset on.
    fn set_task(&mut self, task: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner
            .set_task(parse_task(Some(task))?)
            .map_err(env_error)?;
        Ok(())
    }

    /// The current observation.
    fn observation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.observation().into_pyarray(py)
    }

    /// The simulator telemetry (true state included: for debugging and
    /// plots, never for the policy). Empty before the first reset.
    fn telemetry_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        if let Some(sim) = self.inner.sim() {
            for (name, value) in TELEMETRY_FIELDS.iter().zip(sim.telemetry()) {
                d.set_item(*name, value)?;
            }
        }
        Ok(d)
    }

    /// The environment configuration, as a dict.
    fn env_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        env_config_dict(py, self.inner.config())
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        self.inner.obs_dim()
    }

    #[getter]
    fn action_dim(&self) -> usize {
        ACTION_DIM
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed()
    }

    #[getter]
    fn episode_return(&self) -> f64 {
        self.inner.episode_return()
    }

    #[getter]
    fn episode_length(&self) -> u32 {
        self.inner.episode_length()
    }

    /// Current shaping potential Φ (0 once terminated).
    #[getter]
    fn potential(&self) -> f64 {
        self.inner.potential()
    }

    #[getter]
    fn done(&self) -> bool {
        self.inner.sim().is_some_and(|s| s.outcome().is_some())
    }
}

/// A batch of environments stepped together, in Rust, on every core.
///
/// Args:
///     num_envs: batch size.
///     seed: base seed. Episode ``j`` of environment ``i`` uses
///         ``episode_seed(seed, i, j)`` (top bit set: never an evaluation
///         seed) unless ``reset`` is given explicit seeds.
///     task, env_config: as for ``Env`` (every environment starts with the
///         same task; ``set_task`` can give subsets different ones).
///     autoreset: start the next episode as soon as one ends (training);
///         without it a finished environment stays finished (evaluation).
///     parallel: step on every core (default: batches of 16 or more).
///
/// Threads: ``step`` and ``reset`` release the GIL while the batch runs, so
/// another Python thread touching the same object meanwhile gets
/// ``RuntimeError: Already borrowed`` (use one object per thread). The batch
/// runs on rayon's global pool, one thread per core by default; set
/// ``RAYON_NUM_THREADS`` to share the cores with PyTorch. Create
/// ``VecEnv`` objects after any ``fork`` (rayon's pool does not survive one).
#[pyclass(module = "icarus", name = "VecEnv")]
pub struct PyVecEnv {
    slots: Vec<EnvSlot>,
    obs_dim: usize,
    parallel: bool,
    was_reset: bool,
    config: EnvConfig,
}

impl PyVecEnv {
    fn indices(&self, indices: Option<Vec<usize>>) -> PyResult<Vec<usize>> {
        let n = self.slots.len();
        let indices = indices.unwrap_or_else(|| (0..n).collect());
        if let Some(bad) = indices.iter().find(|&&i| i >= n) {
            return Err(value_error(format!(
                "environment index {bad} out of range for {n} environments"
            )));
        }
        Ok(indices)
    }
}

#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (num_envs, seed = 0, task = None, env_config = None, autoreset = true, parallel = None))]
    fn new(
        num_envs: usize,
        seed: u64,
        task: Option<&Bound<'_, PyAny>>,
        env_config: Option<&Bound<'_, PyAny>>,
        autoreset: bool,
        parallel: Option<bool>,
    ) -> PyResult<Self> {
        if !(1..=MAX_ENVS).contains(&num_envs) {
            return Err(value_error(format!(
                "num_envs must be in 1..={MAX_ENVS}, got {num_envs}"
            )));
        }
        let config = parse_env_config(env_config)?;
        let task = parse_task(task)?;
        let env = Env::new(config.clone(), task).map_err(env_error)?;
        let obs_dim = env.obs_dim();
        let slots = (0..num_envs)
            .map(|i| EnvSlot::new(env.clone(), seed, i as u64, autoreset))
            .collect();
        Ok(Self {
            slots,
            obs_dim,
            parallel: parallel.unwrap_or(num_envs >= PARALLEL_THRESHOLD),
            was_reset: false,
            config,
        })
    }

    /// Starts an episode in every environment (with its next seed, or with
    /// ``seeds[i]``); returns the observations, shape ``(num_envs, obs_dim)``.
    #[pyo3(signature = (seeds = None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seeds: Option<Vec<u64>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = self.slots.len();
        if let Some(s) = &seeds {
            if s.len() != n {
                return Err(value_error(format!("expected {n} seeds, got {}", s.len())));
            }
        }
        let dim = self.obs_dim;
        let mut obs = vec![0.0_f32; n * dim];
        let (slots, parallel) = (&mut self.slots, self.parallel);
        let done: Result<(), EnvError> = py.detach(|| {
            let work = |(i, (slot, row)): (usize, (&mut EnvSlot, &mut [f32]))| {
                slot.reset(seeds.as_ref().map(|s| s[i]), row)
            };
            if parallel {
                slots
                    .par_iter_mut()
                    .zip(obs.par_chunks_mut(dim))
                    .enumerate()
                    .try_for_each(work)
            } else {
                slots
                    .iter_mut()
                    .zip(obs.chunks_mut(dim))
                    .enumerate()
                    .try_for_each(work)
            }
        });
        done.map_err(env_error)?;
        self.was_reset = true;
        Ok(Array2::from_shape_vec((n, dim), obs)
            .expect("buffer sized n * dim")
            .into_pyarray(py))
    }

    /// One decision in every environment. ``actions`` has shape
    /// ``(num_envs, 3)``.
    ///
    /// Returns ``(obs, rewards, terminated, truncated, info)`` where ``info``
    /// is a dict of arrays: ``terminal_obs`` (rows valid where an episode
    /// ended), ``outcome`` (code, 0 while running), ``score``,
    /// ``episode_return``, ``episode_length``, ``seed`` (of the finished
    /// episode) and ``idle`` (already finished, without auto-reset).
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyArrayLike2<'py, f64, AllowTypeChange>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyDict>,
    )> {
        if !self.was_reset {
            return Err(PyRuntimeError::new_err("step() called before reset()"));
        }
        let n = self.slots.len();
        let view = actions.as_array();
        if view.shape() != [n, ACTION_DIM] {
            return Err(value_error(format!(
                "actions must have shape ({n}, {ACTION_DIM}), got {:?}",
                view.shape()
            )));
        }
        if view.iter().any(|v| !v.is_finite()) {
            return Err(value_error("actions must be finite"));
        }
        let acts: Vec<[f64; ACTION_DIM]> = view
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect();
        let dim = self.obs_dim;
        let mut obs = vec![0.0_f32; n * dim];
        let mut terminal = vec![0.0_f32; n * dim];
        let mut results = vec![SlotStep::default(); n];
        let (slots, parallel) = (&mut self.slots, self.parallel);
        let outcome: Result<(), EnvError> = py.detach(|| {
            let work = |(((slot, act), (row, trow)), res): (
                ((&mut EnvSlot, &[f64; ACTION_DIM]), (&mut [f32], &mut [f32])),
                &mut SlotStep,
            )| {
                *res = slot.step(*act, row, trow)?;
                Ok(())
            };
            if parallel {
                slots
                    .par_iter_mut()
                    .zip(acts.par_iter())
                    .zip(obs.par_chunks_mut(dim).zip(terminal.par_chunks_mut(dim)))
                    .zip(results.par_iter_mut())
                    .try_for_each(work)
            } else {
                slots
                    .iter_mut()
                    .zip(acts.iter())
                    .zip(obs.chunks_mut(dim).zip(terminal.chunks_mut(dim)))
                    .zip(results.iter_mut())
                    .try_for_each(work)
            }
        });
        // Only reachable through misuse checked above; kept as an error, not
        // a panic, for safety.
        outcome.map_err(env_error)?;

        let info = PyDict::new(py);
        let col = |f: fn(&SlotStep) -> f64| results.iter().map(f).collect::<Vec<f64>>();
        info.set_item(
            "terminal_obs",
            Array2::from_shape_vec((n, dim), terminal)
                .expect("buffer sized n * dim")
                .into_pyarray(py),
        )?;
        info.set_item(
            "outcome",
            results
                .iter()
                .map(|r| r.outcome)
                .collect::<Vec<u8>>()
                .into_pyarray(py),
        )?;
        info.set_item(
            "score",
            results
                .iter()
                .map(|r| r.score)
                .collect::<Vec<u32>>()
                .into_pyarray(py),
        )?;
        info.set_item("episode_return", col(|r| r.episode_return).into_pyarray(py))?;
        info.set_item(
            "episode_length",
            results
                .iter()
                .map(|r| r.episode_length)
                .collect::<Vec<u32>>()
                .into_pyarray(py),
        )?;
        info.set_item(
            "seed",
            results
                .iter()
                .map(|r| r.seed)
                .collect::<Vec<u64>>()
                .into_pyarray(py),
        )?;
        info.set_item(
            "idle",
            results
                .iter()
                .map(|r| r.idle)
                .collect::<Vec<bool>>()
                .into_pyarray(py),
        )?;
        Ok((
            Array2::from_shape_vec((n, dim), obs)
                .expect("buffer sized n * dim")
                .into_pyarray(py),
            col(|r| r.reward).into_pyarray(py),
            results
                .iter()
                .map(|r| r.terminated)
                .collect::<Vec<bool>>()
                .into_pyarray(py),
            results
                .iter()
                .map(|r| r.truncated)
                .collect::<Vec<bool>>()
                .into_pyarray(py),
            info,
        ))
    }

    /// Replaces the task of the given environments (default: all) from
    /// their next reset on.
    #[pyo3(signature = (task, indices = None))]
    fn set_task(&mut self, task: &Bound<'_, PyAny>, indices: Option<Vec<usize>>) -> PyResult<()> {
        let task = parse_task(Some(task))?;
        // Every environment shares one configuration: checking once is
        // enough, and nothing changes if the task is rejected.
        self.config.check_task(&task).map_err(value_error)?;
        for i in self.indices(indices)? {
            self.slots[i]
                .env_mut()
                .set_task(task.clone())
                .map_err(env_error)?;
        }
        Ok(())
    }

    /// Summary of every environment's current (or finished) episode: one
    /// dict per environment with ``seed``, ``done``, ``outcome``,
    /// ``landed``, ``score``, ``ticks``, ``contact_vx``, ``contact_vy``,
    /// ``position_error``, ``propellant_used``, ``tilt``,
    /// ``episode_return`` and ``episode_length``.
    fn episode_results<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.slots
            .iter()
            .map(|slot| {
                let env = slot.env();
                let d = PyDict::new(py);
                let Some(sim) = env.sim() else {
                    d.set_item("done", false)?;
                    return Ok(d);
                };
                let r = summarize(sim);
                d.set_item("seed", env.seed())?;
                d.set_item("done", sim.outcome().is_some())?;
                d.set_item("outcome", sim.outcome().map(Outcome::name))?;
                d.set_item("landed", r.landed())?;
                d.set_item("score", r.score)?;
                d.set_item("ticks", r.ticks)?;
                d.set_item("contact_vx", r.contact_vx)?;
                d.set_item("contact_vy", r.contact_vy)?;
                d.set_item("position_error", r.position_error)?;
                d.set_item("propellant_used", r.propellant_used)?;
                d.set_item("tilt", r.tilt)?;
                d.set_item("episode_return", env.episode_return())?;
                d.set_item("episode_length", env.episode_length())?;
                Ok(d)
            })
            .collect()
    }

    /// Simulator telemetry of environment ``index`` (true state: for
    /// debugging, never for the policy).
    fn telemetry_dict<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyDict>> {
        let i = self.indices(Some(vec![index]))?[0];
        let d = PyDict::new(py);
        if let Some(sim) = self.slots[i].env().sim() {
            for (name, value) in TELEMETRY_FIELDS.iter().zip(sim.telemetry()) {
                d.set_item(*name, value)?;
            }
        }
        Ok(d)
    }

    /// The environment configuration, as a dict.
    fn env_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        env_config_dict(py, &self.config)
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.slots.len()
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    #[getter]
    fn action_dim(&self) -> usize {
        ACTION_DIM
    }

    #[getter]
    fn parallel(&self) -> bool {
        self.parallel
    }

    #[setter]
    fn set_parallel(&mut self, value: bool) {
        self.parallel = value;
    }

    /// Restarts every environment's seed sequence from a new base seed
    /// (used from the next reset without explicit seeds).
    fn set_base_seed(&mut self, seed: u64) {
        for slot in &mut self.slots {
            slot.set_base_seed(seed);
        }
    }
}

/// Seed of episode ``episode`` of environment ``env_index`` in a batch
/// seeded with ``base`` (what ``VecEnv`` uses).
#[pyfunction]
pub fn episode_seed(base: u64, env_index: u64, episode: u64) -> u64 {
    icarus_core::env::episode_seed(base, env_index, episode)
}
