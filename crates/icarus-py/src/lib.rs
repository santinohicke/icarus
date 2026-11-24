//! Python bindings for the Icarus simulator (`import icarus`).
//!
//! Exposes the same Rust core the web uses, so training, evaluation and the
//! browser run bit-identical physics.
//!
//! Thread safety: objects are `Send + Sync` and the module supports
//! free-threaded Python, but each `Sim`/`Recorder` must be used by one thread
//! at a time (concurrent calls on the same object raise `RuntimeError:
//! Already borrowed`).

use icarus_core::physics::scenario::Outcome;
use icarus_core::physics::state::QUANT_MAX;
use icarus_core::replay::REPLAY_FORMAT_VERSION;
use icarus_core::{
    Controls, Injection, PHYSICS_VERSION, RecordingSim, Replay, Sim, TELEMETRY_FIELDS,
    VehicleConfig,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn parse_config(name: &str) -> PyResult<VehicleConfig> {
    VehicleConfig::from_name(name).ok_or_else(|| {
        let valid: Vec<&str> = VehicleConfig::ALL.iter().map(|c| c.name()).collect();
        PyValueError::new_err(format!(
            "unknown vehicle configuration {name:?}; expected one of {valid:?}"
        ))
    })
}

fn outcome_name(o: Option<Outcome>) -> Option<&'static str> {
    o.map(Outcome::name)
}

/// Validates quantized levels: the binding is strict (`0..=254`) even though
/// the core clamps, so out-of-range values are caught at the source.
fn quantized(throttle: u8, gimbal: u8, rcs: u8) -> PyResult<[u8; 3]> {
    let q = [throttle, gimbal, rcs];
    if q.iter().any(|&v| v > QUANT_MAX) {
        return Err(PyValueError::new_err(format!(
            "quantized levels must be in 0..={QUANT_MAX}, got {q:?}"
        )));
    }
    Ok(q)
}

/// Telemetry as a dict whose keys follow `TELEMETRY_FIELDS` order.
fn telemetry_dict<'py>(py: Python<'py>, sim: &Sim) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, value) in TELEMETRY_FIELDS.iter().zip(sim.telemetry()) {
        dict.set_item(*name, value)?;
    }
    Ok(dict)
}

/// One landing episode.
///
/// Args:
///     config: a configuration name, see ``configs()``.
///     seed: episode seed (initial conditions, wind, sensor noise, faults).
#[pyclass(module = "icarus", name = "Sim")]
pub struct PySim {
    inner: Sim,
    config: VehicleConfig,
}

#[pymethods]
impl PySim {
    #[new]
    #[pyo3(signature = (config = "easy", seed = 0))]
    fn new(config: &str, seed: u64) -> PyResult<Self> {
        let config = parse_config(config)?;
        Ok(Self {
            inner: Sim::from_config(config, seed),
            config,
        })
    }

    /// Advances one tick with normalized controls in ``[-1, 1]`` (values are
    /// clamped; a NaN throttle means engine off, a NaN gimbal/RCS neutral).
    ///
    /// Returns the outcome name once the episode has ended, else ``None``.
    #[pyo3(signature = (throttle, gimbal = 0.0, rcs = 0.0))]
    fn step(&mut self, throttle: f64, gimbal: f64, rcs: f64) -> Option<&'static str> {
        outcome_name(self.inner.step(Controls::new(throttle, gimbal, rcs)))
    }

    /// Advances one tick with quantized controls (``0..=254``, 127 = neutral),
    /// exactly as the web front-end does. Raises ``ValueError`` above 254.
    fn step_quantized(
        &mut self,
        throttle: u8,
        gimbal: u8,
        rcs: u8,
    ) -> PyResult<Option<&'static str>> {
        let q = quantized(throttle, gimbal, rcs)?;
        Ok(outcome_name(self.inner.step(Controls::from_quantized(q))))
    }

    /// Jams the gimbal at ``angle`` radians (clamped to the gimbal range).
    /// Returns whether the event was accepted (``False`` for non-finite
    /// values or a finished episode).
    fn inject_gimbal_stuck(&mut self, angle: f64) -> bool {
        self.inner.inject(Injection::GimbalStuck { angle })
    }

    /// Limits max thrust to ``factor`` (clamped to ``[0, 1]``). Faults never
    /// heal: a milder loss than the current one is accepted but changes
    /// nothing.
    fn inject_thrust_loss(&mut self, factor: f64) -> bool {
        self.inner.inject(Injection::ThrustLoss { factor })
    }

    /// Adds a discrete gust (amplitude clamped to ±50 m/s, duration to 30 s).
    fn inject_gust(&mut self, amplitude: f64, duration: f64) -> bool {
        self.inner.inject(Injection::Gust {
            amplitude,
            duration,
        })
    }

    /// Every observable quantity, ordered like ``icarus.TELEMETRY_FIELDS``.
    fn telemetry(&self) -> Vec<f64> {
        self.inner.telemetry().to_vec()
    }

    /// Telemetry as a ``{name: value}`` dict in ``TELEMETRY_FIELDS`` order.
    fn telemetry_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        telemetry_dict(py, &self.inner)
    }

    /// Delayed, noisy observation ``[x, y, vx, vy, theta, omega]``.
    fn observation(&self) -> Vec<f64> {
        self.inner.observation().to_vec()
    }

    /// Quantized throttle level that keeps the initial engine state.
    fn initial_throttle_level(&self) -> u8 {
        self.inner.initial_throttle_level()
    }

    /// An independent copy of the episode in its current state.
    #[pyo3(name = "clone")]
    fn copy_episode(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            config: self.config,
        }
    }

    fn __copy__(&self) -> Self {
        self.copy_episode()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.copy_episode()
    }

    /// Vehicle configuration name.
    #[getter]
    fn config(&self) -> &'static str {
        self.config.name()
    }

    /// Episode seed.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    /// Ticks simulated so far.
    #[getter]
    fn tick(&self) -> u64 {
        self.inner.tick()
    }

    /// Simulated time in seconds (the exact contact time after a touchdown).
    #[getter]
    fn time(&self) -> f64 {
        self.inner.time()
    }

    /// Tick at which the episode is truncated.
    #[getter]
    fn max_ticks(&self) -> u64 {
        self.inner.max_ticks()
    }

    /// Outcome name once the episode has ended, else ``None``.
    #[getter]
    fn outcome(&self) -> Option<&'static str> {
        outcome_name(self.inner.outcome())
    }

    /// Outcome code (0 while running).
    #[getter]
    fn outcome_code(&self) -> u8 {
        self.inner.outcome().map_or(0, Outcome::code)
    }

    /// ``True`` once the episode has ended for any reason.
    #[getter]
    fn done(&self) -> bool {
        self.inner.outcome().is_some()
    }

    /// ``True`` if the episode reached a terminal state (not a time limit).
    #[getter]
    fn terminated(&self) -> bool {
        self.inner.outcome().is_some_and(|o| !o.is_truncation())
    }

    /// ``True`` if the episode was cut by the time limit.
    #[getter]
    fn truncated(&self) -> bool {
        self.inner.outcome().is_some_and(Outcome::is_truncation)
    }

    /// Score (0 unless landed).
    #[getter]
    fn score(&self) -> u32 {
        self.inner.score()
    }

    /// Whether the engine is running.
    #[getter]
    fn engine_on(&self) -> bool {
        self.inner.engine_on()
    }

    /// Ignitions left.
    #[getter]
    fn ignitions_left(&self) -> u32 {
        self.inner.ignitions_left()
    }

    /// Propellant left (kg).
    #[getter]
    fn propellant(&self) -> f64 {
        self.inner.propellant()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sim(config={:?}, seed={}, tick={}, outcome={:?})",
            self.config.name(),
            self.inner.seed(),
            self.inner.tick(),
            self.outcome()
        )
    }
}

/// A recorded episode that can be exported as a replay string.
#[pyclass(module = "icarus", name = "Recorder")]
pub struct PyRecorder {
    inner: RecordingSim,
    config: VehicleConfig,
}

#[pymethods]
impl PyRecorder {
    #[new]
    #[pyo3(signature = (config = "easy", seed = 0))]
    fn new(config: &str, seed: u64) -> PyResult<Self> {
        let config = parse_config(config)?;
        Ok(Self {
            inner: RecordingSim::new(config, seed),
            config,
        })
    }

    /// Advances one tick with quantized controls and records them (nothing is
    /// recorded once the episode has ended). Raises ``ValueError`` above 254.
    fn step_quantized(
        &mut self,
        throttle: u8,
        gimbal: u8,
        rcs: u8,
    ) -> PyResult<Option<&'static str>> {
        let q = quantized(throttle, gimbal, rcs)?;
        Ok(outcome_name(self.inner.step_quantized(q)))
    }

    /// Applies and records a stuck-gimbal fault (see ``Sim.inject_gimbal_stuck``).
    fn inject_gimbal_stuck(&mut self, angle: f64) -> bool {
        self.inner.inject(Injection::GimbalStuck { angle })
    }

    /// Applies and records a thrust-loss fault (see ``Sim.inject_thrust_loss``).
    fn inject_thrust_loss(&mut self, factor: f64) -> bool {
        self.inner.inject(Injection::ThrustLoss { factor })
    }

    /// Applies and records a gust (see ``Sim.inject_gust``).
    fn inject_gust(&mut self, amplitude: f64, duration: f64) -> bool {
        self.inner.inject(Injection::Gust {
            amplitude,
            duration,
        })
    }

    /// Every observable quantity, ordered like ``icarus.TELEMETRY_FIELDS``.
    fn telemetry(&self) -> Vec<f64> {
        self.inner.sim().telemetry().to_vec()
    }

    /// Telemetry as a ``{name: value}`` dict in ``TELEMETRY_FIELDS`` order.
    fn telemetry_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        telemetry_dict(py, self.inner.sim())
    }

    /// Delayed, noisy observation ``[x, y, vx, vy, theta, omega]``.
    fn observation(&self) -> Vec<f64> {
        self.inner.sim().observation().to_vec()
    }

    /// The recording as a URL-safe replay string.
    fn replay(&self) -> PyResult<String> {
        self.inner
            .replay()
            .encode()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Number of recorded ticks.
    #[getter]
    fn recorded_ticks(&self) -> usize {
        self.inner.replay().inputs.len()
    }

    /// Vehicle configuration name.
    #[getter]
    fn config(&self) -> &'static str {
        self.config.name()
    }

    /// Episode seed.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.sim().seed()
    }

    /// Outcome name once the episode has ended, else ``None``.
    #[getter]
    fn outcome(&self) -> Option<&'static str> {
        outcome_name(self.inner.sim().outcome())
    }

    /// ``True`` once the episode has ended for any reason.
    #[getter]
    fn done(&self) -> bool {
        self.inner.sim().outcome().is_some()
    }

    /// Score (0 unless landed).
    #[getter]
    fn score(&self) -> u32 {
        self.inner.sim().score()
    }

    /// Ticks simulated so far.
    #[getter]
    fn tick(&self) -> u64 {
        self.inner.sim().tick()
    }

    fn __repr__(&self) -> String {
        format!(
            "Recorder(config={:?}, seed={}, tick={}, outcome={:?})",
            self.config.name(),
            self.inner.sim().seed(),
            self.inner.sim().tick(),
            self.outcome()
        )
    }
}

/// Re-simulates a replay string and returns the finished simulation.
///
/// Raises ``ValueError`` if the replay is malformed or was recorded with a
/// different physics version.
#[pyfunction]
fn run_replay(encoded: &str) -> PyResult<PySim> {
    let replay = Replay::decode(encoded).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PySim {
        inner: replay.run(),
        config: replay.config,
    })
}

/// Names of the built-in vehicle configurations.
#[pyfunction]
fn configs() -> Vec<&'static str> {
    VehicleConfig::ALL.iter().map(|c| c.name()).collect()
}

/// Name of an outcome code (``None`` for 0 or any unknown code).
#[pyfunction]
fn outcome_from_code(code: i64) -> Option<&'static str> {
    u8::try_from(code)
        .ok()
        .and_then(Outcome::from_code)
        .map(Outcome::name)
}

#[pymodule]
fn icarus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySim>()?;
    m.add_class::<PyRecorder>()?;
    m.add_function(wrap_pyfunction!(run_replay, m)?)?;
    m.add_function(wrap_pyfunction!(configs, m)?)?;
    m.add_function(wrap_pyfunction!(outcome_from_code, m)?)?;
    m.add("PHYSICS_VERSION", PHYSICS_VERSION)?;
    m.add("REPLAY_FORMAT_VERSION", REPLAY_FORMAT_VERSION)?;
    m.add("TELEMETRY_FIELDS", TELEMETRY_FIELDS.to_vec())?;
    m.add(
        "OUTCOMES",
        Outcome::ALL.iter().map(|o| o.name()).collect::<Vec<_>>(),
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
