//! Python bindings for the Icarus simulator (`import icarus`).
//!
//! Exposes the same Rust core the web uses, so training, evaluation and the
//! browser run bit-identical physics.
//!
//! Thread safety: objects are `Send + Sync` and the module supports
//! free-threaded Python, but each `Sim`/`Recorder` must be used by one thread
//! at a time (concurrent calls on the same object raise `RuntimeError:
//! Already borrowed`).

mod env;

use icarus_core::physics::scenario::Outcome;
use icarus_core::physics::state::QUANT_MAX;
use icarus_core::replay::REPLAY_FORMAT_VERSION;
use std::time::Instant;

use icarus_core::control::{Controller, ControllerKind, Observation};
use icarus_core::eval::{Scenario, summarize};
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

fn parse_scenario(name: &str) -> PyResult<Scenario> {
    Scenario::from_name(name).ok_or_else(|| {
        let valid: Vec<&str> = Scenario::ALL.iter().map(|s| s.name()).collect();
        PyValueError::new_err(format!(
            "unknown scenario {name:?}; expected one of {valid:?}"
        ))
    })
}

fn parse_controller(name: &str) -> PyResult<ControllerKind> {
    ControllerKind::from_name(name).ok_or_else(|| {
        let valid: Vec<&str> = ControllerKind::ALL.iter().map(|c| c.name()).collect();
        PyValueError::new_err(format!(
            "unknown controller {name:?}; expected one of {valid:?}"
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
///     scenario: an evaluation scenario applied on top of the configuration,
///         see ``scenarios()`` (``"nominal"`` keeps it as is).
#[pyclass(module = "icarus", name = "Sim")]
pub struct PySim {
    inner: Sim,
    config: VehicleConfig,
    scenario: Scenario,
}

#[pymethods]
impl PySim {
    #[new]
    #[pyo3(signature = (config = "easy", seed = 0, scenario = "nominal"))]
    fn new(config: &str, seed: u64, scenario: &str) -> PyResult<Self> {
        let config = parse_config(config)?;
        let scenario = parse_scenario(scenario)?;
        let inner = Sim::new(scenario.params(config), seed)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            config,
            scenario,
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
            scenario: self.scenario,
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

    /// Evaluation scenario name (``"nominal"`` unless given).
    #[getter]
    fn scenario(&self) -> &'static str {
        self.scenario.name()
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
        scenario: Scenario::Nominal,
    })
}

/// A classical controller that flies a ``Sim`` through the same quantized
/// interface as a player: ``sim.step_quantized(*ctrl.act(sim))``.
///
/// It only reads what a flight computer would know (delayed, noisy sensors,
/// its engine state and propellant gauge), never the true state or faults.
///
/// Args:
///     controller: a controller name, see ``controllers()``.
///     config: the configuration it will fly (its vehicle parameters).
///     scenario: the scenario it will fly (e.g. the sensor delay it compensates).
#[pyclass(module = "icarus", name = "Controller")]
pub struct PyController {
    inner: Box<dyn Controller + Send + Sync>,
    kind: ControllerKind,
    config: VehicleConfig,
    scenario: Scenario,
    /// Tick of the last decision, to refuse a second decision in one tick
    /// (it would advance the controller's internal state twice).
    last_tick: Option<u64>,
}

impl PyController {
    fn decide(
        &mut self,
        sim: &Sim,
        config: VehicleConfig,
        scenario: Scenario,
    ) -> PyResult<[u8; 3]> {
        if config != self.config || scenario != self.scenario {
            return Err(PyValueError::new_err(format!(
                "controller built for {:?}/{:?} cannot fly a {:?}/{:?} sim",
                self.config.name(),
                self.scenario.name(),
                config.name(),
                scenario.name()
            )));
        }
        let tick = sim.tick();
        if self.last_tick == Some(tick) {
            return Err(PyValueError::new_err(
                "act() was already called for this tick; step the sim first (or reset())",
            ));
        }
        self.last_tick = Some(tick);
        Ok(self.inner.act_quantized(&Observation::from_sim(sim)))
    }
}

#[pymethods]
impl PyController {
    #[new]
    #[pyo3(signature = (controller = "pid", config = "easy", scenario = "nominal"))]
    fn new(controller: &str, config: &str, scenario: &str) -> PyResult<Self> {
        let kind = parse_controller(controller)?;
        let config = parse_config(config)?;
        let scenario = parse_scenario(scenario)?;
        Ok(Self {
            inner: kind.build(&scenario.params(config)),
            kind,
            config,
            scenario,
            last_tick: None,
        })
    }

    /// Quantized commands ``(throttle, gimbal, rcs)`` for the sim's current
    /// tick. Call once per tick, before stepping. Raises ``ValueError`` if
    /// the sim's configuration or scenario differs from the controller's, or
    /// if this tick was already decided.
    fn act(&mut self, sim: PyRef<'_, PySim>) -> PyResult<(u8, u8, u8)> {
        let [a, b, c] = self.decide(&sim.inner, sim.config, sim.scenario)?;
        Ok((a, b, c))
    }

    /// The same for a ``Recorder`` (nominal scenario only), so a controller's
    /// flight can be replayed.
    fn act_recorder(&mut self, recorder: PyRef<'_, PyRecorder>) -> PyResult<(u8, u8, u8)> {
        let [a, b, c] = self.decide(recorder.inner.sim(), recorder.config, Scenario::Nominal)?;
        Ok((a, b, c))
    }

    /// Forgets the internal state, for a new episode.
    fn reset(&mut self) {
        self.inner.reset();
        self.last_tick = None;
    }

    /// Controller name.
    #[getter]
    fn name(&self) -> &'static str {
        self.kind.name()
    }
}

/// Runs one controller on one configuration and scenario over the given
/// seeds, in Rust (the GIL is released while it runs).
///
/// Returns one dict per seed with: ``seed``, ``outcome``, ``landed``,
/// ``score``, ``ticks``, ``contact_vx``, ``contact_vy``, ``position_error``,
/// ``propellant_used``, ``tilt`` and ``act_ns`` (mean wall-clock time of one
/// controller decision in nanoseconds: the only non-deterministic field).
#[pyfunction]
#[pyo3(signature = (controller, config, scenario, seeds))]
fn evaluate<'py>(
    py: Python<'py>,
    controller: &str,
    config: &str,
    scenario: &str,
    seeds: Vec<u64>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let kind = parse_controller(controller)?;
    let params = parse_scenario(scenario)?.params(parse_config(config)?);
    params
        .validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let results = py.detach(|| {
        let mut ctrl = kind.build(&params);
        seeds
            .iter()
            .map(|&seed| {
                let mut sim = Sim::new(params.clone(), seed).expect("parameters validated above");
                ctrl.reset();
                let mut act_time: u128 = 0;
                while sim.outcome().is_none() {
                    let obs = Observation::from_sim(&sim);
                    let start = Instant::now();
                    let q = ctrl.act_quantized(&obs);
                    act_time += start.elapsed().as_nanos();
                    sim.step(Controls::from_quantized(q));
                }
                let ticks = sim.tick().max(1);
                (seed, summarize(&sim), act_time as f64 / ticks as f64)
            })
            .collect::<Vec<_>>()
    });
    results
        .into_iter()
        .map(|(seed, r, act_ns)| {
            let d = PyDict::new(py);
            d.set_item("seed", seed)?;
            d.set_item("outcome", r.outcome.name())?;
            d.set_item("landed", r.landed())?;
            d.set_item("score", r.score)?;
            d.set_item("ticks", r.ticks)?;
            d.set_item("contact_vx", r.contact_vx)?;
            d.set_item("contact_vy", r.contact_vy)?;
            d.set_item("position_error", r.position_error)?;
            d.set_item("propellant_used", r.propellant_used)?;
            d.set_item("tilt", r.tilt)?;
            d.set_item("act_ns", act_ns)?;
            Ok(d)
        })
        .collect()
}

/// Names of the evaluation scenarios.
#[pyfunction]
fn scenarios() -> Vec<&'static str> {
    Scenario::ALL.iter().map(|s| s.name()).collect()
}

/// Names of the built-in controllers.
#[pyfunction]
fn controllers() -> Vec<&'static str> {
    ControllerKind::ALL.iter().map(|c| c.name()).collect()
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
    m.add_class::<PyController>()?;
    m.add_class::<env::PyEnv>()?;
    m.add_class::<env::PyVecEnv>()?;
    m.add_function(wrap_pyfunction!(env::episode_seed, m)?)?;
    m.add_function(wrap_pyfunction!(env::default_env_config, m)?)?;
    m.add_function(wrap_pyfunction!(env::obs_fields, m)?)?;
    m.add_function(wrap_pyfunction!(env::override_keys, m)?)?;
    m.add("ACTION_DIM", icarus_core::env::ACTION_DIM)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(scenarios, m)?)?;
    m.add_function(wrap_pyfunction!(controllers, m)?)?;
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
