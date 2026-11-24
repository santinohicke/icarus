//! Classical controllers that fly the simulator through the same interface as
//! a human or an agent: quantized inputs, one per tick.
//!
//! A controller only sees what a flight computer would know (see
//! [`Observation`]): the delayed, noisy navigation sensors, its own engine
//! state and the propellant gauge. It never reads the true state, the wind or
//! the injected faults. Like the physics, controllers use `libm` for every
//! transcendental function, so they make bit-identical decisions natively, in
//! Python and in WebAssembly.

pub mod pid;

pub use pid::{Pid, PidGains};

use crate::physics::params::Params;
use crate::physics::state::Controls;
use crate::sim::Sim;

/// What the flight computer knows at the start of a tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    /// Measured horizontal position of the center of mass (m), pad at 0.
    pub x: f64,
    /// Measured altitude of the center of mass (m).
    pub y: f64,
    /// Measured horizontal velocity (m/s).
    pub vx: f64,
    /// Measured vertical velocity (m/s).
    pub vy: f64,
    /// Measured tilt, counter-clockwise from vertical (rad).
    pub theta: f64,
    /// Measured angular velocity, counter-clockwise (rad/s).
    pub omega: f64,
    /// Total mass from the propellant gauge (kg).
    pub mass: f64,
    /// Whether the engine is running.
    pub engine_on: bool,
    /// Ignitions left.
    pub ignitions_left: u32,
}

impl Observation {
    /// The observation of a simulation at its current tick.
    pub fn from_sim(sim: &Sim) -> Self {
        let [x, y, vx, vy, theta, omega] = sim.observation();
        Self {
            x,
            y,
            vx,
            vy,
            theta,
            omega,
            mass: sim.params().vehicle.dry_mass + sim.propellant(),
            engine_on: sim.engine_on(),
            ignitions_left: sim.ignitions_left(),
        }
    }
}

/// A controller that maps observations to control commands.
pub trait Controller {
    /// Short identifier (`"pid"`).
    fn name(&self) -> &'static str;

    /// Forgets internal state (integrators, latches) for a new episode.
    fn reset(&mut self);

    /// Commands for this tick. Called exactly once per tick, before
    /// [`Sim::step`].
    fn act(&mut self, obs: &Observation) -> Controls;

    /// Commands for this tick as quantized levels, exactly what a recorded
    /// run stores (so a controller's flight can be replayed).
    fn act_quantized(&mut self, obs: &Observation) -> [u8; 3] {
        self.act(obs).quantized()
    }
}

/// Built-in controllers, by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ControllerKind {
    /// Cascaded PID with an ignition trigger.
    Pid,
}

impl ControllerKind {
    /// Every built-in controller.
    pub const ALL: [ControllerKind; 1] = [ControllerKind::Pid];

    /// Lower-case name.
    pub const fn name(self) -> &'static str {
        match self {
            ControllerKind::Pid => "pid",
        }
    }

    /// Inverse of [`ControllerKind::name`].
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == name)
    }

    /// A fresh controller of this kind for the given parameters.
    pub fn build(self, params: &Params) -> Box<dyn Controller + Send + Sync> {
        match self {
            ControllerKind::Pid => Box::new(Pid::new(params)),
        }
    }
}
