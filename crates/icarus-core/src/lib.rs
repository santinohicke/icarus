//! # icarus-core
//!
//! Deterministic 2D physics for landing a reusable rocket first stage.
//!
//! The same code runs natively (Python bindings, training, evaluation) and in
//! the browser (WebAssembly). Every floating-point operation is designed to be
//! bit-identical across those targets:
//!
//! - fixed time step and fixed order of operations,
//! - every transcendental function comes from the pure-Rust [`libm`] crate
//!   (the standard library versions are banned through `clippy.toml`),
//! - no fused multiply-add,
//! - a custom seeded RNG ([`rng`]) with one independent stream per subsystem.
//!
//! The main entry point is [`sim::Sim`]: create it from a vehicle
//! configuration and a seed, then call [`sim::Sim::step`] once per tick.
//!
//! ```
//! use icarus_core::{Controls, Sim, VehicleConfig};
//!
//! let mut sim = Sim::from_config(VehicleConfig::Easy, 42);
//! while sim.outcome().is_none() {
//!     sim.step(Controls::new(0.2, 0.0, 0.0));
//! }
//! println!("{:?} with score {}", sim.outcome(), sim.score());
//! ```

#![forbid(unsafe_code)]

pub mod physics;
pub mod replay;
pub mod rng;
pub mod sim;

pub use physics::params::{Params, VehicleConfig};
pub use physics::scenario::{CrashReason, Outcome};
pub use physics::state::{Controls, State};
pub use replay::{RecordingSim, Replay, ReplayError};
pub use sim::{Injection, Sim, TELEMETRY_FIELDS, TELEMETRY_LEN};

/// Version of the physics model.
///
/// Bump it whenever a change alters trajectories (dynamics, parameters,
/// integration, RNG usage). Replays and golden parity files record it, so a
/// replay made with another physics version is rejected instead of silently
/// re-simulating to a different result.
pub const PHYSICS_VERSION: u16 = 1;
