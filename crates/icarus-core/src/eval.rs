//! The evaluation battery: fixed scenarios, run identically for every
//! controller (classical baselines now, the RL agent later).
//!
//! A scenario is a vehicle configuration plus a stress applied to its
//! parameters (a large initial tilt, strong gusts, a scheduled failure...).
//! Episodes are fully determined by `(config, scenario, seed, controller)`.

use crate::control::{Controller, Observation};
use crate::physics::params::{FaultParams, Params, Range, VehicleConfig};
use crate::physics::scenario::Outcome;
use crate::physics::state::Controls;
use crate::sim::Sim;

const DEG: f64 = core::f64::consts::PI / 180.0;

/// The scenarios of the evaluation battery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scenario {
    /// The configuration as is: efficiency and smoothness.
    Nominal,
    /// Initial tilt between 30° and 35°: the nonlinear regime.
    InitialTilt,
    /// Frequent strong gusts on top of stronger turbulence.
    StrongGusts,
    /// Sensors delayed by 10 ticks (167 ms).
    DelayedSensors,
    /// The gimbal jams at 1° between 3 and 8 s: survivable only if the RCS
    /// can cancel the torque.
    GimbalStuck,
    /// The engine loses 20% of its thrust between 3 and 8 s.
    ThrustLoss,
    /// Wind, delay and failures beyond every training range.
    OutOfDistribution,
}

impl Scenario {
    /// Every scenario, in battery order.
    pub const ALL: [Scenario; 7] = [
        Scenario::Nominal,
        Scenario::InitialTilt,
        Scenario::StrongGusts,
        Scenario::DelayedSensors,
        Scenario::GimbalStuck,
        Scenario::ThrustLoss,
        Scenario::OutOfDistribution,
    ];

    /// Lower-case name.
    pub const fn name(self) -> &'static str {
        match self {
            Scenario::Nominal => "nominal",
            Scenario::InitialTilt => "initial_tilt",
            Scenario::StrongGusts => "strong_gusts",
            Scenario::DelayedSensors => "delayed_sensors",
            Scenario::GimbalStuck => "gimbal_stuck",
            Scenario::ThrustLoss => "thrust_loss",
            Scenario::OutOfDistribution => "out_of_distribution",
        }
    }

    /// Inverse of [`Scenario::name`].
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The parameters of this scenario on top of a configuration.
    pub fn params(self, config: VehicleConfig) -> Params {
        let mut p = config.params();
        match self {
            Scenario::Nominal => {}
            Scenario::InitialTilt => {
                // One side only (the dynamics are mirror-symmetric):
                // `symmetric(35°)` would put most draws below 30°.
                p.start.tilt = Range::new(30.0 * DEG, 35.0 * DEG);
                p.start.angular_rate = Range::symmetric(0.02);
            }
            Scenario::StrongGusts => {
                p.wind.turbulence_sigma = 3.0;
                p.wind.gust_rate = 1.0 / 5.0;
                p.wind.gust_amplitude = Range::new(10.0, 18.0);
                p.wind.gust_duration = Range::new(2.0, 5.0);
            }
            Scenario::DelayedSensors => {
                p.sensors.delay_ticks = 10;
            }
            Scenario::GimbalStuck => {
                p.faults = FaultParams {
                    probability: 1.0,
                    gimbal_stuck_share: 1.0,
                    trigger_time: Range::new(3.0, 8.0),
                    stuck_angle: Range::fixed(1.0 * DEG),
                    thrust_factor: Range::fixed(1.0),
                };
            }
            Scenario::ThrustLoss => {
                p.faults = FaultParams {
                    probability: 1.0,
                    gimbal_stuck_share: 0.0,
                    trigger_time: Range::new(3.0, 8.0),
                    stuck_angle: Range::fixed(0.0),
                    thrust_factor: Range::fixed(0.8),
                };
            }
            Scenario::OutOfDistribution => {
                p.wind.mean = Range::symmetric(18.0);
                p.wind.turbulence_sigma = 4.0;
                p.wind.gust_rate = 1.0 / 6.0;
                p.wind.gust_amplitude = Range::new(12.0, 20.0);
                p.sensors.delay_ticks = 8;
                p.faults = FaultParams {
                    probability: 1.0,
                    gimbal_stuck_share: 0.5,
                    trigger_time: Range::new(3.0, 8.0),
                    stuck_angle: Range::new(1.2 * DEG, 1.5 * DEG),
                    thrust_factor: Range::new(0.7, 0.75),
                };
            }
        }
        p
    }
}

/// What one episode produced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpisodeResult {
    /// How the episode ended.
    pub outcome: Outcome,
    /// Score (0 unless landed).
    pub score: u32,
    /// Ticks simulated.
    pub ticks: u64,
    /// Vertical speed of the lowest contact point at touchdown (m/s), or the
    /// final vertical speed if there was no touchdown.
    pub contact_vy: f64,
    /// Horizontal speed of the lowest contact point at touchdown (m/s).
    pub contact_vx: f64,
    /// Horizontal distance of the tail from the pad center at the end (m).
    pub position_error: f64,
    /// Propellant burnt (kg).
    pub propellant_used: f64,
    /// Final tilt (rad).
    pub tilt: f64,
}

impl EpisodeResult {
    /// Whether the stage landed.
    pub fn landed(&self) -> bool {
        self.outcome == Outcome::Landed
    }
}

/// Runs one episode of `params` with `seed` flown by `controller`.
///
/// The controller is reset first. Returns `None` only if the parameters are
/// invalid.
pub fn run_episode(
    params: Params,
    seed: u64,
    controller: &mut dyn Controller,
) -> Option<EpisodeResult> {
    let mut sim = Sim::new(params, seed).ok()?;
    controller.reset();
    while sim.outcome().is_none() {
        let q = controller.act_quantized(&Observation::from_sim(&sim));
        sim.step(Controls::from_quantized(q));
    }
    Some(summarize(&sim))
}

/// The result of a finished simulation.
pub fn summarize(sim: &Sim) -> EpisodeResult {
    let geometry = sim.contact().copied().unwrap_or_else(|| sim.geometry());
    EpisodeResult {
        outcome: sim.outcome().unwrap_or(Outcome::TimeLimit),
        score: sim.score(),
        ticks: sim.tick(),
        contact_vy: geometry.point_velocity.1,
        contact_vx: geometry.point_velocity.0,
        position_error: geometry.tail.0.abs(),
        propellant_used: sim.initial_propellant() - sim.propellant(),
        tilt: crate::physics::scenario::wrap_angle(sim.state().theta),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::Pid;

    #[test]
    fn every_scenario_is_valid_on_every_config() {
        for config in VehicleConfig::ALL {
            for scenario in Scenario::ALL {
                scenario
                    .params(config)
                    .validate()
                    .unwrap_or_else(|e| panic!("{} / {}: {e}", config.name(), scenario.name()));
            }
        }
    }

    #[test]
    fn names_round_trip() {
        for s in Scenario::ALL {
            assert_eq!(Scenario::from_name(s.name()), Some(s));
        }
        assert_eq!(Scenario::from_name("nope"), None);
    }

    #[test]
    fn episodes_are_deterministic_and_summarized() {
        let params = Scenario::ThrustLoss.params(VehicleConfig::Easy);
        let mut pid = Pid::new(&params);
        let a = run_episode(params.clone(), 11, &mut pid).unwrap();
        let b = run_episode(params, 11, &mut pid).unwrap();
        assert_eq!(a, b, "reset must make reruns identical");
        assert!(a.ticks > 0 && a.propellant_used > 0.0);
    }

    #[test]
    fn failures_actually_happen() {
        for (scenario, check) in [(Scenario::GimbalStuck, true), (Scenario::ThrustLoss, false)] {
            let mut sim = Sim::new(scenario.params(VehicleConfig::Easy), 2).unwrap();
            for _ in 0..(9 * 60) {
                sim.step(Controls::new(0.5, 0.0, 0.0));
            }
            if check {
                assert!(
                    sim.gimbal_stuck().is_some(),
                    "gimbal should be stuck by 9 s"
                );
            } else {
                assert!(sim.thrust_factor() < 1.0, "thrust should be reduced by 9 s");
            }
        }
    }
}
