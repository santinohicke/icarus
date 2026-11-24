//! Actuator and engine faults.

use crate::physics::params::FaultParams;
use crate::rng::Rng;

/// A fault that changes the vehicle's behavior from the moment it happens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fault {
    /// The gimbal runs to `angle` (rad) at its slew rate and jams there.
    GimbalStuck {
        /// Jam angle (rad).
        angle: f64,
    },
    /// Max thrust is multiplied by `factor` (in `[0, 1]`).
    ThrustLoss {
        /// Remaining fraction of max thrust.
        factor: f64,
    },
}

/// Active faults plus the random schedule of the episode.
#[derive(Clone, Debug, PartialEq)]
pub struct FaultState {
    scheduled: Option<(f64, Fault)>,
    gimbal_stuck: Option<f64>,
    thrust_factor: f64,
}

impl FaultState {
    /// Draws the episode's fault schedule. Always makes the same six draws
    /// (two chances, trigger time, angle, sign, factor), so the stream's use
    /// never depends on the outcome.
    pub fn new(params: &FaultParams, mut rng: Rng) -> Self {
        let has_fault = rng.chance(params.probability);
        let is_gimbal = rng.chance(params.gimbal_stuck_share);
        let time = params.trigger_time.sample(&mut rng);
        let stuck = params.stuck_angle.sample(&mut rng) * rng.sign();
        let factor = params.thrust_factor.sample(&mut rng);
        let scheduled = has_fault.then(|| {
            let fault = if is_gimbal {
                Fault::GimbalStuck { angle: stuck }
            } else {
                Fault::ThrustLoss { factor }
            };
            (time, fault)
        });
        Self {
            scheduled,
            gimbal_stuck: None,
            thrust_factor: 1.0,
        }
    }

    /// The scheduled fault and its trigger time, if it has not happened yet.
    pub fn scheduled(&self) -> Option<(f64, Fault)> {
        self.scheduled
    }

    /// Triggers the scheduled fault once `time` reaches its trigger time.
    pub fn update(&mut self, time: f64) {
        if let Some((t, fault)) = self.scheduled {
            if time >= t {
                self.scheduled = None;
                self.apply(fault);
            }
        }
    }

    /// Applies a fault now. Values are clamped to the physically meaningful
    /// range (`gimbal_max` for the angle, `[0, 1]` for the factor); non-finite
    /// values are ignored.
    pub fn apply_clamped(&mut self, fault: Fault, gimbal_max: f64) -> bool {
        match fault {
            Fault::GimbalStuck { angle } if angle.is_finite() => {
                self.apply(Fault::GimbalStuck {
                    angle: angle.clamp(-gimbal_max, gimbal_max),
                });
                true
            }
            Fault::ThrustLoss { factor } if factor.is_finite() => {
                self.apply(Fault::ThrustLoss {
                    factor: factor.clamp(0.0, 1.0),
                });
                true
            }
            _ => false,
        }
    }

    fn apply(&mut self, fault: Fault) {
        match fault {
            Fault::GimbalStuck { angle } => self.gimbal_stuck = Some(angle),
            // Faults never heal: keep the worst thrust loss.
            Fault::ThrustLoss { factor } => self.thrust_factor = self.thrust_factor.min(factor),
        }
    }

    /// Jam angle of the gimbal, if it is stuck.
    pub fn gimbal_stuck(&self) -> Option<f64> {
        self.gimbal_stuck
    }

    /// Remaining fraction of max thrust.
    pub fn thrust_factor(&self) -> f64 {
        self.thrust_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::params::{DEG, Params};

    #[test]
    fn no_fault_when_probability_is_zero() {
        let p = Params::easy().faults;
        for seed in 0..100 {
            assert_eq!(FaultState::new(&p, Rng::from_seed(seed)).scheduled(), None);
        }
    }

    #[test]
    fn scheduled_fault_triggers_once() {
        let mut p = Params::easy().faults;
        p.probability = 1.0;
        p.gimbal_stuck_share = 0.0;
        let mut f = FaultState::new(&p, Rng::from_seed(1));
        let (t, _) = f.scheduled().expect("a fault is scheduled");
        f.update(t - 0.01);
        assert_eq!(f.thrust_factor(), 1.0);
        f.update(t);
        assert_eq!(f.thrust_factor(), 0.8);
        assert_eq!(f.scheduled(), None);
    }

    #[test]
    fn manual_faults_are_clamped_and_never_heal() {
        let mut f = FaultState::new(&Params::easy().faults, Rng::from_seed(1));
        assert!(f.apply_clamped(Fault::GimbalStuck { angle: 1.0 }, 7.0 * DEG));
        assert_eq!(f.gimbal_stuck(), Some(7.0 * DEG));
        assert!(f.apply_clamped(Fault::ThrustLoss { factor: 0.5 }, 7.0 * DEG));
        assert!(f.apply_clamped(Fault::ThrustLoss { factor: 0.9 }, 7.0 * DEG));
        assert_eq!(f.thrust_factor(), 0.5);
        assert!(!f.apply_clamped(Fault::ThrustLoss { factor: f64::NAN }, 7.0 * DEG));
    }
}
