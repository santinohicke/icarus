//! Navigation sensors: Gaussian noise and a fixed observation delay.
//!
//! A measurement `[x, y, vx, vy, θ, ω]` with fresh noise is taken every tick
//! and stored in a ring buffer; the observation returned is the one taken
//! `delay_ticks` ticks ago. Six normal samples are drawn every tick, so the
//! noise sequence never depends on the controller.

use crate::physics::params::SensorParams;
use crate::physics::state::State;
use crate::rng::Rng;

/// Number of measured quantities.
pub const MEAS_DIM: usize = 6;

/// Maximum supported observation delay (ticks).
pub const MAX_DELAY_TICKS: usize = 12;

const BUFFER_LEN: usize = MAX_DELAY_TICKS + 1;

/// Sensor model with its noise stream and delay line.
#[derive(Clone, Debug, PartialEq)]
pub struct Sensors {
    sigma: [f64; MEAS_DIM],
    delay: usize,
    buffer: [[f64; MEAS_DIM]; BUFFER_LEN],
    head: usize,
    last_noise: [f64; MEAS_DIM],
    rng: Rng,
}

impl Sensors {
    /// Creates the sensors and fills the delay line with a first measurement
    /// of `initial`, so a delayed observation is defined from tick 0.
    pub fn new(params: &SensorParams, rng: Rng, initial: &State) -> Self {
        let mut sensors = Self {
            sigma: params.noise_sigma,
            delay: params.delay_ticks.min(MAX_DELAY_TICKS),
            buffer: [[0.0; MEAS_DIM]; BUFFER_LEN],
            head: 0,
            last_noise: [0.0; MEAS_DIM],
            rng,
        };
        let first = sensors.sample(initial);
        sensors.buffer = [first; BUFFER_LEN];
        sensors
    }

    fn sample(&mut self, s: &State) -> [f64; MEAS_DIM] {
        let truth = [s.x, s.y, s.vx, s.vy, s.theta, s.omega];
        let noise: [f64; MEAS_DIM] = core::array::from_fn(|i| self.sigma[i] * self.rng.normal());
        self.last_noise = noise;
        core::array::from_fn(|i| truth[i] + noise[i])
    }

    /// Takes a new measurement of `s`.
    pub fn measure(&mut self, s: &State) {
        let m = self.sample(s);
        self.head = (self.head + 1) % BUFFER_LEN;
        self.buffer[self.head] = m;
    }

    /// The delayed, noisy observation `[x, y, vx, vy, θ, ω]`.
    pub fn observation(&self) -> [f64; MEAS_DIM] {
        let idx = (self.head + BUFFER_LEN - self.delay) % BUFFER_LEN;
        self.buffer[idx]
    }

    /// Observation delay in ticks.
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// Noise added to the most recent measurement (for tests and debugging).
    pub fn last_noise(&self) -> [f64; MEAS_DIM] {
        self.last_noise
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_at(y: f64) -> State {
        State {
            y,
            ..Default::default()
        }
    }

    #[test]
    fn delay_returns_measurement_from_k_ticks_ago() {
        let params = SensorParams {
            delay_ticks: 3,
            noise_sigma: [0.0; MEAS_DIM],
        };
        let mut s = Sensors::new(&params, Rng::from_seed(1), &state_at(0.0));
        for k in 1..=10 {
            s.measure(&state_at(f64::from(k)));
            let expected = f64::from(k - 3).max(0.0);
            assert_eq!(s.observation()[1], expected, "tick {k}");
        }
    }

    #[test]
    fn zero_delay_returns_latest_measurement() {
        let params = SensorParams {
            delay_ticks: 0,
            noise_sigma: [0.0; MEAS_DIM],
        };
        let mut s = Sensors::new(&params, Rng::from_seed(1), &state_at(5.0));
        assert_eq!(s.observation()[1], 5.0);
        s.measure(&state_at(6.0));
        assert_eq!(s.observation()[1], 6.0);
    }

    #[test]
    fn max_delay_wraps_correctly() {
        let params = SensorParams {
            delay_ticks: MAX_DELAY_TICKS,
            noise_sigma: [0.0; MEAS_DIM],
        };
        let mut s = Sensors::new(&params, Rng::from_seed(1), &state_at(0.0));
        for k in 1..=50u32 {
            s.measure(&state_at(f64::from(k)));
            let expected = (f64::from(k) - MAX_DELAY_TICKS as f64).max(0.0);
            assert_eq!(s.observation()[1], expected);
        }
    }

    #[test]
    fn noise_has_the_configured_scale() {
        let params = SensorParams {
            delay_ticks: 0,
            noise_sigma: [2.0; MEAS_DIM],
        };
        let mut s = Sensors::new(&params, Rng::from_seed(9), &state_at(0.0));
        let n = 50_000;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            s.measure(&state_at(0.0));
            sum_sq += s.observation()[1] * s.observation()[1];
        }
        let var = sum_sq / f64::from(n);
        assert!((var - 4.0).abs() < 0.2, "variance {var}");
    }
}
