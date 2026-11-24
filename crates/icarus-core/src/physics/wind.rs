//! Horizontal wind: mean + Ornstein–Uhlenbeck turbulence + discrete gusts.
//!
//! The wind is advanced exactly once per tick, before the integration step,
//! and held constant during the step. The number of random draws per tick
//! depends only on this stream's own history, never on the controller.

use crate::physics::params::WindParams;
use crate::rng::Rng;

/// Maximum number of simultaneous discrete gusts.
pub const MAX_GUSTS: usize = 4;

/// One "1-cosine" gust: `A/2 · (1 − cos(2π t / D))` for `0 ≤ t ≤ D`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gust {
    /// Peak speed (m/s, signed).
    pub amplitude: f64,
    /// Duration (s).
    pub duration: f64,
    /// Time since the gust started (s).
    pub elapsed: f64,
}

impl Gust {
    fn value(&self) -> f64 {
        let phase = core::f64::consts::TAU * self.elapsed / self.duration;
        0.5 * self.amplitude * (1.0 - libm::cos(phase))
    }
}

/// Wind process.
#[derive(Clone, Debug, PartialEq)]
pub struct Wind {
    params: WindParams,
    dt: f64,
    mean: f64,
    turbulence: f64,
    decay: f64,
    diffusion: f64,
    gusts: [Option<Gust>; MAX_GUSTS],
    current: f64,
    rng: Rng,
}

impl Wind {
    /// Creates the process and draws the episode's mean wind. The turbulence
    /// starts from its stationary distribution.
    pub fn new(params: &WindParams, dt: f64, mut rng: Rng) -> Self {
        let mean = params.mean.sample(&mut rng);
        let sigma = params.turbulence_sigma;
        let turbulence = sigma * rng.normal();
        // Exact discretization of dX = −X/τ dt + σ√(2/τ) dW.
        let decay = libm::exp(-dt / params.turbulence_time_constant);
        let diffusion = sigma * (1.0 - decay * decay).sqrt();
        let mut wind = Self {
            params: params.clone(),
            dt,
            mean,
            turbulence,
            decay,
            diffusion,
            gusts: [None; MAX_GUSTS],
            current: 0.0,
            rng,
        };
        wind.current = wind.evaluate();
        wind
    }

    fn evaluate(&self) -> f64 {
        // Explicit fold in slot order: a fixed summation order and a +0.0
        // starting value on every toolchain.
        let gusts = self
            .gusts
            .iter()
            .flatten()
            .fold(0.0, |acc, g| acc + g.value());
        self.mean + self.turbulence + gusts
    }

    /// Wind speed for the current tick (m/s).
    pub fn current(&self) -> f64 {
        self.current
    }

    /// Mean wind of the episode (m/s).
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the wind for this tick, then advances the processes to the
    /// next tick.
    pub fn advance(&mut self) -> f64 {
        let value = self.current;

        // Turbulence: one normal sample per tick.
        self.turbulence = self.decay * self.turbulence + self.diffusion * self.rng.normal();

        // Gusts: age the active ones, retire finished ones.
        for slot in &mut self.gusts {
            if let Some(g) = slot {
                g.elapsed += self.dt;
                if g.elapsed >= g.duration {
                    *slot = None;
                }
            }
        }

        // New gust arrival: one uniform per tick, three more on arrival.
        if self.rng.chance(self.params.gust_rate * self.dt) {
            let amplitude = self.params.gust_amplitude.sample(&mut self.rng) * self.rng.sign();
            let duration = self.params.gust_duration.sample(&mut self.rng);
            self.add_gust(amplitude, duration);
        }

        self.current = self.evaluate();
        value
    }

    /// Adds a gust (for example injected from the web UI). Ignored if all
    /// slots are busy or the values are not finite and positive.
    pub fn add_gust(&mut self, amplitude: f64, duration: f64) -> bool {
        if !(amplitude.is_finite() && duration.is_finite() && duration > 0.0) {
            return false;
        }
        match self.gusts.iter_mut().find(|g| g.is_none()) {
            Some(slot) => {
                *slot = Some(Gust {
                    amplitude,
                    duration,
                    elapsed: 0.0,
                });
                self.current = self.evaluate();
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::params::{Params, Range};
    use crate::rng::Stream;

    fn calm() -> WindParams {
        WindParams {
            mean: Range::fixed(0.0),
            turbulence_sigma: 0.0,
            turbulence_time_constant: 3.0,
            gust_rate: 0.0,
            gust_amplitude: Range::fixed(5.0),
            gust_duration: Range::fixed(2.0),
        }
    }

    #[test]
    fn turbulence_has_the_stationary_variance() {
        let mut p = calm();
        p.turbulence_sigma = 2.0;
        let mut w = Wind::new(&p, Params::DT, Rng::for_stream(1, Stream::Wind));
        let n = 400_000;
        let (mut sum, mut sum_sq) = (0.0, 0.0);
        for _ in 0..n {
            let v = w.advance();
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / f64::from(n);
        let var = sum_sq / f64::from(n) - mean * mean;
        // Strongly autocorrelated samples: generous tolerance.
        assert!((var - 4.0).abs() < 0.4, "variance {var}");
    }

    #[test]
    fn injected_gust_follows_the_one_minus_cosine_shape() {
        let dt = 0.01;
        let mut w = Wind::new(&calm(), dt, Rng::from_seed(3));
        assert!(w.add_gust(6.0, 2.0));
        let mut peak: f64 = 0.0;
        let mut values = Vec::new();
        for _ in 0..250 {
            let v = w.advance();
            peak = peak.max(v);
            values.push(v);
        }
        assert!((peak - 6.0).abs() < 1e-3, "peak {peak}");
        assert_eq!(values[0], 0.0, "starts from zero");
        assert_eq!(*values.last().unwrap(), 0.0, "gone after its duration");
    }

    #[test]
    fn rejects_invalid_gusts_and_respects_slots() {
        let mut w = Wind::new(&calm(), 0.01, Rng::from_seed(3));
        assert!(!w.add_gust(f64::NAN, 1.0));
        assert!(!w.add_gust(1.0, 0.0));
        for _ in 0..MAX_GUSTS {
            assert!(w.add_gust(1.0, 10.0));
        }
        assert!(!w.add_gust(1.0, 10.0));
    }

    #[test]
    fn random_gusts_arrive_at_the_configured_rate() {
        let mut p = calm();
        p.gust_rate = 0.5;
        p.gust_duration = Range::fixed(0.05);
        let dt = 0.01;
        let mut w = Wind::new(&p, dt, Rng::from_seed(5));
        let mut arrivals = 0;
        let mut was_calm = true;
        for _ in 0..200_000 {
            let v = w.advance();
            if v != 0.0 && was_calm {
                arrivals += 1;
            }
            was_calm = v == 0.0;
        }
        // 2000 s at 0.5 /s → about 1000 gusts.
        // Poisson count with mean ≈ 1000 (σ ≈ 32): ±5σ window.
        assert!((840..1160).contains(&arrivals), "arrivals {arrivals}");
    }
}
