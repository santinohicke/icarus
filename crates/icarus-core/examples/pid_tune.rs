//! Development tool: random search over the PID gains.
//!
//! Tunes on seeds 1_000_000.. (the evaluation battery uses 0..), so the
//! published numbers are not measured on the seeds the gains were fit to.
//!
//!     cargo run --release -p icarus-core --example pid_tune [candidates] [seeds]

use icarus_core::control::PidGains;
use icarus_core::{Controller, Controls, Observation, Outcome, Pid, Sim, VehicleConfig};

const TUNE_SEED_BASE: u64 = 1_000_000;

fn landed(config: VehicleConfig, gains: PidGains, seeds: u64) -> u32 {
    let mut n = 0;
    for s in 0..seeds {
        let mut sim = Sim::from_config(config, TUNE_SEED_BASE + s);
        let mut pid = Pid::with_gains(sim.params(), gains);
        while sim.outcome().is_none() {
            let q = pid.act_quantized(&Observation::from_sim(&sim));
            sim.step(Controls::from_quantized(q));
        }
        n += u32::from(sim.outcome() == Some(Outcome::Landed));
    }
    n
}

/// Small deterministic generator for the search itself.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Perturbs `v` by up to ±spread (relative) with probability 0.35, else keeps it.
    fn around(&mut self, v: f64, spread: f64) -> f64 {
        if self.next() < 0.65 {
            return v;
        }
        v * (1.0 + spread * (2.0 * self.next() - 1.0))
    }
}

fn score(gains: PidGains, seeds: u64) -> (u32, Vec<u32>) {
    let per: Vec<u32> = VehicleConfig::CAMPAIGN
        .iter()
        .map(|&c| landed(c, gains, seeds))
        .collect();
    // Every level counts the same; the worst level counts twice.
    let worst = per.iter().copied().min().unwrap_or(0);
    (per.iter().sum::<u32>() + worst, per)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let candidates: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let seeds: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);
    let mut best = PidGains::default();
    let (mut best_score, per) = score(best, seeds);
    println!("default: {best_score} {per:?}");
    let mut rng = Lcg(12345);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut round = 0;
    while round * threads < candidates {
        let base = best;
        let cands: Vec<PidGains> = (0..threads)
            .map(|_| {
                let mut g = base;
                g.decel_fraction = rng.around(g.decel_fraction, 0.25).clamp(0.3, 0.85);
                g.kp_vy = rng.around(g.kp_vy, 0.4);
                g.ki_vy = rng.around(g.ki_vy, 0.5);
                g.hoverslam_touchdown_fraction = rng
                    .around(g.hoverslam_touchdown_fraction, 0.15)
                    .clamp(0.4, 0.85);
                g.ignition_lag = rng.around(g.ignition_lag, 0.4);
                g.ignition_margin = rng.around(g.ignition_margin, 0.5);
                g.kp_x = rng.around(g.kp_x, 0.5);
                g.kd_x = rng.around(g.kd_x, 0.4);
                g.max_tilt = rng.around(g.max_tilt, 0.3).clamp(0.05, 0.3);
                g.tilt_altitude = rng.around(g.tilt_altitude, 0.4);
                g.min_tilt = rng.around(g.min_tilt, 0.4).clamp(0.01, 0.15);
                g.attitude_omega = rng.around(g.attitude_omega, 0.3);
                g.attitude_ki = rng.around(g.attitude_ki, 0.5);
                g.final_height = rng.around(g.final_height, 0.4);
                g.final_decel_factor = rng.around(g.final_decel_factor, 0.3).clamp(1.1, 4.0);
                g.flare_height = rng.around(g.flare_height, 0.5);
                g.align_height = rng.around(g.align_height, 0.4);
                g.observer_gain = rng.around(g.observer_gain, 0.5).clamp(0.0, 0.1);
                g.position_hold_height = rng.around(g.position_hold_height, 0.5);
                g.min_time_to_go = rng.around(g.min_time_to_go, 0.4).clamp(1.0, 20.0);
                g.upright_height = rng.around(g.upright_height, 0.4);
                g.upright_tilt = rng.around(g.upright_tilt, 0.4).clamp(0.005, 0.06);
                g.upright_kd = rng.around(g.upright_kd, 0.5);
                g.touchdown_speed_fraction =
                    rng.around(g.touchdown_speed_fraction, 0.2).clamp(0.25, 0.8);
                g
            })
            .collect();
        let results: Vec<(u32, Vec<u32>, PidGains)> = std::thread::scope(|s| {
            let handles: Vec<_> = cands
                .iter()
                .map(|&g| {
                    s.spawn(move || {
                        let (sc, per) = score(g, seeds);
                        (sc, per, g)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (sc, per, g) in results {
            if sc > best_score {
                best_score = sc;
                best = g;
                println!("round {round}: {sc} {per:?}");
            }
        }
        round += 1;
    }
    println!("best: {best_score}\n{best:#?}");
}
