//! Determinism and independence of the random streams.

use icarus_core::physics::params::Range;
use icarus_core::rng::Rng;
use icarus_core::sim::tm;
use icarus_core::{Controls, Params, Sim, TELEMETRY_LEN, VehicleConfig};

/// Pseudo-random but reproducible controls.
fn scripted_controls(seed: u64, ticks: usize) -> Vec<Controls> {
    let mut rng = Rng::from_seed(seed);
    let mut out = Vec::with_capacity(ticks);
    let mut c = Controls::new(0.0, 0.0, 0.0);
    for i in 0..ticks {
        if i % 15 == 0 {
            c = Controls::new(
                rng.uniform() * 1.6 - 0.6,
                rng.uniform() * 2.0 - 1.0,
                rng.uniform() * 2.0 - 1.0,
            );
        }
        out.push(c);
    }
    out
}

fn run_bits(config: VehicleConfig, seed: u64, controls: &[Controls]) -> Vec<[u64; TELEMETRY_LEN]> {
    let mut sim = Sim::from_config(config, seed);
    let mut out = vec![sim.telemetry().map(f64::to_bits)];
    for &c in controls {
        let done = sim.step(c).is_some();
        out.push(sim.telemetry().map(f64::to_bits));
        if done {
            break;
        }
    }
    out
}

#[test]
fn same_seed_same_trajectory() {
    let controls = scripted_controls(1, 3_000);
    for config in VehicleConfig::ALL {
        assert_eq!(
            run_bits(config, 42, &controls),
            run_bits(config, 42, &controls)
        );
    }
}

#[test]
fn different_seeds_differ() {
    let controls = scripted_controls(1, 100);
    assert_ne!(
        run_bits(VehicleConfig::Easy, 1, &controls),
        run_bits(VehicleConfig::Easy, 2, &controls)
    );
}

#[test]
fn cloned_simulation_continues_identically() {
    let controls = scripted_controls(3, 600);
    let mut a = Sim::from_config(VehicleConfig::Realistic, 8);
    for &c in &controls[..200] {
        a.step(c);
    }
    let mut b = a.clone();
    for &c in &controls[200..] {
        a.step(c);
        b.step(c);
        assert_eq!(
            a.telemetry().map(f64::to_bits),
            b.telemetry().map(f64::to_bits)
        );
    }
}

fn gusty_with_faults() -> Params {
    let mut p = Params::realistic();
    p.wind.gust_rate = 0.5;
    p.faults.probability = 1.0;
    p.faults.trigger_time = Range::new(1.0, 3.0);
    p
}

#[test]
fn wind_noise_and_faults_do_not_depend_on_the_controller() {
    let p = gusty_with_faults();
    let mut a = Sim::new(p.clone(), 5).unwrap();
    let mut b = Sim::new(p, 5).unwrap();
    let ca = scripted_controls(10, 2_000);
    let cb = scripted_controls(20, 2_000);
    let mut compared = 0;
    for (x, y) in ca.iter().zip(&cb) {
        let (da, db) = (a.step(*x).is_some(), b.step(*y).is_some());
        assert_eq!(
            a.wind().to_bits(),
            b.wind().to_bits(),
            "wind at tick {}",
            a.tick()
        );
        assert_eq!(
            a.last_sensor_noise().map(f64::to_bits),
            b.last_sensor_noise().map(f64::to_bits)
        );
        assert_eq!(a.thrust_factor(), b.thrust_factor());
        assert_eq!(a.gimbal_stuck(), b.gimbal_stuck());
        compared += 1;
        if da || db {
            break;
        }
    }
    assert!(compared > 200, "compared only {compared} ticks");
    assert!(
        a.thrust_factor() < 1.0 || a.gimbal_stuck().is_some(),
        "the fault happened"
    );
}

#[test]
fn random_controls_never_produce_nan() {
    for config in VehicleConfig::ALL {
        for seed in 0..40 {
            let mut sim = Sim::from_config(config, seed);
            let controls = scripted_controls(seed + 1_000, 8_000);
            let mut ended = false;
            for c in controls {
                let done = sim.step(c).is_some();
                assert!(
                    sim.telemetry().iter().all(|v| v.is_finite()),
                    "{config:?} seed {seed} tick {}",
                    sim.tick()
                );
                if done {
                    ended = true;
                    break;
                }
            }
            assert!(
                ended,
                "{config:?} seed {seed} did not end within the time limit"
            );
            assert!(sim.tick() <= sim.max_ticks());
            assert_ne!(sim.telemetry()[tm::OUTCOME], 0.0);
        }
    }
}
