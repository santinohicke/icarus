//! The full simulator converges at fourth order on smooth dynamics.
//!
//! Non-smooth features (gimbal slew limit, throttle cut-off, wind, ground
//! contact) are disabled: they would lower the observed order.

use icarus_core::physics::params::Range;
use icarus_core::{Controls, Params, Sim};

fn smooth_params(dt: f64) -> Params {
    let mut p = Params::easy();
    p.dt = dt;
    p.vehicle.gimbal_rate_max = 1.0e3; // never saturates
    p.wind.mean = Range::fixed(0.0);
    p.wind.turbulence_sigma = 0.0;
    p.wind.gust_rate = 0.0;
    p.start.tilt = Range::fixed(0.05);
    p.start.angular_rate = Range::fixed(0.02);
    p
}

/// Final `[x, y, vx, vy, θ, ω, m]` after 4 s.
fn final_state(dt: f64) -> [f64; 7] {
    let mut sim = Sim::new(smooth_params(dt), 7).expect("valid parameters");
    let ticks = libm::round(4.0 / dt) as u64;
    // Constant commands: thrust and gimbal lags still evolve smoothly.
    let controls = Controls::new(0.3, 0.5, 0.2);
    for _ in 0..ticks {
        assert!(
            sim.step(controls).is_none(),
            "the episode must not end during the test"
        );
    }
    let s = sim.state();
    [s.x, s.y, s.vx, s.vy, s.theta, s.omega, s.mass]
}

fn error(a: &[f64; 7], b: &[f64; 7]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

#[test]
fn rk4_converges_at_fourth_order() {
    let reference = final_state(1.0 / 1920.0);
    let e1 = error(&final_state(1.0 / 30.0), &reference);
    let e2 = error(&final_state(1.0 / 60.0), &reference);
    let e3 = error(&final_state(1.0 / 120.0), &reference);
    let (r1, r2) = (e1 / e2, e2 / e3);
    assert!(
        (12.0..20.0).contains(&r1),
        "error ratio 1/30 → 1/60: {r1} ({e1:e} / {e2:e})"
    );
    assert!(
        (12.0..20.0).contains(&r2),
        "error ratio 1/60 → 1/120: {r2} ({e2:e} / {e3:e})"
    );
}
