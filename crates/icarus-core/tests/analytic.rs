//! The simulator matches closed-form solutions.

use icarus_core::physics::params::{G0, Range};
use icarus_core::{Controls, Params, Sim};

/// Vacuum, calm air, engine off: only gravity acts.
fn ballistic() -> Params {
    let mut p = Params::easy();
    p.environment.rho0 = 0.0;
    p.wind.mean = Range::fixed(0.0);
    p.wind.turbulence_sigma = 0.0;
    p.start.engine_on = false;
    p.start.ignitions = 0;
    p
}

fn close(actual: f64, expected: f64, rel: f64) -> bool {
    (actual - expected).abs() <= rel * expected.abs().max(1.0)
}

#[test]
fn free_fall_is_a_parabola() {
    let p = ballistic();
    let g = p.environment.gravity;
    let mut sim = Sim::new(p, 11).unwrap();
    let s0 = *sim.state();
    let ticks = 180;
    for _ in 0..ticks {
        sim.step(Controls::new(-1.0, 0.0, 0.0));
    }
    let t = sim.time();
    let s = sim.state();
    assert!(close(s.y, s0.y + s0.vy * t - 0.5 * g * t * t, 1e-12), "y");
    assert!(close(s.vy, s0.vy - g * t, 1e-12), "vy");
    assert!(close(s.x, s0.x + s0.vx * t, 1e-12), "x");
    assert_eq!(s.mass, s0.mass, "no propellant used");
}

#[test]
fn torque_free_rotation_is_uniform() {
    let mut p = ballistic();
    p.start.angular_rate = Range::fixed(0.1);
    p.start.tilt = Range::fixed(0.02);
    let mut sim = Sim::new(p, 2).unwrap();
    for _ in 0..240 {
        sim.step(Controls::new(-1.0, 0.0, 0.0));
    }
    let t = sim.time();
    assert!(close(sim.state().theta, 0.02 + 0.1 * t, 1e-12));
    assert!(close(sim.state().omega, 0.1, 1e-12));
}

#[test]
fn thrust_lag_and_propellant_use_match_closed_form() {
    let mut p = ballistic();
    p.start.ignitions = 1;
    let (tau, t_max, isp) = (
        p.vehicle.thrust_time_constant,
        p.vehicle.max_thrust,
        p.vehicle.isp,
    );
    let mut sim = Sim::new(p, 4).unwrap();
    let m0 = sim.state().mass;
    for _ in 0..120 {
        sim.step(Controls::new(1.0, 0.0, 0.0)); // ignite at full throttle
    }
    let t = sim.time();
    let decay = libm::exp(-t / tau);
    let thrust = t_max * (1.0 - decay);
    let mass = m0 - t_max / (isp * G0) * (t - tau * (1.0 - decay));
    assert!(
        close(sim.state().thrust, thrust, 1e-7),
        "thrust {} vs {thrust}",
        sim.state().thrust
    );
    assert!(
        close(sim.state().mass, mass, 1e-9),
        "mass {} vs {mass}",
        sim.state().mass
    );
}

#[test]
fn gimbal_follows_a_first_order_lag_when_not_rate_limited() {
    let mut p = ballistic();
    p.vehicle.gimbal_rate_max = 1.0e3;
    let (tau, max) = (p.vehicle.gimbal_time_constant, p.vehicle.gimbal_max);
    let mut sim = Sim::new(p, 5).unwrap();
    for _ in 0..30 {
        sim.step(Controls::new(-1.0, 0.5, 0.0));
    }
    let expected = 0.5 * max * (1.0 - libm::exp(-sim.time() / tau));
    assert!(close(sim.state().gimbal, expected, 1e-7));
}

#[test]
fn gimbal_slews_at_the_rate_limit() {
    let p = ballistic();
    let rate = p.vehicle.gimbal_rate_max;
    let mut sim = Sim::new(p, 5).unwrap();
    for _ in 0..6 {
        sim.step(Controls::new(-1.0, 1.0, 0.0));
    }
    // Far from the target the rate limit is active: δ = rate · t.
    assert!(close(sim.state().gimbal, rate * sim.time(), 1e-12));
}

#[test]
fn touchdown_speed_and_time_match_free_fall() {
    // Upright vacuum drop: the leg tips hit the ground when the CoM has fallen
    // by (y0 - cm_height); speed and time follow the closed form.
    let mut p = ballistic();
    p.start.tilt = Range::fixed(0.0);
    p.start.angular_rate = Range::fixed(0.0);
    p.start.horizontal_speed = Range::fixed(0.0);
    p.start.horizontal_offset = Range::fixed(0.0);
    let g = p.environment.gravity;
    let mut sim = Sim::new(p, 3).unwrap();
    let s0 = *sim.state();
    let drop = s0.y - sim.geometry().cm_height;
    while sim.step(Controls::new(-1.0, 0.0, 0.0)).is_none() {}
    let speed = (s0.vy * s0.vy + 2.0 * g * drop).sqrt();
    let time = (s0.vy + speed) / g;
    let contact = sim.contact().expect("ended on the ground");
    assert!(
        contact.clearance <= 0.0 && contact.clearance > -1e-9,
        "clearance {}",
        contact.clearance
    );
    assert!(
        close(sim.state().vy, -speed, 1e-9),
        "vy {} vs {}",
        sim.state().vy,
        -speed
    );
    assert!(
        close(sim.time(), time, 1e-9),
        "time {} vs {time}",
        sim.time()
    );
}
