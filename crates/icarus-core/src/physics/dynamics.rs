//! Forces, torques and the state derivative.
//!
//! Sign conventions (see the crate docs): body axis `b = (−sin θ, cos θ)`;
//! thrust direction `(−sin(θ − δ), cos(θ − δ))`; with the engine a distance
//! `l_g` below the center of mass, the thrust torque is `l_g · T · sin δ`
//! (counter-clockwise for positive `δ`).

use crate::physics::params::{Environment, G0, Vehicle};
use crate::physics::state::STATE_DIM;

/// Inputs held constant during one integration step.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct HeldInputs {
    /// Commanded thrust (N).
    pub thrust_cmd: f64,
    /// Commanded gimbal angle (rad).
    pub gimbal_cmd: f64,
    /// RCS torque (N·m).
    pub rcs_torque: f64,
    /// Horizontal wind speed (m/s), sampled once per tick.
    pub wind: f64,
}

/// Mass-dependent properties of the vehicle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    /// Propellant mass (kg), never negative.
    pub propellant: f64,
    /// Height of the center of mass above the tail (m). Also the lever arm of
    /// the engine.
    pub cm_height: f64,
    /// Pitch moment of inertia about the center of mass (kg·m²).
    pub inertia: f64,
}

/// Center of mass and moment of inertia at a given total mass, using the
/// parallel-axis theorem over the dry structure and the propellant.
pub fn mass_properties(v: &Vehicle, mass: f64) -> MassProperties {
    let propellant = (mass - v.dry_mass).max(0.0);
    let total = v.dry_mass + propellant;
    let cm_height = (v.dry_mass * v.dry_cm_height + propellant * v.propellant_cm_height) / total;
    let d_dry = v.dry_cm_height - cm_height;
    let d_prop = v.propellant_cm_height - cm_height;
    let r = v.propellant_gyration_radius;
    let inertia =
        v.dry_inertia + v.dry_mass * d_dry * d_dry + propellant * (r * r + d_prop * d_prop);
    MassProperties {
        propellant,
        cm_height,
        inertia,
    }
}

/// Air density at altitude `y` (exponential atmosphere, clamped at sea level).
pub fn density(env: &Environment, y: f64) -> f64 {
    env.rho0 * libm::exp(-y.max(0.0) / env.scale_height)
}

/// Time derivative of the state vector `[x, y, vx, vy, θ, ω, m, δ, T]`.
pub fn derivatives(
    v: &Vehicle,
    env: &Environment,
    s: &[f64; STATE_DIM],
    u: &HeldInputs,
) -> [f64; STATE_DIM] {
    let [_x, y, vx, vy, theta, omega, mass, gimbal, thrust] = *s;
    let mp = mass_properties(v, mass);

    let sin_t = libm::sin(theta);
    let cos_t = libm::cos(theta);
    let (bx, by) = (-sin_t, cos_t);

    // Engine.
    let phi = theta - gimbal;
    let fx_thrust = -thrust * libm::sin(phi);
    let fy_thrust = thrust * libm::cos(phi);
    let torque_thrust = mp.cm_height * thrust * libm::sin(gimbal);

    // Aerodynamics: drag opposite to the air-relative velocity, applied at the
    // center of pressure, plus pitch damping.
    let rho = density(env, y);
    let vrx = vx - u.wind;
    let vry = vy;
    let speed = (vrx * vrx + vry * vry).sqrt();
    let (fx_drag, fy_drag, torque_aero) = if speed > 0.0 && rho > 0.0 {
        let cos_a = (vrx * bx + vry * by) / speed;
        let cos2 = (cos_a * cos_a).min(1.0);
        let sin2 = 1.0 - cos2;
        let cda = v.cda_axial * cos2 + v.cda_side * sin2;
        let k = 0.5 * rho * cda * speed;
        let fx = -k * vrx;
        let fy = -k * vry;
        let arm = v.cp_height - mp.cm_height;
        let (rx, ry) = (arm * bx, arm * by);
        let torque = (rx * fy - ry * fx) - v.pitch_damping * rho * speed * omega;
        (fx, fy, torque)
    } else {
        (0.0, 0.0, 0.0)
    };

    let ax = (fx_thrust + fx_drag) / mass;
    let ay = (fy_thrust + fy_drag) / mass - env.gravity;
    let alpha = (torque_thrust + torque_aero + u.rcs_torque) / mp.inertia;

    let mass_rate = if mp.propellant > 0.0 {
        -thrust / (v.isp * G0)
    } else {
        0.0
    };
    let gimbal_rate = ((u.gimbal_cmd - gimbal) / v.gimbal_time_constant)
        .clamp(-v.gimbal_rate_max, v.gimbal_rate_max);
    let thrust_rate = (u.thrust_cmd - thrust) / v.thrust_time_constant;

    [
        vx,
        vy,
        ax,
        ay,
        omega,
        alpha,
        mass_rate,
        gimbal_rate,
        thrust_rate,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::params::{DEG, Params};

    fn vacuum() -> Environment {
        Environment {
            gravity: 9.81,
            rho0: 0.0,
            scale_height: 8_500.0,
        }
    }

    #[test]
    fn center_of_mass_moves_toward_the_propellant() {
        let v = Params::easy().vehicle;
        let empty = mass_properties(&v, v.dry_mass);
        let full = mass_properties(&v, v.dry_mass + 6_000.0);
        assert_eq!(empty.cm_height, v.dry_cm_height);
        assert_eq!(empty.inertia, v.dry_inertia);
        assert!(
            full.cm_height < empty.cm_height,
            "propellant sits below the dry CoM"
        );
        assert!(full.inertia > empty.inertia);
        let below_dry = mass_properties(&v, v.dry_mass - 1.0);
        assert_eq!(below_dry.propellant, 0.0);
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-12)
    }

    #[test]
    fn mass_properties_match_hand_computation() {
        let v = Params::easy().vehicle;
        // 25 t dry at 15 m, 5 t propellant at 10 m → CoM at (375 + 50) / 30 = 14.1667 m.
        let mp = mass_properties(&v, 30_000.0);
        let h = (25_000.0 * 15.0 + 5_000.0 * 10.0) / 30_000.0;
        assert!(close(mp.cm_height, h, 1e-12));
        let inertia = v.dry_inertia
            + 25_000.0 * (15.0 - h) * (15.0 - h)
            + 5_000.0 * (3.0 * 3.0 + (10.0 - h) * (10.0 - h));
        assert!(close(mp.inertia, inertia, 1e-12));
    }

    #[test]
    fn thrust_torque_and_force_match_cross_product_for_any_attitude() {
        let p = Params::easy();
        let v = &p.vehicle;
        let mass = v.dry_mass + 1_000.0;
        let mp = mass_properties(v, mass);
        let thrust = 300_000.0;
        for &(theta, delta) in &[(0.0, 2.0 * DEG), (0.3, 0.05), (-1.0, -0.1), (2.0, 0.12)] {
            let s = [0.0, 1_000.0, 0.0, 0.0, theta, 0.0, mass, delta, thrust];
            let u = HeldInputs {
                thrust_cmd: thrust,
                gimbal_cmd: delta,
                ..Default::default()
            };
            let d = derivatives(v, &vacuum(), &s, &u);
            // r × F with r = −l_g·b and F along (−sin(θ−δ), cos(θ−δ)).
            let (bx, by) = (-libm::sin(theta), libm::cos(theta));
            let (rx, ry) = (-mp.cm_height * bx, -mp.cm_height * by);
            let (fx, fy) = (
                -thrust * libm::sin(theta - delta),
                thrust * libm::cos(theta - delta),
            );
            let torque = rx * fy - ry * fx;
            assert!(close(
                torque,
                mp.cm_height * thrust * libm::sin(delta),
                1e-9
            ));
            assert!(
                close(d[5], torque / mp.inertia, 1e-12),
                "angular acceleration"
            );
            assert!(close(d[2], fx / mass, 1e-12), "lateral acceleration");
            assert!(
                close(d[3], fy / mass - 9.81, 1e-12),
                "vertical acceleration"
            );
        }
    }

    #[test]
    fn positive_gimbal_gives_counter_clockwise_torque() {
        let p = Params::easy();
        let mass = p.vehicle.dry_mass + 1_000.0;
        let thrust = 300_000.0;
        let s = [0.0, 1_000.0, 0.0, 0.0, 0.0, 0.0, mass, 2.0 * DEG, thrust];
        let u = HeldInputs {
            thrust_cmd: thrust,
            gimbal_cmd: 2.0 * DEG,
            ..Default::default()
        };
        let d = derivatives(&p.vehicle, &vacuum(), &s, &u);
        assert!(
            d[5] > 0.0,
            "angular acceleration should be counter-clockwise"
        );
        // The tail is pushed to the right: the lateral force is positive.
        assert!(d[2] > 0.0);
    }

    #[test]
    fn leaning_left_under_thrust_accelerates_left() {
        let p = Params::easy();
        let s = [
            0.0,
            1_000.0,
            0.0,
            0.0,
            10.0 * DEG,
            0.0,
            30_000.0,
            0.0,
            300_000.0,
        ];
        let u = HeldInputs {
            thrust_cmd: 300_000.0,
            ..Default::default()
        };
        let d = derivatives(&p.vehicle, &vacuum(), &s, &u);
        assert!(close(
            d[2],
            -300_000.0 * libm::sin(10.0 * DEG) / 30_000.0,
            1e-12
        ));
    }

    #[test]
    fn rcs_gives_a_pure_torque() {
        let p = Params::easy();
        let s = [0.0, 1_000.0, 0.0, 0.0, 0.3, 0.0, 30_000.0, 0.0, 0.0];
        let u = HeldInputs {
            rcs_torque: 50_000.0,
            ..Default::default()
        };
        let d = derivatives(&p.vehicle, &vacuum(), &s, &u);
        let mp = mass_properties(&p.vehicle, 30_000.0);
        assert!(close(d[5], 50_000.0 / mp.inertia, 1e-12));
        assert_eq!(d[2], 0.0);
        assert_eq!(d[3], -9.81);
    }

    #[test]
    fn terminal_velocity_matches_drag_balance() {
        // At terminal velocity (upright, still air, sea level) drag equals weight.
        let p = Params::easy();
        let v = &p.vehicle;
        let mass = v.dry_mass;
        let vt = (2.0 * mass * p.environment.gravity / (p.environment.rho0 * v.cda_axial)).sqrt();
        let s = [0.0, 0.0, 0.0, -vt, 0.0, 0.0, mass, 0.0, 0.0];
        let d = derivatives(v, &p.environment, &s, &HeldInputs::default());
        assert!(d[3].abs() < 1e-9, "net vertical acceleration {}", d[3]);
    }

    #[test]
    fn drag_opposes_motion_and_weathervanes_stably() {
        let p = Params::easy();
        let mass = p.vehicle.dry_mass + 1_000.0;
        // Falling tail-first at 80 m/s, tilted 5° to the left, engine off.
        let s = [0.0, 500.0, 0.0, -80.0, 5.0 * DEG, 0.0, mass, 0.0, 0.0];
        let d = derivatives(&p.vehicle, &p.environment, &s, &HeldInputs::default());
        assert!(d[3] > -p.environment.gravity, "drag decelerates the fall");
        assert!(d[5] < 0.0, "aerodynamic torque restores the tilt");
    }

    #[test]
    fn density_decreases_with_altitude() {
        let env = Params::easy().environment;
        assert_eq!(density(&env, 0.0), env.rho0);
        assert_eq!(density(&env, -10.0), env.rho0);
        assert!(density(&env, 1_000.0) < env.rho0);
    }

    #[test]
    fn gimbal_rate_is_limited() {
        let p = Params::easy();
        let s = [0.0, 1_000.0, 0.0, 0.0, 0.0, 0.0, 30_000.0, 0.0, 0.0];
        let u = HeldInputs {
            gimbal_cmd: 7.0 * DEG,
            ..Default::default()
        };
        let d = derivatives(&p.vehicle, &vacuum(), &s, &u);
        assert_eq!(d[7], p.vehicle.gimbal_rate_max);
    }
}
