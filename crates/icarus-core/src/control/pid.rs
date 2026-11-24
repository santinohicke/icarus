//! Cascaded PID: the simple classical baseline.
//!
//! Loops, from the outside in:
//!
//! 1. **Vertical.** A reference descent speed that shrinks with altitude,
//!    `v_ref(h) = −√(v_td² + 2·a_d·h)`: a constant deceleration `a_d` that
//!    arrives at the touchdown speed `v_td` exactly at the ground. Throttle is
//!    the feed-forward deceleration plus a PI correction on the speed error,
//!    divided by `cos θ` so tilting does not cost altitude control.
//!    Vehicles that cannot hover follow a braking profile whose deceleration
//!    blends down towards the ground, and shut the engine down once stopped
//!    in the last centimeters (the short drop stays under the limit).
//! 2. **Ignition trigger** (vehicles whose engine starts off). The engine is
//!    lit when the stage, projected forward by the ignition lag, would be
//!    falling faster than the braking profile allows. Once lit it stays on
//!    until the touchdown cut-off: with a single ignition an early cut-off
//!    would end the run.
//! 3. **Horizontal.** A PD on the pad offset, scheduled by the time to go,
//!    gives a desired horizontal acceleration, turned into a desired tilt
//!    (thrust leans the stage), limited more and more near the ground so the
//!    stage lands upright.
//! 4. **Attitude.** A PID on the tilt error gives a torque, produced by the
//!    gimbal (through the current lever arm and thrust) and by the RCS. The
//!    integral term cancels steady torques such as a slightly stuck gimbal;
//!    it only integrates while the engine runs and the gimbal is not
//!    saturated (anti-windup).
//!
//! Estimation: the sensor delay is known to the flight computer, so the
//! measurements are propagated forward through the accelerations it expected
//! over the delay; a complementary filter smooths the positions; and a
//! disturbance observer learns what the model misses (drag, wind, a weaker
//! engine) and cancels it.

use crate::control::{Controller, Observation};
use crate::physics::dynamics::mass_properties;
use crate::physics::params::Params;
use crate::physics::scenario::wrap_angle;
use crate::physics::sensors::MAX_DELAY_TICKS;
use crate::physics::state::{Controls, QUANT_MAX, dequantize, quantize};

const DEG: f64 = core::f64::consts::PI / 180.0;

/// Tuning of the cascaded PID. The defaults are used for every level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PidGains {
    /// Target speed of the legs at touchdown, as a fraction of the limit
    /// (hover-capable vehicles).
    pub touchdown_speed_fraction: f64,
    /// The same for vehicles that cannot hover: they cannot slow down and
    /// wait, so they aim a little faster to never stop above the ground.
    pub hoverslam_touchdown_fraction: f64,
    /// Design deceleration as a fraction of the net deceleration at full thrust.
    pub decel_fraction: f64,
    /// Height of the final constant-speed segment for hover-capable vehicles (m).
    pub flare_height: f64,
    /// Complementary-filter gain blending measured positions into the
    /// velocity-propagated estimate (per tick).
    pub position_filter: f64,
    /// Hoverslam: the design deceleration blends linearly from its full
    /// value at this height down to the final value at the ground (m).
    pub final_height: f64,
    /// Hoverslam: deceleration at the ground, as a multiple of the smallest
    /// deceleration the engine can give (minimum thrust − weight).
    pub final_decel_factor: f64,
    /// Hoverslam: shut the engine down when stopped this close to the ground
    /// (m); the short fall stays under the touchdown limit.
    pub cutoff_height: f64,
    /// Hover-capable: below this height, wait (descend slowly) until aligned
    /// with the pad (m).
    pub align_height: f64,
    /// Hover-capable: descent speed while waiting to be aligned (m/s).
    pub align_speed: f64,
    /// Proportional gain on the vertical speed error (1/s).
    pub kp_vy: f64,
    /// Integral gain on the vertical speed error (1/s²).
    pub ki_vy: f64,
    /// Ignition lead (s): the trigger looks this far ahead (thrust build-up).
    pub ignition_lag: f64,
    /// Extra altitude margin of the ignition trigger (m).
    pub ignition_margin: f64,
    /// Proportional gain on the horizontal offset, as `kp_x / t_go²` (t_go:
    /// estimated time to touchdown), so the offset is closed by touchdown.
    pub kp_x: f64,
    /// Derivative gain on the horizontal speed, as `kd_x / t_go`.
    pub kd_x: f64,
    /// Below this height only the horizontal speed is controlled: the last
    /// meters are for arriving still, not for chasing the pad center (m).
    pub position_hold_height: f64,
    /// Below this height the stage stays nearly upright and still: a
    /// rotating stage hits with its legs much faster than its center moves (m).
    pub upright_height: f64,
    /// Largest commanded tilt below `upright_height` (rad): enough to lean
    /// into a steady wind, far below the touchdown limit.
    pub upright_tilt: f64,
    /// Gentle horizontal-speed damping below `upright_height` (1/s): enough
    /// to bleed off drift without rocking the stage.
    pub upright_kd: f64,
    /// Smallest time to go used in the scheduling (s): caps the gains.
    pub min_time_to_go: f64,
    /// Largest time to go used in the scheduling (s): floors the gains high up.
    pub max_time_to_go: f64,
    /// Largest commanded tilt, high up (rad).
    pub max_tilt: f64,
    /// Altitude above which the full tilt is allowed (m).
    pub tilt_altitude: f64,
    /// Smallest tilt allowance near the ground (rad).
    pub min_tilt: f64,
    /// Attitude natural frequency (rad/s).
    pub attitude_omega: f64,
    /// Attitude damping ratio.
    pub attitude_zeta: f64,
    /// Attitude integral gain (relative to the proportional gain, 1/s).
    pub attitude_ki: f64,
    /// Disturbance-observer gain per tick (0 disables it). The observer
    /// compares the measured velocity change with the acceleration this
    /// controller expected, and learns the difference: wind drag, air drag,
    /// a weaker engine.
    pub observer_gain: f64,
}

impl Default for PidGains {
    fn default() -> Self {
        Self {
            touchdown_speed_fraction: 0.8,
            hoverslam_touchdown_fraction: 0.641,
            decel_fraction: 0.403,
            flare_height: 7.15,
            position_filter: 0.04,
            final_height: 102.0,
            final_decel_factor: 1.24,
            cutoff_height: 0.15,
            align_height: 20.6,
            align_speed: 1.5,
            kp_vy: 1.38,
            ki_vy: 0.273,
            ignition_lag: 1.13,
            ignition_margin: 3.24,
            kp_x: 5.8,
            kd_x: 3.81,
            position_hold_height: 7.16,
            upright_height: 1.12,
            upright_tilt: 1.09 * DEG,
            upright_kd: 0.102,
            min_time_to_go: 7.58,
            max_time_to_go: 25.0,
            max_tilt: 12.2 * DEG,
            tilt_altitude: 39.5,
            min_tilt: 5.9 * DEG,
            attitude_omega: 1.52,
            attitude_zeta: 0.9,
            attitude_ki: 0.0738,
            observer_gain: 0.0125,
        }
    }
}

/// Integrator bounds (anti-windup).
const VY_INTEGRAL_MAX: f64 = 5.0;
const VY_INTEGRATE_BAND: f64 = 3.0;
const THETA_INTEGRAL_MAX: f64 = 0.15;
/// Bounds of the estimated disturbance accelerations `(x, y)` (m/s²).
const DISTURBANCE_MAX: (f64, f64) = (4.0, 15.0);
const HISTORY_LEN: usize = MAX_DELAY_TICKS + 2;

/// The cascaded PID controller.
#[derive(Clone, Debug, PartialEq)]
pub struct Pid {
    params: Params,
    gains: PidGains,
    vy_integral: f64,
    theta_integral: f64,
    /// Latched once the trigger fires: the engine stays commanded on.
    ignited: bool,
    /// Latched by the touchdown cut-off: the engine stays commanded off.
    cut_off: bool,
    /// Filtered position estimate `(x, y)`, once initialized.
    position: Option<(f64, f64)>,
    /// Expected accelerations of past ticks (ring buffer), so the observer
    /// can compare delayed measurements with what was expected back then.
    expected: [(f64, f64); HISTORY_LEN],
    head: usize,
    /// Previous raw velocity measurement.
    prev_velocity: Option<(f64, f64)>,
    /// Estimated disturbance accelerations `(x, y)`.
    disturbance: (f64, f64),
    /// Model of the actual thrust fraction (first-order lag of the command).
    thrust_model: f64,
}

/// `a·b + c` without fused multiply-add (bit-exact everywhere).
#[inline]
fn lin(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

impl Pid {
    /// A PID with the default gains for these parameters.
    pub fn new(params: &Params) -> Self {
        Self::with_gains(params, PidGains::default())
    }

    /// A PID with custom gains.
    pub fn with_gains(params: &Params, gains: PidGains) -> Self {
        Self {
            params: params.clone(),
            gains,
            vy_integral: 0.0,
            theta_integral: 0.0,
            ignited: params.start.engine_on,
            cut_off: false,
            position: None,
            expected: [(0.0, 0.0); HISTORY_LEN],
            head: 0,
            prev_velocity: None,
            disturbance: (0.0, 0.0),
            thrust_model: if params.start.engine_on {
                params.start.initial_throttle
            } else {
                0.0
            },
        }
    }

    /// The gains in use.
    pub fn gains(&self) -> &PidGains {
        &self.gains
    }

    /// Normalized throttle command for a thrust fraction (at or above minimum).
    fn throttle_for(&self, fraction: f64) -> f64 {
        let v = &self.params.vehicle;
        if v.min_throttle >= 1.0 {
            return 1.0;
        }
        let t = ((fraction - v.min_throttle) / (1.0 - v.min_throttle)).clamp(0.0, 1.0);
        let u = lin(t, 1.0 - v.throttle_cutoff, v.throttle_cutoff);
        // The command travels quantized: make sure the level it rounds to is
        // still at or above the cut-off (else minimum throttle would shut the
        // engine down on vehicles whose cut-off rounds the wrong way).
        let mut q = quantize(u);
        while q < QUANT_MAX && dequantize(q) < v.throttle_cutoff {
            q += 1;
        }
        dequantize(q)
    }
}

impl Controller for Pid {
    fn name(&self) -> &'static str {
        "pid"
    }

    fn reset(&mut self) {
        // Exactly the constructor's state (checked by a test).
        *self = Self::with_gains(&self.params, self.gains);
    }

    fn act(&mut self, obs: &Observation) -> Controls {
        let p = &self.params;
        let (v, g, k) = (&p.vehicle, p.environment.gravity, self.gains);
        let dt = p.dt;
        let mass = obs.mass.max(v.dry_mass);
        let mp = mass_properties(v, mass);

        // Disturbance observer, on the raw (delayed) measurements: the change
        // since the previous measurement happened `delay + 1` ticks ago.
        let delay = p.sensors.delay_ticks.min(MAX_DELAY_TICKS);
        if let Some((pvx, pvy)) = self.prev_velocity {
            let (ex, ey) = self.expected[(self.head + HISTORY_LEN - delay) % HISTORY_LEN];
            let (dx, dy) = self.disturbance;
            let kobs = k.observer_gain / dt;
            let ix = obs.vx - lin(ex + dx, dt, pvx);
            let iy = obs.vy - lin(ey + dy, dt, pvy);
            self.disturbance = (
                lin(kobs, ix, dx).clamp(-DISTURBANCE_MAX.0, DISTURBANCE_MAX.0),
                lin(kobs, iy, dy).clamp(-DISTURBANCE_MAX.1, DISTURBANCE_MAX.1),
            );
        }
        self.prev_velocity = Some((obs.vx, obs.vy));
        let (dist_x, dist_y) = self.disturbance;

        // Propagate the delayed measurement to "now" through the accelerations
        // expected over the delay (plus the observed disturbance), oldest first.
        let (mut x_meas, mut y_meas, mut vx, mut vy) = (obs.x, obs.y, obs.vx, obs.vy);
        for i in (0..delay).rev() {
            let (ax, ay) = self.expected[(self.head + HISTORY_LEN - i) % HISTORY_LEN];
            let (ax, ay) = (ax + dist_x, ay + dist_y);
            x_meas = lin(lin(0.5 * ax, dt, vx), dt, x_meas);
            y_meas = lin(lin(0.5 * ay, dt, vy), dt, y_meas);
            vx = lin(ax, dt, vx);
            vy = lin(ay, dt, vy);
        }
        let lag = delay as f64 * dt;
        // Complementary filter: positions are noisy (σ ≈ 0.5 m), velocities
        // much less, so integrate the velocities and pull gently towards the
        // measured positions.
        let (x, y) = match self.position {
            None => (x_meas, y_meas),
            Some((px, py)) => {
                let (ex, ey) = (lin(vx, dt, px), lin(vy, dt, py));
                let kf = k.position_filter;
                (lin(kf, x_meas - ex, ex), lin(kf, y_meas - ey, ey))
            }
        };
        self.position = Some((x, y));
        let theta = wrap_angle(lin(obs.omega, lag, obs.theta));
        let omega = obs.omega;
        let cos_t = libm::cos(theta);
        let cos_safe = cos_t.max(0.5);

        // Height of the lower leg tip above the ground: the tips sit at tail
        // level, `leg_half_span` to either side, so a tilt lowers one of them.
        let h =
            (y - mp.cm_height * cos_t - v.leg_half_span * libm::fabs(libm::sin(theta))).max(0.0);
        let a_max = v.max_thrust / mass;
        let a_min = v.min_throttle * a_max;

        // --- Vertical loop -------------------------------------------------
        let can_hover = a_min < g;
        let v_td = if can_hover {
            k.touchdown_speed_fraction
        } else {
            k.hoverslam_touchdown_fraction
        } * p.landing.max_vertical_speed;
        // Design deceleration: a share of what full thrust gives, but always
        // above what minimum thrust gives (else the stage would climb back up).
        let a_d = (k.decel_fraction * (a_max - g))
            .max(1.3 * (a_min - g))
            .max(0.5);
        // Hover-capable vehicles finish with a short constant-speed segment;
        // the others must arrive exactly as the profile reaches the ground.
        // Hoverslam: the deceleration blends from `a_d` (above
        // `final_height`) down to `a_f` at the ground. Gentler braking near
        // the ground makes the arrival speed less sensitive to height errors,
        // and staying above the minimum-thrust deceleration keeps the stage
        // from stopping in mid-air. v² = v_td² + 2·∫₀ʰ a(s) ds.
        let a_f = (k.final_decel_factor * (a_min - g)).clamp(0.5, a_d);
        let big_h = k.final_height.max(1.0);
        // Reference speed (negative) and its deceleration at height `hh`.
        let profile = |hh: f64| -> (f64, f64) {
            if can_hover {
                let v = -libm::sqrt(lin(2.0 * a_d, (hh - k.flare_height).max(0.0), v_td * v_td));
                (v, a_d)
            } else if hh <= big_h {
                let integral = a_f * hh + (a_d - a_f) * hh * hh / (2.0 * big_h);
                (
                    -libm::sqrt(lin(2.0, integral, v_td * v_td)),
                    a_f + (a_d - a_f) * hh / big_h,
                )
            } else {
                let integral = a_f * big_h + 0.5 * (a_d - a_f) * big_h + a_d * (hh - big_h);
                (-libm::sqrt(lin(2.0, integral, v_td * v_td)), a_d)
            }
        };
        let (mut v_ref, a_here) = profile(h);
        // Hover-capable stages wait above the pad until they are aligned.
        let pad = p.landing.pad_half_width;
        let misaligned =
            libm::fabs(x) > 0.5 * pad || libm::fabs(vx) > 0.8 * p.landing.max_horizontal_speed;
        if can_hover && h < k.align_height && h > k.upright_height && misaligned {
            v_ref = v_ref.max(-k.align_speed);
        }
        let a_d = a_here;
        let a_ff = (a_d * vy / v_ref).clamp(0.0, 2.0 * a_d);
        let err_vy = v_ref - vy;
        let engine_running = obs.engine_on;
        // Integrate only near the profile: large transients (the ignition
        // catching up with the profile) must not wind the integrator up.
        if engine_running && err_vy.abs() < VY_INTEGRATE_BAND {
            self.vy_integral =
                lin(err_vy, dt, self.vy_integral).clamp(-VY_INTEGRAL_MAX, VY_INTEGRAL_MAX);
        }
        // The observed disturbance (drag, a weaker engine) is cancelled.
        let a_cmd = g + a_ff + k.kp_vy * err_vy + k.ki_vy * self.vy_integral - dist_y;
        let fraction = (mass * a_cmd / (v.max_thrust * cos_safe)).clamp(v.min_throttle, 1.0);

        // --- Ignition trigger -----------------------------------------------
        if engine_running {
            self.ignited = true;
        } else if !self.ignited && obs.ignitions_left > 0 && vy < 0.0 {
            // Where the stage will be once the thrust has built up: light now
            // if it would be falling faster than the profile allows there.
            let lead = (-vy) * k.ignition_lag + k.ignition_margin;
            let (v_lead, _) = profile((h - lead).max(0.0));
            if vy <= v_lead {
                self.ignited = true;
            }
        }
        // Hoverslam touchdown: stopped just above the ground with an engine
        // that cannot hover: shut it down (for good) and drop the last centimeters.
        if !can_hover && engine_running && h < k.cutoff_height && vy > -0.5 {
            self.cut_off = true;
        }
        let throttle = if self.cut_off {
            -1.0
        } else if self.ignited && (engine_running || obs.ignitions_left > 0) {
            self.throttle_for(fraction)
        } else {
            -1.0 // engine off, waiting for the trigger
        };

        // --- Horizontal loop -> desired tilt --------------------------------
        let thrust_accel = if engine_running {
            (fraction * a_max).max(0.5 * g)
        } else {
            0.0
        };
        let mut theta_ref = 0.0;
        if thrust_accel > 0.0 {
            // Time to touchdown if the vertical loop follows its profile.
            let t_go =
                (2.0 * h / ((-vy).max(0.0) + v_td)).clamp(k.min_time_to_go, k.max_time_to_go);
            let position_term = if h > k.position_hold_height {
                k.kp_x * x / (t_go * t_go)
            } else {
                0.0
            };
            let a_x = if h <= k.upright_height {
                -k.upright_kd * vx - dist_x
            } else {
                -(position_term + k.kd_x * vx / t_go) - dist_x
            };
            // Horizontal acceleration of a tilted thrust: −(T/m)·sin θ.
            let s = (-a_x / thrust_accel).clamp(-1.0, 1.0);
            let allowed = if h <= k.upright_height {
                k.upright_tilt
            } else {
                (k.max_tilt * h / k.tilt_altitude).clamp(k.min_tilt, k.max_tilt)
            };
            theta_ref = libm::asin(s).clamp(-allowed, allowed);
        }

        // --- Attitude loop -> gimbal and RCS --------------------------------
        let err_theta = wrap_angle(theta_ref - theta);
        let (wn, zeta) = (k.attitude_omega, k.attitude_zeta);
        let alpha = wn * wn * err_theta - 2.0 * zeta * wn * omega
            + k.attitude_ki * wn * wn * self.theta_integral;
        let torque = mp.inertia * alpha;

        let thrust_now = if engine_running {
            fraction * v.max_thrust
        } else {
            0.0
        };
        let lever = mp.cm_height * thrust_now;
        let gimbal = if lever > 0.0 {
            // Torque of a deflected engine: l_g·T·sin δ.
            let s = (torque / lever).clamp(-1.0, 1.0);
            libm::asin(s) / v.gimbal_max
        } else {
            0.0
        };
        // Anti-windup: integrate only while the engine runs and the gimbal
        // has room left (the integral would otherwise ramp up during the
        // coast and kick the stage over at ignition).
        if engine_running && gimbal.abs() < 1.0 {
            self.theta_integral = lin(err_theta, dt, self.theta_integral)
                .clamp(-THETA_INTEGRAL_MAX, THETA_INTEGRAL_MAX);
        }
        let rcs = if v.rcs_torque > 0.0 {
            torque / v.rcs_torque
        } else {
            0.0
        };

        // Expected acceleration over this tick (thrust through its lag model,
        // without the disturbance), for the delay compensation and observer.
        let commanded = if engine_running { fraction } else { 0.0 };
        let tau = v.thrust_time_constant.max(dt);
        self.thrust_model = lin(commanded - self.thrust_model, dt / tau, self.thrust_model);
        let a = self.thrust_model * a_max;
        self.head = (self.head + 1) % HISTORY_LEN;
        self.expected[self.head] = (-a * libm::sin(theta), a * cos_t - g);

        Controls::new(throttle, gimbal.clamp(-1.0, 1.0), rcs.clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::params::VehicleConfig;
    use crate::sim::Sim;

    fn fly(config: VehicleConfig, seed: u64) -> (Option<crate::Outcome>, Sim) {
        let mut sim = Sim::from_config(config, seed);
        let mut pid = Pid::new(sim.params());
        while sim.outcome().is_none() {
            let q = pid.act_quantized(&Observation::from_sim(&sim));
            sim.step(Controls::from_quantized(q));
        }
        (sim.outcome(), sim)
    }

    #[test]
    fn lands_the_easy_vehicle() {
        let landed = (0..20)
            .filter(|&s| fly(VehicleConfig::Easy, s).0 == Some(crate::Outcome::Landed))
            .count();
        assert!(landed >= 19, "landed {landed}/20");
    }

    #[test]
    fn decisions_are_deterministic() {
        let (a, sa) = fly(VehicleConfig::Realistic, 7);
        let (b, sb) = fly(VehicleConfig::Realistic, 7);
        assert_eq!(a, b);
        assert_eq!(sa.telemetry(), sb.telemetry());
    }

    #[test]
    fn waits_for_the_trigger_with_the_engine_off() {
        let sim = Sim::from_config(VehicleConfig::Realistic, 3);
        let mut pid = Pid::new(sim.params());
        let c = pid.act(&Observation::from_sim(&sim));
        assert!(
            c.throttle < sim.params().vehicle.throttle_cutoff,
            "lit far too early"
        );
    }

    #[test]
    fn reset_restores_exactly_the_constructor_state() {
        let mut sim = Sim::from_config(VehicleConfig::Storm, 3);
        let fresh = Pid::new(sim.params());
        let mut pid = fresh.clone();
        for _ in 0..900 {
            let q = pid.act_quantized(&Observation::from_sim(&sim));
            sim.step(Controls::from_quantized(q));
        }
        assert_ne!(pid, fresh, "flying must change the internal state");
        pid.reset();
        assert_eq!(pid, fresh);
    }

    #[test]
    fn decisions_depend_only_on_the_observation() {
        let sim = Sim::from_config(VehicleConfig::Realistic, 4);
        let obs = Observation::from_sim(&sim);
        let mut a = Pid::new(sim.params());
        let mut b = Pid::new(sim.params());
        assert_eq!(a.act_quantized(&obs), b.act_quantized(&obs));
    }

    #[test]
    fn touchdown_cutoff_is_latched_even_with_ignitions_left() {
        // A hoverslam vehicle with spare ignitions: once cut off near the
        // ground, the PID must not relight.
        let mut params = VehicleConfig::Realistic.params();
        params.start.ignitions = 3;
        let mut pid = Pid::new(&params);
        let mass = params.vehicle.dry_mass + 2_000.0;
        let cm = mass_properties(&params.vehicle, mass).cm_height;
        let mut obs = Observation {
            x: 0.0,
            y: cm + 0.05,
            vx: 0.0,
            vy: -0.2,
            theta: 0.0,
            omega: 0.0,
            mass,
            engine_on: true,
            ignitions_left: 2,
        };
        let cutoff = params.vehicle.throttle_cutoff;
        assert!(pid.act(&obs).throttle < cutoff, "stopped 5 cm up: cut off");
        obs.engine_on = false;
        obs.vy = -0.5;
        for _ in 0..10 {
            assert!(
                pid.act(&obs).throttle < cutoff,
                "must stay off after the cut-off"
            );
        }
    }

    #[test]
    fn minimum_throttle_survives_quantization() {
        // A cut-off that sits between two quantized levels: the commanded
        // minimum must round to a level that keeps the engine running.
        let mut params = VehicleConfig::Realistic.params();
        params.vehicle.throttle_cutoff = -0.8999;
        let pid = Pid::new(&params);
        let u = pid.throttle_for(params.vehicle.min_throttle);
        assert!(dequantize(quantize(u)) >= params.vehicle.throttle_cutoff);
    }
}
