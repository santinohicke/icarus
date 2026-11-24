//! Initial conditions, contact geometry, episode outcomes and score.

use crate::physics::dynamics::mass_properties;
use crate::physics::params::{LandingCriteria, Params, StartParams, Vehicle};
use crate::physics::state::State;
use crate::rng::Rng;

/// Initial conditions of an episode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InitialConditions {
    /// Continuous state.
    pub state: State,
    /// Engine running at the start.
    pub engine_on: bool,
    /// Ignitions available.
    pub ignitions: u32,
}

/// Draws the initial conditions. Always draws exactly seven numbers.
pub fn sample_initial(params: &Params, rng: &mut Rng) -> InitialConditions {
    let st: &StartParams = &params.start;
    let v: &Vehicle = &params.vehicle;
    let y = st.altitude.sample(rng);
    let vy = st.vertical_speed.sample(rng);
    let vx = st.horizontal_speed.sample(rng);
    let x = st.horizontal_offset.sample(rng);
    let theta = st.tilt.sample(rng);
    let omega = st.angular_rate.sample(rng);
    let propellant = st.propellant.sample(rng);
    let thrust = if st.engine_on {
        st.initial_throttle * v.max_thrust
    } else {
        0.0
    };
    InitialConditions {
        state: State {
            x,
            y,
            vx,
            vy,
            theta,
            omega,
            mass: v.dry_mass + propellant,
            gimbal: 0.0,
            thrust,
        },
        engine_on: st.engine_on,
        ignitions: st.ignitions,
    }
}

/// Which part of the vehicle is lowest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactPart {
    /// Tip of the left leg.
    LeftLeg,
    /// Tip of the right leg.
    RightLeg,
    /// The nose (the stage has tipped over).
    Nose,
}

/// Positions of the vehicle's extreme points and its lowest point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactGeometry {
    /// Height of the lowest point above the ground (m).
    pub clearance: f64,
    /// Which point is lowest.
    pub part: ContactPart,
    /// Position of the lowest point.
    pub point: (f64, f64),
    /// Velocity of the lowest point (`v + ω × r`).
    pub point_velocity: (f64, f64),
    /// Tail (engine) position.
    pub tail: (f64, f64),
    /// Left leg tip position.
    pub left_leg: (f64, f64),
    /// Right leg tip position.
    pub right_leg: (f64, f64),
    /// Center-of-mass height above the tail (m).
    pub cm_height: f64,
}

/// Computes the contact geometry of a state.
pub fn contact_geometry(v: &Vehicle, s: &State) -> ContactGeometry {
    let cm_height = mass_properties(v, s.mass).cm_height;
    let (sin_t, cos_t) = (libm::sin(s.theta), libm::cos(s.theta));
    let (bx, by) = (-sin_t, cos_t); // body axis, tail → nose
    let (nx, ny) = (cos_t, sin_t); // body "right"
    let tail = (s.x - cm_height * bx, s.y - cm_height * by);
    let nose_len = v.length - cm_height;
    let nose = (s.x + nose_len * bx, s.y + nose_len * by);
    let left_leg = (tail.0 - v.leg_half_span * nx, tail.1 - v.leg_half_span * ny);
    let right_leg = (tail.0 + v.leg_half_span * nx, tail.1 + v.leg_half_span * ny);

    let mut part = ContactPart::LeftLeg;
    let mut point = left_leg;
    if right_leg.1 < point.1 {
        part = ContactPart::RightLeg;
        point = right_leg;
    }
    if nose.1 < point.1 {
        part = ContactPart::Nose;
        point = nose;
    }
    let (rx, ry) = (point.0 - s.x, point.1 - s.y);
    let point_velocity = (s.vx - s.omega * ry, s.vy + s.omega * rx);
    ContactGeometry {
        clearance: point.1,
        part,
        point,
        point_velocity,
        tail,
        left_leg,
        right_leg,
        cm_height,
    }
}

/// Wraps an angle to `(−π, π]`.
pub fn wrap_angle(a: f64) -> f64 {
    libm::atan2(libm::sin(a), libm::cos(a))
}

/// Why a touchdown failed. Checked in this order; the first failing check is
/// reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CrashReason {
    /// The nose hit the ground first.
    TippedOver,
    /// A leg tip was outside the pad.
    OffPad,
    /// Vertical contact speed too high.
    VerticalSpeed,
    /// Horizontal contact speed too high.
    HorizontalSpeed,
    /// Tilt too large.
    Tilt,
    /// Rotating too fast.
    AngularRate,
}

/// How an episode ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// Successful landing.
    Landed,
    /// Touchdown outside the success criteria.
    Crashed(CrashReason),
    /// Propellant exhausted while airborne.
    OutOfPropellant,
    /// Left the allowed region.
    OutOfBounds,
    /// Ran out of time (truncation, not a terminal state).
    TimeLimit,
    /// The state became non-finite (should never happen; reported instead of
    /// panicking).
    Diverged,
}

impl Outcome {
    /// Every outcome, in code order.
    pub const ALL: [Outcome; 11] = [
        Outcome::Landed,
        Outcome::Crashed(CrashReason::TippedOver),
        Outcome::Crashed(CrashReason::OffPad),
        Outcome::Crashed(CrashReason::VerticalSpeed),
        Outcome::Crashed(CrashReason::HorizontalSpeed),
        Outcome::Crashed(CrashReason::Tilt),
        Outcome::Crashed(CrashReason::AngularRate),
        Outcome::OutOfPropellant,
        Outcome::OutOfBounds,
        Outcome::TimeLimit,
        Outcome::Diverged,
    ];

    /// Stable numeric code (0 is reserved for "still running").
    pub const fn code(self) -> u8 {
        match self {
            Outcome::Landed => 1,
            Outcome::Crashed(CrashReason::TippedOver) => 2,
            Outcome::Crashed(CrashReason::OffPad) => 3,
            Outcome::Crashed(CrashReason::VerticalSpeed) => 4,
            Outcome::Crashed(CrashReason::HorizontalSpeed) => 5,
            Outcome::Crashed(CrashReason::Tilt) => 6,
            Outcome::Crashed(CrashReason::AngularRate) => 7,
            Outcome::OutOfPropellant => 8,
            Outcome::OutOfBounds => 9,
            Outcome::TimeLimit => 10,
            Outcome::Diverged => 11,
        }
    }

    /// Inverse of [`Outcome::code`].
    pub fn from_code(code: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|o| o.code() == code)
    }

    /// Snake-case name, e.g. `"crashed_off_pad"`.
    pub const fn name(self) -> &'static str {
        match self {
            Outcome::Landed => "landed",
            Outcome::Crashed(CrashReason::TippedOver) => "crashed_tipped_over",
            Outcome::Crashed(CrashReason::OffPad) => "crashed_off_pad",
            Outcome::Crashed(CrashReason::VerticalSpeed) => "crashed_vertical_speed",
            Outcome::Crashed(CrashReason::HorizontalSpeed) => "crashed_horizontal_speed",
            Outcome::Crashed(CrashReason::Tilt) => "crashed_tilt",
            Outcome::Crashed(CrashReason::AngularRate) => "crashed_angular_rate",
            Outcome::OutOfPropellant => "out_of_propellant",
            Outcome::OutOfBounds => "out_of_bounds",
            Outcome::TimeLimit => "time_limit",
            Outcome::Diverged => "diverged",
        }
    }

    /// `true` for a time-limit truncation (the episode did not reach a
    /// terminal state).
    pub const fn is_truncation(self) -> bool {
        matches!(self, Outcome::TimeLimit)
    }
}

/// Judges a touchdown against the landing criteria. Speed, tilt and rate
/// limits are strict (`<`); both leg tips must be on the pad (`≤`).
pub fn evaluate_touchdown(criteria: &LandingCriteria, s: &State, c: &ContactGeometry) -> Outcome {
    let pad = criteria.pad_half_width;
    let failure = if c.part == ContactPart::Nose {
        Some(CrashReason::TippedOver)
    } else if c.left_leg.0.abs() > pad || c.right_leg.0.abs() > pad {
        Some(CrashReason::OffPad)
    } else if c.point_velocity.1.abs() >= criteria.max_vertical_speed {
        Some(CrashReason::VerticalSpeed)
    } else if c.point_velocity.0.abs() >= criteria.max_horizontal_speed {
        Some(CrashReason::HorizontalSpeed)
    } else if wrap_angle(s.theta).abs() >= criteria.max_tilt {
        Some(CrashReason::Tilt)
    } else if s.omega.abs() >= criteria.max_angular_rate {
        Some(CrashReason::AngularRate)
    } else {
        None
    };
    failure.map_or(Outcome::Landed, Outcome::Crashed)
}

/// Score of a finished episode, in `[0, 2000]`.
///
/// Landed: `1000` + up to `500` for propellant left + up to `300` for
/// accuracy (tail distance to the pad center) + up to `200` for a soft
/// vertical contact. Any other outcome: `0`.
pub fn score(
    criteria: &LandingCriteria,
    outcome: Option<Outcome>,
    contact: Option<&ContactGeometry>,
    propellant_left: f64,
    initial_propellant: f64,
) -> u32 {
    let (Some(Outcome::Landed), Some(c)) = (outcome, contact) else {
        return 0;
    };
    let fuel = if initial_propellant > 0.0 {
        (propellant_left / initial_propellant).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let accuracy = (1.0 - c.tail.0.abs() / criteria.pad_half_width).clamp(0.0, 1.0);
    let softness = (1.0 - c.point_velocity.1.abs() / criteria.max_vertical_speed).clamp(0.0, 1.0);
    let total = 1000.0 + 500.0 * fuel + 300.0 * accuracy + 200.0 * softness;
    // `total` is finite and within [1000, 2000]; the cast is exact after rounding.
    libm::round(total) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::params::DEG;

    fn upright(y: f64) -> State {
        let v = Params::easy().vehicle;
        State {
            y,
            mass: v.dry_mass,
            ..Default::default()
        }
    }

    #[test]
    fn upright_geometry() {
        let v = Params::easy().vehicle;
        let s = upright(100.0);
        let c = contact_geometry(&v, &s);
        assert!((c.clearance - (100.0 - v.dry_cm_height)).abs() < 1e-12);
        assert_eq!(c.left_leg.0, -v.leg_half_span);
        assert_eq!(c.right_leg.0, v.leg_half_span);
    }

    #[test]
    fn leaning_left_puts_the_left_leg_lowest() {
        let v = Params::easy().vehicle;
        let s = State {
            theta: 10.0 * DEG,
            ..upright(100.0)
        };
        let c = contact_geometry(&v, &s);
        assert_eq!(c.part, ContactPart::LeftLeg);
        assert!(c.left_leg.1 < c.right_leg.1);
    }

    #[test]
    fn rotation_adds_contact_point_velocity() {
        let v = Params::easy().vehicle;
        let s = State {
            omega: 0.2,
            ..upright(100.0)
        };
        let c = contact_geometry(&v, &s);
        // Lowest point is below the CoM: counter-clockwise rotation moves it right.
        assert!(c.point_velocity.0 > 2.0, "{:?}", c.point_velocity);
    }

    #[test]
    fn touchdown_judgement() {
        let p = Params::easy();
        let crit = &p.landing;
        let judge = |s: State| evaluate_touchdown(crit, &s, &contact_geometry(&p.vehicle, &s));
        let base = upright(p.vehicle.dry_cm_height);
        assert_eq!(judge(State { vy: -1.5, ..base }), Outcome::Landed);
        assert_eq!(
            judge(State { vy: -2.5, ..base }),
            Outcome::Crashed(CrashReason::VerticalSpeed)
        );
        assert_eq!(
            judge(State { vx: 1.5, ..base }),
            Outcome::Crashed(CrashReason::HorizontalSpeed)
        );
        assert_eq!(
            judge(State { x: 35.0, ..base }),
            Outcome::Crashed(CrashReason::OffPad)
        );
        assert_eq!(
            judge(State {
                theta: 4.0 * DEG,
                ..base
            }),
            Outcome::Landed
        );
        assert_eq!(
            judge(State {
                theta: 6.0 * DEG,
                ..base
            }),
            Outcome::Crashed(CrashReason::Tilt)
        );
        assert_eq!(
            judge(State {
                theta: 120.0 * DEG,
                ..base
            }),
            Outcome::Crashed(CrashReason::TippedOver)
        );
        assert_eq!(
            judge(State {
                omega: 0.05,
                ..base
            }),
            Outcome::Landed,
            "leg tip at 0.75 m/s is within limits"
        );
        assert_eq!(
            judge(State {
                omega: 0.08,
                ..base
            }),
            Outcome::Crashed(CrashReason::HorizontalSpeed),
            "0.08 rad/s at a 15 m lever moves the leg tip at 1.2 m/s"
        );
    }

    #[test]
    fn outcome_codes_round_trip_and_are_unique() {
        let mut codes: Vec<u8> = Outcome::ALL.iter().map(|o| o.code()).collect();
        for o in Outcome::ALL {
            assert_eq!(Outcome::from_code(o.code()), Some(o));
        }
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), Outcome::ALL.len());
        assert_eq!(Outcome::from_code(0), None);
    }

    #[test]
    fn score_rewards_fuel_accuracy_and_softness() {
        let p = Params::easy();
        let s = upright(p.vehicle.dry_cm_height);
        let c = contact_geometry(&p.vehicle, &s);
        assert_eq!(
            score(
                &p.landing,
                Some(Outcome::Landed),
                Some(&c),
                6_000.0,
                6_000.0
            ),
            2000
        );
        assert_eq!(
            score(&p.landing, Some(Outcome::Landed), Some(&c), 0.0, 6_000.0),
            1500
        );
        assert_eq!(
            score(
                &p.landing,
                Some(Outcome::OutOfBounds),
                Some(&c),
                6_000.0,
                6_000.0
            ),
            0
        );
        assert_eq!(score(&p.landing, None, None, 6_000.0, 6_000.0), 0);
    }

    #[test]
    fn wrap_angle_range() {
        assert!((wrap_angle(3.0 * core::f64::consts::PI) - core::f64::consts::PI).abs() < 1e-9);
        assert!((wrap_angle(-0.5) + 0.5).abs() < 1e-15);
    }
}
