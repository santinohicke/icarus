//! Continuous state vector and normalized controls.

/// Number of continuous state variables.
pub const STATE_DIM: usize = 9;

/// Continuous state integrated by RK4.
///
/// Conventions: `x` to the right, `y` up, origin at the pad center; `theta`
/// counter-clockwise from vertical (positive = nose leans left).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct State {
    /// Horizontal position of the center of mass (m).
    pub x: f64,
    /// Altitude of the center of mass (m).
    pub y: f64,
    /// Horizontal velocity (m/s).
    pub vx: f64,
    /// Vertical velocity (m/s).
    pub vy: f64,
    /// Tilt from vertical, counter-clockwise (rad).
    pub theta: f64,
    /// Angular velocity, counter-clockwise (rad/s).
    pub omega: f64,
    /// Total mass (kg).
    pub mass: f64,
    /// Actual gimbal angle (rad).
    pub gimbal: f64,
    /// Actual thrust (N).
    pub thrust: f64,
}

impl State {
    /// Packs the state into the integrator vector.
    pub const fn to_array(&self) -> [f64; STATE_DIM] {
        [
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.theta,
            self.omega,
            self.mass,
            self.gimbal,
            self.thrust,
        ]
    }

    /// Unpacks an integrator vector.
    pub const fn from_array(a: &[f64; STATE_DIM]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            vx: a[2],
            vy: a[3],
            theta: a[4],
            omega: a[5],
            mass: a[6],
            gimbal: a[7],
            thrust: a[8],
        }
    }

    /// `true` when no component is NaN or infinite.
    pub fn is_finite(&self) -> bool {
        self.to_array().iter().all(|v| v.is_finite())
    }
}

/// Number of quantization levels per control axis (`0..=254`).
pub const QUANT_LEVELS: u8 = 255;

/// Highest valid quantized level.
pub const QUANT_MAX: u8 = QUANT_LEVELS - 1;

/// The quantized level that maps to exactly `0.0`.
pub const QUANT_CENTER: u8 = 127;

/// Maps a quantized level to a normalized value in `[-1, 1]`.
///
/// Levels above 254 are treated as 254. The mapping is exact at both ends
/// and at the center.
pub fn dequantize(q: u8) -> f64 {
    let q = q.min(QUANT_LEVELS - 1);
    (f64::from(q) - f64::from(QUANT_CENTER)) / f64::from(QUANT_CENTER)
}

/// Maps a normalized value to the nearest quantized level.
pub fn quantize(u: f64) -> u8 {
    if u.is_nan() {
        return QUANT_CENTER;
    }
    let level = libm::round(u.clamp(-1.0, 1.0) * f64::from(QUANT_CENTER) + f64::from(QUANT_CENTER));
    // `level` is an integer in [0, 254]; the cast is exact.
    level as u8
}

/// Normalized controls, each in `[-1, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Controls {
    /// Throttle: below the vehicle's cut-off the engine is commanded off.
    pub throttle: f64,
    /// Gimbal: positive deflection gives a counter-clockwise torque.
    pub gimbal: f64,
    /// Cold-gas attitude thrusters: positive gives a counter-clockwise torque.
    pub rcs: f64,
}

impl Controls {
    /// Builds sanitized controls: values are clamped to `[-1, 1]`; a NaN
    /// throttle means "engine off" and a NaN gimbal or RCS means "neutral".
    pub fn new(throttle: f64, gimbal: f64, rcs: f64) -> Self {
        fn clamp_or(v: f64, fallback: f64) -> f64 {
            if v.is_nan() {
                fallback
            } else {
                v.clamp(-1.0, 1.0)
            }
        }
        Self {
            throttle: clamp_or(throttle, -1.0),
            gimbal: clamp_or(gimbal, 0.0),
            rcs: clamp_or(rcs, 0.0),
        }
    }

    /// Builds controls from quantized levels `[throttle, gimbal, rcs]`.
    pub fn from_quantized(q: [u8; 3]) -> Self {
        Self {
            throttle: dequantize(q[0]),
            gimbal: dequantize(q[1]),
            rcs: dequantize(q[2]),
        }
    }

    /// Quantizes the controls (inverse of [`Controls::from_quantized`] up to
    /// rounding).
    pub fn quantized(&self) -> [u8; 3] {
        [
            quantize(self.throttle),
            quantize(self.gimbal),
            quantize(self.rcs),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_is_exact_at_landmarks() {
        assert_eq!(dequantize(0), -1.0);
        assert_eq!(dequantize(127), 0.0);
        assert_eq!(dequantize(254), 1.0);
        assert_eq!(dequantize(255), 1.0, "255 is clamped to 254");
    }

    #[test]
    fn quantize_round_trips_every_level() {
        for q in 0..=254u8 {
            assert_eq!(quantize(dequantize(q)), q);
        }
        assert_eq!(quantize(f64::NAN), QUANT_CENTER);
        assert_eq!(quantize(5.0), 254);
        assert_eq!(quantize(-5.0), 0);
    }

    #[test]
    fn controls_are_sanitized() {
        let c = Controls::new(f64::NAN, f64::INFINITY, -7.0);
        assert_eq!(
            c,
            Controls {
                throttle: -1.0,
                gimbal: 1.0,
                rcs: -1.0
            }
        );
        let c = Controls::new(0.5, f64::NAN, f64::NAN);
        assert_eq!(
            c,
            Controls {
                throttle: 0.5,
                gimbal: 0.0,
                rcs: 0.0
            }
        );
    }

    #[test]
    fn state_array_round_trip() {
        let s = State {
            x: 1.0,
            y: 2.0,
            vx: 3.0,
            vy: 4.0,
            theta: 5.0,
            omega: 6.0,
            mass: 7.0,
            gimbal: 8.0,
            thrust: 9.0,
        };
        assert_eq!(State::from_array(&s.to_array()), s);
    }
}
