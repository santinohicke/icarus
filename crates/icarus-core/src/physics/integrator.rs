//! Classical fourth-order Runge–Kutta with a fixed step.

/// Advances `s` by `h` with RK4, holding the inputs captured by `f` constant.
///
/// The operation order is fixed, so the result is bit-identical on every
/// target for the same inputs.
pub fn rk4_step<const N: usize, F>(f: F, s: &[f64; N], h: f64) -> [f64; N]
where
    F: Fn(&[f64; N]) -> [f64; N],
{
    let half = 0.5 * h;
    let k1 = f(s);
    let s2: [f64; N] = core::array::from_fn(|i| s[i] + half * k1[i]);
    let k2 = f(&s2);
    let s3: [f64; N] = core::array::from_fn(|i| s[i] + half * k2[i]);
    let k3 = f(&s3);
    let s4: [f64; N] = core::array::from_fn(|i| s[i] + h * k3[i]);
    let k4 = f(&s4);
    let sixth = h / 6.0;
    core::array::from_fn(|i| s[i] + sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrates_exponential_decay_to_fourth_order() {
        // y' = -y, y(0) = 1 → y(1) = e^-1.
        let exact = libm::exp(-1.0);
        let err = |steps: u32| {
            let h = 1.0 / f64::from(steps);
            let mut y = [1.0];
            for _ in 0..steps {
                y = rk4_step(|s: &[f64; 1]| [-s[0]], &y, h);
            }
            (y[0] - exact).abs()
        };
        let ratio = err(10) / err(20);
        assert!((14.0..18.0).contains(&ratio), "error ratio {ratio}");
    }

    #[test]
    fn integrates_cubic_polynomial_exactly() {
        // RK4 is exact for y' = 3t² (written as an autonomous system).
        let mut s = [0.0, 0.0]; // [t, y]
        for _ in 0..10 {
            s = rk4_step(|s: &[f64; 2]| [1.0, 3.0 * s[0] * s[0]], &s, 0.1);
        }
        assert!((s[1] - 1.0).abs() < 1e-12);
    }
}
