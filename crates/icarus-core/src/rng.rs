//! Deterministic random numbers.
//!
//! [`SplitMix64`] expands a 64-bit seed into the 256-bit state of a
//! xoshiro256++ generator ([`Rng`]), following the reference implementations
//! by Blackman and Vigna (<https://prng.di.unimi.it/>). Everything is integer
//! arithmetic plus `libm`, so the output is identical on every target.
//!
//! Each subsystem draws from its own [`Stream`], so for example the wind never
//! depends on how many sensor-noise samples were drawn, and therefore never
//! depends on what the controller does.

/// Independent random streams, one per subsystem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stream {
    /// Initial conditions of the episode.
    InitialConditions,
    /// Mean wind, turbulence and discrete gusts.
    Wind,
    /// Sensor noise.
    Sensors,
    /// Random fault schedule.
    Faults,
}

impl Stream {
    /// Stable numeric identifier mixed into the seed.
    const fn id(self) -> u64 {
        match self {
            Stream::InitialConditions => 1,
            Stream::Wind => 2,
            Stream::Sensors => 3,
            Stream::Faults => 4,
        }
    }
}

/// SplitMix64 generator, used only to seed [`Rng`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator from a seed.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// xoshiro256++ generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Seeds the generator by running SplitMix64 on `seed`.
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let mut s = [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()];
        if s == [0; 4] {
            // The all-zero state is the one invalid state of xoshiro.
            s[0] = 1;
        }
        Self { s }
    }

    /// Creates the generator for one subsystem of an episode.
    ///
    /// The episode seed is first scrambled through SplitMix64 and then mixed
    /// with the stream identifier, so nearby seeds and different streams give
    /// unrelated sequences.
    pub fn for_stream(seed: u64, stream: Stream) -> Self {
        let scrambled = SplitMix64::new(seed).next_u64();
        let mixed = scrambled ^ stream.id().wrapping_mul(0xD1B5_4A32_D192_ED03);
        Self::from_seed(mixed)
    }

    /// Returns the next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform sample in `[0, 1)` with 53 bits of resolution.
    pub fn uniform(&mut self) -> f64 {
        // Both the integer-to-float conversion (53-bit value) and the scaling
        // by a power of two are exact.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Standard normal sample (Box–Muller, one value per two uniforms).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform(); // (0, 1], so the logarithm is finite
        let u2 = self.uniform();
        libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(core::f64::consts::TAU * u2)
    }

    /// Returns `true` with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    /// Returns `+1.0` or `-1.0` with equal probability.
    pub fn sign(&mut self) -> f64 {
        if self.next_u64() >> 63 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_matches_reference_vectors() {
        // Reference outputs for seed 1234567 (published SplitMix64 test vector).
        let mut sm = SplitMix64::new(1_234_567);
        let expected: [u64; 5] = [
            6_457_827_717_110_365_317,
            3_203_168_211_198_807_973,
            9_817_491_932_198_370_423,
            4_593_380_528_125_082_431,
            16_408_922_859_458_223_821,
        ];
        for e in expected {
            assert_eq!(sm.next_u64(), e);
        }
    }

    #[test]
    fn xoshiro_matches_reference_implementation() {
        // Expected values computed with an independent Python port of the C
        // reference implementation (xoshiro256plusplus.c), seeded with
        // SplitMix64(0).
        let mut rng = Rng::from_seed(0);
        let expected: [u64; 4] = XOSHIRO_SEED0_REFERENCE;
        for e in expected {
            assert_eq!(rng.next_u64(), e);
        }
    }

    /// First four outputs of xoshiro256++ whose state is filled by
    /// SplitMix64(0), from the reference algorithm.
    const XOSHIRO_SEED0_REFERENCE: [u64; 4] = [
        0x53175d61490b23df,
        0x61da6f3dc380d507,
        0x5c0fdf91ec9a7bfc,
        0x02eebf8c3bbe5e1a,
    ];

    #[test]
    fn uniform_stays_in_unit_interval_and_is_centered() {
        let mut rng = Rng::from_seed(7);
        let n = 200_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
            sum += u;
        }
        let mean = sum / f64::from(n);
        assert!((mean - 0.5).abs() < 0.005, "mean = {mean}");
    }

    #[test]
    fn normal_has_unit_variance() {
        let mut rng = Rng::from_seed(11);
        let n = 200_000;
        let (mut sum, mut sum_sq) = (0.0, 0.0);
        for _ in 0..n {
            let x = rng.normal();
            assert!(x.is_finite());
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / f64::from(n);
        let var = sum_sq / f64::from(n) - mean * mean;
        assert!(mean.abs() < 0.01, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.02, "var = {var}");
    }

    #[test]
    fn streams_are_distinct_and_reproducible() {
        let a1 = Rng::for_stream(42, Stream::Wind).next_u64();
        let a2 = Rng::for_stream(42, Stream::Wind).next_u64();
        let b = Rng::for_stream(42, Stream::Sensors).next_u64();
        let c = Rng::for_stream(43, Stream::Wind).next_u64();
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
        assert_ne!(a1, c);
    }
}
