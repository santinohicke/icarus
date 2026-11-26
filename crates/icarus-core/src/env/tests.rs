use super::*;
use crate::eval::Scenario;
use crate::physics::params::{Range, VehicleConfig};
use crate::physics::state::dequantize;
use crate::rng::Rng;

fn env(config: VehicleConfig) -> Env {
    Env::new(EnvConfig::default(), Task::from_config(config)).unwrap()
}

/// Runs an episode with a constant action; returns the rewards.
fn rollout(env: &mut Env, seed: u64, action: [f64; ACTION_DIM]) -> (Vec<f64>, Step) {
    env.reset(seed);
    let mut rewards = Vec::new();
    loop {
        let step = env.step(action).unwrap();
        rewards.push(step.reward);
        if step.done() {
            return (rewards, step);
        }
        assert!(rewards.len() < 10_000, "episode never ends");
    }
}

#[test]
fn default_dimensions() {
    let cfg = EnvConfig::default();
    assert_eq!(DEFAULT_HISTORY, 4);
    assert_eq!(cfg.obs_dim(), BASE_OBS_DIM + 12);
    assert_eq!(env(VehicleConfig::Easy).observation().len(), 26);
}

#[test]
fn observation_is_zero_before_reset_and_finite_after() {
    let mut e = env(VehicleConfig::Realistic);
    assert!(e.observation().iter().all(|&v| v == 0.0));
    e.reset(3);
    let o = e.observation();
    assert!(o.iter().all(|v| v.is_finite()));
    assert_eq!(
        o[obs::ENGINE_ON],
        0.0,
        "realistic starts with the engine off"
    );
    assert_eq!(o[obs::IGNITIONS], 1.0);
    assert!((o[obs::DELAY] - 0.25).abs() < 1e-7, "3 of 12 ticks");
    let s = o[obs::SIN_THETA];
    let c = o[obs::COS_THETA];
    assert!((s * s + c * c - 1.0).abs() < 1e-5);
    assert!(o[obs::HEIGHT] > 13.0, "starts above 1.3 km");
    // The realistic engine cannot hover even at minimum throttle: 0.4·TWR > 1.
    assert!(0.4 * o[obs::TWR] > 1.0, "{}", o[obs::TWR]);
    let mut easy = env(VehicleConfig::Easy);
    easy.reset(3);
    assert!(0.4 * easy.observation()[obs::TWR] < 1.0, "easy can hover");
}

#[test]
fn episodes_are_deterministic() {
    let mut a = env(VehicleConfig::Crosswind);
    let mut b = env(VehicleConfig::Crosswind);
    a.reset(42);
    b.reset(42);
    for k in 0..300 {
        let action = [0.1 * libm::sin(f64::from(k)), 0.3, -0.2];
        let sa = a.step(action).unwrap();
        let sb = b.step(action).unwrap();
        assert_eq!(sa.reward.to_bits(), sb.reward.to_bits());
        assert_eq!(a.observation(), b.observation());
        if sa.done() {
            break;
        }
    }
}

#[test]
fn shaping_telescopes_to_minus_initial_potential() {
    // Without fuel cost, the discounted return of a terminated episode is
    // exactly `−Φ(s0) + γ^(T−1)·(terminal reward)`, whenever it ends.
    let mut cfg = EnvConfig::default();
    cfg.reward.fuel_weight = 0.0;
    cfg.reward.time_weight = 0.0;
    cfg.reward.zero_terminal_potential = true;
    let mut e = Env::new(cfg.clone(), Task::from_config(VehicleConfig::Realistic)).unwrap();
    for (seed, action) in [
        (1, [-1.0, 0.0, 0.0]),
        (2, [1.0, 0.5, 0.0]),
        (3, [0.2, 0.0, 1.0]),
    ] {
        e.reset(seed);
        let phi0 = e.potential();
        assert!(phi0 < 0.0);
        let (rewards, last) = rollout(&mut e, seed, action);
        assert!(last.terminated, "{:?}", last.outcome);
        let gamma = cfg.reward.gamma;
        let mut discounted = 0.0;
        let mut g = 1.0;
        for r in &rewards {
            discounted += g * r;
            g *= gamma;
        }
        let terminal = if last.outcome == Some(Outcome::Landed) {
            cfg.reward.landing_bonus
        } else {
            -cfg.reward.crash_penalty
        };
        let last_discount = libm::pow(gamma, (rewards.len() - 1) as f64);
        let expected = -phi0 + last_discount * terminal;
        assert!(
            (discounted - expected).abs() < 1e-6,
            "seed {seed}: {discounted} vs {expected}"
        );
        assert_eq!(e.potential(), 0.0, "Φ(terminal) = 0");
    }
}

#[test]
fn truncation_keeps_the_potential() {
    let mut p = VehicleConfig::Easy.params();
    p.limits.max_time = 2.0;
    let task = Task::new(p, Randomization::none()).unwrap();
    let cfg = EnvConfig::default();
    let mut e = Env::new(cfg.clone(), task).unwrap();
    e.reset(5);
    let reference = VehicleConfig::Easy.params().start.propellant.max;
    loop {
        let before = e.potential();
        let fuel_before = e.sim().unwrap().propellant();
        let t_before = e.sim().unwrap().time();
        let step = e.step([0.0, 0.0, 0.0]).unwrap();
        let fuel_cost =
            cfg.reward.fuel_weight * (fuel_before - e.sim().unwrap().propellant()) / reference;
        let time_cost = cfg.reward.time_weight * (e.sim().unwrap().time() - t_before);
        assert!((time_cost - cfg.reward.time_weight * 3.0 / 60.0).abs() < 1e-12);
        let after = e.potential();
        let expected = cfg.reward.gamma * after - before - fuel_cost - time_cost;
        assert!((step.reward - expected).abs() < 1e-12);
        if step.done() {
            assert!(step.truncated && !step.terminated);
            assert_eq!(step.outcome, Some(Outcome::TimeLimit));
            assert!(after < 0.0, "not reset to zero at truncation");
            break;
        }
    }
    // 2 s at 20 Hz is 40 decisions.
    assert_eq!(e.episode_length(), 40);
}

#[test]
fn landing_earns_the_bonus() {
    // The soft-touchdown setup of the simulator tests, flown one tick per
    // decision by a descent-rate controller reading the true state.
    let mut p = VehicleConfig::Easy.params();
    p.environment.rho0 = 0.0;
    p.wind.mean = Range::fixed(0.0);
    p.wind.turbulence_sigma = 0.0;
    p.start.altitude = Range::fixed(p.vehicle.length + p.vehicle.leg_half_span + 0.5);
    p.start.vertical_speed = Range::fixed(-1.0);
    p.start.horizontal_speed = Range::fixed(0.0);
    p.start.horizontal_offset = Range::fixed(0.0);
    p.start.tilt = Range::fixed(0.0);
    p.start.angular_rate = Range::fixed(0.0);
    let cfg = EnvConfig {
        action_repeat: 1,
        quantize_actions: false,
        ..EnvConfig::default()
    };
    let mut e = Env::new(
        cfg.clone(),
        Task::new(p.clone(), Randomization::none()).unwrap(),
    )
    .unwrap();
    e.reset(1);
    let v = &p.vehicle;
    loop {
        let s = *e.sim().unwrap().state();
        let accel = 2.0 * (-1.0 - s.vy);
        let frac =
            (s.mass * (p.environment.gravity + accel) / v.max_thrust).clamp(v.min_throttle, 1.0);
        let u = v.throttle_cutoff
            + (frac - v.min_throttle) / (1.0 - v.min_throttle) * (1.0 - v.throttle_cutoff);
        let before = e.potential();
        let fuel_before = e.sim().unwrap().propellant();
        let t_before = e.sim().unwrap().time();
        let step = e.step([u, 0.0, 0.0]).unwrap();
        if step.done() {
            assert_eq!(step.outcome, Some(Outcome::Landed));
            assert!(step.terminated);
            let burnt = fuel_before - e.sim().unwrap().propellant();
            let fuel_cost = cfg.reward.fuel_weight * burnt / p.start.propellant.max;
            // Φ(terminal) = 0, so the last reward is the bonus minus Φ(s).
            // The touchdown time is exact, so the last decision is shorter.
            let flown = e.sim().unwrap().time() - t_before;
            assert!(flown > 0.0 && flown <= 1.0 / 60.0 + 1e-12);
            let expected =
                cfg.reward.landing_bonus - before - fuel_cost - cfg.reward.time_weight * flown;
            assert!((step.reward - expected).abs() < 1e-12, "{}", step.reward);
            assert!(before > -1.0, "close to the target just before touchdown");
            break;
        }
        assert!(e.episode_length() < 10_000);
    }
}

#[test]
fn crashes_and_out_of_propellant_pay_the_penalty() {
    let mut e = env(VehicleConfig::Realistic);
    let (_, last) = rollout(&mut e, 9, [-1.0, 0.0, 0.0]);
    assert!(matches!(last.outcome, Some(Outcome::Crashed(_))));
    assert!(last.reward < -50.0, "{}", last.reward);

    let mut p = VehicleConfig::Easy.params();
    p.start.propellant = Range::fixed(50.0);
    let mut e = Env::new(
        EnvConfig::default(),
        Task::new(p, Randomization::none()).unwrap(),
    )
    .unwrap();
    let (_, last) = rollout(&mut e, 1, [1.0, 0.0, 0.0]);
    assert_eq!(last.outcome, Some(Outcome::OutOfPropellant));
    assert!(last.terminated);
    assert!(last.reward < -50.0);
}

#[test]
fn fuel_cost_matches_the_burn() {
    let mut cfg = EnvConfig::default();
    let r = &mut cfg.reward;
    r.position_weight = 0.0;
    r.altitude_weight = 0.0;
    r.velocity_weight = 0.0;
    r.tilt_weight = 0.0;
    r.rate_weight = 0.0;
    r.time_weight = 0.0;
    let mut e = Env::new(cfg.clone(), Task::from_config(VehicleConfig::Easy)).unwrap();
    e.reset(4);
    let before = e.sim().unwrap().propellant();
    let step = e.step([1.0, 0.0, 0.0]).unwrap();
    let burnt = before - e.sim().unwrap().propellant();
    let reference = VehicleConfig::Easy.params().start.propellant.max;
    let expected = -cfg.reward.fuel_weight * burnt / reference;
    assert!(burnt > 0.0);
    assert!((step.reward - expected).abs() < 1e-12);
}

#[test]
fn potential_is_bounded_and_zero_at_rest_on_the_pad() {
    let r = RewardParams::default();
    let p = VehicleConfig::Realistic.params();
    let decel = r.profile_decel(&p);
    let mut rng = Rng::from_seed(7);
    for _ in 0..10_000 {
        let s = State {
            x: (rng.uniform() - 0.5) * 6_000.0,
            y: rng.uniform() * 4_000.0,
            vx: (rng.uniform() - 0.5) * 400.0,
            vy: (rng.uniform() - 0.5) * 400.0,
            theta: (rng.uniform() - 0.5) * 20.0,
            omega: (rng.uniform() - 0.5) * 10.0,
            mass: p.vehicle.dry_mass + rng.uniform() * 5_000.0,
            gimbal: 0.0,
            thrust: 0.0,
        };
        let phi = r.potential(&p, decel, &s);
        assert!(phi <= 0.0 && phi > -r.max_abs_potential(), "{phi}");
    }
    let mass = p.vehicle.dry_mass;
    let cm = mass_properties(&p.vehicle, mass).cm_height;
    let rest = State {
        y: cm,
        mass,
        ..State::default()
    };
    assert_eq!(r.potential(&p, decel, &rest), 0.0);
    // Descending at the profile speed costs nothing; faster does.
    let at_profile = State {
        y: cm + 100.0,
        vy: -libm::sqrt(1.0 + 2.0 * decel * 100.0),
        mass,
        ..State::default()
    };
    let faster = State {
        vy: at_profile.vy - 5.0,
        ..at_profile
    };
    let altitude_only =
        -r.altitude_weight * (100.0 / r.altitude_scale) / (1.0 + 100.0 / r.altitude_scale);
    assert!((r.potential(&p, decel, &at_profile) - altitude_only).abs() < 1e-9);
    assert!(r.potential(&p, decel, &faster) < altitude_only);
}

#[test]
fn history_shows_only_the_actions_within_the_delay() {
    for (delay, shown) in [(0usize, 0usize), (1, 1), (3, 1), (4, 2), (6, 2), (12, 4)] {
        let mut p = VehicleConfig::Easy.params();
        p.sensors.delay_ticks = delay;
        let mut e = Env::new(
            EnvConfig::default(),
            Task::new(p, Randomization::none()).unwrap(),
        )
        .unwrap();
        e.reset(1);
        let actions = [
            [0.5, 0.1, -0.3],
            [0.4, -0.2, 0.3],
            [0.3, 0.3, 0.0],
            [0.2, 0.0, 1.0],
            [0.1, -1.0, -1.0],
        ];
        for a in actions {
            e.step(a).unwrap();
        }
        let o = e.observation();
        for k in 0..DEFAULT_HISTORY {
            let at = obs::HISTORY + k * ACTION_DIM;
            let got = &o[at..at + ACTION_DIM];
            if k < shown {
                // Most recent first, as quantized and applied.
                let a = actions[actions.len() - 1 - k];
                let want = a.map(|v| dequantize(quantize(v)) as f32);
                assert_eq!(got, want, "delay {delay}, k {k}");
            } else {
                assert_eq!(got, [0.0; 3], "delay {delay}, k {k}");
            }
        }
    }
}

#[test]
fn history_ablation_zeroes_it() {
    let cfg = EnvConfig {
        action_history: false,
        ..EnvConfig::default()
    };
    let mut e = Env::new(cfg, Task::from_config(VehicleConfig::Storm)).unwrap();
    e.reset(2);
    e.step([0.5, 0.5, 0.5]).unwrap();
    assert!(e.observation()[obs::HISTORY..].iter().all(|&v| v == 0.0));
}

#[test]
fn actions_are_quantized_like_human_input() {
    let mut e = env(VehicleConfig::Easy);
    e.reset(1);
    e.step([0.5, -0.33, 0.77]).unwrap();
    let c = e.sim().unwrap().controls();
    assert_eq!(c.throttle, dequantize(quantize(0.5)));
    assert_eq!(c.gimbal, dequantize(quantize(-0.33)));
    assert_eq!(c.rcs, dequantize(quantize(0.77)));

    let cfg = EnvConfig {
        quantize_actions: false,
        ..EnvConfig::default()
    };
    let mut e = Env::new(cfg, Task::from_config(VehicleConfig::Easy)).unwrap();
    e.reset(1);
    e.step([0.5, -0.33, 2.0]).unwrap();
    let c = e.sim().unwrap().controls();
    assert_eq!((c.throttle, c.gimbal, c.rcs), (0.5, -0.33, 1.0));
}

#[test]
fn misuse_is_reported() {
    let mut e = env(VehicleConfig::Easy);
    assert_eq!(e.step([0.0; 3]), Err(EnvError::NotReset));
    e.reset(1);
    assert!(matches!(
        e.step([f64::NAN, 0.0, 0.0]),
        Err(EnvError::NonFiniteAction(_))
    ));
    assert!(matches!(
        e.step([0.0, f64::INFINITY, 0.0]),
        Err(EnvError::NonFiniteAction(_))
    ));
    while !e.step([-1.0, 0.0, 0.0]).unwrap().done() {}
    assert_eq!(e.step([0.0; 3]), Err(EnvError::EpisodeOver));
}

#[test]
fn config_validation() {
    let mut cfg = EnvConfig::default();
    cfg.reward.crash_penalty = cfg.reward.max_abs_potential();
    assert!(cfg.validate().is_err(), "crash penalty must exceed max |Φ|");
    let cfg = EnvConfig {
        action_repeat: 0,
        ..EnvConfig::default()
    };
    assert!(cfg.validate().is_err());
    let cfg = EnvConfig {
        history_len: MAX_HISTORY + 1,
        ..EnvConfig::default()
    };
    assert!(cfg.validate().is_err());
    let mut cfg = EnvConfig::default();
    cfg.reward.gamma = 1.5;
    assert!(cfg.validate().is_err());
    let mut cfg = EnvConfig::default();
    cfg.reward.tilt_scale = 0.0;
    assert!(cfg.validate().is_err());
    assert!(EnvConfig::default().validate().is_ok());
}

#[test]
fn set_task_takes_effect_at_the_next_reset() {
    let mut e = env(VehicleConfig::Easy);
    e.reset(1);
    e.set_task(Task::from_config(VehicleConfig::Realistic))
        .unwrap();
    assert!(e.sim().unwrap().engine_on(), "current episode unchanged");
    assert_eq!(e.task().params(), &VehicleConfig::Realistic.params());
    e.reset(1);
    assert!(!e.sim().unwrap().engine_on(), "now realistic");
}

#[test]
fn randomization_leaves_the_other_streams_alone() {
    let r = Randomization {
        dry_mass: Range::new(0.9, 1.1),
        wind: Range::new(0.5, 1.5),
        thrust_time_constant: Range::new(0.8, 1.2),
        ..Randomization::none()
    };
    let plain = Task::from_config(VehicleConfig::Easy);
    let randomized = Task::new(VehicleConfig::Easy.params(), r).unwrap();
    let mut a = Env::new(EnvConfig::default(), plain).unwrap();
    let mut b = Env::new(EnvConfig::default(), randomized).unwrap();
    for seed in 0..20 {
        a.reset(seed);
        b.reset(seed);
        let (sa, sb) = (a.sim().unwrap().state(), b.sim().unwrap().state());
        assert_eq!(
            (sa.x, sa.y, sa.vx, sa.vy, sa.theta),
            (sb.x, sb.y, sb.vx, sb.vy, sb.theta)
        );
        assert_eq!(
            a.sim().unwrap().last_sensor_noise(),
            b.sim().unwrap().last_sensor_noise()
        );
        assert_ne!(a.sim().unwrap().params(), b.sim().unwrap().params());
    }
}

#[test]
fn episode_seeds_are_distinct_and_never_evaluation_seeds() {
    let mut seen = std::collections::HashSet::new();
    for base in 0..4 {
        for env_index in 0..16 {
            for episode in 0..64 {
                let s = episode_seed(base, env_index, episode);
                assert!(s >= 1 << 63);
                assert!(seen.insert(s), "collision");
            }
        }
    }
}

fn slot(autoreset: bool) -> EnvSlot {
    let mut p = VehicleConfig::Easy.params();
    p.limits.max_time = 0.5; // 10 decisions
    let task = Task::new(p, Randomization::none()).unwrap();
    EnvSlot::new(
        Env::new(EnvConfig::default(), task).unwrap(),
        11,
        3,
        autoreset,
    )
}

#[test]
fn slot_autoreset_returns_the_terminal_observation() {
    let mut s = slot(true);
    let dim = s.env().obs_dim();
    let (mut o, mut t) = (vec![0.0; dim], vec![0.0; dim]);
    s.reset(None, &mut o).unwrap();
    assert_eq!(s.env().seed(), Some(episode_seed(11, 3, 0)));
    let mut reference = Env::new(EnvConfig::default(), s.env().task().clone()).unwrap();
    reference.reset(episode_seed(11, 3, 0));
    let mut ret = 0.0;
    for k in 0..10 {
        let st = s.step([0.0, 0.0, 0.0], &mut o, &mut t).unwrap();
        let rs = reference.step([0.0, 0.0, 0.0]).unwrap();
        assert_eq!(st.reward.to_bits(), rs.reward.to_bits());
        ret += st.reward;
        if k < 9 {
            assert!(!st.truncated && st.outcome == 0);
            assert_eq!(o, reference.observation());
        } else {
            assert!(st.truncated && !st.terminated);
            assert_eq!(st.outcome, Outcome::TimeLimit.code());
            assert_eq!(st.episode_length, 10);
            assert_eq!(st.seed, episode_seed(11, 3, 0));
            assert!((st.episode_return - ret).abs() < 1e-9);
            assert_eq!(t, reference.observation(), "terminal observation");
            assert_eq!(s.env().seed(), Some(episode_seed(11, 3, 1)), "auto-reset");
            reference.reset(episode_seed(11, 3, 1));
            assert_eq!(
                o,
                reference.observation(),
                "first observation of the next episode"
            );
        }
    }
}

#[test]
fn slot_without_autoreset_goes_idle() {
    let mut s = slot(false);
    let dim = s.env().obs_dim();
    let (mut o, mut t) = (vec![0.0; dim], vec![0.0; dim]);
    s.reset(Some(77), &mut o).unwrap();
    for _ in 0..10 {
        s.step([0.0; 3], &mut o, &mut t).unwrap();
    }
    assert!(s.finished());
    let last = o.clone();
    assert_eq!(last, t);
    let st = s.step([1.0; 3], &mut o, &mut t).unwrap();
    assert!(st.idle && st.reward == 0.0 && !st.terminated && !st.truncated);
    assert_eq!(o, last);
    assert_eq!(s.env().seed(), Some(77));
}

#[test]
fn evaluation_scenarios_run_through_the_env() {
    for scenario in Scenario::ALL {
        let mut e = Env::new(
            EnvConfig::default(),
            Task::from_scenario(VehicleConfig::Realistic, scenario),
        )
        .unwrap();
        let (_, last) = rollout(&mut e, 0, [0.0, 0.0, 0.0]);
        assert!(last.done());
        assert!(e.observation().iter().all(|v| v.is_finite()));
    }
}

#[test]
fn truncated_shaping_telescopes_to_the_last_potential() {
    let mut cfg = EnvConfig::default();
    cfg.reward.fuel_weight = 0.0;
    cfg.reward.time_weight = 0.0;
    let mut p = VehicleConfig::Easy.params();
    p.limits.max_time = 3.0;
    let mut e = Env::new(cfg.clone(), Task::new(p, Randomization::none()).unwrap()).unwrap();
    e.reset(8);
    let phi0 = e.potential();
    let (rewards, last) = rollout(&mut e, 8, [0.3, 0.1, 0.0]);
    assert!(last.truncated);
    let gamma = cfg.reward.gamma;
    let discounted: f64 = rewards
        .iter()
        .enumerate()
        .map(|(t, r)| libm::pow(gamma, t as f64) * r)
        .sum();
    let expected = libm::pow(gamma, rewards.len() as f64) * e.potential() - phi0;
    assert!(
        (discounted - expected).abs() < 1e-9,
        "{discounted} vs {expected}"
    );
}

#[test]
fn history_must_cover_every_delay_the_task_can_draw() {
    let cfg = EnvConfig {
        action_repeat: 1,
        ..EnvConfig::default()
    };
    // 4 decisions of history at 1 tick per decision cover 4 ticks only.
    assert!(Env::new(cfg.clone(), Task::from_config(VehicleConfig::Realistic)).is_ok());
    let e = Env::new(cfg.clone(), Task::from_config(VehicleConfig::Storm));
    assert!(matches!(e, Err(EnvError::Invalid(_))), "6-tick delay");
    let r = Randomization {
        sensor_delay: Some((0, 12)),
        ..Randomization::none()
    };
    let task = Task::new(VehicleConfig::Easy.params(), r).unwrap();
    assert_eq!(task.max_delay_ticks(), 12);
    let mut e = Env::new(cfg.clone(), Task::from_config(VehicleConfig::Easy)).unwrap();
    assert!(e.set_task(task.clone()).is_err(), "set_task checks it too");
    let full = EnvConfig {
        history_len: MAX_HISTORY,
        ..cfg
    };
    assert!(Env::new(full, task.clone()).is_ok());
    let off = EnvConfig {
        action_repeat: 1,
        action_history: false,
        history_len: 0,
        ..EnvConfig::default()
    };
    assert!(Env::new(off, task).is_ok(), "no history, nothing to cover");
}

#[test]
fn one_tick_per_decision_shows_one_action_per_tick_of_delay() {
    let cfg = EnvConfig {
        action_repeat: 1,
        history_len: 6,
        ..EnvConfig::default()
    };
    let mut e = Env::new(cfg, Task::from_config(VehicleConfig::Storm)).unwrap();
    e.reset(3);
    for k in 0..10 {
        e.step([f64::from(k) / 10.0, 0.0, 0.0]).unwrap();
    }
    let o = e.observation();
    for k in 0..6 {
        let want = dequantize(quantize(f64::from(9 - k) / 10.0)) as f32;
        assert_eq!(o[obs::HISTORY + 3 * k as usize], want);
    }
    assert_eq!(e.episode_length(), 10);
    assert_eq!(e.sim().unwrap().tick(), 10);
}

#[test]
fn history_starts_with_the_command_that_keeps_the_engine_state() {
    let mut e = env(VehicleConfig::Realistic);
    e.reset(1);
    assert_eq!(
        e.observation()[obs::HISTORY],
        -1.0,
        "engine off: the cut-off command"
    );
    let mut p = VehicleConfig::Easy.params();
    p.sensors.delay_ticks = 3;
    let mut e = Env::new(
        EnvConfig::default(),
        Task::new(p, Randomization::none()).unwrap(),
    )
    .unwrap();
    e.reset(1);
    let level = e.sim().unwrap().initial_throttle_level();
    assert_eq!(e.observation()[obs::HISTORY], dequantize(level) as f32);
    assert!(e.observation()[obs::HISTORY] > -0.9, "keeps the engine on");
}

#[test]
fn bad_buffers_are_errors_not_panics() {
    let mut s = slot(true);
    let dim = s.env().obs_dim();
    let mut short = vec![0.0; dim - 1];
    let mut ok = vec![0.0; dim];
    assert_eq!(
        s.reset(None, &mut short),
        Err(EnvError::BadBuffer {
            expected: dim,
            got: dim - 1
        })
    );
    s.reset(None, &mut ok).unwrap();
    let tick = s.env().sim().unwrap().tick();
    let mut ok2 = vec![0.0; dim];
    assert!(s.step([0.0; 3], &mut ok, &mut short).is_err());
    assert!(s.step([0.0; 3], &mut short, &mut ok2).is_err());
    assert_eq!(s.env().sim().unwrap().tick(), tick, "nothing was stepped");
    assert!(s.env().observe(&mut short).is_err());
}

#[test]
fn slot_autoreset_after_a_crash_and_after_set_task() {
    let task = Task::from_config(VehicleConfig::Realistic);
    let mut s = EnvSlot::new(Env::new(EnvConfig::default(), task).unwrap(), 5, 0, true);
    let dim = s.env().obs_dim();
    let (mut o, mut t) = (vec![0.0; dim], vec![0.0; dim]);
    s.reset(None, &mut o).unwrap();
    s.env_mut()
        .set_task(Task::from_config(VehicleConfig::Easy))
        .unwrap();
    loop {
        let st = s.step([-1.0, 0.0, 0.0], &mut o, &mut t).unwrap();
        if st.terminated {
            assert!(matches!(
                Outcome::from_code(st.outcome),
                Some(Outcome::Crashed(_))
            ));
            assert_eq!(
                t[obs::ENGINE_ON],
                0.0,
                "terminal obs of the crashed episode"
            );
            assert_eq!(o[obs::ENGINE_ON], 1.0, "next episode is easy: engine on");
            assert_eq!(s.env().seed(), Some(episode_seed(5, 0, 1)));
            break;
        }
    }
    s.set_base_seed(99);
    s.reset(None, &mut o).unwrap();
    assert_eq!(s.env().seed(), Some(episode_seed(99, 0, 0)));
}

#[test]
fn delay_randomization_never_shifts_the_other_draws() {
    let mut with = Randomization {
        dry_mass: Range::new(0.9, 1.1),
        propellant: Range::new(0.8, 1.2),
        wind: Range::new(0.5, 1.5),
        thrust_time_constant: Range::new(0.8, 1.25),
        gimbal_time_constant: Range::new(0.8, 1.25),
        sensor_noise: Range::new(0.5, 2.0),
        sensor_delay: Some((0, 6)),
    };
    let a = Task::new(VehicleConfig::Easy.params(), with.clone()).unwrap();
    with.sensor_delay = None;
    let b = Task::new(VehicleConfig::Easy.params(), with).unwrap();
    for seed in 0..50 {
        let (mut pa, pb) = (a.episode_params(seed), b.episode_params(seed));
        pa.sensors.delay_ticks = pb.sensors.delay_ticks;
        assert_eq!(pa, pb, "only the delay differs");
    }
}

#[test]
fn every_factor_stays_in_its_range_on_every_config() {
    let r = Randomization {
        dry_mass: Range::new(0.8, 1.2),
        propellant: Range::new(0.7, 1.3),
        wind: Range::new(0.0, 2.0),
        thrust_time_constant: Range::new(0.5, 2.0),
        gimbal_time_constant: Range::new(0.5, 2.0),
        sensor_noise: Range::new(0.0, 3.0),
        sensor_delay: Some((0, 12)),
    };
    let within = |k: f64, range: Range| k >= range.min - 1e-12 && k <= range.max + 1e-12;
    for config in VehicleConfig::ALL {
        let base = config.params();
        let task = Task::new(base.clone(), r.clone()).unwrap();
        for seed in 0..300 {
            let p = task.episode_params(seed);
            p.validate().unwrap();
            let v = &p.vehicle;
            let bv = &base.vehicle;
            assert!(within(v.dry_mass / bv.dry_mass, r.dry_mass));
            assert!(within(
                p.start.propellant.max / base.start.propellant.max,
                r.propellant
            ));
            assert!(within(
                v.thrust_time_constant / bv.thrust_time_constant,
                r.thrust_time_constant
            ));
            assert!(within(
                v.gimbal_time_constant / bv.gimbal_time_constant,
                r.gimbal_time_constant
            ));
            assert!(within(
                p.sensors.noise_sigma[0] / base.sensors.noise_sigma[0],
                r.sensor_noise
            ));
        }
        let cfg = EnvConfig::default();
        let mut env = Env::new(cfg, task).unwrap();
        for seed in 0..3 {
            env.reset(seed);
            env.step([0.0; 3]).unwrap();
        }
    }
}

#[test]
fn randomization_keeps_the_wind_and_fault_streams() {
    // Factors that change the dynamics but not the wind or fault draws.
    let r = Randomization {
        dry_mass: Range::new(0.9, 1.1),
        thrust_time_constant: Range::new(0.8, 1.2),
        gimbal_time_constant: Range::new(0.8, 1.2),
        ..Randomization::none()
    };
    let base = Scenario::ThrustLoss.params(VehicleConfig::Realistic);
    let mut a = Env::new(
        EnvConfig::default(),
        Task::new(base.clone(), Randomization::none()).unwrap(),
    )
    .unwrap();
    let mut b = Env::new(EnvConfig::default(), Task::new(base, r).unwrap()).unwrap();
    for seed in 0..5 {
        a.reset(seed);
        b.reset(seed);
        let (sa, sb) = (a.sim().unwrap().state(), b.sim().unwrap().state());
        assert_eq!(sa.omega, sb.omega);
        // Stored as total mass − dry mass, so only equal to rounding.
        assert!((a.sim().unwrap().propellant() - b.sim().unwrap().propellant()).abs() < 1e-9);
        for _ in 0..200 {
            let action = [0.2, 0.0, 0.0];
            let (da, db) = (a.step(action).unwrap(), b.step(action).unwrap());
            assert_eq!(a.sim().unwrap().wind(), b.sim().unwrap().wind());
            assert_eq!(
                a.sim().unwrap().thrust_factor(),
                b.sim().unwrap().thrust_factor()
            );
            if da.done() || db.done() {
                break;
            }
        }
    }
}

#[test]
fn extreme_states_never_give_nan() {
    let mut cfg = EnvConfig::default();
    cfg.reward.position_scale = MIN_SCALE;
    let p = VehicleConfig::Easy.params();
    let decel = cfg.reward.profile_decel(&p);
    let s = State {
        x: 1e300,
        y: 1e300,
        vx: -1e300,
        vy: 1e300,
        theta: 3.0,
        omega: 1e300,
        mass: p.vehicle.dry_mass,
        gimbal: 0.0,
        thrust: 0.0,
    };
    let phi = cfg.reward.potential(&p, decel, &s);
    assert!(phi.is_finite() && phi > -cfg.reward.max_abs_potential());
    cfg.reward.tilt_scale = 1e-310;
    assert!(cfg.validate().is_err(), "denormal scale rejected");
}

#[test]
fn hovering_never_pays_with_the_default_time_cost() {
    // With the terminal potential kept, crashing now from a state with
    // potential Φ pays γΦ − Φ − c − C; one more decision at the same Φ, then
    // the crash, pays (γΦ − Φ − c) + γ(γΦ − Φ − c − C). The default cost
    // c = (1 − γ)·C per decision leaves (1 − γ)((1 − γ)C + γ|Φ|): the crash
    // timing is neutral up to the shaping's own (1 − γ)·max|Φ|.
    let cfg = EnvConfig::default();
    let r = &cfg.reward;
    let c = r.time_weight * f64::from(cfg.action_repeat) / 60.0;
    let gamma = r.gamma;
    assert!((c - (1.0 - gamma) * r.crash_penalty).abs() < 1e-12);
    for phi in [0.0, -10.0, -49.9] {
        let step = gamma * phi - phi - c;
        let crash_now = step - r.crash_penalty;
        let postpone = step + gamma * (step - r.crash_penalty);
        let hover_forever = step / (1.0 - gamma);
        let gain = postpone - crash_now;
        let exact = (1.0 - gamma) * ((1.0 - gamma) * r.crash_penalty + gamma * -phi);
        assert!((gain - exact).abs() < 1e-9, "Φ = {phi}: {gain} vs {exact}");
        let bound = (1.0 - gamma) * (r.max_abs_potential() + (1.0 - gamma) * r.crash_penalty);
        assert!(gain <= bound);
        assert!(step < 0.0, "hovering costs");
        assert!(hover_forever < 0.0);
    }
}

#[test]
fn by_default_the_terminal_state_keeps_its_potential() {
    // The discounted shaping is γ^T·Φ(s_T) − Φ(s_0): no terminal refund, so a
    // crash far from the pad is worth less than one next to it.
    let mut cfg = EnvConfig::default();
    cfg.reward.fuel_weight = 0.0;
    cfg.reward.time_weight = 0.0;
    assert!(!cfg.reward.zero_terminal_potential);
    let mut e = Env::new(cfg.clone(), Task::from_config(VehicleConfig::Realistic)).unwrap();
    e.reset(2);
    let phi0 = e.potential();
    let (rewards, last) = rollout(&mut e, 2, [1.0, 0.5, 0.0]);
    assert!(last.terminated);
    let phi_t = e.potential();
    assert!(phi_t < -1.0, "terminal potential kept: {phi_t}");
    let gamma = cfg.reward.gamma;
    let discounted: f64 = rewards
        .iter()
        .enumerate()
        .map(|(t, r)| libm::pow(gamma, t as f64) * r)
        .sum();
    let n = rewards.len() as f64;
    let expected =
        libm::pow(gamma, n) * phi_t - phi0 - libm::pow(gamma, n - 1.0) * cfg.reward.crash_penalty;
    assert!(
        (discounted - expected).abs() < 1e-6,
        "{discounted} vs {expected}"
    );
}
