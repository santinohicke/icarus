//! The native Rust environment reproduces the RL-environment golden hashes
//! (`parity/env_golden.json`, generated through the Python extension by
//! `parity/generate_env_golden.py`): observation, reward and termination are
//! bit-identical across bindings.

use std::fs;
use std::path::PathBuf;

use icarus_core::env::{Env, EnvConfig, Randomization, Task, apply_override};
use icarus_core::physics::params::Range;
use icarus_core::{PHYSICS_VERSION, Scenario, VehicleConfig};
use serde_json::Value;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

fn feed(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn hash_obs(mut h: u64, obs: &[f32]) -> u64 {
    for v in obs {
        h = feed(h, &v.to_bits().to_le_bytes());
    }
    h
}

fn action(k: u32) -> [f64; 3] {
    let f = |a: u32, b: u32| f64::from((a * k + b) % 201) / 100.0 - 1.0;
    [f(37, 11), f(53, 7), f(71, 3)]
}

fn range(v: &Value) -> Range {
    let a = v.as_array().expect("[min, max]");
    Range::new(a[0].as_f64().unwrap(), a[1].as_f64().unwrap())
}

fn task(spec: &Value) -> Task {
    let config = VehicleConfig::from_name(spec["config"].as_str().unwrap()).unwrap();
    let scenario = spec.get("scenario").map_or(Scenario::Nominal, |s| {
        Scenario::from_name(s.as_str().unwrap()).unwrap()
    });
    let mut params = scenario.params(config);
    if let Some(overrides) = spec.get("overrides").and_then(Value::as_object) {
        for (key, value) in overrides {
            let values: Vec<f64> = match value {
                Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap()).collect(),
                x => vec![x.as_f64().unwrap()],
            };
            apply_override(&mut params, key, &values).unwrap();
        }
    }
    let mut r = Randomization::none();
    if let Some(spec) = spec.get("randomization").and_then(Value::as_object) {
        for (key, value) in spec {
            match key.as_str() {
                "dry_mass" => r.dry_mass = range(value),
                "propellant" => r.propellant = range(value),
                "wind" => r.wind = range(value),
                "thrust_time_constant" => r.thrust_time_constant = range(value),
                "gimbal_time_constant" => r.gimbal_time_constant = range(value),
                "sensor_noise" => r.sensor_noise = range(value),
                "sensor_delay" => {
                    let a = value.as_array().unwrap();
                    let lo = usize::try_from(a[0].as_u64().unwrap()).unwrap();
                    let hi = usize::try_from(a[1].as_u64().unwrap()).unwrap();
                    r.sensor_delay = Some((lo, hi));
                }
                other => panic!("unknown randomization key {other}"),
            }
        }
    }
    Task::new(params, r).unwrap()
}

fn env_config(spec: &Value) -> EnvConfig {
    let mut c = EnvConfig::default();
    if let Some(spec) = spec.as_object() {
        for (key, value) in spec {
            match key.as_str() {
                "action_repeat" => {
                    c.action_repeat = u32::try_from(value.as_u64().unwrap()).unwrap()
                }
                "quantize_actions" => c.quantize_actions = value.as_bool().unwrap(),
                other => panic!("unsupported env_config key {other}"),
            }
        }
    }
    c
}

#[test]
fn environment_matches_the_golden_hashes() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../parity/env_golden.json");
    let golden: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(
        golden["physics_version"].as_u64(),
        Some(u64::from(PHYSICS_VERSION))
    );
    let max_steps = u32::try_from(golden["max_steps"].as_u64().unwrap()).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert!(cases.len() >= 6);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let mut env = Env::new(env_config(&case["env_config"]), task(&case["task"])).unwrap();
        env.reset(case["seed"].as_u64().unwrap());
        let mut h = hash_obs(FNV_OFFSET, &env.observation());
        let mut steps = 0;
        let mut outcome = None;
        for k in 0..max_steps {
            let step = env.step(action(k)).unwrap();
            h = hash_obs(h, &env.observation());
            h = feed(h, &step.reward.to_bits().to_le_bytes());
            h = feed(
                h,
                &[u8::from(step.terminated) | (u8::from(step.truncated) << 1)],
            );
            steps = k + 1;
            if step.done() {
                outcome = step.outcome.map(|o| o.name());
                break;
            }
        }
        assert_eq!(u64::from(steps), case["steps"].as_u64().unwrap(), "{name}");
        assert_eq!(outcome, case["outcome"].as_str(), "{name}");
        assert_eq!(
            format!("{h:016x}"),
            case["hash"].as_str().unwrap(),
            "{name}"
        );
    }
}
