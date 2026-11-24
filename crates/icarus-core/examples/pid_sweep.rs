//! Development tool: flies the PID on every level and prints the outcomes.
//!
//!     cargo run --release -p icarus-core --example pid_sweep [seeds]

use std::collections::BTreeMap;

use icarus_core::{Controller, Controls, Observation, Pid, Sim, VehicleConfig};

fn main() {
    let seeds: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    for config in VehicleConfig::CAMPAIGN {
        let mut outcomes: BTreeMap<&str, u32> = BTreeMap::new();
        let mut fuel = 0.0;
        for seed in 0..seeds {
            let mut sim = Sim::from_config(config, seed);
            let mut pid = Pid::new(sim.params());
            while sim.outcome().is_none() {
                let q = pid.act_quantized(&Observation::from_sim(&sim));
                sim.step(Controls::from_quantized(q));
            }
            let name = sim.outcome().map_or("running", |o| o.name());
            *outcomes.entry(name).or_default() += 1;
            fuel += sim.propellant();
        }
        println!(
            "{:11} fuel left {:6.0} kg  {:?}",
            config.name(),
            fuel / seeds as f64,
            outcomes
        );
    }
}
