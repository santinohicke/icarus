//! Development tool: prints one PID flight every 30 ticks.
//!
//!     cargo run -p icarus-core --example pid_trace -- <config> <seed>

use icarus_core::{Controller, Controls, Observation, Pid, Sim, VehicleConfig};

fn main() {
    let mut args = std::env::args().skip(1);
    let config = VehicleConfig::from_name(&args.next().unwrap_or_else(|| "training".into()))
        .expect("config");
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut sim = Sim::from_config(config, seed);
    let mut pid = Pid::new(sim.params());
    let t = |sim: &Sim| sim.telemetry();
    while sim.outcome().is_none() {
        let q = pid.act_quantized(&Observation::from_sim(&sim));
        if sim.tick() % 30 == 0 || t(&sim)[13] < 15.0 {
            let tm = t(&sim);
            println!(
                "t={:5.2} clr={:7.2} x={:7.2} vx={:6.2} vy={:7.2} th={:6.2}° thr={:4.2} on={} q={:?}",
                tm[0],
                tm[13],
                tm[2],
                tm[4],
                tm[5],
                tm[6].to_degrees(),
                tm[31],
                tm[14],
                q
            );
        }
        sim.step(Controls::from_quantized(q));
    }
    let tm = t(&sim);
    println!(
        "{:?} contact vx={:.2} vy={:.2} score {}",
        sim.outcome(),
        tm[32],
        tm[33],
        sim.score()
    );
}
