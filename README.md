# Icarus

> A rocket that learns to come home.

A 2D landing simulator for a reusable rocket first stage. The physics is written once in Rust and runs **bit-identically** natively, in Python (PyO3) and in the browser (WebAssembly). This is verified in CI against committed golden trajectories.

The full design, roadmap and research question are in [Icarus.md](Icarus.md).

**Play it: <https://icarus-ai.pages.dev/>**

**Status: Stages 1–3 complete** (core, playable web app, PID baseline and evaluation battery). The reinforcement-learning agent is next.

## What is in Stage 3 (the classical baseline)

- A cascaded PID autopilot in Rust (`icarus_core::control`). It sees only what a flight computer would: delayed, noisy sensors, its engine state and its propellant gauge. It uses a complementary filter, delay compensation, a disturbance observer (drag, wind, lost thrust), a descent speed profile with an ignition trigger for hoverslams, a time-to-go scheduled lateral PD and an attitude PID driving the gimbal and the RCS.
- Bit-identical decisions natively, in Python and in WebAssembly: the parity goldens include PID flights, and every checker re-runs the controller and requires the same command at every tick.
- An evaluation battery (`icarus_core::eval`, `training/evaluate.py`): nominal, 30–35° initial tilt, strong gusts, delayed sensors, a stuck gimbal, a thrust loss and out-of-distribution, with Wilson 95% confidence intervals. Results: [results/pid.md](results/pid.md).
- In the web game, every briefing has a **Watch autopilot** button.
- Gains were tuned by random search (`cargo run --release -p icarus-core --example pid_tune`) on seeds the evaluation never uses.

## What is in Stage 2 (the web app)

- Main menu: the Earth from orbit with the stage circling it, and a six-level campaign (training, easy, crosswind, realistic, storm, engine out), all open from the start (landed levels are marked and the menu suggests the next one).
- Quiet synthesized sound effects (engine, RCS, touchdown, explosion), with a mute toggle.
- White RCS cold-gas jets at the nose, on the side opposite to the push.
- Live telemetry charts (uPlot): altitude, vertical speed, thrust and tilt. Hidden by default.
- Touch controls: a throttle bar, steering and RCS buttons and an engine cut-off, switched on by the first touch.
- Local high scores per mission (localStorage, validated on load), each with a "Watch" button.
- Shareable replay links: the run travels in the URL hash (`#replay=...`, never sent to a server) and plays back at 1×/2×/4×.
- Deployed to Cloudflare Pages (with Cloudflare Web Analytics), only after the WebAssembly build matches the golden trajectories bit for bit.

## What is in Stage 1

- **`icarus-core`** (Rust):
  - RK4 rigid-body dynamics with variable mass and inertia, a gimballed engine with lag and slew limit, a 40% minimum throttle, limited ignitions and cold-gas RCS.
  - Altitude-dependent drag with angle of attack, plus center-of-pressure torque and pitch damping.
  - Wind made of a mean, Ornstein–Uhlenbeck turbulence and 1-cosine gusts.
  - Sensor noise and delay, and actuator and engine faults.
  - Exact touchdown detection with leg-tip contact-point velocity, outcomes and score.
  - Two vehicles: `easy` (can hover) and `realistic` (cannot hover; a hoverslam is required), used by six levels (training, easy, crosswind, realistic, storm, engine out).
  - Deterministic replays (seed + quantized per-tick inputs, RLE + base64url), hardened against untrusted input.
- **`icarus-wasm`**: wasm-bindgen bindings (about 125 KB of `.wasm`).
- **`icarus-py`**: the `icarus` Python module (abi3, Python 3.11+).
- **`web/`**: Vite + TypeScript + Canvas page where you land by hand. It has a fixed 60 Hz timestep independent of the display rate, a webcast-style HUD with large readouts, and a debrief that shows each touchdown value against its limit and how to fix what failed.
- **`parity/`**: golden trajectories (hex `f64` bit patterns + FNV-1a hash of every tick). They are checked against native Rust, the Python module and WebAssembly.

## Determinism, in one paragraph

All transcendental math goes through the pure-Rust `libm` crate, pinned to an exact version, and `clippy.toml` bans the `std` versions and `mul_add`. Steps are fixed, the operation order is fixed, discrete events happen between RK4 steps, and the wind is sampled once per tick. The custom xoshiro256++ RNG has one independent stream per subsystem, so wind, noise and faults never depend on the controller. The result is identical bits on x86_64, arm64 and wasm32.

## Quick start

Requirements: Rust (stable, with the `wasm32-unknown-unknown` target), Python 3.11+, Node 20.19+. On Windows you also need the Visual Studio C++ Build Tools.

Install the WebAssembly CLI once, at exactly the crate version in `Cargo.lock`:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked
```

### Play in the browser

```sh
cd web
npm install
npm run wasm      # builds crates/icarus-wasm into web/src/wasm
npm run dev       # http://localhost:5173
```

Controls:

| Key | Action |
| --- | --- |
| ↑ / ↓ | Throttle |
| ← / → | Steer (gimbal) |
| Q / E | RCS |
| X | Cut the engine |
| Space | Launch |
| Esc | Pause (resume, retry, new scenario, main menu) |

### Use it from Python

```sh
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin pytest
maturin develop --release -m crates/icarus-py/Cargo.toml
```

```python
import icarus

sim = icarus.Sim("realistic", seed=42)
while sim.step(throttle=-1.0) is None:   # engine off: free fall
    pass
print(sim.outcome, sim.score, sim.telemetry_dict()["contact_vy"])

# The PID autopilot, and the evaluation battery
sim, pid = icarus.Sim("realistic", 42), icarus.Controller("pid", "realistic")
while not sim.done:
    sim.step_quantized(*pid.act(sim))
rows = icarus.evaluate("pid", "realistic", "gimbal_stuck", list(range(100)))
```

```sh
python training/evaluate.py --controller pid --seeds 500   # writes results/pid.md and pid.json
```

## Tests

```sh
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                       # physics, convergence, analytic, determinism, replays, goldens
cd training && python -m pytest              # Python API, controllers, evaluation, golden parity
node parity/verify_wasm.mjs                  # WebAssembly matches the goldens bit for bit
cd web && npm test                           # loop, controls, frame-rate independence, playability, scores, links, replays
```

Some highlights:
- **Convergence:** the integrator converges at 4th order.
- **Analytic checks:** free fall, thrust lag and propellant use, and touchdown time all match closed-form solutions to 1e-9.
- **Replay hardening:** the replay decoder survives fuzzing.
- **Playability:** a keyboard-only autopilot, which presses the same keys a player has, must land both vehicles.

## Repository layout

```text
crates/icarus-core   physics, simulation loop, replays (Rust)
crates/icarus-wasm   WebAssembly bindings
crates/icarus-py     Python bindings (module `icarus`)
web/                 browser front-end (Vite + TypeScript)
parity/              golden trajectories and the WebAssembly verifier
training/            Python tests now; RL training and evaluation in later stages
```

## License

MIT, see [LICENSE](LICENSE).
