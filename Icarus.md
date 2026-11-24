# Icarus

> A rocket that learns to come home.

Icarus is a reusable-rocket landing simulator in which a reinforcement learning agent learns to land a first stage on a pad, even with gusty wind, delayed sensors and actuator failures mid-flight. The physics, the environment logic, the classical controllers and the policy inference are written once in Rust and used unchanged both for training (Python) and for playing on the web (WebAssembly), where anyone can try landing by hand and compete against the AI and against classical controllers.

---

## 1. The idea

### What it is

- **A 2D GNC simulator** (Guidance, Navigation & Control) of the final landing phase: from 1–2 km altitude down to touchdown on the pad.
- **An RL agent (PPO)** trained in that simulator on a vectorized Rust environment, and evaluated with statistical rigor against strong classical baselines: PID, LQR and MPC / convex powered-descent guidance.
- **A public website** where anyone can:
  - land by hand with keyboard, mouse or touch (mobile),
  - watch the AI and the classical controllers land in the same scenario,
  - trigger failures (stuck gimbal, reduced thrust, gusts) and see who survives,
  - follow live telemetry like in a mission control console,
  - share a landing as a link that replays it exactly.

### What it is not

It is not "LunarLander with a different skin". The differentiator is engineering, not visuals:

| Typical tutorial | Icarus |
| --- | --- |
| Toy physics (Box2D), instant thrust | RK4 physics, variable mass and inertia, gimballed engine with lag and slew-rate limit, 40% minimum throttle, cold-gas attitude thrusters; in the realistic vehicle the stage cannot hover and must perform a hoverslam |
| No aerodynamics | Nonlinear drag with altitude-dependent density and aerodynamic torque from the center of pressure |
| Perfect observations | Sensor noise and configurable observation delay |
| Fixed scenario | Gusty wind, mass, initial position and random failures every episode |
| "The agent lands" | Multi-seed evaluation with confidence intervals and ablations, against PID, LQR and MPC, including the regimes where each one wins or loses |
| Python env, one step per call | Vectorized environment in Rust: N episodes stepped in parallel per call, zero-copy NumPy arrays |
| Physics duplicated in Python and JS | A single Rust core compiled to Python and WASM, with bit-exact parity verified in CI |

### The question the project answers

> Under what conditions does an RL agent land better than a well-tuned classical controller, and under which does it not?

Expected answer (to be validated with data): classical controllers win in the nominal case (less fuel, smoother; MPC is likely the strongest), and the agent wins in nonlinear regimes and under failures it was trained on: initial tilt above 30°, non-Gaussian gusts, a stuck gimbal or reduced thrust. MPC is the honest opponent: beating only a PID would be a strawman result.

---

## 2. Physical model

### Conventions

- 2D plane, `x` horizontal (right is positive), `y` up, origin at the pad center (sea level), SI units.
- `θ` is measured counter-clockwise from vertical: `θ > 0` means the nose leans **left**. Body axis (tail → nose): `b = (−sin θ, cos θ)`.
- The engine sits at the tail, a distance `l_g` below the center of mass. A positive gimbal angle `δ` deflects the thrust so that it produces a positive (counter-clockwise) torque: thrust direction `(−sin(θ − δ), cos(θ − δ))`, torque `l_g · T · sin δ` (from `r × F` with `r = −l_g · b`).

### State

| Variable | Description | Unit |
| --- | --- | --- |
| `x`, `y` | Center-of-mass position relative to the pad | m |
| `vx`, `vy` | Velocity | m/s |
| `θ` | Tilt (0 = vertical) | rad |
| `ω` | Angular velocity | rad/s |
| `m` | Total mass (dry + propellant) | kg |
| `δ` | Actual gimbal angle (tracks the command with lag and a slew-rate limit) | rad |
| `T` | Actual thrust (tracks the command with first-order lag) | N |

Discrete state alongside it: engine on/off, ignitions left, active faults, tick counter.

Derived from `m`: center-of-mass height along the body and moment of inertia `I(m)` (parallel-axis theorem over the dry structure and the propellant), both of which change as propellant burns.

### Actions (continuous, normalized to [-1, 1])

- **Throttle:** below a cut-off threshold the engine is commanded off; above it, the command maps linearly to 40%–100% of max thrust. Lighting the engine (including the first ignition, if the engine starts off) spends one of a limited number of ignitions.
- **Gimbal:** maps to ±7°, with a slew-rate limit.
- **RCS:** cold-gas attitude thrusters near the top of the stage (the real stage has them too), giving a small pure torque of up to `τ_rcs`. Weak compared with the gimbal: useful for fine attitude control and for surviving a stuck gimbal at low throttle.

Human inputs are quantized to 255 levels per axis before entering the simulation, so a recorded input sequence replays exactly. On the keyboard, throttle is a persistent level ramped by the keys (ramping down stops at minimum throttle; only a dedicated cut-off key shuts the engine down, since an accidental cut-off can end a single-ignition run); gimbal and RCS return to zero when released.

### Vehicle configurations

Both configurations use the same airframe; they differ in engine, starting conditions and disturbances. Parameters live in code (`physics/params.rs`) and every value below is a default that scenarios can override.

| Parameter | `easy` | `realistic` |
| --- | --- | --- |
| Length / diameter | 42 m / 3.7 m | 42 m / 3.7 m |
| Dry mass | 25,000 kg | 25,000 kg |
| Initial propellant | 5,800 – 6,200 kg | 4,800 – 5,200 kg |
| Max thrust `T_max` | 500 kN | 845 kN |
| Minimum throttle | 40% | 40% |
| Isp | 282 s | 282 s |
| Thrust lag τ / gimbal lag / gimbal slew | 0.25 s / 0.1 s / 20°/s | 0.25 s / 0.1 s / 20°/s |
| RCS torque `τ_rcs` | 100 kN·m | 100 kN·m |
| Engine at start / ignitions available | on / 3 | **off** / 1 |
| Start altitude / vertical speed | 700 – 900 m / −45 to −35 m/s | 1,400 – 1,600 m / −100 to −80 m/s |
| Start horizontal offset / speed | ±100 m / ±5 m/s | ±80 m / ±4 m/s (a good boostback leaves the stage roughly aimed at the pad) |
| Pad half-width | 40 m | 25 m |
| Wind | light, no discrete gusts | mean + turbulence + discrete gusts |
| Sensor delay | 0 ticks | 3 ticks (50 ms) |
| `TWR_min` over the propellant range | 0.65 – 0.82 (can hover) | 1.14 – 1.38 (cannot hover) |
| `TWR_max` over the propellant range | 1.63 – 2.04 | 2.85 – 3.45 |

`TWR_min = 0.4 · T_max / (m · g)`. In `realistic`, `TWR_min > 1` for every reachable mass, so there is no hover equilibrium: the stage falls with the engine off and must light it once, at the right moment (hoverslam). That changes which classical controllers are valid:

| Config | Valid baselines |
| --- | --- |
| `easy` | PID, LQR around hover, MPC |
| `realistic` | PID with an ignition trigger, convex guidance + tracking controller, MPC (LQR around hover is not valid, because that equilibrium doesn't exist) |

The web offers both as difficulty levels; development and first training happen on `easy`; the main research results use `realistic`.

### Forces and effects

- Constant gravity.
- Engine thrust along the body axis deflected by the gimbal angle, with first-order dynamics (time constant τ). Gimbal torque uses the current tail-to-CoM distance `l_g(m)`.
- RCS: pure torque, `rcs · τ_rcs`.
- Propellant consumption based on specific impulse: `ṁ = −T / (Isp · g₀)`.
- Drag: `½ · ρ(y) · Cd · A(α) · |v_rel| · v_rel`, with exponential `ρ` by altitude, `v_rel` relative to the wind, and a reference area that blends between the axial and side areas with the angle of attack `α` (`A = A_axial · cos²α + A_side · sin²α`).
- Aerodynamic torque from the offset between the center of pressure and the center of mass, plus a small aerodynamic pitch damping.
- Wind: horizontal mean component plus gusts (Ornstein–Uhlenbeck turbulence, discretized exactly, and discrete "1-cosine" gusts). The wind is **sampled once per tick and held constant during the step**: RK4 never sees a discontinuity inside a step, and the RK4 stages never draw random numbers.
- RK4 integration with fixed step `dt = 1/60 s`. Discrete events (engine cut-off and ignition, faults, flameout) are applied between steps, never inside RK4 stages.
- Ground contact is detected within the step (bisection on the touchdown time, re-integrating with RK4) so landing velocities aren't off by up to one step. The contact points are the two leg tips, 9 m either side of the tail, and the nose (if the stage has tipped over). Near the ground the middle of each step is also checked, so a contact that happens and "un-happens" within one step is not missed; after a touchdown, `time` is the exact contact time.

### Episode end

| Outcome | Condition | Kind |
| --- | --- | --- |
| **Landed** | First ground contact with a leg, both leg tips within the pad half-width, contact-point velocity `|v_y| < 2 m/s` and `|v_x| < 1 m/s`, `|θ| < 5°` and `|ω| < 0.1 rad/s` (all limits strict) | terminated |
| **Crashed** (tipped over) | The nose touches the ground first | terminated |
| **Crashed** | Any other ground contact | terminated |
| **Out of propellant** | Propellant reaches zero while airborne | terminated |
| **Out of bounds** | `|x| > 3 km` or `y > 4 km` | terminated |
| **Time limit** | 120 s of simulated time | **truncated** |

Velocities are checked at the contact point (`v + ω × r`), not at the center of mass: with `l_g` around 15 m, a rotating stage hits the ground much faster than its CoM moves.

### Score (web, replays)

Computed in Rust from the final state, so a re-simulated replay always gives the same score. Landed: `1000` + up to `500` for propellant left + up to `300` for accuracy (distance to the pad center) + up to `200` for softness (vertical contact speed). Anything else: `0`.

### Determinism (bit-exact across native and WASM)

- Fixed time step and fixed order of operations; no floating-point reductions whose order can change (no parallel sums inside a single episode).
- **Every** transcendental or composite math function (`sin`, `cos`, `atan2`, `exp`, `ln`, `pow`, `powi`, `hypot`, `tanh`, …) comes from the `libm` crate on every target; `std` versions are banned in the core (`f64::powi` can even expand differently depending on optimization level). `sqrt` is safe either way: IEEE 754 requires it to be correctly rounded. The MLP uses `libm::tanhf`.
- No `mul_add` / FMA, and no compiler flags that enable fast-math-style reassociation.
- The `libm` version is pinned exactly (`=x.y.z`) and `Cargo.lock` is committed, because the golden trajectories depend on it.
- Trajectories are asserted to contain no NaN (WebAssembly NaN payloads are not deterministic).
- Custom seeded random number generator (xoshiro256++ seeded through SplitMix64), implemented in the core, with **independent streams per subsystem** (wind, sensors, faults, initial conditions). Wind and faults therefore never depend on what the controller does, so duel mode is fair: every controller faces exactly the same scenario.
- **Exception:** the MPC (`clarabel`) uses `std` math internally, so MPC trajectories are reproducible per platform but not bit-exact across native and WASM. The bit-exact claim covers the physics, the environment, the PID/LQR and the MLP policy.

---

## 3. Reinforcement learning

### Environment

The environment logic (observation construction with noise, delay and action history; reward; termination and truncation) lives in the `env` module of **`icarus-core`**, so Python and the web share it. Python never computes per-step logic.

- **Vectorized environment in Rust** (`icarus-py`): one call steps N environments and returns NumPy arrays (observations, rewards, terminated, truncated) without per-environment Python overhead. Parallelized across cores with `rayon`, releasing the GIL (each episode remains deterministic on its own). It implements the SB3 `VecEnv` interface, including `terminal_observation` and `TimeLimit.truncated` in `infos`.
- A thin single-environment `gymnasium.Env` wrapper is kept for `check_env` and debugging.
- The curriculum stays in Python: it only sets scenario parameters on reset.
- Throughput is benchmarked and reported (steps/s: Python env vs. Rust vectorized env).

**Decision rate:** the agent acts at 20 Hz (each action held for 3 physics ticks), so an episode is a few hundred decisions, not thousands. PPO uses `γ ≈ 0.995`.

**Observation:** position and velocity relative to the pad, `sin θ`, `cos θ`, `ω`, propellant fraction, actual gimbal and thrust, engine on/off, ignitions left, the sensor delay `k`, and the **last `k_max` actions** (zero-padded when `k < k_max`). The width is fixed, as an MLP needs, while the delay is randomized per episode; without the action history a delayed observation is not Markov and an MLP can't compensate. Noise and delay are configurable.

**Reward:** potential-based shaping `F = γ·Φ(s') − Φ(s)` with `Φ ≤ 0` built from distance, velocity and tilt toward the target and `Φ(terminal) = 0`, using the same `γ` as PPO so the optimal policy is preserved. The crash penalty is larger than `max |Φ|` so that crashing early is never a shortcut; at truncation the bootstrap uses `Φ(s')`, not 0. Plus +100 for landing, −100 for crashing and a small penalty for propellant used.

### Training

- **Algorithm:** PPO (Stable-Baselines3), small MLP (2 layers of 64 units, `tanh`, SB3's default; 128 as an ablation).
- **Observation normalization:** `VecNormalize` statistics are frozen at the end of training and exported **together with the weights**, with `epsilon` and `clip_obs`. Inference applies `clip((x − mean) / sqrt(var + eps), ±clip_obs)`, uses the deterministic action (the mean) and clips it to [-1, 1].
- **Curriculum:** start on `easy`, close and nearly at rest; then higher, faster, more tilted and with more wind; then `realistic`.
- **Domain randomization:** mass, propellant, wind, sensor delay and engine τ change every episode.
- **Failures:** in a share of the episodes, the gimbal sticks at a small angle or max thrust drops 20% mid-trajectory.
- **Reproducibility:** fixed seeds, versioned YAML config, metric logging in TensorBoard.

### Experiments and ablations

- **10 training seeds** per configuration (5 at the very least). Scenarios are the "tasks" of `rliable`'s runs × tasks matrix: IQM with stratified bootstrap 95% confidence intervals, plus the probability of improvement between methods. Classical baselines have no training seeds, so their success rates get per-episode Wilson confidence intervals.
- **Ablations:**
  - curriculum on / off,
  - domain randomization on / off,
  - action history in the observation on / off (under sensor delay),
  - MLP vs. recurrent policy (LSTM, `sb3-contrib` RecurrentPPO),
  - trained with failures vs. without failures, evaluated on failures.

### Baseline controllers (Rust, `control` module of `icarus-core`)

Implemented once in Rust so that training, evaluation and the web all run the identical controller (no TypeScript copy of the PID).

| Controller | Role |
| --- | --- |
| Cascaded PID | Simple baseline, built first: position → desired angle → gimbal; vertical velocity → throttle. On `realistic` it needs an ignition trigger (light the engine when the stopping distance at ~80% throttle reaches the altitude), since it cannot hover. Validated on `easy` first |
| LQR | Optimal baseline around hover, `easy` config only. Gains computed offline (Python, `scipy.linalg.solve_continuous_are`) and stored in config |
| Convex guidance + tracking | Fuel-optimal powered-descent trajectory solved as a convex problem (second-order cone program, lossless convexification), tracked by a feedback controller. Valid for `realistic` |
| MPC | Re-solves the convex problem in a receding horizon with `clarabel` (pure-Rust conic solver), behind an `mpc` Cargo feature that is off by default, so the solver stays out of the WASM bundle until it is needed. The strongest classical baseline. Solves on a tick-based schedule (every N ticks, fixed apply latency, iteration cap), never on wall-clock deadlines |
| Human | Scores from the website |
| Residual RL (optional) | The agent learns a correction on top of the PID or MPC instead of controlling everything |

**Order of work:** PID → agent → LQR / guidance / MPC. The MPC never blocks the agent, but a strong baseline **must** be finished before the results are written up: the claim "beats a strong classical controller" depends on it. Time-boxed fallback: if the MPC isn't working after its stage, the report uses convex guidance + tracking as the strong baseline and states the limitation explicitly.

### Evaluation

A fixed battery of scenarios with known seeds, run identically for every controller:

| Scenario | What it tests |
| --- | --- |
| Nominal | Easy case: efficiency and smoothness |
| Initial tilt > 30° | Nonlinear regime |
| Strong gusts | Non-Gaussian disturbances |
| Delayed sensors | Robustness to latency |
| Gimbal stuck at 1° | Actuator failure: uncompensated, it spins the stage; survivable only by keeping throttle low enough for the RCS to cancel the torque |
| Thrust −20% | Engine failure |
| Out-of-distribution | Failure magnitudes and wind beyond the training ranges (tests generalization, not memorization) |

**Metrics:** success rate (with confidence intervals), contact velocity, position error, propellant used and compute time per step.

---

## 4. The website

- **Manual mode:** the user lands with the keyboard (throttle, gimbal, RCS), the mouse, or **touch controls** on mobile (many visitors will open the link from a phone).
- **AI mode:** the policy runs in the browser through the Rust/WASM module (hand-written MLP, see §5), deciding at 20 Hz over the 60 Hz simulation.
- **Duel mode:** human, AI and classical controllers in the same scenario with the same seed, side by side.
- **Difficulty:** `easy` (can hover) and `realistic` (hoverslam) vehicle configurations.
- **Failures on demand:** buttons to stick the gimbal, cut thrust or add gusts mid-flight (recorded as events in the replay).
- **Live telemetry:** altitude vs. vertical velocity, tilt, thrust, gimbal and propellant, with a mission-control console look.
- **Fixed-timestep loop:** the simulation advances in fixed 1/60 s ticks using an accumulator, decoupled from `requestAnimationFrame`, so it behaves identically on 60, 120 and 144 Hz screens. Catch-up is capped (at most a few ticks per frame) and the simulation pauses when the tab is hidden. Player inputs are recorded **per tick**, which makes every run replayable.
- **Replays and high scores:** high scores are kept in `localStorage`. Every run can be exported/imported as a replay and shared as a link with the replay encoded in the URL hash. A replay holds a format version, a physics version, the vehicle config, the seed, the tick count, the per-tick quantized inputs and the injected events. Inputs are run-length encoded (keyboard input changes rarely, so a one-minute run is small) and base64url-encoded; analog mouse/touch input is additionally delta-encoded, and compressed with `CompressionStream` if needed. The link carries the inputs, not the score: the browser re-simulates them, so the score proves itself. A link made with a different physics version is rejected instead of silently replaying differently.
- **Classical controllers in the browser:** PID runs live. MPC runs live only if it is fast enough in single-threaded WASM; otherwise it runs at a lower tick-based rate, or stays in native evaluation and the report while web duels use PID vs. AI.
- **Analytics:** privacy-friendly, cookie-free analytics (Cloudflare Web Analytics) to count visitors. This is where the "N users" figure for the CV comes from.
- **No backend:** the site is static (Cloudflare Pages, icarus-ai.pages.dev). An online leaderboard server is an optional last stage.

### WASM boundary

One `Sim` object per running simulation. JS calls `step(...)` once per tick with quantized inputs and reads the whole state back as a single `Float64Array` view (no per-field calls, no JSON per frame). The `wasm-bindgen` CLI version is pinned to exactly the `wasm-bindgen` crate version in `Cargo.lock`.

---

## 5. Stack

| Layer | Technology | Why |
| --- | --- | --- |
| Core | Rust (`icarus-core`): `physics`, `env`, `control`, `policy` and `replay` modules, `libm` (pinned), no heavy dependencies, `#![forbid(unsafe_code)]` | Fast, deterministic, compiles to native and WASM; one crate until there is a real reason to split |
| MPC solver | `clarabel`, behind the `mpc` Cargo feature (off by default) | Pure-Rust conic solver; kept out of the default build. Build the WASM crate with `-p icarus-wasm` so workspace feature unification never pulls it in |
| Policy inference | Hand-written MLP forward pass in Rust (`policy` module), weights loaded as a flat `f32` array | Zero extra dependencies and almost zero WASM size; the same inference code runs in Python and the browser |
| Python binding | `pyo3` + `maturin`, `numpy` crate, `rayon` | Vectorized env with zero-copy arrays and multi-core stepping |
| Web binding | `wasm-bindgen` (CLI pinned to the crate version) | The same core runs in the browser |
| RL environment | Python 3.11+, Gymnasium, NumPy | Industry standard |
| Training | Stable-Baselines3 (PPO), `sb3-contrib` (RecurrentPPO), PyTorch, TensorBoard | Proven implementation; the focus is on the environment, the baselines and the evaluation |
| Statistics | `rliable`, Wilson intervals | IQM, stratified bootstrap CIs and probability of improvement across seeds |
| Model export | Flat little-endian `f32` weights + JSON manifest (layer shapes, activation, normalization mean/var/eps/clip) | Trivial to load in Rust; no ONNX runtime needed for a 2-layer MLP |
| Frontend | Vite + TypeScript, Canvas 2D | Lightweight, no heavy framework |
| Telemetry charts | uPlot | Very fast real-time charts |
| Analytics | Cloudflare Web Analytics | Privacy-friendly usage numbers |
| Tests | `cargo test`, `pytest`, Vitest | One test level per layer |
| CI/CD | GitHub Actions | Automated tests, parity, build and deploy |
| Deploy | Cloudflare Pages (static) | Free and serverless, built-in Web Analytics |
| Leaderboard server (optional, last) | Rust (Axum on Fly.io, or Cloudflare `workers-rs`) | Re-simulates submitted replays with `icarus-core` |

---

## 6. Architecture

```text
        ┌──────────────────────────────────────────────────────────────┐
        │  icarus-core (Rust, libm)                                    │
        │  rng:     seeded generator, independent streams              │
        │  physics: state · params · RK4 · wind · faults · sensors     │
        │  sim:     one tick = events + one RK4 step + touchdown check │
        │  env:     observation · reward · termination (stage 4)       │
        │  control: PID · LQR · guidance · MPC (feature "mpc")         │
        │  policy:  MLP inference + observation normalization          │
        │  replay:  per-tick inputs · events · encoding                │
        └──────────────┬─────────────────────────────┬─────────────────┘
          pyo3/maturin │                wasm-bindgen │
                       ▼                             ▼
     ┌────────────────────────────┐   ┌──────────────────────────────────┐
     │ icarus-py                  │   │ icarus-wasm (.wasm)              │
     │ vectorized env (rayon)     │   └────────────────┬─────────────────┘
     └─────────┬──────────────────┘                    ▼
               ▼                       ┌─────────────────────────────────┐
  ┌───────────────────────────────┐    │  web/ (Vite + TS)               │
  │  training/ (Python)           │    │  render · controls · telemetry  │
  │  PPO · curriculum · ablations │    │  fixed-timestep loop · replays  │
  │  evaluation · rliable · export│    └─────────────────────────────────┘
  └──────────────┬────────────────┘                    ▲
                 └── f32 weights + normalization ──────┘
```

The parity test runs the same seed and the same input sequence through the native module and the `.wasm` (under Node), and requires the trajectories to be identical bit for bit.

---

## 7. Repository structure

```text
icarus/
├── README.md
├── LICENSE
├── Cargo.toml                    # Rust workspace
├── Cargo.lock                    # committed: goldens depend on exact versions
├── crates/
│   ├── icarus-core/              # one crate, no platform dependencies
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── rng.rs            # SplitMix64 + xoshiro256++, independent streams
│   │   │   ├── physics/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── params.rs     # vehicle parameters, `easy` and `realistic`
│   │   │   │   ├── state.rs      # State, Controls (normalized actions)
│   │   │   │   ├── dynamics.rs   # forces, torques, derivatives, I(m), CoM(m)
│   │   │   │   ├── integrator.rs # fixed-step RK4
│   │   │   │   ├── wind.rs       # mean + Ornstein–Uhlenbeck + 1-cosine gusts
│   │   │   │   ├── faults.rs     # stuck gimbal, thrust loss
│   │   │   │   ├── sensors.rs    # observation noise and delay
│   │   │   │   └── scenario.rs   # initial conditions, outcomes, score
│   │   │   ├── sim.rs            # the simulation loop: one tick = one step
│   │   │   ├── replay.rs         # per-tick inputs, events, RLE + base64url
│   │   │   ├── env.rs            # observation, reward, termination (stage 4)
│   │   │   ├── control/          # pid.rs, lqr.rs, guidance.rs, mpc.rs (stages 3 and 6)
│   │   │   └── policy.rs         # MLP forward pass + normalization (stage 5)
│   │   └── tests/
│   │       ├── convergence.rs    # halving dt reduces the error ~16x (4th order, smooth case)
│   │       ├── analytic.rs       # matches closed-form solutions
│   │       └── determinism.rs    # same seed, same trajectory; streams are independent
│   ├── icarus-py/                # pyo3 binding, built with maturin
│   │   ├── Cargo.toml
│   │   ├── pyproject.toml
│   │   └── src/lib.rs            # vec_env.rs arrives in stage 4
│   └── icarus-wasm/              # wasm-bindgen binding
│       ├── Cargo.toml
│       └── src/lib.rs
├── training/
│   ├── pyproject.toml
│   ├── icarus_env/
│   │   ├── env.py                # single-env gymnasium wrapper (debug, check_env)
│   │   ├── vec_env.py            # SB3 VecEnv adapter over the Rust vectorized env
│   │   └── curriculum.py         # sets scenario parameters on reset
│   ├── tools/
│   │   └── design_lqr.py         # computes LQR gains with scipy
│   ├── configs/
│   │   ├── ppo_base.yaml
│   │   └── ablations/*.yaml
│   ├── train.py
│   ├── run_ablations.py          # multiple seeds × configurations
│   ├── evaluate.py               # scenario battery for every controller
│   ├── analyze.py                # rliable: IQM, confidence intervals, plots
│   ├── export_policy.py          # f32 weights + JSON manifest
│   └── tests/
│       ├── test_sim_api.py       # Python binding behaves like the Rust core
│       ├── test_env.py           # complies with the Gymnasium API
│       ├── test_vec_env.py       # vectorized env == N single envs
│       ├── test_policy_parity.py # Rust inference == PyTorch (within float tolerance)
│       └── test_parity.py        # compares against golden trajectories
├── web/
│   ├── package.json
│   ├── index.html
│   ├── public/
│   │   └── models/               # agent.bin (f32 weights) + agent.json (manifest)
│   └── src/
│       ├── main.ts
│       ├── loop.ts               # fixed-timestep accumulator
│       ├── sim.ts                # .wasm wrapper
│       ├── controls.ts           # keyboard, mouse and touch
│       ├── policies.ts           # human (keyboard/touch), AI and PID (via .wasm)
│       ├── render.ts             # rocket, pad, flame, wind
│       ├── hud.ts                # text HUD; telemetry.ts (uPlot charts), touch.ts, scores.ts, share.ts
│       ├── replay.ts             # export/import, URL-hash links
│       └── scores.ts             # localStorage high scores
├── parity/
│   ├── generate_golden.py        # reference trajectories from the native module
│   ├── golden/*.json             # states stored as hex strings of the f64 bits + trajectory hash
│   └── verify_wasm.mjs           # runs the .wasm in Node and compares bits
├── reports/
│   └── results.md                # evaluation tables, ablations and curves
├── server/                       # optional, last stage: Axum leaderboard
└── .github/workflows/
    ├── ci.yml                    # cargo test + clippy + fmt, pytest, parity, web build
    └── deploy.yml                # publishes the site to Cloudflare Pages
```

---

## 8. Quality and CI

- **Rust:** `cargo fmt`, `cargo clippy -D warnings`; integrator convergence-order test (on smooth dynamics: no slew limit, no throttle cut-off, no wind, no clamps, which would lower the observed order) and analytic-solution tests; determinism and RNG-stream independence tests.
- **Python:** `pytest`, Gymnasium API check (`check_env`), `ruff`; the vectorized env must produce exactly the same results as N single environments.
- **Physics parity:** on every PR, the trajectories of the native module (on Linux x86_64 and macOS arm64) and of the `.wasm` must match the golden trajectories bit for bit. States are stored as hex strings of the `f64` bit patterns (JSON numbers can't hold 64-bit integers exactly, and decimal floats can lose precision when serialized) and compared as `BigInt` in Node. Trajectories must contain no NaN.
- **Policy parity:** the Rust MLP output must match PyTorch on a fixed set of observations (within float32 tolerance, e.g. 1e-5), including normalization.
- **Frame-rate independence:** a test drives the real fixed-timestep loop with a fake frame scheduler at 30–240 Hz, feeding the same per-tick inputs, and checks that the WebAssembly trajectory is bit-identical and runs 60 ticks per simulated second at every rate.
- **Playability:** a keyboard-only autopilot (the same key semantics as a player) must keep landing both vehicles on most seeds; a regression here means the physics became unplayable by hand.
- **Agent regression:** the published model is evaluated on the fixed battery. If the success rate drops below the previous version's confidence interval, CI fails.
- **Web:** production build and a smoke test that loads the model and simulates a full landing.

---

## 9. Roadmap

| Stage | Content | Estimated time |
| --- | --- | --- |
| 1. Core + first playable | **Week 1, vertical slice:** core dynamics (RK4, gravity, gimballed thrust, mass flow) → WASM binding → Canvas page you can land with the keyboard at 60 Hz. If landing by hand isn't fun or the physics feels artificial, fix it now: tuning the RL reward on top of bad physics is a nightmare. **Then:** the complete physics model (aerodynamics, RCS, wind, faults, sensors, RNG streams, touchdown detection, outcomes and score, both vehicle configs), replay format, Python binding, bit-exact parity test (native, Python, WASM), CI | ✅ Done |
| 2. Playable web | Telemetry (uPlot), touch controls, difficulty selector, failure buttons, local high scores, replay links in the URL hash, Cloudflare Pages deploy, Cloudflare Web Analytics | ✅ Done |
| 3. PID baseline | Cascaded PID in Rust (with the ignition trigger for `realistic`), evaluation battery; validates the environment on `easy` first | 1 week |
| 4. Agent | `env` module in the core, Rust vectorized env, PPO, curriculum, domain randomization, failures, normalization export | 2–3 weeks |
| 5. AI on the web | Hand-written MLP inference in Rust/WASM, AI mode and duel mode | 1 week |
| 6. Strong baselines | LQR (`easy`), convex guidance and MPC with `clarabel` — only once the agent lands. Time-boxed; required before stages 7–8 (fallback: convex guidance + tracking) | 1–2 weeks |
| 7. Experiments | Multi-seed runs, ablations, `rliable` analysis | 1 week |
| 8. Report | Results, curves, throughput benchmark, final README with GIF and live demo link | 3–5 days |
| 9. Optional | Residual RL, online leaderboard with a Rust re-simulation server | Open |

Total estimate: 10–12 weeks part-time (RL tuning is the least predictable part). Every stage leaves something working and publishable. After stage 5 the **web demo** is complete and shareable without the MPC; the **report** is not, and waits for stage 6.

---

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| The agent doesn't learn the full case | Curriculum from easy cases, potential-based reward, start with the `easy` vehicle and without wind or failures |
| The hoverslam is too hard (for the agent or for humans) | `easy` vehicle as a fallback and a web difficulty level; `realistic` only after `easy` works |
| Failure scenarios that no controller can survive | Failure magnitudes are chosen so that a recovery exists (e.g. a stuck gimbal of ≤ 1° that the RCS can cancel at low throttle); checked with the PID before training |
| Classical controllers beat it everywhere | That's a valid result if documented. The value is in the honest evaluation, the failure scenarios and the ablations |
| Getting stuck formulating the conic MPC (discretization, convexified thrust and angle constraints, solver convergence failures) | MPC is scheduled after the agent works and is time-boxed; start from the published lossless-convexification formulation instead of deriving one; fallback to convex guidance + tracking for the report |
| `clarabel` too slow in single-threaded WASM | MPC at a lower tick-based rate in the browser, or native-only MPC (report + evaluation) |
| Native vs. WASM differences | Pinned `libm` for every transcendental function, no FMA, fixed step and operation order, bit-level parity test in CI from stage 1 |
| Discrete gusts destabilizing fixed-step RK4 | Wind sampled once per tick and held during the step, so each RK4 step sees smooth dynamics |
| Python vs. web policy mismatch | Normalization exported with the weights; policy parity test in CI |
| The physics isn't fun to play | The playable page is the first thing built (week 1); tune with real people before training |
| Rust learning curve and size | The physics core is moderate (≈ 1,500–2,000 lines with tests); the whole Rust side, with controllers, MPC and policy, will be more like 3,000–5,000. Build it module by module with tests |

---

## 11. For the CV

> Built Icarus, a rocket-landing simulator with a Rust physics core compiled to Python and WebAssembly (bit-exact parity verified in CI), and a vectorized Rust RL environment with X× the throughput of a Python implementation. Trained a PPO agent that lands under actuator failures and severe wind gusts, evaluated across 10 seeds with confidence intervals and ablations against PID and MPC baselines (X% vs Y% success under failures). Deployed it in the browser with Rust/WASM policy inference, where N users competed against it.

Shorter variants for different roles:

- **ML / RL:** "Trained a fault-tolerant PPO landing agent with curriculum learning and domain randomization; multi-seed evaluation (IQM, 95% CIs) and ablations against an MPC baseline."
- **Systems / Rust:** "Wrote a deterministic Rust simulation core shared across Python (pyo3) and the browser (WASM), with bit-exact cross-platform parity and a multi-core vectorized environment."
- **Full-stack:** "Shipped an interactive web game (Canvas, TypeScript, WASM) with deterministic, shareable replay links, used by N players."
