// Typed wrapper around the WebAssembly simulation.

import init, {
  WasmSim,
  outcomeName as wasmOutcomeName,
  physicsVersion,
  telemetryFields,
  vehicleInfoFields,
} from "./wasm/icarus_wasm.js";

export const CONFIGS = ["easy", "realistic"] as const;
export type Config = (typeof CONFIGS)[number];

export function isConfig(value: unknown): value is Config {
  return typeof value === "string" && (CONFIGS as readonly string[]).includes(value);
}

const TELEMETRY_KEYS = [
  "time",
  "tick",
  "x",
  "y",
  "vx",
  "vy",
  "theta",
  "omega",
  "mass",
  "gimbal",
  "thrust",
  "propellant",
  "cm_height",
  "clearance",
  "engine_on",
  "ignitions_left",
  "thrust_cmd",
  "gimbal_cmd",
  "rcs_torque",
  "wind",
  "obs_x",
  "obs_y",
  "obs_vx",
  "obs_vy",
  "obs_theta",
  "obs_omega",
  "thrust_factor",
  "gimbal_stuck",
  "gimbal_stuck_angle",
  "outcome",
  "score",
  "thrust_fraction",
  "contact_vx",
  "contact_vy",
  "tail_x",
  "pad_margin",
] as const;
export type TelemetryKey = (typeof TELEMETRY_KEYS)[number];
export type Telemetry = Readonly<Record<TelemetryKey, number>>;

const INFO_KEYS = [
  "length",
  "diameter",
  "leg_half_span",
  "pad_half_width",
  "gimbal_max",
  "max_thrust",
  "min_throttle",
  "throttle_cutoff",
  "rcs_torque",
  "dry_mass",
  "initial_propellant",
  "dt",
  "max_vertical_speed",
  "max_horizontal_speed",
  "max_tilt",
  "max_angular_rate",
  "max_ticks",
  "initial_throttle_level",
] as const;
export type InfoKey = (typeof INFO_KEYS)[number];
export type VehicleInfo = Readonly<Record<InfoKey, number>>;

/** Index of every expected key in the array the WASM module returns. */
function indexOf<K extends string>(expected: readonly K[], actual: string[]): Record<K, number> {
  const out = {} as Record<K, number>;
  for (const key of expected) {
    const i = actual.indexOf(key);
    if (i < 0) {
      throw new Error(`WASM module is missing field "${key}": rebuild it with "npm run wasm"`);
    }
    out[key] = i;
  }
  return out;
}

function toRecord<K extends string>(values: Float64Array, index: Record<K, number>): Record<K, number> {
  const out = {} as Record<K, number>;
  for (const key in index) {
    out[key] = values[index[key]] ?? Number.NaN;
  }
  return out;
}

let telemetryIndex: Record<TelemetryKey, number> | null = null;
let infoIndex: Record<InfoKey, number> | null = null;
let loading: Promise<void> | null = null;

/** Loads and instantiates the WASM module (once). */
export function loadWasm(): Promise<void> {
  loading ??= init().then(() => {
    telemetryIndex = indexOf(TELEMETRY_KEYS, telemetryFields());
    infoIndex = indexOf(INFO_KEYS, vehicleInfoFields());
  });
  return loading;
}

export function outcomeName(code: number): string {
  return wasmOutcomeName(code);
}

export function getPhysicsVersion(): number {
  return physicsVersion();
}

/** One recorded run. Call `dispose()` when done to free WASM memory. */
export class Simulation {
  readonly config: Config;
  readonly seed: bigint;
  readonly info: VehicleInfo;
  private readonly sim: WasmSim;
  private current: Telemetry;
  private previous: Telemetry;

  constructor(config: Config, seed: bigint) {
    if (telemetryIndex === null || infoIndex === null) {
      throw new Error("loadWasm() must complete before creating a Simulation");
    }
    this.config = config;
    this.seed = seed;
    this.sim = new WasmSim(config, seed);
    this.info = toRecord(this.sim.vehicleInfo(), infoIndex);
    this.current = this.read();
    this.previous = this.current;
  }

  private read(): Telemetry {
    // telemetryIndex is set once loadWasm() resolved (checked in the constructor).
    return toRecord(this.sim.telemetry(), telemetryIndex as Record<TelemetryKey, number>);
  }

  /** Advances one tick with quantized inputs. Returns the outcome code (0 while running). */
  step(throttle: number, gimbal: number, rcs: number): number {
    // wasm-bindgen wraps out-of-range u8 values (256 → 0): clamp here.
    const level = (v: number): number => (Number.isFinite(v) ? Math.min(254, Math.max(0, Math.round(v))) : 127);
    this.previous = this.current;
    const code = this.sim.step(level(throttle), level(gimbal), level(rcs));
    this.current = this.read();
    return code;
  }

  injectGimbalStuck(angle: number): boolean {
    return this.sim.injectGimbalStuck(angle);
  }

  injectThrustLoss(factor: number): boolean {
    return this.sim.injectThrustLoss(factor);
  }

  injectGust(amplitude: number, duration: number): boolean {
    return this.sim.injectGust(amplitude, duration);
  }

  get state(): Telemetry {
    return this.current;
  }

  get previousState(): Telemetry {
    return this.previous;
  }

  get outcome(): number {
    return this.sim.outcome();
  }

  get outcomeName(): string {
    return this.sim.outcomeName();
  }

  get score(): number {
    return this.sim.score();
  }

  replay(): string {
    return this.sim.replay();
  }

  dispose(): void {
    this.sim.free();
  }
}
