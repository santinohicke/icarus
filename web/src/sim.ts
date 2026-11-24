// Typed wrapper around the WebAssembly simulation.

import init, {
  WasmReplay,
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

/**
 * Loads and instantiates the WASM module (once). `bytes` replaces the fetch
 * of the .wasm file (tests under Node).
 */
export function loadWasm(bytes?: BufferSource): Promise<void> {
  loading ??= (bytes === undefined ? init() : init({ module_or_path: bytes })).then(() => {
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

/** What the renderer and the HUD need from a live run or a replay. */
export interface FlightView {
  readonly config: Config;
  readonly seed: bigint;
  readonly info: VehicleInfo;
  readonly state: Telemetry;
  readonly previousState: Telemetry;
  readonly outcome: number;
  readonly outcomeName: string;
  readonly score: number;
}

function requireIndices(): [Record<TelemetryKey, number>, Record<InfoKey, number>] {
  if (telemetryIndex === null || infoIndex === null) {
    throw new Error("loadWasm() must complete before creating a simulation");
  }
  return [telemetryIndex, infoIndex];
}

/** One recorded run. Call `dispose()` when done to free WASM memory. */
export class Simulation implements FlightView {
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

/** Plays a replay string back tick by tick. Call `dispose()` when done. */
export class ReplayPlayer implements FlightView {
  readonly config: Config;
  readonly seed: bigint;
  readonly info: VehicleInfo;
  readonly totalTicks: number;
  private readonly player: WasmReplay;
  private readonly telemetryIdx: Record<TelemetryKey, number>;
  private current: Telemetry;
  private previous: Telemetry;

  /** Throws if the string is not a valid replay for this build. */
  constructor(encoded: string) {
    const [tIdx, iIdx] = requireIndices();
    this.telemetryIdx = tIdx;
    const player = new WasmReplay(encoded);
    try {
      const config = player.config();
      if (!isConfig(config)) throw new Error(`replay uses an unknown configuration ${config}`);
      this.config = config;
      this.seed = player.seed();
      this.totalTicks = player.totalTicks();
      this.info = toRecord(player.vehicleInfo(), iIdx);
      this.current = toRecord(player.telemetry(), tIdx);
    } catch (error) {
      player.free();
      throw error;
    }
    this.player = player;
    this.previous = this.current;
  }

  /** Plays one recorded tick. Returns true once the playback is over. */
  step(): boolean {
    if (this.player.isDone()) {
      this.previous = this.current;
      return true;
    }
    this.previous = this.current;
    const done = this.player.step();
    this.current = toRecord(this.player.telemetry(), this.telemetryIdx);
    return done;
  }

  get done(): boolean {
    return this.player.isDone();
  }

  get tick(): number {
    return this.player.tick();
  }

  /** Quantized pilot inputs of the last played tick. */
  get lastInput(): readonly [number, number, number] {
    if (this.player.tick() === 0) return [this.info.initial_throttle_level, 127, 127];
    const q = this.player.lastInput();
    return [q[0] ?? 127, q[1] ?? 127, q[2] ?? 127];
  }

  get state(): Telemetry {
    return this.current;
  }

  get previousState(): Telemetry {
    return this.previous;
  }

  get outcome(): number {
    return this.player.outcome();
  }

  get outcomeName(): string {
    return wasmOutcomeName(this.player.outcome());
  }

  get score(): number {
    return this.player.score();
  }

  dispose(): void {
    this.player.free();
  }
}
