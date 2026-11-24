// Gameplay check: a simple autopilot that can only press the same keys a
// player has (through `InputState`, so throttle ramps and the gimbal is fully
// on or off) must be able to land both vehicles. If this fails, the physics
// or the scenario ranges have become unplayable by hand.

import { readFileSync } from "node:fs";

import { beforeAll, describe, expect, it } from "vitest";

import { InputState, Q_CENTER, minRunningLevel } from "../src/controls";
import { WasmSim, initSync, outcomeName, telemetryFields, vehicleInfoFields } from "../src/wasm/icarus_wasm.js";

type Rec = Record<string, number>;

let tIndex: string[] = [];
let iIndex: string[] = [];

beforeAll(() => {
  initSync({ module: readFileSync(new URL("../src/wasm/icarus_wasm_bg.wasm", import.meta.url)) });
  tIndex = telemetryFields();
  iIndex = vehicleInfoFields();
});

function rec(values: Float64Array, names: string[]): Rec {
  const out: Rec = {};
  names.forEach((n, i) => {
    out[n] = values[i] ?? Number.NaN;
  });
  return out;
}

const G = 9.81;
const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

function hold(input: InputState, code: string, on: boolean): void {
  if (on) input.press(code);
  else input.release(code);
}

/** Quantized throttle level for a thrust fraction (never below the cut-off). */
function levelFor(frac: number, info: Rec): number {
  const min = info["min_throttle"] ?? 0.4;
  const cutoff = info["throttle_cutoff"] ?? -0.9;
  const u = cutoff + ((frac - min) / (1 - min)) * (1 - cutoff);
  return clamp(Math.round(u * Q_CENTER + Q_CENTER), 14, 254);
}

function pilot(t: Rec, info: Rec, input: InputState, realistic: boolean): void {
  const alt = t["clearance"] ?? 0;
  const m = t["mass"] ?? 1;
  const vy = t["vy"] ?? 0;
  const vx = t["vx"] ?? 0;
  const x = t["x"] ?? 0;
  const theta = Math.atan2(Math.sin(t["theta"] ?? 0), Math.cos(t["theta"] ?? 0));
  const omega = t["omega"] ?? 0;
  const tmax = (info["max_thrust"] ?? 1) * (t["thrust_factor"] ?? 1);
  const engineOn = (t["engine_on"] ?? 0) > 0.5;

  // Vertical channel.
  let throttleTarget = 0;
  if (realistic) {
    const decel = (0.7 * tmax) / m - G;
    const stopping = (vy * vy) / (2 * Math.max(decel, 0.1));
    const shouldBurn = engineOn || (vy < 0 && alt - 10 <= stopping);
    if (shouldBurn) {
      const vref = -Math.sqrt(1.0 * 1.0 + 2 * 6 * Math.max(alt, 0));
      const accel = 6 * (vy < vref ? 1 : 0.5) + 2 * (vref - vy);
      throttleTarget = levelFor(clamp((m * (G + accel)) / tmax / Math.cos(theta), 0, 1), info);
    }
  } else {
    const vref = alt < 4 ? -0.8 : -clamp(0.15 * alt, 1.2, 40);
    const accel = 1.5 * (vref - vy);
    throttleTarget = levelFor(clamp((m * (G + accel)) / tmax / Math.cos(theta), 0, 1), info);
  }
  const level = input.throttleLevel;
  hold(input, "ArrowUp", throttleTarget > level + 1);
  hold(input, "ArrowDown", throttleTarget < level - 1 && level > 15);

  // Lateral channel: tilt to accelerate toward the pad, then hold vertical.
  const thrustAccel = Math.max((t["thrust"] ?? 0) / m, 1);
  // High up: steer toward the pad. Low down: only kill the sideways speed
  // (the pad is wide; arriving with lateral speed is what crashes).
  const axDesired = alt > 40 ? clamp(-0.08 * x - 0.7 * vx, -4, 4) : clamp(-0.8 * vx, -2, 2);
  const maxTilt = alt < 15 ? 0.05 : 0.2;
  const thetaDesired = engineOn ? clamp(-axDesired / thrustAccel, -maxTilt, maxTilt) : 0;
  const omegaDesired = clamp(1.0 * (thetaDesired - theta), -0.12, 0.12);
  const err = omegaDesired - omega;
  hold(input, "ArrowLeft", engineOn && err > 0.01);
  hold(input, "ArrowRight", engineOn && err < -0.01);
  hold(input, "KeyQ", err > 0.004);
  hold(input, "KeyE", err < -0.004);
}

function fly(config: "easy" | "realistic", seed: bigint): string {
  const sim = new WasmSim(config, seed);
  const info = rec(sim.vehicleInfo(), iIndex);
  const input = new InputState(info["initial_throttle_level"] ?? 0, minRunningLevel(info["throttle_cutoff"] ?? -0.9));
  let code = 0;
  for (let i = 0; i < 7200 && code === 0; i += 1) {
    pilot(rec(sim.telemetry(), tIndex), info, input, config === "realistic");
    const [th, g, r] = input.sample();
    code = sim.step(th, g, r);
  }
  const end = rec(sim.telemetry(), tIndex);
  if (process.env["ICARUS_DEBUG"]) {
    console.log(config, seed, outcomeName(code), {
      x: end["x"]?.toFixed(2), vx: end["vx"]?.toFixed(2), vy: end["vy"]?.toFixed(2),
      tilt: ((end["theta"] ?? 0) * 57.3).toFixed(2), omega: end["omega"]?.toFixed(3), fuel: end["propellant"]?.toFixed(0),
    });
  }
  sim.free();
  return outcomeName(code);
}

describe("keyboard-only autopilot", () => {
  for (const config of ["easy", "realistic"] as const) {
    it(`lands the ${config} vehicle on most seeds`, () => {
      const results: string[] = [];
      for (let s = 0n; s < 20n; s += 1n) results.push(fly(config, s));
      const landed = results.filter((r) => r === "landed").length;
      const summary = results.reduce<Record<string, number>>((acc, r) => {
        acc[r] = (acc[r] ?? 0) + 1;
        return acc;
      }, {});
      console.log(config, summary);
      // Regression floors for a crude bang-bang pilot, not targets: a
      // competent human does better. Realistic is meant to be hard.
      expect(landed).toBeGreaterThanOrEqual(config === "easy" ? 16 : 10);
    });
  }
});
