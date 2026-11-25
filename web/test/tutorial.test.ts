// The tutorial flown against the real core by a simulated player: every
// lesson freezes time once, in order, and the player can land afterwards.
// The tutorial itself never flies the stage; this test's pilot stands in for
// the human.

import { readFileSync } from "node:fs";

import { beforeAll, describe, expect, it } from "vitest";

import { Q_CENTER, Q_MAX, type Quantized } from "../src/controls";
import { Simulation, type Telemetry, type VehicleInfo, loadWasm } from "../src/sim";
import { TUTORIAL_SEED, TutorialScript } from "../src/tutorial";

beforeAll(async () => {
  await loadWasm(readFileSync(new URL("../src/wasm/icarus_wasm_bg.wasm", import.meta.url)));
});

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

/** A careful human: slow descent, small leans toward the pad, upright at touchdown. */
function pilot(t: Telemetry, info: VehicleInfo): Quantized {
  const alt = Math.max(0, t.clearance);
  const theta = Math.atan2(Math.sin(t.theta), Math.cos(t.theta));
  const vyTarget = -clamp(0.6 + 0.09 * alt, 1.2, 6);
  const f = (t.mass * (9.81 + 1.5 * (vyTarget - t.vy))) / (info.max_thrust * Math.cos(theta));
  const k = (clamp(f, info.min_throttle, 1) - info.min_throttle) / (1 - info.min_throttle);
  const throttle = clamp(Math.ceil(Q_CENTER + (info.throttle_cutoff + k * (1 - info.throttle_cutoff)) * Q_CENTER), 0, Q_MAX);
  const maxLean = (alt < 4 ? 0 : 6) * (Math.PI / 180);
  const lean = clamp(0.035 * (t.vx - clamp(-0.04 * t.x, -2, 2)), -maxLean, maxLean);
  const u = clamp((3 * (lean - theta) - 3 * t.omega) / 0.1, -1, 1);
  const level = clamp(Math.round(Q_CENTER + u * Q_CENTER), 0, Q_MAX);
  return [throttle, level, level];
}

/**
 * `steady`: a beginner who leaves the throttle where it starts (falling about
 * 10 m/s) until the throttle lesson; otherwise the careful pilot throughout.
 */
function fly(seed = TUTORIAL_SEED, steady = false) {
  const sim = new Simulation("training", seed);
  const script = new TutorialScript();
  const freezes: string[] = [];
  for (let i = 0; i < 20_000 && sim.outcome === 0; i += 1) {
    if (script.frozen) {
      const lesson = script.lesson!;
      freezes.push(lesson.channel);
      // Nothing moves while frozen: the same state until the key is pressed.
      const before = sim.state.time;
      expect(script.press("KeyNotIt")).toBe(false);
      expect(script.frozen).toBe(true);
      expect(sim.state.time).toBe(before);
      expect(script.press(lesson.codes[0]!)).toBe(true);
    }
    const p = pilot(sim.state, sim.info);
    const human: Quantized = steady && freezes.length < 3 ? [sim.info.initial_throttle_level, p[1], p[2]] : p;
    sim.step(...script.command(sim.state, sim.info, human));
  }
  const result = { outcome: sim.outcomeName, freezes, done: script.done };
  sim.dispose();
  return result;
}

describe("tutorial", () => {
  it("freezes once per lesson, in order, and the player lands", () => {
    expect(fly()).toEqual({ outcome: "landed", freezes: ["rcs", "gimbal", "throttle"], done: true });
  });

  it("fits every lesson in when the player leaves the throttle alone", () => {
    expect(fly(TUTORIAL_SEED, true).freezes).toEqual(["rcs", "gimbal", "throttle"]);
  });

  it("works from other Training starts too", () => {
    for (const seed of [1n, 2n, 3n, 5n, 7n]) {
      expect(fly(seed).freezes, `seed ${seed}`).toEqual(["rcs", "gimbal", "throttle"]);
    }
  });

  it("only the asked-for key resumes time", () => {
    const sim = new Simulation("training", TUTORIAL_SEED);
    const script = new TutorialScript();
    for (let i = 0; i < 600 && !script.frozen; i += 1) sim.step(...script.command(sim.state, sim.info, [sim.info.initial_throttle_level, Q_CENTER, Q_CENTER]));
    expect(script.lesson?.channel).toBe("rcs");
    expect(script.press("KeyQ")).toBe(script.lesson?.key === "Q");
    sim.dispose();
  });
});
