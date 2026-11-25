// The tutorial against the real core: it starts frozen, already leaning and
// drifting, and exactly three taps (E, →, ↑) land the stage.

import { readFileSync } from "node:fs";

import { beforeAll, describe, expect, it } from "vitest";

import { Simulation, loadWasm } from "../src/sim";
import { TUTORIAL_SEED, TutorialScript } from "../src/tutorial";

beforeAll(async () => {
  await loadWasm(readFileSync(new URL("../src/wasm/icarus_wasm_bg.wasm", import.meta.url)));
});

const DEG = Math.PI / 180;

function fly(seed = TUTORIAL_SEED, log = false) {
  const sim = new Simulation("training", seed);
  const script = new TutorialScript(sim);
  const start = { ...sim.state };
  const presses: string[] = [];
  for (let i = 0; i < 20_000 && sim.outcome === 0; i += 1) {
    if (script.frozen) {
      const lesson = script.lesson!;
      if (log) console.log(`${lesson.channel} t=${sim.state.time.toFixed(1)} alt=${sim.state.clearance.toFixed(0)} th=${(sim.state.theta / DEG).toFixed(1)} vx=${sim.state.vx.toFixed(2)} vy=${sim.state.vy.toFixed(2)} x=${sim.state.x.toFixed(1)}`);
      // A wrong key does nothing; time stays frozen.
      expect(script.press("KeyX")).toBe(false);
      expect(script.frozen).toBe(true);
      presses.push(lesson.codes[0]!);
      script.press(lesson.codes[0]!);
    }
    sim.step(...script.command(sim.state, sim.info));
  }
  const result = { outcome: sim.outcomeName, presses, start, end: { ...sim.state } };
  if (log) console.log(result.outcome, result.end.time.toFixed(1), result.end.contact_vy, result.end.contact_vx, result.end.theta / DEG, result.end.x);
  sim.dispose();
  return result;
}

describe("tutorial", () => {
  it("starts frozen, leaning and drifting away from the pad", () => {
    const sim = new Simulation("training", TUTORIAL_SEED);
    const script = new TutorialScript(sim);
    expect(script.frozen).toBe(true);
    expect(script.lesson?.channel).toBe("rcs");
    const s = sim.state;
    expect(Math.abs(s.theta / DEG)).toBeCloseTo(6, 0);
    expect(Math.abs(s.omega / DEG)).toBeLessThan(0.5);
    expect(s.vx * Math.sign(s.x)).toBeGreaterThanOrEqual(2);
    expect(s.clearance).toBeGreaterThan(100);
    sim.dispose();
  });

  it("lands with exactly three presses: E, then →, then ↑", () => {
    const r = fly(TUTORIAL_SEED, true);
    expect(r.presses).toEqual(["KeyE", "ArrowRight", "ArrowUp"]);
    expect(r.outcome).toBe("landed");
  });

  it("lands with three presses from other Training starts too", () => {
    for (const seed of [1n, 2n, 3n, 5n, 7n]) {
      const r = fly(seed);
      expect(r.presses.length, `seed ${seed}`).toBe(3);
      expect(r.outcome, `seed ${seed}`).toBe("landed");
    }
  });
});
