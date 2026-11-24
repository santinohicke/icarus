// The autopilot (the Rust PID compiled to WebAssembly) flies the live
// simulation, and its flights replay exactly.

import { readFileSync } from "node:fs";

import { beforeAll, describe, expect, it } from "vitest";

import { Autopilot, ReplayPlayer, Simulation, loadWasm } from "../src/sim";

beforeAll(async () => {
  await loadWasm(readFileSync(new URL("../src/wasm/icarus_wasm_bg.wasm", import.meta.url)));
});

function fly(config: "training" | "easy" | "realistic" | "storm" | "engine_out", seed: bigint): Simulation {
  const sim = new Simulation(config, seed);
  const pilot = new Autopilot(config);
  for (let i = 0; i < 10_000 && sim.outcome === 0; i += 1) sim.step(...pilot.act(sim));
  pilot.dispose();
  return sim;
}

describe("autopilot", () => {
  it("lands the easy vehicles", () => {
    let landed = 0;
    for (let s = 0n; s < 10n; s += 1n) {
      const sim = fly(s % 2n === 0n ? "training" : "easy", s);
      if (sim.outcomeName === "landed") landed += 1;
      sim.dispose();
    }
    expect(landed).toBeGreaterThanOrEqual(9);
  });

  it("flies every hoverslam level to an outcome", () => {
    for (const config of ["realistic", "storm", "engine_out"] as const) {
      const sim = fly(config, 3n);
      expect(sim.outcome).not.toBe(0);
      sim.dispose();
    }
  });

  it("makes the same decisions every time", () => {
    const a = fly("realistic", 5n);
    const b = fly("realistic", 5n);
    expect(a.state).toEqual(b.state);
    expect(a.outcomeName).toBe(b.outcomeName);
    a.dispose();
    b.dispose();
  });

  it("records a replay of its flight", () => {
    const sim = fly("easy", 21n);
    const player = new ReplayPlayer(sim.replay());
    while (!player.step()) {
      /* play to the end */
    }
    expect(player.state).toEqual(sim.state);
    expect(player.outcomeName).toBe(sim.outcomeName);
    player.dispose();
    sim.dispose();
  });
});
