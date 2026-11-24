// The trajectory depends only on the per-tick inputs, never on the display
// refresh rate: the real FixedLoop, driven by a fake frame scheduler at
// different frame rates, steps the real WASM simulation through the same
// ticks and gives bit-identical results.

import { readFileSync } from "node:fs";

import { beforeAll, describe, expect, it } from "vitest";

import { FixedLoop } from "../src/loop";
import { WasmSim, initSync } from "../src/wasm/icarus_wasm.js";
import { FakeEnv } from "./fake-env";

beforeAll(() => {
  const wasm = readFileSync(new URL("../src/wasm/icarus_wasm_bg.wasm", import.meta.url));
  initSync({ module: wasm });
});

/** Inputs defined per tick, as the keyboard sampler produces them. */
function inputsAt(tick: number): [number, number, number] {
  const throttle = 100 + ((tick * 7) % 120);
  const gimbal = tick % 90 < 30 ? 254 : tick % 90 < 60 ? 0 : 127;
  const rcs = tick % 50 < 10 ? 200 : 127;
  return [throttle, gimbal, rcs];
}

const TICKS = 900;

/** Runs the loop at `hz` until TICKS ticks; returns the telemetry bits and how many frames it took. */
function runAt(hz: number): { bits: BigUint64Array; frames: number } {
  const sim = new WasmSim("easy", 123n);
  const env = new FakeEnv();
  let ticks = 0;
  let frames = 0;
  const loop = new FixedLoop(
    () => {
      if (ticks >= TICKS) return;
      const [t, g, c] = inputsAt(ticks);
      sim.step(t, g, c);
      ticks += 1;
    },
    () => {
      frames += 1;
    },
    (e) => {
      throw e;
    },
    env,
  );
  loop.start();
  let now = 0;
  while (ticks < TICKS) {
    env.frame(now);
    now += 1000 / hz;
  }
  loop.stop();
  const bits = new BigUint64Array(Float64Array.from(sim.telemetry()).buffer);
  sim.free();
  return { bits, frames };
}

describe("frame-rate independence", () => {
  it("gives bit-identical trajectories at 30, 60, 120, 144 and 240 Hz in the same simulated time", () => {
    const reference = runAt(60);
    for (const hz of [30, 120, 144, 240]) {
      const r = runAt(hz);
      expect(r.bits).toEqual(reference.bits);
      // The same 15 s of simulation took ~15 s of frames at every rate.
      expect(Math.abs(r.frames / hz - reference.frames / 60)).toBeLessThan(0.1);
    }
  });
});
