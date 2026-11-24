import { describe, expect, it } from "vitest";

import { FixedLoop, MAX_TICKS_PER_FRAME, TICK_SECONDS, accumulate } from "../src/loop";
import { FakeEnv } from "./fake-env";

function ticksOver(seconds: number, hz: number): number {
  let acc = 0;
  let ticks = 0;
  for (let i = 0; i < Math.round(seconds * hz); i += 1) {
    const r = accumulate(acc, 1 / hz);
    acc = r.acc;
    ticks += r.ticks;
  }
  return ticks;
}

describe("accumulate", () => {
  it.each([30, 60, 75, 120, 144, 165, 240])("runs 60 ticks per second at %i Hz", (hz) => {
    expect(Math.abs(ticksOver(10, hz) - 600)).toBeLessThanOrEqual(1);
  });

  it("caps catch-up and drops the backlog after a long stall", () => {
    const r = accumulate(0, 2);
    expect(r.ticks).toBe(MAX_TICKS_PER_FRAME);
    expect(r.acc).toBe(0);
  });

  it("keeps a sub-tick remainder", () => {
    const r = accumulate(0, TICK_SECONDS * 1.5);
    expect(r.ticks).toBe(1);
    expect(r.acc).toBeCloseTo(TICK_SECONDS * 0.5, 12);
  });

  it("ignores negative and non-finite frame times", () => {
    expect(accumulate(0, -1)).toEqual({ ticks: 0, acc: 0 });
    expect(accumulate(0, Number.NaN)).toEqual({ ticks: 0, acc: 0 });
    expect(accumulate(0, Number.POSITIVE_INFINITY)).toEqual({ ticks: 0, acc: 0 });
  });
});

describe("FixedLoop", () => {
  function counting(env: FakeEnv): { loop: FixedLoop; ticks: () => number; frames: () => number } {
    let ticks = 0;
    let frames = 0;
    const loop = new FixedLoop(
      () => {
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
    return { loop, ticks: () => ticks, frames: () => frames };
  }

  it.each([30, 60, 120, 144, 240])("ticks at 60 Hz when the display runs at %i Hz", (hz) => {
    const env = new FakeEnv();
    const c = counting(env);
    c.loop.start();
    env.run(hz, 10);
    expect(Math.abs(c.ticks() - 600)).toBeLessThanOrEqual(1);
    expect(c.frames()).toBe(Math.round(10 * hz) + 1);
  });

  it("does not fast-forward through time spent hidden", () => {
    const env = new FakeEnv();
    const c = counting(env);
    c.loop.start();
    const end = env.run(60, 1);
    const before = c.ticks();
    env.setVisible(false);
    for (let t = end; t < end + 30_000; t += 1000) env.frame(t); // throttled background frames
    expect(c.ticks()).toBe(before);
    env.setVisible(true);
    env.run(60, 1, end + 30_000);
    expect(Math.abs(c.ticks() - before - 60)).toBeLessThanOrEqual(1);
  });

  it("stops and reports instead of freezing when a callback throws", () => {
    const env = new FakeEnv();
    const errors: unknown[] = [];
    const loop = new FixedLoop(
      () => {
        throw new Error("boom");
      },
      () => {},
      (e) => errors.push(e),
      env,
    );
    loop.start();
    env.run(60, 0.5);
    expect(errors).toHaveLength(1);
    expect(loop.running).toBe(false);
    expect(env.pending).toBe(false);
  });

  it("stop() cancels the next frame", () => {
    const env = new FakeEnv();
    const c = counting(env);
    c.loop.start();
    env.run(60, 0.1);
    c.loop.stop();
    expect(env.pending).toBe(false);
  });
});
