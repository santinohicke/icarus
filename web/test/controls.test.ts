import { describe, expect, it } from "vitest";

import { InputState, Q_CENTER, Q_MAX, THROTTLE_STEP, minRunningLevel } from "../src/controls";

describe("minRunningLevel", () => {
  it("is the first level at or above the cut-off", () => {
    const q = minRunningLevel(-0.9);
    expect((q - Q_CENTER) / Q_CENTER).toBeGreaterThanOrEqual(-0.9);
    expect((q - 1 - Q_CENTER) / Q_CENTER).toBeLessThan(-0.9);
  });
});

describe("InputState", () => {
  it("is neutral with nothing pressed", () => {
    const s = new InputState(100);
    expect(s.sample()).toEqual([100, Q_CENTER, Q_CENTER]);
  });

  it("ramps the throttle per tick and clamps it", () => {
    const s = new InputState(0);
    s.press("ArrowUp");
    s.sample();
    expect(s.throttleLevel).toBe(THROTTLE_STEP);
    for (let i = 0; i < 200; i += 1) s.sample();
    expect(s.throttleLevel).toBe(Q_MAX);
    s.release("ArrowUp");
    s.press("KeyS");
    for (let i = 0; i < 200; i += 1) s.sample();
    expect(s.throttleLevel).toBe(0);
  });

  it("stops ramping down at minimum throttle: only X cuts the engine", () => {
    const min = minRunningLevel(-0.9);
    const s = new InputState(100, min);
    s.press("ArrowDown");
    for (let i = 0; i < 200; i += 1) s.sample();
    expect(s.throttleLevel).toBe(min);
    s.press("KeyX");
    expect(s.sample()[0]).toBe(0);
  });

  it("cuts and maxes the throttle on key press; both in one tick apply in order", () => {
    const s = new InputState(120);
    s.press("KeyZ");
    expect(s.sample()[0]).toBe(Q_MAX);
    s.release("KeyZ");
    s.press("KeyX");
    expect(s.sample()[0]).toBe(0);
    s.release("KeyX");
    s.press("KeyX");
    s.press("KeyZ");
    expect(s.sample()[0]).toBe(Q_MAX);
  });

  it("maps left to a positive (counter-clockwise) gimbal and right to negative", () => {
    const s = new InputState(0);
    s.press("ArrowLeft");
    expect(s.sample()[1]).toBe(Q_MAX);
    s.press("ArrowRight");
    expect(s.sample()[1]).toBe(Q_CENTER);
    s.release("ArrowLeft");
    expect(s.sample()[1]).toBe(0);
  });

  it("maps Q and E to RCS torque", () => {
    const s = new InputState(0);
    s.press("KeyQ");
    expect(s.sample()[2]).toBe(Q_MAX);
    s.release("KeyQ");
    s.press("KeyE");
    expect(s.sample()[2]).toBe(0);
  });

  it("counts a tap that starts and ends between two ticks", () => {
    const s = new InputState(0);
    s.press("ArrowLeft");
    s.release("ArrowLeft");
    expect(s.sample()[1]).toBe(Q_MAX);
    expect(s.sample()[1]).toBe(Q_CENTER); // only for one tick
  });

  it("keeps physically held keys across a reset but drops queued presses", () => {
    const s = new InputState(50);
    s.press("ArrowUp");
    s.press("KeyZ");
    s.reset(10);
    expect(s.sample()[0]).toBe(10 + THROTTLE_STEP);
  });

  it("releases everything on blur and forgets presses when paused", () => {
    const s = new InputState(50);
    s.press("ArrowUp");
    s.press("ArrowLeft");
    s.releaseAll();
    s.clearPending();
    expect(s.sample()).toEqual([50, Q_CENTER, Q_CENTER]);
    s.press("KeyX");
    s.release("KeyX");
    s.clearPending();
    expect(s.sample()[0]).toBe(50);
  });
});
