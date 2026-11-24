// A fake browser frame scheduler for driving FixedLoop in Node.

import type { LoopEnv } from "../src/loop";

export class FakeEnv implements LoopEnv {
  visible = true;
  private callback: ((now: number) => void) | null = null;
  private visibilityListeners: (() => void)[] = [];
  private nextHandle = 1;

  requestFrame(callback: (now: number) => void): number {
    this.callback = callback;
    return this.nextHandle++;
  }

  cancelFrame(): void {
    this.callback = null;
  }

  isVisible(): boolean {
    return this.visible;
  }

  onVisibilityChange(callback: () => void): void {
    this.visibilityListeners.push(callback);
  }

  setVisible(visible: boolean): void {
    this.visible = visible;
    for (const l of this.visibilityListeners) l();
  }

  /** Delivers one animation frame at time `now` (ms). */
  frame(now: number): void {
    const cb = this.callback;
    this.callback = null;
    cb?.(now);
  }

  /** Delivers frames at `hz` from `startMs` for `seconds`. Returns the end time. */
  run(hz: number, seconds: number, startMs = 0): number {
    const frames = Math.round(seconds * hz);
    let t = startMs;
    for (let i = 0; i <= frames; i += 1) {
      t = startMs + (i * 1000) / hz;
      this.frame(t);
    }
    return t;
  }

  get pending(): boolean {
    return this.callback !== null;
  }
}
