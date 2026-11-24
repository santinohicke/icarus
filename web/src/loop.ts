// Fixed-timestep loop: the simulation advances in fixed 1/60 s ticks,
// independent of the display refresh rate.

export const TICK_SECONDS = 1 / 60;

/** Most ticks run in a single frame; a longer backlog is dropped. */
export const MAX_TICKS_PER_FRAME = 5;

export interface Accumulated {
  /** Ticks to run this frame. */
  ticks: number;
  /** Leftover time carried to the next frame (s). */
  acc: number;
}

/**
 * Adds a frame's elapsed time to the accumulator and returns how many ticks
 * to run. Pure, so it can be tested without a browser.
 */
export function accumulate(acc: number, frameSeconds: number): Accumulated {
  const elapsed = Number.isFinite(frameSeconds) ? Math.max(0, frameSeconds) : 0;
  let a = acc + elapsed;
  let ticks = 0;
  while (a >= TICK_SECONDS && ticks < MAX_TICKS_PER_FRAME) {
    a -= TICK_SECONDS;
    ticks += 1;
  }
  if (ticks === MAX_TICKS_PER_FRAME && a >= TICK_SECONDS) {
    // The page could not keep up (slow device, debugger): drop the backlog
    // instead of spiralling.
    a = 0;
  }
  return { ticks, acc: a };
}

/** What the loop needs from the browser (injectable for tests). */
export interface LoopEnv {
  requestFrame(callback: (now: number) => void): number;
  cancelFrame(handle: number): void;
  isVisible(): boolean;
  onVisibilityChange(callback: () => void): void;
}

export function browserLoopEnv(): LoopEnv {
  return {
    requestFrame: (cb) => window.requestAnimationFrame(cb),
    cancelFrame: (h) => window.cancelAnimationFrame(h),
    isVisible: () => document.visibilityState === "visible",
    onVisibilityChange: (cb) => document.addEventListener("visibilitychange", cb),
  };
}

/**
 * Drives `onTick` at a fixed rate and `onFrame` once per animation frame.
 * Time spent with the page hidden is skipped, never fast-forwarded. If a
 * callback throws, the loop stops and reports the error instead of freezing
 * silently.
 */
export class FixedLoop {
  private acc = 0;
  private last: number | null = null;
  private handle: number | null = null;
  private active = false;

  constructor(
    private readonly onTick: () => void,
    private readonly onFrame: (alpha: number) => void,
    private readonly onError: (error: unknown) => void = (e) => console.error(e),
    private readonly env: LoopEnv = browserLoopEnv(),
  ) {
    env.onVisibilityChange(() => {
      this.last = null;
      this.acc = 0;
    });
  }

  get running(): boolean {
    return this.active;
  }

  start(): void {
    if (this.active) return;
    this.active = true;
    this.last = null;
    this.acc = 0;
    this.handle = this.env.requestFrame(this.frame);
  }

  stop(): void {
    this.active = false;
    if (this.handle !== null) this.env.cancelFrame(this.handle);
    this.handle = null;
  }

  private readonly frame = (now: number): void => {
    if (!this.active) return;
    try {
      if (this.env.isVisible()) {
        const elapsed = this.last === null ? 0 : (now - this.last) / 1000;
        this.last = now;
        const { ticks, acc } = accumulate(this.acc, elapsed);
        this.acc = acc;
        for (let i = 0; i < ticks && this.active; i += 1) this.onTick();
        this.onFrame(this.acc / TICK_SECONDS);
      } else {
        this.last = null;
      }
    } catch (error) {
      this.active = false;
      this.onError(error);
    } finally {
      this.handle = this.active ? this.env.requestFrame(this.frame) : null;
    }
  };
}
