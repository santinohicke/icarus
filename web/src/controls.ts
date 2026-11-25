// Keyboard input, sampled once per simulation tick.
//
// The simulation only ever sees quantized levels (0..=254, 127 = neutral),
// sampled per tick, so a recorded run replays exactly.

export const Q_MAX = 254;
export const Q_CENTER = 127;

/** Throttle levels gained or lost per tick while a key is held (full sweep ≈ 1.4 s). */
export const THROTTLE_STEP = 3;

export type Quantized = readonly [throttle: number, gimbal: number, rcs: number];

const THROTTLE_UP = ["ArrowUp", "KeyW"];
const THROTTLE_DOWN = ["ArrowDown", "KeyS"];
const GIMBAL_LEFT = ["ArrowLeft", "KeyA"];
const GIMBAL_RIGHT = ["ArrowRight", "KeyD"];
const RCS_CCW = ["KeyQ"];
const RCS_CW = ["KeyE"];
const CUT = "KeyX";
const FULL = "KeyZ";

/** Keys the game handles: their default browser action (scrolling) is suppressed. */
export const GAME_KEYS = new Set([
  ...THROTTLE_UP,
  ...THROTTLE_DOWN,
  ...GIMBAL_LEFT,
  ...GIMBAL_RIGHT,
  ...RCS_CCW,
  ...RCS_CW,
  CUT,
  FULL,
  "Space",
]);

function clampLevel(v: number): number {
  return Math.min(Q_MAX, Math.max(0, Math.round(v)));
}

/**
 * Lowest quantized throttle level that keeps the engine running, for a
 * normalized cut-off threshold (levels below it command the engine off).
 */
export function minRunningLevel(cutoff: number): number {
  for (let q = 0; q <= Q_MAX; q += 1) {
    if ((q - Q_CENTER) / Q_CENTER >= cutoff) return q;
  }
  return Q_MAX;
}

/** Pure per-tick input logic, independent of the DOM (testable). */
export class InputState {
  private readonly held = new Set<string>();
  /** Keys pressed since the last sample, even if already released (short taps count). */
  private readonly tapped = new Set<string>();
  private pending: string[] = [];
  private throttle: number;
  private minRunning: number;

  constructor(initialThrottle: number, minRunning = 0) {
    this.throttle = clampLevel(initialThrottle);
    this.minRunning = clampLevel(minRunning);
  }

  isHeld(code: string): boolean {
    return this.held.has(code);
  }

  /** Pressed since the last sample (even if already released): a fresh press, not a key held from before. */
  wasPressed(code: string): boolean {
    return this.tapped.has(code);
  }

  press(code: string): void {
    if (!this.held.has(code)) this.pending.push(code);
    this.held.add(code);
    this.tapped.add(code);
  }

  release(code: string): void {
    this.held.delete(code);
  }

  releaseAll(): void {
    this.held.clear();
  }

  /** Forgets queued presses (cut/full, short taps), e.g. when pausing. */
  clearPending(): void {
    this.pending = [];
    this.tapped.clear();
  }

  /**
   * Starts a new run. Keys still physically held stay held (holding ↑ through
   * a restart keeps working); queued presses are dropped.
   */
  reset(initialThrottle: number, minRunning = this.minRunning): void {
    this.clearPending();
    this.throttle = clampLevel(initialThrottle);
    this.minRunning = clampLevel(minRunning);
  }

  /**
   * Sets the throttle directly (touch slider). Like ramping down, it stops at
   * minimum throttle while the engine runs: only the cut-off control shuts it off.
   */
  setThrottle(level: number): void {
    if (!Number.isFinite(level)) return;
    const target = clampLevel(level);
    const running = this.throttle >= this.minRunning;
    this.throttle = running ? Math.max(target, this.minRunning) : target;
  }

  get throttleLevel(): number {
    return this.throttle;
  }

  private active(codes: readonly string[]): boolean {
    return codes.some((c) => this.held.has(c) || this.tapped.has(c));
  }

  /** Produces this tick's quantized inputs. Call exactly once per tick. */
  sample(): Quantized {
    for (const code of this.pending) {
      if (code === CUT) this.throttle = 0;
      if (code === FULL) this.throttle = Q_MAX;
    }
    this.pending = [];

    const up = this.active(THROTTLE_UP);
    const down = this.active(THROTTLE_DOWN);
    if (up && !down) this.throttle = clampLevel(this.throttle + THROTTLE_STEP);
    if (down && !up) {
      // Ramping down stops at minimum throttle: only X shuts the engine off
      // (with a single ignition, an accidental cut-off would end the run).
      const floor = this.throttle >= this.minRunning ? this.minRunning : 0;
      this.throttle = Math.max(floor, clampLevel(this.throttle - THROTTLE_STEP));
    }

    // Left arrow leans the stage left: positive gimbal gives a
    // counter-clockwise torque (nose to the left).
    const left = this.active(GIMBAL_LEFT);
    const right = this.active(GIMBAL_RIGHT);
    const gimbal = left === right ? Q_CENTER : left ? Q_MAX : 0;

    const ccw = this.active(RCS_CCW);
    const cw = this.active(RCS_CW);
    const rcs = ccw === cw ? Q_CENTER : ccw ? Q_MAX : 0;

    this.tapped.clear();
    return [this.throttle, gimbal, rcs];
  }
}

function isEditable(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement
  );
}

/** Connects an `InputState` to window keyboard events. */
export class KeyboardControls {
  readonly input: InputState;
  private readonly onDown = (e: KeyboardEvent): void => {
    if (e.ctrlKey || e.metaKey || e.altKey || isEditable(e.target)) return;
    if (GAME_KEYS.has(e.code) && !(e.target instanceof HTMLButtonElement && e.code === "Space")) {
      e.preventDefault();
    }
    // Auto-repeat re-registers a key we lost track of (e.g. after a blur).
    if (!e.repeat || !this.input.isHeld(e.code)) this.input.press(e.code);
  };
  private readonly onUp = (e: KeyboardEvent): void => {
    this.input.release(e.code);
  };
  private readonly onBlur = (): void => {
    // Avoid stuck keys when the window loses focus mid-press.
    this.input.releaseAll();
  };

  constructor(initialThrottle: number, minRunning = 0) {
    this.input = new InputState(initialThrottle, minRunning);
    window.addEventListener("keydown", this.onDown);
    window.addEventListener("keyup", this.onUp);
    window.addEventListener("blur", this.onBlur);
  }

  dispose(): void {
    window.removeEventListener("keydown", this.onDown);
    window.removeEventListener("keyup", this.onUp);
    window.removeEventListener("blur", this.onBlur);
  }
}
