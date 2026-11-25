// Tutorial: three key presses land the stage. The flight starts already in
// trouble (leaning and drifting, set up off screen before the first frame)
// with time frozen. Each frozen moment asks for one key; one tap carries out
// that fix to the end:
//   E  fires the side thrusters until the stage stands upright,
//   →  tilts the engine until the drift stops,
//   ↑  raises the throttle and sets the stage down softly.
// Beyond that, the throttle only keeps the descent steady until the last
// lesson and, once → is pressed, the stage holds itself upright. Every other
// key is ignored, so the tutorial cannot be failed. Pure logic, independent of the DOM (testable against the core).

import { Q_CENTER, Q_MAX, type Quantized } from "./controls";
import type { Simulation, Telemetry, VehicleInfo } from "./sim";

const GRAVITY = 9.81;
const DEG = Math.PI / 180;
const TICKS_PER_S = 60;

/** The scenario every tutorial flies: Training, starting high and over the pad. */
export const TUTORIAL_SEED = 347n;

/** Start of the visible flight: the lean, the drift and the descent rate. */
const START_TILT = 6 * DEG;
const START_DRIFT = 2;
const START_DESCENT = 6;
/** Height of the throttle lesson. */
const THROTTLE_HEIGHT = 40;
/** Side-thruster angular acceleration at full command (rad/s²), measured on the core. */
const RCS_ACCEL = 0.029;
/** Gimbal angular acceleration at full deflection (rad/s²), roughly. */
const GIMBAL_ACCEL = 0.1;

export type Channel = "rcs" | "gimbal" | "throttle";

/** One frozen moment: "Press [key] <action>", with HUD items left in colour. */
export interface Lesson {
  channel: Channel;
  action: string;
  /** Key name (alt text), its picture under public/keys; on touch screens, "<touchVerb> <touchKey>" instead. */
  key: string;
  image: string;
  touchVerb: string;
  touchKey: string;
  /** Key codes that answer the lesson (and unfreeze time). */
  codes: readonly string[];
  highlight: readonly string[];
}

type Stage =
  | { kind: "frozen"; lesson: Lesson }
  | { kind: "fixing"; channel: Channel }
  | { kind: "coast" }
  | { kind: "landing" };

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** Wraps an angle to (−π, π]. */
function wrap(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

/** Quantized level for a command in [−1, 1]. */
function level(u: number): number {
  return clamp(Math.round(Q_CENTER + clamp(u, -1, 1) * Q_CENTER), 0, Q_MAX);
}

/** Throttle level commanding thrust fraction `f` (inverse of the HUD's commandedFraction). */
export function throttleFor(f: number, info: VehicleInfo): number {
  const k = (clamp(f, info.min_throttle, 1) - info.min_throttle) / (1 - info.min_throttle);
  return clamp(Math.ceil(Q_CENTER + (info.throttle_cutoff + k * (1 - info.throttle_cutoff)) * Q_CENTER), 0, Q_MAX);
}

/** Throttle that steers the descent rate toward `vy` (m/s, negative down). */
function descend(t: Telemetry, info: VehicleInfo, vy: number): number {
  const f = (t.mass * (GRAVITY + 1.5 * (vy - t.vy))) / (info.max_thrust * Math.max(0.5, Math.cos(wrap(t.theta))));
  return throttleFor(f, info);
}

/** Gimbal command that turns the stage to `target` and holds it. */
function gimbalTo(t: Telemetry, target: number): number {
  return level((3 * (target - wrap(t.theta)) - 3 * t.omega) / GIMBAL_ACCEL);
}

function lessonFor(channel: Channel, t: Telemetry): Lesson {
  if (channel === "rcs") {
    const left = wrap(t.theta) > 0;
    return {
      channel,
      action: "to fire the side thrusters (RCS) and stand upright",
      key: left ? "E" : "Q",
      image: left ? "keys/e.png" : "keys/q.png",
      touchVerb: "Tap",
      touchKey: left ? "RCS ↻" : "↺ RCS",
      codes: [left ? "KeyE" : "KeyQ"],
      highlight: ["tilt", "rcs"],
    };
  }
  if (channel === "gimbal") {
    const left = t.vx < 0;
    return {
      channel,
      action: "to tilt the engine (gimbal) and stop the drift",
      key: left ? "Right arrow" : "Left arrow",
      image: left ? "keys/right.png" : "keys/left.png",
      touchVerb: "Tap",
      touchKey: left ? "Lean ▶" : "◀ Lean",
      codes: left ? ["ArrowRight", "KeyD"] : ["ArrowLeft", "KeyA"],
      highlight: ["gimbal"],
    };
  }
  return {
    channel,
    action: "to raise the throttle and land softly",
    key: "Up arrow",
    image: "keys/up.png",
    touchVerb: "Drag",
    touchKey: "the throttle bar up",
    codes: ["ArrowUp", "KeyW"],
    highlight: ["vs", "thrust"],
  };
}

export class TutorialScript {
  private stage: Stage;

  /**
   * Flies `sim` off screen into the starting position (leaning, drifting away
   * from the pad, descending steadily, not turning) and freezes on the first
   * lesson. Call before the first frame is drawn.
   */
  constructor(sim: Simulation) {
    const away = sim.state.x > 0 ? 1 : -1;
    // Positive theta leans left, which pushes the stage toward −x.
    const tilt = -away * START_TILT;
    for (let i = 0; i < 20 * TICKS_PER_S; i += 1) {
      const t = sim.state;
      const still = Math.abs(wrap(t.theta) - tilt) < 0.3 * DEG && Math.abs(t.omega) < 0.3 * DEG;
      if (still && t.vx * away >= START_DRIFT && Math.abs(t.vy + START_DESCENT) < 0.3) break;
      const g = gimbalTo(t, tilt);
      sim.step(descend(t, sim.info, -START_DESCENT), g, Q_CENTER);
    }
    this.stage = { kind: "frozen", lesson: lessonFor("rcs", sim.state) };
  }

  get frozen(): boolean {
    return this.stage.kind === "frozen";
  }

  /** The lesson waiting for its key while time is frozen, else null. */
  get lesson(): Lesson | null {
    return this.stage.kind === "frozen" ? this.stage.lesson : null;
  }

  /** A key press: resumes time and starts the fix when it is the frozen lesson's key. */
  press(code: string): boolean {
    if (this.stage.kind !== "frozen" || !this.stage.lesson.codes.includes(code)) return false;
    this.stage = { kind: "fixing", channel: this.stage.lesson.channel };
    return true;
  }

  /** Resumes time from a control that has no key code (the touch throttle bar). */
  answer(channel: Channel): boolean {
    if (this.stage.kind !== "frozen" || this.stage.lesson.channel !== channel) return false;
    this.stage = { kind: "fixing", channel };
    return true;
  }

  /** This tick's commands. Never called while frozen; the player's own keys are not used. */
  command(t: Telemetry, info: VehicleInfo): Quantized {
    const s = this.stage;
    const theta = wrap(t.theta);
    // Until the throttle lesson the throttle only keeps the descent steady.
    const cruise = descend(t, info, -START_DESCENT);
    if (s.kind === "fixing" && s.channel === "rcs") {
      // Side thrusters only: turn back to upright and stop there.
      const rcs = level((1.2 * -theta - 2.2 * t.omega) / RCS_ACCEL);
      if (Math.abs(theta) < 0.3 * DEG && Math.abs(t.omega) < 0.3 * DEG) this.freeze("gimbal", t);
      return [cruise, Q_CENTER, rcs];
    }
    // The gimbal fix: lean to stop the drift and glide back over the pad, then
    // stand upright there. Once started it stays on for the rest of the flight.
    const toPad = clamp(-0.05 * t.x, -2, 2);
    const hold = gimbalTo(t, clamp(0.06 * (t.vx - toPad), -6 * DEG, 6 * DEG));
    if (s.kind === "fixing" && s.channel === "gimbal") this.stage = { kind: "coast" };
    if (this.stage.kind === "coast") {
      if (t.clearance < THROTTLE_HEIGHT) this.freeze("throttle", t);
      return [cruise, hold, Q_CENTER];
    }
    // Throttle: slow down with height and touch down gently.
    if (s.kind === "fixing") this.stage = { kind: "landing" };
    const alt = Math.max(0, t.clearance);
    return [descend(t, info, -clamp(0.6 + 0.12 * alt, 1, START_DESCENT)), hold, Q_CENTER];
  }

  private freeze(channel: Channel, t: Telemetry): void {
    // This tick still flies; time stops from the next one.
    this.stage = { kind: "frozen", lesson: lessonFor(channel, t) };
  }
}
