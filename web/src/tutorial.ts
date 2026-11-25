// Tutorial: a Training flight with three lessons, flown by the player the
// whole way. Before each lesson the script causes a small problem with a
// short scripted push (a lean, a sideways drift, a sudden drop in thrust),
// then time freezes until the player presses the one key that fixes it.
// There is no autopilot: outside those pushes every command is the player's.
// Pure logic, independent of the DOM (testable against the WASM core).

import { Q_CENTER, Q_MAX, type Quantized, minRunningLevel } from "./controls";
import type { Telemetry, VehicleInfo } from "./sim";

const DEG = Math.PI / 180;
const TICKS_PER_S = 60;

/** The scenario every tutorial flies: Training, starting high and over the pad. */
export const TUTORIAL_SEED = 347n;

export type Channel = "throttle" | "gimbal" | "rcs";

/**
 * What the player reads next to the rocket. Frozen: "<verb> [key] <action>".
 * While fixing it: "<hintVerb> [key] <hint>".
 */
export interface Lesson {
  channel: Channel;
  verb: string;
  action: string;
  hintVerb: string;
  hint: string;
  /** Key name (alt text), its picture under public/keys, and the touch control instead. */
  key: string;
  image: string;
  touchKey: string;
  /** Key codes that answer the lesson (and unfreeze time). */
  codes: readonly string[];
  /** HUD items left in colour while the rest of the screen goes gray. */
  highlight: readonly string[];
}

type Stage =
  | { kind: "setup"; lesson: Channel; ticks: number; upright?: boolean }
  | { kind: "frozen"; lesson: Lesson }
  | { kind: "practice"; lesson: Lesson }
  | { kind: "gap"; next: Channel; ticks: number }
  | { kind: "done" };

/** Lean the RCS lesson causes. */
const SETUP_TILT = 6 * DEG;
/** Sideways speed the gimbal lesson causes. */
const SETUP_DRIFT = 2.2;
/** Height where the thrust drops for the throttle lesson. */
const THROTTLE_HEIGHT = 45;

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** Wraps an angle to (−π, π]. */
function wrap(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

/**
 * Gimbal and side-thruster levels that turn the stage to `target` (rad) and
 * hold it there: the scripted push that sets up a lesson.
 */
function turnTo(t: Telemetry, target: number): [gimbal: number, rcs: number] {
  const u = clamp((3 * (target - wrap(t.theta)) - 3 * t.omega) / 0.1, -1, 1);
  const level = clamp(Math.round(Q_CENTER + u * Q_CENTER), 0, Q_MAX);
  return [level, level];
}

function lessonFor(channel: Channel, t: Telemetry, info: VehicleInfo): Lesson {
  if (channel === "rcs") {
    const left = wrap(t.theta) > 0;
    const deg = Math.abs(wrap(t.theta) / DEG).toFixed(0);
    return {
      channel,
      verb: "Press",
      action: "to fire the side thrusters (RCS) and stand back upright",
      hintVerb: "Hold",
      hint: "until the tilt is near 0°",
      key: left ? "E" : "Q",
      image: left ? "keys/e.png" : "keys/q.png",
      touchKey: left ? "RCS ↻" : "↺ RCS",
      codes: [left ? "KeyE" : "KeyQ"],
      highlight: ["tilt", "rcs"],
    };
  }
  if (channel === "gimbal") {
    const left = t.vx < 0;
    return {
      channel,
      verb: "Press",
      action: "to tilt the engine (gimbal) and push back toward the pad",
      hintVerb: "Tap",
      hint: "to lean, then the other arrow to stand up once the drift stops",
      key: left ? "Right arrow" : "Left arrow",
      image: left ? "keys/right.png" : "keys/left.png",
      touchKey: left ? "Lean ▶" : "◀ Lean",
      codes: left ? ["ArrowRight", "KeyD"] : ["ArrowLeft", "KeyA"],
      highlight: ["gimbal", "tilt"],
    };
  }
  return {
    channel,
    verb: "Press",
    action: `to raise the throttle: the legs only survive ${info.max_vertical_speed} m/s`,
    hintVerb: "Hold",
    hint: `until you fall slower than ${info.max_vertical_speed} m/s`,
    key: "Up arrow",
    image: "keys/up.png",
    touchKey: "Throttle",
    codes: ["ArrowUp", "KeyW"],
    highlight: ["vs", "thrust"],
  };
}

export class TutorialScript {
  private stage: Stage = { kind: "setup", lesson: "rcs", ticks: 0 };
  /** Lessons fixed so far. */
  completed = 0;

  get frozen(): boolean {
    return this.stage.kind === "frozen";
  }

  /** The lesson waiting for its key (time frozen) or being fixed, else null. */
  get lesson(): Lesson | null {
    return this.stage.kind === "frozen" || this.stage.kind === "practice" ? this.stage.lesson : null;
  }

  get done(): boolean {
    return this.stage.kind === "done";
  }

  /** A key press: resumes time when it is the frozen lesson's key. */
  press(code: string): boolean {
    if (this.stage.kind !== "frozen" || !this.stage.lesson.codes.includes(code)) return false;
    this.stage = { kind: "practice", lesson: this.stage.lesson };
    return true;
  }

  /** Resumes time from a control that has no key code (the touch throttle bar). */
  answer(channel: Channel): boolean {
    if (this.stage.kind !== "frozen" || this.stage.lesson.channel !== channel) return false;
    this.stage = { kind: "practice", lesson: this.stage.lesson };
    return true;
  }

  /**
   * This tick's commands: the player's, except on the channel a lesson setup
   * is pushing. Never called while frozen.
   */
  command(t: Telemetry, info: VehicleInfo, player: Quantized): Quantized {
    // However the earlier lessons went, the throttle lesson comes before touchdown.
    const s = this.stage;
    const throttleLesson = (s.kind === "setup" || s.kind === "practice") && this.channelOf(s) === "throttle";
    if (s.kind !== "done" && s.kind !== "frozen" && !throttleLesson && t.clearance < THROTTLE_HEIGHT) {
      this.stage = { kind: "setup", lesson: "throttle", ticks: 0 };
    }
    return this.step(t, info, player);
  }

  private channelOf(s: Stage): Channel | null {
    return s.kind === "setup" ? s.lesson : s.kind === "practice" ? s.lesson.channel : null;
  }

  private step(t: Telemetry, info: VehicleInfo, player: Quantized): Quantized {
    const s = this.stage;
    const theta = wrap(t.theta);
    const [throttle, gimbal, rcs] = player;
    switch (s.kind) {
      case "setup": {
        s.ticks += 1;
        if (s.lesson === "rcs") {
          // A push leans the stage and leaves it there, still.
          const still = Math.abs(theta - SETUP_TILT) < 1 * DEG && Math.abs(t.omega) < 1 * DEG;
          if (still || s.ticks > 5 * TICKS_PER_S) this.freeze("rcs", t, info);
          return [throttle, ...turnTo(t, SETUP_TILT)];
        }
        if (s.lesson === "gimbal") {
          // Lean away from the pad until the stage slides, then stand it back up.
          const away = t.x > 0 ? 1 : -1;
          if (!s.upright && (t.vx * away > SETUP_DRIFT || s.ticks > 5 * TICKS_PER_S)) s.upright = true;
          const still = Math.abs(theta) < 1 * DEG && Math.abs(t.omega) < 1 * DEG;
          if (s.upright && (still || s.ticks > 9 * TICKS_PER_S)) this.freeze("gimbal", t, info);
          // Positive theta leans left, so leaning toward +x needs a negative angle.
          return [throttle, ...turnTo(t, s.upright ? 0 : -away * 10 * DEG)];
        }
        // The thrust drops to its minimum until the stage falls fast (an engine
        // the player cut stays off: the script never relights it).
        if (t.vy < -6.5 || t.clearance < 30) this.freeze("throttle", t, info);
        return [t.engine_on > 0.5 ? minRunningLevel(info.throttle_cutoff) : throttle, gimbal, rcs];
      }
      case "practice": {
        const c = s.lesson.channel;
        const fixed =
          c === "rcs" ? Math.abs(theta) < 1.5 * DEG : c === "gimbal" ? Math.abs(t.vx) < 0.8 : t.vy > -info.max_vertical_speed;
        if (fixed) {
          this.completed += 1;
          this.stage =
            c === "rcs" ? { kind: "gap", next: "gimbal", ticks: 0 } : c === "gimbal" ? { kind: "gap", next: "throttle", ticks: 0 } : { kind: "done" };
        }
        return player;
      }
      case "gap": {
        s.ticks += 1;
        // A moment to fly freely; the throttle lesson waits for its height.
        if (s.next === "gimbal" && s.ticks > 1 * TICKS_PER_S) this.stage = { kind: "setup", lesson: "gimbal", ticks: 0 };
        return player;
      }
      case "frozen":
      case "done":
        return player;
    }
  }

  private freeze(channel: Channel, t: Telemetry, info: VehicleInfo): void {
    // This tick still flies; time stops from the next one.
    this.stage = { kind: "frozen", lesson: lessonFor(channel, t, info) };
  }
}
