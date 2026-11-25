// On-screen controls for touch screens (and mice). They drive the same
// InputState as the keyboard, so the simulation still sees per-tick samples.

import { type InputState, Q_MAX } from "./controls";

/** Buttons that act like holding a key. */
const HOLD_BUTTONS = [
  ["◀ Lean", "ArrowLeft", "Lean left (gimbal)"],
  ["Lean ▶", "ArrowRight", "Lean right (gimbal)"],
  ["↺ RCS", "KeyQ", "Rotate counter-clockwise (RCS)"],
  ["RCS ↻", "KeyE", "Rotate clockwise (RCS)"],
] as const;

export class TouchControls {
  private readonly fill: HTMLDivElement;
  private readonly track: HTMLDivElement;
  private readonly buttons = new Map<string, HTMLButtonElement>();
  private dragging = false;

  constructor(
    root: HTMLElement,
    private readonly input: () => InputState,
  ) {
    root.replaceChildren();

    const throttle = document.createElement("div");
    throttle.className = "touch-throttle";
    const label = document.createElement("span");
    label.className = "touch-label";
    label.textContent = "Throttle";
    // A pointer-driven slider (not an <input>: a focused input would swallow
    // the keyboard controls). Drag anywhere on the track, bottom = 0.
    const track = document.createElement("div");
    this.track = track;
    track.className = "touch-track";
    // Pointer-only: keyboard users have ↑/↓, so it stays out of the
    // accessibility tree rather than announcing a control it cannot operate.
    track.setAttribute("aria-hidden", "true");
    this.fill = document.createElement("div");
    this.fill.className = "touch-fill";
    track.append(this.fill);
    track.addEventListener("pointerdown", (e) => {
      this.dragging = true;
      track.setPointerCapture(e.pointerId);
      this.fromPointer(track, e);
      e.preventDefault();
    });
    track.addEventListener("pointermove", (e) => {
      if (this.dragging && track.hasPointerCapture(e.pointerId)) this.fromPointer(track, e);
    });
    const stop = (): void => {
      this.dragging = false;
    };
    track.addEventListener("pointerup", stop);
    track.addEventListener("pointercancel", stop);
    track.addEventListener("lostpointercapture", stop);

    const cut = this.button("Cut", "Cut the engine off", "touch-cut");
    // CUT bypasses the minimum-throttle floor, exactly like the X key.
    cut.addEventListener("click", () => {
      this.input().press("KeyX");
      this.input().release("KeyX");
    });
    throttle.append(label, track, cut);

    const pad = document.createElement("div");
    pad.className = "touch-pad";
    for (const [text, code, aria] of HOLD_BUTTONS) {
      const b = this.button(text, aria);
      this.buttons.set(code, b);
      const release = (): void => this.input().release(code);
      b.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        b.setPointerCapture(e.pointerId);
        this.input().press(code);
      });
      b.addEventListener("pointerup", release);
      b.addEventListener("pointercancel", release);
      b.addEventListener("lostpointercapture", release);
      b.addEventListener("contextmenu", (e) => e.preventDefault());
      pad.append(b);
    }
    root.append(throttle, pad);
  }

  private button(text: string, aria: string, className = "touch-btn"): HTMLButtonElement {
    const b = document.createElement("button");
    b.type = "button";
    b.className = className;
    b.textContent = text;
    b.setAttribute("aria-label", aria);
    // Never steal keyboard focus (Space would press the focused button).
    b.tabIndex = -1;
    b.addEventListener("mousedown", (e) => e.preventDefault());
    return b;
  }

  private fromPointer(track: HTMLElement, e: PointerEvent): void {
    const rect = track.getBoundingClientRect();
    if (rect.height <= 0) return;
    const frac = 1 - (e.clientY - rect.top) / rect.height;
    this.input().setThrottle(Math.round(Math.min(1, Math.max(0, frac)) * Q_MAX));
  }

  /** The on-screen control for a key code, or the throttle bar for the throttle keys. */
  target(code: string): HTMLElement | null {
    if (code === "ArrowUp" || code === "ArrowDown") return this.track;
    return this.buttons.get(code) ?? null;
  }

  /** Mirrors the current throttle level (call once per frame). */
  sync(level: number): void {
    this.fill.style.height = `${(level / Q_MAX) * 100}%`;
  }
}
